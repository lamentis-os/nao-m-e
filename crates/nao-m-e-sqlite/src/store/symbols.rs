use std::collections::{BTreeMap, BTreeSet, VecDeque};

use nao_m_e::{EpisodeDraft, Memory, SymbolId};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OptionalExtension, Row, Transaction, params_from_iter};
use unicode_normalization::UnicodeNormalization;

use crate::error::{StoreError, StoreIntegrityError};
use crate::format;

use super::{SqliteStore, read_metadata, read_u64};

const MAX_SYMBOL_QUERY_BINDINGS: usize = 900;
const RECENT_NON_CANONICAL_SYMBOLS: usize = 256;

const _: () = {
    assert!(unicode_case_mapping::UNICODE_VERSION.0 == 16);
    assert!(unicode_case_mapping::UNICODE_VERSION.1 == 0);
    assert!(unicode_case_mapping::UNICODE_VERSION.2 == 0);
    assert!(unicode_normalization::UNICODE_VERSION.0 == 16);
    assert!(unicode_normalization::UNICODE_VERSION.1 == 0);
    assert!(unicode_normalization::UNICODE_VERSION.2 == 0);
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NextSymbolId {
    Next(u64),
    Exhausted,
}

impl NextSymbolId {
    fn allocate_range(self, count: usize) -> Option<(u64, Self)> {
        debug_assert!(count != 0);
        let Self::Next(first) = self else {
            return None;
        };
        let offset = u64::try_from(count.checked_sub(1)?).ok()?;
        let last = first.checked_add(offset)?;
        let next = last.checked_add(1).map_or(Self::Exhausted, Self::Next);
        Some((first, next))
    }

    const fn tail(self) -> Option<u64> {
        match self {
            Self::Next(0) => None,
            Self::Next(next) => Some(next - 1),
            Self::Exhausted => Some(u64::MAX),
        }
    }

    const fn contains(self, id: u64) -> bool {
        match self {
            Self::Next(next) => id < next,
            Self::Exhausted => true,
        }
    }
}

#[derive(Debug)]
pub(super) struct SymbolState {
    persisted_tail: Option<u64>,
    next_id: NextSymbolId,
    pending: BTreeMap<String, u64>,
}

impl SymbolState {
    fn from_tail(tail: Option<u64>) -> Self {
        let next_id = match tail {
            None => NextSymbolId::Next(0),
            Some(u64::MAX) => NextSymbolId::Exhausted,
            Some(tail) => NextSymbolId::Next(tail + 1),
        };
        Self {
            persisted_tail: tail,
            next_id,
            pending: BTreeMap::new(),
        }
    }

    pub(super) fn contains_persisted(&self, id: u64) -> bool {
        self.persisted_tail.is_some_and(|tail| id <= tail)
    }

    const fn contains_current(&self, id: u64) -> bool {
        self.next_id.contains(id)
    }

    pub(super) const fn is_persisted_empty(&self) -> bool {
        self.persisted_tail.is_none()
    }

    pub(super) fn mark_saved(&mut self) {
        self.persisted_tail = self.next_id.tail();
        self.pending.clear();
    }

    #[cfg(test)]
    pub(super) fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

impl SqliteStore {
    /// Normalizes and interns symbol values in input order.
    ///
    /// Duplicate normalized values return the same identifier. New assignments
    /// remain staged in this store until [`Self::save`] commits them atomically
    /// with pending episode and feedback changes. An invalid value or exhausted
    /// identifier space leaves the complete staged symbol state unchanged.
    pub fn intern_symbols(&mut self, values: &[String]) -> Result<Vec<SymbolId>, StoreError> {
        let NormalizedBatch {
            mut unique_values,
            positions,
        } = normalize_batch(values)?;
        if unique_values.is_empty() {
            return Ok(Vec::new());
        }

        let state = &self.symbols;
        let mut assignments = vec![0_u64; unique_values.len()];
        let mut unresolved_indexes = Vec::new();
        for (index, value) in unique_values.iter().enumerate() {
            if let Some(&id) = state.pending.get(value) {
                assignments[index] = id;
            } else {
                unresolved_indexes.push(index);
            }
        }

        let mut persisted = vec![false; unique_values.len()];
        read_symbol_ids_for_values(
            &self.connection,
            &unique_values,
            &unresolved_indexes,
            &mut assignments,
            &mut persisted,
        )?;
        let mut new_count = 0;
        for position in 0..unresolved_indexes.len() {
            let index = unresolved_indexes[position];
            if persisted[index] {
                let id = assignments[index];
                if !state.contains_persisted(id) {
                    let (_, actual_revision, _) = read_metadata(&self.connection)?;
                    if actual_revision != self.expected_revision {
                        return Err(StoreError::ConcurrentModification {
                            expected_revision: self.expected_revision,
                            actual_revision,
                        });
                    }
                    return Err(StoreIntegrityError::InvalidSymbol {
                        id,
                        detail: "symbol row changed outside this store session",
                    }
                    .into());
                }
            } else {
                unresolved_indexes[new_count] = index;
                new_count += 1;
            }
        }
        unresolved_indexes.truncate(new_count);

        let (first_new_id, next_id) = if unresolved_indexes.is_empty() {
            (0, state.next_id)
        } else {
            state
                .next_id
                .allocate_range(unresolved_indexes.len())
                .ok_or(StoreError::SymbolIdExhausted)?
        };
        for (offset, &index) in unresolved_indexes.iter().enumerate() {
            assignments[index] = first_new_id
                .checked_add(u64::try_from(offset).expect("a slice index fits in u64"))
                .expect("the validated symbol range contains every offset");
        }

        let resolved = positions
            .into_iter()
            .map(|position| SymbolId::new(assignments[position]))
            .collect();

        let state = &mut self.symbols;
        for index in unresolved_indexes {
            let id = assignments[index];
            let value = std::mem::take(&mut unique_values[index]);
            let previous = state.pending.insert(value, id);
            debug_assert!(previous.is_none());
        }
        state.next_id = next_id;
        Ok(resolved)
    }

    /// Resolves symbol identifiers in input order.
    ///
    /// Values staged by [`Self::intern_symbols`] are visible before a save.
    /// Only the requested values are loaded from SQLite. Unknown identifiers
    /// produce `None` at their input position.
    pub fn symbol_values(&self, ids: &[SymbolId]) -> Result<Vec<Option<String>>, StoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let state = &self.symbols;
        let requested_pending: BTreeSet<_> = ids
            .iter()
            .map(|id| id.get())
            .filter(|&id| !state.contains_persisted(id) && state.contains_current(id))
            .collect();
        let mut pending_by_id = BTreeMap::new();
        for (value, &id) in &state.pending {
            if requested_pending.contains(&id) {
                pending_by_id.insert(id, value.as_str());
                if pending_by_id.len() == requested_pending.len() {
                    break;
                }
            }
        }
        let mut persisted_ids = Vec::new();
        let mut last_persisted_positions = BTreeMap::new();
        for (position, id) in ids.iter().enumerate() {
            let id = id.get();
            if pending_by_id.contains_key(&id) {
                continue;
            }
            if !state.contains_persisted(id) {
                continue;
            }
            if last_persisted_positions.insert(id, position).is_none() {
                persisted_ids.push(id);
            }
        }
        let mut persisted = read_symbol_values_for_ids(&self.connection, &persisted_ids)?;
        ids.iter()
            .enumerate()
            .map(|(position, id)| {
                let id = id.get();
                if let Some(value) = pending_by_id.get(&id) {
                    return Ok(Some((*value).to_owned()));
                }
                if !state.contains_persisted(id) {
                    return Ok(None);
                }
                let value = if last_persisted_positions.get(&id) == Some(&position) {
                    persisted.remove(&id)
                } else {
                    persisted.get(&id).cloned()
                };
                value.map(Some).ok_or_else(|| {
                    StoreIntegrityError::InvalidSymbol {
                        id,
                        detail: "symbol row is absent",
                    }
                    .into()
                })
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SymbolValueError {
    Empty,
    Control,
    TooLong,
}

impl SymbolValueError {
    const fn detail(self) -> &'static str {
        match self {
            Self::Empty => "normalized value is empty",
            Self::Control => "normalized value contains a control character",
            Self::TooLong => "normalized UTF-8 value exceeds 4096 bytes",
        }
    }
}

fn normalize_symbol(value: &str) -> Result<String, SymbolValueError> {
    let lowercase = value.nfkc().flat_map(unicode_lowercase).nfkc();
    let mut normalized = String::with_capacity(value.len().min(format::MAX_SYMBOL_BYTES));
    let mut whitespace_pending = false;
    for character in lowercase {
        if is_unicode_16_whitespace(character) {
            whitespace_pending = !normalized.is_empty();
            continue;
        }
        if is_unicode_control(character) {
            return Err(SymbolValueError::Control);
        }
        if whitespace_pending {
            normalized.push(' ');
            whitespace_pending = false;
        }
        normalized.push(character);
        if normalized.len() > format::MAX_SYMBOL_BYTES {
            return Err(SymbolValueError::TooLong);
        }
    }
    if normalized.is_empty() {
        return Err(SymbolValueError::Empty);
    }
    Ok(normalized)
}

fn unicode_lowercase(character: char) -> impl Iterator<Item = char> {
    let mapping = unicode_case_mapping::to_lowercase(character);
    let maps_to_self = mapping[0] == 0;
    std::iter::once(character)
        .filter(move |_| maps_to_self)
        .chain(
            mapping
                .into_iter()
                .filter(move |&scalar| !maps_to_self && scalar != 0)
                .map(|scalar| {
                    char::from_u32(scalar).expect("Unicode case mapping contains valid scalars")
                }),
        )
}

const fn is_unicode_16_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'..='\u{000D}'
            | '\u{0020}'
            | '\u{0085}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
    )
}

const fn is_unicode_control(character: char) -> bool {
    matches!(character, '\u{0000}'..='\u{001F}' | '\u{007F}'..='\u{009F}')
}

#[derive(Debug, Eq, PartialEq)]
struct NormalizedBatch {
    unique_values: Vec<String>,
    positions: Vec<usize>,
}

fn normalize_batch(values: &[String]) -> Result<NormalizedBatch, StoreError> {
    let mut canonical_positions = BTreeMap::<String, usize>::new();
    let mut recent_non_canonical = BTreeMap::<&str, usize>::new();
    let mut recent_order = VecDeque::<&str>::new();
    let mut positions = Vec::with_capacity(values.len());

    for (index, value) in values.iter().enumerate() {
        if let Some(&position) = canonical_positions.get(value.as_str()) {
            positions.push(position);
            continue;
        }
        if let Some(&position) = recent_non_canonical.get(value.as_str()) {
            positions.push(position);
            continue;
        }

        let normalized =
            normalize_symbol(value).map_err(|error| StoreError::InvalidSymbolValue {
                index,
                detail: error.detail(),
            })?;
        let is_non_canonical = normalized != value.as_str();
        let next_position = canonical_positions.len();
        let position = *canonical_positions
            .entry(normalized)
            .or_insert(next_position);
        positions.push(position);

        if is_non_canonical {
            if recent_non_canonical.len() == RECENT_NON_CANONICAL_SYMBOLS {
                let oldest = recent_order
                    .pop_front()
                    .expect("a full recent-symbol cache has an oldest entry");
                recent_non_canonical.remove(oldest);
            }
            recent_non_canonical.insert(value.as_str(), position);
            recent_order.push_back(value.as_str());
        }
    }

    let mut unique_values: Vec<Option<String>> = std::iter::repeat_with(|| None)
        .take(canonical_positions.len())
        .collect();
    for (value, position) in canonical_positions {
        unique_values[position] = Some(value);
    }
    Ok(NormalizedBatch {
        unique_values: unique_values
            .into_iter()
            .map(|value| value.expect("every canonical symbol has a first-occurrence position"))
            .collect(),
        positions,
    })
}

pub(super) fn validate_symbol_catalog(connection: &Connection) -> Result<SymbolState, StoreError> {
    let mut statement = connection.prepare("SELECT id, value FROM symbols ORDER BY id")?;
    let mut rows = statement.query([])?;
    let mut next = NextSymbolId::Next(0);
    let mut tail = None;
    while let Some(row) = rows.next()? {
        let NextSymbolId::Next(expected) = next else {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "symbol row follows the maximum unsigned identifier",
            }
            .into());
        };
        let (id, _) = read_symbol_row(row)?;
        if id != expected {
            return Err(StoreIntegrityError::NonContiguousSymbolId {
                expected,
                found: id,
            }
            .into());
        }
        tail = Some(id);
        next = id
            .checked_add(1)
            .map_or(NextSymbolId::Exhausted, NextSymbolId::Next);
    }
    let state = SymbolState::from_tail(tail);
    debug_assert_eq!(state.next_id, next);
    Ok(state)
}

fn read_symbol_row(row: &Row<'_>) -> Result<(u64, String), StoreError> {
    let id = read_u64(row, 0, "symbols", "id")?;
    let ValueRef::Text(bytes) = row.get_ref(1)? else {
        return Err(StoreIntegrityError::InvalidEncoding {
            table: "symbols",
            column: "value",
        }
        .into());
    };
    let value = std::str::from_utf8(bytes).map_err(|_| StoreIntegrityError::InvalidEncoding {
        table: "symbols",
        column: "value",
    })?;
    let canonical =
        normalize_symbol(value).map_err(|error| StoreIntegrityError::InvalidSymbol {
            id,
            detail: error.detail(),
        })?;
    if canonical != value {
        return Err(StoreIntegrityError::InvalidSymbol {
            id,
            detail: "value is not normalized",
        }
        .into());
    }
    Ok((id, canonical))
}

fn placeholders(count: usize) -> String {
    let mut placeholders = String::with_capacity(count * 3);
    if count != 0 {
        placeholders.push('?');
        for _ in 1..count {
            placeholders.push_str(", ?");
        }
    }
    placeholders
}

fn read_symbol_ids_for_values(
    connection: &Connection,
    values: &[String],
    indexes: &[usize],
    assignments: &mut [u64],
    found: &mut [bool],
) -> Result<(), StoreError> {
    let full_end = indexes.len() / MAX_SYMBOL_QUERY_BINDINGS * MAX_SYMBOL_QUERY_BINDINGS;
    let (full_chunks, remainder) = indexes.split_at(full_end);

    if !full_chunks.is_empty() {
        let sql = format!(
            "SELECT id, value FROM symbols WHERE value IN ({}) ORDER BY id",
            placeholders(MAX_SYMBOL_QUERY_BINDINGS)
        );
        let mut statement = connection.prepare(&sql)?;
        for chunk in full_chunks.chunks_exact(MAX_SYMBOL_QUERY_BINDINGS) {
            read_symbol_id_rows(&mut statement, values, chunk, assignments, found)?;
        }
    }

    if !remainder.is_empty() {
        let sql = format!(
            "SELECT id, value FROM symbols WHERE value IN ({}) ORDER BY id",
            placeholders(remainder.len())
        );
        let mut statement = connection.prepare(&sql)?;
        read_symbol_id_rows(&mut statement, values, remainder, assignments, found)?;
    }
    Ok(())
}

fn read_symbol_id_rows(
    statement: &mut rusqlite::Statement<'_>,
    values: &[String],
    indexes: &[usize],
    assignments: &mut [u64],
    found: &mut [bool],
) -> Result<(), StoreError> {
    let requested: BTreeMap<_, _> = indexes
        .iter()
        .map(|&index| (values[index].as_str(), index))
        .collect();
    let mut rows = statement.query(params_from_iter(
        indexes.iter().map(|&index| values[index].as_str()),
    ))?;
    while let Some(row) = rows.next()? {
        let (id, value) = read_symbol_row(row)?;
        let Some(&index) = requested.get(value.as_str()) else {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "symbol query returned an unrequested value",
            }
            .into());
        };
        if found[index] {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "symbol value appears more than once",
            }
            .into());
        }
        assignments[index] = id;
        found[index] = true;
    }
    Ok(())
}

fn read_symbol_values_for_ids(
    connection: &Connection,
    ids: &[u64],
) -> Result<BTreeMap<u64, String>, StoreError> {
    let mut found = BTreeMap::new();
    let full_end = ids.len() / MAX_SYMBOL_QUERY_BINDINGS * MAX_SYMBOL_QUERY_BINDINGS;
    let (full_chunks, remainder) = ids.split_at(full_end);

    if !full_chunks.is_empty() {
        let sql = format!(
            "SELECT id, value FROM symbols WHERE id IN ({}) ORDER BY id",
            placeholders(MAX_SYMBOL_QUERY_BINDINGS)
        );
        let mut statement = connection.prepare(&sql)?;
        for chunk in full_chunks.chunks_exact(MAX_SYMBOL_QUERY_BINDINGS) {
            read_symbol_value_rows(&mut statement, chunk, &mut found)?;
        }
    }

    if !remainder.is_empty() {
        let sql = format!(
            "SELECT id, value FROM symbols WHERE id IN ({}) ORDER BY id",
            placeholders(remainder.len())
        );
        let mut statement = connection.prepare(&sql)?;
        read_symbol_value_rows(&mut statement, remainder, &mut found)?;
    }
    Ok(found)
}

fn read_symbol_value_rows(
    statement: &mut rusqlite::Statement<'_>,
    ids: &[u64],
    found: &mut BTreeMap<u64, String>,
) -> Result<(), StoreError> {
    let encoded: Vec<_> = ids.iter().copied().map(format::encode_u64).collect();
    let mut rows = statement.query(params_from_iter(
        encoded.iter().map(|value| value.as_slice()),
    ))?;
    while let Some(row) = rows.next()? {
        let (id, value) = read_symbol_row(row)?;
        if found.insert(id, value).is_some() {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "symbol identifier appears more than once",
            }
            .into());
        }
    }
    Ok(())
}

pub(super) fn validate_persisted_episode_symbols(
    sequence: u64,
    draft: &EpisodeDraft,
    symbols: &SymbolState,
) -> Result<(), StoreError> {
    for attribute in draft.attributes() {
        if !symbols.contains_persisted(attribute.key().get()) {
            return Err(StoreIntegrityError::InvalidEpisode {
                sequence,
                detail: "attribute key identifier is absent from the symbol catalog",
            }
            .into());
        }
        if attribute
            .values()
            .iter()
            .any(|value| !symbols.contains_persisted(value.get()))
        {
            return Err(StoreIntegrityError::InvalidEpisode {
                sequence,
                detail: "attribute value identifier is absent from the symbol catalog",
            }
            .into());
        }
    }
    Ok(())
}

pub(super) fn validate_new_episode_symbols(
    memory: &Memory,
    start: usize,
    symbols: &SymbolState,
) -> Result<(), StoreError> {
    for atom in memory.episodes().skip(start) {
        for attribute in atom.attributes() {
            let key = attribute.key().get();
            if !symbols.contains_current(key) {
                return Err(StoreError::UnknownSymbolId { id: key });
            }
            if let Some(value) = attribute
                .values()
                .iter()
                .map(|value| value.get())
                .find(|&value| !symbols.contains_current(value))
            {
                return Err(StoreError::UnknownSymbolId { id: value });
            }
        }
    }
    Ok(())
}

pub(super) fn verify_symbol_tail(
    transaction: &Transaction<'_>,
    state: &SymbolState,
) -> Result<(), StoreError> {
    let tail: Option<Vec<u8>> = transaction
        .query_row(
            "SELECT id FROM symbols ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let actual_tail = tail
        .as_deref()
        .map(|bytes| {
            format::decode_u64(bytes).ok_or(StoreIntegrityError::InvalidEncoding {
                table: "symbols",
                column: "id",
            })
        })
        .transpose()?;
    if actual_tail == state.persisted_tail {
        Ok(())
    } else {
        Err(StoreIntegrityError::InvalidMetadata {
            detail: "persisted symbol tail changed outside this store session",
        }
        .into())
    }
}

pub(super) fn insert_pending_symbols(
    transaction: &Transaction<'_>,
    state: &SymbolState,
) -> Result<(), StoreError> {
    if state.pending.is_empty() {
        return Ok(());
    }
    let first_id = state.persisted_tail.map_or(0, |tail| {
        tail.checked_add(1)
            .expect("a non-empty pending range follows a non-maximum persisted tail")
    });
    let mut pending = vec![None; state.pending.len()];
    for (value, &id) in &state.pending {
        let offset = usize::try_from(
            id.checked_sub(first_id)
                .expect("a pending symbol follows the persisted tail"),
        )
        .expect("a pending symbol offset fits in memory");
        let slot = pending
            .get_mut(offset)
            .expect("pending symbol identifiers form one contiguous range");
        debug_assert!(slot.is_none());
        *slot = Some(value);
    }
    let mut insert = transaction.prepare("INSERT INTO symbols (id, value) VALUES (?1, ?2)")?;
    for (offset, value) in pending.into_iter().enumerate() {
        let id = first_id
            .checked_add(u64::try_from(offset).expect("a pending offset fits in u64"))
            .expect("the pending identifier range is valid");
        let id = format::encode_u64(id);
        insert.execute((
            id.as_slice(),
            value.expect("every pending identifier is assigned once"),
        ))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_preserves_the_complete_u64_range_atomically() {
        assert_eq!(
            NextSymbolId::Next(7).allocate_range(3),
            Some((7, NextSymbolId::Next(10)))
        );
        assert_eq!(
            NextSymbolId::Next(u64::MAX).allocate_range(1),
            Some((u64::MAX, NextSymbolId::Exhausted))
        );
        assert_eq!(NextSymbolId::Next(u64::MAX).allocate_range(2), None);
        assert_eq!(NextSymbolId::Exhausted.allocate_range(1), None);
    }

    #[test]
    fn normalized_batches_compact_raw_and_canonical_duplicates_in_input_order() {
        let repeated = "  ZETA\t".to_owned();
        let values = vec![
            repeated.clone(),
            repeated.clone(),
            "ＺＥＴＡ".to_owned(),
            "alpha".to_owned(),
            "ALPHA".to_owned(),
            "zeta".to_owned(),
            "Beta".to_owned(),
            repeated,
        ];

        assert_eq!(
            normalize_batch(&values).unwrap(),
            NormalizedBatch {
                unique_values: vec!["zeta".to_owned(), "alpha".to_owned(), "beta".to_owned()],
                positions: vec![0, 0, 0, 1, 1, 0, 2, 0],
            }
        );
    }
}
