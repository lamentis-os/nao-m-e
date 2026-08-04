use std::collections::{BTreeMap, BTreeSet};

use nao_m_e::{EpisodeAtom, EpisodeDraft, Memory, PredicateId, Statement, TermId};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OptionalExtension, Row, Transaction, params_from_iter};
use unicode_normalization::UnicodeNormalization;

use crate::error::{StoreError, StoreIntegrityError};
use crate::format;

use super::{SqliteStore, read_metadata, read_u64};

const MAX_SYMBOL_QUERY_BINDINGS: usize = 900;

const _: () = {
    assert!(unicode_case_mapping::UNICODE_VERSION.0 == 16);
    assert!(unicode_case_mapping::UNICODE_VERSION.1 == 0);
    assert!(unicode_case_mapping::UNICODE_VERSION.2 == 0);
    assert!(unicode_normalization::UNICODE_VERSION.0 == 16);
    assert!(unicode_normalization::UNICODE_VERSION.1 == 0);
    assert!(unicode_normalization::UNICODE_VERSION.2 == 0);
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SymbolNamespace {
    Predicate,
    Term,
}

impl SymbolNamespace {
    const fn name(self) -> &'static str {
        match self {
            Self::Predicate => "predicate",
            Self::Term => "term",
        }
    }

    const fn table(self) -> &'static str {
        match self {
            Self::Predicate => "predicates",
            Self::Term => "terms",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NextSymbolId {
    Next(u64),
    Exhausted,
}

impl NextSymbolId {
    fn allocate(self, count: usize) -> Option<(Vec<u64>, Self)> {
        let mut next = self;
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            let Self::Next(id) = next else {
                return None;
            };
            ids.push(id);
            next = id.checked_add(1).map_or(Self::Exhausted, Self::Next);
        }
        Some((ids, next))
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

    fn contains_persisted(&self, id: u64) -> bool {
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
    /// Normalizes and interns predicate values in input order.
    ///
    /// Duplicate normalized values return the same identifier. New assignments
    /// remain staged in this store until [`Self::save`] commits them atomically
    /// with pending episode and feedback changes. An invalid value or exhausted
    /// identifier space leaves the complete staged symbol state unchanged.
    pub fn intern_predicates(&mut self, values: &[String]) -> Result<Vec<PredicateId>, StoreError> {
        self.intern_symbols(SymbolNamespace::Predicate, values)
            .map(|ids| ids.into_iter().map(PredicateId::new).collect())
    }

    /// Normalizes and interns term values in input order.
    ///
    /// Duplicate normalized values return the same identifier. New assignments
    /// remain staged in this store until [`Self::save`] commits them atomically
    /// with pending episode and feedback changes. An invalid value or exhausted
    /// identifier space leaves the complete staged symbol state unchanged.
    pub fn intern_terms(&mut self, values: &[String]) -> Result<Vec<TermId>, StoreError> {
        self.intern_symbols(SymbolNamespace::Term, values)
            .map(|ids| ids.into_iter().map(TermId::new).collect())
    }

    /// Resolves predicate identifiers in input order.
    ///
    /// Values staged by [`Self::intern_predicates`] are visible before a save.
    /// Only the requested values are loaded from SQLite. Unknown identifiers
    /// produce `None` at their input position.
    pub fn predicate_values(&self, ids: &[PredicateId]) -> Result<Vec<Option<String>>, StoreError> {
        let ids: Vec<_> = ids.iter().map(|id| id.get()).collect();
        self.symbol_values(SymbolNamespace::Predicate, &ids)
    }

    /// Resolves term identifiers in input order.
    ///
    /// Values staged by [`Self::intern_terms`] are visible before a save. Only
    /// the requested values are loaded from SQLite. Unknown identifiers
    /// produce `None` at their input position.
    pub fn term_values(&self, ids: &[TermId]) -> Result<Vec<Option<String>>, StoreError> {
        let ids: Vec<_> = ids.iter().map(|id| id.get()).collect();
        self.symbol_values(SymbolNamespace::Term, &ids)
    }

    fn intern_symbols(
        &mut self,
        namespace: SymbolNamespace,
        values: &[String],
    ) -> Result<Vec<u64>, StoreError> {
        let NormalizedBatch {
            mut unique_values,
            mut positions,
        } = normalize_batch(namespace, values)?;
        if unique_values.is_empty() {
            return Ok(Vec::new());
        }

        let state = self.symbol_state(namespace);
        let mut assignments = vec![None; unique_values.len()];
        let mut unresolved_indexes = Vec::new();
        for (index, value) in unique_values.iter().enumerate() {
            if let Some(&id) = state.pending.get(value) {
                assignments[index] = Some(id);
            } else {
                unresolved_indexes.push(index);
            }
        }

        let unresolved: Vec<_> = unresolved_indexes
            .iter()
            .map(|&index| unique_values[index].as_str())
            .collect();
        let persisted = read_symbol_ids_for_values(&self.connection, namespace, &unresolved)?;
        let mut new_indexes = Vec::new();
        for index in unresolved_indexes {
            let value = unique_values[index].as_str();
            if let Some(&id) = persisted.get(value) {
                if !state.contains_persisted(id) {
                    let (_, actual_revision) = read_metadata(&self.connection)?;
                    if actual_revision != self.expected_revision {
                        return Err(StoreError::ConcurrentModification {
                            expected_revision: self.expected_revision,
                            actual_revision,
                        });
                    }
                    return Err(StoreIntegrityError::InvalidSymbol {
                        namespace: namespace.name(),
                        id,
                        detail: "symbol row changed outside this store session",
                    }
                    .into());
                }
                assignments[index] = Some(id);
            } else {
                new_indexes.push(index);
            }
        }

        let (new_ids, next_id) =
            state
                .next_id
                .allocate(new_indexes.len())
                .ok_or(StoreError::SymbolIdExhausted {
                    namespace: namespace.name(),
                })?;
        for (&index, &id) in new_indexes.iter().zip(&new_ids) {
            assignments[index] = Some(id);
        }

        for position in &mut positions {
            let index = usize::try_from(*position)
                .expect("a normalized batch position was produced from a usize");
            *position = assignments[index].expect("every normalized symbol was assigned");
        }

        let state = self.symbol_state_mut(namespace);
        for (index, id) in new_indexes.into_iter().zip(new_ids) {
            let value = std::mem::take(&mut unique_values[index]);
            let previous = state.pending.insert(value, id);
            debug_assert!(previous.is_none());
        }
        state.next_id = next_id;
        Ok(positions)
    }

    fn symbol_values(
        &self,
        namespace: SymbolNamespace,
        ids: &[u64],
    ) -> Result<Vec<Option<String>>, StoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let state = self.symbol_state(namespace);
        let pending_by_id: BTreeMap<_, _> = state
            .pending
            .iter()
            .map(|(value, &id)| (id, value.as_str()))
            .collect();
        let mut persisted_ids = Vec::new();
        let mut seen = BTreeSet::new();
        for &id in ids {
            if pending_by_id.contains_key(&id) {
                continue;
            }
            if !state.contains_persisted(id) {
                continue;
            }
            if seen.insert(id) {
                persisted_ids.push(id);
            }
        }
        let persisted = read_symbol_values_for_ids(&self.connection, namespace, &persisted_ids)?;
        ids.iter()
            .map(|id| {
                if let Some(value) = pending_by_id.get(id) {
                    return Ok(Some((*value).to_owned()));
                }
                if !state.contains_persisted(*id) {
                    return Ok(None);
                }
                persisted.get(id).cloned().map(Some).ok_or_else(|| {
                    StoreIntegrityError::InvalidSymbol {
                        namespace: namespace.name(),
                        id: *id,
                        detail: "symbol row is absent",
                    }
                    .into()
                })
            })
            .collect()
    }

    const fn symbol_state(&self, namespace: SymbolNamespace) -> &SymbolState {
        match namespace {
            SymbolNamespace::Predicate => &self.predicates,
            SymbolNamespace::Term => &self.terms,
        }
    }

    const fn symbol_state_mut(&mut self, namespace: SymbolNamespace) -> &mut SymbolState {
        match namespace {
            SymbolNamespace::Predicate => &mut self.predicates,
            SymbolNamespace::Term => &mut self.terms,
        }
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
    positions: Vec<u64>,
}

fn normalize_batch(
    namespace: SymbolNamespace,
    values: &[String],
) -> Result<NormalizedBatch, StoreError> {
    let mut raw_positions = BTreeMap::<&str, usize>::new();
    let mut canonical_positions = BTreeMap::<String, usize>::new();
    let mut positions = Vec::with_capacity(values.len());

    for (index, value) in values.iter().enumerate() {
        if let Some(&position) = raw_positions.get(value.as_str()) {
            positions.push(u64::try_from(position).expect("a slice index fits in a u64"));
            continue;
        }

        let normalized =
            normalize_symbol(value).map_err(|error| StoreError::InvalidSymbolValue {
                namespace: namespace.name(),
                index,
                detail: error.detail(),
            })?;
        let next_position = canonical_positions.len();
        let position = *canonical_positions
            .entry(normalized)
            .or_insert(next_position);
        raw_positions.insert(value.as_str(), position);
        positions.push(u64::try_from(position).expect("a slice index fits in a u64"));
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

pub(super) fn validate_symbol_catalog(
    connection: &Connection,
    namespace: SymbolNamespace,
) -> Result<SymbolState, StoreError> {
    let sql = format!("SELECT id, value FROM {} ORDER BY id", namespace.table());
    let mut statement = connection.prepare(&sql)?;
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
        let (id, _) = read_symbol_row(row, namespace)?;
        if id != expected {
            return Err(StoreIntegrityError::NonContiguousSymbolId {
                namespace: namespace.name(),
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

fn read_symbol_row(row: &Row<'_>, namespace: SymbolNamespace) -> Result<(u64, String), StoreError> {
    let id = read_u64(row, 0, namespace.table(), "id")?;
    let ValueRef::Text(bytes) = row.get_ref(1)? else {
        return Err(StoreIntegrityError::InvalidEncoding {
            table: namespace.table(),
            column: "value",
        }
        .into());
    };
    let value = std::str::from_utf8(bytes).map_err(|_| StoreIntegrityError::InvalidEncoding {
        table: namespace.table(),
        column: "value",
    })?;
    let canonical =
        normalize_symbol(value).map_err(|error| StoreIntegrityError::InvalidSymbol {
            namespace: namespace.name(),
            id,
            detail: error.detail(),
        })?;
    if canonical != value {
        return Err(StoreIntegrityError::InvalidSymbol {
            namespace: namespace.name(),
            id,
            detail: "value is not normalized",
        }
        .into());
    }
    Ok((id, canonical))
}

fn placeholders(count: usize) -> String {
    std::iter::repeat_n("?", count)
        .collect::<Vec<_>>()
        .join(", ")
}

fn read_symbol_ids_for_values(
    connection: &Connection,
    namespace: SymbolNamespace,
    values: &[&str],
) -> Result<BTreeMap<String, u64>, StoreError> {
    let mut found = BTreeMap::new();
    for chunk in values.chunks(MAX_SYMBOL_QUERY_BINDINGS) {
        if chunk.is_empty() {
            continue;
        }
        let sql = format!(
            "SELECT id, value FROM {} WHERE value IN ({}) ORDER BY id",
            namespace.table(),
            placeholders(chunk.len())
        );
        let mut statement = connection.prepare(&sql)?;
        let mut rows = statement.query(params_from_iter(chunk.iter().copied()))?;
        while let Some(row) = rows.next()? {
            let (id, value) = read_symbol_row(row, namespace)?;
            if found.insert(value, id).is_some() {
                return Err(StoreIntegrityError::InvalidMetadata {
                    detail: "symbol value appears more than once",
                }
                .into());
            }
        }
    }
    Ok(found)
}

fn read_symbol_values_for_ids(
    connection: &Connection,
    namespace: SymbolNamespace,
    ids: &[u64],
) -> Result<BTreeMap<u64, String>, StoreError> {
    let mut found = BTreeMap::new();
    for chunk in ids.chunks(MAX_SYMBOL_QUERY_BINDINGS) {
        if chunk.is_empty() {
            continue;
        }
        let encoded: Vec<_> = chunk.iter().copied().map(format::encode_u64).collect();
        let sql = format!(
            "SELECT id, value FROM {} WHERE id IN ({}) ORDER BY id",
            namespace.table(),
            placeholders(chunk.len())
        );
        let mut statement = connection.prepare(&sql)?;
        let mut rows = statement.query(params_from_iter(
            encoded.iter().map(|value| value.as_slice()),
        ))?;
        while let Some(row) = rows.next()? {
            let (id, value) = read_symbol_row(row, namespace)?;
            if found.insert(id, value).is_some() {
                return Err(StoreIntegrityError::InvalidMetadata {
                    detail: "symbol identifier appears more than once",
                }
                .into());
            }
        }
    }
    Ok(found)
}

fn draft_statements(draft: &EpisodeDraft) -> impl Iterator<Item = &Statement> {
    draft
        .context
        .iter()
        .chain(std::iter::once(&draft.observation))
        .chain(draft.action.iter())
        .chain(draft.outcome.iter())
}

fn atom_statements(atom: &EpisodeAtom) -> impl Iterator<Item = &Statement> {
    atom.context()
        .iter()
        .chain(std::iter::once(atom.observation()))
        .chain(atom.action())
        .chain(atom.outcome())
}

pub(super) fn validate_persisted_episode_symbols(
    sequence: u64,
    draft: &EpisodeDraft,
    predicates: &SymbolState,
    terms: &SymbolState,
) -> Result<(), StoreError> {
    for statement in draft_statements(draft) {
        if !predicates.contains_persisted(statement.predicate().get()) {
            return Err(StoreIntegrityError::InvalidEpisode {
                sequence,
                detail: "predicate identifier is absent from the symbol catalog",
            }
            .into());
        }
        if statement
            .arguments()
            .iter()
            .any(|term| !terms.contains_persisted(term.get()))
        {
            return Err(StoreIntegrityError::InvalidEpisode {
                sequence,
                detail: "term identifier is absent from the symbol catalog",
            }
            .into());
        }
    }
    Ok(())
}

pub(super) fn validate_new_episode_symbols(
    memory: &Memory,
    start: usize,
    predicates: &SymbolState,
    terms: &SymbolState,
) -> Result<(), StoreError> {
    for atom in memory.episodes().skip(start) {
        for statement in atom_statements(atom) {
            let predicate = statement.predicate().get();
            if !predicates.contains_current(predicate) {
                return Err(StoreError::UnknownSymbolId {
                    namespace: SymbolNamespace::Predicate.name(),
                    id: predicate,
                });
            }
            if let Some(term) = statement
                .arguments()
                .iter()
                .map(|term| term.get())
                .find(|&term| !terms.contains_current(term))
            {
                return Err(StoreError::UnknownSymbolId {
                    namespace: SymbolNamespace::Term.name(),
                    id: term,
                });
            }
        }
    }
    Ok(())
}

pub(super) fn verify_symbol_tail(
    transaction: &Transaction<'_>,
    namespace: SymbolNamespace,
    state: &SymbolState,
) -> Result<(), StoreError> {
    let sql = format!(
        "SELECT id FROM {} ORDER BY id DESC LIMIT 1",
        namespace.table()
    );
    let tail: Option<Vec<u8>> = transaction
        .query_row(&sql, [], |row| row.get(0))
        .optional()?;
    let actual_tail = tail
        .as_deref()
        .map(|bytes| {
            format::decode_u64(bytes).ok_or(StoreIntegrityError::InvalidEncoding {
                table: namespace.table(),
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
    namespace: SymbolNamespace,
    state: &SymbolState,
) -> Result<(), StoreError> {
    if state.pending.is_empty() {
        return Ok(());
    }
    let mut pending: Vec<_> = state
        .pending
        .iter()
        .map(|(value, &id)| (id, value))
        .collect();
    pending.sort_unstable_by_key(|(id, _)| *id);
    let sql = format!(
        "INSERT INTO {} (id, value) VALUES (?1, ?2)",
        namespace.table()
    );
    let mut insert = transaction.prepare(&sql)?;
    for (id, value) in pending {
        let id = format::encode_u64(id);
        insert.execute((id.as_slice(), value))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_preserves_the_complete_u64_range_atomically() {
        assert_eq!(
            NextSymbolId::Next(u64::MAX).allocate(0),
            Some((Vec::new(), NextSymbolId::Next(u64::MAX)))
        );
        assert_eq!(
            NextSymbolId::Next(u64::MAX).allocate(1),
            Some((vec![u64::MAX], NextSymbolId::Exhausted))
        );
        assert_eq!(NextSymbolId::Next(u64::MAX).allocate(2), None);
        assert_eq!(
            NextSymbolId::Exhausted.allocate(0),
            Some((Vec::new(), NextSymbolId::Exhausted))
        );
        assert_eq!(NextSymbolId::Exhausted.allocate(1), None);
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
            normalize_batch(SymbolNamespace::Term, &values).unwrap(),
            NormalizedBatch {
                unique_values: vec!["zeta".to_owned(), "alpha".to_owned(), "beta".to_owned()],
                positions: vec![0, 0, 0, 1, 1, 0, 2, 0],
            }
        );
    }
}
