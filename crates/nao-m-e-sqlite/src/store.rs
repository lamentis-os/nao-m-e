use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use nao_m_e::{
    AtomId, EpisodeAtom, EpisodeDraft, FeedbackTrace, Memory, MemoryId, PredicateId, Statement,
    TermId,
};
use rusqlite::types::ValueRef;
use rusqlite::{
    Connection, OpenFlags, OptionalExtension, Row, Rows, Transaction, TransactionBehavior,
    params_from_iter,
};
use unicode_normalization::UnicodeNormalization;

use crate::codec;
use crate::error::{StoreError, StoreIntegrityError};
use crate::schema;

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
enum SymbolNamespace {
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
struct SymbolState {
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

    fn mark_saved(&mut self) {
        self.persisted_tail = self.next_id.tail();
        self.pending.clear();
    }
}

/// An explicitly saved SQLite database and its owned in-memory state.
///
/// A store represents one logical memory. Mutating [`Self::memory_mut`] does
/// not write to disk; call [`Self::save`] to atomically persist the changes.
pub struct SqliteStore {
    connection: Connection,
    memory: Memory,
    persisted_episode_count: usize,
    expected_revision: i64,
    predicates: SymbolState,
    terms: SymbolState,
}

impl SqliteStore {
    /// Creates a new empty SQLite memory store at `path`.
    ///
    /// The operation fails rather than opening or replacing an existing file.
    /// A non-zero memory identifier is generated from operating-system entropy
    /// and committed with the initial schema.
    pub fn create(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref();
        let memory_id = random_memory_id()?;
        let staging = build_initial_database(path, memory_id)?;
        let published = staging
            .persist_noclobber(path)
            .map_err(|error| StoreError::Io(error.error))?;
        drop(published);
        Self::open(path)
    }

    /// Opens and validates an existing SQLite memory store.
    ///
    /// Missing files are not created. Invalid and unsupported stores are
    /// rejected without returning a partial memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        let mut connection = Connection::open_with_flags(path, flags)?;
        schema::configure_session(&connection)?;
        verify_application_id(&connection)?;
        verify_format_version(&connection)?;
        schema::verify_durability(&connection)?;

        let loaded = load_memory(&mut connection)?;
        Ok(Self {
            connection,
            memory: loaded.memory,
            persisted_episode_count: loaded.episode_count,
            expected_revision: loaded.revision,
            predicates: loaded.predicates,
            terms: loaded.terms,
        })
    }

    /// Returns the durable identifier of the owned logical memory.
    #[must_use]
    pub const fn memory_id(&self) -> MemoryId {
        self.memory.memory_id()
    }

    /// Returns the owned memory for read-only operations.
    #[must_use]
    pub const fn memory(&self) -> &Memory {
        &self.memory
    }

    /// Returns the owned memory for mutations that remain unsaved until
    /// [`Self::save`] succeeds.
    #[must_use]
    pub const fn memory_mut(&mut self) -> &mut Memory {
        &mut self.memory
    }

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

    /// Atomically persists newly appended episodes and feedback changes.
    ///
    /// A stale store session is rejected instead of overwriting a later
    /// snapshot. On every error, the previous database snapshot remains
    /// committed while the in-memory changes remain available to the caller.
    pub fn save(&mut self) -> Result<(), StoreError> {
        let Self {
            connection,
            memory,
            persisted_episode_count,
            expected_revision,
            predicates,
            terms,
        } = self;

        let episode_count = memory.episodes().len();
        if *persisted_episode_count > episode_count {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "persisted episode count exceeds in-memory episode count",
            }
            .into());
        }

        schema::verify_durability(connection)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        verify_application_id(&transaction)?;
        let (actual_memory_id, actual_revision) = read_metadata(&transaction)?;
        verify_schema(&transaction)?;
        if actual_memory_id != memory.memory_id() {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "persisted memory ID differs from the owned memory",
            }
            .into());
        }
        if actual_revision != *expected_revision {
            return Err(StoreError::ConcurrentModification {
                expected_revision: *expected_revision,
                actual_revision,
            });
        }
        if actual_revision == i64::MAX {
            return Err(StoreError::RevisionExhausted);
        }
        verify_persisted_tail(&transaction, *persisted_episode_count)?;
        verify_symbol_tail(&transaction, SymbolNamespace::Predicate, predicates)?;
        verify_symbol_tail(&transaction, SymbolNamespace::Term, terms)?;
        validate_new_episode_symbols(memory, *persisted_episode_count, predicates, terms)?;

        let next_revision = actual_revision + 1;
        let changed = transaction.execute(
            "UPDATE memory_meta SET snapshot_revision = ?1 WHERE singleton = 1",
            [next_revision],
        )?;
        if changed != 1 {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "metadata singleton changed during the save transaction",
            }
            .into());
        }

        insert_pending_symbols(&transaction, SymbolNamespace::Predicate, predicates)?;
        insert_pending_symbols(&transaction, SymbolNamespace::Term, terms)?;
        append_episodes(&transaction, memory, *persisted_episode_count)?;
        reconcile_feedback(&transaction, memory, *persisted_episode_count)?;
        transaction.commit()?;

        *expected_revision = next_revision;
        *persisted_episode_count = episode_count;
        predicates.mark_saved();
        terms.mark_saved();
        Ok(())
    }
}

fn random_memory_id() -> Result<MemoryId, StoreError> {
    loop {
        let mut bytes = [0_u8; 16];
        getrandom::fill(&mut bytes)?;
        if let Ok(memory_id) = MemoryId::from_be_bytes(bytes) {
            return Ok(memory_id);
        }
    }
}

fn build_initial_database(
    path: &Path,
    memory_id: MemoryId,
) -> Result<tempfile::NamedTempFile, StoreError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let staging = tempfile::Builder::new()
        .prefix(".nao-m-e-")
        .suffix(".sqlite.tmp")
        .tempfile_in(parent)?;

    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE;
    let mut connection = Connection::open_with_flags(staging.path(), flags)?;
    schema::configure_session(&connection)?;
    schema::configure_durability(&connection)?;
    schema::create_schema(&mut connection, memory_id)?;
    let loaded = load_memory(&mut connection)?;
    if loaded.memory.memory_id() != memory_id
        || loaded.episode_count != 0
        || loaded.revision != 0
        || loaded.predicates.persisted_tail.is_some()
        || loaded.terms.persisted_tail.is_some()
    {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "new store differs from its initialized empty snapshot",
        }
        .into());
    }
    connection
        .close()
        .map_err(|(_, error)| StoreError::Database(error))?;
    staging.as_file().sync_all()?;
    Ok(staging)
}

struct LoadedStore {
    memory: Memory,
    episode_count: usize,
    revision: i64,
    predicates: SymbolState,
    terms: SymbolState,
}

fn load_memory(connection: &mut Connection) -> Result<LoadedStore, StoreError> {
    let transaction = connection.transaction()?;
    verify_application_id(&transaction)?;
    let (memory_id, revision) = read_metadata(&transaction)?;
    verify_schema(&transaction)?;
    verify_quick_check(&transaction)?;
    let predicates = validate_symbol_catalog(&transaction, SymbolNamespace::Predicate)?;
    let terms = validate_symbol_catalog(&transaction, SymbolNamespace::Term)?;
    let mut memory = reconstruct_memory(&transaction, memory_id, &predicates, &terms)?;
    restore_feedback(&transaction, &mut memory)?;
    let episode_count = memory.episodes().len();
    transaction.commit()?;
    Ok(LoadedStore {
        memory,
        episode_count,
        revision,
        predicates,
        terms,
    })
}

fn verify_application_id(connection: &Connection) -> Result<(), StoreError> {
    let found = schema::read_application_id(connection)?;
    if found == schema::APPLICATION_ID {
        Ok(())
    } else {
        Err(StoreIntegrityError::ApplicationMismatch { found }.into())
    }
}

fn verify_format_version(connection: &Connection) -> Result<(), StoreError> {
    let mut statement = connection.prepare("SELECT format_version FROM memory_meta")?;
    let mut rows = statement.query([])?;
    let Some(row) = rows.next()? else {
        return Err(StoreIntegrityError::MissingMetadata.into());
    };
    let found = read_integer(row, 0, "memory_meta", "format_version")?;
    if found != schema::FORMAT_VERSION {
        return Err(StoreIntegrityError::UnsupportedFormatVersion { found }.into());
    }
    if rows.next()?.is_some() {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "multiple metadata rows",
        }
        .into());
    }
    Ok(())
}

fn verify_schema(connection: &Connection) -> Result<(), StoreError> {
    if schema::validate_schema(connection)? {
        Ok(())
    } else {
        Err(StoreIntegrityError::InvalidMetadata {
            detail: "database schema differs from the SQLite contract",
        }
        .into())
    }
}

fn verify_quick_check(connection: &Connection) -> Result<(), StoreError> {
    let mut statement = connection.prepare("PRAGMA quick_check")?;
    let mut rows = statement.query([])?;
    let mut diagnostics = Vec::new();
    while let Some(row) = rows.next()? {
        diagnostics.push(row.get::<_, String>(0)?);
    }
    if diagnostics.len() == 1 && diagnostics[0] == "ok" {
        return Ok(());
    }
    Err(StoreIntegrityError::QuickCheckFailed {
        detail: diagnostics.join("; "),
    }
    .into())
}

fn read_metadata(connection: &Connection) -> Result<(MemoryId, i64), StoreError> {
    let mut statement = connection.prepare(
        "SELECT singleton, format_version, memory_id, snapshot_revision
         FROM memory_meta",
    )?;
    let mut rows = statement.query([])?;
    let Some(row) = rows.next()? else {
        return Err(StoreIntegrityError::MissingMetadata.into());
    };
    let singleton = read_integer(row, 0, "memory_meta", "singleton")?;
    if singleton != 1 {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "metadata singleton key is not one",
        }
        .into());
    }
    let format_version = read_integer(row, 1, "memory_meta", "format_version")?;
    if format_version != schema::FORMAT_VERSION {
        return Err(StoreIntegrityError::UnsupportedFormatVersion {
            found: format_version,
        }
        .into());
    }
    let memory_id = read_memory_id(row, 2, "memory_meta", "memory_id")?;
    let revision = read_integer(row, 3, "memory_meta", "snapshot_revision")?;
    if rows.next()?.is_some() {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "multiple metadata rows",
        }
        .into());
    }
    if revision < 0 {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "negative snapshot revision",
        }
        .into());
    }
    Ok((memory_id, revision))
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
    let mut normalized = String::with_capacity(value.len().min(schema::MAX_SYMBOL_BYTES));
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
        if normalized.len() > schema::MAX_SYMBOL_BYTES {
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

fn validate_symbol_catalog(
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
        let encoded: Vec<_> = chunk.iter().copied().map(codec::encode_u64).collect();
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

fn validate_persisted_episode_symbols(
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

fn validate_new_episode_symbols(
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

fn reconstruct_memory(
    connection: &Connection,
    memory_id: MemoryId,
    predicates: &SymbolState,
    terms: &SymbolState,
) -> Result<Memory, StoreError> {
    let mut statement = connection.prepare(
        "SELECT sequence, payload
         FROM episodes
         ORDER BY sequence",
    )?;
    let mut rows = statement.query([])?;
    let mut memory = Memory::new(memory_id);
    let mut expected_sequence = 0_u64;

    while let Some(row) = rows.next()? {
        let sequence = read_u64(row, 0, "episodes", "sequence")?;
        if sequence != expected_sequence {
            return Err(StoreIntegrityError::NonContiguousEpisodeSequence {
                expected: expected_sequence,
                found: sequence,
            }
            .into());
        }
        let ValueRef::Blob(payload) = row.get_ref(1)? else {
            return Err(StoreIntegrityError::InvalidEncoding {
                table: "episodes",
                column: "payload",
            }
            .into());
        };
        let draft = codec::decode_episode(payload).map_err(|error| {
            StoreIntegrityError::InvalidEpisode {
                sequence,
                detail: error.detail(),
            }
        })?;
        validate_persisted_episode_symbols(sequence, &draft, predicates, terms)?;
        let id = memory
            .insert_episode(draft)
            .map_err(|_| StoreIntegrityError::InvalidEpisode {
                sequence,
                detail: "episode sequence cannot be reconstructed",
            })?;
        if id != AtomId::from_parts(memory_id, sequence) {
            return Err(StoreIntegrityError::InvalidEpisode {
                sequence,
                detail: "core assigned a different atom ID",
            }
            .into());
        }
        expected_sequence =
            expected_sequence
                .checked_add(1)
                .ok_or(StoreIntegrityError::InvalidMetadata {
                    detail: "episode sequence space is exhausted",
                })?;
    }
    Ok(memory)
}

fn restore_feedback(connection: &Connection, memory: &mut Memory) -> Result<(), StoreError> {
    let mut statement = connection.prepare(
        "SELECT from_sequence, to_sequence, history_bits, sample_count
         FROM feedback_edges
         ORDER BY from_sequence, to_sequence",
    )?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let edge = read_feedback_edge(row)?;

        let from_id = AtomId::from_parts(memory.memory_id(), edge.from);
        let to_id = AtomId::from_parts(memory.memory_id(), edge.to);
        memory
            .set_feedback_trace(from_id, to_id, edge.trace)
            .map_err(|_| StoreIntegrityError::InvalidFeedback {
                from: edge.from,
                to: edge.to,
                detail: "feedback edge violates core graph invariants",
            })?;
    }
    Ok(())
}

fn verify_persisted_tail(
    transaction: &Transaction<'_>,
    persisted_episode_count: usize,
) -> Result<(), StoreError> {
    let tail: Option<Vec<u8>> = transaction
        .query_row(
            "SELECT sequence FROM episodes ORDER BY sequence DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let expected_tail = persisted_episode_count.checked_sub(1).map(|index| {
        u64::try_from(index).expect("a persisted episode count always fits an atom sequence")
    });
    let actual_tail = tail
        .as_deref()
        .map(|bytes| {
            codec::decode_u64(bytes).ok_or(StoreIntegrityError::InvalidEncoding {
                table: "episodes",
                column: "sequence",
            })
        })
        .transpose()?;
    if actual_tail == expected_tail {
        Ok(())
    } else {
        Err(StoreIntegrityError::InvalidMetadata {
            detail: "persisted episode tail changed outside this store session",
        }
        .into())
    }
}

fn verify_symbol_tail(
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
            codec::decode_u64(bytes).ok_or(StoreIntegrityError::InvalidEncoding {
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

fn insert_pending_symbols(
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
        let id = codec::encode_u64(id);
        insert.execute((id.as_slice(), value))?;
    }
    Ok(())
}

fn append_episodes(
    transaction: &Transaction<'_>,
    memory: &Memory,
    start: usize,
) -> Result<(), StoreError> {
    let mut insert = transaction.prepare(
        "INSERT INTO episodes (sequence, payload)
         VALUES (?1, ?2)",
    )?;
    for episode in memory.episodes().skip(start) {
        let sequence = codec::encode_u64(episode.id().sequence());
        let payload = codec::encode_episode(episode);
        insert.execute((sequence.as_slice(), payload.as_slice()))?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FeedbackRecord {
    from: u64,
    to: u64,
    trace: FeedbackTrace,
}

impl FeedbackRecord {
    const fn key(self) -> (u64, u64) {
        (self.from, self.to)
    }
}

const MAX_BUFFERED_FEEDBACK_MUTATIONS: usize = 16_384;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeedbackMutation {
    Delete(FeedbackRecord),
    Insert(FeedbackRecord),
    Update(FeedbackRecord),
}

#[derive(Debug, Default)]
struct FeedbackPlan {
    mutations: Vec<FeedbackMutation>,
    replace_all: bool,
}

impl FeedbackPlan {
    fn push(&mut self, mutation: FeedbackMutation) {
        if self.replace_all {
            return;
        }
        if self.mutations.len() == MAX_BUFFERED_FEEDBACK_MUTATIONS {
            self.mutations.clear();
            self.replace_all = true;
            return;
        }
        self.mutations.push(mutation);
    }
}

fn reconcile_feedback(
    transaction: &Transaction<'_>,
    memory: &Memory,
    persisted_episode_count: usize,
) -> Result<(), StoreError> {
    let mut plan = FeedbackPlan::default();

    {
        let mut statement = transaction.prepare(
            "SELECT from_sequence, to_sequence, history_bits, sample_count
             FROM feedback_edges
             ORDER BY from_sequence, to_sequence",
        )?;
        let mut rows = statement.query([])?;
        let persisted_episode_count = u64::try_from(persisted_episode_count)
            .expect("an episode count always fits the atom sequence space");
        let mut persisted = next_feedback_edge(&mut rows, persisted_episode_count)?;
        let mut current = memory_feedback_records(memory);
        let mut expected = current.next();

        loop {
            if plan.replace_all {
                while persisted.is_some() {
                    persisted = next_feedback_edge(&mut rows, persisted_episode_count)?;
                }
                break;
            }
            match (persisted, expected) {
                (Some(found), Some(wanted)) => match found.key().cmp(&wanted.key()) {
                    Ordering::Less => {
                        plan.push(FeedbackMutation::Delete(found));
                        persisted = next_feedback_edge(&mut rows, persisted_episode_count)?;
                    }
                    Ordering::Equal => {
                        if found.trace != wanted.trace {
                            plan.push(FeedbackMutation::Update(wanted));
                        }
                        persisted = next_feedback_edge(&mut rows, persisted_episode_count)?;
                        expected = current.next();
                    }
                    Ordering::Greater => {
                        plan.push(FeedbackMutation::Insert(wanted));
                        expected = current.next();
                    }
                },
                (Some(found), None) => {
                    plan.push(FeedbackMutation::Delete(found));
                    persisted = next_feedback_edge(&mut rows, persisted_episode_count)?;
                }
                (None, Some(wanted)) => {
                    plan.push(FeedbackMutation::Insert(wanted));
                    expected = current.next();
                }
                (None, None) => break,
            }
        }
    }

    if plan.replace_all {
        transaction.execute("DELETE FROM feedback_edges", [])?;
        let mut insert = transaction.prepare(
            "INSERT INTO feedback_edges (
                from_sequence,
                to_sequence,
                history_bits,
                sample_count
             ) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for edge in memory_feedback_records(memory) {
            let from = codec::encode_u64(edge.from);
            let to = codec::encode_u64(edge.to);
            insert.execute((
                from.as_slice(),
                to.as_slice(),
                i64::from(edge.trace.history_bits()),
                i64::from(edge.trace.sample_count()),
            ))?;
        }
        return Ok(());
    }

    if plan.mutations.is_empty() {
        return Ok(());
    }

    let mut delete = transaction.prepare(
        "DELETE FROM feedback_edges
         WHERE from_sequence = ?1 AND to_sequence = ?2",
    )?;
    let mut insert = transaction.prepare(
        "INSERT INTO feedback_edges (
            from_sequence,
            to_sequence,
            history_bits,
            sample_count
         ) VALUES (?1, ?2, ?3, ?4)",
    )?;
    let mut update = transaction.prepare(
        "UPDATE feedback_edges SET history_bits = ?3, sample_count = ?4
         WHERE from_sequence = ?1 AND to_sequence = ?2",
    )?;
    for mutation in plan.mutations {
        match mutation {
            FeedbackMutation::Delete(edge) => {
                let from = codec::encode_u64(edge.from);
                let to = codec::encode_u64(edge.to);
                delete.execute((from.as_slice(), to.as_slice()))?;
            }
            FeedbackMutation::Insert(edge) => {
                let from = codec::encode_u64(edge.from);
                let to = codec::encode_u64(edge.to);
                insert.execute((
                    from.as_slice(),
                    to.as_slice(),
                    i64::from(edge.trace.history_bits()),
                    i64::from(edge.trace.sample_count()),
                ))?;
            }
            FeedbackMutation::Update(edge) => {
                let from = codec::encode_u64(edge.from);
                let to = codec::encode_u64(edge.to);
                update.execute((
                    from.as_slice(),
                    to.as_slice(),
                    i64::from(edge.trace.history_bits()),
                    i64::from(edge.trace.sample_count()),
                ))?;
            }
        }
    }
    Ok(())
}

fn memory_feedback_records(memory: &Memory) -> impl Iterator<Item = FeedbackRecord> + '_ {
    memory.feedback_edges().map(|edge| FeedbackRecord {
        from: edge.from().sequence(),
        to: edge.to().sequence(),
        trace: edge.trace(),
    })
}

fn next_feedback_edge(
    rows: &mut Rows<'_>,
    persisted_episode_count: u64,
) -> Result<Option<FeedbackRecord>, StoreError> {
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    let edge = read_feedback_edge(row)?;
    if edge.from >= persisted_episode_count || edge.to >= persisted_episode_count {
        return Err(StoreIntegrityError::InvalidFeedback {
            from: edge.from,
            to: edge.to,
            detail: "edge endpoint is absent",
        }
        .into());
    }
    Ok(Some(edge))
}

fn read_feedback_edge(row: &Row<'_>) -> Result<FeedbackRecord, StoreError> {
    let from = read_u64(row, 0, "feedback_edges", "from_sequence")?;
    let to = read_u64(row, 1, "feedback_edges", "to_sequence")?;
    if from == to {
        return Err(StoreIntegrityError::InvalidFeedback {
            from,
            to,
            detail: "self-edge",
        }
        .into());
    }
    let history_bits = read_integer(row, 2, "feedback_edges", "history_bits")?;
    let sample_count = read_integer(row, 3, "feedback_edges", "sample_count")?;
    let trace = u16::try_from(history_bits)
        .ok()
        .zip(u8::try_from(sample_count).ok())
        .and_then(|(history_bits, sample_count)| {
            FeedbackTrace::from_parts(history_bits, sample_count)
        })
        .ok_or(StoreIntegrityError::InvalidFeedback {
            from,
            to,
            detail: "feedback trace is not canonical",
        })?;
    Ok(FeedbackRecord { from, to, trace })
}

fn read_integer(
    row: &Row<'_>,
    index: usize,
    table: &'static str,
    column: &'static str,
) -> Result<i64, StoreError> {
    match row.get_ref(index)? {
        ValueRef::Integer(value) => Ok(value),
        _ => Err(StoreIntegrityError::InvalidEncoding { table, column }.into()),
    }
}

fn read_u64(
    row: &Row<'_>,
    index: usize,
    table: &'static str,
    column: &'static str,
) -> Result<u64, StoreError> {
    let ValueRef::Blob(bytes) = row.get_ref(index)? else {
        return Err(StoreIntegrityError::InvalidEncoding { table, column }.into());
    };
    codec::decode_u64(bytes)
        .ok_or(StoreIntegrityError::InvalidEncoding { table, column })
        .map_err(Into::into)
}

fn read_memory_id(
    row: &Row<'_>,
    index: usize,
    table: &'static str,
    column: &'static str,
) -> Result<MemoryId, StoreError> {
    let ValueRef::Blob(bytes) = row.get_ref(index)? else {
        return Err(StoreIntegrityError::InvalidEncoding { table, column }.into());
    };
    codec::decode_memory_id(bytes).ok_or_else(|| {
        if bytes.len() == 16 {
            StoreIntegrityError::InvalidMemoryId.into()
        } else {
            StoreIntegrityError::InvalidEncoding { table, column }.into()
        }
    })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use nao_m_e::{EpisodeDraft, PredicateId, SourceId, Statement, TermId, TimestampMs};
    use rusqlite::{Connection, params};
    use tempfile::{TempDir, tempdir};

    use super::*;

    fn statement(predicate: u64, arguments: &[u64]) -> Statement {
        Statement::new(
            PredicateId::new(predicate),
            arguments.iter().copied().map(TermId::new).collect(),
        )
        .expect("test statement has arguments")
    }

    fn draft(seed: u64) -> EpisodeDraft {
        EpisodeDraft {
            occurred_at: TimestampMs::new(i64::try_from(seed).expect("small test seed")),
            recorded_at: TimestampMs::new(-i64::try_from(seed).expect("small test seed")),
            context: vec![statement(0, &[0])],
            observation: statement(1, &[1, 2]),
            action: Some(statement(2, &[3])),
            outcome: None,
            source: SourceId::new(40 + seed),
        }
    }

    fn insert(store: &mut SqliteStore, episode: EpisodeDraft) -> AtomId {
        store
            .intern_predicates(&[
                "context".to_owned(),
                "observation".to_owned(),
                "action".to_owned(),
            ])
            .unwrap();
        store
            .intern_terms(&[
                "context-term".to_owned(),
                "observation-left".to_owned(),
                "observation-right".to_owned(),
                "action-term".to_owned(),
            ])
            .unwrap();
        store.memory_mut().insert_episode(episode).unwrap()
    }

    fn trace(history_bits: u16, sample_count: u8) -> FeedbackTrace {
        FeedbackTrace::from_parts(history_bits, sample_count)
            .expect("test feedback trace is canonical")
    }

    fn saved_store(directory: &TempDir, episode_count: u64) -> PathBuf {
        let path = directory.path().join("memory.sqlite3");
        let mut store = SqliteStore::create(&path).expect("test store is created");
        for seed in 0..episode_count {
            insert(&mut store, draft(seed));
        }
        store.save().expect("test snapshot saves");
        drop(store);
        path
    }

    fn integrity_error(path: &Path) -> StoreIntegrityError {
        match SqliteStore::open(path) {
            Err(StoreError::InvalidStore(error)) => error,
            Err(error) => panic!("expected persisted-data error, got {error}"),
            Ok(store) => {
                drop(store);
                panic!("corrupt store unexpectedly opened")
            }
        }
    }

    #[test]
    fn symbol_cursor_preserves_the_complete_u64_range_atomically() {
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

    #[test]
    fn failed_episode_dml_rolls_back_pending_symbols_and_revision() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("memory.sqlite3");
        let mut store = SqliteStore::create(&path).unwrap();
        let predicate = store.intern_predicates(&["predicate".to_owned()]).unwrap()[0];
        let term = store.intern_terms(&["term".to_owned()]).unwrap()[0];
        store
            .memory_mut()
            .insert_episode(EpisodeDraft {
                occurred_at: TimestampMs::new(0),
                recorded_at: TimestampMs::new(0),
                context: Vec::new(),
                observation: statement(predicate.get(), &[term.get()]),
                action: None,
                outcome: None,
                source: SourceId::new(0),
            })
            .unwrap();
        store
            .connection
            .execute_batch(
                "CREATE TEMP TRIGGER abort_episode_insert
                 BEFORE INSERT ON main.episodes
                 BEGIN SELECT RAISE(ABORT, 'test abort'); END;",
            )
            .unwrap();

        assert!(store.save().is_err());
        assert_eq!(store.predicates.pending.len(), 1);
        assert_eq!(store.terms.pending.len(), 1);
        assert_eq!(store.expected_revision, 0);
        let persisted: (i64, i64, i64, i64) = store
            .connection
            .query_row(
                "SELECT snapshot_revision,
                        (SELECT count(*) FROM predicates),
                        (SELECT count(*) FROM terms),
                        (SELECT count(*) FROM episodes)
                 FROM memory_meta",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(persisted, (0, 0, 0, 0));
    }

    #[test]
    fn non_contiguous_symbol_identifiers_are_rejected() {
        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 0);
        let raw = Connection::open(&path).unwrap();
        raw.execute(
            "INSERT INTO predicates (id, value) VALUES (?1, 'gap')",
            [codec::encode_u64(1).as_slice()],
        )
        .unwrap();
        drop(raw);

        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::NonContiguousSymbolId {
                namespace: "predicate",
                expected: 0,
                found: 1
            }
        ));
    }

    #[test]
    fn metadata_and_unsupported_formats_fail_closed_with_specific_errors() {
        for unsupported in [1, 2, 3, schema::FORMAT_VERSION + 1] {
            let directory = tempdir().unwrap();
            let path = saved_store(&directory, 0);
            let raw = Connection::open(&path).unwrap();
            raw.pragma_update(None, "ignore_check_constraints", true)
                .unwrap();
            raw.execute("UPDATE memory_meta SET format_version = ?1", [unsupported])
                .unwrap();
            drop(raw);
            assert!(matches!(
                integrity_error(&path),
                StoreIntegrityError::UnsupportedFormatVersion { found } if found == unsupported
            ));
        }

        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 0);
        let raw = Connection::open(&path).unwrap();
        raw.execute("DELETE FROM memory_meta", []).unwrap();
        drop(raw);
        assert_eq!(integrity_error(&path), StoreIntegrityError::MissingMetadata);

        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 0);
        let raw = Connection::open(&path).unwrap();
        raw.pragma_update(None, "application_id", 1).unwrap();
        drop(raw);
        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::ApplicationMismatch { found: 1 }
        ));
    }

    #[test]
    fn rejected_identity_format_or_journal_mode_never_rewrites_the_file_mode() {
        for unsupported_format in [
            None,
            Some(1),
            Some(2),
            Some(3),
            Some(schema::FORMAT_VERSION),
            Some(schema::FORMAT_VERSION + 1),
        ] {
            let directory = tempdir().unwrap();
            let path = if unsupported_format.is_some() {
                saved_store(&directory, 0)
            } else {
                let path = directory.path().join("unrelated.sqlite3");
                let raw = Connection::open(&path).unwrap();
                raw.execute("CREATE TABLE unrelated (value INTEGER)", [])
                    .unwrap();
                drop(raw);
                path
            };
            let raw = Connection::open(&path).unwrap();
            if let Some(version) = unsupported_format {
                raw.pragma_update(None, "ignore_check_constraints", true)
                    .unwrap();
                raw.execute("UPDATE memory_meta SET format_version = ?1", [version])
                    .unwrap();
            }
            let mode: String = raw
                .query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))
                .unwrap();
            assert_eq!(mode.to_ascii_lowercase(), "wal");
            drop(raw);
            let before = std::fs::read(&path).unwrap();

            assert!(SqliteStore::open(&path).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), before);

            let raw = Connection::open(&path).unwrap();
            let mode: String = raw
                .pragma_query_value(None, "journal_mode", |row| row.get(0))
                .unwrap();
            assert_eq!(mode.to_ascii_lowercase(), "wal");
        }
    }

    #[test]
    fn payload_and_sequence_corruption_are_rejected() {
        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 1);
        let raw = Connection::open(&path).unwrap();
        raw.execute(
            "UPDATE episodes SET sequence = ?1",
            [codec::encode_u64(1).as_slice()],
        )
        .unwrap();
        drop(raw);
        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::NonContiguousEpisodeSequence {
                expected: 0,
                found: 1
            }
        ));

        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 1);
        let raw = Connection::open(&path).unwrap();
        raw.execute("UPDATE episodes SET payload = x'00'", [])
            .unwrap();
        drop(raw);
        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::InvalidEpisode { sequence: 0, .. }
        ));
    }

    #[test]
    fn multiple_feedback_edges_per_source_are_accepted_but_persistent_triggers_are_rejected() {
        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 3);
        let raw = Connection::open(&path).unwrap();
        for to in [1, 2] {
            raw.execute(
                "INSERT INTO feedback_edges VALUES (?1, ?2, 65535, 16)",
                params![
                    codec::encode_u64(0).as_slice(),
                    codec::encode_u64(to).as_slice()
                ],
            )
            .unwrap();
        }
        drop(raw);
        assert_eq!(
            SqliteStore::open(&path)
                .unwrap()
                .memory()
                .feedback_edges()
                .count(),
            2
        );

        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 0);
        let raw = Connection::open(&path).unwrap();
        raw.execute_batch(
            "CREATE TRIGGER mutate_revision AFTER UPDATE ON memory_meta
             BEGIN UPDATE memory_meta SET snapshot_revision = 0; END;",
        )
        .unwrap();
        drop(raw);
        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::InvalidMetadata { .. }
        ));
    }

    #[test]
    fn missing_feedback_endpoint_is_rejected_during_reconstruction() {
        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 1);
        let raw = Connection::open(&path).unwrap();
        raw.pragma_update(None, "foreign_keys", false).unwrap();
        raw.execute(
            "INSERT INTO feedback_edges VALUES (?1, ?2, 1, 1)",
            params![
                codec::encode_u64(0).as_slice(),
                codec::encode_u64(1).as_slice()
            ],
        )
        .unwrap();
        drop(raw);

        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::InvalidFeedback { from: 0, to: 1, .. }
        ));
    }

    #[test]
    fn feedback_save_writes_only_changed_rows() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("memory.sqlite3");
        let mut store = SqliteStore::create(&path).unwrap();
        let ids: Vec<_> = (0..5).map(|seed| insert(&mut store, draft(seed))).collect();
        for (from, to, initial_trace) in [
            (ids[0], ids[2], trace(1, 1)),
            (ids[1], ids[2], trace(2, 2)),
            (ids[2], ids[3], trace(3, 2)),
        ] {
            store
                .memory_mut()
                .set_feedback_trace(from, to, initial_trace)
                .unwrap();
        }
        store.save().unwrap();
        store
            .connection
            .execute_batch(
                "CREATE TEMP TABLE feedback_mutations (kind TEXT NOT NULL);
                 CREATE TEMP TRIGGER audit_feedback_insert
                 AFTER INSERT ON main.feedback_edges
                 BEGIN INSERT INTO feedback_mutations VALUES ('insert'); END;
                 CREATE TEMP TRIGGER audit_feedback_update
                 AFTER UPDATE ON main.feedback_edges
                 BEGIN INSERT INTO feedback_mutations VALUES ('update'); END;
                 CREATE TEMP TRIGGER audit_feedback_delete
                 AFTER DELETE ON main.feedback_edges
                 BEGIN INSERT INTO feedback_mutations VALUES ('delete'); END;",
            )
            .unwrap();

        let raw = Connection::open(&path).unwrap();
        raw.execute(
            "INSERT INTO feedback_edges VALUES (?1, ?2, 0, 1)",
            params![
                codec::encode_u64(ids[4].sequence()).as_slice(),
                codec::encode_u64(ids[0].sequence()).as_slice()
            ],
        )
        .unwrap();
        drop(raw);

        store
            .memory_mut()
            .set_feedback_trace(ids[0], ids[2], trace(2, 2))
            .unwrap();
        store
            .memory_mut()
            .set_feedback_trace(ids[0], ids[1], trace(0, 1))
            .unwrap();
        store
            .memory_mut()
            .set_feedback_trace(ids[3], ids[0], trace(1, 1))
            .unwrap();
        store.save().unwrap();

        let mutations: Vec<(String, i64)> = store
            .connection
            .prepare(
                "SELECT kind, count(*)
                 FROM feedback_mutations
                 GROUP BY kind
                 ORDER BY kind",
            )
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            mutations,
            [
                ("delete".to_owned(), 1),
                ("insert".to_owned(), 2),
                ("update".to_owned(), 1)
            ]
        );

        store
            .connection
            .execute("DELETE FROM feedback_mutations", [])
            .unwrap();
        store
            .memory_mut()
            .set_feedback_trace(ids[0], ids[2], trace(2, 2))
            .unwrap();
        store.save().unwrap();
        let mutation_count: i64 = store
            .connection
            .query_row("SELECT count(*) FROM feedback_mutations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(mutation_count, 0);
    }

    #[test]
    fn feedback_plan_bounds_buffer_before_bulk_replacement() {
        let mut plan = FeedbackPlan::default();
        for to in 0..=MAX_BUFFERED_FEEDBACK_MUTATIONS {
            plan.push(FeedbackMutation::Insert(FeedbackRecord {
                from: 0,
                to: u64::try_from(to).unwrap(),
                trace: trace(1, 1),
            }));
        }

        assert!(plan.replace_all);
        assert!(plan.mutations.is_empty());
        plan.push(FeedbackMutation::Delete(FeedbackRecord {
            from: 1,
            to: 2,
            trace: trace(0, 1),
        }));
        assert!(plan.mutations.is_empty());
    }

    #[test]
    fn failed_save_rolls_back_revision_episodes_and_feedback() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("memory.sqlite3");
        let mut store = SqliteStore::create(&path).unwrap();
        let first = insert(&mut store, draft(0));
        store.save().unwrap();
        let second = insert(&mut store, draft(1));
        store
            .memory_mut()
            .set_feedback_trace(first, second, trace(1, 1))
            .unwrap();
        store
            .connection
            .execute_batch(
                "CREATE TEMP TRIGGER abort_feedback_insert
                 BEFORE INSERT ON main.feedback_edges
                 BEGIN SELECT RAISE(ABORT, 'test abort'); END;",
            )
            .unwrap();

        assert!(store.save().is_err());
        assert_eq!(store.memory().episodes().len(), 2);
        drop(store);

        let reopened = SqliteStore::open(&path).unwrap();
        assert_eq!(reopened.memory().episodes().len(), 1);
        assert_eq!(reopened.memory().feedback_edges().count(), 0);
    }

    #[test]
    fn failed_feedback_updates_and_deletes_roll_back_the_graph() {
        for operation in ["UPDATE", "DELETE"] {
            let directory = tempdir().unwrap();
            let path = directory.path().join("memory.sqlite3");
            let mut store = SqliteStore::create(&path).unwrap();
            let first = insert(&mut store, draft(0));
            let second = insert(&mut store, draft(1));
            store
                .memory_mut()
                .set_feedback_trace(first, second, trace(1, 1))
                .unwrap();
            store.save().unwrap();

            match operation {
                "UPDATE" => {
                    store
                        .memory_mut()
                        .set_feedback_trace(first, second, trace(2, 2))
                        .unwrap();
                }
                "DELETE" => {
                    let third = insert(&mut store, draft(2));
                    store.save().unwrap();
                    let raw = Connection::open(&path).unwrap();
                    raw.execute(
                        "INSERT INTO feedback_edges VALUES (?1, ?2, 0, 1)",
                        params![
                            codec::encode_u64(second.sequence()).as_slice(),
                            codec::encode_u64(third.sequence()).as_slice()
                        ],
                    )
                    .unwrap();
                    drop(raw);
                }
                _ => unreachable!(),
            }
            let expected_revision = store.expected_revision;
            store
                .connection
                .execute_batch(&format!(
                    "CREATE TEMP TRIGGER abort_feedback_change
                     BEFORE {operation} ON main.feedback_edges
                     BEGIN SELECT RAISE(ABORT, 'test abort'); END;"
                ))
                .unwrap();

            assert!(store.save().is_err());
            assert_eq!(store.expected_revision, expected_revision);
            drop(store);

            let reopened = SqliteStore::open(&path).unwrap();
            assert_eq!(
                reopened.memory().feedback_trace(first, second),
                Some(trace(1, 1))
            );
        }
    }

    #[test]
    fn save_reconciles_unrevisioned_feedback_divergence() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("memory.sqlite3");
        let mut store = SqliteStore::create(&path).unwrap();
        let first = insert(&mut store, draft(0));
        let second = insert(&mut store, draft(1));
        let third = insert(&mut store, draft(2));
        store
            .memory_mut()
            .set_feedback_trace(first, second, trace(1, 1))
            .unwrap();
        store.save().unwrap();

        let raw = Connection::open(&path).unwrap();
        raw.execute(
            "UPDATE feedback_edges SET history_bits = 0, sample_count = 1",
            [],
        )
        .unwrap();
        raw.execute(
            "INSERT INTO feedback_edges VALUES (?1, ?2, 3, 2)",
            params![
                codec::encode_u64(second.sequence()).as_slice(),
                codec::encode_u64(third.sequence()).as_slice()
            ],
        )
        .unwrap();
        drop(raw);

        store.save().unwrap();
        drop(store);
        let reopened = SqliteStore::open(&path).unwrap();
        assert_eq!(
            reopened.memory().feedback_trace(first, second),
            Some(trace(1, 1))
        );
        assert_eq!(reopened.memory().feedback_trace(second, third), None);
        assert_eq!(reopened.memory().feedback_edges().count(), 1);
    }

    #[test]
    fn save_rejects_invalid_persisted_feedback_without_advancing_revision() {
        fn assert_rejected(
            persisted_episodes: u64,
            unsaved_episodes: u64,
            corrupt: impl FnOnce(&Connection),
            expected_target: u64,
        ) {
            let directory = tempdir().unwrap();
            let path = directory.path().join("memory.sqlite3");
            let mut store = SqliteStore::create(&path).unwrap();
            for seed in 0..persisted_episodes {
                insert(&mut store, draft(seed));
            }
            store.save().unwrap();
            let expected_revision = store.expected_revision;
            for seed in persisted_episodes..persisted_episodes + unsaved_episodes {
                insert(&mut store, draft(seed));
            }

            let raw = Connection::open(&path).unwrap();
            corrupt(&raw);
            drop(raw);

            assert!(matches!(
                store.save(),
                Err(StoreError::InvalidStore(
                    StoreIntegrityError::InvalidFeedback {
                        from: 0,
                        to,
                        ..
                    }
                )) if to == expected_target
            ));
            assert_eq!(store.expected_revision, expected_revision);
            let raw = Connection::open(&path).unwrap();
            let persisted: (i64, i64) = raw
                .query_row(
                    "SELECT snapshot_revision, (SELECT count(*) FROM episodes)
                     FROM memory_meta",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            assert_eq!(persisted.0, expected_revision);
            assert_eq!(persisted.1, i64::try_from(persisted_episodes).unwrap());
        }

        assert_rejected(
            3,
            0,
            |raw| {
                raw.pragma_update(None, "ignore_check_constraints", true)
                    .unwrap();
                raw.execute(
                    "INSERT INTO feedback_edges VALUES (?1, ?2, 2, 1)",
                    params![
                        codec::encode_u64(0).as_slice(),
                        codec::encode_u64(1).as_slice()
                    ],
                )
                .unwrap();
            },
            1,
        );
        for sample_count in [0, 17] {
            assert_rejected(
                3,
                0,
                |raw| {
                    raw.pragma_update(None, "ignore_check_constraints", true)
                        .unwrap();
                    raw.execute(
                        "INSERT INTO feedback_edges VALUES (?1, ?2, 0, ?3)",
                        params![
                            codec::encode_u64(0).as_slice(),
                            codec::encode_u64(1).as_slice(),
                            sample_count
                        ],
                    )
                    .unwrap();
                },
                1,
            );
        }
        assert_rejected(
            3,
            0,
            |raw| {
                raw.pragma_update(None, "foreign_keys", false).unwrap();
                raw.execute(
                    "INSERT INTO feedback_edges VALUES (?1, ?2, 1, 1)",
                    params![
                        codec::encode_u64(0).as_slice(),
                        codec::encode_u64(3).as_slice()
                    ],
                )
                .unwrap();
            },
            3,
        );
        assert_rejected(
            1,
            1,
            |raw| {
                raw.pragma_update(None, "foreign_keys", false).unwrap();
                raw.execute(
                    "INSERT INTO feedback_edges VALUES (?1, ?2, 1, 1)",
                    params![
                        codec::encode_u64(0).as_slice(),
                        codec::encode_u64(1).as_slice()
                    ],
                )
                .unwrap();
            },
            1,
        );
    }

    #[test]
    fn save_rejects_changed_identity_and_exhausted_revision() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("memory.sqlite3");
        let mut store = SqliteStore::create(&path).unwrap();
        let raw = Connection::open(&path).unwrap();
        raw.execute(
            "UPDATE memory_meta SET memory_id = ?1",
            [codec::encode_memory_id(MemoryId::new(1).unwrap()).as_slice()],
        )
        .unwrap();
        drop(raw);
        assert!(matches!(
            store.save(),
            Err(StoreError::InvalidStore(
                StoreIntegrityError::InvalidMetadata { .. }
            ))
        ));

        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 0);
        let raw = Connection::open(&path).unwrap();
        raw.execute("UPDATE memory_meta SET snapshot_revision = ?1", [i64::MAX])
            .unwrap();
        drop(raw);
        let mut store = SqliteStore::open(&path).unwrap();
        assert!(matches!(store.save(), Err(StoreError::RevisionExhausted)));
    }
}
