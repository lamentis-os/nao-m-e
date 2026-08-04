use std::path::Path;

use nao_m_e::{AtomId, Memory, MemoryId};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, OptionalExtension, Row, Transaction, TransactionBehavior};

use crate::codec;
use crate::error::{StoreError, StoreIntegrityError};
use crate::schema;

mod feedback;
mod symbols;

/// An explicitly saved SQLite database and its owned in-memory state.
///
/// A store represents one logical memory. Mutating [`Self::memory_mut`] does
/// not write to disk; call [`Self::save`] to atomically persist the changes.
pub struct SqliteStore {
    connection: Connection,
    memory: Memory,
    persisted_episode_count: usize,
    expected_revision: i64,
    predicates: symbols::SymbolState,
    terms: symbols::SymbolState,
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
        symbols::verify_symbol_tail(
            &transaction,
            symbols::SymbolNamespace::Predicate,
            predicates,
        )?;
        symbols::verify_symbol_tail(&transaction, symbols::SymbolNamespace::Term, terms)?;
        symbols::validate_new_episode_symbols(memory, *persisted_episode_count, predicates, terms)?;

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

        symbols::insert_pending_symbols(
            &transaction,
            symbols::SymbolNamespace::Predicate,
            predicates,
        )?;
        symbols::insert_pending_symbols(&transaction, symbols::SymbolNamespace::Term, terms)?;
        append_episodes(&transaction, memory, *persisted_episode_count)?;
        feedback::reconcile(&transaction, memory, *persisted_episode_count)?;
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
        || !loaded.predicates.is_persisted_empty()
        || !loaded.terms.is_persisted_empty()
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
    predicates: symbols::SymbolState,
    terms: symbols::SymbolState,
}

fn load_memory(connection: &mut Connection) -> Result<LoadedStore, StoreError> {
    let transaction = connection.transaction()?;
    verify_application_id(&transaction)?;
    let (memory_id, revision) = read_metadata(&transaction)?;
    verify_schema(&transaction)?;
    verify_quick_check(&transaction)?;
    let predicates =
        symbols::validate_symbol_catalog(&transaction, symbols::SymbolNamespace::Predicate)?;
    let terms = symbols::validate_symbol_catalog(&transaction, symbols::SymbolNamespace::Term)?;
    let mut memory = reconstruct_memory(&transaction, memory_id, &predicates, &terms)?;
    feedback::restore(&transaction, &mut memory)?;
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

fn reconstruct_memory(
    connection: &Connection,
    memory_id: MemoryId,
    predicates: &symbols::SymbolState,
    terms: &symbols::SymbolState,
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
        symbols::validate_persisted_episode_symbols(sequence, &draft, predicates, terms)?;
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

    use nao_m_e::{
        EpisodeDraft, FeedbackTrace, PredicateId, SourceId, Statement, TermId, TimestampMs,
    };
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
        assert_eq!(store.predicates.pending_len(), 1);
        assert_eq!(store.terms.pending_len(), 1);
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
