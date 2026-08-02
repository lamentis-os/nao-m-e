use std::path::Path;

use nao_m_e::{Activation, AtomId, GraphError, InfluenceWeight, MemoryId, MemoryV0, SCALE};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, OptionalExtension, Row, Transaction, TransactionBehavior};

use crate::codec;
use crate::error::{StoreError, StoreIntegrityError};
use crate::schema;

/// An explicitly saved SQLite database and its owned in-memory state.
///
/// A store represents one logical memory. Mutating [`Self::memory_mut`] does
/// not write to disk; call [`Self::save`] to atomically persist the changes.
pub struct SqliteStore {
    connection: Connection,
    memory: MemoryV0,
    persisted_episode_count: usize,
    expected_revision: i64,
}

impl SqliteStore {
    /// Creates a new empty SQLite V2 store at `path`.
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

    /// Opens and validates an existing SQLite V2 memory store.
    ///
    /// Missing files are not created. V1, invalid, and unsupported stores are
    /// rejected without returning a partial memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        let mut connection = Connection::open_with_flags(path, flags)?;
        schema::configure_session(&connection)?;
        verify_application_id(&connection)?;
        verify_format_version(&connection)?;
        schema::configure_durability(&connection)?;

        let (memory, persisted_episode_count, expected_revision) = load_memory(&mut connection)?;
        Ok(Self {
            connection,
            memory,
            persisted_episode_count,
            expected_revision,
        })
    }

    /// Returns the durable identifier of the owned logical memory.
    #[must_use]
    pub const fn memory_id(&self) -> MemoryId {
        self.memory.memory_id()
    }

    /// Returns the owned memory for read-only operations.
    #[must_use]
    pub const fn memory(&self) -> &MemoryV0 {
        &self.memory
    }

    /// Returns the owned memory for mutations that remain unsaved until
    /// [`Self::save`] succeeds.
    #[must_use]
    pub const fn memory_mut(&mut self) -> &mut MemoryV0 {
        &mut self.memory
    }

    /// Atomically persists newly appended episodes and the complete mutable
    /// activation and relevance state.
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
        } = self;

        let episode_count = memory.episodes().len();
        if *persisted_episode_count > episode_count {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "persisted episode count exceeds in-memory episode count",
            }
            .into());
        }

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

        append_episodes(&transaction, memory, *persisted_episode_count)?;
        replace_activations(&transaction, memory)?;
        replace_relevance(&transaction, memory)?;
        transaction.commit()?;

        *expected_revision = next_revision;
        *persisted_episode_count = episode_count;
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
    let (memory, episode_count, revision) = load_memory(&mut connection)?;
    if memory.memory_id() != memory_id || episode_count != 0 || revision != 0 {
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

fn load_memory(connection: &mut Connection) -> Result<(MemoryV0, usize, i64), StoreError> {
    let transaction = connection.transaction()?;
    verify_application_id(&transaction)?;
    let (memory_id, revision) = read_metadata(&transaction)?;
    verify_schema(&transaction)?;
    verify_quick_check(&transaction)?;
    verify_foreign_keys(&transaction)?;
    let mut memory = reconstruct_memory(&transaction, memory_id)?;
    restore_activations(&transaction, &mut memory)?;
    restore_relevance(&transaction, &mut memory)?;
    let episode_count = memory.episodes().len();
    transaction.commit()?;
    Ok((memory, episode_count, revision))
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
            detail: "database schema differs from the SQLite V2 contract",
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

fn verify_foreign_keys(connection: &Connection) -> Result<(), StoreError> {
    let mut statement = connection.prepare("PRAGMA foreign_key_check")?;
    let mut rows = statement.query([])?;
    let Some(row) = rows.next()? else {
        return Ok(());
    };
    let table: String = row.get(0)?;
    let row_id: Option<i64> = row.get(1)?;
    let parent: String = row.get(2)?;
    let foreign_key: i64 = row.get(3)?;
    Err(StoreIntegrityError::ForeignKeyViolation {
        detail: format!("{table} row {row_id:?} -> {parent} constraint {foreign_key}"),
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
) -> Result<MemoryV0, StoreError> {
    let mut statement = connection.prepare(
        "SELECT sequence, payload
         FROM episodes
         ORDER BY sequence",
    )?;
    let mut rows = statement.query([])?;
    let mut memory = MemoryV0::new(memory_id);
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

fn restore_activations(connection: &Connection, memory: &mut MemoryV0) -> Result<(), StoreError> {
    let mut statement = connection.prepare(
        "SELECT episode_sequence, activation_ppm
         FROM activations
         ORDER BY episode_sequence",
    )?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let sequence = read_u64(row, 0, "activations", "episode_sequence")?;
        let value = read_integer(row, 1, "activations", "activation_ppm")?;
        let ppm = u32::try_from(value)
            .ok()
            .filter(|value| (1..=SCALE).contains(value))
            .ok_or(StoreIntegrityError::InvalidActivation {
                sequence,
                detail: "activation is outside the positive fixed-point range",
            })?;
        let id = AtomId::from_parts(memory.memory_id(), sequence);
        let activation =
            Activation::from_ppm(ppm).map_err(|_| StoreIntegrityError::InvalidActivation {
                sequence,
                detail: "activation is outside the positive fixed-point range",
            })?;
        memory
            .stimulate(id, activation)
            .map_err(|_| StoreIntegrityError::InvalidActivation {
                sequence,
                detail: "activation references an absent episode",
            })?;
    }
    Ok(())
}

fn restore_relevance(connection: &Connection, memory: &mut MemoryV0) -> Result<(), StoreError> {
    let mut statement = connection.prepare(
        "SELECT from_sequence, to_sequence, weight_ppm
         FROM relevance_edges
         ORDER BY from_sequence, to_sequence",
    )?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let from = read_u64(row, 0, "relevance_edges", "from_sequence")?;
        let to = read_u64(row, 1, "relevance_edges", "to_sequence")?;
        if from == to {
            return Err(StoreIntegrityError::InvalidRelevance {
                from,
                to,
                detail: "self-edge",
            }
            .into());
        }
        let value = read_integer(row, 2, "relevance_edges", "weight_ppm")?;
        let ppm = u32::try_from(value)
            .ok()
            .filter(|value| (1..=SCALE).contains(value))
            .ok_or(StoreIntegrityError::InvalidRelevance {
                from,
                to,
                detail: "weight is outside the fixed-point range",
            })?;

        let from_id = AtomId::from_parts(memory.memory_id(), from);
        let to_id = AtomId::from_parts(memory.memory_id(), to);
        let weight =
            InfluenceWeight::from_ppm(ppm).map_err(|_| StoreIntegrityError::InvalidRelevance {
                from,
                to,
                detail: "weight is outside the fixed-point range",
            })?;
        memory
            .set_relevance(from_id, to_id, weight)
            .map_err(|error| StoreIntegrityError::InvalidRelevance {
                from,
                to,
                detail: match error {
                    GraphError::UnknownAtom(_) => "edge endpoint is absent",
                    GraphError::FeedbackTargetLimitExceeded { .. } => {
                        "feedback target limit was exceeded during reconstruction"
                    }
                    GraphError::SelfEdge(_) => "self-edge",
                    GraphError::OutgoingWeightBudgetExceeded { .. } => {
                        "outgoing weight budget is exceeded"
                    }
                },
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

fn append_episodes(
    transaction: &Transaction<'_>,
    memory: &MemoryV0,
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

fn replace_activations(transaction: &Transaction<'_>, memory: &MemoryV0) -> Result<(), StoreError> {
    transaction.execute("DELETE FROM activations", [])?;
    let mut insert = transaction.prepare(
        "INSERT INTO activations (episode_sequence, activation_ppm)
         VALUES (?1, ?2)",
    )?;
    for episode in memory.episodes() {
        let activation = memory
            .activation(episode.id())
            .expect("an in-memory episode always has parallel activation storage");
        if activation == Activation::ZERO {
            continue;
        }
        let sequence = codec::encode_u64(episode.id().sequence());
        insert.execute((sequence.as_slice(), i64::from(activation.as_ppm())))?;
    }
    Ok(())
}

fn replace_relevance(transaction: &Transaction<'_>, memory: &MemoryV0) -> Result<(), StoreError> {
    transaction.execute("DELETE FROM relevance_edges", [])?;
    let mut insert = transaction.prepare(
        "INSERT INTO relevance_edges (from_sequence, to_sequence, weight_ppm)
         VALUES (?1, ?2, ?3)",
    )?;
    for edge in memory.relevance_edges() {
        let from = codec::encode_u64(edge.from().sequence());
        let to = codec::encode_u64(edge.to().sequence());
        insert.execute((
            from.as_slice(),
            to.as_slice(),
            i64::from(edge.weight().as_ppm()),
        ))?;
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
            context: vec![statement(10 + seed, &[100 + seed])],
            observation: statement(20 + seed, &[200 + seed, 201 + seed]),
            action: Some(statement(30 + seed, &[300 + seed])),
            outcome: None,
            source: SourceId::new(40 + seed),
        }
    }

    fn saved_store(directory: &TempDir, episode_count: u64) -> PathBuf {
        let path = directory.path().join("memory.sqlite3");
        let mut store = SqliteStore::create(&path).expect("test store is created");
        for seed in 0..episode_count {
            store
                .memory_mut()
                .insert_episode(draft(seed))
                .expect("test episode inserts");
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
    fn metadata_and_v1_format_fail_closed_with_specific_errors() {
        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 0);
        let raw = Connection::open(&path).unwrap();
        raw.pragma_update(None, "ignore_check_constraints", true)
            .unwrap();
        raw.execute("UPDATE memory_meta SET format_version = 1", [])
            .unwrap();
        drop(raw);
        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::UnsupportedFormatVersion { found: 1 }
        ));

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
    fn rejected_application_and_format_do_not_change_journal_mode() {
        for unsupported_format in [false, true] {
            let directory = tempdir().unwrap();
            let path = if unsupported_format {
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
            if unsupported_format {
                raw.pragma_update(None, "ignore_check_constraints", true)
                    .unwrap();
                raw.execute("UPDATE memory_meta SET format_version = 1", [])
                    .unwrap();
            }
            let mode: String = raw
                .query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))
                .unwrap();
            assert_eq!(mode.to_ascii_lowercase(), "wal");
            drop(raw);

            assert!(SqliteStore::open(&path).is_err());

            let raw = Connection::open(&path).unwrap();
            let mode: String = raw
                .pragma_query_value(None, "journal_mode", |row| row.get(0))
                .unwrap();
            assert_eq!(mode.to_ascii_lowercase(), "wal");
        }
    }

    #[test]
    fn payload_sequence_and_sparse_state_corruption_are_rejected() {
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

        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 1);
        let raw = Connection::open(&path).unwrap();
        raw.pragma_update(None, "ignore_check_constraints", true)
            .unwrap();
        raw.execute(
            "INSERT INTO activations VALUES (?1, 0)",
            [codec::encode_u64(0).as_slice()],
        )
        .unwrap();
        drop(raw);
        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::QuickCheckFailed { .. }
        ));
    }

    #[test]
    fn sparse_activations_store_only_positive_values() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("memory.sqlite3");
        let mut store = SqliteStore::create(&path).unwrap();
        let ids: Vec<_> = (0..3)
            .map(|seed| store.memory_mut().insert_episode(draft(seed)).unwrap())
            .collect();
        store
            .memory_mut()
            .stimulate(ids[1], Activation::from_ppm(42).unwrap())
            .unwrap();
        store.save().unwrap();

        let raw = Connection::open(&path).unwrap();
        let rows: i64 = raw
            .query_row("SELECT count(*) FROM activations", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 1);
        drop(raw);
        drop(store);

        let mut reopened = SqliteStore::open(&path).unwrap();
        assert_eq!(reopened.memory().activation(ids[0]), Some(Activation::ZERO));
        assert_eq!(
            reopened.memory().activation(ids[1]),
            Some(Activation::from_ppm(42).unwrap())
        );
        reopened.memory_mut().reset_activations();
        reopened.save().unwrap();
        drop(reopened);

        let raw = Connection::open(&path).unwrap();
        let rows: i64 = raw
            .query_row("SELECT count(*) FROM activations", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 0);
    }

    #[test]
    fn relevance_budget_and_persistent_triggers_are_rejected() {
        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 3);
        let raw = Connection::open(&path).unwrap();
        for (to, weight) in [(1, 600_000), (2, 600_000)] {
            raw.execute(
                "INSERT INTO relevance_edges VALUES (?1, ?2, ?3)",
                params![
                    codec::encode_u64(0).as_slice(),
                    codec::encode_u64(to).as_slice(),
                    weight
                ],
            )
            .unwrap();
        }
        drop(raw);
        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::InvalidRelevance { from: 0, to: 2, .. }
        ));

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
    fn failed_save_rolls_back_revision_episodes_and_mutable_state() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("memory.sqlite3");
        let mut store = SqliteStore::create(&path).unwrap();
        let first = store.memory_mut().insert_episode(draft(0)).unwrap();
        store.save().unwrap();
        let second = store.memory_mut().insert_episode(draft(1)).unwrap();
        store
            .memory_mut()
            .stimulate(first, Activation::ONE)
            .unwrap();
        store
            .memory_mut()
            .set_relevance(first, second, InfluenceWeight::from_ppm(1).unwrap())
            .unwrap();
        store
            .connection
            .execute_batch(
                "CREATE TEMP TRIGGER abort_activation_insert
                 BEFORE INSERT ON main.activations
                 BEGIN SELECT RAISE(ABORT, 'test abort'); END;",
            )
            .unwrap();

        assert!(store.save().is_err());
        assert_eq!(store.memory().episodes().len(), 2);
        drop(store);

        let reopened = SqliteStore::open(&path).unwrap();
        assert_eq!(reopened.memory().episodes().len(), 1);
        assert_eq!(reopened.memory().activation(first), Some(Activation::ZERO));
        assert_eq!(reopened.memory().relevance_edges().count(), 0);
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
