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
mod tests;
