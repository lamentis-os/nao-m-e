use std::path::Path;

use nao_m_e::{AtomId, Memory, MemoryId};
use nao_m_e_semantic::{E5_SMALL_PROFILE, SemanticEncoder};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, OptionalExtension, Row, Transaction, TransactionBehavior};

use crate::error::{StoreError, StoreIntegrityError};
use crate::format;

mod feedback;
mod semantic;
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
    symbols: symbols::SymbolState,
    semantic: semantic::SemanticState,
    encoder: SemanticEncoder,
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
        format::configure_session(&connection)?;
        verify_application_id(&connection)?;
        verify_format_version(&connection)?;
        format::verify_durability(&connection)?;

        let loaded = load_memory(&mut connection)?;
        Ok(Self {
            connection,
            memory: loaded.memory,
            persisted_episode_count: loaded.episode_count,
            expected_revision: loaded.revision,
            symbols: loaded.symbols,
            semantic: semantic::SemanticState::new(loaded.semantic_cue_count),
            encoder: SemanticEncoder::new(),
        })
    }

    /// Performs a complete, physically read-only integrity audit.
    ///
    /// Unlike [`Self::open`], this scans the complete SQLite file, all foreign
    /// keys, every semantic vector, and the exact episode-to-cue projection.
    /// The semantic model is neither resolved nor loaded.
    pub fn check(path: impl AsRef<Path>) -> Result<(), StoreError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY;
        let mut connection = Connection::open_with_flags(path, flags)?;
        format::configure_session(&connection)?;
        verify_application_id(&connection)?;
        verify_format_version(&connection)?;
        format::verify_durability(&connection)?;
        let transaction = connection.transaction()?;
        verify_application_id(&transaction)?;
        verify_format_version(&transaction)?;
        verify_integrity_check(&transaction)?;
        verify_foreign_keys(&transaction)?;
        let loaded = load_snapshot(&transaction)?;
        semantic::full_audit(&transaction, &loaded.memory, loaded.semantic_cue_count)?;
        transaction.commit()?;
        Ok(())
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
        let mut encoder = std::mem::take(&mut self.encoder);
        let result = self.save_with_encoder(&mut encoder);
        self.encoder = encoder;
        result
    }

    fn save_with_encoder<E: semantic::CueEncoder>(
        &mut self,
        encoder: &mut E,
    ) -> Result<(), StoreError> {
        symbols::validate_new_episode_symbols(
            &self.memory,
            self.persisted_episode_count,
            &self.symbols,
        )?;
        let prepared_semantic = semantic::prepare(self, encoder)?;

        let Self {
            connection,
            memory,
            persisted_episode_count,
            expected_revision,
            symbols,
            semantic,
            encoder: _,
        } = self;

        let episode_count = memory.episodes().len();
        if *persisted_episode_count > episode_count {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "persisted episode count exceeds in-memory episode count",
            }
            .into());
        }

        format::verify_durability(connection)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        verify_application_id(&transaction)?;
        let (actual_memory_id, actual_revision, actual_semantic_cue_count) =
            read_metadata(&transaction)?;
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
        if actual_semantic_cue_count != semantic.persisted_count {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "persisted semantic cue count changed outside this store session",
            }
            .into());
        }
        if actual_revision == i64::MAX {
            return Err(StoreError::RevisionExhausted);
        }
        verify_persisted_tail(&transaction, *persisted_episode_count)?;
        symbols::verify_symbol_tail(&transaction, symbols)?;
        semantic::verify_tail(&transaction, semantic)?;

        let next_revision = actual_revision + 1;
        let next_semantic_cue_count = semantic.current_count()?;
        let changed = transaction.execute(
            "UPDATE memory_meta
             SET snapshot_revision = ?1, semantic_cue_count = ?2
             WHERE singleton = 1",
            rusqlite::params![
                next_revision,
                format::encode_u64(next_semantic_cue_count).as_slice()
            ],
        )?;
        if changed != 1 {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "metadata singleton changed during the save transaction",
            }
            .into());
        }

        symbols::insert_pending_symbols(&transaction, symbols)?;
        let committed_semantic_count = semantic::insert_pending_cues(&transaction, semantic)?;
        debug_assert_eq!(committed_semantic_count, next_semantic_cue_count);
        append_episodes(&transaction, memory, *persisted_episode_count)?;
        semantic::insert_postings(&transaction, &prepared_semantic, memory)?;
        feedback::reconcile(&transaction, memory, *persisted_episode_count)?;
        transaction.commit()?;

        *expected_revision = next_revision;
        *persisted_episode_count = episode_count;
        symbols.mark_saved();
        semantic.mark_saved();
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
    format::configure_session(&connection)?;
    format::configure_durability(&connection)?;
    format::create_schema(&mut connection, memory_id, &E5_SMALL_PROFILE.fingerprint())?;
    let loaded = load_memory(&mut connection)?;
    if loaded.memory.memory_id() != memory_id
        || loaded.episode_count != 0
        || loaded.revision != 0
        || loaded.semantic_cue_count != 0
        || !loaded.symbols.is_persisted_empty()
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
    symbols: symbols::SymbolState,
    semantic_cue_count: u64,
}

fn load_memory(connection: &mut Connection) -> Result<LoadedStore, StoreError> {
    let transaction = connection.transaction()?;
    let loaded = load_snapshot(&transaction)?;
    transaction.commit()?;
    Ok(loaded)
}

fn load_snapshot(connection: &Connection) -> Result<LoadedStore, StoreError> {
    verify_application_id(connection)?;
    let (memory_id, revision, semantic_cue_count) = read_metadata(connection)?;
    verify_schema(connection)?;
    verify_quick_check(connection)?;
    semantic::verify_tail(
        connection,
        &semantic::SemanticState::new(semantic_cue_count),
    )?;
    let symbols = symbols::validate_symbol_catalog(connection)?;
    let mut memory = reconstruct_memory(connection, memory_id, &symbols)?;
    feedback::restore(connection, &mut memory)?;
    let episode_count = memory.episodes().len();
    Ok(LoadedStore {
        memory,
        episode_count,
        revision,
        symbols,
        semantic_cue_count,
    })
}

fn verify_application_id(connection: &Connection) -> Result<(), StoreError> {
    let found = format::read_application_id(connection)?;
    if found == format::APPLICATION_ID {
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
    if found != format::FORMAT_VERSION {
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
    if format::validate_schema(connection)? {
        Ok(())
    } else {
        Err(StoreIntegrityError::InvalidMetadata {
            detail: "database schema differs from the SQLite contract",
        }
        .into())
    }
}

fn verify_quick_check(connection: &Connection) -> Result<(), StoreError> {
    let mut diagnostics = Vec::new();
    for table in ["memory_meta", "symbols", "episodes", "feedback_edges"] {
        let mut statement = connection.prepare(&format!("PRAGMA quick_check({table})"))?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let diagnostic = row.get::<_, String>(0)?;
            if diagnostic != "ok" {
                diagnostics.push(format!("{table}: {diagnostic}"));
            }
        }
    }
    if diagnostics.is_empty() {
        return Ok(());
    }
    Err(StoreIntegrityError::QuickCheckFailed {
        detail: diagnostics.join("; "),
    }
    .into())
}

fn verify_integrity_check(connection: &Connection) -> Result<(), StoreError> {
    let mut statement = connection.prepare("PRAGMA integrity_check")?;
    let mut rows = statement.query([])?;
    let mut diagnostics = Vec::new();
    while let Some(row) = rows.next()? {
        diagnostics.push(row.get::<_, String>(0)?);
    }
    if diagnostics.len() == 1 && diagnostics[0] == "ok" {
        Ok(())
    } else {
        Err(StoreIntegrityError::IntegrityCheckFailed {
            detail: diagnostics.join("; "),
        }
        .into())
    }
}

fn verify_foreign_keys(connection: &Connection) -> Result<(), StoreError> {
    let mut statement = connection.prepare("PRAGMA foreign_key_check")?;
    let mut rows = statement.query([])?;
    if let Some(row) = rows.next()? {
        return Err(StoreIntegrityError::ForeignKeyCheckFailed {
            detail: format!(
                "{} row {} references {}",
                row.get::<_, String>(0)?,
                row.get::<_, Option<i64>>(1)?
                    .map_or_else(|| "without rowid".to_owned(), |value| value.to_string()),
                row.get::<_, String>(2)?,
            ),
        }
        .into());
    }
    Ok(())
}

fn read_metadata(connection: &Connection) -> Result<(MemoryId, i64, u64), StoreError> {
    let mut statement = connection.prepare(
        "SELECT singleton, format_version, memory_id, snapshot_revision,
                semantic_profile_fingerprint, semantic_cue_count
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
    if format_version != format::FORMAT_VERSION {
        return Err(StoreIntegrityError::UnsupportedFormatVersion {
            found: format_version,
        }
        .into());
    }
    let memory_id = read_memory_id(row, 2, "memory_meta", "memory_id")?;
    let revision = read_integer(row, 3, "memory_meta", "snapshot_revision")?;
    let ValueRef::Blob(fingerprint) = row.get_ref(4)? else {
        return Err(StoreIntegrityError::InvalidEncoding {
            table: "memory_meta",
            column: "semantic_profile_fingerprint",
        }
        .into());
    };
    if fingerprint != E5_SMALL_PROFILE.fingerprint() {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "semantic profile fingerprint differs from SQLite format V7",
        }
        .into());
    }
    let semantic_cue_count = read_u64(row, 5, "memory_meta", "semantic_cue_count")?;
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
    Ok((memory_id, revision, semantic_cue_count))
}

fn reconstruct_memory(
    connection: &Connection,
    memory_id: MemoryId,
    symbols: &symbols::SymbolState,
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
        let draft = format::decode_episode(payload).map_err(|error| {
            StoreIntegrityError::InvalidEpisode {
                sequence,
                detail: error.detail(),
            }
        })?;
        symbols::validate_persisted_episode_symbols(sequence, &draft, symbols)?;
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
            format::decode_u64(bytes).ok_or(StoreIntegrityError::InvalidEncoding {
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
    if start == memory.episodes().len() {
        return Ok(());
    }
    let mut insert = transaction.prepare(
        "INSERT INTO episodes (sequence, payload)
         VALUES (?1, ?2)",
    )?;
    let mut payload = Vec::with_capacity(format::MIN_EPISODE_PAYLOAD_BYTES);
    for episode in memory.episodes().skip(start) {
        let sequence = format::encode_u64(episode.id().sequence());
        format::encode_episode(episode, &mut payload);
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
    format::decode_u64(bytes)
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
    format::decode_memory_id(bytes).ok_or_else(|| {
        if bytes.len() == 16 {
            StoreIntegrityError::InvalidMemoryId.into()
        } else {
            StoreIntegrityError::InvalidEncoding { table, column }.into()
        }
    })
}

#[cfg(test)]
mod tests;
