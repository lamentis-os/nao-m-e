use std::io::Write;
use std::path::Path;

use nao_m_e::{
    Activation, AtomId, EpisodeAtom, EpisodeDraft, GraphError, InfluenceWeight, MemoryId, MemoryV0,
    PredicateId, SCALE, SourceId, Statement, TermId, TimestampMs,
};
use rusqlite::types::ValueRef;
use rusqlite::{
    Connection, MAIN_DB, OpenFlags, OptionalExtension, Row, Transaction, TransactionBehavior,
};

use crate::codec;
use crate::error::{StoreError, StoreIntegrityError};
use crate::schema;

const ROLE_CONTEXT: i64 = 0;
const ROLE_OBSERVATION: i64 = 1;
const ROLE_ACTION: i64 = 2;
const ROLE_OUTCOME: i64 = 3;

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
    /// Creates a new empty store at `path`.
    ///
    /// The operation fails rather than opening or replacing an existing file.
    /// A non-zero memory identifier is generated from operating-system entropy
    /// and committed with the initial schema.
    pub fn create(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref();
        let memory_id = random_memory_id()?;
        let connection = build_initial_database(memory_id)?;
        let database = connection.serialize(MAIN_DB)?;
        publish_database(path, &database)?;
        Self::open(path)
    }

    /// Opens and validates an existing SQLite V1 memory store.
    ///
    /// Missing files are not created. Invalid or unsupported stores are
    /// rejected without returning a partial memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        let mut connection = Connection::open_with_flags(path, flags)?;
        schema::configure_session(&connection)?;
        let application_id = schema::read_application_id(&connection)?;
        if application_id != schema::APPLICATION_ID {
            return Err(StoreIntegrityError::ApplicationMismatch {
                found: application_id,
            }
            .into());
        }
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
        verify_schema(&transaction)?;
        let (actual_memory_id, actual_revision) = read_metadata(&transaction)?;
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
        let memory_id = codec::encode_memory_id(memory.memory_id());
        let changed = transaction.execute(
            "UPDATE memory_meta
             SET snapshot_revision = ?1
             WHERE singleton = 1
               AND snapshot_revision = ?2
               AND memory_id = ?3",
            rusqlite::params![next_revision, actual_revision, memory_id.as_slice()],
        )?;
        if changed != 1 {
            let (found_memory_id, found_revision) = read_metadata(&transaction)?;
            if found_memory_id != memory.memory_id() {
                return Err(StoreIntegrityError::InvalidMetadata {
                    detail: "persisted memory ID differs from the owned memory",
                }
                .into());
            }
            return Err(StoreError::ConcurrentModification {
                expected_revision: actual_revision,
                actual_revision: found_revision,
            });
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

fn build_initial_database(memory_id: MemoryId) -> Result<Connection, StoreError> {
    let mut connection = Connection::open_in_memory()?;
    schema::configure_session(&connection)?;
    schema::create_schema(&mut connection, memory_id)?;
    verify_application_id(&connection)?;
    verify_schema(&connection)?;
    verify_quick_check(&connection)?;
    verify_foreign_keys(&connection)?;
    let (stored_memory_id, revision) = read_metadata(&connection)?;
    if stored_memory_id != memory_id || revision != 0 {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "new store metadata differs from its initialized identity",
        }
        .into());
    }
    Ok(connection)
}

fn publish_database(path: &Path, database: &[u8]) -> Result<(), StoreError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut staging = tempfile::Builder::new()
        .prefix(".nao-m-e-")
        .suffix(".sqlite.tmp")
        .tempfile_in(parent)?;
    staging.write_all(database)?;
    staging.flush()?;
    staging.as_file().sync_all()?;
    let published = staging
        .persist_noclobber(path)
        .map_err(|error| StoreError::Io(error.error))?;
    drop(published);
    Ok(())
}

fn load_memory(connection: &mut Connection) -> Result<(MemoryV0, usize, i64), StoreError> {
    let transaction = connection.transaction()?;
    verify_application_id(&transaction)?;
    verify_schema(&transaction)?;
    let (memory_id, revision) = read_metadata(&transaction)?;
    verify_quick_check(&transaction)?;
    verify_foreign_keys(&transaction)?;
    let episode_rows = read_episodes(&transaction)?;
    let mut statement_rows = read_statements(&transaction)?;
    attach_terms(&transaction, &mut statement_rows)?;
    let mut memory = reconstruct_memory(memory_id, episode_rows, statement_rows)?;
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

fn verify_schema(connection: &Connection) -> Result<(), StoreError> {
    if schema::validate_schema(connection)? {
        Ok(())
    } else {
        Err(StoreIntegrityError::InvalidMetadata {
            detail: "database schema differs from the SQLite V1 contract",
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

#[derive(Debug)]
struct EpisodeRow {
    sequence: u64,
    occurred_at: TimestampMs,
    recorded_at: TimestampMs,
    source: SourceId,
}

fn read_episodes(connection: &Connection) -> Result<Vec<EpisodeRow>, StoreError> {
    let mut statement = connection.prepare(
        "SELECT sequence, occurred_at_ms, recorded_at_ms, source_id
         FROM episodes
         ORDER BY sequence",
    )?;
    let mut rows = statement.query([])?;
    let mut episodes = Vec::new();
    let mut expected = 0_u64;
    while let Some(row) = rows.next()? {
        let sequence = read_u64(row, 0, "episodes", "sequence")?;
        if sequence != expected {
            return Err(StoreIntegrityError::NonContiguousEpisodeSequence {
                expected,
                found: sequence,
            }
            .into());
        }
        episodes.push(EpisodeRow {
            sequence,
            occurred_at: TimestampMs::new(read_integer(row, 1, "episodes", "occurred_at_ms")?),
            recorded_at: TimestampMs::new(read_integer(row, 2, "episodes", "recorded_at_ms")?),
            source: SourceId::new(read_u64(row, 3, "episodes", "source_id")?),
        });
        expected = expected
            .checked_add(1)
            .ok_or(StoreIntegrityError::InvalidMetadata {
                detail: "episode sequence space is exhausted",
            })?;
    }
    Ok(episodes)
}

#[derive(Debug)]
struct StoredStatement {
    episode_sequence: u64,
    role: i64,
    ordinal: u64,
    predicate: PredicateId,
    arguments: Vec<TermId>,
}

impl StoredStatement {
    const fn key(&self) -> (u64, i64, u64) {
        (self.episode_sequence, self.role, self.ordinal)
    }
}

fn read_statements(connection: &Connection) -> Result<Vec<StoredStatement>, StoreError> {
    let mut statement = connection.prepare(
        "SELECT episode_sequence, role, statement_ordinal, predicate_id
         FROM episode_statements
         ORDER BY episode_sequence, role, statement_ordinal",
    )?;
    let mut rows = statement.query([])?;
    let mut statements = Vec::new();
    while let Some(row) = rows.next()? {
        let episode_sequence = read_u64(row, 0, "episode_statements", "episode_sequence")?;
        let role = read_integer(row, 1, "episode_statements", "role")?;
        let ordinal = read_non_negative_u64(
            row,
            2,
            "episode_statements",
            "statement_ordinal",
            episode_sequence,
        )?;
        if !(ROLE_CONTEXT..=ROLE_OUTCOME).contains(&role) {
            return Err(StoreIntegrityError::InvalidEpisode {
                sequence: episode_sequence,
                detail: "unknown statement role",
            }
            .into());
        }
        if role != ROLE_CONTEXT && ordinal != 0 {
            return Err(StoreIntegrityError::InvalidEpisode {
                sequence: episode_sequence,
                detail: "singleton statement role has a non-zero ordinal",
            }
            .into());
        }
        statements.push(StoredStatement {
            episode_sequence,
            role,
            ordinal,
            predicate: PredicateId::new(read_u64(row, 3, "episode_statements", "predicate_id")?),
            arguments: Vec::new(),
        });
    }
    Ok(statements)
}

#[derive(Debug)]
struct TermRow {
    episode_sequence: u64,
    role: i64,
    statement_ordinal: u64,
    term_ordinal: u64,
    term: TermId,
}

impl TermRow {
    const fn statement_key(&self) -> (u64, i64, u64) {
        (self.episode_sequence, self.role, self.statement_ordinal)
    }
}

fn read_next_term(rows: &mut rusqlite::Rows<'_>) -> Result<Option<TermRow>, StoreError> {
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    let episode_sequence = read_u64(row, 0, "statement_terms", "episode_sequence")?;
    Ok(Some(TermRow {
        episode_sequence,
        role: read_integer(row, 1, "statement_terms", "role")?,
        statement_ordinal: read_non_negative_u64(
            row,
            2,
            "statement_terms",
            "statement_ordinal",
            episode_sequence,
        )?,
        term_ordinal: read_non_negative_u64(
            row,
            3,
            "statement_terms",
            "term_ordinal",
            episode_sequence,
        )?,
        term: TermId::new(read_u64(row, 4, "statement_terms", "term_id")?),
    }))
}

fn attach_terms(
    connection: &Connection,
    statements: &mut [StoredStatement],
) -> Result<(), StoreError> {
    let mut query = connection.prepare(
        "SELECT episode_sequence, role, statement_ordinal, term_ordinal, term_id
         FROM statement_terms
         ORDER BY episode_sequence, role, statement_ordinal, term_ordinal",
    )?;
    let mut rows = query.query([])?;
    let mut next_term = read_next_term(&mut rows)?;

    for statement in statements {
        let mut expected_ordinal = 0_u64;
        while next_term
            .as_ref()
            .is_some_and(|term| term.statement_key() == statement.key())
        {
            let term = next_term
                .take()
                .expect("a matching term was checked before extraction");
            if term.term_ordinal != expected_ordinal {
                return Err(StoreIntegrityError::InvalidEpisode {
                    sequence: statement.episode_sequence,
                    detail: "term ordinals are not contiguous",
                }
                .into());
            }
            statement.arguments.push(term.term);
            expected_ordinal =
                expected_ordinal
                    .checked_add(1)
                    .ok_or(StoreIntegrityError::InvalidEpisode {
                        sequence: statement.episode_sequence,
                        detail: "term ordinal space is exhausted",
                    })?;
            next_term = read_next_term(&mut rows)?;
        }
        if statement.arguments.is_empty() {
            return Err(StoreIntegrityError::InvalidEpisode {
                sequence: statement.episode_sequence,
                detail: "statement has no terms",
            }
            .into());
        }
    }
    if let Some(term) = next_term {
        return Err(StoreIntegrityError::InvalidEpisode {
            sequence: term.episode_sequence,
            detail: "term references an absent statement",
        }
        .into());
    }
    Ok(())
}

fn reconstruct_memory(
    memory_id: MemoryId,
    episodes: Vec<EpisodeRow>,
    statements: Vec<StoredStatement>,
) -> Result<MemoryV0, StoreError> {
    let mut statements = statements.into_iter().peekable();
    let mut memory = MemoryV0::new(memory_id);
    for episode in episodes {
        let mut context = Vec::new();
        let mut observation = None;
        let mut action = None;
        let mut outcome = None;
        let mut next_context_ordinal = 0_u64;

        while statements
            .peek()
            .is_some_and(|statement| statement.episode_sequence == episode.sequence)
        {
            let statement = statements.next().expect("peeked statement exists");
            let value = Statement::new(statement.predicate, statement.arguments).map_err(|_| {
                StoreIntegrityError::InvalidEpisode {
                    sequence: episode.sequence,
                    detail: "statement has no terms",
                }
            })?;
            match statement.role {
                ROLE_CONTEXT => {
                    if statement.ordinal != next_context_ordinal {
                        return Err(StoreIntegrityError::InvalidEpisode {
                            sequence: episode.sequence,
                            detail: "context ordinals are not contiguous",
                        }
                        .into());
                    }
                    next_context_ordinal += 1;
                    context.push(value);
                }
                ROLE_OBSERVATION => set_singleton(
                    &mut observation,
                    value,
                    episode.sequence,
                    "multiple observations",
                )?,
                ROLE_ACTION => {
                    set_singleton(&mut action, value, episode.sequence, "multiple actions")?
                }
                ROLE_OUTCOME => {
                    set_singleton(&mut outcome, value, episode.sequence, "multiple outcomes")?
                }
                _ => unreachable!("statement roles were validated while reading"),
            }
        }

        if context.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(StoreIntegrityError::InvalidEpisode {
                sequence: episode.sequence,
                detail: "context is not strictly sorted and duplicate-free",
            }
            .into());
        }
        let observation = observation.ok_or(StoreIntegrityError::InvalidEpisode {
            sequence: episode.sequence,
            detail: "observation is missing",
        })?;
        let sequence = episode.sequence;
        let draft = EpisodeDraft {
            occurred_at: episode.occurred_at,
            recorded_at: episode.recorded_at,
            context,
            observation,
            action,
            outcome,
            source: episode.source,
        };
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
    }
    if let Some(statement) = statements.next() {
        return Err(StoreIntegrityError::InvalidEpisode {
            sequence: statement.episode_sequence,
            detail: "statement references an absent episode",
        }
        .into());
    }
    Ok(memory)
}

fn set_singleton(
    slot: &mut Option<Statement>,
    value: Statement,
    sequence: u64,
    detail: &'static str,
) -> Result<(), StoreError> {
    if slot.replace(value).is_some() {
        Err(StoreIntegrityError::InvalidEpisode { sequence, detail }.into())
    } else {
        Ok(())
    }
}

fn restore_activations(connection: &Connection, memory: &mut MemoryV0) -> Result<(), StoreError> {
    let mut statement = connection.prepare(
        "SELECT episode_sequence, activation_ppm
         FROM activations
         ORDER BY episode_sequence",
    )?;
    let mut rows = statement.query([])?;
    let episode_count = memory.episodes().len();
    let mut expected = 0_usize;
    while let Some(row) = rows.next()? {
        let sequence = read_u64(row, 0, "activations", "episode_sequence")?;
        let expected_sequence =
            u64::try_from(expected).map_err(|_| StoreIntegrityError::InvalidMetadata {
                detail: "episode count exceeds the atom ID sequence space",
            })?;
        if sequence != expected_sequence {
            return Err(StoreIntegrityError::InvalidActivation {
                sequence: expected_sequence,
                detail: "activation rows are missing or out of order",
            }
            .into());
        }
        let value = read_integer(row, 1, "activations", "activation_ppm")?;
        let ppm = u32::try_from(value)
            .ok()
            .filter(|value| *value <= SCALE)
            .ok_or(StoreIntegrityError::InvalidActivation {
                sequence,
                detail: "activation is outside the fixed-point range",
            })?;
        if expected >= episode_count {
            return Err(StoreIntegrityError::InvalidActivation {
                sequence,
                detail: "activation references an absent episode",
            }
            .into());
        }
        let id = AtomId::from_parts(memory.memory_id(), sequence);
        if ppm != 0 {
            let activation =
                Activation::from_ppm(ppm).map_err(|_| StoreIntegrityError::InvalidActivation {
                    sequence,
                    detail: "activation is outside the fixed-point range",
                })?;
            memory.stimulate(id, activation).map_err(|_| {
                StoreIntegrityError::InvalidActivation {
                    sequence,
                    detail: "activation references an absent episode",
                }
            })?;
        }
        expected += 1;
    }
    if expected != episode_count {
        return Err(StoreIntegrityError::InvalidActivation {
            sequence: u64::try_from(expected).unwrap_or(u64::MAX),
            detail: "activation row is missing",
        }
        .into());
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
    let mut previous = None;
    while let Some(row) = rows.next()? {
        let from = read_u64(row, 0, "relevance_edges", "from_sequence")?;
        let to = read_u64(row, 1, "relevance_edges", "to_sequence")?;
        if previous.is_some_and(|last| (from, to) <= last) {
            return Err(StoreIntegrityError::InvalidRelevance {
                from,
                to,
                detail: "edges are duplicated or not canonically ordered",
            }
            .into());
        }
        previous = Some((from, to));
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
        u64::try_from(index).expect("a persisted episode count always fit an atom sequence")
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
    let mut insert_episode = transaction.prepare(
        "INSERT INTO episodes (
             sequence, occurred_at_ms, recorded_at_ms, source_id
         ) VALUES (?1, ?2, ?3, ?4)",
    )?;
    let mut insert_statement = transaction.prepare(
        "INSERT INTO episode_statements (
             episode_sequence, role, statement_ordinal, predicate_id
         ) VALUES (?1, ?2, ?3, ?4)",
    )?;
    let mut insert_term = transaction.prepare(
        "INSERT INTO statement_terms (
             episode_sequence, role, statement_ordinal, term_ordinal, term_id
         ) VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;

    for episode in memory.episodes().skip(start) {
        let sequence = codec::encode_u64(episode.id().sequence());
        let source = codec::encode_u64(episode.source().get());
        insert_episode.execute((
            sequence.as_slice(),
            episode.occurred_at().get(),
            episode.recorded_at().get(),
            source.as_slice(),
        ))?;

        for (ordinal, statement) in episode.context().iter().enumerate() {
            insert_statement_value(
                &mut insert_statement,
                &mut insert_term,
                &sequence,
                ROLE_CONTEXT,
                ordinal,
                statement,
                episode,
            )?;
        }
        for (role, statement) in [
            (ROLE_OBSERVATION, Some(episode.observation())),
            (ROLE_ACTION, episode.action()),
            (ROLE_OUTCOME, episode.outcome()),
        ] {
            if let Some(statement) = statement {
                insert_statement_value(
                    &mut insert_statement,
                    &mut insert_term,
                    &sequence,
                    role,
                    0,
                    statement,
                    episode,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn insert_statement_value(
    insert_statement: &mut rusqlite::Statement<'_>,
    insert_term: &mut rusqlite::Statement<'_>,
    sequence: &[u8; 8],
    role: i64,
    ordinal: usize,
    statement: &Statement,
    episode: &EpisodeAtom,
) -> Result<(), StoreError> {
    let ordinal = i64::try_from(ordinal).map_err(|_| StoreIntegrityError::InvalidEpisode {
        sequence: episode.id().sequence(),
        detail: "statement ordinal exceeds SQLite INTEGER",
    })?;
    let predicate = codec::encode_u64(statement.predicate().get());
    insert_statement.execute((sequence.as_slice(), role, ordinal, predicate.as_slice()))?;
    for (term_ordinal, term) in statement.arguments().iter().enumerate() {
        let term_ordinal =
            i64::try_from(term_ordinal).map_err(|_| StoreIntegrityError::InvalidEpisode {
                sequence: episode.id().sequence(),
                detail: "term ordinal exceeds SQLite INTEGER",
            })?;
        let term = codec::encode_u64(term.get());
        insert_term.execute((
            sequence.as_slice(),
            role,
            ordinal,
            term_ordinal,
            term.as_slice(),
        ))?;
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
        let sequence = codec::encode_u64(episode.id().sequence());
        let activation = memory
            .activation(episode.id())
            .expect("an in-memory episode always has parallel activation storage");
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

fn read_non_negative_u64(
    row: &Row<'_>,
    index: usize,
    table: &'static str,
    column: &'static str,
    sequence: u64,
) -> Result<u64, StoreError> {
    let value = read_integer(row, index, table, column)?;
    u64::try_from(value).map_err(|_| {
        StoreIntegrityError::InvalidEpisode {
            sequence,
            detail: "negative ordinal",
        }
        .into()
    })
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
            context: vec![
                statement(10 + seed, &[100 + seed]),
                statement(20 + seed, &[200 + seed]),
            ],
            observation: statement(30 + seed, &[300 + seed, 301 + seed]),
            action: None,
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

    #[derive(Clone, Copy, Debug)]
    enum MetadataCorruption {
        ApplicationId,
        MissingRow,
        UnsupportedVersion,
        ShortMemoryId,
        ZeroMemoryId,
    }

    #[test]
    fn metadata_corruption_is_rejected_with_specific_errors() {
        for corruption in [
            MetadataCorruption::ApplicationId,
            MetadataCorruption::MissingRow,
            MetadataCorruption::UnsupportedVersion,
            MetadataCorruption::ShortMemoryId,
            MetadataCorruption::ZeroMemoryId,
        ] {
            let directory = tempdir().expect("temporary directory is available");
            let path = saved_store(&directory, 0);
            let connection = Connection::open(&path).expect("fixture database opens");
            match corruption {
                MetadataCorruption::ApplicationId => connection
                    .pragma_update(None, "application_id", 0)
                    .expect("application ID can be corrupted"),
                MetadataCorruption::MissingRow => {
                    connection
                        .execute("DELETE FROM memory_meta", [])
                        .expect("metadata can be removed");
                }
                MetadataCorruption::UnsupportedVersion => {
                    connection
                        .execute("UPDATE memory_meta SET format_version = 2", [])
                        .expect("future format version fits structural schema");
                }
                MetadataCorruption::ShortMemoryId => {
                    connection
                        .pragma_update(None, "ignore_check_constraints", true)
                        .expect("fixture can bypass checks");
                    connection
                        .execute("UPDATE memory_meta SET memory_id = x'01'", [])
                        .expect("short ID can be injected");
                }
                MetadataCorruption::ZeroMemoryId => {
                    connection
                        .pragma_update(None, "ignore_check_constraints", true)
                        .expect("fixture can bypass checks");
                    connection
                        .execute(
                            "UPDATE memory_meta SET memory_id = ?1",
                            params![[0_u8; 16].as_slice()],
                        )
                        .expect("zero ID can be injected");
                }
            }
            drop(connection);

            let error = integrity_error(&path);
            let expected_variant = match corruption {
                MetadataCorruption::ApplicationId => {
                    matches!(error, StoreIntegrityError::ApplicationMismatch { .. })
                }
                MetadataCorruption::MissingRow => {
                    matches!(error, StoreIntegrityError::MissingMetadata)
                }
                MetadataCorruption::UnsupportedVersion => matches!(
                    error,
                    StoreIntegrityError::UnsupportedFormatVersion { found: 2 }
                ),
                MetadataCorruption::ShortMemoryId => {
                    matches!(error, StoreIntegrityError::InvalidEncoding { .. })
                }
                MetadataCorruption::ZeroMemoryId => {
                    matches!(error, StoreIntegrityError::InvalidMemoryId)
                }
            };
            assert!(
                expected_variant,
                "unexpected error for {corruption:?}: {error}"
            );
        }
    }

    #[test]
    fn rejecting_an_unrelated_database_does_not_change_its_journal_mode() {
        let directory = tempdir().expect("temporary directory is available");
        let path = directory.path().join("unrelated.sqlite3");
        let raw = Connection::open(&path).expect("unrelated database opens");
        raw.execute("CREATE TABLE unrelated (value INTEGER)", [])
            .expect("unrelated schema is created");
        let journal_mode: String = raw
            .query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))
            .expect("WAL mode can be selected for the fixture");
        assert_eq!(journal_mode, "wal");
        drop(raw);

        assert!(matches!(
            SqliteStore::open(&path),
            Err(StoreError::InvalidStore(
                StoreIntegrityError::ApplicationMismatch { found: 0 }
            ))
        ));

        let raw = Connection::open(&path).expect("unrelated database reopens");
        let journal_mode: String = raw
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .expect("journal mode remains readable");
        assert_eq!(journal_mode, "wal");
        raw.pragma_update(None, "journal_mode", "DELETE")
            .expect("fixture leaves no WAL sidecars");
    }

    #[derive(Clone, Copy, Debug)]
    enum StateCorruption {
        SequenceGap,
        MissingObservation,
        EmptyStatement,
        InvalidRole,
        ContextOrdinalGap,
        TermGap,
        NonCanonicalContext,
        MissingActivation,
        ForeignKey,
        BrokenCheck,
        OutgoingBudget,
    }

    #[test]
    fn state_corruption_is_rejected_without_canonicalization_or_repair() {
        for corruption in [
            StateCorruption::SequenceGap,
            StateCorruption::MissingObservation,
            StateCorruption::EmptyStatement,
            StateCorruption::InvalidRole,
            StateCorruption::ContextOrdinalGap,
            StateCorruption::TermGap,
            StateCorruption::NonCanonicalContext,
            StateCorruption::MissingActivation,
            StateCorruption::ForeignKey,
            StateCorruption::BrokenCheck,
            StateCorruption::OutgoingBudget,
        ] {
            let directory = tempdir().expect("temporary directory is available");
            let path = saved_store(&directory, 3);
            let zero = codec::encode_u64(0);
            let one = codec::encode_u64(1);
            let two = codec::encode_u64(2);
            let absent = codec::encode_u64(99);
            let connection = Connection::open(&path).expect("fixture database opens");
            match corruption {
                StateCorruption::SequenceGap => {
                    connection
                        .execute(
                            "DELETE FROM statement_terms WHERE episode_sequence = ?1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                    connection
                        .execute(
                            "DELETE FROM episode_statements WHERE episode_sequence = ?1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                    connection
                        .execute(
                            "DELETE FROM activations WHERE episode_sequence = ?1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                    connection
                        .execute(
                            "DELETE FROM episodes WHERE sequence = ?1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                }
                StateCorruption::MissingObservation => {
                    connection
                        .execute(
                            "DELETE FROM statement_terms
                             WHERE episode_sequence = ?1 AND role = 1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                    connection
                        .execute(
                            "DELETE FROM episode_statements
                             WHERE episode_sequence = ?1 AND role = 1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                }
                StateCorruption::EmptyStatement => {
                    connection
                        .execute(
                            "DELETE FROM statement_terms
                             WHERE episode_sequence = ?1 AND role = 1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                }
                StateCorruption::InvalidRole => {
                    connection
                        .pragma_update(None, "foreign_keys", false)
                        .unwrap();
                    connection
                        .pragma_update(None, "ignore_check_constraints", true)
                        .unwrap();
                    connection
                        .execute(
                            "UPDATE statement_terms
                             SET role = 4
                             WHERE episode_sequence = ?1 AND role = 1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                    connection
                        .execute(
                            "UPDATE episode_statements
                             SET role = 4
                             WHERE episode_sequence = ?1 AND role = 1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                }
                StateCorruption::ContextOrdinalGap => {
                    connection
                        .pragma_update(None, "foreign_keys", false)
                        .unwrap();
                    connection
                        .execute(
                            "UPDATE statement_terms
                             SET statement_ordinal = 2
                             WHERE episode_sequence = ?1
                               AND role = 0
                               AND statement_ordinal = 1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                    connection
                        .execute(
                            "UPDATE episode_statements
                             SET statement_ordinal = 2
                             WHERE episode_sequence = ?1
                               AND role = 0
                               AND statement_ordinal = 1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                }
                StateCorruption::TermGap => {
                    connection
                        .execute(
                            "UPDATE statement_terms
                             SET term_ordinal = 2
                             WHERE episode_sequence = ?1
                               AND role = 1
                               AND term_ordinal = 1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                }
                StateCorruption::NonCanonicalContext => {
                    let high = codec::encode_u64(u64::MAX);
                    let low = codec::encode_u64(0);
                    connection
                        .execute(
                            "UPDATE episode_statements
                             SET predicate_id = ?1
                             WHERE episode_sequence = ?2
                               AND role = 0
                               AND statement_ordinal = 0",
                            params![high.as_slice(), zero.as_slice()],
                        )
                        .unwrap();
                    connection
                        .execute(
                            "UPDATE episode_statements
                             SET predicate_id = ?1
                             WHERE episode_sequence = ?2
                               AND role = 0
                               AND statement_ordinal = 1",
                            params![low.as_slice(), zero.as_slice()],
                        )
                        .unwrap();
                }
                StateCorruption::MissingActivation => {
                    connection
                        .execute(
                            "DELETE FROM activations WHERE episode_sequence = ?1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                }
                StateCorruption::ForeignKey => {
                    connection
                        .pragma_update(None, "foreign_keys", false)
                        .unwrap();
                    connection
                        .execute(
                            "INSERT INTO relevance_edges
                             (from_sequence, to_sequence, weight_ppm)
                             VALUES (?1, ?2, 1)",
                            params![zero.as_slice(), absent.as_slice()],
                        )
                        .unwrap();
                }
                StateCorruption::BrokenCheck => {
                    connection
                        .pragma_update(None, "ignore_check_constraints", true)
                        .unwrap();
                    connection
                        .execute(
                            "UPDATE activations
                             SET activation_ppm = 1000001
                             WHERE episode_sequence = ?1",
                            [zero.as_slice()],
                        )
                        .unwrap();
                }
                StateCorruption::OutgoingBudget => {
                    connection
                        .execute(
                            "INSERT INTO relevance_edges
                             (from_sequence, to_sequence, weight_ppm)
                             VALUES (?1, ?2, 600000), (?1, ?3, 500000)",
                            params![zero.as_slice(), one.as_slice(), two.as_slice()],
                        )
                        .unwrap();
                }
            }
            drop(connection);

            let error = integrity_error(&path);
            let expected_variant = match corruption {
                StateCorruption::SequenceGap => matches!(
                    error,
                    StoreIntegrityError::NonContiguousEpisodeSequence { .. }
                ),
                StateCorruption::EmptyStatement => matches!(
                    error,
                    StoreIntegrityError::InvalidEpisode {
                        sequence: 0,
                        detail: "statement has no terms"
                    }
                ),
                StateCorruption::InvalidRole => {
                    matches!(error, StoreIntegrityError::QuickCheckFailed { .. })
                }
                StateCorruption::ContextOrdinalGap => matches!(
                    error,
                    StoreIntegrityError::InvalidEpisode {
                        sequence: 0,
                        detail: "context ordinals are not contiguous"
                    }
                ),
                StateCorruption::MissingObservation
                | StateCorruption::TermGap
                | StateCorruption::NonCanonicalContext => {
                    matches!(error, StoreIntegrityError::InvalidEpisode { .. })
                }
                StateCorruption::MissingActivation => {
                    matches!(error, StoreIntegrityError::InvalidActivation { .. })
                }
                StateCorruption::ForeignKey => {
                    matches!(error, StoreIntegrityError::ForeignKeyViolation { .. })
                }
                StateCorruption::BrokenCheck => {
                    matches!(error, StoreIntegrityError::QuickCheckFailed { .. })
                }
                StateCorruption::OutgoingBudget => {
                    matches!(error, StoreIntegrityError::InvalidRelevance { .. })
                }
            };
            assert!(
                expected_variant,
                "unexpected error for {corruption:?}: {error}"
            );
        }
    }

    #[test]
    fn late_episode_corruption_discards_local_reconstruction() {
        let directory = tempdir().expect("temporary directory is available");
        let path = saved_store(&directory, 3);
        let two = codec::encode_u64(2);
        let connection = Connection::open(&path).expect("fixture database opens");
        connection
            .execute(
                "DELETE FROM statement_terms
                 WHERE episode_sequence = ?1 AND role = 1",
                [two.as_slice()],
            )
            .unwrap();
        connection
            .execute(
                "DELETE FROM episode_statements
                 WHERE episode_sequence = ?1 AND role = 1",
                [two.as_slice()],
            )
            .unwrap();
        drop(connection);

        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::InvalidEpisode {
                sequence: 2,
                detail: "observation is missing"
            }
        ));
    }

    #[test]
    fn save_rejects_a_changed_persisted_memory_identity() {
        let directory = tempdir().expect("temporary directory is available");
        let path = saved_store(&directory, 1);
        let mut store = SqliteStore::open(&path).expect("base store opens");
        let original_memory_id = store.memory_id();
        let replacement_memory_id = if original_memory_id.get() == 1 {
            MemoryId::new(2).unwrap()
        } else {
            MemoryId::new(1).unwrap()
        };

        let raw = Connection::open(&path).expect("fixture database opens");
        raw.execute(
            "UPDATE memory_meta SET memory_id = ?1 WHERE singleton = 1",
            [replacement_memory_id.to_be_bytes().as_slice()],
        )
        .expect("persisted identity can be changed out of band");
        drop(raw);

        store
            .memory_mut()
            .insert_episode(draft(10))
            .expect("unsaved episode inserts");
        assert!(matches!(
            store.save(),
            Err(StoreError::InvalidStore(
                StoreIntegrityError::InvalidMetadata { .. }
            ))
        ));
        assert_eq!(store.memory().episodes().len(), 2);

        let raw = Connection::open(&path).expect("fixture database reopens");
        raw.execute(
            "UPDATE memory_meta SET memory_id = ?1 WHERE singleton = 1",
            [original_memory_id.to_be_bytes().as_slice()],
        )
        .expect("fixture identity can be restored");
        drop(raw);

        store
            .save()
            .expect("unchanged in-memory state can be retried after restoring identity");
        drop(store);

        let reopened = SqliteStore::open(&path).expect("retried snapshot reopens");
        assert_eq!(reopened.memory_id(), original_memory_id);
        assert_eq!(reopened.memory().episodes().len(), 2);
    }

    #[test]
    fn save_rejects_a_noncanonical_metadata_singleton() {
        let directory = tempdir().expect("temporary directory is available");
        let path = saved_store(&directory, 1);
        let mut store = SqliteStore::open(&path).expect("base store opens");
        let raw = Connection::open(&path).expect("fixture database opens");
        raw.pragma_update(None, "ignore_check_constraints", true)
            .expect("fixture can bypass the singleton check");
        raw.execute("UPDATE memory_meta SET singleton = 2", [])
            .expect("singleton key can be changed out of band");
        drop(raw);

        store
            .memory_mut()
            .insert_episode(draft(10))
            .expect("unsaved episode inserts");
        assert!(matches!(
            store.save(),
            Err(StoreError::InvalidStore(
                StoreIntegrityError::InvalidMetadata { .. }
            ))
        ));
        assert_eq!(store.memory().episodes().len(), 2);

        let raw = Connection::open(&path).expect("fixture database reopens");
        raw.execute("UPDATE memory_meta SET singleton = 1", [])
            .expect("fixture singleton can be restored");
        drop(raw);
        store
            .save()
            .expect("same in-memory changes can be retried after metadata repair");
    }

    #[test]
    fn persistent_trigger_is_rejected_before_open_or_save() {
        const TRIGGER: &str = "CREATE TRIGGER erase_inserted_activation
            AFTER INSERT ON activations
            BEGIN
                DELETE FROM activations
                WHERE episode_sequence = NEW.episode_sequence;
            END;";

        let directory = tempdir().expect("temporary directory is available");
        let path = saved_store(&directory, 1);
        let raw = Connection::open(&path).expect("fixture database opens");
        raw.execute_batch(TRIGGER)
            .expect("persistent trigger installs");
        drop(raw);

        assert!(matches!(
            integrity_error(&path),
            StoreIntegrityError::InvalidMetadata { .. }
        ));

        let raw = Connection::open(&path).expect("fixture database reopens");
        raw.execute("DROP TRIGGER erase_inserted_activation", [])
            .expect("fixture trigger can be removed");
        drop(raw);

        let mut store = SqliteStore::open(&path).expect("canonical store opens");
        store
            .memory_mut()
            .insert_episode(draft(10))
            .expect("unsaved episode inserts");
        let raw = Connection::open(&path).expect("fixture database reopens");
        raw.execute_batch(TRIGGER)
            .expect("persistent trigger reinstalls after open");
        drop(raw);

        assert!(matches!(
            store.save(),
            Err(StoreError::InvalidStore(
                StoreIntegrityError::InvalidMetadata { .. }
            ))
        ));
        assert_eq!(store.memory().episodes().len(), 2);

        let raw = Connection::open(&path).expect("fixture database reopens");
        raw.execute("DROP TRIGGER erase_inserted_activation", [])
            .expect("fixture trigger can be removed");
        drop(raw);
        store
            .save()
            .expect("same in-memory changes can be retried after schema repair");
        drop(store);

        let reopened = SqliteStore::open(&path).expect("retried snapshot reopens");
        assert_eq!(reopened.memory().episodes().len(), 2);
    }

    #[test]
    fn failed_save_rolls_back_revision_atoms_and_mutable_state() {
        let directory = tempdir().expect("temporary directory is available");
        let path = saved_store(&directory, 1);
        let first_id;
        {
            let mut initial = SqliteStore::open(&path).expect("base store opens");
            first_id = initial
                .memory()
                .episodes()
                .next()
                .expect("base episode exists")
                .id();
            initial
                .memory_mut()
                .stimulate(first_id, Activation::from_ppm(123).unwrap())
                .unwrap();
            initial.save().expect("base activation saves");
        }

        let mut store = SqliteStore::open(&path).expect("base store reopens");
        store
            .connection
            .execute_batch(
                "CREATE TEMP TRIGGER abort_activation_insert
             BEFORE INSERT ON main.activations
             BEGIN
                 SELECT RAISE(ABORT, 'injected save failure');
             END;",
            )
            .expect("connection-local abort trigger installs");

        store
            .memory_mut()
            .insert_episode(draft(10))
            .expect("unsaved episode inserts");
        store.memory_mut().reset_activations();
        assert!(matches!(store.save(), Err(StoreError::Database(_))));
        assert_eq!(store.memory().episodes().len(), 2);

        let reopened = SqliteStore::open(&path).expect("previous snapshot remains valid");
        assert_eq!(reopened.memory().episodes().len(), 1);
        assert_eq!(
            reopened.memory().activation(first_id),
            Some(Activation::from_ppm(123).unwrap())
        );
        drop(reopened);

        let revision: i64 = store
            .connection
            .query_row(
                "SELECT snapshot_revision FROM memory_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(revision, 2);
        store
            .connection
            .execute("DROP TRIGGER temp.abort_activation_insert", [])
            .expect("connection-local failure can be removed");

        store
            .save()
            .expect("same in-memory changes can be retried after rollback");
        drop(store);

        let reopened = SqliteStore::open(&path).expect("retried snapshot reopens");
        assert_eq!(reopened.memory().episodes().len(), 2);
        assert_eq!(
            reopened.memory().activation(first_id),
            Some(Activation::ZERO)
        );
    }

    #[test]
    fn exhausted_revision_is_rejected_without_wraparound() {
        let directory = tempdir().expect("temporary directory is available");
        let path = saved_store(&directory, 0);
        let raw = Connection::open(&path).expect("fixture database opens");
        raw.execute(
            "UPDATE memory_meta SET snapshot_revision = ?1 WHERE singleton = 1",
            [i64::MAX],
        )
        .unwrap();
        drop(raw);

        let mut store = SqliteStore::open(&path).expect("maximum revision is structurally valid");
        assert!(matches!(store.save(), Err(StoreError::RevisionExhausted)));
    }
}
