mod format;

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use nao_m_e::{MemoryId, SymbolId};
use nao_m_e_sqlite::SqliteStore;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, Row, TransactionBehavior, params};

use crate::{CueEmbedder, CueText, EmbeddingProfile, IndexError, IndexIntegrityError, IndexStats};

const EMBEDDING_BATCH_SIZE: usize = 256;

type CueIds = BTreeMap<(u64, u64), u64>;

/// A validated, rebuildable semantic projection of one committed memory.
///
/// The index owns a separate SQLite sidecar. It never changes the authoritative
/// memory database and contains no episode or symbol text.
pub struct SemanticCueIndex {
    connection: Connection,
    memory_id: MemoryId,
    profile: EmbeddingProfile,
    stats: IndexStats,
}

impl SemanticCueIndex {
    /// Builds and publishes a new sidecar for the committed memory snapshot.
    ///
    /// The destination must not exist. Construction uses a staging file in the
    /// destination directory and publishes it only after all cue embeddings and
    /// postings have committed successfully.
    pub fn create<E: CueEmbedder>(
        index_path: impl AsRef<Path>,
        memory_path: impl AsRef<Path>,
        embedder: &mut E,
    ) -> Result<Self, IndexError> {
        let index_path = index_path.as_ref();
        let memory_path = memory_path.as_ref();
        if index_path.try_exists()? {
            return Err(IndexError::Io(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "semantic cue index already exists",
            )));
        }

        let memory = SqliteStore::open(memory_path)?;
        let profile = embedder.profile();
        let parent = index_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let staging = tempfile::Builder::new()
            .prefix(".nao-m-e-semantic-")
            .suffix(".sqlite.tmp")
            .tempfile_in(parent)?;

        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE;
        let mut connection = Connection::open_with_flags(staging.path(), flags)?;
        format::configure_session(&connection)?;
        format::configure_durability(&connection)?;
        format::create_schema(&mut connection, memory.memory_id(), profile)?;
        let mut index = Self {
            connection,
            memory_id: memory.memory_id(),
            profile,
            stats: IndexStats::new(0, 0, 0),
        };
        index.synchronize_store(&memory, embedder)?;
        index.validate_store(&memory, Some(index.stats))?;

        let memory_id = index.memory_id;
        let stats = index.stats;

        index
            .connection
            .close()
            .map_err(|(_, error)| IndexError::Database(error))?;
        staging.as_file().sync_all()?;
        let published = staging
            .persist_noclobber(index_path)
            .map_err(|error| IndexError::Io(error.error))?;
        drop(published);
        Self::reopen_published(index_path, memory_path, memory_id, profile, stats)
    }

    /// Opens and fully validates an existing sidecar against a committed memory.
    ///
    /// A sidecar may cover a strict prefix of the current append-only episode
    /// sequence. Profile changes, divergent memory identity, malformed vectors,
    /// and incomplete cue postings are rejected before an index is returned.
    pub fn open(
        index_path: impl AsRef<Path>,
        memory_path: impl AsRef<Path>,
        profile: EmbeddingProfile,
    ) -> Result<Self, IndexError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        let connection = Connection::open_with_flags(index_path, flags)?;
        format::configure_session(&connection)?;
        verify_application_id(&connection)?;
        verify_format_version(&connection)?;
        format::verify_durability(&connection)?;

        let memory = SqliteStore::open(memory_path)?;
        let mut index = Self {
            connection,
            memory_id: memory.memory_id(),
            profile,
            stats: IndexStats::new(0, 0, 0),
        };
        let validated = index.validate_store(&memory, None)?;
        index.stats = validated.stats;
        Ok(index)
    }

    /// Extends the sidecar through the committed episode tail.
    ///
    /// Only previously unseen bound key/value cues are sent to the embedder.
    /// Model failure, malformed output, stale writers, and SQLite errors leave
    /// the prior committed sidecar unchanged. The memory database is read only.
    pub fn synchronize<E: CueEmbedder>(
        &mut self,
        memory_path: impl AsRef<Path>,
        embedder: &mut E,
    ) -> Result<IndexStats, IndexError> {
        let memory = SqliteStore::open(memory_path)?;
        self.synchronize_store(&memory, embedder)
    }

    /// Returns the committed coverage and row counts of this sidecar session.
    #[must_use]
    pub const fn stats(&self) -> IndexStats {
        self.stats
    }

    fn reopen_published(
        index_path: &Path,
        memory_path: &Path,
        memory_id: MemoryId,
        profile: EmbeddingProfile,
        stats: IndexStats,
    ) -> Result<Self, IndexError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        let connection = Connection::open_with_flags(index_path, flags)?;
        format::configure_session(&connection)?;
        verify_application_id(&connection)?;
        verify_format_version(&connection)?;
        format::verify_durability(&connection)?;
        if !format::validate_schema(&connection)? {
            return Err(IndexIntegrityError::InvalidMetadata {
                detail: "published sidecar schema differs from its validated staging file",
            }
            .into());
        }
        let metadata = read_metadata(&connection)?;
        verify_metadata(&metadata, memory_id, profile)?;
        if metadata.indexed_episode_count != stats.indexed_episode_count() {
            drop(connection);
            return Self::open(index_path, memory_path, profile);
        }
        Ok(Self {
            connection,
            memory_id,
            profile,
            stats,
        })
    }

    fn synchronize_store<E: CueEmbedder>(
        &mut self,
        memory: &SqliteStore,
        embedder: &mut E,
    ) -> Result<IndexStats, IndexError> {
        if embedder.profile() != self.profile {
            return Err(IndexError::ProfileMismatch {
                expected: self.profile,
                found: embedder.profile(),
            });
        }
        let validated = self.validate_store(memory, Some(self.stats))?;
        let current_episode_count = u64::try_from(memory.memory().episodes().len())
            .map_err(|_| IndexError::EpisodeCountExhausted)?;
        if validated.stats.indexed_episode_count() == current_episode_count {
            return Ok(self.stats);
        }

        let previous_stats = validated.stats;
        let old_episode_count = previous_stats.indexed_episode_count();
        let plan = plan_tail(memory, validated, current_episode_count)?;
        let symbol_values = resolve_new_cue_symbols(memory, &plan.new_cues)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        verify_application_id(&transaction)?;
        verify_format_version(&transaction)?;
        if !format::validate_schema(&transaction)? {
            return Err(IndexIntegrityError::InvalidMetadata {
                detail: "sidecar schema changed before synchronization",
            }
            .into());
        }
        let metadata = read_metadata(&transaction)?;
        verify_metadata(&metadata, self.memory_id, self.profile)?;
        if metadata.indexed_episode_count != old_episode_count {
            return Err(IndexError::ConcurrentModification {
                expected_episode_count: old_episode_count,
                actual_episode_count: metadata.indexed_episode_count,
            });
        }

        {
            let mut statement = transaction.prepare(
                "INSERT INTO semantic_cues (cue_id, key_id, value_id, vector)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            let mut vector_bytes = Vec::with_capacity(usize::from(self.profile.dimensions()) * 2);
            for chunk in plan.new_cues.chunks(EMBEDDING_BATCH_SIZE) {
                if embedder.profile() != self.profile {
                    return Err(IndexError::ProfileMismatch {
                        expected: self.profile,
                        found: embedder.profile(),
                    });
                }
                let texts: Vec<_> = chunk
                    .iter()
                    .map(|cue| {
                        CueText::new(
                            resolved_symbol(&symbol_values, cue.key_id),
                            resolved_symbol(&symbol_values, cue.value_id),
                        )
                    })
                    .collect();
                let embeddings = embedder.embed_batch(&texts).map_err(IndexError::Embedder)?;
                if embeddings.len() != chunk.len() {
                    return Err(IndexError::EmbeddingBatchLength {
                        expected: chunk.len(),
                        found: embeddings.len(),
                    });
                }
                for (cue, embedding) in chunk.iter().zip(embeddings) {
                    if embedding.profile() != self.profile {
                        return Err(IndexError::ProfileMismatch {
                            expected: self.profile,
                            found: embedding.profile(),
                        });
                    }
                    format::encode_vector(embedding.values(), &mut vector_bytes);
                    statement.execute(params![
                        format::encode_u64(cue.cue_id).as_slice(),
                        format::encode_u64(cue.key_id).as_slice(),
                        format::encode_u64(cue.value_id).as_slice(),
                        vector_bytes.as_slice(),
                    ])?;
                }
            }
        }

        {
            let mut statement = transaction
                .prepare("INSERT INTO episode_cues (sequence, cue_id) VALUES (?1, ?2)")?;
            for &(sequence, cue_id) in &plan.postings {
                statement.execute(params![
                    format::encode_u64(sequence).as_slice(),
                    format::encode_u64(cue_id).as_slice(),
                ])?;
            }
        }

        let changed = transaction.execute(
            "UPDATE semantic_meta
             SET indexed_episode_count = ?1
             WHERE singleton = 1 AND indexed_episode_count = ?2",
            params![
                format::encode_u64(current_episode_count).as_slice(),
                format::encode_u64(old_episode_count).as_slice(),
            ],
        )?;
        if changed != 1 {
            let actual = read_metadata(&transaction)?.indexed_episode_count;
            return Err(IndexError::ConcurrentModification {
                expected_episode_count: old_episode_count,
                actual_episode_count: actual,
            });
        }
        transaction.commit()?;

        self.stats = IndexStats::new(
            current_episode_count,
            u64::try_from(plan.cue_ids.len()).map_err(|_| IndexError::CueIdExhausted)?,
            previous_stats
                .posting_count()
                .checked_add(
                    u64::try_from(plan.postings.len())
                        .map_err(|_| IndexError::EpisodeCountExhausted)?,
                )
                .ok_or(IndexError::EpisodeCountExhausted)?,
        );
        Ok(self.stats)
    }

    fn validate_store(
        &self,
        memory: &SqliteStore,
        expected_stats: Option<IndexStats>,
    ) -> Result<ValidatedIndex, IndexError> {
        let transaction = self.connection.unchecked_transaction()?;
        verify_application_id(&transaction)?;
        let metadata = read_metadata(&transaction)?;
        verify_metadata(&metadata, memory.memory_id(), self.profile)?;
        if !format::validate_schema(&transaction)? {
            return Err(IndexIntegrityError::InvalidMetadata {
                detail: "database schema differs from the semantic index contract",
            }
            .into());
        }
        verify_quick_check(&transaction)?;
        verify_foreign_keys(&transaction)?;

        let memory_episode_count = u64::try_from(memory.memory().episodes().len())
            .map_err(|_| IndexError::EpisodeCountExhausted)?;
        if metadata.indexed_episode_count > memory_episode_count {
            return Err(IndexIntegrityError::InvalidMetadata {
                detail: "indexed episode count exceeds the committed memory",
            }
            .into());
        }

        let (cue_ids, cue_count) = read_cues(&transaction, self.profile)?;
        let posting_count = validate_postings(
            &transaction,
            memory,
            metadata.indexed_episode_count,
            &cue_ids,
            cue_count,
        )?;
        let stats = IndexStats::new(metadata.indexed_episode_count, cue_count, posting_count);
        if let Some(expected) = expected_stats
            && expected != stats
        {
            return Err(IndexError::ConcurrentModification {
                expected_episode_count: expected.indexed_episode_count(),
                actual_episode_count: stats.indexed_episode_count(),
            });
        }
        transaction.commit()?;
        Ok(ValidatedIndex { cue_ids, stats })
    }
}

struct Metadata {
    memory_id: MemoryId,
    profile: EmbeddingProfile,
    indexed_episode_count: u64,
}

struct ValidatedIndex {
    cue_ids: CueIds,
    stats: IndexStats,
}

struct NewCue {
    cue_id: u64,
    key_id: u64,
    value_id: u64,
}

struct SyncPlan {
    cue_ids: CueIds,
    new_cues: Vec<NewCue>,
    postings: Vec<(u64, u64)>,
}

fn plan_tail(
    memory: &SqliteStore,
    validated: ValidatedIndex,
    current_episode_count: u64,
) -> Result<SyncPlan, IndexError> {
    let mut cue_ids = validated.cue_ids;
    let mut new_cues = Vec::new();
    let mut postings = Vec::new();
    let start = usize::try_from(validated.stats.indexed_episode_count())
        .map_err(|_| IndexError::EpisodeCountExhausted)?;

    for episode in memory.memory().episodes().skip(start) {
        let sequence = episode.id().sequence();
        debug_assert!(sequence < current_episode_count);
        for attribute in episode.attributes() {
            let key_id = attribute.key().get();
            for value in attribute.values() {
                let value_id = value.get();
                let cue_id = if let Some(&cue_id) = cue_ids.get(&(key_id, value_id)) {
                    cue_id
                } else {
                    let cue_id =
                        u64::try_from(cue_ids.len()).map_err(|_| IndexError::CueIdExhausted)?;
                    cue_ids.insert((key_id, value_id), cue_id);
                    new_cues.push(NewCue {
                        cue_id,
                        key_id,
                        value_id,
                    });
                    cue_id
                };
                postings.push((sequence, cue_id));
            }
        }
    }
    postings.sort_unstable();
    Ok(SyncPlan {
        cue_ids,
        new_cues,
        postings,
    })
}

fn resolve_new_cue_symbols(
    memory: &SqliteStore,
    cues: &[NewCue],
) -> Result<Vec<(u64, String)>, IndexError> {
    let mut ids: Vec<_> = cues
        .iter()
        .flat_map(|cue| [SymbolId::new(cue.key_id), SymbolId::new(cue.value_id)])
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let values = memory.symbol_values(&ids)?;
    ids.into_iter()
        .zip(values)
        .map(|(id, value)| {
            let Some(value) = value else {
                return Err(IndexIntegrityError::InvalidCue {
                    cue_id: 0,
                    detail: "planned cue refers to an absent symbol",
                }
                .into());
            };
            Ok((id.get(), value))
        })
        .collect()
}

fn resolved_symbol(symbols: &[(u64, String)], id: u64) -> &str {
    let index = symbols
        .binary_search_by_key(&id, |&(symbol_id, _)| symbol_id)
        .expect("every planned cue symbol was resolved");
    &symbols[index].1
}

fn read_metadata(connection: &Connection) -> Result<Metadata, IndexError> {
    let mut statement = connection.prepare(
        "SELECT format_version, memory_id, profile_fingerprint, dimensions,
                indexed_episode_count
         FROM semantic_meta",
    )?;
    let mut rows = statement.query([])?;
    let Some(row) = rows.next()? else {
        return Err(IndexIntegrityError::MissingMetadata.into());
    };
    let found_format = read_i64(row, 0, "semantic_meta", "format_version")?;
    if found_format != format::FORMAT_VERSION {
        return Err(IndexIntegrityError::UnsupportedFormatVersion {
            found: found_format,
        }
        .into());
    }
    let memory_bytes = read_blob::<16>(row, 1, "semantic_meta", "memory_id")?;
    let memory_id = MemoryId::from_be_bytes(memory_bytes).map_err(|_| {
        IndexIntegrityError::InvalidMetadata {
            detail: "memory identifier is zero",
        }
    })?;
    let fingerprint = read_blob::<32>(row, 2, "semantic_meta", "profile_fingerprint")?;
    let dimensions = read_i64(row, 3, "semantic_meta", "dimensions")?;
    let dimensions = u16::try_from(dimensions)
        .ok()
        .filter(|&value| value != 0)
        .ok_or(IndexIntegrityError::InvalidMetadata {
            detail: "embedding dimensions are outside 1..=65535",
        })?;
    let profile = EmbeddingProfile::new(fingerprint, dimensions).ok_or(
        IndexIntegrityError::InvalidMetadata {
            detail: "embedding profile fingerprint is zero",
        },
    )?;
    let indexed_episode_count = read_u64(row, 4, "semantic_meta", "indexed_episode_count")?;
    if rows.next()?.is_some() {
        return Err(IndexIntegrityError::InvalidMetadata {
            detail: "multiple semantic metadata rows",
        }
        .into());
    }
    Ok(Metadata {
        memory_id,
        profile,
        indexed_episode_count,
    })
}

fn verify_metadata(
    metadata: &Metadata,
    memory_id: MemoryId,
    profile: EmbeddingProfile,
) -> Result<(), IndexError> {
    if metadata.memory_id != memory_id {
        return Err(IndexIntegrityError::MemoryMismatch.into());
    }
    if metadata.profile != profile {
        return Err(IndexError::ProfileMismatch {
            expected: metadata.profile,
            found: profile,
        });
    }
    Ok(())
}

fn read_cues(
    connection: &Connection,
    profile: EmbeddingProfile,
) -> Result<(CueIds, u64), IndexError> {
    let mut statement = connection.prepare(
        "SELECT cue_id, key_id, value_id, vector
         FROM semantic_cues
         ORDER BY cue_id",
    )?;
    let mut rows = statement.query([])?;
    let mut cue_ids = BTreeMap::new();
    let mut expected = 0_u64;
    while let Some(row) = rows.next()? {
        let cue_id = read_u64(row, 0, "semantic_cues", "cue_id")?;
        if cue_id != expected {
            return Err(IndexIntegrityError::InvalidCue {
                cue_id,
                detail: "cue identifiers are not a contiguous zero-based prefix",
            }
            .into());
        }
        let key_id = read_u64(row, 1, "semantic_cues", "key_id")?;
        let value_id = read_u64(row, 2, "semantic_cues", "value_id")?;
        let ValueRef::Blob(vector) = row.get_ref(3)? else {
            return Err(IndexIntegrityError::InvalidEncoding {
                table: "semantic_cues",
                column: "vector",
            }
            .into());
        };
        if !format::validate_vector(vector, profile.dimensions()) {
            return Err(IndexIntegrityError::InvalidCue {
                cue_id,
                detail: "vector length, encoding, or non-zero invariant is invalid",
            }
            .into());
        }
        if cue_ids.insert((key_id, value_id), cue_id).is_some() {
            return Err(IndexIntegrityError::InvalidCue {
                cue_id,
                detail: "bound key/value cue appears more than once",
            }
            .into());
        }
        expected = expected.checked_add(1).ok_or(IndexError::CueIdExhausted)?;
    }
    Ok((cue_ids, expected))
}

fn validate_postings(
    connection: &Connection,
    memory: &SqliteStore,
    indexed_episode_count: u64,
    cue_ids: &CueIds,
    cue_count: u64,
) -> Result<u64, IndexError> {
    let mut statement = connection
        .prepare("SELECT sequence, cue_id FROM episode_cues ORDER BY sequence, cue_id")?;
    let mut rows = statement.query([])?;
    let mut next = rows.next()?;
    let mut used = vec![false; usize::try_from(cue_count).map_err(|_| IndexError::CueIdExhausted)?];
    let mut next_first_cue_id = 0_u64;
    let mut posting_count = 0_u64;
    let take =
        usize::try_from(indexed_episode_count).map_err(|_| IndexError::EpisodeCountExhausted)?;

    for episode in memory.memory().episodes().take(take) {
        let sequence = episode.id().sequence();
        let mut expected = Vec::new();
        for attribute in episode.attributes() {
            for value in attribute.values() {
                let pair = (attribute.key().get(), value.get());
                let Some(&cue_id) = cue_ids.get(&pair) else {
                    return Err(IndexIntegrityError::InvalidPosting {
                        sequence,
                        cue_id: 0,
                        detail: "episode cue has no embedding row",
                    }
                    .into());
                };
                let cue_index = usize::try_from(cue_id).map_err(|_| IndexError::CueIdExhausted)?;
                if !used[cue_index] {
                    if cue_id != next_first_cue_id {
                        return Err(IndexIntegrityError::InvalidCue {
                            cue_id,
                            detail: "cue identifier differs from deterministic first occurrence",
                        }
                        .into());
                    }
                    used[cue_index] = true;
                    next_first_cue_id = next_first_cue_id
                        .checked_add(1)
                        .ok_or(IndexError::CueIdExhausted)?;
                }
                expected.push(cue_id);
            }
        }
        expected.sort_unstable();
        for cue_id in expected {
            let Some(row) = next else {
                return Err(IndexIntegrityError::InvalidPosting {
                    sequence,
                    cue_id,
                    detail: "required episode cue posting is absent",
                }
                .into());
            };
            let found_sequence = read_u64(row, 0, "episode_cues", "sequence")?;
            let found_cue = read_u64(row, 1, "episode_cues", "cue_id")?;
            if (found_sequence, found_cue) != (sequence, cue_id) {
                return Err(IndexIntegrityError::InvalidPosting {
                    sequence: found_sequence,
                    cue_id: found_cue,
                    detail: "posting stream differs from the committed episode cues",
                }
                .into());
            }
            posting_count = posting_count
                .checked_add(1)
                .ok_or(IndexError::EpisodeCountExhausted)?;
            next = rows.next()?;
        }
    }
    if let Some(row) = next {
        return Err(IndexIntegrityError::InvalidPosting {
            sequence: read_u64(row, 0, "episode_cues", "sequence")?,
            cue_id: read_u64(row, 1, "episode_cues", "cue_id")?,
            detail: "posting lies outside the indexed episode prefix",
        }
        .into());
    }
    if let Some(cue_id) = used.iter().position(|used| !used) {
        return Err(IndexIntegrityError::InvalidCue {
            cue_id: u64::try_from(cue_id).map_err(|_| IndexError::CueIdExhausted)?,
            detail: "cue embedding has no episode posting",
        }
        .into());
    }
    Ok(posting_count)
}

fn verify_application_id(connection: &Connection) -> Result<(), IndexError> {
    let found = format::read_application_id(connection)?;
    if found == format::APPLICATION_ID {
        Ok(())
    } else {
        Err(IndexIntegrityError::ApplicationMismatch { found }.into())
    }
}

fn verify_format_version(connection: &Connection) -> Result<(), IndexError> {
    let mut statement = connection.prepare("SELECT format_version FROM semantic_meta")?;
    let mut rows = statement.query([])?;
    let Some(row) = rows.next()? else {
        return Err(IndexIntegrityError::MissingMetadata.into());
    };
    let found = read_i64(row, 0, "semantic_meta", "format_version")?;
    if found != format::FORMAT_VERSION {
        return Err(IndexIntegrityError::UnsupportedFormatVersion { found }.into());
    }
    if rows.next()?.is_some() {
        return Err(IndexIntegrityError::InvalidMetadata {
            detail: "multiple semantic metadata rows",
        }
        .into());
    }
    Ok(())
}

fn verify_quick_check(connection: &Connection) -> Result<(), IndexError> {
    let detail: String = connection.query_row("PRAGMA quick_check", [], |row| row.get(0))?;
    if detail == "ok" {
        Ok(())
    } else {
        Err(IndexIntegrityError::QuickCheckFailed { detail }.into())
    }
}

fn verify_foreign_keys(connection: &Connection) -> Result<(), IndexError> {
    let mut statement = connection.prepare("PRAGMA foreign_key_check")?;
    let mut rows = statement.query([])?;
    if rows.next()?.is_none() {
        Ok(())
    } else {
        Err(IndexIntegrityError::ForeignKeyCheckFailed.into())
    }
}

fn read_i64(
    row: &Row<'_>,
    index: usize,
    table: &'static str,
    column: &'static str,
) -> Result<i64, IndexError> {
    let ValueRef::Integer(value) = row.get_ref(index)? else {
        return Err(IndexIntegrityError::InvalidEncoding { table, column }.into());
    };
    Ok(value)
}

fn read_u64(
    row: &Row<'_>,
    index: usize,
    table: &'static str,
    column: &'static str,
) -> Result<u64, IndexError> {
    Ok(u64::from_be_bytes(read_blob::<8>(
        row, index, table, column,
    )?))
}

fn read_blob<const N: usize>(
    row: &Row<'_>,
    index: usize,
    table: &'static str,
    column: &'static str,
) -> Result<[u8; N], IndexError> {
    let ValueRef::Blob(bytes) = row.get_ref(index)? else {
        return Err(IndexIntegrityError::InvalidEncoding { table, column }.into());
    };
    bytes
        .try_into()
        .map_err(|_| IndexIntegrityError::InvalidEncoding { table, column }.into())
}
