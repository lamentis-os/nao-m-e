use std::collections::{BTreeMap, BTreeSet};

use nao_m_e::{Memory, SymbolId};
use nao_m_e_semantic::{CueText, Embedding, MAX_EMBEDDING_BATCH_SIZE, SemanticEncoder};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OptionalExtension, Row, Transaction, params_from_iter};

use crate::error::{StoreError, StoreIntegrityError};
use crate::format;

use super::{SqliteStore, read_metadata};

const MAX_PAIR_QUERY_BINDINGS: usize = 900;

pub(super) trait CueEncoder {
    fn encode(&mut self, cues: &[CueText<'_>]) -> Result<Vec<Embedding>, StoreError>;
}

impl CueEncoder for SemanticEncoder {
    fn encode(&mut self, cues: &[CueText<'_>]) -> Result<Vec<Embedding>, StoreError> {
        SemanticEncoder::encode(self, cues).map_err(|error| StoreError::SemanticEncoding {
            detail: error.to_string(),
        })
    }
}

#[derive(Debug)]
struct PendingCue {
    cue_id: u64,
    vector: Embedding,
}

#[derive(Debug)]
pub(super) struct SemanticState {
    pub(super) persisted_count: u64,
    pending: BTreeMap<(u64, u64), PendingCue>,
}

impl SemanticState {
    pub(super) const fn new(persisted_count: u64) -> Self {
        Self {
            persisted_count,
            pending: BTreeMap::new(),
        }
    }

    pub(super) fn mark_saved(&mut self) {
        self.persisted_count = self
            .persisted_count
            .checked_add(u64::try_from(self.pending.len()).expect("pending cue count fits in u64"))
            .expect("pending cue identifiers were allocated without overflow");
        self.pending.clear();
    }

    pub(super) fn current_count(&self) -> Result<u64, StoreError> {
        self.persisted_count
            .checked_add(
                u64::try_from(self.pending.len())
                    .map_err(|_| StoreError::SemanticCueIdExhausted)?,
            )
            .ok_or(StoreError::SemanticCueIdExhausted)
    }

    #[cfg(test)]
    pub(super) fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

pub(super) struct PreparedSemantic {
    episode_start: usize,
    cue_ids: Vec<((u64, u64), u64)>,
}

impl PreparedSemantic {
    fn cue_id(&self, pair: (u64, u64)) -> Option<u64> {
        self.cue_ids
            .binary_search_by_key(&pair, |(candidate, _)| *candidate)
            .ok()
            .map(|index| self.cue_ids[index].1)
    }
}

pub(super) fn prepare<E: CueEncoder>(
    store: &mut SqliteStore,
    encoder: &mut E,
) -> Result<PreparedSemantic, StoreError> {
    let start = store.persisted_episode_count;
    let mut unique_pairs = Vec::new();
    let mut seen_pairs = BTreeSet::new();
    for episode in store.memory.episodes().skip(start) {
        for attribute in episode.attributes() {
            let key = attribute.key().get();
            for value in attribute.values() {
                let pair = (key, value.get());
                if seen_pairs.insert(pair) {
                    unique_pairs.push(pair);
                }
            }
        }
    }
    if unique_pairs.is_empty() {
        return Ok(PreparedSemantic {
            episode_start: start,
            cue_ids: Vec::new(),
        });
    }

    verify_current_revision(store)?;

    let pairs = unique_pairs;
    let mut resolved = read_persisted_cues(&store.connection, &pairs)?;
    for (&(key, value), &cue_id) in &resolved {
        let detail = if cue_id >= store.semantic.persisted_count {
            Some("cue identifier lies outside the committed prefix")
        } else if !store.symbols.contains_persisted(key) || !store.symbols.contains_persisted(value)
        {
            Some("cue endpoint lies outside the committed symbol prefix")
        } else {
            None
        };
        if let Some(detail) = detail {
            verify_current_revision(store)?;
            return Err(StoreIntegrityError::InvalidSemanticCue { cue_id, detail }.into());
        }
    }
    for (&pair, pending) in &store.semantic.pending {
        if seen_pairs.contains(&pair) {
            resolved.insert(pair, pending.cue_id);
        }
    }
    drop(seen_pairs);

    let mut missing = pairs;
    missing.retain(|pair| !resolved.contains_key(pair));
    if !missing.is_empty() {
        let first_id = store.semantic.current_count()?;
        first_id
            .checked_add(
                u64::try_from(missing.len()).map_err(|_| StoreError::SemanticCueIdExhausted)?,
            )
            .ok_or(StoreError::SemanticCueIdExhausted)?;

        let mut ids = Vec::with_capacity(missing.len() * 2);
        let mut seen_ids = BTreeSet::new();
        for &(key, value) in &missing {
            if seen_ids.insert(key) {
                ids.push(SymbolId::new(key));
            }
            if seen_ids.insert(value) {
                ids.push(SymbolId::new(value));
            }
        }
        drop(seen_ids);
        let values = store.symbol_values(&ids)?;
        let mut text_by_id = BTreeMap::new();
        for (id, value) in ids.into_iter().zip(values) {
            let id = id.get();
            let value = value.ok_or(StoreError::UnknownSymbolId { id })?;
            text_by_id.insert(id, value);
        }

        let mut embeddings = Vec::with_capacity(missing.len());
        for chunk in missing.chunks(MAX_EMBEDDING_BATCH_SIZE) {
            let cue_text: Vec<_> = chunk
                .iter()
                .map(|&(key, value)| {
                    let key = text_by_id
                        .get(&key)
                        .ok_or(StoreError::UnknownSymbolId { id: key })?;
                    let value = text_by_id
                        .get(&value)
                        .ok_or(StoreError::UnknownSymbolId { id: value })?;
                    Ok(CueText::new(key, value))
                })
                .collect::<Result<_, StoreError>>()?;
            let encoded = encoder.encode(&cue_text)?;
            validate_embeddings(&encoded, chunk.len())?;
            embeddings.extend(encoded);
        }
        drop(text_by_id);
        debug_assert_eq!(embeddings.len(), missing.len());

        for (offset, (pair, vector)) in missing.into_iter().zip(embeddings).enumerate() {
            let cue_id = first_id
                .checked_add(u64::try_from(offset).expect("embedding count fits in u64"))
                .expect("the cue range was checked before mutation");
            resolved.insert(pair, cue_id);
            let previous = store
                .semantic
                .pending
                .insert(pair, PendingCue { cue_id, vector });
            debug_assert!(previous.is_none());
        }
    }

    Ok(PreparedSemantic {
        episode_start: start,
        cue_ids: resolved.into_iter().collect(),
    })
}

fn verify_current_revision(store: &SqliteStore) -> Result<(), StoreError> {
    let (_, actual_revision, _) = read_metadata(&store.connection)?;
    if actual_revision == store.expected_revision {
        Ok(())
    } else {
        Err(StoreError::ConcurrentModification {
            expected_revision: store.expected_revision,
            actual_revision,
        })
    }
}

fn validate_embeddings(embeddings: &[Embedding], expected: usize) -> Result<(), StoreError> {
    if embeddings.len() != expected {
        return Err(StoreError::SemanticEncoding {
            detail: format!(
                "encoder returned {} vectors for {expected} cues",
                embeddings.len()
            ),
        });
    }
    Ok(())
}

fn read_persisted_cues(
    connection: &Connection,
    pairs: &[(u64, u64)],
) -> Result<BTreeMap<(u64, u64), u64>, StoreError> {
    let mut found = BTreeMap::new();
    let pairs_per_chunk = MAX_PAIR_QUERY_BINDINGS / 2;
    let full_end = pairs.len() / pairs_per_chunk * pairs_per_chunk;
    let (full_chunks, remainder) = pairs.split_at(full_end);
    if !full_chunks.is_empty() {
        let sql = pair_lookup_sql(pairs_per_chunk);
        let mut statement = connection.prepare(&sql)?;
        for chunk in full_chunks.chunks_exact(pairs_per_chunk) {
            read_persisted_cue_chunk(&mut statement, chunk, &mut found)?;
        }
    }
    if !remainder.is_empty() {
        let sql = pair_lookup_sql(remainder.len());
        let mut statement = connection.prepare(&sql)?;
        read_persisted_cue_chunk(&mut statement, remainder, &mut found)?;
    }
    Ok(found)
}

fn pair_lookup_sql(pair_count: usize) -> String {
    let mut sql = String::with_capacity(128 + pair_count * 6);
    sql.push_str(
        "SELECT cue_id, key_id, value_id, vector FROM semantic_cues WHERE (key_id, value_id) IN (",
    );
    for index in 0..pair_count {
        if index != 0 {
            sql.push(',');
        }
        sql.push_str("(?,?)");
    }
    sql.push_str(") ORDER BY key_id, value_id");
    sql
}

fn read_persisted_cue_chunk(
    statement: &mut rusqlite::Statement<'_>,
    pairs: &[(u64, u64)],
    found: &mut BTreeMap<(u64, u64), u64>,
) -> Result<(), StoreError> {
    let encoded: Vec<_> = pairs
        .iter()
        .flat_map(|&(key, value)| [format::encode_u64(key), format::encode_u64(value)])
        .collect();
    let mut rows = statement.query(params_from_iter(
        encoded.iter().map(|bytes| bytes.as_slice()),
    ))?;
    while let Some(row) = rows.next()? {
        let (cue_id, pair) = read_cue_row(row)?;
        if found.insert(pair, cue_id).is_some() {
            return Err(StoreIntegrityError::InvalidMetadata {
                detail: "semantic cue lookup returned an unexpected or duplicate pair",
            }
            .into());
        }
    }
    Ok(())
}

fn read_cue_row(row: &Row<'_>) -> Result<(u64, (u64, u64)), StoreError> {
    let cue_id = read_blob_u64(row, 0, "semantic_cues", "cue_id")?;
    let key = read_blob_u64(row, 1, "semantic_cues", "key_id")?;
    let value = read_blob_u64(row, 2, "semantic_cues", "value_id")?;
    let ValueRef::Blob(vector) = row.get_ref(3)? else {
        return Err(StoreIntegrityError::InvalidEncoding {
            table: "semantic_cues",
            column: "vector",
        }
        .into());
    };
    validate_vector(cue_id, vector)?;
    Ok((cue_id, (key, value)))
}

fn read_blob_u64(
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

fn validate_vector(cue_id: u64, bytes: &[u8]) -> Result<(), StoreError> {
    if bytes.len() != format::SEMANTIC_VECTOR_BYTES {
        return Err(StoreIntegrityError::InvalidSemanticCue {
            cue_id,
            detail: "vector has the wrong byte width",
        }
        .into());
    }
    let mut has_non_zero = false;
    for value in vector_components(bytes) {
        if value == i16::MIN {
            return Err(StoreIntegrityError::InvalidSemanticCue {
                cue_id,
                detail: "vector contains a component outside the profile range",
            }
            .into());
        }
        has_non_zero |= value != 0;
    }
    if !has_non_zero {
        return Err(StoreIntegrityError::InvalidSemanticCue {
            cue_id,
            detail: "vector is all zero",
        }
        .into());
    }
    Ok(())
}

fn vector_components(bytes: &[u8]) -> impl Iterator<Item = i16> + '_ {
    bytes
        .chunks_exact(size_of::<i16>())
        .map(|component| i16::from_le_bytes([component[0], component[1]]))
}

fn encode_vector(vector: &Embedding, encoded: &mut Vec<u8>) {
    encoded.clear();
    encoded.extend(
        vector
            .values()
            .iter()
            .flat_map(|component| component.to_le_bytes()),
    );
    debug_assert_eq!(encoded.len(), format::SEMANTIC_VECTOR_BYTES);
}

pub(super) fn verify_tail(
    connection: &Connection,
    state: &SemanticState,
) -> Result<(), StoreError> {
    let tail: Option<Vec<u8>> = connection
        .query_row(
            "SELECT cue_id FROM semantic_cues ORDER BY cue_id DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let actual = tail
        .as_deref()
        .map(|bytes| {
            format::decode_u64(bytes).ok_or(StoreIntegrityError::InvalidEncoding {
                table: "semantic_cues",
                column: "cue_id",
            })
        })
        .transpose()?;
    let expected = state.persisted_count.checked_sub(1);
    if actual == expected {
        Ok(())
    } else {
        Err(StoreIntegrityError::InvalidMetadata {
            detail: "persisted semantic cue tail differs from metadata",
        }
        .into())
    }
}

pub(super) fn insert_pending_cues(
    transaction: &Transaction<'_>,
    state: &SemanticState,
) -> Result<u64, StoreError> {
    if !state.pending.is_empty() {
        let mut pending: Vec<_> = state.pending.iter().collect();
        pending.sort_unstable_by_key(|(_, cue)| cue.cue_id);
        let mut insert = transaction.prepare(
            "INSERT INTO semantic_cues (cue_id, key_id, value_id, vector) VALUES (?1, ?2, ?3, ?4)",
        )?;
        let mut encoded = Vec::with_capacity(format::SEMANTIC_VECTOR_BYTES);
        for (&(key, value), cue) in pending {
            encode_vector(&cue.vector, &mut encoded);
            insert.execute((
                format::encode_u64(cue.cue_id).as_slice(),
                format::encode_u64(key).as_slice(),
                format::encode_u64(value).as_slice(),
                encoded.as_slice(),
            ))?;
        }
    }
    state
        .persisted_count
        .checked_add(u64::try_from(state.pending.len()).expect("pending count fits u64"))
        .ok_or(StoreError::SemanticCueIdExhausted)
}

pub(super) fn insert_postings(
    transaction: &Transaction<'_>,
    prepared: &PreparedSemantic,
    memory: &Memory,
) -> Result<(), StoreError> {
    if !prepared.cue_ids.is_empty() {
        let mut insert =
            transaction.prepare("INSERT INTO episode_cues (sequence, cue_id) VALUES (?1, ?2)")?;
        for episode in memory.episodes().skip(prepared.episode_start) {
            for attribute in episode.attributes() {
                for value in attribute.values() {
                    let pair = (attribute.key().get(), value.get());
                    let cue_id =
                        prepared
                            .cue_id(pair)
                            .ok_or(StoreIntegrityError::InvalidMetadata {
                                detail: "prepared semantic cue lookup is incomplete",
                            })?;
                    insert.execute((
                        format::encode_u64(episode.id().sequence()).as_slice(),
                        format::encode_u64(cue_id).as_slice(),
                    ))?;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn full_audit(
    connection: &Connection,
    memory: &Memory,
    metadata_count: u64,
) -> Result<(), StoreError> {
    let mut statement = connection
        .prepare("SELECT cue_id, key_id, value_id, vector FROM semantic_cues ORDER BY cue_id")?;
    let mut rows = statement.query([])?;
    let mut expected_id = 0_u64;
    while let Some(row) = rows.next()? {
        let (cue_id, _) = read_cue_row(row)?;
        if cue_id != expected_id {
            return Err(StoreIntegrityError::NonContiguousSemanticCueId {
                expected: expected_id,
                found: cue_id,
            }
            .into());
        }
        expected_id = expected_id
            .checked_add(1)
            .ok_or(StoreIntegrityError::InvalidMetadata {
                detail: "semantic cue identifier space is exhausted",
            })?;
    }
    if expected_id != metadata_count {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "semantic cue count differs from the cue catalog",
        }
        .into());
    }
    let unused_cue: Option<Vec<u8>> = connection
        .query_row(
            "SELECT semantic_cues.cue_id
             FROM semantic_cues
             WHERE NOT EXISTS (
                 SELECT 1 FROM episode_cues
                 WHERE episode_cues.cue_id = semantic_cues.cue_id
             )
             ORDER BY semantic_cues.cue_id
             LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(cue_id) = unused_cue {
        let cue_id = format::decode_u64(&cue_id).ok_or(StoreIntegrityError::InvalidEncoding {
            table: "semantic_cues",
            column: "cue_id",
        })?;
        return Err(StoreIntegrityError::InvalidSemanticCue {
            cue_id,
            detail: "cue is not referenced by any episode",
        }
        .into());
    }

    let mut actual_statement = connection.prepare(
        "SELECT episode_cues.sequence, semantic_cues.key_id, semantic_cues.value_id
         FROM episode_cues
         JOIN semantic_cues USING (cue_id)
         ORDER BY episode_cues.sequence, semantic_cues.key_id, semantic_cues.value_id",
    )?;
    let mut rows = actual_statement.query([])?;
    let mut actual = next_posting(&mut rows)?;
    for episode in memory.episodes() {
        let sequence = episode.id().sequence();
        for attribute in episode.attributes() {
            for value in attribute.values() {
                let expected = (sequence, attribute.key().get(), value.get());
                if actual != Some(expected) {
                    return Err(StoreIntegrityError::InvalidSemanticPostings {
                        sequence,
                        detail: "postings do not exactly match the episode attributes",
                    }
                    .into());
                }
                actual = next_posting(&mut rows)?;
            }
        }
    }
    if let Some((sequence, _, _)) = actual {
        return Err(StoreIntegrityError::InvalidSemanticPostings {
            sequence,
            detail: "postings contain an episode or cue absent from memory",
        }
        .into());
    }
    Ok(())
}

fn next_posting(rows: &mut rusqlite::Rows<'_>) -> Result<Option<(u64, u64, u64)>, StoreError> {
    rows.next()?
        .map(|row| {
            Ok((
                read_blob_u64(row, 0, "episode_cues", "sequence")?,
                read_blob_u64(row, 1, "semantic_cues", "key_id")?,
                read_blob_u64(row, 2, "semantic_cues", "value_id")?,
            ))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use nao_m_e_semantic::{EMBEDDING_DIMENSIONS, Embedding};

    use super::{encode_vector, validate_vector, vector_components};

    #[test]
    fn persisted_vector_codec_is_little_endian_and_roundtrips_extremes() {
        let mut values = vec![0_i16; EMBEDDING_DIMENSIONS];
        values[..5].copy_from_slice(&[-32_767, -1, 0, 1, i16::MAX]);
        let embedding = Embedding::new(values).unwrap();
        let mut encoded = Vec::new();

        encode_vector(&embedding, &mut encoded);

        assert_eq!(
            &encoded[..10],
            &[0x01, 0x80, 0xff, 0xff, 0x00, 0x00, 0x01, 0x00, 0xff, 0x7f]
        );
        validate_vector(0, &encoded).unwrap();
        assert_eq!(
            vector_components(&encoded).take(5).collect::<Vec<_>>(),
            [-32_767, -1, 0, 1, i16::MAX]
        );

        encoded[..2].copy_from_slice(&i16::MIN.to_le_bytes());
        assert!(validate_vector(0, &encoded).is_err());
    }
}
