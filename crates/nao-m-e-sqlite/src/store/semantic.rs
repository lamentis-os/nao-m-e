use std::collections::BTreeSet;

use nao_m_e::SymbolId;
use nao_m_e_semantic::{Embedding, EpisodeText, SemanticEncoder};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OptionalExtension, Row, Transaction};

use crate::error::{StoreError, StoreIntegrityError};
use crate::format;

use super::{SqliteStore, read_metadata};

pub(super) trait EpisodeEncoder {
    fn encode_episode(&mut self, episode: EpisodeText<'_>) -> Result<Embedding, StoreError>;
}

impl EpisodeEncoder for SemanticEncoder {
    fn encode_episode(&mut self, episode: EpisodeText<'_>) -> Result<Embedding, StoreError> {
        SemanticEncoder::encode_episode(self, episode).map_err(|error| {
            StoreError::SemanticEncoding {
                detail: error.to_string(),
            }
        })
    }
}

#[derive(Debug, Default)]
pub(super) struct SemanticState {
    pending: Vec<Embedding>,
}

impl SemanticState {
    pub(super) const fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    pub(super) fn mark_saved(&mut self) {
        self.pending.clear();
    }

    #[cfg(test)]
    pub(super) fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

pub(super) fn prepare<E: EpisodeEncoder>(
    store: &mut SqliteStore,
    encoder: &mut E,
) -> Result<(), StoreError> {
    let episode_count = store.memory.episodes().len();
    let first_unprepared = store
        .persisted_episode_count
        .checked_add(store.semantic.pending.len())
        .ok_or(StoreIntegrityError::InvalidMetadata {
            detail: "prepared semantic episode count overflowed",
        })?;
    if first_unprepared > episode_count {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "prepared semantic episode count exceeds in-memory episode count",
        }
        .into());
    }
    if first_unprepared == episode_count {
        return Ok(());
    }

    let mut ids = BTreeSet::new();
    for episode in store.memory.episodes().skip(first_unprepared) {
        for attribute in episode.attributes() {
            let key = attribute.key();
            if ids.insert(key) && !store.symbols.contains_current(key.get()) {
                return Err(StoreError::UnknownSymbolId { id: key.get() });
            }
            for &value in attribute.values() {
                if ids.insert(value) && !store.symbols.contains_current(value.get()) {
                    return Err(StoreError::UnknownSymbolId { id: value.get() });
                }
            }
        }
    }
    verify_current_revision(store)?;
    let ids = ids.into_iter().collect::<Vec<_>>();
    let values = store
        .symbol_values(&ids)?
        .into_iter()
        .zip(&ids)
        .map(|(value, id)| value.ok_or(StoreError::UnknownSymbolId { id: id.get() }))
        .collect::<Result<Vec<_>, _>>()?;

    let mut encoded = Vec::with_capacity(episode_count - first_unprepared);
    for episode in store.memory.episodes().skip(first_unprepared) {
        let mut attributes = Vec::new();
        for attribute in episode.attributes() {
            let key = symbol_text(attribute.key(), &ids, &values)?;
            for &value in attribute.values() {
                attributes.push((key, symbol_text(value, &ids, &values)?));
            }
        }
        let episode =
            EpisodeText::new(&attributes).ok_or(StoreIntegrityError::InvalidMetadata {
                detail: "an episode has no bound semantic attributes",
            })?;
        encoded.push(encoder.encode_episode(episode)?);
    }

    store.semantic.pending.extend(encoded);
    Ok(())
}

fn symbol_text<'a>(
    id: SymbolId,
    ids: &[SymbolId],
    values: &'a [String],
) -> Result<&'a str, StoreError> {
    ids.binary_search(&id)
        .ok()
        .and_then(|index| values.get(index))
        .map(String::as_str)
        .ok_or(StoreError::UnknownSymbolId { id: id.get() })
}

fn verify_current_revision(store: &SqliteStore) -> Result<(), StoreError> {
    let (_, actual_revision) = read_metadata(&store.connection)?;
    if actual_revision == store.expected_revision {
        Ok(())
    } else {
        Err(StoreError::ConcurrentModification {
            expected_revision: store.expected_revision,
            actual_revision,
        })
    }
}

pub(super) fn verify_tail(
    connection: &Connection,
    persisted_episode_count: usize,
) -> Result<(), StoreError> {
    let tail: Option<Vec<u8>> = connection
        .query_row(
            "SELECT sequence FROM episode_vectors ORDER BY sequence DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let actual = tail
        .as_deref()
        .map(|bytes| {
            format::decode_u64(bytes).ok_or(StoreIntegrityError::InvalidEncoding {
                table: "episode_vectors",
                column: "sequence",
            })
        })
        .transpose()?;
    let expected = persisted_episode_count
        .checked_sub(1)
        .map(|sequence| u64::try_from(sequence).expect("episode count fits an atom sequence"));
    if actual == expected {
        Ok(())
    } else {
        Err(StoreIntegrityError::InvalidMetadata {
            detail: "persisted episode-vector tail differs from the episode prefix",
        }
        .into())
    }
}

pub(super) fn insert_pending_vectors(
    transaction: &Transaction<'_>,
    state: &SemanticState,
    episode_start: usize,
) -> Result<(), StoreError> {
    if state.pending.is_empty() {
        return Ok(());
    }
    let mut insert = transaction.prepare(
        "INSERT INTO episode_vectors (sequence, vector)
         VALUES (?1, ?2)",
    )?;
    let mut encoded = Vec::with_capacity(format::SEMANTIC_VECTOR_BYTES);
    for (offset, vector) in state.pending.iter().enumerate() {
        let sequence = episode_start
            .checked_add(offset)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or(StoreIntegrityError::InvalidMetadata {
                detail: "episode vector sequence space is exhausted",
            })?;
        encode_vector(vector, &mut encoded);
        insert.execute((format::encode_u64(sequence).as_slice(), encoded.as_slice()))?;
    }
    Ok(())
}

pub(super) fn full_audit(
    connection: &Connection,
    expected_episode_count: usize,
) -> Result<(), StoreError> {
    let mut statement =
        connection.prepare("SELECT sequence, vector FROM episode_vectors ORDER BY sequence")?;
    let mut rows = statement.query([])?;
    let mut expected = 0_u64;
    while let Some(row) = rows.next()? {
        let sequence = read_sequence(row)?;
        if sequence != expected {
            return Err(StoreIntegrityError::NonContiguousEpisodeVector {
                expected,
                found: sequence,
            }
            .into());
        }
        validate_vector(sequence, read_vector(row)?)?;
        expected = expected
            .checked_add(1)
            .ok_or(StoreIntegrityError::InvalidMetadata {
                detail: "episode vector sequence space is exhausted",
            })?;
    }
    let episode_count = u64::try_from(expected_episode_count).map_err(|_| {
        StoreIntegrityError::InvalidMetadata {
            detail: "episode count does not fit the persisted sequence space",
        }
    })?;
    if expected == episode_count {
        Ok(())
    } else {
        Err(StoreIntegrityError::InvalidMetadata {
            detail: "episode-vector count differs from the episode prefix",
        }
        .into())
    }
}

pub(super) fn read_sequence(row: &Row<'_>) -> Result<u64, StoreError> {
    let ValueRef::Blob(bytes) = row.get_ref(0)? else {
        return Err(StoreIntegrityError::InvalidEncoding {
            table: "episode_vectors",
            column: "sequence",
        }
        .into());
    };
    format::decode_u64(bytes)
        .ok_or(StoreIntegrityError::InvalidEncoding {
            table: "episode_vectors",
            column: "sequence",
        })
        .map_err(Into::into)
}

pub(super) fn read_vector<'a>(row: &'a Row<'_>) -> Result<&'a [u8], StoreError> {
    let ValueRef::Blob(bytes) = row.get_ref(1)? else {
        return Err(StoreIntegrityError::InvalidEncoding {
            table: "episode_vectors",
            column: "vector",
        }
        .into());
    };
    Ok(bytes)
}

pub(super) fn fold_vector<T>(
    sequence: u64,
    bytes: &[u8],
    initial: T,
    mut fold: impl FnMut(T, i16) -> T,
) -> Result<T, StoreError> {
    if bytes.len() != format::SEMANTIC_VECTOR_BYTES {
        return Err(StoreIntegrityError::InvalidEpisodeVector {
            sequence,
            detail: "vector has the wrong byte width",
        }
        .into());
    }
    let mut has_non_zero = false;
    let mut state = initial;
    for value in vector_components(bytes) {
        if value == i16::MIN {
            return Err(StoreIntegrityError::InvalidEpisodeVector {
                sequence,
                detail: "vector contains a component outside the profile range",
            }
            .into());
        }
        has_non_zero |= value != 0;
        state = fold(state, value);
    }
    if !has_non_zero {
        return Err(StoreIntegrityError::InvalidEpisodeVector {
            sequence,
            detail: "vector is all zero",
        }
        .into());
    }
    Ok(state)
}

fn validate_vector(sequence: u64, bytes: &[u8]) -> Result<(), StoreError> {
    fold_vector(sequence, bytes, (), |(), _| ())
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
