use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use nao_m_e::{Activation, AtomId, MemoryId, RecallHit, SCALE};
use nao_m_e_semantic::{Embedding, QueryText, SemanticEncoder};
use rusqlite::Connection;

use crate::error::{StoreError, StoreIntegrityError};

use super::{SqliteStore, read_metadata, semantic, symbols};

pub(super) trait QueryEncoder {
    fn encode_query(&mut self, query: QueryText<'_>) -> Result<Embedding, StoreError>;
}

impl QueryEncoder for SemanticEncoder {
    fn encode_query(&mut self, query: QueryText<'_>) -> Result<Embedding, StoreError> {
        SemanticEncoder::encode_query(self, query).map_err(|error| {
            StoreError::SemanticQueryEncoding {
                detail: error.to_string(),
            }
        })
    }
}

pub(super) fn recall<E: QueryEncoder>(
    store: &mut SqliteStore,
    encoder: &mut E,
    query: &str,
    limit: usize,
) -> Result<Vec<RecallHit>, StoreError> {
    let normalized = symbols::normalize_query(query)?;
    if limit == 0 {
        return Ok(Vec::new());
    }

    let memory_id = store.memory.memory_id();
    let expected_revision = store.expected_revision;
    let expected_episode_count = store.persisted_episode_count;
    verify_snapshot(&store.connection, memory_id, expected_revision)?;
    if expected_episode_count == 0 {
        return Ok(Vec::new());
    }

    let query = encoder.encode_query(QueryText::new(&normalized))?;
    let transaction = store.connection.transaction()?;
    verify_snapshot(&transaction, memory_id, expected_revision)?;
    let hits = rank_episode_vectors(
        &transaction,
        memory_id,
        &query,
        expected_episode_count,
        limit,
    )?;
    transaction.commit()?;
    Ok(hits)
}

fn verify_snapshot(
    connection: &Connection,
    memory_id: MemoryId,
    expected_revision: i64,
) -> Result<(), StoreError> {
    let (actual_memory_id, actual_revision) = read_metadata(connection)?;
    if actual_memory_id != memory_id {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "persisted memory ID differs from the owned memory",
        }
        .into());
    }
    if actual_revision != expected_revision {
        return Err(StoreError::ConcurrentModification {
            expected_revision,
            actual_revision,
        });
    }
    Ok(())
}

fn rank_episode_vectors(
    connection: &Connection,
    memory_id: MemoryId,
    query: &Embedding,
    expected_episode_count: usize,
    limit: usize,
) -> Result<Vec<RecallHit>, StoreError> {
    let expected_episode_count = u64::try_from(expected_episode_count).map_err(|_| {
        StoreIntegrityError::InvalidMetadata {
            detail: "episode count does not fit the persisted sequence space",
        }
    })?;
    let query_norm = squared_norm(query.values());
    let mut statement =
        connection.prepare("SELECT sequence, vector FROM episode_vectors ORDER BY sequence")?;
    let mut rows = statement.query([])?;
    let mut ranking = BoundedHits::new(limit);
    let mut expected_sequence = 0_u64;
    while let Some(row) = rows.next()? {
        let sequence = semantic::read_sequence(row)?;
        if sequence != expected_sequence {
            return Err(StoreIntegrityError::NonContiguousEpisodeVector {
                expected: expected_sequence,
                found: sequence,
            }
            .into());
        }
        if sequence >= expected_episode_count {
            return Err(StoreIntegrityError::InvalidEpisodeVector {
                sequence,
                detail: "vector lies outside the committed episode prefix",
            }
            .into());
        }
        let bytes = semantic::read_vector(row)?;
        let score = cosine_ppm(sequence, query.values(), query_norm, bytes)?;
        let atom_id = AtomId::from_parts(memory_id, sequence);
        ranking.push(RecallHit {
            atom_id,
            activation: Activation::from_ppm(score)
                .expect("normalized semantic similarity is bounded by SCALE"),
        });
        expected_sequence =
            expected_sequence
                .checked_add(1)
                .ok_or(StoreIntegrityError::InvalidMetadata {
                    detail: "episode vector sequence space is exhausted",
                })?;
    }
    if expected_sequence != expected_episode_count {
        return Err(StoreIntegrityError::InvalidMetadata {
            detail: "episode-vector count differs from the committed episode prefix",
        }
        .into());
    }
    Ok(ranking.finish())
}

fn squared_norm(values: &[i16]) -> u64 {
    values
        .iter()
        .map(|&value| {
            let value = i64::from(value);
            u64::try_from(value * value).expect("a squared i16 is non-negative")
        })
        .sum()
}

fn cosine_ppm(
    sequence: u64,
    query: &[i16],
    query_norm: u64,
    candidate: &[u8],
) -> Result<u32, StoreError> {
    let mut query = query.iter().copied();
    let (dot, candidate_norm) = semantic::fold_vector(
        sequence,
        candidate,
        (0_i64, 0_u64),
        |(dot, norm), candidate| {
            let query = i64::from(query.next().expect("profile dimensions agree"));
            let candidate = i64::from(candidate);
            (
                dot + query * candidate,
                norm.checked_add(
                    u64::try_from(candidate * candidate).expect("a squared i16 is non-negative"),
                )
                .expect("a canonical embedding norm fits u64"),
            )
        },
    )?;
    debug_assert_eq!(query.next(), None);
    if dot <= 0 {
        return Ok(0);
    }
    let denominator = (u128::from(query_norm) * u128::from(candidate_norm)).isqrt();
    debug_assert_ne!(denominator, 0);
    let score = u128::try_from(dot).expect("positive dot product fits u128") * u128::from(SCALE)
        / denominator;
    Ok(u32::try_from(score.min(u128::from(SCALE))).expect("semantic score is bounded by SCALE"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RankedHit(RecallHit);

impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .activation
            .cmp(&other.0.activation)
            .then_with(|| other.0.atom_id.cmp(&self.0.atom_id))
    }
}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct BoundedHits {
    limit: usize,
    best: BinaryHeap<Reverse<RankedHit>>,
}

impl BoundedHits {
    fn new(limit: usize) -> Self {
        debug_assert_ne!(limit, 0);
        Self {
            limit,
            best: BinaryHeap::new(),
        }
    }

    fn push(&mut self, hit: RecallHit) {
        if hit.activation == Activation::ZERO {
            return;
        }
        let ranked = RankedHit(hit);
        if self.best.len() < self.limit {
            self.best.push(Reverse(ranked));
            return;
        }
        if self
            .best
            .peek()
            .is_some_and(|Reverse(worst)| ranked > *worst)
        {
            self.best.pop();
            self.best.push(Reverse(ranked));
        }
    }

    fn finish(self) -> Vec<RecallHit> {
        self.best
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(hit)| hit)
            .map(|hit| hit.0)
            .collect()
    }
}

#[cfg(test)]
mod tests;
