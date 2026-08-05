use std::cell::Cell;

use nao_m_e::{Attribute, EpisodeDraft, SymbolId, TimestampMs};
use nao_m_e_semantic::{EMBEDDING_DIMENSIONS, EpisodeText};
use rusqlite::params;
use tempfile::{TempDir, tempdir};

use crate::format;

use super::*;

struct FixtureEpisodeEncoder;

impl semantic::EpisodeEncoder for FixtureEpisodeEncoder {
    fn encode_episode(&mut self, episode: EpisodeText<'_>) -> Result<Embedding, StoreError> {
        let value = episode
            .attributes()
            .iter()
            .find_map(|(key, value)| (*key == "vector").then_some(*value))
            .expect("fixture episode has a vector selector");
        let components = match value {
            "exact-a" | "exact-b" => &[1, 0][..],
            "high" => &[4, 3][..],
            "medium" => &[3, 4][..],
            "orthogonal" => &[0, 1][..],
            "negative" => &[-1, 0][..],
            other => panic!("unexpected fixture vector {other:?}"),
        };
        Ok(embedding(components))
    }
}

struct FixtureQueryEncoder {
    embedding: Embedding,
    calls: usize,
    last_query: Option<String>,
}

impl FixtureQueryEncoder {
    fn new(components: &[i16]) -> Self {
        Self {
            embedding: embedding(components),
            calls: 0,
            last_query: None,
        }
    }
}

impl QueryEncoder for FixtureQueryEncoder {
    fn encode_query(&mut self, query: QueryText<'_>) -> Result<Embedding, StoreError> {
        self.calls += 1;
        self.last_query = Some(query.value().to_owned());
        Ok(self.embedding.clone())
    }
}

struct RejectCalls(Cell<usize>);

impl QueryEncoder for RejectCalls {
    fn encode_query(&mut self, _: QueryText<'_>) -> Result<Embedding, StoreError> {
        self.0.set(self.0.get() + 1);
        panic!("query encoder must not be called")
    }
}

fn embedding(components: &[i16]) -> Embedding {
    let mut values = vec![0; EMBEDDING_DIMENSIONS];
    values[..components.len()].copy_from_slice(components);
    Embedding::new(values).expect("fixture embedding is canonical")
}

fn encoded(components: &[i16]) -> Vec<u8> {
    embedding(components)
        .values()
        .iter()
        .flat_map(|component| component.to_le_bytes())
        .collect()
}

fn hit(memory_id: MemoryId, sequence: u64, activation_ppm: u32) -> RecallHit {
    RecallHit {
        atom_id: AtomId::from_parts(memory_id, sequence),
        activation: Activation::from_ppm(activation_ppm).unwrap(),
    }
}

fn insert_episode(store: &mut SqliteStore, timestamp: i64, selector: SymbolId, value: SymbolId) {
    store
        .memory_mut()
        .insert_episode(
            EpisodeDraft::new(
                TimestampMs::new(timestamp),
                vec![Attribute::new(selector, vec![value]).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
}

fn fixture_store() -> (TempDir, SqliteStore) {
    let directory = tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).unwrap();
    let ids = store
        .intern_symbols(
            &[
                "vector",
                "exact-a",
                "high",
                "medium",
                "orthogonal",
                "negative",
                "exact-b",
            ]
            .map(str::to_owned),
        )
        .unwrap();
    for (timestamp, value) in ids[1..].iter().copied().enumerate() {
        insert_episode(&mut store, timestamp as i64, ids[0], value);
    }
    store.save_with_encoder(&mut FixtureEpisodeEncoder).unwrap();
    (directory, store)
}

#[test]
fn cosine_projection_is_exact_and_nonpositive_scores_are_excluded() {
    let query = embedding(&[1, 0]);
    let query_norm = squared_norm(query.values());
    assert_eq!(query_norm, 1);
    for (candidate, expected) in [
        (&[1, 0][..], 1_000_000),
        (&[4, 3][..], 800_000),
        (&[3, 4][..], 600_000),
        (&[0, 1][..], 0),
        (&[-1, 0][..], 0),
    ] {
        assert_eq!(
            cosine_ppm(0, query.values(), query_norm, &encoded(candidate)).unwrap(),
            expected
        );
    }
}

#[test]
fn normalization_ranking_ties_limits_and_nonmutation_are_exact() {
    let (_directory, mut store) = fixture_store();
    let path = store.connection.path().unwrap().to_owned();
    let before = std::fs::read(&path).unwrap();
    let mut encoder = FixtureQueryEncoder::new(&[1, 0]);

    let hits = recall(&mut store, &mut encoder, "  Focused\tQUERY  ", 99).unwrap();

    assert_eq!(encoder.calls, 1);
    assert_eq!(encoder.last_query.as_deref(), Some("focused query"));
    assert_eq!(
        hits.iter()
            .map(|hit| (hit.atom_id.sequence(), hit.activation.as_ppm()))
            .collect::<Vec<_>>(),
        [(0, 1_000_000), (5, 1_000_000), (1, 800_000), (2, 600_000)]
    );
    for limit in [1, 2, 3, 4, 10] {
        let prefix = recall(&mut store, &mut encoder, "focused query", limit).unwrap();
        assert_eq!(prefix, hits[..hits.len().min(limit)]);
    }
    assert_eq!(std::fs::read(path).unwrap(), before);
}

#[test]
fn bounded_heap_replaces_the_cutoff_and_retains_smaller_atom_ids_on_ties() {
    let memory_id = MemoryId::new(1).unwrap();
    let mut scores = BoundedHits::new(2);
    scores.push(hit(memory_id, 0, 900_000));
    scores.push(hit(memory_id, 1, 500_000));
    scores.push(hit(memory_id, 2, 800_000));
    assert_eq!(
        scores
            .finish()
            .iter()
            .map(|hit| (hit.atom_id.sequence(), hit.activation.as_ppm()))
            .collect::<Vec<_>>(),
        [(0, 900_000), (2, 800_000)]
    );

    let mut ties = BoundedHits::new(2);
    ties.push(hit(memory_id, 9, 500_000));
    ties.push(hit(memory_id, 8, 500_000));
    ties.push(hit(memory_id, 7, 500_000));
    ties.push(hit(memory_id, 10, 500_000));
    assert_eq!(
        ties.finish()
            .iter()
            .map(|hit| hit.atom_id.sequence())
            .collect::<Vec<_>>(),
        [7, 8]
    );
}

#[test]
fn zero_limit_and_empty_store_validate_without_encoding() {
    let (_directory, mut store) = fixture_store();
    let mut encoder = RejectCalls(Cell::new(0));
    assert!(
        recall(&mut store, &mut encoder, "valid", 0)
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        recall(&mut store, &mut encoder, " \t\n ", 0),
        Err(StoreError::InvalidSemanticQuery { .. })
    ));

    let directory = tempdir().unwrap();
    let mut empty = SqliteStore::create(directory.path().join("empty.sqlite3")).unwrap();
    assert!(
        recall(&mut empty, &mut encoder, "valid", 10)
            .unwrap()
            .is_empty()
    );
    assert_eq!(encoder.0.get(), 0);
}

#[test]
fn vector_scan_uses_primary_key_order_without_a_temp_sort() {
    let (_directory, store) = fixture_store();
    let plan = store
        .connection
        .prepare(
            "EXPLAIN QUERY PLAN
             SELECT sequence, vector FROM episode_vectors ORDER BY sequence",
        )
        .unwrap()
        .query_map([], |row| row.get::<_, String>(3))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(
        plan.iter().all(|detail| !detail.contains("TEMP B-TREE")),
        "episode-vector scan unexpectedly uses a temporary sort: {plan:?}"
    );
}

#[test]
fn stale_reader_is_rejected_before_query_encoding() {
    let (directory, store) = fixture_store();
    let path = directory.path().join("memory.sqlite3");
    drop(store);
    let mut stale = SqliteStore::open(&path).unwrap();
    let mut writer = SqliteStore::open(&path).unwrap();
    let source = AtomId::from_parts(writer.memory_id(), 0);
    let target = AtomId::from_parts(writer.memory_id(), 1);
    writer
        .memory_mut()
        .apply_feedback(source, &[target], true)
        .unwrap();
    writer
        .save_with_encoder(&mut FixtureEpisodeEncoder)
        .unwrap();

    let mut encoder = FixtureQueryEncoder::new(&[1, 0]);
    assert!(matches!(
        recall(&mut stale, &mut encoder, "focused", 10),
        Err(StoreError::ConcurrentModification {
            expected_revision: 1,
            actual_revision: 2
        })
    ));
    assert_eq!(encoder.calls, 0);
}

#[test]
fn revision_change_during_query_encoding_is_rejected_before_vector_scan() {
    struct CommitRevision<'a> {
        writer: &'a mut SqliteStore,
        calls: usize,
    }

    impl QueryEncoder for CommitRevision<'_> {
        fn encode_query(&mut self, _: QueryText<'_>) -> Result<Embedding, StoreError> {
            self.calls += 1;
            self.writer
                .save_with_encoder(&mut FixtureEpisodeEncoder)
                .unwrap();
            Ok(embedding(&[1, 0]))
        }
    }

    let (directory, store) = fixture_store();
    let path = directory.path().join("memory.sqlite3");
    drop(store);
    let mut reader = SqliteStore::open(&path).unwrap();
    let mut writer = SqliteStore::open(&path).unwrap();
    writer
        .intern_symbols(&["concurrent-symbol".to_owned()])
        .unwrap();
    let mut encoder = CommitRevision {
        writer: &mut writer,
        calls: 0,
    };

    assert!(matches!(
        recall(&mut reader, &mut encoder, "focused", 10),
        Err(StoreError::ConcurrentModification {
            expected_revision: 1,
            actual_revision: 2
        })
    ));
    assert_eq!(encoder.calls, 1);
    assert_eq!(reader.expected_revision, 1);
}

#[test]
fn episode_feedback_does_not_change_semantic_ranking() {
    let (_directory, mut store) = fixture_store();
    let mut encoder = FixtureQueryEncoder::new(&[1, 0]);
    let before = recall(&mut store, &mut encoder, "focused", 10).unwrap();
    let source = AtomId::from_parts(store.memory_id(), 3);
    let target = AtomId::from_parts(store.memory_id(), 5);
    store
        .memory_mut()
        .apply_feedback(source, &[target], true)
        .unwrap();
    store.save_with_encoder(&mut FixtureEpisodeEncoder).unwrap();
    assert_eq!(
        recall(&mut store, &mut encoder, "focused", 10).unwrap(),
        before
    );
}

#[test]
fn corrupt_or_noncontiguous_vectors_fail_closed_without_recall_mutation() {
    for corruption in ["zero", "gap"] {
        let (_directory, mut store) = fixture_store();
        match corruption {
            "zero" => {
                store
                    .connection
                    .execute(
                        "UPDATE episode_vectors SET vector = zeroblob(768) WHERE sequence = ?1",
                        [format::encode_u64(0).as_slice()],
                    )
                    .unwrap();
            }
            "gap" => {
                store
                    .connection
                    .execute(
                        "DELETE FROM episode_vectors WHERE sequence = ?1",
                        [format::encode_u64(1).as_slice()],
                    )
                    .unwrap();
            }
            _ => unreachable!(),
        }
        let path = store.connection.path().unwrap().to_owned();
        let before = std::fs::read(&path).unwrap();
        let mut encoder = FixtureQueryEncoder::new(&[1, 0]);
        let result = recall(&mut store, &mut encoder, "focused", 10);
        assert!(matches!(
            (corruption, result),
            (
                "zero",
                Err(StoreError::InvalidStore(
                    StoreIntegrityError::InvalidEpisodeVector { sequence: 0, .. }
                ))
            ) | (
                "gap",
                Err(StoreError::InvalidStore(
                    StoreIntegrityError::NonContiguousEpisodeVector {
                        expected: 1,
                        found: 2
                    }
                ))
            )
        ));
        assert_eq!(std::fs::read(path).unwrap(), before);
    }
}

#[test]
fn malformed_late_candidates_fail_closed_after_the_heap_cutoff_is_full() {
    for corruption in ["wrong-width", "zero", "minimum"] {
        let (_directory, mut store) = fixture_store();
        let sequence = format::encode_u64(5);
        let (vector, detail) = match corruption {
            "wrong-width" => (
                vec![0_u8; format::SEMANTIC_VECTOR_BYTES - 1],
                "vector has the wrong byte width",
            ),
            "zero" => (
                vec![0_u8; format::SEMANTIC_VECTOR_BYTES],
                "vector is all zero",
            ),
            "minimum" => {
                let mut vector = vec![0_u8; format::SEMANTIC_VECTOR_BYTES];
                vector[..size_of::<i16>()].copy_from_slice(&i16::MIN.to_le_bytes());
                (
                    vector,
                    "vector contains a component outside the profile range",
                )
            }
            _ => unreachable!(),
        };
        if corruption == "wrong-width" {
            store
                .connection
                .pragma_update(None, "ignore_check_constraints", true)
                .unwrap();
        }
        store
            .connection
            .execute(
                "UPDATE episode_vectors SET vector = ?1 WHERE sequence = ?2",
                params![vector, sequence.as_slice()],
            )
            .unwrap();
        store
            .connection
            .pragma_update(None, "ignore_check_constraints", false)
            .unwrap();
        let path = store.connection.path().unwrap().to_owned();
        let before = std::fs::read(&path).unwrap();
        let mut encoder = FixtureQueryEncoder::new(&[1, 0]);

        match recall(&mut store, &mut encoder, "focused", 1) {
            Err(StoreError::InvalidStore(StoreIntegrityError::InvalidEpisodeVector {
                sequence: 5,
                detail: found,
            })) => assert_eq!(found, detail, "unexpected {corruption} diagnostic"),
            result => panic!("late {corruption} candidate did not fail closed: {result:?}"),
        }
        assert_eq!(encoder.calls, 1);
        assert_eq!(std::fs::read(path).unwrap(), before);
    }
}

#[test]
fn vector_for_a_pending_episode_cannot_join_the_committed_view() {
    let (_directory, mut store) = fixture_store();
    let ids = store
        .intern_symbols(&["vector".to_owned(), "exact-a".to_owned()])
        .unwrap();
    insert_episode(&mut store, 6, ids[0], ids[1]);
    store
        .connection
        .pragma_update(None, "foreign_keys", false)
        .unwrap();
    store
        .connection
        .execute(
            "INSERT INTO episode_vectors (sequence, vector) VALUES (?1, ?2)",
            params![format::encode_u64(6).as_slice(), encoded(&[1, 0])],
        )
        .unwrap();
    store
        .connection
        .pragma_update(None, "foreign_keys", true)
        .unwrap();

    let mut encoder = FixtureQueryEncoder::new(&[1, 0]);
    assert!(matches!(
        recall(&mut store, &mut encoder, "focused", 10),
        Err(StoreError::InvalidStore(
            StoreIntegrityError::InvalidEpisodeVector { sequence: 6, .. }
        ))
    ));
}

#[test]
fn squared_norm_is_exact_for_signed_components() {
    assert_eq!(squared_norm(&[-3, 0, 4]), 25);
}
