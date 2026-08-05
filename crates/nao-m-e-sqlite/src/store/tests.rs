use std::cell::Cell;
use std::path::{Path, PathBuf};

use nao_m_e::{Attribute, EpisodeDraft, FeedbackTrace, SymbolId, TimestampMs};
use nao_m_e_semantic::{EMBEDDING_DIMENSIONS, Embedding, EpisodeText};
use rusqlite::{Connection, params};
use tempfile::{TempDir, tempdir};

use super::*;

mod contract;

struct FakeEncoder;

impl semantic::EpisodeEncoder for FakeEncoder {
    fn encode_episode(&mut self, _episode: EpisodeText<'_>) -> Result<Embedding, StoreError> {
        Ok(one_hot_embedding(0))
    }
}

fn save_for_test(store: &mut SqliteStore) -> Result<(), StoreError> {
    store.save_with_encoder(&mut FakeEncoder)
}

fn one_hot_embedding(component: usize) -> Embedding {
    let mut values = vec![0; EMBEDDING_DIMENSIONS];
    values[component % EMBEDDING_DIMENSIONS] = 1;
    Embedding::new(values).unwrap()
}

fn attribute(key: u64, values: &[u64]) -> Attribute {
    Attribute::new(
        SymbolId::new(key),
        values.iter().copied().map(SymbolId::new).collect(),
    )
    .expect("test attribute has values")
}

fn draft(seed: u64) -> EpisodeDraft {
    EpisodeDraft::new(
        TimestampMs::new(-i64::try_from(seed).expect("small test seed")),
        vec![attribute(0, &[1, 2]), attribute(3, &[4])],
    )
    .expect("test episode has attributes")
}

fn insert(store: &mut SqliteStore, episode: EpisodeDraft) -> AtomId {
    store
        .intern_symbols(&[
            "first-key".to_owned(),
            "first-value".to_owned(),
            "second-value".to_owned(),
            "second-key".to_owned(),
            "third-value".to_owned(),
        ])
        .unwrap();
    store.memory_mut().insert_episode(episode).unwrap()
}

fn trace(history_bits: u16, sample_count: u8) -> FeedbackTrace {
    FeedbackTrace::from_parts(history_bits, sample_count).expect("test feedback trace is canonical")
}

fn saved_store(directory: &TempDir, episode_count: u64) -> PathBuf {
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).expect("test store is created");
    for seed in 0..episode_count {
        insert(&mut store, draft(seed));
    }
    save_for_test(&mut store).expect("test snapshot saves");
    drop(store);
    path
}

fn check_without_file_mutation(path: &Path) -> Result<(), StoreError> {
    let before = std::fs::read(path).unwrap();
    let result = SqliteStore::check(path);
    assert_eq!(std::fs::read(path).unwrap(), before);
    result
}

fn check_integrity_error(path: &Path) -> StoreIntegrityError {
    match check_without_file_mutation(path) {
        Err(StoreError::InvalidStore(error)) => error,
        Err(error) => panic!("expected persisted-data error, got {error}"),
        Ok(()) => panic!("corrupt store passed its full check"),
    }
}

#[test]
fn episode_vectors_follow_episode_sequences_and_roundtrip() {
    let directory = tempdir().unwrap();
    let path = saved_store(&directory, 2);
    let rows: Vec<(Vec<u8>, i64)> = Connection::open(&path)
        .unwrap()
        .prepare("SELECT sequence, length(vector) FROM episode_vectors ORDER BY sequence")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(format::decode_u64(&rows[0].0), Some(0));
    assert_eq!(format::decode_u64(&rows[1].0), Some(1));
    assert!(
        rows.iter()
            .all(|(_, bytes)| { *bytes == i64::try_from(format::SEMANTIC_VECTOR_BYTES).unwrap() })
    );
    check_without_file_mutation(&path).unwrap();
}

#[test]
fn stale_writer_fails_before_episode_encoding() {
    struct RejectEncoder(Cell<usize>);

    impl semantic::EpisodeEncoder for RejectEncoder {
        fn encode_episode(&mut self, _: EpisodeText<'_>) -> Result<Embedding, StoreError> {
            self.0.set(self.0.get() + 1);
            panic!("stale writer must fail before semantic encoding")
        }
    }

    let directory = tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    drop(SqliteStore::create(&path).unwrap());
    let mut first = SqliteStore::open(&path).unwrap();
    let mut stale = SqliteStore::open(&path).unwrap();
    insert(&mut first, draft(0));
    insert(&mut stale, draft(1));

    save_for_test(&mut first).unwrap();
    let committed = std::fs::read(&path).unwrap();
    let mut encoder = RejectEncoder(Cell::new(0));
    assert!(matches!(
        stale.save_with_encoder(&mut encoder),
        Err(StoreError::ConcurrentModification {
            expected_revision: 0,
            actual_revision: 1
        })
    ));
    assert_eq!(encoder.0.get(), 0);
    assert_eq!(stale.semantic.pending_len(), 0);
    assert_eq!(std::fs::read(path).unwrap(), committed);
}

#[test]
fn later_episode_encoding_failure_stages_nothing_and_changes_no_database_bytes() {
    struct FailSecond {
        calls: usize,
    }

    impl semantic::EpisodeEncoder for FailSecond {
        fn encode_episode(&mut self, _: EpisodeText<'_>) -> Result<Embedding, StoreError> {
            self.calls += 1;
            if self.calls == 2 {
                Err(StoreError::SemanticEncoding {
                    detail: "test failure".to_owned(),
                })
            } else {
                Ok(one_hot_embedding(self.calls))
            }
        }
    }

    let directory = tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).unwrap();
    insert(&mut store, draft(0));
    insert(&mut store, draft(1));
    let before = std::fs::read(&path).unwrap();
    let mut encoder = FailSecond { calls: 0 };

    assert!(matches!(
        store.save_with_encoder(&mut encoder),
        Err(StoreError::SemanticEncoding { .. })
    ));
    assert_eq!(encoder.calls, 2);
    assert_eq!(store.semantic.pending_len(), 0);
    assert_eq!(std::fs::read(&path).unwrap(), before);

    save_for_test(&mut store).unwrap();
    let counts: (i64, i64, i64) = store
        .connection
        .query_row(
            "SELECT snapshot_revision,
                    (SELECT count(*) FROM episodes),
                    (SELECT count(*) FROM episode_vectors)
             FROM memory_meta",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(counts, (1, 2, 2));
}

#[test]
fn failed_vector_insert_retains_preparation_and_retry_does_not_reencode() {
    struct RecordingEncoder {
        calls: usize,
    }

    impl semantic::EpisodeEncoder for RecordingEncoder {
        fn encode_episode(&mut self, _: EpisodeText<'_>) -> Result<Embedding, StoreError> {
            self.calls += 1;
            Ok(one_hot_embedding(self.calls))
        }
    }

    struct RejectEncoder;

    impl semantic::EpisodeEncoder for RejectEncoder {
        fn encode_episode(&mut self, _: EpisodeText<'_>) -> Result<Embedding, StoreError> {
            panic!("retry must reuse prepared episode vectors")
        }
    }

    let directory = tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).unwrap();
    insert(&mut store, draft(0));
    store
        .connection
        .execute_batch(
            "CREATE TEMP TRIGGER abort_vector_insert
             BEFORE INSERT ON main.episode_vectors
             BEGIN SELECT RAISE(ABORT, 'test abort'); END;",
        )
        .unwrap();
    let mut encoder = RecordingEncoder { calls: 0 };

    assert!(store.save_with_encoder(&mut encoder).is_err());
    assert_eq!(encoder.calls, 1);
    assert_eq!(store.semantic.pending_len(), 1);
    let counts: (i64, i64, i64, i64) = store
        .connection
        .query_row(
            "SELECT snapshot_revision,
                    (SELECT count(*) FROM symbols),
                    (SELECT count(*) FROM episodes),
                    (SELECT count(*) FROM episode_vectors)
             FROM memory_meta",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(counts, (0, 0, 0, 0));

    store
        .connection
        .execute_batch("DROP TRIGGER abort_vector_insert")
        .unwrap();
    store.save_with_encoder(&mut RejectEncoder).unwrap();
    assert_eq!(store.semantic.pending_len(), 0);
}

#[test]
fn fast_open_skips_vector_bodies_but_full_check_validates_them() {
    let directory = tempdir().unwrap();
    let path = saved_store(&directory, 1);
    Connection::open(&path)
        .unwrap()
        .execute(
            "UPDATE episode_vectors SET vector = zeroblob(768) WHERE sequence = ?1",
            [format::encode_u64(0).as_slice()],
        )
        .unwrap();

    drop(SqliteStore::open(&path).unwrap());
    assert!(matches!(
        check_integrity_error(&path),
        StoreIntegrityError::InvalidEpisodeVector { sequence: 0, .. }
    ));
}

#[test]
fn interior_vector_gap_is_deferred_to_full_check() {
    let directory = tempdir().unwrap();
    let path = saved_store(&directory, 3);
    Connection::open(&path)
        .unwrap()
        .execute(
            "DELETE FROM episode_vectors WHERE sequence = ?1",
            [format::encode_u64(1).as_slice()],
        )
        .unwrap();

    drop(SqliteStore::open(&path).unwrap());
    assert!(matches!(
        check_integrity_error(&path),
        StoreIntegrityError::NonContiguousEpisodeVector {
            expected: 1,
            found: 2
        }
    ));
}

#[test]
fn profile_and_vector_tail_corruptions_fail_operational_open() {
    for corruption in ["profile", "tail"] {
        let directory = tempdir().unwrap();
        let path = saved_store(&directory, 2);
        let raw = Connection::open(&path).unwrap();
        match corruption {
            "profile" => {
                raw.execute(
                    "UPDATE memory_meta SET semantic_profile_fingerprint = ?1",
                    [[0xa5_u8; 32].as_slice()],
                )
                .unwrap();
            }
            "tail" => {
                raw.execute(
                    "DELETE FROM episode_vectors WHERE sequence = ?1",
                    [format::encode_u64(1).as_slice()],
                )
                .unwrap();
            }
            _ => unreachable!(),
        }
        drop(raw);

        assert!(
            matches!(
                SqliteStore::open(&path),
                Err(StoreError::InvalidStore(
                    StoreIntegrityError::InvalidMetadata { .. }
                ))
            ),
            "operational open accepted {corruption} corruption"
        );
    }
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
    let symbols = store
        .intern_symbols(&["key".to_owned(), "value".to_owned()])
        .unwrap();
    store
        .memory_mut()
        .insert_episode(
            EpisodeDraft::new(
                TimestampMs::new(0),
                vec![Attribute::new(symbols[0], vec![symbols[1]]).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    store
        .connection
        .execute_batch(
            "CREATE TEMP TRIGGER abort_episode_insert
             BEFORE INSERT ON main.episodes
             BEGIN SELECT RAISE(ABORT, 'test abort'); END;",
        )
        .unwrap();

    assert!(save_for_test(&mut store).is_err());
    assert_eq!(store.symbols.pending_len(), 2);
    assert_eq!(store.expected_revision, 0);
    let persisted: (i64, i64, i64) = store
        .connection
        .query_row(
            "SELECT snapshot_revision,
                    (SELECT count(*) FROM symbols),
                    (SELECT count(*) FROM episodes)
             FROM memory_meta",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(persisted, (0, 0, 0));
}

#[test]
fn non_contiguous_symbol_identifiers_are_rejected() {
    let directory = tempdir().unwrap();
    let path = saved_store(&directory, 0);
    let raw = Connection::open(&path).unwrap();
    raw.execute(
        "INSERT INTO symbols (id, value) VALUES (?1, 'gap')",
        [format::encode_u64(1).as_slice()],
    )
    .unwrap();
    drop(raw);

    assert!(matches!(
        integrity_error(&path),
        StoreIntegrityError::NonContiguousSymbolId {
            expected: 0,
            found: 1
        }
    ));
}

#[test]
fn metadata_and_unsupported_formats_fail_closed_with_specific_errors() {
    for unsupported in [1, 2, 3, 4, 5, 6, 7, format::FORMAT_VERSION + 1] {
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
        Some(4),
        Some(5),
        Some(6),
        Some(7),
        Some(format::FORMAT_VERSION),
        Some(format::FORMAT_VERSION + 1),
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

        let result = SqliteStore::open(&path);
        match unsupported_format {
            Some(version) if version != format::FORMAT_VERSION => assert!(matches!(
                result,
                Err(StoreError::InvalidStore(
                    StoreIntegrityError::UnsupportedFormatVersion { found }
                )) if found == version
            )),
            _ => assert!(result.is_err()),
        }
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
    raw.pragma_update(None, "foreign_keys", false).unwrap();
    raw.execute(
        "UPDATE episodes SET sequence = ?1",
        [format::encode_u64(1).as_slice()],
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
    raw.execute(
        "UPDATE episodes SET payload = ?1",
        [[0_u8; format::MIN_EPISODE_PAYLOAD_BYTES].as_slice()],
    )
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
                format::encode_u64(0).as_slice(),
                format::encode_u64(to).as_slice()
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
            format::encode_u64(0).as_slice(),
            format::encode_u64(1).as_slice()
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
    save_for_test(&mut store).unwrap();
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
            format::encode_u64(ids[4].sequence()).as_slice(),
            format::encode_u64(ids[0].sequence()).as_slice()
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
    save_for_test(&mut store).unwrap();

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
    save_for_test(&mut store).unwrap();
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
    save_for_test(&mut store).unwrap();
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

    assert!(save_for_test(&mut store).is_err());
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
        save_for_test(&mut store).unwrap();

        match operation {
            "UPDATE" => {
                store
                    .memory_mut()
                    .set_feedback_trace(first, second, trace(2, 2))
                    .unwrap();
            }
            "DELETE" => {
                let third = insert(&mut store, draft(2));
                save_for_test(&mut store).unwrap();
                let raw = Connection::open(&path).unwrap();
                raw.execute(
                    "INSERT INTO feedback_edges VALUES (?1, ?2, 0, 1)",
                    params![
                        format::encode_u64(second.sequence()).as_slice(),
                        format::encode_u64(third.sequence()).as_slice()
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

        assert!(save_for_test(&mut store).is_err());
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
    save_for_test(&mut store).unwrap();

    let raw = Connection::open(&path).unwrap();
    raw.execute(
        "UPDATE feedback_edges SET history_bits = 0, sample_count = 1",
        [],
    )
    .unwrap();
    raw.execute(
        "INSERT INTO feedback_edges VALUES (?1, ?2, 3, 2)",
        params![
            format::encode_u64(second.sequence()).as_slice(),
            format::encode_u64(third.sequence()).as_slice()
        ],
    )
    .unwrap();
    drop(raw);

    save_for_test(&mut store).unwrap();
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
        save_for_test(&mut store).unwrap();
        let expected_revision = store.expected_revision;
        for seed in persisted_episodes..persisted_episodes + unsaved_episodes {
            insert(&mut store, draft(seed));
        }

        let raw = Connection::open(&path).unwrap();
        corrupt(&raw);
        drop(raw);

        assert!(matches!(
            save_for_test(&mut store),
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
                    format::encode_u64(0).as_slice(),
                    format::encode_u64(1).as_slice()
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
                        format::encode_u64(0).as_slice(),
                        format::encode_u64(1).as_slice(),
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
                    format::encode_u64(0).as_slice(),
                    format::encode_u64(3).as_slice()
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
                    format::encode_u64(0).as_slice(),
                    format::encode_u64(1).as_slice()
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
        [MemoryId::new(1).unwrap().to_be_bytes().as_slice()],
    )
    .unwrap();
    drop(raw);
    assert!(matches!(
        save_for_test(&mut store),
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
    assert!(matches!(
        save_for_test(&mut store),
        Err(StoreError::RevisionExhausted)
    ));
}
