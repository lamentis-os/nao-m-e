use std::path::{Path, PathBuf};

use nao_m_e::{Attribute, EpisodeDraft, FeedbackTrace, SymbolId, TimestampMs};
use nao_m_e_semantic::{CueText, EMBEDDING_DIMENSIONS, Embedding};
use rusqlite::{Connection, params};
use tempfile::{TempDir, tempdir};

use super::*;

mod contract;

struct FakeEncoder;

impl semantic::CueEncoder for FakeEncoder {
    fn encode(&mut self, cues: &[CueText<'_>]) -> Result<Vec<Embedding>, StoreError> {
        Ok(cues
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let mut values = vec![0; EMBEDDING_DIMENSIONS];
                values[index % EMBEDDING_DIMENSIONS] = 1;
                Embedding::new(values).unwrap()
            })
            .collect())
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

fn saved_distinct_semantic_store(directory: &TempDir) -> PathBuf {
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).unwrap();
    let symbols = store
        .intern_symbols(&[
            "first-key".to_owned(),
            "first-value".to_owned(),
            "second-key".to_owned(),
            "second-value".to_owned(),
        ])
        .unwrap();
    for (timestamp, key, value) in [
        (0, symbols[0], symbols[1]),
        (1, symbols[2], symbols[3]),
        (2, symbols[0], symbols[1]),
    ] {
        store
            .memory_mut()
            .insert_episode(
                EpisodeDraft::new(
                    TimestampMs::new(timestamp),
                    vec![Attribute::new(key, vec![value]).unwrap()],
                )
                .unwrap(),
            )
            .unwrap();
    }
    save_for_test(&mut store).unwrap();
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
fn semantic_cues_follow_first_episode_occurrence_and_roundtrip() {
    struct CueRow {
        cue_id: Vec<u8>,
        key_id: Vec<u8>,
        value_id: Vec<u8>,
        vector_bytes: i64,
    }

    let directory = tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).unwrap();
    let symbols = store
        .intern_symbols(&[
            "old-key".to_owned(),
            "old-value".to_owned(),
            "new-key".to_owned(),
            "new-value".to_owned(),
        ])
        .unwrap();
    store
        .memory_mut()
        .insert_episode(
            EpisodeDraft::new(
                TimestampMs::new(1),
                vec![Attribute::new(symbols[2], vec![symbols[3]]).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    store
        .memory_mut()
        .insert_episode(
            EpisodeDraft::new(
                TimestampMs::new(2),
                vec![Attribute::new(symbols[0], vec![symbols[1]]).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();

    save_for_test(&mut store).unwrap();
    let cues: Vec<CueRow> = store
        .connection
        .prepare(
            "SELECT cue_id, key_id, value_id, length(vector) FROM semantic_cues ORDER BY cue_id",
        )
        .unwrap()
        .query_map([], |row| {
            Ok(CueRow {
                cue_id: row.get(0)?,
                key_id: row.get(1)?,
                value_id: row.get(2)?,
                vector_bytes: row.get(3)?,
            })
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(cues.len(), 2);
    assert_eq!(format::decode_u64(&cues[0].cue_id), Some(0));
    assert_eq!(format::decode_u64(&cues[0].key_id), Some(2));
    assert_eq!(format::decode_u64(&cues[0].value_id), Some(3));
    assert_eq!(
        cues[0].vector_bytes,
        i64::try_from(format::SEMANTIC_VECTOR_BYTES).unwrap()
    );
    assert_eq!(format::decode_u64(&cues[1].cue_id), Some(1));
    assert_eq!(format::decode_u64(&cues[1].key_id), Some(0));
    assert_eq!(format::decode_u64(&cues[1].value_id), Some(1));
    drop(store);
    check_without_file_mutation(&path).unwrap();
}

#[test]
fn known_cues_save_without_loading_the_production_encoder() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).unwrap();
    let symbols = store
        .intern_symbols(&["key".to_owned(), "value".to_owned()])
        .unwrap();
    let episode = |timestamp| {
        EpisodeDraft::new(
            TimestampMs::new(timestamp),
            vec![Attribute::new(symbols[0], vec![symbols[1]]).unwrap()],
        )
        .unwrap()
    };
    store.memory_mut().insert_episode(episode(1)).unwrap();
    save_for_test(&mut store).unwrap();
    assert!(!store.encoder.is_loaded());

    store.memory_mut().insert_episode(episode(2)).unwrap();
    store.save().unwrap();
    assert!(!store.encoder.is_loaded());
    drop(store);
    check_without_file_mutation(&path).unwrap();
}

#[test]
fn stale_writer_with_a_new_cue_reports_concurrent_modification() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    drop(SqliteStore::create(&path).unwrap());
    let mut first = SqliteStore::open(&path).unwrap();
    let mut stale = SqliteStore::open(&path).unwrap();

    for store in [&mut first, &mut stale] {
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
    }

    save_for_test(&mut first).unwrap();
    let committed = std::fs::read(&path).unwrap();
    assert!(matches!(
        save_for_test(&mut stale),
        Err(StoreError::ConcurrentModification {
            expected_revision: 0,
            actual_revision: 1
        })
    ));
    assert_eq!(std::fs::read(&path).unwrap(), committed);
    assert_eq!(stale.semantic.pending_len(), 0);
}

#[test]
fn persisted_cue_cannot_be_repaired_by_interning_its_missing_endpoint() {
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
    save_for_test(&mut store).unwrap();
    drop(store);

    let raw = Connection::open(&path).unwrap();
    raw.pragma_update(None, "foreign_keys", false).unwrap();
    raw.execute(
        "UPDATE semantic_cues SET value_id = ?1 WHERE cue_id = ?2",
        params![
            format::encode_u64(2).as_slice(),
            format::encode_u64(0).as_slice()
        ],
    )
    .unwrap();
    drop(raw);

    let mut reopened = SqliteStore::open(&path).unwrap();
    let future = reopened.intern_symbols(&["future".to_owned()]).unwrap()[0];
    assert_eq!(future, SymbolId::new(2));
    reopened
        .memory_mut()
        .insert_episode(
            EpisodeDraft::new(
                TimestampMs::new(1),
                vec![Attribute::new(SymbolId::new(0), vec![future]).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    let before = std::fs::read(&path).unwrap();

    assert!(matches!(
        save_for_test(&mut reopened),
        Err(StoreError::InvalidStore(
            StoreIntegrityError::InvalidSemanticCue { cue_id: 0, .. }
        ))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(reopened.semantic.pending_len(), 0);
}

#[test]
fn semantic_preparation_batches_at_32_and_is_atomic_on_later_failure() {
    struct FailingSecondBatch {
        batch_sizes: Vec<usize>,
    }
    impl semantic::CueEncoder for FailingSecondBatch {
        fn encode(&mut self, cues: &[CueText<'_>]) -> Result<Vec<Embedding>, StoreError> {
            self.batch_sizes.push(cues.len());
            if self.batch_sizes.len() == 2 {
                return Err(StoreError::SemanticEncoding {
                    detail: "test failure".to_owned(),
                });
            }
            Ok((0..cues.len()).map(one_hot_embedding).collect())
        }
    }

    let directory = tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).unwrap();
    let mut names = vec!["key".to_owned()];
    names.extend((0..33).map(|index| format!("value-{index}")));
    let symbols = store.intern_symbols(&names).unwrap();
    store
        .memory_mut()
        .insert_episode(
            EpisodeDraft::new(
                TimestampMs::new(0),
                vec![Attribute::new(symbols[0], symbols[1..].to_vec()).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    let mut encoder = FailingSecondBatch {
        batch_sizes: Vec::new(),
    };

    assert!(store.save_with_encoder(&mut encoder).is_err());
    assert_eq!(encoder.batch_sizes, [32, 1]);
    assert_eq!(store.semantic.pending_len(), 0);
    let counts: (i64, i64, Vec<u8>) = store
        .connection
        .query_row(
            "SELECT (SELECT count(*) FROM semantic_cues),
                    (SELECT count(*) FROM episode_cues), semantic_cue_count
             FROM memory_meta",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(counts.0, 0);
    assert_eq!(counts.1, 0);
    assert_eq!(format::decode_u64(&counts.2), Some(0));
}

#[test]
fn full_check_rejects_an_orphan_semantic_cue() {
    let directory = tempdir().unwrap();
    let path = saved_store(&directory, 1);
    let raw = Connection::open(&path).unwrap();
    let vector = vec![1_u8; format::SEMANTIC_VECTOR_BYTES];
    raw.execute(
        "INSERT INTO semantic_cues (cue_id, key_id, value_id, vector)
         VALUES (?1, ?2, ?3, ?4)",
        params![
            format::encode_u64(3).as_slice(),
            format::encode_u64(0).as_slice(),
            format::encode_u64(4).as_slice(),
            vector,
        ],
    )
    .unwrap();
    raw.execute(
        "UPDATE memory_meta SET semantic_cue_count = ?1",
        [format::encode_u64(4).as_slice()],
    )
    .unwrap();
    drop(raw);

    assert!(matches!(
        check_integrity_error(&path),
        StoreIntegrityError::InvalidSemanticCue { cue_id: 3, .. }
    ));
}

#[test]
fn semantic_posting_corruptions_are_deferred_to_full_check() {
    #[derive(Clone, Copy)]
    enum Corruption {
        Missing,
        Additional,
        WrongCue,
        ForeignKey,
    }

    for (name, corruption) in [
        ("missing", Corruption::Missing),
        ("additional", Corruption::Additional),
        ("wrong-cue", Corruption::WrongCue),
        ("foreign-key", Corruption::ForeignKey),
    ] {
        let directory = tempdir().unwrap();
        let path = saved_distinct_semantic_store(&directory);
        let raw = Connection::open(&path).unwrap();
        raw.pragma_update(None, "foreign_keys", false).unwrap();
        match corruption {
            Corruption::Missing => {
                raw.execute(
                    "DELETE FROM episode_cues WHERE sequence = ?1 AND cue_id = ?2",
                    params![
                        format::encode_u64(0).as_slice(),
                        format::encode_u64(0).as_slice()
                    ],
                )
                .unwrap();
            }
            Corruption::Additional => {
                raw.execute(
                    "INSERT INTO episode_cues (sequence, cue_id) VALUES (?1, ?2)",
                    params![
                        format::encode_u64(0).as_slice(),
                        format::encode_u64(1).as_slice()
                    ],
                )
                .unwrap();
            }
            Corruption::WrongCue => {
                raw.execute(
                    "UPDATE episode_cues SET cue_id = ?1
                     WHERE sequence = ?2 AND cue_id = ?3",
                    params![
                        format::encode_u64(1).as_slice(),
                        format::encode_u64(0).as_slice(),
                        format::encode_u64(0).as_slice()
                    ],
                )
                .unwrap();
                raw.execute(
                    "UPDATE episode_cues SET cue_id = ?1
                     WHERE sequence = ?2 AND cue_id = ?3",
                    params![
                        format::encode_u64(0).as_slice(),
                        format::encode_u64(1).as_slice(),
                        format::encode_u64(1).as_slice()
                    ],
                )
                .unwrap();
            }
            Corruption::ForeignKey => {
                raw.execute(
                    "INSERT INTO episode_cues (sequence, cue_id) VALUES (?1, ?2)",
                    params![
                        format::encode_u64(0).as_slice(),
                        format::encode_u64(2).as_slice()
                    ],
                )
                .unwrap();
            }
        }
        drop(raw);

        let opened = SqliteStore::open(&path).unwrap_or_else(|error| {
            panic!("operational open rejected deferred {name} corruption: {error}")
        });
        drop(opened);
        let error = check_integrity_error(&path);
        match corruption {
            Corruption::ForeignKey => assert!(
                matches!(error, StoreIntegrityError::ForeignKeyCheckFailed { .. }),
                "unexpected {name} error: {error}"
            ),
            _ => assert!(
                matches!(error, StoreIntegrityError::InvalidSemanticPostings { .. }),
                "unexpected {name} error: {error}"
            ),
        }
    }
}

#[test]
fn semantic_metadata_and_tail_corruptions_fail_operational_open() {
    #[derive(Clone, Copy)]
    enum Corruption {
        Profile,
        CueCount,
        CueTail,
    }

    for (name, corruption) in [
        ("profile", Corruption::Profile),
        ("cue-count", Corruption::CueCount),
        ("cue-tail", Corruption::CueTail),
    ] {
        let directory = tempdir().unwrap();
        let path = saved_distinct_semantic_store(&directory);
        let raw = Connection::open(&path).unwrap();
        raw.pragma_update(None, "foreign_keys", false).unwrap();
        match corruption {
            Corruption::Profile => {
                raw.execute(
                    "UPDATE memory_meta SET semantic_profile_fingerprint = ?1",
                    [[0xa5_u8; 32].as_slice()],
                )
                .unwrap();
            }
            Corruption::CueCount => {
                raw.execute(
                    "UPDATE memory_meta SET semantic_cue_count = ?1",
                    [format::encode_u64(3).as_slice()],
                )
                .unwrap();
            }
            Corruption::CueTail => {
                raw.execute(
                    "UPDATE semantic_cues SET cue_id = ?1 WHERE cue_id = ?2",
                    params![
                        format::encode_u64(2).as_slice(),
                        format::encode_u64(1).as_slice()
                    ],
                )
                .unwrap();
                raw.execute(
                    "UPDATE episode_cues SET cue_id = ?1 WHERE cue_id = ?2",
                    params![
                        format::encode_u64(2).as_slice(),
                        format::encode_u64(1).as_slice()
                    ],
                )
                .unwrap();
            }
        }
        drop(raw);

        assert!(
            matches!(
                SqliteStore::open(&path),
                Err(StoreError::InvalidStore(
                    StoreIntegrityError::InvalidMetadata { .. }
                ))
            ),
            "operational open accepted {name} corruption"
        );
        assert!(
            matches!(
                check_integrity_error(&path),
                StoreIntegrityError::InvalidMetadata { .. }
            ),
            "full check returned the wrong {name} error"
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
    for unsupported in [1, 2, 3, 4, 5, 6, format::FORMAT_VERSION + 1] {
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
fn former_semantic_sidecar_identity_is_rejected_without_mutation() {
    const SIDECAR_APPLICATION_ID: i64 = 0x4E41_4F53;

    let directory = tempdir().unwrap();
    let path = saved_store(&directory, 0);
    let raw = Connection::open(&path).unwrap();
    raw.pragma_update(None, "application_id", SIDECAR_APPLICATION_ID)
        .unwrap();
    drop(raw);
    let before = std::fs::read(&path).unwrap();

    assert!(matches!(
        SqliteStore::open(&path),
        Err(StoreError::InvalidStore(
            StoreIntegrityError::ApplicationMismatch {
                found: SIDECAR_APPLICATION_ID
            }
        ))
    ));
    assert!(matches!(
        check_integrity_error(&path),
        StoreIntegrityError::ApplicationMismatch {
            found: SIDECAR_APPLICATION_ID
        }
    ));
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn fast_open_skips_vectors_but_reuse_and_full_check_validate_them() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).unwrap();
    let symbols = store
        .intern_symbols(&["key".to_owned(), "value".to_owned()])
        .unwrap();
    let first = store
        .memory_mut()
        .insert_episode(
            EpisodeDraft::new(
                TimestampMs::new(1),
                vec![Attribute::new(symbols[0], vec![symbols[1]]).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    save_for_test(&mut store).unwrap();
    drop(store);

    let raw = Connection::open(&path).unwrap();
    raw.execute(
        "UPDATE semantic_cues SET vector = zeroblob(768) WHERE cue_id = ?1",
        [format::encode_u64(0).as_slice()],
    )
    .unwrap();
    drop(raw);

    let mut reopened = SqliteStore::open(&path).unwrap();
    assert!(reopened.memory().recall_from(first, 10).unwrap().is_empty());
    reopened
        .memory_mut()
        .insert_episode(
            EpisodeDraft::new(
                TimestampMs::new(2),
                vec![Attribute::new(symbols[0], vec![symbols[1]]).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(matches!(
        save_for_test(&mut reopened),
        Err(StoreError::InvalidStore(
            StoreIntegrityError::InvalidSemanticCue { cue_id: 0, .. }
        ))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    drop(reopened);
    assert!(matches!(
        check_integrity_error(&path),
        StoreIntegrityError::InvalidSemanticCue { cue_id: 0, .. }
    ));
}

#[test]
fn exhausted_semantic_count_fails_before_encoder_work() {
    struct RecordingEncoder {
        called: bool,
    }
    impl semantic::CueEncoder for RecordingEncoder {
        fn encode(&mut self, _cues: &[CueText<'_>]) -> Result<Vec<Embedding>, StoreError> {
            self.called = true;
            Ok(Vec::new())
        }
    }

    let directory = tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).unwrap();
    insert(&mut store, draft(0));
    store.semantic.persisted_count = u64::MAX;
    let mut encoder = RecordingEncoder { called: false };

    assert!(matches!(
        store.save_with_encoder(&mut encoder),
        Err(StoreError::SemanticCueIdExhausted)
    ));
    assert!(!encoder.called);
    assert_eq!(store.semantic.pending_len(), 0);
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
