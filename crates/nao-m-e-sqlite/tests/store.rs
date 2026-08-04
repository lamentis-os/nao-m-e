use std::fs;
use std::sync::{Arc, Barrier};
use std::thread;

use nao_m_e::{
    AtomId, EpisodeAtom, EpisodeDraft, FeedbackEdge, FeedbackTrace, Memory, MemoryId, PredicateId,
    SourceId, Statement, TermId, TimestampMs,
};
use nao_m_e_sqlite::{SqliteStore, StoreError, StoreIntegrityError};
use rusqlite::Connection;
use tempfile::{TempDir, tempdir};

#[derive(Debug, Eq, PartialEq)]
struct MemorySnapshot {
    memory_id: MemoryId,
    episodes: Vec<EpisodeAtom>,
    feedback_edges: Vec<FeedbackEdge>,
}

fn snapshot(memory: &Memory) -> MemorySnapshot {
    MemorySnapshot {
        memory_id: memory.memory_id(),
        episodes: memory.episodes().cloned().collect(),
        feedback_edges: memory.feedback_edges().collect(),
    }
}

fn database_path(directory: &TempDir) -> std::path::PathBuf {
    directory.path().join("memory.sqlite3")
}

fn statement(predicate: u64, arguments: &[u64]) -> Statement {
    Statement::new(
        PredicateId::new(predicate),
        arguments.iter().copied().map(TermId::new).collect(),
    )
    .expect("test statement has arguments")
}

fn draft(seed: u64) -> EpisodeDraft {
    let timestamp = i64::try_from(seed).expect("test seed fits an i64");
    EpisodeDraft {
        occurred_at: TimestampMs::new(timestamp),
        recorded_at: TimestampMs::new(timestamp + 10),
        context: vec![statement(0, &[0])],
        observation: statement(1, &[1, 2]),
        action: Some(statement(2, &[3])),
        outcome: Some(statement(3, &[4])),
        source: SourceId::new(50 + seed),
    }
}

fn observation_draft(seed: u64, predicate: u64, arguments: &[u64]) -> EpisodeDraft {
    let timestamp = i64::try_from(seed).expect("test seed fits an i64");
    EpisodeDraft {
        occurred_at: TimestampMs::new(timestamp),
        recorded_at: TimestampMs::new(timestamp + 1),
        context: Vec::new(),
        observation: statement(predicate, arguments),
        action: None,
        outcome: None,
        source: SourceId::new(seed + 100),
    }
}

fn insert(store: &mut SqliteStore, episode: EpisodeDraft) -> AtomId {
    let statements = episode
        .context
        .iter()
        .chain(std::iter::once(&episode.observation))
        .chain(episode.action.iter())
        .chain(episode.outcome.iter());
    let predicate_max = statements
        .clone()
        .map(|statement| statement.predicate().get())
        .max()
        .expect("an episode always has an observation");
    let term_max = statements
        .flat_map(|statement| statement.arguments())
        .map(|term| term.get())
        .max()
        .expect("every episode statement has arguments");
    let predicate_values: Vec<_> = (0..=predicate_max)
        .map(|id| format!("predicate-{id:020}"))
        .collect();
    let term_values: Vec<_> = (0..=term_max).map(|id| format!("term-{id:020}")).collect();
    store
        .intern_predicates(&predicate_values)
        .expect("test predicate catalog stages");
    store
        .intern_terms(&term_values)
        .expect("test term catalog stages");
    store
        .memory_mut()
        .insert_episode(episode)
        .expect("test memory has identifier capacity")
}

fn trace(history_bits: u16, sample_count: u8) -> FeedbackTrace {
    FeedbackTrace::from_parts(history_bits, sample_count).expect("test feedback trace is canonical")
}

#[test]
fn symbol_batches_normalize_stage_resolve_and_round_trip() {
    let directory = tempdir().unwrap();
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).unwrap();

    let predicates = store
        .intern_predicates(&[
            "  KELVIN\t\nNAME  ".to_owned(),
            "kelvin name".to_owned(),
            "zeta".to_owned(),
            "alpha".to_owned(),
        ])
        .unwrap();
    let terms = store
        .intern_terms(&[
            "kelvin name".to_owned(),
            "ＣＡＦÉ".to_owned(),
            "café".to_owned(),
            "İ".to_owned(),
            "i\u{0307}".to_owned(),
            "left\u{0085}right".to_owned(),
            "left right".to_owned(),
        ])
        .unwrap();
    assert_eq!(
        predicates,
        [
            PredicateId::new(0),
            PredicateId::new(0),
            PredicateId::new(1),
            PredicateId::new(2)
        ]
    );
    assert_eq!(
        terms,
        [
            TermId::new(0),
            TermId::new(1),
            TermId::new(1),
            TermId::new(2),
            TermId::new(2),
            TermId::new(3),
            TermId::new(3)
        ]
    );
    assert_eq!(
        store
            .predicate_values(&[predicates[0], PredicateId::new(99), predicates[1]])
            .unwrap(),
        [
            Some("kelvin name".to_owned()),
            None,
            Some("kelvin name".to_owned())
        ]
    );
    assert_eq!(
        store
            .term_values(&[terms[0], terms[1], terms[3], terms[5]])
            .unwrap(),
        [
            Some("kelvin name".to_owned()),
            Some("café".to_owned()),
            Some("i\u{0307}".to_owned()),
            Some("left right".to_owned())
        ]
    );

    let raw = Connection::open(&path).unwrap();
    let staged_counts: (i64, i64, i64) = raw
        .query_row(
            "SELECT snapshot_revision,
                    (SELECT count(*) FROM predicates),
                    (SELECT count(*) FROM terms)
             FROM memory_meta",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(staged_counts, (0, 0, 0));
    drop(raw);

    store
        .memory_mut()
        .insert_episode(observation_draft(0, predicates[0].get(), &[terms[0].get()]))
        .unwrap();
    store.save().unwrap();
    drop(store);

    let reopened = SqliteStore::open(&path).unwrap();
    assert_eq!(
        reopened.predicate_values(&[PredicateId::new(0)]).unwrap(),
        [Some("kelvin name".to_owned())]
    );
    assert_eq!(
        reopened
            .term_values(&[
                TermId::new(0),
                TermId::new(1),
                TermId::new(2),
                TermId::new(3)
            ])
            .unwrap(),
        [
            Some("kelvin name".to_owned()),
            Some("café".to_owned()),
            Some("i\u{0307}".to_owned()),
            Some("left right".to_owned())
        ]
    );
}

#[test]
fn invalid_symbol_batch_is_atomic_and_resolve_chunks_large_requests() {
    let directory = tempdir().unwrap();
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).unwrap();

    assert!(store.intern_predicates(&[]).unwrap().is_empty());
    assert!(store.intern_terms(&[]).unwrap().is_empty());
    assert!(store.predicate_values(&[]).unwrap().is_empty());
    assert!(store.term_values(&[]).unwrap().is_empty());

    for (values, expected_detail) in [
        (
            vec!["valid".to_owned(), " \t\n ".to_owned()],
            "normalized value is empty",
        ),
        (
            vec!["valid".to_owned(), "nul\0value".to_owned()],
            "normalized value contains a control character",
        ),
        (
            vec!["valid".to_owned(), "x".repeat(4_097)],
            "normalized UTF-8 value exceeds 4096 bytes",
        ),
    ] {
        assert!(matches!(
            store.intern_predicates(&values),
            Err(StoreError::InvalidSymbolValue { index: 1, detail, .. })
                if detail == expected_detail
        ));
    }

    let repeated_before_invalid = vec![
        "  repeated value  ".to_owned(),
        "  repeated value  ".to_owned(),
        "REPEATED VALUE".to_owned(),
        "another valid value".to_owned(),
        "late\0control".to_owned(),
    ];
    assert!(matches!(
        store.intern_predicates(&repeated_before_invalid),
        Err(StoreError::InvalidSymbolValue {
            namespace: "predicate",
            index: 4,
            detail: "normalized value contains a control character"
        })
    ));

    let values: Vec<_> = (0..1_801).map(|index| format!("symbol-{index}")).collect();
    let ids = store.intern_predicates(&values).unwrap();
    assert_eq!(ids.first(), Some(&PredicateId::new(0)));
    assert_eq!(ids.last(), Some(&PredicateId::new(1_800)));
    store.save().unwrap();
    drop(store);

    let mut reopened = SqliteStore::open(&path).unwrap();
    assert_eq!(
        reopened.intern_predicates(&values[..1_800]).unwrap(),
        ids[..1_800]
    );
    assert_eq!(reopened.intern_predicates(&values).unwrap(), ids);

    let unknown = PredicateId::new(9_999);
    let mut requested = Vec::with_capacity(ids.len() + 6);
    requested.extend([ids[900], unknown, ids[1_800]]);
    requested.extend(ids.iter().rev().copied());
    requested.extend([ids[900], ids[0], unknown]);
    let resolved = reopened.predicate_values(&requested).unwrap();
    assert_eq!(resolved.len(), requested.len());
    for (id, value) in requested.iter().zip(&resolved) {
        let expected = usize::try_from(id.get())
            .ok()
            .and_then(|index| values.get(index));
        assert_eq!(value.as_ref(), expected);
    }
}

#[test]
fn missing_symbol_after_open_fails_without_changing_store_and_resolution_can_retry() {
    let directory = tempdir().unwrap();
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).unwrap();
    let values = ["first".to_owned(), "temporarily missing".to_owned()];
    let ids = store.intern_predicates(&values).unwrap();
    store.save().unwrap();
    drop(store);

    let store = SqliteStore::open(&path).unwrap();
    let memory_before = snapshot(store.memory());
    let raw = Connection::open(&path).unwrap();
    let revision_before: i64 = raw
        .query_row(
            "SELECT snapshot_revision FROM memory_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let missing_id: Vec<u8> = raw
        .query_row(
            "SELECT id FROM predicates WHERE value = ?1",
            [&values[1]],
            |row| row.get(0),
        )
        .unwrap();
    raw.execute("DELETE FROM predicates WHERE id = ?1", [&missing_id])
        .unwrap();

    let requested = [ids[0], ids[1], ids[0]];
    assert!(matches!(
        store.predicate_values(&requested),
        Err(StoreError::InvalidStore(
            StoreIntegrityError::InvalidSymbol {
                namespace: "predicate",
                id: 1,
                detail: "symbol row is absent"
            }
        ))
    ));
    assert_eq!(snapshot(store.memory()), memory_before);

    raw.execute(
        "INSERT INTO predicates (id, value) VALUES (?1, ?2)",
        (missing_id, &values[1]),
    )
    .unwrap();
    let revision_after: i64 = raw
        .query_row(
            "SELECT snapshot_revision FROM memory_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(revision_after, revision_before);
    assert_eq!(
        store.predicate_values(&requested).unwrap(),
        [
            Some(values[0].clone()),
            Some(values[1].clone()),
            Some(values[0].clone())
        ]
    );
    assert_eq!(snapshot(store.memory()), memory_before);
}

#[test]
fn symbol_byte_limit_and_lowercase_without_case_folding_are_exact() {
    let directory = tempdir().unwrap();
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).unwrap();

    let exact_limit = "é".repeat(2_048);
    let over_limit = format!("{exact_limit}x");
    let ids = store
        .intern_terms(&[
            exact_limit.clone(),
            "Straße".to_owned(),
            "STRASSE".to_owned(),
            "comma,value".to_owned(),
            "comma value".to_owned(),
        ])
        .unwrap();
    assert_eq!(ids, (0..5).map(TermId::new).collect::<Vec<_>>());
    assert!(matches!(
        store.intern_terms(&[over_limit]),
        Err(StoreError::InvalidSymbolValue {
            namespace: "term",
            index: 0,
            detail: "normalized UTF-8 value exceeds 4096 bytes"
        })
    ));
    assert_eq!(
        store.term_values(&ids).unwrap(),
        [
            Some(exact_limit),
            Some("straße".to_owned()),
            Some("strasse".to_owned()),
            Some("comma,value".to_owned()),
            Some("comma value".to_owned()),
        ]
    );
}

#[test]
fn save_rejects_uninterned_symbols_atomically_and_can_retry() {
    let directory = tempdir().unwrap();
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).unwrap();
    store
        .memory_mut()
        .insert_episode(observation_draft(0, 0, &[0]))
        .unwrap();

    assert!(matches!(
        store.save(),
        Err(StoreError::UnknownSymbolId {
            namespace: "predicate",
            id: 0
        })
    ));
    let raw = Connection::open(&path).unwrap();
    let state: (i64, i64, i64, i64) = raw
        .query_row(
            "SELECT snapshot_revision,
                    (SELECT count(*) FROM predicates),
                    (SELECT count(*) FROM terms),
                    (SELECT count(*) FROM episodes)
             FROM memory_meta",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(state, (0, 0, 0, 0));
    drop(raw);

    assert_eq!(
        store.intern_predicates(&["predicate".to_owned()]).unwrap(),
        [PredicateId::new(0)]
    );
    assert!(matches!(
        store.save(),
        Err(StoreError::UnknownSymbolId {
            namespace: "term",
            id: 0
        })
    ));
    let raw = Connection::open(&path).unwrap();
    let state: (i64, i64, i64, i64) = raw
        .query_row(
            "SELECT snapshot_revision,
                    (SELECT count(*) FROM predicates),
                    (SELECT count(*) FROM terms),
                    (SELECT count(*) FROM episodes)
             FROM memory_meta",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(state, (0, 0, 0, 0));
    drop(raw);

    assert_eq!(
        store.intern_terms(&["term".to_owned()]).unwrap(),
        [TermId::new(0)]
    );
    store.save().unwrap();
    drop(store);
    let reopened = SqliteStore::open(&path).unwrap();
    assert_eq!(reopened.memory().episodes().len(), 1);
    assert_eq!(
        reopened.predicate_values(&[PredicateId::new(0)]).unwrap(),
        [Some("predicate".to_owned())]
    );
    assert_eq!(
        reopened.term_values(&[TermId::new(0)]).unwrap(),
        [Some("term".to_owned())]
    );
}

#[test]
fn symbol_corruption_and_missing_episode_references_fail_closed() {
    for corrupt in ["value", "reference"] {
        let directory = tempdir().unwrap();
        let path = database_path(&directory);
        let mut store = SqliteStore::create(&path).unwrap();
        let predicate = store.intern_predicates(&["predicate".to_owned()]).unwrap()[0];
        let term = store.intern_terms(&["term".to_owned()]).unwrap()[0];
        store
            .memory_mut()
            .insert_episode(observation_draft(0, predicate.get(), &[term.get()]))
            .unwrap();
        store.save().unwrap();
        drop(store);

        let raw = Connection::open(&path).unwrap();
        match corrupt {
            "value" => {
                raw.execute("UPDATE predicates SET value = 'NOT NORMALIZED'", [])
                    .unwrap();
            }
            "reference" => {
                raw.execute("DELETE FROM terms", []).unwrap();
            }
            _ => unreachable!(),
        }
        drop(raw);

        match SqliteStore::open(&path) {
            Err(StoreError::InvalidStore(StoreIntegrityError::InvalidSymbol { .. }))
                if corrupt == "value" => {}
            Err(StoreError::InvalidStore(StoreIntegrityError::InvalidEpisode { .. }))
                if corrupt == "reference" => {}
            Err(error) => panic!("unexpected integrity error: {error}"),
            Ok(_) => panic!("corrupt symbol store opened"),
        }
    }
}

#[test]
fn concurrent_symbol_allocators_are_serialized_by_snapshot_cas() {
    let directory = tempdir().unwrap();
    let path = database_path(&directory);
    drop(SqliteStore::create(&path).unwrap());
    let mut first = SqliteStore::open(&path).unwrap();
    let mut stale = SqliteStore::open(&path).unwrap();
    assert_eq!(
        first.intern_predicates(&["first".to_owned()]).unwrap(),
        [PredicateId::new(0)]
    );
    assert_eq!(
        stale.intern_predicates(&["stale".to_owned()]).unwrap(),
        [PredicateId::new(0)]
    );
    first.save().unwrap();
    assert!(matches!(
        stale.save(),
        Err(StoreError::ConcurrentModification {
            expected_revision: 0,
            actual_revision: 1
        })
    ));
    drop(first);
    drop(stale);

    let reopened = SqliteStore::open(&path).unwrap();
    assert_eq!(
        reopened.predicate_values(&[PredicateId::new(0)]).unwrap(),
        [Some("first".to_owned())]
    );
}

#[test]
fn create_and_open_preserve_identity_and_respect_path_lifecycle() {
    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    assert!(!path.exists());

    let memory_id = {
        let store = SqliteStore::create(&path).expect("new store is created");
        assert_eq!(store.memory().episodes().len(), 0);
        assert_ne!(store.memory_id().get(), 0);
        store.memory_id()
    };
    assert!(path.exists());

    let bytes_before = fs::read(&path).expect("created database can be read");
    assert!(SqliteStore::create(&path).is_err());
    let bytes_after = fs::read(&path).expect("existing database can still be read");
    assert_eq!(bytes_after, bytes_before);

    let reopened = SqliteStore::open(&path).expect("created store reopens");
    assert_eq!(reopened.memory_id(), memory_id);
    drop(reopened);

    let missing = directory.path().join("missing.sqlite3");
    assert!(SqliteStore::open(&missing).is_err());
    assert!(!missing.exists());
}

#[test]
fn concurrent_creators_publish_exactly_one_complete_store() {
    const CREATOR_COUNT: usize = 8;

    let directory = tempdir().expect("temporary directory is available");
    let path = Arc::new(database_path(&directory));
    let barrier = Arc::new(Barrier::new(CREATOR_COUNT));
    let handles: Vec<_> = (0..CREATOR_COUNT)
        .map(|_| {
            let path = Arc::clone(&path);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                SqliteStore::create(path.as_ref())
                    .map(|store| store.memory_id())
                    .map_err(|error| error.to_string())
            })
        })
        .collect();

    let outcomes: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("creator thread does not panic"))
        .collect();
    let successful_ids: Vec<_> = outcomes.into_iter().filter_map(Result::ok).collect();
    assert_eq!(successful_ids.len(), 1);

    let reopened = SqliteStore::open(path.as_ref()).expect("published store is complete");
    assert_eq!(reopened.memory_id(), successful_ids[0]);
    drop(reopened);

    let entries: Vec<_> = fs::read_dir(directory.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(entries, [path.file_name().unwrap()]);
}

#[test]
fn full_snapshot_round_trips_exactly_and_continues_the_sequence() {
    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).expect("new store is created");

    let low_context = statement(0, &[0]);
    let high_context = statement(1, &[1, 2]);
    let first = insert(
        &mut store,
        EpisodeDraft {
            occurred_at: TimestampMs::new(i64::MIN),
            recorded_at: TimestampMs::new(i64::MAX),
            context: vec![high_context.clone(), low_context.clone(), high_context],
            observation: statement(2, &[0, 2, 3]),
            action: Some(statement(3, &[3, 0])),
            outcome: Some(statement(0, &[1])),
            source: SourceId::new(u64::MAX),
        },
    );
    let second = insert(
        &mut store,
        EpisodeDraft {
            occurred_at: TimestampMs::new(i64::MAX),
            recorded_at: TimestampMs::new(i64::MIN),
            context: Vec::new(),
            observation: statement(1, &[3]),
            action: None,
            outcome: None,
            source: SourceId::new(1_u64 << 63),
        },
    );
    let third = insert(&mut store, draft(7));

    store
        .memory_mut()
        .set_feedback_trace(first, second, trace(1, 1))
        .expect("first trace is valid");
    store
        .memory_mut()
        .set_feedback_trace(first, third, trace(2, 2))
        .expect("second trace is valid");
    store
        .memory_mut()
        .set_feedback_trace(second, first, trace(u16::MAX, 16))
        .expect("reverse trace is valid");

    let persisted = snapshot(store.memory());
    store.save().expect("snapshot is saved atomically");
    drop(store);

    let mut reopened = SqliteStore::open(&path).expect("saved store reopens");
    assert_eq!(snapshot(reopened.memory()), persisted);

    let next = insert(&mut reopened, draft(8));
    assert_eq!(next.memory_id(), reopened.memory_id());
    assert_eq!(next.sequence(), 3);
}

#[test]
fn feedback_trace_forms_round_trip_and_continue_after_reopen() {
    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).expect("new store is created");
    let ids: Vec<_> = (0..5).map(|seed| insert(&mut store, draft(seed))).collect();
    for (target, feedback_trace) in [
        (ids[1], trace(0, 1)),
        (ids[2], trace(1, 1)),
        (ids[3], trace(0xaaaa, 16)),
        (ids[4], trace(0x8000, 16)),
    ] {
        store
            .memory_mut()
            .set_feedback_trace(ids[0], target, feedback_trace)
            .expect("feedback trace endpoints are known");
    }
    store.save().expect("all feedback trace forms save");
    drop(store);

    let mut reopened = SqliteStore::open(&path).expect("feedback traces reopen");
    assert_eq!(
        reopened.memory().feedback_trace(ids[0], ids[1]),
        Some(trace(0, 1))
    );
    assert_eq!(
        reopened.memory().feedback_trace(ids[0], ids[2]),
        Some(trace(1, 1))
    );
    assert_eq!(
        reopened.memory().feedback_trace(ids[0], ids[3]),
        Some(trace(0xaaaa, 16))
    );
    assert_eq!(
        reopened.memory().feedback_trace(ids[0], ids[4]),
        Some(trace(0x8000, 16))
    );

    reopened
        .memory_mut()
        .apply_feedback(ids[0], &[ids[4]], true)
        .expect("continued feedback is accepted");
    assert_eq!(
        reopened.memory().feedback_trace(ids[0], ids[4]),
        Some(trace(1, 16))
    );
    reopened.save().expect("continued trace saves");
    drop(reopened);

    let reopened = SqliteStore::open(&path).expect("continued trace reopens");
    assert_eq!(
        reopened.memory().feedback_trace(ids[0], ids[4]),
        Some(trace(1, 16))
    );
}

#[test]
fn cue_derived_recall_is_identical_after_reopen_without_feedback() {
    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).expect("new store is created");
    let source = insert(&mut store, observation_draft(1, 10, &[7, 8]));
    let target = insert(&mut store, observation_draft(2, 10, &[7, 9]));
    insert(&mut store, observation_draft(3, 20, &[30]));

    let before = store
        .memory()
        .recall_from(source, usize::MAX)
        .expect("source is known");
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].atom_id, target);
    assert_eq!(before[0].activation.as_ppm(), 177_777);
    assert_eq!(store.memory().feedback_edges().count(), 0);
    let persisted = snapshot(store.memory());

    store.save().expect("episodes are saved");
    drop(store);

    let reopened = SqliteStore::open(&path).expect("saved store reopens");
    assert_eq!(snapshot(reopened.memory()), persisted);
    assert_eq!(reopened.memory().feedback_edges().count(), 0);
    assert_eq!(
        reopened
            .memory()
            .recall_from(source, usize::MAX)
            .expect("source is reconstructed"),
        before
    );
}

#[test]
fn repeated_saves_persist_feedback_deltas() {
    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).expect("new store is created");
    let first = insert(&mut store, draft(1));
    let second = insert(&mut store, draft(2));
    let third = insert(&mut store, draft(3));

    store
        .memory_mut()
        .set_feedback_trace(first, second, trace(1, 1))
        .expect("first edge inserts");
    store
        .memory_mut()
        .set_feedback_trace(second, third, trace(0, 1))
        .expect("second edge inserts");
    store.save().expect("initial snapshot is saved");

    store
        .memory_mut()
        .set_feedback_trace(first, second, trace(2, 2))
        .expect("known trace can be updated");
    store
        .memory_mut()
        .set_feedback_trace(third, first, trace(5, 3))
        .expect("replacement topology is valid");
    store.save().expect("replacement snapshot is saved");
    store
        .save()
        .expect("an unchanged snapshot can be saved again");
    drop(store);

    let reopened = SqliteStore::open(&path).expect("updated store reopens");
    assert_eq!(reopened.memory().episodes().len(), 3);
    assert_eq!(
        reopened.memory().feedback_trace(first, second),
        Some(trace(2, 2))
    );
    assert_eq!(
        reopened.memory().feedback_trace(second, third),
        Some(trace(0, 1))
    );
    assert_eq!(
        reopened.memory().feedback_trace(third, first),
        Some(trace(5, 3))
    );
    assert_eq!(reopened.memory().feedback_edges().count(), 3);
}

#[test]
fn stale_writer_fails_without_overwriting_the_committed_snapshot() {
    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    let mut original = SqliteStore::create(&path).expect("new store is created");
    insert(&mut original, draft(0));
    original.save().expect("base snapshot is saved");
    drop(original);

    let mut first_writer = SqliteStore::open(&path).expect("first writer opens");
    let mut stale_writer = SqliteStore::open(&path).expect("second writer opens");
    insert(&mut first_writer, draft(1));
    insert(&mut stale_writer, draft(99));

    first_writer.save().expect("first writer advances revision");
    let error = stale_writer
        .save()
        .expect_err("stale writer must not overwrite a newer snapshot");
    match error {
        StoreError::ConcurrentModification {
            expected_revision,
            actual_revision,
        } => assert_eq!(actual_revision, expected_revision + 1),
        other => panic!("unexpected stale-writer error: {other}"),
    }
    assert_eq!(stale_writer.memory().episodes().len(), 2);
    assert_eq!(
        stale_writer
            .memory()
            .episodes()
            .nth(1)
            .expect("unsaved episode remains in memory")
            .observation(),
        &draft(99).observation
    );
    drop(first_writer);
    drop(stale_writer);

    let reopened = SqliteStore::open(&path).expect("committed snapshot reopens");
    assert_eq!(reopened.memory().episodes().len(), 2);
    assert_eq!(
        reopened
            .memory()
            .episodes()
            .nth(1)
            .expect("first writer episode was persisted")
            .observation(),
        &draft(1).observation
    );
}

#[test]
fn identical_episodes_remain_distinct_occurrences_after_reopen() {
    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).expect("new store is created");
    let repeated = draft(42);
    let first = insert(&mut store, repeated.clone());
    let second = insert(&mut store, repeated);
    assert_eq!(first.sequence(), 0);
    assert_eq!(second.sequence(), 1);
    assert_ne!(first, second);
    store.save().expect("both occurrences are saved");
    let memory_id = store.memory_id();
    drop(store);

    let reopened = SqliteStore::open(&path).expect("saved store reopens");
    let episodes: Vec<_> = reopened.memory().episodes().collect();
    assert_eq!(episodes.len(), 2);
    assert_eq!(episodes[0].id(), AtomId::from_parts(memory_id, 0));
    assert_eq!(episodes[1].id(), AtomId::from_parts(memory_id, 1));
    assert_ne!(episodes[0], episodes[1]);
    assert_eq!(episodes[0].occurred_at(), episodes[1].occurred_at());
    assert_eq!(episodes[0].recorded_at(), episodes[1].recorded_at());
    assert_eq!(episodes[0].context(), episodes[1].context());
    assert_eq!(episodes[0].observation(), episodes[1].observation());
    assert_eq!(episodes[0].action(), episodes[1].action());
    assert_eq!(episodes[0].outcome(), episodes[1].outcome());
    assert_eq!(episodes[0].source(), episodes[1].source());
}

#[test]
fn empty_memory_can_be_saved_and_reopened() {
    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).expect("new store is created");
    let memory_id = store.memory_id();
    store.save().expect("empty snapshot is saved");
    drop(store);

    let reopened = SqliteStore::open(&path).expect("empty store reopens");
    assert_eq!(reopened.memory_id(), memory_id);
    assert_eq!(snapshot(reopened.memory()).episodes.len(), 0);
    assert_eq!(reopened.memory().feedback_edges().count(), 0);
}

#[test]
fn thousand_episode_snapshot_reopens_without_semantic_drift() {
    const EPISODE_COUNT: usize = 1_000;

    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).expect("new store is created");
    let mut ids = Vec::with_capacity(EPISODE_COUNT);

    for sequence in 0..EPISODE_COUNT {
        let seed = u64::try_from(sequence).expect("fixture sequence fits u64");
        ids.push(insert(&mut store, draft(seed)));
    }
    for pair in ids.windows(2) {
        store
            .memory_mut()
            .set_feedback_trace(pair[0], pair[1], trace(1, 1))
            .expect("chain feedback trace is valid");
    }
    store.save().expect("large deterministic fixture is saved");
    let expected = snapshot(store.memory());
    drop(store);

    let reopened = SqliteStore::open(&path).expect("large deterministic fixture reopens");
    assert_eq!(reopened.memory().episodes().len(), EPISODE_COUNT);
    assert_eq!(
        reopened.memory().feedback_edges().count(),
        EPISODE_COUNT - 1
    );
    assert_eq!(snapshot(reopened.memory()), expected);
}
