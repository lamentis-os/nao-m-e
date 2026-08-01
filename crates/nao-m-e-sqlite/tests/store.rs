use std::fs;
use std::sync::{Arc, Barrier};
use std::thread;

use nao_m_e::{
    Activation, AtomId, EpisodeAtom, EpisodeDraft, InfluenceWeight, MemoryId, MemoryV0,
    PredicateId, RecallHit, RelevanceEdge, SCALE, SourceId, Statement, TermId, TimestampMs,
};
use nao_m_e_sqlite::{SqliteStore, StoreError};
use tempfile::{TempDir, tempdir};

#[derive(Debug, Eq, PartialEq)]
struct MemorySnapshot {
    memory_id: MemoryId,
    episodes: Vec<EpisodeAtom>,
    activations: Vec<(AtomId, Activation)>,
    relevance_edges: Vec<RelevanceEdge>,
    recall: Vec<RecallHit>,
}

fn snapshot(memory: &MemoryV0) -> MemorySnapshot {
    let episodes: Vec<_> = memory.episodes().cloned().collect();
    let activations = episodes
        .iter()
        .map(|episode| {
            (
                episode.id(),
                memory
                    .activation(episode.id())
                    .expect("snapshot episode belongs to its memory"),
            )
        })
        .collect();

    MemorySnapshot {
        memory_id: memory.memory_id(),
        episodes,
        activations,
        relevance_edges: memory.relevance_edges().collect(),
        recall: memory.top_k(usize::MAX),
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
        context: vec![statement(10 + seed, &[100 + seed])],
        observation: statement(20 + seed, &[200 + seed, 201 + seed]),
        action: Some(statement(30 + seed, &[300 + seed])),
        outcome: Some(statement(40 + seed, &[400 + seed])),
        source: SourceId::new(50 + seed),
    }
}

fn insert(store: &mut SqliteStore, episode: EpisodeDraft) -> AtomId {
    store
        .memory_mut()
        .insert_episode(episode)
        .expect("test memory has identifier capacity")
}

fn activation(value: u32) -> Activation {
    Activation::from_ppm(value).expect("test activation is in range")
}

fn weight(value: u32) -> InfluenceWeight {
    InfluenceWeight::from_ppm(value).expect("test weight is positive and in range")
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
fn full_snapshot_and_next_transition_round_trip_exactly() {
    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).expect("new store is created");

    let low_context = statement(0, &[u64::MAX]);
    let high_context = statement(u64::MAX, &[0, 1_u64 << 63]);
    let first = insert(
        &mut store,
        EpisodeDraft {
            occurred_at: TimestampMs::new(i64::MIN),
            recorded_at: TimestampMs::new(i64::MAX),
            context: vec![high_context.clone(), low_context.clone(), high_context],
            observation: statement(u64::MAX, &[0, 1_u64 << 63, u64::MAX]),
            action: Some(statement(1_u64 << 63, &[u64::MAX, 0])),
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
            observation: statement(1_u64 << 63, &[u64::MAX]),
            action: None,
            outcome: None,
            source: SourceId::new(1_u64 << 63),
        },
    );
    let third = insert(&mut store, draft(7));

    store
        .memory_mut()
        .stimulate(first, Activation::ONE)
        .expect("first atom is local");
    store
        .memory_mut()
        .stimulate(second, activation(424_242))
        .expect("second atom is local");
    store
        .memory_mut()
        .set_relevance(first, second, weight(600_000))
        .expect("first edge fits its source budget");
    store
        .memory_mut()
        .set_relevance(first, third, weight(400_000))
        .expect("second edge fills its source budget");
    store
        .memory_mut()
        .set_relevance(second, first, weight(SCALE))
        .expect("reverse edge has an independent budget");

    let persisted = snapshot(store.memory());
    store.save().expect("snapshot is saved atomically");

    store.memory_mut().step();
    let expected_after_step = snapshot(store.memory());
    drop(store);

    let mut reopened = SqliteStore::open(&path).expect("saved store reopens");
    assert_eq!(snapshot(reopened.memory()), persisted);
    reopened.memory_mut().step();
    assert_eq!(snapshot(reopened.memory()), expected_after_step);

    let next = insert(&mut reopened, draft(8));
    assert_eq!(next.memory_id(), reopened.memory_id());
    assert_eq!(next.sequence(), 3);
}

#[test]
fn repeated_saves_persist_edge_removal_and_activation_reset() {
    let directory = tempdir().expect("temporary directory is available");
    let path = database_path(&directory);
    let mut store = SqliteStore::create(&path).expect("new store is created");
    let first = insert(&mut store, draft(1));
    let second = insert(&mut store, draft(2));
    let third = insert(&mut store, draft(3));

    store
        .memory_mut()
        .set_relevance(first, second, weight(600_000))
        .expect("first edge inserts");
    store
        .memory_mut()
        .set_relevance(second, third, weight(700_000))
        .expect("second edge inserts");
    store
        .memory_mut()
        .stimulate(first, Activation::ONE)
        .expect("first atom is stimulated");
    store
        .memory_mut()
        .stimulate(second, activation(500_000))
        .expect("second atom is stimulated");
    store.save().expect("initial snapshot is saved");

    assert_eq!(
        store
            .memory_mut()
            .remove_relevance(first, second)
            .expect("known edge can be removed"),
        Some(weight(600_000))
    );
    store
        .memory_mut()
        .set_relevance(third, first, weight(123_456))
        .expect("replacement topology is valid");
    store.memory_mut().reset_activations();
    store.save().expect("replacement snapshot is saved");
    store
        .save()
        .expect("an unchanged snapshot can be saved again");
    drop(store);

    let reopened = SqliteStore::open(&path).expect("updated store reopens");
    assert_eq!(reopened.memory().episodes().len(), 3);
    for episode in reopened.memory().episodes() {
        assert_eq!(
            reopened.memory().activation(episode.id()),
            Some(Activation::ZERO)
        );
    }
    assert_eq!(reopened.memory().relevance(first, second), None);
    assert_eq!(
        reopened.memory().relevance(second, third),
        Some(weight(700_000))
    );
    assert_eq!(
        reopened.memory().relevance(third, first),
        Some(weight(123_456))
    );
    assert_eq!(reopened.memory().relevance_edges().count(), 2);
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
    store
        .memory_mut()
        .stimulate(second, activation(7))
        .expect("second occurrence is independently addressable");
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
    assert_eq!(reopened.memory().activation(first), Some(Activation::ZERO));
    assert_eq!(reopened.memory().activation(second), Some(activation(7)));
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
    assert!(reopened.memory().top_k(10).is_empty());
    assert_eq!(reopened.memory().relevance_edges().count(), 0);
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
            .set_relevance(pair[0], pair[1], weight(1))
            .expect("chain edge has an independent source budget");
    }
    for id in ids.iter().step_by(100) {
        store
            .memory_mut()
            .stimulate(*id, activation(123_456))
            .expect("sampled atom is local");
    }

    store.save().expect("large deterministic fixture is saved");
    let expected = snapshot(store.memory());
    drop(store);

    let reopened = SqliteStore::open(&path).expect("large deterministic fixture reopens");
    assert_eq!(reopened.memory().episodes().len(), EPISODE_COUNT);
    assert_eq!(
        reopened.memory().relevance_edges().count(),
        EPISODE_COUNT - 1
    );
    assert_eq!(snapshot(reopened.memory()), expected);
}
