use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use nao_m_e::{Attribute, EpisodeDraft, TimestampMs};
use nao_m_e_semantic_index::{
    CueEmbedder, CueText, EmbedderError, Embedding, EmbeddingProfile, IndexError,
    IndexIntegrityError, IndexStats, SemanticCueIndex,
};
use nao_m_e_sqlite::SqliteStore;
use rusqlite::{Connection, params};
use tempfile::{TempDir, tempdir};

const PROFILE_FINGERPRINT: [u8; 32] = [0x35; 32];
const OTHER_PROFILE_FINGERPRINT: [u8; 32] = [0xa7; 32];
const DIMENSIONS: u16 = 3;

struct Fixture {
    _directory: TempDir,
    memory_path: PathBuf,
    index_path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempdir().unwrap();
        let memory_path = directory.path().join("memory.sqlite3");
        let index_path = directory.path().join("semantic.sqlite3");
        Self {
            _directory: directory,
            memory_path,
            index_path,
        }
    }
}

#[derive(Clone, Copy)]
enum EmbedBehavior {
    Normal,
    Fail,
    WrongLength,
}

#[derive(Debug)]
struct TestEmbedderError;

impl fmt::Display for TestEmbedderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("deliberate test embedder failure")
    }
}

impl Error for TestEmbedderError {}

struct RecordingEmbedder {
    profile: EmbeddingProfile,
    behavior: EmbedBehavior,
    calls: Vec<Vec<(String, String)>>,
}

impl RecordingEmbedder {
    fn new(profile: EmbeddingProfile) -> Self {
        Self {
            profile,
            behavior: EmbedBehavior::Normal,
            calls: Vec::new(),
        }
    }

    fn failing(profile: EmbeddingProfile) -> Self {
        Self {
            profile,
            behavior: EmbedBehavior::Fail,
            calls: Vec::new(),
        }
    }

    fn wrong_length(profile: EmbeddingProfile) -> Self {
        Self {
            profile,
            behavior: EmbedBehavior::WrongLength,
            calls: Vec::new(),
        }
    }

    fn calls(&self) -> &[Vec<(String, String)>] {
        &self.calls
    }

    fn embedding(&self, cue: CueText<'_>) -> Embedding {
        let key = checksum(cue.key());
        let value = checksum(cue.value());
        Embedding::new(self.profile, vec![key, value, key.wrapping_add(value)])
            .expect("test vectors have the profile width and are non-zero")
    }
}

impl CueEmbedder for RecordingEmbedder {
    fn profile(&self) -> EmbeddingProfile {
        self.profile
    }

    fn embed_batch(&mut self, cues: &[CueText<'_>]) -> Result<Vec<Embedding>, EmbedderError> {
        self.calls.push(
            cues.iter()
                .map(|cue| (cue.key().to_owned(), cue.value().to_owned()))
                .collect(),
        );
        match self.behavior {
            EmbedBehavior::Normal => Ok(cues
                .iter()
                .copied()
                .map(|cue| self.embedding(cue))
                .collect()),
            EmbedBehavior::Fail => Err(Box::new(TestEmbedderError)),
            EmbedBehavior::WrongLength => Ok(Vec::new()),
        }
    }
}

fn checksum(value: &str) -> i16 {
    let sum = value.bytes().fold(1_u32, |sum, byte| {
        sum.wrapping_mul(31).wrapping_add(u32::from(byte))
    });
    i16::try_from((sum % 30_000) + 1).unwrap()
}

fn profile() -> EmbeddingProfile {
    EmbeddingProfile::new(PROFILE_FINGERPRINT, DIMENSIONS).unwrap()
}

fn other_profile() -> EmbeddingProfile {
    EmbeddingProfile::new(OTHER_PROFILE_FINGERPRINT, DIMENSIONS).unwrap()
}

fn create_memory(path: &Path) -> SqliteStore {
    SqliteStore::create(path).unwrap()
}

fn append_episode(store: &mut SqliteStore, attributes: &[(&str, &[&str])]) {
    let mut texts = Vec::new();
    for &(key, values) in attributes {
        texts.push(key.to_owned());
        texts.extend(values.iter().map(|value| (*value).to_owned()));
    }
    let ids = store.intern_symbols(&texts).unwrap();
    let mut ids = ids.into_iter();
    let attributes = attributes
        .iter()
        .map(|(_, values)| {
            let key = ids.next().expect("each key was interned");
            let values = (0..values.len())
                .map(|_| ids.next().expect("each value was interned"))
                .collect();
            Attribute::new(key, values).unwrap()
        })
        .collect();
    assert!(ids.next().is_none());

    let timestamp = i64::try_from(store.memory().episodes().len()).unwrap();
    store
        .memory_mut()
        .insert_episode(EpisodeDraft::new(TimestampMs::new(timestamp), attributes).unwrap())
        .unwrap();
}

fn append_and_save(store: &mut SqliteStore, attributes: &[(&str, &[&str])]) {
    append_episode(store, attributes);
    store.save().unwrap();
}

fn assert_stats(stats: IndexStats, episodes: u64, cues: u64, postings: u64) {
    assert_eq!(stats.indexed_episode_count(), episodes);
    assert_eq!(stats.cue_count(), cues);
    assert_eq!(stats.posting_count(), postings);
}

fn expect_index_error<T>(result: Result<T, IndexError>) -> IndexError {
    match result {
        Ok(_) => panic!("operation unexpectedly accepted an invalid index state"),
        Err(error) => error,
    }
}

#[test]
fn empty_memory_creates_and_reopens_an_empty_index() {
    let fixture = Fixture::new();
    drop(create_memory(&fixture.memory_path));
    let mut embedder = RecordingEmbedder::new(profile());

    let index =
        SemanticCueIndex::create(&fixture.index_path, &fixture.memory_path, &mut embedder).unwrap();
    assert_stats(index.stats(), 0, 0, 0);
    assert!(embedder.calls().is_empty());
    drop(index);

    let reopened =
        SemanticCueIndex::open(&fixture.index_path, &fixture.memory_path, profile()).unwrap();
    assert_stats(reopened.stats(), 0, 0, 0);
}

#[test]
fn bound_cues_are_deduplicated_but_keep_every_episode_posting() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_episode(
        &mut store,
        &[("topic", &["rust", "sqlite"]), ("state", &["open"])],
    );
    append_episode(&mut store, &[("topic", &["rust"]), ("alias", &["rust"])]);
    store.save().unwrap();
    drop(store);
    let mut embedder = RecordingEmbedder::new(profile());

    let index =
        SemanticCueIndex::create(&fixture.index_path, &fixture.memory_path, &mut embedder).unwrap();

    assert_stats(index.stats(), 2, 4, 5);
    assert_eq!(
        embedder.calls(),
        &[vec![
            ("topic".to_owned(), "rust".to_owned()),
            ("topic".to_owned(), "sqlite".to_owned()),
            ("state".to_owned(), "open".to_owned()),
            ("alias".to_owned(), "rust".to_owned()),
        ]]
    );
    drop(index);

    let reopened =
        SemanticCueIndex::open(&fixture.index_path, &fixture.memory_path, profile()).unwrap();
    assert_stats(reopened.stats(), 2, 4, 5);
}

#[test]
fn synchronize_embeds_only_new_cues_and_persists_the_extended_prefix() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_and_save(&mut store, &[("project", &["lamentis"])]);
    drop(store);
    let mut initial_embedder = RecordingEmbedder::new(profile());
    let mut index = SemanticCueIndex::create(
        &fixture.index_path,
        &fixture.memory_path,
        &mut initial_embedder,
    )
    .unwrap();
    assert_stats(index.stats(), 1, 1, 1);

    let mut store = SqliteStore::open(&fixture.memory_path).unwrap();
    append_and_save(
        &mut store,
        &[("project", &["lamentis"]), ("error", &["http 404"])],
    );
    drop(store);
    let mut incremental_embedder = RecordingEmbedder::new(profile());

    let stats = index
        .synchronize(&fixture.memory_path, &mut incremental_embedder)
        .unwrap();
    assert_stats(stats, 2, 2, 3);
    assert_eq!(
        incremental_embedder.calls(),
        &[vec![("error".to_owned(), "http 404".to_owned())]]
    );
    drop(index);

    let reopened =
        SemanticCueIndex::open(&fixture.index_path, &fixture.memory_path, profile()).unwrap();
    assert_stats(reopened.stats(), 2, 2, 3);
}

#[test]
fn synchronize_with_only_known_cues_never_calls_the_embedder() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_and_save(&mut store, &[("project", &["lamentis"])]);
    drop(store);
    let mut initial_embedder = RecordingEmbedder::new(profile());
    let mut index = SemanticCueIndex::create(
        &fixture.index_path,
        &fixture.memory_path,
        &mut initial_embedder,
    )
    .unwrap();

    let mut store = SqliteStore::open(&fixture.memory_path).unwrap();
    append_and_save(&mut store, &[("project", &["lamentis"])]);
    drop(store);
    let mut incremental_embedder = RecordingEmbedder::new(profile());

    let stats = index
        .synchronize(&fixture.memory_path, &mut incremental_embedder)
        .unwrap();
    assert_stats(stats, 2, 1, 2);
    assert!(incremental_embedder.calls().is_empty());
}

#[test]
fn synchronization_batches_257_new_cues_as_256_plus_one() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    let values: Vec<_> = (0..257).map(|index| format!("value-{index:03}")).collect();
    let value_refs: Vec<_> = values.iter().map(String::as_str).collect();
    append_and_save(&mut store, &[("tag", value_refs.as_slice())]);
    drop(store);
    let mut embedder = RecordingEmbedder::new(profile());

    let index =
        SemanticCueIndex::create(&fixture.index_path, &fixture.memory_path, &mut embedder).unwrap();

    assert_stats(index.stats(), 1, 257, 257);
    assert_eq!(
        embedder.calls().iter().map(Vec::len).collect::<Vec<_>>(),
        [256, 1]
    );
    assert_eq!(
        embedder.calls()[0][0],
        ("tag".to_owned(), values[0].clone())
    );
    assert_eq!(
        embedder.calls()[1][0],
        ("tag".to_owned(), values[256].clone())
    );
    drop(index);

    let reopened =
        SemanticCueIndex::open(&fixture.index_path, &fixture.memory_path, profile()).unwrap();
    assert_stats(reopened.stats(), 1, 257, 257);
}

#[test]
fn synchronize_observes_only_the_freshly_opened_committed_memory() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_and_save(&mut store, &[("project", &["lamentis"])]);
    drop(store);
    let mut initial_embedder = RecordingEmbedder::new(profile());
    let mut index = SemanticCueIndex::create(
        &fixture.index_path,
        &fixture.memory_path,
        &mut initial_embedder,
    )
    .unwrap();

    let mut unsaved = SqliteStore::open(&fixture.memory_path).unwrap();
    append_episode(&mut unsaved, &[("error", &["unsaved 404"])]);
    assert_eq!(unsaved.memory().episodes().len(), 2);
    let mut incremental_embedder = RecordingEmbedder::new(profile());

    let stats = index
        .synchronize(&fixture.memory_path, &mut incremental_embedder)
        .unwrap();
    assert_stats(stats, 1, 1, 1);
    assert!(incremental_embedder.calls().is_empty());
}

#[test]
fn stale_sidecar_session_is_rejected_and_a_reopened_session_can_continue() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_and_save(&mut store, &[("project", &["lamentis"])]);
    drop(store);
    let mut initial_embedder = RecordingEmbedder::new(profile());
    let mut first = SemanticCueIndex::create(
        &fixture.index_path,
        &fixture.memory_path,
        &mut initial_embedder,
    )
    .unwrap();
    let mut stale =
        SemanticCueIndex::open(&fixture.index_path, &fixture.memory_path, profile()).unwrap();

    let mut store = SqliteStore::open(&fixture.memory_path).unwrap();
    append_and_save(&mut store, &[("error", &["http 404"])]);
    drop(store);
    let mut first_embedder = RecordingEmbedder::new(profile());
    assert_stats(
        first
            .synchronize(&fixture.memory_path, &mut first_embedder)
            .unwrap(),
        2,
        2,
        2,
    );

    let mut stale_embedder = RecordingEmbedder::new(profile());
    let error = stale
        .synchronize(&fixture.memory_path, &mut stale_embedder)
        .expect_err("stale sidecar session must not overwrite a newer index prefix");
    assert!(matches!(
        error,
        IndexError::ConcurrentModification {
            expected_episode_count: 1,
            actual_episode_count: 2
        }
    ));
    assert!(stale_embedder.calls().is_empty());
    drop(stale);
    drop(first);

    let mut reopened =
        SemanticCueIndex::open(&fixture.index_path, &fixture.memory_path, profile()).unwrap();
    let mut store = SqliteStore::open(&fixture.memory_path).unwrap();
    append_and_save(&mut store, &[("status", &["fixed"])]);
    drop(store);
    let mut reopened_embedder = RecordingEmbedder::new(profile());
    assert_stats(
        reopened
            .synchronize(&fixture.memory_path, &mut reopened_embedder)
            .unwrap(),
        3,
        3,
        3,
    );
    assert_eq!(
        reopened_embedder.calls(),
        &[vec![("status".to_owned(), "fixed".to_owned())]]
    );
}

#[test]
fn unchanged_memory_is_a_byte_exact_no_op_sync() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_and_save(&mut store, &[("project", &["lamentis"])]);
    drop(store);
    let mut initial_embedder = RecordingEmbedder::new(profile());
    let mut index = SemanticCueIndex::create(
        &fixture.index_path,
        &fixture.memory_path,
        &mut initial_embedder,
    )
    .unwrap();
    let before = fs::read(&fixture.index_path).unwrap();
    let before_stats = index.stats();
    let mut no_op_embedder = RecordingEmbedder::new(profile());

    let after_stats = index
        .synchronize(&fixture.memory_path, &mut no_op_embedder)
        .unwrap();

    assert_eq!(after_stats, before_stats);
    assert!(no_op_embedder.calls().is_empty());
    assert_eq!(fs::read(&fixture.index_path).unwrap(), before);
}

#[test]
fn create_never_clobbers_an_existing_destination() {
    let fixture = Fixture::new();
    drop(create_memory(&fixture.memory_path));
    let original = b"existing unrelated bytes";
    fs::write(&fixture.index_path, original).unwrap();
    let mut embedder = RecordingEmbedder::new(profile());

    let error = expect_index_error(SemanticCueIndex::create(
        &fixture.index_path,
        &fixture.memory_path,
        &mut embedder,
    ));

    assert!(matches!(
        error,
        IndexError::Io(ref error) if error.kind() == std::io::ErrorKind::AlreadyExists
    ));
    assert!(embedder.calls().is_empty());
    assert_eq!(fs::read(&fixture.index_path).unwrap(), original);
}

#[test]
fn embedder_failures_and_wrong_batch_lengths_are_atomic_and_retryable() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_and_save(&mut store, &[("project", &["lamentis"])]);
    drop(store);

    let failed_create_path = fixture._directory.path().join("failed.sqlite3");
    let mut failing_create = RecordingEmbedder::failing(profile());
    let error = expect_index_error(SemanticCueIndex::create(
        &failed_create_path,
        &fixture.memory_path,
        &mut failing_create,
    ));
    assert!(matches!(error, IndexError::Embedder(_)));
    assert!(!failed_create_path.exists());

    let wrong_create_path = fixture._directory.path().join("wrong.sqlite3");
    let mut wrong_create = RecordingEmbedder::wrong_length(profile());
    let error = expect_index_error(SemanticCueIndex::create(
        &wrong_create_path,
        &fixture.memory_path,
        &mut wrong_create,
    ));
    assert!(matches!(
        error,
        IndexError::EmbeddingBatchLength {
            expected: 1,
            found: 0
        }
    ));
    assert!(!wrong_create_path.exists());

    let mut initial_embedder = RecordingEmbedder::new(profile());
    let mut index = SemanticCueIndex::create(
        &fixture.index_path,
        &fixture.memory_path,
        &mut initial_embedder,
    )
    .unwrap();
    let original_stats = index.stats();

    let mut store = SqliteStore::open(&fixture.memory_path).unwrap();
    append_and_save(&mut store, &[("error", &["http 404"])]);
    drop(store);

    let mut failing_sync = RecordingEmbedder::failing(profile());
    let error = index
        .synchronize(&fixture.memory_path, &mut failing_sync)
        .expect_err("embedder failure must abort synchronization");
    assert!(matches!(error, IndexError::Embedder(_)));
    assert_eq!(index.stats(), original_stats);
    let reopened =
        SemanticCueIndex::open(&fixture.index_path, &fixture.memory_path, profile()).unwrap();
    assert_eq!(reopened.stats(), original_stats);
    drop(reopened);

    let mut wrong_sync = RecordingEmbedder::wrong_length(profile());
    let error = index
        .synchronize(&fixture.memory_path, &mut wrong_sync)
        .expect_err("wrong output count must abort synchronization");
    assert!(matches!(
        error,
        IndexError::EmbeddingBatchLength {
            expected: 1,
            found: 0
        }
    ));
    assert_eq!(index.stats(), original_stats);
    let reopened =
        SemanticCueIndex::open(&fixture.index_path, &fixture.memory_path, profile()).unwrap();
    assert_eq!(reopened.stats(), original_stats);
    drop(reopened);

    let mut retry = RecordingEmbedder::new(profile());
    assert_stats(
        index.synchronize(&fixture.memory_path, &mut retry).unwrap(),
        2,
        2,
        2,
    );
}

#[test]
fn profile_and_memory_mismatches_fail_closed() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_and_save(&mut store, &[("project", &["lamentis"])]);
    drop(store);
    let mut embedder = RecordingEmbedder::new(profile());
    drop(
        SemanticCueIndex::create(&fixture.index_path, &fixture.memory_path, &mut embedder).unwrap(),
    );

    let error = expect_index_error(SemanticCueIndex::open(
        &fixture.index_path,
        &fixture.memory_path,
        other_profile(),
    ));
    assert!(matches!(
        error,
        IndexError::ProfileMismatch { expected, found }
            if expected == profile() && found == other_profile()
    ));

    let other_memory_path = fixture._directory.path().join("other-memory.sqlite3");
    drop(create_memory(&other_memory_path));
    let error = expect_index_error(SemanticCueIndex::open(
        &fixture.index_path,
        &other_memory_path,
        profile(),
    ));
    assert!(matches!(
        error,
        IndexError::InvalidIndex(IndexIntegrityError::MemoryMismatch)
    ));
}

#[test]
fn create_open_and_synchronize_never_change_the_authoritative_database() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_and_save(&mut store, &[("project", &["lamentis"])]);
    drop(store);
    let before_create = fs::read(&fixture.memory_path).unwrap();
    let mut embedder = RecordingEmbedder::new(profile());

    let mut index =
        SemanticCueIndex::create(&fixture.index_path, &fixture.memory_path, &mut embedder).unwrap();
    assert_eq!(fs::read(&fixture.memory_path).unwrap(), before_create);

    let reopened =
        SemanticCueIndex::open(&fixture.index_path, &fixture.memory_path, profile()).unwrap();
    assert_eq!(fs::read(&fixture.memory_path).unwrap(), before_create);
    drop(reopened);

    let mut store = SqliteStore::open(&fixture.memory_path).unwrap();
    append_and_save(&mut store, &[("error", &["http 404"])]);
    drop(store);
    let before_sync = fs::read(&fixture.memory_path).unwrap();
    let mut incremental_embedder = RecordingEmbedder::new(profile());
    index
        .synchronize(&fixture.memory_path, &mut incremental_embedder)
        .unwrap();
    assert_eq!(fs::read(&fixture.memory_path).unwrap(), before_sync);
}

#[test]
fn an_additional_schema_object_is_rejected() {
    let fixture = populated_index();
    let connection = Connection::open(&fixture.index_path).unwrap();
    connection
        .execute("CREATE TABLE unexpected (value INTEGER)", [])
        .unwrap();
    drop(connection);

    let error = expect_index_error(SemanticCueIndex::open(
        &fixture.index_path,
        &fixture.memory_path,
        profile(),
    ));
    assert!(matches!(
        error,
        IndexError::InvalidIndex(IndexIntegrityError::InvalidMetadata { .. })
    ));
}

#[test]
fn an_invalid_vector_is_rejected() {
    let fixture = populated_index();
    let connection = Connection::open(&fixture.index_path).unwrap();
    connection
        .execute(
            "UPDATE semantic_cues SET vector = ?1",
            params![vec![0_u8; usize::from(DIMENSIONS) * 2]],
        )
        .unwrap();
    drop(connection);

    let error = expect_index_error(SemanticCueIndex::open(
        &fixture.index_path,
        &fixture.memory_path,
        profile(),
    ));
    assert!(matches!(
        error,
        IndexError::InvalidIndex(IndexIntegrityError::InvalidCue { .. })
    ));
}

#[test]
fn a_missing_required_posting_is_rejected() {
    let fixture = populated_index();
    let connection = Connection::open(&fixture.index_path).unwrap();
    connection.execute("DELETE FROM episode_cues", []).unwrap();
    drop(connection);

    let error = expect_index_error(SemanticCueIndex::open(
        &fixture.index_path,
        &fixture.memory_path,
        profile(),
    ));
    assert!(matches!(
        error,
        IndexError::InvalidIndex(IndexIntegrityError::InvalidPosting { .. })
    ));
}

#[test]
fn missing_metadata_uses_the_specific_integrity_error() {
    let fixture = populated_index();
    let connection = Connection::open(&fixture.index_path).unwrap();
    connection.execute("DELETE FROM semantic_meta", []).unwrap();
    drop(connection);

    let error = expect_index_error(SemanticCueIndex::open(
        &fixture.index_path,
        &fixture.memory_path,
        profile(),
    ));
    assert!(matches!(
        error,
        IndexError::InvalidIndex(IndexIntegrityError::MissingMetadata)
    ));
}

#[test]
fn cue_identifiers_cannot_be_rebound_with_matching_posting_changes() {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_episode(&mut store, &[("key a", &["value a"])]);
    append_episode(&mut store, &[("key b", &["value b"])]);
    store.save().unwrap();
    drop(store);
    let mut embedder = RecordingEmbedder::new(profile());
    drop(
        SemanticCueIndex::create(&fixture.index_path, &fixture.memory_path, &mut embedder).unwrap(),
    );

    let mut connection = Connection::open(&fixture.index_path).unwrap();
    let transaction = connection.transaction().unwrap();
    let cues: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = {
        let mut statement = transaction
            .prepare("SELECT cue_id, key_id, value_id FROM semantic_cues ORDER BY cue_id")
            .unwrap();
        statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    assert_eq!(cues.len(), 2);
    let temporary_key = u64::MAX.to_be_bytes();
    let temporary_value = (u64::MAX - 1).to_be_bytes();
    transaction
        .execute(
            "UPDATE semantic_cues SET key_id = ?1, value_id = ?2 WHERE cue_id = ?3",
            params![
                temporary_key.as_slice(),
                temporary_value.as_slice(),
                &cues[0].0
            ],
        )
        .unwrap();
    transaction
        .execute(
            "UPDATE semantic_cues SET key_id = ?1, value_id = ?2 WHERE cue_id = ?3",
            params![&cues[0].1, &cues[0].2, &cues[1].0],
        )
        .unwrap();
    transaction
        .execute(
            "UPDATE semantic_cues SET key_id = ?1, value_id = ?2 WHERE cue_id = ?3",
            params![&cues[1].1, &cues[1].2, &cues[0].0],
        )
        .unwrap();

    let temporary_sequence = u64::MAX.to_be_bytes();
    transaction
        .execute(
            "UPDATE episode_cues SET sequence = ?1 WHERE sequence = ?2",
            params![
                temporary_sequence.as_slice(),
                0_u64.to_be_bytes().as_slice()
            ],
        )
        .unwrap();
    transaction
        .execute(
            "UPDATE episode_cues SET sequence = ?1 WHERE sequence = ?2",
            params![
                0_u64.to_be_bytes().as_slice(),
                1_u64.to_be_bytes().as_slice()
            ],
        )
        .unwrap();
    transaction
        .execute(
            "UPDATE episode_cues SET sequence = ?1 WHERE sequence = ?2",
            params![
                1_u64.to_be_bytes().as_slice(),
                temporary_sequence.as_slice()
            ],
        )
        .unwrap();
    transaction.commit().unwrap();
    drop(connection);

    let error = expect_index_error(SemanticCueIndex::open(
        &fixture.index_path,
        &fixture.memory_path,
        profile(),
    ));
    assert!(matches!(
        error,
        IndexError::InvalidIndex(IndexIntegrityError::InvalidCue { .. })
    ));
}

fn populated_index() -> Fixture {
    let fixture = Fixture::new();
    let mut store = create_memory(&fixture.memory_path);
    append_and_save(&mut store, &[("project", &["lamentis"])]);
    drop(store);
    let mut embedder = RecordingEmbedder::new(profile());
    drop(
        SemanticCueIndex::create(&fixture.index_path, &fixture.memory_path, &mut embedder).unwrap(),
    );
    fixture
}
