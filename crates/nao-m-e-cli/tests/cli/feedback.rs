use std::path::Path;

use nao_m_e::{AtomId, FeedbackTrace, MAX_FEEDBACK_TARGETS};
use nao_m_e_sqlite::SqliteStore;
use tempfile::TempDir;

use super::support::{
    add_minimal, assert_silent_success, cli, failure, feedback, init, invoke, recall, success_text,
};

fn add_episode(database: &Path, timestamp: i64, attribute: &str, values: &[&str]) {
    let mut command = cli();
    command
        .arg("add")
        .arg(database)
        .arg("--timestamp")
        .arg(timestamp.to_string())
        .arg("--attribute")
        .arg(attribute);
    for value in values {
        command.arg("--value").arg(value);
    }
    command.arg("--quiet");
    assert_silent_success(invoke(command, None));
}

fn recall_scores(database: &Path, source: u64) -> Vec<(u64, u32)> {
    let output = success_text(recall(database, source, None));
    if output.is_empty() {
        return Vec::new();
    }
    output
        .trim_end()
        .split("\n\n")
        .map(|block| {
            let mut lines = block.lines();
            let sequence = lines
                .next()
                .and_then(|line| line.strip_prefix("sequence "))
                .expect("recall block starts with sequence")
                .parse()
                .expect("sequence is numeric");
            let activation = lines
                .next()
                .and_then(|line| line.strip_prefix("activation_ppm "))
                .expect("recall block continues with activation")
                .parse()
                .expect("activation is numeric");
            (sequence, activation)
        })
        .collect()
}

fn score_for(scores: &[(u64, u32)], sequence: u64) -> Option<u32> {
    scores
        .iter()
        .find_map(|&(candidate, score)| (candidate == sequence).then_some(score))
}

#[test]
fn bounded_feedback_learns_reverses_and_suppresses_structural_matches_across_processes() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    for (timestamp, attribute, values) in [
        (1, "category", &["seven", "eight"][..]),
        (3, "category", &["seven", "nine"][..]),
        (5, "category", &["nine", "ten"][..]),
        (7, "other", &["seven"][..]),
        (9, "learned only", &["thirty"][..]),
    ] {
        add_episode(&database, timestamp, attribute, values);
    }

    assert_eq!(
        recall_scores(&database, 0),
        vec![(1, 171_428), (3, 57_142), (2, 44_444)]
    );

    let positive_checkpoints = [
        (1, 71_875, 1),
        (2, 127_777, 1),
        (3, 172_500, 0),
        (4, 209_090, 0),
        (8, 306_666, 0),
        (16, 400_000, 0),
        (17, 400_000, 0),
    ];
    for sample in 1..=17 {
        assert_silent_success(feedback(&database, 0, true, "4"));
        if let Some(&(_, expected_score, expected_rank)) = positive_checkpoints
            .iter()
            .find(|&&(checkpoint, _, _)| checkpoint == sample)
        {
            let scores = recall_scores(&database, 0);
            assert_eq!(score_for(&scores, 4), Some(expected_score));
            assert_eq!(scores[expected_rank].0, 4);
        }
    }

    for sample in 1..=16 {
        assert_silent_success(feedback(&database, 0, false, "4"));
        if sample == 1 {
            let scores = recall_scores(&database, 0);
            assert_eq!(score_for(&scores, 4), Some(350_000));
            assert_eq!(scores[0].0, 4);
        }
        if matches!(sample, 8 | 9 | 16) {
            assert_eq!(score_for(&recall_scores(&database, 0), 4), None);
        }
    }

    for expected_score in [Some(99_553), Some(43_651), None, None] {
        assert_silent_success(feedback(&database, 0, false, "1"));
        assert_eq!(score_for(&recall_scores(&database, 0), 1), expected_score);
    }

    let store = SqliteStore::open(&database).expect("feedback histories reopen");
    let memory_id = store.memory_id();
    assert_eq!(
        store.memory().feedback_trace(
            AtomId::from_parts(memory_id, 0),
            AtomId::from_parts(memory_id, 4),
        ),
        Some(FeedbackTrace::from_parts(0, 16).unwrap())
    );
}

#[test]
fn feedback_is_silent_persists_and_changes_the_next_recall() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    assert_eq!(success_text(add_minimal(&database, 1, false)), "0\n");
    assert_eq!(success_text(add_minimal(&database, 2, false)), "1\n");

    assert_silent_success(feedback(&database, 0, true, "1,1,0"));
    let store = SqliteStore::open(&database).unwrap();
    let source = AtomId::from_parts(store.memory_id(), 0);
    let target = AtomId::from_parts(store.memory_id(), 1);
    assert_eq!(
        store.memory().feedback_trace(source, target),
        Some(FeedbackTrace::from_parts(1, 1).unwrap())
    );
    drop(store);

    let recalled = success_text(recall(&database, 0, None));
    assert!(recalled.starts_with("sequence 1\nactivation_ppm 71875\n"));

    assert_silent_success(feedback(&database, 0, false, "1"));
    assert_silent_success(recall(&database, 0, None));
    assert_eq!(
        SqliteStore::open(&database)
            .unwrap()
            .memory()
            .feedback_trace(source, target),
        Some(FeedbackTrace::from_parts(2, 2).unwrap())
    );
}

#[test]
fn feedback_runtime_failure_leaves_learning_unchanged() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    assert_silent_success(add_minimal(&database, 1, true));
    assert_silent_success(add_minimal(&database, 2, true));

    let stderr = failure(feedback(&database, 0, true, "1,99"), 1);
    assert!(stderr.contains("unknown atom"));
    let store = SqliteStore::open(&database).unwrap();
    let source = AtomId::from_parts(store.memory_id(), 0);
    let target = AtomId::from_parts(store.memory_id(), 1);
    assert_eq!(store.memory().feedback_trace(source, target), None);
}

#[test]
fn feedback_enforces_the_raw_target_limit_without_changing_a_rejected_snapshot() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    assert_silent_success(add_minimal(&database, 1, true));
    assert_silent_success(add_minimal(&database, 2, true));

    let maximum_targets_with_trailing_comma = "1,".repeat(MAX_FEEDBACK_TARGETS);
    let maximum_targets = maximum_targets_with_trailing_comma
        .strip_suffix(',')
        .expect("the target limit is non-zero");
    assert_silent_success(feedback(&database, 0, true, maximum_targets));

    let mut revision_witness = SqliteStore::open(&database).unwrap();
    let too_many_targets = format!("{maximum_targets},1");
    let stderr = failure(feedback(&database, 0, false, &too_many_targets), 1);
    assert!(stderr.contains("feedback target count"));

    let reopened = SqliteStore::open(&database).unwrap();
    assert_eq!(reopened.memory_id(), revision_witness.memory_id());
    assert!(
        reopened
            .memory()
            .episodes()
            .eq(revision_witness.memory().episodes())
    );
    assert!(
        reopened
            .memory()
            .feedback_edges()
            .eq(revision_witness.memory().feedback_edges())
    );
    drop(reopened);
    revision_witness
        .save()
        .expect("rejected feedback did not advance the snapshot revision");
}
