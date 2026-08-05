use std::fs;

use nao_m_e::{AtomId, FeedbackTrace};
use nao_m_e_sqlite::SqliteStore;
use tempfile::TempDir;

use super::support::{
    assert_silent_success, check, cli, failure, feedback, init, invoke, semantic_recall,
    snapshot_revision, success_text,
};

#[test]
fn feedback_errors_are_offline_and_leave_an_empty_store_unchanged() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    let absent_cache = directory.path().join("absent-model-cache");
    init(&database);
    let before = fs::read(&database).expect("empty store is readable before feedback");
    let revision = snapshot_revision(&database);

    for arguments in [
        vec!["feedback", database.to_str().unwrap()],
        vec![
            "feedback",
            database.to_str().unwrap(),
            "--from",
            "invalid",
            "--helpful",
            "1",
        ],
        vec![
            "feedback",
            database.to_str().unwrap(),
            "--from",
            "0",
            "--helpful",
            "",
        ],
        vec![
            "feedback",
            database.to_str().unwrap(),
            "--from",
            "0",
            "--helpful",
            "1,invalid",
        ],
        vec![
            "feedback",
            database.to_str().unwrap(),
            "--from",
            "0",
            "--maybe",
            "1",
        ],
    ] {
        let mut command = cli();
        command.args(arguments);
        failure(invoke(command, None), 2);
    }

    let mut runtime_failure = cli();
    runtime_failure
        .arg("feedback")
        .arg(&database)
        .args(["--from", "0", "--helpful", "1"])
        .env("HF_HUB_OFFLINE", "1")
        .env_remove("HF_HUB_CACHE")
        .env_remove("HUGGINGFACE_HUB_CACHE")
        .env("HF_HOME", &absent_cache);
    let stderr = failure(invoke(runtime_failure, None), 1);
    assert!(stderr.contains("unknown atom"));

    assert_eq!(fs::read(&database).unwrap(), before);
    assert_eq!(snapshot_revision(&database), revision);
    assert!(!absent_cache.exists());
}

#[test]
#[ignore = "requires and executes the provisioned pinned 470 MB E5 Small model"]
fn provisioned_semantic_recall_and_feedback_workflow_is_deterministic_and_durable() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let mut direct = cli();
    direct.arg("add").arg(&database).args([
        "--timestamp",
        "-5",
        "--attribute",
        " Project ",
        "--value",
        " Lamentis ",
        "--attribute",
        "Problem",
        "--value",
        "Login request returned HTTP 404",
    ]);
    assert_eq!(success_text(invoke(direct, None)), "0\n");

    let input = "\
--timestamp 1 --attribute component --value authentication --attribute event --value 'login request returned http 401 unauthorized' --attribute project --value lamentis --attribute status --value failed\n\
--timestamp 2 --attribute activity --value 'walked with the dog on the beach' --attribute context --value 'rainy afternoon' --attribute place --value beach --attribute type --value 'personal memory'\n\
--timestamp 3 --attribute problem --value 'linker cannot find sqlite library' --attribute project --value 'rust workspace' --attribute status --value failed --attribute tool --value 'cargo build'\n\
--timestamp 4 --attribute genre --value 'fictional movie' --attribute plot --value 'a dog walks on a beach while hackers investigate a lamentis login http 404' --attribute title --value 'the signal shore' --attribute type --value 'movie synopsis'\n";
    let mut add = cli();
    add.arg("add").arg(&database).args(["--many", "--quiet"]);
    assert_silent_success(invoke(add, Some(input)));
    assert_silent_success(check(&database));

    let query = "login request in lamentis returns http 404";
    let read_only_snapshot = fs::read(&database).expect("semantic snapshot is readable");
    let read_only_revision = snapshot_revision(&database);
    let semantic_before = success_text(semantic_recall(&database, query, Some(5)));
    assert_top_hit(&semantic_before, 0, "value login request returned http 404");
    assert_eq!(
        success_text(semantic_recall(&database, query, Some(5))),
        semantic_before
    );
    let dog = success_text(semantic_recall(
        &database,
        "personal memory walking the dog on the beach on a rainy afternoon",
        Some(1),
    ));
    assert_top_hit(&dog, 2, "value rainy afternoon");
    assert_eq!(fs::read(&database).unwrap(), read_only_snapshot);
    assert_eq!(snapshot_revision(&database), read_only_revision);

    let revision_before_feedback = snapshot_revision(&database);
    assert_silent_success(feedback(&database, 0, true, "1"));
    assert_eq!(snapshot_revision(&database), revision_before_feedback + 1);

    let committed = fs::read(&database).expect("feedback snapshot is readable");
    let committed_revision = snapshot_revision(&database);
    let semantic_after = success_text(semantic_recall(&database, query, Some(5)));
    assert_eq!(semantic_after, semantic_before);
    assert_eq!(fs::read(&database).unwrap(), committed);
    assert_eq!(snapshot_revision(&database), committed_revision);

    let store = SqliteStore::open(&database).expect("feedback snapshot reopens");
    assert_eq!(store.memory().episodes().len(), 5);
    assert_eq!(
        store.memory().episodes().next().unwrap().timestamp().get(),
        -5
    );
    let source = AtomId::from_parts(store.memory_id(), 0);
    let target = AtomId::from_parts(store.memory_id(), 1);
    assert_eq!(
        store.memory().feedback_trace(source, target),
        Some(FeedbackTrace::from_parts(1, 1).unwrap())
    );
    drop(store);
    assert_silent_success(check(&database));
    assert_eq!(fs::read(&database).unwrap(), committed);
    assert_eq!(snapshot_revision(&database), committed_revision);
}

fn assert_top_hit(output: &str, expected_sequence: u64, identifying_value: &str) {
    let mut lines = output.lines();
    let expected = format!("sequence {expected_sequence}");
    assert_eq!(lines.next(), Some(expected.as_str()));
    let score = lines
        .next()
        .and_then(|line| line.strip_prefix("activation_ppm "))
        .and_then(|score| score.parse::<u32>().ok())
        .expect("semantic hit has a numeric activation");
    assert!(score > 0);
    assert!(lines.any(|line| line == identifying_value));
}
