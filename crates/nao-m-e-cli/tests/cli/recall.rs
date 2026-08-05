use std::fs;

use nao_m_e_sqlite::SqliteStore;
use tempfile::TempDir;

use super::support::{
    assert_silent_success, cli, failure, init, invoke, semantic_recall, snapshot_revision,
};

#[test]
fn semantic_recall_empty_and_zero_limit_paths_are_offline_and_read_only() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    let absent_cache = directory.path().join("absent-model-cache");
    init(&database);
    let before = fs::read(&database).expect("empty store is readable before recall");
    let revision = snapshot_revision(&database);
    let mut revision_witness = SqliteStore::open(&database).expect("revision witness opens");

    for limit in [None, Some(0), Some(1)] {
        let mut command = cli();
        command
            .arg("recall")
            .arg(&database)
            .args(["--query", "login bug in lamentis with http 404"])
            .env("HF_HUB_OFFLINE", "1")
            .env_remove("HF_HUB_CACHE")
            .env_remove("HUGGINGFACE_HUB_CACHE")
            .env("HF_HOME", &absent_cache);
        if let Some(limit) = limit {
            command.arg("--limit").arg(limit.to_string());
        }
        assert_silent_success(invoke(command, None));
    }

    assert_eq!(fs::read(&database).unwrap(), before);
    assert_eq!(snapshot_revision(&database), revision);
    assert!(!absent_cache.exists());
    revision_witness
        .save()
        .expect("read-only recall did not advance the snapshot revision");
}

#[test]
fn semantic_recall_rejects_invalid_queries_and_removed_source_syntax_without_mutation() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    let before = fs::read(&database).expect("empty store is readable before rejected recall");
    let revision = snapshot_revision(&database);

    for query in ["", " \t\n ", "\u{7}"] {
        let stderr = failure(semantic_recall(&database, query, Some(0)), 1);
        assert!(stderr.contains("invalid semantic query"));
    }
    let oversized = "x".repeat(4_097);
    let stderr = failure(semantic_recall(&database, &oversized, Some(0)), 1);
    assert!(stderr.contains("exceeds 4096 bytes"));

    let mut removed = cli();
    removed.arg("recall").arg(&database).args(["--from", "0"]);
    let stderr = failure(invoke(removed, None), 2);
    assert!(stderr.contains("--query <TEXT>"));

    let mut mixed = cli();
    mixed
        .arg("recall")
        .arg(&database)
        .args(["--query", "query", "--from", "0"]);
    failure(invoke(mixed, None), 2);

    assert_eq!(fs::read(&database).unwrap(), before);
    assert_eq!(snapshot_revision(&database), revision);
}

#[test]
fn semantic_recall_syntax_and_option_like_query_keep_exit_contracts() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    let before = fs::read(&database).expect("empty store is readable before syntax checks");

    for arguments in [
        vec!["recall", database.to_str().unwrap()],
        vec!["recall", database.to_str().unwrap(), "--query"],
        vec![
            "recall",
            database.to_str().unwrap(),
            "--query",
            "query",
            "--limit",
        ],
        vec![
            "recall",
            database.to_str().unwrap(),
            "--query",
            "query",
            "--limit",
            "invalid",
        ],
        vec![
            "recall",
            database.to_str().unwrap(),
            "--limit",
            "0",
            "--query",
            "query",
        ],
    ] {
        let mut command = cli();
        command.args(arguments);
        failure(invoke(command, None), 2);
    }

    assert_silent_success(semantic_recall(&database, "--from", Some(0)));
    assert_eq!(fs::read(&database).unwrap(), before);
}
