use std::fs;

use nao_m_e_sqlite::SqliteStore;
use tempfile::TempDir;

use super::support::{cli, failure, init, invoke, snapshot_revision};

#[test]
fn add_requires_provisioned_assets_and_publishes_nothing_on_cache_miss() {
    let directory = TempDir::new().unwrap();
    let database = directory.path().join("memory.sqlite3");
    let absent_cache = directory.path().join("absent-model-cache");
    init(&database);
    let before = fs::read(&database).unwrap();
    let revision = snapshot_revision(&database);

    let mut command = cli();
    command
        .arg("add")
        .arg(&database)
        .args([
            "--timestamp",
            "1",
            "--attribute",
            "project",
            "--value",
            "lamentis",
        ])
        .env_remove("HF_HUB_CACHE")
        .env_remove("HUGGINGFACE_HUB_CACHE")
        .env("HF_HOME", &absent_cache);
    let stderr = failure(invoke(command, None), 1);
    assert!(stderr.contains("required semantic artifact onnx/model.onnx is not provisioned"));

    assert_eq!(fs::read(&database).unwrap(), before);
    assert_eq!(snapshot_revision(&database), revision);
    assert_eq!(
        SqliteStore::open(&database)
            .unwrap()
            .memory()
            .episodes()
            .len(),
        0
    );
}

#[test]
fn add_many_parses_every_row_before_opening_or_encoding() {
    let directory = TempDir::new().unwrap();
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    let before = fs::read(&database).unwrap();

    let mut invalid = cli();
    invalid.arg("add").arg(&database).arg("--many");
    let stderr = failure(
        invoke(
            invalid,
            Some(
                "--timestamp 1 --attribute project --value lamentis\n\
                 --timestamp 2 --attribute broken --value\n",
            ),
        ),
        1,
    );
    assert!(stderr.contains("`--value` requires a value"));
    assert_eq!(fs::read(&database).unwrap(), before);

    let mut quote = cli();
    quote.arg("add").arg(&database).arg("--many");
    let stderr = failure(
        invoke(
            quote,
            Some("--timestamp 1 --attribute 'unterminated --value value\n"),
        ),
        1,
    );
    assert!(stderr.contains("invalid shell quoting"));
    assert_eq!(fs::read(&database).unwrap(), before);
}

#[test]
fn removed_episode_role_and_metadata_flags_are_syntax_errors() {
    let directory = TempDir::new().unwrap();
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    let before = fs::read(&database).unwrap();

    for obsolete in [
        "--occurred",
        "--recorded",
        "--source",
        "--predicate",
        "--term",
        "--context",
        "--action",
        "--outcome",
    ] {
        let mut command = cli();
        command.arg("add").arg(&database).arg(obsolete).arg("value");
        failure(invoke(command, None), 2);
    }
    assert_eq!(fs::read(&database).unwrap(), before);
}
