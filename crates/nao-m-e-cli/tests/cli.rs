use std::fs;
use std::path::Path;

use nao_m_e_sqlite::SqliteStore;
use tempfile::TempDir;

#[path = "cli/add.rs"]
mod add;
#[path = "cli/feedback.rs"]
mod feedback;
#[path = "cli/recall.rs"]
mod recall;
#[path = "cli/support.rs"]
mod support;

use support::{add_minimal, cli, failure, init, invoke, recall, success_text};

fn rewrite_format_version(database: &Path, memory_id: [u8; 16], from: u8, to: u8) {
    let mut bytes = fs::read(database).expect("database is readable");
    let positions = bytes
        .windows(memory_id.len())
        .enumerate()
        .filter_map(|(position, candidate)| (candidate == memory_id).then_some(position))
        .collect::<Vec<_>>();
    assert_eq!(
        positions.len(),
        1,
        "memory ID occurs once in the SQLite file"
    );
    let version_position = positions[0]
        .checked_sub(1)
        .expect("format version precedes the memory ID");
    assert_eq!(bytes[version_position], from);
    bytes[version_position] = to;
    fs::write(database, bytes).expect("test format version is rewritten");
}

#[test]
fn help_version_and_top_level_syntax_have_stable_exit_categories() {
    for arguments in [
        &["--help"][..],
        &["init", "--help"][..],
        &["add", "--help"][..],
        &["recall", "--help"][..],
        &["feedback", "--help"][..],
    ] {
        let mut command = cli();
        command.args(arguments);
        let output = invoke(command, None);
        let stdout = success_text(output);
        assert!(stdout.contains("Usage:"));
        assert!(!stdout.contains("JSON"));
    }

    let mut recall_help = cli();
    recall_help.args(["recall", "--help"]);
    let recall_help = success_text(invoke(recall_help, None));
    assert!(recall_help.contains("Symbolic cue overlap provides cold candidates"));
    assert!(recall_help.contains("Direct learned feedback"));
    assert!(recall_help.contains("suppress structural matches"));

    let mut version = cli();
    version.arg("--version");
    assert!(success_text(invoke(version, None)).starts_with("nao-m-e 0.0.1"));

    let stderr = failure(invoke(cli(), None), 2);
    assert!(stderr.contains("command"));

    let mut unknown = cli();
    unknown.arg("unknown");
    failure(invoke(unknown, None), 2);
}

#[test]
fn init_is_silent_creates_current_format_and_never_clobbers() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");

    init(&database);
    let store = SqliteStore::open(&database).expect("created SQLite store opens");
    assert_eq!(store.memory().episodes().len(), 0);
    drop(store);

    let original = fs::read(&database).expect("database is readable");
    let mut again = cli();
    again.arg("init").arg(&database);
    let stderr = failure(invoke(again, None), 1);
    assert!(stderr.contains("could not create"));
    assert_eq!(fs::read(&database).unwrap(), original);
}

#[test]
fn commands_reject_unsupported_format_before_execution_without_changing_it() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let store = SqliteStore::open(&database).expect("created SQLite store opens");
    let memory_id = store.memory_id().to_be_bytes();
    drop(store);
    rewrite_format_version(&database, memory_id, 4, 5);
    let before = fs::read(&database).expect("unsupported-format store is readable");

    let stderr = failure(recall(&database, 0, None), 1);
    assert!(stderr.contains("unsupported SQLite memory format version 5"));
    assert_eq!(fs::read(&database).unwrap(), before);
}

#[test]
fn malformed_direct_arguments_are_syntax_errors_but_store_errors_are_runtime_errors() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let invalid_commands = [
        vec!["add", database.to_str().unwrap()],
        vec!["recall", database.to_str().unwrap(), "--limit", "1"],
        vec![
            "feedback",
            database.to_str().unwrap(),
            "--from",
            "0",
            "--helpful",
            "1",
            "--unhelpful",
            "1",
        ],
    ];
    for arguments in invalid_commands {
        let mut command = cli();
        command.args(arguments);
        failure(invoke(command, None), 2);
    }

    let mut malformed_statement = cli();
    malformed_statement.arg("add").arg(&database).args([
        "--occurred",
        "1",
        "--recorded",
        "2",
        "--source",
        "3",
        "--predicate",
        "4",
        "--terms",
        "1,",
    ]);
    failure(invoke(malformed_statement, None), 2);

    for episode_options in [
        vec![
            "--predicate",
            "observation",
            "--context",
            "context",
            "--term",
            "late observation term",
            "--context-term",
            "context term",
        ],
        vec![
            "--predicate",
            "observation",
            "--term",
            "observation term",
            "--context",
            "context",
            "--context-term",
            "context term",
            "--term",
            "late observation term",
        ],
        vec![
            "--predicate",
            "observation",
            "--term",
            "observation term",
            "--action",
            "action",
            "--action-term",
            "action term",
            "--outcome",
            "outcome",
            "--outcome-term",
            "outcome term",
            "--action-term",
            "late action term",
        ],
    ] {
        let mut command = cli();
        command
            .arg("add")
            .arg(&database)
            .args(["--occurred", "1", "--recorded", "2", "--source", "3"])
            .args(episode_options);
        failure(invoke(command, None), 2);
    }

    let stderr = failure(recall(&database, 99, None), 1);
    assert!(stderr.contains("unknown atom"));

    let missing = directory.path().join("missing.sqlite3");
    let stderr = failure(add_minimal(&missing, 1, false), 1);
    assert!(stderr.contains("could not open"));
}
