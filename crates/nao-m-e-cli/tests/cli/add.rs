use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use nao_m_e::{EpisodeAtom, Statement};
use nao_m_e_sqlite::SqliteStore;
use tempfile::TempDir;

use super::support::{
    add_minimal, assert_silent_success, cli, failure, init, invoke, success_text,
};

fn assert_statement(store: &SqliteStore, actual: &Statement, predicate: &str, terms: &[&str]) {
    assert_eq!(
        store
            .predicate_values(&[actual.predicate()])
            .expect("predicate symbol resolves"),
        [Some(predicate.to_owned())]
    );
    assert_eq!(
        store
            .term_values(actual.arguments())
            .expect("term symbols resolve"),
        terms
            .iter()
            .map(|term| Some((*term).to_owned()))
            .collect::<Vec<_>>()
    );
}

fn assert_minimal_episode(store: &SqliteStore, atom: &EpisodeAtom, seed: u64) {
    assert_eq!(atom.timestamp().get(), i64::try_from(seed).unwrap());
    assert!(atom.context().is_empty());
    assert_statement(
        store,
        atom.observation(),
        &format!("predicate-{seed}"),
        &[&format!("term-{seed}")],
    );
    assert_eq!(atom.action(), None);
    assert_eq!(atom.outcome(), None);
}

#[test]
fn direct_add_outputs_only_the_assigned_sequence_and_quiet_is_silent() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let mut rich = cli();
    rich.arg("add")
        .arg(&database)
        .arg("--timestamp")
        .arg("-5")
        .arg("--predicate")
        .arg("  OBSERVE: Build  ")
        .arg("--term")
        .arg(" Linux, ARM64 ")
        .arg("--term")
        .arg("Release Candidate")
        .arg("--context")
        .arg(" Environment ")
        .arg("--context-term")
        .arg(" CI : Linux ")
        .arg("--context")
        .arg("Project")
        .arg("--context-term")
        .arg(" Nao M E ")
        .arg("--context")
        .arg("environment")
        .arg("--context-term")
        .arg("ci : linux")
        .arg("--action")
        .arg(" Execute ")
        .arg("--action-term")
        .arg(" Cargo Test ")
        .arg("--outcome")
        .arg(" Result ")
        .arg("--outcome-term")
        .arg(" Pass ")
        .arg("--outcome-term")
        .arg("No Warnings");
    assert_eq!(success_text(invoke(rich, None)), "0\n");
    assert_silent_success(add_minimal(&database, 8, true));
    let mut option_like_text = cli();
    option_like_text
        .arg("add")
        .arg(&database)
        .args(["--timestamp", "9"])
        .arg("--predicate")
        .arg("--quiet")
        .arg("--term")
        .arg("--many")
        .arg("--quiet");
    assert_silent_success(invoke(option_like_text, None));

    let store = SqliteStore::open(&database).expect("added episodes reopen");
    let atoms: Vec<_> = store.memory().episodes().collect();
    assert_eq!(atoms.len(), 3);
    assert_eq!(atoms[0].id().sequence(), 0);
    assert_eq!(atoms[0].timestamp().get(), -5);
    assert_eq!(atoms[0].context().len(), 2);
    assert_statement(
        &store,
        &atoms[0].context()[0],
        "environment",
        &["ci : linux"],
    );
    assert_statement(&store, &atoms[0].context()[1], "project", &["nao m e"]);
    assert_statement(
        &store,
        atoms[0].observation(),
        "observe: build",
        &["linux, arm64", "release candidate"],
    );
    assert_statement(
        &store,
        atoms[0].action().expect("action is stored"),
        "execute",
        &["cargo test"],
    );
    assert_statement(
        &store,
        atoms[0].outcome().expect("outcome is stored"),
        "result",
        &["pass", "no warnings"],
    );
    assert_eq!(atoms[1].id().sequence(), 1);
    assert_minimal_episode(&store, atoms[1], 8);
    assert_statement(&store, atoms[2].observation(), "--quiet", &["--many"]);
}

#[test]
fn direct_add_round_trips_integer_boundaries() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    for (sequence, timestamp) in [i64::MIN, i64::MAX].into_iter().enumerate() {
        let mut command = cli();
        command
            .arg("add")
            .arg(&database)
            .arg("--timestamp")
            .arg(timestamp.to_string())
            .arg("--predicate")
            .arg("Boundary")
            .arg("--term")
            .arg(timestamp.to_string());
        assert_eq!(success_text(invoke(command, None)), format!("{sequence}\n"));
    }

    let store = SqliteStore::open(&database).unwrap();
    let atoms = store.memory().episodes().collect::<Vec<_>>();
    assert_eq!(atoms[0].timestamp().get(), i64::MIN);
    assert_eq!(atoms[1].timestamp().get(), i64::MAX);
}

#[test]
fn add_defaults_to_current_unix_milliseconds() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let before = unix_milliseconds_now();
    let mut command = cli();
    command
        .arg("add")
        .arg(&database)
        .args(["--predicate", "Now", "--term", "Default"]);
    assert_eq!(success_text(invoke(command, None)), "0\n");
    let after = unix_milliseconds_now();

    let store = SqliteStore::open(&database).unwrap();
    let timestamp = store
        .memory()
        .episodes()
        .next()
        .expect("episode is stored")
        .timestamp()
        .get();
    assert!((before..=after).contains(&timestamp));
}

fn unix_milliseconds_now() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test clock follows the Unix epoch")
            .as_millis(),
    )
    .expect("test clock fits the supported timestamp range")
}

#[test]
fn many_add_shares_one_default_timestamp_preserves_explicit_values_and_is_atomic() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let input = "# ignored comment\n\n--predicate 'Predicate-1' --term 'Term-1' # trailing comment\r\n  \n--timestamp -3 --predicate \"Predicate 2\" --term 'Term 2' --term 'Term, 3' --context 'Context: Value' --context-term 'Context Term' --action Action --action-term 'Action Term' --outcome Outcome --outcome-term 'Outcome Term'\n--predicate '--quiet' --term '--many' --term escaped\\ value --term \"quoted \\\"value\\\"\"\n";
    let before_default = unix_milliseconds_now();
    let mut many = cli();
    many.arg("add").arg(&database).arg("--many");
    assert_eq!(success_text(invoke(many, Some(input))), "0\n1\n2\n");
    let after_default = unix_milliseconds_now();

    let mut quiet = cli();
    quiet.arg("add").arg(&database).arg("--many").arg("--quiet");
    assert_silent_success(invoke(
        quiet,
        Some("--timestamp 3 --predicate 'Predicate-3' --term 'Term-3'\n"),
    ));

    let store = SqliteStore::open(&database).unwrap();
    let atoms: Vec<_> = store.memory().episodes().collect();
    assert_eq!(atoms.len(), 4);
    let default_timestamp = atoms[0].timestamp().get();
    assert!((before_default..=after_default).contains(&default_timestamp));
    assert_eq!(atoms[2].timestamp().get(), default_timestamp);
    assert_statement(&store, atoms[0].observation(), "predicate-1", &["term-1"]);
    assert_eq!(atoms[1].id().sequence(), 1);
    assert_eq!(atoms[1].timestamp().get(), -3);
    assert_statement(
        &store,
        atoms[1].observation(),
        "predicate 2",
        &["term 2", "term, 3"],
    );
    assert_statement(
        &store,
        &atoms[1].context()[0],
        "context: value",
        &["context term"],
    );
    assert_statement(
        &store,
        atoms[1].action().unwrap(),
        "action",
        &["action term"],
    );
    assert_statement(
        &store,
        atoms[1].outcome().unwrap(),
        "outcome",
        &["outcome term"],
    );
    assert_statement(
        &store,
        atoms[2].observation(),
        "--quiet",
        &["--many", "escaped value", "quoted \"value\""],
    );
    assert_minimal_episode(&store, atoms[3], 3);
    drop(store);

    let before = fs::read(&database).expect("database is readable before rejected batch");
    let invalid =
        "--timestamp 4 --predicate Four --term Four\n--timestamp 5 --predicate Five --term\n";
    let mut rejected = cli();
    rejected.arg("add").arg(&database).arg("--many");
    failure(invoke(rejected, Some(invalid)), 1);
    assert_eq!(fs::read(&database).unwrap(), before);

    let mut invalid_quote = cli();
    invalid_quote.arg("add").arg(&database).arg("--many");
    let stderr = failure(
        invoke(
            invalid_quote,
            Some("--timestamp 4 --predicate 'unterminated --term value\n"),
        ),
        1,
    );
    assert!(stderr.contains("invalid shell quoting"));

    let mut invalid_symbol = cli();
    invalid_symbol.arg("add").arg(&database).arg("--many");
    failure(
        invoke(
            invalid_symbol,
            Some("--timestamp 4 --predicate Four --term '   '\n"),
        ),
        1,
    );
    assert_eq!(fs::read(&database).unwrap(), before);

    let mut empty = cli();
    empty.arg("add").arg(&database).arg("--many");
    failure(invoke(empty, Some("\n  \r\n")), 1);
    assert_eq!(
        SqliteStore::open(&database)
            .unwrap()
            .memory()
            .episodes()
            .len(),
        4
    );
}

#[test]
fn many_add_rejects_duplicate_flags_and_episode_options_without_changing_the_store() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    let before = fs::read(&database).expect("database is readable before rejected commands");

    for (options, error) in [
        (
            &["--many", "--many"][..],
            "`--many` may be specified only once",
        ),
        (
            &["--many", "--quiet", "--quiet"][..],
            "`--quiet` may be specified only once",
        ),
        (
            &["--many", "--timestamp"][..],
            "`--many` cannot be combined with episode option `--timestamp`",
        ),
    ] {
        let mut command = cli();
        command.arg("add").arg(&database).args(options);
        let stderr = failure(invoke(command, None), 2);
        assert!(stderr.starts_with(&format!("nao-m-e: {error}\n\n")));
        assert_eq!(fs::read(&database).unwrap(), before);
    }
}

#[test]
fn removed_episode_metadata_flags_are_rejected_without_changing_the_store() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    let before = fs::read(&database).expect("database is readable before rejected commands");

    for option in ["--occurred", "--recorded", "--source"] {
        let mut command = cli();
        command
            .arg("add")
            .arg(&database)
            .arg(option)
            .arg("1")
            .args(["--predicate", "Observation", "--term", "Value"]);
        let stderr = failure(invoke(command, None), 2);
        assert!(stderr.contains(&format!("unknown episode option `{option}`")));
        assert_eq!(fs::read(&database).unwrap(), before);
    }
}
