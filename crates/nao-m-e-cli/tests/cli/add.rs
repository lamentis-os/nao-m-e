use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use nao_m_e::{Attribute, EpisodeAtom};
use nao_m_e_sqlite::SqliteStore;
use tempfile::TempDir;

use super::support::{
    add_minimal, assert_silent_success, check, cli, failure, init, invoke, seed_cues,
    snapshot_revision, success_text,
};

fn assert_attribute(store: &SqliteStore, actual: &Attribute, key: &str, values: &[&str]) {
    assert_eq!(
        store
            .symbol_values(&[actual.key()])
            .expect("attribute key resolves"),
        [Some(key.to_owned())]
    );
    assert_eq!(
        store
            .symbol_values(actual.values())
            .expect("attribute values resolve"),
        values
            .iter()
            .map(|value| Some((*value).to_owned()))
            .collect::<Vec<_>>()
    );
}

fn assert_minimal_episode(store: &SqliteStore, atom: &EpisodeAtom, seed: u64) {
    assert_eq!(atom.timestamp().get(), i64::try_from(seed).unwrap());
    assert_eq!(atom.attributes().len(), 1);
    assert_attribute(
        store,
        &atom.attributes()[0],
        &format!("attribute-{seed}"),
        &[&format!("value-{seed}")],
    );
}

#[test]
fn direct_add_outputs_only_the_assigned_sequence_and_quiet_is_silent() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    seed_cues(
        &database,
        [
            (" Environment ", " CI : Linux "),
            ("Project", " Nao M E "),
            ("  OBSERVE: Build  ", " Linux, ARM64 "),
            ("  OBSERVE: Build  ", "Release Candidate"),
            (" Execute ", " Cargo Test "),
            (" Result ", " Pass "),
            (" Result ", "No Warnings"),
            ("--quiet", "--many"),
        ],
    );
    let revision_before_rich = snapshot_revision(&database);

    let mut rich = cli();
    rich.arg("add")
        .arg(&database)
        .arg("--timestamp")
        .arg("-5")
        .arg("--attribute")
        .arg(" Environment ")
        .arg("--value")
        .arg(" CI : Linux ")
        .arg("--attribute")
        .arg("Project")
        .arg("--value")
        .arg(" Nao M E ")
        .arg("--attribute")
        .arg("environment")
        .arg("--value")
        .arg("ci : linux")
        .arg("--attribute")
        .arg("  OBSERVE: Build  ")
        .arg("--value")
        .arg(" Linux, ARM64 ")
        .arg("--value")
        .arg("Release Candidate")
        .arg("--attribute")
        .arg(" Execute ")
        .arg("--value")
        .arg(" Cargo Test ")
        .arg("--attribute")
        .arg(" Result ")
        .arg("--value")
        .arg(" Pass ")
        .arg("--value")
        .arg("No Warnings");
    assert_eq!(success_text(invoke(rich, None)), "0\n");
    assert_eq!(snapshot_revision(&database), revision_before_rich + 1);
    assert_silent_success(add_minimal(&database, 8, true));
    let mut option_like_text = cli();
    option_like_text
        .arg("add")
        .arg(&database)
        .args(["--timestamp", "9"])
        .arg("--attribute")
        .arg("--quiet")
        .arg("--value")
        .arg("--many")
        .arg("--quiet");
    assert_silent_success(invoke(option_like_text, None));

    let store = SqliteStore::open(&database).expect("added episodes reopen");
    let atoms: Vec<_> = store.memory().episodes().collect();
    assert_eq!(atoms.len(), 3);
    assert_eq!(atoms[0].id().sequence(), 0);
    assert_eq!(atoms[0].timestamp().get(), -5);
    assert_eq!(atoms[0].attributes().len(), 5);
    assert_attribute(
        &store,
        &atoms[0].attributes()[0],
        "environment",
        &["ci : linux"],
    );
    assert_attribute(&store, &atoms[0].attributes()[1], "project", &["nao m e"]);
    assert_attribute(
        &store,
        &atoms[0].attributes()[2],
        "observe: build",
        &["linux, arm64", "release candidate"],
    );
    assert_attribute(
        &store,
        &atoms[0].attributes()[3],
        "execute",
        &["cargo test"],
    );
    assert_attribute(
        &store,
        &atoms[0].attributes()[4],
        "result",
        &["pass", "no warnings"],
    );
    assert_eq!(atoms[1].id().sequence(), 1);
    assert_minimal_episode(&store, atoms[1], 8);
    assert_attribute(&store, &atoms[2].attributes()[0], "--quiet", &["--many"]);
    drop(store);
    assert_silent_success(check(&database));
}

#[test]
fn direct_add_round_trips_integer_boundaries() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    for (sequence, timestamp) in [i64::MIN, i64::MAX].into_iter().enumerate() {
        seed_cues(&database, [("Boundary", timestamp.to_string())]);
        let mut command = cli();
        command
            .arg("add")
            .arg(&database)
            .arg("--timestamp")
            .arg(timestamp.to_string())
            .arg("--attribute")
            .arg("Boundary")
            .arg("--value")
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
    seed_cues(&database, [("Now", "Default")]);

    let before = unix_milliseconds_now();
    let mut command = cli();
    command
        .arg("add")
        .arg(&database)
        .args(["--attribute", "Now", "--value", "Default"]);
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
    seed_cues(
        &database,
        [
            ("Attribute-1", "Value-1"),
            ("Attribute 2", "Value 2"),
            ("Attribute 2", "Value, 3"),
            ("Context: Value", "Context Value"),
            ("Action", "Action Value"),
            ("Outcome", "Outcome Value"),
            ("--quiet", "--many"),
            ("--quiet", "escaped value"),
            ("--quiet", "quoted \"value\""),
            ("Attribute-3", "Value-3"),
        ],
    );

    let input = "# ignored comment\n\n--attribute 'Attribute-1' --value 'Value-1' # trailing comment\r\n  \n--timestamp -3 --attribute \"Attribute 2\" --value 'Value 2' --value 'Value, 3' --attribute 'Context: Value' --value 'Context Value' --attribute Action --value 'Action Value' --attribute Outcome --value 'Outcome Value'\n--attribute '--quiet' --value '--many' --value escaped\\ value --value \"quoted \\\"value\\\"\"\n";
    let before_default = unix_milliseconds_now();
    let mut many = cli();
    many.arg("add").arg(&database).arg("--many");
    assert_eq!(success_text(invoke(many, Some(input))), "0\n1\n2\n");
    let after_default = unix_milliseconds_now();

    let mut quiet = cli();
    quiet.arg("add").arg(&database).arg("--many").arg("--quiet");
    assert_silent_success(invoke(
        quiet,
        Some("--timestamp 3 --attribute 'Attribute-3' --value 'Value-3'\n"),
    ));

    let store = SqliteStore::open(&database).unwrap();
    let atoms: Vec<_> = store.memory().episodes().collect();
    assert_eq!(atoms.len(), 4);
    let default_timestamp = atoms[0].timestamp().get();
    assert!((before_default..=after_default).contains(&default_timestamp));
    assert_eq!(atoms[2].timestamp().get(), default_timestamp);
    assert_attribute(
        &store,
        &atoms[0].attributes()[0],
        "attribute-1",
        &["value-1"],
    );
    assert_eq!(atoms[1].id().sequence(), 1);
    assert_eq!(atoms[1].timestamp().get(), -3);
    assert_eq!(atoms[1].attributes().len(), 4);
    assert_attribute(
        &store,
        &atoms[1].attributes()[0],
        "attribute 2",
        &["value 2", "value, 3"],
    );
    assert_attribute(
        &store,
        &atoms[1].attributes()[1],
        "context: value",
        &["context value"],
    );
    assert_attribute(
        &store,
        &atoms[1].attributes()[2],
        "action",
        &["action value"],
    );
    assert_attribute(
        &store,
        &atoms[1].attributes()[3],
        "outcome",
        &["outcome value"],
    );
    assert_attribute(
        &store,
        &atoms[2].attributes()[0],
        "--quiet",
        &["--many", "escaped value", "quoted \"value\""],
    );
    assert_minimal_episode(&store, atoms[3], 3);
    drop(store);

    let before = fs::read(&database).expect("database is readable before rejected batch");
    let invalid =
        "--timestamp 4 --attribute Four --value Four\n--timestamp 5 --attribute Five --value\n";
    let mut rejected = cli();
    rejected.arg("add").arg(&database).arg("--many");
    failure(invoke(rejected, Some(invalid)), 1);
    assert_eq!(fs::read(&database).unwrap(), before);

    let mut invalid_quote = cli();
    invalid_quote.arg("add").arg(&database).arg("--many");
    let stderr = failure(
        invoke(
            invalid_quote,
            Some("--timestamp 4 --attribute 'unterminated --value value\n"),
        ),
        1,
    );
    assert!(stderr.contains("invalid shell quoting"));

    let mut row_quiet = cli();
    row_quiet.arg("add").arg(&database).arg("--many");
    let stderr = failure(
        invoke(row_quiet, Some("--attribute value --value value --quiet\n")),
        1,
    );
    assert!(stderr.contains("`--quiet` is not valid inside an add --many row"));
    assert_eq!(fs::read(&database).unwrap(), before);

    let mut invalid_symbol = cli();
    invalid_symbol.arg("add").arg(&database).arg("--many");
    failure(
        invoke(
            invalid_symbol,
            Some("--timestamp 4 --attribute Four --value '   '\n"),
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
fn removed_episode_role_and_metadata_flags_are_rejected_without_changing_the_store() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    let before = fs::read(&database).expect("database is readable before rejected commands");

    for option in [
        "--predicate",
        "--term",
        "--context",
        "--context-term",
        "--action",
        "--action-term",
        "--outcome",
        "--outcome-term",
        "--occurred",
        "--recorded",
        "--source",
    ] {
        let mut command = cli();
        command
            .arg("add")
            .arg(&database)
            .arg(option)
            .arg("1")
            .args(["--attribute", "valid", "--value", "value"]);
        let stderr = failure(invoke(command, None), 2);
        assert!(stderr.contains(&format!("unknown episode option `{option}`")));
        assert_eq!(fs::read(&database).unwrap(), before);
    }
}
