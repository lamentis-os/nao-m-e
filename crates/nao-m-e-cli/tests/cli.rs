use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use nao_m_e::{
    AtomId, EpisodeAtom, EpisodeDraft, InfluenceWeight, PredicateId, SourceId, Statement, TermId,
    TimestampMs,
};
use nao_m_e_sqlite::SqliteStore;
use tempfile::TempDir;

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_nao-m-e"))
}

fn invoke(mut command: Command, stdin: Option<&str>) -> Output {
    let Some(stdin) = stdin else {
        return command.output().expect("CLI process starts");
    };

    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("CLI process starts");
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(stdin.as_bytes())
        .expect("test input is written");
    child.wait_with_output().expect("CLI process exits")
}

fn success_text(output: Output) -> String {
    assert!(
        output.status.success(),
        "expected success, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    String::from_utf8(output.stdout).expect("stdout is UTF-8")
}

fn assert_silent_success(output: Output) {
    assert_eq!(success_text(output), "");
}

fn failure(output: Output, expected_code: i32) -> String {
    assert_eq!(output.status.code(), Some(expected_code));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).expect("stderr is UTF-8");
    assert!(!stderr.is_empty());
    stderr
}

fn init(database: &Path) {
    let mut command = cli();
    command.arg("init").arg(database);
    assert_silent_success(invoke(command, None));
}

fn add_minimal(database: &Path, seed: i64, quiet: bool) -> Output {
    let source = u64::try_from(seed).expect("test seed is non-negative");
    let mut command = cli();
    command
        .arg("add")
        .arg(database)
        .arg("--occurred")
        .arg(seed.to_string())
        .arg("--recorded")
        .arg((seed + 1).to_string())
        .arg("--source")
        .arg(source.to_string())
        .arg("--predicate")
        .arg((source + 10).to_string())
        .arg("--terms")
        .arg((source + 100).to_string());
    if quiet {
        command.arg("--quiet");
    }
    invoke(command, None)
}

fn recall(database: &Path, source: u64, limit: Option<usize>) -> Output {
    let mut command = cli();
    command
        .arg("recall")
        .arg(database)
        .arg("--from")
        .arg(source.to_string());
    if let Some(limit) = limit {
        command.arg("--limit").arg(limit.to_string());
    }
    invoke(command, None)
}

fn feedback(database: &Path, source: u64, helpful: bool, targets: &str) -> Output {
    let mut command = cli();
    command
        .arg("feedback")
        .arg(database)
        .arg("--from")
        .arg(source.to_string())
        .arg(if helpful { "--helpful" } else { "--unhelpful" })
        .arg(targets);
    invoke(command, None)
}

fn statement(predicate: u64, terms: &[u64]) -> Statement {
    Statement::new(
        PredicateId::new(predicate),
        terms.iter().copied().map(TermId::new).collect(),
    )
    .expect("test statement has terms")
}

fn draft(seed: u64) -> EpisodeDraft {
    EpisodeDraft {
        occurred_at: TimestampMs::new(i64::try_from(seed).expect("small test seed")),
        recorded_at: TimestampMs::new(i64::try_from(seed + 1).expect("small test seed")),
        context: Vec::new(),
        observation: statement(seed + 10, &[seed + 100]),
        action: None,
        outcome: None,
        source: SourceId::new(seed),
    }
}

fn assert_statement(actual: &Statement, predicate: u64, terms: &[u64]) {
    assert_eq!(actual.predicate().get(), predicate);
    assert_eq!(
        actual
            .arguments()
            .iter()
            .map(|term| term.get())
            .collect::<Vec<_>>(),
        terms
    );
}

fn assert_minimal_episode(atom: &EpisodeAtom, seed: u64) {
    assert_eq!(atom.occurred_at().get(), i64::try_from(seed).unwrap());
    assert_eq!(atom.recorded_at().get(), i64::try_from(seed + 1).unwrap());
    assert_eq!(atom.source().get(), seed);
    assert!(atom.context().is_empty());
    assert_statement(atom.observation(), seed + 10, &[seed + 100]);
    assert_eq!(atom.action(), None);
    assert_eq!(atom.outcome(), None);
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
fn init_is_silent_creates_v2_and_never_clobbers() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");

    init(&database);
    let store = SqliteStore::open(&database).expect("created SQLite V2 store opens");
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
fn direct_add_outputs_only_the_assigned_sequence_and_quiet_is_silent() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let mut rich = cli();
    rich.arg("add")
        .arg(&database)
        .arg("--occurred")
        .arg("-5")
        .arg("--recorded")
        .arg("6")
        .arg("--source")
        .arg("7")
        .arg("--predicate")
        .arg("20")
        .arg("--terms")
        .arg("200,201")
        .arg("--context")
        .arg("12:120,121")
        .arg("--context")
        .arg("11:110")
        .arg("--context")
        .arg("11:110")
        .arg("--action")
        .arg("21:210")
        .arg("--outcome")
        .arg("22:220,221");
    assert_eq!(success_text(invoke(rich, None)), "0\n");
    assert_silent_success(add_minimal(&database, 8, true));

    let store = SqliteStore::open(&database).expect("added episodes reopen");
    let atoms: Vec<_> = store.memory().episodes().collect();
    assert_eq!(atoms.len(), 2);
    assert_eq!(atoms[0].id().sequence(), 0);
    assert_eq!(atoms[0].occurred_at().get(), -5);
    assert_eq!(atoms[0].recorded_at().get(), 6);
    assert_eq!(atoms[0].source().get(), 7);
    assert_eq!(atoms[0].context().len(), 2);
    assert_statement(&atoms[0].context()[0], 11, &[110]);
    assert_statement(&atoms[0].context()[1], 12, &[120, 121]);
    assert_statement(atoms[0].observation(), 20, &[200, 201]);
    assert_statement(atoms[0].action().expect("action is stored"), 21, &[210]);
    assert_statement(
        atoms[0].outcome().expect("outcome is stored"),
        22,
        &[220, 221],
    );
    assert_eq!(atoms[1].id().sequence(), 1);
    assert_minimal_episode(atoms[1], 8);
}

#[test]
fn direct_add_round_trips_integer_boundaries() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let mut command = cli();
    command
        .arg("add")
        .arg(&database)
        .arg("--occurred")
        .arg(i64::MIN.to_string())
        .arg("--recorded")
        .arg(i64::MAX.to_string())
        .arg("--source")
        .arg(u64::MAX.to_string())
        .arg("--predicate")
        .arg((u64::MAX - 1).to_string())
        .arg("--terms")
        .arg(format!("0,{}", u64::MAX));
    assert_eq!(success_text(invoke(command, None)), "0\n");

    let store = SqliteStore::open(&database).unwrap();
    let atom = store.memory().episodes().next().expect("episode is stored");
    assert_eq!(atom.occurred_at().get(), i64::MIN);
    assert_eq!(atom.recorded_at().get(), i64::MAX);
    assert_eq!(atom.source().get(), u64::MAX);
    assert_statement(atom.observation(), u64::MAX - 1, &[0, u64::MAX]);
}

#[test]
fn many_add_ignores_empty_lines_is_atomic_and_assigns_input_order() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let input = "\n--occurred 1 --recorded 2 --source 1 --predicate 11 --terms 101\r\n  \n--occurred -3 --recorded -2 --source 2 --predicate 12 --terms 102,103 --context 20:200 --action 30:300 --outcome 40:400\n";
    let mut many = cli();
    many.arg("add").arg(&database).arg("--many");
    assert_eq!(success_text(invoke(many, Some(input))), "0\n1\n");

    let mut quiet = cli();
    quiet.arg("add").arg(&database).arg("--many").arg("--quiet");
    assert_silent_success(invoke(
        quiet,
        Some("--occurred 3 --recorded 4 --source 3 --predicate 13 --terms 103\n"),
    ));

    let store = SqliteStore::open(&database).unwrap();
    let atoms: Vec<_> = store.memory().episodes().collect();
    assert_eq!(atoms.len(), 3);
    assert_minimal_episode(atoms[0], 1);
    assert_eq!(atoms[1].id().sequence(), 1);
    assert_eq!(atoms[1].occurred_at().get(), -3);
    assert_statement(atoms[1].observation(), 12, &[102, 103]);
    assert_statement(&atoms[1].context()[0], 20, &[200]);
    assert_statement(atoms[1].action().unwrap(), 30, &[300]);
    assert_statement(atoms[1].outcome().unwrap(), 40, &[400]);
    assert_minimal_episode(atoms[2], 3);
    drop(store);

    let invalid = "--occurred 4 --recorded 5 --source 4 --predicate 14 --terms 104\n--occurred 5 --recorded 6 --source 5 --predicate 15 --terms\n";
    let mut rejected = cli();
    rejected.arg("add").arg(&database).arg("--many");
    failure(invoke(rejected, Some(invalid)), 1);

    let mut empty = cli();
    empty.arg("add").arg(&database).arg("--many");
    failure(invoke(empty, Some("\n  \r\n")), 1);
    assert_eq!(
        SqliteStore::open(&database)
            .unwrap()
            .memory()
            .episodes()
            .len(),
        3
    );
}

#[test]
fn recall_emits_exact_ranked_blocks_and_honors_limit() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&database).expect("existing SQLite V2 is created");
    let source = store.memory_mut().insert_episode(draft(0)).unwrap();
    let rich = store
        .memory_mut()
        .insert_episode(EpisodeDraft {
            occurred_at: TimestampMs::new(-7),
            recorded_at: TimestampMs::new(8),
            context: vec![
                statement(12, &[120, 121]),
                statement(11, &[110]),
                statement(11, &[110]),
            ],
            observation: statement(20, &[200, 201]),
            action: Some(statement(21, &[210])),
            outcome: Some(statement(22, &[220, 221])),
            source: SourceId::new(9),
        })
        .unwrap();
    let first_tie = store.memory_mut().insert_episode(draft(2)).unwrap();
    let second_tie = store.memory_mut().insert_episode(draft(3)).unwrap();
    store
        .memory_mut()
        .set_relevance(source, rich, InfluenceWeight::from_ppm(600_000).unwrap())
        .unwrap();
    for target in [first_tie, second_tie] {
        store
            .memory_mut()
            .set_relevance(source, target, InfluenceWeight::from_ppm(200_000).unwrap())
            .unwrap();
    }
    store.save().unwrap();
    drop(store);

    let expected = "sequence 1\nactivation_ppm 240000\noccurred -7\nrecorded 8\nsource 9\ncontext 11:110\ncontext 12:120,121\npredicate 20\nterms 200,201\naction 21:210\noutcome 22:220,221\n\nsequence 2\nactivation_ppm 80000\noccurred 2\nrecorded 3\nsource 2\npredicate 12\nterms 102\n\nsequence 3\nactivation_ppm 80000\noccurred 3\nrecorded 4\nsource 3\npredicate 13\nterms 103\n";
    assert_eq!(success_text(recall(&database, 0, None)), expected);

    let first_two = expected
        .rsplit_once("\n\nsequence 3")
        .expect("third block is present")
        .0
        .to_owned()
        + "\n";
    assert_eq!(success_text(recall(&database, 0, Some(2))), first_two);
}

#[test]
fn recall_with_no_hits_is_silent_and_does_not_advance_the_store() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    assert_eq!(success_text(add_minimal(&database, 1, false)), "0\n");
    assert_silent_success(recall(&database, 0, None));
    assert_silent_success(recall(&database, 0, Some(0)));

    let mut writer = SqliteStore::open(&database).expect("writer opens before recall");
    let source = AtomId::from_parts(writer.memory_id(), 0);
    assert_silent_success(recall(&database, 0, None));
    let target = writer.memory_mut().insert_episode(draft(2)).unwrap();
    writer
        .memory_mut()
        .set_relevance(source, target, InfluenceWeight::from_ppm(1_000).unwrap())
        .unwrap();
    writer
        .save()
        .expect("read-only recall did not advance the revision");
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
        store.memory().relevance(source, target),
        Some(InfluenceWeight::from_ppm(1_000).unwrap())
    );
    drop(store);

    let recalled = success_text(recall(&database, 0, None));
    assert!(recalled.starts_with("sequence 1\nactivation_ppm 400\n"));

    assert_silent_success(feedback(&database, 0, false, "1"));
    assert_silent_success(recall(&database, 0, None));
    assert_eq!(
        SqliteStore::open(&database)
            .unwrap()
            .memory()
            .relevance(source, target),
        None
    );
}

#[test]
fn feedback_runtime_failure_leaves_relevance_unchanged() {
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
    assert_eq!(store.memory().relevance(source, target), None);
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

    let stderr = failure(recall(&database, 99, None), 1);
    assert!(stderr.contains("unknown atom"));

    let missing = directory.path().join("missing.sqlite3");
    let stderr = failure(add_minimal(&missing, 1, false), 1);
    assert!(stderr.contains("could not open"));
}

#[test]
fn old_run_json_and_recall_v2_syntax_are_rejected() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let mut run = cli();
    run.arg("run").arg(&database).arg("--input").arg("-");
    failure(
        invoke(run, Some("{\"schema_version\":2,\"operations\":[]}")),
        2,
    );

    let mut old_recall = cli();
    old_recall
        .arg("recall")
        .arg(&database)
        .arg("--from-sequence")
        .arg("0");
    failure(invoke(old_recall, None), 2);

    let mut json_many = cli();
    json_many.arg("add").arg(&database).arg("--many");
    failure(
        invoke(
            json_many,
            Some("{\"op\":\"insert_episode\",\"episode\":{}}\n"),
        ),
        1,
    );
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
fn preexisting_sqlite_v2_store_needs_no_migration() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&database).unwrap();
    let source = store.memory_mut().insert_episode(draft(1)).unwrap();
    let target = store.memory_mut().insert_episode(draft(2)).unwrap();
    store
        .memory_mut()
        .set_relevance(source, target, InfluenceWeight::from_ppm(1_000).unwrap())
        .unwrap();
    store.save().unwrap();
    drop(store);

    assert!(
        success_text(recall(&database, 0, None)).starts_with("sequence 1\nactivation_ppm 400\n")
    );
    assert_eq!(success_text(add_minimal(&database, 3, false)), "2\n");
    assert_silent_success(feedback(&database, 0, false, "1"));

    let reopened = SqliteStore::open(&database).unwrap();
    assert_eq!(reopened.memory().episodes().len(), 3);
    assert_eq!(reopened.memory().relevance(source, target), None);
}
