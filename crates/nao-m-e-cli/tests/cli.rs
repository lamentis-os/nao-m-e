use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use nao_m_e::{AtomId, EpisodeAtom, FeedbackTrace, MAX_FEEDBACK_TARGETS, Statement};
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
        .arg(format!("predicate-{source}"))
        .arg("--term")
        .arg(format!("term-{source}"));
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
    assert_eq!(atom.occurred_at().get(), i64::try_from(seed).unwrap());
    assert_eq!(atom.recorded_at().get(), i64::try_from(seed + 1).unwrap());
    assert_eq!(atom.source().get(), seed);
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

fn minimal_recall_block(sequence: u64, activation_ppm: u32) -> String {
    format!(
        "sequence {sequence}\nactivation_ppm {activation_ppm}\noccurred {sequence}\nrecorded {}\nsource {sequence}\npredicate predicate-{sequence}\nterm term-{sequence}",
        sequence + 1
    )
}

fn add_observation(database: &Path, occurred: i64, predicate: &str, terms: &[&str]) {
    let mut command = cli();
    command
        .arg("add")
        .arg(database)
        .arg("--occurred")
        .arg(occurred.to_string())
        .arg("--recorded")
        .arg((occurred + 1).to_string())
        .arg("--source")
        .arg(occurred.to_string())
        .arg("--predicate")
        .arg(predicate);
    for term in terms {
        command.arg("--term").arg(term);
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
        .args(["--occurred", "9", "--recorded", "10", "--source", "9"])
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
    assert_eq!(atoms[0].occurred_at().get(), -5);
    assert_eq!(atoms[0].recorded_at().get(), 6);
    assert_eq!(atoms[0].source().get(), 7);
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
        .arg("Boundary")
        .arg("--term")
        .arg("Zero")
        .arg("--term")
        .arg("Maximum");
    assert_eq!(success_text(invoke(command, None)), "0\n");

    let store = SqliteStore::open(&database).unwrap();
    let atom = store.memory().episodes().next().expect("episode is stored");
    assert_eq!(atom.occurred_at().get(), i64::MIN);
    assert_eq!(atom.recorded_at().get(), i64::MAX);
    assert_eq!(atom.source().get(), u64::MAX);
    assert_statement(&store, atom.observation(), "boundary", &["zero", "maximum"]);
}

#[test]
fn many_add_ignores_empty_lines_is_atomic_and_assigns_input_order() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let input = "# ignored comment\n\n--occurred 1 --recorded 2 --source 1 --predicate 'Predicate-1' --term 'Term-1' # trailing comment\r\n  \n--occurred -3 --recorded -2 --source 2 --predicate \"Predicate 2\" --term 'Term 2' --term 'Term, 3' --context 'Context: Value' --context-term 'Context Term' --action Action --action-term 'Action Term' --outcome Outcome --outcome-term 'Outcome Term'\n";
    let mut many = cli();
    many.arg("add").arg(&database).arg("--many");
    assert_eq!(success_text(invoke(many, Some(input))), "0\n1\n");

    let mut quiet = cli();
    quiet.arg("add").arg(&database).arg("--many").arg("--quiet");
    assert_silent_success(invoke(
        quiet,
        Some("--occurred 3 --recorded 4 --source 3 --predicate 'Predicate-3' --term 'Term-3'\n"),
    ));

    let store = SqliteStore::open(&database).unwrap();
    let atoms: Vec<_> = store.memory().episodes().collect();
    assert_eq!(atoms.len(), 3);
    assert_minimal_episode(&store, atoms[0], 1);
    assert_eq!(atoms[1].id().sequence(), 1);
    assert_eq!(atoms[1].occurred_at().get(), -3);
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
    assert_minimal_episode(&store, atoms[2], 3);
    drop(store);

    let before = fs::read(&database).expect("database is readable before rejected batch");
    let invalid = "--occurred 4 --recorded 5 --source 4 --predicate Four --term Four\n--occurred 5 --recorded 6 --source 5 --predicate Five --term\n";
    let mut rejected = cli();
    rejected.arg("add").arg(&database).arg("--many");
    failure(invoke(rejected, Some(invalid)), 1);
    assert_eq!(fs::read(&database).unwrap(), before);

    let mut invalid_quote = cli();
    invalid_quote.arg("add").arg(&database).arg("--many");
    let stderr = failure(
        invoke(
            invalid_quote,
            Some("--occurred 4 --recorded 5 --source 4 --predicate 'unterminated --term value\n"),
        ),
        1,
    );
    assert!(stderr.contains("invalid shell quoting"));

    let mut invalid_symbol = cli();
    invalid_symbol.arg("add").arg(&database).arg("--many");
    failure(
        invoke(
            invalid_symbol,
            Some("--occurred 4 --recorded 5 --source 4 --predicate Four --term '   '\n"),
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
        3
    );
}

#[test]
fn recall_emits_exact_ranked_blocks_and_honors_limit() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    let mut input = String::from(
        "--occurred 0 --recorded 1 --source 0 --predicate predicate-0 --term term-0\n\
         --occurred -7 --recorded 8 --source 9 --context context-b --context-term context-b-1 --context-term context-b-2 --context context-a --context-term context-a-1 --context context-a --context-term context-a-1 --predicate observation --term observation-1 --term observation-2 --action action --action-term action-1 --outcome outcome --outcome-term outcome-1 --outcome-term outcome-2\n",
    );
    for seed in 2..=11 {
        input.push_str(&format!(
            "--occurred {seed} --recorded {} --source {seed} --predicate predicate-{seed} --term term-{seed}\n",
            seed + 1
        ));
    }
    let mut many = cli();
    many.arg("add").arg(&database).arg("--many").arg("--quiet");
    assert_silent_success(invoke(many, Some(&input)));

    let mut store = SqliteStore::open(&database).expect("seeded store opens");
    let memory_id = store.memory_id();
    let source = AtomId::from_parts(memory_id, 0);
    let rich = AtomId::from_parts(memory_id, 1);
    let targets = (2..=11)
        .map(|sequence| AtomId::from_parts(memory_id, sequence))
        .collect::<Vec<_>>();
    for _ in 0..16 {
        store
            .memory_mut()
            .apply_feedback(source, &[rich], true)
            .unwrap();
    }
    for (&target, helpful_count) in targets.iter().zip([8, 8, 15, 14, 13, 12, 11, 10, 9, 7]) {
        for _ in 0..helpful_count {
            store
                .memory_mut()
                .apply_feedback(source, &[target], true)
                .unwrap();
        }
    }
    store.save().unwrap();
    drop(store);
    let before = fs::read(&database).expect("database is readable before recall");

    let rich_block = "sequence 1\nactivation_ppm 400000\noccurred -7\nrecorded 8\nsource 9\ncontext context-b\ncontext-term context-b-1\ncontext-term context-b-2\ncontext context-a\ncontext-term context-a-1\npredicate observation\nterm observation-1\nterm observation-2\naction action\naction-term action-1\noutcome outcome\noutcome-term outcome-1\noutcome-term outcome-2";
    let mut blocks = vec![rich_block.to_owned()];
    blocks.extend(
        [
            (4, 392_045),
            (5, 383_333),
            (6, 373_750),
            (7, 363_157),
            (8, 351_388),
            (9, 338_235),
            (10, 323_437),
            (2, 306_666),
            (3, 306_666),
            (11, 287_500),
        ]
        .into_iter()
        .map(|(sequence, activation_ppm)| minimal_recall_block(sequence, activation_ppm)),
    );
    assert_eq!(blocks.len(), 11);

    let expected_default = blocks[..10].join("\n\n") + "\n";
    assert_eq!(success_text(recall(&database, 0, None)), expected_default);

    let expected_eleven = blocks.join("\n\n") + "\n";
    assert_eq!(
        success_text(recall(&database, 0, Some(11))),
        expected_eleven
    );
    assert_eq!(
        fs::read(&database).expect("database is readable after recall"),
        before
    );
}

#[test]
fn cold_recall_rebuilds_cue_candidates_without_feedback() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    for (sequence, occurred, source, predicate, terms) in [
        (0, 1, 100, "category", &["seven", "eight"][..]),
        (1, 3, 101, "category", &["seven", "nine"][..]),
        (2, 5, 102, "other", &["thirty"][..]),
    ] {
        let mut command = cli();
        command
            .arg("add")
            .arg(&database)
            .arg("--occurred")
            .arg(occurred.to_string())
            .arg("--recorded")
            .arg((occurred + 1).to_string())
            .arg("--source")
            .arg(source.to_string())
            .arg("--predicate")
            .arg(predicate);
        for term in terms {
            command.arg("--term").arg(term);
        }
        assert_eq!(success_text(invoke(command, None)), format!("{sequence}\n"));
    }

    assert_eq!(
        success_text(recall(&database, 0, None)),
        "sequence 1\nactivation_ppm 177777\noccurred 3\nrecorded 4\nsource 101\npredicate category\nterm seven\nterm nine\n"
    );
    assert_eq!(
        SqliteStore::open(&database)
            .unwrap()
            .memory()
            .feedback_edges()
            .count(),
        0
    );
}

#[test]
fn bounded_feedback_learns_reverses_and_suppresses_structural_matches_across_processes() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    for (occurred, predicate, terms) in [
        (1, "category", &["seven", "eight"][..]),
        (3, "category", &["seven", "nine"][..]),
        (5, "category", &["nine", "ten"][..]),
        (7, "other", &["seven"][..]),
        (9, "learned only", &["thirty"][..]),
    ] {
        add_observation(&database, occurred, predicate, terms);
    }

    assert_eq!(
        recall_scores(&database, 0),
        vec![(1, 177_777), (2, 52_173), (3, 20_000)]
    );

    let positive_checkpoints = [
        (1, 71_875, 1),
        (2, 127_777, 1),
        (3, 172_500, 1),
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

    for expected_score in [Some(105_902), Some(50_000), Some(5_277), None] {
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
fn recall_with_no_hits_is_silent_and_does_not_advance_the_store() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    assert_eq!(success_text(add_minimal(&database, 1, false)), "0\n");
    assert_eq!(success_text(add_minimal(&database, 2, false)), "1\n");
    let before = fs::read(&database).expect("database is readable before recall");
    assert_silent_success(recall(&database, 0, None));
    assert_silent_success(recall(&database, 0, Some(0)));
    assert_eq!(
        fs::read(&database).expect("database is readable after recall"),
        before
    );

    let mut writer = SqliteStore::open(&database).expect("writer opens before recall");
    let source = AtomId::from_parts(writer.memory_id(), 0);
    let target = AtomId::from_parts(writer.memory_id(), 1);
    assert_silent_success(recall(&database, 0, None));
    writer
        .memory_mut()
        .apply_feedback(source, &[target], true)
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
