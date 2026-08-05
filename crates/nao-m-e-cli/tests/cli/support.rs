use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use rusqlite::{Connection, OptionalExtension, params};

use nao_m_e::SymbolId;
use nao_m_e_sqlite::SqliteStore;

pub(super) fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_nao-m-e"))
}

pub(super) fn invoke(mut command: Command, stdin: Option<&str>) -> Output {
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

pub(super) fn success_text(output: Output) -> String {
    assert!(
        output.status.success(),
        "expected success, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    String::from_utf8(output.stdout).expect("stdout is UTF-8")
}

pub(super) fn assert_silent_success(output: Output) {
    assert_eq!(success_text(output), "");
}

pub(super) fn failure(output: Output, expected_code: i32) -> String {
    assert_eq!(output.status.code(), Some(expected_code));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).expect("stderr is UTF-8");
    assert!(!stderr.is_empty());
    stderr
}

pub(super) fn init(database: &Path) {
    let mut command = cli();
    command.arg("init").arg(database);
    assert_silent_success(invoke(command, None));
}

pub(super) fn check(database: &Path) -> Output {
    let mut command = cli();
    command.arg("check").arg(database);
    invoke(command, None)
}

pub(super) fn add_minimal(database: &Path, seed: i64, quiet: bool) -> Output {
    if database.exists() {
        seed_cues(
            database,
            [(format!("attribute-{seed}"), format!("value-{seed}"))],
        );
    }
    let mut command = cli();
    command
        .arg("add")
        .arg(database)
        .arg("--timestamp")
        .arg(seed.to_string())
        .arg("--attribute")
        .arg(format!("attribute-{seed}"))
        .arg("--value")
        .arg(format!("value-{seed}"));
    if quiet {
        command.arg("--quiet");
    }
    invoke(command, None)
}

pub(super) fn seed_cues<K, V>(database: &Path, pairs: impl IntoIterator<Item = (K, V)>)
where
    K: Into<String>,
    V: Into<String>,
{
    let pairs = pairs
        .into_iter()
        .map(|(key, value)| (key.into(), value.into()))
        .collect::<Vec<_>>();
    if pairs.is_empty() {
        return;
    }
    let baseline_revision = snapshot_revision(database);

    let values = pairs
        .iter()
        .flat_map(|(key, value)| [key.clone(), value.clone()])
        .collect::<Vec<_>>();
    let mut store = SqliteStore::open(database).expect("store opens before semantic cue seeding");
    let ids = store
        .intern_symbols(&values)
        .expect("fixture symbols are valid");
    store.save().expect("fixture symbols are saved");
    drop(store);

    let mut connection = Connection::open(database).expect("fixture database opens directly");
    connection
        .pragma_update(None, "foreign_keys", true)
        .expect("fixture foreign keys are enabled");
    let transaction = connection
        .transaction()
        .expect("fixture cue transaction begins");
    let count: Vec<u8> = transaction
        .query_row(
            "SELECT semantic_cue_count FROM memory_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .expect("semantic cue count exists");
    let mut next_cue = u64::from_be_bytes(
        count
            .try_into()
            .expect("semantic cue count has canonical width"),
    );
    let mut vector = vec![0_u8; 384 * size_of::<i16>()];
    vector[..2].copy_from_slice(&1_i16.to_le_bytes());

    for pair in ids.chunks_exact(2) {
        let key = encode_symbol(pair[0]);
        let value = encode_symbol(pair[1]);
        let existing: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT cue_id FROM semantic_cues WHERE key_id = ?1 AND value_id = ?2",
                params![key.as_slice(), value.as_slice()],
                |row| row.get(0),
            )
            .optional()
            .expect("fixture cue lookup succeeds");
        if existing.is_some() {
            continue;
        }
        let cue_id = next_cue.to_be_bytes();
        transaction
            .execute(
                "INSERT INTO semantic_cues (cue_id, key_id, value_id, vector)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    cue_id.as_slice(),
                    key.as_slice(),
                    value.as_slice(),
                    vector.as_slice(),
                ],
            )
            .expect("fixture cue is inserted");
        next_cue = next_cue.checked_add(1).expect("fixture cue IDs fit u64");
    }
    // Fixture preparation is revision-neutral so the command under test remains
    // the only observable save transition.
    transaction
        .execute(
            "UPDATE memory_meta
             SET semantic_cue_count = ?1,
                 snapshot_revision = ?2
             WHERE singleton = 1",
            rusqlite::params![next_cue.to_be_bytes().as_slice(), baseline_revision],
        )
        .expect("fixture metadata is updated");
    transaction
        .commit()
        .expect("fixture semantic cues are committed");
}

pub(super) fn snapshot_revision(database: &Path) -> i64 {
    Connection::open(database)
        .expect("fixture database opens for revision lookup")
        .query_row(
            "SELECT snapshot_revision FROM memory_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .expect("snapshot revision exists")
}

fn encode_symbol(id: SymbolId) -> [u8; 8] {
    id.get().to_be_bytes()
}

pub(super) fn recall(database: &Path, source: u64, limit: Option<usize>) -> Output {
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

pub(super) fn feedback(database: &Path, source: u64, helpful: bool, targets: &str) -> Output {
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
