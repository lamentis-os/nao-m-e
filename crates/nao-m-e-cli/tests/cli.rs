use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use nao_m_e::AtomId;
use nao_m_e_sqlite::SqliteStore;
use serde_json::{Value, json};
use tempfile::TempDir;

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_nao-m-e"))
}

fn invoke(mut command: Command, stdin: Option<&str>) -> Output {
    if stdin.is_none() {
        return command.output().expect("CLI process starts");
    }

    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("CLI process starts");
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(stdin.expect("checked above").as_bytes())
        .expect("scenario is written to stdin");
    child.wait_with_output().expect("CLI process exits")
}

fn assert_success(output: Output) -> Value {
    assert!(
        output.status.success(),
        "expected success, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    serde_json::from_slice(&output.stdout).expect("success output is JSON")
}

fn assert_runtime_failure(output: Output) -> String {
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).expect("stderr is UTF-8");
    assert!(!stderr.is_empty());
    stderr
}

fn init(database: &Path) -> Value {
    let mut command = cli();
    command.arg("init").arg(database);
    assert_success(invoke(command, None))
}

fn run_stdin(database: &Path, scenario: &Value) -> Output {
    let mut command = cli();
    command.arg("run").arg(database).arg("--input").arg("-");
    invoke(command, Some(&scenario.to_string()))
}

fn recall_limit(database: &Path, limit: usize) -> Value {
    let mut command = cli();
    command
        .arg("recall")
        .arg(database)
        .arg("--limit")
        .arg(limit.to_string());
    assert_success(invoke(command, None))
}

fn recall_sequence(database: &Path, sequence: u64) -> Output {
    let mut command = cli();
    command
        .arg("recall")
        .arg(database)
        .arg("--sequence")
        .arg(sequence.to_string());
    invoke(command, None)
}

fn episode(seed: u64) -> Value {
    json!({
        "occurred_at_ms": i64::try_from(seed).expect("small seed"),
        "recorded_at_ms": i64::try_from(seed + 1).expect("small seed"),
        "source_id": seed,
        "observation": {
            "predicate_id": seed + 10,
            "term_ids": [seed + 100]
        }
    })
}

fn insert(label: Option<&str>, seed: u64) -> Value {
    let mut operation = json!({
        "op": "insert_episode",
        "episode": episode(seed)
    });
    if let Some(label) = label {
        operation["label"] = json!(label);
    }
    operation
}

#[test]
fn help_version_and_usage_have_stable_exit_categories() {
    for arguments in [
        &["--help"][..],
        &["init", "--help"][..],
        &["run", "--help"][..],
    ] {
        let mut command = cli();
        command.args(arguments);
        let output = invoke(command, None);
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("Usage:"));
        assert!(output.stderr.is_empty());
    }

    let mut version = cli();
    version.arg("--version");
    let output = invoke(version, None);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("nao-m-e 0.0.1"));

    let output = invoke(cli(), None);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("a command is required"));

    let mut conflicting = cli();
    conflicting
        .arg("recall")
        .arg("memory.sqlite3")
        .arg("--limit")
        .arg("1")
        .arg("--sequence")
        .arg("0");
    let output = invoke(conflicting, None);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
}

#[test]
fn init_creates_a_valid_store_and_never_clobbers_it() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");

    let response = init(&database);
    assert_eq!(response["schema_version"], 1);
    assert_eq!(response["episode_count"], 0);
    let memory_id = response["memory_id"].as_str().expect("memory ID is text");
    assert_eq!(memory_id.len(), 32);
    assert!(
        memory_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
    assert_eq!(
        SqliteStore::open(&database)
            .unwrap()
            .memory_id()
            .to_string(),
        memory_id
    );

    let original = fs::read(&database).expect("database is readable");
    let mut command = cli();
    command.arg("init").arg(&database);
    let error = assert_runtime_failure(invoke(command, None));
    assert!(error.contains("could not create"));
    assert_eq!(fs::read(&database).unwrap(), original);
}

#[test]
fn separate_file_and_stdin_batches_append_monotonic_sequences() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    let input = directory.path().join("batch.json");
    init(&database);

    let file_batch = json!({
        "schema_version": 1,
        "operations": [insert(Some("first"), 1), insert(None, 2)]
    });
    fs::write(&input, file_batch.to_string()).expect("batch file is written");
    let mut command = cli();
    command.arg("run").arg(&database).arg("--input").arg(&input);
    let response = assert_success(invoke(command, None));
    assert_eq!(response["episode_count"], 2);
    assert_eq!(response["inserted"][0]["label"], "first");
    assert_eq!(response["inserted"][0]["sequence"], 0);
    assert!(response["inserted"][1]["label"].is_null());
    assert_eq!(response["inserted"][1]["sequence"], 1);

    let stdin_batch = json!({
        "schema_version": 1,
        "operations": [insert(Some("third"), 3)]
    });
    let response = assert_success(run_stdin(&database, &stdin_batch));
    assert_eq!(response["episode_count"], 3);
    assert_eq!(response["inserted"][0]["sequence"], 2);

    let store = SqliteStore::open(&database).unwrap();
    assert_eq!(store.memory().episodes().len(), 3);
}

#[test]
fn batch_dynamics_round_trip_through_ranked_recall() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let failure = json!({
        "op": "insert_episode",
        "label": "failure",
        "episode": {
            "occurred_at_ms": 1000,
            "recorded_at_ms": 1001,
            "source_id": 7,
            "context": [
                {"predicate_id": 11, "term_ids": [2]},
                {"predicate_id": 10, "term_ids": [1]},
                {"predicate_id": 11, "term_ids": [2]}
            ],
            "observation": {"predicate_id": 20, "term_ids": [1001, 3001]}
        }
    });
    let recovery = json!({
        "op": "insert_episode",
        "label": "recovery",
        "episode": {
            "occurred_at_ms": 2000,
            "recorded_at_ms": 2001,
            "source_id": 8,
            "observation": {"predicate_id": 21, "term_ids": [1001]},
            "action": {"predicate_id": 30, "term_ids": [1001]},
            "outcome": {"predicate_id": 40, "term_ids": [1001]}
        }
    });
    let scenario = json!({
        "schema_version": 1,
        "operations": [
            failure,
            recovery,
            {
                "op": "set_relevance",
                "from": {"label": "failure"},
                "to": {"label": "recovery"},
                "weight_ppm": 600000
            },
            {
                "op": "stimulate",
                "atom": {"label": "failure"},
                "amount_ppm": 1000000
            },
            {"op": "step", "count": 1}
        ]
    });
    assert_success(run_stdin(&database, &scenario));

    let recall = recall_limit(&database, 10);
    assert_eq!(recall["hits"].as_array().unwrap().len(), 2);
    assert_eq!(recall["hits"][0]["sequence"], 0);
    assert_eq!(recall["hits"][0]["activation_ppm"], 500000);
    assert_eq!(recall["hits"][1]["sequence"], 1);
    assert_eq!(recall["hits"][1]["activation_ppm"], 240000);
    let context = recall["hits"][0]["episode"]["context"].as_array().unwrap();
    assert_eq!(context.len(), 2);
    assert_eq!(context[0]["predicate_id"], 10);
    assert_eq!(context[1]["predicate_id"], 11);
}

#[test]
fn persisted_sequences_support_edge_removal_reset_and_inactive_lookup() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let setup = json!({
        "schema_version": 1,
        "operations": [
            insert(Some("from"), 1),
            insert(Some("to"), 2),
            {
                "op": "set_relevance",
                "from": {"label": "from"},
                "to": {"label": "to"},
                "weight_ppm": 250000
            },
            {
                "op": "stimulate",
                "atom": {"label": "from"},
                "amount_ppm": 750000
            }
        ]
    });
    assert_success(run_stdin(&database, &setup));

    let cleanup = json!({
        "schema_version": 1,
        "operations": [
            {
                "op": "remove_relevance",
                "from": {"sequence": 0},
                "to": {"sequence": 1}
            },
            {"op": "reset_activations"}
        ]
    });
    assert_success(run_stdin(&database, &cleanup));
    assert!(
        recall_limit(&database, 10)["hits"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    let episode = assert_success(recall_sequence(&database, 0));
    assert_eq!(episode["activation_ppm"], 0);
    assert_eq!(episode["sequence"], 0);

    let store = SqliteStore::open(&database).unwrap();
    let from = AtomId::from_parts(store.memory_id(), 0);
    let to = AtomId::from_parts(store.memory_id(), 1);
    assert_eq!(store.memory().relevance(from, to), None);
}

#[test]
fn an_operation_failure_discards_the_whole_in_memory_batch() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let invalid = json!({
        "schema_version": 1,
        "operations": [insert(Some("duplicate"), 1), insert(Some("duplicate"), 2)]
    });
    let stderr = assert_runtime_failure(run_stdin(&database, &invalid));
    assert!(stderr.contains("operations[1] (insert_episode)"));
    assert_eq!(
        SqliteStore::open(&database)
            .unwrap()
            .memory()
            .episodes()
            .len(),
        0
    );
    assert_runtime_failure(recall_sequence(&database, 0));

    let response = assert_success(run_stdin(
        &database,
        &json!({"schema_version": 1, "operations": [insert(None, 3)]}),
    ));
    assert_eq!(response["inserted"][0]["sequence"], 0);
}

#[test]
fn malformed_and_invalid_scenarios_fail_closed() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let invalid = [
        json!({"schema_version": 2, "operations": [insert(None, 1)]}),
        json!({"schema_version": 1, "operations": [insert(Some(""), 1)]}),
        json!({"schema_version": 1, "operations": [], "extra": true}),
        json!({
            "schema_version": 1,
            "operations": [{
                "op": "insert_episode",
                "episode": {
                    "occurred_at_ms": 1,
                    "recorded_at_ms": 2,
                    "source_id": 3,
                    "observation": {"predicate_id": 4, "term_ids": []}
                }
            }]
        }),
        json!({
            "schema_version": 1,
            "operations": [{
                "op": "stimulate",
                "atom": {"sequence": 0},
                "amount_ppm": 1
            }]
        }),
        json!({
            "schema_version": 1,
            "operations": [
                insert(Some("new"), 1),
                {
                    "op": "stimulate",
                    "atom": {"sequence": 0},
                    "amount_ppm": 1
                }
            ]
        }),
        json!({
            "schema_version": 1,
            "operations": [{
                "op": "set_relevance",
                "from": {"label": "later"},
                "to": {"label": "later"},
                "weight_ppm": 1
            }, insert(Some("later"), 1)]
        }),
        json!({
            "schema_version": 1,
            "operations": [{"op": "step", "count": 0}]
        }),
        json!({
            "schema_version": 1,
            "operations": [{
                "op": "stimulate",
                "atom": {"sequence": 0, "extra": true},
                "amount_ppm": 1
            }]
        }),
        json!({
            "schema_version": 1,
            "operations": [
                insert(Some("new"), 1),
                {
                    "op": "stimulate",
                    "atom": {"label": "new"},
                    "amount_ppm": 1000001
                }
            ]
        }),
        json!({
            "schema_version": 1,
            "operations": [
                insert(Some("a"), 1),
                insert(Some("b"), 2),
                {
                    "op": "set_relevance",
                    "from": {"label": "a"},
                    "to": {"label": "b"},
                    "weight_ppm": 1000001
                }
            ]
        }),
        json!({
            "schema_version": 1,
            "operations": [
                insert(Some("a"), 1),
                insert(Some("b"), 2),
                {
                    "op": "set_relevance",
                    "from": {"label": "a"},
                    "to": {"label": "b"},
                    "weight_ppm": 0
                }
            ]
        }),
    ];

    for scenario in invalid {
        assert_runtime_failure(run_stdin(&database, &scenario));
        assert_eq!(
            SqliteStore::open(&database)
                .unwrap()
                .memory()
                .episodes()
                .len(),
            0
        );
    }

    let unknown_operation = json!({
        "schema_version": 1,
        "operations": [{"op": "unknown"}]
    });
    let stderr = assert_runtime_failure(run_stdin(&database, &unknown_operation));
    assert!(stderr.contains("operations[0] is invalid"));

    let mut command = cli();
    command.arg("run").arg(&database).arg("--input").arg("-");
    assert_runtime_failure(invoke(command, Some("{")));
}

#[test]
fn valid_fixed_point_boundaries_and_default_recall_are_supported() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let scenario = json!({
        "schema_version": 1,
        "operations": [
            insert(Some("from"), 1),
            insert(Some("to"), 2),
            {
                "op": "set_relevance",
                "from": {"label": "from"},
                "to": {"label": "to"},
                "weight_ppm": 1000000
            },
            {
                "op": "stimulate",
                "atom": {"label": "from"},
                "amount_ppm": 0
            }
        ]
    });
    assert_success(run_stdin(&database, &scenario));

    let mut recall = cli();
    recall.arg("recall").arg(&database);
    let response = assert_success(invoke(recall, None));
    assert!(response["hits"].as_array().unwrap().is_empty());

    let store = SqliteStore::open(&database).unwrap();
    let from = AtomId::from_parts(store.memory_id(), 0);
    let to = AtomId::from_parts(store.memory_id(), 1);
    assert_eq!(
        store.memory().relevance(from, to).unwrap().as_ppm(),
        1000000
    );
    assert_eq!(store.memory().activation(from).unwrap().as_ppm(), 0);
}

#[test]
fn integer_boundaries_round_trip_exactly() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let scenario = json!({
        "schema_version": 1,
        "operations": [{
            "op": "insert_episode",
            "episode": {
                "occurred_at_ms": i64::MIN,
                "recorded_at_ms": i64::MAX,
                "source_id": u64::MAX,
                "observation": {
                    "predicate_id": u64::MAX - 1,
                    "term_ids": [u64::MAX]
                }
            }
        }]
    });
    assert_success(run_stdin(&database, &scenario));

    let response = assert_success(recall_sequence(&database, 0));
    let episode = &response["episode"];
    assert_eq!(episode["occurred_at_ms"].as_i64(), Some(i64::MIN));
    assert_eq!(episode["recorded_at_ms"].as_i64(), Some(i64::MAX));
    assert_eq!(episode["source_id"].as_u64(), Some(u64::MAX));
    assert_eq!(
        episode["observation"]["predicate_id"].as_u64(),
        Some(u64::MAX - 1)
    );
    assert_eq!(
        episode["observation"]["term_ids"][0].as_u64(),
        Some(u64::MAX)
    );
}

#[test]
fn syntax_errors_use_exit_two_without_success_output() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");

    let mut command = cli();
    command.arg("run").arg(&database).arg("--wrong").arg("-");
    let output = invoke(command, None);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires <DATABASE>"));
}
