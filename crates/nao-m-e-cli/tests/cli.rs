use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use nao_m_e::{AtomId, InfluenceWeight, MAX_FEEDBACK_TARGETS};
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

fn recall_source(database: &Path, source_sequence: u64, limit: Option<usize>) -> Output {
    let mut command = cli();
    command
        .arg("recall")
        .arg(database)
        .arg("--from-sequence")
        .arg(source_sequence.to_string());
    if let Some(limit) = limit {
        command.arg("--limit").arg(limit.to_string());
    }
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
        &["recall", "--help"][..],
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
        .arg("1");
    let output = invoke(conflicting, None);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
}

#[test]
fn init_creates_a_valid_store_and_never_clobbers_it() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");

    let response = init(&database);
    assert_eq!(response["schema_version"], 2);
    assert_eq!(response["episode_count"], 0);
    let memory_id = response["memory_id"]
        .as_str()
        .expect("memory ID is text")
        .to_owned();
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
    assert_eq!(
        response,
        json!({
            "schema_version": 2,
            "memory_id": memory_id,
            "episode_count": 0
        })
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
        "schema_version": 2,
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
        "schema_version": 2,
        "operations": [insert(Some("third"), 3)]
    });
    let response = assert_success(run_stdin(&database, &stdin_batch));
    assert_eq!(response["episode_count"], 3);
    assert_eq!(response["inserted"][0]["sequence"], 2);

    let store = SqliteStore::open(&database).unwrap();
    assert_eq!(store.memory().episodes().len(), 3);
}

#[test]
fn source_recall_default_and_limit_return_exact_ranked_json() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    let memory_id = init(&database)["memory_id"].clone();

    let source = json!({
        "op": "insert_episode",
        "label": "source",
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
    let strongest = json!({
        "op": "insert_episode",
        "label": "strongest",
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
        "schema_version": 2,
        "operations": [
            source,
            strongest,
            insert(Some("middle"), 3),
            insert(Some("weakest"), 4),
            {
                "op": "set_relevance",
                "from": {"label": "source"},
                "to": {"label": "strongest"},
                "weight_ppm": 600000
            },
            {
                "op": "set_relevance",
                "from": {"label": "source"},
                "to": {"label": "middle"},
                "weight_ppm": 250000
            },
            {
                "op": "set_relevance",
                "from": {"label": "source"},
                "to": {"label": "weakest"},
                "weight_ppm": 150000
            }
        ]
    });
    let run = assert_success(run_stdin(&database, &scenario));
    assert_eq!(
        run,
        json!({
            "schema_version": 2,
            "memory_id": memory_id.clone(),
            "operations_applied": 7,
            "episode_count": 4,
            "inserted": [
                {"label": "source", "sequence": 0},
                {"label": "strongest", "sequence": 1},
                {"label": "middle", "sequence": 2},
                {"label": "weakest", "sequence": 3}
            ]
        })
    );

    let recall = assert_success(recall_source(&database, 0, None));
    assert_eq!(
        recall,
        json!({
            "schema_version": 2,
            "memory_id": memory_id,
            "hits": [
                {
                    "sequence": 1,
                    "activation_ppm": 240000,
                    "episode": {
                        "occurred_at_ms": 2000,
                        "recorded_at_ms": 2001,
                        "source_id": 8,
                        "context": [],
                        "observation": {"predicate_id": 21, "term_ids": [1001]},
                        "action": {"predicate_id": 30, "term_ids": [1001]},
                        "outcome": {"predicate_id": 40, "term_ids": [1001]}
                    }
                },
                {
                    "sequence": 2,
                    "activation_ppm": 100000,
                    "episode": {
                        "occurred_at_ms": 3,
                        "recorded_at_ms": 4,
                        "source_id": 3,
                        "context": [],
                        "observation": {"predicate_id": 13, "term_ids": [103]},
                        "action": null,
                        "outcome": null
                    }
                },
                {
                    "sequence": 3,
                    "activation_ppm": 60000,
                    "episode": {
                        "occurred_at_ms": 4,
                        "recorded_at_ms": 5,
                        "source_id": 4,
                        "context": [],
                        "observation": {"predicate_id": 14, "term_ids": [104]},
                        "action": null,
                        "outcome": null
                    }
                }
            ]
        })
    );

    let limited = assert_success(recall_source(&database, 0, Some(2)));
    let mut expected_limited = recall;
    expected_limited["hits"]
        .as_array_mut()
        .expect("expected hits are an array")
        .truncate(2);
    assert_eq!(limited, expected_limited);
}

#[test]
fn source_recall_rejects_an_unknown_source_without_success_output() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    assert_success(run_stdin(
        &database,
        &json!({"schema_version": 2, "operations": [insert(None, 1)]}),
    ));

    let stderr = assert_runtime_failure(recall_source(&database, 99, None));
    assert!(
        stderr.contains("unknown atom"),
        "unexpected diagnostic: {stderr}"
    );
}

#[test]
fn source_recall_does_not_mutate_state_or_advance_the_revision() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    assert_success(run_stdin(
        &database,
        &json!({
            "schema_version": 2,
            "operations": [
                insert(Some("source"), 1),
                insert(Some("direct"), 2),
                insert(Some("ambient"), 3)
            ]
        }),
    ));

    let mut writer = SqliteStore::open(&database).expect("writer opens");
    let source = AtomId::from_parts(writer.memory_id(), 0);
    let direct = AtomId::from_parts(writer.memory_id(), 1);
    let ambient = AtomId::from_parts(writer.memory_id(), 2);
    writer
        .memory_mut()
        .set_relevance(source, direct, InfluenceWeight::from_ppm(600_000).unwrap())
        .unwrap();
    writer.save().expect("initial relevance is persisted");

    let recall = assert_success(recall_source(&database, 0, None));
    assert_eq!(recall["hits"].as_array().unwrap().len(), 1);
    assert_eq!(recall["hits"][0]["sequence"], 1);
    assert_eq!(recall["hits"][0]["activation_ppm"], 240_000);

    writer
        .memory_mut()
        .set_relevance(source, ambient, InfluenceWeight::from_ppm(250_000).unwrap())
        .unwrap();
    writer
        .save()
        .expect("read-only recall did not advance the snapshot revision");

    let after = assert_success(recall_source(&database, 0, None));
    assert_eq!(after["hits"].as_array().unwrap().len(), 2);
    assert_eq!(after["hits"][0]["sequence"], 1);
    assert_eq!(after["hits"][0]["activation_ppm"], 240_000);
    assert_eq!(after["hits"][1]["sequence"], 2);
    assert_eq!(after["hits"][1]["activation_ppm"], 100_000);
}

#[test]
fn feedback_uses_the_explicit_target_list_and_persists_exact_updates() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    let memory_id = init(&database)["memory_id"].clone();

    let setup = json!({
        "schema_version": 2,
        "operations": [
            insert(Some("source"), 1),
            insert(Some("first"), 2),
            insert(Some("second"), 3),
            insert(Some("existing"), 4),
            {
                "op": "set_relevance",
                "from": {"label": "source"},
                "to": {"label": "existing"},
                "weight_ppm": 1000000
            }
        ]
    });
    assert_success(run_stdin(&database, &setup));
    let prior_hits = assert_success(recall_source(&database, 0, None));
    assert_eq!(prior_hits["hits"].as_array().unwrap().len(), 1);
    assert_eq!(prior_hits["hits"][0]["sequence"], 3);

    let positive = json!({
        "schema_version": 2,
        "operations": [{
            "op": "apply_feedback",
            "source": {"sequence": 0},
            "targets": [
                {"sequence": 0},
                {"sequence": 1},
                {"sequence": 2},
                {"sequence": 2}
            ],
            "feedback": 1
        }]
    });
    assert_eq!(
        assert_success(run_stdin(&database, &positive)),
        json!({
            "schema_version": 2,
            "memory_id": memory_id.clone(),
            "operations_applied": 1,
            "episode_count": 4,
            "inserted": []
        })
    );

    let store = SqliteStore::open(&database).expect("positive feedback snapshot reopens");
    let source = AtomId::from_parts(store.memory_id(), 0);
    let first = AtomId::from_parts(store.memory_id(), 1);
    let second = AtomId::from_parts(store.memory_id(), 2);
    let existing = AtomId::from_parts(store.memory_id(), 3);
    assert_eq!(store.memory().relevance(source, source), None);
    assert_eq!(
        store
            .memory()
            .relevance(source, first)
            .map(|weight| weight.as_ppm()),
        Some(1_000)
    );
    assert_eq!(
        store
            .memory()
            .relevance(source, second)
            .map(|weight| weight.as_ppm()),
        Some(1_000)
    );
    assert_eq!(
        store
            .memory()
            .relevance(source, existing)
            .map(|weight| weight.as_ppm()),
        Some(998_000)
    );
    drop(store);

    let mut negative = positive;
    negative["operations"][0]["feedback"] = json!(0);
    assert_eq!(
        assert_success(run_stdin(&database, &negative)),
        json!({
            "schema_version": 2,
            "memory_id": memory_id,
            "operations_applied": 1,
            "episode_count": 4,
            "inserted": []
        })
    );

    let store = SqliteStore::open(&database).expect("negative feedback snapshot reopens");
    assert_eq!(store.memory().relevance(source, first), None);
    assert_eq!(store.memory().relevance(source, second), None);
    assert_eq!(
        store
            .memory()
            .relevance(source, existing)
            .map(|weight| weight.as_ppm()),
        Some(998_000)
    );
    assert_eq!(store.memory().episodes().len(), 4);
}

#[test]
fn feedback_changes_the_next_source_conditioned_query() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    let memory_id = init(&database)["memory_id"].clone();
    assert_success(run_stdin(
        &database,
        &json!({
            "schema_version": 2,
            "operations": [insert(Some("source"), 1), insert(Some("target"), 2)]
        }),
    ));

    assert!(
        assert_success(recall_source(&database, 0, None))["hits"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    let feedback = |value| {
        json!({
            "schema_version": 2,
            "operations": [{
                "op": "apply_feedback",
                "source": {"sequence": 0},
                "targets": [{"sequence": 1}],
                "feedback": value
            }]
        })
    };
    assert_success(run_stdin(&database, &feedback(1)));
    assert_eq!(
        assert_success(recall_source(&database, 0, None)),
        json!({
            "schema_version": 2,
            "memory_id": memory_id,
            "hits": [{
                "sequence": 1,
                "activation_ppm": 400,
                "episode": {
                    "occurred_at_ms": 2,
                    "recorded_at_ms": 3,
                    "source_id": 2,
                    "context": [],
                    "observation": {"predicate_id": 12, "term_ids": [102]},
                    "action": null,
                    "outcome": null
                }
            }]
        })
    );

    assert_success(run_stdin(&database, &feedback(0)));
    assert!(
        assert_success(recall_source(&database, 0, None))["hits"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn feedback_target_order_produces_the_same_relevance_graph() {
    let directory = TempDir::new().expect("temporary directory");
    let base = directory.path().join("base.sqlite3");
    let forward = directory.path().join("forward.sqlite3");
    let reversed = directory.path().join("reversed.sqlite3");
    init(&base);
    assert_success(run_stdin(
        &base,
        &json!({
            "schema_version": 2,
            "operations": [
                insert(Some("source"), 1),
                insert(Some("first"), 2),
                insert(Some("second"), 3),
                insert(Some("existing"), 4),
                {
                    "op": "set_relevance",
                    "from": {"label": "source"},
                    "to": {"label": "existing"},
                    "weight_ppm": 1000000
                }
            ]
        }),
    ));
    fs::copy(&base, &forward).expect("forward store copy");
    fs::copy(&base, &reversed).expect("reversed store copy");

    let feedback = |targets: [u64; 2]| {
        json!({
            "schema_version": 2,
            "operations": [{
                "op": "apply_feedback",
                "source": {"sequence": 0},
                "targets": [
                    {"sequence": targets[0]},
                    {"sequence": targets[1]}
                ],
                "feedback": 1
            }]
        })
    };
    assert_success(run_stdin(&forward, &feedback([1, 2])));
    assert_success(run_stdin(&reversed, &feedback([2, 1])));

    let forward = SqliteStore::open(&forward).expect("forward snapshot reopens");
    let reversed = SqliteStore::open(&reversed).expect("reversed snapshot reopens");
    assert_eq!(
        forward.memory().relevance_edges().collect::<Vec<_>>(),
        reversed.memory().relevance_edges().collect::<Vec<_>>()
    );
}

#[test]
fn feedback_target_limit_accepts_the_boundary_and_rejects_excess() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    assert_success(run_stdin(
        &database,
        &json!({
            "schema_version": 2,
            "operations": [insert(None, 1), insert(None, 2)]
        }),
    ));

    let target = json!({"sequence": 1});
    let at_limit = json!({
        "schema_version": 2,
        "operations": [{
            "op": "apply_feedback",
            "source": {"sequence": 0},
            "targets": vec![target.clone(); MAX_FEEDBACK_TARGETS],
            "feedback": 1
        }]
    });
    assert_success(run_stdin(&database, &at_limit));

    let over_limit = json!({
        "schema_version": 2,
        "operations": [{
            "op": "apply_feedback",
            "source": {"sequence": 0},
            "targets": vec![target; MAX_FEEDBACK_TARGETS + 1],
            "feedback": 0
        }]
    });
    let stderr = assert_runtime_failure(run_stdin(&database, &over_limit));
    assert!(stderr.contains("feedback target count 10001 exceeds 10000"));

    let store = SqliteStore::open(&database).expect("limit failure leaves snapshot valid");
    let source = AtomId::from_parts(store.memory_id(), 0);
    let target = AtomId::from_parts(store.memory_id(), 1);
    assert_eq!(
        store
            .memory()
            .relevance(source, target)
            .map(|weight| weight.as_ppm()),
        Some(1_000)
    );
}

#[test]
fn feedback_validation_is_strict_and_batch_failures_roll_back() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    assert_success(run_stdin(
        &database,
        &json!({
            "schema_version": 2,
            "operations": [insert(None, 1), insert(None, 2)]
        }),
    ));

    let invalid_feedback = json!({
        "schema_version": 2,
        "operations": [{
            "op": "apply_feedback",
            "source": {"sequence": 99},
            "targets": [{"sequence": 98}],
            "feedback": 2
        }]
    });
    let stderr = assert_runtime_failure(run_stdin(&database, &invalid_feedback));
    assert!(stderr.contains("operations[0] (apply_feedback) failed: feedback must be 0 or 1"));

    for (scenario, expected) in [
        (
            json!({
                "schema_version": 2,
                "operations": [{
                    "op": "apply_feedback",
                    "source": {"sequence": 99},
                    "targets": [{"sequence": 1}],
                    "feedback": 1
                }]
            }),
            "episode sequence 99 was not persisted before this batch",
        ),
        (
            json!({
                "schema_version": 2,
                "operations": [{
                    "op": "apply_feedback",
                    "source": {"sequence": 0},
                    "targets": [{"sequence": 99}],
                    "feedback": 1
                }]
            }),
            "episode sequence 99 was not persisted before this batch",
        ),
    ] {
        let stderr = assert_runtime_failure(run_stdin(&database, &scenario));
        assert!(stderr.contains(expected), "unexpected diagnostic: {stderr}");
    }

    for operation in [
        json!({
            "op": "apply_feedback",
            "source": {"sequence": 0},
            "targets": [{"sequence": 1}],
            "feedback": 1,
            "extra": true
        }),
        json!({
            "op": "apply_feedback",
            "source": {"sequence": 0, "label": "ambiguous"},
            "targets": [{"sequence": 1}],
            "feedback": 1
        }),
        json!({
            "op": "apply_feedback",
            "source": {"sequence": 0},
            "targets": [{"sequence": 1, "extra": true}],
            "feedback": 1
        }),
    ] {
        let scenario = json!({"schema_version": 2, "operations": [operation]});
        let stderr = assert_runtime_failure(run_stdin(&database, &scenario));
        assert!(stderr.contains("operations[0] is invalid"));
    }

    let rollback = json!({
        "schema_version": 2,
        "operations": [
            {
                "op": "apply_feedback",
                "source": {"sequence": 0},
                "targets": [{"sequence": 1}],
                "feedback": 1
            },
            {
                "op": "set_relevance",
                "from": {"sequence": 0},
                "to": {"sequence": 0},
                "weight_ppm": 123456
            }
        ]
    });
    let stderr = assert_runtime_failure(run_stdin(&database, &rollback));
    assert!(stderr.contains("operations[1] (set_relevance) failed"));

    let store = SqliteStore::open(&database).expect("failed feedback batch leaves store valid");
    let source = AtomId::from_parts(store.memory_id(), 0);
    let target = AtomId::from_parts(store.memory_id(), 1);
    assert_eq!(store.memory().relevance(source, target), None);
    assert_eq!(store.memory().episodes().len(), 2);
}

#[test]
fn persisted_sequences_support_edge_removal_and_source_recall() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let setup = json!({
        "schema_version": 2,
        "operations": [
            insert(Some("from"), 1),
            insert(Some("to"), 2),
            {
                "op": "set_relevance",
                "from": {"label": "from"},
                "to": {"label": "to"},
                "weight_ppm": 250000
            }
        ]
    });
    assert_success(run_stdin(&database, &setup));
    let before = assert_success(recall_source(&database, 0, None));
    assert_eq!(before["hits"].as_array().unwrap().len(), 1);
    assert_eq!(before["hits"][0]["sequence"], 1);
    assert_eq!(before["hits"][0]["activation_ppm"], 100_000);

    let cleanup = json!({
        "schema_version": 2,
        "operations": [{
            "op": "remove_relevance",
            "from": {"sequence": 0},
            "to": {"sequence": 1}
        }]
    });
    assert_success(run_stdin(&database, &cleanup));
    assert!(
        assert_success(recall_source(&database, 0, None))["hits"]
            .as_array()
            .unwrap()
            .is_empty()
    );

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
        "schema_version": 2,
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
    assert_runtime_failure(recall_source(&database, 0, None));

    let response = assert_success(run_stdin(
        &database,
        &json!({"schema_version": 2, "operations": [insert(None, 3)]}),
    ));
    assert_eq!(response["inserted"][0]["sequence"], 0);
}

#[test]
fn malformed_and_invalid_scenarios_fail_closed() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let invalid = [
        json!({"schema_version": 2, "operations": [insert(Some(""), 1)]}),
        json!({"schema_version": 2, "operations": [], "extra": true}),
        json!({
            "schema_version": 2,
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
            "schema_version": 2,
            "operations": [
                insert(Some("new"), 1),
                {
                    "op": "set_relevance",
                    "from": {"sequence": 0},
                    "to": {"label": "new"},
                    "weight_ppm": 1
                }
            ]
        }),
        json!({
            "schema_version": 2,
            "operations": [{
                "op": "set_relevance",
                "from": {"label": "later"},
                "to": {"label": "later"},
                "weight_ppm": 1
            }, insert(Some("later"), 1)]
        }),
        json!({
            "schema_version": 2,
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
            "schema_version": 2,
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

    for atom in [
        json!({"sequence": 0, "label": "ambiguous"}),
        json!({"label": "missing", "extra": true}),
    ] {
        let scenario = json!({
            "schema_version": 2,
            "operations": [{
                "op": "set_relevance",
                "from": atom,
                "to": {"sequence": 0},
                "weight_ppm": 1
            }]
        });
        let stderr = assert_runtime_failure(run_stdin(&database, &scenario));
        assert!(stderr.contains("operations[0] is invalid"));
    }

    let unknown_operation = json!({
        "schema_version": 2,
        "operations": [{"op": "unknown"}]
    });
    let stderr = assert_runtime_failure(run_stdin(&database, &unknown_operation));
    assert!(stderr.contains("operations[0] is invalid"));

    let mut command = cli();
    command.arg("run").arg(&database).arg("--input").arg("-");
    assert_runtime_failure(invoke(command, Some("{")));
}

#[test]
fn v1_scenarios_and_removed_activation_operations_are_rejected() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let v1 = json!({"schema_version": 1, "operations": [insert(None, 1)]});
    let stderr = assert_runtime_failure(run_stdin(&database, &v1));
    assert!(stderr.contains("unsupported scenario schema_version 1; expected 2"));

    for operation in [
        json!({
            "op": "stimulate",
            "atom": {"sequence": 0},
            "amount_ppm": 1
        }),
        json!({"op": "step", "count": 1}),
        json!({"op": "reset_activations"}),
    ] {
        let scenario = json!({"schema_version": 2, "operations": [operation]});
        let stderr = assert_runtime_failure(run_stdin(&database, &scenario));
        assert!(stderr.contains("operations[0] is invalid"));
    }

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
fn maximum_weight_produces_the_exact_source_recall_score() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let scenario = json!({
        "schema_version": 2,
        "operations": [
            insert(Some("from"), 1),
            insert(Some("to"), 2),
            {
                "op": "set_relevance",
                "from": {"label": "from"},
                "to": {"label": "to"},
                "weight_ppm": 1000000
            }
        ]
    });
    assert_success(run_stdin(&database, &scenario));

    let response = assert_success(recall_source(&database, 0, None));
    assert_eq!(response["hits"].as_array().unwrap().len(), 1);
    assert_eq!(response["hits"][0]["sequence"], 1);
    assert_eq!(response["hits"][0]["activation_ppm"], 400_000);

    let store = SqliteStore::open(&database).unwrap();
    let from = AtomId::from_parts(store.memory_id(), 0);
    let to = AtomId::from_parts(store.memory_id(), 1);
    assert_eq!(
        store.memory().relevance(from, to).unwrap().as_ppm(),
        1000000
    );
}

#[test]
fn integer_boundaries_round_trip_exactly() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);

    let scenario = json!({
        "schema_version": 2,
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

    let store = SqliteStore::open(&database).expect("boundary snapshot reopens");
    let atom = store.memory().episodes().next().expect("episode is stored");
    assert_eq!(atom.occurred_at().get(), i64::MIN);
    assert_eq!(atom.recorded_at().get(), i64::MAX);
    assert_eq!(atom.source().get(), u64::MAX);
    assert_eq!(atom.observation().predicate().get(), u64::MAX - 1);
    assert_eq!(atom.observation().arguments()[0].get(), u64::MAX);
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

    let database = database.to_str().expect("temporary path is UTF-8");
    for arguments in [
        vec!["recall", database],
        vec!["recall", database, "--limit", "1"],
        vec!["recall", database, "--sequence", "0"],
        vec!["recall", database, "--from-sequence", "0", "--limit"],
        vec!["recall", database, "--limit", "1", "--from-sequence", "0"],
    ] {
        let mut command = cli();
        command.args(arguments);
        let output = invoke(command, None);
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
    }
}
