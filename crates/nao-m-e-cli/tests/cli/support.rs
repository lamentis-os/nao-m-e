use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

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

pub(super) fn add_minimal(database: &Path, seed: i64, quiet: bool) -> Output {
    let mut command = cli();
    command
        .arg("add")
        .arg(database)
        .arg("--timestamp")
        .arg(seed.to_string())
        .arg("--predicate")
        .arg(format!("predicate-{seed}"))
        .arg("--term")
        .arg(format!("term-{seed}"));
    if quiet {
        command.arg("--quiet");
    }
    invoke(command, None)
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
