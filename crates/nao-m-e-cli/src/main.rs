#![forbid(unsafe_code)]

mod add;
mod recall;

use std::env;
use std::ffi::{OsStr, OsString};
use std::io::{self, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nao_m_e::AtomId;
use nao_m_e_sqlite::SqliteStore;

const ROOT_HELP: &str = "NAO-M-E symbolic memory command-line interface

Usage:
  nao-m-e init <DATABASE>
  nao-m-e add <DATABASE> [--quiet] --occurred <MS> --recorded <MS> --source <ID> --predicate <TEXT> --term <TEXT>... [EPISODE OPTIONS]
  nao-m-e add <DATABASE> --many [--quiet]
  nao-m-e recall <DATABASE> --from <SEQUENCE> [--limit <N>]
  nao-m-e feedback <DATABASE> --from <SEQUENCE> --helpful <SEQUENCE,...>
  nao-m-e feedback <DATABASE> --from <SEQUENCE> --unhelpful <SEQUENCE,...>

Commands:
  init      Create a new SQLite memory store without replacing an existing file
  add       Append one episode, or atomic line-delimited episodes with --many
  recall    Rank cue-derived and learned source-conditioned episodes read-only
  feedback  Learn from one explicit helpful or unhelpful target set

Options:
  -h, --help     Show help
  --version      Show the CLI version
";

const INIT_HELP: &str = "Create a new SQLite memory store.

Usage:
  nao-m-e init <DATABASE>
";

const FEEDBACK_HELP: &str = "Learn from one explicit binary assessment and save atomically.

Usage:
  nao-m-e feedback <DATABASE> --from <SEQUENCE> --helpful <SEQUENCE,...>
  nao-m-e feedback <DATABASE> --from <SEQUENCE> --unhelpful <SEQUENCE,...>

Every effective target receives one complete bounded-history sample with the
same assessment. Successful feedback produces no standard output.
";

fn main() -> ExitCode {
    let parsed = match parse_args(env::args_os().skip(1).collect()) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("nao-m-e: {error}\n\n{ROOT_HELP}");
            return ExitCode::from(2);
        }
    };

    let output = match parsed {
        ParsedArgs::Print(output) => Ok(output.into_bytes()),
        ParsedArgs::Execute(command) => execute(command),
    };

    match output.and_then(write_stdout) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("nao-m-e: {error}");
            ExitCode::from(1)
        }
    }
}

type CliResult<T> = Result<T, String>;

enum ParsedArgs {
    Execute(Command),
    Print(String),
}

enum Command {
    Init {
        database: PathBuf,
    },
    Add {
        database: PathBuf,
        draft: Box<add::TextEpisodeDraft>,
        quiet: bool,
    },
    AddMany {
        database: PathBuf,
        quiet: bool,
    },
    Recall {
        database: PathBuf,
        source_sequence: u64,
        limit: usize,
    },
    Feedback {
        database: PathBuf,
        source_sequence: u64,
        target_sequences: Vec<u64>,
        helpful: bool,
    },
}

fn parse_args(args: Vec<OsString>) -> Result<ParsedArgs, String> {
    let Some(command) = args.first().and_then(|value| value.to_str()) else {
        return Err("a command is required".to_owned());
    };

    match command {
        "-h" | "--help" if args.len() == 1 => Ok(ParsedArgs::Print(ROOT_HELP.to_owned())),
        "--version" if args.len() == 1 => Ok(ParsedArgs::Print(format!(
            "nao-m-e {}\n",
            env!("CARGO_PKG_VERSION")
        ))),
        "init" => parse_init_args(&args[1..]),
        "add" => add::parse_args(&args[1..]),
        "recall" => recall::parse_args(&args[1..]),
        "feedback" => parse_feedback_args(&args[1..]),
        _ => Err(format!("unknown command or option `{command}`")),
    }
}

fn parse_init_args(args: &[OsString]) -> Result<ParsedArgs, String> {
    if is_help_request(args) {
        return Ok(ParsedArgs::Print(INIT_HELP.to_owned()));
    }
    if args.len() != 1 {
        return Err("`init` requires exactly one database path".to_owned());
    }

    Ok(ParsedArgs::Execute(Command::Init {
        database: PathBuf::from(&args[0]),
    }))
}

fn parse_feedback_args(args: &[OsString]) -> Result<ParsedArgs, String> {
    if is_help_request(args) {
        return Ok(ParsedArgs::Print(FEEDBACK_HELP.to_owned()));
    }
    let [database, source_option, source, assessment, targets] = args else {
        return Err(
            "`feedback` requires <DATABASE> --from <SEQUENCE> (--helpful|--unhelpful) <SEQUENCE,...>"
                .to_owned(),
        );
    };
    if source_option.as_os_str() != OsStr::new("--from") {
        return Err("`feedback` source must use `--from <SEQUENCE>`".to_owned());
    }
    let helpful = match assessment.to_str() {
        Some("--helpful") => true,
        Some("--unhelpful") => false,
        _ => return Err("feedback assessment must be `--helpful` or `--unhelpful`".to_owned()),
    };

    Ok(ParsedArgs::Execute(Command::Feedback {
        database: PathBuf::from(database),
        source_sequence: parse_number(source, "source sequence")?,
        target_sequences: parse_sequence_list(targets)?,
        helpful,
    }))
}

fn parse_sequence_list(value: &OsStr) -> CliResult<Vec<u64>> {
    let value = value
        .to_str()
        .ok_or_else(|| "feedback targets must be valid UTF-8".to_owned())?;
    if value.is_empty() {
        return Err("feedback targets must not be empty".to_owned());
    }
    value
        .split(',')
        .enumerate()
        .map(|(index, sequence)| {
            sequence
                .parse()
                .map_err(|_| format!("invalid feedback target sequence at index {index}"))
        })
        .collect()
}

fn is_help_request(args: &[OsString]) -> bool {
    matches!(args, [value] if value.as_os_str() == OsStr::new("-h") || value.as_os_str() == OsStr::new("--help"))
}

fn parse_number<T>(value: &OsStr, description: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .to_str()
        .ok_or_else(|| format!("{description} must be valid UTF-8"))?
        .parse()
        .map_err(|_| format!("invalid {description}"))
}

fn execute(command: Command) -> CliResult<Vec<u8>> {
    match command {
        Command::Init { database } => execute_init(&database),
        Command::Add {
            database,
            draft,
            quiet,
        } => add::execute(&database, vec![*draft], quiet),
        Command::AddMany { database, quiet } => {
            let drafts = add::read_many_drafts()?;
            add::execute(&database, drafts, quiet)
        }
        Command::Recall {
            database,
            source_sequence,
            limit,
        } => recall::execute(&database, source_sequence, limit),
        Command::Feedback {
            database,
            source_sequence,
            target_sequences,
            helpful,
        } => execute_feedback(&database, source_sequence, &target_sequences, helpful),
    }
}

fn execute_init(database: &Path) -> CliResult<Vec<u8>> {
    SqliteStore::create(database)
        .map_err(|error| format!("could not create `{}`: {error}", database.display()))?;
    Ok(Vec::new())
}

fn execute_feedback(
    database: &Path,
    source_sequence: u64,
    target_sequences: &[u64],
    helpful: bool,
) -> CliResult<Vec<u8>> {
    let mut store = open_store(database)?;
    let memory_id = store.memory_id();
    let source = AtomId::from_parts(memory_id, source_sequence);
    let targets = target_sequences
        .iter()
        .map(|&sequence| AtomId::from_parts(memory_id, sequence))
        .collect::<Vec<_>>();
    store
        .memory_mut()
        .apply_feedback(source, &targets, helpful)
        .map_err(|error| error.to_string())?;
    save(&mut store, database)?;
    Ok(Vec::new())
}

fn open_store(database: &Path) -> CliResult<SqliteStore> {
    SqliteStore::open(database)
        .map_err(|error| format!("could not open `{}`: {error}", database.display()))
}

fn save(store: &mut SqliteStore, database: &Path) -> CliResult<()> {
    store.save().map_err(|error| {
        format!(
            "could not save `{}`: {error}; the operation was not committed",
            database.display()
        )
    })
}

fn write_stdout(output: Vec<u8>) -> CliResult<()> {
    if output.is_empty() {
        return Ok(());
    }
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(&output)
        .and_then(|()| stdout.flush())
        .map_err(|error| format!("could not write standard output: {error}"))
}
