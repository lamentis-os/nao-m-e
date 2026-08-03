#![forbid(unsafe_code)]

use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt::Write as FmtWrite;
use std::io::{self, BufRead, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nao_m_e::{
    AtomId, EpisodeAtom, EpisodeDraft, MemoryV0, PredicateId, SourceId, Statement, TermId,
    TimestampMs,
};
use nao_m_e_sqlite::SqliteStore;

const DEFAULT_RECALL_LIMIT: usize = 10;

const ROOT_HELP: &str = "NAO-M-E symbolic memory command-line interface

Usage:
  nao-m-e init <DATABASE>
  nao-m-e add <DATABASE> [--quiet] --occurred <MS> --recorded <MS> --source <ID> --predicate <ID> --terms <ID,...> [EPISODE OPTIONS]
  nao-m-e add <DATABASE> --many [--quiet]
  nao-m-e recall <DATABASE> --from <SEQUENCE> [--limit <N>]
  nao-m-e feedback <DATABASE> --from <SEQUENCE> --helpful <SEQUENCE,...>
  nao-m-e feedback <DATABASE> --from <SEQUENCE> --unhelpful <SEQUENCE,...>

Commands:
  init      Create a new SQLite memory store without replacing an existing file
  add       Append one episode, or atomic line-delimited episodes with --many
  recall    Return direct source-conditioned episodes without changing state
  feedback  Learn from one explicit helpful or unhelpful target set

Options:
  -h, --help     Show help
  --version      Show the CLI version
";

const INIT_HELP: &str = "Create a new SQLite memory store.

Usage:
  nao-m-e init <DATABASE>
";

const ADD_HELP: &str = "Append symbolic episodes and save atomically.

Usage:
  nao-m-e add <DATABASE> [--quiet] --occurred <MS> --recorded <MS> --source <ID> --predicate <ID> --terms <ID,...> [EPISODE OPTIONS]
  nao-m-e add <DATABASE> --many [--quiet]

Episode options:
  --context <PREDICATE:TERM,...>   Add one context statement; repeatable
  --action <PREDICATE:TERM,...>    Set the optional action statement
  --outcome <PREDICATE:TERM,...>   Set the optional outcome statement

With --many, standard input contains one non-empty single-episode flag row per
episode. The command saves all rows once or saves none. Successful add writes
the assigned sequence per episode unless --quiet is present.
";

const RECALL_HELP: &str = "Return direct source-conditioned episodes without changing state.

Usage:
  nao-m-e recall <DATABASE> --from <SEQUENCE>
  nao-m-e recall <DATABASE> --from <SEQUENCE> --limit <N>

The default recall limit is 10. Hits are separated by one blank line. No hits
produce no standard output.
";

const FEEDBACK_HELP: &str = "Learn from one explicit binary assessment and save atomically.

Usage:
  nao-m-e feedback <DATABASE> --from <SEQUENCE> --helpful <SEQUENCE,...>
  nao-m-e feedback <DATABASE> --from <SEQUENCE> --unhelpful <SEQUENCE,...>

Every listed target receives the same assessment. Successful feedback produces
no standard output.
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
        draft: EpisodeDraft,
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
        "add" => parse_add_args(&args[1..]),
        "recall" => parse_recall_args(&args[1..]),
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

fn parse_add_args(args: &[OsString]) -> Result<ParsedArgs, String> {
    if is_help_request(args) {
        return Ok(ParsedArgs::Print(ADD_HELP.to_owned()));
    }
    let Some(database) = args.first() else {
        return Err("`add` requires a database path and episode flags".to_owned());
    };
    let options = &args[1..];
    let many_count = options
        .iter()
        .filter(|value| value.as_os_str() == OsStr::new("--many"))
        .count();

    if many_count != 0 {
        if many_count != 1 {
            return Err("`--many` may be specified only once".to_owned());
        }
        let quiet = parse_mode_options(options, "--many")?;
        return Ok(ParsedArgs::Execute(Command::AddMany {
            database: PathBuf::from(database),
            quiet,
        }));
    }

    let (episode_options, quiet) = strip_quiet(options)?;
    let draft = parse_episode_flags(&episode_options)?;
    Ok(ParsedArgs::Execute(Command::Add {
        database: PathBuf::from(database),
        draft,
        quiet,
    }))
}

fn parse_mode_options(options: &[OsString], required: &str) -> Result<bool, String> {
    let mut quiet = false;
    for option in options {
        match option.to_str() {
            Some(value) if value == required => {}
            Some("--quiet") if !quiet => quiet = true,
            Some("--quiet") => return Err("`--quiet` may be specified only once".to_owned()),
            Some(value) => {
                return Err(format!(
                    "`{required}` cannot be combined with episode option `{value}`"
                ));
            }
            None => return Err("add options must be valid UTF-8".to_owned()),
        }
    }
    Ok(quiet)
}

fn strip_quiet(options: &[OsString]) -> Result<(Vec<OsString>, bool), String> {
    let mut episode_options = Vec::with_capacity(options.len());
    let mut quiet = false;
    for option in options {
        if option.as_os_str() == OsStr::new("--quiet") {
            if quiet {
                return Err("`--quiet` may be specified only once".to_owned());
            }
            quiet = true;
        } else {
            episode_options.push(option.clone());
        }
    }
    Ok((episode_options, quiet))
}

fn parse_episode_flags(args: &[OsString]) -> CliResult<EpisodeDraft> {
    let mut occurred_at = None;
    let mut recorded_at = None;
    let mut source = None;
    let mut predicate = None;
    let mut terms = None;
    let mut context = Vec::new();
    let mut action = None;
    let mut outcome = None;
    let mut cursor = 0;

    while cursor < args.len() {
        let option = args[cursor]
            .to_str()
            .ok_or_else(|| "episode options must be valid UTF-8".to_owned())?;
        cursor += 1;
        let value = args
            .get(cursor)
            .ok_or_else(|| format!("`{option}` requires a value"))?;
        cursor += 1;

        match option {
            "--occurred" => set_once(
                &mut occurred_at,
                TimestampMs::new(parse_number(value, "occurred timestamp")?),
                option,
            )?,
            "--recorded" => set_once(
                &mut recorded_at,
                TimestampMs::new(parse_number(value, "recorded timestamp")?),
                option,
            )?,
            "--source" => set_once(
                &mut source,
                SourceId::new(parse_number(value, "source ID")?),
                option,
            )?,
            "--predicate" => set_once(
                &mut predicate,
                PredicateId::new(parse_number(value, "predicate ID")?),
                option,
            )?,
            "--terms" => set_once(&mut terms, parse_terms(value, "observation terms")?, option)?,
            "--context" => context.push(parse_statement(value, "context statement")?),
            "--action" => set_once(
                &mut action,
                parse_statement(value, "action statement")?,
                option,
            )?,
            "--outcome" => set_once(
                &mut outcome,
                parse_statement(value, "outcome statement")?,
                option,
            )?,
            _ => return Err(format!("unknown episode option `{option}`")),
        }
    }

    let occurred_at = occurred_at.ok_or_else(|| "episode requires `--occurred`".to_owned())?;
    let recorded_at = recorded_at.ok_or_else(|| "episode requires `--recorded`".to_owned())?;
    let source = source.ok_or_else(|| "episode requires `--source`".to_owned())?;
    let predicate = predicate.ok_or_else(|| "episode requires `--predicate`".to_owned())?;
    let terms = terms.ok_or_else(|| "episode requires `--terms`".to_owned())?;
    let observation = Statement::new(predicate, terms).map_err(|error| error.to_string())?;

    Ok(EpisodeDraft {
        occurred_at,
        recorded_at,
        context,
        observation,
        action,
        outcome,
        source,
    })
}

fn set_once<T>(slot: &mut Option<T>, value: T, option: &str) -> CliResult<()> {
    if slot.replace(value).is_some() {
        return Err(format!("`{option}` may be specified only once"));
    }
    Ok(())
}

fn parse_statement(value: &OsStr, description: &str) -> CliResult<Statement> {
    let value = value
        .to_str()
        .ok_or_else(|| format!("{description} must be valid UTF-8"))?;
    let (predicate, terms) = value
        .split_once(':')
        .ok_or_else(|| format!("invalid {description}; expected PREDICATE:TERM,..."))?;
    let predicate = predicate
        .parse()
        .map(PredicateId::new)
        .map_err(|_| format!("invalid {description} predicate"))?;
    let terms = parse_term_text(terms, description)?;
    Statement::new(predicate, terms).map_err(|error| format!("invalid {description}: {error}"))
}

fn parse_terms(value: &OsStr, description: &str) -> CliResult<Vec<TermId>> {
    let value = value
        .to_str()
        .ok_or_else(|| format!("{description} must be valid UTF-8"))?;
    parse_term_text(value, description)
}

fn parse_term_text(value: &str, description: &str) -> CliResult<Vec<TermId>> {
    if value.is_empty() {
        return Err(format!("{description} must not be empty"));
    }
    value
        .split(',')
        .enumerate()
        .map(|(index, term)| {
            term.parse()
                .map(TermId::new)
                .map_err(|_| format!("invalid {description} term at index {index}"))
        })
        .collect()
}

fn parse_recall_args(args: &[OsString]) -> Result<ParsedArgs, String> {
    if is_help_request(args) {
        return Ok(ParsedArgs::Print(RECALL_HELP.to_owned()));
    }
    let Some(database) = args.first() else {
        return Err("`recall` requires a database path".to_owned());
    };

    let (source_sequence, limit) = match &args[1..] {
        [option, source] if option.as_os_str() == OsStr::new("--from") => (
            parse_number(source, "source sequence")?,
            DEFAULT_RECALL_LIMIT,
        ),
        [source_option, source, limit_option, limit]
            if source_option.as_os_str() == OsStr::new("--from")
                && limit_option.as_os_str() == OsStr::new("--limit") =>
        {
            (
                parse_number(source, "source sequence")?,
                parse_number(limit, "recall limit")?,
            )
        }
        _ => {
            return Err("`recall` requires <DATABASE> --from <SEQUENCE> [--limit <N>]".to_owned());
        }
    };

    Ok(ParsedArgs::Execute(Command::Recall {
        database: PathBuf::from(database),
        source_sequence,
        limit,
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
        } => execute_add(&database, vec![draft], quiet),
        Command::AddMany { database, quiet } => {
            let drafts = read_many_drafts()?;
            execute_add(&database, drafts, quiet)
        }
        Command::Recall {
            database,
            source_sequence,
            limit,
        } => execute_recall(&database, source_sequence, limit),
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

fn read_many_drafts() -> CliResult<Vec<EpisodeDraft>> {
    let mut drafts = Vec::new();
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let line_number = index + 1;
        let line =
            line.map_err(|error| format!("could not read add --many line {line_number}: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let arguments = line
            .split_ascii_whitespace()
            .map(OsString::from)
            .collect::<Vec<_>>();
        let draft = parse_episode_flags(&arguments)
            .map_err(|error| format!("add --many line {line_number}: {error}"))?;
        drafts.push(draft);
    }
    if drafts.is_empty() {
        return Err("add --many requires at least one non-empty input line".to_owned());
    }
    Ok(drafts)
}

fn execute_add(database: &Path, drafts: Vec<EpisodeDraft>, quiet: bool) -> CliResult<Vec<u8>> {
    let mut store = SqliteStore::open(database)
        .map_err(|error| format!("could not open `{}`: {error}", database.display()))?;
    let mut sequences = Vec::with_capacity(drafts.len());
    for draft in drafts {
        let atom_id = store
            .memory_mut()
            .insert_episode(draft)
            .map_err(|error| error.to_string())?;
        sequences.push(atom_id.sequence());
    }
    let output = format_sequences(&sequences, quiet);
    save(&mut store, database)?;
    Ok(output)
}

fn format_sequences(sequences: &[u64], quiet: bool) -> Vec<u8> {
    if quiet {
        return Vec::new();
    }
    let mut output = String::with_capacity(sequences.len().saturating_mul(21));
    for sequence in sequences {
        writeln!(output, "{sequence}").expect("writing to a String cannot fail");
    }
    output.into_bytes()
}

fn execute_recall(database: &Path, source_sequence: u64, limit: usize) -> CliResult<Vec<u8>> {
    let store = SqliteStore::open(database)
        .map_err(|error| format!("could not open `{}`: {error}", database.display()))?;
    let source = AtomId::from_parts(store.memory_id(), source_sequence);
    let hits = store
        .memory()
        .recall_from(source, limit)
        .map_err(|error| error.to_string())?;
    Ok(format_recall(store.memory(), &hits))
}

fn format_recall(memory: &MemoryV0, hits: &[nao_m_e::RecallHit]) -> Vec<u8> {
    let mut output = String::new();
    for (index, hit) in hits.iter().enumerate() {
        if index != 0 {
            output.push('\n');
        }
        let episode = memory
            .episode(hit.atom_id)
            .expect("recall hits always reference stored episodes");
        write_recall_hit(&mut output, episode, hit.activation.as_ppm());
    }
    output.into_bytes()
}

fn write_recall_hit(output: &mut String, episode: &EpisodeAtom, activation_ppm: u32) {
    writeln!(output, "sequence {}", episode.id().sequence())
        .expect("writing to a String cannot fail");
    writeln!(output, "activation_ppm {activation_ppm}").expect("writing to a String cannot fail");
    writeln!(output, "occurred {}", episode.occurred_at().get())
        .expect("writing to a String cannot fail");
    writeln!(output, "recorded {}", episode.recorded_at().get())
        .expect("writing to a String cannot fail");
    writeln!(output, "source {}", episode.source().get()).expect("writing to a String cannot fail");
    for statement in episode.context() {
        write_statement_line(output, "context", statement);
    }
    writeln!(
        output,
        "predicate {}",
        episode.observation().predicate().get()
    )
    .expect("writing to a String cannot fail");
    write_terms_line(output, "terms", episode.observation().arguments());
    if let Some(action) = episode.action() {
        write_statement_line(output, "action", action);
    }
    if let Some(outcome) = episode.outcome() {
        write_statement_line(output, "outcome", outcome);
    }
}

fn write_statement_line(output: &mut String, name: &str, statement: &Statement) {
    write!(output, "{name} {}:", statement.predicate().get())
        .expect("writing to a String cannot fail");
    write_term_values(output, statement.arguments());
    output.push('\n');
}

fn write_terms_line(output: &mut String, name: &str, terms: &[TermId]) {
    write!(output, "{name} ").expect("writing to a String cannot fail");
    write_term_values(output, terms);
    output.push('\n');
}

fn write_term_values(output: &mut String, terms: &[TermId]) {
    for (index, term) in terms.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{}", term.get()).expect("writing to a String cannot fail");
    }
}

fn execute_feedback(
    database: &Path,
    source_sequence: u64,
    target_sequences: &[u64],
    helpful: bool,
) -> CliResult<Vec<u8>> {
    let mut store = SqliteStore::open(database)
        .map_err(|error| format!("could not open `{}`: {error}", database.display()))?;
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
