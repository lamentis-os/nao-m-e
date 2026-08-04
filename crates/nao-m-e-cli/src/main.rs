#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt::Write as FmtWrite;
use std::io::{self, BufRead, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nao_m_e::{
    AtomId, EpisodeAtom, EpisodeDraft, PredicateId, SourceId, Statement, TermId, TimestampMs,
};
use nao_m_e_sqlite::SqliteStore;

const DEFAULT_RECALL_LIMIT: usize = 10;

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

const ADD_HELP: &str = "Append symbolic episodes and save atomically.

Usage:
  nao-m-e add <DATABASE> [--quiet] --occurred <MS> --recorded <MS> --source <ID> --predicate <TEXT> --term <TEXT>... [EPISODE OPTIONS]
  nao-m-e add <DATABASE> --many [--quiet]

Episode options:
  --context <TEXT> --context-term <TEXT>...   Add one context; repeatable
  --action <TEXT> --action-term <TEXT>...      Set the optional action
  --outcome <TEXT> --outcome-term <TEXT>...   Set the optional outcome

With --many, standard input contains one shell-quoted single-episode flag row
per episode. Blank lines and shell comments are ignored. The command parses and
saves all rows once or saves none. Successful add writes the assigned sequence
per episode unless --quiet is present.
";

const RECALL_HELP: &str = "Rank source-conditioned episodes without changing state.

Usage:
  nao-m-e recall <DATABASE> --from <SEQUENCE>
  nao-m-e recall <DATABASE> --from <SEQUENCE> --limit <N>

Symbolic cue overlap provides cold candidates. Direct learned feedback can add
candidates, boost their score, or suppress structural matches. The default
recall limit is 10. Hits are separated by one blank line. No hits produce no
standard output.
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
        draft: Box<TextEpisodeDraft>,
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

struct TextStatement {
    predicate: String,
    terms: Vec<String>,
}

struct TextEpisodeDraft {
    occurred_at: TimestampMs,
    recorded_at: TimestampMs,
    context: Vec<TextStatement>,
    observation: TextStatement,
    action: Option<TextStatement>,
    outcome: Option<TextStatement>,
    source: SourceId,
}

struct EpisodeShape {
    occurred_at: TimestampMs,
    recorded_at: TimestampMs,
    context_term_counts: Vec<usize>,
    observation_term_count: usize,
    action_term_count: Option<usize>,
    outcome_term_count: Option<usize>,
    source: SourceId,
}

#[derive(Clone, Copy)]
enum ActiveStatement {
    Context(usize),
    Observation,
    Action,
    Outcome,
}

impl ActiveStatement {
    fn term_option(self) -> &'static str {
        match self {
            Self::Context(_) => "--context-term",
            Self::Observation => "--term",
            Self::Action => "--action-term",
            Self::Outcome => "--outcome-term",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Context(_) => "context",
            Self::Observation => "observation",
            Self::Action => "action",
            Self::Outcome => "outcome",
        }
    }
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
    if options.first().is_some_and(|value| value == "--many") {
        let quiet = parse_mode_options(options, "--many")?;
        return Ok(ParsedArgs::Execute(Command::AddMany {
            database: PathBuf::from(database),
            quiet,
        }));
    }

    let (episode_options, quiet) = extract_quiet(options)?;
    let draft = parse_episode_flags(episode_options)?;
    Ok(ParsedArgs::Execute(Command::Add {
        database: PathBuf::from(database),
        draft: Box::new(draft),
        quiet,
    }))
}

fn extract_quiet(options: &[OsString]) -> CliResult<(Vec<&OsStr>, bool)> {
    let mut episode_options = Vec::with_capacity(options.len());
    let mut quiet = false;
    let mut index = 0;
    while index < options.len() {
        if options[index].as_os_str() == OsStr::new("--quiet") {
            if quiet {
                return Err("`--quiet` may be specified only once".to_owned());
            }
            quiet = true;
            index += 1;
            continue;
        }
        episode_options.push(options[index].as_os_str());
        if let Some(value) = options.get(index + 1) {
            episode_options.push(value.as_os_str());
        }
        index += 2;
    }
    Ok((episode_options, quiet))
}

fn parse_mode_options(options: &[OsString], required: &str) -> Result<bool, String> {
    let mut required_seen = false;
    let mut quiet = false;
    for option in options {
        match option.to_str() {
            Some(value) if value == required && !required_seen => required_seen = true,
            Some(value) if value == required => {
                return Err(format!("`{required}` may be specified only once"));
            }
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

fn parse_episode_flags<'a>(
    args: impl IntoIterator<Item = &'a OsStr>,
) -> CliResult<TextEpisodeDraft> {
    let mut occurred_at = None;
    let mut recorded_at = None;
    let mut source = None;
    let mut predicate = None;
    let mut terms = Vec::new();
    let mut context = Vec::new();
    let mut action = None;
    let mut outcome = None;
    let mut active: Option<ActiveStatement> = None;
    let mut args = args.into_iter();

    while let Some(option) = args.next() {
        let option = option
            .to_str()
            .ok_or_else(|| "episode options must be valid UTF-8".to_owned())?;
        let value = args
            .next()
            .ok_or_else(|| format!("`{option}` requires a value"))?;

        if matches!(
            option,
            "--term" | "--context-term" | "--action-term" | "--outcome-term"
        ) {
            let active_statement = active.ok_or_else(|| {
                format!("`{option}` must immediately follow its statement and terms")
            })?;
            if option != active_statement.term_option() {
                return Err(format!(
                    "`{option}` cannot be used in an active {} statement; expected `{}`",
                    active_statement.description(),
                    active_statement.term_option()
                ));
            }
            active_terms_mut(
                active_statement,
                &mut terms,
                &mut context,
                &mut action,
                &mut outcome,
            )
            .push(parse_text(
                value,
                &format!("{} term", active_statement.description()),
            )?);
            continue;
        }

        close_active_statement(
            active.take(),
            &mut terms,
            &mut context,
            &mut action,
            &mut outcome,
        )?;

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
            "--predicate" => {
                set_once(&mut predicate, parse_text(value, "predicate")?, option)?;
                active = Some(ActiveStatement::Observation);
            }
            "--context" => {
                context.push(TextStatement {
                    predicate: parse_text(value, "context predicate")?,
                    terms: Vec::new(),
                });
                active = Some(ActiveStatement::Context(context.len() - 1));
            }
            "--action" => {
                set_once(
                    &mut action,
                    TextStatement {
                        predicate: parse_text(value, "action predicate")?,
                        terms: Vec::new(),
                    },
                    option,
                )?;
                active = Some(ActiveStatement::Action);
            }
            "--outcome" => {
                set_once(
                    &mut outcome,
                    TextStatement {
                        predicate: parse_text(value, "outcome predicate")?,
                        terms: Vec::new(),
                    },
                    option,
                )?;
                active = Some(ActiveStatement::Outcome);
            }
            _ => return Err(format!("unknown episode option `{option}`")),
        }
    }

    close_active_statement(active, &mut terms, &mut context, &mut action, &mut outcome)?;

    let occurred_at = occurred_at.ok_or_else(|| "episode requires `--occurred`".to_owned())?;
    let recorded_at = recorded_at.ok_or_else(|| "episode requires `--recorded`".to_owned())?;
    let source = source.ok_or_else(|| "episode requires `--source`".to_owned())?;
    let predicate = predicate.ok_or_else(|| "episode requires `--predicate`".to_owned())?;

    Ok(TextEpisodeDraft {
        occurred_at,
        recorded_at,
        context,
        observation: TextStatement { predicate, terms },
        action,
        outcome,
        source,
    })
}

fn active_terms_mut<'a>(
    active: ActiveStatement,
    observation_terms: &'a mut Vec<String>,
    context: &'a mut [TextStatement],
    action: &'a mut Option<TextStatement>,
    outcome: &'a mut Option<TextStatement>,
) -> &'a mut Vec<String> {
    match active {
        ActiveStatement::Context(index) => &mut context[index].terms,
        ActiveStatement::Observation => observation_terms,
        ActiveStatement::Action => &mut action.as_mut().expect("an active action exists").terms,
        ActiveStatement::Outcome => &mut outcome.as_mut().expect("an active outcome exists").terms,
    }
}

fn close_active_statement(
    active: Option<ActiveStatement>,
    observation_terms: &mut Vec<String>,
    context: &mut [TextStatement],
    action: &mut Option<TextStatement>,
    outcome: &mut Option<TextStatement>,
) -> CliResult<()> {
    if let Some(active) = active {
        require_terms(
            active_terms_mut(active, observation_terms, context, action, outcome),
            active.description(),
            active.term_option(),
        )?;
    }
    Ok(())
}

fn set_once<T>(slot: &mut Option<T>, value: T, option: &str) -> CliResult<()> {
    if slot.replace(value).is_some() {
        return Err(format!("`{option}` may be specified only once"));
    }
    Ok(())
}

fn parse_text(value: &OsStr, description: &str) -> CliResult<String> {
    value
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{description} must be valid UTF-8"))
}

fn require_terms(terms: &[String], statement: &str, option: &str) -> CliResult<()> {
    if terms.is_empty() {
        return Err(format!(
            "{statement} requires at least one `{option}` value"
        ));
    }
    Ok(())
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
        } => execute_add(&database, vec![*draft], quiet),
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

fn read_many_drafts() -> CliResult<Vec<TextEpisodeDraft>> {
    let mut drafts = Vec::new();
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let line_number = index + 1;
        let line =
            line.map_err(|error| format!("could not read add --many line {line_number}: {error}"))?;
        let words = shlex::split(&line)
            .ok_or_else(|| format!("add --many line {line_number}: invalid shell quoting"))?;
        if words.is_empty() {
            continue;
        }
        let draft = parse_episode_flags(words.iter().map(String::as_str).map(OsStr::new))
            .map_err(|error| format!("add --many line {line_number}: {error}"))?;
        drafts.push(draft);
    }
    if drafts.is_empty() {
        return Err("add --many requires at least one non-empty input line".to_owned());
    }
    Ok(drafts)
}

fn execute_add(database: &Path, drafts: Vec<TextEpisodeDraft>, quiet: bool) -> CliResult<Vec<u8>> {
    let mut store = SqliteStore::open(database)
        .map_err(|error| format!("could not open `{}`: {error}", database.display()))?;
    let drafts = intern_drafts(&mut store, drafts)?;
    let mut output = if quiet {
        String::new()
    } else {
        String::with_capacity(drafts.len().saturating_mul(21))
    };
    for draft in drafts {
        let atom_id = store
            .memory_mut()
            .insert_episode(draft)
            .map_err(|error| error.to_string())?;
        if !quiet {
            writeln!(output, "{}", atom_id.sequence()).expect("writing to a String cannot fail");
        }
    }
    save(&mut store, database)?;
    Ok(output.into_bytes())
}

fn intern_drafts(
    store: &mut SqliteStore,
    drafts: Vec<TextEpisodeDraft>,
) -> CliResult<Vec<EpisodeDraft>> {
    let (predicate_values, term_values, shapes) = flatten_drafts(drafts);
    let predicate_ids = store
        .intern_predicates(&predicate_values)
        .map_err(|error| error.to_string())?;
    let term_ids = store
        .intern_terms(&term_values)
        .map_err(|error| error.to_string())?;
    resolve_drafts(shapes, predicate_ids, term_ids)
}

fn flatten_drafts(drafts: Vec<TextEpisodeDraft>) -> (Vec<String>, Vec<String>, Vec<EpisodeShape>) {
    let statement_count = drafts
        .iter()
        .map(|draft| {
            draft.context.len()
                + 1
                + usize::from(draft.action.is_some())
                + usize::from(draft.outcome.is_some())
        })
        .sum();
    let term_count = drafts
        .iter()
        .map(|draft| {
            draft
                .context
                .iter()
                .map(|statement| statement.terms.len())
                .sum::<usize>()
                + draft.observation.terms.len()
                + draft
                    .action
                    .as_ref()
                    .map_or(0, |statement| statement.terms.len())
                + draft
                    .outcome
                    .as_ref()
                    .map_or(0, |statement| statement.terms.len())
        })
        .sum();
    let mut predicate_values = Vec::with_capacity(statement_count);
    let mut term_values = Vec::with_capacity(term_count);
    let mut shapes = Vec::with_capacity(drafts.len());

    for draft in drafts {
        let context_term_counts = draft
            .context
            .into_iter()
            .map(|statement| flatten_statement(statement, &mut predicate_values, &mut term_values))
            .collect();
        let observation_term_count =
            flatten_statement(draft.observation, &mut predicate_values, &mut term_values);
        let action_term_count = draft
            .action
            .map(|statement| flatten_statement(statement, &mut predicate_values, &mut term_values));
        let outcome_term_count = draft
            .outcome
            .map(|statement| flatten_statement(statement, &mut predicate_values, &mut term_values));
        shapes.push(EpisodeShape {
            occurred_at: draft.occurred_at,
            recorded_at: draft.recorded_at,
            context_term_counts,
            observation_term_count,
            action_term_count,
            outcome_term_count,
            source: draft.source,
        });
    }

    (predicate_values, term_values, shapes)
}

fn flatten_statement(
    statement: TextStatement,
    predicates: &mut Vec<String>,
    terms: &mut Vec<String>,
) -> usize {
    predicates.push(statement.predicate);
    let term_count = statement.terms.len();
    terms.extend(statement.terms);
    term_count
}

fn resolve_drafts(
    shapes: Vec<EpisodeShape>,
    predicate_ids: Vec<PredicateId>,
    term_ids: Vec<TermId>,
) -> CliResult<Vec<EpisodeDraft>> {
    let mut predicates = predicate_ids.into_iter();
    let mut terms = term_ids.into_iter();
    let drafts = shapes
        .into_iter()
        .map(|shape| {
            let context = shape
                .context_term_counts
                .into_iter()
                .map(|term_count| resolve_statement(&mut predicates, &mut terms, term_count))
                .collect::<CliResult<Vec<_>>>()?;
            let observation =
                resolve_statement(&mut predicates, &mut terms, shape.observation_term_count)?;
            let action = shape
                .action_term_count
                .map(|term_count| resolve_statement(&mut predicates, &mut terms, term_count))
                .transpose()?;
            let outcome = shape
                .outcome_term_count
                .map(|term_count| resolve_statement(&mut predicates, &mut terms, term_count))
                .transpose()?;
            Ok(EpisodeDraft {
                occurred_at: shape.occurred_at,
                recorded_at: shape.recorded_at,
                context,
                observation,
                action,
                outcome,
                source: shape.source,
            })
        })
        .collect::<CliResult<Vec<_>>>()?;
    debug_assert!(predicates.next().is_none());
    debug_assert!(terms.next().is_none());
    Ok(drafts)
}

fn resolve_statement(
    predicates: &mut impl Iterator<Item = PredicateId>,
    terms: &mut impl Iterator<Item = TermId>,
    term_count: usize,
) -> CliResult<Statement> {
    let predicate = predicates
        .next()
        .ok_or_else(|| "predicate interning returned an incomplete result".to_owned())?;
    let arguments = terms.by_ref().take(term_count).collect::<Vec<_>>();
    if arguments.len() != term_count {
        return Err("term interning returned an incomplete result".to_owned());
    }
    Statement::new(predicate, arguments).map_err(|error| error.to_string())
}

fn execute_recall(database: &Path, source_sequence: u64, limit: usize) -> CliResult<Vec<u8>> {
    let store = SqliteStore::open(database)
        .map_err(|error| format!("could not open `{}`: {error}", database.display()))?;
    let source = AtomId::from_parts(store.memory_id(), source_sequence);
    let hits = store
        .memory()
        .recall_from(source, limit)
        .map_err(|error| error.to_string())?;
    format_recall(&store, &hits)
}

fn format_recall(store: &SqliteStore, hits: &[nao_m_e::RecallHit]) -> CliResult<Vec<u8>> {
    let mut predicate_ids = BTreeSet::new();
    let mut term_ids = BTreeSet::new();
    for hit in hits {
        let episode = store
            .memory()
            .episode(hit.atom_id)
            .expect("recall hits always reference stored episodes");
        collect_episode_symbols(episode, &mut predicate_ids, &mut term_ids);
    }
    let predicate_ids: Vec<_> = predicate_ids.into_iter().collect();
    let term_ids: Vec<_> = term_ids.into_iter().collect();
    let predicate_values = store
        .predicate_values(&predicate_ids)
        .map_err(|error| error.to_string())?;
    let term_values = store
        .term_values(&term_ids)
        .map_err(|error| error.to_string())?;
    let predicate_values = resolved_predicates(predicate_ids, predicate_values)?;
    let term_values = resolved_terms(term_ids, term_values)?;
    let mut output = String::new();
    for (index, hit) in hits.iter().enumerate() {
        if index != 0 {
            output.push('\n');
        }
        let episode = store
            .memory()
            .episode(hit.atom_id)
            .expect("recall hits always reference stored episodes");
        write_recall_hit(
            &mut output,
            episode,
            hit.activation.as_ppm(),
            &predicate_values,
            &term_values,
        );
    }
    Ok(output.into_bytes())
}

fn resolved_predicates(
    ids: Vec<PredicateId>,
    values: Vec<Option<String>>,
) -> CliResult<BTreeMap<PredicateId, String>> {
    if ids.len() != values.len() {
        return Err("predicate lookup returned an incomplete result".to_owned());
    }
    ids.into_iter()
        .zip(values)
        .map(|(id, value)| {
            value.map(|value| (id, value)).ok_or_else(|| {
                format!(
                    "stored episode references unresolved predicate symbol {}",
                    id.get()
                )
            })
        })
        .collect()
}

fn resolved_terms(
    ids: Vec<TermId>,
    values: Vec<Option<String>>,
) -> CliResult<BTreeMap<TermId, String>> {
    if ids.len() != values.len() {
        return Err("term lookup returned an incomplete result".to_owned());
    }
    ids.into_iter()
        .zip(values)
        .map(|(id, value)| {
            value.map(|value| (id, value)).ok_or_else(|| {
                format!(
                    "stored episode references unresolved term symbol {}",
                    id.get()
                )
            })
        })
        .collect()
}

fn collect_episode_symbols(
    episode: &EpisodeAtom,
    predicates: &mut BTreeSet<PredicateId>,
    terms: &mut BTreeSet<TermId>,
) {
    for statement in episode
        .context()
        .iter()
        .chain(std::iter::once(episode.observation()))
        .chain(episode.action())
        .chain(episode.outcome())
    {
        predicates.insert(statement.predicate());
        terms.extend(statement.arguments());
    }
}

fn write_recall_hit(
    output: &mut String,
    episode: &EpisodeAtom,
    activation_ppm: u32,
    predicates: &BTreeMap<PredicateId, String>,
    terms: &BTreeMap<TermId, String>,
) {
    writeln!(output, "sequence {}", episode.id().sequence())
        .expect("writing to a String cannot fail");
    writeln!(output, "activation_ppm {activation_ppm}").expect("writing to a String cannot fail");
    writeln!(output, "occurred {}", episode.occurred_at().get())
        .expect("writing to a String cannot fail");
    writeln!(output, "recorded {}", episode.recorded_at().get())
        .expect("writing to a String cannot fail");
    writeln!(output, "source {}", episode.source().get()).expect("writing to a String cannot fail");
    for statement in episode.context() {
        write_statement(
            output,
            "context",
            "context-term",
            statement,
            predicates,
            terms,
        );
    }
    write_statement(
        output,
        "predicate",
        "term",
        episode.observation(),
        predicates,
        terms,
    );
    if let Some(action) = episode.action() {
        write_statement(output, "action", "action-term", action, predicates, terms);
    }
    if let Some(outcome) = episode.outcome() {
        write_statement(
            output,
            "outcome",
            "outcome-term",
            outcome,
            predicates,
            terms,
        );
    }
}

fn write_statement(
    output: &mut String,
    predicate_label: &str,
    term_label: &str,
    statement: &Statement,
    predicates: &BTreeMap<PredicateId, String>,
    terms: &BTreeMap<TermId, String>,
) {
    let predicate = predicates
        .get(&statement.predicate())
        .expect("every recalled predicate was resolved");
    writeln!(output, "{predicate_label} {predicate}").expect("writing to a String cannot fail");
    for term_id in statement.arguments() {
        let term = terms
            .get(term_id)
            .expect("every recalled term was resolved");
        writeln!(output, "{term_label} {term}").expect("writing to a String cannot fail");
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
