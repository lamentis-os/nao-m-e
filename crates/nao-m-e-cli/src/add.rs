use std::ffi::{OsStr, OsString};
use std::fmt::Write as FmtWrite;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

use nao_m_e::{EpisodeDraft, PredicateId, SourceId, Statement, TermId, TimestampMs};
use nao_m_e_sqlite::SqliteStore;

use super::{CliResult, Command, ParsedArgs, is_help_request, open_store, parse_number, save};

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

struct TextStatement {
    predicate: String,
    terms: Vec<String>,
}

enum EpisodeToken<'a> {
    Borrowed(&'a OsStr),
    Owned(String),
}

impl EpisodeToken<'_> {
    fn as_os_str(&self) -> &OsStr {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => OsStr::new(value.as_str()),
        }
    }

    fn into_text(self, description: &str) -> CliResult<String> {
        match self {
            Self::Borrowed(value) => value
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{description} must be valid UTF-8")),
            Self::Owned(value) => Ok(value),
        }
    }
}

pub(super) struct TextEpisodeDraft {
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

    fn term_description(self) -> &'static str {
        match self {
            Self::Context(_) => "context term",
            Self::Observation => "observation term",
            Self::Action => "action term",
            Self::Outcome => "outcome term",
        }
    }
}

pub(super) fn parse_args(args: &[OsString]) -> Result<ParsedArgs, String> {
    if is_help_request(args) {
        return Ok(ParsedArgs::Print(ADD_HELP.to_owned()));
    }
    let Some(database) = args.first() else {
        return Err("`add` requires a database path and episode flags".to_owned());
    };
    let options = &args[1..];
    if options.first().is_some_and(|value| value == "--many") {
        let quiet = parse_many_options(&options[1..])?;
        return Ok(ParsedArgs::Execute(Command::AddMany {
            database: PathBuf::from(database),
            quiet,
        }));
    }

    let (episode_options, quiet) = extract_quiet(options)?;
    let draft = parse_episode_flags(episode_options.into_iter().map(EpisodeToken::Borrowed))?;
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

fn parse_many_options(options: &[OsString]) -> Result<bool, String> {
    let mut quiet = false;
    for option in options {
        match option.to_str() {
            Some("--many") => return Err("`--many` may be specified only once".to_owned()),
            Some("--quiet") if !quiet => quiet = true,
            Some("--quiet") => return Err("`--quiet` may be specified only once".to_owned()),
            Some(value) => {
                return Err(format!(
                    "`--many` cannot be combined with episode option `{value}`"
                ));
            }
            None => return Err("add options must be valid UTF-8".to_owned()),
        }
    }
    Ok(quiet)
}

fn parse_episode_flags<'a>(
    args: impl IntoIterator<Item = EpisodeToken<'a>>,
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
            .as_os_str()
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
            .push(value.into_text(active_statement.term_description())?);
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
                TimestampMs::new(parse_number(value.as_os_str(), "occurred timestamp")?),
                option,
            )?,
            "--recorded" => set_once(
                &mut recorded_at,
                TimestampMs::new(parse_number(value.as_os_str(), "recorded timestamp")?),
                option,
            )?,
            "--source" => set_once(
                &mut source,
                SourceId::new(parse_number(value.as_os_str(), "source ID")?),
                option,
            )?,
            "--predicate" => {
                set_once(&mut predicate, value.into_text("predicate")?, option)?;
                active = Some(ActiveStatement::Observation);
            }
            "--context" => {
                context.push(TextStatement {
                    predicate: value.into_text("context predicate")?,
                    terms: Vec::new(),
                });
                active = Some(ActiveStatement::Context(context.len() - 1));
            }
            "--action" => {
                set_once(
                    &mut action,
                    TextStatement {
                        predicate: value.into_text("action predicate")?,
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
                        predicate: value.into_text("outcome predicate")?,
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

fn require_terms(terms: &[String], statement: &str, option: &str) -> CliResult<()> {
    if terms.is_empty() {
        return Err(format!(
            "{statement} requires at least one `{option}` value"
        ));
    }
    Ok(())
}

pub(super) fn read_many_drafts() -> CliResult<Vec<TextEpisodeDraft>> {
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
        let draft = parse_episode_flags(words.into_iter().map(EpisodeToken::Owned))
            .map_err(|error| format!("add --many line {line_number}: {error}"))?;
        drafts.push(draft);
    }
    if drafts.is_empty() {
        return Err("add --many requires at least one non-empty input line".to_owned());
    }
    Ok(drafts)
}

pub(super) fn execute(
    database: &Path,
    drafts: Vec<TextEpisodeDraft>,
    quiet: bool,
) -> CliResult<Vec<u8>> {
    let mut store = open_store(database)?;
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
