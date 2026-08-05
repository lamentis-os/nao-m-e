use std::ffi::{OsStr, OsString};
use std::fmt::Write as FmtWrite;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nao_m_e::{Attribute, EpisodeDraft, SymbolId, TimestampMs};
use nao_m_e_sqlite::SqliteStore;

use super::{CliResult, Command, ParsedArgs, is_help_request, open_store, parse_number, save};

const ADD_HELP: &str = "Append symbolic episodes and save atomically.

Usage:
  nao-m-e add <DATABASE> [--quiet] [--timestamp <UNIX_MS>] --attribute <TEXT> --value <TEXT>... [ATTRIBUTE OPTIONS]
  nao-m-e add <DATABASE> --many [--quiet]

Attribute options:
  --attribute <TEXT> --value <TEXT>...   Add one set-valued attribute; repeatable
  --timestamp <UNIX_MS>                  Set signed Unix milliseconds; defaults to current time

With --many, standard input contains one shell-quoted single-episode flag row
per episode. Blank lines and shell comments are ignored. The command parses and
saves all rows once or saves none. Successful add writes the assigned sequence
per episode unless --quiet is present. Missing timestamps share one current-time
default per invocation.

Every new episode must be encoded before it can commit. The pinned model assets
are an installation prerequisite and are never downloaded by this command.
Missing, invalid, or unusable assets reject the complete add without publishing
symbols, an episode, its semantic vector, or a revision.
";

struct TextAttribute {
    key: String,
    values: Vec<String>,
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
    timestamp: Option<TimestampMs>,
    attributes: Vec<TextAttribute>,
}

struct EpisodeShape {
    timestamp: TimestampMs,
    value_counts: Vec<usize>,
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

    let (draft, quiet) = parse_episode_flags(
        options.iter().map(|value| EpisodeToken::Borrowed(value)),
        true,
    )?;
    Ok(ParsedArgs::Execute(Command::Add {
        database: PathBuf::from(database),
        draft: Box::new(draft),
        quiet,
    }))
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
    allow_quiet: bool,
) -> CliResult<(TextEpisodeDraft, bool)> {
    let mut timestamp = None;
    let mut attributes: Vec<TextAttribute> = Vec::new();
    let mut active_attribute = None;
    let mut quiet = false;
    let mut args = args.into_iter();

    while let Some(option) = args.next() {
        let option = option
            .as_os_str()
            .to_str()
            .ok_or_else(|| "episode options must be valid UTF-8".to_owned())?;

        if option == "--quiet" {
            if !allow_quiet {
                return Err("`--quiet` is not valid inside an add --many row".to_owned());
            }
            close_attribute(active_attribute.take(), &attributes)?;
            if quiet {
                return Err("`--quiet` may be specified only once".to_owned());
            }
            quiet = true;
            continue;
        }

        let value = args
            .next()
            .ok_or_else(|| format!("`{option}` requires a value"))?;

        if option == "--value" {
            let index = active_attribute.ok_or_else(|| {
                "`--value` must immediately follow its attribute and values".to_owned()
            })?;
            attributes[index]
                .values
                .push(value.into_text("attribute value")?);
            continue;
        }

        close_attribute(active_attribute.take(), &attributes)?;
        match option {
            "--timestamp" => {
                if timestamp
                    .replace(TimestampMs::new(parse_number(
                        value.as_os_str(),
                        "Unix timestamp",
                    )?))
                    .is_some()
                {
                    return Err("`--timestamp` may be specified only once".to_owned());
                }
            }
            "--attribute" => {
                attributes.push(TextAttribute {
                    key: value.into_text("attribute key")?,
                    values: Vec::new(),
                });
                active_attribute = Some(attributes.len() - 1);
            }
            _ => return Err(format!("unknown episode option `{option}`")),
        }
    }

    close_attribute(active_attribute, &attributes)?;
    if attributes.is_empty() {
        return Err("episode requires at least one `--attribute`".to_owned());
    }
    Ok((
        TextEpisodeDraft {
            timestamp,
            attributes,
        },
        quiet,
    ))
}

fn close_attribute(active: Option<usize>, attributes: &[TextAttribute]) -> CliResult<()> {
    if active.is_some_and(|index| attributes[index].values.is_empty()) {
        return Err("attribute requires at least one `--value` value".to_owned());
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
        drop(line);
        if words.is_empty() {
            continue;
        }
        let (draft, quiet) = parse_episode_flags(words.into_iter().map(EpisodeToken::Owned), false)
            .map_err(|error| format!("add --many line {line_number}: {error}"))?;
        debug_assert!(!quiet);
        drafts.push(draft);
    }
    if drafts.is_empty() {
        return Err("add --many requires at least one non-empty input line".to_owned());
    }
    Ok(drafts)
}

pub(super) fn execute(
    database: &Path,
    mut drafts: Vec<TextEpisodeDraft>,
    quiet: bool,
) -> CliResult<Vec<u8>> {
    fill_missing_timestamps(&mut drafts)?;
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

fn fill_missing_timestamps(drafts: &mut [TextEpisodeDraft]) -> CliResult<()> {
    if drafts.iter().all(|draft| draft.timestamp.is_some()) {
        return Ok(());
    }

    let milliseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "system clock is before the Unix epoch".to_owned())?
        .as_millis();
    let timestamp = TimestampMs::new(
        i64::try_from(milliseconds)
            .map_err(|_| "system clock exceeds the supported Unix timestamp range".to_owned())?,
    );
    for draft in drafts {
        draft.timestamp.get_or_insert(timestamp);
    }
    Ok(())
}

fn intern_drafts(
    store: &mut SqliteStore,
    drafts: Vec<TextEpisodeDraft>,
) -> CliResult<Vec<EpisodeDraft>> {
    let (symbol_values, shapes) = flatten_drafts(drafts);
    let symbol_ids = store
        .intern_symbols(&symbol_values)
        .map_err(|error| error.to_string())?;
    drop(symbol_values);
    resolve_drafts(shapes, symbol_ids)
}

fn flatten_drafts(drafts: Vec<TextEpisodeDraft>) -> (Vec<String>, Vec<EpisodeShape>) {
    let symbol_count = drafts
        .iter()
        .flat_map(|draft| &draft.attributes)
        .map(|attribute| 1 + attribute.values.len())
        .sum();
    let mut symbol_values = Vec::with_capacity(symbol_count);
    let mut shapes = Vec::with_capacity(drafts.len());

    for draft in drafts {
        let mut value_counts = Vec::with_capacity(draft.attributes.len());
        for attribute in draft.attributes {
            symbol_values.push(attribute.key);
            value_counts.push(attribute.values.len());
            symbol_values.extend(attribute.values);
        }
        shapes.push(EpisodeShape {
            timestamp: draft
                .timestamp
                .expect("missing timestamps are filled before interning"),
            value_counts,
        });
    }
    (symbol_values, shapes)
}

fn resolve_drafts(
    shapes: Vec<EpisodeShape>,
    symbol_ids: Vec<SymbolId>,
) -> CliResult<Vec<EpisodeDraft>> {
    let mut symbols = symbol_ids.into_iter();
    let drafts = shapes
        .into_iter()
        .map(|shape| {
            let attributes = shape
                .value_counts
                .into_iter()
                .map(|value_count| resolve_attribute(&mut symbols, value_count))
                .collect::<CliResult<Vec<_>>>()?;
            EpisodeDraft::new(shape.timestamp, attributes).map_err(|error| error.to_string())
        })
        .collect::<CliResult<Vec<_>>>()?;
    if symbols.next().is_some() {
        return Err("symbol interning returned an oversized result".to_owned());
    }
    Ok(drafts)
}

fn resolve_attribute(
    symbols: &mut impl Iterator<Item = SymbolId>,
    value_count: usize,
) -> CliResult<Attribute> {
    let key = symbols
        .next()
        .ok_or_else(|| "symbol interning returned an incomplete result".to_owned())?;
    let values = symbols.by_ref().take(value_count).collect::<Vec<_>>();
    if values.len() != value_count {
        return Err("symbol interning returned an incomplete result".to_owned());
    }
    Attribute::new(key, values).map_err(|error| error.to_string())
}
