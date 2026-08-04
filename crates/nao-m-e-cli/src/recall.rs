use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};

use nao_m_e::{AtomId, EpisodeAtom, PredicateId, Statement, TermId};
use nao_m_e_sqlite::SqliteStore;

use super::{CliResult, Command, ParsedArgs, is_help_request, open_store, parse_number};

const DEFAULT_RECALL_LIMIT: usize = 10;

const RECALL_HELP: &str = "Rank source-conditioned episodes without changing state.

Usage:
  nao-m-e recall <DATABASE> --from <SEQUENCE>
  nao-m-e recall <DATABASE> --from <SEQUENCE> --limit <N>

Symbolic cue overlap provides cold candidates. Direct learned feedback can add
candidates, boost their score, or suppress structural matches. The default
recall limit is 10. Hits are separated by one blank line. No hits produce no
standard output.
";

pub(super) fn parse_args(args: &[OsString]) -> Result<ParsedArgs, String> {
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

pub(super) fn execute(database: &Path, source_sequence: u64, limit: usize) -> CliResult<Vec<u8>> {
    let store = open_store(database)?;
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
    let predicate_values = resolve_symbols(
        predicate_ids,
        predicate_values,
        "predicate",
        PredicateId::get,
    )?;
    let term_values = resolve_symbols(term_ids, term_values, "term", TermId::get)?;
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

struct ResolvedSymbols<I> {
    ids: Vec<I>,
    values: Vec<Option<String>>,
}

impl<I: Ord> ResolvedSymbols<I> {
    fn get(&self, id: &I) -> Option<&str> {
        let index = self.ids.binary_search(id).ok()?;
        self.values[index].as_deref()
    }
}

fn resolve_symbols<I: Copy + Ord>(
    ids: Vec<I>,
    values: Vec<Option<String>>,
    namespace: &str,
    raw_id: impl Fn(I) -> u64,
) -> CliResult<ResolvedSymbols<I>> {
    if ids.len() != values.len() {
        return Err(format!("{namespace} lookup returned an incomplete result"));
    }
    debug_assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
    if let Some(index) = values.iter().position(Option::is_none) {
        return Err(format!(
            "stored episode references unresolved {namespace} symbol {}",
            raw_id(ids[index])
        ));
    }
    Ok(ResolvedSymbols { ids, values })
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
    predicates: &ResolvedSymbols<PredicateId>,
    terms: &ResolvedSymbols<TermId>,
) {
    writeln!(output, "sequence {}", episode.id().sequence())
        .expect("writing to a String cannot fail");
    writeln!(output, "activation_ppm {activation_ppm}").expect("writing to a String cannot fail");
    writeln!(output, "timestamp {}", episode.timestamp().get())
        .expect("writing to a String cannot fail");
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
    predicates: &ResolvedSymbols<PredicateId>,
    terms: &ResolvedSymbols<TermId>,
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
