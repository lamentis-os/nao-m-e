use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};

use nao_m_e::{EpisodeAtom, SymbolId};
use nao_m_e_sqlite::SqliteStore;

use super::{CliResult, Command, ParsedArgs, is_help_request, open_store, parse_number};

const DEFAULT_RECALL_LIMIT: usize = 10;

const RECALL_HELP: &str = "Rank episodes for one semantic free-text query without changing state.

Usage:
  nao-m-e recall <DATABASE> --query <TEXT>
  nao-m-e recall <DATABASE> --query <TEXT> --limit <N>

One fixed language-agnostic local model path compares the query with each
committed episode vector. Its pinned assets must be provisioned during product
installation and are never downloaded by recall. Existing episode-to-episode
feedback does not affect this query. The default recall limit is 10. Hits are
separated by one blank line. No hits produce no standard output.
";

pub(super) fn parse_args(args: &[OsString]) -> Result<ParsedArgs, String> {
    if is_help_request(args) {
        return Ok(ParsedArgs::Print(RECALL_HELP.to_owned()));
    }
    let Some(database) = args.first() else {
        return Err("`recall` requires a database path".to_owned());
    };

    let (query, limit) = match &args[1..] {
        [option, query] if option.as_os_str() == OsStr::new("--query") => {
            (query, DEFAULT_RECALL_LIMIT)
        }
        [query_option, query, limit_option, limit]
            if query_option.as_os_str() == OsStr::new("--query")
                && limit_option.as_os_str() == OsStr::new("--limit") =>
        {
            (query, parse_number(limit, "recall limit")?)
        }
        _ => {
            return Err("`recall` requires <DATABASE> --query <TEXT> [--limit <N>]".to_owned());
        }
    };
    let query = query
        .to_str()
        .ok_or_else(|| "semantic query must be valid UTF-8".to_owned())?
        .to_owned();

    Ok(ParsedArgs::Execute(Command::Recall {
        database: PathBuf::from(database),
        query,
        limit,
    }))
}

pub(super) fn execute(database: &Path, query: &str, limit: usize) -> CliResult<Vec<u8>> {
    let mut store = open_store(database)?;
    let hits = store
        .recall_semantic(query, limit)
        .map_err(|error| error.to_string())?;
    format_recall(&store, &hits)
}

fn format_recall(store: &SqliteStore, hits: &[nao_m_e::RecallHit]) -> CliResult<Vec<u8>> {
    let mut symbol_ids = BTreeSet::new();
    for hit in hits {
        let episode = store
            .memory()
            .episode(hit.atom_id)
            .expect("recall hits always reference stored episodes");
        collect_episode_symbols(episode, &mut symbol_ids);
    }
    let symbol_ids: Vec<_> = symbol_ids.into_iter().collect();
    let symbol_values = store
        .symbol_values(&symbol_ids)
        .map_err(|error| error.to_string())?;
    let symbols = resolve_symbols(symbol_ids, symbol_values)?;
    let mut output = String::new();
    for (index, hit) in hits.iter().enumerate() {
        if index != 0 {
            output.push('\n');
        }
        let episode = store
            .memory()
            .episode(hit.atom_id)
            .expect("recall hits always reference stored episodes");
        write_recall_hit(&mut output, episode, hit.activation.as_ppm(), &symbols);
    }
    Ok(output.into_bytes())
}

struct ResolvedSymbols {
    ids: Vec<SymbolId>,
    values: Vec<String>,
}

impl ResolvedSymbols {
    fn get(&self, id: &SymbolId) -> Option<&str> {
        let index = self.ids.binary_search(id).ok()?;
        Some(self.values[index].as_str())
    }
}

fn resolve_symbols(ids: Vec<SymbolId>, values: Vec<Option<String>>) -> CliResult<ResolvedSymbols> {
    if ids.len() != values.len() {
        return Err("symbol lookup returned an incomplete result".to_owned());
    }
    debug_assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
    if let Some(index) = values.iter().position(Option::is_none) {
        return Err(format!(
            "stored episode references unresolved symbol {}",
            ids[index].get()
        ));
    }
    let values = values
        .into_iter()
        .map(|value| value.expect("missing symbol values were rejected"))
        .collect();
    Ok(ResolvedSymbols { ids, values })
}

fn collect_episode_symbols(episode: &EpisodeAtom, symbols: &mut BTreeSet<SymbolId>) {
    for attribute in episode.attributes() {
        symbols.insert(attribute.key());
        symbols.extend(attribute.values());
    }
}

fn write_recall_hit(
    output: &mut String,
    episode: &EpisodeAtom,
    activation_ppm: u32,
    symbols: &ResolvedSymbols,
) {
    writeln!(output, "sequence {}", episode.id().sequence())
        .expect("writing to a String cannot fail");
    writeln!(output, "activation_ppm {activation_ppm}").expect("writing to a String cannot fail");
    writeln!(output, "timestamp {}", episode.timestamp().get())
        .expect("writing to a String cannot fail");
    for attribute in episode.attributes() {
        let key = symbols
            .get(&attribute.key())
            .expect("every recalled attribute key was resolved");
        writeln!(output, "attribute {key}").expect("writing to a String cannot fail");
        for value_id in attribute.values() {
            let value = symbols
                .get(value_id)
                .expect("every recalled attribute value was resolved");
            writeln!(output, "value {value}").expect("writing to a String cannot fail");
        }
    }
}
