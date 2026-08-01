#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nao_m_e::{
    Activation, AtomId, EpisodeAtom, EpisodeDraft, InfluenceWeight, MemoryId, MemoryV0,
    PredicateId, SourceId, Statement, TermId, TimestampMs,
};
use nao_m_e_sqlite::SqliteStore;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_RECALL_LIMIT: usize = 10;

const ROOT_HELP: &str = "NAO-M-E symbolic memory command-line interface

Usage:
  nao-m-e init <DATABASE>
  nao-m-e run <DATABASE> --input <FILE|->
  nao-m-e recall <DATABASE> [--limit <N>]
  nao-m-e recall <DATABASE> --sequence <N>

Commands:
  init      Create a new SQLite memory store without replacing an existing file
  run       Apply one JSON operation batch and save it atomically
  recall    Return ranked active episodes or inspect one episode by sequence

Options:
  -h, --help     Show help
  --version      Show the CLI version
";

const INIT_HELP: &str = "Create a new SQLite memory store.

Usage:
  nao-m-e init <DATABASE>
";

const RUN_HELP: &str = "Apply a versioned JSON operation batch and save it atomically.

Usage:
  nao-m-e run <DATABASE> --input <FILE|->

Use '-' as the input path to read JSON from standard input.
";

const RECALL_HELP: &str = "Return ranked active episodes or inspect one episode by sequence.

Usage:
  nao-m-e recall <DATABASE> [--limit <N>]
  nao-m-e recall <DATABASE> --sequence <N>

The default recall limit is 10. The two modes are mutually exclusive.
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

#[derive(Debug)]
struct CliError(String);

impl CliError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CliError {}

type CliResult<T> = Result<T, CliError>;

enum ParsedArgs {
    Execute(Command),
    Print(String),
}

enum Command {
    Init { database: PathBuf },
    Run { database: PathBuf, input: PathBuf },
    Recall { database: PathBuf, mode: RecallMode },
}

enum RecallMode {
    Top { limit: usize },
    Sequence { sequence: u64 },
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
        "run" => parse_run_args(&args[1..]),
        "recall" => parse_recall_args(&args[1..]),
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

fn parse_run_args(args: &[OsString]) -> Result<ParsedArgs, String> {
    if is_help_request(args) {
        return Ok(ParsedArgs::Print(RUN_HELP.to_owned()));
    }
    if args.len() != 3 || args[1].as_os_str() != OsStr::new("--input") {
        return Err("`run` requires <DATABASE> --input <FILE|->".to_owned());
    }

    Ok(ParsedArgs::Execute(Command::Run {
        database: PathBuf::from(&args[0]),
        input: PathBuf::from(&args[2]),
    }))
}

fn parse_recall_args(args: &[OsString]) -> Result<ParsedArgs, String> {
    if is_help_request(args) {
        return Ok(ParsedArgs::Print(RECALL_HELP.to_owned()));
    }
    let Some(database) = args.first() else {
        return Err("`recall` requires a database path".to_owned());
    };

    let mode = match &args[1..] {
        [] => RecallMode::Top {
            limit: DEFAULT_RECALL_LIMIT,
        },
        [option, value] if option.as_os_str() == OsStr::new("--limit") => RecallMode::Top {
            limit: parse_number(value, "recall limit")?,
        },
        [option, value] if option.as_os_str() == OsStr::new("--sequence") => RecallMode::Sequence {
            sequence: parse_number(value, "episode sequence")?,
        },
        _ => {
            return Err("`recall` accepts either `--limit <N>` or `--sequence <N>`".to_owned());
        }
    };

    Ok(ParsedArgs::Execute(Command::Recall {
        database: PathBuf::from(database),
        mode,
    }))
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
        Command::Run { database, input } => execute_run(&database, &input),
        Command::Recall { database, mode } => execute_recall(&database, mode),
    }
}

fn execute_init(database: &Path) -> CliResult<Vec<u8>> {
    let store = SqliteStore::create(database).map_err(|error| {
        CliError::new(format!(
            "could not create `{}`: {error}",
            database.display()
        ))
    })?;
    let response = InitResponse {
        schema_version: SCHEMA_VERSION,
        memory_id: encode_memory_id(store.memory_id()),
        episode_count: 0,
    };
    serialize_response(&response)
}

fn execute_run(database: &Path, input: &Path) -> CliResult<Vec<u8>> {
    let scenario = read_scenario(input)?;
    if scenario.schema_version != SCHEMA_VERSION {
        return Err(CliError::new(format!(
            "unsupported scenario schema_version {}; expected {SCHEMA_VERSION}",
            scenario.schema_version
        )));
    }
    if scenario.operations.is_empty() {
        return Err(CliError::new("scenario operations must not be empty"));
    }

    let operations = scenario
        .operations
        .into_iter()
        .enumerate()
        .map(|(index, operation)| {
            serde_json::from_value(operation)
                .map_err(|error| CliError::new(format!("operations[{index}] is invalid: {error}")))
        })
        .collect::<CliResult<Vec<OperationInput>>>()?;
    let operation_count = operations.len();
    let mut store = SqliteStore::open(database).map_err(|error| {
        CliError::new(format!("could not open `{}`: {error}", database.display()))
    })?;
    let baseline_episode_count = store.memory().episodes().len();
    let mut labels = BTreeMap::new();
    let mut inserted = Vec::new();

    for (index, operation) in operations.into_iter().enumerate() {
        let name = operation.name();
        if let Err(error) = apply_operation(
            operation,
            &mut store,
            baseline_episode_count,
            &mut labels,
            &mut inserted,
        ) {
            return Err(CliError::new(format!(
                "operations[{index}] ({name}) failed: {error}"
            )));
        }
    }

    let response = RunResponse {
        schema_version: SCHEMA_VERSION,
        memory_id: encode_memory_id(store.memory_id()),
        operations_applied: u64::try_from(operation_count)
            .map_err(|_| CliError::new("operation count exceeds the JSON protocol range"))?,
        episode_count: u64::try_from(store.memory().episodes().len())
            .map_err(|_| CliError::new("episode count exceeds the JSON protocol range"))?,
        inserted,
    };
    let output = serialize_response(&response)?;

    store.save().map_err(|error| {
        CliError::new(format!(
            "could not save `{}`: {error}; the batch was not committed",
            database.display()
        ))
    })?;
    Ok(output)
}

fn execute_recall(database: &Path, mode: RecallMode) -> CliResult<Vec<u8>> {
    let store = SqliteStore::open(database).map_err(|error| {
        CliError::new(format!("could not open `{}`: {error}", database.display()))
    })?;
    let memory_id = encode_memory_id(store.memory_id());

    match mode {
        RecallMode::Top { limit } => {
            let hits = store
                .memory()
                .top_k(limit)
                .into_iter()
                .map(|hit| {
                    let episode = store
                        .memory()
                        .episode(hit.atom_id)
                        .expect("recall hits always reference stored episodes");
                    RecallEpisode::from_atom(episode, hit.activation.as_ppm())
                })
                .collect();
            serialize_response(&RecallResponse {
                schema_version: SCHEMA_VERSION,
                memory_id,
                hits,
            })
        }
        RecallMode::Sequence { sequence } => {
            let id = AtomId::from_parts(store.memory_id(), sequence);
            let episode = store
                .memory()
                .episode(id)
                .ok_or_else(|| CliError::new(format!("unknown episode sequence {sequence}")))?;
            let activation = store
                .memory()
                .activation(id)
                .expect("stored episodes always have activation state");
            serialize_response(&SequenceResponse {
                schema_version: SCHEMA_VERSION,
                memory_id,
                sequence,
                activation_ppm: activation.as_ppm(),
                episode: EpisodeDocument::from(episode),
            })
        }
    }
}

fn read_scenario(input: &Path) -> CliResult<ScenarioInput> {
    if input.as_os_str() == OsStr::new("-") {
        serde_json::from_reader(io::stdin().lock())
            .map_err(|error| CliError::new(format!("invalid scenario JSON from stdin: {error}")))
    } else {
        let file = File::open(input).map_err(|error| {
            CliError::new(format!("could not read `{}`: {error}", input.display()))
        })?;
        serde_json::from_reader(BufReader::new(file)).map_err(|error| {
            CliError::new(format!(
                "invalid scenario JSON in `{}`: {error}",
                input.display()
            ))
        })
    }
}

fn apply_operation(
    operation: OperationInput,
    store: &mut SqliteStore,
    baseline_episode_count: usize,
    labels: &mut BTreeMap<String, AtomId>,
    inserted: &mut Vec<InsertedEpisode>,
) -> CliResult<()> {
    match operation {
        OperationInput::InsertEpisode { label, episode } => {
            if let Some(label) = label.as_deref() {
                if label.is_empty() {
                    return Err(CliError::new("insert label must not be empty"));
                }
                if labels.contains_key(label) {
                    return Err(CliError::new(format!(
                        "insert label `{label}` is already defined"
                    )));
                }
            }

            let draft = episode.into_draft()?;
            let id = store
                .memory_mut()
                .insert_episode(draft)
                .map_err(|error| CliError::new(error.to_string()))?;
            if let Some(label) = label.as_ref() {
                labels.insert(label.clone(), id);
            }
            inserted.push(InsertedEpisode {
                label,
                sequence: id.sequence(),
            });
        }
        OperationInput::SetRelevance {
            from,
            to,
            weight_ppm,
        } => {
            let from = resolve_atom(from, store.memory(), baseline_episode_count, labels)?;
            let to = resolve_atom(to, store.memory(), baseline_episode_count, labels)?;
            let weight = InfluenceWeight::from_ppm(weight_ppm)
                .map_err(|error| CliError::new(error.to_string()))?;
            store
                .memory_mut()
                .set_relevance(from, to, weight)
                .map_err(|error| CliError::new(error.to_string()))?;
        }
        OperationInput::RemoveRelevance { from, to } => {
            let from = resolve_atom(from, store.memory(), baseline_episode_count, labels)?;
            let to = resolve_atom(to, store.memory(), baseline_episode_count, labels)?;
            store
                .memory_mut()
                .remove_relevance(from, to)
                .map_err(|error| CliError::new(error.to_string()))?;
        }
        OperationInput::Stimulate { atom, amount_ppm } => {
            let atom = resolve_atom(atom, store.memory(), baseline_episode_count, labels)?;
            let amount = Activation::from_ppm(amount_ppm)
                .map_err(|error| CliError::new(error.to_string()))?;
            store
                .memory_mut()
                .stimulate(atom, amount)
                .map_err(|error| CliError::new(error.to_string()))?;
        }
        OperationInput::Step { count } => {
            if count == 0 {
                return Err(CliError::new("step count must be positive"));
            }
            for _ in 0..count {
                store.memory_mut().step();
            }
        }
        OperationInput::ResetActivations {} => store.memory_mut().reset_activations(),
    }
    Ok(())
}

fn resolve_atom(
    reference: AtomReferenceInput,
    memory: &MemoryV0,
    baseline_episode_count: usize,
    labels: &BTreeMap<String, AtomId>,
) -> CliResult<AtomId> {
    match reference {
        AtomReferenceInput::Sequence(SequenceReferenceInput { sequence }) => {
            let index = usize::try_from(sequence).map_err(|_| {
                CliError::new(format!("episode sequence {sequence} is not addressable"))
            })?;
            if index >= baseline_episode_count {
                return Err(CliError::new(format!(
                    "episode sequence {sequence} was not persisted before this batch; use an insert label"
                )));
            }
            let id = AtomId::from_parts(memory.memory_id(), sequence);
            if memory.episode(id).is_none() {
                return Err(CliError::new(format!(
                    "unknown episode sequence {sequence}"
                )));
            }
            Ok(id)
        }
        AtomReferenceInput::Label(LabelReferenceInput { label }) => {
            if label.is_empty() {
                return Err(CliError::new("atom label must not be empty"));
            }
            labels
                .get(&label)
                .copied()
                .ok_or_else(|| CliError::new(format!("unknown or forward label `{label}`")))
        }
    }
}

fn serialize_response<T: Serialize>(response: &T) -> CliResult<Vec<u8>> {
    let mut output = serde_json::to_vec_pretty(response)
        .map_err(|error| CliError::new(format!("could not serialize JSON output: {error}")))?;
    output.push(b'\n');
    Ok(output)
}

fn write_stdout(output: Vec<u8>) -> CliResult<()> {
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(&output)
        .and_then(|()| stdout.flush())
        .map_err(|error| CliError::new(format!("could not write standard output: {error}")))
}

fn encode_memory_id(memory_id: MemoryId) -> String {
    format!("{:032x}", memory_id.get())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioInput {
    schema_version: u32,
    operations: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum OperationInput {
    InsertEpisode {
        #[serde(default)]
        label: Option<String>,
        episode: EpisodeDocument,
    },
    SetRelevance {
        from: AtomReferenceInput,
        to: AtomReferenceInput,
        weight_ppm: u32,
    },
    RemoveRelevance {
        from: AtomReferenceInput,
        to: AtomReferenceInput,
    },
    Stimulate {
        atom: AtomReferenceInput,
        amount_ppm: u32,
    },
    Step {
        count: u32,
    },
    ResetActivations {},
}

impl OperationInput {
    const fn name(&self) -> &'static str {
        match self {
            Self::InsertEpisode { .. } => "insert_episode",
            Self::SetRelevance { .. } => "set_relevance",
            Self::RemoveRelevance { .. } => "remove_relevance",
            Self::Stimulate { .. } => "stimulate",
            Self::Step { .. } => "step",
            Self::ResetActivations {} => "reset_activations",
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum AtomReferenceInput {
    Sequence(SequenceReferenceInput),
    Label(LabelReferenceInput),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SequenceReferenceInput {
    sequence: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LabelReferenceInput {
    label: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EpisodeDocument {
    occurred_at_ms: i64,
    recorded_at_ms: i64,
    source_id: u64,
    #[serde(default)]
    context: Vec<StatementDocument>,
    observation: StatementDocument,
    #[serde(default)]
    action: Option<StatementDocument>,
    #[serde(default)]
    outcome: Option<StatementDocument>,
}

impl EpisodeDocument {
    fn into_draft(self) -> CliResult<EpisodeDraft> {
        let context = self
            .context
            .into_iter()
            .enumerate()
            .map(|(index, statement)| {
                statement
                    .into_statement()
                    .map_err(|error| CliError::new(format!("context[{index}]: {error}")))
            })
            .collect::<CliResult<Vec<_>>>()?;
        let observation = self
            .observation
            .into_statement()
            .map_err(|error| CliError::new(format!("observation: {error}")))?;
        let action = self
            .action
            .map(StatementDocument::into_statement)
            .transpose()
            .map_err(|error| CliError::new(format!("action: {error}")))?;
        let outcome = self
            .outcome
            .map(StatementDocument::into_statement)
            .transpose()
            .map_err(|error| CliError::new(format!("outcome: {error}")))?;

        Ok(EpisodeDraft {
            occurred_at: TimestampMs::new(self.occurred_at_ms),
            recorded_at: TimestampMs::new(self.recorded_at_ms),
            context,
            observation,
            action,
            outcome,
            source: SourceId::new(self.source_id),
        })
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StatementDocument {
    predicate_id: u64,
    term_ids: Vec<u64>,
}

impl StatementDocument {
    fn into_statement(self) -> Result<Statement, nao_m_e::ModelError> {
        Statement::new(
            PredicateId::new(self.predicate_id),
            self.term_ids.into_iter().map(TermId::new).collect(),
        )
    }
}

#[derive(Serialize)]
struct InitResponse {
    schema_version: u32,
    memory_id: String,
    episode_count: u64,
}

#[derive(Serialize)]
struct RunResponse {
    schema_version: u32,
    memory_id: String,
    operations_applied: u64,
    episode_count: u64,
    inserted: Vec<InsertedEpisode>,
}

#[derive(Serialize)]
struct InsertedEpisode {
    label: Option<String>,
    sequence: u64,
}

#[derive(Serialize)]
struct RecallResponse {
    schema_version: u32,
    memory_id: String,
    hits: Vec<RecallEpisode>,
}

#[derive(Serialize)]
struct SequenceResponse {
    schema_version: u32,
    memory_id: String,
    sequence: u64,
    activation_ppm: u32,
    episode: EpisodeDocument,
}

#[derive(Serialize)]
struct RecallEpisode {
    sequence: u64,
    activation_ppm: u32,
    episode: EpisodeDocument,
}

impl RecallEpisode {
    fn from_atom(atom: &EpisodeAtom, activation_ppm: u32) -> Self {
        Self {
            sequence: atom.id().sequence(),
            activation_ppm,
            episode: EpisodeDocument::from(atom),
        }
    }
}

impl From<&EpisodeAtom> for EpisodeDocument {
    fn from(atom: &EpisodeAtom) -> Self {
        Self {
            occurred_at_ms: atom.occurred_at().get(),
            recorded_at_ms: atom.recorded_at().get(),
            source_id: atom.source().get(),
            context: atom.context().iter().map(StatementDocument::from).collect(),
            observation: StatementDocument::from(atom.observation()),
            action: atom.action().map(StatementDocument::from),
            outcome: atom.outcome().map(StatementDocument::from),
        }
    }
}

impl From<&Statement> for StatementDocument {
    fn from(statement: &Statement) -> Self {
        Self {
            predicate_id: statement.predicate().get(),
            term_ids: statement
                .arguments()
                .iter()
                .map(|term| term.get())
                .collect(),
        }
    }
}
