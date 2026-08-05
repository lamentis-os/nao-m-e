# NAO-M-E

[![CI](https://github.com/lamentis-os/nao-m-e/actions/workflows/ci.yml/badge.svg)](https://github.com/lamentis-os/nao-m-e/actions/workflows/ci.yml)

NAO-M-E is a deterministic Rust kernel for symbolic episode memory. The core
operates entirely in memory; an optional SQLite adapter provides explicit,
transactional snapshots with integrated semantic cue vectors, and a small CLI
exposes the normal add, recall, and feedback workflow. It is a research
mechanism rather than an LLM memory product.

## Model

- An episode is an immutable signed Unix timestamp in milliseconds plus one or
  more set-valued symbolic attributes.
- Attribute keys and values share one numeric `SymbolId` type. The core stores
  only IDs and has no runtime dependencies.
- SQLite owns one normalized, append-only text catalog and stores compact IDs in
  episode payloads.
- SQLite stores one fixed-width multilingual-E5 vector for each distinct bound
  attribute cue `(key, value)` and reuses it across episodes.
- Source-conditioned recall combines ordinary Jaccard overlap over symbolic
  cues with sparse directed feedback histories.
- Each feedback relationship retains at most 16 helpful or unhelpful samples.
- Caller-supplied memory IDs namespace the local atom sequences allocated by a
  memory.

An attribute is an unordered set-valued mapping, not a tuple or ordered
relation. Repeated keys are merged and repeated values are removed. If value
order or pairwise roles matter, use distinct attribute keys or separate
episodes.

## Core quick start

```rust
use std::error::Error;

use nao_m_e::{Attribute, EpisodeDraft, Memory, MemoryId, SymbolId, TimestampMs};

fn main() -> Result<(), Box<dyn Error>> {
    let episode = EpisodeDraft::new(
        TimestampMs::new(1_000),
        vec![Attribute::new(
            SymbolId::new(1),
            vec![SymbolId::new(10), SymbolId::new(11)],
        )?],
    )?;
    let memory_id = MemoryId::new(0x7b4f_6be0_32c2_4be8_96b8_7394_f734_85af)?;
    let mut memory = Memory::new(memory_id);
    let source = memory.insert_episode(episode.clone())?;
    let target = memory.insert_episode(episode)?;

    let cold_hits = memory.recall_from(source, 1)?;
    assert_eq!(cold_hits[0].atom_id, target);
    assert_eq!(cold_hits[0].activation.as_ppm(), 400_000);

    memory.apply_feedback(source, &[target], true)?;
    let learned_hits = memory.recall_from(source, 1)?;
    assert_eq!(learned_hits[0].activation.as_ppm(), 471_875);
    Ok(())
}
```

## SQLite snapshots

`nao-m-e-sqlite` owns one logical memory per database. Symbols are normalized
and staged through the adapter, then published with episodes and feedback by
one explicit `save()`:

```rust
use std::error::Error;
use std::path::Path;

use nao_m_e::{Attribute, EpisodeDraft, TimestampMs};
use nao_m_e_sqlite::SqliteStore;

fn main() -> Result<(), Box<dyn Error>> {
    let path = Path::new("memory.sqlite3"); // Must not exist on create.
    let mut store = SqliteStore::create(path)?;
    let symbols = store.intern_symbols(&[
        "repository".to_owned(),
        "nao-m-e".to_owned(),
    ])?;
    let episode = EpisodeDraft::new(
        TimestampMs::new(1_000),
        vec![Attribute::new(symbols[0], vec![symbols[1]])?],
    )?;
    let source = store.memory_mut().insert_episode(episode.clone())?;
    let target = store.memory_mut().insert_episode(episode)?;
    store.save()?;

    let memory_id = store.memory_id();
    drop(store);

    let reopened = SqliteStore::open(path)?;
    assert_eq!(reopened.memory_id(), memory_id);
    let hits = reopened.memory().recall_from(source, 1)?;
    assert_eq!(hits[0].atom_id, target);
    assert_eq!(hits[0].activation.as_ppm(), 400_000);
    Ok(())
}
```

`save()` appends only new episodes and reconciles feedback changes without
rewriting equal rows on the bounded-delta path. New symbols remain staged until
the same transaction publishes them with the episodes that reference their
IDs. `open()` validates and reconstructs the complete core snapshot before
exposing memory state, using bounded operational semantic checks;
`check()` performs the exhaustive vector and posting audit. The symbolic-recall
posting cache and scores are still derived in memory and are not persisted.
Integrated semantic postings are separate V7 projection data and current core
recall does not read them.

## Integrated semantic cues

SQLite format V7 stores semantic vectors and their episode postings in the same
database as symbols, episodes, and feedback. Each distinct normalized bound cue
`(attribute key, value)` is embedded once and reused by every episode containing
that pair. The fixed profile is
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
with 384 signed 16-bit coordinates; model revision, tokenizer, projection,
pooling, normalization, and quantization are part of the format contract.
There is no locale setting, language detection, translation, or localized
execution path. Text from every script passes through the same fixed profile.

The embedding runtime is lazy. The first `add` that introduces a new cue
downloads and verifies the pinned model assets when they are not already in the
local cache. That operation therefore needs network access and can be much
slower than later additions. A missing network, corrupt download, model error,
or invalid vector aborts the complete save. Reused cues, `init`, `check`,
`recall --from`, and `feedback` do not load the model.

The vectors are an integrated, validated projection of authoritative normalized
text. They do not change symbolic core scoring, and this version deliberately
does not provide `recall <TEXT>` or any other semantic-text query. See the
[semantic cue contract](docs/semantic-contract.md) for the exact model and
failure boundary.

## Command-line interface

Install the workspace binary from a checkout:

```sh
cargo install --locked --path crates/nao-m-e-cli
```

The complete command surface is:

```text
nao-m-e init <DATABASE>
nao-m-e check <DATABASE>
nao-m-e add <DATABASE> [--quiet] [--timestamp <UNIX_MS>]
  --attribute <TEXT> --value <TEXT>...
  [--attribute <TEXT> --value <TEXT>...]...
nao-m-e add <DATABASE> --many [--quiet]
nao-m-e recall <DATABASE> --from <SEQUENCE> [--limit <N>]
nao-m-e feedback <DATABASE> --from <SEQUENCE> --helpful <SEQUENCE,...>
nao-m-e feedback <DATABASE> --from <SEQUENCE> --unhelpful <SEQUENCE,...>
```

`UNIX_MS` is a signed decimal count of milliseconds since
`1970-01-01T00:00:00Z`. Omitting it records the current system time. All missing
timestamps within one command receive the same value. Unix time is independent
of local time zones and daylight-saving transitions, but system time is not
guaranteed to be monotonic; the episode sequence is the canonical insertion
order.

Each `--attribute` requires at least one immediately following `--value`.
Attributes and values are text; quote shell values containing spaces. Text that
looks numeric or option-like remains text when supplied as the required
argument, for example `--attribute --quiet --value --many`.

Create a store and add two episodes:

```sh
nao-m-e init incident-memory.sqlite3

nao-m-e add incident-memory.sqlite3 \
  --timestamp 1785596400000 \
  --attribute repository \
  --value nao-m-e \
  --attribute "build status" \
  --value failed
# stdout: 0

nao-m-e add incident-memory.sqlite3 \
  --timestamp 1785596460000 \
  --attribute repository \
  --value nao-m-e \
  --attribute "incident status" \
  --value open
# stdout: 1
```

Successful `init`, `check`, and feedback are silent. `check` performs the full
read-only database validation without loading the embedding model. `add` prints
the assigned local sequence followed by a newline; `--quiet` suppresses it
without changing the saved state.

`add --many` reads one shell-quoted episode flag row per non-empty standard-input
line. It performs quoting and escaping but no variable, command, or glob
expansion; an unquoted `#` begins a comment. The complete input is parsed before
the store is opened, then all symbols and episodes are committed by exactly one
save or none are committed:

```sh
printf '%s\n' \
  '--timestamp 1785596520000 --attribute "build status" --value queued' \
  '--attribute "build status" --value running --attribute command --value "cargo test"' \
  | nao-m-e add incident-memory.sqlite3 --many
# stdout:
# 2
# 3
```

Recall is read-only and defaults to ten hits. Every hit uses this deterministic
line grammar, with one blank line between hit blocks:

```sh
nao-m-e recall incident-memory.sqlite3 --from 0 --limit 5
```

```text
sequence 1
activation_ppm 133333
timestamp 1785596460000
attribute repository
value nao-m-e
attribute incident status
value open
```

The two episodes share the three cues produced by `repository = {nao-m-e}`.
Each also has three distinct cues, so ordinary Jaccard projection is
`floor(3 * 400000 / 9) = 133333` ppm. Attribute order is canonical numeric-ID
order, not input or lexical order. A recall with no hits writes zero bytes and
does not save or mutate the database.

One helpful assessment adds the first learned contribution of `71,875 ppm`:

```sh
nao-m-e feedback incident-memory.sqlite3 --from 0 --helpful 1
nao-m-e recall incident-memory.sqlite3 --from 0 --limit 5
```

The hit content is unchanged and its score becomes `205208`. Repeated helpful
or unhelpful feedback updates the bounded 16-sample history. Positive learned
feedback can introduce a target with no shared cues; negative feedback can
suppress a structural match. Feedback changes retrieval accessibility, not
truth or confidence, and never runs recall implicitly.

Errors detected before a command commits leave standard output empty. Exit code
`2` denotes invalid CLI syntax; model, store, input, and output failures use exit
code `1`. Add sequences are emitted after the save commits, so a later output
failure cannot roll back the append. Add and feedback are not idempotent; callers
must resolve an unknown completion state before retrying.

## Boundaries

- The core performs no I/O, networking, random recall, embeddings, or LLM calls.
- SQLite format V7 has no automatic or heuristic migration from V1-V6.
- Semantic cue vectors are integrated into V7. There is no secondary database,
  synchronization step, caller-defined embedding profile, or repair path.
- Symbol normalization applies NFKC, Unicode lowercase, NFKC again, and Unicode
  whitespace collapse. Only the normalized value is retained.
- There is no stemming, fuzzy matching, aliasing, synonym inference, or episode
  deduplication.
- Every CLI process reconstructs the complete snapshot. The adapter opens
  read-write even for logically read-only recall.
- Recall is one-hop and source-conditioned. There is no persistent activation
  vector or global activation ranking.
- There is no text-query or semantic-vector contribution to recall in this
  version; the integrated vectors establish the durable cue foundation only.
- Same-platform encoding is byte-repeatable. Independently generated vectors
  across supported platforms may differ by at most one `i16` bin per component
  at cosine similarity `>= 0.999999`; universal byte equality is not promised.
- Concurrent or stale writers are rejected rather than merged.

See the [core contract](docs/core-contract.md), the
[SQLite contract](docs/sqlite-contract.md), and the
[semantic cue contract](docs/semantic-contract.md) for exact
semantics. Generate local API documentation with:

```sh
cargo doc --workspace --no-deps --all-features --locked --open
```

## Development

Run the primary local test suite with:

```sh
cargo test --workspace --all-targets --all-features --locked --no-fail-fast
```

Repository rules and the complete gate list are in `AGENTS.md`.

## License

Licensed under the [Apache License 2.0](LICENSE).
