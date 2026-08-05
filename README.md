# NAO-M-E

[![CI](https://github.com/lamentis-os/nao-m-e/actions/workflows/ci.yml/badge.svg)](https://github.com/lamentis-os/nao-m-e/actions/workflows/ci.yml)

NAO-M-E is a deterministic Rust kernel for symbolic episode memory. The core
operates entirely in memory; an optional SQLite adapter provides explicit,
transactional snapshots with one mandatory semantic vector per episode, and a
small CLI exposes add, free-text recall, and feedback. It is a research
mechanism rather than an LLM memory product.

## Model

- An episode is an immutable signed Unix timestamp in milliseconds plus one or
  more set-valued symbolic attributes.
- Attribute keys and values share one numeric `SymbolId` type. The core stores
  only IDs and has no runtime dependencies.
- SQLite owns one normalized, append-only text catalog and exactly one
  fixed-width semantic vector for every committed episode.
- CLI recall ranks committed episode vectors against one free-text query.
  Feedback does not affect this semantic ranking.
- Programmatic `Memory::recall_from` combines ordinary Jaccard overlap over
  symbolic cues with sparse directed feedback histories.
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

## SQLite V8 snapshots

`nao-m-e-sqlite` owns one logical memory per database. Symbols are normalized
and staged through the adapter. `save()` encodes every new episode before
opening a write transaction, then atomically publishes staged symbols, episodes,
their same-sequence vectors, and feedback changes:

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
    store.memory_mut().insert_episode(episode)?;
    store.save()?;

    drop(store);
    assert_eq!(SqliteStore::open(path)?.memory().episodes().len(), 1);
    Ok(())
}
```

A failed preparation or transaction leaves the previous database snapshot
authoritative. `open()` validates and reconstructs the complete core snapshot;
`check()` performs a physically read-only exhaustive vector audit. The core's
symbolic cue postings are derived in memory and are not persisted.

## Semantic episode recall

SQLite V8 builds each episode vector from all normalized bound
`(attribute key, value)` pairs. Equal episode text still receives one vector per
episode sequence. The fixed profile is
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
with 384 signed 16-bit coordinates; its model revision, tokenizer, projection,
pooling, normalization, quantization, and fingerprint are format constants.

The pinned model and tokenizer are installation prerequisites. Runtime commands
use the local Hugging Face cache only and never download or repair assets. The
first episode encoding or positive-limit query against a non-empty store in a
process verifies the exact cached assets and loads the model lazily; `init` and
`check` do not load it. Missing, invalid, or unusable assets fail clearly without
a partial save or symbolic fallback.

The model path is language-agnostic. It performs no language detection, routing,
translation, locale selection, or language-specific scoring. Semantic recall is
an exact scan of committed vectors with positive integer cosine scores, ordered
by score and then atom ID. Pending episodes and feedback do not contribute.

See the [semantic episode contract](docs/semantic-contract.md) for the exact
projection, runtime, scoring, and failure boundaries.

## Command-line interface

Install the workspace binary from a checkout:

```sh
cargo install --locked --path crates/nao-m-e-cli
```

Provision the pinned model and tokenizer in the local Hugging Face cache before
adding an episode or running a positive-limit query against a non-empty store.
This repository does not define an installer.

The complete command surface is:

```text
nao-m-e init <DATABASE>
nao-m-e check <DATABASE>
nao-m-e add <DATABASE> [--quiet] [--timestamp <UNIX_MS>]
  --attribute <TEXT> --value <TEXT>...
  [--attribute <TEXT> --value <TEXT>...]...
nao-m-e add <DATABASE> --many [--quiet]
nao-m-e recall <DATABASE> --query <TEXT> [--limit <N>]
nao-m-e feedback <DATABASE> --from <SEQUENCE> --helpful <SEQUENCE,...>
nao-m-e feedback <DATABASE> --from <SEQUENCE> --unhelpful <SEQUENCE,...>
```

A minimal workflow is:

```sh
nao-m-e init incident-memory.sqlite3

nao-m-e add incident-memory.sqlite3 \
  --timestamp 1785596400000 \
  --attribute repository --value nao-m-e \
  --attribute "build status" --value failed
# stdout: 0

nao-m-e add incident-memory.sqlite3 \
  --timestamp 1785596460000 \
  --attribute repository --value nao-m-e \
  --attribute "incident status" --value open
# stdout: 1

nao-m-e recall incident-memory.sqlite3 \
  --query "open nao-m-e incident" --limit 5

nao-m-e feedback incident-memory.sqlite3 --from 0 --helpful 1
nao-m-e check incident-memory.sqlite3
```

Every singular or batch add calls `save()` exactly once, so all normalized
symbols, episodes, and mandatory vectors commit together or none do. `add
--many` reads one shell-quoted episode flag row per non-empty standard-input
line and parses the complete input before opening the store. `--timestamp` is a
signed decimal Unix-millisecond value; omitting it uses the current system time.

Recall is logically read-only and defaults to ten hits. Hits are ordered by
semantic score descending and then atom ID ascending. Each uses this line
grammar, with one blank line between blocks:

```text
sequence <SEQUENCE>
activation_ppm <POSITIVE_PPM>
timestamp <UNIX_MS>
attribute <NORMALIZED_KEY>
value <NORMALIZED_VALUE>
```

A recall with no hits writes zero bytes. Feedback updates the bounded history
used by programmatic `Memory::recall_from`; CLI semantic recall ignores it.
Successful `init`, `check`, and feedback are silent. Exit code `2` denotes
invalid CLI syntax; model, store, input, and output failures use exit code `1`.

## Boundaries

- The core performs no I/O, networking, random recall, embeddings, or LLM calls.
- SQLite format V8 rejects V1-V7 without automatic or heuristic migration.
- Episode vectors are mandatory V8 rows. There is no secondary semantic index
  or database, synchronization step, caller-defined profile, runtime download,
  or repair path.
- Symbol normalization applies NFKC, Unicode lowercase, NFKC again, and Unicode
  whitespace collapse. Only the normalized value is retained.
- There is no stemming, fuzzy matching, aliasing, synonym inference, episode
  deduplication, symbolic-semantic score fusion, or approximate vector index.
- CLI recall accepts free text only. Programmatic core recall remains one-hop
  and source-conditioned, with explicit feedback and no persistent activation
  vector.
- Concurrent or stale writers are rejected rather than merged.

See the [core contract](docs/core-contract.md), the
[SQLite contract](docs/sqlite-contract.md), and the
[semantic episode contract](docs/semantic-contract.md) for exact semantics.

## Development

Run the primary local test suite with:

```sh
cargo test --workspace --all-targets --all-features --locked --no-fail-fast
```

Repository rules and the complete gate list are in `AGENTS.md`. Generate local
API documentation with:

```sh
cargo doc --workspace --no-deps --all-features --locked --open
```

## License

Licensed under the [Apache License 2.0](LICENSE).
