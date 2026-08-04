# Semantic cue index contract

This document defines the durable format and observable lifecycle of the
optional `nao-m-e-semantic-index` sidecar. The authoritative episode and
feedback state remains the SQLite V6 database defined by the
[SQLite contract](sqlite-contract.md); symbolic recall remains defined by the
[core contract](core-contract.md).

The current semantic sidecar format version is `1`. This is an independent
version namespace. Semantic sidecar V1 neither changes nor supersedes the main
SQLite format version `6`.

## Boundary and logical state

A semantic cue index is a rebuildable projection of one committed main-memory
snapshot under exactly one caller-supplied embedding profile. It contains:

```text
S = (memory identity, embedding profile, indexed episode prefix, cues, postings)
```

It is not part of the logical core state `M = (A, F)` and is never authoritative
for episode content, symbol text, insertion sequence, or feedback. Deleting a
sidecar loses only derived data; it can be recreated from its matching main
database and the same embedding profile.

Every set-valued episode attribute with key `k` and distinct value `v` produces
one bound semantic cue `(k, v)`. A cue is shared by every episode containing the
same pair. The index deliberately does not embed an entire episode and does not
create separate key-only or value-only vectors. Binding the two identifiers
prevents a value reused under different keys from collapsing those meanings.

Episode timestamps and feedback traces produce no semantic cues. Repeated
attributes and values have already been canonicalized by the core and therefore
cannot duplicate an episode-to-cue posting.

The sidecar persists only main-database identifiers, vectors, and their
relationships. Attribute-key and value text remains solely in the main V6
symbol catalog; the sidecar contains no text copy, display value, alias, token,
or model input.

## Embedding profile and caller boundary

`EmbeddingProfile` consists of:

- a non-zero, opaque 32-byte fingerprint; and
- a vector dimension in `1..=65,535`.

The fingerprint is a caller-owned compatibility identity. It must change when
anything that can change vector meaning or bytes changes, including model or
weight revision, tokenizer, bound-cue projection, preprocessing, dimension,
or quantization. The adapter checks only exact bytes and dimension; it cannot
prove that the caller assigned the fingerprint correctly.

`CueEmbedder` receives `CueText` values resolved from the normalized key and
value text in the main database. The two fields remain structurally separate;
the sidecar does not prescribe or persist a delimiter-rendered sentence. The
embedder returns one `Embedding` in the same order for every input cue. Each
embedding contains exactly the profile dimension of signed 16-bit coordinates
and must contain at least one non-zero coordinate.

The caller owns model loading, tokenization, inference, and quantization. The
index performs no networking, model download, floating-point conversion, or
LLM call. A failed batch or a missing, extra, profile-mismatched, wrongly
dimensioned, or all-zero result rejects the complete operation before derived
rows are published. The adapter cannot detect a caller returning otherwise
valid vectors in the wrong cue order; preserving input order is part of the
`CueEmbedder` contract.

Vector coordinates are stored as consecutive signed two's-complement `i16`
values in little-endian byte order. Thus the vector BLOB length is exactly
`2 * dimensions`, between 2 and 131,070 bytes. No norm, similarity metric, or
semantic threshold is defined by this format.

## Adapter lifecycle

The public boundary centers on `SemanticCueIndex`, `EmbeddingProfile`,
`Embedding`, `CueText`, `CueEmbedder`, `IndexStats`, and `IndexError`. Its three
stateful operations have these semantics:

- `create` requires an absent sidecar target, opens the main database through a
  fresh `SqliteStore::open()`, derives and embeds its complete committed episode
  prefix, validates all output, and atomically publishes a complete sidecar.
- `open` opens both files afresh, validates the complete sidecar format and its
  binding to the committed main snapshot and requested profile, and exposes no
  index if any check fails.
- `synchronize` opens the main database afresh, derives only the committed
  episode suffix after the indexed prefix, embeds only previously unseen bound
  cues, and atomically appends cues and postings before advancing the committed
  prefix.

These operations intentionally accept a path to the authoritative snapshot,
not a caller-held `SqliteStore`. Unsaved in-process symbols, episodes, and
feedback therefore never enter the sidecar. A no-op synchronization neither
calls the embedder nor writes the sidecar.

`IndexStats` describes committed index quantities. It is operational metadata,
not a retrieval-quality, coverage-quality, or model-quality measurement.

## Identity and snapshot binding

One sidecar belongs to exactly one main `MemoryId` and one embedding profile.
Opening or synchronizing with a different memory identity, profile fingerprint,
or dimension fails closed.

`indexed_episode_count` is the exclusive end of the append-only episode prefix
represented by the sidecar. It may not exceed the currently committed episode
count. Synchronization advances it only after every cue and posting for the
new suffix has been committed.

The sidecar is not bound to the main snapshot revision. Feedback-only saves and
unreferenced symbol additions can advance that revision without changing the
immutable episode prefix, so they do not invalidate or rewrite semantic cues.
The main V6 adapter validates the complete authoritative snapshot before the
semantic adapter consumes it.

## SQLite identity and schema

The sidecar SQLite application ID is `0x4E414F53` (`NAOS`, decimal
`1312902995`). The only accepted semantic `format_version` is `1`. Other
application IDs, versions, or schema objects are rejected; there is no
migration or heuristic repair.

The closed schema contains these logical objects:

| Object | Purpose |
|---|---|
| `semantic_meta` | Singleton sidecar version, main `MemoryId`, profile fingerprint, dimension, and indexed episode count. |
| `semantic_cues` | Gapless local cue ID, main key `SymbolId`, main value `SymbolId`, and exactly one profile-bound vector. |
| `semantic_cue_pair_unique` | Binary uniqueness of `(key_id, value_id)`. |
| `episode_cues` | Unique posting from a main episode sequence to a local cue ID. |
| `episode_cues_by_cue` | Ordered reverse traversal from one cue to its episode sequences. |

`semantic_meta` has exactly one row. Cue IDs begin at zero and form a gapless,
append-only prefix. A newly encountered pair receives the next cue ID in the
adapter's deterministic traversal order. Existing cue IDs and vectors are never
changed, deleted, or rebound by synchronization.

The posting primary key is `(sequence, cue_id)`. Its foreign key resolves every
cue ID inside the sidecar. SQLite cannot enforce a cross-file foreign key to the
main database, so open-time validation is responsible for main episode and
symbol membership.

Canonical scalar encodings are:

- main `MemoryId`: fixed-width 16-byte big-endian BLOB;
- main `SymbolId`, episode sequence, cue ID, and indexed episode count:
  fixed-width 8-byte big-endian unsigned BLOBs;
- profile fingerprint: exactly 32 opaque bytes; and
- vector coordinates: fixed-width 2-byte little-endian signed values.

Native-endian identifiers, SQLite row IDs, decimal identifiers, variable-width
identifiers, and `AUTOINCREMENT` are not valid format representations.

## Validation and failure atomicity

Before exposing an opened index, the adapter validates at least:

- SQLite application ID, semantic format version, closed schema, singleton
  metadata, connection policy, and canonical scalar widths;
- non-zero memory identity and profile fingerprint plus valid dimension;
- exact requested memory/profile binding and a valid indexed episode prefix;
- gapless cue IDs, unique and resolvable main key/value pairs, exact vector
  byte length, and at least one non-zero vector coordinate;
- unique postings whose episode sequences are inside the indexed prefix and
  whose cue IDs exist; and
- exact agreement between each indexed episode's bound pairs and its persisted
  postings, with no missing or additional relationship.

Malformed, truncated, stale, foreign, or partially synchronized files fail
closed with `IndexError`; no partially reconstructed semantic index is exposed.
Validation is eager rather than lazy: opening reconstructs the authoritative
snapshot and compares the complete posting stream for the indexed prefix. Its
work is therefore linear in the validated main snapshot and semantic postings,
even though no embedding inference runs during open.

Creation uses a private file-backed staging database and publishes it only after
the complete sidecar has been written, validated, closed, and synchronized. It
does not replace an existing target. Synchronization resolves all new cue text
and plans all postings before beginning its write transaction. It then embeds
new cues in bounded batches of at most 256 and writes each batch inside that one
transaction, followed by the postings and new indexed prefix. Cue rows, posting
rows, and metadata therefore commit together or not at all, while vector peak
memory remains bounded. A failed synchronization leaves the previous committed
sidecar usable and the operation retryable.

This choice keeps the derived sidecar write transaction open across caller
embedding calls. It can therefore hold the sidecar's exclusive writer position
for the complete inference duration. It never locks or writes the authoritative
main database, and concurrent sidecar writers fail immediately rather than wait
or observe partial rows.

The sidecar uses immediate lock failure rather than waiting or retrying. It uses
foreign-key enforcement, an untrusted schema, enforced checks, rollback-journal
mode, and the adapter's strongest configured synchronous policy. These settings
describe SQLite transaction and file-boundary behavior; they are not a general
claim of physical power-loss resilience on every filesystem or device.

## Explicit non-goals

Semantic cue index V1 provides only a durable, profile-bound candidate-index
foundation. It does not provide:

- a built-in embedding model, tokenizer, model download, or model registry;
- semantic text input to the CLI or changes to `add`, `recall`, or `feedback`;
- approximate nearest-neighbor search, vector similarity queries, or an ANN
  extension;
- fusion of semantic similarity with symbolic recall or learned feedback;
- whole-episode, timestamp, key-only, or value-only embeddings;
- multiple profiles in one sidecar, background indexing, automatic watching,
  migration, or repair; or
- a claim that the supplied model, vectors, candidates, or eventual retrieval
  are semantically correct or improve retrieval quality.

Those are separate product and evaluation decisions. Their introduction must
preserve the authority of the main database and explicitly define model
provenance, candidate scoring, deterministic ordering, failure behavior, and
quality evidence.
