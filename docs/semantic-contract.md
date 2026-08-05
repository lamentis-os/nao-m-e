# Semantic cue contract

This document defines the fixed semantic projection integrated into the SQLite
V7 database described by the [SQLite contract](sqlite-contract.md). The
in-memory state and observable recall behavior remain defined by the
[core contract](core-contract.md).

Semantic rows are required persisted V7 state, but normalized symbol text and
episodes remain authoritative. The core does not ingest vectors or postings,
and semantic data does not change source-conditioned recall, feedback, scores,
or ordering in this version.

## Bound cue projection

Every distinct attribute binding with key `k` and value `v` produces exactly
one semantic cue `(k, v)`. Every episode containing that binding has exactly one
posting to the cue. Repeated use of the same binding reuses its vector; using
the same value under a different key produces a different cue.

Timestamps, key-only symbols, feedback traces, and whole episodes do not produce
semantic vectors. Core canonicalization has already merged repeated attributes
and removed repeated values, so one episode cannot produce a duplicate posting.

The encoder receives the normalized key and value text resolved from the V7
symbol catalog. Text is not copied into semantic rows. Cue identifiers are local
unsigned 64-bit values beginning at zero and forming a gapless, append-only
prefix. They are never deleted, reused, renamed, or rebound.

## Fixed embedding profile

V7 has exactly one built-in embedding profile:

- model: `intfloat/multilingual-e5-small`;
- model revision: `0e60b8d9d2166d80387f86e3b48ec9ced55f4d15`;
- ONNX artifact: `onnx/model.onnx`, SHA-256
  `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665`;
- tokenizer artifact: `onnx/tokenizer.json`, SHA-256
  `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`;
- tokenizer runtime: `tokenizers` 0.23.1;
- model input: `passage: {normalized key}: {normalized value}`;
- special tokens enabled; maximum input length 512 tokens, with longest-first
  right truncation and zero stride, batch-longest right padding with no
  multiple, `<pad>` token ID `1`, and padding type ID `0`;
- ONNX Runtime 1.28 through `ort` 2.0.0-rc.13, exclusively bound to the unique
  CPU device of `CPUExecutionProvider`, level-three graph optimization, sequential
  execution, one intra-op and one inter-op thread, deterministic compute
  enabled, memory patterns and the CPU arena disabled, environment execution
  providers ignored, and one cue per model invocation;
- `input_ids`, `attention_mask`, and `token_type_ids` as signed 64-bit tensors,
  with `last_hidden_state` read as 32-bit floating point output;
- masked token vectors accumulated left-to-right in `f64`, divided by the
  unmasked token count, then L2-normalized in `f64`; and
- 384 signed 16-bit coordinates.

There is no locale selection, language detection, translation, or
language-specific execution path. Every normalized Unicode input passes through
the same fixed projection, tokenizer, model, pooling, and quantization pipeline.
Cross-script fixtures only verify that this single path remains stable; they do
not select localized behavior.

The profile fingerprint stored in `memory_meta` is SHA-256
`1297d8d28c7dbe9c624a13a0afa6bf8aa6eb7a43c3235359089ceed7df1f5b25`.
It binds these artifacts and the complete preprocessing, pooling,
normalization, and quantization pipeline. It is a file-format constant, not
caller configuration. Opening a database with a different fingerprint fails
closed.

The text rendering above is model input, not a stored identifier or reversible
serialization. Key and value IDs remain structurally separate in
`semantic_cues`, so punctuation resembling the separator cannot change cue
identity.

Each coordinate is persisted as signed two's-complement `i16` in little-endian
order. Quantization multiplies each normalized component by `32,767`, applies
Rust `f64::round()` (halfway values away from zero), clamps to
`[-32,767, 32,767]`, and converts to `i16`. A vector is therefore exactly 768
bytes and must contain at least one non-zero coordinate. The database contains
no floating-point vector, norm, model output, token sequence, confidence value,
or semantic label.

## Determinism boundary

For one supported platform, fixed artifacts and profile settings must reproduce
the exact same quantized vector bytes for the same canonical cue text. Every
cue uses the same singleton model-input shape, so request grouping, order, and
outer batches of up to 32 cues cannot change its tensor shape. This is the
strict repeatability boundary used by same-platform contract tests. The V7
release targets are Linux x86_64, macOS arm64, and Windows x86_64.

ONNX floating-point kernels are not specified to be byte-identical across CPU
architectures and operating systems. Cross-platform qualification therefore
accepts independently generated vectors only when every corresponding `i16`
coordinate differs by at most one quantization bin and their cosine similarity
is at least `0.999999`. Universal text-to-vector byte equality is not promised.

Persisted vector bytes remain unchanged when a database moves between
platforms. Core behavior and current recall stay deterministic for that fixed
database because neither regenerates nor scores semantic vectors. Any future
semantic score fusion must define its own observable cross-platform ordering
and near-tie behavior before it can become part of recall.

## Model acquisition and runtime

The encoder and model are initialized lazily. Creating or opening a store,
running the full integrity check, resolving text, recalling from an episode, and
applying feedback do not resolve model assets. A save also avoids the runtime
when every cue needed by its new episodes already exists.

When the first missing cue must be encoded, the runtime resolves the pinned
artifacts through its local Hugging Face cache. Cache discovery uses
`HF_HUB_CACHE` when set, otherwise `$HF_HOME/hub`, otherwise
`~/.cache/huggingface/hub`. Missing artifacts trigger a download from the fixed
model revision. Each artifact is hash-checked before model loading. A hash
mismatch triggers exactly one forced download and one further check, after which
the operation fails. There is no unpinned model selection, caller-supplied
profile, approximate substitute, or silent symbolic fallback.

Consequently, the first addition of a new cue can require network access and a
large download. Offline operation succeeds only when the verified artifacts are
already cached. Runtime, network, tokenizer, model, non-finite output, wrong
dimension, all-zero vector, and quantization failures reject the save before
any database transaction begins.

Encoding preserves input order and accepts at most 32 cues per call. An empty
call remains offline and does not load the runtime. The encoder returns exactly
one canonical vector per requested cue. A missing, extra, wrongly profiled,
wrongly dimensioned, or all-zero result rejects the complete operation. The
store splits larger missing-cue sets into ordered calls of at most 32 and
validates every result before adding any cue to its pending state.

## Integrated persistence

`memory_meta` stores the fixed non-zero profile fingerprint and the canonical
big-endian semantic cue count. `semantic_cues` stores the gapless cue ID, bound
key and value IDs, and one fixed-width vector. `(key_id, value_id)` is unique and
both IDs reference the shared symbol catalog. `episode_cues` stores the exact
many-to-many posting relation and references existing episodes and cues. The
reverse `(cue_id, sequence)` index is persisted for future candidate traversal.

The complete DDL and scalar encodings are part of the closed SQLite V7 schema.
There is no second database, synchronization watermark, independent format
version, cross-file identity, or separately writable semantic state.

## Save lifecycle and atomicity

Before beginning its SQLite write transaction, `save()` validates all new
episode symbol references and derives their bound pairs in deterministic order.
It resolves existing cues with bounded indexed queries and embeds only missing
pairs. New normalized symbol text staged in the same store is eligible for this
projection.

Only after every required vector has been produced and validated does the store
begin its existing immediate transaction. It rechecks memory identity, expected
revision, episode and symbol tails, semantic cue count and tail, and the closed
schema. The transaction then advances the revision and cue count, inserts staged
symbols and new cues, appends episodes and exact postings, reconciles feedback,
and commits them together.

No network or model inference occurs while the SQLite write transaction is
held. Any error before commit leaves the previous database byte state committed;
no symbol, cue, vector, episode, posting, feedback change, count, or revision is
partially published. A stale writer is rejected rather than merging its locally
prepared semantic state.

## Two validation tiers

`SqliteStore::open()` is the operational path. It checks file identity, V7
metadata and fixed profile, durability policy, exact schema, targeted SQLite
quick checks for the authoritative tables, symbol and episode reconstruction,
feedback restoration, and agreement between the semantic cue count and catalog
tail. Semantic cue rows returned by an operational lookup are validated before
use. This path deliberately does not scan every vector and posting.

`SqliteStore::check()` is the exhaustive audit path. It opens the file
physically read-only, performs SQLite `integrity_check` and `foreign_key_check`,
reconstructs the same authoritative state, then scans the complete semantic
catalog and postings. It requires:

- cue IDs to be the exact prefix `0..semantic_cue_count`;
- every bound key and value to resolve and every pair to be unique;
- every vector to have the canonical width and a non-zero coordinate;
- every episode's postings to equal its attribute bindings exactly; and
- the global cue catalog to contain neither an unused nor a missing bound pair.

The exhaustive audit never resolves, downloads, or loads the model and never
changes connection-persistent settings, revision, rows, or file contents. No
partially validated store is returned by either tier.

## CLI boundary and non-goals

`nao-m-e check <DATABASE>` exposes the exhaustive tier. Success is silent;
invalid data is a runtime failure with empty standard output. The existing
`init`, `add`, `recall --from`, and `feedback` grammar and successful output stay
unchanged. New cues are embedded as part of the existing atomic add save.

V7 deliberately does not provide:

- `recall <TEXT>` or any other semantic-text query;
- vector similarity search, approximate nearest neighbors, or score fusion;
- whole-episode, timestamp, key-only, or feedback embeddings;
- multiple models, profiles, dimensions, or caller-provided vectors;
- background indexing, automatic repair, migration, or garbage collection; or
- evidence that the fixed model improves retrieval quality or is semantically
  correct for a particular language, domain, or task.

Those retrieval and evaluation questions require separate contracts and
evidence. Persisting vectors now establishes the atomic, deduplicated cue
foundation without changing current recall behavior.
