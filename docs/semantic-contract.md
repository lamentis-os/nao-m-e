# Semantic episode contract

This document defines the fixed semantic projection and free-text retrieval
integrated into the SQLite V8 database described by the
[SQLite contract](sqlite-contract.md). The dependency-free kernel and its
source-conditioned symbolic recall remain defined by the
[core contract](core-contract.md).

Every committed episode has exactly one mandatory semantic vector. An add is
successful only when its normalized symbols, episode, and vector are published
by the same SQLite transaction. There is no searchable-later fallback, optional
semantic cache, sidecar, or independently writable index.

## Episode and query projection

For an episode, every bound normalized attribute pair `(key, value)` becomes
one line:

```text
passage: key: value
other key: other value
```

Pairs are sorted lexically and deduplicated before rendering. Symbol allocation,
attribute input order, and repeated values therefore cannot change the passage.
The timestamp and feedback traces are excluded. Equal episode content is encoded
for every distinct episode because the persisted vector is owned by its episode
sequence rather than shared by text identity.

A normalized free-text query is rendered as:

```text
query: normalized query
```

Both inputs use the SQLite symbol normalizer: Unicode 16 NFKC, locale-independent
lowercase mapping, NFKC again, Unicode whitespace collapse, control-character
rejection, and a 4,096-byte normalized UTF-8 limit. The model path is
language-agnostic: there is no language detection, routing, translation, locale
selection, or language-specific scoring branch.

## Fixed embedding profile

V8 has one built-in profile:

- model: `intfloat/multilingual-e5-small`;
- revision: `0e60b8d9d2166d80387f86e3b48ec9ced55f4d15`;
- FP32 ONNX artifact: `onnx/model.onnx`, SHA-256
  `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665`;
- tokenizer artifact: `onnx/tokenizer.json`, SHA-256
  `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`;
- tokenizer runtime: `tokenizers` 0.23.1, special tokens enabled, maximum
  length 512, longest-first right truncation with zero stride, and no padding;
- ONNX Runtime 1.28 through `ort` 2.0.0-rc.13, the unique CPU device from
  `CPUExecutionProvider`, level-three graph optimization, sequential execution,
  one intra-op and inter-op thread, deterministic compute enabled, memory
  patterns and the CPU arena disabled, and environment providers ignored;
- signed 64-bit `input_ids`, `attention_mask`, and `token_type_ids`, with
  `last_hidden_state` read as FP32;
- attention-masked token vectors accumulated left-to-right in `f64`, divided by
  the unmasked count, then L2-normalized in `f64`; and
- 384 signed 16-bit coordinates.

Queries are right-truncated at the fixed context boundary. Episodes are rejected
instead of truncated because silently removing attributes would make committed
semantic state incomplete.

The V8 profile fingerprint is SHA-256
`79a5437c9d41cb7022451c2c3bc4708c769e7a073506c5d368c2f27e258fe18b`.
It binds the artifacts and complete projection, tokenizer, runtime, pooling,
normalization, and quantization policy. It is a file-format constant, not caller
configuration. A different fingerprint is rejected.

Each normalized component is multiplied by `32,767`, rounded with halfway
values away from zero, clamped to `[-32,767, 32,767]`, and persisted as
little-endian two's-complement `i16`. A vector is exactly 768 bytes, contains no
`i16::MIN`, and has at least one non-zero component. No floating-point vector,
tokens, norm, model output, confidence, or semantic label is stored.

For fixed artifacts and runtime on one platform, repeated encoding must produce
identical quantized bytes. Persisted bytes remain stable
when a database moves. This contract does not promise byte-identical fresh
inference across operating systems or CPU architectures; observable recall over
a particular persisted database is nevertheless integer-only and deterministic.

## Installation and runtime boundary

The pinned model and tokenizer are product installation prerequisites. Runtime
commands never download or repair them. Cache discovery follows the Hugging Face
cache rules (`HF_HUB_CACHE`, otherwise `$HF_HOME/hub`, otherwise the default user
cache), then requires both exact revision paths and SHA-256 values.

Constructing an encoder and running `init` or `check` perform no model I/O. The
first episode encoding or positive-limit query against a non-empty store in a
process verifies the cached artifacts, loads one lazy ONNX session, and performs
inference. Missing, unreadable, hash-mismatched, unusable, or invalid artifacts
fail clearly. Provisioning belongs to product packaging or release setup; this
repository does not define an installer.

## Atomic save lifecycle

`save()` encodes the complete unprepared episode suffix before its single SQLite
write transaction. An episode and its vector commit together or neither does;
model work never holds the write lock. Preparation is batch-atomic, while fully
prepared vectors survive a later transactional failure for retry. The exact
validation, CAS, insertion order, and post-commit state transition are defined
once in the [SQLite save lifecycle](sqlite-contract.md#save-lifecycle).

## Exact semantic recall

`SqliteStore::recall_semantic(query, limit)` operates only on committed episode
vectors. Pending episodes are invisible. It normalizes the query, rejects
invalid input even when `limit == 0`, and returns immediately for zero limit.
For a positive limit it verifies the opened revision, avoids model work for an
empty store, encodes the query outside a transaction, then opens a read
transaction and rechecks the revision before scanning.

For query vector `q` and episode vector `e`, with 384 signed components:

```text
dot = sum(q[i] * e[i])
denominator = isqrt(sum(q[i]^2) * sum(e[i]^2))
score = 0                                      when dot <= 0
score = min(1_000_000,
            floor(dot * 1_000_000 / denominator)) otherwise
```

All accumulation and ranking are integer-only. Zero scores are omitted. Positive
hits sort by score descending, then complete `AtomId` ascending. A bounded heap
retains at most `limit` results without changing the final order. The scan still
validates every row, including candidates that would score zero.

Recall is an exact `O(E * 384)` scan over `E` committed episodes with
`O(min(E, limit))` ranking memory. There is no ANN index, score threshold,
field weighting, cue pooling, symbolic fusion, or feedback contribution.
The reported `activation_ppm` is the query-local cosine projection, not stored
core activation, confidence, or truth.

## Validation tiers

Operational `open`, exhaustive `check`, and recall-time validation are specified
in the [SQLite contract](sqlite-contract.md#open-lifecycle). In particular,
`open` avoids vector-body scans, while `check` and semantic recall validate the
exact vector prefix and every accessed component without loading the model.

The audit can prove structural and codec integrity, not that valid-looking bytes
were originally produced from the corresponding text. Proving that would require
model-backed re-encoding, which is outside this offline integrity contract.

## CLI boundary and non-goals

The only CLI recall form is:

```text
nao-m-e recall <DATABASE> --query <TEXT> [--limit <N>]
```

It is logically read-only and renders normalized episode text for ranked hits.
Source-conditioned retrieval is a separate programmatic core API,
`Memory::recall_from`. Existing episode-to-episode helpful/unhelpful feedback
remains persisted and affects that core API, but not semantic query ranking.

V8 deliberately provides no approximate search, model selection, quantization
alternative, caller-supplied vector, score fusion, learned semantic ranking,
background indexing, migration, automatic repair, or general retrieval-quality
claim. Curated retrieval fixtures are regression evidence for declared cases,
not proof of production or domain-wide quality.
