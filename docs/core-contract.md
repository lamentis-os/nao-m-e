# Core contract

This document defines the cross-cutting, observable semantics of the memory
kernel. Symbol-specific behavior remains documented in the Rust API.

## State

The memory state is:

```text
M = (A, F)
```

- `A` is an append-only sequence of immutable episode atoms.
- `F` is a sparse matrix of directed, bounded feedback traces.

Feedback is mutable state separate from episode content. Inserting identical
episode content creates distinct atoms. Implementations may maintain a private
cue index derived completely from `A`; that index is not additional logical
state and is rebuilt by replaying the atom sequence.

Normalized symbol text, persisted semantic cue vectors, and their episode
postings remain adapter-owned projections outside `M`. The core neither creates
nor reads them. They do not change the scoring or ordering defined by this
contract.

## Symbolic episodes

An episode consists of one timestamp and a non-empty set of symbolic
attributes. A caller-owned `SymbolId` can identify either an attribute key or
an attribute value. The kernel does not interpret these identifiers or assign
meaning to particular keys.

An attribute contains one key and a non-empty, unordered set of values.
Construction sorts and deduplicates each value set. Episode construction sorts
attributes by key and merges repeated keys by set union. The canonical episode
therefore has strictly increasing keys, and each attribute has strictly
increasing values. Attribute or value input order is not retained, and there is
no positional meaning within a value set.

The timestamp is a signed number of milliseconds since the Unix epoch,
`1970-01-01T00:00:00Z`. It does not drive recall or feedback, establish
insertion order, or require the kernel to read a clock.

Recall derives these unit-weight cues from every attribute with key `k` and
each value `v`:

```text
Key(k)
Value(v)
KeyValue(k, v)
```

Each distinct cue occurs at most once in an episode's cue set. Reusing a value
under different keys can match through `Value(v)`, while `KeyValue(k, v)`
preserves the exact binding. The cue variants remain distinct, so an identifier
used as a key does not match the same identifier used only as a value. The
timestamp does not produce cues.

## Identity and membership

A `MemoryId` is a caller-allocated, non-zero 128-bit value. Its canonical
representation is exactly 16 bytes in big-endian order. Its diagnostic display
is exactly 32 lowercase hexadecimal digits and is not a storage format.

An `AtomId` is the pair `(MemoryId, sequence)`, where `sequence` is an unsigned
64-bit insertion sequence. A new memory starts at sequence zero. Successful
insertions advance the sequence, and sequences are not reused within that
memory instance. Ordering compares `MemoryId` first and sequence second. The
diagnostic display is `<memory-id>:<sequence>` and is not a database or wire
format; the core defines no combined byte representation for `AtomId`.

Constructing an `AtomId` does not prove that the atom exists. Read operations
return `None` for a foreign memory identifier or absent local sequence. Recall
and feedback mutations reject those identifiers with `GraphError::UnknownAtom`.

The caller must assign one `MemoryId` to one logical memory. Reopening that
memory requires the same identifier and reconstruction of its complete atom
sequence in the original order before appending. Independent writers under one
identifier are unsupported because they can allocate the same sequence. IDs
are references, not permissions, content hashes, or evidence about an episode.

## Feedback graph

An existing directed edge contains one `FeedbackTrace`. A trace represents
between one and 16 recent binary assessments:

```text
1 = helpful
0 = unhelpful
```

`history_bits` is an unsigned 16-bit value. Bit zero is the newest represented
sample and bit `sample_count - 1` is the oldest. `sample_count` is in `1..=16`,
and every bit at or above `sample_count` must be zero. Thus an all-unhelpful
trace is valid even though its bit value is zero. There is no empty trace.

`FeedbackTrace::from_parts(history_bits, sample_count)` accepts exactly these
canonical representations. Its getters return the bits, sample count, and the
helpful and unhelpful sample counts. Helpful count is population count;
unhelpful count is `sample_count - helpful_count`.

Edges are directed, and self-edges are rejected. `set_feedback_trace` validates
both endpoints before inserting or replacing a trace and returns the previous
trace when one existed. There is no regular removal operation: negative and
balanced traces are retained because they remain learned state.
`feedback_trace` returns `None` for an absent edge or unknown endpoint.
`feedback_edges` visits edges by ascending source `AtomId` and then ascending
target `AtomId`.

Feedback traces encode source-conditioned accessibility learned from explicit
assessments. They do not encode truth, probability, semantic confidence,
causality, or factual support. Traces are independent per edge; there is no
outgoing source budget and changing one target does not redistribute another
target's history.

## External feedback

`apply_feedback(source, targets, helpful)` appends one caller-supplied binary
assessment to every effective directed `source -> target` trace. The kernel
accepts the assessment as input; it does not infer whether a recall was useful.

The operation first validates the source. A supplied target list longer than
`MAX_FEEDBACK_TARGETS = 10,000` is rejected without mutation. Every entry of an
accepted list is then validated; any unknown atom rejects the complete
operation. Targets are sorted and deduplicated and the source itself is removed.
Target order, recall rank, duplicates, and self-hits therefore have no effect.
An empty effective target set is a successful no-op.

For feedback bit `x`, adding a sample updates each effective trace as follows:

```text
history_bits' = (history_bits << 1) | x
sample_count' = min(sample_count + 1, 16)
```

The operation is on a 16-bit value, so when the trace is full the shift drops
exactly bit 15, its oldest sample. A previously absent edge is created with
`history_bits = x` and `sample_count = 1`. Every effective target receives one
complete sample regardless of batch size.

All identifiers and the raw target-count limit are validated before any trace
is mutated. Every returned error therefore leaves the complete feedback graph
unchanged. Feedback does not change immutable episode content.

## Source-conditioned recall

Recall scores use integer parts per million with these fixed parameters:

```text
SCALE                       = 1,000,000
STRUCTURAL_GAIN_PPM         =   400,000
LEARNED_GAIN_PPM            =   400,000
FEEDBACK_HISTORY_CAPACITY   =        16
FEEDBACK_PRIOR_MASS         =         7
```

`recall_from(source, limit)` performs a read-only, source-conditioned query. It
validates `source` even when `limit` is zero. Its candidates are the union of
all other episodes sharing at least one cue with the source and every target in
the source's direct outgoing feedback row. The first successful recall with a
positive limit may build the derived index by scanning all atoms once. Warm
candidate lookup in that `Memory` then traverses only the source cues and
their postings plus its direct feedback row rather than globally scanning
episodes. A target present through several cues or both paths occurs only once.

For source cue set `C_s` and target cue set `C_t`, intersection and union are:

```text
intersection = |C_s intersect C_t|
union        = |C_s union C_t|
```

The structural contribution is ordinary Jaccard similarity with integer floor:

```text
structural = floor(intersection * STRUCTURAL_GAIN_PPM / union)
```

For a direct trace with sample count `m`, helpful count `h`, and signed balance
`d = 2h - m`, the learned contribution is:

```text
learned = trunc_toward_zero(
    LEARNED_GAIN_PPM * d *
        (FEEDBACK_HISTORY_CAPACITY + FEEDBACK_PRIOR_MASS)
    ---------------------------------------------------
    FEEDBACK_HISTORY_CAPACITY * (m + FEEDBACK_PRIOR_MASS)
)
```

An absent edge contributes learned zero. Signed integer division truncates
toward zero, not toward negative infinity. With the fixed constants this is:

```text
learned = trunc_toward_zero(400000 * d * 23 / (16 * (m + 7)))
score   = clamp(structural + learned, 0, 800000)
```

One, two, three, four, eight, and sixteen consistently helpful samples project
to `71,875`, `127,777`, `172,500`, `209,090`, `306,666`, and `400,000` ppm.
Starting from sixteen helpful samples, one, eight, nine, and sixteen unhelpful
samples project to `350,000`, `0`, `-50,000`, and `-400,000` ppm because the
bounded history progressively replaces the previous samples.

A learned-only candidate has structural zero and is returned only when its
learned contribution is positive. Negative and balanced traces remain stored
but do not independently produce hits. For a structural candidate, negative
learned feedback can lower or fully suppress the combined score. Each
contribution has magnitude at most 400,000 ppm, and the clamped score is in
`0..=800,000` ppm.

The source itself and targets whose final score is zero are excluded. At most
`limit` hits are returned by descending score and then ascending `AtomId`. The
memory stores no activation vector. Recall does not mutate logical atom or
feedback state, but may initialize and reuse its private derived cue cache.
Incoming edges, other source rows, and multi-hop paths do not contribute.

## Kernel boundary

The kernel's logical state contains atoms and bounded feedback traces only.
`Memory` may also hold cue postings and cue counts derived
deterministically from immutable atoms; they are neither independently mutable
nor persistent. The kernel performs no persistence, loading, synchronization,
or multi-writer coordination. It also performs no free-text processing,
embedding or LLM calls, time-driven decay, or autonomous learning. Feedback
changes only through explicit trace reconstruction or caller-supplied binary
assessments. Persistence adapters remain outside the kernel; the format and
lifecycle of the optional SQLite adapter are specified separately in the
[SQLite contract](sqlite-contract.md).
