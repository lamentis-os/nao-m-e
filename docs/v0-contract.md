# V0 contract

This document defines the cross-cutting, observable semantics of the V0 kernel.
Symbol-specific behavior remains documented in the Rust API.

## State

The memory state is:

```text
M = (A, W)
```

- `A` is an append-only sequence of immutable episode atoms.
- `W` is a sparse matrix of directed, positive relevance weights.

Relevance is mutable state separate from episode content. Inserting identical
episode content creates distinct atoms. Implementations may maintain a private
cue index derived completely from `A`; that index is not additional logical
state and is rebuilt by replaying the atom sequence.

## Symbolic episodes

A statement consists of a caller-owned predicate identifier and one or more
ordered, caller-owned term identifiers. An episode contains occurrence and
recording timestamps, context statements, one observation, an optional action,
an optional outcome, and a caller-owned provenance source identifier.

Insertion sorts and deduplicates the context list by predicate identifier and
then lexicographically by the ordered argument identifiers. It does not reorder
a statement's arguments or otherwise interpret caller-owned identifiers.
Timestamps are signed milliseconds on a caller-defined timeline and do not
drive recall or feedback.

Recall derives a set of symbolic cues from every statement in an episode. The
statement roles are `Context`, `Observation`, `Action`, and `Outcome`. For a
statement with role `r`, predicate `p`, and zero-based ordered arguments `t_i`,
the cues and their fixed weights are:

```text
Predicate(p)                 1
Term(t_i)                    1
RolePredicate(r, p)          2
RoleArgument(r, p, i, t_i)   4
```

Each distinct cue occurs at most once in an episode's cue set, even if several
statements produce it. Predicate and term cues can therefore match across
roles, while role-predicate and role-argument cues preserve their stated role;
role-argument cues additionally preserve predicate and argument position.
Timestamps and provenance source identifiers do not produce cues.

## Identity and membership

A `MemoryId` is a caller-allocated, non-zero 128-bit value. Its canonical
representation is exactly 16 bytes in big-endian order. Its diagnostic display
is exactly 32 lowercase hexadecimal digits and is not a storage format.

An `AtomId` is the pair `(MemoryId, sequence)`, where `sequence` is an unsigned
64-bit insertion sequence. A new memory starts at sequence zero. Successful
insertions advance the sequence, and sequences are not reused within that
memory instance. Ordering compares `MemoryId` first and sequence second. The
diagnostic display is `<memory-id>:<sequence>` and is not a database or wire
format; V0 defines no combined byte representation for `AtomId`.

Constructing an `AtomId` does not prove that the atom exists. Read operations
return `None` for a foreign memory identifier or absent local sequence. Recall
and graph mutations reject those identifiers with `GraphError::UnknownAtom`.

The caller must assign one `MemoryId` to one logical memory. Reopening that
memory requires the same identifier and reconstruction of its complete atom
sequence in the original order before appending. Independent writers under one
identifier are unsupported because they can allocate the same sequence. IDs
are references, not permissions, content hashes, or evidence about an episode.

## Relevance graph

An existing relevance edge has a weight from 1 through `SCALE`; absence
represents no edge. Edges are directed, self-edges are rejected, and the sum of
outgoing weights from one atom cannot exceed `SCALE`. Setting or replacing an
edge validates both endpoints and the resulting budget before mutation.

A positive weight controls source-conditioned accessibility only. It does not
encode truth, probability, confidence, evidence, causality, or semantic
support.

## External feedback

`apply_feedback(source, targets, helpful)` applies one caller-supplied binary
assessment to the source's outgoing relevance. `helpful = true` is positive
feedback and `helpful = false` is negative feedback. The kernel accepts the
assessment as input; it does not infer whether a recall was useful.

The operation first validates the source. A supplied target list longer than
`MAX_FEEDBACK_TARGETS` is rejected without mutation. Every entry of an accepted
list is then validated; any unknown atom rejects the complete operation. The
targets are sorted and deduplicated and the source itself is removed. Target
input order, rank, duplicates, and self-hits therefore have no effect. An empty
effective target set is a successful no-op.

The fixed feedback parameters are:

```text
FEEDBACK_TARGET_STEP_PPM = 1,000
FEEDBACK_MAX_EVENT_PPM = 10,000
MAX_FEEDBACK_TARGETS = 10,000
```

For a non-empty effective target set `T` of size `n`, positive feedback computes:

```text
target_total = sum(weight[source,target] for target in T)
per_target = min(
    FEEDBACK_TARGET_STEP_PPM,
    floor(FEEDBACK_MAX_EVENT_PPM / n),
    floor((SCALE - target_total) / n)
)
total_award = per_target * n
```

Thus a normal list of at most ten effective targets changes each target by
1,000 ppm, while longer lists share at most 10,000 ppm of direct target
adjustment. `FEEDBACK_MAX_EVENT_PPM` bounds that aggregate target award or
reduction; positive feedback can additionally move weight away from non-targets
to fund the award. The entry limit ensures that the event-budget share is at
least one ppm. Remaining outgoing capacity can still make positive feedback a
no-op when it cannot fund an equal one-ppm increase for every effective target.

The award first consumes unused outgoing budget. If the unused budget is less
than `total_award`, the deficit is funded by scaling only non-target edges.
Let `non_target_total` be their old total and `needed` be the deficit.
Non-target edges are visited in ascending target identifier order with
`remainder` initially zero. For each old weight, the proportional reduction,
replacement, and next remainder are:

```text
numerator = weight * needed + remainder
reduction = floor(numerator / non_target_total)
remainder = numerator mod non_target_total
replacement = weight - reduction
```

Carrying the remainder makes the reductions sum to exactly `needed`, so
fragmentation cannot remove additional weight.
Each target weight is then increased by `per_target`. A zero result is
represented by edge absence. Positive feedback therefore moves at most
`FEEDBACK_MAX_EVENT_PPM` into targets and at most that same amount out of
non-targets; the sum of absolute edge changes can include both sides.

Negative feedback computes `per_target = min(FEEDBACK_TARGET_STEP_PPM,
floor(FEEDBACK_MAX_EVENT_PPM / n))` and replaces each target weight with
`max(0, weight - per_target)`. Missing target edges and all non-target edges
remain unchanged.

All identifiers and the target-count limit are validated before relevance is
mutated, so every returned error leaves the complete graph unchanged. Feedback
does not change immutable episode content.

## Source-conditioned recall

Recall scores use integer parts per million with these fixed parameters:

```text
SCALE                  = 1,000,000
STRUCTURAL_GAIN_PPM    =   400,000
PROPAGATION_GAIN_PPM   =   400,000
```

`recall_from(source, limit)` performs a read-only, source-conditioned query. It
validates `source` even when `limit` is zero. Its candidates are the union of
all other episodes sharing at least one cue with the source and all targets in
the source's direct outgoing relevance row. The first successful recall with a
positive limit may build the derived index by scanning all atoms once. Warm
candidate lookup in that `MemoryV0` then traverses only the source cues and
their postings rather than globally scanning episodes. A target present through
several cues or through both paths occurs only once.

For source cue set `C_s` and target cue set `C_t`, let `w(c)` be the fixed cue
weight above. Weighted intersection and union are:

```text
intersection = sum(w(c) for c in C_s intersect C_t)
union        = sum(w(c) for c in C_s union C_t)
```

The structural contribution is weighted Jaccard similarity scaled with integer
flooring. A direct learned relevance edge contributes the existing one-hop
projection; an absent edge contributes zero:

```text
structural[target] = floor(intersection * STRUCTURAL_GAIN_PPM / union)
learned[target]    = floor(weight[source,target] * PROPAGATION_GAIN_PPM / SCALE)
score[target]      = structural[target] + learned[target]
```

For a learned-only candidate, `intersection` and `structural` are zero. Each
contribution is at most 400,000 ppm, so the combined score is at most 800,000
ppm. The source itself and targets whose combined score is zero are excluded.
At most `limit` hits are returned by descending score and then ascending
`AtomId`. The memory stores no activation vector. Recall does not mutate the
logical atom or relevance state, but may initialize and reuse a private derived
cue cache. Incoming edges, other source rows, and multi-hop paths do not
contribute. Unhelpful feedback can reduce or remove only the learned
contribution; it cannot suppress a target's independently derived structural
contribution.

## Kernel boundary

The V0 kernel's logical state contains atoms and relevance only. `MemoryV0` may
also hold cue postings and cue-weight totals derived deterministically from its
immutable atoms; they are neither independently mutable nor persistent. The
kernel performs no persistence, loading, synchronization, or multi-writer
coordination. It also performs no free-text processing, embedding or LLM calls,
or autonomous relevance learning. Relevance changes only through explicit graph
mutation or caller-supplied feedback. Persistence adapters remain outside the
kernel; the format and lifecycle of the optional SQLite adapter are specified
separately in the [SQLite V2 contract](sqlite-v2-contract.md).
