# V0 contract

This document defines the cross-cutting, observable semantics of the V0 kernel.
Symbol-specific behavior remains documented in the Rust API.

## State

At logical step `t`, a memory has the state:

```text
M(t) = (A, a(t), W)
```

- `A` is an append-only sequence of immutable episode atoms.
- `a(t)` contains one fixed-point activation value per atom.
- `W` is a sparse matrix of directed, positive relevance weights.

Activation and relevance are mutable state separate from episode content.
Inserting identical episode content creates distinct atoms.

## Symbolic episodes

A statement consists of a caller-owned predicate identifier and one or more
ordered, caller-owned term identifiers. An episode contains occurrence and
recording timestamps, context statements, one observation, an optional action,
an optional outcome, and a caller-owned provenance source identifier.

Insertion sorts and deduplicates the context list by predicate identifier and
then lexicographically by the ordered argument identifiers. It does not reorder
a statement's arguments or otherwise interpret caller-owned identifiers.
Timestamps are signed milliseconds on a caller-defined timeline and do not
drive state transitions.

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
return `None` for a foreign memory identifier or absent local sequence.
Stimulation and graph mutations reject those identifiers with
`GraphError::UnknownAtom`.

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

A positive weight controls activation flow only. It does not encode truth,
probability, confidence, evidence, causality, or semantic support.

## Activation dynamics

All unit-interval values use integer parts per million:

```text
SCALE            = 1,000,000
RETENTION        =   500,000
PROPAGATION_GAIN =   400,000
```

External stimulation adds activation and saturates at `SCALE`. One explicit,
synchronous logical step computes every target from the previous activation
vector:

```text
next[j] = floor(
    min(
        SCALE^3,
        RETENTION * SCALE * current[j]
            + PROPAGATION_GAIN * sum(weight[i,j] * current[i])
    ) / SCALE^2
)
```

Retention and all incoming contributions share one rounding operation per
target. The numerator is clamped at `SCALE^3`, so activation stays within the
unit interval even when many edges converge. Because each source has an
outgoing budget of at most `SCALE`, an unstimulated step retains at most 90% of
the previous total activation across the memory, although an individual target
can grow through incoming flow.

A step is a caller-triggered logical tick. It has no wall-clock dependency, and
episode timestamps do not affect retention or propagation.

## Recall and reset

`top_k(limit)` returns at most `limit` atoms with non-zero activation. Results
are ordered by descending activation and then ascending `AtomId`; within one
memory, ties therefore preserve insertion order. A zero limit returns no hits.

Resetting activation leaves atoms and relevance edges unchanged.

## Kernel boundary

The V0 kernel stores atoms, activation, and relevance only in a `MemoryV0`
instance. It performs no persistence, loading, synchronization, or multi-writer
coordination. It also performs no free-text processing, embedding or LLM calls,
or automatic relevance learning. Persistence adapters remain outside the
kernel; the format and lifecycle of the optional SQLite adapter are specified
separately in the [SQLite V1 contract](sqlite-v1-contract.md).
