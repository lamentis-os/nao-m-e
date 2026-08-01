# NAO-M-E

[![CI](https://github.com/lamentis-os/nao-m-e/actions/workflows/ci.yml/badge.svg)](https://github.com/lamentis-os/nao-m-e/actions/workflows/ci.yml)

NAO-M-E V0 is a deterministic in-memory kernel for structured memory atoms.
It is intentionally a research mechanism, not an LLM memory product.

## V0 contract

The memory state is:

    M(t) = (A, a(t), W)

- A is an append-only set of immutable, structured episode atoms.
- a(t) is persistent fixed-point activation.
- W is a sparse, directed, non-negative relevance matrix.

Episode content is symbolic. Predicates, terms, and provenance sources are
caller-owned numeric identifiers; V0 never stores free-form text and does not
generate those identifiers.

An episode contains:

- occurrence and recording time;
- canonical contextual statements;
- one required observation;
- an optional action;
- an optional outcome;
- one provenance source identifier.

The atom is immutable after insertion. Activation and relevance edges are
stored separately.

## Dynamics

All unit-interval values use integer parts per million:

    SCALE = 1,000,000
    RETENTION = 500,000
    PROPAGATION_GAIN = 400,000

One synchronous step computes:

    next[j] =
        floor(RETENTION * current[j] / SCALE)
        + sum floor(
            PROPAGATION_GAIN * weight[i,j] * current[i]
            / SCALE^2
          )

The result is clamped to SCALE. Each source has an outgoing weight budget of
at most SCALE. Because retention plus propagation gain is 0.9, total
activation cannot grow without an external stimulus.

Positive relevance only means that activating one atom makes another atom
more accessible. It does not encode truth, evidence, causality, or semantic
support.

## Minimal use

    use nao_m_e::{
        Activation, EpisodeDraft, InfluenceWeight, MemoryV0, PredicateId,
        SourceId, Statement, TermId, TimestampMs,
    };

    let observation = Statement::new(
        PredicateId::new(1),
        vec![TermId::new(10), TermId::new(11)],
    )?;

    let episode = EpisodeDraft {
        occurred_at: TimestampMs::new(1_000),
        recorded_at: TimestampMs::new(1_001),
        context: Vec::new(),
        observation,
        action: None,
        outcome: None,
        source: SourceId::new(7),
    };

    let mut memory = MemoryV0::new();
    let first = memory.insert_episode(episode.clone())?;
    let second = memory.insert_episode(episode)?;
    memory.set_relevance(
        first,
        second,
        InfluenceWeight::from_ppm(1_000_000)?,
    )?;
    memory.stimulate(first, Activation::ONE)?;
    memory.step();

The example uses the question-mark operator and is intended for a function
whose error type can contain the constructor and graph errors.

## Deliberate exclusions

V0 has no persistence, CLI, network, free text, embeddings, LLM calls,
negative edges, confidence, semantic consolidation, automatic edge learning,
authentication, encryption, or FFI.

## Verification

The repository has no runtime dependencies. CI uses the pinned Rust toolchain
to verify Linux x86_64, macOS ARM64, and Windows x86_64. Its required local
gates are:

    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
    cargo test --workspace --all-targets --all-features --locked --no-fail-fast
    cargo test --workspace --doc --all-features --locked
    cargo build --workspace --release --all-targets --all-features --locked
    cargo test --workspace --release --all-targets --all-features --locked --no-fail-fast
    cargo doc --workspace --no-deps --all-features --locked

Run the documentation command with `RUSTDOCFLAGS` set to `-D warnings`.
The crate-level minimal example is compiled as a documentation test. The
stable `Rust CI` check succeeds only after the quality job and every platform
matrix entry have succeeded.

The tests include exact golden graphs, graph and value invariants, ordering
checks, and 10,000 deterministic generated graphs compared with an independent
dense reference evaluator.
