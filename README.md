# NAO-M-E

[![CI](https://github.com/lamentis-os/nao-m-e/actions/workflows/ci.yml/badge.svg)](https://github.com/lamentis-os/nao-m-e/actions/workflows/ci.yml)

NAO-M-E V0 is a deterministic in-memory kernel for structured memory atoms.
It is intentionally a research mechanism, not an LLM memory product.

## V0 contract

The memory state is:

    M(t) = (A, a(t), W)

- A is an append-only set of immutable, structured episode atoms.
- a(t) is fixed-point activation retained across logical steps.
- W is a sparse, directed, non-negative relevance matrix.

Episode content is symbolic. Predicates, terms, and provenance sources are
caller-owned numeric identifiers; V0 never stores free-form text and does not
generate those identifiers.

Each logical memory has a caller-allocated, non-zero 128-bit `MemoryId`.
An `AtomId` combines that memory identifier with a monotonic insertion sequence
starting at zero. Its canonical persistence components are the 16-byte
big-endian memory identifier and the unsigned 64-bit sequence. `Display` uses a
fixed 32-digit lowercase hexadecimal memory ID, a colon, and the decimal
sequence; that text is diagnostic and is not a storage format.

The caller must assign one `MemoryId` to exactly one logical memory, persist it,
and reuse it when reopening that memory. Independent or concurrently written
copies under the same identifier are unsupported because they could allocate
the same next sequence. IDs are references, not permissions, content hashes,
or evidence about an episode.

V0 still stores all state only in memory. The durable ID contract makes a later
storage adapter possible, but IDs and episodes do not survive a process restart
until that adapter persists and reconstructs them.

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

One synchronous logical step computes:

    next[j] =
        floor(
            min(SCALE^3,
                RETENTION * SCALE * current[j]
                + PROPAGATION_GAIN * sum(weight[i,j] * current[i])
            ) / SCALE^2
        )

Retention and every incoming contribution are therefore accumulated before one
rounding operation per target. The numerator is clamped at SCALE cubed, so
converging inputs cannot overflow the activation range.
Each source has an outgoing weight budget of at most SCALE. Because retention
plus propagation gain is 0.9, total activation cannot grow without an external
stimulus.

A step is an explicit logical tick, not elapsed wall-clock time. Episode
timestamps are immutable metadata and do not alter retention. Callers that map
steps to real time must define and keep that cadence themselves.

Positive relevance only means that activating one atom makes another atom
more accessible. It does not encode truth, evidence, causality, or semantic
support.

## Minimal use

    use nao_m_e::{
        Activation, EpisodeDraft, InfluenceWeight, MemoryId, MemoryV0,
        PredicateId, SourceId, Statement, TermId, TimestampMs,
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

    let memory_id = MemoryId::new(0x7b4f_6be0_32c2_4be8_96b8_7394_f734_85af)?;
    let mut memory = MemoryV0::new(memory_id);
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

V0 has no persistence, migration, synchronization, multi-writer access, CLI,
network, free text, embeddings, LLM calls, negative edges, confidence, semantic
consolidation, automatic edge learning, authentication, encryption, or FFI.

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
Documentation tests are part of the gate when present. The stable `Rust CI`
check succeeds only after the quality job and every platform matrix entry have
succeeded.

The tests include exact golden graphs, graph and value invariants, ordering
checks, and four transitions across 10,000 deterministic generated graphs
compared with an independent dense reference evaluator.
