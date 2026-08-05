# Repository instructions

## Scope and purpose

These instructions apply to the entire repository. The workspace contains a
deterministic, in-memory Rust 2024 kernel for symbolic episode memory, an
optional SQLite snapshot adapter, and a user- and agent-facing command-line
adapter. SQLite V7 atomically stores a fixed multilingual-E5 projection for
bound attribute cues, while a dedicated semantic crate owns lazy local model
inference. Core behavior is defined in `docs/core-contract.md`; authoritative
persistence is defined in `docs/sqlite-contract.md`; the fixed model and cue
projection are defined in `docs/semantic-contract.md`.

## Source layout

- `src/model.rs` owns public identifiers, episode values, fixed-point values,
  and errors.
- `src/memory.rs` owns atom storage and bounded feedback traces.
- `src/memory/recall.rs` owns cue indexing, source-conditioned scoring, and
  deterministic recall ranking.
- `src/parameters.rs` owns fixed-point constants.
- `tests/core_contract.rs` and `tests/core_contract` form one public core
  contract target.
- `crates/nao-m-e-sqlite/src/format.rs` owns SQLite file identity, session and
  durability policy, and closed-schema creation and validation.
- `crates/nao-m-e-sqlite/src/format/codec.rs` owns canonical identifier and
  episode-payload encoding and decoding.
- `crates/nao-m-e-sqlite/src/format/tests.rs` and
  `crates/nao-m-e-sqlite/src/format/codec/tests.rs` exercise the private format
  and codec invariants.
- `crates/nao-m-e-sqlite/src/store.rs` owns SQLite store lifecycle, metadata and
  revision validation, and atomic save orchestration.
- `crates/nao-m-e-sqlite/src/store/symbols.rs` owns text-symbol normalization,
  allocation, staging, validation, and resolution.
- `crates/nao-m-e-sqlite/src/store/semantic.rs` owns integrated cue allocation,
  embedding preparation, transactional cue/posting publication, and exhaustive
  semantic audit.
- `crates/nao-m-e-sqlite/src/store/feedback.rs` owns feedback restoration and
  transactional reconciliation.
- `crates/nao-m-e-sqlite/src/store/tests.rs` exercises private lifecycle,
  revision, corruption, and transaction invariants.
- `crates/nao-m-e-sqlite/tests` exercises the public adapter contract.
- `crates/nao-m-e-semantic/src/profile.rs` owns the fixed model artifact,
  tokenizer, projection, runtime, pooling, normalization, quantization, and
  fingerprint contract.
- `crates/nao-m-e-semantic/src/model.rs` owns bound cue text and fixed-width
  embedding values.
- `crates/nao-m-e-semantic/src/encoder.rs` owns lazy verified asset resolution,
  tokenization, ONNX inference, and canonical vector production.
- `crates/nao-m-e-semantic/src/error.rs` owns semantic runtime failures.
- `crates/nao-m-e-semantic/tests` exercises the public fixed-profile runtime
  boundary without requiring the production model in ordinary test runs.
- `crates/nao-m-e-cli/src/main.rs` owns the process boundary, root dispatch,
  initialization, exhaustive check, feedback, and shared save/output handling.
- `crates/nao-m-e-cli/src/add.rs` owns Add grammar, text-episode parsing,
  symbol interning, and atomic Add execution.
- `crates/nao-m-e-cli/src/recall.rs` owns Recall grammar, symbol resolution, and
  deterministic text output.
- `crates/nao-m-e-cli/tests/cli.rs` and `tests/cli` form one cross-process CLI
  contract target.

## Contract guardrails

The exact semantics live in the applicable contract document. Preserve these
repository-level guardrails when changing the implementation:

- Preserve `#![forbid(unsafe_code)]` and deterministic behavior across supported
  platforms.
- Use integer fixed-point arithmetic and ordered traversal for observable recall,
  feedback, and results.
- Keep episode atoms append-only and immutable. Bounded feedback traces remain
  separate mutable state.
- Keep caller-owned memory identity distinct from local insertion sequence;
  foreign or absent atom IDs must never alias local atoms.
- Keep feedback traces directed, bounded, and atomically validated. Their
  signed projection affects retrieval accessibility only.
- Keep recall ordering deterministic and feedback semantically distinct from
  truth or confidence.
- Keep persistence outside the core crate. Decode and validate a complete
  snapshot before exposing reconstructed state, and commit a save atomically.
- Keep attribute-key and value text in one append-only SQLite symbol catalog.
  The core and episode payloads remain numeric; every persisted symbol
  reference must resolve to that catalog before a snapshot is exposed.
- Keep the fixed semantic profile and its cue vectors outside the dependency-free
  core. SQLite V7 publishes symbols, vectors, episodes, and postings atomically;
  normalized text and episodes remain authoritative, and vectors must not change
  current core scoring.
- Keep text processing locale-free: do not add locale selection, language
  detection, translation, localized profiles, or language-specific execution
  branches. Unicode and cross-script text are data for one fixed pipeline, not
  product localization.
- Resolve and encode only missing cues before taking the SQLite write
  transaction. Model download or inference failure must fail closed without a
  partially published database state or silent symbolic fallback.
- Preserve canonical fixed-width identifier encodings and reject stale writers
  rather than silently merging or overwriting their snapshots.

## Change discipline

- Cleanup and refactoring must preserve the public API and runtime behavior
  unless the task explicitly changes the contract.
- Add dependencies, randomness, external I/O, networking, or wall-clock-driven
  dynamics only when the task explicitly requires them. Keep integrations out
  of deterministic recall and feedback logic.
- Update implementation, the applicable contract document, and contract tests
  together for every intentional behavior change.
- Do not split coherent modules or add abstraction without a concrete ownership
  or maintenance benefit.

## Documentation ownership

- `README.md` is the durable user entry point: purpose, overview, minimal use,
  boundaries, and links.
- `docs/core-contract.md` is the single cross-cutting core specification.
- `docs/sqlite-contract.md` is the single SQLite format and lifecycle
  specification.
- `docs/semantic-contract.md` is the single fixed-profile semantic cue,
  runtime, projection, and validation specification integrated with SQLite V7.
- Rustdoc describes symbol-local semantics, errors, and non-obvious behavior.
  Public items remain documented under `#![deny(missing_docs)]`.
- Ordinary comments explain only non-obvious reasons or invariants, not control
  flow, review history, or status.
- `AGENTS.md` contains repository workflow, guardrails, and verification. Do not
  use these documents as task logs, changelogs, PR reports, or roadmaps.

## Verification

Run the repository gates:

```sh
cargo fmt --all -- --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked --no-fail-fast
cargo test --workspace --doc --all-features --locked
cargo build --workspace --release --all-targets --all-features --locked
cargo test --workspace --release --all-targets --all-features --locked --no-fail-fast
git diff --check
```

Treat `.github/workflows/ci.yml` as the operational source of truth when the CI
matrix changes.

When persistence dependencies change, also inspect
`cargo tree -p nao-m-e --edges normal` and confirm that the core package still
has no runtime dependencies.

## Evidence

- Report only checks that were actually executed.
- Separate local verification from remote CI and its platform results.
- Do not claim persistence, physical power-loss resilience, learned retrieval
  quality, semantic correctness, production readiness, or cross-platform
  verification without direct evidence.
