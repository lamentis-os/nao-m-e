# Repository instructions

## Scope and purpose

These instructions apply to the entire repository. The workspace contains a
deterministic, in-memory Rust 2024 kernel for symbolic episode memory, an
optional SQLite snapshot adapter, and a user- and agent-facing command-line
adapter. Core behavior is defined in `docs/core-contract.md`; the persistence
format and lifecycle are defined in `docs/sqlite-contract.md`.

## Source layout

- `src/model.rs` owns public identifiers, episode values, fixed-point values,
  and errors.
- `src/memory.rs` owns atom storage, bounded feedback traces, and recall.
- `src/parameters.rs` owns fixed-point constants.
- `tests/core_contract.rs` exercises the public core contract.
- `crates/nao-m-e-sqlite` owns SQLite connection handling, format validation,
  snapshot transactions, and adapter tests.
- `crates/nao-m-e-cli` owns the strict CLI argument and text-output grammar,
  command execution, and cross-process CLI tests.

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
