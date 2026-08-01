# Repository instructions

## Scope and purpose

These instructions apply to the entire repository. `nao_m_e` is a Rust 2024
research library for deterministic, in-memory symbolic episode memory. The
cross-cutting V0 behavior is defined in `docs/v0-contract.md`.

## Source layout

- `src/model.rs` owns public identifiers, episode values, fixed-point values,
  and errors.
- `src/memory.rs` owns atom storage, activation transitions, relevance, and
  recall.
- `src/parameters.rs` owns fixed-point constants.
- `tests/v0_contract.rs` exercises the public V0 contract and independent
  differential reference behavior.

## Contract guardrails

The exact semantics live only in `docs/v0-contract.md`. Preserve these
repository-level guardrails when changing the implementation:

- Preserve `#![forbid(unsafe_code)]` and deterministic behavior across supported
  platforms.
- Use integer fixed-point arithmetic and ordered traversal for observable state
  transitions and results.
- Keep episode atoms append-only and immutable. Activation and relevance remain
  separate mutable state.
- Keep caller-owned memory identity distinct from local insertion sequence;
  foreign or absent atom IDs must never alias local atoms.
- Keep relevance directed, positive, budgeted, and atomically validated.
- Keep recall ordering deterministic and relevance semantically distinct from
  truth or confidence.

## Change discipline

- Cleanup and refactoring must preserve the public API and runtime behavior
  unless the task explicitly changes the contract.
- Add dependencies, randomness, external I/O, networking, or wall-clock-driven
  dynamics only when the task explicitly requires them. Keep integrations out
  of deterministic transition logic.
- Update implementation, `docs/v0-contract.md`, and contract tests together for
  every intentional behavior change.
- Do not split coherent modules or add abstraction without a concrete ownership
  or maintenance benefit.

## Documentation ownership

- `README.md` is the durable user entry point: purpose, overview, minimal use,
  boundaries, and links.
- `docs/v0-contract.md` is the single cross-cutting technical specification.
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

## Evidence

- Report only checks that were actually executed.
- Separate local verification from remote CI and its platform results.
- Do not claim persistence, learned relevance, semantic correctness,
  production readiness, or cross-platform verification without direct evidence.
