# SQLite contract

This document defines the durable format and observable lifecycle of the
`nao-m-e-sqlite` adapter. Logical kernel state is defined by the
[core contract](core-contract.md); the fixed episode-vector and free-text
retrieval rules are defined by the
[semantic episode contract](semantic-contract.md). The current format is V8.

## Adapter boundary

One database represents one logical memory. `SqliteStore` owns its SQLite
connection, a completely reconstructed numeric `Memory`, staged symbols and
episode vectors, and one lazy semantic encoder:

```rust
impl SqliteStore {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, StoreError>;
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError>;
    pub fn check(path: impl AsRef<Path>) -> Result<(), StoreError>;
    pub fn memory_id(&self) -> MemoryId;
    pub fn memory(&self) -> &Memory;
    pub fn memory_mut(&mut self) -> &mut Memory;
    pub fn intern_symbols(
        &mut self,
        values: &[String],
    ) -> Result<Vec<SymbolId>, StoreError>;
    pub fn symbol_values(
        &self,
        ids: &[SymbolId],
    ) -> Result<Vec<Option<String>>, StoreError>;
    pub fn recall_semantic(
        &mut self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RecallHit>, StoreError>;
    pub fn save(&mut self) -> Result<(), StoreError>;
}
```

Mutation and interning change process state only. `save()` is the sole
persistence boundary; there is no save on mutation or drop. A failed save keeps
caller mutations and any completely prepared episode-vector suffix available
for explicit retry.

Batch interning returns one ID per input value in input order. It validates the
complete normalized batch before staging assignments. Resolution returns one
optional value per input ID and sees persisted and locally staged symbols.
Interned but unreferenced symbols remain valid append-only state.

Episodes inserted directly through `memory_mut()` may use only persisted or
staged symbols. `save()` rejects unknown references and automatically creates
the required semantic vector; there is no separate prepare API. Feedback remains
directed mutable core state and has no receipt, timestamp, provenance, query
binding, or idempotency key.

The adapter does not expose its connection, accept independently constructed
memories, migrate files, or merge independently modified copies.

## Identity and connection settings

The SQLite application ID is `0x4E414F4D` (`NAOM`, decimal `1312902989`). The
only accepted `format_version` is `8`; every other value fails with
`StoreIntegrityError::UnsupportedFormatVersion`.

Immediately after connecting, before schema or memory access, the adapter sets
and verifies:

```text
busy_timeout = 0
foreign_keys = ON
trusted_schema = OFF
ignore_check_constraints = OFF
```

For an existing file it next verifies the application ID and format. Only after
accepting V8 does it verify rollback-journal durability:

```text
journal_mode = DELETE
synchronous = EXTRA
```

Unsupported files are therefore rejected before persistent journal settings
can change. Lock failures return immediately; the adapter neither waits nor
retries.

Creation builds a complete private file-backed staging database, closes and
synchronizes it, and publishes without replacing an existing destination. An
empty database requires no semantic model assets.

## Closed V8 schema

Exactly five tables and one explicit index may exist:

```sql
CREATE TABLE memory_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL CHECK (format_version = 8),
    memory_id BLOB NOT NULL
        CHECK (
            typeof(memory_id) = 'blob'
            AND length(memory_id) = 16
            AND memory_id != zeroblob(16)
        ),
    snapshot_revision INTEGER NOT NULL CHECK (snapshot_revision >= 0),
    semantic_profile_fingerprint BLOB NOT NULL
        CHECK (
            typeof(semantic_profile_fingerprint) = 'blob'
            AND length(semantic_profile_fingerprint) = 32
            AND semantic_profile_fingerprint != zeroblob(32)
        )
) STRICT, WITHOUT ROWID;

CREATE TABLE symbols (
    id BLOB PRIMARY KEY
        CHECK (typeof(id) = 'blob' AND length(id) = 8),
    value TEXT NOT NULL
        CHECK (
            typeof(value) = 'text'
            AND length(CAST(value AS BLOB)) BETWEEN 1 AND 4096
        )
) STRICT, WITHOUT ROWID;

CREATE UNIQUE INDEX symbols_value_unique
    ON symbols(value COLLATE BINARY);

CREATE TABLE episodes (
    sequence BLOB PRIMARY KEY
        CHECK (typeof(sequence) = 'blob' AND length(sequence) = 8),
    payload BLOB NOT NULL
        CHECK (typeof(payload) = 'blob' AND length(payload) >= 26)
) STRICT, WITHOUT ROWID;

CREATE TABLE feedback_edges (
    from_sequence BLOB NOT NULL
        CHECK (typeof(from_sequence) = 'blob' AND length(from_sequence) = 8),
    to_sequence BLOB NOT NULL
        CHECK (typeof(to_sequence) = 'blob' AND length(to_sequence) = 8),
    history_bits INTEGER NOT NULL CHECK (history_bits BETWEEN 0 AND 65535),
    sample_count INTEGER NOT NULL CHECK (sample_count BETWEEN 1 AND 16),
    PRIMARY KEY (from_sequence, to_sequence),
    FOREIGN KEY (from_sequence) REFERENCES episodes(sequence)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (to_sequence) REFERENCES episodes(sequence)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (from_sequence != to_sequence),
    CHECK (history_bits < (1 << sample_count))
) STRICT, WITHOUT ROWID;

CREATE TABLE episode_vectors (
    sequence BLOB PRIMARY KEY
        CHECK (typeof(sequence) = 'blob' AND length(sequence) = 8),
    vector BLOB NOT NULL
        CHECK (typeof(vector) = 'blob' AND length(vector) = 768),
    FOREIGN KEY (sequence) REFERENCES episodes(sequence)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT, WITHOUT ROWID;
```

SQLite primary-key autoindexes are expected. Any extra table, index, view,
trigger, virtual table, or altered definition makes the store invalid.
`memory_meta` has exactly one singleton row containing V8, the non-zero memory
identity, non-negative revision, and fixed non-zero semantic-profile fingerprint.

## Canonical scalar encoding

- `MemoryId` is its 16-byte big-endian representation.
- Episode sequences and `SymbolId` values are fixed 8-byte big-endian unsigned
  integers.
- Timestamps are fixed 8-byte big-endian two's-complement signed Unix
  milliseconds; the full `i64` range is preserved.
- Feedback history is an unsigned 16-bit bitset stored as a SQLite integer.
- Feedback sample count is an integer in `1..=16`.
- One semantic vector is 384 little-endian two's-complement `i16` components,
  exactly 768 bytes.

Fixed-width big-endian identifiers preserve the complete unsigned ranges and
numeric lexicographic order. Native-endian, decimal-text, and variable-width
identifier representations are invalid.

Payload counts use canonical unsigned LEB128. Each byte contributes its low
seven bits, least-significant group first, and the high bit denotes
continuation. Zero is exactly `00`; multi-byte values may not end with a zero
group. Truncation, overflow, overlong encoding, impossible remaining lengths,
and counts not fitting `usize` fail before allocation.

## Symbols and normalization

Attribute keys and values share one append-only symbol namespace. IDs start at
zero, form the exact prefix `0..N`, and are assigned by first normalized
occurrence. They are never renamed, deleted, reused, rebound, or portable across
independently created databases. After `u64::MAX`, the namespace is exhausted.

Normalization is part of V8 and fixed to Unicode 16.0:

1. Apply NFKC.
2. Apply locale-independent Unicode lowercase mapping to every scalar.
3. Apply NFKC again.
4. Collapse Unicode whitespace runs to one ASCII space and trim the ends.
5. Reject every remaining Unicode control scalar.
6. Reject an empty result or normalized UTF-8 longer than 4,096 bytes.

The lowercase mapping includes unconditional multi-scalar expansions and is
neither context-sensitive conversion nor full case folding. Punctuation,
accents, and other surviving scalars remain. Only normalized text is stored
with binary collation. There is no display spelling, alias, stemming, fuzzy
matching, accent removal, or synonym mapping.

Equal normalized key and value text receives one ID. Indexed lookup and
resolution use bounded batches of at most 900 SQL parameters and do not retain
the complete text catalog in core memory.

## Episode payload

Every `episodes.payload` is exactly:

```text
EpisodePayload =
    timestamp_ms          i64be
    attribute_count       uleb128
    attributes            Attribute * attribute_count

Attribute =
    key_id                u64be
    value_count           uleb128
    values                u64be * value_count
```

The minimum is 26 bytes. Attribute count and every value count are positive;
keys and each value set are strictly increasing and duplicate-free; every ID
resolves in `symbols`; and the decoder must consume the full payload. Stored
order is validated rather than repaired through core canonicalization.

The timestamp is signed Unix milliseconds. It is content, not recall score or
insertion identity. The episode sequence remains canonical append order.

## Feedback rows

`feedback_edges` stores sparse directed traces between existing episodes.
Self-edges are invalid. `sample_count` is `1..=16`; bit zero is the newest
assessment, `1` means helpful, and bits above the count are zero. Neutral traces
remain persisted. Every row is reconstructed through the validated core setter.

Feedback affects only the programmatic source-conditioned core recall. It is
stored unchanged by V8 and does not enter semantic query scoring.

## Episode vectors

`episode_vectors` owns exactly one fixed-profile vector for every committed
episode, keyed by the same sequence. Its rows must form the exact episode prefix
and can neither outlive nor precede an episode. Identical episode text still has
one row per episode; there is no cue catalog, posting table, shared-vector key,
persisted norm, or approximate-search index.

The vector projection, model, fingerprint, codec, validation, and scoring rules
are defined by the [semantic episode contract](semantic-contract.md).

## Open lifecycle

Opening an existing store performs, in order:

1. Apply and verify session settings.
2. Verify the application ID and require V8.
3. Verify durability without changing journal mode.
4. Begin one consistent read transaction.
5. Validate metadata, fixed profile, closed schema, and targeted SQLite quick
   checks for `memory_meta`, `symbols`, `episodes`, and `feedback_edges`.
6. Stream `symbols` by ID and require contiguous IDs, canonical text, bounds,
   and unique values.
7. Stream `episodes` by sequence, require the exact prefix, strictly decode each
   payload, resolve every symbol, and rebuild immutable core atoms.
8. Validate and restore the complete feedback graph.
9. Compare the indexed last episode-vector sequence with the episode tail.
10. Commit the read transaction and expose the store.

Operational open intentionally does not scan vector bodies or the entire vector
key set. A missing or extra tail fails immediately; an interior gap or malformed
body can remain dormant until recall or full check. This tier keeps ordinary
process startup independent of semantic payload bytes.

The adapter retains numeric memory, symbol boundaries, locally staged symbol
assignments, and prepared vectors. It does not retain complete text or persisted
vector maps. Core symbolic indexes are private caches rebuilt from episodes.

## Full check lifecycle

`SqliteStore::check(path)` opens with SQLite's physical read-only flag. It runs
the same identity, format, schema, snapshot, and reconstruction checks plus
whole-file `integrity_check`, `foreign_key_check`, and a complete ordered vector
scan. Vector sequences must equal `0..episode_count`; every vector must have the
canonical width, component range, and non-zero content.

The check never constructs the semantic encoder, resolves model assets,
re-encodes text, saves, or changes revision, rows, journal mode, or file bytes.

## Save lifecycle

Each store remembers its opened revision, persisted episode count, symbol tail,
pending symbol assignments, and completely prepared vector suffix. Before a
write transaction, `save()`:

1. validates every new episode symbol reference;
2. rejects an already stale writer before model work;
3. resolves normalized texts for only the unprepared episode suffix;
4. creates one canonical episode passage and vector per new episode; and
5. adds those vectors to retry state only after the complete preparation batch
   succeeds.

It then verifies durability and starts one `BEGIN IMMEDIATE` transaction:

1. Recheck application ID, memory ID, expected revision, V8 profile, and closed
   schema.
2. Reject revision exhaustion.
3. Verify remembered episode, symbol, and vector tails with indexed last-row
   queries.
4. Advance the singleton revision exactly once.
5. Insert staged symbols in ID order.
6. Append episodes and their same-sequence vectors.
7. Reconcile feedback in deterministic source/target order.
8. Commit, then update local persisted boundaries and clear pending state.

Every successful save increments the revision, including a logical no-op.
Inference never runs while the write transaction is held. A transactional
failure rolls back all database changes but retains prepared vectors for retry;
newly appended episodes after such a failure encode only their unprepared
suffix. A stale writer is never merged or overwritten.

Feedback reconciliation compares the ordered persisted and in-memory graphs
because callers can mutate through `memory_mut()` and the core has no persistence
dirty tracking. Small deltas issue planned insert/update/delete statements after
the read cursor closes; a bounded bulk fallback replaces the table for very
large changes. Every observed persisted row is validated before DML.

## Semantic recall lifecycle

Semantic recall first normalizes the query. Limit zero returns no hits after
query validation without loading the model, rereading the revision, or scanning
vectors. For positive limits it verifies memory identity and revision, returns
offline when the committed store is empty, and encodes the query outside a
transaction. It then starts a read transaction, repeats the identity/revision
check, and streams every vector in primary-key order.

The scan requires the exact vector prefix, validates all accessed bytes, computes
fixed-point cosine scores, and keeps a bounded deterministic ranking. Pending
unsaved episodes are excluded. A concurrent commit during query encoding fails
the second revision check rather than combining different snapshots. The read
transaction and database are not mutated.

## Fail-closed behavior and compatibility

`StoreError` distinguishes I/O, SQLite, entropy, invalid stores, concurrent
modification, revision or symbol-ID exhaustion, invalid symbols or semantic
queries, unknown symbol references, and episode/query encoding failure.
`StoreIntegrityError` distinguishes file identity, metadata, schema, encodings,
symbol, episode, feedback, episode-vector, SQLite-integrity, and foreign-key
violations. Both enums are non-exhaustive.

The adapter never accepts a partial snapshot, renumbers IDs, repairs references,
canonicalizes stored data, drops invalid rows, clamps feedback, or mutates an
unsupported file. V1-V3 lack source text; V4-V5 have incompatible symbol and
episode contracts; V6 lacks mandatory integrated vectors; V7 stores per-cue
vectors and postings under a different profile. Constructing V8 episode vectors
requires model-backed re-encoding, so V1-V7 are rejected without migration.

## CLI boundary

The CLI creates and accepts V8 only. `add` and `feedback` require an existing
database and call exactly one save. Singular and batch add intern symbols,
encode every new episode, and atomically publish symbols, episodes, and vectors.
If the installed model prerequisite is missing or invalid, add fails with empty
standard output and no committed state.

CLI recall accepts only a free-text `--query`, performs no interning or save,
resolves text only for returned hits, and reports the semantic score as
`activation_ppm`. It uses the read-write operational store connection while
remaining logically non-mutating. `check` uses the physically read-only full
audit and is silent on success.

## Durability boundary

A save is one rollback-journal transaction with `synchronous=EXTRA`, SQLite's
strongest applicable rollback-journal synchronization policy. This does not make
unsaved mutations durable or prove sudden-power-loss behavior for every device,
filesystem, or operating system.

Creation synchronizes the private staging file before no-clobber publication.
It does not synchronize the parent directory. A hard-link publication fallback
can leave a temporary link after cleanup failure or process crash; the caller
owns the containing directory's trust and lifecycle.
