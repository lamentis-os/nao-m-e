# SQLite contract

This document defines the durable format and observable lifecycle of the
`nao-m-e-sqlite` adapter. The logical state being stored and reconstructed is
defined by the [core contract](core-contract.md). Integrated semantic cue
projection is defined by the [semantic cue contract](semantic-contract.md).
The current persisted format version is `7`.

## Adapter boundary

One database represents exactly one logical memory. `SqliteStore` owns both its
SQLite connection and the completely reconstructed `Memory`:

```rust
pub struct SqliteStore { /* private fields */ }

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
    pub fn save(&mut self) -> Result<(), StoreError>;
}
```

Mutating the returned memory or interning symbols changes only process state.
`save()` is the sole persistence boundary. There is no automatic save on
mutation or drop. A failed save leaves all in-memory episode, symbol, and
feedback changes available for an explicit retry.

Batch interning returns one ID per input value in the same order. It fully
normalizes and validates the batch before staging any new assignment. Resolution
returns one optional value per input ID in the same order and can see both
persisted and locally staged values. Interned but unreferenced values are valid
snapshot state and are never garbage-collected.

An episode inserted directly through `memory_mut()` may refer only to persisted
or staged `SymbolId` values. `save()` rejects every new episode containing an
unknown key or value ID.

Feedback remains separate mutable core state. A save persists each directed,
bounded trace but no receipt, timestamp, provenance record, query binding, or
idempotency key. Repeating feedback after a successful save appends another
sample.

The adapter does not expose its connection and cannot save an independently
constructed memory. Database copies with the same `MemoryId` must not be
modified independently and later merged.

## Identity and connection settings

The SQLite application ID is `0x4E414F4D` (`NAOM`, decimal `1312902989`) and the
only accepted `format_version` is `7`. Every other version is rejected with
`StoreIntegrityError::UnsupportedFormatVersion`. There is no migrator.

Immediately after connecting, before schema or memory access, the adapter
applies and verifies these connection-local settings:

```text
busy_timeout = 0
foreign_keys = ON
trusted_schema = OFF
ignore_check_constraints = OFF
```

For an existing file it then verifies the application ID and reads the metadata
format version. Only after accepting V7 does it verify rollback-journal mode and
apply synchronization policy:

```text
journal_mode = DELETE
synchronous = EXTRA
```

An unsupported file is therefore rejected before persistent journal settings
can be changed. A zero busy timeout means SQLite lock failures are returned
immediately; the adapter neither waits nor retries.

Creation uses a private file-backed staging database. It applies the same
settings, writes and validates a complete initial store, closes and synchronizes
the staging file, and publishes it without replacing an existing target.

## Closed V7 schema

Only these six tables and three explicit indexes may exist:

```sql
CREATE TABLE memory_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL CHECK (format_version = 7),
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
        ),
    semantic_cue_count BLOB NOT NULL
        CHECK (
            typeof(semantic_cue_count) = 'blob'
            AND length(semantic_cue_count) = 8
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
        CHECK (
            typeof(from_sequence) = 'blob'
            AND length(from_sequence) = 8
        ),
    to_sequence BLOB NOT NULL
        CHECK (
            typeof(to_sequence) = 'blob'
            AND length(to_sequence) = 8
        ),
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

CREATE TABLE semantic_cues (
    cue_id BLOB PRIMARY KEY
        CHECK (typeof(cue_id) = 'blob' AND length(cue_id) = 8),
    key_id BLOB NOT NULL
        CHECK (typeof(key_id) = 'blob' AND length(key_id) = 8),
    value_id BLOB NOT NULL
        CHECK (typeof(value_id) = 'blob' AND length(value_id) = 8),
    vector BLOB NOT NULL
        CHECK (typeof(vector) = 'blob' AND length(vector) = 768),
    FOREIGN KEY (key_id) REFERENCES symbols(id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (value_id) REFERENCES symbols(id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT, WITHOUT ROWID;

CREATE UNIQUE INDEX semantic_cues_pair_unique
    ON semantic_cues(key_id, value_id);

CREATE TABLE episode_cues (
    sequence BLOB NOT NULL
        CHECK (typeof(sequence) = 'blob' AND length(sequence) = 8),
    cue_id BLOB NOT NULL
        CHECK (typeof(cue_id) = 'blob' AND length(cue_id) = 8),
    PRIMARY KEY (sequence, cue_id),
    FOREIGN KEY (sequence) REFERENCES episodes(sequence)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (cue_id) REFERENCES semantic_cues(cue_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT, WITHOUT ROWID;

CREATE INDEX episode_cues_by_cue
    ON episode_cues(cue_id, sequence);
```

SQLite's primary-key autoindexes are expected. Any additional table, index,
view, trigger, virtual table, or altered schema definition makes the store
invalid. `memory_meta` contains exactly one singleton row with version `7`, the
memory identity, snapshot revision, fixed semantic profile fingerprint, and
canonical semantic cue count.

## Canonical scalar encoding

- `MemoryId` is its 16-byte big-endian representation.
- Episode sequences, `SymbolId` values, semantic cue IDs, and the semantic cue
  count are fixed-width 8-byte big-endian unsigned integers.
- Timestamps are fixed-width 8-byte big-endian two's-complement signed Unix
  milliseconds. The complete `i64` range is preserved.
- Feedback history is an unsigned 16-bit bitset stored as a SQLite integer.
- Feedback sample count is an integer in `1..=16`.
- A semantic vector is exactly 384 consecutive signed two's-complement `i16`
  coordinates in little-endian order, for a total of 768 bytes.

Fixed-width big-endian BLOBs preserve the complete unsigned ranges and numeric
lexicographic ordering. Native-endian, decimal-text, and variable-width
identifier encodings are not accepted.

Payload counts use canonical unsigned LEB128. Each byte contributes its low
seven bits, least-significant group first; the high bit denotes continuation.
Zero is exactly `00`, and a multi-byte encoding may not end in a zero seven-bit
group. Truncation, `u64` overflow, overlong encoding, impossible remaining-byte
counts, and counts that do not fit `usize` are rejected before allocation.

## Symbol identity

There is one shared append-only symbol namespace for attribute keys and values.
IDs begin at zero and always form the exact prefix `0..N`. A new normalized
value receives the next ID according to its first occurrence in the input batch.
IDs are never renamed, deleted, reused, or rebound. After assigning `u64::MAX`,
the namespace is exhausted.

Symbol IDs are stable only within a logical database and its continued copies;
independently created stores need not assign the same ID to equal text.

The normalization algorithm is part of format V7 and is fixed to Unicode 16.0:

1. Apply NFKC.
2. Apply locale-independent Unicode lowercase mapping to every scalar.
3. Apply NFKC again.
4. Split on Unicode whitespace, remove empty outer components, and join the
   remaining components with one ASCII space.
5. Reject every remaining Unicode control scalar.
6. Reject an empty result or normalized UTF-8 longer than 4,096 bytes.

Lowercasing includes unconditional multi-scalar expansions and is neither
context-sensitive conversion nor full case folding. Punctuation, accents, and
all other surviving scalars remain unchanged. The normalized value is the only
stored text and uses SQLite binary collation. There is no display spelling,
alias, stemming, fuzzy matching, accent removal, or synonym mapping. Semantic
projection consumes this normalized text but does not change symbol identity.

Equal normalized key and value text receives the same ID. Persisted lookup and
resolution use indexed queries with at most 900 parameters and do not load the
full text catalog into the core.

## Episode payload

Each `episodes.payload` contains exactly one episode:

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

The minimum is 26 bytes: an 8-byte timestamp, a 1-byte attribute count, an
8-byte key, a 1-byte value count, and one 8-byte value.

Persisted payloads obey all of these invariants:

- `attribute_count >= 1`;
- every `value_count >= 1`;
- keys are strictly ascending and occur once;
- values within an attribute are strictly ascending and duplicate-free;
- every key and value resolves in `symbols`;
- the decoder consumes every byte and rejects trailing data.

The encoder receives already canonical core atoms. The decoder checks ordering
before constructing a core draft; it never uses core canonicalization to repair
corrupt stored data. The payload has no local magic or version field because
`memory_meta.format_version` selects its codec.

## Feedback rows

`feedback_edges` stores sparse directed traces between existing episode
sequences. Self-edges are invalid. `sample_count` is `1..=16`; bit zero is the
newest assessment, `1` means helpful, and all bits above `sample_count` must be
zero. Neutral traces remain stored state. The SQLite adapter reconstructs every
row through the validated core setter and exposes no partial graph.

## Semantic rows

`semantic_cues` stores one fixed-profile vector for every distinct key/value
binding used by a committed episode. Cue IDs form the exact append-only prefix
selected by `memory_meta.semantic_cue_count`; key/value pairs are unique and
reference the shared symbol catalog. `episode_cues` is the exact posting
projection from each episode to all of its bound cues. Its reverse index does
not change the posting identity or ordering contract.

The count is itself an unsigned 64-bit prefix length. It can therefore reach
`u64::MAX`, making `u64::MAX - 1` the largest assignable cue ID; a further cue
fails before text resolution or model inference.

The exact text projection, model artifacts, profile fingerprint, quantization,
lazy acquisition, and validation tiers are defined by the
[semantic cue contract](semantic-contract.md).

## Open lifecycle

Opening an existing store performs, in order:

1. Apply and verify session settings.
2. Verify the SQLite application ID.
3. Read metadata and require format V7.
4. Verify durability settings without changing persistent journal mode.
5. Begin one consistent read transaction.
6. Validate memory identity, revision, the fixed semantic profile, exact schema,
   targeted SQLite quick checks, and agreement between the semantic cue count
   and cue-catalog tail.
7. Stream `symbols` in ID order and require fixed-width contiguous IDs,
   canonical normalized text, length bounds, and unique values.
8. Stream `episodes` in sequence order and require the exact prefix `0..N`.
9. Strictly decode every payload, validate every symbol reference, and rebuild
   the corresponding immutable core atom.
10. Validate and restore the complete feedback graph.
11. Commit the read transaction and only then expose the store.

Operational open does not scan every semantic vector or posting. Persisted cue
rows reached while preparing a later save are validated before use. The adapter
keeps only catalog boundaries, the semantic cue count, and locally pending
assignments in long-lived state, not full text, cue, or posting maps. The
reconstructed memory stores only numeric symbols. Symbolic recall indexes remain
private core caches rebuilt from immutable episodes.

## Full check lifecycle

`SqliteStore::check(path)` opens the database with SQLite's read-only flag. It
performs the same identity, version, settings, metadata, schema, snapshot, and
operational reconstruction checks, plus whole-file `integrity_check`,
`foreign_key_check`, and a complete semantic audit. The semantic audit requires
gapless cue IDs, canonical vectors, unique and resolvable pairs, exact postings
for every episode, and neither unused nor missing cue pairs.

The full check never constructs the semantic encoder or resolves model assets.
It neither saves nor changes the revision, rows, journal mode, or file contents.
It is intentionally more expensive than operational open.

## Save lifecycle

Each store remembers its opened revision, persisted episode count, symbol tail,
and persisted semantic cue count. Before taking a write transaction, save:

1. Validates every new episode key and value against the
   persisted-or-staged symbol prefix.
2. Derives new episode bindings in deterministic order and resolves existing
   cue pairs through bounded indexed queries.
3. Resolves normalized text and encodes only missing cues through the fixed
   profile, outside the SQLite write lock.
4. Validates the complete encoder result and stages new cue IDs, vectors, and
   episode postings in process memory.

It then verifies durability and runs one `BEGIN IMMEDIATE` transaction:

1. Recheck application ID, memory ID, expected revision, persisted semantic cue
   count, and the closed schema.
2. Reject revision exhaustion before any mutation.
3. Verify the remembered episode, symbol, and semantic cue tails against indexed
   last-row queries.
4. Increment the singleton revision exactly once, publish the next semantic cue
   count, and require one changed metadata row.
5. Insert staged symbols and semantic cues in ascending ID order.
6. Append only episodes at or beyond the remembered episode count and insert
   their exact semantic postings.
7. Reconcile feedback in deterministic source/target order.
8. Commit.
9. Only after commit update remembered boundaries and clear staged symbols and
   cues.

Every successful save increments the revision, including a logical no-op.
Interning never starts an independent transaction. Two sessions may locally
stage different text under the same next ID; the first save advances the shared
revision, and the stale session fails before publishing any symbol, semantic
cue, episode, posting, or feedback state. Semantic inference can already have
completed before that compare-and-swap rejection; it never runs while the write
transaction is held.

Feedback reconciliation performs an ordered full comparison because callers
can mutate the graph through `memory_mut()` and the adapter does not duplicate a
persisted graph or introduce persistence dirty tracking into the core. Small
deltas perform only planned insert, update, and delete statements after the read
cursor closes. A bounded bulk fallback replaces the table for very large
changes. Every encountered persisted row is validated before any feedback DML.

Any failure before commit rolls back the database while retaining the local
pending state for retry. The adapter does not automatically retry, merge
divergent histories, or use last-writer-wins behavior.

## Fail-closed behavior

`StoreError` distinguishes I/O, SQLite, entropy, invalid persisted state,
concurrent modification, revision exhaustion, invalid symbol input, symbol-ID
or semantic cue-ID exhaustion, unknown symbol references, and semantic encoding
failure. `StoreIntegrityError` reports file identity, metadata, schema,
encoding, symbol, episode, feedback, semantic cue, posting, SQLite integrity,
and foreign-key violations. Both enums are non-exhaustive.

The adapter never migrates unsupported formats, accepts a partially valid
snapshot, renumbers identifiers, repairs references, canonicalizes stored
payloads or symbols, drops invalid rows, clamps feedback, or exposes state
before complete validation. Direct SQL modification is outside the supported
API.

V1-V3 lack source text. V4 and V5 have incompatible symbol namespaces and
episode semantics. Converting their role-specific statements to unordered open
attributes would be a potentially lossy semantic transformation, not a safe
format migration. V6 contains the current text and episode semantics but lacks
the fixed integrated semantic profile, cue vectors, postings, and metadata.
Creating those rows would require a model-backed mutating migration, which this
adapter does not perform. V1-V6 are therefore rejected.

## CLI boundary

The CLI creates and accepts only format V7. `add`, `recall`, and `feedback`
require an existing database and reconstruct it fully. Singular and batch add
intern keys and values through the shared catalog, encode previously unseen
bound cues, and publish symbols, cues, vectors, episodes, and postings using
exactly one save. Feedback also uses exactly one save.

Recall performs no interning and no save. It resolves only distinct symbols in
the returned hits through bounded indexed batches. It emits normalized text and
does not change the revision or database contents. The current implementation
uses the same read-write `SqliteStore::open()` path, so logical read-only behavior
does not imply a read-only SQLite connection. `check` uses the physically
read-only exhaustive path, is silent on success, and never loads the model.

## Durability boundary

A save is one rollback-journal transaction with `synchronous=EXTRA`. This
targets SQLite's strongest applicable rollback-journal synchronization behavior,
including additional directory synchronization after journal removal. It does
not make unsaved mutations durable and is not evidence for every storage
device, filesystem, operating system, or sudden-power-loss scenario.

Creation synchronizes the complete private staging file before no-clobber
publication. The adapter does not `fsync` the parent directory. Where the
platform uses a hard-link publication fallback, a temporary link can remain
after cleanup failure or process crash. The caller owns the trust and lifecycle
of the containing directory.
