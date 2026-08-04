# SQLite contract

This document defines the durable format and observable lifecycle of the
`nao-m-e-sqlite` adapter. The logical state being stored and reconstructed is
defined by the [core contract](core-contract.md). The current persisted format
version is `6`.

## Adapter boundary

One database represents exactly one logical memory. `SqliteStore` owns both its
SQLite connection and the completely reconstructed `Memory`:

```rust
pub struct SqliteStore { /* private fields */ }

impl SqliteStore {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, StoreError>;
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError>;
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
only accepted `format_version` is `6`. Every other version is rejected with
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
format version. Only after accepting V6 does it verify rollback-journal mode and
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

## Closed V6 schema

Only these four tables and one explicit index may exist:

```sql
CREATE TABLE memory_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL CHECK (format_version = 6),
    memory_id BLOB NOT NULL
        CHECK (
            typeof(memory_id) = 'blob'
            AND length(memory_id) = 16
            AND memory_id != zeroblob(16)
        ),
    snapshot_revision INTEGER NOT NULL CHECK (snapshot_revision >= 0)
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
```

SQLite's primary-key autoindexes are expected. Any additional table, index,
view, trigger, virtual table, or altered schema definition makes the store
invalid. `memory_meta` contains exactly the singleton row `(1, 6, memory_id,
revision)`.

## Canonical scalar encoding

- `MemoryId` is its 16-byte big-endian representation.
- Episode sequences and `SymbolId` values are fixed-width 8-byte big-endian
  unsigned integers.
- Timestamps are fixed-width 8-byte big-endian two's-complement signed Unix
  milliseconds. The complete `i64` range is preserved.
- Feedback history is an unsigned 16-bit bitset stored as a SQLite integer.
- Feedback sample count is an integer in `1..=16`.

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

The normalization algorithm is part of format V6 and is fixed to Unicode 16.0:

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
alias, stemming, fuzzy matching, accent removal, synonym mapping, or embedding.

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

## Open lifecycle

Opening an existing store performs, in order:

1. Apply and verify session settings.
2. Verify the SQLite application ID.
3. Read metadata and require format V6.
4. Verify durability settings without changing persistent journal mode.
5. Begin one consistent read transaction.
6. Validate memory identity, revision, exact schema, and SQLite `quick_check`.
7. Stream `symbols` in ID order and require fixed-width contiguous IDs,
   canonical normalized text, length bounds, and unique values.
8. Stream `episodes` in sequence order and require the exact prefix `0..N`.
9. Strictly decode every payload, validate every symbol reference, and rebuild
   the corresponding immutable core atom.
10. Validate and restore the complete feedback graph.
11. Commit the read transaction and only then expose the store.

The adapter keeps only the catalog boundary and locally pending assignments in
its long-lived state, not a full text map. The reconstructed memory stores only
numeric symbols. Cue indexes are private derived caches rebuilt from immutable
episodes when recall needs them.

## Save lifecycle

Each store remembers its opened revision, persisted episode count, and symbol
tail. A save verifies durability, then runs one `BEGIN IMMEDIATE` transaction:

1. Recheck application ID, memory ID, expected revision, and the closed schema.
2. Reject revision exhaustion before any mutation.
3. Verify the remembered episode and symbol tails against indexed last-row
   queries.
4. Validate every new episode key and value against the persisted-or-staged
   symbol prefix.
5. Increment the singleton revision exactly once and require one changed row.
6. Insert staged symbols in ascending ID order.
7. Append only episodes at or beyond the remembered episode count.
8. Reconcile feedback in deterministic source/target order.
9. Commit.
10. Only after commit update remembered boundaries and clear staged symbols.

Every successful save increments the revision, including a logical no-op.
Interning never starts an independent transaction. Two sessions may locally
stage different text under the same next ID; the first save advances the shared
revision, and the stale session fails before publishing any symbol, episode, or
feedback state.

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
exhaustion, and unknown symbol references. `StoreIntegrityError` reports file
identity, metadata, schema, encoding, symbol, episode, and feedback violations.
Both enums are non-exhaustive.

The adapter never migrates unsupported formats, accepts a partially valid
snapshot, renumbers identifiers, repairs references, canonicalizes stored
payloads or symbols, drops invalid rows, clamps feedback, or exposes state
before complete validation. Direct SQL modification is outside the supported
API.

V1-V3 lack source text. V4 and V5 have incompatible symbol namespaces and
episode semantics. Converting their role-specific statements to unordered open
attributes would be a potentially lossy semantic transformation, not a safe
format migration; they are therefore rejected.

## CLI boundary

The CLI creates and accepts only format V6. `add`, `recall`, and `feedback`
require an existing database and reconstruct it fully. Singular and batch add
intern keys and values through the shared catalog and publish them with all new
episodes using exactly one save. Feedback also uses exactly one save.

Recall performs no interning and no save. It resolves only distinct symbols in
the returned hits through bounded indexed batches. It emits normalized text and
does not change the revision or database contents. The current implementation
uses the same read-write `SqliteStore::open()` path, so logical read-only behavior
does not imply a read-only SQLite connection.

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
