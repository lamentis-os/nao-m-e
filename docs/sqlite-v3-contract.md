# SQLite V3 contract

This document defines the durable format and observable lifecycle of the
`nao-m-e-sqlite` adapter. The state being stored and reconstructed remains the
state defined by the [V0 core contract](v0-contract.md).

## Adapter boundary

One SQLite database represents exactly one logical memory. `SqliteStore` owns
both the database connection and the reconstructed `MemoryV0`. The public API
is:

```rust
pub struct SqliteStore { /* private fields */ }

impl SqliteStore {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, StoreError>;
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError>;
    pub fn memory_id(&self) -> MemoryId;
    pub fn memory(&self) -> &MemoryV0;
    pub fn memory_mut(&mut self) -> &mut MemoryV0;
    pub fn save(&mut self) -> Result<(), StoreError>;
}
```

Mutating the returned `MemoryV0` changes only process memory. `save()` is the
only operation that persists those changes. There is no automatic save on
mutation or drop. A failed save leaves the in-memory state available for a
retry, while unsaved state is lost when its process ends.

Explicit feedback appends one sample to each addressed in-memory feedback
trace. `save()` stores the resulting bounded feedback graph as part of the
complete snapshot transaction; it does not use a separate persistence path.
The database stores the trace's bounded history and sample count, but no
feedback receipt, timestamp, provenance record, or idempotency key. Reapplying
the same feedback after a successful save is a new sample.

The core's cue postings and per-episode cue-weight totals are derived in-memory
data, not snapshot state. SQLite stores only the episode content from which the
core rebuilds them and never persists a cue table, posting list, or structural
recall score.

The adapter does not expose its SQLite connection and does not accept an
independently constructed `MemoryV0` for saving. Database copies carrying the
same `MemoryId` must not be modified independently and later merged.

## Format identity and connection settings

The format uses SQLite application ID `0x4E414F4D` (`NAOM`, decimal
`1312902989`) and `format_version = 3`. This adapter accepts only that format
version. Metadata rows containing `format_version = 1` or `2` are rejected
with `StoreIntegrityError::UnsupportedFormatVersion`, wrapped in
`StoreError::InvalidStore`. The adapter has no migrator and never rewrites an
unsupported store.

Immediately after opening a connection, and before reading or creating schema
or memory state, the adapter applies and verifies these connection-local safety
settings:

```text
busy_timeout = 0
foreign_keys = ON
trusted_schema = OFF
ignore_check_constraints = OFF
```

For an existing file-backed target, the adapter next reads the SQLite header's
application ID and then the metadata format version. Only after accepting the
ID as `NAOM` and the format as V3 does it apply and verify the settings that
affect persistent journaling behavior:

```text
journal_mode = DELETE
synchronous = EXTRA
```

This ordering prevents an attempted open of an unrelated, V1, or V2 SQLite
database from changing its journaling mode. A zero busy timeout means the
adapter neither waits nor retries on its own; SQLite locking failures remain
immediately visible to the caller as database errors.

A newly created database is also file-backed while it is staged. It receives
both the connection-local safety settings and the file durability settings
before its schema and initial snapshot are committed.

## Canonical scalar encoding

- `MemoryId` is its canonical 16-byte big-endian representation.
- Episode sequences, `PredicateId`, `TermId`, and `SourceId` are fixed-width
  8-byte big-endian unsigned integers.
- Timestamps in an episode payload are fixed-width 8-byte big-endian
  two's-complement signed integers containing the unchanged `i64` value.
- Feedback history is an unsigned 16-bit bitset stored as a SQLite integer.
- Feedback sample count is an unsigned integer in `1..=16`.
- An `AtomId` is reconstructed from the database `MemoryId` and an episode
  sequence. Its diagnostic display and Rust memory layout are never stored.

Fixed-width big-endian BLOBs preserve the complete unsigned integer ranges and
sort lexicographically in numeric order. Text, variable-width integer BLOBs,
and native-endian representations are non-canonical.

Payload counts use unsigned LEB128. Each byte contributes its low seven bits,
least-significant group first; bit seven means that another byte follows. The
encoding must be minimal: zero is the single byte `00`, and no multi-byte
encoding may end in a zero seven-bit group. A decoder rejects truncated,
overflowing, overlong, or otherwise non-minimal encodings. Counts are decoded
with checked arithmetic and must fit both the remaining payload and the
in-memory collection size before allocation.

## Whole-episode payload codec

Each `episodes.payload` BLOB contains exactly one complete episode. The byte
layout is the concatenation below; no padding or native-layout bytes occur:

```text
EpisodePayload =
    flags                u8
    occurred_at_ms       i64be
    recorded_at_ms       i64be
    source_id            u64be
    context_count        uleb128
    context              Statement * context_count
    observation          Statement
    action               Statement * flags.bit0
    outcome              Statement * flags.bit1

Statement =
    predicate_id         u64be
    argument_count       uleb128
    arguments            u64be * argument_count
```

In `flags`, bit 0 denotes one following action statement and bit 1 denotes one
following outcome statement. All other bits are reserved and must be zero. An
unset bit means that the corresponding statement is absent; there are no
separate presence tags. Every statement has at least one argument. The context
count may be zero. Stored context must already be strictly increasing and
duplicate-free under the statement order in the V0 core contract. The decoder
does not repair or canonicalize it.

The decoder must consume the payload exactly. Missing bytes, trailing bytes,
reserved flag bits, empty statements, impossible counts, arithmetic overflow,
and non-canonical scalar encodings reject the complete store. The format
version in `memory_meta` versions this codec; an episode payload has no
independent magic or version field.

One encoded episode must fit SQLite's effective BLOB and row-length limit. V3
does not split an oversized episode across rows or introduce overflow chunks;
if SQLite rejects the bound payload, the enclosing save transaction rolls back
and the unsaved in-memory state remains available to the caller.

## Schema

The following three tables are the complete application schema. Every table is
both `STRICT` and `WITHOUT ROWID`. All foreign-key actions are restrictive
because episode rows are append-only.

```sql
BEGIN IMMEDIATE;

PRAGMA application_id = 1312902989;

CREATE TABLE memory_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL CHECK (format_version = 3),
    memory_id BLOB NOT NULL
        CHECK (
            typeof(memory_id) = 'blob'
            AND length(memory_id) = 16
            AND memory_id != zeroblob(16)
        ),
    snapshot_revision INTEGER NOT NULL CHECK (snapshot_revision >= 0)
) STRICT, WITHOUT ROWID;

CREATE TABLE episodes (
    sequence BLOB PRIMARY KEY
        CHECK (typeof(sequence) = 'blob' AND length(sequence) = 8),
    payload BLOB NOT NULL
        CHECK (typeof(payload) = 'blob' AND length(payload) > 0)
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

INSERT INTO memory_meta (
    singleton,
    format_version,
    memory_id,
    snapshot_revision
) VALUES (1, 3, :memory_id, 0);

COMMIT;
```

`memory_meta` contains exactly one row with `singleton = 1`. Creation inserts a
randomly generated non-zero memory ID and snapshot revision `0`. `:memory_id`
denotes the bound canonical 16-byte BLOB. The application ID, all three table
definitions, and this singleton row are committed together or not at all.

Episode rows are append-only and their sequences form the exact prefix
`0..N-1`. The payload holds the complete immutable episode associated with its
sequence. There are no statement or term tables and no content dictionaries or
content deduplication.

Feedback is sparse and stores one non-empty bounded trace per directed edge.
Bit zero is the newest sample; one means helpful and zero means unhelpful. Bits
at or above `sample_count` must be zero. The schema rejects invalid storage
classes, lengths, ranges, non-canonical bitsets, duplicate keys, self-edges,
and missing endpoints. Feedback traces are independent and have no cross-row
outgoing budget.

The application schema is closed. It contains exactly these three user tables
and no additional persistent user table, view, trigger, or user-defined index.
SQLite-owned internal objects and automatic indexes are not user extensions.
On `open()` and before every `save()`, the adapter validates the complete
`main.sqlite_schema` object inventory and compares whitespace-normalized
`CREATE TABLE` SQL with the three canonical definitions above. The canonical
text includes every column, type, constraint, primary key, foreign key,
`STRICT`, and `WITHOUT ROWID` clause. Whitespace-only formatting differences
may be accepted; any object, token, or constraint drift is rejected.

## Creation

`SqliteStore::create` requires an absent target path and never initializes that
path in place. It generates a `MemoryId` from operating-system entropy,
retrying only if the all-zero value is returned. It then creates a private,
file-backed staging database in the target directory and opens SQLite directly
on that staging path.

The staging connection receives the required session and durability settings.
One immediate transaction commits the application ID, schema, metadata, and
empty snapshot. Before publication, the adapter verifies the application ID,
supported format, canonical closed schema, singleton metadata, revision zero,
empty episode and feedback tables, `PRAGMA quick_check`, and
complete core reconstruction. It then closes the SQLite connection, flushes
the staging file, and calls `sync_all()` on that file.

The validated staging file is published to the requested target with a
no-clobber operation such as `persist_noclobber`. A pre-existing target, or one
created concurrently, is never replaced or truncated. Before publication,
ordinary errors clean up only private staging artifacts. The adapter never
deletes the requested target on an error path, including an error from the
normal `open()` validation performed after publication.

The directory containing the target is a caller-controlled trust boundary.
Another actor able to mutate that directory can alter the path after
publication.

## Opening and reconstruction

`SqliteStore::open` requires an existing target and never creates a missing
database. It opens the file read-write because every returned store supports
`save()`; there is no read-only mode. After applying the connection settings,
it opens a consistent read transaction and validates the snapshot before
returning a store.

After the application ID is accepted, metadata is read before format-specific
schema validation so that a well-formed unsupported `format_version` is
classified as `UnsupportedFormatVersion`. The remaining checks are:

1. `memory_meta` contains exactly one row with a non-zero canonical memory ID
   and a non-negative revision.
2. The application schema is the canonical closed three-table schema.
3. `PRAGMA quick_check` returns exactly one row containing `ok`.
4. Episode sequences are exactly `0..N-1` without gaps, and every payload is a
   canonical, exactly consumed whole-episode encoding.
5. Stored context is already strictly increasing and duplicate-free. Every
   reconstructed episode is accepted by the core without changing its content,
   and the returned `AtomId` has the expected sequence and memory ID.
6. Feedback endpoints exist, edges are unique and non-reflexive, and every
   history bitset and sample count form a canonical `FeedbackTrace`.

Rows are consumed in canonical key order. Reconstruction occurs only in local
state:

1. Create a `MemoryV0` with the stored `MemoryId`.
2. Decode and insert episodes in sequence order, checking every returned
   `AtomId`.
3. Install feedback traces in source and target order through the core API.

Any storage-level or core rejection invalidates the complete snapshot. The
adapter does not expose the reconstructed `MemoryV0` until all rows and all
invariants have been validated. A late feedback error therefore discards the
local reconstruction rather than returning a partial memory. If a snapshot
contains multiple independent violations, which violation is reported first is
unspecified.

## Saving and writer exclusion

Each opened store remembers its loaded `snapshot_revision` and immutable
episode count. `save()` starts `BEGIN IMMEDIATE` and performs one transaction:

1. Revalidate the application ID, supported format, and canonical closed
   schema. An added persistent trigger, view, table, or index rejects the save
   before state changes.
2. Read the persisted memory ID and current revision. A memory ID different
   from the owned `MemoryV0` is invalid metadata. A changed revision is a stale
   writer and fails with `StoreError::ConcurrentModification`.
3. Reject `i64::MAX` with `StoreError::RevisionExhausted`.
4. Read the greatest stored episode sequence and compare it with the remembered
   episode count. A changed append boundary is invalid stored data.
5. Update the singleton revision directly to its successor and require exactly
   one changed row. `BEGIN IMMEDIATE` already excludes another writer, and the
   identity and expected revision were checked inside that same transaction.
6. Append only episodes at or beyond the remembered count, encoding one
   complete payload BLOB per episode.
7. Compare persisted and in-memory feedback in canonical source/target order,
   then delete absent edges, update changed traces, and insert new edges.
8. Commit, then update the store's remembered revision and episode count.

Every successful save increments the revision exactly once, including a save
with no logical state change. The append-boundary query uses the ordered episode
primary key and does not count or scan the immutable prefix. Direct SQL edits
remain unsupported; the next `open()` performs complete fail-closed validation.

Feedback reconciliation performs one ordered `O(E)` comparison because
`memory_mut()` permits direct edge insertion and replacement, and the adapter
deliberately does not duplicate the complete persisted graph or put persistence
dirty tracking into the core. A bounded delta plan retains small change sets
until the SQLite read cursor is closed, then performs `O(D)` row mutations for
`D` inserted, updated, or deleted edges; equal rows are not rewritten on this
path.
If the delta exceeds that fixed internal bound, the adapter discards the plan
and replaces the complete feedback table after validation. This keeps
transient reconciliation memory bounded and avoids pathological row-by-row DML
for a wholesale graph replacement.

Every persisted feedback record encountered by the comparison is decoded and
validated against the previously persisted episode prefix before any feedback
DML begins. An invalid persisted graph
therefore fails the save and rolls back the transaction rather than being
silently repaired.

If any operation fails before commit, the transaction rolls back and the
previous database snapshot remains visible. The in-memory changes remain
available for retry. SQLite locking excludes simultaneous writes, while the
persisted revision rejects a store opened before another supported writer
committed. The adapter does not automatically retry, merge divergent histories,
or use last-writer-wins behavior.

## Errors and fail-closed behavior

`StoreError` distinguishes operating-system I/O, SQLite, entropy,
invalid-store, concurrent-modification, and exhausted-revision failures.
Opening a missing path and creating an existing path both fail, but the wrapped
platform error category is not part of the portable contract. Wrapped source
errors remain available through `std::error::Error::source`.

`StoreIntegrityError` reports application mismatch, missing metadata,
unsupported format version, failed quick check, invalid fixed-width or payload
encoding, invalid memory ID, invalid metadata, non-contiguous episode
sequences, invalid episodes, or invalid feedback. A non-canonical schema is
invalid metadata. Both error enums are non-exhaustive so later releases can
report corruption more precisely
without making callers treat the variants as a closed format definition.

The adapter fails closed. It does not migrate an unsupported format, accept a
non-canonical schema or payload, renumber episodes, canonicalize malformed
stored context, drop invalid rows, clamp stored values, discard feedback
samples, or expose a partially reconstructed memory. Direct SQL modification
is not a supported API.

## CLI boundary

CLI V3 is an argument and text-output contract independent of SQLite
`format_version = 3`. Changing the CLI syntax does not implicitly change,
accept, or migrate the persisted format. `init` creates an empty database in
the format defined here and is silent on success. `add`, `recall`, and
`feedback` require an existing store and open and reconstruct its complete
snapshot before executing. Singular `add` commits one episode in one save,
while `add --many` commits all validated input episodes in one save or none of
them. Feedback also uses one save. Recall does not save. Add output begins only
after its save commits and is not part of the SQLite transaction; a later
standard-output failure cannot roll back that commit. Recall can return
cue-derived hits without feedback edges or a schema migration.

## Durability boundary

A save is one SQLite transaction in rollback-journal mode with
`synchronous=EXTRA`. This configuration targets SQLite's strongest applicable
rollback-journal synchronization behavior, including additional directory
synchronization after journal removal. It does not make unsaved mutations
durable and is not evidence of behavior for every storage device, filesystem,
operating system, or sudden-power-loss scenario.

Creation applies the same file durability settings directly to the private
file-backed staging database. The committed and validated staging database is
closed and its file is synchronized before no-clobber publication. A successful
publication therefore exposes a complete staged SQLite database rather than an
incrementally initialized target.

The adapter does not `fsync` the parent directory. On a platform or filesystem
where no-clobber publication uses a hard-link fallback, a temporary hard link
can remain after a cleanup failure or process crash. These publication
mechanics do not establish survival of the target directory entry under
physical power loss. The caller is responsible for the trust and lifecycle of
the parent directory.
