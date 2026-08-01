# SQLite V1 contract

This document defines the durable format and observable lifecycle of the
`nao-m-e-sqlite` adapter. The state being stored and reconstructed remains the
state defined by the [V0 core contract](v0-contract.md).

## Adapter boundary

One SQLite database represents exactly one logical memory. `SqliteStore` owns
both the database connection and the reconstructed `MemoryV0`:

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
retry, while an unsaved state is lost when its process ends.

The adapter does not expose its SQLite connection and does not accept an
independently constructed `MemoryV0` for saving. Database copies carrying the
same `MemoryId` must not be modified independently and later merged.

## Database identity and connection settings

V1 uses SQLite application ID `0x4E414F4D` (`NAOM`, decimal `1312902989`) and
format version `1`. A different application ID or format version is rejected;
V1 performs no migration.

Immediately after opening a connection, and before reading or creating schema
or memory state, the adapter applies and verifies these connection-local safety
settings:

```text
busy_timeout = 0
foreign_keys = ON
trusted_schema = OFF
ignore_check_constraints = OFF
```

For a file-backed target, the adapter then reads the SQLite header's application
ID. Only after accepting that ID as `NAOM` does it apply and verify the settings
that affect persistent journaling behavior:

```text
journal_mode = DELETE
synchronous = EXTRA
```

This ordering prevents an attempted open of an unrelated SQLite database from
changing its journaling mode. A zero busy timeout means the adapter neither
waits nor retries on its own; SQLite locking failures remain immediately
visible to the caller as database errors.

The new-store builder is an in-memory SQLite connection. It receives the safety
settings but deliberately does not receive the file durability settings:
in-memory SQLite uses `journal_mode=MEMORY`, and durability is configured only
when the published target is opened as a file-backed store.

## Canonical encoding

- `MemoryId` is its canonical 16-byte big-endian representation.
- Episode sequences, `PredicateId`, `TermId`, and `SourceId` are fixed-width
  8-byte big-endian unsigned integers.
- Timestamps are SQLite integers containing the unchanged signed `i64` value.
- Activation and influence are SQLite integers containing parts per million.
- An `AtomId` is reconstructed from the database `MemoryId` and an episode
  sequence. Its diagnostic display and Rust memory layout are never stored.

Fixed-width big-endian BLOBs preserve the complete unsigned integer ranges and
sort lexicographically in numeric order. Text, variable-width integer BLOBs,
and native-endian representations are non-canonical.

## Schema

The schema below is the complete V1 storage shape. All foreign-key actions are
restrictive because episode rows are append-only.

```sql
BEGIN IMMEDIATE;

PRAGMA application_id = 1312902989;

CREATE TABLE memory_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL CHECK (format_version >= 1),
    memory_id BLOB NOT NULL
        CHECK (
            typeof(memory_id) = 'blob'
            AND length(memory_id) = 16
            AND memory_id != zeroblob(16)
        ),
    snapshot_revision INTEGER NOT NULL CHECK (snapshot_revision >= 0)
) STRICT;

CREATE TABLE episodes (
    sequence BLOB PRIMARY KEY
        CHECK (typeof(sequence) = 'blob' AND length(sequence) = 8),
    occurred_at_ms INTEGER NOT NULL,
    recorded_at_ms INTEGER NOT NULL,
    source_id BLOB NOT NULL
        CHECK (typeof(source_id) = 'blob' AND length(source_id) = 8)
) STRICT, WITHOUT ROWID;

CREATE TABLE episode_statements (
    episode_sequence BLOB NOT NULL
        CHECK (
            typeof(episode_sequence) = 'blob'
            AND length(episode_sequence) = 8
        ),
    role INTEGER NOT NULL CHECK (role BETWEEN 0 AND 3),
    statement_ordinal INTEGER NOT NULL CHECK (statement_ordinal >= 0),
    predicate_id BLOB NOT NULL
        CHECK (typeof(predicate_id) = 'blob' AND length(predicate_id) = 8),
    PRIMARY KEY (episode_sequence, role, statement_ordinal),
    FOREIGN KEY (episode_sequence) REFERENCES episodes(sequence)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (role = 0 OR statement_ordinal = 0)
) STRICT, WITHOUT ROWID;

CREATE TABLE statement_terms (
    episode_sequence BLOB NOT NULL
        CHECK (
            typeof(episode_sequence) = 'blob'
            AND length(episode_sequence) = 8
        ),
    role INTEGER NOT NULL CHECK (role BETWEEN 0 AND 3),
    statement_ordinal INTEGER NOT NULL CHECK (statement_ordinal >= 0),
    term_ordinal INTEGER NOT NULL CHECK (term_ordinal >= 0),
    term_id BLOB NOT NULL
        CHECK (typeof(term_id) = 'blob' AND length(term_id) = 8),
    PRIMARY KEY (
        episode_sequence,
        role,
        statement_ordinal,
        term_ordinal
    ),
    FOREIGN KEY (episode_sequence, role, statement_ordinal)
        REFERENCES episode_statements (
            episode_sequence,
            role,
            statement_ordinal
        )
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT, WITHOUT ROWID;

CREATE TABLE activations (
    episode_sequence BLOB PRIMARY KEY
        CHECK (
            typeof(episode_sequence) = 'blob'
            AND length(episode_sequence) = 8
        ),
    activation_ppm INTEGER NOT NULL
        CHECK (activation_ppm BETWEEN 0 AND 1000000),
    FOREIGN KEY (episode_sequence) REFERENCES episodes(sequence)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT, WITHOUT ROWID;

CREATE TABLE relevance_edges (
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
    weight_ppm INTEGER NOT NULL CHECK (weight_ppm BETWEEN 1 AND 1000000),
    PRIMARY KEY (from_sequence, to_sequence),
    FOREIGN KEY (from_sequence) REFERENCES episodes(sequence)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (to_sequence) REFERENCES episodes(sequence)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (from_sequence != to_sequence)
) STRICT, WITHOUT ROWID;

INSERT INTO memory_meta (
    singleton,
    format_version,
    memory_id,
    snapshot_revision
) VALUES (1, 1, :memory_id, 0);

COMMIT;
```

`memory_meta` contains exactly one row with `singleton = 1`. Creation inserts
format version `1`, a randomly generated non-zero memory ID, and snapshot
revision `0`. `:memory_id` denotes the bound canonical 16-byte BLOB. The
application ID, all six table definitions, and this singleton row are committed
together or not at all.

Statement roles are:

| Value | Episode field | Cardinality |
|---:|---|---|
| `0` | context | zero or more, ordered by `statement_ordinal` |
| `1` | observation | exactly one, ordinal `0` |
| `2` | action | zero or one, ordinal `0` |
| `3` | outcome | zero or one, ordinal `0` |

Each statement has one or more terms with contiguous ordinals beginning at
zero. Activations are dense: every episode has exactly one activation row,
including zero activation. Relevance is sparse: only positive edges exist.
There are no content dictionaries, content deduplication, or secondary indexes
in V1.

The schema rejects invalid storage classes, lengths, ranges, duplicate keys,
self-edges, and missing endpoints. The adapter additionally validates the
cross-row invariants that SQL does not enforce efficiently. In particular, no
trigger sums outgoing relevance weights for each inserted edge.

The persisted V1 application schema is closed. It contains exactly the six
user tables above and no additional persistent user table, view, trigger, or
user-defined index. SQLite-owned internal objects and automatic indexes are not
user extensions. On `open()` and before every `save()`, the adapter validates
the schema through SQLite PRAGMA and `main.sqlite_schema` introspection. It
compares whitespace-normalized `CREATE TABLE` SQL with the canonical statements
derived from the same internal `SCHEMA` constant, so whitespace-only formatting
changes are accepted but token or CHECK-constraint drift is rejected. It also
verifies the table inventory, `STRICT` and `WITHOUT ROWID` properties, column
names, declared types, nullability, default and hidden-column absence,
primary-key positions, and complete foreign-key definitions.

## Create and open

`SqliteStore::create` requires an absent target path but never initializes that
path in place. It generates a `MemoryId` from operating-system entropy,
retrying only if the all-zero value is returned, and builds the canonical empty
database in `Connection::open_in_memory()`. The in-memory connection receives
the safety settings; one transaction commits the application ID, schema,
metadata, and initial empty snapshot. The adapter then verifies identity,
canonical schema, quick check, foreign keys, and revision zero before
serializing SQLite's main database image.

The serialized bytes are written through a privately held `NamedTempFile` in
the target directory. The adapter flushes the stream and calls `sync_all()` on
that file before `persist_noclobber` publishes it at the requested path without
replacing an existing entry. A pre-existing target, or one created
concurrently, is never replaced or truncated. After publication, the returned
file handle is dropped and `SqliteStore::open(target)` performs the normal
file-backed validation and applies the durability settings.

The adapter never deletes the requested target on an error path, including an
error from the final `open()`. Before publication, ordinary failures clean up
only the private staging file. The directory containing the target is a
caller-controlled trust boundary; another actor able to mutate that directory
can alter the path after publication.

`SqliteStore::open` requires an existing target and never creates a missing
database. It opens the file read-write because every returned store supports
`save()`; V1 has no read-only mode. Before the first schema or state read, it
applies the safety settings. It reads and accepts the application ID before
applying the durability settings, then opens a consistent read transaction and
performs these checks:

1. The persisted application schema is canonical V1 and contains no additional
   persistent user object, including a trigger or view.
2. `memory_meta` contains exactly one supported metadata row with a non-zero
   16-byte ID.
3. `PRAGMA quick_check` returns only `ok` and `PRAGMA foreign_key_check`
   returns no violations.
4. Episode sequences are exactly `0..N-1` without gaps.
5. Every episode has exactly one observation, at most one action and outcome,
   and contiguous context ordinals.
6. Every statement has contiguous term ordinals and at least one term.
7. Stored context is already strictly increasing and duplicate-free under the
   statement order defined by the V0 core contract. Corrupt context is rejected
   rather than silently repaired by insertion canonicalization.
8. Every episode has exactly one activation row.
9. Relevance endpoints exist, edges are unique and non-reflexive, and each
   source's total outgoing weight is at most `SCALE`.

The adapter reads each table in ordered, set-based passes and reconstructs
episodes with linear cursors; it does not issue one query per episode. Episode
rows and statement structure are validated as they are consumed into a local
`MemoryV0`:

1. Create it with the stored `MemoryId`.
2. Insert episodes in sequence order and verify every returned `AtomId`.
3. Apply each positive stored activation once to its initially zero atom.
4. Install relevance edges in source and target order through the core API.

Any rejection by the core during reconstruction is invalid stored data. The
local reconstruction is not exposed until every storage-level and core
validation succeeds; any later activation or relevance error discards it, so
the adapter returns no partial memory. If a snapshot contains multiple
independent violations, which violation is reported first is unspecified. The
transition scratch buffer is not stored because every logical `step()`
overwrites it before use.

## Saving and writer exclusion

Each opened store remembers its loaded `snapshot_revision` and immutable
episode count. `save()` starts `BEGIN IMMEDIATE` and performs one transaction:

1. Revalidate the application ID, canonical V1 schema, and format version. Any
   additional persistent trigger or view rejects the save before state is
   changed.
2. Read the persisted memory ID and current revision. A memory ID different
   from the owned `MemoryV0` is invalid metadata; a changed revision is a stale
   writer and fails with `StoreError::ConcurrentModification`.
3. Reject `i64::MAX` with `StoreError::RevisionExhausted`.
4. Read only the greatest stored episode sequence and compare it with the
   remembered episode count. A changed append boundary is invalid stored data.
5. Compare-and-swap the expected revision to its successor with both the
   expected revision and owned canonical memory-ID bytes in the SQL predicate.
   Zero changed rows are re-read and classified as identity corruption or a
   concurrent revision change.
6. Append only episodes at or beyond the remembered count, including their
   statements and terms, through reused prepared statements.
7. Replace the complete dense activation table.
8. Replace the complete sparse relevance table.
9. Commit, then update the store's remembered revision and episode count.

The append-boundary query uses the ordered episode primary key and does not
count or scan the immutable prefix. It protects the supported writer path in
`O(log N)`; direct SQL edits remain unsupported and the next `open` performs
the complete fail-closed validation.

The expected save cost is:

```text
O(log N + new episode data + all activations + all relevance edges)
```

The immutable episode prefix is not rewritten. Dirty tracking and an event log
are outside V1. If any statement fails, the transaction rolls back and the
previous database snapshot remains the visible durable state. The in-memory
changes are retained for retry.

SQLite locking excludes simultaneous writes, while the persisted revision also
rejects a store that was opened before another supported writer committed. V1
does not merge divergent histories or provide last-writer-wins behavior.

## Errors and corruption

`StoreError` distinguishes operating-system I/O, SQLite, entropy, invalid-store,
concurrent-modification, and exhausted-revision failures. Opening a missing
path and creating an existing path both fail, but the wrapped platform error
category is not part of the portable contract. Wrapped source errors remain
available through `std::error::Error::source`.

`StoreIntegrityError` reports application mismatch, missing metadata,
unsupported format version, failed quick check, foreign-key violations,
invalid fixed-width encoding, invalid memory ID, invalid metadata,
non-contiguous episode sequences, invalid episodes, invalid activations, or
invalid relevance. A non-canonical schema is classified as invalid metadata.
Both error enums are non-exhaustive so that future releases can report
corruption more precisely without making callers treat the variants as a
closed format definition.

The adapter fails closed: it does not accept a non-canonical persisted schema,
renumber episodes, canonicalize malformed stored context, drop invalid rows,
infer missing activation, reduce relevance weights, or expose a partially
reconstructed memory. Direct SQL modification is not a supported API.

## Worked snapshot

Suppose a database has memory ID
`00112233445566778899aabbccddeeff`. It contains these atoms:

- Sequence `0`: timestamps `1000` and `1001`, source `7`, context statement
  `10(100, 101)`, and observation `20(200)`.
- Sequence `1`: timestamps `2000` and `2001`, source `8`, observation
  `21(201)`, action `30(300)`, and outcome `40(400)`.
- Activations are `750000` and `125000`; sequence `0` influences sequence `1`
  with weight `250000`.

After the first save, selecting BLOB columns through SQLite's `hex()` function
shows the following concrete rows. The displayed hexadecimal strings are a
diagnostic rendering of the stored BLOBs, not TEXT values in the schema.

`memory_meta`:

| singleton | format_version | hex(memory_id) | snapshot_revision |
|---:|---:|---|---:|
| 1 | 1 | `00112233445566778899AABBCCDDEEFF` | 1 |

`episodes`:

| hex(sequence) | occurred_at_ms | recorded_at_ms | hex(source_id) |
|---|---:|---:|---|
| `0000000000000000` | 1000 | 1001 | `0000000000000007` |
| `0000000000000001` | 2000 | 2001 | `0000000000000008` |

`episode_statements`:

| episode | role | statement_ordinal | predicate |
|---|---:|---:|---|
| `0000000000000000` | 0 | 0 | `000000000000000A` |
| `0000000000000000` | 1 | 0 | `0000000000000014` |
| `0000000000000001` | 1 | 0 | `0000000000000015` |
| `0000000000000001` | 2 | 0 | `000000000000001E` |
| `0000000000000001` | 3 | 0 | `0000000000000028` |

`statement_terms`:

| episode | role | statement | term_ordinal | term |
|---|---:|---:|---:|---|
| `0000000000000000` | 0 | 0 | 0 | `0000000000000064` |
| `0000000000000000` | 0 | 0 | 1 | `0000000000000065` |
| `0000000000000000` | 1 | 0 | 0 | `00000000000000C8` |
| `0000000000000001` | 1 | 0 | 0 | `00000000000000C9` |
| `0000000000000001` | 2 | 0 | 0 | `000000000000012C` |
| `0000000000000001` | 3 | 0 | 0 | `0000000000000190` |

`activations`:

| episode | activation_ppm |
|---|---:|
| `0000000000000000` | 750000 |
| `0000000000000001` | 125000 |

`relevance_edges`:

| from | to | weight_ppm |
|---|---|---:|
| `0000000000000000` | `0000000000000001` | 250000 |

Reopening reconstructs atom IDs `(memory ID, 0)` and `(memory ID, 1)`, their
current activations, and the directed edge. A subsequent insertion receives
sequence `2`. Running the next logical step before or after reopening therefore
has the same observable result.

## Durability boundary

A save is one SQLite transaction in rollback-journal mode with
`synchronous=EXTRA`. This configuration targets SQLite's strongest applicable
rollback-journal synchronization behavior, including additional directory
synchronization after journal removal. It does not make unsaved mutations
durable and is not evidence of behavior for every storage device, filesystem,
operating system, or sudden-power-loss scenario.

Creation does not apply these file durability PRAGMAs to the in-memory builder.
Instead, it serializes a validated database image, writes it into the private
staging file, flushes it, and calls `sync_all()` before target-no-clobber
publication. A successful publication therefore exposes the complete staged
image rather than an incrementally initialized target.

The adapter does not `fsync` the parent directory. On a platform or filesystem
where `persist_noclobber` uses a hard-link fallback, a temporary hard link can
remain after a cleanup failure or process crash. These publication mechanics do
not establish survival of the target directory entry under physical power loss.
The caller is responsible for the trust and lifecycle of the parent directory.
