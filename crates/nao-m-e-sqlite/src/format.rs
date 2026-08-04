mod codec;

use codec::encode_memory_id;
pub(crate) use codec::{
    MIN_EPISODE_PAYLOAD_BYTES, decode_episode, decode_memory_id, decode_u64, encode_episode,
    encode_u64,
};

use std::sync::OnceLock;
use std::time::Duration;

use nao_m_e::MemoryId;
use rusqlite::{Connection, Error, Result, TransactionBehavior, params};

pub(crate) const APPLICATION_ID: i64 = 0x4E41_4F4D;
pub(crate) const FORMAT_VERSION: i64 = 5;
pub(crate) const MAX_SYMBOL_BYTES: usize = 4_096;

const SCHEMA: &str = r#"
CREATE TABLE memory_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL CHECK (format_version = 5),
    memory_id BLOB NOT NULL
        CHECK (
            typeof(memory_id) = 'blob'
            AND length(memory_id) = 16
            AND memory_id != zeroblob(16)
        ),
    snapshot_revision INTEGER NOT NULL CHECK (snapshot_revision >= 0)
) STRICT, WITHOUT ROWID;

CREATE TABLE predicates (
    id BLOB PRIMARY KEY
        CHECK (typeof(id) = 'blob' AND length(id) = 8),
    value TEXT NOT NULL
        CHECK (
            typeof(value) = 'text'
            AND length(CAST(value AS BLOB)) BETWEEN 1 AND 4096
        )
) STRICT, WITHOUT ROWID;

CREATE UNIQUE INDEX predicates_value_unique
    ON predicates(value COLLATE BINARY);

CREATE TABLE terms (
    id BLOB PRIMARY KEY
        CHECK (typeof(id) = 'blob' AND length(id) = 8),
    value TEXT NOT NULL
        CHECK (
            typeof(value) = 'text'
            AND length(CAST(value AS BLOB)) BETWEEN 1 AND 4096
        )
) STRICT, WITHOUT ROWID;

CREATE UNIQUE INDEX terms_value_unique
    ON terms(value COLLATE BINARY);

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
"#;

static CANONICAL_SCHEMA: OnceLock<Vec<SchemaObject>> = OnceLock::new();
const SCHEMA_OBJECTS: [(&str, &str, &str); 7] = [
    ("table", "memory_meta", "memory_meta"),
    ("table", "predicates", "predicates"),
    ("index", "predicates_value_unique", "predicates"),
    ("table", "terms", "terms"),
    ("index", "terms_value_unique", "terms"),
    ("table", "episodes", "episodes"),
    ("table", "feedback_edges", "feedback_edges"),
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct SchemaObject {
    object_type: String,
    name: String,
    table_name: String,
    normalized_sql: Option<String>,
}

pub(crate) fn configure_session(connection: &Connection) -> Result<()> {
    connection.busy_timeout(Duration::ZERO)?;
    connection.pragma_update(None, "foreign_keys", true)?;
    connection.pragma_update(None, "trusted_schema", false)?;
    connection.pragma_update(None, "ignore_check_constraints", false)?;

    verify_integer_pragma(connection, "foreign_keys", 1)?;
    verify_integer_pragma(connection, "trusted_schema", 0)?;
    verify_integer_pragma(connection, "ignore_check_constraints", 0)?;
    verify_integer_pragma(connection, "busy_timeout", 0)
}

pub(crate) fn configure_durability(connection: &Connection) -> Result<()> {
    let journal_mode: String =
        connection.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
    if !journal_mode.eq_ignore_ascii_case("delete") {
        let configured: String =
            connection.query_row("PRAGMA journal_mode = DELETE", [], |row| row.get(0))?;
        if !configured.eq_ignore_ascii_case("delete") {
            return Err(Error::InvalidQuery);
        }
    }

    configure_synchronous(connection)
}

pub(crate) fn verify_durability(connection: &Connection) -> Result<()> {
    let journal_mode: String =
        connection.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
    if !journal_mode.eq_ignore_ascii_case("delete") {
        return Err(Error::InvalidQuery);
    }

    configure_synchronous(connection)
}

fn configure_synchronous(connection: &Connection) -> Result<()> {
    connection.pragma_update(None, "synchronous", "EXTRA")?;
    verify_integer_pragma(connection, "synchronous", 3)
}

pub(crate) fn read_application_id(connection: &Connection) -> Result<i64> {
    connection.pragma_query_value(None, "application_id", |row| row.get(0))
}

pub(crate) fn create_schema(connection: &mut Connection, memory_id: MemoryId) -> Result<()> {
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.pragma_update(None, "application_id", APPLICATION_ID)?;
    transaction.execute_batch(SCHEMA)?;
    transaction.execute(
        "INSERT INTO memory_meta (
            singleton,
            format_version,
            memory_id,
            snapshot_revision
        ) VALUES (1, ?1, ?2, 0)",
        params![FORMAT_VERSION, encode_memory_id(memory_id).as_slice()],
    )?;
    transaction.commit()?;
    verify_integer_pragma(connection, "application_id", APPLICATION_ID)
}

pub(crate) fn validate_schema(connection: &Connection) -> Result<bool> {
    Ok(read_schema_objects(connection)? == *canonical_schema_objects())
}

fn canonical_schema_objects() -> &'static Vec<SchemaObject> {
    CANONICAL_SCHEMA.get_or_init(|| {
        let definitions: Vec<_> = SCHEMA
            .split(';')
            .map(str::trim)
            .filter(|definition| !definition.is_empty())
            .collect();
        assert_eq!(definitions.len(), SCHEMA_OBJECTS.len());
        let mut objects: Vec<_> = SCHEMA_OBJECTS
            .into_iter()
            .zip(definitions)
            .map(
                |((object_type, name, table_name), definition)| SchemaObject {
                    object_type: object_type.to_owned(),
                    name: name.to_owned(),
                    table_name: table_name.to_owned(),
                    normalized_sql: Some(normalize_sql(definition)),
                },
            )
            .collect();
        objects.sort_unstable_by(|left, right| {
            left.object_type
                .cmp(&right.object_type)
                .then_with(|| left.name.cmp(&right.name))
        });
        objects
    })
}

fn read_schema_objects(connection: &Connection) -> Result<Vec<SchemaObject>> {
    let mut statement = connection.prepare(
        "SELECT type, name, tbl_name, sql
         FROM main.sqlite_schema
         ORDER BY type, name",
    )?;
    statement
        .query_map([], |row| {
            Ok(SchemaObject {
                object_type: row.get(0)?,
                name: row.get(1)?,
                table_name: row.get(2)?,
                normalized_sql: row
                    .get::<_, Option<String>>(3)?
                    .map(|sql| normalize_sql(&sql)),
            })
        })?
        .collect()
}

fn normalize_sql(sql: &str) -> String {
    sql.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn verify_integer_pragma(connection: &Connection, name: &str, expected: i64) -> Result<()> {
    let actual: i64 = connection.pragma_query_value(None, name, |row| row.get(0))?;
    if actual == expected {
        Ok(())
    } else {
        Err(Error::InvalidQuery)
    }
}

#[cfg(test)]
mod tests;
