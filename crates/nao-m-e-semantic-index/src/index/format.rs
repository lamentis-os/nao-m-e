use std::sync::OnceLock;
use std::time::Duration;

use nao_m_e::MemoryId;
use rusqlite::{Connection, Error, Result, TransactionBehavior, params};

use crate::EmbeddingProfile;

pub(super) const APPLICATION_ID: i64 = 0x4E41_4F53;
pub(super) const FORMAT_VERSION: i64 = 1;

const SCHEMA: &str = r#"
CREATE TABLE semantic_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL CHECK (format_version = 1),
    memory_id BLOB NOT NULL
        CHECK (
            typeof(memory_id) = 'blob'
            AND length(memory_id) = 16
            AND memory_id != zeroblob(16)
        ),
    profile_fingerprint BLOB NOT NULL
        CHECK (
            typeof(profile_fingerprint) = 'blob'
            AND length(profile_fingerprint) = 32
            AND profile_fingerprint != zeroblob(32)
        ),
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 65535),
    indexed_episode_count BLOB NOT NULL
        CHECK (
            typeof(indexed_episode_count) = 'blob'
            AND length(indexed_episode_count) = 8
        )
) STRICT, WITHOUT ROWID;

CREATE TABLE semantic_cues (
    cue_id BLOB PRIMARY KEY
        CHECK (typeof(cue_id) = 'blob' AND length(cue_id) = 8),
    key_id BLOB NOT NULL
        CHECK (typeof(key_id) = 'blob' AND length(key_id) = 8),
    value_id BLOB NOT NULL
        CHECK (typeof(value_id) = 'blob' AND length(value_id) = 8),
    vector BLOB NOT NULL
        CHECK (
            typeof(vector) = 'blob'
            AND length(vector) BETWEEN 2 AND 131070
            AND length(vector) % 2 = 0
        )
) STRICT, WITHOUT ROWID;

CREATE UNIQUE INDEX semantic_cue_pair_unique
    ON semantic_cues(key_id, value_id);

CREATE TABLE episode_cues (
    sequence BLOB NOT NULL
        CHECK (typeof(sequence) = 'blob' AND length(sequence) = 8),
    cue_id BLOB NOT NULL
        CHECK (typeof(cue_id) = 'blob' AND length(cue_id) = 8),
    PRIMARY KEY (sequence, cue_id),
    FOREIGN KEY (cue_id) REFERENCES semantic_cues(cue_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT, WITHOUT ROWID;

CREATE INDEX episode_cues_by_cue
    ON episode_cues(cue_id, sequence);
"#;

static CANONICAL_SCHEMA: OnceLock<Vec<SchemaObject>> = OnceLock::new();

const SCHEMA_OBJECTS: [(&str, &str, &str); 5] = [
    ("table", "semantic_meta", "semantic_meta"),
    ("table", "semantic_cues", "semantic_cues"),
    ("index", "semantic_cue_pair_unique", "semantic_cues"),
    ("table", "episode_cues", "episode_cues"),
    ("index", "episode_cues_by_cue", "episode_cues"),
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct SchemaObject {
    object_type: String,
    name: String,
    table_name: String,
    normalized_sql: Option<String>,
}

pub(super) fn configure_session(connection: &Connection) -> Result<()> {
    connection.busy_timeout(Duration::ZERO)?;
    connection.pragma_update(None, "foreign_keys", true)?;
    connection.pragma_update(None, "trusted_schema", false)?;
    connection.pragma_update(None, "ignore_check_constraints", false)?;

    verify_integer_pragma(connection, "foreign_keys", 1)?;
    verify_integer_pragma(connection, "trusted_schema", 0)?;
    verify_integer_pragma(connection, "ignore_check_constraints", 0)?;
    verify_integer_pragma(connection, "busy_timeout", 0)
}

pub(super) fn configure_durability(connection: &Connection) -> Result<()> {
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

pub(super) fn verify_durability(connection: &Connection) -> Result<()> {
    let journal_mode: String =
        connection.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
    if !journal_mode.eq_ignore_ascii_case("delete") {
        return Err(Error::InvalidQuery);
    }
    configure_synchronous(connection)
}

pub(super) fn read_application_id(connection: &Connection) -> Result<i64> {
    connection.pragma_query_value(None, "application_id", |row| row.get(0))
}

pub(super) fn create_schema(
    connection: &mut Connection,
    memory_id: MemoryId,
    profile: EmbeddingProfile,
) -> Result<()> {
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.pragma_update(None, "application_id", APPLICATION_ID)?;
    transaction.execute_batch(SCHEMA)?;
    transaction.execute(
        "INSERT INTO semantic_meta (
            singleton,
            format_version,
            memory_id,
            profile_fingerprint,
            dimensions,
            indexed_episode_count
        ) VALUES (1, ?1, ?2, ?3, ?4, ?5)",
        params![
            FORMAT_VERSION,
            memory_id.to_be_bytes().as_slice(),
            profile.fingerprint().as_slice(),
            i64::from(profile.dimensions()),
            encode_u64(0).as_slice(),
        ],
    )?;
    transaction.commit()?;
    verify_integer_pragma(connection, "application_id", APPLICATION_ID)
}

pub(super) fn validate_schema(connection: &Connection) -> Result<bool> {
    Ok(read_schema_objects(connection)? == *canonical_schema_objects())
}

pub(super) const fn encode_u64(value: u64) -> [u8; 8] {
    value.to_be_bytes()
}

pub(super) fn encode_vector(values: &[i16], bytes: &mut Vec<u8>) {
    bytes.clear();
    bytes.reserve(values.len() * 2);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

pub(super) fn validate_vector(bytes: &[u8], dimensions: u16) -> bool {
    if bytes.len() != usize::from(dimensions) * 2 {
        return false;
    }
    bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .any(|value| value != 0)
}

fn configure_synchronous(connection: &Connection) -> Result<()> {
    connection.pragma_update(None, "synchronous", "EXTRA")?;
    verify_integer_pragma(connection, "synchronous", 3)
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
mod tests {
    use super::*;

    #[test]
    fn vectors_use_canonical_little_endian_i16() {
        let values = [-32_768, -1, 1, 32_767];
        let mut bytes = Vec::new();
        encode_vector(&values, &mut bytes);
        assert_eq!(bytes, [0x00, 0x80, 0xff, 0xff, 0x01, 0x00, 0xff, 0x7f]);
        assert!(validate_vector(&bytes, 4));
        assert!(!validate_vector(&bytes, 3));
        assert!(!validate_vector(&[0, 0, 0, 0], 2));
    }
}
