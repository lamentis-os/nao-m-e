use std::sync::OnceLock;
use std::time::Duration;

use nao_m_e::MemoryId;
use rusqlite::{Connection, Error, Result, TransactionBehavior, params};

use crate::codec::encode_memory_id;

pub(crate) const APPLICATION_ID: i64 = 0x4E41_4F4D;
pub(crate) const FORMAT_VERSION: i64 = 2;

const SCHEMA: &str = r#"
CREATE TABLE memory_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL CHECK (format_version = 2),
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
"#;

static CANONICAL_SCHEMA: OnceLock<Vec<SchemaObject>> = OnceLock::new();
const SCHEMA_TABLE_NAMES: [&str; 3] = ["memory_meta", "episodes", "relevance_edges"];

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
        assert_eq!(definitions.len(), SCHEMA_TABLE_NAMES.len());
        let mut objects: Vec<_> = SCHEMA_TABLE_NAMES
            .into_iter()
            .zip(definitions)
            .map(|(name, definition)| SchemaObject {
                object_type: "table".to_owned(),
                name: name.to_owned(),
                table_name: name.to_owned(),
                normalized_sql: Some(normalize_sql(definition)),
            })
            .collect();
        objects.sort_unstable_by(|left, right| left.name.cmp(&right.name));
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

    fn open_temporary_database() -> (tempfile::TempDir, Connection) {
        let directory = tempfile::tempdir().expect("temporary directory is available");
        let connection = Connection::open(directory.path().join("memory.sqlite"))
            .expect("temporary database opens");
        (directory, connection)
    }

    fn configure(connection: &Connection) {
        configure_session(connection).expect("session configuration succeeds");
        configure_durability(connection).expect("durability configuration succeeds");
    }

    #[test]
    fn session_and_durability_settings_are_verified() {
        let (_directory, connection) = open_temporary_database();

        configure(&connection);

        for (name, expected) in [
            ("foreign_keys", 1),
            ("trusted_schema", 0),
            ("ignore_check_constraints", 0),
            ("busy_timeout", 0),
            ("synchronous", 3),
        ] {
            let actual: i64 = connection
                .pragma_query_value(None, name, |row| row.get(0))
                .expect("configured pragma can be read");
            assert_eq!(actual, expected, "unexpected {name}");
        }
        let journal_mode: String = connection
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .expect("journal mode can be read");
        assert_eq!(journal_mode.to_ascii_lowercase(), "delete");
    }

    #[test]
    fn schema_creation_commits_v2_identity_and_closed_shape() {
        let (_directory, mut connection) = open_temporary_database();
        let memory_id = MemoryId::new(7).unwrap();
        configure(&connection);

        create_schema(&mut connection, memory_id).unwrap();

        assert_eq!(read_application_id(&connection).unwrap(), APPLICATION_ID);
        assert!(validate_schema(&connection).unwrap());
        let metadata: (i64, Vec<u8>, i64) = connection
            .query_row(
                "SELECT format_version, memory_id, snapshot_revision FROM memory_meta",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(metadata.0, FORMAT_VERSION);
        assert_eq!(metadata.1, encode_memory_id(memory_id));
        assert_eq!(metadata.2, 0);
        let tables: Vec<String> = connection
            .prepare("SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_>>()
            .unwrap();
        assert_eq!(tables, ["episodes", "memory_meta", "relevance_edges"]);
    }

    #[test]
    fn failed_schema_creation_rolls_back_identity_and_ddl() {
        let (_directory, mut connection) = open_temporary_database();
        connection
            .execute("CREATE TABLE episodes (value INTEGER)", [])
            .unwrap();
        configure(&connection);

        assert!(create_schema(&mut connection, MemoryId::new(1).unwrap()).is_err());

        assert_eq!(read_application_id(&connection).unwrap(), 0);
        assert!(
            connection
                .query_row("SELECT 1 FROM memory_meta", [], |_| Ok(()))
                .is_err()
        );
    }

    #[test]
    fn validation_accepts_whitespace_only_variation() {
        let (_directory, connection) = open_temporary_database();
        connection
            .execute_batch(&SCHEMA.replace("CREATE TABLE", "CREATE  TABLE"))
            .unwrap();

        assert!(validate_schema(&connection).unwrap());
    }

    #[test]
    fn validation_rejects_schema_drift() {
        let mutations = [
            SCHEMA.replacen("payload BLOB NOT NULL", "payload TEXT NOT NULL", 1),
            SCHEMA.replacen(
                "CHECK (weight_ppm BETWEEN 1 AND 1000000)",
                "CHECK (weight_ppm BETWEEN 0 AND 1000000)",
                1,
            ),
            SCHEMA.replacen(
                "PRIMARY KEY (from_sequence, to_sequence)",
                "PRIMARY KEY (to_sequence, from_sequence)",
                1,
            ),
            SCHEMA.replacen(
                "ON UPDATE RESTRICT ON DELETE RESTRICT",
                "ON UPDATE RESTRICT ON DELETE CASCADE",
                1,
            ),
        ];

        for schema in mutations {
            let (_directory, connection) = open_temporary_database();
            connection.execute_batch(&schema).unwrap();
            assert!(!validate_schema(&connection).unwrap());
        }
    }

    #[test]
    fn validation_rejects_additional_persistent_objects() {
        for object in [
            "CREATE TABLE extra (value INTEGER) STRICT",
            "CREATE INDEX extra_index ON relevance_edges(weight_ppm)",
            "CREATE VIEW extra_view AS SELECT * FROM memory_meta",
            "CREATE TRIGGER extra_trigger AFTER UPDATE ON memory_meta BEGIN SELECT 1; END",
        ] {
            let (_directory, connection) = open_temporary_database();
            connection.execute_batch(SCHEMA).unwrap();
            connection.execute_batch(object).unwrap();
            assert!(!validate_schema(&connection).unwrap(), "accepted {object}");
        }
    }

    #[test]
    fn constraints_enforce_canonical_keys_and_edges() {
        let (_directory, mut connection) = open_temporary_database();
        configure(&connection);
        create_schema(&mut connection, MemoryId::new(1).unwrap()).unwrap();
        let zero = [0_u8; 8];
        let one = [0_u8, 0, 0, 0, 0, 0, 0, 1];

        assert!(
            connection
                .execute(
                    "INSERT INTO episodes (sequence, payload) VALUES (?1, ?2)",
                    params![[0_u8; 7].as_slice(), [1_u8].as_slice()],
                )
                .is_err()
        );
        assert!(
            connection
                .execute(
                    "INSERT INTO episodes (sequence, payload) VALUES (?1, ?2)",
                    params![zero.as_slice(), [].as_slice()],
                )
                .is_err()
        );
        connection
            .execute(
                "INSERT INTO episodes (sequence, payload) VALUES (?1, ?2)",
                params![zero.as_slice(), [1_u8].as_slice()],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO episodes (sequence, payload) VALUES (?1, ?2)",
                params![one.as_slice(), [1_u8].as_slice()],
            )
            .unwrap();
        assert!(
            connection
                .execute(
                    "INSERT INTO relevance_edges VALUES (?1, ?1, 1)",
                    [zero.as_slice()],
                )
                .is_err()
        );
        connection
            .execute(
                "INSERT INTO relevance_edges VALUES (?1, ?2, 1000000)",
                params![zero.as_slice(), one.as_slice()],
            )
            .unwrap();
        assert!(
            connection
                .execute(
                    "INSERT INTO relevance_edges VALUES (?1, ?2, 1)",
                    params![zero.as_slice(), [0xff_u8; 8].as_slice()],
                )
                .is_err()
        );
    }
}
