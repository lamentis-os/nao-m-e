use std::sync::OnceLock;
use std::time::Duration;

use nao_m_e::MemoryId;
use rusqlite::{Connection, Error, Result, TransactionBehavior, params};

use crate::codec::encode_memory_id;

pub(crate) const APPLICATION_ID: i64 = 0x4E41_4F4D;
pub(crate) const FORMAT_VERSION: i64 = 1;

const SCHEMA: &str = r#"
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
"#;

static CANONICAL_SCHEMA: OnceLock<SchemaShape> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq)]
struct SchemaShape {
    objects: Vec<SchemaObject>,
    tables: Vec<TableShape>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SchemaObject {
    object_type: String,
    name: String,
    table_name: String,
    normalized_sql: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TableShape {
    name: String,
    table_type: String,
    column_count: i64,
    without_rowid: i64,
    strict: i64,
    columns: Vec<ColumnShape>,
    foreign_keys: Vec<ForeignKeyShape>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ColumnShape {
    cid: i64,
    name: String,
    declared_type: String,
    not_null: i64,
    default_value: Option<String>,
    primary_key_ordinal: i64,
    hidden: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ForeignKeyShape {
    id: i64,
    sequence: i64,
    parent_table: String,
    child_column: String,
    parent_column: String,
    on_update: String,
    on_delete: String,
    match_clause: String,
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
        connection.query_row("PRAGMA journal_mode = DELETE", [], |row| row.get(0))?;
    if !journal_mode.eq_ignore_ascii_case("delete") {
        return Err(Error::InvalidQuery);
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
    Ok(read_schema_shape(connection)? == *canonical_schema_shape()?)
}

fn canonical_schema_shape() -> Result<&'static SchemaShape> {
    if let Some(schema) = CANONICAL_SCHEMA.get() {
        return Ok(schema);
    }
    let connection = Connection::open_in_memory()?;
    connection.execute_batch(SCHEMA)?;
    let schema = read_schema_shape(&connection)?;
    Ok(CANONICAL_SCHEMA.get_or_init(|| schema))
}

fn read_schema_shape(connection: &Connection) -> Result<SchemaShape> {
    let mut statement = connection.prepare(
        "SELECT type, name, tbl_name, sql
         FROM main.sqlite_schema
         ORDER BY type, name",
    )?;
    let mut rows = statement.query([])?;
    let mut objects = Vec::new();
    while let Some(row) = rows.next()? {
        objects.push(SchemaObject {
            object_type: row.get(0)?,
            name: row.get(1)?,
            table_name: row.get(2)?,
            normalized_sql: row
                .get::<_, Option<String>>(3)?
                .map(|sql| normalize_sql(&sql)),
        });
    }
    let tables = objects
        .iter()
        .filter(|object| object.object_type == "table")
        .map(|object| read_table_shape(connection, &object.name))
        .collect::<Result<Vec<_>>>()?;
    Ok(SchemaShape { objects, tables })
}

fn normalize_sql(sql: &str) -> String {
    sql.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn read_table_shape(connection: &Connection, table_name: &str) -> Result<TableShape> {
    let (table_type, column_count, without_rowid, strict) = connection.query_row(
        "SELECT type, ncol, wr, strict
         FROM pragma_table_list
         WHERE schema = 'main' AND name = ?1",
        [table_name],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )?;
    let mut statement = connection.prepare(
        "SELECT cid, name, type, \"notnull\", dflt_value, pk, hidden
         FROM pragma_table_xinfo(?1)
         ORDER BY cid",
    )?;
    let columns = statement
        .query_map([table_name], |row| {
            Ok(ColumnShape {
                cid: row.get(0)?,
                name: row.get(1)?,
                declared_type: row.get(2)?,
                not_null: row.get(3)?,
                default_value: row.get(4)?,
                primary_key_ordinal: row.get(5)?,
                hidden: row.get(6)?,
            })
        })?
        .collect::<Result<Vec<_>>>()?;
    let mut statement = connection.prepare(
        "SELECT id, seq, \"table\", \"from\", \"to\", on_update, on_delete, \"match\"
         FROM pragma_foreign_key_list(?1)
         ORDER BY id, seq",
    )?;
    let foreign_keys = statement
        .query_map([table_name], |row| {
            Ok(ForeignKeyShape {
                id: row.get(0)?,
                sequence: row.get(1)?,
                parent_table: row.get(2)?,
                child_column: row.get(3)?,
                parent_column: row.get(4)?,
                on_update: row.get(5)?,
                on_delete: row.get(6)?,
                match_clause: row.get(7)?,
            })
        })?
        .collect::<Result<Vec<_>>>()?;
    Ok(TableShape {
        name: table_name.to_owned(),
        table_type,
        column_count,
        without_rowid,
        strict,
        columns,
        foreign_keys,
    })
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
    fn session_and_durability_configuration_are_independent() {
        let (_directory, connection) = open_temporary_database();
        connection
            .pragma_update(None, "journal_mode", "MEMORY")
            .unwrap();
        connection
            .pragma_update(None, "synchronous", "NORMAL")
            .unwrap();
        connection
            .pragma_update(None, "trusted_schema", true)
            .unwrap();
        connection
            .pragma_update(None, "ignore_check_constraints", true)
            .unwrap();

        configure_session(&connection).expect("session configuration succeeds");

        assert_eq!(
            connection
                .pragma_query_value(None, "busy_timeout", |row| row.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .pragma_query_value(None, "foreign_keys", |row| row.get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .pragma_query_value(None, "trusted_schema", |row| row.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .pragma_query_value(None, "ignore_check_constraints", |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .pragma_query_value(None, "journal_mode", |row| row.get::<_, String>(0))
                .unwrap(),
            "memory"
        );
        assert_eq!(
            connection
                .pragma_query_value(None, "synchronous", |row| row.get::<_, i64>(0))
                .unwrap(),
            1
        );

        configure_durability(&connection).expect("durability configuration succeeds");

        assert_eq!(
            connection
                .pragma_query_value(None, "journal_mode", |row| row.get::<_, String>(0))
                .unwrap(),
            "delete"
        );
        assert_eq!(
            connection
                .pragma_query_value(None, "synchronous", |row| row.get::<_, i64>(0))
                .unwrap(),
            3
        );
    }

    #[test]
    fn schema_creation_commits_identity_metadata_and_canonical_shape() {
        let (_directory, mut connection) = open_temporary_database();
        let memory_id = MemoryId::new(u128::MAX).unwrap();
        configure(&connection);

        assert_eq!(read_application_id(&connection).unwrap(), 0);

        create_schema(&mut connection, memory_id).expect("schema creation succeeds");

        assert_eq!(read_application_id(&connection).unwrap(), APPLICATION_ID);
        assert!(validate_schema(&connection).unwrap());
        let (version, stored_id, revision): (i64, Vec<u8>, i64) = connection
            .query_row(
                "SELECT format_version, memory_id, snapshot_revision
                 FROM memory_meta
                 WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(version, FORMAT_VERSION);
        assert_eq!(stored_id, memory_id.to_be_bytes());
        assert_eq!(revision, 0);
    }

    #[test]
    fn failed_schema_creation_rolls_back_application_identity_and_ddl() {
        let (_directory, mut connection) = open_temporary_database();
        configure(&connection);
        connection
            .execute_batch(
                "CREATE TABLE episodes (
                    sequence BLOB PRIMARY KEY
                 ) STRICT, WITHOUT ROWID;",
            )
            .unwrap();

        assert!(create_schema(&mut connection, MemoryId::new(1).unwrap()).is_err());
        assert_eq!(read_application_id(&connection).unwrap(), 0);
        let metadata_tables: i64 = connection
            .query_row(
                "SELECT count(*) FROM sqlite_schema WHERE name = 'memory_meta'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(metadata_tables, 0);
    }

    #[test]
    fn validation_accepts_whitespace_only_ddl_variation() {
        let (_directory, connection) = open_temporary_database();
        let reformatted = SCHEMA
            .replace("CREATE TABLE", "CREATE    TABLE")
            .replace('\n', "\n    ");

        connection.execute_batch(&reformatted).unwrap();

        assert!(validate_schema(&connection).unwrap());
    }

    #[test]
    fn validation_rejects_structural_and_check_constraint_drift() {
        let mutations = [
            ("STRICT", SCHEMA.replacen(") STRICT;", ");", 1)),
            (
                "WITHOUT ROWID",
                SCHEMA.replacen(") STRICT, WITHOUT ROWID;", ") STRICT;", 1),
            ),
            (
                "column type",
                SCHEMA.replacen(
                    "occurred_at_ms INTEGER NOT NULL",
                    "occurred_at_ms TEXT NOT NULL",
                    1,
                ),
            ),
            (
                "primary key",
                SCHEMA.replacen(
                    "PRIMARY KEY (from_sequence, to_sequence)",
                    "PRIMARY KEY (to_sequence, from_sequence)",
                    1,
                ),
            ),
            (
                "foreign key action",
                SCHEMA.replacen(
                    "ON UPDATE RESTRICT ON DELETE RESTRICT",
                    "ON UPDATE RESTRICT ON DELETE CASCADE",
                    1,
                ),
            ),
            (
                "check constraint",
                SCHEMA.replacen(
                    "CHECK (activation_ppm BETWEEN 0 AND 1000000)",
                    "CHECK (activation_ppm BETWEEN 0 AND 999999)",
                    1,
                ),
            ),
        ];

        for (name, schema) in mutations {
            let (_directory, connection) = open_temporary_database();
            connection
                .execute_batch(&schema)
                .unwrap_or_else(|error| panic!("{name} fixture must be valid SQLite: {error}"));
            assert!(
                !validate_schema(&connection).unwrap(),
                "{name} drift must be rejected"
            );
        }
    }

    #[test]
    fn validation_rejects_persisted_user_schema_objects() {
        let objects = [
            "CREATE TABLE extra (value INTEGER) STRICT",
            "CREATE INDEX extra_index ON activations(activation_ppm)",
            "CREATE VIEW extra_view AS SELECT * FROM memory_meta",
            "CREATE TRIGGER extra_trigger AFTER UPDATE ON memory_meta
             BEGIN SELECT 1; END",
        ];

        for object in objects {
            let (_directory, connection) = open_temporary_database();
            connection.execute_batch(SCHEMA).unwrap();
            connection.execute_batch(object).unwrap();
            assert!(!validate_schema(&connection).unwrap(), "accepted {object}");
        }
    }

    #[test]
    fn database_constraints_reject_noncanonical_storage_values() {
        let (_directory, mut connection) = open_temporary_database();
        let memory_id = MemoryId::new(1).unwrap();
        configure(&connection);
        create_schema(&mut connection, memory_id).unwrap();

        assert!(
            connection
                .execute(
                    "INSERT INTO episodes (
                        sequence,
                        occurred_at_ms,
                        recorded_at_ms,
                        source_id
                    ) VALUES (?1, 0, 0, ?2)",
                    params![[0_u8; 7].as_slice(), [0_u8; 8].as_slice()],
                )
                .is_err()
        );
        connection
            .execute(
                "INSERT INTO episodes (
                    sequence,
                    occurred_at_ms,
                    recorded_at_ms,
                    source_id
                ) VALUES (?1, 0, 0, ?2)",
                params![[0_u8; 8].as_slice(), [0_u8; 8].as_slice()],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO episodes (
                    sequence,
                    occurred_at_ms,
                    recorded_at_ms,
                    source_id
                ) VALUES (?1, 0, 0, ?2)",
                params![[0_u8, 0, 0, 0, 0, 0, 0, 1].as_slice(), [0_u8; 8].as_slice()],
            )
            .unwrap();
        assert!(
            connection
                .execute(
                    "INSERT INTO activations (episode_sequence, activation_ppm)
                     VALUES (?1, 1000001)",
                    params![[0_u8; 8].as_slice()],
                )
                .is_err()
        );
        assert!(
            connection
                .execute(
                    "INSERT INTO episode_statements (
                        episode_sequence,
                        role,
                        statement_ordinal,
                        predicate_id
                    ) VALUES (?1, 1, 1, ?2)",
                    params![[0_u8; 8].as_slice(), [0_u8; 8].as_slice()],
                )
                .is_err()
        );
        assert!(
            connection
                .execute(
                    "INSERT INTO relevance_edges (
                        from_sequence,
                        to_sequence,
                        weight_ppm
                    ) VALUES (?1, ?1, 1)",
                    params![[0_u8; 8].as_slice()],
                )
                .is_err()
        );
        assert!(
            connection
                .execute(
                    "INSERT INTO relevance_edges (
                        from_sequence,
                        to_sequence,
                        weight_ppm
                    ) VALUES (?1, ?2, 1)",
                    params![[0_u8; 8].as_slice(), [0xff_u8; 8].as_slice()],
                )
                .is_err()
        );
    }
}
