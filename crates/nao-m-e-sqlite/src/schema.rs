use std::time::Duration;

use nao_m_e::MemoryId;
use rusqlite::{Connection, Error, Result, Transaction, TransactionBehavior, params};

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

pub(crate) fn configure_new_connection(connection: &Connection) -> Result<()> {
    configure_connection(connection)?;
    connection.pragma_update(None, "application_id", APPLICATION_ID)?;
    verify_integer_pragma(connection, "application_id", APPLICATION_ID)
}

pub(crate) fn configure_existing_connection(connection: &Connection) -> Result<()> {
    configure_connection(connection)
}

pub(crate) fn read_application_id(connection: &Connection) -> Result<i64> {
    connection.pragma_query_value(None, "application_id", |row| row.get(0))
}

pub(crate) fn create_schema(connection: &Connection, memory_id: MemoryId) -> Result<()> {
    let transaction = Transaction::new_unchecked(connection, TransactionBehavior::Immediate)?;
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
    transaction.commit()
}

fn configure_connection(connection: &Connection) -> Result<()> {
    connection.busy_timeout(Duration::ZERO)?;
    connection.pragma_update(None, "foreign_keys", true)?;
    connection.pragma_update(None, "trusted_schema", false)?;
    connection.pragma_update(None, "ignore_check_constraints", false)?;

    let journal_mode: String =
        connection.query_row("PRAGMA journal_mode = DELETE", [], |row| row.get(0))?;
    if !journal_mode.eq_ignore_ascii_case("delete") {
        return Err(Error::InvalidQuery);
    }

    connection.pragma_update(None, "synchronous", "EXTRA")?;

    verify_integer_pragma(connection, "foreign_keys", 1)?;
    verify_integer_pragma(connection, "trusted_schema", 0)?;
    verify_integer_pragma(connection, "ignore_check_constraints", 0)?;
    verify_integer_pragma(connection, "busy_timeout", 0)?;
    verify_integer_pragma(connection, "synchronous", 3)
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

    #[test]
    fn connection_configuration_is_applied_and_verified() {
        let (_directory, connection) = open_temporary_database();

        configure_new_connection(&connection).expect("configuration succeeds");

        assert_eq!(read_application_id(&connection).unwrap(), APPLICATION_ID);
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
    fn schema_records_memory_metadata_and_uses_strict_tables() {
        let (_directory, connection) = open_temporary_database();
        let memory_id = MemoryId::new(u128::MAX).unwrap();
        configure_new_connection(&connection).unwrap();

        create_schema(&connection, memory_id).expect("schema creation succeeds");

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

        let mut statement = connection
            .prepare(
                "SELECT name, wr, strict
                 FROM pragma_table_list
                 WHERE name IN (
                    'memory_meta',
                    'episodes',
                    'episode_statements',
                    'statement_terms',
                    'activations',
                    'relevance_edges'
                 )
                 ORDER BY name",
            )
            .unwrap();
        let properties = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(properties.len(), 6);
        for (name, without_rowid, strict) in properties {
            assert_eq!(strict, 1, "{name} must be STRICT");
            assert_eq!(
                without_rowid,
                i64::from(name != "memory_meta"),
                "unexpected rowid policy for {name}"
            );
        }
    }

    #[test]
    fn database_constraints_reject_noncanonical_storage_values() {
        let (_directory, connection) = open_temporary_database();
        let memory_id = MemoryId::new(1).unwrap();
        configure_new_connection(&connection).unwrap();
        create_schema(&connection, memory_id).unwrap();

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
