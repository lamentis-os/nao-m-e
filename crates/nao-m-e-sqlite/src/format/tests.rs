use super::*;

fn open_temporary_database() -> (tempfile::TempDir, Connection) {
    let directory = tempfile::tempdir().expect("temporary directory is available");
    let connection =
        Connection::open(directory.path().join("memory.sqlite")).expect("temporary database opens");
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
fn schema_creation_commits_current_format_identity_and_closed_shape() {
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
    assert_eq!(
        tables,
        [
            "episodes",
            "feedback_edges",
            "memory_meta",
            "predicates",
            "terms"
        ]
    );
    let indexes: Vec<String> = connection
        .prepare(
            "SELECT name FROM sqlite_schema
             WHERE type = 'index' AND sql IS NOT NULL
             ORDER BY name",
        )
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_>>()
        .unwrap();
    assert_eq!(indexes, ["predicates_value_unique", "terms_value_unique"]);
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
            "CHECK (history_bits BETWEEN 0 AND 65535)",
            "CHECK (history_bits BETWEEN 0 AND 65536)",
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
        "CREATE INDEX extra_index ON feedback_edges(history_bits)",
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
                "INSERT INTO predicates (id, value) VALUES (?1, 'bad-id')",
                [[0_u8; 7].as_slice()],
            )
            .is_err()
    );
    for value in [String::new(), "x".repeat(MAX_SYMBOL_BYTES + 1)] {
        assert!(
            connection
                .execute(
                    "INSERT INTO predicates (id, value) VALUES (?1, ?2)",
                    params![zero.as_slice(), value],
                )
                .is_err()
        );
    }
    connection
        .execute(
            "INSERT INTO predicates (id, value) VALUES (?1, 'value')",
            [zero.as_slice()],
        )
        .unwrap();
    assert!(
        connection
            .execute(
                "INSERT INTO predicates (id, value) VALUES (?1, 'value')",
                [one.as_slice()],
            )
            .is_err()
    );

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
                "INSERT INTO feedback_edges VALUES (?1, ?1, 1, 1)",
                [zero.as_slice()],
            )
            .is_err()
    );
    connection
        .execute(
            "INSERT INTO feedback_edges VALUES (?1, ?2, 65535, 16)",
            params![zero.as_slice(), one.as_slice()],
        )
        .unwrap();
    assert!(
        connection
            .execute(
                "INSERT INTO feedback_edges VALUES (?1, ?2, 1, 1)",
                params![zero.as_slice(), [0xff_u8; 8].as_slice()],
            )
            .is_err()
    );
    for (history_bits, sample_count) in [(2, 1), (0, 0), (0, 17)] {
        assert!(
            connection
                .execute(
                    "UPDATE feedback_edges
                     SET history_bits = ?1, sample_count = ?2",
                    params![history_bits, sample_count],
                )
                .is_err()
        );
    }
}
