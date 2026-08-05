use std::fmt::Write as _;
use std::fs;

use nao_m_e::AtomId;
use nao_m_e_sqlite::SqliteStore;
use tempfile::TempDir;

use super::support::{
    add_minimal, assert_silent_success, cli, init, invoke, recall, seed_cues, success_text,
};

fn minimal_recall_block(sequence: u64, activation_ppm: u32) -> String {
    format!(
        "sequence {sequence}\nactivation_ppm {activation_ppm}\ntimestamp {sequence}\nattribute attribute-{sequence}\nvalue value-{sequence}"
    )
}

#[test]
fn recall_emits_exact_ranked_blocks_and_honors_limit() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    let mut initial_cues = vec![("attribute-0".to_owned(), "value-0".to_owned())];
    initial_cues.extend(
        [
            ("context-b", "context-b-1"),
            ("context-b", "context-b-2"),
            ("context-a", "context-a-1"),
            ("observation", "observation-1"),
            ("observation", "observation-2"),
            ("action", "action-1"),
            ("outcome", "outcome-1"),
            ("outcome", "outcome-2"),
        ]
        .map(|(key, value)| (key.to_owned(), value.to_owned())),
    );
    initial_cues
        .extend((2..=11).map(|seed| (format!("attribute-{seed}"), format!("value-{seed}"))));
    seed_cues(&database, initial_cues);
    let mut input = String::from(
        "--timestamp 0 --attribute attribute-0 --value value-0\n\
         --timestamp -7 --attribute context-b --value context-b-1 --value context-b-2 --attribute context-a --value context-a-1 --attribute context-a --value context-a-1 --attribute observation --value observation-1 --value observation-2 --attribute action --value action-1 --attribute outcome --value outcome-1 --value outcome-2\n",
    );
    for seed in 2..=11 {
        input.push_str(&format!(
            "--timestamp {seed} --attribute attribute-{seed} --value value-{seed}\n"
        ));
    }
    let mut many = cli();
    many.arg("add").arg(&database).arg("--many").arg("--quiet");
    assert_silent_success(invoke(many, Some(&input)));

    let mut store = SqliteStore::open(&database).expect("seeded store opens");
    let memory_id = store.memory_id();
    let source = AtomId::from_parts(memory_id, 0);
    let rich = AtomId::from_parts(memory_id, 1);
    let targets = (2..=11)
        .map(|sequence| AtomId::from_parts(memory_id, sequence))
        .collect::<Vec<_>>();
    for _ in 0..16 {
        store
            .memory_mut()
            .apply_feedback(source, &[rich], true)
            .unwrap();
    }
    for (&target, helpful_count) in targets.iter().zip([8, 8, 15, 14, 13, 12, 11, 10, 9, 7]) {
        for _ in 0..helpful_count {
            store
                .memory_mut()
                .apply_feedback(source, &[target], true)
                .unwrap();
        }
    }
    store.save().unwrap();
    drop(store);
    let before = fs::read(&database).expect("database is readable before recall");

    let rich_block = "sequence 1\nactivation_ppm 400000\ntimestamp -7\nattribute context-b\nvalue context-b-1\nvalue context-b-2\nattribute context-a\nvalue context-a-1\nattribute observation\nvalue observation-1\nvalue observation-2\nattribute action\nvalue action-1\nattribute outcome\nvalue outcome-1\nvalue outcome-2";
    let mut blocks = vec![rich_block.to_owned()];
    blocks.extend(
        [
            (4, 392_045),
            (5, 383_333),
            (6, 373_750),
            (7, 363_157),
            (8, 351_388),
            (9, 338_235),
            (10, 323_437),
            (2, 306_666),
            (3, 306_666),
            (11, 287_500),
        ]
        .into_iter()
        .map(|(sequence, activation_ppm)| minimal_recall_block(sequence, activation_ppm)),
    );
    assert_eq!(blocks.len(), 11);

    let expected_default = blocks[..10].join("\n\n") + "\n";
    assert_eq!(success_text(recall(&database, 0, None)), expected_default);

    let expected_eleven = blocks.join("\n\n") + "\n";
    assert_eq!(
        success_text(recall(&database, 0, Some(11))),
        expected_eleven
    );
    assert_eq!(
        fs::read(&database).expect("database is readable after recall"),
        before
    );

    const UNIQUE_VALUE_COUNT: usize = 901;
    let mut wide_cues = vec![("batch-symbol".to_owned(), "shared-symbol".to_owned())];
    wide_cues.extend((0..UNIQUE_VALUE_COUNT).map(|index| {
        (
            "batch-symbol".to_owned(),
            format!("unique-value-{index:04}"),
        )
    }));
    seed_cues(&database, wide_cues);
    let mut wide_input = String::from(
        "--timestamp 12 --attribute batch-symbol --value shared-symbol\n\
         --timestamp 13 --attribute batch-symbol --value shared-symbol --value shared-symbol",
    );
    for index in 0..UNIQUE_VALUE_COUNT {
        write!(wide_input, " --value unique-value-{index:04}")
            .expect("writing to a String cannot fail");
    }
    wide_input.push('\n');
    let mut many = cli();
    many.arg("add").arg(&database).arg("--many").arg("--quiet");
    assert_silent_success(invoke(many, Some(&wide_input)));

    let before_wide_recall = fs::read(&database).expect("wide database is readable before recall");
    let mut expected_wide = String::from(
        "sequence 13\nactivation_ppm 664\ntimestamp 13\nattribute batch-symbol\nvalue shared-symbol\n",
    );
    for index in 0..UNIQUE_VALUE_COUNT {
        writeln!(expected_wide, "value unique-value-{index:04}")
            .expect("writing to a String cannot fail");
    }
    assert_eq!(success_text(recall(&database, 12, Some(1))), expected_wide);
    assert_eq!(
        fs::read(&database).expect("wide database is readable after recall"),
        before_wide_recall
    );
}

#[test]
fn cold_recall_rebuilds_cue_candidates_without_feedback() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    seed_cues(
        &database,
        [
            ("category", "seven"),
            ("category", "eight"),
            ("category", "nine"),
            ("other", "thirty"),
        ],
    );

    for (sequence, timestamp, attribute, values) in [
        (0, 1, "category", &["seven", "eight"][..]),
        (1, 3, "category", &["seven", "nine"][..]),
        (2, 5, "other", &["thirty"][..]),
    ] {
        let mut command = cli();
        command
            .arg("add")
            .arg(&database)
            .arg("--timestamp")
            .arg(timestamp.to_string())
            .arg("--attribute")
            .arg(attribute);
        for value in values {
            command.arg("--value").arg(value);
        }
        assert_eq!(success_text(invoke(command, None)), format!("{sequence}\n"));
    }

    assert_eq!(
        success_text(recall(&database, 0, None)),
        "sequence 1\nactivation_ppm 171428\ntimestamp 3\nattribute category\nvalue seven\nvalue nine\n"
    );
    assert_eq!(
        SqliteStore::open(&database)
            .unwrap()
            .memory()
            .feedback_edges()
            .count(),
        0
    );
}

#[test]
fn recall_with_no_hits_is_silent_and_does_not_advance_the_store() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    assert_eq!(success_text(add_minimal(&database, 1, false)), "0\n");
    assert_eq!(success_text(add_minimal(&database, 2, false)), "1\n");
    let before = fs::read(&database).expect("database is readable before recall");
    assert_silent_success(recall(&database, 0, None));
    assert_silent_success(recall(&database, 0, Some(0)));
    assert_eq!(
        fs::read(&database).expect("database is readable after recall"),
        before
    );

    let mut writer = SqliteStore::open(&database).expect("writer opens before recall");
    let source = AtomId::from_parts(writer.memory_id(), 0);
    let target = AtomId::from_parts(writer.memory_id(), 1);
    assert_silent_success(recall(&database, 0, None));
    writer
        .memory_mut()
        .apply_feedback(source, &[target], true)
        .unwrap();
    writer
        .save()
        .expect("read-only recall did not advance the revision");
}
