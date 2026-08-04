use std::fmt::Write as _;
use std::fs;

use nao_m_e::AtomId;
use nao_m_e_sqlite::SqliteStore;
use tempfile::TempDir;

use super::support::{add_minimal, assert_silent_success, cli, init, invoke, recall, success_text};

fn minimal_recall_block(sequence: u64, activation_ppm: u32) -> String {
    format!(
        "sequence {sequence}\nactivation_ppm {activation_ppm}\noccurred {sequence}\nrecorded {}\nsource {sequence}\npredicate predicate-{sequence}\nterm term-{sequence}",
        sequence + 1
    )
}

#[test]
fn recall_emits_exact_ranked_blocks_and_honors_limit() {
    let directory = TempDir::new().expect("temporary directory");
    let database = directory.path().join("memory.sqlite3");
    init(&database);
    let mut input = String::from(
        "--occurred 0 --recorded 1 --source 0 --predicate predicate-0 --term term-0\n\
         --occurred -7 --recorded 8 --source 9 --context context-b --context-term context-b-1 --context-term context-b-2 --context context-a --context-term context-a-1 --context context-a --context-term context-a-1 --predicate observation --term observation-1 --term observation-2 --action action --action-term action-1 --outcome outcome --outcome-term outcome-1 --outcome-term outcome-2\n",
    );
    for seed in 2..=11 {
        input.push_str(&format!(
            "--occurred {seed} --recorded {} --source {seed} --predicate predicate-{seed} --term term-{seed}\n",
            seed + 1
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

    let rich_block = "sequence 1\nactivation_ppm 400000\noccurred -7\nrecorded 8\nsource 9\ncontext context-b\ncontext-term context-b-1\ncontext-term context-b-2\ncontext context-a\ncontext-term context-a-1\npredicate observation\nterm observation-1\nterm observation-2\naction action\naction-term action-1\noutcome outcome\noutcome-term outcome-1\noutcome-term outcome-2";
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

    const UNIQUE_TERM_COUNT: usize = 901;
    let mut wide_input = String::from(
        "--occurred 12 --recorded 13 --source 12 --predicate batch-symbol --term shared-symbol\n\
         --occurred 13 --recorded 14 --source 13 --predicate batch-symbol --term shared-symbol --term shared-symbol",
    );
    for index in 0..UNIQUE_TERM_COUNT {
        write!(wide_input, " --term unique-term-{index:04}")
            .expect("writing to a String cannot fail");
    }
    wide_input.push('\n');
    let mut many = cli();
    many.arg("add").arg(&database).arg("--many").arg("--quiet");
    assert_silent_success(invoke(many, Some(&wide_input)));

    let before_wide_recall = fs::read(&database).expect("wide database is readable before recall");
    let mut expected_wide = String::from(
        "sequence 13\nactivation_ppm 708\noccurred 13\nrecorded 14\nsource 13\npredicate batch-symbol\nterm shared-symbol\nterm shared-symbol\n",
    );
    for index in 0..UNIQUE_TERM_COUNT {
        writeln!(expected_wide, "term unique-term-{index:04}")
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

    for (sequence, occurred, source, predicate, terms) in [
        (0, 1, 100, "category", &["seven", "eight"][..]),
        (1, 3, 101, "category", &["seven", "nine"][..]),
        (2, 5, 102, "other", &["thirty"][..]),
    ] {
        let mut command = cli();
        command
            .arg("add")
            .arg(&database)
            .arg("--occurred")
            .arg(occurred.to_string())
            .arg("--recorded")
            .arg((occurred + 1).to_string())
            .arg("--source")
            .arg(source.to_string())
            .arg("--predicate")
            .arg(predicate);
        for term in terms {
            command.arg("--term").arg(term);
        }
        assert_eq!(success_text(invoke(command, None)), format!("{sequence}\n"));
    }

    assert_eq!(
        success_text(recall(&database, 0, None)),
        "sequence 1\nactivation_ppm 177777\noccurred 3\nrecorded 4\nsource 101\npredicate category\nterm seven\nterm nine\n"
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
