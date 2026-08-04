use nao_m_e::{AtomId, EpisodeDraft, GraphError, Memory, TimestampMs};

use super::support::{draft, insert, memory_id, new_memory, statement, trace};

#[test]
fn replaying_the_same_atom_sequence_recreates_durable_ids() {
    let durable_id = memory_id(0xfeed);
    let original_ids = {
        let mut original = Memory::new(durable_id);
        assert_eq!(original.memory_id(), durable_id);
        [insert(&mut original, 1), insert(&mut original, 2)]
    };

    let mut reopened = Memory::new(durable_id);
    let reopened_ids = [insert(&mut reopened, 1), insert(&mut reopened, 2)];
    assert_eq!(reopened_ids, original_ids);
    assert_eq!(reopened_ids[0].sequence(), 0);
    assert_eq!(reopened_ids[1].sequence(), 1);
}

#[test]
fn insertion_canonicalizes_context_but_keeps_occurrences_distinct() {
    let low = statement(1, &[1]);
    let high = statement(2, &[1]);
    let mut episode = draft(1);
    episode.context = vec![high.clone(), low.clone(), high];

    let mut memory = new_memory(1);
    let first = memory
        .insert_episode(episode.clone())
        .expect("first inserts");
    let second = memory.insert_episode(episode).expect("duplicate inserts");
    assert_ne!(first, second);
    assert_eq!(
        memory.episode(first).expect("atom exists").context(),
        &[low, statement(2, &[1])]
    );
}

#[test]
fn statements_and_timestamp_extremes_preserve_caller_semantics() {
    let ordered = statement(7, &[30, 20, 10]);
    let mut memory = new_memory(1);
    let earliest = memory
        .insert_episode(EpisodeDraft {
            timestamp: TimestampMs::new(i64::MIN),
            context: Vec::new(),
            observation: ordered.clone(),
            action: Some(statement(8, &[1])),
            outcome: Some(statement(9, &[2])),
        })
        .expect("episode inserts");
    let latest = memory
        .insert_episode(EpisodeDraft {
            timestamp: TimestampMs::new(i64::MAX),
            context: Vec::new(),
            observation: ordered.clone(),
            action: Some(statement(8, &[1])),
            outcome: Some(statement(9, &[2])),
        })
        .expect("episode inserts");
    let stored = memory.episode(earliest).expect("episode exists");

    assert_eq!(stored.timestamp(), TimestampMs::new(i64::MIN));
    assert_eq!(stored.observation(), &ordered);
    assert_eq!(stored.action(), Some(&statement(8, &[1])));
    assert_eq!(stored.outcome(), Some(&statement(9, &[2])));
    assert_eq!(
        memory.episode(latest).expect("episode exists").timestamp(),
        TimestampMs::new(i64::MAX)
    );
}

#[test]
fn feedback_changes_do_not_mutate_episode_content() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    let original = memory.episode(source).expect("atom exists").clone();

    memory
        .set_feedback_trace(source, target, trace(1, 1))
        .expect("trace inserts");
    memory
        .apply_feedback(source, &[target], false)
        .expect("feedback applies");

    assert_eq!(memory.episode(source), Some(&original));
    assert_eq!(memory.feedback_trace(source, target), Some(trace(0b10, 2)));
}

#[test]
fn empty_memory_has_no_episodes_or_feedback() {
    let memory = new_memory(1);
    assert_eq!(memory.episodes().len(), 0);
    assert_eq!(memory.feedback_edges().count(), 0);
}

#[test]
fn foreign_and_absent_ids_never_alias_local_atoms_or_feedback() {
    let mut memory = new_memory(1);
    let local = insert(&mut memory, 1);
    let other = insert(&mut memory, 2);
    let foreign = AtomId::from_parts(memory_id(2), local.sequence());
    let absent = AtomId::from_parts(memory.memory_id(), u64::MAX);

    assert_eq!(memory.episode(foreign), None);
    assert_eq!(memory.episode(absent), None);
    assert_eq!(memory.feedback_trace(local, foreign), None);
    assert_eq!(memory.feedback_trace(absent, other), None);
    assert_eq!(
        memory.set_feedback_trace(local, foreign, trace(1, 1)),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(
        memory.apply_feedback(absent, &[other], true),
        Err(GraphError::UnknownAtom(absent))
    );
    assert_eq!(memory.feedback_edges().count(), 0);
}
