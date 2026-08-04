use nao_m_e::{AtomId, GraphError, MAX_FEEDBACK_TARGETS};

use super::support::{
    activation, feedback_snapshot, insert, insert_observation, memory_id, new_memory, trace,
};

#[test]
fn feedback_trace_setter_validates_atomically_and_returns_the_previous_trace() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    let other = insert(&mut memory, 3);
    let absent = AtomId::from_parts(memory.memory_id(), u64::MAX);
    let foreign = AtomId::from_parts(memory_id(2), 0);

    assert_eq!(
        memory
            .set_feedback_trace(source, target, trace(1, 1))
            .expect("trace inserts"),
        None
    );
    assert_eq!(
        memory
            .set_feedback_trace(source, target, trace(0b01, 2))
            .expect("trace replaces"),
        Some(trace(1, 1))
    );
    let before = feedback_snapshot(&memory);
    assert_eq!(
        memory.set_feedback_trace(source, source, trace(0, 1)),
        Err(GraphError::SelfEdge(source))
    );
    assert_eq!(
        memory.set_feedback_trace(source, absent, trace(0, 1)),
        Err(GraphError::UnknownAtom(absent))
    );
    assert_eq!(
        memory.set_feedback_trace(foreign, other, trace(0, 1)),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(feedback_snapshot(&memory), before);
    assert_eq!(memory.feedback_trace(absent, target), None);
    assert_eq!(memory.feedback_trace(source, foreign), None);
}

#[test]
fn feedback_edges_are_directed_and_sorted() {
    let mut memory = new_memory(1);
    let a = insert(&mut memory, 1);
    let b = insert(&mut memory, 2);
    let c = insert(&mut memory, 3);
    memory
        .set_feedback_trace(b, c, trace(0, 1))
        .expect("B to C");
    memory
        .set_feedback_trace(a, c, trace(1, 1))
        .expect("A to C");
    memory
        .set_feedback_trace(a, b, trace(0b01, 2))
        .expect("A to B");

    assert_eq!(memory.feedback_trace(b, a), None);
    assert_eq!(
        feedback_snapshot(&memory),
        vec![
            (a, b, trace(0b01, 2)),
            (a, c, trace(1, 1)),
            (b, c, trace(0, 1)),
        ]
    );
}

#[test]
fn feedback_validates_source_limit_and_every_target_before_mutation() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    let absent = AtomId::from_parts(memory.memory_id(), u64::MAX);
    let foreign = AtomId::from_parts(memory_id(2), 0);
    memory
        .set_feedback_trace(source, target, trace(1, 1))
        .expect("initial trace inserts");
    let before = feedback_snapshot(&memory);

    assert_eq!(
        memory.apply_feedback(foreign, &[target], true),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(
        memory.apply_feedback(source, &[target, absent], false),
        Err(GraphError::UnknownAtom(absent))
    );
    assert_eq!(
        memory.apply_feedback(source, &[target, foreign], true),
        Err(GraphError::UnknownAtom(foreign))
    );
    let too_many = vec![target; MAX_FEEDBACK_TARGETS + 1];
    assert_eq!(
        memory.apply_feedback(source, &too_many, true),
        Err(GraphError::FeedbackTargetLimitExceeded {
            count: MAX_FEEDBACK_TARGETS + 1,
            max: MAX_FEEDBACK_TARGETS,
        })
    );
    assert_eq!(feedback_snapshot(&memory), before);
}

#[test]
fn feedback_treats_targets_as_a_set_and_ignores_the_source() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let first = insert(&mut memory, 2);
    let second = insert(&mut memory, 3);

    memory
        .apply_feedback(source, &[], true)
        .expect("empty feedback is a no-op");
    memory
        .apply_feedback(source, &[source, source], false)
        .expect("self-only feedback is a no-op");
    memory
        .apply_feedback(source, &[second, source, first, second, first], true)
        .expect("deduplicated feedback applies");

    assert_eq!(
        feedback_snapshot(&memory),
        vec![(source, first, trace(1, 1)), (source, second, trace(1, 1))]
    );
}

#[test]
fn every_target_receives_a_full_sample_at_the_limit() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 0);
    let targets: Vec<_> = (0..MAX_FEEDBACK_TARGETS)
        .map(|index| insert(&mut memory, u64::try_from(index + 1).expect("small index")))
        .collect();

    memory
        .apply_feedback(source, &targets, false)
        .expect("maximum feedback batch applies");

    assert_eq!(memory.feedback_edges().count(), MAX_FEEDBACK_TARGETS);
    assert!(
        targets
            .iter()
            .all(|&target| memory.feedback_trace(source, target) == Some(trace(0, 1)))
    );
}

#[test]
fn the_seventeenth_sample_drops_exactly_the_oldest_bit() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    memory
        .set_feedback_trace(source, target, trace(0xaaaa, 16))
        .expect("full trace inserts");

    memory
        .apply_feedback(source, &[target], true)
        .expect("seventeenth sample applies");
    assert_eq!(
        memory.feedback_trace(source, target),
        Some(trace(0x5555, 16))
    );
    memory
        .apply_feedback(source, &[target], false)
        .expect("next sample applies");
    assert_eq!(
        memory.feedback_trace(source, target),
        Some(trace(0xaaaa, 16))
    );
}

#[test]
fn helpful_feedback_follows_the_exact_saturating_learning_curve() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    let checkpoints = [
        (1, 71_875),
        (2, 127_777),
        (3, 172_500),
        (4, 209_090),
        (8, 306_666),
        (16, 400_000),
        (17, 400_000),
    ];

    for sample in 1..=17 {
        memory
            .apply_feedback(source, &[target], true)
            .expect("helpful sample applies");
        if let Some((_, expected)) = checkpoints.iter().find(|(count, _)| *count == sample) {
            assert_eq!(
                memory.recall_from(source, 1).expect("source is known"),
                vec![nao_m_e::RecallHit {
                    atom_id: target,
                    activation: activation(*expected),
                }],
                "sample {sample}"
            );
        }
    }
    assert_eq!(
        memory.feedback_trace(source, target),
        Some(trace(u16::MAX, 16))
    );
}

#[test]
fn saturated_learning_requires_eight_opposing_samples_to_become_neutral() {
    let mut memory = new_memory(1);
    let source = insert_observation(&mut memory, 1, 10, &[100]);
    let target = insert_observation(&mut memory, 2, 10, &[100]);
    for _ in 0..16 {
        memory
            .apply_feedback(source, &[target], true)
            .expect("helpful sample applies");
    }
    assert_eq!(
        memory.recall_from(source, 1).expect("source is known")[0].activation,
        activation(800_000)
    );

    for opposing in 1..=16 {
        memory
            .apply_feedback(source, &[target], false)
            .expect("unhelpful sample applies");
        let expected = match opposing {
            1 => Some(750_000),
            8 => Some(400_000),
            9 => Some(350_000),
            16 => None,
            _ => continue,
        };
        let hits = memory.recall_from(source, 1).expect("source is known");
        assert_eq!(hits.first().map(|hit| hit.activation.as_ppm()), expected);
    }
    assert_eq!(memory.feedback_trace(source, target), Some(trace(0, 16)));
}

#[test]
fn unhelpful_feedback_suppresses_structural_recall_with_exact_signed_scores() {
    let mut memory = new_memory(1);
    let source = insert_observation(&mut memory, 1, 10, &[100, 200]);
    let target = insert_observation(&mut memory, 2, 10, &[100, 999]);
    let expected = [Some(105_902), Some(50_000), Some(5_277), None];

    for (index, expected_score) in expected.into_iter().enumerate() {
        memory
            .apply_feedback(source, &[target], false)
            .expect("unhelpful sample applies");
        let hits = memory.recall_from(source, 1).expect("source is known");
        assert_eq!(
            hits.first().map(|hit| hit.activation.as_ppm()),
            expected_score,
            "sample {}",
            index + 1
        );
    }
}

#[test]
fn neutral_and_negative_learned_only_traces_remain_state_but_not_candidates() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let neutral = insert(&mut memory, 2);
    let negative = insert(&mut memory, 3);
    memory
        .set_feedback_trace(source, neutral, trace(0b01, 2))
        .expect("neutral trace inserts");
    memory
        .set_feedback_trace(source, negative, trace(0, 1))
        .expect("negative trace inserts");

    assert!(
        memory
            .recall_from(source, usize::MAX)
            .expect("source is known")
            .is_empty()
    );
    assert_eq!(memory.feedback_edges().count(), 2);
}
