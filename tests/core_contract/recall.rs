use nao_m_e::{AtomId, GraphError, TimestampMs};

use super::support::{
    activation, feedback_snapshot, insert, insert_observation, memory_id, new_memory,
    observation_draft, statement, trace,
};

#[test]
fn recall_unifies_structural_learned_and_combined_candidates_once() {
    let mut memory = new_memory(1);
    let source = insert_observation(&mut memory, 1, 10, &[100]);
    let learned_only = insert_observation(&mut memory, 2, 20, &[300]);
    let structural_only = insert_observation(&mut memory, 3, 10, &[100]);
    let combined = insert_observation(&mut memory, 4, 10, &[200]);
    memory
        .set_feedback_trace(source, combined, trace(1, 1))
        .expect("combined trace inserts");
    memory
        .set_feedback_trace(source, learned_only, trace(1, 1))
        .expect("learned-only trace inserts");

    let expected = vec![
        nao_m_e::RecallHit {
            atom_id: structural_only,
            activation: activation(400_000),
        },
        nao_m_e::RecallHit {
            atom_id: combined,
            activation: activation(164_182),
        },
        nao_m_e::RecallHit {
            atom_id: learned_only,
            activation: activation(71_875),
        },
    ];
    assert_eq!(
        memory
            .recall_from(source, usize::MAX)
            .expect("source is known"),
        expected
    );
    assert_eq!(
        memory.recall_from(source, 2).expect("source is known"),
        expected[..2]
    );
}

#[test]
fn recall_uses_only_the_sources_direct_feedback_row_without_mutating_state() {
    let mut memory = new_memory(1);
    let incoming = insert(&mut memory, 1);
    let source = insert(&mut memory, 2);
    let direct = insert(&mut memory, 3);
    let downstream = insert(&mut memory, 4);
    memory
        .set_feedback_trace(incoming, source, trace(u16::MAX, 16))
        .expect("incoming trace inserts");
    memory
        .set_feedback_trace(source, direct, trace(1, 1))
        .expect("direct trace inserts");
    memory
        .set_feedback_trace(direct, downstream, trace(u16::MAX, 16))
        .expect("downstream trace inserts");
    let episodes_before: Vec<_> = memory.episodes().cloned().collect();
    let feedback_before = feedback_snapshot(&memory);

    assert_eq!(
        memory
            .recall_from(source, usize::MAX)
            .expect("source is known"),
        vec![nao_m_e::RecallHit {
            atom_id: direct,
            activation: activation(71_875),
        }]
    );
    assert_eq!(
        memory.episodes().cloned().collect::<Vec<_>>(),
        episodes_before
    );
    assert_eq!(feedback_snapshot(&memory), feedback_before);
}

#[test]
fn recall_ranking_is_deterministic_for_every_limit() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let first_tie = insert(&mut memory, 2);
    let second_tie = insert(&mut memory, 3);
    let strongest = insert(&mut memory, 4);
    memory
        .set_feedback_trace(source, first_tie, trace(1, 1))
        .expect("first tie inserts");
    memory
        .set_feedback_trace(source, second_tie, trace(1, 1))
        .expect("second tie inserts");
    memory
        .set_feedback_trace(source, strongest, trace(0b11, 2))
        .expect("strongest inserts");

    let complete = [
        nao_m_e::RecallHit {
            atom_id: strongest,
            activation: activation(127_777),
        },
        nao_m_e::RecallHit {
            atom_id: first_tie,
            activation: activation(71_875),
        },
        nao_m_e::RecallHit {
            atom_id: second_tie,
            activation: activation(71_875),
        },
    ];
    for limit in 0..=complete.len() + 1 {
        assert_eq!(
            memory.recall_from(source, limit).expect("source is known"),
            complete.iter().copied().take(limit).collect::<Vec<_>>(),
            "limit {limit}"
        );
    }
}

#[test]
fn timestamps_do_not_change_structural_scores_or_tie_breaking() {
    let mut memory = new_memory(1);
    let mut source_draft = observation_draft(1, 10, &[100]);
    source_draft.timestamp = TimestampMs::new(i64::MIN);
    let source = memory.insert_episode(source_draft).expect("source inserts");

    let mut first_draft = observation_draft(2, 10, &[100]);
    first_draft.timestamp = TimestampMs::new(i64::MAX);
    let first = memory
        .insert_episode(first_draft)
        .expect("first target inserts");

    let mut second_draft = observation_draft(3, 10, &[100]);
    second_draft.timestamp = TimestampMs::new(0);
    let second = memory
        .insert_episode(second_draft)
        .expect("second target inserts");

    assert_eq!(
        memory
            .recall_from(source, usize::MAX)
            .expect("source is known"),
        vec![
            nao_m_e::RecallHit {
                atom_id: first,
                activation: activation(400_000),
            },
            nao_m_e::RecallHit {
                atom_id: second,
                activation: activation(400_000),
            },
        ]
    );
}

#[test]
fn recall_validates_the_source_before_the_limit_and_handles_isolated_sources() {
    let memory = {
        let mut memory = new_memory(1);
        insert(&mut memory, 1);
        memory
    };
    let known = memory.episodes().next().expect("one atom exists").id();
    let absent = AtomId::from_parts(memory.memory_id(), u64::MAX);
    let foreign = AtomId::from_parts(memory_id(2), 0);

    assert_eq!(memory.recall_from(known, 0), Ok(Vec::new()));
    assert_eq!(memory.recall_from(known, usize::MAX), Ok(Vec::new()));
    assert_eq!(
        memory.recall_from(absent, 0),
        Err(GraphError::UnknownAtom(absent))
    );
    assert_eq!(
        memory.recall_from(foreign, 0),
        Err(GraphError::UnknownAtom(foreign))
    );
}

#[test]
fn recall_derives_length_normalized_structural_candidates_without_feedback() {
    let mut memory = new_memory(1);
    let source = insert_observation(&mut memory, 1, 10, &[100]);
    let exact = insert_observation(&mut memory, 2, 10, &[100]);
    let mut longer_draft = observation_draft(3, 10, &[100]);
    longer_draft.context.push(statement(30, &[300]));
    let longer = memory.insert_episode(longer_draft).expect("target inserts");
    let predicate_only = insert_observation(&mut memory, 4, 10, &[200]);
    let term_only = insert_observation(&mut memory, 5, 20, &[100]);
    let unrelated = insert_observation(&mut memory, 6, 20, &[200]);

    let hits = memory
        .recall_from(source, usize::MAX)
        .expect("source is known");
    assert_eq!(
        hits,
        vec![
            nao_m_e::RecallHit {
                atom_id: exact,
                activation: activation(400_000),
            },
            nao_m_e::RecallHit {
                atom_id: longer,
                activation: activation(200_000),
            },
            nao_m_e::RecallHit {
                atom_id: predicate_only,
                activation: activation(92_307),
            },
            nao_m_e::RecallHit {
                atom_id: term_only,
                activation: activation(26_666),
            },
        ]
    );
    assert!(!hits.iter().any(|hit| hit.atom_id == unrelated));
    assert_eq!(memory.feedback_edges().count(), 0);
}

#[test]
fn structural_recall_separates_namespaces_roles_and_argument_positions() {
    let mut memory = new_memory(1);
    let source = insert_observation(&mut memory, 1, 10, &[100, 200]);
    let positional = insert_observation(&mut memory, 2, 10, &[100, 999]);
    let reordered = insert_observation(&mut memory, 3, 10, &[200, 100]);
    let mut other_role_draft = observation_draft(4, 20, &[300]);
    other_role_draft.action = Some(statement(10, &[100, 200]));
    let other_role = memory
        .insert_episode(other_role_draft)
        .expect("target inserts");
    insert_observation(&mut memory, 5, 100, &[10]);

    assert_eq!(
        memory
            .recall_from(source, usize::MAX)
            .expect("source is known"),
        vec![
            nao_m_e::RecallHit {
                atom_id: positional,
                activation: activation(177_777),
            },
            nao_m_e::RecallHit {
                atom_id: reordered,
                activation: activation(95_238),
            },
            nao_m_e::RecallHit {
                atom_id: other_role,
                activation: activation(38_709),
            },
        ]
    );
}

#[test]
fn cue_index_uses_canonical_context_and_rebuilds_from_atom_sequence() {
    let mut repeated = observation_draft(1, 10, &[100]);
    repeated.context = vec![statement(20, &[200]), statement(20, &[200])];
    let mut canonical = observation_draft(2, 10, &[100]);
    canonical.context = vec![statement(20, &[200])];

    let mut original = new_memory(1);
    let source = original
        .insert_episode(repeated.clone())
        .expect("source inserts");
    let target = original
        .insert_episode(canonical.clone())
        .expect("target inserts");
    let expected = vec![nao_m_e::RecallHit {
        atom_id: target,
        activation: activation(400_000),
    }];
    assert_eq!(
        original.recall_from(source, usize::MAX),
        Ok(expected.clone())
    );

    let mut reconstructed = new_memory(1);
    let reconstructed_source = reconstructed
        .insert_episode(repeated)
        .expect("source reconstructs");
    assert_eq!(
        reconstructed.recall_from(reconstructed_source, usize::MAX),
        Ok(Vec::new())
    );
    reconstructed
        .insert_episode(canonical)
        .expect("target updates initialized index");
    assert_eq!(
        reconstructed.recall_from(reconstructed_source, usize::MAX),
        Ok(expected)
    );
}
