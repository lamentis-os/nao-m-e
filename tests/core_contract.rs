//! Executable contract tests for the public memory kernel.

use nao_m_e::{
    Activation, AtomId, EpisodeDraft, FEEDBACK_HISTORY_CAPACITY, FEEDBACK_PRIOR_MASS,
    FeedbackTrace, GraphError, LEARNED_GAIN_PPM, MAX_FEEDBACK_TARGETS, Memory, MemoryId,
    MemoryIdError, ModelError, PredicateId, SCALE, STRUCTURAL_GAIN_PPM, SourceId, Statement,
    TermId, TimestampMs, ValueError,
};

fn memory_id(value: u128) -> MemoryId {
    MemoryId::new(value).expect("test memory identifier is non-zero")
}

fn new_memory(id: u128) -> Memory {
    Memory::new(memory_id(id))
}

fn statement(predicate: u64, arguments: &[u64]) -> Statement {
    Statement::new(
        PredicateId::new(predicate),
        arguments.iter().copied().map(TermId::new).collect(),
    )
    .expect("test statement is valid")
}

fn draft(seed: u64) -> EpisodeDraft {
    EpisodeDraft {
        occurred_at: TimestampMs::new(i64::try_from(seed).expect("small test seed")),
        recorded_at: TimestampMs::new(i64::try_from(seed + 10).expect("small test seed")),
        context: vec![statement(10 + seed, &[100 + seed])],
        observation: statement(20 + seed, &[200 + seed]),
        action: Some(statement(30 + seed, &[300 + seed])),
        outcome: Some(statement(40 + seed, &[400 + seed])),
        source: SourceId::new(50 + seed),
    }
}

fn insert(memory: &mut Memory, seed: u64) -> AtomId {
    memory
        .insert_episode(draft(seed))
        .expect("identifier space is available")
}

fn observation_draft(seed: u64, predicate: u64, arguments: &[u64]) -> EpisodeDraft {
    EpisodeDraft {
        occurred_at: TimestampMs::new(i64::try_from(seed).expect("small test seed")),
        recorded_at: TimestampMs::new(i64::try_from(seed + 1).expect("small test seed")),
        context: Vec::new(),
        observation: statement(predicate, arguments),
        action: None,
        outcome: None,
        source: SourceId::new(seed),
    }
}

fn insert_observation(memory: &mut Memory, seed: u64, predicate: u64, arguments: &[u64]) -> AtomId {
    memory
        .insert_episode(observation_draft(seed, predicate, arguments))
        .expect("identifier space is available")
}

fn activation(value: u32) -> Activation {
    Activation::from_ppm(value).expect("test activation is bounded")
}

fn trace(history_bits: u16, sample_count: u8) -> FeedbackTrace {
    FeedbackTrace::from_parts(history_bits, sample_count).expect("test trace is canonical")
}

fn feedback_snapshot(memory: &Memory) -> Vec<(AtomId, AtomId, FeedbackTrace)> {
    memory
        .feedback_edges()
        .map(|edge| (edge.from(), edge.to(), edge.trace()))
        .collect()
}

#[test]
fn model_constructors_and_feedback_parameters_enforce_their_boundaries() {
    assert_eq!(FEEDBACK_HISTORY_CAPACITY, 16);
    assert_eq!(FEEDBACK_PRIOR_MASS, 7);
    assert_eq!(LEARNED_GAIN_PPM, 400_000);
    assert_eq!(MAX_FEEDBACK_TARGETS, 10_000);
    assert_eq!(STRUCTURAL_GAIN_PPM, 400_000);
    assert_eq!(
        Statement::new(PredicateId::new(1), Vec::new()),
        Err(ModelError::EmptyArguments)
    );
    assert_eq!(
        Activation::from_ppm(SCALE + 1),
        Err(ValueError::OutOfRange { value: SCALE + 1 })
    );

    assert_eq!(FeedbackTrace::from_parts(0, 0), None);
    assert_eq!(FeedbackTrace::from_parts(0, 17), None);
    assert_eq!(FeedbackTrace::from_parts(0b10, 1), None);
    assert_eq!(FeedbackTrace::from_parts(u16::MAX, 15), None);
    let all_helpful = trace(u16::MAX, 16);
    assert_eq!(all_helpful.history_bits(), u16::MAX);
    assert_eq!(all_helpful.sample_count(), 16);
    assert_eq!(all_helpful.helpful_count(), 16);
    assert_eq!(all_helpful.unhelpful_count(), 0);
    let all_unhelpful = trace(0, 16);
    assert_eq!(all_unhelpful.helpful_count(), 0);
    assert_eq!(all_unhelpful.unhelpful_count(), 16);
    let mixed = trace(0b0101, 4);
    assert_eq!(mixed.helpful_count(), 2);
    assert_eq!(mixed.unhelpful_count(), 2);
}

#[test]
fn memory_ids_reject_zero_and_round_trip_their_canonical_bytes() {
    const VALUE: u128 = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
    const BYTES: [u8; 16] = [
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32,
        0x10,
    ];

    assert_eq!(MemoryId::new(0), Err(MemoryIdError::Zero));
    assert_eq!(MemoryId::from_be_bytes([0; 16]), Err(MemoryIdError::Zero));
    assert_eq!(
        MemoryIdError::Zero.to_string(),
        "a memory identifier must be non-zero"
    );

    let id = MemoryId::new(VALUE).expect("value is non-zero");
    assert_eq!(id.get(), VALUE);
    assert_eq!(id.to_be_bytes(), BYTES);
    assert_eq!(MemoryId::from_be_bytes(BYTES), Ok(id));
    assert_eq!(id.to_string(), "0123456789abcdeffedcba9876543210");
    assert!(memory_id(u128::from(u64::MAX)) < memory_id(1_u128 << 64));
}

#[test]
fn atom_ids_round_trip_components_and_have_canonical_ordering() {
    let first_memory = memory_id(1);
    let second_memory = memory_id(2);
    let first = AtomId::from_parts(first_memory, 0);
    let second = AtomId::from_parts(first_memory, 1);
    let foreign = AtomId::from_parts(second_memory, 0);

    assert_eq!(first.memory_id(), first_memory);
    assert_eq!(first.sequence(), 0);
    assert_eq!(
        AtomId::from_parts(first.memory_id(), first.sequence()),
        first
    );
    assert_eq!(first.to_string(), "00000000000000000000000000000001:0");
    assert!(first < second);
    assert!(second < foreign);
}

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
fn statements_and_episode_metadata_preserve_caller_semantics() {
    let ordered = statement(7, &[30, 20, 10]);
    let mut memory = new_memory(1);
    let atom = memory
        .insert_episode(EpisodeDraft {
            occurred_at: TimestampMs::new(-50),
            recorded_at: TimestampMs::new(75),
            context: Vec::new(),
            observation: ordered.clone(),
            action: Some(statement(8, &[1])),
            outcome: Some(statement(9, &[2])),
            source: SourceId::new(99),
        })
        .expect("episode inserts");
    let stored = memory.episode(atom).expect("episode exists");

    assert_eq!(stored.occurred_at(), TimestampMs::new(-50));
    assert_eq!(stored.recorded_at(), TimestampMs::new(75));
    assert_eq!(stored.observation(), &ordered);
    assert_eq!(stored.action(), Some(&statement(8, &[1])));
    assert_eq!(stored.outcome(), Some(&statement(9, &[2])));
    assert_eq!(stored.source(), SourceId::new(99));
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
fn empty_memory_has_no_episodes_or_feedback() {
    let memory = new_memory(1);
    assert_eq!(memory.episodes().len(), 0);
    assert_eq!(memory.feedback_edges().count(), 0);
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

#[test]
fn recall_unifies_structural_learned_and_combined_candidates_once() {
    let mut memory = new_memory(1);
    let source = insert_observation(&mut memory, 1, 10, &[100]);
    let structural_only = insert_observation(&mut memory, 2, 10, &[100]);
    let combined = insert_observation(&mut memory, 3, 10, &[200]);
    let learned_only = insert_observation(&mut memory, 4, 20, &[300]);
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
