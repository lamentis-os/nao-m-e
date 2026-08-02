use nao_m_e::{
    Activation, AtomId, EpisodeDraft, FEEDBACK_MAX_EVENT_PPM, FEEDBACK_TARGET_STEP_PPM, GraphError,
    InfluenceWeight, MAX_FEEDBACK_TARGETS, MemoryId, MemoryIdError, MemoryV0, ModelError,
    PredicateId, SCALE, SourceId, Statement, TermId, TimestampMs, ValueError,
};

fn memory_id(value: u128) -> MemoryId {
    MemoryId::new(value).expect("test memory identifier is non-zero")
}

fn new_memory(id: u128) -> MemoryV0 {
    MemoryV0::new(memory_id(id))
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

fn insert(memory: &mut MemoryV0, seed: u64) -> AtomId {
    memory
        .insert_episode(draft(seed))
        .expect("identifier space is available")
}

fn activation(value: u32) -> Activation {
    Activation::from_ppm(value).expect("test activation is bounded")
}

fn weight(value: u32) -> InfluenceWeight {
    InfluenceWeight::from_ppm(value).expect("test weight is positive and bounded")
}

fn relevance_snapshot(memory: &MemoryV0) -> Vec<(AtomId, AtomId, u32)> {
    memory
        .relevance_edges()
        .map(|edge| (edge.from(), edge.to(), edge.weight().as_ppm()))
        .collect()
}

#[test]
fn model_constructors_enforce_their_boundaries() {
    assert_eq!(FEEDBACK_TARGET_STEP_PPM, 1_000);
    assert_eq!(FEEDBACK_MAX_EVENT_PPM, 10_000);
    assert_eq!(MAX_FEEDBACK_TARGETS, 10_000);
    assert_eq!(
        Statement::new(PredicateId::new(1), Vec::new()),
        Err(ModelError::EmptyArguments)
    );
    assert_eq!(
        Activation::from_ppm(SCALE + 1),
        Err(ValueError::OutOfRange { value: SCALE + 1 })
    );
    assert_eq!(InfluenceWeight::from_ppm(0), Err(ValueError::ZeroWeight));
    assert_eq!(
        InfluenceWeight::from_ppm(SCALE + 1),
        Err(ValueError::OutOfRange { value: SCALE + 1 })
    );
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

    let lower_half_max = memory_id(u128::from(u64::MAX));
    let upper_half_min = memory_id(1_u128 << 64);
    assert!(lower_half_max < upper_half_min);
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
        let mut original = MemoryV0::new(durable_id);
        assert_eq!(original.memory_id(), durable_id);
        [insert(&mut original, 1), insert(&mut original, 2)]
    };

    let mut reopened = MemoryV0::new(durable_id);
    let reopened_ids = [insert(&mut reopened, 1), insert(&mut reopened, 2)];

    assert_eq!(reopened.memory_id(), durable_id);
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
        .expect("first atom inserts");
    let second = memory
        .insert_episode(episode)
        .expect("identical occurrence also inserts");

    assert_ne!(first, second);
    assert_eq!(
        memory.episode(first).expect("atom exists").context(),
        &[low, statement(2, &[1])]
    );
}

#[test]
fn statements_and_episode_metadata_preserve_caller_semantics() {
    let ordered = statement(7, &[30, 20, 10]);
    assert_eq!(
        ordered.arguments(),
        &[TermId::new(30), TermId::new(20), TermId::new(10)]
    );

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
fn atom_content_is_immutable_across_graph_changes() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    let original = memory.episode(source).expect("atom exists").clone();

    memory
        .set_relevance(source, target, weight(SCALE))
        .expect("edge inserts");

    assert_eq!(memory.episode(source), Some(&original));
    assert_eq!(memory.relevance(source, target), Some(weight(SCALE)));
}

#[test]
fn graph_validation_is_atomic_and_budgeted() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let first = insert(&mut memory, 2);
    let second = insert(&mut memory, 3);

    assert_eq!(
        memory
            .set_relevance(source, first, weight(700_000))
            .expect("first edge fits"),
        None
    );
    let rejected = memory.set_relevance(source, second, weight(400_000));
    assert_eq!(
        rejected,
        Err(GraphError::OutgoingWeightBudgetExceeded {
            from: source,
            attempted_ppm: 1_100_000,
        })
    );
    assert_eq!(memory.relevance(source, first), Some(weight(700_000)));
    assert_eq!(memory.relevance(source, second), None);
    assert!(
        rejected
            .expect_err("budget violation is retained for diagnostics")
            .to_string()
            .contains(&source.to_string())
    );

    assert_eq!(
        memory.set_relevance(source, source, weight(1)),
        Err(GraphError::SelfEdge(source))
    );
    assert!(
        GraphError::SelfEdge(source)
            .to_string()
            .contains(&source.to_string())
    );

    let mut other = new_memory(2);
    let unknown = (0..4)
        .fold(None, |_, seed| Some(insert(&mut other, seed)))
        .unwrap();
    assert_eq!(
        memory.set_relevance(source, unknown, weight(1)),
        Err(GraphError::UnknownAtom(unknown))
    );

    assert_eq!(
        memory
            .set_relevance(source, first, weight(600_000))
            .expect("replacement frees budget"),
        Some(weight(700_000))
    );
    memory
        .set_relevance(source, second, weight(400_000))
        .expect("remaining budget fits");
    assert_eq!(
        memory.set_relevance(source, first, weight(700_000)),
        Err(GraphError::OutgoingWeightBudgetExceeded {
            from: source,
            attempted_ppm: 1_100_000,
        })
    );
    assert_eq!(memory.relevance(source, first), Some(weight(600_000)));
    assert_eq!(memory.relevance(source, second), Some(weight(400_000)));
    assert_eq!(
        memory
            .remove_relevance(source, first)
            .expect("endpoints are known"),
        Some(weight(600_000))
    );
    assert_eq!(memory.relevance(source, first), None);
    memory
        .set_relevance(source, first, weight(600_000))
        .expect("removal restores cached budget");
    memory
        .remove_relevance(source, first)
        .expect("first cached edge removes");
    memory
        .remove_relevance(source, second)
        .expect("last cached edge removes its sparse source");
    assert_eq!(memory.relevance_edges().count(), 0);
    memory
        .set_relevance(source, first, weight(SCALE))
        .expect("removed sparse source can be recreated");
}

#[test]
fn feedback_validates_source_and_every_target_before_mutation() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let first = insert(&mut memory, 2);
    let second = insert(&mut memory, 3);
    memory
        .set_relevance(source, first, weight(600_000))
        .expect("first edge fits");
    memory
        .set_relevance(source, second, weight(400_000))
        .expect("second edge fits");
    let before = relevance_snapshot(&memory);

    let mut foreign_memory = new_memory(2);
    let foreign = insert(&mut foreign_memory, 4);
    let absent = AtomId::from_parts(memory.memory_id(), u64::MAX);
    assert_eq!(
        memory.apply_feedback(source, &[first, foreign], true),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(relevance_snapshot(&memory), before);
    assert_eq!(
        memory.apply_feedback(source, &[second, foreign], false),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(relevance_snapshot(&memory), before);
    assert_eq!(
        memory.apply_feedback(foreign, &[first], true),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(relevance_snapshot(&memory), before);
    assert_eq!(
        memory.apply_feedback(source, &[first, absent], true),
        Err(GraphError::UnknownAtom(absent))
    );
    assert_eq!(relevance_snapshot(&memory), before);
    assert_eq!(
        memory.apply_feedback(absent, &[first], false),
        Err(GraphError::UnknownAtom(absent))
    );
    assert_eq!(relevance_snapshot(&memory), before);
}

#[test]
fn feedback_target_limit_keeps_the_event_share_positive() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let targets: Vec<_> = (0..MAX_FEEDBACK_TARGETS)
        .map(|index| {
            insert(
                &mut memory,
                u64::try_from(index + 2).expect("small target index"),
            )
        })
        .collect();

    memory
        .apply_feedback(source, &targets, true)
        .expect("the maximum target count is accepted");
    assert_eq!(memory.relevance(source, targets[0]), Some(weight(1)));
    assert_eq!(
        memory.relevance(source, targets[MAX_FEEDBACK_TARGETS - 1]),
        Some(weight(1))
    );
    assert_eq!(
        memory
            .relevance_edges()
            .map(|edge| edge.weight().as_ppm())
            .sum::<u32>(),
        FEEDBACK_MAX_EVENT_PPM
    );

    let before = relevance_snapshot(&memory);
    let duplicate = targets[0];
    let mut too_many = targets;
    too_many.push(duplicate);
    assert_eq!(
        memory.apply_feedback(source, &too_many, false),
        Err(GraphError::FeedbackTargetLimitExceeded {
            count: MAX_FEEDBACK_TARGETS + 1,
            max: MAX_FEEDBACK_TARGETS,
        })
    );
    assert_eq!(relevance_snapshot(&memory), before);

    memory
        .apply_feedback(source, &too_many[..MAX_FEEDBACK_TARGETS], false)
        .expect("the maximum target count keeps a non-zero negative share");
    assert_eq!(memory.relevance_edges().count(), 0);
}

#[test]
fn feedback_ignores_self_duplicates_and_empty_effective_lists() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let other = insert(&mut memory, 2);
    memory
        .set_relevance(source, other, weight(300_000))
        .expect("edge fits");
    let before = relevance_snapshot(&memory);

    memory
        .apply_feedback(source, &[], true)
        .expect("empty list is a no-op");
    memory
        .apply_feedback(source, &[source, source], true)
        .expect("positive self-only list is a no-op");
    memory
        .apply_feedback(source, &[source, source], false)
        .expect("negative self-only list is a no-op");

    assert_eq!(relevance_snapshot(&memory), before);
    assert_eq!(memory.relevance(source, source), None);
}

#[test]
fn positive_feedback_uses_free_budget_before_rebalancing() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let first = insert(&mut memory, 2);
    let second = insert(&mut memory, 3);
    let non_target = insert(&mut memory, 4);
    memory
        .set_relevance(source, first, weight(100_000))
        .expect("target edge fits");
    memory
        .set_relevance(source, non_target, weight(800_000))
        .expect("non-target edge fits");
    let episodes_before: Vec<_> = memory.episodes().cloned().collect();

    memory
        .apply_feedback(source, &[second, source, first, second, first], true)
        .expect("known feedback targets");

    assert_eq!(memory.relevance(source, first), Some(weight(101_000)));
    assert_eq!(memory.relevance(source, second), Some(weight(1_000)));
    assert_eq!(memory.relevance(source, non_target), Some(weight(800_000)));
    assert_eq!(memory.relevance(source, source), None);
    assert_eq!(
        memory.episodes().cloned().collect::<Vec<_>>(),
        episodes_before
    );
}

#[test]
fn repeated_positive_feedback_uses_target_step_and_event_budget() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let targets = [
        insert(&mut memory, 2),
        insert(&mut memory, 3),
        insert(&mut memory, 4),
    ];
    let non_target = insert(&mut memory, 5);
    memory
        .set_relevance(source, non_target, weight(SCALE))
        .expect("initial edge fills the budget");

    for iteration in 0..4 {
        let before: u32 = targets
            .iter()
            .filter_map(|&target| memory.relevance(source, target))
            .map(InfluenceWeight::as_ppm)
            .sum();
        memory
            .apply_feedback(source, &targets, true)
            .expect("known feedback targets");
        let after: u32 = targets
            .iter()
            .filter_map(|&target| memory.relevance(source, target))
            .map(InfluenceWeight::as_ppm)
            .sum();

        assert!(after >= before);
        assert_eq!(after - before, 3 * FEEDBACK_TARGET_STEP_PPM);
        assert!(after - before <= FEEDBACK_MAX_EVENT_PPM);
        if iteration == 0 {
            for &target in &targets {
                assert_eq!(
                    memory.relevance(source, target),
                    Some(weight(FEEDBACK_TARGET_STEP_PPM))
                );
            }
        }
    }
}

#[test]
fn learned_shortcut_changes_the_next_source_conditioned_recall() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    memory
        .apply_feedback(source, &[target], true)
        .expect("known feedback target");

    assert_eq!(
        memory.relevance(source, target),
        Some(weight(FEEDBACK_TARGET_STEP_PPM))
    );
    assert_eq!(
        memory.recall_from(source, 1).expect("source is known"),
        vec![nao_m_e::RecallHit {
            atom_id: target,
            activation: activation(400),
        }]
    );
}

#[test]
fn positive_feedback_funds_exactly_with_carried_remainders() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let missing_before = insert(&mut memory, 2);
    let non_target_a = insert(&mut memory, 3);
    let existing_target = insert(&mut memory, 4);
    let missing_between = insert(&mut memory, 5);
    let non_target_b = insert(&mut memory, 6);
    let missing_after = insert(&mut memory, 7);
    memory
        .set_relevance(source, non_target_a, weight(400_001))
        .expect("first non-target fits");
    memory
        .set_relevance(source, existing_target, weight(100_000))
        .expect("existing target fits");
    memory
        .set_relevance(source, non_target_b, weight(499_999))
        .expect("second non-target fills the budget");

    memory
        .apply_feedback(
            source,
            &[
                missing_after,
                existing_target,
                missing_before,
                missing_between,
                existing_target,
                source,
                missing_after,
            ],
            true,
        )
        .expect("target order and duplicates do not matter");

    assert_eq!(
        memory.relevance(source, non_target_a),
        Some(weight(398_224))
    );
    assert_eq!(
        memory.relevance(source, existing_target),
        Some(weight(101_000))
    );
    assert_eq!(
        memory.relevance(source, non_target_b),
        Some(weight(497_776))
    );
    for target in [missing_before, missing_between, missing_after] {
        assert_eq!(memory.relevance(source, target), Some(weight(1_000)));
    }
    assert_eq!(
        memory
            .relevance_edges()
            .filter(|edge| edge.from() == source)
            .map(|edge| edge.weight().as_ppm())
            .sum::<u32>(),
        SCALE
    );
}

#[test]
fn positive_feedback_funding_is_independent_of_non_target_fragmentation() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    for seed in 3..103 {
        let non_target = insert(&mut memory, seed);
        memory
            .set_relevance(source, non_target, weight(1))
            .expect("fragmented edge fits");
    }
    let large_non_target = insert(&mut memory, 103);
    memory
        .set_relevance(source, large_non_target, weight(999_900))
        .expect("large edge fills the outgoing budget");

    memory
        .apply_feedback(source, &[target], true)
        .expect("known feedback target");

    assert_eq!(memory.relevance(source, target), Some(weight(1_000)));
    assert_eq!(
        memory.relevance(source, large_non_target),
        Some(weight(998_900))
    );
    assert_eq!(
        memory
            .relevance_edges()
            .filter(|edge| edge.from() == source)
            .map(|edge| edge.weight().as_ppm())
            .sum::<u32>(),
        SCALE
    );
}

#[test]
fn positive_feedback_caps_award_by_remaining_target_capacity() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let first = insert(&mut memory, 2);
    let second = insert(&mut memory, 3);
    let non_target = insert(&mut memory, 4);
    memory
        .set_relevance(source, first, weight(600_000))
        .expect("first target fits");
    memory
        .set_relevance(source, second, weight(399_000))
        .expect("second target fits");
    memory
        .set_relevance(source, non_target, weight(1_000))
        .expect("non-target fits");

    memory
        .apply_feedback(source, &[first, second], true)
        .expect("known feedback targets");

    assert_eq!(memory.relevance(source, first), Some(weight(600_500)));
    assert_eq!(memory.relevance(source, second), Some(weight(399_500)));
    assert_eq!(memory.relevance(source, non_target), None);
}

#[test]
fn negative_feedback_splits_the_target_step_drops_zero_and_preserves_non_targets() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let first = insert(&mut memory, 2);
    let second = insert(&mut memory, 3);
    let non_target = insert(&mut memory, 4);
    let missing_target = insert(&mut memory, 5);
    memory
        .set_relevance(source, first, weight(1_500))
        .expect("first target fits");
    memory
        .set_relevance(source, second, weight(500))
        .expect("second target fits");
    memory
        .set_relevance(source, non_target, weight(998_000))
        .expect("non-target fits");

    memory
        .apply_feedback(source, &[second, source, first, first], false)
        .expect("known feedback targets");

    assert_eq!(memory.relevance(source, first), Some(weight(500)));
    assert_eq!(memory.relevance(source, second), None);
    assert_eq!(memory.relevance(source, non_target), Some(weight(998_000)));
    memory
        .apply_feedback(source, &[first], false)
        .expect("remaining target can reach zero");
    assert_eq!(memory.relevance(source, first), None);
    assert_eq!(memory.relevance(source, non_target), Some(weight(998_000)));
    let before_missing = relevance_snapshot(&memory);
    memory
        .apply_feedback(source, &[missing_target], false)
        .expect("missing target edge has no effect");
    assert_eq!(relevance_snapshot(&memory), before_missing);
}

#[test]
fn foreign_and_absent_ids_cannot_alias_local_atoms() {
    let mut first_memory = new_memory(1);
    let local = insert(&mut first_memory, 1);
    let mut second_memory = new_memory(2);
    let foreign = insert(&mut second_memory, 1);
    let absent_local = AtomId::from_parts(first_memory.memory_id(), u64::MAX);

    assert_eq!(local.sequence(), foreign.sequence());
    assert_ne!(local.memory_id(), foreign.memory_id());
    assert_eq!(
        local.to_string(),
        format!("{}:{}", local.memory_id(), local.sequence())
    );
    assert_ne!(local.to_string(), foreign.to_string());
    assert_eq!(first_memory.episode(foreign), None);
    assert_eq!(first_memory.relevance(local, foreign), None);
    assert_eq!(first_memory.relevance(foreign, local), None);
    assert_eq!(first_memory.episode(absent_local), None);
    assert_eq!(first_memory.relevance(local, absent_local), None);
    assert_eq!(first_memory.relevance(absent_local, local), None);
    assert_eq!(
        first_memory.set_relevance(local, foreign, weight(1)),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(
        first_memory.set_relevance(foreign, local, weight(1)),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(
        first_memory.set_relevance(local, absent_local, weight(1)),
        Err(GraphError::UnknownAtom(absent_local))
    );
    assert_eq!(
        first_memory.set_relevance(absent_local, local, weight(1)),
        Err(GraphError::UnknownAtom(absent_local))
    );
    assert_eq!(
        first_memory.remove_relevance(local, foreign),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(
        first_memory.remove_relevance(foreign, local),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(
        first_memory.remove_relevance(local, absent_local),
        Err(GraphError::UnknownAtom(absent_local))
    );
    assert_eq!(
        first_memory.remove_relevance(absent_local, local),
        Err(GraphError::UnknownAtom(absent_local))
    );
    assert!(
        GraphError::UnknownAtom(foreign)
            .to_string()
            .contains(&foreign.to_string())
    );
    assert_eq!(first_memory.relevance_edges().count(), 0);
}

#[test]
fn empty_memory_has_no_episodes_or_relevance() {
    let memory = new_memory(1);

    assert_eq!(memory.episodes().len(), 0);
    assert_eq!(memory.relevance_edges().count(), 0);
}

#[test]
fn relevance_is_directed_sorted_and_removal_of_absent_edge_is_a_noop() {
    let mut memory = new_memory(1);
    let a = insert(&mut memory, 1);
    let b = insert(&mut memory, 2);
    let c = insert(&mut memory, 3);
    assert_eq!(
        memory
            .episodes()
            .rev()
            .map(|episode| episode.id())
            .collect::<Vec<_>>(),
        vec![c, b, a]
    );
    memory.set_relevance(b, c, weight(200_000)).expect("B to C");
    memory.set_relevance(a, c, weight(300_000)).expect("A to C");
    memory.set_relevance(a, b, weight(400_000)).expect("A to B");

    assert_eq!(memory.relevance(b, a), None);
    let ordered: Vec<_> = memory
        .relevance_edges()
        .map(|edge| (edge.from(), edge.to(), edge.weight()))
        .collect();
    assert_eq!(
        ordered,
        vec![
            (a, b, weight(400_000)),
            (a, c, weight(300_000)),
            (b, c, weight(200_000)),
        ]
    );
    assert_eq!(
        memory.remove_relevance(c, a).expect("known endpoints"),
        None
    );
}

#[test]
fn recall_from_projects_only_the_source_row_without_mutating_state() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let strongest = insert(&mut memory, 2);
    let weaker = insert(&mut memory, 3);
    let weakest = insert(&mut memory, 4);
    let incoming = insert(&mut memory, 5);
    let downstream = insert(&mut memory, 6);
    memory
        .set_relevance(source, strongest, weight(600_000))
        .expect("strongest direct edge fits");
    memory
        .set_relevance(source, weaker, weight(250_000))
        .expect("weaker direct edge fits");
    memory
        .set_relevance(source, weakest, weight(150_000))
        .expect("weakest direct edge fits");
    memory
        .set_relevance(incoming, source, weight(SCALE))
        .expect("incoming edge fits");
    memory
        .set_relevance(strongest, downstream, weight(SCALE))
        .expect("downstream edge fits");
    let episodes_before: Vec<_> = memory.episodes().cloned().collect();
    let relevance_before = relevance_snapshot(&memory);

    assert_eq!(
        memory
            .recall_from(source, usize::MAX)
            .expect("source is known"),
        vec![
            nao_m_e::RecallHit {
                atom_id: strongest,
                activation: activation(240_000),
            },
            nao_m_e::RecallHit {
                atom_id: weaker,
                activation: activation(100_000),
            },
            nao_m_e::RecallHit {
                atom_id: weakest,
                activation: activation(60_000),
            },
        ]
    );
    assert_eq!(
        memory.episodes().cloned().collect::<Vec<_>>(),
        episodes_before
    );
    assert_eq!(relevance_snapshot(&memory), relevance_before);
}

#[test]
fn recall_from_excludes_rounded_zero_and_matches_full_ranking_for_every_limit() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let strongest = insert(&mut memory, 2);
    let first_tie = insert(&mut memory, 3);
    let second_tie = insert(&mut memory, 4);
    let one_ppm_edge = insert(&mut memory, 5);
    let two_ppm_edge = insert(&mut memory, 6);
    let three_ppm_edge = insert(&mut memory, 7);
    memory
        .set_relevance(source, strongest, weight(400_000))
        .expect("strongest edge fits");
    memory
        .set_relevance(source, first_tie, weight(250_000))
        .expect("first tied edge fits");
    memory
        .set_relevance(source, second_tie, weight(250_000))
        .expect("second tied edge fits");
    memory
        .set_relevance(source, one_ppm_edge, weight(1))
        .expect("one-ppm edge fits");
    memory
        .set_relevance(source, two_ppm_edge, weight(2))
        .expect("two-ppm edge fits");
    memory
        .set_relevance(source, three_ppm_edge, weight(3))
        .expect("three-ppm edge fits");

    let complete = memory
        .recall_from(source, usize::MAX)
        .expect("source is known");
    assert_eq!(
        complete,
        vec![
            nao_m_e::RecallHit {
                atom_id: strongest,
                activation: activation(160_000),
            },
            nao_m_e::RecallHit {
                atom_id: first_tie,
                activation: activation(100_000),
            },
            nao_m_e::RecallHit {
                atom_id: second_tie,
                activation: activation(100_000),
            },
            nao_m_e::RecallHit {
                atom_id: three_ppm_edge,
                activation: activation(1),
            },
        ]
    );
    assert!(!complete.iter().any(|hit| hit.atom_id == one_ppm_edge));
    assert!(!complete.iter().any(|hit| hit.atom_id == two_ppm_edge));

    for limit in 0..=complete.len() + 1 {
        assert_eq!(
            memory.recall_from(source, limit).expect("source is known"),
            complete.iter().copied().take(limit).collect::<Vec<_>>(),
            "limit {limit}"
        );
    }
}

#[test]
fn recall_from_validates_the_source_before_the_limit_and_handles_isolated_sources() {
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
fn feedback_updates_source_conditioned_recall_directly() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);

    memory
        .apply_feedback(source, &[target], true)
        .expect("positive feedback applies");
    assert_eq!(
        memory.recall_from(source, 1).expect("source is known"),
        vec![nao_m_e::RecallHit {
            atom_id: target,
            activation: activation(400),
        }]
    );

    memory
        .apply_feedback(source, &[target], false)
        .expect("negative feedback applies");
    assert!(
        memory
            .recall_from(source, 1)
            .expect("source is known")
            .is_empty()
    );
}
