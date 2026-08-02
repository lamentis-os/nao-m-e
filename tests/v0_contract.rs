use nao_m_e::{
    Activation, AtomId, EpisodeDraft, FEEDBACK_MAX_EVENT_PPM, FEEDBACK_TARGET_STEP_PPM, GraphError,
    InfluenceWeight, MAX_FEEDBACK_TARGETS, MemoryId, MemoryIdError, MemoryV0, ModelError,
    PROPAGATION_GAIN_PPM, PredicateId, RETENTION_PPM, SCALE, SourceId, Statement, TermId,
    TimestampMs, ValueError,
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

fn assert_activation(memory: &MemoryV0, id: AtomId, expected: u32) {
    assert_eq!(
        memory.activation(id).map(Activation::as_ppm),
        Some(expected)
    );
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
    assert_eq!(memory.activation(first), Some(Activation::ZERO));
    assert_eq!(memory.activation(second), Some(Activation::ZERO));
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
fn atom_content_is_immutable_across_graph_and_state_changes() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    let original = memory.episode(source).expect("atom exists").clone();

    memory
        .set_relevance(source, target, weight(SCALE))
        .expect("edge inserts");
    memory
        .stimulate(source, Activation::ONE)
        .expect("stimulus applies");
    memory.step();
    memory.reset_activations();

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
fn positive_feedback_splits_the_step_and_uses_free_budget_first() {
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
    memory
        .stimulate(source, activation(700_000))
        .expect("source stimulus applies");
    memory
        .stimulate(second, activation(300_000))
        .expect("target stimulus applies");
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
    assert_activation(&memory, source, 700_000);
    assert_activation(&memory, second, 300_000);
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
fn learned_shortcut_changes_the_next_step_by_the_existing_formula() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    memory
        .stimulate(source, Activation::ONE)
        .expect("source stimulus applies");
    memory
        .apply_feedback(source, &[target], true)
        .expect("known feedback target");

    assert_eq!(
        memory.relevance(source, target),
        Some(weight(FEEDBACK_TARGET_STEP_PPM))
    );
    memory.step();
    assert_activation(&memory, source, RETENTION_PPM);
    let expected_target = u32::try_from(
        u64::from(FEEDBACK_TARGET_STEP_PPM) * u64::from(PROPAGATION_GAIN_PPM) / u64::from(SCALE),
    )
    .expect("one propagated activation fits in ppm");
    assert_activation(&memory, target, expected_target);
}

#[test]
fn positive_feedback_funds_exactly_with_carried_remainders() {
    fn prepared_memory() -> (MemoryV0, [AtomId; 5]) {
        let mut memory = new_memory(1);
        let source = insert(&mut memory, 1);
        let first = insert(&mut memory, 2);
        let second = insert(&mut memory, 3);
        let non_target_a = insert(&mut memory, 4);
        let non_target_b = insert(&mut memory, 5);
        memory
            .set_relevance(source, first, weight(100_000))
            .expect("first target fits");
        memory
            .set_relevance(source, second, weight(200_000))
            .expect("second target fits");
        memory
            .set_relevance(source, non_target_a, weight(400_000))
            .expect("first non-target fits");
        memory
            .set_relevance(source, non_target_b, weight(300_000))
            .expect("second non-target fits");
        (memory, [source, first, second, non_target_a, non_target_b])
    }

    let (mut forward, [source, first, second, non_target_a, non_target_b]) = prepared_memory();
    let (mut reordered, reordered_ids) = prepared_memory();
    forward
        .apply_feedback(source, &[second, first], true)
        .expect("known feedback targets");
    reordered
        .apply_feedback(
            reordered_ids[0],
            &[
                reordered_ids[1],
                reordered_ids[0],
                reordered_ids[2],
                reordered_ids[1],
            ],
            true,
        )
        .expect("rank and duplicates do not matter");

    assert_eq!(relevance_snapshot(&forward), relevance_snapshot(&reordered));
    assert_eq!(forward.relevance(source, first), Some(weight(101_000)));
    assert_eq!(forward.relevance(source, second), Some(weight(201_000)));
    assert_eq!(
        forward.relevance(source, non_target_a),
        Some(weight(398_858))
    );
    assert_eq!(
        forward.relevance(source, non_target_b),
        Some(weight(299_142))
    );
    assert_eq!(
        forward
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
fn negative_feedback_splits_the_step_drops_zero_and_preserves_non_targets() {
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
    assert_eq!(first_memory.activation(foreign), None);
    assert_eq!(first_memory.relevance(local, foreign), None);
    assert_eq!(first_memory.relevance(foreign, local), None);
    assert_eq!(first_memory.episode(absent_local), None);
    assert_eq!(first_memory.activation(absent_local), None);
    assert_eq!(first_memory.relevance(local, absent_local), None);
    assert_eq!(first_memory.relevance(absent_local, local), None);
    assert_eq!(
        first_memory.stimulate(foreign, Activation::ONE),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(
        first_memory.stimulate(absent_local, Activation::ONE),
        Err(GraphError::UnknownAtom(absent_local))
    );
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
    assert_eq!(first_memory.activation(local), Some(Activation::ZERO));
    assert_eq!(first_memory.relevance_edges().count(), 0);
}

#[test]
fn empty_memory_operations_are_noops() {
    let mut memory = new_memory(1);

    assert_eq!(memory.episodes().len(), 0);
    assert_eq!(memory.relevance_edges().count(), 0);
    assert!(memory.top_k(0).is_empty());
    assert!(memory.top_k(10).is_empty());
    memory.step();
    memory.reset_activations();
    assert!(memory.top_k(10).is_empty());
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
fn stimulation_adds_and_saturates() {
    let mut memory = new_memory(1);
    let atom = insert(&mut memory, 1);

    assert_eq!(
        memory
            .stimulate(atom, activation(700_000))
            .expect("first stimulus"),
        activation(700_000)
    );
    assert_eq!(
        memory
            .stimulate(atom, activation(600_000))
            .expect("second stimulus"),
        Activation::ONE
    );
}

#[test]
fn golden_chain_propagates_synchronously() {
    let mut memory = new_memory(1);
    let a = insert(&mut memory, 1);
    let b = insert(&mut memory, 2);
    let c = insert(&mut memory, 3);
    memory.set_relevance(a, b, weight(SCALE)).expect("A to B");
    memory.set_relevance(b, c, weight(SCALE)).expect("B to C");
    memory
        .stimulate(a, Activation::ONE)
        .expect("seed activation");

    memory.step();
    assert_activation(&memory, a, 500_000);
    assert_activation(&memory, b, 400_000);
    assert_activation(&memory, c, 0);

    memory.step();
    assert_activation(&memory, a, 250_000);
    assert_activation(&memory, b, 400_000);
    assert_activation(&memory, c, 160_000);
}

#[test]
fn golden_branch_respects_weight_allocation() {
    let mut memory = new_memory(1);
    let a = insert(&mut memory, 1);
    let b = insert(&mut memory, 2);
    let c = insert(&mut memory, 3);
    memory
        .set_relevance(a, b, weight(750_000))
        .expect("first branch");
    memory
        .set_relevance(a, c, weight(250_000))
        .expect("second branch");
    memory
        .stimulate(a, Activation::ONE)
        .expect("seed activation");

    memory.step();
    assert_activation(&memory, a, 500_000);
    assert_activation(&memory, b, 300_000);
    assert_activation(&memory, c, 100_000);
}

#[test]
fn incoming_contributions_are_aggregated_before_rounding() {
    let mut memory = new_memory(1);
    let sources: Vec<_> = (0..3).map(|seed| insert(&mut memory, seed)).collect();
    let target = insert(&mut memory, 10);
    for source in sources {
        memory
            .set_relevance(source, target, weight(SCALE))
            .expect("independent source edge");
        memory
            .stimulate(source, activation(1))
            .expect("one ppm stimulus");
    }

    memory.step();
    assert_activation(&memory, target, 1);
}

#[test]
fn individually_sub_ppm_weights_can_combine_at_one_target() {
    let mut memory = new_memory(1);
    let sources: Vec<_> = (0..3).map(|seed| insert(&mut memory, seed)).collect();
    let target = insert(&mut memory, 10);
    for source in sources {
        memory
            .set_relevance(source, target, weight(1))
            .expect("positive sub-ppm contribution is retained");
        memory
            .stimulate(source, Activation::ONE)
            .expect("full source stimulus");
    }

    memory.step();
    assert_activation(&memory, target, 1);
}

#[test]
fn retention_and_propagation_share_one_rounding_boundary() {
    let mut memory = new_memory(1);
    let source = insert(&mut memory, 1);
    let target = insert(&mut memory, 2);
    memory
        .set_relevance(source, target, weight(SCALE))
        .expect("full edge");
    memory
        .stimulate(source, activation(2))
        .expect("two ppm source stimulus");
    memory
        .stimulate(target, activation(1))
        .expect("one ppm retained stimulus");

    memory.step();
    assert_activation(&memory, target, 1);
}

#[test]
fn converging_incoming_edges_clamp_target_at_full_activation() {
    let mut memory = new_memory(1);
    let sources: Vec<_> = (0..3).map(|seed| insert(&mut memory, seed)).collect();
    let target = insert(&mut memory, 10);
    for source in sources {
        memory
            .set_relevance(source, target, weight(SCALE))
            .expect("independent source edge");
        memory
            .stimulate(source, Activation::ONE)
            .expect("full source stimulus");
    }

    memory.step();
    assert_activation(&memory, target, SCALE);
}

#[test]
fn cycles_are_bounded_and_lose_total_activation() {
    let mut memory = new_memory(1);
    let a = insert(&mut memory, 1);
    let b = insert(&mut memory, 2);
    memory.set_relevance(a, b, weight(SCALE)).expect("A to B");
    memory.set_relevance(b, a, weight(SCALE)).expect("B to A");
    memory
        .stimulate(a, Activation::ONE)
        .expect("seed activation");

    let mut previous_total = u128::from(SCALE);
    for _ in 0..32 {
        memory.step();
        let current_total = u128::from(
            memory.activation(a).expect("A exists").as_ppm()
                + memory.activation(b).expect("B exists").as_ppm(),
        );
        assert!(
            current_total * u128::from(SCALE)
                <= previous_total * u128::from(RETENTION_PPM + PROPAGATION_GAIN_PPM)
        );
        previous_total = current_total;
    }
}

#[test]
fn disconnected_components_do_not_receive_activation() {
    let mut memory = new_memory(1);
    let a = insert(&mut memory, 1);
    let b = insert(&mut memory, 2);
    let x = insert(&mut memory, 3);
    let y = insert(&mut memory, 4);
    memory
        .set_relevance(a, b, weight(SCALE))
        .expect("active component");
    memory
        .set_relevance(x, y, weight(SCALE))
        .expect("disconnected component");
    memory
        .stimulate(a, Activation::ONE)
        .expect("seed activation");

    for _ in 0..4 {
        memory.step();
    }
    assert_activation(&memory, x, 0);
    assert_activation(&memory, y, 0);
}

#[test]
fn top_k_excludes_zero_and_breaks_ties_by_atom_id() {
    let mut memory = new_memory(1);
    let first = insert(&mut memory, 1);
    let second = insert(&mut memory, 2);
    let zero = insert(&mut memory, 3);
    memory
        .stimulate(first, activation(500_000))
        .expect("first stimulus");
    memory
        .stimulate(second, activation(500_000))
        .expect("second stimulus");

    let hits = memory.top_k(10);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].atom_id, first);
    assert_eq!(hits[1].atom_id, second);
    assert!(!hits.iter().any(|hit| hit.atom_id == zero));
    assert!(memory.top_k(0).is_empty());
}

#[test]
fn top_k_small_limit_keeps_only_the_best_ranked_hits() {
    let mut memory = new_memory(1);
    let ids: Vec<_> = (0..6).map(|seed| insert(&mut memory, seed)).collect();
    for (id, value) in ids
        .iter()
        .copied()
        .zip([100_000, 900_000, 500_000, 900_000, 700_000, 900_000])
    {
        memory.stimulate(id, activation(value)).expect("known atom");
    }

    let hits = memory.top_k(2);
    assert_eq!(
        hits.iter().map(|hit| hit.atom_id).collect::<Vec<_>>(),
        vec![ids[1], ids[3]]
    );
    assert!(hits.iter().all(|hit| hit.activation == activation(900_000)));
}

#[test]
fn top_k_matches_full_ranking_for_every_limit() {
    let mut memory = new_memory(1);
    let ids: Vec<_> = (0..12).map(|seed| insert(&mut memory, seed)).collect();
    for (id, value) in ids
        .iter()
        .copied()
        .zip([0, 20, 20, 1, 900_000, 0, 700_000, 20, 3, 900_000, 2, 0])
    {
        memory.stimulate(id, activation(value)).expect("known atom");
    }

    let complete = memory.top_k(ids.len());
    for limit in 0..=ids.len() + 1 {
        let expected: Vec<_> = complete.iter().copied().take(limit).collect();
        assert_eq!(memory.top_k(limit), expected, "limit {limit}");
    }
}

#[test]
fn step_buffers_remain_aligned_after_insertion_and_reset() {
    let mut memory = new_memory(1);
    let first = insert(&mut memory, 1);
    let second = insert(&mut memory, 2);
    memory
        .set_relevance(first, second, weight(SCALE))
        .expect("first edge");
    memory
        .stimulate(first, Activation::ONE)
        .expect("initial stimulus");
    memory.step();

    let inserted_after_step = insert(&mut memory, 3);
    memory
        .set_relevance(second, inserted_after_step, weight(SCALE))
        .expect("edge to newly inserted atom");
    memory.step();
    assert_activation(&memory, first, 250_000);
    assert_activation(&memory, second, 400_000);
    assert_activation(&memory, inserted_after_step, 160_000);

    memory.reset_activations();
    memory.step();
    assert!(memory.top_k(3).is_empty());
}

#[test]
fn edge_insertion_order_does_not_change_dynamics() {
    let mut forward = new_memory(1);
    let mut reverse = new_memory(2);
    let forward_ids: Vec<_> = (0..4).map(|seed| insert(&mut forward, seed)).collect();
    let reverse_ids: Vec<_> = (0..4).map(|seed| insert(&mut reverse, seed)).collect();
    let edges = [
        (0, 1, 400_000),
        (0, 2, 600_000),
        (1, 3, SCALE),
        (2, 3, SCALE),
    ];

    for &(from, to, ppm) in &edges {
        forward
            .set_relevance(forward_ids[from], forward_ids[to], weight(ppm))
            .expect("forward edge");
    }
    for &(from, to, ppm) in edges.iter().rev() {
        reverse
            .set_relevance(reverse_ids[from], reverse_ids[to], weight(ppm))
            .expect("reverse edge");
    }
    forward
        .stimulate(forward_ids[0], Activation::ONE)
        .expect("forward stimulus");
    reverse
        .stimulate(reverse_ids[0], Activation::ONE)
        .expect("reverse stimulus");

    for _ in 0..5 {
        forward.step();
        reverse.step();
        for index in 0..4 {
            assert_eq!(
                forward.activation(forward_ids[index]),
                reverse.activation(reverse_ids[index])
            );
        }
        let forward_ranking: Vec<_> = forward
            .top_k(4)
            .into_iter()
            .map(|hit| (hit.atom_id.sequence(), hit.activation))
            .collect();
        let reverse_ranking: Vec<_> = reverse
            .top_k(4)
            .into_iter()
            .map(|hit| (hit.atom_id.sequence(), hit.activation))
            .collect();
        assert_eq!(forward_ranking, reverse_ranking);
    }
}

fn reference_step(current: &[u32], edges: &[(usize, usize, u32)]) -> Vec<u32> {
    let scale = u128::from(SCALE);
    let denominator = scale * scale;
    let saturation = denominator * scale;
    let mut numerators: Vec<u128> = current
        .iter()
        .map(|&value| u128::from(value) * u128::from(RETENTION_PPM) * scale)
        .collect();

    for &(from, to, edge_weight) in edges {
        numerators[to] +=
            u128::from(current[from]) * u128::from(edge_weight) * u128::from(PROPAGATION_GAIN_PPM);
        numerators[to] = numerators[to].min(saturation);
    }

    numerators
        .into_iter()
        .map(|value| u32::try_from(value / denominator).expect("reference activation is bounded"))
        .collect()
}

struct DeterministicGenerator(u64);

impl DeterministicGenerator {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
}

#[test]
fn ten_thousand_graphs_match_independent_dense_reference() {
    let mut generator = DeterministicGenerator(0x4e41_4f4d_4500_0001);
    let candidates = [100_000, 250_000, 400_000, 500_000, 750_000, SCALE];

    for case_index in 0..10_000_u64 {
        let atom_count = usize::try_from(1 + generator.next() % 8).expect("small count");
        let mut memory = new_memory(u128::from(case_index) + 1);
        let ids: Vec<_> = (0..atom_count)
            .map(|index| insert(&mut memory, case_index * 16 + index as u64))
            .collect();
        let mut before = Vec::with_capacity(atom_count);

        for &id in &ids {
            let value = u32::try_from(generator.next() % u64::from(SCALE + 1))
                .expect("bounded generated activation");
            before.push(value);
            memory.stimulate(id, activation(value)).expect("known atom");
        }

        for from in 0..atom_count {
            let mut remaining = SCALE;
            for to in 0..atom_count {
                if from == to || generator.next().is_multiple_of(3) || remaining == 0 {
                    continue;
                }
                let candidate = candidates
                    [usize::try_from(generator.next() % candidates.len() as u64).expect("index")];
                let edge_weight = candidate.min(remaining);
                memory
                    .set_relevance(ids[from], ids[to], weight(edge_weight))
                    .expect("generated budget is valid");
                remaining -= edge_weight;
            }
        }

        let edges: Vec<_> = memory
            .relevance_edges()
            .map(|edge| {
                (
                    usize::try_from(edge.from().sequence()).expect("small source id"),
                    usize::try_from(edge.to().sequence()).expect("small target id"),
                    edge.weight().as_ppm(),
                )
            })
            .collect();
        let mut previous = before;
        for step_index in 0..4 {
            let expected = reference_step(&previous, &edges);
            memory.step();

            let actual: Vec<_> = ids
                .iter()
                .map(|&id| memory.activation(id).expect("known atom").as_ppm())
                .collect();
            assert_eq!(
                actual, expected,
                "differential case {case_index}, step {step_index}"
            );

            let mut expected_ranking: Vec<_> = ids
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .filter(|(_, value)| *value != 0)
                .collect();
            expected_ranking.sort_unstable_by(|left, right| {
                right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
            });
            let actual_ranking: Vec<_> = memory
                .top_k(atom_count)
                .into_iter()
                .map(|hit| (hit.atom_id, hit.activation.as_ppm()))
                .collect();
            assert_eq!(
                actual_ranking, expected_ranking,
                "ranking case {case_index}, step {step_index}"
            );

            let before_total: u128 = previous.iter().copied().map(u128::from).sum();
            let after_total: u128 = actual.iter().copied().map(u128::from).sum();
            assert!(
                after_total * u128::from(SCALE)
                    <= before_total * u128::from(RETENTION_PPM + PROPAGATION_GAIN_PPM),
                "mass bound case {case_index}, step {step_index}"
            );
            previous = expected;
        }
    }
}
