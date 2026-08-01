use nao_m_e::{
    Activation, AtomId, EpisodeDraft, GraphError, InfluenceWeight, MemoryV0, ModelError,
    PROPAGATION_GAIN_PPM, PredicateId, RETENTION_PPM, SCALE, SourceId, Statement, TermId,
    TimestampMs, ValueError,
};

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

#[test]
fn model_constructors_enforce_their_boundaries() {
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
fn insertion_canonicalizes_context_but_keeps_occurrences_distinct() {
    let low = statement(1, &[1]);
    let high = statement(2, &[1]);
    let mut episode = draft(1);
    episode.context = vec![high.clone(), low.clone(), high];

    let mut memory = MemoryV0::new();
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

    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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

    let mut other = MemoryV0::new();
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
fn ids_from_another_memory_cannot_alias_local_atoms() {
    let mut first_memory = MemoryV0::new();
    let local = insert(&mut first_memory, 1);
    let mut second_memory = MemoryV0::new();
    let foreign = insert(&mut second_memory, 1);

    assert_eq!(local.get(), foreign.get());
    assert_ne!(local.memory_namespace(), foreign.memory_namespace());
    assert_eq!(
        local.to_string(),
        format!("{}:{}", local.memory_namespace(), local.get())
    );
    assert_ne!(local.to_string(), foreign.to_string());
    assert_eq!(first_memory.episode(foreign), None);
    assert_eq!(first_memory.activation(foreign), None);
    assert_eq!(first_memory.relevance(local, foreign), None);
    assert_eq!(first_memory.relevance(foreign, local), None);
    assert_eq!(
        first_memory.stimulate(foreign, Activation::ONE),
        Err(GraphError::UnknownAtom(foreign))
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
        first_memory.remove_relevance(local, foreign),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert_eq!(
        first_memory.remove_relevance(foreign, local),
        Err(GraphError::UnknownAtom(foreign))
    );
    assert!(
        GraphError::UnknownAtom(foreign)
            .to_string()
            .contains(&foreign.to_string())
    );
    assert_eq!(first_memory.activation(local), Some(Activation::ZERO));
}

#[test]
fn empty_memory_operations_are_noops() {
    let mut memory = MemoryV0::new();

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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut memory = MemoryV0::new();
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
    let mut forward = MemoryV0::new();
    let mut reverse = MemoryV0::new();
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
            .map(|hit| (hit.atom_id.get(), hit.activation))
            .collect();
        let reverse_ranking: Vec<_> = reverse
            .top_k(4)
            .into_iter()
            .map(|hit| (hit.atom_id.get(), hit.activation))
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
        let mut memory = MemoryV0::new();
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
                    usize::try_from(edge.from().get()).expect("small source id"),
                    usize::try_from(edge.to().get()).expect("small target id"),
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
