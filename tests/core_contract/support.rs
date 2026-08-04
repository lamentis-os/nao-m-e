use nao_m_e::{
    Activation, AtomId, EpisodeDraft, FeedbackTrace, Memory, MemoryId, PredicateId, SourceId,
    Statement, TermId, TimestampMs,
};

pub(super) fn memory_id(value: u128) -> MemoryId {
    MemoryId::new(value).expect("test memory identifier is non-zero")
}

pub(super) fn new_memory(id: u128) -> Memory {
    Memory::new(memory_id(id))
}

pub(super) fn statement(predicate: u64, arguments: &[u64]) -> Statement {
    Statement::new(
        PredicateId::new(predicate),
        arguments.iter().copied().map(TermId::new).collect(),
    )
    .expect("test statement is valid")
}

pub(super) fn draft(seed: u64) -> EpisodeDraft {
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

pub(super) fn insert(memory: &mut Memory, seed: u64) -> AtomId {
    memory
        .insert_episode(draft(seed))
        .expect("identifier space is available")
}

pub(super) fn observation_draft(seed: u64, predicate: u64, arguments: &[u64]) -> EpisodeDraft {
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

pub(super) fn insert_observation(
    memory: &mut Memory,
    seed: u64,
    predicate: u64,
    arguments: &[u64],
) -> AtomId {
    memory
        .insert_episode(observation_draft(seed, predicate, arguments))
        .expect("identifier space is available")
}

pub(super) fn activation(value: u32) -> Activation {
    Activation::from_ppm(value).expect("test activation is bounded")
}

pub(super) fn trace(history_bits: u16, sample_count: u8) -> FeedbackTrace {
    FeedbackTrace::from_parts(history_bits, sample_count).expect("test trace is canonical")
}

pub(super) fn feedback_snapshot(memory: &Memory) -> Vec<(AtomId, AtomId, FeedbackTrace)> {
    memory
        .feedback_edges()
        .map(|edge| (edge.from(), edge.to(), edge.trace()))
        .collect()
}
