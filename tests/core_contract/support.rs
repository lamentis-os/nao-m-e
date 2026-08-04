use nao_m_e::{
    Activation, AtomId, Attribute, EpisodeDraft, FeedbackTrace, Memory, MemoryId, SymbolId,
    TimestampMs,
};

pub(super) fn memory_id(value: u128) -> MemoryId {
    MemoryId::new(value).expect("test memory identifier is non-zero")
}

pub(super) fn new_memory(id: u128) -> Memory {
    Memory::new(memory_id(id))
}

pub(super) fn attribute(key: u64, values: &[u64]) -> Attribute {
    Attribute::new(
        SymbolId::new(key),
        values.iter().copied().map(SymbolId::new).collect(),
    )
    .expect("test attribute is valid")
}

pub(super) fn insert(memory: &mut Memory, seed: u64) -> AtomId {
    memory
        .insert_episode(attribute_draft(seed, 10 + seed, &[100 + seed]))
        .expect("identifier space is available")
}

pub(super) fn attribute_draft(seed: u64, key: u64, values: &[u64]) -> EpisodeDraft {
    EpisodeDraft::new(
        TimestampMs::new(i64::try_from(seed).expect("small test seed")),
        vec![attribute(key, values)],
    )
    .expect("test episode is valid")
}

pub(super) fn insert_attribute(memory: &mut Memory, seed: u64, key: u64, values: &[u64]) -> AtomId {
    memory
        .insert_episode(attribute_draft(seed, key, values))
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
