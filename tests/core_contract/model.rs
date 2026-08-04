use nao_m_e::{
    Activation, AtomId, Attribute, EpisodeDraft, FEEDBACK_HISTORY_CAPACITY, FEEDBACK_PRIOR_MASS,
    FeedbackTrace, LEARNED_GAIN_PPM, MAX_FEEDBACK_TARGETS, MemoryId, MemoryIdError, ModelError,
    SCALE, STRUCTURAL_GAIN_PPM, SymbolId, TimestampMs, ValueError,
};

use super::support::{memory_id, trace};

#[test]
fn model_constructors_and_feedback_parameters_enforce_their_boundaries() {
    assert_eq!(FEEDBACK_HISTORY_CAPACITY, 16);
    assert_eq!(FEEDBACK_PRIOR_MASS, 7);
    assert_eq!(LEARNED_GAIN_PPM, 400_000);
    assert_eq!(MAX_FEEDBACK_TARGETS, 10_000);
    assert_eq!(STRUCTURAL_GAIN_PPM, 400_000);
    let symbol = SymbolId::from(u64::MAX);
    assert_eq!(symbol.get(), u64::MAX);

    let attribute = Attribute::new(
        SymbolId::new(7),
        vec![SymbolId::new(3), SymbolId::new(1), SymbolId::new(3)],
    )
    .expect("attribute has values");
    assert_eq!(attribute.key(), SymbolId::new(7));
    assert_eq!(attribute.values(), &[SymbolId::new(1), SymbolId::new(3)]);
    assert_eq!(
        Attribute::new(SymbolId::new(1), Vec::new()),
        Err(ModelError::EmptyAttributeValues)
    );
    assert_eq!(
        EpisodeDraft::new(TimestampMs::new(0), Vec::new()),
        Err(ModelError::EmptyEpisodeAttributes)
    );
    assert_eq!(
        ModelError::EmptyAttributeValues.to_string(),
        "an attribute requires at least one value"
    );
    assert_eq!(
        ModelError::EmptyEpisodeAttributes.to_string(),
        "an episode requires at least one attribute"
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
