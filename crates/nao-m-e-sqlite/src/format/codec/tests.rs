use nao_m_e::{AtomId, Memory};

use super::*;

fn attribute(key: u64, values: &[u64]) -> Attribute {
    Attribute::new(
        SymbolId::new(key),
        values.iter().copied().map(SymbolId::new).collect(),
    )
    .expect("test attributes have values")
}

fn draft(timestamp: i64, attributes: Vec<Attribute>) -> EpisodeDraft {
    EpisodeDraft::new(TimestampMs::new(timestamp), attributes)
        .expect("test episodes have attributes")
}

fn episode(draft: EpisodeDraft) -> EpisodeAtom {
    let memory_id = MemoryId::new(1).expect("test memory ID is non-zero");
    let mut memory = Memory::new(memory_id);
    let id = memory.insert_episode(draft).expect("test episode inserts");
    assert_eq!(id, AtomId::from_parts(memory_id, 0));
    memory.episode(id).expect("inserted episode exists").clone()
}

fn draft_from_episode(episode: &EpisodeAtom) -> EpisodeDraft {
    EpisodeDraft::new(episode.timestamp(), episode.attributes().to_vec())
        .expect("stored episodes are canonical")
}

fn fixed_prefix() -> Vec<u8> {
    0_i64.to_be_bytes().to_vec()
}

fn encoded_episode(episode: &EpisodeAtom) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(MIN_EPISODE_PAYLOAD_BYTES);
    encode_episode(episode, &mut encoded);
    encoded
}

fn assert_decode_error(bytes: &[u8], expected: &'static str) {
    let error = decode_episode(bytes).expect_err("malformed episode must be rejected");
    assert_eq!(error.detail(), expected);
}

#[test]
fn memory_id_roundtrips_in_canonical_form() {
    for value in [1, u128::from(u64::MAX) + 1, u128::MAX] {
        let memory_id = MemoryId::new(value).expect("test identifier is non-zero");
        let encoded = encode_memory_id(memory_id);
        assert_eq!(decode_memory_id(&encoded), Some(memory_id));
    }
}

#[test]
fn memory_id_decoder_rejects_wrong_lengths_and_zero() {
    assert_eq!(decode_memory_id(&[]), None);
    assert_eq!(decode_memory_id(&[0; MEMORY_ID_BYTES - 1]), None);
    assert_eq!(decode_memory_id(&[0; MEMORY_ID_BYTES]), None);
    assert_eq!(decode_memory_id(&[0; MEMORY_ID_BYTES + 1]), None);
}

#[test]
fn u64_roundtrips_across_the_full_storage_range() {
    for value in [0, i64::MAX as u64, i64::MAX as u64 + 1, u64::MAX] {
        let encoded = encode_u64(value);
        assert_eq!(decode_u64(&encoded), Some(value));
    }
}

#[test]
fn u64_decoder_rejects_wrong_lengths_and_big_endian_preserves_order() {
    assert_eq!(decode_u64(&[]), None);
    assert_eq!(decode_u64(&[0; U64_BYTES - 1]), None);
    assert_eq!(decode_u64(&[0; U64_BYTES + 1]), None);
    let values = [0, 1, i64::MAX as u64, i64::MAX as u64 + 1, u64::MAX];
    for pair in values.map(encode_u64).windows(2) {
        assert!(pair[0] < pair[1]);
    }
}

#[test]
fn episode_encoding_matches_golden_bytes() {
    let atom = episode(draft(-1, vec![attribute(1, &[2])]));
    let expected = vec![
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // timestamp
        0x01, // attribute count
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, // key
        0x01, // value count
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, // value
    ];
    assert_eq!(expected.len(), MIN_EPISODE_PAYLOAD_BYTES);

    let mut encoded = vec![0xaa; MIN_EPISODE_PAYLOAD_BYTES + 10];
    encode_episode(&atom, &mut encoded);
    assert_eq!(encoded, expected);
    assert_eq!(decode_episode(&expected), Ok(draft_from_episode(&atom)));
}

#[test]
fn counts_cross_the_canonical_uleb128_boundary() {
    for (count, expected) in [(127_usize, &[0x7f][..]), (128, &[0x80, 0x01][..])] {
        let attributes = (0..count)
            .map(|index| attribute(index as u64, &[index as u64]))
            .collect();
        let atom = episode(draft(0, attributes));
        let encoded = encoded_episode(&atom);
        assert_eq!(
            &encoded[FIXED_EPISODE_PREFIX_BYTES..FIXED_EPISODE_PREFIX_BYTES + expected.len()],
            expected
        );
        assert_eq!(decode_episode(&encoded), Ok(draft_from_episode(&atom)));

        let values = (0..count).map(|index| index as u64).collect::<Vec<_>>();
        let atom = episode(draft(0, vec![attribute(0, &values)]));
        let encoded = encoded_episode(&atom);
        let value_count_offset = FIXED_EPISODE_PREFIX_BYTES + 1 + U64_BYTES;
        assert_eq!(
            &encoded[value_count_offset..value_count_offset + expected.len()],
            expected
        );
        assert_eq!(decode_episode(&encoded), Ok(draft_from_episode(&atom)));
    }
}

#[test]
fn signed_timestamp_and_unsigned_symbols_roundtrip_at_extremes() {
    for timestamp in [i64::MIN, i64::MAX] {
        let atom = episode(draft(
            timestamp,
            vec![attribute(0, &[u64::MAX]), attribute(u64::MAX, &[0])],
        ));
        let encoded = encoded_episode(&atom);
        assert_eq!(decode_episode(&encoded), Ok(draft_from_episode(&atom)));
    }
}

#[test]
fn malformed_counts_empty_collections_and_trailing_bytes_fail_closed() {
    let atom = episode(draft(1, vec![attribute(10, &[11]), attribute(20, &[21])]));
    let valid = encoded_episode(&atom);
    for end in 0..valid.len() {
        assert!(
            decode_episode(&valid[..end]).is_err(),
            "accepted truncated prefix of length {end}"
        );
    }

    let mut trailing = valid;
    trailing.push(0);
    assert_decode_error(&trailing, TRAILING_BYTES);

    let mut truncated_count = fixed_prefix();
    truncated_count.push(0x80);
    assert_decode_error(&truncated_count, TRUNCATED_COUNT);

    let mut overflowing_count = fixed_prefix();
    overflowing_count.extend_from_slice(&[0xff; 9]);
    overflowing_count.push(0x02);
    assert_decode_error(&overflowing_count, OVERFLOWING_COUNT);

    let mut non_canonical_count = fixed_prefix();
    non_canonical_count.extend_from_slice(&[0x80, 0x00]);
    assert_decode_error(&non_canonical_count, NON_CANONICAL_COUNT);

    let mut empty_attributes = fixed_prefix();
    empty_attributes.push(0);
    assert_decode_error(&empty_attributes, EMPTY_ATTRIBUTES);

    let mut impossible_attribute_count = fixed_prefix();
    encode_uleb128(u64::MAX, &mut impossible_attribute_count);
    assert_decode_error(&impossible_attribute_count, ATTRIBUTE_COUNT_EXCEEDS_REST);

    let mut empty_values = fixed_prefix();
    empty_values.push(1);
    empty_values.extend_from_slice(&0_u64.to_be_bytes());
    empty_values.push(0);
    empty_values.extend_from_slice(&0_u64.to_be_bytes());
    assert_decode_error(&empty_values, EMPTY_VALUES);

    let mut impossible_value_count = fixed_prefix();
    impossible_value_count.push(1);
    impossible_value_count.extend_from_slice(&0_u64.to_be_bytes());
    impossible_value_count.push(2);
    impossible_value_count.extend_from_slice(&0_u64.to_be_bytes());
    assert_decode_error(&impossible_value_count, VALUE_COUNT_EXCEEDS_REST);
}

#[test]
fn non_canonical_attribute_and_value_order_fail_closed() {
    let atom = episode(draft(0, vec![attribute(10, &[11]), attribute(20, &[21])]));
    let valid = encoded_episode(&atom);
    let attributes_start = FIXED_EPISODE_PREFIX_BYTES + 1;
    let attribute_bytes = MIN_ATTRIBUTE_BYTES;

    let mut unsorted_keys = valid.clone();
    let first = unsorted_keys[attributes_start..attributes_start + attribute_bytes].to_vec();
    let second = unsorted_keys
        [attributes_start + attribute_bytes..attributes_start + 2 * attribute_bytes]
        .to_vec();
    unsorted_keys[attributes_start..attributes_start + attribute_bytes].copy_from_slice(&second);
    unsorted_keys[attributes_start + attribute_bytes..attributes_start + 2 * attribute_bytes]
        .copy_from_slice(&first);
    assert_decode_error(&unsorted_keys, NON_CANONICAL_ATTRIBUTE_KEYS);

    let mut duplicate_keys = valid;
    let first = duplicate_keys[attributes_start..attributes_start + attribute_bytes].to_vec();
    duplicate_keys[attributes_start + attribute_bytes..attributes_start + 2 * attribute_bytes]
        .copy_from_slice(&first);
    assert_decode_error(&duplicate_keys, NON_CANONICAL_ATTRIBUTE_KEYS);

    for values in [[2_u64, 1], [1, 1]] {
        let mut bytes = fixed_prefix();
        bytes.push(1);
        bytes.extend_from_slice(&0_u64.to_be_bytes());
        bytes.push(2);
        for value in values {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        assert_decode_error(&bytes, NON_CANONICAL_ATTRIBUTE_VALUES);
    }
}

#[test]
fn roundtrip_uses_the_core_canonical_attribute_union() {
    let atom = episode(draft(
        5,
        vec![
            attribute(3, &[4, 2]),
            attribute(1, &[9]),
            attribute(3, &[1, 2]),
        ],
    ));
    assert_eq!(
        atom.attributes(),
        &[attribute(1, &[9]), attribute(3, &[1, 2, 4])]
    );
    let encoded = encoded_episode(&atom);
    assert_eq!(decode_episode(&encoded), Ok(draft_from_episode(&atom)));
}
