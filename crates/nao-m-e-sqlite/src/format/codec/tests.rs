use nao_m_e::{AtomId, Memory};

use super::*;

fn statement(predicate: u64, arguments: &[u64]) -> Statement {
    Statement::new(
        PredicateId::new(predicate),
        arguments.iter().copied().map(TermId::new).collect(),
    )
    .expect("test statements have arguments")
}

fn episode(draft: EpisodeDraft) -> EpisodeAtom {
    let memory_id = MemoryId::new(1).expect("test memory ID is non-zero");
    let mut memory = Memory::new(memory_id);
    let id = memory.insert_episode(draft).expect("test episode inserts");
    assert_eq!(id, AtomId::from_parts(memory_id, 0));
    memory.episode(id).expect("inserted episode exists").clone()
}

fn draft_from_episode(episode: &EpisodeAtom) -> EpisodeDraft {
    EpisodeDraft {
        occurred_at: episode.occurred_at(),
        recorded_at: episode.recorded_at(),
        context: episode.context().to_vec(),
        observation: episode.observation().clone(),
        action: episode.action().cloned(),
        outcome: episode.outcome().cloned(),
        source: episode.source(),
    }
}

fn fixed_prefix(flags: u8) -> Vec<u8> {
    let mut bytes = vec![flags];
    bytes.extend_from_slice(&0_i64.to_be_bytes());
    bytes.extend_from_slice(&0_i64.to_be_bytes());
    bytes.extend_from_slice(&0_u64.to_be_bytes());
    bytes
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
fn u64_decoder_rejects_wrong_lengths() {
    assert_eq!(decode_u64(&[]), None);
    assert_eq!(decode_u64(&[0; U64_BYTES - 1]), None);
    assert_eq!(decode_u64(&[0; U64_BYTES + 1]), None);
}

#[test]
fn big_endian_encoding_preserves_numeric_order() {
    let values = [0, 1, i64::MAX as u64, i64::MAX as u64 + 1, u64::MAX];
    let encoded = values.map(encode_u64);

    for pair in encoded.windows(2) {
        assert!(pair[0] < pair[1]);
    }
}

#[test]
fn episode_encoding_matches_golden_bytes() {
    let atom = episode(EpisodeDraft {
        occurred_at: TimestampMs::new(0),
        recorded_at: TimestampMs::new(-1),
        context: Vec::new(),
        observation: statement(1, &[2]),
        action: None,
        outcome: None,
        source: SourceId::new(u64::MAX),
    });
    let expected = vec![
        0x00, // flags
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // occurred_at
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // recorded_at
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // source
        0x00, // context count
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, // predicate
        0x01, // argument count
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, // term
    ];

    assert_eq!(encode_episode(&atom), expected);
    assert_eq!(decode_episode(&expected), Ok(draft_from_episode(&atom)));
}

#[test]
fn all_optional_statement_combinations_roundtrip() {
    for action_present in [false, true] {
        for outcome_present in [false, true] {
            let atom = episode(EpisodeDraft {
                occurred_at: TimestampMs::new(10),
                recorded_at: TimestampMs::new(20),
                context: vec![statement(1, &[2]), statement(3, &[4, 5])],
                observation: statement(6, &[7]),
                action: action_present.then(|| statement(8, &[9])),
                outcome: outcome_present.then(|| statement(10, &[11])),
                source: SourceId::new(12),
            });
            let encoded = encode_episode(&atom);
            let expected_flags = (u8::from(action_present) * ACTION_PRESENT)
                | (u8::from(outcome_present) * OUTCOME_PRESENT);

            assert_eq!(encoded[0], expected_flags);
            assert_eq!(decode_episode(&encoded), Ok(draft_from_episode(&atom)));
        }
    }
}

#[test]
fn counts_cross_the_canonical_uleb128_boundary() {
    for (count, expected) in [(127_usize, &[0x7f][..]), (128, &[0x80, 0x01][..])] {
        let context: Vec<_> = (0..count)
            .map(|index| statement(index as u64, &[index as u64]))
            .collect();
        let context_atom = episode(EpisodeDraft {
            occurred_at: TimestampMs::new(0),
            recorded_at: TimestampMs::new(0),
            context,
            observation: statement(u64::MAX, &[0]),
            action: None,
            outcome: None,
            source: SourceId::new(0),
        });
        let encoded = encode_episode(&context_atom);
        assert_eq!(
            &encoded[FIXED_EPISODE_PREFIX_BYTES..FIXED_EPISODE_PREFIX_BYTES + expected.len()],
            expected
        );
        assert_eq!(
            decode_episode(&encoded),
            Ok(draft_from_episode(&context_atom))
        );

        let arguments: Vec<_> = (0..count).map(|index| index as u64).collect();
        let argument_atom = episode(EpisodeDraft {
            occurred_at: TimestampMs::new(0),
            recorded_at: TimestampMs::new(0),
            context: Vec::new(),
            observation: statement(0, &arguments),
            action: None,
            outcome: None,
            source: SourceId::new(0),
        });
        let encoded = encode_episode(&argument_atom);
        let argument_count_offset = FIXED_EPISODE_PREFIX_BYTES + 1 + U64_BYTES;
        assert_eq!(
            &encoded[argument_count_offset..argument_count_offset + expected.len()],
            expected
        );
        assert_eq!(
            decode_episode(&encoded),
            Ok(draft_from_episode(&argument_atom))
        );
    }
}

#[test]
fn signed_timestamps_and_unsigned_identifiers_roundtrip_at_extremes() {
    let atom = episode(EpisodeDraft {
        occurred_at: TimestampMs::new(i64::MIN),
        recorded_at: TimestampMs::new(i64::MAX),
        context: vec![statement(0, &[u64::MAX])],
        observation: statement(u64::MAX, &[0, u64::MAX]),
        action: Some(statement(u64::MAX - 1, &[u64::MAX])),
        outcome: Some(statement(u64::MAX, &[0])),
        source: SourceId::new(u64::MAX),
    });
    let encoded = encode_episode(&atom);

    assert_eq!(decode_episode(&encoded), Ok(draft_from_episode(&atom)));
}

#[test]
fn malformed_episode_encodings_fail_closed() {
    let atom = episode(EpisodeDraft {
        occurred_at: TimestampMs::new(1),
        recorded_at: TimestampMs::new(2),
        context: vec![statement(10, &[11]), statement(20, &[21])],
        observation: statement(30, &[31]),
        action: None,
        outcome: None,
        source: SourceId::new(3),
    });
    let valid = encode_episode(&atom);
    for end in 0..valid.len() {
        assert!(
            decode_episode(&valid[..end]).is_err(),
            "accepted truncated prefix of length {end}"
        );
    }

    let mut reserved_flags = valid.clone();
    reserved_flags[0] |= 1 << 2;
    assert_decode_error(&reserved_flags, RESERVED_FLAGS);

    let mut trailing = valid.clone();
    trailing.push(0);
    assert_decode_error(&trailing, TRAILING_BYTES);

    let mut truncated_count = fixed_prefix(0);
    truncated_count.push(0x80);
    assert_decode_error(&truncated_count, TRUNCATED_COUNT);

    let mut overflowing_count = fixed_prefix(0);
    overflowing_count.extend_from_slice(&[0xff; 9]);
    overflowing_count.push(0x02);
    assert_decode_error(&overflowing_count, OVERFLOWING_COUNT);

    let mut non_canonical_count = fixed_prefix(0);
    non_canonical_count.extend_from_slice(&[0x80, 0x00]);
    assert_decode_error(&non_canonical_count, NON_CANONICAL_COUNT);

    let mut impossible_context_count = fixed_prefix(0);
    encode_uleb128(u64::MAX, &mut impossible_context_count);
    assert_decode_error(&impossible_context_count, STATEMENT_COUNT_EXCEEDS_REST);

    let mut empty_arguments = fixed_prefix(0);
    empty_arguments.push(0);
    empty_arguments.extend_from_slice(&0_u64.to_be_bytes());
    empty_arguments.push(0);
    empty_arguments.extend_from_slice(&0_u64.to_be_bytes());
    assert_decode_error(&empty_arguments, EMPTY_ARGUMENTS);

    let mut impossible_argument_count = fixed_prefix(0);
    impossible_argument_count.push(0);
    impossible_argument_count.extend_from_slice(&0_u64.to_be_bytes());
    impossible_argument_count.push(2);
    impossible_argument_count.extend_from_slice(&0_u64.to_be_bytes());
    assert_decode_error(&impossible_argument_count, ARGUMENT_COUNT_EXCEEDS_REST);

    let context_start = FIXED_EPISODE_PREFIX_BYTES + 1;
    let statement_bytes = MIN_STATEMENT_BYTES;
    let mut unsorted = valid.clone();
    let first = unsorted[context_start..context_start + statement_bytes].to_vec();
    let second =
        unsorted[context_start + statement_bytes..context_start + 2 * statement_bytes].to_vec();
    unsorted[context_start..context_start + statement_bytes].copy_from_slice(&second);
    unsorted[context_start + statement_bytes..context_start + 2 * statement_bytes]
        .copy_from_slice(&first);
    assert_decode_error(&unsorted, NON_CANONICAL_CONTEXT);

    let mut duplicate = valid;
    let first = duplicate[context_start..context_start + statement_bytes].to_vec();
    duplicate[context_start + statement_bytes..context_start + 2 * statement_bytes]
        .copy_from_slice(&first);
    assert_decode_error(&duplicate, NON_CANONICAL_CONTEXT);
}

#[test]
fn roundtrip_uses_the_atoms_canonical_context() {
    let first = statement(1, &[2]);
    let second = statement(3, &[4]);
    let atom = episode(EpisodeDraft {
        occurred_at: TimestampMs::new(5),
        recorded_at: TimestampMs::new(6),
        context: vec![second.clone(), first.clone(), second],
        observation: statement(7, &[8]),
        action: None,
        outcome: None,
        source: SourceId::new(9),
    });
    assert_eq!(atom.context(), &[first, statement(3, &[4])]);

    let encoded = encode_episode(&atom);
    assert_eq!(decode_episode(&encoded), Ok(draft_from_episode(&atom)));
}
