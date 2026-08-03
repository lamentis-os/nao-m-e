use nao_m_e::{
    EpisodeAtom, EpisodeDraft, MemoryId, PredicateId, SourceId, Statement, TermId, TimestampMs,
};

pub(crate) const MEMORY_ID_BYTES: usize = 16;
pub(crate) const U64_BYTES: usize = 8;

const ACTION_PRESENT: u8 = 1 << 0;
const OUTCOME_PRESENT: u8 = 1 << 1;
const KNOWN_FLAGS: u8 = ACTION_PRESENT | OUTCOME_PRESENT;
#[cfg(test)]
const FIXED_EPISODE_PREFIX_BYTES: usize = 1 + 3 * U64_BYTES;
const MIN_STATEMENT_BYTES: usize = U64_BYTES + 1 + U64_BYTES;

const TRUNCATED_EPISODE: &str = "episode blob is truncated";
const RESERVED_FLAGS: &str = "episode flags contain reserved bits";
const TRUNCATED_COUNT: &str = "count ULEB128 is truncated";
const OVERFLOWING_COUNT: &str = "count ULEB128 overflows u64";
const NON_CANONICAL_COUNT: &str = "count ULEB128 is not canonical";
const STATEMENT_COUNT_EXCEEDS_REST: &str = "episode statement count exceeds remaining bytes";
const EMPTY_ARGUMENTS: &str = "statement has no arguments";
const ARGUMENT_COUNT_EXCEEDS_REST: &str = "statement argument count exceeds remaining bytes";
const NON_CANONICAL_CONTEXT: &str = "context is not strictly sorted and duplicate-free";
const TRAILING_BYTES: &str = "episode blob has trailing bytes";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EpisodeDecodeError {
    detail: &'static str,
}

impl EpisodeDecodeError {
    const fn new(detail: &'static str) -> Self {
        Self { detail }
    }

    pub(crate) const fn detail(&self) -> &'static str {
        self.detail
    }
}

pub(crate) fn encode_memory_id(memory_id: MemoryId) -> [u8; MEMORY_ID_BYTES] {
    memory_id.to_be_bytes()
}

pub(crate) fn decode_memory_id(bytes: &[u8]) -> Option<MemoryId> {
    let bytes = <[u8; MEMORY_ID_BYTES]>::try_from(bytes).ok()?;
    MemoryId::from_be_bytes(bytes).ok()
}

pub(crate) const fn encode_u64(value: u64) -> [u8; U64_BYTES] {
    value.to_be_bytes()
}

pub(crate) fn decode_u64(bytes: &[u8]) -> Option<u64> {
    <[u8; U64_BYTES]>::try_from(bytes)
        .ok()
        .map(u64::from_be_bytes)
}

pub(crate) fn encode_episode(episode: &EpisodeAtom) -> Vec<u8> {
    let mut encoded = Vec::new();
    let mut flags = 0;
    if episode.action().is_some() {
        flags |= ACTION_PRESENT;
    }
    if episode.outcome().is_some() {
        flags |= OUTCOME_PRESENT;
    }
    encoded.push(flags);
    encoded.extend_from_slice(&episode.occurred_at().get().to_be_bytes());
    encoded.extend_from_slice(&episode.recorded_at().get().to_be_bytes());
    encoded.extend_from_slice(&episode.source().get().to_be_bytes());
    encode_count(episode.context().len(), &mut encoded);
    for statement in episode.context() {
        encode_statement(statement, &mut encoded);
    }
    encode_statement(episode.observation(), &mut encoded);
    if let Some(action) = episode.action() {
        encode_statement(action, &mut encoded);
    }
    if let Some(outcome) = episode.outcome() {
        encode_statement(outcome, &mut encoded);
    }
    encoded
}

pub(crate) fn decode_episode(bytes: &[u8]) -> Result<EpisodeDraft, EpisodeDecodeError> {
    let mut decoder = Decoder::new(bytes);
    let flags = decoder.read_byte(TRUNCATED_EPISODE)?;
    if flags & !KNOWN_FLAGS != 0 {
        return Err(EpisodeDecodeError::new(RESERVED_FLAGS));
    }

    let occurred_at = TimestampMs::new(decoder.read_i64()?);
    let recorded_at = TimestampMs::new(decoder.read_i64()?);
    let source = SourceId::new(decoder.read_u64()?);
    let context_count = decoder.read_uleb128()?;
    let trailing_statement_count =
        1_u64 + u64::from(flags & ACTION_PRESENT != 0) + u64::from(flags & OUTCOME_PRESENT != 0);
    let statement_count = context_count
        .checked_add(trailing_statement_count)
        .ok_or_else(|| EpisodeDecodeError::new(STATEMENT_COUNT_EXCEEDS_REST))?;
    if !count_fits_remaining(
        statement_count,
        MIN_STATEMENT_BYTES,
        decoder.remaining_len(),
    ) {
        return Err(EpisodeDecodeError::new(STATEMENT_COUNT_EXCEEDS_REST));
    }
    let context_count = usize::try_from(context_count)
        .map_err(|_| EpisodeDecodeError::new(STATEMENT_COUNT_EXCEEDS_REST))?;

    let mut context = Vec::with_capacity(context_count);
    for _ in 0..context_count {
        context.push(decoder.read_statement()?);
    }
    if context.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(EpisodeDecodeError::new(NON_CANONICAL_CONTEXT));
    }

    let observation = decoder.read_statement()?;
    let action = if flags & ACTION_PRESENT != 0 {
        Some(decoder.read_statement()?)
    } else {
        None
    };
    let outcome = if flags & OUTCOME_PRESENT != 0 {
        Some(decoder.read_statement()?)
    } else {
        None
    };
    if decoder.remaining_len() != 0 {
        return Err(EpisodeDecodeError::new(TRAILING_BYTES));
    }

    Ok(EpisodeDraft {
        occurred_at,
        recorded_at,
        context,
        observation,
        action,
        outcome,
        source,
    })
}

fn encode_count(count: usize, encoded: &mut Vec<u8>) {
    let count = u64::try_from(count).expect("collection lengths fit u64 on supported platforms");
    encode_uleb128(count, encoded);
}

fn encode_uleb128(mut value: u64, encoded: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        encoded.push(byte);
        if value == 0 {
            return;
        }
    }
}

fn encode_statement(statement: &Statement, encoded: &mut Vec<u8>) {
    encoded.extend_from_slice(&statement.predicate().get().to_be_bytes());
    encode_count(statement.arguments().len(), encoded);
    for term in statement.arguments() {
        encoded.extend_from_slice(&term.get().to_be_bytes());
    }
}

fn count_fits_remaining(count: u64, item_bytes: usize, remaining_bytes: usize) -> bool {
    usize::try_from(count).is_ok_and(|count| count <= remaining_bytes / item_bytes)
}

struct Decoder<'a> {
    remaining: &'a [u8],
}

impl<'a> Decoder<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    const fn remaining_len(&self) -> usize {
        self.remaining.len()
    }

    fn take(&mut self, count: usize, detail: &'static str) -> Result<&'a [u8], EpisodeDecodeError> {
        if self.remaining.len() < count {
            return Err(EpisodeDecodeError::new(detail));
        }
        let (value, remaining) = self.remaining.split_at(count);
        self.remaining = remaining;
        Ok(value)
    }

    fn read_byte(&mut self, detail: &'static str) -> Result<u8, EpisodeDecodeError> {
        Ok(self.take(1, detail)?[0])
    }

    fn read_i64(&mut self) -> Result<i64, EpisodeDecodeError> {
        let bytes = self
            .take(U64_BYTES, TRUNCATED_EPISODE)?
            .try_into()
            .expect("the decoder took exactly eight bytes");
        Ok(i64::from_be_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, EpisodeDecodeError> {
        let bytes = self
            .take(U64_BYTES, TRUNCATED_EPISODE)?
            .try_into()
            .expect("the decoder took exactly eight bytes");
        Ok(u64::from_be_bytes(bytes))
    }

    fn read_uleb128(&mut self) -> Result<u64, EpisodeDecodeError> {
        let mut value = 0_u64;
        for index in 0..10 {
            let byte = self.read_byte(TRUNCATED_COUNT)?;
            let payload = byte & 0x7f;
            if index == 9 && payload > 1 {
                return Err(EpisodeDecodeError::new(OVERFLOWING_COUNT));
            }
            value |= u64::from(payload) << (index * 7);
            if byte & 0x80 == 0 {
                if index != 0 && payload == 0 {
                    return Err(EpisodeDecodeError::new(NON_CANONICAL_COUNT));
                }
                return Ok(value);
            }
        }
        Err(EpisodeDecodeError::new(OVERFLOWING_COUNT))
    }

    fn read_statement(&mut self) -> Result<Statement, EpisodeDecodeError> {
        let predicate = PredicateId::new(self.read_u64()?);
        let argument_count = self.read_uleb128()?;
        if argument_count == 0 {
            return Err(EpisodeDecodeError::new(EMPTY_ARGUMENTS));
        }
        if !count_fits_remaining(argument_count, U64_BYTES, self.remaining_len()) {
            return Err(EpisodeDecodeError::new(ARGUMENT_COUNT_EXCEEDS_REST));
        }
        let argument_count = usize::try_from(argument_count)
            .map_err(|_| EpisodeDecodeError::new(ARGUMENT_COUNT_EXCEEDS_REST))?;
        let mut arguments = Vec::with_capacity(argument_count);
        for _ in 0..argument_count {
            arguments.push(TermId::new(self.read_u64()?));
        }
        Statement::new(predicate, arguments).map_err(|_| EpisodeDecodeError::new(EMPTY_ARGUMENTS))
    }
}

#[cfg(test)]
mod tests {
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
}
