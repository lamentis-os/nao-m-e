use nao_m_e::{
    EpisodeAtom, EpisodeDraft, MemoryId, PredicateId, SourceId, Statement, TermId, TimestampMs,
};

const MEMORY_ID_BYTES: usize = 16;
const U64_BYTES: usize = 8;

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
mod tests;
