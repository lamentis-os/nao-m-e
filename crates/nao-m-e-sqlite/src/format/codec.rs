use nao_m_e::{Attribute, EpisodeAtom, EpisodeDraft, MemoryId, SymbolId, TimestampMs};

const MEMORY_ID_BYTES: usize = 16;
const U64_BYTES: usize = 8;

const FIXED_EPISODE_PREFIX_BYTES: usize = U64_BYTES;
const MIN_ATTRIBUTE_BYTES: usize = U64_BYTES + 1 + U64_BYTES;
pub(crate) const MIN_EPISODE_PAYLOAD_BYTES: usize =
    FIXED_EPISODE_PREFIX_BYTES + 1 + MIN_ATTRIBUTE_BYTES;

const TRUNCATED_EPISODE: &str = "episode blob is truncated";
const TRUNCATED_COUNT: &str = "count ULEB128 is truncated";
const OVERFLOWING_COUNT: &str = "count ULEB128 overflows u64";
const NON_CANONICAL_COUNT: &str = "count ULEB128 is not canonical";
const EMPTY_ATTRIBUTES: &str = "episode has no attributes";
const ATTRIBUTE_COUNT_EXCEEDS_REST: &str = "episode attribute count exceeds remaining bytes";
const EMPTY_VALUES: &str = "attribute has no values";
const VALUE_COUNT_EXCEEDS_REST: &str = "attribute value count exceeds remaining bytes";
const NON_CANONICAL_ATTRIBUTE_KEYS: &str = "attribute keys are not strictly sorted and unique";
const NON_CANONICAL_ATTRIBUTE_VALUES: &str = "attribute values are not strictly sorted and unique";
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

pub(crate) fn encode_episode(episode: &EpisodeAtom, encoded: &mut Vec<u8>) {
    encoded.clear();
    encoded.extend_from_slice(&episode.timestamp().get().to_be_bytes());
    encode_count(episode.attributes().len(), encoded);
    for attribute in episode.attributes() {
        encode_attribute(attribute, encoded);
    }
}

pub(crate) fn decode_episode(bytes: &[u8]) -> Result<EpisodeDraft, EpisodeDecodeError> {
    let mut decoder = Decoder::new(bytes);
    let timestamp = TimestampMs::new(decoder.read_i64()?);
    let attribute_count = decoder.read_uleb128()?;
    if attribute_count == 0 {
        return Err(EpisodeDecodeError::new(EMPTY_ATTRIBUTES));
    }
    if !count_fits_remaining(
        attribute_count,
        MIN_ATTRIBUTE_BYTES,
        decoder.remaining_len(),
    ) {
        return Err(EpisodeDecodeError::new(ATTRIBUTE_COUNT_EXCEEDS_REST));
    }
    let attribute_count = usize::try_from(attribute_count)
        .map_err(|_| EpisodeDecodeError::new(ATTRIBUTE_COUNT_EXCEEDS_REST))?;

    let mut attributes = Vec::with_capacity(attribute_count);
    for _ in 0..attribute_count {
        attributes.push(decoder.read_attribute()?);
    }
    if attributes
        .windows(2)
        .any(|pair| pair[0].key() >= pair[1].key())
    {
        return Err(EpisodeDecodeError::new(NON_CANONICAL_ATTRIBUTE_KEYS));
    }
    if decoder.remaining_len() != 0 {
        return Err(EpisodeDecodeError::new(TRAILING_BYTES));
    }

    EpisodeDraft::new(timestamp, attributes).map_err(|_| EpisodeDecodeError::new(EMPTY_ATTRIBUTES))
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

fn encode_attribute(attribute: &Attribute, encoded: &mut Vec<u8>) {
    encoded.extend_from_slice(&attribute.key().get().to_be_bytes());
    encode_count(attribute.values().len(), encoded);
    for value in attribute.values() {
        encoded.extend_from_slice(&value.get().to_be_bytes());
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

    fn read_attribute(&mut self) -> Result<Attribute, EpisodeDecodeError> {
        let key = SymbolId::new(self.read_u64()?);
        let value_count = self.read_uleb128()?;
        if value_count == 0 {
            return Err(EpisodeDecodeError::new(EMPTY_VALUES));
        }
        if !count_fits_remaining(value_count, U64_BYTES, self.remaining_len()) {
            return Err(EpisodeDecodeError::new(VALUE_COUNT_EXCEEDS_REST));
        }
        let value_count = usize::try_from(value_count)
            .map_err(|_| EpisodeDecodeError::new(VALUE_COUNT_EXCEEDS_REST))?;
        let mut values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            values.push(SymbolId::new(self.read_u64()?));
        }
        if values.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(EpisodeDecodeError::new(NON_CANONICAL_ATTRIBUTE_VALUES));
        }
        Attribute::new(key, values).map_err(|_| EpisodeDecodeError::new(EMPTY_VALUES))
    }
}

#[cfg(test)]
mod tests;
