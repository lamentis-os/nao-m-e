use nao_m_e::MemoryId;

pub(crate) const MEMORY_ID_BYTES: usize = 16;
pub(crate) const U64_BYTES: usize = 8;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
