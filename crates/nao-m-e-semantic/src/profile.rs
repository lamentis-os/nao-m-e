/// Number of signed components in every semantic embedding.
pub const EMBEDDING_DIMENSIONS: usize = 384;

/// Maximum number of cues accepted by one encoder call.
pub const MAX_EMBEDDING_BATCH_SIZE: usize = 32;

pub(crate) const MODEL_OWNER: &str = "intfloat";
pub(crate) const MODEL_NAME: &str = "multilingual-e5-small";
pub(crate) const MODEL_REVISION: &str = "0e60b8d9d2166d80387f86e3b48ec9ced55f4d15";
pub(crate) const MODEL_FILE: &str = "onnx/model.onnx";
pub(crate) const TOKENIZER_FILE: &str = "onnx/tokenizer.json";
pub(crate) const MODEL_SHA256: [u8; 32] = [
    0xca, 0x45, 0x6c, 0x06, 0xb3, 0xa9, 0x50, 0x5d, 0xdf, 0xd9, 0x13, 0x14, 0x08, 0x91, 0x6d, 0xd7,
    0x92, 0x90, 0x36, 0x83, 0x31, 0xe7, 0xd7, 0x6b, 0xb6, 0x21, 0xf1, 0xcb, 0xa6, 0xbc, 0x86, 0x65,
];
pub(crate) const TOKENIZER_SHA256: [u8; 32] = [
    0x0b, 0x44, 0xa9, 0xd7, 0xb5, 0x1c, 0x3c, 0x62, 0x62, 0x66, 0x40, 0xcd, 0xa0, 0xe2, 0xc2, 0xf7,
    0x0f, 0xda, 0xcd, 0xc2, 0x5b, 0xbb, 0xd6, 0x80, 0x38, 0x36, 0x9d, 0x14, 0xeb, 0xdf, 0x4c, 0x39,
];

const PROFILE_MANIFEST: &str = concat!(
    "nao-m-e-semantic-profile-v1\n",
    "model.repository=intfloat/multilingual-e5-small\n",
    "model.revision=0e60b8d9d2166d80387f86e3b48ec9ced55f4d15\n",
    "model.artifact=onnx/model.onnx\n",
    "model.sha256=ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665\n",
    "tokenizer.artifact=onnx/tokenizer.json\n",
    "tokenizer.sha256=0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39\n",
    "tokenizer.runtime=tokenizers-0.23.1\n",
    "tokenizer.max_tokens=512\n",
    "tokenizer.add-special-tokens=true\n",
    "tokenizer.truncation=longest-first-right\n",
    "tokenizer.truncation-stride=0\n",
    "tokenizer.padding=batch-longest-right\n",
    "tokenizer.padding-token=<pad>:1\n",
    "tokenizer.padding-type-id=0\n",
    "tokenizer.padding-multiple=none\n",
    "projection=passage: {key}: {value}\n",
    "runtime=onnxruntime-1.28.0/ort-2.0.0-rc.13\n",
    "runtime.execution-provider=CPUExecutionProvider\n",
    "runtime.execution-device=unique-cpu-provider-device\n",
    "runtime.provider-selection=explicit-provider-and-device-type\n",
    "runtime.environment-execution-providers=false\n",
    "runtime.cpu-arena=false\n",
    "runtime.graph-optimization=level3\n",
    "runtime.execution=sequential\n",
    "runtime.model-batch=singleton\n",
    "runtime.intra-threads=1\n",
    "runtime.inter-threads=1\n",
    "runtime.deterministic-compute=true\n",
    "runtime.memory-pattern=false\n",
    "runtime.inputs=input_ids:i64,attention_mask:i64,token_type_ids:i64\n",
    "runtime.output=last_hidden_state:f32\n",
    "pooling=attention-mask-weighted-mean-f64-left-to-right\n",
    "normalization=l2-f64\n",
    "quantization=i16:round-ties-away-from-zero:scale-32767:clamp-minus32767-plus32767\n",
    "dimensions=384\n",
);

const PROFILE_FINGERPRINT: [u8; 32] = [
    0x12, 0x97, 0xd8, 0xd2, 0x8c, 0x7d, 0xbe, 0x9c, 0x62, 0x4a, 0x13, 0xa0, 0xaf, 0xa6, 0xbf, 0x8a,
    0xa6, 0xeb, 0x7a, 0x43, 0xc3, 0x23, 0x53, 0x59, 0x08, 0x9c, 0xee, 0xd7, 0xdf, 0x1f, 0x5b, 0x25,
];

/// Stable identity of the fixed E5 Small encoding contract.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmbeddingProfile {
    fingerprint: [u8; 32],
}

impl EmbeddingProfile {
    /// Returns the SHA-256 identity of the canonical profile manifest.
    #[must_use]
    pub const fn fingerprint(self) -> [u8; 32] {
        self.fingerprint
    }

    /// Returns the fixed vector width.
    #[must_use]
    pub const fn dimensions(self) -> usize {
        EMBEDDING_DIMENSIONS
    }

    /// Returns the canonical manifest whose SHA-256 is the profile fingerprint.
    #[must_use]
    pub const fn manifest(self) -> &'static str {
        PROFILE_MANIFEST
    }
}

/// Fixed E5 Small FP32 profile used by this crate.
pub const E5_SMALL_PROFILE: EmbeddingProfile = EmbeddingProfile {
    fingerprint: PROFILE_FINGERPRINT,
};

#[cfg(test)]
mod tests {
    use super::{E5_SMALL_PROFILE, EMBEDDING_DIMENSIONS};
    use sha2::{Digest, Sha256};

    #[test]
    fn manifest_hash_is_the_published_profile_fingerprint() {
        let digest: [u8; 32] = Sha256::digest(E5_SMALL_PROFILE.manifest().as_bytes()).into();
        assert_eq!(digest, E5_SMALL_PROFILE.fingerprint());
        assert_eq!(E5_SMALL_PROFILE.dimensions(), EMBEDDING_DIMENSIONS);
    }
}
