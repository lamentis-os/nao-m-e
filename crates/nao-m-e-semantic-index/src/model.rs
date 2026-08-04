use crate::EmbedderError;

/// Stable identity and vector width of one embedding configuration.
///
/// The fingerprint must identify every setting that can affect emitted
/// vectors, including the model, weights, tokenizer, cue projection, and
/// quantization. It is an identity token, not a cryptographic trust claim.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmbeddingProfile {
    fingerprint: [u8; 32],
    dimension: u16,
}

impl EmbeddingProfile {
    /// Constructs a profile from a non-zero fingerprint and vector dimension.
    ///
    /// Returns `None` when the fingerprint is all zero or `dimension` is zero.
    /// A `u16` dimension therefore represents the supported range
    /// `1..=65_535` exactly.
    #[must_use]
    pub fn new(fingerprint: [u8; 32], dimension: u16) -> Option<Self> {
        if dimension == 0 || fingerprint.iter().all(|byte| *byte == 0) {
            return None;
        }
        Some(Self {
            fingerprint,
            dimension,
        })
    }

    /// Returns the stable 32-byte configuration fingerprint.
    #[must_use]
    pub const fn fingerprint(self) -> [u8; 32] {
        self.fingerprint
    }

    /// Returns the number of signed 16-bit components in every vector.
    #[must_use]
    pub const fn dimensions(self) -> u16 {
        self.dimension
    }
}

/// One non-zero, fixed-width embedding in signed 16-bit representation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Embedding {
    profile: EmbeddingProfile,
    values: Box<[i16]>,
}

impl Embedding {
    /// Constructs an embedding for `profile`.
    ///
    /// Returns `None` unless `values` has exactly the profile dimension and at
    /// least one component is non-zero.
    #[must_use]
    pub fn new(profile: EmbeddingProfile, values: Vec<i16>) -> Option<Self> {
        if values.len() != usize::from(profile.dimensions())
            || values.iter().all(|value| *value == 0)
        {
            return None;
        }
        Some(Self {
            profile,
            values: values.into_boxed_slice(),
        })
    }

    pub(crate) const fn profile(&self) -> EmbeddingProfile {
        self.profile
    }

    /// Returns the ordered signed components.
    #[must_use]
    pub fn values(&self) -> &[i16] {
        &self.values
    }
}

/// Borrowed normalized text for one bound attribute-key/value cue.
///
/// Keeping key and value bound avoids conflating an identical value used under
/// different attribute keys.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CueText<'a> {
    key: &'a str,
    value: &'a str,
}

impl<'a> CueText<'a> {
    /// Creates one borrowed key/value cue.
    #[must_use]
    pub(crate) const fn new(key: &'a str, value: &'a str) -> Self {
        Self { key, value }
    }

    /// Returns the normalized attribute-key text.
    #[must_use]
    pub const fn key(self) -> &'a str {
        self.key
    }

    /// Returns the normalized attribute-value text.
    #[must_use]
    pub const fn value(self) -> &'a str {
        self.value
    }
}

/// Supplies embeddings without coupling the index to a model runtime.
pub trait CueEmbedder {
    /// Returns the immutable profile of every vector produced by this embedder.
    fn profile(&self) -> EmbeddingProfile;

    /// Embeds one ordered batch of bound key/value cues.
    ///
    /// A successful implementation must return exactly one embedding per cue,
    /// in the same order. The index validates that count and every vector.
    fn embed_batch(&mut self, cues: &[CueText<'_>]) -> Result<Vec<Embedding>, EmbedderError>;
}

/// Counts describing the committed contents of a semantic cue index.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IndexStats {
    indexed_episode_count: u64,
    cue_count: u64,
    posting_count: u64,
}

impl IndexStats {
    pub(crate) const fn new(
        indexed_episode_count: u64,
        cue_count: u64,
        posting_count: u64,
    ) -> Self {
        Self {
            indexed_episode_count,
            cue_count,
            posting_count,
        }
    }

    /// Returns the number of authoritative episodes covered by the index.
    #[must_use]
    pub const fn indexed_episode_count(self) -> u64 {
        self.indexed_episode_count
    }

    /// Returns the number of distinct bound key/value cues.
    #[must_use]
    pub const fn cue_count(self) -> u64 {
        self.cue_count
    }

    /// Returns the number of cue-to-episode postings.
    #[must_use]
    pub const fn posting_count(self) -> u64 {
        self.posting_count
    }
}

#[cfg(test)]
mod tests {
    use super::{Embedding, EmbeddingProfile};

    const FINGERPRINT: [u8; 32] = [7; 32];

    #[test]
    fn profile_rejects_zero_components_and_preserves_valid_parts() {
        assert_eq!(EmbeddingProfile::new([0; 32], 1), None);
        assert_eq!(EmbeddingProfile::new(FINGERPRINT, 0), None);

        let profile = EmbeddingProfile::new(FINGERPRINT, u16::MAX).unwrap();
        assert_eq!(profile.fingerprint(), FINGERPRINT);
        assert_eq!(profile.dimensions(), u16::MAX);
    }

    #[test]
    fn embedding_requires_exact_dimension_and_a_non_zero_component() {
        let profile = EmbeddingProfile::new(FINGERPRINT, 3).unwrap();

        assert_eq!(Embedding::new(profile, vec![1, 2]), None);
        assert_eq!(Embedding::new(profile, vec![1, 2, 3, 4]), None);
        assert_eq!(Embedding::new(profile, vec![0, 0, 0]), None);

        let embedding = Embedding::new(profile, vec![i16::MIN, 0, i16::MAX]).unwrap();
        assert_eq!(embedding.profile(), profile);
        assert_eq!(embedding.values(), &[i16::MIN, 0, i16::MAX]);
    }
}
