use crate::{E5_SMALL_PROFILE, EMBEDDING_DIMENSIONS, EmbeddingProfile};

/// Borrowed normalized free-text query for semantic retrieval.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct QueryText<'a> {
    value: &'a str,
}

impl<'a> QueryText<'a> {
    /// Creates one borrowed normalized query.
    #[must_use]
    pub const fn new(value: &'a str) -> Self {
        Self { value }
    }

    /// Returns the normalized query text.
    #[must_use]
    pub const fn value(self) -> &'a str {
        self.value
    }

    pub(crate) fn project(self) -> String {
        format!("query: {}", self.value)
    }
}

/// Borrowed normalized text for one bound attribute-key/value cue.
///
/// The semantic projection keeps the key and value together so identical
/// values under different attribute keys remain distinguishable.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CueText<'a> {
    key: &'a str,
    value: &'a str,
}

impl<'a> CueText<'a> {
    /// Creates one borrowed key/value cue.
    #[must_use]
    pub const fn new(key: &'a str, value: &'a str) -> Self {
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

    pub(crate) fn project(self) -> String {
        format!("passage: {}: {}", self.key, self.value)
    }
}

/// One non-zero 384-dimensional embedding in the canonical signed 16-bit
/// representation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Embedding {
    values: Box<[i16; EMBEDDING_DIMENSIONS]>,
}

impl Embedding {
    /// Constructs an embedding from its ordered signed components.
    ///
    /// Returns `None` unless `values` has exactly 384 components, every
    /// component is in `-32_767..=32_767`, and at least one is non-zero.
    #[must_use]
    pub fn new(values: Vec<i16>) -> Option<Self> {
        if values.len() != EMBEDDING_DIMENSIONS
            || values.contains(&i16::MIN)
            || values.iter().all(|value| *value == 0)
        {
            return None;
        }
        let values = values.into_boxed_slice().try_into().ok()?;
        Some(Self { values })
    }

    /// Returns the fixed profile of this vector.
    #[must_use]
    pub const fn profile(&self) -> EmbeddingProfile {
        E5_SMALL_PROFILE
    }

    /// Returns the ordered signed components.
    #[must_use]
    pub fn values(&self) -> &[i16] {
        self.values.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::{CueText, Embedding, QueryText};
    use crate::{E5_SMALL_PROFILE, EMBEDDING_DIMENSIONS};

    #[test]
    fn cue_projection_binds_key_and_value() {
        let cue = CueText::new("problem", "http 404");
        assert_eq!(cue.key(), "problem");
        assert_eq!(cue.value(), "http 404");
        assert_eq!(cue.project(), "passage: problem: http 404");
    }

    #[test]
    fn query_projection_uses_the_retrieval_prefix() {
        let query = QueryText::new("login bug in lamentis");
        assert_eq!(query.value(), "login bug in lamentis");
        assert_eq!(query.project(), "query: login bug in lamentis");
    }

    #[test]
    fn embedding_requires_fixed_width_and_non_zero_content() {
        assert_eq!(Embedding::new(vec![1; EMBEDDING_DIMENSIONS - 1]), None);
        assert_eq!(Embedding::new(vec![0; EMBEDDING_DIMENSIONS]), None);
        assert_eq!(Embedding::new(vec![i16::MIN; EMBEDDING_DIMENSIONS]), None);

        let embedding = Embedding::new(vec![1; EMBEDDING_DIMENSIONS]).unwrap();
        assert_eq!(embedding.profile(), E5_SMALL_PROFILE);
        assert_eq!(embedding.values().len(), EMBEDDING_DIMENSIONS);
    }
}
