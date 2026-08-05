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

/// Borrowed normalized attributes for one canonical episode passage.
///
/// Attribute pairs are treated as an unordered set. Projection sorts and
/// deduplicates them lexically, so symbol allocation and caller order cannot
/// change the encoded passage.
#[derive(Clone, Copy, Debug)]
pub struct EpisodeText<'a> {
    attributes: &'a [(&'a str, &'a str)],
}

impl<'a> EpisodeText<'a> {
    /// Creates one borrowed non-empty episode attribute set.
    ///
    /// Returns `None` when no bound attribute pair is supplied.
    #[must_use]
    pub const fn new(attributes: &'a [(&'a str, &'a str)]) -> Option<Self> {
        if attributes.is_empty() {
            None
        } else {
            Some(Self { attributes })
        }
    }

    /// Returns the normalized bound attribute pairs.
    #[must_use]
    pub const fn attributes(self) -> &'a [(&'a str, &'a str)] {
        self.attributes
    }

    pub(crate) fn project(self) -> String {
        let mut attributes = self.attributes.to_vec();
        attributes.sort_unstable();
        attributes.dedup();

        let text_bytes = attributes
            .iter()
            .map(|(key, value)| key.len() + value.len())
            .sum::<usize>();
        let mut projected = String::with_capacity(9 + text_bytes + attributes.len() * 3);
        projected.push_str("passage: ");
        for (index, (key, value)) in attributes.into_iter().enumerate() {
            if index != 0 {
                projected.push('\n');
            }
            projected.push_str(key);
            projected.push_str(": ");
            projected.push_str(value);
        }
        projected
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
    use super::{Embedding, EpisodeText, QueryText};
    use crate::{E5_SMALL_PROFILE, EMBEDDING_DIMENSIONS};

    #[test]
    fn episode_projection_is_non_empty_ordered_and_duplicate_free() {
        assert!(EpisodeText::new(&[]).is_none());
        let attributes = [
            ("status", "failed"),
            ("problem", "http 404"),
            ("status", "failed"),
        ];
        let episode = EpisodeText::new(&attributes).unwrap();
        assert_eq!(episode.attributes(), &attributes);
        assert_eq!(
            episode.project(),
            "passage: problem: http 404\nstatus: failed"
        );
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
