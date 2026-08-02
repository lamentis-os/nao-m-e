use std::error::Error;
use std::fmt;

use crate::parameters::SCALE;

macro_rules! unsigned_id {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(u64);

        impl $name {
            #[doc = concat!("Creates a new ", stringify!($name), ".")]
            #[must_use]
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            #[doc = concat!("Returns the numeric value of this ", stringify!($name), ".")]
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl From<u64> for $name {
            fn from(value: u64) -> Self {
                Self::new(value)
            }
        }
    };
}

/// Non-zero 128-bit identifier for one logical memory.
///
/// Persist its canonical bytes. [`fmt::Display`] emits exactly 32 lowercase
/// hexadecimal digits for diagnostics only.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MemoryId {
    high: u64,
    low: u64,
}

impl MemoryId {
    /// Creates a non-zero memory identifier.
    pub const fn new(value: u128) -> Result<Self, MemoryIdError> {
        if value == 0 {
            return Err(MemoryIdError::Zero);
        }
        Ok(Self {
            high: (value >> 64) as u64,
            low: value as u64,
        })
    }

    /// Reconstructs a memory identifier from its canonical big-endian bytes.
    pub const fn from_be_bytes(bytes: [u8; 16]) -> Result<Self, MemoryIdError> {
        Self::new(u128::from_be_bytes(bytes))
    }

    /// Returns the underlying numeric value.
    #[must_use]
    pub const fn get(self) -> u128 {
        ((self.high as u128) << 64) | self.low as u128
    }

    /// Returns the canonical 16-byte big-endian representation.
    #[must_use]
    pub const fn to_be_bytes(self) -> [u8; 16] {
        self.get().to_be_bytes()
    }
}

impl fmt::Display for MemoryId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:016x}{:016x}", self.high, self.low)
    }
}

/// Durable atom reference composed of a memory ID and insertion sequence.
///
/// [`fmt::Display`] joins the memory ID and decimal sequence with a colon for
/// diagnostics only. The ID is a reference, not a permission or content hash.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AtomId {
    memory_id: MemoryId,
    sequence: u64,
}

// Keep frequently copied IDs padding-free without promising a stable Rust ABI.
const _: () = {
    assert!(std::mem::size_of::<MemoryId>() == 2 * std::mem::size_of::<u64>());
    assert!(std::mem::align_of::<MemoryId>() <= std::mem::align_of::<u64>());
    assert!(
        std::mem::size_of::<AtomId>()
            == std::mem::size_of::<MemoryId>() + std::mem::size_of::<u64>()
    );
};

impl AtomId {
    /// Constructs an identifier without checking that the atom exists.
    #[must_use]
    pub const fn from_parts(memory_id: MemoryId, sequence: u64) -> Self {
        Self {
            memory_id,
            sequence,
        }
    }

    /// Returns the owning logical memory identifier.
    #[must_use]
    pub const fn memory_id(self) -> MemoryId {
        self.memory_id
    }

    /// Returns the monotonic insertion sequence within the memory.
    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
}

impl fmt::Display for AtomId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.memory_id, self.sequence)
    }
}

unsigned_id!(
    PredicateId,
    "Caller-owned identifier for the predicate of a statement."
);
unsigned_id!(
    TermId,
    "Caller-owned identifier for an entity, value, or other symbolic term."
);
unsigned_id!(
    SourceId,
    "Caller-owned identifier for the provenance source of an episode."
);

/// Signed milliseconds on a caller-defined timeline.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TimestampMs(i64);

impl TimestampMs {
    /// Creates a timestamp without imposing clock or ordering semantics.
    #[must_use]
    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    /// Returns the underlying millisecond value.
    #[must_use]
    pub const fn get(self) -> i64 {
        self.0
    }
}

impl From<i64> for TimestampMs {
    fn from(value: i64) -> Self {
        Self::new(value)
    }
}

/// A symbolic predicate applied to one or more ordered arguments.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Statement {
    predicate: PredicateId,
    arguments: Box<[TermId]>,
}

impl Statement {
    /// Constructs a statement, rejecting an empty argument list.
    pub fn new(predicate: PredicateId, arguments: Vec<TermId>) -> Result<Self, ModelError> {
        if arguments.is_empty() {
            return Err(ModelError::EmptyArguments);
        }

        Ok(Self {
            predicate,
            arguments: arguments.into_boxed_slice(),
        })
    }

    /// Returns the predicate identifier.
    #[must_use]
    pub const fn predicate(&self) -> PredicateId {
        self.predicate
    }

    /// Returns the ordered argument identifiers.
    #[must_use]
    pub fn arguments(&self) -> &[TermId] {
        &self.arguments
    }
}

/// Construction data for one episode atom.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpisodeDraft {
    /// Time at which the represented event occurred.
    pub occurred_at: TimestampMs,
    /// Time at which the event was recorded.
    pub recorded_at: TimestampMs,
    /// Context statements, sorted and deduplicated on insertion.
    pub context: Vec<Statement>,
    /// Observation recorded by the episode.
    pub observation: Statement,
    /// Action performed during the episode, if any.
    pub action: Option<Statement>,
    /// Observed outcome, if any.
    pub outcome: Option<Statement>,
    /// Caller-defined provenance source.
    pub source: SourceId,
}

/// An immutable episode stored in a [`crate::MemoryV0`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpisodeAtom {
    id: AtomId,
    occurred_at: TimestampMs,
    recorded_at: TimestampMs,
    context: Box<[Statement]>,
    observation: Statement,
    action: Option<Statement>,
    outcome: Option<Statement>,
    source: SourceId,
}

impl EpisodeAtom {
    pub(crate) fn from_draft(id: AtomId, mut draft: EpisodeDraft) -> Self {
        draft.context.sort_unstable();
        draft.context.dedup();

        Self {
            id,
            occurred_at: draft.occurred_at,
            recorded_at: draft.recorded_at,
            context: draft.context.into_boxed_slice(),
            observation: draft.observation,
            action: draft.action,
            outcome: draft.outcome,
            source: draft.source,
        }
    }

    /// Returns the atom identifier.
    #[must_use]
    pub const fn id(&self) -> AtomId {
        self.id
    }

    /// Returns when the represented event occurred.
    #[must_use]
    pub const fn occurred_at(&self) -> TimestampMs {
        self.occurred_at
    }

    /// Returns when the event was recorded.
    #[must_use]
    pub const fn recorded_at(&self) -> TimestampMs {
        self.recorded_at
    }

    /// Returns the sorted, duplicate-free context.
    #[must_use]
    pub fn context(&self) -> &[Statement] {
        &self.context
    }

    /// Returns the required observation.
    #[must_use]
    pub const fn observation(&self) -> &Statement {
        &self.observation
    }

    /// Returns the optional action.
    #[must_use]
    pub const fn action(&self) -> Option<&Statement> {
        self.action.as_ref()
    }

    /// Returns the optional outcome.
    #[must_use]
    pub const fn outcome(&self) -> Option<&Statement> {
        self.outcome.as_ref()
    }

    /// Returns the provenance source.
    #[must_use]
    pub const fn source(&self) -> SourceId {
        self.source
    }
}

/// Query-local recall activation measured in parts per million.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Activation(u32);

impl Activation {
    /// Zero projected activation.
    pub const ZERO: Self = Self(0);

    /// Creates a projected activation from zero through [`crate::SCALE`].
    pub const fn from_ppm(value: u32) -> Result<Self, ValueError> {
        if value > SCALE {
            return Err(ValueError::OutOfRange { value });
        }
        Ok(Self(value))
    }

    /// Returns the parts-per-million representation.
    #[must_use]
    pub const fn as_ppm(self) -> u32 {
        self.0
    }
}

/// Positive relevance influence measured in parts per million.
///
/// A weight controls source-conditioned accessibility; it is not a probability
/// or confidence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InfluenceWeight(u32);

impl InfluenceWeight {
    /// Creates a weight from one through [`crate::SCALE`].
    pub const fn from_ppm(value: u32) -> Result<Self, ValueError> {
        if value == 0 {
            return Err(ValueError::ZeroWeight);
        }
        if value > SCALE {
            return Err(ValueError::OutOfRange { value });
        }
        Ok(Self(value))
    }

    /// Returns the parts-per-million representation.
    #[must_use]
    pub const fn as_ppm(self) -> u32 {
        self.0
    }
}

/// Failure while constructing a symbolic model value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelError {
    /// A statement was constructed without arguments.
    EmptyArguments,
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyArguments => {
                formatter.write_str("a statement requires at least one argument")
            }
        }
    }
}

impl Error for ModelError {}

/// Failure while constructing a durable memory identifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryIdError {
    /// The all-zero value is reserved and cannot identify a memory.
    Zero,
}

impl fmt::Display for MemoryIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => formatter.write_str("a memory identifier must be non-zero"),
        }
    }
}

impl Error for MemoryIdError {}

/// Failure while constructing a fixed-point value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueError {
    /// A value exceeded [`crate::SCALE`].
    OutOfRange {
        /// Rejected parts-per-million value.
        value: u32,
    },
    /// Zero cannot represent an existing relevance edge.
    ZeroWeight,
}

impl fmt::Display for ValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { value } => {
                write!(formatter, "fixed-point value {value} exceeds {SCALE}")
            }
            Self::ZeroWeight => formatter.write_str("an influence weight must be positive"),
        }
    }
}

impl Error for ValueError {}

/// Failure while inserting an episode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryError {
    /// The monotonic identifier space is exhausted.
    IdExhausted,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdExhausted => formatter.write_str("atom identifier space is exhausted"),
        }
    }
}

impl Error for MemoryError {}

/// Failure while recalling or changing relevance topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphError {
    /// An operation referred to an atom absent from this memory.
    UnknownAtom(AtomId),
    /// A feedback event supplied more target entries than allowed.
    FeedbackTargetLimitExceeded {
        /// Number of supplied target entries.
        count: usize,
        /// Maximum accepted target entries.
        max: usize,
    },
    /// A relevance edge attempted to connect an atom to itself.
    SelfEdge(AtomId),
    /// Updated outgoing weights would exceed [`crate::SCALE`].
    OutgoingWeightBudgetExceeded {
        /// Source whose outgoing budget would be exceeded.
        from: AtomId,
        /// Rejected outgoing total.
        attempted_ppm: u64,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownAtom(id) => write!(formatter, "unknown atom {id}"),
            Self::FeedbackTargetLimitExceeded { count, max } => {
                write!(formatter, "feedback target count {count} exceeds {max}")
            }
            Self::SelfEdge(id) => write!(formatter, "atom {id} cannot influence itself"),
            Self::OutgoingWeightBudgetExceeded {
                from,
                attempted_ppm,
            } => write!(
                formatter,
                "atom {from} outgoing weight {attempted_ppm} exceeds {SCALE}",
            ),
        }
    }
}

impl Error for GraphError {}
