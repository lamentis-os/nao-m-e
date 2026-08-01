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

/// Stable identifier assigned monotonically within one memory namespace.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AtomId {
    memory_namespace: u64,
    local_id: u64,
}

impl AtomId {
    pub(crate) const fn from_raw(memory_namespace: u64, local_id: u64) -> Self {
        Self {
            memory_namespace,
            local_id,
        }
    }

    /// Returns the memory-local monotonic identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.local_id
    }

    /// Returns the process-local namespace of the owning memory.
    #[must_use]
    pub const fn memory_namespace(self) -> u64 {
        self.memory_namespace
    }
}

impl fmt::Display for AtomId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.memory_namespace, self.local_id)
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

/// Milliseconds on a caller-defined timeline.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TimestampMs(i64);

impl TimestampMs {
    /// Creates a timestamp from a signed millisecond value.
    #[must_use]
    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    /// Returns the signed millisecond value.
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

/// A symbolic, ordered predicate application.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Statement {
    predicate: PredicateId,
    arguments: Vec<TermId>,
}

impl Statement {
    /// Constructs a statement.
    ///
    /// Argument order is meaningful. At least one argument is required.
    pub fn new(predicate: PredicateId, arguments: Vec<TermId>) -> Result<Self, ModelError> {
        if arguments.is_empty() {
            return Err(ModelError::EmptyArguments);
        }

        Ok(Self {
            predicate,
            arguments,
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

/// Mutable construction input for one immutable episode atom.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpisodeDraft {
    /// Time at which the represented event occurred.
    pub occurred_at: TimestampMs,
    /// Time at which the event was recorded.
    pub recorded_at: TimestampMs,
    /// Unordered contextual statements, canonicalized on insertion.
    pub context: Vec<Statement>,
    /// Required observation that bounds the episode.
    pub observation: Statement,
    /// Optional action performed during the episode.
    pub action: Option<Statement>,
    /// Optional observed outcome of the action or event.
    pub outcome: Option<Statement>,
    /// Provenance source supplied by the caller.
    pub source: SourceId,
}

/// Immutable, addressable episode stored by the memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpisodeAtom {
    id: AtomId,
    occurred_at: TimestampMs,
    recorded_at: TimestampMs,
    context: Vec<Statement>,
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
            context: draft.context,
            observation: draft.observation,
            action: draft.action,
            outcome: draft.outcome,
            source: draft.source,
        }
    }

    /// Returns the memory-local identifier.
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

    /// Returns canonical, sorted, duplicate-free context.
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

/// Fixed-point activation measured in parts per million.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Activation(u32);

impl Activation {
    /// Zero activation.
    pub const ZERO: Self = Self(0);

    /// Full activation.
    pub const ONE: Self = Self(SCALE);

    /// Creates activation in the inclusive range zero through one million.
    pub const fn from_ppm(value: u32) -> Result<Self, ValueError> {
        if value > SCALE {
            return Err(ValueError::OutOfRange { value });
        }
        Ok(Self(value))
    }

    pub(crate) const fn from_clamped_ppm(value: u32) -> Self {
        if value > SCALE {
            Self(SCALE)
        } else {
            Self(value)
        }
    }

    /// Returns the parts-per-million representation.
    #[must_use]
    pub const fn as_ppm(self) -> u32 {
        self.0
    }
}

/// Positive relevance influence measured in parts per million.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InfluenceWeight(u32);

impl InfluenceWeight {
    /// Creates a weight in the inclusive range one through one million.
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

/// Failure while constructing a fixed-point value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueError {
    /// A value exceeded the inclusive upper bound of one million.
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

/// Failure while changing the atom store.
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

/// Failure while changing activation or relevance topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphError {
    /// An operation referred to an atom absent from this memory.
    UnknownAtom(AtomId),
    /// A relevance edge attempted to connect an atom to itself.
    SelfEdge(AtomId),
    /// Updated outgoing weights would exceed one million.
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
