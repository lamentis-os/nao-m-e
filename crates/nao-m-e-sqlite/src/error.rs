use std::error::Error;
use std::fmt;

/// Failure while creating, opening, or saving a SQLite memory store.
#[derive(Debug)]
#[non_exhaustive]
pub enum StoreError {
    /// Filesystem work outside SQLite failed.
    Io(std::io::Error),
    /// SQLite rejected an operation.
    Database(rusqlite::Error),
    /// Operating-system entropy was unavailable while allocating a memory ID.
    Entropy(getrandom::Error),
    /// Persisted data did not satisfy the SQLite V2 contract.
    InvalidStore(StoreIntegrityError),
    /// Another store session committed after this session was opened or saved.
    ConcurrentModification {
        /// Revision from which this session attempted to save.
        expected_revision: i64,
        /// Revision currently stored in the database.
        actual_revision: i64,
    },
    /// The non-negative SQLite revision counter reached its maximum value.
    RevisionExhausted,
}

impl fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "SQLite store I/O failed: {error}"),
            Self::Database(error) => write!(formatter, "SQLite operation failed: {error}"),
            Self::Entropy(error) => write!(formatter, "memory ID generation failed: {error}"),
            Self::InvalidStore(error) => write!(formatter, "invalid SQLite memory store: {error}"),
            Self::ConcurrentModification {
                expected_revision,
                actual_revision,
            } => write!(
                formatter,
                "SQLite memory revision changed from {expected_revision} to {actual_revision}",
            ),
            Self::RevisionExhausted => {
                formatter.write_str("SQLite memory snapshot revision is exhausted")
            }
        }
    }
}

impl Error for StoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Database(error) => Some(error),
            Self::Entropy(error) => Some(error),
            Self::InvalidStore(error) => Some(error),
            Self::ConcurrentModification { .. } | Self::RevisionExhausted => None,
        }
    }
}

impl From<std::io::Error> for StoreError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<rusqlite::Error> for StoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error)
    }
}

impl From<getrandom::Error> for StoreError {
    fn from(error: getrandom::Error) -> Self {
        Self::Entropy(error)
    }
}

impl From<StoreIntegrityError> for StoreError {
    fn from(error: StoreIntegrityError) -> Self {
        Self::InvalidStore(error)
    }
}

/// Persisted-data violation detected while opening or saving a store.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StoreIntegrityError {
    /// The SQLite header does not identify a NAO-M-E database.
    ApplicationMismatch {
        /// Application identifier found in the SQLite header.
        found: i64,
    },
    /// The required singleton metadata row is absent.
    MissingMetadata,
    /// The database uses a format this adapter does not implement.
    UnsupportedFormatVersion {
        /// Format version stored in the database.
        found: i64,
    },
    /// SQLite's structural quick check did not report `ok`.
    QuickCheckFailed {
        /// Diagnostic returned by SQLite.
        detail: String,
    },
    /// SQLite reported at least one foreign-key violation.
    ForeignKeyViolation {
        /// Compact diagnostic identifying the violation.
        detail: String,
    },
    /// A column did not contain its canonical fixed-width encoding.
    InvalidEncoding {
        /// Table containing the invalid value.
        table: &'static str,
        /// Column containing the invalid value.
        column: &'static str,
    },
    /// The persisted memory identifier is zero or malformed.
    InvalidMemoryId,
    /// Metadata outside the separately classified fields is invalid.
    InvalidMetadata {
        /// Violated metadata invariant.
        detail: &'static str,
    },
    /// Episode sequences do not form the exact prefix `0..N`.
    NonContiguousEpisodeSequence {
        /// Sequence required at this position.
        expected: u64,
        /// Sequence read from the database.
        found: u64,
    },
    /// An episode or one of its statements is not canonical.
    InvalidEpisode {
        /// Sequence of the affected episode.
        sequence: u64,
        /// Violated episode invariant.
        detail: &'static str,
    },
    /// A persisted positive activation is invalid.
    InvalidActivation {
        /// Sequence of the affected episode.
        sequence: u64,
        /// Violated activation invariant.
        detail: &'static str,
    },
    /// A relevance edge is invalid.
    InvalidRelevance {
        /// Source sequence.
        from: u64,
        /// Target sequence.
        to: u64,
        /// Violated graph invariant.
        detail: &'static str,
    },
}

impl fmt::Display for StoreIntegrityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApplicationMismatch { found } => {
                write!(formatter, "unexpected SQLite application ID {found:#010x}")
            }
            Self::MissingMetadata => formatter.write_str("memory metadata row is missing"),
            Self::UnsupportedFormatVersion { found } => {
                write!(
                    formatter,
                    "unsupported SQLite memory format version {found}"
                )
            }
            Self::QuickCheckFailed { detail } => {
                write!(formatter, "SQLite quick check failed: {detail}")
            }
            Self::ForeignKeyViolation { detail } => {
                write!(formatter, "SQLite foreign-key check failed: {detail}")
            }
            Self::InvalidEncoding { table, column } => {
                write!(formatter, "invalid encoding in {table}.{column}")
            }
            Self::InvalidMemoryId => formatter.write_str("invalid persisted memory ID"),
            Self::InvalidMetadata { detail } => write!(formatter, "invalid metadata: {detail}"),
            Self::NonContiguousEpisodeSequence { expected, found } => write!(
                formatter,
                "expected episode sequence {expected}, found {found}",
            ),
            Self::InvalidEpisode { sequence, detail } => {
                write!(formatter, "invalid episode {sequence}: {detail}")
            }
            Self::InvalidActivation { sequence, detail } => {
                write!(
                    formatter,
                    "invalid activation for episode {sequence}: {detail}"
                )
            }
            Self::InvalidRelevance { from, to, detail } => {
                write!(formatter, "invalid relevance edge {from}->{to}: {detail}")
            }
        }
    }
}

impl Error for StoreIntegrityError {}
