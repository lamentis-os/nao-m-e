use std::error::Error;
use std::fmt;

use crate::EmbeddingProfile;

/// Type-erased failure returned by a caller-supplied cue embedder.
pub type EmbedderError = Box<dyn Error + Send + Sync + 'static>;

/// Failure while creating, opening, or synchronizing a semantic cue index.
#[derive(Debug)]
#[non_exhaustive]
pub enum IndexError {
    /// Filesystem work outside SQLite failed.
    Io(std::io::Error),
    /// SQLite rejected an operation.
    Database(rusqlite::Error),
    /// The authoritative memory store could not be read.
    MemoryStore(nao_m_e_sqlite::StoreError),
    /// Persisted sidecar data violated the semantic-index contract.
    InvalidIndex(IndexIntegrityError),
    /// The caller-supplied embedding implementation failed.
    Embedder(EmbedderError),
    /// The embedder returned a different number of vectors than requested.
    EmbeddingBatchLength {
        /// Number of vectors required for the input cue batch.
        expected: usize,
        /// Number of vectors returned by the embedder.
        found: usize,
    },
    /// An operation used a different embedding configuration than the index.
    ProfileMismatch {
        /// Profile persisted by the index.
        expected: EmbeddingProfile,
        /// Profile supplied for the operation.
        found: EmbeddingProfile,
    },
    /// Another sidecar session changed the indexed episode coverage.
    ConcurrentModification {
        /// Indexed episode count owned by this sidecar session.
        expected_episode_count: u64,
        /// Indexed episode count currently stored in the sidecar.
        actual_episode_count: u64,
    },
    /// The semantic cue identifier space is exhausted.
    CueIdExhausted,
    /// The authoritative episode count cannot be represented by this index.
    EpisodeCountExhausted,
}

impl fmt::Display for IndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "semantic index I/O failed: {error}"),
            Self::Database(error) => {
                write!(formatter, "semantic index SQLite operation failed: {error}")
            }
            Self::MemoryStore(error) => write!(formatter, "memory store operation failed: {error}"),
            Self::InvalidIndex(error) => write!(formatter, "invalid semantic cue index: {error}"),
            Self::Embedder(error) => write!(formatter, "cue embedding failed: {error}"),
            Self::EmbeddingBatchLength { expected, found } => write!(
                formatter,
                "cue embedder returned {found} vectors for a batch of {expected} cues",
            ),
            Self::ProfileMismatch { expected, found } => write!(
                formatter,
                "embedding profile mismatch: expected dimension {} and fingerprint {:02x?}, found dimension {} and fingerprint {:02x?}",
                expected.dimensions(),
                expected.fingerprint(),
                found.dimensions(),
                found.fingerprint(),
            ),
            Self::ConcurrentModification {
                expected_episode_count,
                actual_episode_count,
            } => write!(
                formatter,
                "semantic index coverage changed from {expected_episode_count} to {actual_episode_count} episodes",
            ),
            Self::CueIdExhausted => {
                formatter.write_str("semantic cue identifier space is exhausted")
            }
            Self::EpisodeCountExhausted => {
                formatter.write_str("memory episode count exceeds the semantic index range")
            }
        }
    }
}

impl Error for IndexError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Database(error) => Some(error),
            Self::MemoryStore(error) => Some(error),
            Self::InvalidIndex(error) => Some(error),
            Self::Embedder(error) => Some(error.as_ref()),
            Self::EmbeddingBatchLength { .. }
            | Self::ProfileMismatch { .. }
            | Self::ConcurrentModification { .. }
            | Self::CueIdExhausted
            | Self::EpisodeCountExhausted => None,
        }
    }
}

impl From<std::io::Error> for IndexError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<rusqlite::Error> for IndexError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error)
    }
}

impl From<nao_m_e_sqlite::StoreError> for IndexError {
    fn from(error: nao_m_e_sqlite::StoreError) -> Self {
        Self::MemoryStore(error)
    }
}

impl From<EmbedderError> for IndexError {
    fn from(error: EmbedderError) -> Self {
        Self::Embedder(error)
    }
}

impl From<IndexIntegrityError> for IndexError {
    fn from(error: IndexIntegrityError) -> Self {
        Self::InvalidIndex(error)
    }
}

/// Persisted-data violation detected in a semantic cue sidecar.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexIntegrityError {
    /// The SQLite header does not identify a NAO-M-E semantic index.
    ApplicationMismatch {
        /// Application identifier found in the SQLite header.
        found: i64,
    },
    /// The required singleton metadata row is absent.
    MissingMetadata,
    /// The sidecar uses a format this adapter does not implement.
    UnsupportedFormatVersion {
        /// Format version stored in the sidecar.
        found: i64,
    },
    /// Semantic-index metadata is invalid.
    InvalidMetadata {
        /// Violated metadata invariant.
        detail: &'static str,
    },
    /// A column did not contain its canonical fixed-width encoding.
    InvalidEncoding {
        /// Table containing the invalid value.
        table: &'static str,
        /// Column containing the invalid value.
        column: &'static str,
    },
    /// The sidecar belongs to a different authoritative logical memory.
    MemoryMismatch,
    /// A persisted cue is malformed or not canonical.
    InvalidCue {
        /// Identifier of the malformed cue.
        cue_id: u64,
        /// Violated cue invariant.
        detail: &'static str,
    },
    /// A persisted cue-to-episode posting is invalid.
    InvalidPosting {
        /// Authoritative episode sequence referenced by the posting.
        sequence: u64,
        /// Semantic cue identifier referenced by the posting.
        cue_id: u64,
        /// Violated posting invariant.
        detail: &'static str,
    },
    /// SQLite's structural quick check did not report `ok`.
    QuickCheckFailed {
        /// Diagnostic returned by SQLite.
        detail: String,
    },
    /// SQLite reported at least one broken foreign-key reference.
    ForeignKeyCheckFailed,
}

impl fmt::Display for IndexIntegrityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApplicationMismatch { found } => {
                write!(formatter, "unexpected SQLite application ID {found:#010x}")
            }
            Self::MissingMetadata => formatter.write_str("semantic index metadata row is missing"),
            Self::UnsupportedFormatVersion { found } => {
                write!(
                    formatter,
                    "unsupported semantic index format version {found}"
                )
            }
            Self::InvalidMetadata { detail } => write!(formatter, "invalid metadata: {detail}"),
            Self::InvalidEncoding { table, column } => {
                write!(formatter, "invalid encoding in {table}.{column}")
            }
            Self::MemoryMismatch => {
                formatter.write_str("semantic index belongs to a different logical memory")
            }
            Self::InvalidCue { cue_id, detail } => {
                write!(formatter, "invalid semantic cue {cue_id}: {detail}")
            }
            Self::InvalidPosting {
                sequence,
                cue_id,
                detail,
            } => write!(
                formatter,
                "invalid semantic posting {cue_id}->{sequence}: {detail}",
            ),
            Self::QuickCheckFailed { detail } => {
                write!(formatter, "SQLite quick check failed: {detail}")
            }
            Self::ForeignKeyCheckFailed => formatter.write_str("SQLite foreign-key check failed"),
        }
    }
}

impl Error for IndexIntegrityError {}
