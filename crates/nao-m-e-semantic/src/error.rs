use std::error::Error;
use std::fmt;

/// Failure while loading the fixed model or encoding semantic cues.
#[derive(Debug)]
#[non_exhaustive]
pub enum SemanticError {
    /// One request exceeded the fixed batch bound.
    BatchTooLarge {
        /// Maximum number of accepted cues.
        maximum: usize,
        /// Number of supplied cues.
        found: usize,
    },
    /// Hugging Face cache lookup or download failed.
    ArtifactDownload {
        /// Repository path of the required artifact.
        file: &'static str,
        /// Underlying Hub failure.
        source: hf_hub::HFError,
    },
    /// Reading a cached artifact failed.
    ArtifactRead {
        /// Repository path of the required artifact.
        file: &'static str,
        /// Underlying filesystem failure.
        source: std::io::Error,
    },
    /// An artifact remained different from its pinned SHA-256 after one forced retry.
    ArtifactHashMismatch {
        /// Repository path of the invalid artifact.
        file: &'static str,
        /// Required digest.
        expected: [u8; 32],
        /// Digest read after the retry.
        found: [u8; 32],
    },
    /// The pinned tokenizer could not be loaded or configured.
    Tokenizer {
        /// Stable diagnostic from the tokenizer runtime.
        detail: String,
    },
    /// ONNX Runtime rejected model loading, input construction, or inference.
    Runtime(ort::Error),
    /// The fixed CPU-only execution device could not be selected uniquely.
    RuntimeConfiguration {
        /// Violated runtime invariant.
        detail: &'static str,
    },
    /// The pinned artifacts exposed a shape or value outside the fixed profile.
    InvalidModelOutput {
        /// Violated invariant.
        detail: &'static str,
    },
}

impl fmt::Display for SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchTooLarge { maximum, found } => {
                write!(
                    formatter,
                    "semantic batch contains {found} cues; maximum is {maximum}"
                )
            }
            Self::ArtifactDownload { file, source } => {
                write!(
                    formatter,
                    "failed to obtain semantic artifact {file}: {source}"
                )
            }
            Self::ArtifactRead { file, source } => {
                write!(
                    formatter,
                    "failed to read semantic artifact {file}: {source}"
                )
            }
            Self::ArtifactHashMismatch {
                file,
                expected,
                found,
            } => write!(
                formatter,
                "semantic artifact {file} has SHA-256 {found:02x?}; expected {expected:02x?}",
            ),
            Self::Tokenizer { detail } => write!(formatter, "semantic tokenizer failed: {detail}"),
            Self::Runtime(error) => write!(formatter, "semantic ONNX inference failed: {error}"),
            Self::RuntimeConfiguration { detail } => {
                write!(
                    formatter,
                    "semantic runtime configuration is invalid: {detail}"
                )
            }
            Self::InvalidModelOutput { detail } => {
                write!(formatter, "semantic model output is invalid: {detail}")
            }
        }
    }
}

impl Error for SemanticError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ArtifactDownload { source, .. } => Some(source),
            Self::ArtifactRead { source, .. } => Some(source),
            Self::Runtime(error) => Some(error),
            Self::BatchTooLarge { .. }
            | Self::ArtifactHashMismatch { .. }
            | Self::Tokenizer { .. }
            | Self::RuntimeConfiguration { .. }
            | Self::InvalidModelOutput { .. } => None,
        }
    }
}

impl From<ort::Error> for SemanticError {
    fn from(error: ort::Error) -> Self {
        Self::Runtime(error)
    }
}
