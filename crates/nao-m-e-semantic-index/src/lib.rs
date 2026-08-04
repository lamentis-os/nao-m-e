#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Rebuildable semantic cue indexing for a persisted `nao_m_e` memory.
//!
//! The index is a derived SQLite sidecar: the authoritative episodes, symbols,
//! and feedback remain in [`nao_m_e_sqlite::SqliteStore`]. Embedding generation
//! is supplied by the caller through [`CueEmbedder`], so this crate neither
//! selects nor downloads a machine-learning model.

mod error;
mod index;
mod model;

pub use error::{EmbedderError, IndexError, IndexIntegrityError};
pub use index::SemanticCueIndex;
pub use model::{CueEmbedder, CueText, Embedding, EmbeddingProfile, IndexStats};
