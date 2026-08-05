#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Fixed-profile local semantic encoding for `nao_m_e` attribute cues.
//!
//! The crate owns the exact multilingual E5 Small model, tokenizer, projection,
//! pooling, normalization, and fixed-point conversion contract. Model artifacts
//! are loaded lazily through the Hugging Face cache on the first non-empty
//! encoding request. It owns no memory, SQLite, or retrieval state.

mod encoder;
mod error;
mod model;
mod profile;

pub use encoder::SemanticEncoder;
pub use error::SemanticError;
pub use model::{CueText, Embedding};
pub use profile::{
    E5_SMALL_PROFILE, EMBEDDING_DIMENSIONS, EmbeddingProfile, MAX_EMBEDDING_BATCH_SIZE,
};
