#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Fixed-profile local semantic encoding for `nao_m_e` episodes and queries.
//!
//! The crate owns the exact fixed E5 Small model, tokenizer, projection,
//! pooling, normalization, and fixed-point conversion contract. Pinned model
//! artifacts are an installation prerequisite: the first encoding request
//! verifies and loads them from the local Hugging Face cache without
//! network fallback. The crate owns no memory, SQLite, or retrieval state.

mod encoder;
mod error;
mod model;
mod profile;

pub use encoder::SemanticEncoder;
pub use error::SemanticError;
pub use model::{Embedding, EpisodeText, QueryText};
pub use profile::{E5_SMALL_PROFILE, EMBEDDING_DIMENSIONS, EmbeddingProfile};
