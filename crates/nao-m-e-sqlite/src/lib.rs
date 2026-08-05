#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! SQLite persistence for the deterministic `nao_m_e` memory kernel.
//!
//! A [`SqliteStore`] owns one [`nao_m_e::Memory`]. Mutations remain in
//! memory until an explicit [`SqliteStore::save`] atomically commits staged
//! symbols, semantic cue embeddings, newly appended episodes, and feedback
//! changes. [`SqliteStore::check`] performs the deliberately separate complete
//! file and semantic-projection audit.

mod error;
mod format;
mod store;

pub use error::{StoreError, StoreIntegrityError};
pub use store::SqliteStore;
