#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! SQLite persistence for the deterministic `nao_m_e` memory kernel.
//!
//! A [`SqliteStore`] owns one [`nao_m_e::MemoryV0`]. Mutations remain in
//! memory until an explicit [`SqliteStore::save`] commits the relevance graph
//! and any newly appended episodes.

mod codec;
mod error;
mod schema;
mod store;

pub use error::{StoreError, StoreIntegrityError};
pub use store::SqliteStore;
