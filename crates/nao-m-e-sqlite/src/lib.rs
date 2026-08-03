#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! SQLite persistence for the deterministic `nao_m_e` memory kernel.
//!
//! A [`SqliteStore`] owns one [`nao_m_e::Memory`]. Mutations remain in
//! memory until an explicit [`SqliteStore::save`] atomically commits staged
//! predicate and term symbols, newly appended episodes, and feedback changes.

mod codec;
mod error;
mod schema;
mod store;

pub use error::{StoreError, StoreIntegrityError};
pub use store::SqliteStore;
