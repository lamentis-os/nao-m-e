#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Deterministic in-memory kernel for immutable, symbolic episode atoms.
//!
//! [`MemoryV0`] keeps atom content separate from mutable activation and directed
//! relevance. Fixed-point arithmetic and ordered storage make state transitions
//! reproducible. The repository README defines the full V0 contract and usage.

mod memory;
mod model;
mod parameters;

pub use memory::{MemoryV0, RecallHit, RelevanceEdge};
pub use model::{Activation, InfluenceWeight, ValueError};
pub use model::{AtomId, PredicateId, SourceId, TermId, TimestampMs};
pub use model::{EpisodeAtom, EpisodeDraft, Statement};
pub use model::{GraphError, MemoryError, ModelError};
pub use parameters::{PROPAGATION_GAIN_PPM, RETENTION_PPM, SCALE};
