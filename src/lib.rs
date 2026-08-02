#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Deterministic in-memory kernel for immutable symbolic episode atoms.
//!
//! Episode content is immutable; activation and directed relevance are mutable.
//! Fixed-point arithmetic and ordered storage make transitions reproducible.
//! Cross-cutting V0 semantics are specified in `docs/v0-contract.md`.

mod memory;
mod model;
mod parameters;

pub use memory::{MemoryV0, RecallHit, RelevanceEdge};
pub use model::{Activation, InfluenceWeight, ValueError};
pub use model::{AtomId, MemoryId, PredicateId, SourceId, TermId, TimestampMs};
pub use model::{EpisodeAtom, EpisodeDraft, Statement};
pub use model::{GraphError, MemoryError, MemoryIdError, ModelError};
pub use parameters::{
    FEEDBACK_MAX_EVENT_PPM, FEEDBACK_TARGET_STEP_PPM, MAX_FEEDBACK_TARGETS, PROPAGATION_GAIN_PPM,
    RETENTION_PPM, SCALE,
};
