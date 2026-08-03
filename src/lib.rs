#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Deterministic in-memory kernel for immutable symbolic episode atoms.
//!
//! Episode content is immutable; directed feedback traces are mutable. Cue-derived
//! candidates, fixed-point arithmetic, and ordered storage make recall and
//! feedback reproducible.
//! Cross-cutting semantics are specified in `docs/core-contract.md`.

mod memory;
mod model;
mod parameters;

pub use memory::{FeedbackEdge, Memory, RecallHit};
pub use model::{Activation, FeedbackTrace, ValueError};
pub use model::{AtomId, MemoryId, PredicateId, SourceId, TermId, TimestampMs};
pub use model::{EpisodeAtom, EpisodeDraft, Statement};
pub use model::{GraphError, MemoryError, MemoryIdError, ModelError};
pub use parameters::{
    FEEDBACK_HISTORY_CAPACITY, FEEDBACK_PRIOR_MASS, LEARNED_GAIN_PPM, MAX_FEEDBACK_TARGETS, SCALE,
    STRUCTURAL_GAIN_PPM,
};
