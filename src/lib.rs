#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Deterministic V0 memory-atom kernel.
//!
//! The kernel stores immutable, symbolically structured episode atoms and keeps
//! activation plus directed relevance edges outside those atoms. All dynamic
//! arithmetic uses fixed-point integers, making a state transition independent
//! of floating-point implementations and hash-map iteration order.
//!
//! # Example
//!
//! ```
//! use nao_m_e::{
//!     Activation, EpisodeDraft, InfluenceWeight, MemoryV0, PredicateId,
//!     SourceId, Statement, TermId, TimestampMs,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let observation = Statement::new(
//!     PredicateId::new(1),
//!     vec![TermId::new(10), TermId::new(11)],
//! )?;
//! let episode = EpisodeDraft {
//!     occurred_at: TimestampMs::new(1_000),
//!     recorded_at: TimestampMs::new(1_001),
//!     context: Vec::new(),
//!     observation,
//!     action: None,
//!     outcome: None,
//!     source: SourceId::new(7),
//! };
//!
//! let mut memory = MemoryV0::new();
//! let first = memory.insert_episode(episode.clone())?;
//! let second = memory.insert_episode(episode)?;
//! memory.set_relevance(
//!     first,
//!     second,
//!     InfluenceWeight::from_ppm(1_000_000)?,
//! )?;
//! memory.stimulate(first, Activation::ONE)?;
//! memory.step();
//!
//! assert_eq!(memory.activation(second), Some(Activation::from_ppm(400_000)?));
//! # Ok(())
//! # }
//! ```

mod memory;
mod model;

pub use memory::{MemoryV0, PROPAGATION_GAIN_PPM, RETENTION_PPM, RecallHit, RelevanceEdge, SCALE};
pub use model::{
    Activation, AtomId, EpisodeAtom, EpisodeDraft, GraphError, InfluenceWeight, MemoryError,
    ModelError, PredicateId, SourceId, Statement, TermId, TimestampMs, ValueError,
};
