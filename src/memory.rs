mod recall;

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::fmt;
use std::sync::OnceLock;

use self::recall::CueIndex;
use crate::model::{
    Activation, AtomId, EpisodeAtom, EpisodeDraft, FeedbackTrace, GraphError, MemoryError, MemoryId,
};
use crate::parameters::MAX_FEEDBACK_TARGETS;

/// A directed bounded feedback trace between two atoms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeedbackEdge {
    from: AtomId,
    to: AtomId,
    trace: FeedbackTrace,
}

impl FeedbackEdge {
    /// Returns the episode used as the feedback source.
    #[must_use]
    pub const fn from(self) -> AtomId {
        self.from
    }

    /// Returns the episode assessed relative to the source.
    #[must_use]
    pub const fn to(self) -> AtomId {
        self.to
    }

    /// Returns the bounded binary feedback history.
    #[must_use]
    pub const fn trace(self) -> FeedbackTrace {
        self.trace
    }
}

/// A non-zero query-local activation score returned by source-conditioned recall.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecallHit {
    /// Identifier of the recalled atom.
    pub atom_id: AtomId,
    /// Query-local fixed-point activation used for ranking.
    pub activation: Activation,
}

/// Append-only atom storage with derived cue postings and sparse feedback.
///
/// Each memory ID must name one exclusively written logical memory. Reopening
/// requires reconstructing its complete atom sequence before appending. Atom
/// identifiers from another memory are rejected rather than aliased.
pub struct Memory {
    memory_id: MemoryId,
    atoms: Vec<EpisodeAtom>,
    cue_index: OnceLock<CueIndex>,
    outgoing: BTreeMap<usize, BTreeMap<usize, FeedbackTrace>>,
}

impl Memory {
    /// Creates an empty memory with the caller-owned durable identifier.
    #[must_use]
    pub fn new(memory_id: MemoryId) -> Self {
        Self {
            memory_id,
            atoms: Vec::new(),
            cue_index: OnceLock::new(),
            outgoing: BTreeMap::new(),
        }
    }

    /// Returns the durable identifier of this logical memory.
    #[must_use]
    pub const fn memory_id(&self) -> MemoryId {
        self.memory_id
    }

    /// Canonicalizes the draft context and appends the immutable episode.
    pub fn insert_episode(&mut self, draft: EpisodeDraft) -> Result<AtomId, MemoryError> {
        let sequence = u64::try_from(self.atoms.len()).map_err(|_| MemoryError::IdExhausted)?;
        let id = AtomId::from_parts(self.memory_id, sequence);
        let atom = EpisodeAtom::from_draft(id, draft);
        let atom_index = self.atoms.len();

        self.atoms.push(atom);
        if let Some(index) = self.cue_index.get_mut() {
            index.insert(atom_index, &self.atoms[atom_index]);
        }
        Ok(id)
    }

    /// Returns the episode, or `None` if the identifier does not belong here.
    #[must_use]
    pub fn episode(&self, id: AtomId) -> Option<&EpisodeAtom> {
        self.local_index(id).map(|index| &self.atoms[index])
    }

    /// Iterates over episodes in ascending local identifier order.
    pub fn episodes(
        &self,
    ) -> impl ExactSizeIterator<Item = &EpisodeAtom> + DoubleEndedIterator + '_ {
        self.atoms.iter()
    }

    /// Sets or replaces a directed feedback trace.
    ///
    /// The operation is atomic: unknown endpoints and self-edges are rejected
    /// without mutation.
    pub fn set_feedback_trace(
        &mut self,
        from: AtomId,
        to: AtomId,
        trace: FeedbackTrace,
    ) -> Result<Option<FeedbackTrace>, GraphError> {
        let from_index = self.require_atom(from)?;
        let to_index = self.require_atom(to)?;
        if from == to {
            return Err(GraphError::SelfEdge(from));
        }

        Ok(self
            .outgoing
            .entry(from_index)
            .or_default()
            .insert(to_index, trace))
    }

    /// Returns a trace, or `None` for an absent edge or unknown endpoint.
    #[must_use]
    pub fn feedback_trace(&self, from: AtomId, to: AtomId) -> Option<FeedbackTrace> {
        let from_index = self.local_index(from)?;
        let to_index = self.local_index(to)?;
        self.outgoing
            .get(&from_index)
            .and_then(|outgoing| outgoing.get(&to_index))
            .copied()
    }

    /// Iterates over edges in ascending source and target identifier order.
    pub fn feedback_edges(&self) -> impl Iterator<Item = FeedbackEdge> + '_ {
        self.outgoing
            .iter()
            .flat_map(move |(&from_index, outgoing)| {
                let from = self.atoms[from_index].id();
                outgoing
                    .iter()
                    .map(move |(&to_index, &trace)| FeedbackEdge {
                        from,
                        to: self.atoms[to_index].id(),
                        trace,
                    })
            })
    }

    /// Applies one external binary assessment to a source and recalled targets.
    ///
    /// At most [`crate::MAX_FEEDBACK_TARGETS`] entries are accepted. Targets are
    /// then treated as an unordered set: duplicates and the source atom are
    /// ignored. Every effective target receives one complete sample. Once a
    /// trace reaches [`crate::FEEDBACK_HISTORY_CAPACITY`], adding a sample drops
    /// exactly its oldest sample.
    ///
    /// All identifiers are validated before feedback is mutated. Episode
    /// content is not changed.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::UnknownAtom`] if the source or any target does not
    /// belong to this memory, or [`GraphError::FeedbackTargetLimitExceeded`] if
    /// the supplied slice is too long. Failure leaves all feedback unchanged.
    pub fn apply_feedback(
        &mut self,
        source: AtomId,
        targets: &[AtomId],
        helpful: bool,
    ) -> Result<(), GraphError> {
        let source_index = self.require_atom(source)?;
        if targets.len() > MAX_FEEDBACK_TARGETS {
            return Err(GraphError::FeedbackTargetLimitExceeded {
                count: targets.len(),
                max: MAX_FEEDBACK_TARGETS,
            });
        }
        let mut target_indices = Vec::with_capacity(targets.len());
        for &target in targets {
            target_indices.push(self.require_atom(target)?);
        }
        target_indices.sort_unstable();
        target_indices.dedup();
        target_indices.retain(|&target_index| target_index != source_index);

        if target_indices.is_empty() {
            return Ok(());
        }

        let outgoing = self.outgoing.entry(source_index).or_default();
        for target_index in target_indices {
            match outgoing.entry(target_index) {
                Entry::Vacant(entry) => {
                    entry.insert(FeedbackTrace::from_feedback(helpful));
                }
                Entry::Occupied(mut entry) => entry.get_mut().push(helpful),
            }
        }
        Ok(())
    }

    fn local_index(&self, id: AtomId) -> Option<usize> {
        if id.memory_id() != self.memory_id {
            return None;
        }

        let index = usize::try_from(id.sequence()).ok()?;
        (index < self.atoms.len()).then_some(index)
    }

    fn require_atom(&self, id: AtomId) -> Result<usize, GraphError> {
        self.local_index(id).ok_or(GraphError::UnknownAtom(id))
    }
}

impl fmt::Debug for Memory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Memory")
            .field("memory_id", &self.memory_id)
            .field("atoms", &self.atoms)
            .field("outgoing", &self.outgoing)
            .finish_non_exhaustive()
    }
}
