use std::cmp::{Ordering as ComparisonOrdering, Reverse};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;
use std::sync::OnceLock;

use crate::model::{
    Activation, AtomId, EpisodeAtom, EpisodeDraft, FeedbackTrace, GraphError, MemoryError,
    MemoryId, PredicateId, Statement, TermId,
};
use crate::parameters::{
    FEEDBACK_HISTORY_CAPACITY, FEEDBACK_PRIOR_MASS, LEARNED_GAIN_PPM, MAX_FEEDBACK_TARGETS,
    STRUCTURAL_GAIN_PPM,
};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RankedRecallHit(RecallHit);

impl Ord for RankedRecallHit {
    fn cmp(&self, other: &Self) -> ComparisonOrdering {
        self.0
            .activation
            .cmp(&other.0.activation)
            .then_with(|| other.0.atom_id.cmp(&self.0.atom_id))
    }
}

impl PartialOrd for RankedRecallHit {
    fn partial_cmp(&self, other: &Self) -> Option<ComparisonOrdering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum StatementRole {
    Context,
    Observation,
    Action,
    Outcome,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Cue {
    Predicate(PredicateId),
    Term(TermId),
    RolePredicate {
        role: StatementRole,
        predicate: PredicateId,
    },
    RoleArgument {
        role: StatementRole,
        predicate: PredicateId,
        position: u64,
        term: TermId,
    },
}

impl Cue {
    const fn weight(self) -> u64 {
        match self {
            Self::Predicate(_) | Self::Term(_) => 1,
            Self::RolePredicate { .. } => 2,
            Self::RoleArgument { .. } => 4,
        }
    }
}

fn episode_cues(episode: &EpisodeAtom) -> Vec<Cue> {
    let mut cues = Vec::new();
    for statement in episode.context() {
        append_statement_cues(&mut cues, StatementRole::Context, statement);
    }
    append_statement_cues(&mut cues, StatementRole::Observation, episode.observation());
    if let Some(statement) = episode.action() {
        append_statement_cues(&mut cues, StatementRole::Action, statement);
    }
    if let Some(statement) = episode.outcome() {
        append_statement_cues(&mut cues, StatementRole::Outcome, statement);
    }
    cues.sort_unstable();
    cues.dedup();
    cues
}

fn append_statement_cues(cues: &mut Vec<Cue>, role: StatementRole, statement: &Statement) {
    let predicate = statement.predicate();
    cues.push(Cue::Predicate(predicate));
    cues.push(Cue::RolePredicate { role, predicate });
    for (position, term) in statement.arguments().iter().copied().enumerate() {
        cues.push(Cue::Term(term));
        cues.push(Cue::RoleArgument {
            role,
            predicate,
            position: u64::try_from(position)
                .expect("a statement argument position fits in u64 on supported platforms"),
            term,
        });
    }
}

fn cue_weight_total(cues: &[Cue]) -> u64 {
    cues.iter().copied().fold(0_u64, |total, cue| {
        total
            .checked_add(cue.weight())
            .expect("an in-memory episode's cue weight fits in u64")
    })
}

enum PostingList {
    One(usize),
    Many(Vec<usize>),
}

impl PostingList {
    fn push(&mut self, atom_index: usize) {
        match self {
            Self::One(first) => {
                debug_assert!(*first < atom_index);
                let mut postings = Vec::with_capacity(4);
                postings.extend([*first, atom_index]);
                *self = Self::Many(postings);
            }
            Self::Many(postings) => {
                debug_assert!(postings.last().is_some_and(|last| *last < atom_index));
                postings.push(atom_index);
            }
        }
    }

    fn as_slice(&self) -> &[usize] {
        match self {
            Self::One(atom_index) => std::slice::from_ref(atom_index),
            Self::Many(postings) => postings,
        }
    }
}

struct CueIndex {
    postings: BTreeMap<Cue, PostingList>,
    weight_totals: Vec<u64>,
}

impl CueIndex {
    fn from_atoms(atoms: &[EpisodeAtom]) -> Self {
        let mut index = Self {
            postings: BTreeMap::new(),
            weight_totals: Vec::with_capacity(atoms.len()),
        };
        for (atom_index, atom) in atoms.iter().enumerate() {
            index.insert(atom_index, atom);
        }
        index
    }

    fn insert(&mut self, atom_index: usize, atom: &EpisodeAtom) {
        debug_assert_eq!(atom_index, self.weight_totals.len());
        let cues = episode_cues(atom);
        self.weight_totals.push(cue_weight_total(&cues));
        for cue in cues {
            match self.postings.entry(cue) {
                Entry::Vacant(entry) => {
                    entry.insert(PostingList::One(atom_index));
                }
                Entry::Occupied(mut entry) => entry.get_mut().push(atom_index),
            }
        }
    }
}

/// Append-only atom storage with derived cue postings and sparse feedback.
///
/// Each memory ID must name one exclusively written logical memory. Reopening
/// requires reconstructing its complete atom sequence before appending. Atom
/// identifiers from another memory are rejected rather than aliased.
pub struct MemoryV0 {
    memory_id: MemoryId,
    atoms: Vec<EpisodeAtom>,
    cue_index: OnceLock<CueIndex>,
    outgoing: BTreeMap<usize, BTreeMap<usize, FeedbackTrace>>,
}

impl MemoryV0 {
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

    /// Ranks cue-derived similarity and direct learned feedback from `source`.
    ///
    /// Cue postings generate structural candidates from the source episode's
    /// symbolic content. Direct outgoing feedback can add candidates and a
    /// signed learned score. Both contributions use deterministic fixed-point
    /// arithmetic. The query may initialize a private derived cue cache, but
    /// leaves episode and feedback state unchanged. Zero scores and the source
    /// itself are omitted; equal scores are ordered by ascending atom identifier.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::UnknownAtom`] if `source` does not belong to this
    /// memory. The source is validated even when `limit` is zero.
    pub fn recall_from(&self, source: AtomId, limit: usize) -> Result<Vec<RecallHit>, GraphError> {
        let source_index = self.require_atom(source)?;
        if limit == 0 {
            return Ok(Vec::new());
        }

        let cue_index = self
            .cue_index
            .get_or_init(|| CueIndex::from_atoms(&self.atoms));
        let source_cues = episode_cues(&self.atoms[source_index]);
        let mut candidates = BTreeMap::<usize, u64>::new();
        for cue in source_cues {
            let Some(postings) = cue_index.postings.get(&cue) else {
                continue;
            };
            for &target_index in postings.as_slice() {
                if target_index == source_index {
                    continue;
                }
                let shared_weight = candidates.entry(target_index).or_default();
                *shared_weight = shared_weight
                    .checked_add(cue.weight())
                    .expect("shared cue weight is bounded by each episode's cue weight");
            }
        }

        let outgoing = self.outgoing.get(&source_index);
        if let Some(outgoing) = outgoing {
            for &target_index in outgoing.keys() {
                candidates.entry(target_index).or_default();
            }
        }

        let candidate_bound = candidates.len();
        let source_cue_weight = cue_index.weight_totals[source_index];
        let hits = candidates
            .into_iter()
            .filter_map(|(target_index, shared_weight)| {
                let structural_score = Self::structural_score(
                    shared_weight,
                    source_cue_weight,
                    cue_index.weight_totals[target_index],
                );
                let learned_score = outgoing
                    .and_then(|outgoing| outgoing.get(&target_index))
                    .copied()
                    .map_or(0, Self::project_feedback);
                let score = (i64::from(structural_score) + i64::from(learned_score)).clamp(
                    0,
                    i64::from(STRUCTURAL_GAIN_PPM) + i64::from(LEARNED_GAIN_PPM),
                );
                let score = u32::try_from(score).expect("clamped recall score fits in u32");
                let activation = Activation::from_ppm(score)
                    .expect("combined recall activation is bounded by SCALE");
                (activation != Activation::ZERO).then_some(RecallHit {
                    atom_id: self.atoms[target_index].id(),
                    activation,
                })
            });

        Ok(Self::rank_hits(hits, candidate_bound, limit))
    }

    fn structural_score(shared_weight: u64, source_weight: u64, target_weight: u64) -> u32 {
        if shared_weight == 0 {
            return 0;
        }
        debug_assert!(shared_weight <= source_weight);
        debug_assert!(shared_weight <= target_weight);
        let union_weight =
            u128::from(source_weight) + u128::from(target_weight) - u128::from(shared_weight);
        let score = u128::from(shared_weight) * u128::from(STRUCTURAL_GAIN_PPM) / union_weight;
        u32::try_from(score).expect("structural recall score is bounded by its gain")
    }

    fn project_feedback(trace: FeedbackTrace) -> i32 {
        let sample_count = i64::from(trace.sample_count());
        let helpful_count = i64::from(trace.helpful_count());
        let balance = helpful_count * 2 - sample_count;
        let numerator = i64::from(LEARNED_GAIN_PPM)
            * balance
            * i64::from(FEEDBACK_HISTORY_CAPACITY + FEEDBACK_PRIOR_MASS);
        let denominator = i64::from(FEEDBACK_HISTORY_CAPACITY)
            * i64::from(trace.sample_count() + FEEDBACK_PRIOR_MASS);
        i32::try_from(numerator / denominator)
            .expect("projected feedback is bounded by its signed gain")
    }

    fn rank_hits(
        hits: impl Iterator<Item = RecallHit>,
        candidate_bound: usize,
        limit: usize,
    ) -> Vec<RecallHit> {
        if limit >= candidate_bound {
            let mut hits: Vec<_> = hits.collect();
            Self::sort_hits(&mut hits);
            return hits;
        }

        let mut best = BinaryHeap::with_capacity(limit);
        for hit in hits {
            let ranked = RankedRecallHit(hit);
            if best.len() < limit {
                best.push(Reverse(ranked));
                continue;
            }

            let worst = best.peek().expect("non-empty heap at its capacity").0;
            if ranked > worst {
                best.pop();
                best.push(Reverse(ranked));
            }
        }

        let mut hits: Vec<_> = best.into_iter().map(|Reverse(ranked)| ranked.0).collect();
        Self::sort_hits(&mut hits);
        hits
    }

    fn sort_hits(hits: &mut [RecallHit]) {
        hits.sort_unstable_by_key(|hit| Reverse(RankedRecallHit(*hit)));
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

impl fmt::Debug for MemoryV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryV0")
            .field("memory_id", &self.memory_id)
            .field("atoms", &self.atoms)
            .field("outgoing", &self.outgoing)
            .finish_non_exhaustive()
    }
}
