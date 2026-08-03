use std::cmp::{Ordering as ComparisonOrdering, Reverse};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;
use std::sync::OnceLock;

use crate::model::{
    Activation, AtomId, EpisodeAtom, EpisodeDraft, GraphError, InfluenceWeight, MemoryError,
    MemoryId, PredicateId, Statement, TermId,
};
use crate::parameters::{
    FEEDBACK_MAX_EVENT_PPM, FEEDBACK_TARGET_STEP_PPM, MAX_FEEDBACK_TARGETS, PROPAGATION_GAIN_PPM,
    SCALE, STRUCTURAL_GAIN_PPM,
};

/// A directed positive relevance edge between two atoms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelevanceEdge {
    from: AtomId,
    to: AtomId,
    weight: InfluenceWeight,
}

impl RelevanceEdge {
    /// Returns the influencing atom.
    #[must_use]
    pub const fn from(self) -> AtomId {
        self.from
    }

    /// Returns the influenced atom.
    #[must_use]
    pub const fn to(self) -> AtomId {
        self.to
    }

    /// Returns the edge weight.
    #[must_use]
    pub const fn weight(self) -> InfluenceWeight {
        self.weight
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

#[derive(Debug, Default)]
struct OutgoingRelevance {
    total_ppm: u32,
    targets: BTreeMap<usize, InfluenceWeight>,
}

/// Append-only atom storage with derived cue postings and sparse relevance.
///
/// Each memory ID must name one exclusively written logical memory. Reopening
/// requires reconstructing its complete atom sequence before appending. Atom
/// identifiers from another memory are rejected rather than aliased.
pub struct MemoryV0 {
    memory_id: MemoryId,
    atoms: Vec<EpisodeAtom>,
    cue_index: OnceLock<CueIndex>,
    outgoing: BTreeMap<usize, OutgoingRelevance>,
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

    /// Sets or replaces a directed relevance edge.
    ///
    /// The operation is atomic: unknown endpoints, self-edges, and outgoing
    /// weight totals above [`crate::SCALE`] are rejected without mutation.
    pub fn set_relevance(
        &mut self,
        from: AtomId,
        to: AtomId,
        weight: InfluenceWeight,
    ) -> Result<Option<InfluenceWeight>, GraphError> {
        let from_index = self.require_atom(from)?;
        let to_index = self.require_atom(to)?;
        if from == to {
            return Err(GraphError::SelfEdge(from));
        }

        let outgoing = self.outgoing.entry(from_index).or_default();
        let existing = outgoing.targets.get(&to_index).copied();
        let attempted_total =
            outgoing.total_ppm - existing.map_or(0, InfluenceWeight::as_ppm) + weight.as_ppm();

        if attempted_total > SCALE {
            return Err(GraphError::OutgoingWeightBudgetExceeded {
                from,
                attempted_ppm: u64::from(attempted_total),
            });
        }

        outgoing.targets.insert(to_index, weight);
        outgoing.total_ppm = attempted_total;
        Ok(existing)
    }

    /// Removes an edge between known, distinct endpoints if it exists.
    pub fn remove_relevance(
        &mut self,
        from: AtomId,
        to: AtomId,
    ) -> Result<Option<InfluenceWeight>, GraphError> {
        let from_index = self.require_atom(from)?;
        let to_index = self.require_atom(to)?;
        if from == to {
            return Err(GraphError::SelfEdge(from));
        }

        match self.outgoing.entry(from_index) {
            Entry::Vacant(_) => Ok(None),
            Entry::Occupied(mut entry) => {
                let outgoing = entry.get_mut();
                let removed = outgoing.targets.remove(&to_index);
                if let Some(weight) = removed {
                    outgoing.total_ppm -= weight.as_ppm();
                }
                let is_empty = outgoing.targets.is_empty();
                if is_empty {
                    entry.remove();
                }
                Ok(removed)
            }
        }
    }

    /// Returns the edge weight, or `None` for an absent edge or unknown endpoint.
    #[must_use]
    pub fn relevance(&self, from: AtomId, to: AtomId) -> Option<InfluenceWeight> {
        let from_index = self.local_index(from)?;
        let to_index = self.local_index(to)?;
        self.outgoing
            .get(&from_index)
            .and_then(|outgoing| outgoing.targets.get(&to_index))
            .copied()
    }

    /// Iterates over edges in ascending source and target identifier order.
    pub fn relevance_edges(&self) -> impl Iterator<Item = RelevanceEdge> + '_ {
        self.outgoing
            .iter()
            .flat_map(move |(&from_index, outgoing)| {
                let from = self.atoms[from_index].id();
                outgoing
                    .targets
                    .iter()
                    .map(move |(&to_index, &weight)| RelevanceEdge {
                        from,
                        to: self.atoms[to_index].id(),
                        weight,
                    })
            })
    }

    /// Applies one external binary assessment to a source and recalled targets.
    ///
    /// At most [`crate::MAX_FEEDBACK_TARGETS`] entries are accepted. Targets are
    /// then treated as an unordered set: duplicates and the source atom are
    /// ignored. Each effective target changes by at most
    /// [`crate::FEEDBACK_TARGET_STEP_PPM`], while their aggregate direct change
    /// is bounded by [`crate::FEEDBACK_MAX_EVENT_PPM`]. Positive feedback uses
    /// free outgoing budget first; if more is needed, only non-target edges are
    /// proportionally reduced. Integer remainders are carried across non-target
    /// edges in ascending identifier order, so their aggregate reduction exactly
    /// funds the award. Negative feedback removes edges that reach zero and does
    /// not redistribute weight.
    ///
    /// All identifiers are validated before relevance is mutated. Episode
    /// content is not changed.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::UnknownAtom`] if the source or any target does not
    /// belong to this memory, or [`GraphError::FeedbackTargetLimitExceeded`] if
    /// the supplied slice is too long. Failure leaves all relevance unchanged.
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

        let target_count = u64::try_from(target_indices.len())
            .expect("target count fits in u64 on supported platforms");
        let base_per_target = u64::from(FEEDBACK_TARGET_STEP_PPM)
            .min(u64::from(FEEDBACK_MAX_EVENT_PPM) / target_count);
        let current_outgoing = self.outgoing.get(&source_index);
        let current_total = current_outgoing.map_or(0, |outgoing| u64::from(outgoing.total_ppm));

        if helpful {
            let target_total = target_indices.iter().fold(0_u64, |total, target_index| {
                total
                    + current_outgoing
                        .and_then(|outgoing| outgoing.targets.get(target_index))
                        .map_or(0, |weight| u64::from(weight.as_ppm()))
            });
            let per_target = base_per_target.min((u64::from(SCALE) - target_total) / target_count);
            if per_target == 0 {
                return Ok(());
            }

            let total_award = per_target * target_count;
            let free_budget = u64::from(SCALE) - current_total;
            let funding_needed = total_award.saturating_sub(free_budget);
            let outgoing = self.outgoing.entry(source_index).or_default();
            if funding_needed != 0 {
                let non_target_total = current_total - target_total;
                let mut remainder = 0_u64;
                let mut distributed_reduction = 0_u64;
                let mut target_cursor = 0;
                outgoing.targets.retain(|target_index, weight| {
                    // `BTreeMap::retain` and `target_indices` both visit ascending keys.
                    while target_indices
                        .get(target_cursor)
                        .is_some_and(|target| target < target_index)
                    {
                        target_cursor += 1;
                    }
                    if target_indices.get(target_cursor) == Some(target_index) {
                        target_cursor += 1;
                        return true;
                    }

                    let numerator = u64::from(weight.as_ppm()) * funding_needed + remainder;
                    let reduction = numerator / non_target_total;
                    remainder = numerator % non_target_total;
                    distributed_reduction += reduction;
                    let updated = u64::from(weight.as_ppm()) - reduction;
                    if updated == 0 {
                        return false;
                    }
                    *weight = InfluenceWeight::from_ppm(
                        u32::try_from(updated).expect("scaled relevance is bounded by SCALE"),
                    )
                    .expect("scaled non-zero relevance is valid");
                    true
                });
                debug_assert_eq!(distributed_reduction, funding_needed);
                debug_assert_eq!(remainder, 0);
            }

            for target_index in target_indices {
                let existing = outgoing
                    .targets
                    .get(&target_index)
                    .map_or(0, |weight| u64::from(weight.as_ppm()));
                let updated = existing + per_target;
                outgoing.targets.insert(
                    target_index,
                    InfluenceWeight::from_ppm(
                        u32::try_from(updated).expect("feedback relevance is bounded by SCALE"),
                    )
                    .expect("positive feedback creates non-zero relevance"),
                );
            }
            let updated_total = current_total + total_award - funding_needed;
            outgoing.total_ppm =
                u32::try_from(updated_total).expect("outgoing relevance is bounded by SCALE");
        } else {
            let per_target =
                u32::try_from(base_per_target).expect("feedback step is bounded by SCALE");
            let remove_source = {
                let Some(outgoing) = self.outgoing.get_mut(&source_index) else {
                    return Ok(());
                };
                let mut removed_total = 0_u32;
                for target_index in target_indices {
                    if let Entry::Occupied(mut entry) = outgoing.targets.entry(target_index) {
                        let previous = entry.get().as_ppm();
                        let updated = previous.saturating_sub(per_target);
                        removed_total += previous - updated;
                        if updated == 0 {
                            entry.remove();
                        } else {
                            *entry.get_mut() = InfluenceWeight::from_ppm(updated)
                                .expect("remaining relevance is non-zero and bounded");
                        }
                    }
                }
                outgoing.total_ppm -= removed_total;
                outgoing.targets.is_empty()
            };
            if remove_source {
                self.outgoing.remove(&source_index);
            }
        }
        Ok(())
    }

    /// Ranks cue-derived similarity and direct learned relevance from `source`.
    ///
    /// Cue postings generate structural candidates from the source episode's
    /// symbolic content. Direct outgoing relevance can add candidates and a
    /// learned score. Both contributions use deterministic fixed-point
    /// arithmetic. The query may initialize a private derived cue cache, but
    /// leaves episode and relevance state unchanged. Zero scores and the source
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
            for &target_index in outgoing.targets.keys() {
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
                    .and_then(|outgoing| outgoing.targets.get(&target_index))
                    .copied()
                    .map_or(0, Self::project_relevance);
                let score = structural_score + learned_score;
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

    fn project_relevance(weight: InfluenceWeight) -> u32 {
        let score = u64::from(weight.as_ppm()) * u64::from(PROPAGATION_GAIN_PPM) / u64::from(SCALE);
        u32::try_from(score).expect("projected relevance is bounded by its gain")
    }

    fn rank_hits(
        hits: impl Iterator<Item = RecallHit>,
        candidate_bound: usize,
        limit: usize,
    ) -> Vec<RecallHit> {
        if limit == 0 {
            return Vec::new();
        }

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
