use std::cmp::{Ordering as ComparisonOrdering, Reverse};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;

use crate::model::{
    Activation, AtomId, EpisodeAtom, EpisodeDraft, GraphError, InfluenceWeight, MemoryError,
    MemoryId,
};
use crate::parameters::{
    FEEDBACK_MAX_EVENT_PPM, FEEDBACK_TARGET_STEP_PPM, MAX_FEEDBACK_TARGETS, PROPAGATION_GAIN_PPM,
    RETENTION_PPM, SCALE, SCALE_CUBED, SCALE_SQUARED,
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

/// A non-zero activation returned by recall ranking.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecallHit {
    /// Identifier of the active atom.
    pub atom_id: AtomId,
    /// Current fixed-point activation.
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

#[derive(Debug, Default)]
struct OutgoingRelevance {
    total_ppm: u32,
    targets: BTreeMap<usize, InfluenceWeight>,
}

/// Append-only atom storage with mutable activation and sparse relevance edges.
///
/// Each memory ID must name one exclusively written logical memory. Reopening
/// requires reconstructing its complete atom sequence before appending. Atom
/// identifiers from another memory are rejected rather than aliased.
pub struct MemoryV0 {
    memory_id: MemoryId,
    atoms: Vec<EpisodeAtom>,
    activations: Vec<Activation>,
    transition_numerators: Vec<u64>,
    outgoing: BTreeMap<usize, OutgoingRelevance>,
}

impl MemoryV0 {
    /// Creates an empty memory with the caller-owned durable identifier.
    #[must_use]
    pub fn new(memory_id: MemoryId) -> Self {
        Self {
            memory_id,
            atoms: Vec::new(),
            activations: Vec::new(),
            transition_numerators: Vec::new(),
            outgoing: BTreeMap::new(),
        }
    }

    /// Returns the durable identifier of this logical memory.
    #[must_use]
    pub const fn memory_id(&self) -> MemoryId {
        self.memory_id
    }

    /// Canonicalizes the draft context and appends an immutable episode.
    pub fn insert_episode(&mut self, draft: EpisodeDraft) -> Result<AtomId, MemoryError> {
        self.debug_assert_storage();
        let sequence = u64::try_from(self.atoms.len()).map_err(|_| MemoryError::IdExhausted)?;
        let id = AtomId::from_parts(self.memory_id, sequence);
        let atom = EpisodeAtom::from_draft(id, draft);

        self.atoms.push(atom);
        self.activations.push(Activation::ZERO);
        self.debug_assert_storage();
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
    /// content and activation are not changed.
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
                outgoing.targets.retain(|target_index, weight| {
                    if target_indices.binary_search(target_index).is_ok() {
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

    /// Adds external activation, saturating at [`Activation::ONE`].
    pub fn stimulate(&mut self, id: AtomId, amount: Activation) -> Result<Activation, GraphError> {
        let index = self.require_atom(id)?;
        let current = &mut self.activations[index];
        let next = current.as_ppm().saturating_add(amount.as_ppm()).min(SCALE);
        *current = Activation::from_clamped_ppm(next);
        Ok(*current)
    }

    /// Returns activation, or `None` if the identifier does not belong here.
    #[must_use]
    pub fn activation(&self, id: AtomId) -> Option<Activation> {
        self.local_index(id).map(|index| self.activations[index])
    }

    /// Advances every activation by one synchronous logical step.
    ///
    /// Retention and incoming influence share one rounding operation per atom.
    pub fn step(&mut self) {
        self.debug_assert_storage();
        self.transition_numerators.resize(self.activations.len(), 0);
        for (index, &activation) in self.activations.iter().enumerate() {
            self.transition_numerators[index] =
                u64::from(activation.as_ppm()) * u64::from(RETENTION_PPM) * u64::from(SCALE);
        }

        for (&from_index, outgoing) in &self.outgoing {
            let source = self.activations[from_index].as_ppm();
            if source == 0 {
                continue;
            }

            for (&to_index, &weight) in &outgoing.targets {
                let numerator = u64::from(source)
                    * u64::from(weight.as_ppm())
                    * u64::from(PROPAGATION_GAIN_PPM);
                self.transition_numerators[to_index] = self.transition_numerators[to_index]
                    .saturating_add(numerator)
                    .min(SCALE_CUBED);
            }
        }

        for (index, &transition_numerator) in self.transition_numerators.iter().enumerate() {
            let next = transition_numerator / SCALE_SQUARED;
            self.activations[index] = Activation::from_clamped_ppm(
                u32::try_from(next).expect("transition activation is bounded by SCALE"),
            );
        }
        self.debug_assert_storage();
    }

    /// Returns at most `limit` non-zero activations, highest first.
    ///
    /// Equal activations are ordered by ascending atom identifier.
    #[must_use]
    pub fn top_k(&self, limit: usize) -> Vec<RecallHit> {
        if limit == 0 {
            return Vec::new();
        }

        if limit >= self.activations.len() {
            let mut hits: Vec<_> = self.active_hits().collect();
            Self::sort_hits(&mut hits);
            return hits;
        }

        let mut best = BinaryHeap::with_capacity(limit);
        for hit in self.active_hits() {
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

    /// Resets activation without changing atoms or relevance edges.
    pub fn reset_activations(&mut self) {
        self.activations.fill(Activation::ZERO);
    }

    fn active_hits(&self) -> impl Iterator<Item = RecallHit> + '_ {
        self.activations
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, activation)| {
                (activation != Activation::ZERO).then_some(RecallHit {
                    atom_id: self.atoms[index].id(),
                    activation,
                })
            })
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

    fn debug_assert_storage(&self) {
        debug_assert_eq!(self.activations.len(), self.atoms.len());
        debug_assert!(self.transition_numerators.len() <= self.atoms.len());
    }
}

impl fmt::Debug for MemoryV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryV0")
            .field("memory_id", &self.memory_id)
            .field("atoms", &self.atoms)
            .field("activations", &self.activations)
            .field("outgoing", &self.outgoing)
            .finish_non_exhaustive()
    }
}
