use std::cmp::{Ordering as ComparisonOrdering, Reverse};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::model::{
    Activation, AtomId, EpisodeAtom, EpisodeDraft, GraphError, InfluenceWeight, MemoryError,
};
use crate::parameters::{PROPAGATION_GAIN_PPM, RETENTION_PPM, SCALE, SCALE_CUBED, SCALE_SQUARED};

static NEXT_MEMORY_NAMESPACE: AtomicU64 = AtomicU64::new(0);

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
/// Atom identifiers belong to the memory instance that created them. Passing an
/// identifier from another instance is rejected instead of aliasing a local atom.
pub struct MemoryV0 {
    memory_namespace: u64,
    atoms: Vec<EpisodeAtom>,
    activations: Vec<Activation>,
    transition_numerators: Vec<u64>,
    outgoing: BTreeMap<usize, OutgoingRelevance>,
}

impl MemoryV0 {
    /// Creates an empty memory with a new process-local namespace.
    ///
    /// # Panics
    ///
    /// Panics if the process-local namespace counter is exhausted.
    #[must_use]
    pub fn new() -> Self {
        let memory_namespace = NEXT_MEMORY_NAMESPACE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .unwrap_or_else(|_| panic!("memory namespace space is exhausted"));

        Self {
            memory_namespace,
            atoms: Vec::new(),
            activations: Vec::new(),
            transition_numerators: Vec::new(),
            outgoing: BTreeMap::new(),
        }
    }

    /// Canonicalizes the draft context and appends an immutable episode.
    pub fn insert_episode(&mut self, draft: EpisodeDraft) -> Result<AtomId, MemoryError> {
        self.debug_assert_parallel_storage();
        let local_id = u64::try_from(self.atoms.len()).map_err(|_| MemoryError::IdExhausted)?;
        let id = AtomId::from_raw(self.memory_namespace, local_id);
        let atom = EpisodeAtom::from_draft(id, draft);

        self.atoms.push(atom);
        self.activations.push(Activation::ZERO);
        self.transition_numerators.push(0);
        self.debug_assert_parallel_storage();
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

        match self.outgoing.entry(from_index) {
            Entry::Vacant(entry) => {
                let mut outgoing = OutgoingRelevance {
                    total_ppm: weight.as_ppm(),
                    ..OutgoingRelevance::default()
                };
                outgoing.targets.insert(to_index, weight);
                entry.insert(outgoing);
                Ok(None)
            }
            Entry::Occupied(mut entry) => {
                let outgoing = entry.get_mut();
                let existing = outgoing.targets.get(&to_index).copied();
                let attempted_total = u64::from(outgoing.total_ppm)
                    - existing.map_or(0, |value| u64::from(value.as_ppm()))
                    + u64::from(weight.as_ppm());

                if attempted_total > u64::from(SCALE) {
                    return Err(GraphError::OutgoingWeightBudgetExceeded {
                        from,
                        attempted_ppm: attempted_total,
                    });
                }

                outgoing.targets.insert(to_index, weight);
                outgoing.total_ppm =
                    u32::try_from(attempted_total).expect("validated outgoing total fits u32");
                Ok(existing)
            }
        }
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
        self.debug_assert_parallel_storage();
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
        self.debug_assert_parallel_storage();
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

        let mut best = BinaryHeap::new();
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
        hits.sort_unstable_by(|left, right| {
            right
                .activation
                .cmp(&left.activation)
                .then_with(|| left.atom_id.cmp(&right.atom_id))
        });
    }

    fn local_index(&self, id: AtomId) -> Option<usize> {
        if id.memory_namespace() != self.memory_namespace {
            return None;
        }

        let index = usize::try_from(id.get()).ok()?;
        (index < self.atoms.len()).then_some(index)
    }

    fn require_atom(&self, id: AtomId) -> Result<usize, GraphError> {
        self.local_index(id).ok_or(GraphError::UnknownAtom(id))
    }

    fn debug_assert_parallel_storage(&self) {
        debug_assert_eq!(self.activations.len(), self.atoms.len());
        debug_assert_eq!(self.transition_numerators.len(), self.atoms.len());
    }
}

impl fmt::Debug for MemoryV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryV0")
            .field("memory_namespace", &self.memory_namespace)
            .field("atoms", &self.atoms)
            .field("activations", &self.activations)
            .field("outgoing", &self.outgoing)
            .finish_non_exhaustive()
    }
}

impl Default for MemoryV0 {
    fn default() -> Self {
        Self::new()
    }
}
