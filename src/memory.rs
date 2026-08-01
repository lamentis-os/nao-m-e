use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::model::{
    Activation, AtomId, EpisodeAtom, EpisodeDraft, GraphError, InfluenceWeight, MemoryError,
};

/// Fixed-point unit representing one.
pub const SCALE: u32 = 1_000_000;

/// Activation retained by an atom during one step.
pub const RETENTION_PPM: u32 = 500_000;

/// Activation made available for propagation during one step.
pub const PROPAGATION_GAIN_PPM: u32 = 400_000;

static NEXT_MEMORY_NAMESPACE: AtomicU64 = AtomicU64::new(0);

/// One directed positive relevance edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelevanceEdge {
    from: AtomId,
    to: AtomId,
    weight: InfluenceWeight,
}

impl RelevanceEdge {
    /// Returns the source atom.
    #[must_use]
    pub const fn from(self) -> AtomId {
        self.from
    }

    /// Returns the target atom.
    #[must_use]
    pub const fn to(self) -> AtomId {
        self.to
    }

    /// Returns the positive relevance weight.
    #[must_use]
    pub const fn weight(self) -> InfluenceWeight {
        self.weight
    }
}

/// One ranked active atom.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecallHit {
    /// Identifier of the active atom.
    pub atom_id: AtomId,
    /// Current fixed-point activation.
    pub activation: Activation,
}

/// In-memory V0 atom store and deterministic relevance dynamics.
#[derive(Debug)]
pub struct MemoryV0 {
    memory_namespace: u64,
    next_atom_id: Option<u64>,
    atoms: BTreeMap<AtomId, EpisodeAtom>,
    activations: BTreeMap<AtomId, Activation>,
    outgoing: BTreeMap<AtomId, BTreeMap<AtomId, InfluenceWeight>>,
}

impl MemoryV0 {
    /// Creates an empty memory.
    ///
    /// # Panics
    ///
    /// Panics only if all process-local memory namespaces have been exhausted.
    #[must_use]
    pub fn new() -> Self {
        let memory_namespace = NEXT_MEMORY_NAMESPACE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .unwrap_or_else(|_| panic!("memory namespace space is exhausted"));

        Self {
            memory_namespace,
            next_atom_id: Some(0),
            atoms: BTreeMap::new(),
            activations: BTreeMap::new(),
            outgoing: BTreeMap::new(),
        }
    }

    /// Inserts one immutable episode and returns its new monotonic identifier.
    pub fn insert_episode(&mut self, draft: EpisodeDraft) -> Result<AtomId, MemoryError> {
        let local_id = self.next_atom_id.ok_or(MemoryError::IdExhausted)?;
        let id = AtomId::from_raw(self.memory_namespace, local_id);
        let atom = EpisodeAtom::from_draft(id, draft);

        self.next_atom_id = local_id.checked_add(1);
        self.atoms.insert(id, atom);
        self.activations.insert(id, Activation::ZERO);
        Ok(id)
    }

    /// Returns an immutable episode by identifier.
    #[must_use]
    pub fn episode(&self, id: AtomId) -> Option<&EpisodeAtom> {
        self.atoms.get(&id)
    }

    /// Iterates over all episodes in ascending identifier order.
    pub fn episodes(
        &self,
    ) -> impl ExactSizeIterator<Item = &EpisodeAtom> + DoubleEndedIterator + '_ {
        self.atoms.values()
    }

    /// Sets or replaces one relevance edge.
    ///
    /// The update is rejected without mutation if either endpoint is unknown,
    /// the edge is a self-edge, or the source outgoing budget would exceed one.
    pub fn set_relevance(
        &mut self,
        from: AtomId,
        to: AtomId,
        weight: InfluenceWeight,
    ) -> Result<Option<InfluenceWeight>, GraphError> {
        self.require_atom(from)?;
        self.require_atom(to)?;
        if from == to {
            return Err(GraphError::SelfEdge(from));
        }

        let existing = self
            .outgoing
            .get(&from)
            .and_then(|targets| targets.get(&to))
            .copied();
        let current_total: u64 = self
            .outgoing
            .get(&from)
            .into_iter()
            .flat_map(|targets| targets.values())
            .map(|value| u64::from(value.as_ppm()))
            .sum();
        let attempted_total = current_total - existing.map_or(0, |value| u64::from(value.as_ppm()))
            + u64::from(weight.as_ppm());

        if attempted_total > u64::from(SCALE) {
            return Err(GraphError::OutgoingWeightBudgetExceeded {
                from,
                attempted_ppm: attempted_total,
            });
        }

        self.outgoing.entry(from).or_default().insert(to, weight);
        Ok(existing)
    }

    /// Removes one relevance edge if present.
    pub fn remove_relevance(
        &mut self,
        from: AtomId,
        to: AtomId,
    ) -> Result<Option<InfluenceWeight>, GraphError> {
        self.require_atom(from)?;
        self.require_atom(to)?;
        if from == to {
            return Err(GraphError::SelfEdge(from));
        }

        let removed = self
            .outgoing
            .get_mut(&from)
            .and_then(|targets| targets.remove(&to));
        if self.outgoing.get(&from).is_some_and(BTreeMap::is_empty) {
            self.outgoing.remove(&from);
        }
        Ok(removed)
    }

    /// Returns one relevance weight, or none when the edge is absent.
    #[must_use]
    pub fn relevance(&self, from: AtomId, to: AtomId) -> Option<InfluenceWeight> {
        self.outgoing
            .get(&from)
            .and_then(|targets| targets.get(&to))
            .copied()
    }

    /// Iterates over relevance edges in ascending source and target order.
    pub fn relevance_edges(&self) -> impl Iterator<Item = RelevanceEdge> + '_ {
        self.outgoing.iter().flat_map(|(&from, targets)| {
            targets
                .iter()
                .map(move |(&to, &weight)| RelevanceEdge { from, to, weight })
        })
    }

    /// Adds an external stimulus, saturating at full activation.
    pub fn stimulate(&mut self, id: AtomId, amount: Activation) -> Result<Activation, GraphError> {
        self.require_atom(id)?;
        let current = self
            .activations
            .get_mut(&id)
            .expect("every stored atom has runtime activation");
        let next = current.as_ppm().saturating_add(amount.as_ppm()).min(SCALE);
        *current = Activation::from_clamped_ppm(next);
        Ok(*current)
    }

    /// Returns current activation for a known atom.
    #[must_use]
    pub fn activation(&self, id: AtomId) -> Option<Activation> {
        self.activations.get(&id).copied()
    }

    /// Advances the complete activation vector by one synchronous step.
    pub fn step(&mut self) {
        let previous = self.activations.clone();
        let mut next = BTreeMap::new();

        for (&id, &activation) in &previous {
            let retained =
                u128::from(activation.as_ppm()) * u128::from(RETENTION_PPM) / u128::from(SCALE);
            next.insert(
                id,
                Activation::from_clamped_ppm(
                    u32::try_from(retained).expect("retained activation is bounded by SCALE"),
                ),
            );
        }

        let scale_squared = u128::from(SCALE) * u128::from(SCALE);
        for (&from, targets) in &self.outgoing {
            let source = u128::from(
                previous
                    .get(&from)
                    .expect("edge source exists in activation map")
                    .as_ppm(),
            );
            for (&to, &weight) in targets {
                let contribution =
                    source * u128::from(weight.as_ppm()) * u128::from(PROPAGATION_GAIN_PPM)
                        / scale_squared;
                let target = next
                    .get_mut(&to)
                    .expect("edge target exists in activation map");
                let combined = u128::from(target.as_ppm()) + contribution;
                let clamped = combined.min(u128::from(SCALE));
                *target = Activation::from_clamped_ppm(
                    u32::try_from(clamped).expect("clamped activation is bounded by SCALE"),
                );
            }
        }

        self.activations = next;
    }

    /// Returns up to limit active atoms ordered by activation then identifier.
    #[must_use]
    pub fn top_k(&self, limit: usize) -> Vec<RecallHit> {
        let mut hits: Vec<_> = self
            .activations
            .iter()
            .filter_map(|(&atom_id, &activation)| {
                (activation != Activation::ZERO).then_some(RecallHit {
                    atom_id,
                    activation,
                })
            })
            .collect();

        hits.sort_unstable_by(|left, right| {
            right
                .activation
                .cmp(&left.activation)
                .then_with(|| left.atom_id.cmp(&right.atom_id))
        });
        hits.truncate(limit);
        hits
    }

    /// Resets all activation to zero without changing atoms or edges.
    pub fn reset_activations(&mut self) {
        for activation in self.activations.values_mut() {
            *activation = Activation::ZERO;
        }
    }

    fn require_atom(&self, id: AtomId) -> Result<(), GraphError> {
        if self.atoms.contains_key(&id) {
            Ok(())
        } else {
            Err(GraphError::UnknownAtom(id))
        }
    }
}

impl Default for MemoryV0 {
    fn default() -> Self {
        Self::new()
    }
}
