use std::cmp::{Ordering as ComparisonOrdering, Reverse};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BinaryHeap};

use super::{Memory, RecallHit};
use crate::model::{Activation, AtomId, EpisodeAtom, FeedbackTrace, GraphError, SymbolId};
use crate::parameters::{
    FEEDBACK_HISTORY_CAPACITY, FEEDBACK_PRIOR_MASS, LEARNED_GAIN_PPM, STRUCTURAL_GAIN_PPM,
};

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
enum Cue {
    Key(SymbolId),
    Value(SymbolId),
    KeyValue { key: SymbolId, value: SymbolId },
}

fn episode_cues(episode: &EpisodeAtom) -> Vec<Cue> {
    let value_count = episode
        .attributes()
        .iter()
        .map(|attribute| attribute.values().len())
        .sum::<usize>();
    let capacity = episode
        .attributes()
        .len()
        .saturating_add(value_count.saturating_mul(2));
    let mut cues = Vec::with_capacity(capacity);

    for attribute in episode.attributes() {
        cues.push(Cue::Key(attribute.key()));
    }
    let values_start = cues.len();
    for attribute in episode.attributes() {
        for value in attribute.values().iter().copied() {
            cues.push(Cue::Value(value));
        }
    }
    cues[values_start..].sort_unstable();
    cues.dedup();
    for attribute in episode.attributes() {
        let key = attribute.key();
        for value in attribute.values().iter().copied() {
            cues.push(Cue::KeyValue { key, value });
        }
    }
    cues
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

struct PostingCursor<'a> {
    postings: &'a [usize],
    position: usize,
}

struct PostingMerge<'a> {
    cursors: Vec<PostingCursor<'a>>,
    heads: BinaryHeap<Reverse<(usize, usize)>>,
    excluded_target: usize,
}

impl<'a> PostingMerge<'a> {
    fn new(streams: impl IntoIterator<Item = &'a [usize]>, excluded_target: usize) -> Self {
        let mut cursors = Vec::new();
        let mut heads = BinaryHeap::new();
        for postings in streams {
            let position = usize::from(postings.first() == Some(&excluded_target));
            let Some(&target_index) = postings.get(position) else {
                continue;
            };
            let cursor_index = cursors.len();
            cursors.push(PostingCursor { postings, position });
            heads.push(Reverse((target_index, cursor_index)));
        }
        Self {
            cursors,
            heads,
            excluded_target,
        }
    }

    fn pop_head(&mut self) -> Option<usize> {
        let Reverse((target_index, cursor_index)) = self.heads.pop()?;
        let cursor = &mut self.cursors[cursor_index];
        debug_assert_eq!(cursor.postings[cursor.position], target_index);
        cursor.position += 1;
        if cursor.postings.get(cursor.position) == Some(&self.excluded_target) {
            cursor.position += 1;
        }
        if let Some(&next_target) = cursor.postings.get(cursor.position) {
            self.heads.push(Reverse((next_target, cursor_index)));
        }
        Some(target_index)
    }
}

impl Iterator for PostingMerge<'_> {
    type Item = (usize, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let target_index = self.pop_head()?;
        let mut shared_count = 1_u64;
        while self
            .heads
            .peek()
            .is_some_and(|Reverse((next_target, _))| *next_target == target_index)
        {
            self.pop_head()
                .expect("a matching posting head remains available");
            shared_count += 1;
        }
        Some((target_index, shared_count))
    }
}

pub(super) struct CueIndex {
    postings: BTreeMap<Cue, PostingList>,
    cue_counts: Vec<u64>,
}

impl CueIndex {
    fn from_atoms(atoms: &[EpisodeAtom]) -> Self {
        let mut index = Self {
            postings: BTreeMap::new(),
            cue_counts: Vec::with_capacity(atoms.len()),
        };
        for (atom_index, atom) in atoms.iter().enumerate() {
            index.insert(atom_index, atom);
        }
        index
    }

    pub(super) fn insert(&mut self, atom_index: usize, atom: &EpisodeAtom) {
        debug_assert_eq!(atom_index, self.cue_counts.len());
        let cues = episode_cues(atom);
        let cue_count =
            u64::try_from(cues.len()).expect("an in-memory episode's cue count fits in u64");
        for cue in cues {
            match self.postings.entry(cue) {
                Entry::Vacant(entry) => {
                    entry.insert(PostingList::One(atom_index));
                }
                Entry::Occupied(mut entry) => entry.get_mut().push(atom_index),
            }
        }
        self.cue_counts.push(cue_count);
    }
}

impl Memory {
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
        let outgoing = self.outgoing.get(&source_index);
        let source_cues = episode_cues(&self.atoms[source_index]);
        let posting_streams = source_cues.into_iter().filter_map(|cue| {
            let postings = cue_index.postings.get(&cue)?.as_slice();
            Some(postings)
        });
        let mut structural = PostingMerge::new(posting_streams, source_index).peekable();
        let mut feedback = outgoing.map(|row| row.iter().peekable());
        let candidates = std::iter::from_fn(|| {
            let structural_target = structural.peek().map(|(target, _)| *target);
            let feedback_target = feedback
                .as_mut()
                .and_then(|row| row.peek().map(|entry| *entry.0));
            let target = match (structural_target, feedback_target) {
                (Some(structural), Some(feedback)) => structural.min(feedback),
                (Some(structural), None) => structural,
                (None, Some(feedback)) => feedback,
                (None, None) => return None,
            };
            let shared_count = if structural_target == Some(target) {
                structural
                    .next()
                    .expect("peeked structural candidate remains available")
                    .1
            } else {
                0
            };
            let trace = if feedback_target == Some(target) {
                Some(
                    *feedback
                        .as_mut()
                        .and_then(Iterator::next)
                        .expect("peeked feedback candidate remains available")
                        .1,
                )
            } else {
                None
            };
            Some((target, shared_count, trace))
        });

        let source_cue_count = cue_index.cue_counts[source_index];
        let hits = candidates.filter_map(|(target_index, shared_count, trace)| {
            let structural_score = Self::structural_score(
                shared_count,
                source_cue_count,
                cue_index.cue_counts[target_index],
            );
            let learned_score = trace.map_or(0, Self::project_feedback);
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

        Ok(Self::rank_hits(hits, limit))
    }

    fn structural_score(shared_count: u64, source_count: u64, target_count: u64) -> u32 {
        if shared_count == 0 {
            return 0;
        }
        debug_assert!(shared_count <= source_count);
        debug_assert!(shared_count <= target_count);
        let union_count =
            u128::from(source_count) + u128::from(target_count) - u128::from(shared_count);
        let score = u128::from(shared_count) * u128::from(STRUCTURAL_GAIN_PPM) / union_count;
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

    fn rank_hits(hits: impl Iterator<Item = RecallHit>, limit: usize) -> Vec<RecallHit> {
        debug_assert_ne!(limit, 0);
        let mut collected = Vec::new();
        let mut bounded = None;
        for hit in hits {
            if let Some(best) = &mut bounded {
                Self::retain_better_hit(best, hit);
                continue;
            }
            if collected.len() < limit {
                collected.push(hit);
                continue;
            }

            let ranked = std::mem::take(&mut collected)
                .into_iter()
                .map(|hit| Reverse(RankedRecallHit(hit)))
                .collect::<Vec<_>>();
            let mut best = BinaryHeap::from(ranked);
            Self::retain_better_hit(&mut best, hit);
            bounded = Some(best);
        }

        let mut hits = bounded.map_or(collected, |best| {
            best.into_iter().map(|Reverse(ranked)| ranked.0).collect()
        });
        Self::sort_hits(&mut hits);
        hits
    }

    fn retain_better_hit(best: &mut BinaryHeap<Reverse<RankedRecallHit>>, hit: RecallHit) {
        let ranked = RankedRecallHit(hit);
        let worst = best.peek().expect("bounded ranking is at its limit").0;
        if ranked > worst {
            best.pop();
            best.push(Reverse(ranked));
        }
    }

    fn sort_hits(hits: &mut [RecallHit]) {
        hits.sort_unstable_by_key(|hit| Reverse(RankedRecallHit(*hit)));
    }
}

#[cfg(test)]
mod tests {
    use super::PostingMerge;

    #[test]
    fn posting_merge_aggregates_sorted_streams_once() {
        let first = [1, 3, 5];
        let second = [1, 2, 3, 5];
        let third = [2, 3, 4];
        let source_last = [0, 3];
        let source_only = [3];

        assert_eq!(
            PostingMerge::new(
                [
                    &first[..],
                    &second[..],
                    &third[..],
                    &source_last[..],
                    &source_only[..],
                ],
                3,
            )
            .collect::<Vec<_>>(),
            [(0, 1), (1, 2), (2, 2), (4, 1), (5, 2)]
        );
    }
}
