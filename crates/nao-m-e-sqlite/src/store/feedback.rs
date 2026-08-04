use std::cmp::Ordering;

use nao_m_e::{AtomId, FeedbackTrace, Memory};
use rusqlite::{Connection, Row, Rows, Transaction};

use crate::error::{StoreError, StoreIntegrityError};
use crate::format;

use super::{read_integer, read_u64};

pub(super) fn restore(connection: &Connection, memory: &mut Memory) -> Result<(), StoreError> {
    let mut statement = connection.prepare(
        "SELECT from_sequence, to_sequence, history_bits, sample_count
         FROM feedback_edges
         ORDER BY from_sequence, to_sequence",
    )?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let edge = read_feedback_edge(row)?;

        let from_id = AtomId::from_parts(memory.memory_id(), edge.from);
        let to_id = AtomId::from_parts(memory.memory_id(), edge.to);
        memory
            .set_feedback_trace(from_id, to_id, edge.trace)
            .map_err(|_| StoreIntegrityError::InvalidFeedback {
                from: edge.from,
                to: edge.to,
                detail: "feedback edge violates core graph invariants",
            })?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FeedbackRecord {
    from: u64,
    to: u64,
    trace: FeedbackTrace,
}

impl FeedbackRecord {
    const fn key(self) -> (u64, u64) {
        (self.from, self.to)
    }
}

const MAX_BUFFERED_FEEDBACK_MUTATIONS: usize = 16_384;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeedbackMutation {
    Delete(FeedbackRecord),
    Insert(FeedbackRecord),
    Update(FeedbackRecord),
}

#[derive(Debug, Default)]
struct FeedbackPlan {
    mutations: Vec<FeedbackMutation>,
    replace_all: bool,
}

impl FeedbackPlan {
    fn push(&mut self, mutation: FeedbackMutation) {
        if self.replace_all {
            return;
        }
        if self.mutations.len() == MAX_BUFFERED_FEEDBACK_MUTATIONS {
            self.mutations.clear();
            self.replace_all = true;
            return;
        }
        self.mutations.push(mutation);
    }
}

pub(super) fn reconcile(
    transaction: &Transaction<'_>,
    memory: &Memory,
    persisted_episode_count: usize,
) -> Result<(), StoreError> {
    let mut plan = FeedbackPlan::default();

    {
        let mut statement = transaction.prepare(
            "SELECT from_sequence, to_sequence, history_bits, sample_count
             FROM feedback_edges
             ORDER BY from_sequence, to_sequence",
        )?;
        let mut rows = statement.query([])?;
        let persisted_episode_count = u64::try_from(persisted_episode_count)
            .expect("an episode count always fits the atom sequence space");
        let mut persisted = next_feedback_edge(&mut rows, persisted_episode_count)?;
        let mut current = memory_feedback_records(memory);
        let mut expected = current.next();

        loop {
            if plan.replace_all {
                while persisted.is_some() {
                    persisted = next_feedback_edge(&mut rows, persisted_episode_count)?;
                }
                break;
            }
            match (persisted, expected) {
                (Some(found), Some(wanted)) => match found.key().cmp(&wanted.key()) {
                    Ordering::Less => {
                        plan.push(FeedbackMutation::Delete(found));
                        persisted = next_feedback_edge(&mut rows, persisted_episode_count)?;
                    }
                    Ordering::Equal => {
                        if found.trace != wanted.trace {
                            plan.push(FeedbackMutation::Update(wanted));
                        }
                        persisted = next_feedback_edge(&mut rows, persisted_episode_count)?;
                        expected = current.next();
                    }
                    Ordering::Greater => {
                        plan.push(FeedbackMutation::Insert(wanted));
                        expected = current.next();
                    }
                },
                (Some(found), None) => {
                    plan.push(FeedbackMutation::Delete(found));
                    persisted = next_feedback_edge(&mut rows, persisted_episode_count)?;
                }
                (None, Some(wanted)) => {
                    plan.push(FeedbackMutation::Insert(wanted));
                    expected = current.next();
                }
                (None, None) => break,
            }
        }
    }

    if plan.replace_all {
        transaction.execute("DELETE FROM feedback_edges", [])?;
        let mut insert = transaction.prepare(
            "INSERT INTO feedback_edges (
                from_sequence,
                to_sequence,
                history_bits,
                sample_count
             ) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for edge in memory_feedback_records(memory) {
            let from = format::encode_u64(edge.from);
            let to = format::encode_u64(edge.to);
            insert.execute((
                from.as_slice(),
                to.as_slice(),
                i64::from(edge.trace.history_bits()),
                i64::from(edge.trace.sample_count()),
            ))?;
        }
        return Ok(());
    }

    if plan.mutations.is_empty() {
        return Ok(());
    }

    let mut delete = transaction.prepare(
        "DELETE FROM feedback_edges
         WHERE from_sequence = ?1 AND to_sequence = ?2",
    )?;
    let mut insert = transaction.prepare(
        "INSERT INTO feedback_edges (
            from_sequence,
            to_sequence,
            history_bits,
            sample_count
         ) VALUES (?1, ?2, ?3, ?4)",
    )?;
    let mut update = transaction.prepare(
        "UPDATE feedback_edges SET history_bits = ?3, sample_count = ?4
         WHERE from_sequence = ?1 AND to_sequence = ?2",
    )?;
    for mutation in plan.mutations {
        match mutation {
            FeedbackMutation::Delete(edge) => {
                let from = format::encode_u64(edge.from);
                let to = format::encode_u64(edge.to);
                delete.execute((from.as_slice(), to.as_slice()))?;
            }
            FeedbackMutation::Insert(edge) => {
                let from = format::encode_u64(edge.from);
                let to = format::encode_u64(edge.to);
                insert.execute((
                    from.as_slice(),
                    to.as_slice(),
                    i64::from(edge.trace.history_bits()),
                    i64::from(edge.trace.sample_count()),
                ))?;
            }
            FeedbackMutation::Update(edge) => {
                let from = format::encode_u64(edge.from);
                let to = format::encode_u64(edge.to);
                update.execute((
                    from.as_slice(),
                    to.as_slice(),
                    i64::from(edge.trace.history_bits()),
                    i64::from(edge.trace.sample_count()),
                ))?;
            }
        }
    }
    Ok(())
}

fn memory_feedback_records(memory: &Memory) -> impl Iterator<Item = FeedbackRecord> + '_ {
    memory.feedback_edges().map(|edge| FeedbackRecord {
        from: edge.from().sequence(),
        to: edge.to().sequence(),
        trace: edge.trace(),
    })
}

fn next_feedback_edge(
    rows: &mut Rows<'_>,
    persisted_episode_count: u64,
) -> Result<Option<FeedbackRecord>, StoreError> {
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    let edge = read_feedback_edge(row)?;
    if edge.from >= persisted_episode_count || edge.to >= persisted_episode_count {
        return Err(StoreIntegrityError::InvalidFeedback {
            from: edge.from,
            to: edge.to,
            detail: "edge endpoint is absent",
        }
        .into());
    }
    Ok(Some(edge))
}

fn read_feedback_edge(row: &Row<'_>) -> Result<FeedbackRecord, StoreError> {
    let from = read_u64(row, 0, "feedback_edges", "from_sequence")?;
    let to = read_u64(row, 1, "feedback_edges", "to_sequence")?;
    if from == to {
        return Err(StoreIntegrityError::InvalidFeedback {
            from,
            to,
            detail: "self-edge",
        }
        .into());
    }
    let history_bits = read_integer(row, 2, "feedback_edges", "history_bits")?;
    let sample_count = read_integer(row, 3, "feedback_edges", "sample_count")?;
    let trace = u16::try_from(history_bits)
        .ok()
        .zip(u8::try_from(sample_count).ok())
        .and_then(|(history_bits, sample_count)| {
            FeedbackTrace::from_parts(history_bits, sample_count)
        })
        .ok_or(StoreIntegrityError::InvalidFeedback {
            from,
            to,
            detail: "feedback trace is not canonical",
        })?;
    Ok(FeedbackRecord { from, to, trace })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(history_bits: u16, sample_count: u8) -> FeedbackTrace {
        FeedbackTrace::from_parts(history_bits, sample_count)
            .expect("test feedback trace is canonical")
    }

    #[test]
    fn plan_bounds_buffer_before_bulk_replacement() {
        let mut plan = FeedbackPlan::default();
        for to in 0..=MAX_BUFFERED_FEEDBACK_MUTATIONS {
            plan.push(FeedbackMutation::Insert(FeedbackRecord {
                from: 0,
                to: u64::try_from(to).unwrap(),
                trace: trace(1, 1),
            }));
        }

        assert!(plan.replace_all);
        assert!(plan.mutations.is_empty());
        plan.push(FeedbackMutation::Delete(FeedbackRecord {
            from: 1,
            to: 2,
            trace: trace(0, 1),
        }));
        assert!(plan.mutations.is_empty());
    }
}
