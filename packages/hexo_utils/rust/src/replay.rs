//! Replay boundary sketch.
//!
//! The old `ReplaySample` type mixed three decisions in one place:
//! encoding shape, MCTS visit targets, and the training-file schema. That made
//! it too easy for the Python runner redesign to inherit a replay format before
//! the contract was actually settled.
//!
//! Keep this module as a small marker for now. The next implementation should
//! define replay as an explicit versioned contract between Rust self-play and
//! whichever trainer consumes it.

/// Current placeholder schema version.
pub const REPLAY_SCHEMA_DRAFT: u32 = 0;

/// Minimal manifest-like replay batch description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayBatchDraft {
    /// Schema draft used by this placeholder.
    pub schema: u32,
    /// Number of samples the eventual writer claims to contain.
    pub samples: usize,
}

impl ReplayBatchDraft {
    /// Create an empty draft batch.
    pub fn empty() -> Self {
        Self {
            schema: REPLAY_SCHEMA_DRAFT,
            samples: 0,
        }
    }
}
