//! Samples boundary sketch.
//!
//! Older sample sketches mixed three decisions in one place:
//! encoding shape, MCTS visit targets, and the training-file schema. That made
//! it too easy for the Python runner redesign to inherit a sample format before
//! the contract was actually settled.
//!
//! Keep this module as a small marker for now. The next implementation should
//! define samples as an explicit versioned contract between self-play and
//! whichever trainer consumes it.

/// Current placeholder schema version.
pub const SAMPLE_SCHEMA_DRAFT: u32 = 0;

/// Minimal manifest-like sample batch description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleBatchDraft {
    /// Schema draft used by this placeholder.
    pub schema: u32,
    /// Number of samples the eventual writer claims to contain.
    pub samples: usize,
}

impl SampleBatchDraft {
    /// Create an empty draft batch.
    pub fn empty() -> Self {
        Self {
            schema: SAMPLE_SCHEMA_DRAFT,
            samples: 0,
        }
    }
}
