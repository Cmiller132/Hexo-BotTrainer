//! Self-play boundary sketch.
//!
//! The concrete self-play loop has been removed while the Python/control-plane
//! layer is redesigned. MCTS, encoding, and game rules remain available in the
//! crate; this module only records the intended boundary for the next runner.

/// Draft configuration for a future Rust-owned self-play runner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfplayPlan {
    /// Number of games the control plane wants from one cycle.
    pub games: u32,
    /// Maximum placements before a game is capped.
    pub max_placements: u32,
    /// Encoder crop size requested by the model contract.
    pub crop_size: usize,
}

impl Default for SelfplayPlan {
    fn default() -> Self {
        Self {
            games: 0,
            max_placements: 0,
            crop_size: 0,
        }
    }
}

/// Draft cycle summary returned by a future runner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfplayCycleDraft {
    /// Zero-based cycle identifier.
    pub cycle: u32,
    /// Games actually completed.
    pub games: u32,
    /// Replay samples actually written.
    pub samples: usize,
}

impl SelfplayCycleDraft {
    /// Empty placeholder cycle.
    pub fn empty(cycle: u32) -> Self {
        Self {
            cycle,
            games: 0,
            samples: 0,
        }
    }
}
