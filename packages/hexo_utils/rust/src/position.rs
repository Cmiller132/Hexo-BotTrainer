//! Search-owned mutable game position.
//!
//! MCTS should never mutate the caller's primary `HexoState`. A
//! `SearchPosition` wraps a cloned state for one rollout and delegates every
//! mutation back to `hexo_engine::apply_placement`, keeping the engine as the
//! single source of truth for legality, phase changes, windows, and wins.

use hexo_engine::{apply_placement, ApplyResult, HexCoord, HexoState, MoveError, Placement};

/// Mutable state used inside search simulations.
#[derive(Clone, Debug)]
pub struct SearchPosition {
    state: HexoState,
}

impl SearchPosition {
    /// Create a search position from an owned engine state.
    pub fn new(state: HexoState) -> Self {
        Self { state }
    }

    /// Clone the immutable root state into a rollout-owned position.
    pub fn from_root(root: &HexoState) -> Self {
        Self {
            state: root.clone(),
        }
    }

    /// Read the wrapped engine state.
    pub fn state(&self) -> &HexoState {
        &self.state
    }

    /// Consume the wrapper and return the final engine state.
    pub fn into_state(self) -> HexoState {
        self.state
    }

    /// Apply a single-stone placement through the engine rule transition.
    pub fn apply_placement(&mut self, placement: Placement) -> Result<ApplyResult, MoveError> {
        apply_placement(&mut self.state, placement)
    }

    /// Convenience helper for applying a coordinate as a placement.
    pub fn place_coord(&mut self, coord: HexCoord) -> Result<ApplyResult, MoveError> {
        self.apply_placement(Placement { coord })
    }
}
