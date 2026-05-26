//! Search-owned mutable game position.
//!
//! MCTS should never mutate the caller's primary `HexoState`. A
//! `SearchPosition` wraps one cloned root state for a whole search and records
//! apply deltas as simulations descend the tree. Rewinding those deltas returns
//! the position to the root without cloning `HexoState` for every rollout.

use hexo_engine::{ApplyDelta, ApplyResult, HexCoord, HexoState, MoveError, Placement};

/// Mutable state used inside search simulations.
#[derive(Clone, Debug)]
pub struct SearchPosition {
    state: HexoState,
    deltas: Vec<ApplyDelta>,
}

impl SearchPosition {
    /// Create a search position from an owned engine state.
    pub fn new(state: HexoState) -> Self {
        Self {
            state,
            deltas: Vec::new(),
        }
    }

    /// Clone the immutable root state into a rollout-owned position.
    pub fn from_root(root: &HexoState) -> Self {
        Self {
            state: root.clone(),
            deltas: Vec::new(),
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
        let (result, delta) = self.state.apply_with_delta(placement)?;
        self.deltas.push(delta);
        Ok(result)
    }

    /// Convenience helper for applying a coordinate as a placement.
    pub fn place_coord(&mut self, coord: HexCoord) -> Result<ApplyResult, MoveError> {
        self.apply_placement(Placement { coord })
    }

    /// Number of applied placements since the root position.
    pub fn depth(&self) -> usize {
        self.deltas.len()
    }

    /// Undo the most recent search placement.
    pub fn undo_last(&mut self) -> bool {
        let Some(delta) = self.deltas.pop() else {
            return false;
        };
        self.state.undo(delta);
        true
    }

    /// Undo every placement applied during the current simulation.
    pub fn rewind(&mut self) {
        while self.undo_last() {}
    }
}
