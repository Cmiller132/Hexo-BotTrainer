//! Sparse board storage.
//!
//! Hexo has no fixed board bounds, so the board stores only occupied cells.
//! A hash map gives O(1)-ish lookup by coordinate, while `occupied` preserves a
//! compact list for frontier generation, encoding, and board summaries.

use super::coord::HexCoord;
use super::state::Player;
use super::windows::{WindowStore, WindowUpdate};
use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// In the game engine, a stone is just the owning player.
pub type Stone = Player;

/// Sparse representation of all placed stones.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Board {
    /// Coordinate -> owner lookup for legality and window updates.
    stones: AHashMap<HexCoord, Stone>,
    /// Placement coordinates in insertion order.
    occupied: Vec<HexCoord>,
    /// Incrementally maintained six-cell window state.
    windows: WindowStore,
}

/// Errors produced when a placement violates the rules.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MoveError {
    #[error("cannot apply a move to a terminal state")]
    TerminalState,
    #[error("opening placement must be at (0, 0)")]
    IllegalOpening,
    #[error("cell {0:?} is already occupied")]
    Occupied(HexCoord),
    #[error("cell {0:?} is not a legal placement")]
    IllegalPlacement(HexCoord),
    #[error("second placement cannot reuse the first placement")]
    ReusedFirstStone,
}

impl Board {
    /// Create an empty board.
    pub fn new() -> Self {
        Self::default()
    }

    /// True when no stone occupies `coord`.
    pub fn is_empty(&self, coord: HexCoord) -> bool {
        !self.stones.contains_key(&coord)
    }

    /// Return the owner of a cell, if occupied.
    pub fn get(&self, coord: HexCoord) -> Option<Stone> {
        self.stones.get(&coord).copied()
    }

    /// Place one stone without checking higher-level turn rules.
    ///
    /// Callers should validate game legality before calling this method. This
    /// method only protects the board invariant that a cell cannot be occupied
    /// twice.
    pub fn place(&mut self, coord: HexCoord, stone: Stone) -> Result<WindowUpdate, MoveError> {
        if !self.is_empty(coord) {
            return Err(MoveError::Occupied(coord));
        }
        self.stones.insert(coord, stone);
        self.occupied.push(coord);
        Ok(self.windows.update_for_placement(coord, stone))
    }

    /// Incremental active/threat/win window state.
    pub fn windows(&self) -> &WindowStore {
        &self.windows
    }

    /// All occupied coordinates in placement order.
    pub fn occupied_cells(&self) -> &[HexCoord] {
        &self.occupied
    }

    /// Number of stones currently on the board.
    pub fn len(&self) -> usize {
        self.occupied.len()
    }

    /// True when the board contains no stones.
    pub fn is_empty_board(&self) -> bool {
        self.occupied.is_empty()
    }

    /// Axis-aligned axial bounds around occupied cells.
    ///
    /// This is not a playable board boundary; it is a convenience for encoding
    /// and diagnostics.
    pub fn bounds(&self) -> Option<(HexCoord, HexCoord)> {
        let first = *self.occupied.first()?;
        let mut min_q = first.q;
        let mut max_q = first.q;
        let mut min_r = first.r;
        let mut max_r = first.r;

        for coord in &self.occupied {
            min_q = min_q.min(coord.q);
            max_q = max_q.max(coord.q);
            min_r = min_r.min(coord.r);
            max_r = max_r.max(coord.r);
        }

        Some((
            HexCoord { q: min_q, r: min_r },
            HexCoord { q: max_q, r: max_r },
        ))
    }
}
