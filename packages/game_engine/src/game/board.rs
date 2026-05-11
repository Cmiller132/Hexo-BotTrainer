//! Sparse board storage.
//!
//! Hexo has no fixed board bounds, so the board stores only occupied cells.
//! A hash map gives O(1)-ish lookup by coordinate, while `occupied` preserves a
//! compact list for frontier generation, encoding, and board summaries.

use super::coord::{coords_within_radius, HexCoord};
use super::state::Player;
use super::windows::{WindowStore, WindowUpdate};
use ahash::{AHashMap, AHashSet};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Maximum distance from any existing stone for non-opening placements.
pub const LEGAL_FRONTIER_RADIUS: i16 = 8;

/// In the game engine, a stone is just the owning player.
pub type Stone = Player;

/// Sparse representation of all placed stones.
#[derive(Clone, Debug, Default)]
pub struct Board {
    /// Coordinate -> owner lookup for legality and window updates.
    stones: AHashMap<HexCoord, Stone>,
    /// Placement coordinates in insertion order.
    occupied: Vec<HexCoord>,
    /// Incrementally maintained six-cell window state.
    windows: WindowStore,
    /// Incrementally maintained legal non-opening placements.
    frontier: AHashSet<HexCoord>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct BoardStone {
    coord: HexCoord,
    stone: Stone,
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
    pub(crate) fn place(
        &mut self,
        coord: HexCoord,
        stone: Stone,
    ) -> Result<WindowUpdate, MoveError> {
        if !self.is_empty(coord) {
            return Err(MoveError::Occupied(coord));
        }
        self.stones.insert(coord, stone);
        self.occupied.push(coord);
        self.update_frontier_for_placement(coord);
        Ok(self.windows.update_for_placement(coord, stone))
    }

    /// Incremental threat/win window state.
    pub fn windows(&self) -> &WindowStore {
        &self.windows
    }

    /// True when `coord` is a legal non-opening frontier cell.
    pub fn is_frontier_cell(&self, coord: HexCoord) -> bool {
        self.frontier.contains(&coord)
    }

    /// Iterate legal non-opening frontier cells.
    pub fn frontier_cells(&self) -> impl Iterator<Item = HexCoord> + '_ {
        self.frontier.iter().copied()
    }

    /// All occupied coordinates in placement order.
    pub fn occupied_cells(&self) -> &[HexCoord] {
        &self.occupied
    }

    /// Number of stones currently on the board.
    pub fn len(&self) -> usize {
        self.occupied.len()
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

    fn update_frontier_for_placement(&mut self, coord: HexCoord) {
        self.frontier.remove(&coord);
        for candidate in coords_within_radius(coord, LEGAL_FRONTIER_RADIUS) {
            if self.is_empty(candidate) {
                self.frontier.insert(candidate);
            }
        }
    }
}

impl Serialize for Board {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let placements: Vec<BoardStone> = self
            .occupied
            .iter()
            .filter_map(|coord| {
                self.get(*coord).map(|stone| BoardStone {
                    coord: *coord,
                    stone,
                })
            })
            .collect();
        placements.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Board {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let placements = Vec::<BoardStone>::deserialize(deserializer)?;
        let mut board = Self::new();
        for placement in placements {
            board
                .place(placement.coord, placement.stone)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(board)
    }
}
