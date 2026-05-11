//! Legal placement generation and validation.
//!
//! The prototype uses a simple radius frontier: after the opening move, every
//! new stone must be empty and within eight hex steps of an existing stone.

use super::board::MoveError;
use super::coord::{coords_within_radius, hex_distance, HexCoord};
use super::state::{HexoState, TurnPhase};
use ahash::AHashSet;

/// Maximum distance from any existing stone for non-opening placements.
pub const FRONTIER_RADIUS: i16 = 8;

/// Fill `out` with all legal single-stone placements for the current state.
///
/// This respects the autoregressive phase model:
/// - opening: only `(0, 0)`
/// - first stone: any empty frontier cell
/// - second stone: any empty frontier cell other than the first stone
pub fn legal_placements(state: &HexoState, out: &mut Vec<HexCoord>) {
    out.clear();

    if state.terminal().is_some() {
        return;
    }

    match state.phase() {
        TurnPhase::Opening => {
            if state.board().is_empty(HexCoord::ZERO) {
                out.push(HexCoord::ZERO);
            }
        }
        TurnPhase::FirstStone | TurnPhase::SecondStone { .. } => {
            let mut seen = AHashSet::new();
            for occupied in state.board().occupied_cells() {
                // Prototype implementation: rescan radius-8 neighborhoods and
                // deduplicate. This is simple and correct; an incremental
                // frontier can replace it later if profiling says it matters.
                for candidate in coords_within_radius(*occupied, FRONTIER_RADIUS) {
                    if state.board().is_empty(candidate) && seen.insert(candidate) {
                        out.push(candidate);
                    }
                }
            }
            out.sort_by_key(|coord| (coord.q, coord.r));
        }
    }
}

/// Validate one coordinate against the current state and phase.
pub fn is_legal_placement(state: &HexoState, coord: HexCoord) -> Result<(), MoveError> {
    if state.terminal().is_some() {
        return Err(MoveError::TerminalState);
    }

    match state.phase() {
        TurnPhase::Opening => {
            if coord == HexCoord::ZERO && state.board().is_empty(coord) {
                Ok(())
            } else {
                Err(MoveError::IllegalOpening)
            }
        }
        TurnPhase::FirstStone => legal_non_opening_placement(state, coord),
        TurnPhase::SecondStone { first } => {
            if coord == first {
                return Err(MoveError::ReusedFirstStone);
            }
            legal_non_opening_placement(state, coord)
        }
    }
}

/// Shared validation for all non-opening placements.
fn legal_non_opening_placement(state: &HexoState, coord: HexCoord) -> Result<(), MoveError> {
    if !state.board().is_empty(coord) {
        return Err(MoveError::Occupied(coord));
    }

    if state
        .board()
        .occupied_cells()
        .iter()
        .any(|occupied| hex_distance(*occupied, coord) <= FRONTIER_RADIUS)
    {
        Ok(())
    } else {
        Err(MoveError::IllegalPlacement(coord))
    }
}
