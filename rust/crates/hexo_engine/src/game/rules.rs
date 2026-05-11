//! Legal placement generation and validation.
//!
//! The board maintains a radius-8 frontier incrementally. After the opening
//! move, legal placement generation is a query over that frontier rather than a
//! full-board rescan.

use super::board::{MoveError, LEGAL_FRONTIER_RADIUS};
use super::coord::HexCoord;
use super::state::{HexoState, TurnPhase};

/// Maximum distance from any existing stone for non-opening placements.
pub const FRONTIER_RADIUS: i16 = LEGAL_FRONTIER_RADIUS;

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
            out.extend(state.board().frontier_cells());
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

    if state.board().is_frontier_cell(coord) {
        Ok(())
    } else {
        Err(MoveError::IllegalPlacement(coord))
    }
}
