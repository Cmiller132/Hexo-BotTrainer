//! Threat-window helpers.
//!
//! A threat is any six-cell window containing at least four stones from one
//! player and zero stones from the opponent. The current MCTS does not depend
//! on this; it is useful for metrics, debugging, and future move ordering.

use super::board::Board;
use super::coord::{HexCoord, AXES};
use super::state::Player;
use serde::{Deserialize, Serialize};

const WINDOW: i16 = 6;

/// One open six-cell tactical window for a player.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Threat {
    /// Player who owns the stones in this window.
    pub player: Player,
    /// The six coordinates that make up the window.
    pub cells: [HexCoord; WINDOW as usize],
    /// Number of `player` stones in the window. Always at least four.
    pub own_count: u8,
}

/// Find all currently open threat windows for `player`.
pub fn find_threats(board: &Board, player: Player) -> Vec<Threat> {
    let opponent = player.other();
    let mut threats = Vec::new();

    for &anchor in board.occupied_cells() {
        for axis in AXES {
            for offset in 0..WINDOW {
                // Slide a length-6 window along each axis so every occupied
                // anchor can appear at every possible position in the window.
                let start = anchor - axis.scale(offset);
                let mut cells = [HexCoord::ZERO; WINDOW as usize];
                let mut own_count = 0;
                let mut blocked = false;

                for i in 0..WINDOW {
                    let cell = start + axis.scale(i);
                    cells[i as usize] = cell;
                    match board.get(cell) {
                        Some(stone) if stone == player => own_count += 1,
                        Some(stone) if stone == opponent => {
                            blocked = true;
                            break;
                        }
                        _ => {}
                    }
                }

                if !blocked && own_count >= 4 {
                    threats.push(Threat {
                        player,
                        cells,
                        own_count,
                    });
                }
            }
        }
    }

    threats
}
