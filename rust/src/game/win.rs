//! Six-in-line win detection.
//!
//! A placement can only create a new win on a line that passes through that
//! newly placed stone, so this file never scans the whole board.

use super::board::Board;
use super::coord::{HexCoord, AXES};
use super::state::Player;

const WIN_LENGTH: i16 = 6;

/// Return true when `coord` completes a connected line of six for `player`.
pub fn is_winning_placement(board: &Board, coord: HexCoord, player: Player) -> bool {
    AXES.iter().copied().any(|dir| {
        let forward = count_in_direction(board, coord, dir, player);
        let backward = count_in_direction(board, coord, -dir, player);
        1 + forward + backward >= WIN_LENGTH
    })
}

/// Count connected stones from `origin + dir` until the line breaks.
fn count_in_direction(board: &Board, origin: HexCoord, dir: HexCoord, player: Player) -> i16 {
    let mut count = 0;
    let mut cursor = origin + dir;

    while board.get(cursor) == Some(player) {
        count += 1;
        cursor = cursor + dir;
    }

    count
}
