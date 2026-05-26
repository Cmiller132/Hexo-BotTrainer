//! Optional shared default encoder for crop-based model inputs.
//!
//! The encoder creates a fixed square crop around the occupied board area. The
//! crop is serialized as flat planes (`plane, row, col`) so it can cross the
//! Rust/Python boundary as plain JSON/msgpack-friendly data. Model packages may
//! use this default when its plane semantics match their architecture, or ignore
//! it and own a model-specific encoder.

use hexo_engine::{HexCoord, HexoState, Player, Stone, TurnPhase};
use serde::{Deserialize, Serialize};

/// Default crop size used by development configs.
pub const DEFAULT_CROP_SIZE: usize = 31;
/// Number of feature planes produced by the encoder.
pub const PLANE_COUNT: usize = 12;

/// Plane indices in `EncodedState.planes`.
pub mod planes {
    /// Stones owned by the player to act.
    pub const CURRENT_PLAYER_STONES: usize = 0;
    /// Stones owned by the opponent.
    pub const OPPONENT_STONES: usize = 1;
    /// Legal single-stone moves inside the crop.
    pub const LEGAL_MOVES: usize = 2;
    /// First stone of the current two-stone turn, if in `SecondStone` phase.
    pub const FIRST_STONE_THIS_TURN: usize = 3;
    /// Most recent stone by current player.
    pub const LAST_OWN_STONE_1: usize = 4;
    /// Second-most recent stone by current player.
    pub const LAST_OWN_STONE_2: usize = 5;
    /// Most recent stone by opponent.
    pub const LAST_OPPONENT_STONE_1: usize = 6;
    /// Second-most recent stone by opponent.
    pub const LAST_OPPONENT_STONE_2: usize = 7;
    /// Whole-plane phase marker for opening.
    pub const PHASE_OPENING: usize = 8;
    /// Whole-plane phase marker for first stone.
    pub const PHASE_FIRST_STONE: usize = 9;
    /// Whole-plane phase marker for second stone.
    pub const PHASE_SECOND_STONE: usize = 10;
    /// Whole-plane mask for valid crop cells.
    pub const VALID_CROP_MASK: usize = 11;
}

/// Fixed-crop tensor plus coordinate mapping metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodedState {
    /// Height and width of the square crop.
    pub crop_size: usize,
    /// Number of feature planes.
    pub plane_count: usize,
    /// Axial coordinate represented by row 0, column 0.
    pub origin: HexCoord,
    /// Axial coordinate at the crop center.
    pub center: HexCoord,
    /// Flat storage in plane-major order: `plane * H * W + row * W + col`.
    pub planes: Vec<f32>,
}

impl EncodedState {
    /// Number of cells per plane.
    pub fn cell_count(&self) -> usize {
        self.crop_size * self.crop_size
    }

    /// Flat index where one plane starts.
    pub fn plane_offset(&self, plane: usize) -> usize {
        plane * self.cell_count()
    }

    /// Write one value if the target index is in bounds.
    pub fn set(&mut self, plane: usize, index: usize, value: f32) {
        let offset = self.plane_offset(plane) + index;
        if let Some(cell) = self.planes.get_mut(offset) {
            *cell = value;
        }
    }

    /// Convert a board coordinate to a flat cell index inside the crop.
    pub fn index_of_coord(&self, coord: HexCoord) -> Option<usize> {
        let col = coord.q as i32 - self.origin.q as i32;
        let row = coord.r as i32 - self.origin.r as i32;
        if col < 0 || row < 0 {
            return None;
        }

        let col = col as usize;
        let row = row as usize;
        if col >= self.crop_size || row >= self.crop_size {
            return None;
        }

        Some(row * self.crop_size + col)
    }

    /// Convert a flat crop cell index back to a board coordinate.
    pub fn coord_at_index(&self, index: usize) -> Option<HexCoord> {
        if index >= self.cell_count() {
            return None;
        }

        let row = index / self.crop_size;
        let col = index % self.crop_size;
        Some(HexCoord {
            q: self.origin.q + col as i16,
            r: self.origin.r + row as i16,
        })
    }
}

/// Encode `state` into a fixed square crop.
pub fn encode_state(state: &HexoState, crop_size: usize) -> EncodedState {
    let crop_size = crop_size.max(1);
    let center = crop_center(state);
    let half = (crop_size / 2) as i16;
    let origin = HexCoord {
        q: center.q - half,
        r: center.r - half,
    };

    let mut encoded = EncodedState {
        crop_size,
        plane_count: PLANE_COUNT,
        origin,
        center,
        planes: vec![0.0; PLANE_COUNT * crop_size * crop_size],
    };

    // The current crop is always a full square, so every cell is valid. This
    // plane exists so future dynamic/irregular crops can reuse the same model
    // interface.
    for index in 0..encoded.cell_count() {
        encoded.set(planes::VALID_CROP_MASK, index, 1.0);
    }

    encode_stones(state, &mut encoded);
    encode_legal_moves(state, &mut encoded);
    encode_phase(state, &mut encoded);
    encode_turn_memory(state, &mut encoded);

    encoded
}

/// Fill `out` with legal placements that are representable in `encoded`.
pub fn legal_moves_in_crop(state: &HexoState, encoded: &EncodedState, out: &mut Vec<HexCoord>) {
    state.write_legal_moves(out);
    out.retain(|coord| encoded.index_of_coord(*coord).is_some());
}

/// Mark stones as current-player or opponent stones.
fn encode_stones(state: &HexoState, encoded: &mut EncodedState) {
    for &coord in state.board().occupied_cells() {
        let Some(index) = encoded.index_of_coord(coord) else {
            continue;
        };

        let Some(stone) = state.board().get(coord) else {
            continue;
        };

        let plane = if stone_player(stone) == state.current_player() {
            planes::CURRENT_PLAYER_STONES
        } else {
            planes::OPPONENT_STONES
        };
        encoded.set(plane, index, 1.0);
    }
}

/// Mark legal moves that fall inside the crop.
fn encode_legal_moves(state: &HexoState, encoded: &mut EncodedState) {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);

    for coord in legal {
        if let Some(index) = encoded.index_of_coord(coord) {
            encoded.set(planes::LEGAL_MOVES, index, 1.0);
        }
    }
}

/// Fill one phase plane so the model knows where it is in the turn.
fn encode_phase(state: &HexoState, encoded: &mut EncodedState) {
    let plane = match state.phase() {
        TurnPhase::Opening => planes::PHASE_OPENING,
        TurnPhase::FirstStone => planes::PHASE_FIRST_STONE,
        TurnPhase::SecondStone { .. } => planes::PHASE_SECOND_STONE,
    };

    for index in 0..encoded.cell_count() {
        encoded.set(plane, index, 1.0);
    }
}

/// Mark the first stone this turn and the last two stones for each side.
fn encode_turn_memory(state: &HexoState, encoded: &mut EncodedState) {
    if let TurnPhase::SecondStone { first } = state.phase() {
        if let Some(index) = encoded.index_of_coord(first) {
            encoded.set(planes::FIRST_STONE_THIS_TURN, index, 1.0);
        }
    }

    let mut own_count = 0;
    let mut opponent_count = 0;

    for record in state.placement_history().iter().rev() {
        if record.player == state.current_player() {
            let plane = match own_count {
                0 => planes::LAST_OWN_STONE_1,
                1 => planes::LAST_OWN_STONE_2,
                _ => continue,
            };
            if let Some(index) = encoded.index_of_coord(record.coord) {
                encoded.set(plane, index, 1.0);
            }
            own_count += 1;
        } else {
            let plane = match opponent_count {
                0 => planes::LAST_OPPONENT_STONE_1,
                1 => planes::LAST_OPPONENT_STONE_2,
                _ => continue,
            };
            if let Some(index) = encoded.index_of_coord(record.coord) {
                encoded.set(plane, index, 1.0);
            }
            opponent_count += 1;
        }

        if own_count >= 2 && opponent_count >= 2 {
            break;
        }
    }
}

/// Choose a deterministic crop center from occupied-cell bounds.
fn crop_center(state: &HexoState) -> HexCoord {
    match state.board().bounds() {
        Some((min, max)) => HexCoord {
            q: min.q + (max.q - min.q) / 2,
            r: min.r + (max.r - min.r) / 2,
        },
        None => HexCoord::ZERO,
    }
}

/// Convert a board stone alias back to its player.
fn stone_player(stone: Stone) -> Player {
    stone
}
