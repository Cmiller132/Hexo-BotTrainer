//! Dense CNN Model1 input encoding.
//!
//! This file owns the 13-plane dense tensor contract consumed by the PyTorch
//! network. MCTS and inference call this encoder, while sample generation uses
//! a few coordinate helpers to keep compact sample facts aligned with training
//! expansion.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use hexo_engine::{
    pack_coord, Axis, HexCoord, HexoState as RustHexoState, PackedCoord, Player, TurnPhase,
};

use super::constants::*;
use super::state::states_from_py_states;

pub(crate) struct Model1EncodedState {
    pub(crate) planes: Vec<f32>,
    pub(crate) half_planes: Vec<u16>,
    pub(crate) all_legal_action_count: usize,
    pub(crate) crop_legal_action_count: usize,
    pub(crate) legal_action_ids: Vec<PackedCoord>,
    pub(crate) legal_flat_indices: Vec<i64>,
    pub(crate) center: HexCoord,
}

#[pyfunction(signature = (states))]
pub fn model1_batch_inputs(py: Python<'_>, states: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let states = states_from_py_states(py, states)?;
    let mut planes: Vec<f32> = Vec::new();
    let legal_action_rows = PyList::empty(py);
    let legal_flat_rows = PyList::empty(py);
    let centers = PyList::empty(py);

    for state in &states {
        let encoded = encode_model1_state(state);
        planes.extend_from_slice(&encoded.planes);
        legal_action_rows.append(PyTuple::new(py, encoded.legal_action_ids)?)?;
        legal_flat_rows.append(PyTuple::new(py, encoded.legal_flat_indices)?)?;
        centers.append((encoded.center.q, encoded.center.r))?;
    }

    let byte_len = planes.len() * std::mem::size_of::<f32>();
    let bytes = unsafe { std::slice::from_raw_parts(planes.as_ptr() as *const u8, byte_len) };
    let dict = PyDict::new(py);
    dict.set_item("inputs", PyBytes::new(py, bytes))?;
    dict.set_item(
        "shape",
        (
            states.len(),
            MODEL1_INPUT_CHANNELS,
            MODEL1_BOARD_SIZE,
            MODEL1_BOARD_SIZE,
        ),
    )?;
    dict.set_item("legal_action_ids", legal_action_rows)?;
    dict.set_item("legal_flat_indices", legal_flat_rows)?;
    dict.set_item("centers", centers)?;
    Ok(dict.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(model1_batch_inputs, module)?)?;
    Ok(())
}

pub(crate) fn encode_model1_state(state: &RustHexoState) -> Model1EncodedState {
    encode_model1_state_inner(state, true)
}

pub(crate) fn encode_model1_state_for_mcts(
    state: &RustHexoState,
    include_crop_legal_lists: bool,
) -> Model1EncodedState {
    encode_model1_state_inner(state, include_crop_legal_lists)
}

pub(crate) fn encode_model1_state_half_for_mcts(state: &RustHexoState) -> Model1EncodedState {
    let center = model1_crop_center(state);
    let mut half_planes = model1_half_base_planes().to_vec();

    let current_player = state.current_player();
    for &coord in state.board().occupied_cells() {
        let Some(flat) = model1_flat_index(coord, center) else {
            continue;
        };
        let owner = state.board().get(coord).unwrap_or(current_player);
        let plane = if owner == current_player {
            MODEL1_PLANE_OWN_STONES
        } else {
            MODEL1_PLANE_OPPONENT_STONES
        };
        set_half_plane(&mut half_planes, plane, flat, half_one());
        set_half_plane(&mut half_planes, MODEL1_PLANE_EMPTY, flat, 0);
    }

    let mut legal_coords = Vec::with_capacity(state.legal_move_count());
    state.write_legal_moves(&mut legal_coords);
    let all_legal_action_count = legal_coords.len();
    let mut crop_legal_action_count = 0usize;
    for coord in legal_coords {
        if let Some(flat) = model1_flat_index(coord, center) {
            crop_legal_action_count += 1;
            set_half_plane(&mut half_planes, MODEL1_PLANE_LEGAL, flat, half_one());
        }
    }

    match state.phase() {
        TurnPhase::SecondStone { first } => {
            fill_half_plane(&mut half_planes, MODEL1_PLANE_SECOND_PLACEMENT, half_one());
            if let Some(flat) = model1_flat_index(first, center) {
                set_half_plane(&mut half_planes, MODEL1_PLANE_FIRST_STONE, flat, half_one());
            }
        }
        TurnPhase::Opening | TurnPhase::FirstStone => {}
    }

    if current_player == Player::Player0 {
        fill_half_plane(&mut half_planes, MODEL1_PLANE_PLAYER_COLOUR, half_one());
    }

    let latest_index = state.placements_made();
    for record in state.placement_history().iter().rev() {
        let Some(flat) = model1_flat_index(record.coord, center) else {
            continue;
        };
        let age = latest_index.saturating_sub(record.placement_index);
        let weight = f32_to_f16_bits(1.0 / (1.0 + age as f32));
        let plane = if record.player == current_player {
            MODEL1_PLANE_OWN_RECENCY
        } else {
            MODEL1_PLANE_OPPONENT_RECENCY
        };
        let offset = plane_offset(plane, flat);
        half_planes[offset] = half_planes[offset].max(weight);
    }

    fill_hot_cells_half(state, current_player, center, &mut half_planes);
    fill_opponent_last_turn_half(state, current_player, center, &mut half_planes);

    Model1EncodedState {
        planes: Vec::new(),
        half_planes,
        all_legal_action_count,
        crop_legal_action_count,
        legal_action_ids: Vec::new(),
        legal_flat_indices: Vec::new(),
        center,
    }
}

fn encode_model1_state_inner(
    state: &RustHexoState,
    include_crop_legal_lists: bool,
) -> Model1EncodedState {
    let center = model1_crop_center(state);
    let mut planes = model1_base_planes().to_vec();

    // The plane order mirrors `dense_cnn/python/.../input.py`; changing it is a
    // model contract change, not an engine concern.
    let current_player = state.current_player();
    for &coord in state.board().occupied_cells() {
        let Some(flat) = model1_flat_index(coord, center) else {
            continue;
        };
        let owner = state.board().get(coord).unwrap_or(current_player);
        let plane = if owner == current_player {
            MODEL1_PLANE_OWN_STONES
        } else {
            MODEL1_PLANE_OPPONENT_STONES
        };
        set_plane(&mut planes, plane, flat, 1.0);
        set_plane(&mut planes, MODEL1_PLANE_EMPTY, flat, 0.0);
    }

    let mut legal_coords = Vec::with_capacity(state.legal_move_count());
    state.write_legal_moves(&mut legal_coords);
    let all_legal_action_count = legal_coords.len();
    let mut legal_action_ids = if include_crop_legal_lists {
        Vec::with_capacity(legal_coords.len())
    } else {
        Vec::new()
    };
    let mut legal_flat_indices = if include_crop_legal_lists {
        Vec::with_capacity(legal_coords.len())
    } else {
        Vec::new()
    };
    let mut crop_legal_action_count = 0usize;
    for coord in legal_coords {
        let action_id = pack_coord(coord);
        if let Some(flat) = model1_flat_index(coord, center) {
            crop_legal_action_count += 1;
            set_plane(&mut planes, MODEL1_PLANE_LEGAL, flat, 1.0);
            if include_crop_legal_lists {
                legal_action_ids.push(action_id);
                legal_flat_indices.push(flat as i64);
            }
        }
    }

    match state.phase() {
        TurnPhase::SecondStone { first } => {
            fill_plane(&mut planes, MODEL1_PLANE_SECOND_PLACEMENT, 1.0);
            if let Some(flat) = model1_flat_index(first, center) {
                set_plane(&mut planes, MODEL1_PLANE_FIRST_STONE, flat, 1.0);
            }
        }
        TurnPhase::Opening | TurnPhase::FirstStone => {}
    }

    if current_player == Player::Player0 {
        fill_plane(&mut planes, MODEL1_PLANE_PLAYER_COLOUR, 1.0);
    }

    let latest_index = state.placements_made();
    for record in state.placement_history().iter().rev() {
        let Some(flat) = model1_flat_index(record.coord, center) else {
            continue;
        };
        let age = latest_index.saturating_sub(record.placement_index);
        let weight = 1.0 / (1.0 + age as f32);
        let plane = if record.player == current_player {
            MODEL1_PLANE_OWN_RECENCY
        } else {
            MODEL1_PLANE_OPPONENT_RECENCY
        };
        let offset = plane_offset(plane, flat);
        planes[offset] = planes[offset].max(weight);
    }

    fill_hot_cells(state, current_player, center, &mut planes);
    fill_opponent_last_turn(state, current_player, center, &mut planes);

    Model1EncodedState {
        planes,
        half_planes: Vec::new(),
        all_legal_action_count,
        crop_legal_action_count,
        legal_action_ids,
        legal_flat_indices,
        center,
    }
}

fn model1_base_planes() -> &'static [f32] {
    MODEL1_BASE_PLANES.get_or_init(|| {
        let mut planes = vec![0.0; MODEL1_INPUT_CHANNELS * MODEL1_BOARD_AREA];
        fill_plane(&mut planes, MODEL1_PLANE_EMPTY, 1.0);
        fill_distance_plane(&mut planes);
        planes
    })
}

fn model1_half_base_planes() -> &'static [u16] {
    MODEL1_BASE_HALF_PLANES.get_or_init(|| {
        let mut planes = vec![0u16; MODEL1_INPUT_CHANNELS * MODEL1_BOARD_AREA];
        fill_half_plane(&mut planes, MODEL1_PLANE_EMPTY, half_one());
        fill_half_distance_plane(&mut planes);
        planes
    })
}

fn fill_hot_cells(
    state: &RustHexoState,
    current_player: Player,
    center: HexCoord,
    planes: &mut [f32],
) {
    if state.placements_made() < 7 {
        return;
    }
    for entry in state.board().windows().entries() {
        let p0_count = entry.mask(Player::Player0).count_ones();
        let p1_count = entry.mask(Player::Player1).count_ones();
        if (p0_count > 0 && p1_count > 0) || (p0_count < 4 && p1_count < 4) {
            continue;
        }
        let player = if p0_count >= 4 {
            Player::Player0
        } else {
            Player::Player1
        };
        let plane = if player == current_player {
            MODEL1_PLANE_OWN_HOT
        } else {
            MODEL1_PLANE_OPPONENT_HOT
        };
        let key = entry.key();
        let (dq, dr) = axis_delta(key.axis);
        for index in 0..6 {
            let coord = HexCoord {
                q: key.start.q + dq * index,
                r: key.start.r + dr * index,
            };
            if !state.board().is_cell_empty(coord) {
                continue;
            }
            if let Some(flat) = model1_flat_index(coord, center) {
                set_plane(planes, plane, flat, 1.0);
            }
        }
    }
}

fn fill_opponent_last_turn(
    state: &RustHexoState,
    current_player: Player,
    center: HexCoord,
    planes: &mut [f32],
) {
    let opponent = current_player.other();
    for record in state.placement_history().iter().rev() {
        if record.player != opponent {
            continue;
        }
        match record.phase {
            TurnPhase::SecondStone { first } => {
                if let Some(flat) = model1_flat_index(first, center) {
                    set_plane(planes, MODEL1_PLANE_OPPONENT_LAST_TURN, flat, 1.0);
                }
                if let Some(flat) = model1_flat_index(record.coord, center) {
                    set_plane(planes, MODEL1_PLANE_OPPONENT_LAST_TURN, flat, 1.0);
                }
                return;
            }
            TurnPhase::Opening => {
                if let Some(flat) = model1_flat_index(record.coord, center) {
                    set_plane(planes, MODEL1_PLANE_OPPONENT_LAST_TURN, flat, 1.0);
                }
                return;
            }
            TurnPhase::FirstStone => {}
        }
    }
}

fn fill_hot_cells_half(
    state: &RustHexoState,
    current_player: Player,
    center: HexCoord,
    planes: &mut [u16],
) {
    if state.placements_made() < 7 {
        return;
    }
    for entry in state.board().windows().entries() {
        let p0_count = entry.mask(Player::Player0).count_ones();
        let p1_count = entry.mask(Player::Player1).count_ones();
        if (p0_count > 0 && p1_count > 0) || (p0_count < 4 && p1_count < 4) {
            continue;
        }
        let player = if p0_count >= 4 {
            Player::Player0
        } else {
            Player::Player1
        };
        let plane = if player == current_player {
            MODEL1_PLANE_OWN_HOT
        } else {
            MODEL1_PLANE_OPPONENT_HOT
        };
        let key = entry.key();
        let (dq, dr) = axis_delta(key.axis);
        for index in 0..6 {
            let coord = HexCoord {
                q: key.start.q + dq * index,
                r: key.start.r + dr * index,
            };
            if !state.board().is_cell_empty(coord) {
                continue;
            }
            if let Some(flat) = model1_flat_index(coord, center) {
                set_half_plane(planes, plane, flat, half_one());
            }
        }
    }
}

fn fill_opponent_last_turn_half(
    state: &RustHexoState,
    current_player: Player,
    center: HexCoord,
    planes: &mut [u16],
) {
    let opponent = current_player.other();
    for record in state.placement_history().iter().rev() {
        if record.player != opponent {
            continue;
        }
        match record.phase {
            TurnPhase::SecondStone { first } => {
                if let Some(flat) = model1_flat_index(first, center) {
                    set_half_plane(planes, MODEL1_PLANE_OPPONENT_LAST_TURN, flat, half_one());
                }
                if let Some(flat) = model1_flat_index(record.coord, center) {
                    set_half_plane(planes, MODEL1_PLANE_OPPONENT_LAST_TURN, flat, half_one());
                }
                return;
            }
            TurnPhase::Opening => {
                if let Some(flat) = model1_flat_index(record.coord, center) {
                    set_half_plane(planes, MODEL1_PLANE_OPPONENT_LAST_TURN, flat, half_one());
                }
                return;
            }
            TurnPhase::FirstStone => {}
        }
    }
}

pub(crate) fn model1_crop_center(state: &RustHexoState) -> HexCoord {
    let occupied = state.board().occupied_cells();
    if occupied.is_empty() {
        return HexCoord { q: 0, r: 0 };
    }
    let q_sum: i32 = occupied.iter().map(|coord| coord.q as i32).sum();
    let r_sum: i32 = occupied.iter().map(|coord| coord.r as i32).sum();
    HexCoord {
        q: python_round(q_sum, occupied.len() as i32) as i16,
        r: python_round(r_sum, occupied.len() as i32) as i16,
    }
}

fn python_round(numerator: i32, denominator: i32) -> i32 {
    let quotient = numerator.div_euclid(denominator);
    let remainder = numerator.rem_euclid(denominator);
    let doubled = remainder * 2;
    if doubled < denominator {
        quotient
    } else if doubled > denominator {
        quotient + 1
    } else if quotient % 2 == 0 {
        quotient
    } else {
        quotient + 1
    }
}

pub(crate) fn model1_flat_index(coord: HexCoord, center: HexCoord) -> Option<usize> {
    let half = (MODEL1_BOARD_SIZE / 2) as i32;
    let row = coord.r as i32 - center.r as i32 + half;
    let col = coord.q as i32 - center.q as i32 + half;
    if row < 0 || col < 0 || row >= MODEL1_BOARD_SIZE as i32 || col >= MODEL1_BOARD_SIZE as i32 {
        return None;
    }
    Some(row as usize * MODEL1_BOARD_SIZE + col as usize)
}

pub(crate) fn model1_coord_from_flat(flat: usize, center: HexCoord) -> Option<HexCoord> {
    if flat >= MODEL1_BOARD_AREA {
        return None;
    }
    let half = (MODEL1_BOARD_SIZE / 2) as i32;
    let row = flat / MODEL1_BOARD_SIZE;
    let col = flat % MODEL1_BOARD_SIZE;
    Some(HexCoord {
        q: (col as i32 - half + center.q as i32) as i16,
        r: (row as i32 - half + center.r as i32) as i16,
    })
}

fn fill_distance_plane(planes: &mut [f32]) {
    let half = (MODEL1_BOARD_SIZE / 2) as i32;
    for row in 0..MODEL1_BOARD_SIZE {
        for col in 0..MODEL1_BOARD_SIZE {
            let r = row as i32 - half;
            let q = col as i32 - half;
            let s = -r - q;
            let distance =
                r.abs().max(q.abs()).max(s.abs()) as f32 / (MODEL1_BOARD_SIZE - 1) as f32;
            set_plane(
                planes,
                MODEL1_PLANE_CENTER_DISTANCE,
                row * MODEL1_BOARD_SIZE + col,
                distance,
            );
        }
    }
}

fn fill_half_distance_plane(planes: &mut [u16]) {
    let half = (MODEL1_BOARD_SIZE / 2) as i32;
    for row in 0..MODEL1_BOARD_SIZE {
        for col in 0..MODEL1_BOARD_SIZE {
            let r = row as i32 - half;
            let q = col as i32 - half;
            let s = -r - q;
            let distance =
                r.abs().max(q.abs()).max(s.abs()) as f32 / (MODEL1_BOARD_SIZE - 1) as f32;
            set_half_plane(
                planes,
                MODEL1_PLANE_CENTER_DISTANCE,
                row * MODEL1_BOARD_SIZE + col,
                f32_to_f16_bits(distance),
            );
        }
    }
}

fn fill_plane(planes: &mut [f32], plane: usize, value: f32) {
    let start = plane * MODEL1_BOARD_AREA;
    let end = start + MODEL1_BOARD_AREA;
    planes[start..end].fill(value);
}

fn set_plane(planes: &mut [f32], plane: usize, flat: usize, value: f32) {
    planes[plane_offset(plane, flat)] = value;
}

fn fill_half_plane(planes: &mut [u16], plane: usize, value: u16) {
    let start = plane * MODEL1_BOARD_AREA;
    let end = start + MODEL1_BOARD_AREA;
    planes[start..end].fill(value);
}

fn set_half_plane(planes: &mut [u16], plane: usize, flat: usize, value: u16) {
    planes[plane_offset(plane, flat)] = value;
}

fn plane_offset(plane: usize, flat: usize) -> usize {
    plane * MODEL1_BOARD_AREA + flat
}

fn half_one() -> u16 {
    0x3c00
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if exponent == 0xff {
        return if mantissa == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = 14 - half_exponent;
        let mut half_mantissa = (mantissa >> shift) as u16;
        if ((mantissa >> (shift - 1)) & 1) != 0 {
            half_mantissa = half_mantissa.saturating_add(1);
        }
        return sign | half_mantissa;
    }

    let mut half = sign | ((half_exponent as u16) << 10) | ((mantissa >> 13) as u16);
    if (mantissa & 0x1000) != 0 {
        half = half.saturating_add(1);
    }
    half
}

fn axis_delta(axis: Axis) -> (i16, i16) {
    match axis {
        Axis::Q => (1, 0),
        Axis::R => (0, 1),
        Axis::QR => (1, -1),
    }
}
