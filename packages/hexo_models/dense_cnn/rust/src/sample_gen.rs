//! Dense CNN compact sample generation.
//!
//! Self-play records compact facts rather than dense tensors. This module reads
//! the state-derived facts for a live position by cloning the live engine state
//! through the generic engine state capsule.
//!
//! Search targets (`policy`, `root_prior_policy`) are attached by Python from the
//! MCTS result, and outcome targets (`value`, `opp_policy`, `short_term_value`)
//! are computed by Python finalization once a game ends. Those steps are pure
//! arithmetic over the game's decision sequence and do not need engine state, so
//! they live in Python rather than crossing this boundary.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use hexo_engine::{
    pack_coord, Axis, HexCoord, HexoState as RustHexoState, Player, TurnPhase,
};

#[pyfunction(signature = (state, game_id, turn_index, metadata))]
pub fn model1_sample_from_state(
    py: Python<'_>,
    state: &Bound<'_, PyAny>,
    game_id: String,
    turn_index: u32,
    metadata: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // Clone the live Python state through the engine capsule before reading any
    // facts. The returned dict is compact state-derived sample data, not a dense
    // tensor and not a training target.
    let state = super::state::state_from_py_state(py, state)?;
    let center = super::encoding::model1_crop_center(&state);

    let dict = PyDict::new(py);
    dict.set_item("game_id", game_id)?;
    dict.set_item("turn_index", turn_index)?;
    dict.set_item("current_player", player_label(state.current_player()))?;
    dict.set_item("phase", phase_label(state.phase()))?;
    dict.set_item("center", (center.q, center.r))?;
    dict.set_item("stones", stones_obj(py, &state)?)?;
    dict.set_item("legal_action_ids", legal_action_ids_obj(py, &state, center)?)?;
    dict.set_item("placement_history", placement_history_obj(py, &state)?)?;
    dict.set_item("first_stone", first_stone_tuple(state.phase()))?;

    let (own_hot, opponent_hot) = hot_cells(&state, state.current_player());
    dict.set_item("own_hot", coord_list_obj(py, &own_hot)?)?;
    dict.set_item("opponent_hot", coord_list_obj(py, &opponent_hot)?)?;
    dict.set_item("opponent_last_turn", coord_list_obj(py, &opponent_last_turn(&state))?)?;
    dict.set_item("metadata", mapping_to_dict(py, metadata)?)?;
    Ok(dict.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(model1_sample_from_state, module)?)?;
    Ok(())
}

fn stones_obj(py: Python<'_>, state: &RustHexoState) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    let mut occupied = state.board().occupied_cells().to_vec();
    occupied.sort_by_key(|coord| (coord.q, coord.r));
    for coord in occupied {
        if let Some(player) = state.board().get(coord) {
            list.append((coord.q, coord.r, player_label(player)))?;
        }
    }
    Ok(list.into_any().unbind())
}

fn legal_action_ids_obj(
    py: Python<'_>,
    state: &RustHexoState,
    center: HexCoord,
) -> PyResult<Py<PyAny>> {
    // The training policy head can represent only in-crop legal actions. The
    // engine still owns total legality; this compact sample stores the action ids
    // needed to build dense targets for the fixed crop.
    let list = PyList::empty(py);
    let mut legal = Vec::with_capacity(state.legal_move_count());
    state.write_legal_moves(&mut legal);
    for coord in legal {
        if super::encoding::model1_flat_index(coord, center).is_some() {
            list.append(pack_coord(coord))?;
        }
    }
    Ok(list.into_any().unbind())
}

fn placement_history_obj(py: Python<'_>, state: &RustHexoState) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for record in state.placement_history() {
        let (first_q, first_r) = match record.phase {
            TurnPhase::SecondStone { first } => (Some(first.q), Some(first.r)),
            TurnPhase::Opening | TurnPhase::FirstStone => (None, None),
        };
        list.append((
            record.coord.q,
            record.coord.r,
            player_label(record.player),
            phase_label(record.phase),
            record.placement_index,
            first_q,
            first_r,
        ))?;
    }
    Ok(list.into_any().unbind())
}

fn coord_list_obj(py: Python<'_>, coords: &[HexCoord]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for coord in coords {
        list.append((coord.q, coord.r))?;
    }
    Ok(list.into_any().unbind())
}

fn hot_cells(state: &RustHexoState, current_player: Player) -> (Vec<HexCoord>, Vec<HexCoord>) {
    if state.placements_made() < 7 {
        return (Vec::new(), Vec::new());
    }

    let mut own = Vec::new();
    let mut opponent = Vec::new();
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
        let target = if player == current_player {
            &mut own
        } else {
            &mut opponent
        };
        let key = entry.key();
        let (dq, dr) = axis_delta(key.axis);
        for index in 0..6 {
            let coord = HexCoord {
                q: key.start.q + dq * index,
                r: key.start.r + dr * index,
            };
            if state.board().is_cell_empty(coord) {
                target.push(coord);
            }
        }
    }
    sort_dedup_coords(&mut own);
    sort_dedup_coords(&mut opponent);
    (own, opponent)
}

fn opponent_last_turn(state: &RustHexoState) -> Vec<HexCoord> {
    let opponent = state.current_player().other();
    for record in state.placement_history().iter().rev() {
        if record.player != opponent {
            continue;
        }
        match record.phase {
            TurnPhase::SecondStone { first } => {
                return vec![first, record.coord];
            }
            TurnPhase::Opening => {
                return vec![record.coord];
            }
            TurnPhase::FirstStone => {}
        }
    }
    Vec::new()
}

fn sort_dedup_coords(coords: &mut Vec<HexCoord>) {
    coords.sort_by_key(|coord| (coord.q, coord.r));
    coords.dedup();
}

fn mapping_to_dict<'py>(
    py: Python<'py>,
    mapping: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    if mapping.is_none() {
        return Ok(dict);
    }
    if !mapping.hasattr("items")? {
        return Err(PyValueError::new_err("metadata must be a mapping"));
    }
    let items = mapping.call_method0("items")?;
    for item in items.try_iter()? {
        let item = item?;
        dict.set_item(item.get_item(0)?, item.get_item(1)?)?;
    }
    Ok(dict)
}

fn first_stone_tuple(phase: TurnPhase) -> Option<(i16, i16)> {
    match phase {
        TurnPhase::SecondStone { first } => Some((first.q, first.r)),
        TurnPhase::Opening | TurnPhase::FirstStone => None,
    }
}

fn player_label(player: Player) -> &'static str {
    match player {
        Player::Player0 => "player0",
        Player::Player1 => "player1",
    }
}

fn phase_label(phase: TurnPhase) -> &'static str {
    match phase {
        TurnPhase::Opening => "Opening",
        TurnPhase::FirstStone => "FirstStone",
        TurnPhase::SecondStone { .. } => "SecondStone",
    }
}

fn axis_delta(axis: Axis) -> (i16, i16) {
    match axis {
        Axis::Q => (1, 0),
        Axis::R => (0, 1),
        Axis::QR => (1, -1),
    }
}
