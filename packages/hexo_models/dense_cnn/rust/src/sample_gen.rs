//! Dense CNN compact sample generation and finalization.
//!
//! Self-play records compact facts rather than dense tensors. This module owns
//! those facts for live positions and game outcomes, cloning live engine states
//! through the generic engine state capsule before reading model-specific facts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use hexo_engine::{
    pack_coord, Axis, HexCoord, HexoState as RustHexoState, PackedCoord, Player, TurnPhase,
};

const TARGET_SCHEMA_VERSION: i64 = 2;
const LOOKAHEAD_TARGET_SEMANTICS: &str = "mcts_prefix_future_root_only_v2";

#[pyfunction(
    signature = (state, game_id, turn_index, policy, value, opp_policy, lookahead, metadata)
)]
pub fn model1_sample_from_state(
    py: Python<'_>,
    state: &Bound<'_, PyAny>,
    game_id: String,
    turn_index: u32,
    policy: &Bound<'_, PyAny>,
    value: f32,
    opp_policy: &Bound<'_, PyAny>,
    lookahead: &Bound<'_, PyAny>,
    metadata: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let state = super::state::state_from_py_state(py, state)?;
    let center = super::encoding::model1_crop_center(&state);

    let dict = PyDict::new(py);
    dict.set_item("game_id", game_id)?;
    dict.set_item("turn_index", turn_index)?;
    dict.set_item("current_player", player_label(state.current_player()))?;
    dict.set_item("phase", phase_label(state.phase()))?;
    dict.set_item("center", (center.q, center.r))?;
    dict.set_item("stones", stones_obj(py, &state)?)?;
    dict.set_item(
        "legal_action_ids",
        legal_action_ids_obj(py, &state, center)?,
    )?;
    dict.set_item("placement_history", placement_history_obj(py, &state)?)?;
    dict.set_item("first_stone", first_stone_tuple(state.phase()))?;

    let (own_hot, opponent_hot) = hot_cells(&state, state.current_player());
    dict.set_item("own_hot", coord_list_obj(py, &own_hot)?)?;
    dict.set_item("opponent_hot", coord_list_obj(py, &opponent_hot)?)?;
    dict.set_item(
        "opponent_last_turn",
        coord_list_obj(py, &opponent_last_turn(&state))?,
    )?;

    dict.set_item(
        "policy",
        action_weight_pairs_obj(py, &action_weight_pairs(policy)?)?,
    )?;
    dict.set_item(
        "opp_policy",
        action_weight_pairs_obj(py, &action_weight_pairs(opp_policy)?)?,
    )?;
    dict.set_item("value", value)?;
    dict.set_item(
        "lookahead",
        lookahead_pairs_obj(py, &lookahead_pairs(lookahead)?)?,
    )?;
    dict.set_item("metadata", mapping_to_dict(py, metadata)?)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
pub fn model1_finalize_game_samples(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    winner: Option<String>,
    horizons: &Bound<'_, PyAny>,
    truncated: bool,
) -> PyResult<Py<PyAny>> {
    let decisions = pending_decisions(pending)?;
    let horizons = horizon_values(horizons)?;
    let output = PyList::empty(py);

    for index in 0..decisions.len() {
        let decision = &decisions[index];
        let value = match winner.as_deref() {
            Some(winner) if winner == decision.player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };

        let opp_policy = decisions[index + 1..]
            .iter()
            .find(|future| future.player != decision.player)
            .map(|future| future.policy.as_slice())
            .unwrap_or(&[]);

        let lookahead = finalized_lookahead(&decisions, index, &horizons);
        let sample = sample_to_dict(py, decision.sample.bind(py))?;
        sample.set_item("value", value)?;
        sample.set_item("opp_policy", action_weight_pairs_obj(py, opp_policy)?)?;
        sample.set_item("lookahead", lookahead_pairs_obj(py, &lookahead)?)?;

        let metadata_obj = sample_field(decision.sample.bind(py), "metadata")?;
        let metadata = mapping_to_dict(py, &metadata_obj)?;
        metadata.set_item(
            "opp_policy_source",
            if opp_policy.is_empty() {
                "none"
            } else {
                "future_opponent_mcts"
            },
        )?;
        if truncated {
            metadata.set_item("truncated", true)?;
            metadata.set_item("value_target_reason", "max_actions_draw")?;
        }
        metadata.set_item("target_schema_version", TARGET_SCHEMA_VERSION)?;
        metadata.set_item("lookahead_target_semantics", LOOKAHEAD_TARGET_SEMANTICS)?;
        sample.set_item("metadata", metadata)?;
        output.append(sample)?;
    }

    Ok(output.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(model1_sample_from_state, module)?)?;
    module.add_function(wrap_pyfunction!(model1_finalize_game_samples, module)?)?;
    Ok(())
}

struct PendingDecision {
    player: String,
    sample: Py<PyAny>,
    root_value: f32,
    policy: Vec<(PackedCoord, f32)>,
}

fn pending_decisions(pending: &Bound<'_, PyAny>) -> PyResult<Vec<PendingDecision>> {
    let mut decisions = Vec::new();
    for item in pending.try_iter()? {
        let item = item?;
        let player = item.get_item(0)?.extract::<String>()?;
        let sample = item.get_item(1)?;
        let root_value = item.get_item(2)?.extract::<f32>()?;
        let policy_obj = sample_field(&sample, "policy")?;
        let policy = action_weight_pairs(&policy_obj)?;
        decisions.push(PendingDecision {
            player,
            sample: sample.unbind(),
            root_value,
            policy,
        });
    }
    Ok(decisions)
}

fn finalized_lookahead(
    decisions: &[PendingDecision],
    index: usize,
    horizons: &[usize],
) -> Vec<(i64, f32)> {
    let mut lookahead = Vec::with_capacity(horizons.len());
    if decisions.is_empty() || index >= decisions.len() {
        return lookahead;
    }
    for &horizon in horizons {
        let Some(future_index) = index.checked_add(horizon) else {
            continue;
        };
        if future_index >= decisions.len() {
            continue;
        }
        let future = &decisions[future_index];
        let lookahead_value = if future.player == decisions[index].player {
            future.root_value
        } else {
            -future.root_value
        };
        lookahead.push((horizon as i64, lookahead_value));
    }
    lookahead
}

fn sample_to_dict<'py>(
    py: Python<'py>,
    sample: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    const SAMPLE_KEYS: &[&str] = &[
        "game_id",
        "turn_index",
        "current_player",
        "phase",
        "center",
        "stones",
        "legal_action_ids",
        "placement_history",
        "first_stone",
        "own_hot",
        "opponent_hot",
        "opponent_last_turn",
        "policy",
        "opp_policy",
        "value",
        "lookahead",
        "metadata",
    ];

    let dict = PyDict::new(py);
    for key in SAMPLE_KEYS {
        dict.set_item(*key, sample_field(sample, key)?)?;
    }
    Ok(dict)
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

fn action_weight_pairs_obj(py: Python<'_>, pairs: &[(PackedCoord, f32)]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for &(action_id, weight) in pairs {
        list.append((action_id, weight))?;
    }
    Ok(list.into_any().unbind())
}

fn lookahead_pairs_obj(py: Python<'_>, pairs: &[(i64, f32)]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for &(horizon, value) in pairs {
        list.append((horizon, value))?;
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

fn action_weight_pairs(weights: &Bound<'_, PyAny>) -> PyResult<Vec<(PackedCoord, f32)>> {
    if weights.is_none() {
        return Ok(Vec::new());
    }
    if let Some(compact) = compact_action_weight_pairs(weights)? {
        return Ok(compact);
    }
    let iterable = if weights.hasattr("items")? {
        weights.call_method0("items")?
    } else {
        weights.clone()
    };
    let mut pairs = Vec::new();
    for item in iterable.try_iter()? {
        let item = item?;
        pairs.push((
            item.get_item(0)?.extract::<PackedCoord>()?,
            item.get_item(1)?.extract::<f32>()?,
        ));
    }
    Ok(pairs)
}

fn compact_action_weight_pairs(
    weights: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<(PackedCoord, f32)>>> {
    if !weights.hasattr("action_ids_bytes")? || !weights.hasattr("weights_bytes")? {
        return Ok(None);
    }
    let action_obj = weights.getattr("action_ids_bytes")?;
    let weight_obj = weights.getattr("weights_bytes")?;
    let action_bytes = action_obj.downcast::<PyBytes>()?.as_bytes();
    let weight_bytes = weight_obj.downcast::<PyBytes>()?.as_bytes();
    if action_bytes.len() % 4 != 0 || weight_bytes.len() % 4 != 0 {
        return Err(PyValueError::new_err(
            "compact visit policy byte lengths must be multiples of four",
        ));
    }
    let count = action_bytes.len() / 4;
    if weight_bytes.len() / 4 != count {
        return Err(PyValueError::new_err(
            "compact visit policy action and weight counts differ",
        ));
    }
    let mut pairs = Vec::with_capacity(count);
    for index in 0..count {
        pairs.push((read_u32(action_bytes, index), read_f32(weight_bytes, index)));
    }
    Ok(Some(pairs))
}

fn read_u32(bytes: &[u8], index: usize) -> u32 {
    let start = index * 4;
    let chunk = &bytes[start..start + 4];
    u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
}

fn read_f32(bytes: &[u8], index: usize) -> f32 {
    let start = index * 4;
    let chunk = &bytes[start..start + 4];
    f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
}

fn lookahead_pairs(lookahead: &Bound<'_, PyAny>) -> PyResult<Vec<(i64, f32)>> {
    if lookahead.is_none() {
        return Ok(Vec::new());
    }
    let iterable = if lookahead.hasattr("items")? {
        lookahead.call_method0("items")?
    } else {
        lookahead.clone()
    };
    let mut pairs = Vec::new();
    for item in iterable.try_iter()? {
        let item = item?;
        pairs.push((
            item.get_item(0)?.extract::<i64>()?,
            item.get_item(1)?.extract::<f32>()?,
        ));
    }
    Ok(pairs)
}

fn horizon_values(horizons: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if horizons.is_none() {
        return Ok(Vec::new());
    }
    let mut values = Vec::new();
    for item in horizons.try_iter()? {
        let horizon = item?.extract::<i64>()?;
        if horizon < 0 {
            return Err(PyValueError::new_err(
                "lookahead horizons must be non-negative",
            ));
        }
        values.push(horizon as usize);
    }
    Ok(values)
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

fn sample_field<'py>(sample: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(value) = sample.get_item(key) {
        return Ok(value);
    }
    sample.getattr(key)
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
