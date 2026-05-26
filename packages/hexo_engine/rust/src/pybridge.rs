//! PyO3 bridge for the Hexo rules engine.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::{
    apply_placement, legal_placements, Axis, GameOutcome, HexCoord, HexoState as RustHexoState,
    MoveError, Placement, Player, TurnPhase,
};

/// Python-owned opaque handle to a Rust Hexo state.
#[pyclass(name = "HexoState", module = "hexo_engine._rust", skip_from_py_object)]
#[derive(Clone)]
pub struct PyHexoState {
    state: RustHexoState,
}

#[pymethods]
impl PyHexoState {
    fn __repr__(&self) -> String {
        format!(
            "HexoState(placements_made={}, terminal={})",
            self.state.placements_made(),
            self.state.is_terminal()
        )
    }
}

#[pyfunction(signature = (seed=None, scenario=None))]
pub fn new_game(seed: Option<u64>, scenario: Option<Py<PyAny>>) -> PyHexoState {
    let _ = seed;
    let _ = scenario;
    PyHexoState {
        state: RustHexoState::new(),
    }
}

#[pyfunction]
pub fn clone_state(state: PyRef<'_, PyHexoState>) -> PyHexoState {
    PyHexoState {
        state: state.state.clone(),
    }
}

#[pyfunction]
pub fn current_player(state: PyRef<'_, PyHexoState>) -> &'static str {
    player_label(state.state.current_player())
}

#[pyfunction]
pub fn legal_actions(state: PyRef<'_, PyHexoState>) -> Vec<(i16, i16)> {
    let mut actions = Vec::new();
    legal_placements(&state.state, &mut actions);
    actions.sort_by_key(|coord| (coord.q, coord.r));
    actions.into_iter().map(|coord| (coord.q, coord.r)).collect()
}

#[pyfunction]
pub fn apply_action(
    py: Python<'_>,
    mut state: PyRefMut<'_, PyHexoState>,
    q: i16,
    r: i16,
) -> PyResult<Py<PyAny>> {
    let result = apply_placement(&mut state.state, Placement { coord: HexCoord { q, r } })
        .map_err(move_error)?;

    let dict = PyDict::new(py);
    dict.set_item("placed", coord_obj(py, result.placed)?)?;
    dict.set_item("player", player_label(result.player))?;
    dict.set_item("phase_before", phase_label(result.phase_before))?;
    dict.set_item("phase_after", phase_label(result.phase_after))?;
    dict.set_item("terminal", result.outcome.is_some())?;
    dict.set_item(
        "next_player",
        result
            .outcome
            .is_none()
            .then(|| player_label(state.state.current_player())),
    )?;
    dict.set_item("outcome", outcome_obj(py, result.outcome)?)?;

    let metadata = PyDict::new(py);
    metadata.set_item("placements_made", state.state.placements_made())?;
    dict.set_item("metadata", metadata)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
pub fn terminal(py: Python<'_>, state: PyRef<'_, PyHexoState>) -> PyResult<Option<Py<PyAny>>> {
    outcome_obj(py, state.state.terminal())
}

#[pyfunction]
pub fn to_python_state(py: Python<'_>, state: PyRef<'_, PyHexoState>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("board", board_obj(py, &state.state)?)?;
    dict.set_item("current_player", player_label(state.state.current_player()))?;
    dict.set_item("phase", phase_label(state.state.phase()))?;
    dict.set_item("placements_made", state.state.placements_made())?;
    dict.set_item("terminal", outcome_obj(py, state.state.terminal())?)?;
    dict.set_item("last_turn", last_turn_obj(py, &state.state)?)?;
    dict.set_item("placement_history", placement_history_obj(py, &state.state)?)?;
    dict.set_item("first_stone", first_stone_obj(py, state.state.phase())?)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
pub fn action_id(q: i16, r: i16) -> String {
    format!("{q},{r}")
}

#[pyfunction]
pub fn engine_metadata(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("engine_api", true)?;
    dict.set_item("backend", "rust-pyo3")?;
    dict.set_item("rules_version", RustHexoState::new().snapshot().rules_version())?;
    Ok(dict.into_any().unbind())
}

#[pymodule]
pub fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHexoState>()?;
    module.add_function(wrap_pyfunction!(new_game, module)?)?;
    module.add_function(wrap_pyfunction!(clone_state, module)?)?;
    module.add_function(wrap_pyfunction!(current_player, module)?)?;
    module.add_function(wrap_pyfunction!(legal_actions, module)?)?;
    module.add_function(wrap_pyfunction!(apply_action, module)?)?;
    module.add_function(wrap_pyfunction!(terminal, module)?)?;
    module.add_function(wrap_pyfunction!(to_python_state, module)?)?;
    module.add_function(wrap_pyfunction!(action_id, module)?)?;
    module.add_function(wrap_pyfunction!(engine_metadata, module)?)?;
    Ok(())
}

fn move_error(error: MoveError) -> PyErr {
    PyValueError::new_err(error.to_string())
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

fn axis_label(axis: Axis) -> &'static str {
    match axis {
        Axis::Q => "Q",
        Axis::R => "R",
        Axis::QR => "QR",
    }
}

fn coord_obj(py: Python<'_>, coord: HexCoord) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("q", coord.q)?;
    dict.set_item("r", coord.r)?;
    Ok(dict.into_any().unbind())
}

fn outcome_obj(py: Python<'_>, outcome: Option<GameOutcome>) -> PyResult<Option<Py<PyAny>>> {
    let Some(outcome) = outcome else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    dict.set_item("winner", player_label(outcome.winner))?;
    dict.set_item("reason", "six_in_line")?;
    let metadata = PyDict::new(py);
    metadata.set_item("placements", outcome.placements)?;
    dict.set_item("metadata", metadata)?;
    Ok(Some(dict.into_any().unbind()))
}

fn board_obj(py: Python<'_>, state: &RustHexoState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    let stones = PyList::empty(py);
    let mut occupied = state.board().occupied_cells().to_vec();
    occupied.sort_by_key(|coord| (coord.q, coord.r));
    for coord in occupied {
        if let Some(player) = state.board().get(coord) {
            let item = PyDict::new(py);
            item.set_item("coord", coord_obj(py, coord)?)?;
            item.set_item("player", player_label(player))?;
            stones.append(item)?;
        }
    }
    dict.set_item("stones", stones)?;

    let occupied_ordered = PyList::empty(py);
    for coord in state.board().occupied_cells() {
        occupied_ordered.append(coord_obj(py, *coord)?)?;
    }
    dict.set_item("occupied", occupied_ordered)?;

    let legal = PyList::empty(py);
    let mut legal_coords = Vec::new();
    legal_placements(state, &mut legal_coords);
    legal_coords.sort_by_key(|coord| (coord.q, coord.r));
    for coord in legal_coords {
        legal.append(coord_obj(py, coord)?)?;
    }
    dict.set_item("legal", legal)?;

    dict.set_item("windows", window_entries_obj(py, state)?)?;
    Ok(dict.into_any().unbind())
}

fn window_entries_obj(py: Python<'_>, state: &RustHexoState) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    let mut entries: Vec<_> = state.board().windows().entries().collect();
    entries.sort_by_key(|entry| {
        let key = entry.key();
        (key.axis.index(), key.start.q, key.start.r)
    });

    for entry in entries {
        let key = entry.key();
        let item = PyDict::new(py);
        item.set_item("start", coord_obj(py, key.start)?)?;
        item.set_item("axis", axis_label(key.axis))?;
        item.set_item(
            "masks",
            (
                entry.mask(Player::Player0),
                entry.mask(Player::Player1),
            ),
        )?;
        list.append(item)?;
    }
    Ok(list.into_any().unbind())
}

fn last_turn_obj(py: Python<'_>, state: &RustHexoState) -> PyResult<Option<Py<PyAny>>> {
    let Some(record) = state.last_turn() else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    dict.set_item("player", player_label(record.player))?;
    let placements = PyList::empty(py);
    for coord in &record.placements {
        placements.append(coord_obj(py, *coord)?)?;
    }
    dict.set_item("placements", placements)?;
    Ok(Some(dict.into_any().unbind()))
}

fn placement_history_obj(py: Python<'_>, state: &RustHexoState) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for record in state.placement_history() {
        let item = PyDict::new(py);
        item.set_item("player", player_label(record.player))?;
        item.set_item("coord", coord_obj(py, record.coord)?)?;
        item.set_item("phase", phase_label(record.phase))?;
        item.set_item("placement_index", record.placement_index)?;
        item.set_item("first_stone", first_stone_obj(py, record.phase)?)?;
        list.append(item)?;
    }
    Ok(list.into_any().unbind())
}

fn first_stone_obj(py: Python<'_>, phase: TurnPhase) -> PyResult<Option<Py<PyAny>>> {
    match phase {
        TurnPhase::SecondStone { first } => Ok(Some(coord_obj(py, first)?)),
        TurnPhase::Opening | TurnPhase::FirstStone => Ok(None),
    }
}
