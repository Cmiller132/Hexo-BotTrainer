//! PyO3 bridge for the Hexo rules engine.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::{
    apply_placement, pack_coord, Axis, GameOutcome, HexCoord, HexoState as RustHexoState,
    MoveError, Placement, Player, TurnPhase,
};

const MODEL1_BOARD_SIZE: usize = 41;
const MODEL1_BOARD_AREA: usize = MODEL1_BOARD_SIZE * MODEL1_BOARD_SIZE;
const MODEL1_INPUT_CHANNELS: usize = 13;
const MODEL1_PLANE_OWN_STONES: usize = 0;
const MODEL1_PLANE_OPPONENT_STONES: usize = 1;
const MODEL1_PLANE_EMPTY: usize = 2;
const MODEL1_PLANE_LEGAL: usize = 3;
const MODEL1_PLANE_SECOND_PLACEMENT: usize = 4;
const MODEL1_PLANE_FIRST_STONE: usize = 5;
const MODEL1_PLANE_PLAYER_COLOUR: usize = 6;
const MODEL1_PLANE_OWN_RECENCY: usize = 7;
const MODEL1_PLANE_OPPONENT_RECENCY: usize = 8;
const MODEL1_PLANE_OPPONENT_HOT: usize = 9;
const MODEL1_PLANE_OWN_HOT: usize = 10;
const MODEL1_PLANE_CENTER_DISTANCE: usize = 11;
const MODEL1_PLANE_OPPONENT_LAST_TURN: usize = 12;
static MODEL1_BASE_PLANES: OnceLock<Vec<f32>> = OnceLock::new();

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
pub fn legal_action_ids(py: Python<'_>, state: PyRef<'_, PyHexoState>) -> PyResult<Py<PyAny>> {
    let mut actions = Vec::with_capacity(state.state.legal_move_count());
    state.state.write_legal_action_ids(&mut actions);
    Ok(PyTuple::new(py, actions)?.into_any().unbind())
}

#[pyfunction]
pub fn legal_action_count(state: PyRef<'_, PyHexoState>) -> usize {
    state.state.legal_move_count()
}

#[pyfunction]
pub fn is_legal_action(state: PyRef<'_, PyHexoState>, q: i16, r: i16) -> bool {
    crate::is_legal_placement(&state.state, HexCoord { q, r }).is_ok()
}

#[pyfunction]
pub fn apply_action(
    py: Python<'_>,
    mut state: PyRefMut<'_, PyHexoState>,
    q: i16,
    r: i16,
) -> PyResult<Py<PyAny>> {
    let result = apply_placement(
        &mut state.state,
        Placement {
            coord: HexCoord { q, r },
        },
    )
    .map_err(move_error)?;

    let dict = PyDict::new(py);
    dict.set_item("terminal", result.outcome.is_some())?;
    dict.set_item(
        "next_player",
        result
            .outcome
            .is_none()
            .then(|| player_label(state.state.current_player())),
    )?;

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
    dict.set_item(
        "placement_history",
        placement_history_obj(py, &state.state)?,
    )?;
    dict.set_item("first_stone", first_stone_obj(py, state.state.phase())?)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
pub fn model1_batch_inputs(py: Python<'_>, states: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mut planes: Vec<f32> = Vec::new();
    let legal_action_rows = PyList::empty(py);
    let legal_flat_rows = PyList::empty(py);
    let centers = PyList::empty(py);
    let mut state_count = 0usize;

    for item in states.try_iter()? {
        let item = item?;
        let state_ref = item.extract::<PyRef<'_, PyHexoState>>()?;
        let encoded = encode_model1_state(&state_ref.state);
        planes.extend_from_slice(&encoded.planes);
        legal_action_rows.append(PyTuple::new(py, encoded.legal_action_ids)?)?;
        legal_flat_rows.append(PyTuple::new(py, encoded.legal_flat_indices)?)?;
        centers.append((encoded.center.q, encoded.center.r))?;
        state_count += 1;
    }

    let byte_len = planes.len() * std::mem::size_of::<f32>();
    let bytes = unsafe { std::slice::from_raw_parts(planes.as_ptr() as *const u8, byte_len) };
    let dict = PyDict::new(py);
    dict.set_item("inputs", PyBytes::new(py, bytes))?;
    dict.set_item(
        "shape",
        (
            state_count,
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

#[pyfunction(signature = (states, visits, c_puct, temperature, seed, evaluator, virtual_batch_size=None))]
pub fn model1_batched_mcts(
    py: Python<'_>,
    states: &Bound<'_, PyAny>,
    visits: u32,
    c_puct: f32,
    temperature: f32,
    seed: u64,
    evaluator: &Bound<'_, PyAny>,
    virtual_batch_size: Option<u32>,
) -> PyResult<Py<PyAny>> {
    let mut roots = Vec::new();
    for item in states.try_iter()? {
        let item = item?;
        let state_ref = item.extract::<PyRef<'_, PyHexoState>>()?;
        roots.push(state_ref.state.clone());
    }
    if roots.is_empty() {
        return Ok(PyTuple::empty(py).into_any().unbind());
    }

    let mut evaluation_cache: HashMap<Vec<u32>, RustEvaluation> = HashMap::new();
    let root_evals = evaluate_model1_states_cached(py, evaluator, &roots, &mut evaluation_cache)?;
    let mut searches: Vec<RustSearch> = roots
        .iter()
        .zip(root_evals.iter())
        .map(|(state, eval)| RustSearch::new(state, eval))
        .collect();

    let target_visits = visits.max(1);
    let leaf_batch_per_root = virtual_batch_size.unwrap_or(target_visits).max(1);
    let mut completed = vec![0u32; searches.len()];
    while completed.iter().any(|count| *count < target_visits) {
        let mut leaves = Vec::new();
        let mut immediate = Vec::new();
        let mut made_progress = false;

        for root_index in 0..searches.len() {
            if completed[root_index] >= target_visits || searches[root_index].root_edges_empty() {
                continue;
            }
            let budget = leaf_batch_per_root.min(target_visits - completed[root_index]);
            for _ in 0..budget {
                let selected = searches[root_index].select_pending_leaf(c_puct)?;
                let Some(selected) = selected else {
                    break;
                };
                searches[root_index].apply_virtual_visit(&selected.path);
                completed[root_index] += 1;
                made_progress = true;
                if let Some(outcome) = selected.terminal {
                    let leaf_player = selected.state.current_player();
                    let leaf_value = terminal_value(outcome, leaf_player);
                    immediate.push((root_index, selected.path, leaf_player, leaf_value));
                } else if let Some(node_id) = selected.existing_node {
                    let node = &searches[root_index].nodes[node_id];
                    immediate.push((root_index, selected.path, node.player, node.value()));
                } else {
                    searches[root_index].mark_pending(selected.parent_node, selected.edge_index, 1);
                    leaves.push(RustLeaf {
                        root_index,
                        parent_node: selected.parent_node,
                        edge_index: selected.edge_index,
                        path: selected.path,
                        state: selected.state,
                    });
                }
            }
        }

        for (root_index, path, leaf_player, leaf_value) in immediate {
            searches[root_index].backup_virtual(&path, leaf_player, leaf_value);
        }

        if !leaves.is_empty() {
            let leaf_states: Vec<_> = leaves.iter().map(|leaf| leaf.state.clone()).collect();
            let evaluations =
                evaluate_model1_states_cached(py, evaluator, &leaf_states, &mut evaluation_cache)?;
            for (leaf, evaluation) in leaves.into_iter().zip(evaluations.iter()) {
                let search = &mut searches[leaf.root_index];
                let child_id = search.add_node_from_eval(&leaf.state, evaluation);
                search.nodes[leaf.parent_node].edges[leaf.edge_index].child = Some(child_id);
                search.mark_pending(leaf.parent_node, leaf.edge_index, -1);
                let child_player = search.nodes[child_id].player;
                let child_value = search.nodes[child_id].value();
                search.backup_virtual(&leaf.path, child_player, child_value);
            }
        }

        if !made_progress {
            break;
        }
    }

    let results = PyList::empty(py);
    for (index, search) in searches.iter().enumerate() {
        let result = PyDict::new(py);
        let root = &search.nodes[0];
        let policy_total: u32 = root.edges.iter().map(|edge| edge.visits).sum();
        let policy = PyList::empty(py);
        for edge in &root.edges {
            let weight = if policy_total > 0 {
                edge.visits as f32 / policy_total as f32
            } else {
                edge.prior
            };
            policy.append((edge.action_id, weight))?;
        }
        let selected = select_root_action(root, temperature, seed.wrapping_add(index as u64));
        result.set_item("action_id", selected.unwrap_or(0))?;
        result.set_item("visit_policy", policy)?;
        result.set_item("root_value", root.value())?;
        result.set_item("visits", policy_total)?;
        results.append(result)?;
    }

    Ok(results.into_any().unbind())
}

#[pyfunction]
pub fn action_id(q: i16, r: i16) -> u32 {
    pack_coord(HexCoord { q, r })
}

#[pyfunction]
pub fn engine_metadata(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("engine_api", true)?;
    dict.set_item("backend", "rust-pyo3")?;
    dict.set_item(
        "rules_version",
        RustHexoState::new().snapshot().rules_version(),
    )?;
    Ok(dict.into_any().unbind())
}

#[pymodule]
pub fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHexoState>()?;
    module.add_function(wrap_pyfunction!(new_game, module)?)?;
    module.add_function(wrap_pyfunction!(clone_state, module)?)?;
    module.add_function(wrap_pyfunction!(current_player, module)?)?;
    module.add_function(wrap_pyfunction!(legal_action_ids, module)?)?;
    module.add_function(wrap_pyfunction!(legal_action_count, module)?)?;
    module.add_function(wrap_pyfunction!(is_legal_action, module)?)?;
    module.add_function(wrap_pyfunction!(apply_action, module)?)?;
    module.add_function(wrap_pyfunction!(terminal, module)?)?;
    module.add_function(wrap_pyfunction!(to_python_state, module)?)?;
    module.add_function(wrap_pyfunction!(model1_batch_inputs, module)?)?;
    module.add_function(wrap_pyfunction!(model1_batched_mcts, module)?)?;
    module.add_function(wrap_pyfunction!(action_id, module)?)?;
    module.add_function(wrap_pyfunction!(engine_metadata, module)?)?;
    Ok(())
}

struct Model1EncodedState {
    planes: Vec<f32>,
    legal_action_ids: Vec<u32>,
    legal_flat_indices: Vec<i64>,
    center: HexCoord,
}

#[derive(Clone, Debug)]
struct RustEvaluation {
    value: f32,
    priors: Vec<(u32, f32)>,
}

#[derive(Clone, Debug)]
struct RustEdge {
    action_id: u32,
    action: HexCoord,
    prior: f32,
    visits: u32,
    value_sum: f32,
    pending: u32,
    child: Option<usize>,
}

impl RustEdge {
    fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

#[derive(Clone, Debug)]
struct RustNode {
    player: Player,
    visits: u32,
    value_sum: f32,
    edges: Vec<RustEdge>,
}

impl RustNode {
    fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

#[derive(Clone, Debug)]
struct RustSearch {
    root_state: RustHexoState,
    nodes: Vec<RustNode>,
}

struct RustSelectedLeaf {
    path: Vec<(usize, usize)>,
    state: RustHexoState,
    parent_node: usize,
    edge_index: usize,
    terminal: Option<GameOutcome>,
    existing_node: Option<usize>,
}

struct RustLeaf {
    root_index: usize,
    parent_node: usize,
    edge_index: usize,
    path: Vec<(usize, usize)>,
    state: RustHexoState,
}

impl RustSearch {
    fn new(root_state: &RustHexoState, evaluation: &RustEvaluation) -> Self {
        Self {
            root_state: root_state.clone(),
            nodes: vec![node_from_evaluation(root_state, evaluation)],
        }
    }

    fn root_edges_empty(&self) -> bool {
        self.nodes[0].edges.is_empty()
    }

    fn add_node_from_eval(&mut self, state: &RustHexoState, evaluation: &RustEvaluation) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node_from_evaluation(state, evaluation));
        id
    }

    fn select_pending_leaf(&self, c_puct: f32) -> PyResult<Option<RustSelectedLeaf>> {
        let mut state = self.root_state.clone();
        let mut node_id = 0usize;
        let mut path = Vec::new();
        let mut last_parent = None;
        let mut last_edge = None;

        loop {
            let node = &self.nodes[node_id];
            if node.edges.is_empty() {
                let Some(parent_node) = last_parent else {
                    return Ok(None);
                };
                let edge_index = last_edge.expect("edge index exists with parent");
                return Ok(Some(RustSelectedLeaf {
                    path,
                    state,
                    parent_node,
                    edge_index,
                    terminal: None,
                    existing_node: Some(node_id),
                }));
            }

            let Some(edge_index) = self.select_edge(node_id, c_puct) else {
                return Ok(None);
            };
            let edge = &node.edges[edge_index];
            if edge.pending > 0 && edge.child.is_none() {
                return Ok(None);
            }
            let action = edge.action;
            let child = edge.child;
            apply_placement(&mut state, Placement { coord: action }).map_err(move_error)?;
            path.push((node_id, edge_index));
            last_parent = Some(node_id);
            last_edge = Some(edge_index);

            if let Some(child_id) = child {
                node_id = child_id;
                continue;
            }

            return Ok(Some(RustSelectedLeaf {
                path,
                state: state.clone(),
                parent_node: node_id,
                edge_index,
                terminal: state.terminal(),
                existing_node: None,
            }));
        }
    }

    fn select_edge(&self, node_id: usize, c_puct: f32) -> Option<usize> {
        let node = &self.nodes[node_id];
        let exploration_scale = c_puct * (node.visits.max(1) as f32).sqrt();
        let mut best: Option<(usize, f32, u32, u32)> = None;
        for (index, edge) in node.edges.iter().enumerate() {
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            let score = edge.value() + edge.prior * exploration_scale / (1.0 + edge.visits as f32);
            let candidate = (index, score, edge.visits, edge.action_id);
            let replace = best.is_none_or(|current| compare_edge_score(candidate, current) == Ordering::Greater);
            if replace {
                best = Some(candidate);
            }
        }
        best.map(|item| item.0)
    }

    fn apply_virtual_visit(&mut self, path: &[(usize, usize)]) {
        for &(node_id, edge_index) in path {
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].edges[edge_index].visits += 1;
        }
    }

    fn backup_virtual(&mut self, path: &[(usize, usize)], leaf_player: Player, leaf_value: f32) {
        for &(node_id, edge_index) in path {
            let value = if self.nodes[node_id].player == leaf_player {
                leaf_value
            } else {
                -leaf_value
            };
            self.nodes[node_id].value_sum += value;
            self.nodes[node_id].edges[edge_index].value_sum += value;
        }
    }

    fn mark_pending(&mut self, node_id: usize, edge_index: usize, delta: i32) {
        let edge = &mut self.nodes[node_id].edges[edge_index];
        if delta >= 0 {
            edge.pending = edge.pending.saturating_add(delta as u32);
        } else {
            edge.pending = edge.pending.saturating_sub((-delta) as u32);
        }
    }
}

fn evaluate_model1_states(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[RustHexoState],
) -> PyResult<Vec<RustEvaluation>> {
    let encoded: Vec<_> = states.iter().map(encode_model1_state).collect();
    let mut planes = Vec::with_capacity(encoded.len() * MODEL1_INPUT_CHANNELS * MODEL1_BOARD_AREA);
    let mut legal_flat_indices = Vec::new();
    let mut legal_row_offsets = Vec::with_capacity(encoded.len() + 1);
    legal_row_offsets.push(0i64);
    for row in &encoded {
        planes.extend_from_slice(&row.planes);
        legal_flat_indices.extend_from_slice(&row.legal_flat_indices);
        legal_row_offsets.push(legal_flat_indices.len() as i64);
    }

    let byte_len = planes.len() * std::mem::size_of::<f32>();
    let bytes = unsafe { std::slice::from_raw_parts(planes.as_ptr() as *const u8, byte_len) };
    let flat_byte_len = legal_flat_indices.len() * std::mem::size_of::<i64>();
    let flat_bytes = unsafe {
        std::slice::from_raw_parts(legal_flat_indices.as_ptr() as *const u8, flat_byte_len)
    };
    let payload = PyDict::new(py);
    payload.set_item("inputs", PyBytes::new(py, bytes))?;
    payload.set_item(
        "shape",
        (
            encoded.len(),
            MODEL1_INPUT_CHANNELS,
            MODEL1_BOARD_SIZE,
            MODEL1_BOARD_SIZE,
        ),
    )?;
    payload.set_item("legal_flat_indices_bytes", PyBytes::new(py, flat_bytes))?;
    payload.set_item("legal_row_offsets", PyTuple::new(py, legal_row_offsets)?)?;

    let output = evaluator.call1((payload,))?;
    if let (Ok(values_obj), Ok(priors_obj)) = (
        output.get_item("values_bytes"),
        output.get_item("priors_bytes"),
    ) {
        let value_bytes = values_obj.downcast::<PyBytes>()?.as_bytes();
        let prior_bytes = priors_obj.downcast::<PyBytes>()?.as_bytes();
        let mut evaluations = Vec::with_capacity(encoded.len());
        let mut prior_offset = 0usize;
        for (index, row) in encoded.iter().enumerate() {
            let value = read_f32(value_bytes, index).unwrap_or(0.0).clamp(-1.0, 1.0);
            let mut row_priors = Vec::with_capacity(row.legal_action_ids.len());
            for action_id in row.legal_action_ids.iter().copied() {
                let prior = read_f32(prior_bytes, prior_offset).unwrap_or(0.0).max(0.0);
                row_priors.push((action_id, prior));
                prior_offset += 1;
            }
            evaluations.push(RustEvaluation {
                value,
                priors: row_priors,
            });
        }
        return Ok(evaluations);
    }

    let values = output.get_item("values")?;
    let priors = output.get_item("priors")?;
    let mut evaluations = Vec::with_capacity(encoded.len());
    for (index, row) in encoded.iter().enumerate() {
        let value = values.get_item(index)?.extract::<f32>()?.clamp(-1.0, 1.0);
        let prior_row = priors.get_item(index)?;
        let mut row_priors = Vec::with_capacity(row.legal_action_ids.len());
        for (action_id, prior_item) in row.legal_action_ids.iter().copied().zip(prior_row.try_iter()?) {
            row_priors.push((action_id, prior_item?.extract::<f32>()?.max(0.0)));
        }
        evaluations.push(RustEvaluation {
            value,
            priors: row_priors,
        });
    }
    Ok(evaluations)
}

fn evaluate_model1_states_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[RustHexoState],
    cache: &mut HashMap<Vec<u32>, RustEvaluation>,
) -> PyResult<Vec<RustEvaluation>> {
    let mut result_slots: Vec<Option<RustEvaluation>> = vec![None; states.len()];
    let mut unique_states: Vec<RustHexoState> = Vec::new();
    let mut unique_keys: Vec<Vec<u32>> = Vec::new();
    let mut unique_index_by_key: HashMap<Vec<u32>, usize> = HashMap::new();

    for (index, state) in states.iter().enumerate() {
        let key = model1_state_cache_key(state);
        if let Some(cached) = cache.get(&key) {
            result_slots[index] = Some(cached.clone());
            continue;
        }
        if unique_index_by_key.contains_key(&key) {
            continue;
        }
        unique_index_by_key.insert(key.clone(), unique_states.len());
        unique_keys.push(key);
        unique_states.push(state.clone());
    }

    if !unique_states.is_empty() {
        let unique_evals = evaluate_model1_states(py, evaluator, &unique_states)?;
        for (key, evaluation) in unique_keys.into_iter().zip(unique_evals.into_iter()) {
            cache.insert(key.clone(), evaluation.clone());
        }
        for (index, state) in states.iter().enumerate() {
            if result_slots[index].is_some() {
                continue;
            }
            let key = model1_state_cache_key(state);
            if let Some(evaluation) = cache.get(&key) {
                result_slots[index] = Some(evaluation.clone());
            }
        }
    }

    Ok(result_slots
        .into_iter()
        .map(|item| item.expect("every model1 evaluation slot must be populated"))
        .collect())
}

fn model1_state_cache_key(state: &RustHexoState) -> Vec<u32> {
    state
        .placement_history()
        .iter()
        .map(|record| pack_coord(record.coord))
        .collect()
}

fn read_f32(bytes: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    let chunk = bytes.get(start..start + 4)?;
    Some(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn node_from_evaluation(state: &RustHexoState, evaluation: &RustEvaluation) -> RustNode {
    let mut edges: Vec<_> = evaluation
        .priors
        .iter()
        .map(|(action_id, prior)| RustEdge {
            action_id: *action_id,
            action: unpack_packed_coord(*action_id),
            prior: prior.max(0.0),
            visits: 0,
            value_sum: 0.0,
            pending: 0,
            child: None,
        })
        .collect();
    normalize_rust_priors(&mut edges);
    RustNode {
        player: state.current_player(),
        visits: 1,
        value_sum: evaluation.value,
        edges,
    }
}

fn normalize_rust_priors(edges: &mut [RustEdge]) {
    let total: f32 = edges.iter().map(|edge| edge.prior.max(0.0)).sum();
    if total <= f32::EPSILON || !total.is_finite() {
        let prior = if edges.is_empty() {
            0.0
        } else {
            1.0 / edges.len() as f32
        };
        for edge in edges {
            edge.prior = prior;
        }
        return;
    }
    for edge in edges {
        edge.prior = edge.prior.max(0.0) / total;
    }
}

fn compare_edge_score(left: (usize, f32, u32, u32), right: (usize, f32, u32, u32)) -> Ordering {
    left.1
        .partial_cmp(&right.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| right.2.cmp(&left.2))
        .then_with(|| right.3.cmp(&left.3))
}

fn terminal_value(outcome: GameOutcome, player: Player) -> f32 {
    if outcome.winner == player {
        1.0
    } else {
        -1.0
    }
}

fn select_root_action(node: &RustNode, temperature: f32, seed: u64) -> Option<u32> {
    if node.edges.is_empty() {
        return None;
    }
    if temperature <= 1.0e-6 {
        return node
            .edges
            .iter()
            .max_by_key(|edge| (edge.visits, std::cmp::Reverse(edge.action_id)))
            .map(|edge| edge.action_id);
    }
    let inv_temperature = 1.0 / temperature.max(1.0e-3);
    let mut total = 0.0f64;
    let mut weights = Vec::with_capacity(node.edges.len());
    for edge in &node.edges {
        let weight = (edge.visits.max(1) as f32).powf(inv_temperature) as f64;
        total += weight;
        weights.push(weight);
    }
    let mut threshold = random_unit(seed) * total;
    for (edge, weight) in node.edges.iter().zip(weights) {
        threshold -= weight;
        if threshold <= 0.0 {
            return Some(edge.action_id);
        }
    }
    node.edges.last().map(|edge| edge.action_id)
}

fn random_unit(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn unpack_packed_coord(action_id: u32) -> HexCoord {
    HexCoord {
        q: ((action_id >> 16) as i32 - 32768) as i16,
        r: ((action_id & 0xFFFF) as i32 - 32768) as i16,
    }
}

fn encode_model1_state(state: &RustHexoState) -> Model1EncodedState {
    let center = model1_crop_center(state);
    let mut planes = model1_base_planes().to_vec();

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
    let mut legal_action_ids = Vec::with_capacity(legal_coords.len());
    let mut legal_flat_indices = Vec::with_capacity(legal_coords.len());
    for coord in legal_coords {
        if let Some(flat) = model1_flat_index(coord, center) {
            set_plane(&mut planes, MODEL1_PLANE_LEGAL, flat, 1.0);
            legal_action_ids.push(pack_coord(coord));
            legal_flat_indices.push(flat as i64);
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

fn model1_crop_center(state: &RustHexoState) -> HexCoord {
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

fn model1_flat_index(coord: HexCoord, center: HexCoord) -> Option<usize> {
    let half = (MODEL1_BOARD_SIZE / 2) as i32;
    let row = coord.r as i32 - center.r as i32 + half;
    let col = coord.q as i32 - center.q as i32 + half;
    if row < 0 || col < 0 || row >= MODEL1_BOARD_SIZE as i32 || col >= MODEL1_BOARD_SIZE as i32 {
        return None;
    }
    Some(row as usize * MODEL1_BOARD_SIZE + col as usize)
}

fn fill_distance_plane(planes: &mut [f32]) {
    let half = (MODEL1_BOARD_SIZE / 2) as i32;
    for row in 0..MODEL1_BOARD_SIZE {
        for col in 0..MODEL1_BOARD_SIZE {
            let r = row as i32 - half;
            let q = col as i32 - half;
            let s = -r - q;
            let distance = r.abs().max(q.abs()).max(s.abs()) as f32 / (MODEL1_BOARD_SIZE - 1) as f32;
            set_plane(
                planes,
                MODEL1_PLANE_CENTER_DISTANCE,
                row * MODEL1_BOARD_SIZE + col,
                distance,
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

fn plane_offset(plane: usize, flat: usize) -> usize {
    plane * MODEL1_BOARD_AREA + flat
}

fn axis_delta(axis: Axis) -> (i16, i16) {
    match axis {
        Axis::Q => (1, 0),
        Axis::R => (0, 1),
        Axis::QR => (1, -1),
    }
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
    state.write_legal_moves(&mut legal_coords);
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
            (entry.mask(Player::Player0), entry.mask(Player::Player1)),
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
