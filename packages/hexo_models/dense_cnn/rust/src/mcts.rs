//! Dense CNN MCTS Python boundary.
//!
//! Tree mechanics live in `mcts_tree`, and evaluator payload parsing lives in
//! `mcts_eval`. This module wires those pieces into the PyO3 functions consumed
//! by dense CNN self-play.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;

use hexo_engine::PackedCoord;

use crate::mcts_eval::{evaluate_model1_states_cached, RustEvaluation};
use crate::mcts_tree::{select_root_action, terminal_value, RustLeaf, RustSearch};
use crate::state::states_from_history_rows;

#[pyfunction(signature = (history_rows, visits, c_puct, temperature, seed, evaluator, virtual_batch_size=None))]
pub fn model1_batched_mcts(
    py: Python<'_>,
    history_rows: &Bound<'_, PyAny>,
    visits: u32,
    c_puct: f32,
    temperature: f32,
    seed: u64,
    evaluator: &Bound<'_, PyAny>,
    virtual_batch_size: Option<u32>,
) -> PyResult<Py<PyAny>> {
    let roots = states_from_history_rows(history_rows)?;
    if roots.is_empty() {
        return Ok(PyTuple::empty(py).into_any().unbind());
    }

    let mut evaluation_cache: HashMap<Vec<PackedCoord>, RustEvaluation> = HashMap::new();
    let root_evals = evaluate_model1_states_cached(py, evaluator, &roots, &mut evaluation_cache)?;
    let mut searches: Vec<RustSearch> = roots
        .iter()
        .zip(root_evals.iter())
        .map(|(state, eval)| RustSearch::new(state, eval))
        .collect();
    if searches.iter().any(RustSearch::root_edges_empty) {
        return Err(PyValueError::new_err("MCTS root has no legal actions"));
    }

    let target_visits = visits.max(1);
    let leaf_batch_per_root = virtual_batch_size.unwrap_or(target_visits).max(1);
    let mut completed = vec![0u32; searches.len()];

    // Each outer pass gathers up to `leaf_batch_per_root` leaves per root,
    // evaluates the unique uncached leaves together, then backs results into
    // each independent root tree. This keeps Python/Torch calls coarse without
    // sharing tree state between games.
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


pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(model1_batched_mcts, module)?)?;
    Ok(())
}
