//! Hexformer AR MCTS Python boundary.
//!
//! The heavy search mechanics live in `mcts_tree`, and evaluator payload parsing
//! lives in `mcts_eval`. This module wires those pieces into the PyO3 functions
//! consumed by Python self-play.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;

use crate::engine_state::clone_py_engine_states;
use crate::mcts_eval::evaluate_states_cached;
use crate::mcts_tree::{select_root_action, terminal_value, RustLeaf, RustSearch};
use crate::sample_gen::{ArchitectureConfig, CandidateConfig};

#[pyfunction(signature = (root_state, visits, c_puct, temperature, seed, evaluator, architecture, candidates))]
pub fn hexformer_ar_mcts(
    py: Python<'_>,
    root_state: &Bound<'_, PyAny>,
    visits: u32,
    c_puct: f32,
    temperature: f32,
    seed: u64,
    evaluator: &Bound<'_, PyAny>,
    architecture: &Bound<'_, PyAny>,
    candidates: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let root_states = PyTuple::new(py, [root_state])?;
    let results = hexformer_ar_batched_mcts(
        py,
        root_states.as_any(),
        visits,
        c_puct,
        temperature,
        seed,
        evaluator,
        architecture,
        candidates,
        Some(visits.max(1)),
    )?;
    let results = results.bind(py);
    Ok(results.get_item(0)?.unbind())
}

#[pyfunction(signature = (root_states, visits, c_puct, temperature, seed, evaluator, architecture, candidates, virtual_batch_size=None))]
pub fn hexformer_ar_batched_mcts(
    py: Python<'_>,
    root_states: &Bound<'_, PyAny>,
    visits: u32,
    c_puct: f32,
    temperature: f32,
    seed: u64,
    evaluator: &Bound<'_, PyAny>,
    architecture: &Bound<'_, PyAny>,
    candidates: &Bound<'_, PyAny>,
    virtual_batch_size: Option<u32>,
) -> PyResult<Py<PyAny>> {
    let roots = clone_py_engine_states(py, root_states)?;
    if roots.is_empty() {
        return Ok(PyTuple::empty(py).into_any().unbind());
    }
    if roots
        .iter()
        .any(|state| state.terminal().is_some() || state.legal_move_count() == 0)
    {
        return Err(PyValueError::new_err(
            "Hexformer MCTS root has no legal candidate actions",
        ));
    }

    let arch = ArchitectureConfig::from_py(architecture)?;
    let candidate_cfg = CandidateConfig::from_py(candidates)?;
    let mut evaluation_cache = HashMap::new();
    let root_evals = evaluate_states_cached(
        py,
        evaluator,
        &roots,
        &arch,
        &candidate_cfg,
        &mut evaluation_cache,
    )?;
    let mut searches: Vec<RustSearch> = roots
        .iter()
        .zip(root_evals.iter())
        .map(|(state, eval)| RustSearch::new(state, eval))
        .collect();
    if searches.iter().any(RustSearch::root_edges_empty) {
        return Err(PyValueError::new_err(
            "Hexformer MCTS root has no legal candidate actions",
        ));
    }

    let target_visits = visits.max(1);
    let leaf_batch_per_root = virtual_batch_size.unwrap_or(target_visits).max(1);
    let mut completed = vec![0u32; searches.len()];

    // Batch MCTS by collecting pending leaves from each root, evaluating the
    // uncached states in one Python callback, and backing values into only the
    // tree that produced each leaf.
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
                let Some(selected) = searches[root_index].select_pending_leaf(c_puct)? else {
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
                    searches[root_index].mark_pending(
                        selected.parent_node,
                        selected.edge_index,
                        1,
                    );
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
            let evaluations = evaluate_states_cached(
                py,
                evaluator,
                &leaf_states,
                &arch,
                &candidate_cfg,
                &mut evaluation_cache,
            )?;
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
        result.set_item("root_candidate_count", root.edges.len())?;
        result.set_item(
            "root_prior_mass",
            root.edges.iter().map(|edge| edge.prior).sum::<f32>(),
        )?;
        results.append(result)?;
    }

    Ok(results.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hexformer_ar_mcts, module)?)?;
    module.add_function(wrap_pyfunction!(hexformer_ar_batched_mcts, module)?)?;
    Ok(())
}
