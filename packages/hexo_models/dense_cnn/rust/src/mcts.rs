//! Dense CNN MCTS Python boundary.
//!
//! Python hands live engine states to this module. The dense-cnn state bridge is
//! responsible for cloning those states into authoritative Rust `HexoState`
//! values; from there the search never mutates the live Python game. Tree
//! mechanics live in `mcts_tree`, and evaluator payload parsing lives in
//! `mcts_eval`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::constants::MODEL1_EVAL_CACHE_MAX_STATES;
use super::mcts_eval::{
    evaluate_model1_states_cached, new_shared_evaluation_cache, new_shared_evaluation_stats,
    EvaluationStats, Model1MctsEvaluationCache,
};
use super::mcts_tree::{
    select_root_action, terminal_value, ProgressiveWideningConfig, RustLeaf, RustSearch,
    RustSearchDiagnostics,
};
use super::state::states_from_py_states;

#[pyfunction(signature = (states, visits, c_puct, temperature, seed, evaluator, virtual_batch_size=None, progressive_widening_initial_actions=None, progressive_widening_child_initial_actions=None, progressive_widening_growth_interval=None, progressive_widening_growth_base=None, progressive_widening_candidate_actions=None, evaluation_cache=None))]
pub fn model1_batched_mcts(
    py: Python<'_>,
    states: &Bound<'_, PyAny>,
    visits: u32,
    c_puct: f32,
    temperature: f32,
    seed: u64,
    evaluator: &Bound<'_, PyAny>,
    virtual_batch_size: Option<u32>,
    progressive_widening_initial_actions: Option<u32>,
    progressive_widening_child_initial_actions: Option<u32>,
    progressive_widening_growth_interval: Option<f32>,
    progressive_widening_growth_base: Option<f32>,
    progressive_widening_candidate_actions: Option<u32>,
    evaluation_cache: Option<PyRef<'_, Model1MctsEvaluationCache>>,
) -> PyResult<Py<PyAny>> {
    let roots = states_from_py_states(py, states)?;
    if roots.is_empty() {
        return Ok(PyTuple::empty(py).into_any().unbind());
    }

    let target_visits = visits.max(1);
    let leaf_batch_per_root = virtual_batch_size.unwrap_or(target_visits).max(1);
    let widening = progressive_widening_initial_actions
        .filter(|count| *count > 0)
        .and_then(|count| {
            ProgressiveWideningConfig::new(
                count as usize,
                progressive_widening_child_initial_actions.unwrap_or(count) as usize,
                progressive_widening_growth_interval.unwrap_or(40.0),
                progressive_widening_growth_base.unwrap_or(1.3),
            )
        });
    let (evaluation_cache, cache_max_states) = if let Some(cache) = evaluation_cache.as_ref() {
        (cache.shared_cache(), cache.max_state_count())
    } else {
        (new_shared_evaluation_cache(), MODEL1_EVAL_CACHE_MAX_STATES)
    };
    let evaluation_stats = new_shared_evaluation_stats();
    let root_evals = evaluate_model1_states_cached(
        py,
        evaluator,
        &roots,
        &evaluation_cache,
        Some(&evaluation_stats),
        progressive_widening_candidate_actions.map(|count| count.max(1) as usize),
        cache_max_states,
    )?;
    let mut searches: Vec<RustSearch> = roots
        .into_iter()
        .zip(root_evals.iter())
        .map(|(state, eval)| {
            RustSearch::new(
                state,
                eval,
                target_visits,
                evaluation_cache.clone(),
                widening,
            )
        })
        .collect();

    if searches.iter().any(RustSearch::root_edges_empty) {
        return Err(PyValueError::new_err("MCTS root has no legal actions"));
    }

    // Each outer pass gathers several pending leaves per root, evaluates the
    // unique uncached states as one Python/Torch batch, and then backs values
    // into independent root trees. The shared cache handle deduplicates exact
    // states across roots while each tree keeps its own visit statistics.
    while searches.iter().any(RustSearch::needs_visits) {
        let mut leaves = Vec::new();
        let mut immediate = Vec::new();
        let mut made_progress = false;

        for root_index in 0..searches.len() {
            if !searches[root_index].needs_visits() {
                continue;
            }
            let budget = leaf_batch_per_root.min(searches[root_index].remaining_visits());
            for _ in 0..budget {
                let selected = searches[root_index].select_pending_leaf(c_puct)?;
                let Some(selected) = selected else {
                    break;
                };
                searches[root_index].apply_virtual_visit(&selected.path);
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
                        state_hash: selected.state_hash,
                    });
                }
            }
        }

        for (root_index, path, leaf_player, leaf_value) in immediate {
            searches[root_index].backup_virtual(&path, leaf_player, leaf_value);
        }

        if !leaves.is_empty() {
            let leaf_states: Vec<_> = leaves.iter().map(|leaf| leaf.state.clone()).collect();
            let search_cache = searches[0].evaluator_cache.clone();
            let evaluations = evaluate_model1_states_cached(
                py,
                evaluator,
                &leaf_states,
                &search_cache,
                Some(&evaluation_stats),
                progressive_widening_candidate_actions.map(|count| count.max(1) as usize),
                cache_max_states,
            )?;
            for (leaf, evaluation) in leaves.into_iter().zip(evaluations.iter()) {
                let search = &mut searches[leaf.root_index];
                let child_id = search.add_node_from_eval(&leaf.state, leaf.state_hash, evaluation);
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

    let cache_len = evaluation_cache.borrow().len();
    let evaluation_stats = evaluation_stats.borrow().clone();
    let batch_diagnostics = build_batch_diagnostics(
        py,
        &searches,
        &evaluation_stats,
        target_visits,
        leaf_batch_per_root,
        cache_len,
    )?;
    let results = PyList::empty(py);
    for (index, search) in searches.iter().enumerate() {
        let result = PyDict::new(py);
        let root = search.root();
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
        result.set_item(
            "diagnostics",
            build_result_diagnostics(py, &search.diagnostics(), &batch_diagnostics)?,
        )?;
        results.append(result)?;
    }

    Ok(results.into_any().unbind())
}

fn build_result_diagnostics<'py>(
    py: Python<'py>,
    search: &RustSearchDiagnostics,
    batch: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let diagnostics = PyDict::new(py);
    let root = PyDict::new(py);
    root.set_item("node_count", search.node_count)?;
    root.set_item("active_edge_count", search.active_edge_count)?;
    root.set_item("hidden_prior_count", search.hidden_prior_count)?;
    root.set_item("root_active_edges", search.root_active_edges)?;
    root.set_item("root_hidden_priors", search.root_hidden_priors)?;
    root.set_item(
        "max_active_edges_per_node",
        search.max_active_edges_per_node,
    )?;
    root.set_item(
        "max_hidden_priors_per_node",
        search.max_hidden_priors_per_node,
    )?;
    root.set_item("widened_edges_total", search.widened_edges_total)?;
    diagnostics.set_item("root", root)?;
    diagnostics.set_item("batch", batch)?;
    Ok(diagnostics)
}

fn build_batch_diagnostics<'py>(
    py: Python<'py>,
    searches: &[RustSearch],
    evaluation: &EvaluationStats,
    target_visits: u32,
    leaf_batch_per_root: u32,
    cache_len: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mut aggregate = RustSearchDiagnostics::default();
    let mut completed_visits = 0u64;
    let mut max_nodes_per_root = 0usize;
    let mut max_active_edges_per_root = 0usize;
    let mut max_hidden_priors_per_root = 0usize;
    for search in searches {
        let stats = search.diagnostics();
        aggregate.node_count += stats.node_count;
        aggregate.active_edge_count += stats.active_edge_count;
        aggregate.hidden_prior_count += stats.hidden_prior_count;
        aggregate.root_active_edges += stats.root_active_edges;
        aggregate.root_hidden_priors += stats.root_hidden_priors;
        aggregate.max_active_edges_per_node = aggregate
            .max_active_edges_per_node
            .max(stats.max_active_edges_per_node);
        aggregate.max_hidden_priors_per_node = aggregate
            .max_hidden_priors_per_node
            .max(stats.max_hidden_priors_per_node);
        aggregate.widened_edges_total += stats.widened_edges_total;
        completed_visits += search.completed_visits as u64;
        max_nodes_per_root = max_nodes_per_root.max(stats.node_count);
        max_active_edges_per_root = max_active_edges_per_root.max(stats.active_edge_count);
        max_hidden_priors_per_root = max_hidden_priors_per_root.max(stats.hidden_prior_count);
    }

    let tree = PyDict::new(py);
    tree.set_item("root_count", searches.len())?;
    tree.set_item("target_visits", target_visits)?;
    tree.set_item("leaf_batch_per_root", leaf_batch_per_root)?;
    tree.set_item("completed_visits", completed_visits)?;
    tree.set_item("node_count", aggregate.node_count)?;
    tree.set_item("active_edge_count", aggregate.active_edge_count)?;
    tree.set_item("hidden_prior_count", aggregate.hidden_prior_count)?;
    tree.set_item("root_active_edges", aggregate.root_active_edges)?;
    tree.set_item("root_hidden_priors", aggregate.root_hidden_priors)?;
    tree.set_item("max_nodes_per_root", max_nodes_per_root)?;
    tree.set_item("max_active_edges_per_root", max_active_edges_per_root)?;
    tree.set_item("max_hidden_priors_per_root", max_hidden_priors_per_root)?;
    tree.set_item(
        "max_active_edges_per_node",
        aggregate.max_active_edges_per_node,
    )?;
    tree.set_item(
        "max_hidden_priors_per_node",
        aggregate.max_hidden_priors_per_node,
    )?;
    tree.set_item("widened_edges_total", aggregate.widened_edges_total)?;

    let eval = PyDict::new(py);
    eval.set_item("requested_states", evaluation.requested_states)?;
    eval.set_item("cache_hits", evaluation.cache_hits)?;
    eval.set_item("duplicate_hits", evaluation.duplicate_hits)?;
    eval.set_item("unique_states", evaluation.unique_states)?;
    eval.set_item("evaluator_chunks", evaluation.evaluator_chunks)?;
    eval.set_item("encoded_states", evaluation.encoded_states)?;
    eval.set_item("encoded_legal_actions", evaluation.encoded_legal_actions)?;
    eval.set_item("max_chunk_states", evaluation.max_chunk_states)?;
    eval.set_item(
        "max_chunk_legal_actions",
        evaluation.max_chunk_legal_actions,
    )?;
    eval.set_item("input_bytes", evaluation.input_bytes)?;
    eval.set_item("legal_index_bytes", evaluation.legal_index_bytes)?;
    eval.set_item("value_bytes", evaluation.value_bytes)?;
    eval.set_item("prior_bytes", evaluation.prior_bytes)?;
    eval.set_item("cache_inserts", evaluation.cache_inserts)?;
    eval.set_item("cache_insert_skipped", evaluation.cache_insert_skipped)?;
    eval.set_item("cache_size", cache_len)?;
    eval.set_item("cache_size_peak", evaluation.cache_size_peak.max(cache_len))?;

    let diagnostics = PyDict::new(py);
    diagnostics.set_item("tree", tree)?;
    diagnostics.set_item("evaluation", eval)?;
    Ok(diagnostics)
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model1MctsEvaluationCache>()?;
    module.add_function(wrap_pyfunction!(model1_batched_mcts, module)?)?;
    Ok(())
}
