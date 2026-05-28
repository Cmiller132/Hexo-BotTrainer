//! Dense CNN MCTS Python boundary.
//!
//! Python hands live engine states to this module. The dense-cnn state bridge is
//! responsible for cloning those states into authoritative Rust `HexoState`
//! values; from there the search never mutates the live Python game. Tree
//! mechanics live in `mcts_tree`, and evaluator payload parsing lives in
//! `mcts_eval`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use rayon::prelude::*;
use std::collections::HashMap;

use hexo_engine::PackedCoord;

use super::constants::{MODEL1_ACTIVE_ROOT_LIMIT, MODEL1_EVAL_CACHE_MAX_STATES};
use super::mcts_eval::{
    evaluate_model1_state_refs_cached, evaluate_model1_states_cached, new_shared_evaluation_cache,
    new_shared_evaluation_stats, state_hash, EvaluationStats, Model1MctsEvaluationCache,
    RustEvaluationRequest, SharedEvaluationCache, SharedEvaluationStats,
};
use super::mcts_tree::{
    select_root_action, terminal_value, ProgressiveWideningConfig, RustLeaf, RustSearch,
    RustSearchDiagnostics,
};
use super::state::states_from_py_states;

struct RootSelectionWork {
    leaves: Vec<RustLeaf>,
    made_progress: bool,
}

#[pyfunction(signature = (states, visits, c_puct, temperature, seed, evaluator, virtual_batch_size=None, progressive_widening_initial_actions=None, progressive_widening_child_initial_actions=None, progressive_widening_growth_interval=None, progressive_widening_growth_base=None, progressive_widening_candidate_actions=None, evaluation_cache=None, active_root_limit=None))]
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
    active_root_limit: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let roots = states_from_py_states(py, states)?;
    if roots.is_empty() {
        return Ok(PyTuple::empty(py).into_any().unbind());
    }
    let root_limit = active_root_limit.unwrap_or(MODEL1_ACTIVE_ROOT_LIMIT).max(1);
    if roots.len() > root_limit {
        return Err(PyValueError::new_err(format!(
            "dense_cnn MCTS received {} active roots, above strict limit {}",
            roots.len(),
            root_limit
        )));
    }

    let target_visits = visits.max(1);
    let leaf_batch_per_root = virtual_batch_size.unwrap_or(target_visits).max(1);
    let widening = progressive_widening_initial_actions
        .filter(|count| *count > 0)
        .and_then(|count| {
            ProgressiveWideningConfig::new(
                count as usize,
                progressive_widening_child_initial_actions.unwrap_or(count) as usize,
                progressive_widening_growth_interval.unwrap_or(256.0),
                progressive_widening_growth_base.unwrap_or(1.3),
            )
        });
    let (evaluation_cache, cache_max_states) = if let Some(cache) = evaluation_cache.as_ref() {
        (cache.shared_cache(), cache.max_state_count())
    } else {
        (new_shared_evaluation_cache(), MODEL1_EVAL_CACHE_MAX_STATES)
    };
    let evaluation_stats = new_shared_evaluation_stats();
    let prior_candidate_limit =
        progressive_widening_candidate_actions.map(|count| count.max(1) as usize);
    let root_evals = evaluate_model1_states_cached(
        py,
        evaluator,
        &roots,
        &evaluation_cache,
        Some(&evaluation_stats),
        prior_candidate_limit,
        cache_max_states,
    )?;
    let mut searches: Vec<RustSearch> = roots
        .into_iter()
        .zip(root_evals.iter())
        .map(|(state, eval)| RustSearch::new(state, eval, target_visits, widening))
        .collect();

    if searches.iter().any(RustSearch::root_edges_empty) {
        return Err(PyValueError::new_err("MCTS root has no legal actions"));
    }

    run_searches_to_targets(
        py,
        evaluator,
        &mut searches,
        c_puct,
        leaf_batch_per_root,
        &evaluation_cache,
        &evaluation_stats,
        prior_candidate_limit,
        cache_max_states,
    )?;
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
    build_search_result_payloads(py, &searches, &batch_diagnostics, temperature, seed, None)
}

#[pyclass(unsendable)]
pub(crate) struct Model1MctsSession {
    searches: HashMap<u64, RustSearch>,
    evaluation_cache: SharedEvaluationCache,
    cache_max_states: usize,
}

#[pymethods]
impl Model1MctsSession {
    #[new]
    #[pyo3(signature = (max_states=None))]
    fn new(max_states: Option<usize>) -> Self {
        Self {
            searches: HashMap::new(),
            evaluation_cache: new_shared_evaluation_cache(),
            cache_max_states: max_states.unwrap_or(MODEL1_EVAL_CACHE_MAX_STATES).max(1),
        }
    }

    fn clear(&mut self) {
        self.searches.clear();
        self.evaluation_cache.borrow_mut().clear();
    }

    fn discard(&mut self, game_key: u64) {
        self.searches.remove(&game_key);
    }

    fn len(&self) -> usize {
        self.searches.len()
    }

    #[pyo3(signature = (game_keys, states, visits, c_puct, temperature, seed, evaluator, virtual_batch_size=None, progressive_widening_initial_actions=None, progressive_widening_child_initial_actions=None, progressive_widening_growth_interval=None, progressive_widening_growth_base=None, progressive_widening_candidate_actions=None, active_root_limit=None))]
    fn search(
        &mut self,
        py: Python<'_>,
        game_keys: Vec<u64>,
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
        active_root_limit: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let roots = states_from_py_states(py, states)?;
        if roots.is_empty() {
            return Ok(PyTuple::empty(py).into_any().unbind());
        }
        if roots.len() != game_keys.len() {
            return Err(PyValueError::new_err(format!(
                "dense_cnn MCTS session received {} game keys for {} states",
                game_keys.len(),
                roots.len()
            )));
        }
        let root_limit = active_root_limit.unwrap_or(MODEL1_ACTIVE_ROOT_LIMIT).max(1);
        if roots.len() > root_limit {
            return Err(PyValueError::new_err(format!(
                "dense_cnn MCTS session received {} active roots, above strict limit {}",
                roots.len(),
                root_limit
            )));
        }

        let target_visits = visits.max(1);
        let leaf_batch_per_root = virtual_batch_size.unwrap_or(target_visits).max(1);
        let widening = progressive_widening_initial_actions
            .filter(|count| *count > 0)
            .and_then(|count| {
                ProgressiveWideningConfig::new(
                    count as usize,
                    progressive_widening_child_initial_actions.unwrap_or(count) as usize,
                    progressive_widening_growth_interval.unwrap_or(256.0),
                    progressive_widening_growth_base.unwrap_or(1.3),
                )
            });
        let prior_candidate_limit =
            progressive_widening_candidate_actions.map(|count| count.max(1) as usize);
        let evaluation_stats = new_shared_evaluation_stats();

        let mut searches: Vec<Option<RustSearch>> = Vec::with_capacity(roots.len());
        let mut missing_indices = Vec::new();
        let mut missing_roots = Vec::new();
        for (index, (game_key, root)) in game_keys.iter().zip(roots.iter()).enumerate() {
            let root_hash = state_hash(root);
            if let Some(mut search) = self.searches.remove(game_key) {
                if search.root_hash == root_hash {
                    search.set_additional_visits(target_visits);
                    searches.push(Some(search));
                    continue;
                }
            }
            missing_indices.push(index);
            missing_roots.push(root.clone());
            searches.push(None);
        }

        if !missing_roots.is_empty() {
            let root_evals = evaluate_model1_states_cached(
                py,
                evaluator,
                &missing_roots,
                &self.evaluation_cache,
                Some(&evaluation_stats),
                prior_candidate_limit,
                self.cache_max_states,
            )?;
            for ((index, root), evaluation) in missing_indices
                .into_iter()
                .zip(missing_roots.into_iter())
                .zip(root_evals.iter())
            {
                searches[index] = Some(RustSearch::new(root, evaluation, target_visits, widening));
            }
        }

        let mut searches: Vec<RustSearch> = searches
            .into_iter()
            .map(|search| search.expect("session search initialized"))
            .collect();
        if searches.iter().any(RustSearch::root_edges_empty) {
            return Err(PyValueError::new_err("MCTS root has no legal actions"));
        }

        let baselines: Vec<HashMap<PackedCoord, u32>> = searches
            .iter()
            .map(|search| search.root_edge_visits().into_iter().collect())
            .collect();
        run_searches_to_targets(
            py,
            evaluator,
            &mut searches,
            c_puct,
            leaf_batch_per_root,
            &self.evaluation_cache,
            &evaluation_stats,
            prior_candidate_limit,
            self.cache_max_states,
        )?;
        let cache_len = self.evaluation_cache.borrow().len();
        let evaluation_stats = evaluation_stats.borrow().clone();
        let batch_diagnostics = build_batch_diagnostics(
            py,
            &searches,
            &evaluation_stats,
            target_visits,
            leaf_batch_per_root,
            cache_len,
        )?;
        let selected_actions: Vec<_> = searches
            .iter()
            .enumerate()
            .map(|(index, search)| {
                select_root_action(search.root(), temperature, seed.wrapping_add(index as u64))
            })
            .collect();
        let results = build_search_result_payloads(
            py,
            &searches,
            &batch_diagnostics,
            temperature,
            seed,
            Some(&baselines),
        )?;

        for ((game_key, mut search), selected) in game_keys
            .into_iter()
            .zip(searches.into_iter())
            .zip(selected_actions.into_iter())
        {
            if let Some(action_id) = selected {
                if search.advance_root(action_id)? {
                    self.searches.insert(game_key, search);
                }
            }
        }

        Ok(results)
    }
}

fn run_searches_to_targets(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    searches: &mut [RustSearch],
    c_puct: f32,
    leaf_batch_per_root: u32,
    evaluation_cache: &SharedEvaluationCache,
    evaluation_stats: &SharedEvaluationStats,
    prior_candidate_limit: Option<usize>,
    cache_max_states: usize,
) -> PyResult<()> {
    // Each outer pass gathers several pending leaves per root, evaluates the
    // unique uncached states as one Python/Torch batch, and then backs values
    // into independent root trees. Root selection is independent per tree, so
    // it runs across CPU workers while the shared cache remains only at eval
    // boundaries.
    while searches.iter().any(RustSearch::needs_visits) {
        let work_results: Vec<PyResult<RootSelectionWork>> = searches
            .par_iter_mut()
            .enumerate()
            .map(|(root_index, search)| {
                let mut leaves = Vec::new();
                let mut made_progress = false;
                if !search.needs_visits() {
                    return Ok(RootSelectionWork {
                        leaves,
                        made_progress,
                    });
                }
                let budget = leaf_batch_per_root.min(search.remaining_visits());
                for _ in 0..budget {
                    let selected = search.select_pending_leaf(c_puct)?;
                    let Some(selected) = selected else {
                        break;
                    };
                    search.apply_virtual_visit(&selected.path);
                    made_progress = true;

                    if let Some(outcome) = selected.terminal {
                        let leaf_player = selected.state.current_player();
                        let leaf_value = terminal_value(outcome, leaf_player);
                        search.backup_virtual(&selected.path, leaf_player, leaf_value);
                    } else if let Some(node_id) = selected.existing_node {
                        let node = &search.nodes[node_id];
                        search.backup_virtual(&selected.path, node.player, node.value());
                    } else {
                        search.mark_pending(selected.parent_node, selected.edge_index, 1);
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
                Ok(RootSelectionWork {
                    leaves,
                    made_progress,
                })
            })
            .collect();
        let mut leaves = Vec::new();
        let mut made_progress = false;
        for work in work_results {
            let work = work?;
            made_progress |= work.made_progress;
            leaves.extend(work.leaves);
        }

        if !leaves.is_empty() {
            let leaf_requests: Vec<_> = leaves
                .iter()
                .map(|leaf| RustEvaluationRequest {
                    state: &leaf.state,
                    state_hash: leaf.state_hash,
                })
                .collect();
            let evaluations = evaluate_model1_state_refs_cached(
                py,
                evaluator,
                &leaf_requests,
                evaluation_cache,
                Some(evaluation_stats),
                prior_candidate_limit,
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
    Ok(())
}

fn build_search_result_payloads(
    py: Python<'_>,
    searches: &[RustSearch],
    batch_diagnostics: &Bound<'_, PyDict>,
    temperature: f32,
    seed: u64,
    baselines: Option<&[HashMap<PackedCoord, u32>]>,
) -> PyResult<Py<PyAny>> {
    let results = PyList::empty(py);
    for (index, search) in searches.iter().enumerate() {
        let result = PyDict::new(py);
        let root = search.root();
        let baseline = baselines.and_then(|items| items.get(index));
        let policy_total: u32 = root
            .edges
            .iter()
            .map(|edge| {
                let before = baseline
                    .and_then(|visits| visits.get(&edge.action_id).copied())
                    .unwrap_or(0);
                edge.visits.saturating_sub(before)
            })
            .sum();
        let mut policy_action_ids = Vec::with_capacity(root.edges.len());
        let mut policy_weights = Vec::with_capacity(root.edges.len());
        for edge in &root.edges {
            let before = baseline
                .and_then(|visits| visits.get(&edge.action_id).copied())
                .unwrap_or(0);
            let visits = edge.visits.saturating_sub(before);
            if baseline.is_some() && visits == 0 {
                continue;
            }
            let weight = if policy_total > 0 {
                visits as f32 / policy_total as f32
            } else {
                edge.prior
            };
            policy_action_ids.push(edge.action_id);
            policy_weights.push(weight);
        }
        let selected = select_root_action(root, temperature, seed.wrapping_add(index as u64));
        result.set_item("action_id", selected.unwrap_or(0))?;
        let action_byte_len = policy_action_ids.len() * std::mem::size_of::<u32>();
        let weight_byte_len = policy_weights.len() * std::mem::size_of::<f32>();
        let action_bytes = unsafe {
            std::slice::from_raw_parts(policy_action_ids.as_ptr() as *const u8, action_byte_len)
        };
        let weight_bytes = unsafe {
            std::slice::from_raw_parts(policy_weights.as_ptr() as *const u8, weight_byte_len)
        };
        result.set_item(
            "visit_policy_action_ids_bytes",
            PyBytes::new(py, action_bytes),
        )?;
        result.set_item("visit_policy_weights_bytes", PyBytes::new(py, weight_bytes))?;
        result.set_item("visit_policy_count", policy_action_ids.len())?;
        result.set_item("root_value", root.value())?;
        result.set_item("visits", policy_total)?;
        if index == 0 {
            result.set_item(
                "diagnostics",
                build_result_diagnostics(py, &search.diagnostics(), batch_diagnostics)?,
            )?;
        }
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
    eval.set_item("encoding_seconds", evaluation.encoding_seconds)?;
    eval.set_item("evaluator_seconds", evaluation.evaluator_seconds)?;

    let diagnostics = PyDict::new(py);
    diagnostics.set_item("tree", tree)?;
    diagnostics.set_item("evaluation", eval)?;
    Ok(diagnostics)
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model1MctsEvaluationCache>()?;
    module.add_class::<Model1MctsSession>()?;
    module.add_function(wrap_pyfunction!(model1_batched_mcts, module)?)?;
    Ok(())
}
