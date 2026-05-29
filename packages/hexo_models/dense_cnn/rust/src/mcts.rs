//! Dense CNN MCTS Python boundary.
//!
//! Python hands live engine states to this module. The dense_cnn state bridge is
//! responsible for cloning those states into authoritative Rust `HexoState`
//! values; from there the search never mutates the live Python game. Tree
//! mechanics live in `mcts_tree`, and evaluator payload parsing lives in
//! `mcts_eval`.
//!
//! `Model1MctsSession` is intentionally stateful. A caller provides a stable
//! game key for each active game, and the session promotes the selected child
//! subtree after every search. If the next call sends a root whose hash differs
//! from the promoted tree, the old tree is discarded and a new one is evaluated.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use hexo_engine::PackedCoord;

use super::constants::{MODEL1_ACTIVE_ROOT_LIMIT, MODEL1_EVAL_CACHE_MAX_STATES};
use super::mcts_eval::{
    evaluate_model1_state_refs_cached, evaluate_model1_states_cached, new_shared_evaluation_cache,
    new_shared_evaluation_stats, state_hash, EvaluationStats, RustEvaluation,
    RustEvaluationRequest, SharedEvaluationCache, SharedEvaluationStats,
};
use super::mcts_tree::{
    terminal_value, RootDirichletNoise, RustEdge, RustLeaf, RustNode, RustSearch,
    RustSearchDiagnostics, Widening,
};
use super::state::states_from_py_states;

struct RootSelectionWork {
    // Leaves selected by one root during one virtual batch.
    leaves: Vec<RustLeaf>,
    // False means the root could not select any new or existing visit. The outer
    // loop uses this to stop if every root is blocked.
    made_progress: bool,
}

#[pyclass(unsendable)]
pub(crate) struct Model1MctsSession {
    // Keyed by Python self-play game ids. Each search stores one promoted root.
    searches: HashMap<u64, RustSearch>,
    // Shared across active roots so transpositions and duplicate leaf requests
    // evaluate once per exact state hash.
    evaluation_cache: SharedEvaluationCache,
    cache_max_states: usize,
}

#[pymethods]
impl Model1MctsSession {
    #[new]
    #[pyo3(signature = (max_states=None))]
    fn new(max_states: Option<usize>) -> PyResult<Self> {
        let cache_max_states = validate_positive_usize(
            "max_states",
            max_states.unwrap_or(MODEL1_EVAL_CACHE_MAX_STATES),
        )?;
        Ok(Self {
            searches: HashMap::new(),
            evaluation_cache: new_shared_evaluation_cache(),
            cache_max_states,
        })
    }

    fn clear(&mut self) {
        self.searches.clear();
        self.evaluation_cache
            .lock()
            .expect("evaluation cache mutex poisoned")
            .clear();
    }

    fn discard(&mut self, game_key: u64) {
        self.searches.remove(&game_key);
    }

    fn len(&self) -> usize {
        self.searches.len()
    }

    #[pyo3(signature = (game_keys, states, visits, c_puct, temperature, seed, evaluator, virtual_batch_size=None, active_root_limit=None, root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None, root_policy_temperature=None, fpu_reduction=None, virtual_loss=None, widening_policy_mass=None, widening_max_children=None, widening_min_children=None))]
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
        active_root_limit: Option<usize>,
        root_dirichlet_total_alpha: Option<f32>,
        root_dirichlet_noise_fraction: Option<f32>,
        root_policy_temperature: Option<f32>,
        fpu_reduction: Option<f32>,
        virtual_loss: Option<f32>,
        widening_policy_mass: Option<f32>,
        widening_max_children: Option<u32>,
        widening_min_children: Option<u32>,
    ) -> PyResult<Py<PyAny>> {
        // Validate true native-search boundaries here. The Python wrapper is a
        // transport layer and intentionally does not duplicate these checks.
        validate_search_inputs(visits, c_puct, temperature)?;
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
        let root_limit = validate_positive_usize(
            "active_root_limit",
            active_root_limit.unwrap_or(MODEL1_ACTIVE_ROOT_LIMIT),
        )?;
        if roots.len() > root_limit {
            return Err(PyValueError::new_err(format!(
                "dense_cnn MCTS session received {} active roots, above strict limit {}",
                roots.len(),
                root_limit
            )));
        }

        let target_visits = visits;
        let leaf_batch_per_root = validate_positive_u32(
            "virtual_batch_size",
            virtual_batch_size.unwrap_or(target_visits),
        )?;
        let evaluation_stats = new_shared_evaluation_stats();
        let root_policy_temperature =
            validate_positive_f32("root_policy_temperature", root_policy_temperature.unwrap_or(1.0))?;
        let fpu_reduction = validate_nonnegative_f32("fpu_reduction", fpu_reduction.unwrap_or(0.20))?;
        let virtual_loss = validate_nonnegative_f32("virtual_loss", virtual_loss.unwrap_or(1.0))?;
        let root_noise_config =
            root_noise_config(root_dirichlet_total_alpha, root_dirichlet_noise_fraction)?;

        let widening_mass = widening_policy_mass.unwrap_or(0.95);
        if !widening_mass.is_finite() || widening_mass <= 0.0 || widening_mass > 1.0 {
            return Err(PyValueError::new_err("widening_policy_mass must be in (0, 1]"));
        }
        let widening = Widening {
            mass: widening_mass,
            min_children: validate_positive_u32(
                "widening_min_children",
                widening_min_children.unwrap_or(2),
            )? as usize,
            max_children: validate_positive_u32(
                "widening_max_children",
                widening_max_children.unwrap_or(32),
            )? as usize,
        };
        if widening.min_children > widening.max_children {
            return Err(PyValueError::new_err(
                "widening_min_children must be <= widening_max_children",
            ));
        }

        let mut searches: Vec<Option<RustSearch>> = Vec::with_capacity(roots.len());
        let mut missing_indices = Vec::new();
        let mut missing_roots = Vec::new();
        for (index, (game_key, root)) in game_keys.iter().zip(roots.iter()).enumerate() {
            let root_hash = state_hash(root);
            // Reuse only when the promoted native root exactly matches the live
            // engine state Python just handed us. This avoids stale-subtree
            // reuse if an external caller advanced or reset the game.
            if let Some(mut search) = self.searches.remove(game_key) {
                if search.root_hash == root_hash {
                    search.set_additional_visits(target_visits);
                    if let Some(noise) = root_noise(root_noise_config, seed, index) {
                        search.apply_root_dirichlet_noise(noise);
                    }
                    searches.push(Some(search));
                    continue;
                }
            }
            missing_indices.push(index);
            missing_roots.push(root.clone());
            searches.push(None);
        }

        if !missing_roots.is_empty() {
            // Missing roots are evaluated as one batch before tree construction.
            let root_evals = evaluate_model1_states_cached(
                py,
                evaluator,
                &missing_roots,
                &self.evaluation_cache,
                Some(&evaluation_stats),
                self.cache_max_states,
            )?;
            for ((index, root), evaluation) in missing_indices
                .into_iter()
                .zip(missing_roots.into_iter())
                .zip(root_evals.iter())
            {
                searches[index] = Some(RustSearch::new(
                    root,
                    &**evaluation,
                    target_visits,
                    fpu_reduction,
                    root_policy_temperature,
                    root_noise(root_noise_config, seed, index),
                    widening,
                )?);
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
        // Baselines let the returned visit policy describe only the visits added
        // by this call. That is what self-play wants for the current move target
        // when a root already carried visits from previous turns.
        run_searches_to_targets(
            py,
            evaluator,
            &mut searches,
            c_puct,
            leaf_batch_per_root,
            &self.evaluation_cache,
            &evaluation_stats,
            self.cache_max_states,
            virtual_loss,
        )?;
        let cache_len = self
            .evaluation_cache
            .lock()
            .expect("evaluation cache mutex poisoned")
            .len();
        let evaluation_stats = evaluation_stats
            .lock()
            .expect("evaluation stats mutex poisoned")
            .clone();
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
                select_search_action(
                    search,
                    baselines.get(index),
                    temperature,
                    seed.wrapping_add(index as u64),
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
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
                // Store the child subtree for the next call. Terminal or
                // unexpanded children are intentionally not retained.
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
    cache_max_states: usize,
    virtual_loss: f32,
) -> PyResult<()> {
    // A1 (select↔eval pipeline). The serial form selected a leaf batch, ran one
    // Python/Torch forward while every CPU worker sat idle, then backed up — so
    // GPU duty was ~29%. Here we overlap the two halves with a two-stage software
    // pipeline: the *current* batch is evaluated on this GIL-holding thread while
    // the *next* batch is selected on a scoped worker thread (which itself fans
    // out across rayon). Virtual loss (already applied at selection) is the sync
    // primitive — the next batch sees the in-flight leaves as pending and avoids
    // them, exactly as virtual-loss parallel MCTS intends.
    //
    // Tree-access discipline (no lock needed): during the scope ONLY the select
    // thread touches the trees; the eval thread reads only the owned leaf batch +
    // the (thread-safe) cache. Backup runs after the scope joins, with exclusive
    // access on this thread. So the trees have a single mutator at every instant.
    //
    // This is deterministic for a fixed seed but NOT bit-identical to the serial
    // barrier: the next batch is selected before the current batch is backed up,
    // extending the virtual-loss window by one batch. That is the intended,
    // search-quality-neutral behavior of pipelined MCTS.
    //
    // Termination is keyed off `needs_visits` (visit budget counts virtual visits
    // at selection time), NOT off the prefetch making progress. The prefetch can
    // legitimately come up empty on a *narrow* tree — every selectable path is
    // already in-flight (pending) for the batch currently being evaluated, so the
    // next batch is blocked until that batch is backed up. When that happens we
    // fall back to a synchronous select after backup so the search always advances
    // to its visit target (overlap is best-effort, correctness is not).
    // Optional diagnostics (HEXO_MCTS_TRACE): per-pass eval vs select vs scope
    // wall, to confirm the overlap is real and size the overlappable fraction.
    let trace = std::env::var("HEXO_MCTS_TRACE").is_ok();
    let mut tr_passes = 0u64;
    let mut tr_leaves = 0u64;
    let mut tr_eval = 0f64;
    let mut tr_select = 0f64;
    let mut tr_scope = 0f64;
    let mut tr_backup = 0f64;

    let (mut pending_leaves, _primed_progress) =
        select_leaf_batch(searches, c_puct, leaf_batch_per_root, virtual_loss)?;

    loop {
        if pending_leaves.is_empty() {
            if !searches.iter().any(RustSearch::needs_visits) {
                break;
            }
            // Nothing in flight to overlap with; select synchronously to advance.
            let (leaves, made_progress) =
                select_leaf_batch(searches, c_puct, leaf_batch_per_root, virtual_loss)?;
            if leaves.is_empty() {
                // No leaf needs evaluation. Either inline (terminal/existing)
                // backups advanced the search and we retry, or nothing at all
                // could be selected (every needs-visits root is structurally
                // blocked) — break rather than spin.
                if !made_progress {
                    break;
                }
                continue;
            }
            pending_leaves = leaves;
        }

        // Overlap: evaluate the in-flight batch on this GIL thread while the next
        // batch is selected on a worker thread. Prefetch only if visits remain;
        // each root self-limits to its remaining budget inside `select_leaf_batch`.
        let prefetch_next = searches.iter().any(RustSearch::needs_visits);
        let scope_start = std::time::Instant::now();
        let (evaluations, next_leaves, eval_s, select_s) = std::thread::scope(
            |scope| -> PyResult<(Vec<Arc<RustEvaluation>>, Vec<RustLeaf>, f64, f64)> {
                let select_handle = if prefetch_next {
                    let select_searches: &mut [RustSearch] = &mut *searches;
                    Some(scope.spawn(move || {
                        let started = std::time::Instant::now();
                        let result =
                            select_leaf_batch(select_searches, c_puct, leaf_batch_per_root, virtual_loss);
                        (result, started.elapsed().as_secs_f64())
                    }))
                } else {
                    None
                };

                // Building the requests here keeps their borrow of `pending_leaves`
                // local to the scope, so it is free to move into backup after.
                let leaf_requests: Vec<_> = pending_leaves
                    .iter()
                    .map(|leaf| RustEvaluationRequest {
                        state: &leaf.state,
                        state_hash: leaf.state_hash,
                    })
                    .collect();
                let eval_start = std::time::Instant::now();
                let evaluations = evaluate_model1_state_refs_cached(
                    py,
                    evaluator,
                    &leaf_requests,
                    evaluation_cache,
                    Some(evaluation_stats),
                    cache_max_states,
                )?;
                let eval_s = eval_start.elapsed().as_secs_f64();

                let (next_leaves, select_s) = match select_handle {
                    Some(handle) => {
                        let (result, select_s) =
                            handle.join().expect("mcts select worker panicked");
                        (result?.0, select_s)
                    }
                    None => (Vec::new(), 0.0),
                };
                Ok((evaluations, next_leaves, eval_s, select_s))
            },
        )?;

        if trace {
            tr_passes += 1;
            tr_leaves += pending_leaves.len() as u64;
            tr_eval += eval_s;
            tr_select += select_s;
            tr_scope += scope_start.elapsed().as_secs_f64();
        }

        // Backup the batch we just evaluated (exclusive tree access on this thread).
        let backup_start = std::time::Instant::now();
        apply_eval_backups(searches, pending_leaves, &evaluations, virtual_loss)?;
        if trace {
            tr_backup += backup_start.elapsed().as_secs_f64();
        }
        pending_leaves = next_leaves;
    }

    if trace {
        eprintln!(
            "[mcts-trace] passes={} leaves={} eval={:.1}ms select={:.1}ms scope={:.1}ms backup={:.1}ms \
             overlap_saved~={:.1}ms (eval+select-scope)",
            tr_passes,
            tr_leaves,
            tr_eval * 1e3,
            tr_select * 1e3,
            tr_scope * 1e3,
            tr_backup * 1e3,
            (tr_eval + tr_select - tr_scope) * 1e3,
        );
    }
    Ok(())
}

// Select up to one virtual batch of leaves across every root that still needs
// visits. Pure Rust (no Python, no cache): terminal and already-expanded leaves
// are backed up inline; only leaves that still need a network evaluation are
// returned. `made_progress` is true if any root selected at least one leaf.
fn select_leaf_batch(
    searches: &mut [RustSearch],
    c_puct: f32,
    leaf_batch_per_root: u32,
    virtual_loss: f32,
) -> PyResult<(Vec<RustLeaf>, bool)> {
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
                search.apply_virtual_visit(&selected.path, virtual_loss);
                made_progress = true;

                if let Some(outcome) = selected.terminal {
                    let leaf_player = selected.state.current_player();
                    let leaf_value = terminal_value(outcome, leaf_player);
                    search.backup_virtual(&selected.path, leaf_player, leaf_value, virtual_loss);
                } else if let Some(node_id) = selected.existing_node {
                    let node = &search.nodes[node_id];
                    search.backup_virtual(&selected.path, node.player, node.value(), virtual_loss);
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
    Ok((leaves, made_progress))
}

// Attach the evaluated children to their parent edges and back up their values.
// Exclusive `&mut searches` access — called only after the select scope joins.
fn apply_eval_backups(
    searches: &mut [RustSearch],
    leaves: Vec<RustLeaf>,
    evaluations: &[Arc<RustEvaluation>],
    virtual_loss: f32,
) -> PyResult<()> {
    for (leaf, evaluation) in leaves.into_iter().zip(evaluations.iter()) {
        let search = &mut searches[leaf.root_index];
        let child_id =
            search.add_node_from_eval(&leaf.state, leaf.state_hash, Arc::clone(evaluation))?;
        search.nodes[leaf.parent_node].edges[leaf.edge_index].child = Some(child_id);
        search.mark_pending(leaf.parent_node, leaf.edge_index, -1);
        let child_player = search.nodes[child_id].player;
        let child_value = search.nodes[child_id].value();
        search.backup_virtual(&leaf.path, child_player, child_value, virtual_loss);
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
    // The Python wrapper expects byte-backed policies. This avoids allocating a
    // Python tuple for every legal move while still supporting lazy iteration.
    let results = PyList::empty(py);
    for (index, search) in searches.iter().enumerate() {
        let result = PyDict::new(py);
        let root = search.root();
        let baseline = baselines.and_then(|items| items.get(index));
        let (policy_action_ids, policy_weights, policy_total) = visit_policy(root, baseline);
        let (root_prior_action_ids, root_prior_weights) = root_prior_policy(root);
        let selected = select_action_from_policy(
            &policy_action_ids,
            &policy_weights,
            temperature,
            seed.wrapping_add(index as u64),
        )?;
        result.set_item("action_id", selected.unwrap_or(0))?;
        result.set_item(
            "action_selection",
            if baseline.is_some() {
                "delta_visit_policy"
            } else {
                "cumulative_visit_policy"
            },
        )?;
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
        let prior_action_byte_len = root_prior_action_ids.len() * std::mem::size_of::<u32>();
        let prior_weight_byte_len = root_prior_weights.len() * std::mem::size_of::<f32>();
        let prior_action_bytes = unsafe {
            std::slice::from_raw_parts(
                root_prior_action_ids.as_ptr() as *const u8,
                prior_action_byte_len,
            )
        };
        let prior_weight_bytes = unsafe {
            std::slice::from_raw_parts(
                root_prior_weights.as_ptr() as *const u8,
                prior_weight_byte_len,
            )
        };
        result.set_item(
            "root_prior_policy_action_ids_bytes",
            PyBytes::new(py, prior_action_bytes),
        )?;
        result.set_item(
            "root_prior_policy_weights_bytes",
            PyBytes::new(py, prior_weight_bytes),
        )?;
        result.set_item("root_prior_policy_count", root_prior_action_ids.len())?;
        result.set_item("root_value", root.value())?;
        result.set_item("visits", policy_total)?;
        result.set_item(
            "diagnostics",
            build_result_diagnostics(py, &search.diagnostics(), batch_diagnostics)?,
        )?;
        results.append(result)?;
    }

    Ok(results.into_any().unbind())
}

fn root_prior_policy(root: &RustNode) -> (Vec<PackedCoord>, Vec<f32>) {
    // Every in-crop legal move is staged as either a materialized edge or an
    // unexpanded prior, so the root prior is exactly their union normalized. This
    // holds for both an owned root and a reused (shared) root: `remaining_priors`
    // returns the not-yet-materialized candidates for whichever variant the root
    // is, so the exported distribution stays byte-identical to the pre-refactor
    // edges ∪ unexpanded union.
    let remaining = root.remaining_priors();
    let mut priors: HashMap<PackedCoord, f32> =
        HashMap::with_capacity(root.edges.len() + remaining.len());
    for edge in &root.edges {
        if edge.prior.is_finite() && edge.prior > 0.0 {
            priors.insert(edge.action_id, edge.prior);
        }
    }
    for (action_id, prior) in remaining {
        if prior.is_finite() && prior > 0.0 {
            priors.insert(action_id, prior);
        }
    }
    let mut pairs: Vec<(PackedCoord, f32)> = priors.into_iter().collect();
    pairs.sort_unstable_by_key(|(action_id, _prior)| *action_id);
    let action_ids: Vec<PackedCoord> = pairs
        .iter()
        .map(|(action_id, _prior)| *action_id)
        .collect();
    let mut weights: Vec<f32> = pairs.into_iter().map(|(_action_id, prior)| prior).collect();
    let total: f32 = weights.iter().copied().sum();
    if total > 0.0 {
        for weight in &mut weights {
            *weight /= total;
        }
    }
    (action_ids, weights)
}

fn validate_search_inputs(visits: u32, c_puct: f32, temperature: f32) -> PyResult<()> {
    if visits == 0 {
        return Err(PyValueError::new_err("visits must be > 0"));
    }
    if !c_puct.is_finite() || c_puct <= 0.0 {
        return Err(PyValueError::new_err("c_puct must be finite and > 0"));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(PyValueError::new_err("temperature must be finite and >= 0"));
    }
    Ok(())
}

fn validate_positive_u32(name: &str, value: u32) -> PyResult<u32> {
    if value == 0 {
        return Err(PyValueError::new_err(format!("{name} must be > 0")));
    }
    Ok(value)
}

fn validate_positive_usize(name: &str, value: usize) -> PyResult<usize> {
    if value == 0 {
        return Err(PyValueError::new_err(format!("{name} must be > 0")));
    }
    Ok(value)
}

fn validate_positive_f32(name: &str, value: f32) -> PyResult<f32> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PyValueError::new_err(format!("{name} must be finite and > 0")));
    }
    Ok(value)
}

fn validate_nonnegative_f32(name: &str, value: f32) -> PyResult<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!("{name} must be finite and >= 0")));
    }
    Ok(value)
}

fn validate_bounded_f32(name: &str, value: f32, minimum: f32, maximum: f32) -> PyResult<f32> {
    if !value.is_finite() || value < minimum || value > maximum {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and in [{minimum}, {maximum}]"
        )));
    }
    Ok(value)
}

#[derive(Clone, Copy)]
struct RootNoiseConfig {
    total_alpha: f32,
    fraction: f32,
}

fn root_noise_config(
    total_alpha: Option<f32>,
    fraction: Option<f32>,
) -> PyResult<Option<RootNoiseConfig>> {
    match (total_alpha, fraction) {
        (None, None) => Ok(None),
        (Some(total_alpha), Some(fraction)) => {
            let total_alpha = validate_positive_f32("root_dirichlet_total_alpha", total_alpha)?;
            let fraction =
                validate_bounded_f32("root_dirichlet_noise_fraction", fraction, 0.0, 1.0)?;
            if fraction == 0.0 {
                return Ok(None);
            }
            Ok(Some(RootNoiseConfig {
                total_alpha,
                fraction,
            }))
        }
        _ => Err(PyValueError::new_err(
            "root_dirichlet_total_alpha and root_dirichlet_noise_fraction must be provided together",
        )),
    }
}

fn root_noise(
    config: Option<RootNoiseConfig>,
    seed: u64,
    index: usize,
) -> Option<RootDirichletNoise> {
    let config = config?;
    Some(RootDirichletNoise {
        total_alpha: config.total_alpha,
        fraction: config.fraction,
        seed: seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    })
}

fn select_search_action(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    temperature: f32,
    seed: u64,
) -> PyResult<Option<PackedCoord>> {
    let (action_ids, weights, _total) = visit_policy(search.root(), baseline);
    select_action_from_policy(&action_ids, &weights, temperature, seed)
}

fn visit_policy(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
) -> (Vec<PackedCoord>, Vec<f32>, u32) {
    // With a baseline, policy weights are normalized over visits added during
    // this search call. Without one, they are normalized over cumulative root
    // visits and are mostly useful for diagnostics.
    let policy_total: u32 = root
        .edges
        .iter()
        .map(|edge| edge_delta_visits(edge, baseline))
        .sum();
    let mut policy_action_ids = Vec::with_capacity(root.edges.len());
    let mut policy_weights = Vec::with_capacity(root.edges.len());
    for edge in &root.edges {
        let visits = edge_delta_visits(edge, baseline);
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
    (policy_action_ids, policy_weights, policy_total)
}

fn edge_delta_visits(edge: &RustEdge, baseline: Option<&HashMap<PackedCoord, u32>>) -> u32 {
    let before = baseline
        .and_then(|visits| visits.get(&edge.action_id).copied())
        .unwrap_or(0);
    edge.visits.saturating_sub(before)
}

fn select_action_from_policy(
    action_ids: &[PackedCoord],
    weights: &[f32],
    temperature: f32,
    seed: u64,
) -> PyResult<Option<PackedCoord>> {
    // Temperature zero is deterministic argmax. Positive temperature samples
    // from visit weights raised by 1 / temperature, matching self-play action
    // selection from the MCTS visit policy.
    if action_ids.is_empty() || weights.is_empty() {
        return Ok(None);
    }
    if action_ids.len() != weights.len() {
        return Err(PyValueError::new_err("visit policy action and weight lengths differ"));
    }
    let total_weight: f32 = weights.iter().copied().sum();
    for weight in weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(PyValueError::new_err(format!(
                "visit policy weights must be finite and >= 0, got {weight}"
            )));
        }
    }
    if total_weight <= 0.0 {
        return Err(PyValueError::new_err(
            "visit policy must contain positive weight mass",
        ));
    }
    if temperature == 0.0 {
        return Ok(action_ids
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(action_id, _)| action_id));
    }
    let inv_temperature = 1.0 / temperature;
    let mut total = 0.0f64;
    let mut adjusted = Vec::with_capacity(weights.len());
    for weight in weights {
        let value = weight.powf(inv_temperature) as f64;
        total += value;
        adjusted.push(value);
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(PyValueError::new_err(
            "temperature-adjusted visit policy must contain positive finite mass",
        ));
    }
    let mut threshold = random_unit(seed) * total;
    for (action_id, weight) in action_ids.iter().copied().zip(adjusted) {
        threshold -= weight;
        if threshold <= 0.0 {
            return Ok(Some(action_id));
        }
    }
    Ok(action_ids.last().copied())
}

fn random_unit(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
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
    root.set_item("max_active_edges_per_node", search.max_active_edges_per_node)?;
    root.set_item(
        "max_hidden_priors_per_node",
        search.max_hidden_priors_per_node,
    )?;
    root.set_item("active_edge_bytes", search.active_edge_bytes)?;
    root.set_item("hidden_prior_bytes", search.hidden_prior_bytes)?;
    root.set_item("shared_prior_nodes", search.shared_prior_nodes)?;
    root.set_item("shared_prior_refs", search.shared_prior_refs)?;
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
        aggregate.active_edge_bytes += stats.active_edge_bytes;
        aggregate.hidden_prior_bytes += stats.hidden_prior_bytes;
        aggregate.shared_prior_nodes += stats.shared_prior_nodes;
        aggregate.shared_prior_refs += stats.shared_prior_refs;
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
    tree.set_item("max_active_edges_per_node", aggregate.max_active_edges_per_node)?;
    tree.set_item(
        "max_hidden_priors_per_node",
        aggregate.max_hidden_priors_per_node,
    )?;
    tree.set_item("active_edge_bytes", aggregate.active_edge_bytes)?;
    tree.set_item("hidden_prior_bytes", aggregate.hidden_prior_bytes)?;
    tree.set_item("shared_prior_nodes", aggregate.shared_prior_nodes)?;
    tree.set_item("shared_prior_refs", aggregate.shared_prior_refs)?;

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
    eval.set_item(
        "cache_prior_pair_bytes",
        evaluation.encoded_legal_actions * std::mem::size_of::<(PackedCoord, f32)>(),
    )?;
    eval.set_item("cache_inserts", evaluation.cache_inserts)?;
    eval.set_item("cache_insert_skipped", evaluation.cache_insert_skipped)?;
    eval.set_item("cache_size", cache_len)?;
    eval.set_item("cache_size_peak", evaluation.cache_size_peak.max(cache_len))?;
    eval.set_item("encoding_seconds", evaluation.encoding_seconds)?;
    eval.set_item("evaluator_seconds", evaluation.evaluator_seconds)?;
    eval.set_item("parse_seconds", evaluation.parse_seconds)?;

    let diagnostics = PyDict::new(py);
    diagnostics.set_item("tree", tree)?;
    diagnostics.set_item("evaluation", eval)?;
    Ok(diagnostics)
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model1MctsSession>()?;
    Ok(())
}
