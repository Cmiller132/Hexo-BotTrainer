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

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use hexo_engine::{
    apply_placement, unpack_coord, HexoState as RustHexoState, PackedCoord, Placement,
};

// Threat-Space Search core, shared with the hexgt lineage (leaf value override +
// root move-selection guard). Pure board geometry; no graph/network coupling.
use crate::threats_shared as threats;

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

enum ContinuousPhase {
    Active,
    AwaitRootEval,
    Empty,
}

// Deterministic per-(game, ply) RNG streams for the continuous scheduler: root
// Dirichlet noise and played-move sampling must draw independently (the lockstep
// path got this from separate seed bases; here one mixer with a stream tag).
const SEED_STREAM_ROOT_NOISE: u64 = 0;
const SEED_STREAM_MOVE_SELECT: u64 = 1;

struct ContinuousSlot {
    game_key: u64,
    ply: u32,
    search: Option<RustSearch>,
    phase: ContinuousPhase,
    in_flight: u32,
    baseline: HashMap<PackedCoord, u32>,
}

enum ContinuousEvalItem {
    Leaf(RustLeaf),
    RootInit {
        slot_index: usize,
        state: RustHexoState,
        state_hash: hexo_utils::StateHash,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContinuousFlushDecision {
    Hold,
    Flush { no_progress: bool },
    Stop,
}

#[derive(Default)]
struct ContinuousSchedulerStats {
    flush_count: u64,
    no_progress_flushes: u64,
    queued_states: u64,
    flushed_states: u64,
    // Keyed by power-of-two bucket (next_power_of_two of the unique state
    // count), so the epoch JSON stays a handful of entries instead of one per
    // distinct flush size. The exact mean is flushed_states / flush_count.
    flush_size_histogram: HashMap<usize, u64>,
    on_move_seconds: f64,
    moves_decided: u64,
}

fn continuous_flush_decision(
    queue_len: usize,
    flush_target: usize,
    made_progress: bool,
) -> ContinuousFlushDecision {
    if queue_len == 0 {
        return if made_progress {
            ContinuousFlushDecision::Hold
        } else {
            ContinuousFlushDecision::Stop
        };
    }
    if queue_len >= flush_target {
        return ContinuousFlushDecision::Flush {
            no_progress: !made_progress,
        };
    }
    if !made_progress {
        return ContinuousFlushDecision::Flush { no_progress: true };
    }
    ContinuousFlushDecision::Hold
}

fn continuous_completion_ready(completed_visits: u32, target_visits: u32, in_flight: u32) -> bool {
    completed_visits >= target_visits && in_flight == 0
}

#[derive(Default)]
struct ContinuousTreeStats {
    root_count: usize,
    completed_visits: u64,
    node_count: usize,
    active_edge_count: usize,
    hidden_prior_count: usize,
    root_active_edges: usize,
    root_hidden_priors: usize,
    max_nodes_per_root: usize,
    max_active_edges_per_root: usize,
    max_hidden_priors_per_root: usize,
    max_active_edges_per_node: usize,
    max_hidden_priors_per_node: usize,
    active_edge_bytes: usize,
    hidden_prior_bytes: usize,
    shared_prior_nodes: usize,
    shared_prior_refs: usize,
}

impl ContinuousTreeStats {
    fn record(&mut self, search: &RustSearch) {
        let stats = search.diagnostics();
        self.root_count += 1;
        self.completed_visits += search.completed_visits as u64;
        self.node_count += stats.node_count;
        self.active_edge_count += stats.active_edge_count;
        self.hidden_prior_count += stats.hidden_prior_count;
        self.root_active_edges += stats.root_active_edges;
        self.root_hidden_priors += stats.root_hidden_priors;
        self.max_nodes_per_root = self.max_nodes_per_root.max(stats.node_count);
        self.max_active_edges_per_root =
            self.max_active_edges_per_root.max(stats.active_edge_count);
        self.max_hidden_priors_per_root =
            self.max_hidden_priors_per_root.max(stats.hidden_prior_count);
        self.max_active_edges_per_node =
            self.max_active_edges_per_node.max(stats.max_active_edges_per_node);
        self.max_hidden_priors_per_node =
            self.max_hidden_priors_per_node.max(stats.max_hidden_priors_per_node);
        self.active_edge_bytes += stats.active_edge_bytes;
        self.hidden_prior_bytes += stats.hidden_prior_bytes;
        self.shared_prior_nodes += stats.shared_prior_nodes;
        self.shared_prior_refs += stats.shared_prior_refs;
    }
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

    #[pyo3(signature = (game_keys, states, visits, c_puct, temperature, seed, evaluator, virtual_batch_size=None, active_root_limit=None, root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None, root_policy_temperature=None, fpu_reduction=None, virtual_loss=None, widening_policy_mass=None, widening_max_children=None, widening_min_children=None, forced_playout_k=None, move_temperatures=None, root_policy_temperatures=None))]
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
        forced_playout_k: Option<f32>,
        // Optional per-root move-selection temperature (one entry per game key,
        // aligned with `game_keys`). When provided, each root's played action is
        // sampled at its own temperature instead of the scalar `temperature`; this
        // is how self-play applies a per-move temperature decay across a batch of
        // games that are each at a different move number. `temperature` remains the
        // validated fallback used when this is None.
        move_temperatures: Option<Vec<f32>>,
        // Optional per-root root-policy softmax temperature (one entry per game
        // key). Lets self-play run a KataGo-style early ramp (e.g. 1.25 at the
        // opening decaying to 1.1) where each game sits at its own ply. The scalar
        // `root_policy_temperature` is the fallback when this is None.
        root_policy_temperatures: Option<Vec<f32>>,
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
        // Resolve the per-root move-selection temperatures once: an explicit vector
        // (validated to match the root count and be finite/non-negative) or the
        // scalar `temperature` broadcast to every root.
        let move_temps: Vec<f32> = match move_temperatures {
            Some(values) => {
                if values.len() != roots.len() {
                    return Err(PyValueError::new_err(format!(
                        "move_temperatures has {} entries for {} roots",
                        values.len(),
                        roots.len()
                    )));
                }
                for value in &values {
                    if !value.is_finite() || *value < 0.0 {
                        return Err(PyValueError::new_err(
                            "move_temperatures entries must be finite and >= 0",
                        ));
                    }
                }
                values
            }
            None => vec![temperature; roots.len()],
        };
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
        let root_policy_temperature = validate_positive_f32(
            "root_policy_temperature",
            root_policy_temperature.unwrap_or(1.0),
        )?;
        // Per-root root-policy temperatures (early-ramp support): an explicit
        // vector aligned with `game_keys`, or the validated scalar broadcast.
        let root_policy_temps: Vec<f32> = match root_policy_temperatures {
            Some(values) => {
                if values.len() != roots.len() {
                    return Err(PyValueError::new_err(format!(
                        "root_policy_temperatures has {} entries for {} roots",
                        values.len(),
                        roots.len()
                    )));
                }
                for value in &values {
                    if !value.is_finite() || *value <= 0.0 {
                        return Err(PyValueError::new_err(
                            "root_policy_temperatures entries must be finite and > 0",
                        ));
                    }
                }
                values
            }
            None => vec![root_policy_temperature; roots.len()],
        };
        let fpu_reduction =
            validate_nonnegative_f32("fpu_reduction", fpu_reduction.unwrap_or(0.20))?;
        let virtual_loss = validate_nonnegative_f32("virtual_loss", virtual_loss.unwrap_or(1.0))?;
        let forced_playout_k =
            validate_nonnegative_f32("forced_playout_k", forced_playout_k.unwrap_or(0.0))?;
        let root_noise_config =
            root_noise_config(root_dirichlet_total_alpha, root_dirichlet_noise_fraction)?;
        // KataGo zeroes the root FPU reduction while Dirichlet root noise is on
        // (`rootFpuReductionMax = 0`) so unvisited noise-boosted root children are
        // not suppressed below `parent_value - fpu` before their first visit.
        // Noise-free searches (eval/play, PCR fast searches) keep the normal FPU.
        let root_fpu_reduction = if root_noise_config.is_some() {
            0.0
        } else {
            fpu_reduction
        };

        let widening_mass = widening_policy_mass.unwrap_or(0.95);
        if !widening_mass.is_finite() || widening_mass <= 0.0 || widening_mass > 1.0 {
            return Err(PyValueError::new_err(
                "widening_policy_mass must be in (0, 1]",
            ));
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
                    search.set_forced_playout_k(forced_playout_k);
                    search.set_root_fpu_reduction(root_fpu_reduction);
                    // Reused roots carry raw eval priors; apply the root-policy
                    // temperature here (before noise, matching the fresh-root
                    // order) so it covers every searched position, not just each
                    // game's first root.
                    search.apply_root_policy_temperature(root_policy_temps[index]);
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
                    root_fpu_reduction,
                    root_policy_temps[index],
                    root_noise(root_noise_config, seed, index),
                    widening,
                    forced_playout_k,
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
                    move_temps[index],
                    seed.wrapping_add(index as u64),
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let results = build_search_result_payloads(
            py,
            &searches,
            Some(&batch_diagnostics),
            &move_temps,
            seed,
            Some(&baselines),
            c_puct,
            forced_playout_k,
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

    #[pyo3(signature = (game_keys, states, evaluator, on_move, visits, c_puct, base_seed, virtual_batch_size, flush_target, active_root_limit, temperature_by_ply, root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None, root_policy_temperature=None, fpu_reduction=None, virtual_loss=None, widening_policy_mass=None, widening_max_children=None, widening_min_children=None, forced_playout_k=None))]
    fn run_continuous(
        &mut self,
        py: Python<'_>,
        game_keys: Vec<u64>,
        states: &Bound<'_, PyAny>,
        evaluator: &Bound<'_, PyAny>,
        on_move: &Bound<'_, PyAny>,
        visits: u32,
        c_puct: f32,
        base_seed: u64,
        virtual_batch_size: u32,
        flush_target: usize,
        active_root_limit: usize,
        temperature_by_ply: Vec<f32>,
        root_dirichlet_total_alpha: Option<f32>,
        root_dirichlet_noise_fraction: Option<f32>,
        root_policy_temperature: Option<f32>,
        fpu_reduction: Option<f32>,
        virtual_loss: Option<f32>,
        widening_policy_mass: Option<f32>,
        widening_max_children: Option<u32>,
        widening_min_children: Option<u32>,
        forced_playout_k: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        validate_search_inputs(visits, c_puct, 0.0)?;
        let roots = states_from_py_states(py, states)?;
        if roots.len() != game_keys.len() {
            return Err(PyValueError::new_err(format!(
                "dense_cnn continuous MCTS received {} game keys for {} states",
                game_keys.len(),
                roots.len()
            )));
        }
        let root_limit = validate_positive_usize("active_root_limit", active_root_limit)?;
        if roots.len() > root_limit {
            return Err(PyValueError::new_err(format!(
                "dense_cnn continuous MCTS received {} active roots, above strict limit {}",
                roots.len(),
                root_limit
            )));
        }
        let leaf_batch_per_root = validate_positive_u32("virtual_batch_size", virtual_batch_size)?;
        let flush_target = validate_positive_usize("flush_target", flush_target)?;
        let root_policy_temperature = validate_positive_f32(
            "root_policy_temperature",
            root_policy_temperature.unwrap_or(1.0),
        )?;
        let fpu_reduction =
            validate_nonnegative_f32("fpu_reduction", fpu_reduction.unwrap_or(0.20))?;
        let virtual_loss = validate_nonnegative_f32("virtual_loss", virtual_loss.unwrap_or(1.0))?;
        let forced_playout_k =
            validate_nonnegative_f32("forced_playout_k", forced_playout_k.unwrap_or(0.0))?;
        let root_noise_config =
            root_noise_config(root_dirichlet_total_alpha, root_dirichlet_noise_fraction)?;
        // Root-only FPU: zero under Dirichlet noise (KataGo `rootFpuReductionMax`),
        // mirroring the lockstep `search` derivation above.
        let root_fpu_reduction = if root_noise_config.is_some() {
            0.0
        } else {
            fpu_reduction
        };
        let widening_mass = widening_policy_mass.unwrap_or(0.95);
        if !widening_mass.is_finite() || widening_mass <= 0.0 || widening_mass > 1.0 {
            return Err(PyValueError::new_err(
                "widening_policy_mass must be in (0, 1]",
            ));
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
        if temperature_by_ply.is_empty() {
            return Err(PyValueError::new_err(
                "temperature_by_ply must not be empty",
            ));
        }
        if temperature_by_ply
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(PyValueError::new_err(
                "temperature_by_ply entries must be finite and >= 0",
            ));
        }

        let evaluation_stats = new_shared_evaluation_stats();
        let mut slots = Vec::with_capacity(roots.len());
        let mut queue: Vec<ContinuousEvalItem> = Vec::new();
        for (slot_index, (game_key, root)) in
            game_keys.into_iter().zip(roots.into_iter()).enumerate()
        {
            let root_hash = state_hash(&root);
            let mut slot = ContinuousSlot {
                game_key,
                ply: 0,
                search: None,
                phase: ContinuousPhase::AwaitRootEval,
                in_flight: 0,
                baseline: HashMap::new(),
            };
            if let Some(mut search) = self.searches.remove(&game_key) {
                if search.root_hash == root_hash {
                    search.set_additional_visits(visits);
                    search.set_forced_playout_k(forced_playout_k);
                    search.set_root_fpu_reduction(root_fpu_reduction);
                    // Reused roots carry raw eval priors; apply the root-policy
                    // temperature before noise (fresh-root order of operations).
                    search.apply_root_policy_temperature(root_policy_temperature);
                    if let Some(noise) = root_noise_exact(
                        root_noise_config,
                        mix_seed(base_seed, game_key, 0, SEED_STREAM_ROOT_NOISE),
                    ) {
                        search.apply_root_dirichlet_noise(noise);
                    }
                    slot.baseline = search.root_edge_visits().into_iter().collect();
                    slot.search = Some(search);
                    slot.phase = ContinuousPhase::Active;
                }
            }
            if matches!(slot.phase, ContinuousPhase::AwaitRootEval) {
                queue.push(ContinuousEvalItem::RootInit {
                    slot_index,
                    state: root,
                    state_hash: root_hash,
                });
            }
            slots.push(slot);
        }

        let mut stats = ContinuousSchedulerStats::default();
        let mut tree_stats = ContinuousTreeStats::default();
        // Select↔eval overlap (the lockstep A1 pipeline, slot-scheduler form):
        // while a flush batch is being evaluated on the GIL thread, the NEXT
        // select pass runs on a scoped worker. Safe for the same reason lockstep
        // is — virtual loss is applied at selection — plus one continuous-only
        // guard: a slot with prefetched leaves has in_flight > 0, so it cannot
        // complete (and thus cannot advance_root and invalidate those leaves'
        // node ids) until they are backed up. A prefetch that made no progress
        // is discarded so the next iteration re-selects synchronously AFTER the
        // backup freed the in-flight paths (lockstep's narrow-tree fallback).
        let mut prefetched: Option<(Vec<RustLeaf>, bool)> = None;
        while continuous_has_work(&slots) || !queue.is_empty() {
            // SELECT: every Active slot tops its in-flight leaves up to the
            // virtual-batch cap, in parallel across slots; results fold into the
            // queue in slot order, so scheduling stays a pure function of
            // (slots, seeds) — no timing dependence. Every slot gets a full turn
            // each pass; an early queue-capacity break here would permanently
            // starve high slots (low slots refill their in-flight budget every
            // pass and would hit the cap alone). The GIL is released so the
            // Python writer thread can drain finished games while selection runs.
            let (new_leaves, made_progress) = match prefetched.take() {
                Some(result) => result,
                None => py.detach(|| {
                    select_continuous_pass(&mut slots, c_puct, leaf_batch_per_root, virtual_loss)
                })?,
            };
            queue.extend(new_leaves.into_iter().map(ContinuousEvalItem::Leaf));

            let decision = continuous_flush_decision(queue.len(), flush_target, made_progress);
            if let ContinuousFlushDecision::Flush { no_progress } = decision {
                if no_progress {
                    stats.no_progress_flushes += 1;
                }
                let items = std::mem::take(&mut queue);
                stats.flush_count += 1;
                stats.queued_states += items.len() as u64;
                let unique_before = evaluation_stats
                    .lock()
                    .expect("evaluation stats mutex poisoned")
                    .unique_states;
                let requests: Vec<RustEvaluationRequest> = items
                    .iter()
                    .map(|item| match item {
                        ContinuousEvalItem::Leaf(leaf) => RustEvaluationRequest {
                            state: &leaf.state,
                            state_hash: leaf.state_hash,
                        },
                        ContinuousEvalItem::RootInit {
                            state, state_hash, ..
                        } => RustEvaluationRequest {
                            state,
                            state_hash: *state_hash,
                        },
                    })
                    .collect();
                // Tree-access discipline inside the scope: ONLY the select
                // worker touches slots/trees; the eval reads the owned flush
                // items + the (thread-safe) cache. Backup runs after the join
                // with exclusive access, so trees have one mutator at every
                // instant. An eval error propagates after the scope's implicit
                // join, so the worker never outlives its borrows.
                let slots_worker: &mut Vec<ContinuousSlot> = &mut slots;
                let (evaluations, prefetch_result) =
                    std::thread::scope(|scope| -> PyResult<_> {
                        let select_handle = scope.spawn(move || {
                            select_continuous_pass(
                                slots_worker,
                                c_puct,
                                leaf_batch_per_root,
                                virtual_loss,
                            )
                        });
                        let evaluations = evaluate_model1_state_refs_cached(
                            py,
                            evaluator,
                            &requests,
                            &self.evaluation_cache,
                            Some(&evaluation_stats),
                            self.cache_max_states,
                        )?;
                        let prefetch = select_handle
                            .join()
                            .expect("continuous select worker panicked")?;
                        Ok((evaluations, prefetch))
                    })?;
                let unique_after = evaluation_stats
                    .lock()
                    .expect("evaluation stats mutex poisoned")
                    .unique_states;
                let unique_flushed = unique_after.saturating_sub(unique_before);
                stats.flushed_states += unique_flushed as u64;
                *stats
                    .flush_size_histogram
                    .entry(unique_flushed.max(1).next_power_of_two())
                    .or_insert(0) += 1;
                backup_continuous_items(
                    &mut slots,
                    items,
                    &evaluations,
                    visits,
                    fpu_reduction,
                    root_fpu_reduction,
                    root_policy_temperature,
                    root_noise_config,
                    widening,
                    forced_playout_k,
                    base_seed,
                    virtual_loss,
                )?;
                // A no-progress prefetch is stale advice (paths it saw as
                // pending were just backed up): drop it and re-select fresh.
                prefetched = if prefetch_result.1 {
                    Some(prefetch_result)
                } else {
                    None
                };
            }

            let moves_decided = complete_continuous_slots(
                py,
                on_move,
                &mut slots,
                visits,
                c_puct,
                forced_playout_k,
                root_fpu_reduction,
                root_policy_temperature,
                &temperature_by_ply,
                base_seed,
                root_noise_config,
                &mut queue,
                &mut stats,
                &mut tree_stats,
            )?;

            // Stop fires only when the queue is empty AND no root could select —
            // healthy schedules never reach it (slots exhaust to Empty and the
            // loop condition exits). If a completion pass also made no progress,
            // the scheduler is genuinely wedged (e.g. a root with no legal
            // actions): fail loud rather than return a silently-partial epoch.
            if matches!(decision, ContinuousFlushDecision::Stop) && moves_decided == 0 {
                let stuck = slots
                    .iter()
                    .filter(|slot| !matches!(slot.phase, ContinuousPhase::Empty))
                    .count();
                return Err(PyRuntimeError::new_err(format!(
                    "dense_cnn continuous MCTS scheduler stalled with {stuck} unfinished slots \
                     (queue empty, no selectable leaves, no completable roots)"
                )));
            }
        }

        for slot in slots {
            if let Some(search) = slot.search {
                if matches!(slot.phase, ContinuousPhase::Active) {
                    self.searches.insert(slot.game_key, search);
                }
            }
        }

        let dict = PyDict::new(py);
        dict.set_item("flush_count", stats.flush_count)?;
        dict.set_item("queued_states", stats.queued_states)?;
        dict.set_item("flushed_states", stats.flushed_states)?;
        dict.set_item(
            "mean_flush_states",
            if stats.flush_count > 0 {
                stats.flushed_states as f64 / stats.flush_count as f64
            } else {
                0.0
            },
        )?;
        dict.set_item("no_progress_flushes", stats.no_progress_flushes)?;
        dict.set_item("moves_decided", stats.moves_decided)?;
        let hist = PyDict::new(py);
        let mut hist_items: Vec<_> = stats.flush_size_histogram.into_iter().collect();
        hist_items.sort_unstable_by_key(|(size, _)| *size);
        for (size, count) in hist_items {
            hist.set_item(size, count)?;
        }
        dict.set_item("flush_size_histogram", hist)?;
        dict.set_item("on_move_seconds", stats.on_move_seconds)?;
        let eval_snapshot = evaluation_stats
            .lock()
            .expect("evaluation stats mutex poisoned")
            .clone();
        let cache_len = self
            .evaluation_cache
            .lock()
            .expect("evaluation cache mutex poisoned")
            .len();
        let batch_diagnostics = build_continuous_batch_diagnostics(
            py,
            &tree_stats,
            &eval_snapshot,
            visits,
            leaf_batch_per_root,
            cache_len,
        )?;
        dict.set_item("mcts_batch_diagnostics", batch_diagnostics)?;
        Ok(dict.into_any().unbind())
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
                        let result = select_leaf_batch(
                            select_searches,
                            c_puct,
                            leaf_batch_per_root,
                            virtual_loss,
                        );
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
                } else if let Some(verdict) = threats::analyze(&selected.state).verdict() {
                    // TSS phase-aware hitting-set leaf value OVERRIDE. This fresh
                    // leaf is a proven 1-ply forced win/loss for the side to move
                    // (own win completable with B placements, or opponent threats
                    // whose minimum hitting set exceeds B). Back up the hard +-1
                    // through the same terminal path and SKIP the network
                    // eval/node creation — a proven node is as good as terminal,
                    // and expansion injection has already guaranteed the covering
                    // children exist for the must-answer (non-proven) case. Fires
                    // at EVERY fresh leaf (root child and every inner leaf), reading
                    // only board occupancy + phase, so it is crop-independent and
                    // recognizes even out-of-crop forced losses.
                    let leaf_player = selected.state.current_player();
                    search.backup_virtual(&selected.path, leaf_player, verdict, virtual_loss);
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

fn select_continuous_leaves(
    search: &mut RustSearch,
    slot_index: usize,
    c_puct: f32,
    budget: u32,
    virtual_loss: f32,
) -> PyResult<(Vec<RustLeaf>, bool, u32)> {
    let mut leaves = Vec::new();
    let mut made_progress = false;
    let mut added_in_flight = 0u32;
    let budget = budget.min(search.remaining_visits());
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
        } else if let Some(verdict) = threats::analyze(&selected.state).verdict() {
            let leaf_player = selected.state.current_player();
            search.backup_virtual(&selected.path, leaf_player, verdict, virtual_loss);
        } else {
            search.mark_pending(selected.parent_node, selected.edge_index, 1);
            added_in_flight += 1;
            leaves.push(RustLeaf {
                root_index: slot_index,
                parent_node: selected.parent_node,
                edge_index: selected.edge_index,
                path: selected.path,
                state: selected.state,
                state_hash: selected.state_hash,
            });
        }
    }
    Ok((leaves, made_progress, added_in_flight))
}

#[allow(clippy::too_many_arguments)]
// One parallel selection sweep over every Active slot: top each slot's
// in-flight leaves up to the virtual-batch cap (rayon across slots — each
// closure owns one slot's tree, no shared mutable state; results concatenate in
// slot order so the sweep is deterministic). Pure Rust: callers run it under
// `py.detach` (synchronous form) or on the scoped prefetch worker (overlap form).
fn select_continuous_pass(
    slots: &mut [ContinuousSlot],
    c_puct: f32,
    leaf_batch_per_root: u32,
    virtual_loss: f32,
) -> PyResult<(Vec<RustLeaf>, bool)> {
    let per_slot: PyResult<Vec<(Vec<RustLeaf>, bool)>> = slots
        .par_iter_mut()
        .enumerate()
        .map(|(slot_index, slot)| {
            if !matches!(slot.phase, ContinuousPhase::Active) {
                return Ok((Vec::new(), false));
            }
            let cap = leaf_batch_per_root.saturating_sub(slot.in_flight);
            if cap == 0 {
                return Ok((Vec::new(), false));
            }
            let Some(search) = slot.search.as_mut() else {
                return Ok((Vec::new(), false));
            };
            if !search.needs_visits() {
                return Ok((Vec::new(), false));
            }
            let (leaves, progressed, added_in_flight) =
                select_continuous_leaves(search, slot_index, c_puct, cap, virtual_loss)?;
            slot.in_flight = slot.in_flight.saturating_add(added_in_flight);
            Ok((leaves, progressed))
        })
        .collect();
    let mut leaves = Vec::new();
    let mut made_progress = false;
    for (slot_leaves, progressed) in per_slot? {
        made_progress |= progressed;
        leaves.extend(slot_leaves);
    }
    Ok((leaves, made_progress))
}

// Attach one flush's evaluations back onto their trees: NN leaves expand and
// back up; RootInit results construct the slot's new search. Exclusive
// `&mut slots` access — called only after the eval/prefetch scope joins.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn backup_continuous_items(
    slots: &mut [ContinuousSlot],
    items: Vec<ContinuousEvalItem>,
    evaluations: &[Arc<RustEvaluation>],
    visits: u32,
    fpu_reduction: f32,
    root_fpu_reduction: f32,
    root_policy_temperature: f32,
    root_noise_config: Option<RootNoiseConfig>,
    widening: Widening,
    forced_playout_k: f32,
    base_seed: u64,
    virtual_loss: f32,
) -> PyResult<()> {
    for (item, evaluation) in items.into_iter().zip(evaluations.iter()) {
        match item {
            ContinuousEvalItem::Leaf(leaf) => {
                let slot = &mut slots[leaf.root_index];
                let Some(search) = slot.search.as_mut() else {
                    return Err(PyValueError::new_err(
                        "continuous MCTS leaf resolved for empty slot",
                    ));
                };
                let child_id = search.add_node_from_eval(
                    &leaf.state,
                    leaf.state_hash,
                    Arc::clone(evaluation),
                )?;
                search.nodes[leaf.parent_node].edges[leaf.edge_index].child = Some(child_id);
                search.mark_pending(leaf.parent_node, leaf.edge_index, -1);
                slot.in_flight = slot.in_flight.saturating_sub(1);
                let child_player = search.nodes[child_id].player;
                let child_value = search.nodes[child_id].value();
                search.backup_virtual(&leaf.path, child_player, child_value, virtual_loss);
            }
            ContinuousEvalItem::RootInit {
                slot_index, state, ..
            } => {
                let slot = &mut slots[slot_index];
                let search = RustSearch::new(
                    state,
                    &**evaluation,
                    visits,
                    fpu_reduction,
                    root_fpu_reduction,
                    root_policy_temperature,
                    root_noise_exact(
                        root_noise_config,
                        mix_seed(base_seed, slot.game_key, slot.ply, SEED_STREAM_ROOT_NOISE),
                    ),
                    widening,
                    forced_playout_k,
                )?;
                if search.root_edges_empty() {
                    return Err(PyValueError::new_err(
                        "dense_cnn continuous MCTS root has no legal actions",
                    ));
                }
                slot.baseline = search.root_edge_visits().into_iter().collect();
                slot.search = Some(search);
                slot.phase = ContinuousPhase::Active;
                slot.in_flight = 0;
            }
        }
    }
    Ok(())
}

// Decide moves for every root that reached its visit target. Returns the number
// of moves decided so the caller can distinguish a healthy pass from a stall.
//
// Per-move payloads carry root-only diagnostics (batch=None): a per-move batch
// snapshot would clone the cumulative evaluation stats ~once per position and
// the Python driver would then sum thousands of overlapping snapshots. The
// epoch-wide aggregate is built once in `run_continuous` instead.
#[allow(clippy::too_many_arguments)]
fn complete_continuous_slots(
    py: Python<'_>,
    on_move: &Bound<'_, PyAny>,
    slots: &mut [ContinuousSlot],
    visits: u32,
    c_puct: f32,
    forced_playout_k: f32,
    root_fpu_reduction: f32,
    root_policy_temperature: f32,
    temperature_by_ply: &[f32],
    base_seed: u64,
    root_noise_config: Option<RootNoiseConfig>,
    queue: &mut Vec<ContinuousEvalItem>,
    stats: &mut ContinuousSchedulerStats,
    tree_stats: &mut ContinuousTreeStats,
) -> PyResult<u64> {
    let mut moves_decided = 0u64;
    for slot_index in 0..slots.len() {
        if !matches!(slots[slot_index].phase, ContinuousPhase::Active) {
            continue;
        }
        let complete = slots[slot_index]
            .search
            .as_ref()
            .map(|search| {
                continuous_completion_ready(
                    search.completed_visits,
                    search.target_visits,
                    slots[slot_index].in_flight,
                )
            })
            .unwrap_or(false);
        if !complete {
            continue;
        }

        let game_key = slots[slot_index].game_key;
        let ply = slots[slot_index].ply;
        let move_seed = mix_seed(base_seed, game_key, ply, SEED_STREAM_MOVE_SELECT);
        let temperature = temperature_for_ply(temperature_by_ply, ply);
        let baseline = slots[slot_index].baseline.clone();
        let payloads = {
            let search = slots[slot_index]
                .search
                .as_ref()
                .expect("active continuous slot has search");
            tree_stats.record(search);
            let baselines = vec![baseline];
            build_search_result_payloads(
                py,
                std::slice::from_ref(search),
                None,
                &[temperature],
                move_seed,
                Some(&baselines),
                c_puct,
                forced_playout_k,
            )?
        };
        let payloads = payloads.bind(py).downcast::<PyList>()?;
        let payload = payloads.get_item(0)?;
        let payload_dict = payload.downcast::<PyDict>()?;
        let action_id: PackedCoord = payload_dict
            .get_item("action_id")?
            .ok_or_else(|| PyValueError::new_err("continuous payload missing action_id"))?
            .extract()?;

        moves_decided += 1;
        stats.moves_decided += 1;
        let started = std::time::Instant::now();
        let response = on_move.call1((game_key, payload_dict))?;
        stats.on_move_seconds += started.elapsed().as_secs_f64();
        if response.is_none() {
            slots[slot_index].search = None;
            slots[slot_index].phase = ContinuousPhase::Empty;
            continue;
        }
        let tuple = response.downcast::<PyTuple>()?;
        if tuple.is_empty() {
            return Err(PyValueError::new_err(
                "continuous on_move response tuple is empty",
            ));
        }
        let action: String = tuple.get_item(0)?.extract()?;
        match action.as_str() {
            "advance" => {
                if tuple.len() != 2 {
                    return Err(PyValueError::new_err(
                        "advance response must be ('advance', state)",
                    ));
                }
                let next_state = single_state_from_py(py, &tuple.get_item(1)?)?;
                let next_hash = state_hash(&next_state);
                let mut keep_promoted = false;
                if let Some(search) = slots[slot_index].search.as_mut() {
                    if search.advance_root(action_id)? && search.root_hash == next_hash {
                        search.set_additional_visits(visits);
                        search.set_forced_playout_k(forced_playout_k);
                        search.set_root_fpu_reduction(root_fpu_reduction);
                        // Reused roots carry raw eval priors; apply the
                        // root-policy temperature before noise.
                        search.apply_root_policy_temperature(root_policy_temperature);
                        if let Some(noise) = root_noise_exact(
                            root_noise_config,
                            mix_seed(base_seed, game_key, ply + 1, SEED_STREAM_ROOT_NOISE),
                        ) {
                            search.apply_root_dirichlet_noise(noise);
                        }
                        slots[slot_index].baseline =
                            search.root_edge_visits().into_iter().collect();
                        keep_promoted = true;
                    }
                }
                slots[slot_index].ply = slots[slot_index].ply.saturating_add(1);
                slots[slot_index].in_flight = 0;
                if keep_promoted {
                    slots[slot_index].phase = ContinuousPhase::Active;
                } else {
                    slots[slot_index].search = None;
                    slots[slot_index].phase = ContinuousPhase::AwaitRootEval;
                    queue.push(ContinuousEvalItem::RootInit {
                        slot_index,
                        state: next_state,
                        state_hash: next_hash,
                    });
                }
            }
            "replace" => {
                if tuple.len() != 3 {
                    return Err(PyValueError::new_err(
                        "replace response must be ('replace', new_key, state)",
                    ));
                }
                let new_key: u64 = tuple.get_item(1)?.extract()?;
                let next_state = single_state_from_py(py, &tuple.get_item(2)?)?;
                let next_hash = state_hash(&next_state);
                slots[slot_index].game_key = new_key;
                slots[slot_index].ply = 0;
                slots[slot_index].search = None;
                slots[slot_index].baseline.clear();
                slots[slot_index].in_flight = 0;
                slots[slot_index].phase = ContinuousPhase::AwaitRootEval;
                queue.push(ContinuousEvalItem::RootInit {
                    slot_index,
                    state: next_state,
                    state_hash: next_hash,
                });
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "continuous on_move returned unsupported action {other:?}"
                )));
            }
        }
    }
    Ok(moves_decided)
}

fn continuous_has_work(slots: &[ContinuousSlot]) -> bool {
    slots
        .iter()
        .any(|slot| !matches!(slot.phase, ContinuousPhase::Empty))
}

fn temperature_for_ply(values: &[f32], ply: u32) -> f32 {
    let index = (ply as usize).min(values.len().saturating_sub(1));
    values[index]
}

fn single_state_from_py(py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<RustHexoState> {
    let tuple = PyTuple::new(py, [state])?;
    let states = states_from_py_states(py, tuple.as_any())?;
    states
        .into_iter()
        .next()
        .ok_or_else(|| PyValueError::new_err("expected one state"))
}

fn build_search_result_payloads(
    py: Python<'_>,
    searches: &[RustSearch],
    batch_diagnostics: Option<&Bound<'_, PyDict>>,
    temperatures: &[f32],
    seed: u64,
    baselines: Option<&[HashMap<PackedCoord, u32>]>,
    c_puct: f32,
    forced_playout_k: f32,
) -> PyResult<Py<PyAny>> {
    // The Python wrapper expects byte-backed policies. This avoids allocating a
    // Python tuple for every legal move while still supporting lazy iteration.
    let results = PyList::empty(py);
    for (index, search) in searches.iter().enumerate() {
        let result = PyDict::new(py);
        let root = search.root();
        let baseline = baselines.and_then(|items| items.get(index));
        // RAW visit policy: drives the played move (kept identical to the action
        // used to advance the native root in `run`, so play stays diverse) and the
        // reported visit total.
        let (policy_action_ids, policy_weights, policy_total) = visit_policy(root, baseline);
        // EXPORTED policy = training target: with forced playouts on, prune the
        // forced visits back out (KataGo policy-target pruning) so the network is
        // not taught to like the artificially-explored moves. With k==0 this is a
        // byte-identical copy of the raw policy.
        let (export_action_ids, export_weights) = if forced_playout_k > 0.0 {
            pruned_visit_policy(root, baseline, forced_playout_k, c_puct)
        } else {
            (policy_action_ids.clone(), policy_weights.clone())
        };
        let (root_prior_action_ids, root_prior_weights) = root_prior_policy(root);
        // TSS move-selection guard: identical inputs + determinism as
        // `select_search_action`, so the reported action matches the one used to
        // advance the native root. Masks the RAW selection weights only; the
        // exported training policy below is left untouched.
        let guarded_weights =
            tactical_guard_weights(&search.root_state, &policy_action_ids, &policy_weights);
        let selected = select_action_from_policy(
            &policy_action_ids,
            &guarded_weights,
            temperatures[index],
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
        let action_byte_len = export_action_ids.len() * std::mem::size_of::<u32>();
        let weight_byte_len = export_weights.len() * std::mem::size_of::<f32>();
        let action_bytes = unsafe {
            std::slice::from_raw_parts(export_action_ids.as_ptr() as *const u8, action_byte_len)
        };
        let weight_bytes = unsafe {
            std::slice::from_raw_parts(export_weights.as_ptr() as *const u8, weight_byte_len)
        };
        result.set_item(
            "visit_policy_action_ids_bytes",
            PyBytes::new(py, action_bytes),
        )?;
        result.set_item("visit_policy_weights_bytes", PyBytes::new(py, weight_bytes))?;
        result.set_item("visit_policy_count", export_action_ids.len())?;
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
    let action_ids: Vec<PackedCoord> = pairs.iter().map(|(action_id, _prior)| *action_id).collect();
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
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and > 0"
        )));
    }
    Ok(value)
}

fn validate_nonnegative_f32(name: &str, value: f32) -> PyResult<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and >= 0"
        )));
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

fn root_noise_exact(config: Option<RootNoiseConfig>, seed: u64) -> Option<RootDirichletNoise> {
    let config = config?;
    Some(RootDirichletNoise {
        total_alpha: config.total_alpha,
        fraction: config.fraction,
        seed,
    })
}

fn mix_seed(base_seed: u64, game_key: u64, ply: u32, stream: u64) -> u64 {
    let mut value = base_seed ^ 0xA076_1D64_78BD_642F;
    value ^= game_key.wrapping_mul(0xE703_7ED1_A0B4_28DB);
    value ^= (ply as u64).wrapping_mul(0x8EBC_6AF0_9C88_C6E3);
    value ^= stream.wrapping_mul(0x5899_65CC_7537_4CC3);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Tactical-safety classification of a candidate ROOT move (TSS move-selection
/// guard): `+1` if playing it is a proven 1-ply WIN for the side to move, `-1` if
/// it is a proven 1-ply LOSS (an immediate terminal loss, or it leaves the
/// opponent a forced win), else `0`. Exact and cheap — one placement + one
/// phase-aware threat analysis — matching the depth-1 scope of the leaf override.
fn classify_root_move(root_state: &RustHexoState, action_id: PackedCoord) -> i8 {
    let me = root_state.current_player();
    let mut child = root_state.clone();
    let coord = unpack_coord(action_id);
    match apply_placement(&mut child, Placement { coord }) {
        Err(_) => 0, // not a legal move (shouldn't occur for a materialized edge)
        Ok(res) => {
            if let Some(outcome) = res.outcome {
                return if outcome.winner == me { 1 } else { -1 };
            }
            match threats::analyze(&child).verdict() {
                // verdict() is from the child's side to move; flip to `me` when the
                // turn has passed to the opponent.
                Some(v) => {
                    let ours = if child.current_player() == me { v } else { -v };
                    if ours > 0.5 {
                        1
                    } else if ours < -0.5 {
                        -1
                    } else {
                        0
                    }
                }
                None => 0,
            }
        }
    }
}

/// Apply the 1-ply tactical-safety guard to the visit-policy weights used for the
/// PLAYED move (TSS move-selection guard). Temperature/Dirichlet visit sampling
/// alone can select a move the search has effectively proven losing (forced
/// playouts keep a loser's visit count nonzero) or miss a proven win; this guard
/// ensures a proven win is played and a proven-losing move is never selected when
/// a non-losing move exists. It masks ONLY the selection weights — the exported
/// TRAINING policy is left untouched (do not collapse the policy target). Gated on
/// the root carrying a threat / own win, so the quiet common case is a no-op.
/// Deterministic in (root_state, action_ids, weights), so both selection sites
/// (advance + result payload) stay identical.
fn tactical_guard_weights(
    root_state: &RustHexoState,
    action_ids: &[PackedCoord],
    weights: &[f32],
) -> Vec<f32> {
    let analysis = threats::analyze(root_state);
    if !analysis.own_win_now && analysis.opp_threat_count == 0 {
        return weights.to_vec(); // no move can be a 1-ply win/loss here
    }
    let classes: Vec<i8> = action_ids
        .iter()
        .map(|&id| classify_root_move(root_state, id))
        .collect();
    let mut guarded = weights.to_vec();
    if classes.iter().any(|&c| c == 1) {
        // A proven win is available: restrict selection to winning moves.
        for (i, &c) in classes.iter().enumerate() {
            if c != 1 {
                guarded[i] = 0.0;
            }
        }
    } else if classes.iter().any(|&c| c != -1) {
        // No win, but a non-losing move exists: exclude proven-losing moves.
        for (i, &c) in classes.iter().enumerate() {
            if c == -1 {
                guarded[i] = 0.0;
            }
        }
    }
    // Safety net: never mask out the entire move set (e.g. a winning move with no
    // visits) — fall back to the raw weights so a legal move is still selectable.
    if guarded.iter().all(|&w| w <= 0.0) {
        return weights.to_vec();
    }
    guarded
}

fn select_search_action(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    temperature: f32,
    seed: u64,
) -> PyResult<Option<PackedCoord>> {
    let (action_ids, weights, _total) = visit_policy(search.root(), baseline);
    let guarded = tactical_guard_weights(&search.root_state, &action_ids, &weights);
    select_action_from_policy(&action_ids, &guarded, temperature, seed)
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

/// KataGo policy-target pruning. Starts from the raw per-edge DELTA visits (the
/// visits this search call added) and subtracts each non-best root child's forced
/// playouts back out, so the exported TRAINING target reflects what PUCT would
/// have chosen on its own — not the visits forced purely to keep root noise alive.
/// For every child except the most-visited one, remove up to
/// `n_forced = floor(sqrt(k * prior * root_visits))` visits, stopping early if a
/// further removal would raise that child's PUCT selection value above the
/// most-visited child's (meaning PUCT genuinely preferred it, so the visit is
/// real). Children pruned to zero drop out of the target. The PLAYED move is
/// chosen from the un-pruned policy elsewhere, so play stays diverse.
fn pruned_visit_policy(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    forced_playout_k: f32,
    c_puct: f32,
) -> (Vec<PackedCoord>, Vec<f32>) {
    let edges = &root.edges;
    let deltas: Vec<u32> = edges
        .iter()
        .map(|edge| edge_delta_visits(edge, baseline))
        .collect();
    let priors: Vec<f32> = edges.iter().map(|edge| edge.prior).collect();
    let cumulative: Vec<u32> = edges.iter().map(|edge| edge.visits).collect();
    let values: Vec<f32> = edges.iter().map(|edge| edge.value()).collect();
    let pruned = prune_forced_delta_counts(
        &deltas,
        &priors,
        &cumulative,
        &values,
        root.visits,
        forced_playout_k,
        c_puct,
    );
    let total: u32 = pruned.iter().sum();
    if total == 0 {
        // Defensive: pruning never removes the most-visited child, so this only
        // triggers when no visits were added this call. Fall back to raw policy.
        let (ids, weights, _total) = visit_policy(root, baseline);
        return (ids, weights);
    }
    let mut out_ids = Vec::with_capacity(edges.len());
    let mut weights = Vec::with_capacity(edges.len());
    for (index, edge) in edges.iter().enumerate() {
        if pruned[index] == 0 {
            continue;
        }
        out_ids.push(edge.action_id);
        weights.push(pruned[index] as f32 / total as f32);
    }
    (out_ids, weights)
}

/// Pure core of policy-target pruning (no tree types, so it is unit-testable).
/// Returns the pruned per-child DELTA visit counts. The most-visited child (ties
/// -> lower action id) is never pruned; for each other child remove up to
/// `floor(sqrt(k * prior * root_visits))` visits, stopping when a further removal
/// would lift that child's PUCT value above the most-visited child's.
fn prune_forced_delta_counts(
    deltas: &[u32],
    priors: &[f32],
    cumulative: &[u32],
    values: &[f32],
    root_visits: u32,
    forced_playout_k: f32,
    c_puct: f32,
) -> Vec<u32> {
    let mut pruned = deltas.to_vec();
    if forced_playout_k <= 0.0 {
        return pruned;
    }
    // Reference child = most visits added this call; ties resolve to the lower
    // index (we scan ascending and only replace on a strictly greater delta).
    let mut best_idx: Option<usize> = None;
    for index in 0..deltas.len() {
        if deltas[index] == 0 {
            continue;
        }
        best_idx = match best_idx {
            None => Some(index),
            Some(current) => {
                if deltas[index] > deltas[current] {
                    Some(index)
                } else {
                    Some(current)
                }
            }
        };
    }
    let Some(best_idx) = best_idx else {
        return pruned;
    };
    let root_n = root_visits.max(1) as f32;
    let explore = c_puct * root_n.sqrt();
    let u_best =
        values[best_idx] + priors[best_idx] * explore / (1.0 + cumulative[best_idx] as f32);
    for index in 0..deltas.len() {
        if index == best_idx || pruned[index] == 0 {
            continue;
        }
        if !(priors[index].is_finite() && priors[index] > 0.0) {
            continue;
        }
        let n_forced = (forced_playout_k * priors[index] * root_n).sqrt().floor() as u32;
        if n_forced == 0 {
            continue;
        }
        let q = values[index];
        let mut removed = 0u32;
        while removed < n_forced && pruned[index] > 0 {
            // Reduce the child's CUMULATIVE visit count for the PUCT test (Q held
            // fixed). `pruned[index] > 0` keeps the reduced count >= baseline.
            let reduced = cumulative[index].saturating_sub(removed + 1);
            let u = q + priors[index] * explore / (1.0 + reduced as f32);
            if u > u_best {
                break; // PUCT would genuinely have made this visit; keep it.
            }
            removed += 1;
            pruned[index] -= 1;
        }
    }
    pruned
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
        return Err(PyValueError::new_err(
            "visit policy action and weight lengths differ",
        ));
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
    batch: Option<&Bound<'py, PyDict>>,
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
    root.set_item("active_edge_bytes", search.active_edge_bytes)?;
    root.set_item("hidden_prior_bytes", search.hidden_prior_bytes)?;
    root.set_item("shared_prior_nodes", search.shared_prior_nodes)?;
    root.set_item("shared_prior_refs", search.shared_prior_refs)?;
    diagnostics.set_item("root", root)?;
    if let Some(batch) = batch {
        diagnostics.set_item("batch", batch)?;
    }
    Ok(diagnostics)
}

fn build_continuous_batch_diagnostics<'py>(
    py: Python<'py>,
    tree_stats: &ContinuousTreeStats,
    evaluation: &EvaluationStats,
    target_visits: u32,
    leaf_batch_per_root: u32,
    cache_len: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let tree = PyDict::new(py);
    tree.set_item("root_count", tree_stats.root_count)?;
    tree.set_item("target_visits", target_visits)?;
    tree.set_item("leaf_batch_per_root", leaf_batch_per_root)?;
    tree.set_item("completed_visits", tree_stats.completed_visits)?;
    tree.set_item("node_count", tree_stats.node_count)?;
    tree.set_item("active_edge_count", tree_stats.active_edge_count)?;
    tree.set_item("hidden_prior_count", tree_stats.hidden_prior_count)?;
    tree.set_item("root_active_edges", tree_stats.root_active_edges)?;
    tree.set_item("root_hidden_priors", tree_stats.root_hidden_priors)?;
    tree.set_item("max_nodes_per_root", tree_stats.max_nodes_per_root)?;
    tree.set_item("max_active_edges_per_root", tree_stats.max_active_edges_per_root)?;
    tree.set_item(
        "max_hidden_priors_per_root",
        tree_stats.max_hidden_priors_per_root,
    )?;
    tree.set_item(
        "max_active_edges_per_node",
        tree_stats.max_active_edges_per_node,
    )?;
    tree.set_item(
        "max_hidden_priors_per_node",
        tree_stats.max_hidden_priors_per_node,
    )?;
    tree.set_item("active_edge_bytes", tree_stats.active_edge_bytes)?;
    tree.set_item("hidden_prior_bytes", tree_stats.hidden_prior_bytes)?;
    tree.set_item("shared_prior_nodes", tree_stats.shared_prior_nodes)?;
    tree.set_item("shared_prior_refs", tree_stats.shared_prior_refs)?;

    let diagnostics = PyDict::new(py);
    diagnostics.set_item("tree", tree)?;
    diagnostics.set_item("evaluation", build_evaluation_diagnostics(py, evaluation, cache_len)?)?;
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
    tree.set_item(
        "max_active_edges_per_node",
        aggregate.max_active_edges_per_node,
    )?;
    tree.set_item(
        "max_hidden_priors_per_node",
        aggregate.max_hidden_priors_per_node,
    )?;
    tree.set_item("active_edge_bytes", aggregate.active_edge_bytes)?;
    tree.set_item("hidden_prior_bytes", aggregate.hidden_prior_bytes)?;
    tree.set_item("shared_prior_nodes", aggregate.shared_prior_nodes)?;
    tree.set_item("shared_prior_refs", aggregate.shared_prior_refs)?;

    let eval = build_evaluation_diagnostics(py, evaluation, cache_len)?;

    let diagnostics = PyDict::new(py);
    diagnostics.set_item("tree", tree)?;
    diagnostics.set_item("evaluation", eval)?;
    Ok(diagnostics)
}

fn build_evaluation_diagnostics<'py>(
    py: Python<'py>,
    evaluation: &EvaluationStats,
    cache_len: usize,
) -> PyResult<Bound<'py, PyDict>> {
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
    eval.set_item("payload_seconds", evaluation.payload_seconds)?;
    eval.set_item("dedup_seconds", evaluation.dedup_seconds)?;
    eval.set_item("cache_insert_seconds", evaluation.cache_insert_seconds)?;
    Ok(eval)
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model1MctsSession>()?;
    Ok(())
}

#[cfg(test)]
mod forced_playout_tests {
    use super::{
        continuous_completion_ready, continuous_flush_decision, mix_seed,
        prune_forced_delta_counts, ContinuousFlushDecision,
    };

    const K: f32 = 2.0;
    const CPUCT: f32 = 1.5;

    #[test]
    fn k_zero_is_identity() {
        let deltas = [80u32, 10, 3];
        let priors = [0.6f32, 0.05, 0.02];
        let cumulative = [80u32, 10, 3];
        let values = [0.5f32, 0.1, -0.2];
        let pruned =
            prune_forced_delta_counts(&deltas, &priors, &cumulative, &values, 100, 0.0, CPUCT);
        assert_eq!(pruned, deltas.to_vec(), "k=0 must not change any count");
    }

    #[test]
    fn removes_forced_tail_keeps_best() {
        // best = idx 0 (most delta). u_best = 0.5 + 0.6*1.5*sqrt(100)/(1+80) = 0.6111.
        // child1 prior .05: n_forced=floor(sqrt(2*.05*100))=3, all 3 removable (U stays < u_best) -> 10-3=7.
        // child2 prior .02: n_forced=floor(sqrt(2*.02*100))=2, both removable -> 3-2=1.
        let deltas = [80u32, 10, 3];
        let priors = [0.6f32, 0.05, 0.02];
        let cumulative = [80u32, 10, 3];
        let values = [0.5f32, 0.1, -0.2];
        let pruned =
            prune_forced_delta_counts(&deltas, &priors, &cumulative, &values, 100, K, CPUCT);
        assert_eq!(pruned, vec![80u32, 7, 1]);
        assert_eq!(pruned[0], 80, "most-visited child is never pruned");
        let raw_tail: u32 = deltas[1..].iter().sum();
        let pruned_tail: u32 = pruned[1..].iter().sum();
        assert!(pruned_tail < raw_tail, "forced tail visits must shrink");
    }

    #[test]
    fn keeps_puct_preferred_high_value_child() {
        // child1 has high Q (0.7) and few visits: PUCT genuinely prefers it, so even
        // though it is not the most-visited, its visits must NOT be pruned away.
        // u_best (idx0) = 0.4 + 0.5*1.5*10/(1+90)=0.4824. child1 reduced-by-1 U
        // = 0.7 + 0.3*15/(1+19)=0.925 > u_best -> stop immediately, remove 0.
        let deltas = [90u32, 20];
        let priors = [0.5f32, 0.3];
        let cumulative = [90u32, 20];
        let values = [0.4f32, 0.7];
        let pruned =
            prune_forced_delta_counts(&deltas, &priors, &cumulative, &values, 100, K, CPUCT);
        assert_eq!(
            pruned,
            vec![90u32, 20],
            "a PUCT-preferred child keeps its visits"
        );
    }

    #[test]
    fn never_prunes_below_baseline_or_to_negative() {
        // delta < cumulative (reused root): only the delta portion is prunable.
        let deltas = [50u32, 4]; // child1 added 4 this call (cumulative 30)
        let priors = [0.7f32, 0.05];
        let cumulative = [200u32, 30];
        let values = [0.5f32, 0.0];
        let pruned =
            prune_forced_delta_counts(&deltas, &priors, &cumulative, &values, 230, K, CPUCT);
        assert!(pruned[1] <= 4, "cannot remove more than the delta visits");
        assert_eq!(pruned[0], 50, "best untouched");
    }

    #[test]
    fn continuous_seed_mixer_has_golden_values() {
        assert_eq!(mix_seed(0, 0, 0, 0), 0x7de5_3de7_72ea_694c);
        assert_eq!(mix_seed(1, 2, 3, 4), 0xa6a9_b091_cff9_d67a);
        assert_eq!(
            mix_seed(123_456_789, 987_654_321, 42, 1),
            0x9fc4_975d_734f_10b3
        );
        assert_ne!(
            mix_seed(123_456_789, 987_654_321, 42, 0),
            mix_seed(123_456_789, 987_654_321, 42, 1),
            "Dirichlet and move-sampling streams must diverge",
        );
    }

    #[test]
    fn continuous_flush_decision_table() {
        use ContinuousFlushDecision::{Flush, Hold, Stop};

        let cases = [
            (0, 4, true, Hold),
            (0, 4, false, Stop),
            (3, 4, true, Hold),
            (3, 4, false, Flush { no_progress: true }),
            (4, 4, true, Flush { no_progress: false }),
            (4, 4, false, Flush { no_progress: true }),
            (7, 4, true, Flush { no_progress: false }),
        ];
        for (queue_len, flush_target, made_progress, expected) in cases {
            assert_eq!(
                continuous_flush_decision(queue_len, flush_target, made_progress),
                expected
            );
        }
    }

    #[test]
    fn continuous_completion_requires_zero_in_flight() {
        assert!(!continuous_completion_ready(64, 64, 1));
        assert!(continuous_completion_ready(64, 64, 0));
        assert!(continuous_completion_ready(65, 64, 0));
        assert!(!continuous_completion_ready(63, 64, 0));
    }
}
