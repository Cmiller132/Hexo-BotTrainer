//! hexfield search drivers — lockstep + continuous scheduler, ported from the
//! as-built dense_cnn mcts.rs (the semantic reference for the M5/M6
//! differential harness) with:
//!
//! - the §5.1 exploration-knob QUARANTINE: `root_fpu_zero_under_noise`
//!   defaults FALSE and the root-policy-temperature schedule defaults OFF
//!   (1.0 / no ramp). The differential harness passes dense's as-built values
//!   explicitly to both sides; production simply never enables them.
//! - the §5.4 divergences (LCB greedy selection, early-stop by move class,
//!   visit-scaled c_puct, moves-left utility), default ON in production,
//!   forced off by `search_parity_mode`.
//!
//! Seed discipline: the exact dense `mix_seed` hash and stream ids 0-5 are a
//! written contract (golden vectors in tests).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use hexo_engine::{
    apply_placement, unpack_coord, HexoState as RustHexoState, PackedCoord, Placement,
};

use crate::cache::{
    new_shared_evaluation_cache, new_shared_evaluation_stats, state_hash, EvaluationStats,
    RustEvaluation, RustEvaluationRequest, SharedEvaluationCache, SharedEvaluationStats,
    EVAL_CACHE_MAX_STATES,
};
use crate::payload::{evaluate_state_refs_cached, finish_eval_cached, submit_eval_cached};
use crate::state::states_from_py_states;
use crate::threats_shared as threats;
use crate::tree::{
    random_unit, terminal_value, Divergences, RootDirichletNoise, RustEdge, RustLeaf, RustNode,
    RustSearch, Widening,
};

pub const ACTIVE_ROOT_LIMIT: usize = 512;

pub const SEED_STREAM_ROOT_NOISE: u64 = 0;
pub const SEED_STREAM_MOVE_SELECT: u64 = 1;
pub const SEED_STREAM_PCR: u64 = 2;
pub const SEED_STREAM_POLICY_INIT_SELECT: u64 = 3;
pub const SEED_STREAM_POLICY_INIT_COUNT: u64 = 4;
pub const SEED_STREAM_POLICY_INIT_SAMPLE: u64 = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MoveClass {
    Full,
    Fast,
    Init,
}

#[derive(Clone, Copy)]
struct ContinuousMovePolicy {
    full_visits: u32,
    fast_visits: u32,
    pcr_full_proportion: f32,
    policy_init_fraction: f32,
    policy_init_avg_plies: f32,
    policy_init_max_plies: u32,
    policy_init_temperature: f32,
    root_policy_temperature: f32,
    root_policy_temperature_early: f32,
    root_policy_temperature_halflife: f32,
    fpu_reduction: f32,
    forced_playout_k: f32,
    noise: Option<RootNoiseConfig>,
    tss_enabled: bool,
    root_fpu_zero_under_noise: bool,
    /// [6] first-class root FPU reduction (KataGo rootFpuReductionMax). When
    /// Some it takes precedence over the legacy noise-conditioned mechanism;
    /// self-play sets 0.0. When None, the legacy `root_fpu_zero_under_noise`
    /// path applies (parity).
    root_fpu_reduction: Option<f32>,
    divergences: Divergences,
}

impl ContinuousMovePolicy {
    fn policy_init_plies(&self, base_seed: u64, game_key: u64) -> u32 {
        if self.policy_init_fraction <= 0.0
            || self.policy_init_avg_plies <= 0.0
            || self.policy_init_max_plies == 0
        {
            return 0;
        }
        if self.policy_init_fraction < 1.0 {
            let select =
                random_unit(mix_seed(base_seed, game_key, 0, SEED_STREAM_POLICY_INIT_SELECT));
            if select >= self.policy_init_fraction as f64 {
                return 0;
            }
        }
        let unit = random_unit(mix_seed(base_seed, game_key, 1, SEED_STREAM_POLICY_INIT_COUNT));
        let count =
            (-(self.policy_init_avg_plies as f64) * (1.0 - unit).max(1.0e-12).ln()).floor();
        (count.max(0.0) as u32).min(self.policy_init_max_plies)
    }

    fn classify(
        &self,
        base_seed: u64,
        game_key: u64,
        ply: u32,
        policy_init_remaining: u32,
    ) -> MoveClass {
        if policy_init_remaining > 0 {
            return MoveClass::Init;
        }
        if self.pcr_full_proportion >= 1.0 {
            return MoveClass::Full;
        }
        let unit = random_unit(mix_seed(base_seed, game_key, ply, SEED_STREAM_PCR));
        if unit < self.pcr_full_proportion as f64 {
            MoveClass::Full
        } else {
            MoveClass::Fast
        }
    }

    fn visits_for(&self, class: MoveClass) -> u32 {
        match class {
            MoveClass::Full => self.full_visits,
            MoveClass::Fast => self.fast_visits,
            MoveClass::Init => 1,
        }
    }

    fn forced_k_for(&self, class: MoveClass) -> f32 {
        match class {
            MoveClass::Full => self.forced_playout_k,
            _ => 0.0,
        }
    }

    fn noise_for(&self, class: MoveClass) -> Option<RootNoiseConfig> {
        match class {
            MoveClass::Full => self.noise,
            _ => None,
        }
    }

    fn root_fpu_for(&self, class: MoveClass) -> f32 {
        // [6] first-class root FPU reduction takes precedence (KataGo
        // rootFpuReductionMax; self-play 0.0). Applies to every move class — the
        // root descent always uses the root-specific reduction, not a
        // noise-conditioned special case.
        if let Some(value) = self.root_fpu_reduction {
            return value;
        }
        // Legacy (parity): zero FPU only at noised Full roots.
        if matches!(class, MoveClass::Full)
            && self.noise.is_some()
            && self.root_fpu_zero_under_noise
        {
            0.0
        } else {
            self.fpu_reduction
        }
    }

    fn root_temp_for(&self, class: MoveClass, ply: u32) -> f32 {
        if !matches!(class, MoveClass::Full) {
            return 1.0;
        }
        if self.root_policy_temperature_early <= 0.0
            || self.root_policy_temperature_halflife <= 0.0
        {
            return self.root_policy_temperature;
        }
        self.root_policy_temperature
            + (self.root_policy_temperature_early - self.root_policy_temperature)
                * 0.5f32.powf(ply as f32 / self.root_policy_temperature_halflife)
    }

    fn request_moves_left(&self) -> bool {
        self.divergences.moves_left_utility
    }
}

enum ContinuousPhase {
    Active,
    AwaitRootEval,
    Empty,
}

struct ContinuousSlot {
    game_key: u64,
    ply: u32,
    search: Option<RustSearch>,
    phase: ContinuousPhase,
    in_flight: u32,
    baseline: HashMap<PackedCoord, u32>,
    policy_init_remaining: u32,
    move_class: MoveClass,
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
    flush_size_histogram: HashMap<usize, u64>,
    on_move_seconds: f64,
    moves_decided: u64,
    full_moves: u64,
    fast_moves: u64,
    init_moves: u64,
    early_stops_fast: u64,
    early_stops_full: u64,
    early_stop_visits_saved: u64,
    lcb_overrides: u64,
}

pub fn continuous_flush_decision_pub(
    queue_len: usize,
    flush_target: usize,
    made_progress: bool,
) -> u8 {
    match continuous_flush_decision(queue_len, flush_target, made_progress) {
        ContinuousFlushDecision::Hold => 0,
        ContinuousFlushDecision::Flush { .. } => 1,
        ContinuousFlushDecision::Stop => 2,
    }
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

/// §5.4.2 early-stop. Greedy unrecorded searches (Fast / eval-arena): stop
/// when the remaining budget cannot overtake the visit leader AND, when LCB
/// selection is active, the LCB winner currently equals the visit winner
/// (conservative-heuristic w.r.t. LCB — the M6 gate for the LCB arm is
/// statistical). Recorded Full roots: a conservative 75% visit floor first.
fn early_stop_ready(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    recorded_full: bool,
    in_flight: u32,
) -> bool {
    let dv = search.divergences;
    if !dv.early_stop || in_flight > 0 {
        return false;
    }
    let remaining = search.remaining_visits();
    if remaining == 0 {
        return false;
    }
    if recorded_full {
        let floor = (search.target_visits as f32 * dv.full_visit_floor).ceil() as u32;
        if search.completed_visits < floor {
            return false;
        }
    }
    let root = search.root();
    // Build the per-edge stats vec ONCE (delta + LCB inputs) and derive
    // best/second/best_id from it, instead of scanning root.edges here and
    // then re-scanning it inside lcb_pick. The derivation below preserves the
    // original `delta > best` (strictly-greater) tie-break — the first edge at
    // the max delta stays best_id — so this is bit-identical to the prior code.
    let stats = lcb_stats(root, baseline);
    let mut best = 0u32;
    let mut second = 0u32;
    let mut best_id: Option<PackedCoord> = None;
    for &(action_id, delta, _visits, _value_sum, _value_sq_sum) in &stats {
        if delta > best {
            second = best;
            best = delta;
            best_id = Some(action_id as PackedCoord);
        } else if delta > second {
            second = delta;
        }
    }
    let Some(best_id) = best_id else {
        return false;
    };
    if best.saturating_sub(second) <= remaining {
        return false;
    }
    if dv.lcb_move_selection && !recorded_full {
        if let Some(lcb_id) =
            debug_lcb_from_stats(&stats, dv.lcb_z, dv.lcb_min_visits, dv.lcb_visit_fraction)
                .map(|id| id as PackedCoord)
        {
            if lcb_id != best_id {
                return false;
            }
        }
    }
    true
}

/// Per-edge LCB inputs over root edges: (action_id, delta_visits, visits,
/// value_sum, value_sq_sum), in edge order. Shared by lcb_pick and
/// early_stop_ready so the edge scan happens once per decision.
fn lcb_stats(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
) -> Vec<(u64, u32, u32, f32, f32)> {
    root.edges
        .iter()
        .map(|edge| {
            (
                edge.action_id as u64,
                edge_delta_visits(edge, baseline),
                edge.visits,
                edge.value_sum,
                edge.value_sq_sum,
            )
        })
        .collect()
}

/// LCB pick among eligible root edges: Q - z * sigma / sqrt(n), eligibility
/// delta >= max(lcb_min_visits, lcb_visit_fraction * max_child_delta). None
/// when no child qualifies (caller falls back to max-visits). Delegates to
/// the same core the M6 closed-form table tests exercise.
fn lcb_pick(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    dv: &Divergences,
) -> Option<PackedCoord> {
    let stats = lcb_stats(root, baseline);
    debug_lcb_from_stats(&stats, dv.lcb_z, dv.lcb_min_visits, dv.lcb_visit_fraction)
        .map(|id| id as PackedCoord)
}

/// §5.4.4 final-move decisiveness tie-break. Among root moves whose LCB is
/// within `ml_final_pick_band` of the LCB leader AND are guard-positive, prefer
/// the most decisive one: fewest moves-left when the root is clearly winning
/// (root value > ml_q_gate), most moves-left when clearly losing (< -ml_q_gate).
/// Returns None in the |value| <= gate dead-zone or when no candidate carries a
/// moves-left mean (head inert) — the caller then keeps the plain LCB pick. By
/// construction it only ever re-picks among value-equivalent moves, so a
/// miscalibrated head costs at most `ml_final_pick_band` of value.
fn ml_final_pick(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    dv: &Divergences,
    action_ids: &[PackedCoord],
    guarded_weights: &[f32],
) -> Option<PackedCoord> {
    let root_v = root.value();
    let dir: i32 = if root_v > dv.ml_q_gate {
        1
    } else if root_v < -dv.ml_q_gate {
        -1
    } else {
        return None;
    };
    let stats = lcb_stats(root, baseline);
    let max_delta = stats.iter().map(|s| s.1).max().unwrap_or(0);
    if max_delta == 0 {
        return None;
    }
    let threshold = (dv.lcb_min_visits as f32).max(dv.lcb_visit_fraction * max_delta as f32);
    let mut best_lcb = f32::NEG_INFINITY;
    let mut eligible: Vec<(PackedCoord, f32)> = Vec::new();
    for &(action_id, delta, visits, value_sum, value_sq_sum) in &stats {
        if (delta as f32) < threshold || visits == 0 {
            continue;
        }
        let n = visits as f32;
        let q = value_sum / n;
        let variance = (value_sq_sum / n - q * q).max(0.0);
        let lcb = q - dv.lcb_z * variance.sqrt() / n.sqrt();
        eligible.push((action_id as PackedCoord, lcb));
        if lcb > best_lcb {
            best_lcb = lcb;
        }
    }
    let mut pick: Option<(PackedCoord, f32)> = None;
    for &(id, lcb) in &eligible {
        if lcb < best_lcb - dv.ml_final_pick_band {
            continue;
        }
        let guard_positive = action_ids
            .iter()
            .zip(guarded_weights.iter())
            .any(|(&aid, &w)| aid == id && w > 0.0);
        if !guard_positive {
            continue;
        }
        let Some(m) = root
            .edges
            .iter()
            .find(|e| e.action_id == id)
            .and_then(|e| e.ml_mean())
        else {
            continue;
        };
        let better = match pick {
            None => true,
            Some((_, bm)) => {
                if dir == 1 {
                    m < bm
                } else {
                    m > bm
                }
            }
        };
        if better {
            pick = Some((id, m));
        }
    }
    pick.map(|(id, _)| id)
}

#[pyclass(unsendable)]
pub struct HexfieldMctsSession {
    searches: HashMap<u64, RustSearch>,
    evaluation_cache: SharedEvaluationCache,
    cache_max_states: usize,
}

#[pymethods]
impl HexfieldMctsSession {
    #[new]
    #[pyo3(signature = (max_states=None))]
    fn new(max_states: Option<usize>) -> PyResult<Self> {
        let cache_max_states =
            validate_positive_usize("max_states", max_states.unwrap_or(EVAL_CACHE_MAX_STATES))?;
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

    /// Debug-only: dump the stored tree for a game key (parity forensics).
    fn debug_dump(&self, py: Python<'_>, game_key: u64) -> PyResult<Py<PyAny>> {
        let Some(search) = self.searches.get(&game_key) else {
            return Ok(py.None());
        };
        let nodes = PyList::empty(py);
        for node in &search.nodes {
            let edges = PyList::empty(py);
            for edge in &node.edges {
                edges.append((
                    edge.action_id,
                    edge.visits,
                    edge.value_sum,
                    edge.prior,
                    edge.child,
                    edge.forced,
                ))?;
            }
            let entry = PyDict::new(py);
            entry.set_item("visits", node.visits)?;
            entry.set_item("value_sum", node.value_sum)?;
            entry.set_item("eval_value", node.eval_value)?;
            entry.set_item("remaining", node.remaining_prior_count())?;
            entry.set_item("max_children", node.max_eligible_children)?;
            entry.set_item("next_candidate", node.remaining_priors().first().copied())?;
            entry.set_item("edges", edges)?;
            nodes.append(entry)?;
        }
        let out = PyDict::new(py);
        out.set_item("completed", search.completed_visits)?;
        out.set_item("target", search.target_visits)?;
        out.set_item("nodes", nodes)?;
        Ok(out.into_any().unbind())
    }

    fn len(&self) -> usize {
        self.searches.len()
    }

    /// Lockstep batched search (eval ladder / arena / differential harness).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (game_keys, states, visits, c_puct, temperature, seed, evaluator, virtual_batch_size=None, active_root_limit=None, root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None, root_policy_temperature=None, fpu_reduction=None, virtual_loss=None, widening_policy_mass=None, widening_max_children=None, widening_min_children=None, forced_playout_k=None, move_temperatures=None, root_policy_temperatures=None, tss_enabled=None, root_fpu_zero_under_noise=None, root_fpu_reduction=None, search_parity_mode=None, divergence_overrides=None, debug_no_advance=None))]
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
        move_temperatures: Option<Vec<f32>>,
        root_policy_temperatures: Option<Vec<f32>>,
        tss_enabled: Option<bool>,
        // QUARANTINED (spec §5.1): hexfield production default is FALSE (no
        // FPU zeroing at noised roots). The differential harness passes true
        // to reproduce dense's as-built behavior.
        root_fpu_zero_under_noise: Option<bool>,
        // [6] SPEC CORRECTION: modern KataGo has NO "zero FPU under noise"
        // branch; it uses a separate rootFpuReductionMax that self-play sets to
        // 0.0. When provided this is the first-class root FPU reduction and
        // takes precedence over the legacy noise-conditioned mechanism.
        root_fpu_reduction: Option<f32>,
        search_parity_mode: Option<bool>,
        divergence_overrides: Option<&Bound<'_, PyDict>>,
        debug_no_advance: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        validate_search_inputs(visits, c_puct, temperature)?;
        let divergences = resolve_divergences(search_parity_mode, divergence_overrides)?;
        let roots = states_from_py_states(py, states)?;
        if roots.is_empty() {
            return Ok(PyTuple::empty(py).into_any().unbind());
        }
        if roots.len() != game_keys.len() {
            return Err(PyValueError::new_err(format!(
                "hexfield MCTS session received {} game keys for {} states",
                game_keys.len(),
                roots.len()
            )));
        }
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
            active_root_limit.unwrap_or(ACTIVE_ROOT_LIMIT),
        )?;
        if roots.len() > root_limit {
            return Err(PyValueError::new_err(format!(
                "hexfield MCTS session received {} active roots, above strict limit {}",
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
        // QUARANTINE: root policy temperature defaults to 1.0 (schedule off).
        let root_policy_temperature = validate_positive_f32(
            "root_policy_temperature",
            root_policy_temperature.unwrap_or(1.0),
        )?;
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
        let tss_enabled = tss_enabled.unwrap_or(true);
        // [6] root FPU reduction. If `root_fpu_reduction` is given explicitly it
        // is the first-class KataGo rootFpuReductionMax (self-play sets 0.0) and
        // takes precedence. Otherwise fall back to the legacy noise-conditioned
        // mechanism (parity): zero FPU only at noised roots when the quarantined
        // `root_fpu_zero_under_noise` knob is set.
        let root_fpu_reduction = match root_fpu_reduction {
            Some(value) => validate_nonnegative_f32("root_fpu_reduction", value)?,
            None => {
                if root_noise_config.is_some() && root_fpu_zero_under_noise.unwrap_or(false) {
                    0.0
                } else {
                    fpu_reduction
                }
            }
        };
        let widening = build_widening(
            widening_policy_mass,
            widening_min_children,
            widening_max_children,
        )?;
        let request_ml = divergences.moves_left_utility;

        let mut searches: Vec<Option<RustSearch>> = Vec::with_capacity(roots.len());
        let mut missing_indices = Vec::new();
        let mut missing_roots = Vec::new();
        for (index, (game_key, root)) in game_keys.iter().zip(roots.iter()).enumerate() {
            let root_hash = state_hash(root);
            if let Some(mut search) = self.searches.remove(game_key) {
                if search.root_hash == root_hash {
                    search.set_additional_visits(target_visits);
                    search.set_forced_playout_k(forced_playout_k);
                    search.set_root_fpu_reduction(root_fpu_reduction);
                    search.set_tss_enabled(tss_enabled);
                    search.set_divergences(divergences);
                    search.apply_root_policy_temperature(root_policy_temps[index]);
                    if let Some(noise) =
                        root_noise(root_noise_config, seed, index, divergences.dirichlet_shaped)
                    {
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
            let requests: Vec<RustEvaluationRequest> = missing_roots
                .iter()
                .map(|state| RustEvaluationRequest {
                    state,
                    state_hash: state_hash(state),
                })
                .collect();
            let root_evals = evaluate_state_refs_cached(
                py,
                evaluator,
                &requests,
                &self.evaluation_cache,
                Some(&evaluation_stats),
                self.cache_max_states,
                request_ml,
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
                    root_noise(root_noise_config, seed, index, divergences.dirichlet_shaped),
                    widening,
                    forced_playout_k,
                    tss_enabled,
                    divergences,
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
            request_ml,
            &move_temps,
            &baselines,
        )?;
        let cache_len = self
            .evaluation_cache
            .lock()
            .expect("evaluation cache mutex poisoned")
            .len();
        let evaluation_stats_snapshot = evaluation_stats
            .lock()
            .expect("evaluation stats mutex poisoned")
            .clone();
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
            Some(&evaluation_stats_snapshot),
            Some(cache_len),
            &move_temps,
            seed,
            Some(&baselines),
            c_puct,
            forced_playout_k,
        )?;

        let no_advance = debug_no_advance.unwrap_or(false);
        for ((game_key, mut search), selected) in game_keys
            .into_iter()
            .zip(searches.into_iter())
            .zip(selected_actions.into_iter())
        {
            if no_advance {
                // Forensics only: store the searched tree as-is for debug_dump.
                self.searches.insert(game_key, search);
                continue;
            }
            if let Some(action_id) = selected {
                if search.advance_root(action_id)? {
                    self.searches.insert(game_key, search);
                }
            }
        }

        Ok(results)
    }

    /// Continuous per-slot scheduler (the production self-play driver).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (game_keys, states, evaluator, on_move, visits, c_puct, base_seed, virtual_batch_size, flush_target, active_root_limit, temperature_by_ply, root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None, root_policy_temperature=None, fpu_reduction=None, virtual_loss=None, widening_policy_mass=None, widening_max_children=None, widening_min_children=None, forced_playout_k=None, root_policy_temperature_early=None, root_policy_temperature_halflife=None, pcr_full_proportion=None, pcr_fast_visits=None, policy_init_fraction=None, policy_init_avg_plies=None, policy_init_max_plies=None, policy_init_temperature=None, tss_enabled=None, root_fpu_zero_under_noise=None, root_fpu_reduction=None, search_parity_mode=None, divergence_overrides=None))]
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
        root_policy_temperature_early: Option<f32>,
        root_policy_temperature_halflife: Option<f32>,
        pcr_full_proportion: Option<f32>,
        pcr_fast_visits: Option<u32>,
        policy_init_fraction: Option<f32>,
        policy_init_avg_plies: Option<f32>,
        policy_init_max_plies: Option<u32>,
        policy_init_temperature: Option<f32>,
        tss_enabled: Option<bool>,
        root_fpu_zero_under_noise: Option<bool>,
        // [6] first-class KataGo rootFpuReductionMax (self-play 0.0); precedence
        // over the legacy noise-conditioned knob when provided.
        root_fpu_reduction: Option<f32>,
        search_parity_mode: Option<bool>,
        divergence_overrides: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        validate_search_inputs(visits, c_puct, 0.0)?;
        let divergences = resolve_divergences(search_parity_mode, divergence_overrides)?;
        let roots = states_from_py_states(py, states)?;
        if roots.len() != game_keys.len() {
            return Err(PyValueError::new_err(format!(
                "hexfield continuous MCTS received {} game keys for {} states",
                game_keys.len(),
                roots.len()
            )));
        }
        let root_limit = validate_positive_usize("active_root_limit", active_root_limit)?;
        if roots.len() > root_limit {
            return Err(PyValueError::new_err(format!(
                "hexfield continuous MCTS received {} active roots, above strict limit {}",
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
        let root_policy_temperature_early = validate_nonnegative_f32(
            "root_policy_temperature_early",
            root_policy_temperature_early.unwrap_or(0.0),
        )?;
        let root_policy_temperature_halflife = validate_nonnegative_f32(
            "root_policy_temperature_halflife",
            root_policy_temperature_halflife.unwrap_or(0.0),
        )?;
        if root_policy_temperature_early > 0.0 && root_policy_temperature_halflife <= 0.0 {
            return Err(PyValueError::new_err(
                "root_policy_temperature_halflife must be > 0 when root_policy_temperature_early is set",
            ));
        }
        let pcr_full_proportion = pcr_full_proportion.unwrap_or(1.0);
        if !pcr_full_proportion.is_finite()
            || pcr_full_proportion <= 0.0
            || pcr_full_proportion > 1.0
        {
            return Err(PyValueError::new_err(
                "pcr_full_proportion must be in (0, 1]",
            ));
        }
        let pcr_fast_visits = pcr_fast_visits.unwrap_or(visits);
        if pcr_full_proportion < 1.0 && pcr_fast_visits == 0 {
            return Err(PyValueError::new_err(
                "pcr_fast_visits must be >= 1 when PCR is enabled",
            ));
        }
        let policy_init_fraction = policy_init_fraction.unwrap_or(0.0);
        if !policy_init_fraction.is_finite() || !(0.0..=1.0).contains(&policy_init_fraction) {
            return Err(PyValueError::new_err(
                "policy_init_fraction must be in [0, 1]",
            ));
        }
        let policy_init_avg_plies = policy_init_avg_plies.unwrap_or(0.0);
        let policy_init_max_plies = policy_init_max_plies.unwrap_or(0);
        let policy_init_temperature = policy_init_temperature.unwrap_or(1.0);
        if policy_init_fraction > 0.0 {
            if !policy_init_avg_plies.is_finite() || policy_init_avg_plies <= 0.0 {
                return Err(PyValueError::new_err(
                    "policy_init_avg_plies must be > 0 when policy-init openings are enabled",
                ));
            }
            if policy_init_max_plies == 0 {
                return Err(PyValueError::new_err(
                    "policy_init_max_plies must be >= 1 when policy-init openings are enabled",
                ));
            }
            if !policy_init_temperature.is_finite() || policy_init_temperature <= 0.0 {
                return Err(PyValueError::new_err("policy_init_temperature must be > 0"));
            }
        }
        let move_policy = ContinuousMovePolicy {
            full_visits: visits,
            fast_visits: pcr_fast_visits,
            pcr_full_proportion,
            policy_init_fraction,
            policy_init_avg_plies,
            policy_init_max_plies,
            policy_init_temperature,
            root_policy_temperature,
            root_policy_temperature_early,
            root_policy_temperature_halflife,
            fpu_reduction,
            forced_playout_k,
            noise: root_noise_config,
            tss_enabled: tss_enabled.unwrap_or(true),
            // QUARANTINE default false (dense as-built default is true).
            root_fpu_zero_under_noise: root_fpu_zero_under_noise.unwrap_or(false),
            // [6] first-class root FPU reduction (validated >= 0 when provided).
            root_fpu_reduction: match root_fpu_reduction {
                Some(value) => Some(validate_nonnegative_f32("root_fpu_reduction", value)?),
                None => None,
            },
            divergences,
        };
        let widening = build_widening(
            widening_policy_mass,
            widening_min_children,
            widening_max_children,
        )?;
        if temperature_by_ply.is_empty() {
            return Err(PyValueError::new_err("temperature_by_ply must not be empty"));
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
            let policy_init_remaining = move_policy.policy_init_plies(base_seed, game_key);
            let move_class = move_policy.classify(base_seed, game_key, 0, policy_init_remaining);
            let mut slot = ContinuousSlot {
                game_key,
                ply: 0,
                search: None,
                phase: ContinuousPhase::AwaitRootEval,
                in_flight: 0,
                baseline: HashMap::new(),
                policy_init_remaining,
                move_class,
            };
            if let Some(mut search) = self.searches.remove(&game_key) {
                if search.root_hash == root_hash {
                    search.set_additional_visits(move_policy.visits_for(move_class));
                    search.set_forced_playout_k(move_policy.forced_k_for(move_class));
                    search.set_root_fpu_reduction(move_policy.root_fpu_for(move_class));
                    search.set_tss_enabled(move_policy.tss_enabled);
                    search.set_divergences(divergences);
                    search.apply_root_policy_temperature(move_policy.root_temp_for(move_class, 0));
                    if let Some(noise) = root_noise_exact(
                        move_policy.noise_for(move_class),
                        mix_seed(base_seed, game_key, 0, SEED_STREAM_ROOT_NOISE),
                        divergences.dirichlet_shaped,
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
        // dense's select↔eval overlap, serial form: the NEXT select pass runs
        // with the flush's virtual losses still pending (pre-backup tree
        // state); a no-progress prefetch is stale advice and is discarded so
        // the next iteration re-selects after the backup freed the paths.
        let mut prefetched: Option<(Vec<RustLeaf>, bool)> = None;
        // HEXFIELD_ASYNC_EVAL: real GPU/host overlap. The forward is ENQUEUED
        // (submit, no device sync), the pre-backup select runs with the GIL
        // released while those kernels execute, then the forward is drained
        // (finish). Off => the original synchronous eval-then-select. Results
        // are identical either way (only the sync point moves); the flag exists
        // so the path can be parity-gated before it owns the live run.
        // HEXFIELD_NO_PREFETCH is a parity-debugging lever only.
        let async_eval = std::env::var("HEXFIELD_ASYNC_EVAL").is_ok();
        let no_prefetch = std::env::var("HEXFIELD_NO_PREFETCH").is_ok();
        while continuous_has_work(&slots) || !queue.is_empty() {
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
                // Eval the flush and run the pre-backup select (on pre-backup
                // tree state). Async: submit -> select (overlaps GPU) -> finish.
                // Sync: eval -> select. Both yield (prefetch_result, evaluations).
                let (prefetch_result, evaluations) = if async_eval {
                    let pending = submit_eval_cached(
                        py,
                        evaluator,
                        &requests,
                        &self.evaluation_cache,
                        Some(&evaluation_stats),
                        move_policy.request_moves_left(),
                    )?;
                    let prefetch_result = if no_prefetch {
                        (Vec::new(), false)
                    } else {
                        py.detach(|| {
                            select_continuous_pass(
                                &mut slots,
                                c_puct,
                                leaf_batch_per_root,
                                virtual_loss,
                            )
                        })?
                    };
                    let evaluations = finish_eval_cached(
                        py,
                        evaluator,
                        pending,
                        &self.evaluation_cache,
                        Some(&evaluation_stats),
                        self.cache_max_states,
                    )?;
                    (prefetch_result, evaluations)
                } else {
                    let evaluations = evaluate_state_refs_cached(
                        py,
                        evaluator,
                        &requests,
                        &self.evaluation_cache,
                        Some(&evaluation_stats),
                        self.cache_max_states,
                        move_policy.request_moves_left(),
                    )?;
                    let prefetch_result = if no_prefetch {
                        (Vec::new(), false)
                    } else {
                        select_continuous_pass(
                            &mut slots,
                            c_puct,
                            leaf_batch_per_root,
                            virtual_loss,
                        )?
                    };
                    (prefetch_result, evaluations)
                };
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
                    &move_policy,
                    widening,
                    base_seed,
                    virtual_loss,
                    divergences,
                )?;
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
                c_puct,
                &move_policy,
                &temperature_by_ply,
                base_seed,
                &mut queue,
                &mut stats,
            )?;

            if matches!(decision, ContinuousFlushDecision::Stop) && moves_decided == 0 {
                let stuck = slots
                    .iter()
                    .filter(|slot| !matches!(slot.phase, ContinuousPhase::Empty))
                    .count();
                return Err(PyRuntimeError::new_err(format!(
                    "hexfield continuous MCTS scheduler stalled with {stuck} unfinished slots \
                     (queue empty, no selectable leaves, no completable roots)"
                )));
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
        dict.set_item("full_moves", stats.full_moves)?;
        dict.set_item("fast_moves", stats.fast_moves)?;
        dict.set_item("init_moves", stats.init_moves)?;
        dict.set_item("early_stops_fast", stats.early_stops_fast)?;
        dict.set_item("early_stops_full", stats.early_stops_full)?;
        dict.set_item("early_stop_visits_saved", stats.early_stop_visits_saved)?;
        dict.set_item("lcb_overrides", stats.lcb_overrides)?;
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
        dict.set_item("evaluation", eval_stats_dict(py, &eval_snapshot)?)?;
        let cache_len = self
            .evaluation_cache
            .lock()
            .expect("evaluation cache mutex poisoned")
            .len();
        dict.set_item("cache_len", cache_len)?;
        Ok(dict.into_any().unbind())
    }
}

// === Lockstep internals ===

#[allow(clippy::too_many_arguments)]
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
    request_moves_left: bool,
    move_temps: &[f32],
    baselines: &[HashMap<PackedCoord, u32>],
) -> PyResult<()> {
    // dense's lockstep is a two-stage pipeline: the NEXT batch is selected
    // BEFORE the current batch is backed up (the select worker runs during
    // the eval). That ordering is SEMANTIC — it extends the virtual-loss
    // window by one batch — so the serial form here replicates it exactly:
    // select(N+1) runs after evaluate(N) and before backup(N). (Running the
    // select on a worker thread to reclaim the wall-clock is an M8 perf item
    // with identical semantics.)
    // §5.4.2 early-stop. in_flight is passed as 0 here BY DESIGN (audit
    // 2026-06-13 reviewed): the visit-overtake test inside early_stop_ready is
    // already in-flight-safe — apply_virtual_visit increments BOTH
    // completed_visits and the selected edge's visit count at selection time,
    // so best/second (per-edge delta visits) include pending leaves while
    // remaining = target - completed excludes them. Thus best-second > remaining
    // correctly proves the visit leader is unbeatable by ALL un-selected visits
    // regardless of how many are currently pending; the pending batch is still
    // evaluated+backed-up by the loop below before exit (no leaked virtual
    // loss). The continuous path's in_flight==0 guard is about slot-advance
    // safety (node-id invalidation), a different concern. test_early_stop_
    // without_lcb_is_exact pins chosen-move equality AND non-vacuity.
    let early_stop_pass = |searches: &mut [RustSearch]| {
        for (index, search) in searches.iter_mut().enumerate() {
            if search.needs_visits()
                && move_temps.get(index).copied().unwrap_or(1.0) == 0.0
                && early_stop_ready(search, baselines.get(index), false, 0)
            {
                search.early_stopped = true;
                search.target_visits = search.completed_visits;
            }
        }
    };

    early_stop_pass(searches);
    let (mut pending_leaves, _primed_progress) =
        select_leaf_batch(searches, c_puct, leaf_batch_per_root, virtual_loss)?;

    loop {
        // §5.4.2: check between every batch (a parity-mode no-op); see the
        // in-flight-safety note on early_stop_pass above.
        early_stop_pass(searches);
        if pending_leaves.is_empty() {
            if !searches.iter().any(RustSearch::needs_visits) {
                break;
            }
            let (leaves, made_progress) =
                select_leaf_batch(searches, c_puct, leaf_batch_per_root, virtual_loss)?;
            if leaves.is_empty() {
                if !made_progress {
                    break;
                }
                continue;
            }
            pending_leaves = leaves;
        }

        let leaf_requests: Vec<_> = pending_leaves
            .iter()
            .map(|leaf| RustEvaluationRequest {
                state: &leaf.state,
                state_hash: leaf.state_hash,
            })
            .collect();
        let evaluations = evaluate_state_refs_cached(
            py,
            evaluator,
            &leaf_requests,
            evaluation_cache,
            Some(evaluation_stats),
            cache_max_states,
            request_moves_left,
        )?;
        // Prefetch select with the current batch still pending (pre-backup
        // tree state) — dense's overlap semantics, serial form.
        let next_leaves = if searches.iter().any(RustSearch::needs_visits) {
            select_leaf_batch(searches, c_puct, leaf_batch_per_root, virtual_loss)?.0
        } else {
            Vec::new()
        };
        apply_eval_backups(searches, pending_leaves, &evaluations, virtual_loss)?;
        pending_leaves = next_leaves;
    }
    Ok(())
}

fn select_leaf_batch(
    searches: &mut [RustSearch],
    c_puct: f32,
    leaf_batch_per_root: u32,
    virtual_loss: f32,
) -> PyResult<(Vec<RustLeaf>, bool)> {
    let mut leaves = Vec::new();
    let mut made_progress = false;
    for (root_index, search) in searches.iter_mut().enumerate() {
        if !search.needs_visits() {
            continue;
        }
        let budget = leaf_batch_per_root.min(search.remaining_visits());
        for _ in 0..budget {
            let selected = search.select_pending_leaf(c_puct)?;
            let Some(selected) = selected else {
                break;
            };
            search.apply_virtual_visit(&selected.path, virtual_loss);
            made_progress = true;

            let ml_on = search.divergences.moves_left_utility;
            if let Some(outcome) = selected.terminal {
                let leaf_player = selected.state.current_player();
                let leaf_value = terminal_value(outcome, leaf_player);
                let leaf_ml = ml_on.then_some(0.0);
                search.backup_virtual(&selected.path, leaf_player, leaf_value, virtual_loss, leaf_ml);
            } else if let Some(node_id) = selected.existing_node {
                let node = &search.nodes[node_id];
                let player = node.player;
                let value = node.value();
                let leaf_ml = if ml_on { node.ml_mean() } else { None };
                search.backup_virtual(&selected.path, player, value, virtual_loss, leaf_ml);
            } else if let Some(verdict) = search
                .tss_enabled
                .then(|| threats::analyze(&selected.state).verdict())
                .flatten()
            {
                let leaf_player = selected.state.current_player();
                search.backup_virtual(&selected.path, leaf_player, verdict, virtual_loss, None);
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
    }
    Ok((leaves, made_progress))
}

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
        let leaf_ml = if search.divergences.moves_left_utility {
            search.nodes[child_id].ml_mean()
        } else {
            None
        };
        search.backup_virtual(&leaf.path, child_player, child_value, virtual_loss, leaf_ml);
    }
    Ok(())
}

// === Continuous internals ===

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
        let ml_on = search.divergences.moves_left_utility;
        if let Some(outcome) = selected.terminal {
            let leaf_player = selected.state.current_player();
            let leaf_value = terminal_value(outcome, leaf_player);
            let leaf_ml = ml_on.then_some(0.0);
            search.backup_virtual(&selected.path, leaf_player, leaf_value, virtual_loss, leaf_ml);
        } else if let Some(node_id) = selected.existing_node {
            let node = &search.nodes[node_id];
            let player = node.player;
            let value = node.value();
            let leaf_ml = if ml_on { node.ml_mean() } else { None };
            search.backup_virtual(&selected.path, player, value, virtual_loss, leaf_ml);
        } else if let Some(verdict) = search
            .tss_enabled
            .then(|| threats::analyze(&selected.state).verdict())
            .flatten()
        {
            let leaf_player = selected.state.current_player();
            search.backup_virtual(&selected.path, leaf_player, verdict, virtual_loss, None);
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

fn select_continuous_pass(
    slots: &mut [ContinuousSlot],
    c_puct: f32,
    leaf_batch_per_root: u32,
    virtual_loss: f32,
) -> PyResult<(Vec<RustLeaf>, bool)> {
    // Per-slot selection is independent (each closure owns one slot's tree via
    // &mut; the RNG is seeded by slot_index, not execution order), so fan it
    // across cores with rayon. Results fold in slot order, so the leaf sweep is
    // byte-identical to the serial form (dense_cnn mcts.rs:1458 port — the layer
    // the hexfield port had dropped).
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

#[allow(clippy::too_many_arguments)]
fn backup_continuous_items(
    slots: &mut [ContinuousSlot],
    items: Vec<ContinuousEvalItem>,
    evaluations: &[Arc<RustEvaluation>],
    move_policy: &ContinuousMovePolicy,
    widening: Widening,
    base_seed: u64,
    virtual_loss: f32,
    divergences: Divergences,
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
                let child_id =
                    search.add_node_from_eval(&leaf.state, leaf.state_hash, Arc::clone(evaluation))?;
                search.nodes[leaf.parent_node].edges[leaf.edge_index].child = Some(child_id);
                search.mark_pending(leaf.parent_node, leaf.edge_index, -1);
                slot.in_flight = slot.in_flight.saturating_sub(1);
                let child_player = search.nodes[child_id].player;
                let child_value = search.nodes[child_id].value();
                let leaf_ml = if search.divergences.moves_left_utility {
                    search.nodes[child_id].ml_mean()
                } else {
                    None
                };
                search.backup_virtual(&leaf.path, child_player, child_value, virtual_loss, leaf_ml);
            }
            ContinuousEvalItem::RootInit {
                slot_index, state, ..
            } => {
                let slot = &mut slots[slot_index];
                let move_class = move_policy.classify(
                    base_seed,
                    slot.game_key,
                    slot.ply,
                    slot.policy_init_remaining,
                );
                slot.move_class = move_class;
                let search = RustSearch::new(
                    state,
                    &**evaluation,
                    move_policy.visits_for(move_class),
                    move_policy.fpu_reduction,
                    move_policy.root_fpu_for(move_class),
                    move_policy.root_temp_for(move_class, slot.ply),
                    root_noise_exact(
                        move_policy.noise_for(move_class),
                        mix_seed(base_seed, slot.game_key, slot.ply, SEED_STREAM_ROOT_NOISE),
                        divergences.dirichlet_shaped,
                    ),
                    widening,
                    move_policy.forced_k_for(move_class),
                    move_policy.tss_enabled,
                    divergences,
                )?;
                if search.root_edges_empty() {
                    return Err(PyValueError::new_err(
                        "hexfield continuous MCTS root has no legal actions",
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

#[allow(clippy::too_many_arguments)]
fn complete_continuous_slots(
    py: Python<'_>,
    on_move: &Bound<'_, PyAny>,
    slots: &mut [ContinuousSlot],
    c_puct: f32,
    move_policy: &ContinuousMovePolicy,
    temperature_by_ply: &[f32],
    base_seed: u64,
    queue: &mut Vec<ContinuousEvalItem>,
    stats: &mut ContinuousSchedulerStats,
) -> PyResult<u64> {
    let mut moves_decided = 0u64;
    for slot_index in 0..slots.len() {
        if !matches!(slots[slot_index].phase, ContinuousPhase::Active) {
            continue;
        }
        let move_class = slots[slot_index].move_class;
        let in_flight = slots[slot_index].in_flight;
        let (complete, early) = slots[slot_index]
            .search
            .as_ref()
            .map(|search| {
                let normal = continuous_completion_ready(
                    search.completed_visits,
                    search.target_visits,
                    in_flight,
                );
                if normal {
                    return (true, false);
                }
                // §5.4.2: Fast moves stop unrestricted; recorded Full roots
                // keep the conservative visit floor.
                let early = early_stop_ready(
                    search,
                    Some(&slots[slot_index].baseline),
                    matches!(move_class, MoveClass::Full),
                    in_flight,
                );
                (early, early)
            })
            .unwrap_or((false, false));
        if !complete {
            continue;
        }
        if early {
            let search = slots[slot_index].search.as_mut().expect("active slot");
            stats.early_stop_visits_saved += search.remaining_visits() as u64;
            match move_class {
                MoveClass::Full => stats.early_stops_full += 1,
                _ => stats.early_stops_fast += 1,
            }
            search.early_stopped = true;
            search.target_visits = search.completed_visits;
        }

        let game_key = slots[slot_index].game_key;
        let ply = slots[slot_index].ply;
        let move_seed = mix_seed(base_seed, game_key, ply, SEED_STREAM_MOVE_SELECT);
        let temperature = match move_class {
            MoveClass::Full => temperature_for_ply(temperature_by_ply, ply),
            MoveClass::Fast => 0.0,
            MoveClass::Init => 0.0,
        };
        let baseline = slots[slot_index].baseline.clone();
        let payloads = {
            let search = slots[slot_index]
                .search
                .as_ref()
                .expect("active continuous slot has search");
            let baselines = vec![baseline];
            build_search_result_payloads(
                py,
                std::slice::from_ref(search),
                None,
                None,
                &[temperature],
                move_seed,
                Some(&baselines),
                c_puct,
                move_policy.forced_k_for(move_class),
            )?
        };
        let payloads = payloads.bind(py).downcast::<PyList>()?;
        let payload = payloads.get_item(0)?;
        let payload_dict = payload.downcast::<PyDict>()?;
        payload_dict.set_item("pcr_full", matches!(move_class, MoveClass::Full))?;
        payload_dict.set_item("policy_init", matches!(move_class, MoveClass::Init))?;
        if let Some(true) = payload_dict
            .get_item("lcb_override")?
            .map(|v| v.extract::<bool>().unwrap_or(false))
        {
            stats.lcb_overrides += 1;
        }
        if matches!(move_class, MoveClass::Init) {
            let search = slots[slot_index]
                .search
                .as_ref()
                .expect("active continuous slot has search");
            let (prior_ids, prior_weights) = root_prior_policy(search.root());
            let sampled = select_action_from_policy(
                &prior_ids,
                &prior_weights,
                move_policy.policy_init_temperature,
                mix_seed(base_seed, game_key, ply, SEED_STREAM_POLICY_INIT_SAMPLE),
            )?
            .ok_or_else(|| {
                PyValueError::new_err("policy-init sampling found no positive prior mass")
            })?;
            payload_dict.set_item("action_id", sampled)?;
            payload_dict.set_item("action_selection", "policy_init_prior")?;
        }
        let action_id: PackedCoord = payload_dict
            .get_item("action_id")?
            .ok_or_else(|| PyValueError::new_err("continuous payload missing action_id"))?
            .extract()?;

        moves_decided += 1;
        stats.moves_decided += 1;
        match move_class {
            MoveClass::Full => stats.full_moves += 1,
            MoveClass::Fast => stats.fast_moves += 1,
            MoveClass::Init => stats.init_moves += 1,
        }
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
                if matches!(move_class, MoveClass::Init) {
                    slots[slot_index].policy_init_remaining =
                        slots[slot_index].policy_init_remaining.saturating_sub(1);
                }
                let next_ply = ply.saturating_add(1);
                let next_class = move_policy.classify(
                    base_seed,
                    game_key,
                    next_ply,
                    slots[slot_index].policy_init_remaining,
                );
                slots[slot_index].move_class = next_class;
                let mut keep_promoted = false;
                if let Some(search) = slots[slot_index].search.as_mut() {
                    if search.advance_root(action_id)? && search.root_hash == next_hash {
                        search.set_additional_visits(move_policy.visits_for(next_class));
                        search.set_forced_playout_k(move_policy.forced_k_for(next_class));
                        search.set_root_fpu_reduction(move_policy.root_fpu_for(next_class));
                        search.set_tss_enabled(move_policy.tss_enabled);
                        search
                            .apply_root_policy_temperature(move_policy.root_temp_for(next_class, next_ply));
                        if let Some(noise) = root_noise_exact(
                            move_policy.noise_for(next_class),
                            mix_seed(base_seed, game_key, next_ply, SEED_STREAM_ROOT_NOISE),
                            move_policy.divergences.dirichlet_shaped,
                        ) {
                            search.apply_root_dirichlet_noise(noise);
                        }
                        slots[slot_index].baseline =
                            search.root_edge_visits().into_iter().collect();
                        keep_promoted = true;
                    }
                }
                slots[slot_index].ply = next_ply;
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
                slots[slot_index].policy_init_remaining =
                    move_policy.policy_init_plies(base_seed, new_key);
                slots[slot_index].move_class = move_policy.classify(
                    base_seed,
                    new_key,
                    0,
                    slots[slot_index].policy_init_remaining,
                );
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

// === Shared helpers ===

fn single_state_from_py(py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<RustHexoState> {
    let tuple = PyTuple::new(py, [state])?;
    let states = states_from_py_states(py, tuple.as_any())?;
    states
        .into_iter()
        .next()
        .ok_or_else(|| PyValueError::new_err("expected one state"))
}

fn resolve_divergences(
    search_parity_mode: Option<bool>,
    overrides: Option<&Bound<'_, PyDict>>,
) -> PyResult<Divergences> {
    let mut dv = if search_parity_mode.unwrap_or(false) {
        Divergences::parity()
    } else {
        Divergences::production()
    };
    if let Some(overrides) = overrides {
        // Test-only per-divergence toggles (M6 property gates / M10 lesions).
        if let Some(v) = overrides.get_item("lcb_move_selection")? {
            dv.lcb_move_selection = v.extract()?;
        }
        if let Some(v) = overrides.get_item("early_stop")? {
            dv.early_stop = v.extract()?;
        }
        if let Some(v) = overrides.get_item("visit_scaled_c_puct")? {
            dv.visit_scaled_c_puct = v.extract()?;
        }
        if let Some(v) = overrides.get_item("moves_left_utility")? {
            dv.moves_left_utility = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_weight")? {
            dv.ml_weight = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_scale")? {
            dv.ml_scale = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_q_gate")? {
            dv.ml_q_gate = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_two_sided")? {
            dv.ml_two_sided = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_final_pick")? {
            dv.ml_final_pick = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_final_pick_band")? {
            dv.ml_final_pick_band = v.extract()?;
        }
        if let Some(v) = overrides.get_item("lcb_z")? {
            dv.lcb_z = v.extract()?;
        }
        if let Some(v) = overrides.get_item("c_scale")? {
            dv.c_scale = v.extract()?;
        }
        if let Some(v) = overrides.get_item("c_base")? {
            dv.c_base = v.extract()?;
        }
        // main_4 KataGo-faithful search divergences (ledger items [1]-[7]) —
        // individually flippable for the M6 property gates / M10 lesions.
        if let Some(v) = overrides.get_item("nucleus_f64")? {
            dv.nucleus_f64 = v.extract()?;
        }
        if let Some(v) = overrides.get_item("new_child_fpu")? {
            dv.new_child_fpu = v.extract()?;
        }
        if let Some(v) = overrides.get_item("lazy_widening")? {
            dv.lazy_widening = v.extract()?;
        }
        if let Some(v) = overrides.get_item("clean_root_prior_cache")? {
            dv.clean_root_prior_cache = v.extract()?;
        }
        if let Some(v) = overrides.get_item("dirichlet_shaped")? {
            dv.dirichlet_shaped = v.extract()?;
        }
        if let Some(v) = overrides.get_item("pruned_dynamic_cpuct")? {
            dv.pruned_dynamic_cpuct = v.extract()?;
        }
    }
    Ok(dv)
}

fn build_widening(
    mass: Option<f32>,
    min_children: Option<u32>,
    max_children: Option<u32>,
) -> PyResult<Widening> {
    let widening_mass = mass.unwrap_or(0.95);
    if !widening_mass.is_finite() || widening_mass <= 0.0 || widening_mass > 1.0 {
        return Err(PyValueError::new_err("widening_policy_mass must be in (0, 1]"));
    }
    let widening = Widening {
        mass: widening_mass,
        min_children: validate_positive_u32("widening_min_children", min_children.unwrap_or(2))?
            as usize,
        max_children: validate_positive_u32("widening_max_children", max_children.unwrap_or(32))?
            as usize,
    };
    if widening.min_children > widening.max_children {
        return Err(PyValueError::new_err(
            "widening_min_children must be <= widening_max_children",
        ));
    }
    Ok(widening)
}

#[allow(clippy::too_many_arguments)]
fn build_search_result_payloads(
    py: Python<'_>,
    searches: &[RustSearch],
    eval_stats: Option<&EvaluationStats>,
    cache_len: Option<usize>,
    temperatures: &[f32],
    seed: u64,
    baselines: Option<&[HashMap<PackedCoord, u32>]>,
    c_puct: f32,
    forced_playout_k: f32,
) -> PyResult<Py<PyAny>> {
    let results = PyList::empty(py);
    for (index, search) in searches.iter().enumerate() {
        let result = PyDict::new(py);
        let root = search.root();
        let baseline = baselines.and_then(|items| items.get(index));
        let (policy_action_ids, policy_weights, _policy_q, policy_total) =
            visit_policy(root, baseline);
        let (export_action_ids, export_weights, export_q) = if forced_playout_k > 0.0 {
            // [7] align the recorded-target pruning with selection's c_for(N)
            // when pruned_dynamic_cpuct is on; otherwise static c_puct (parity).
            let effective_c = search.effective_pruning_c_puct(c_puct, root.visits);
            pruned_visit_policy(root, baseline, forced_playout_k, effective_c)
        } else {
            let (ids, w, q, _t) = visit_policy(root, baseline);
            (ids, w, q)
        };
        let (root_prior_action_ids, root_prior_weights) = root_prior_policy(root);
        let guarded_weights = if search.tss_enabled {
            tactical_guard_weights(&search.root_state, &policy_action_ids, &policy_weights)
        } else {
            policy_weights.clone()
        };
        let (selected, lcb_override) = select_action_with_lcb(
            search,
            baseline,
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
        result.set_item("lcb_override", lcb_override)?;
        result.set_item("early_stopped", search.early_stopped)?;
        let to_bytes = |data: &[u32]| -> Bound<'_, PyBytes> {
            let len = std::mem::size_of_val(data);
            let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) };
            PyBytes::new(py, raw)
        };
        let to_bytes_f32 = |data: &[f32]| -> Bound<'_, PyBytes> {
            let len = std::mem::size_of_val(data);
            let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) };
            PyBytes::new(py, raw)
        };
        result.set_item("visit_policy_action_ids_bytes", to_bytes(&export_action_ids))?;
        result.set_item("visit_policy_weights_bytes", to_bytes_f32(&export_weights))?;
        result.set_item("visit_policy_q_bytes", to_bytes_f32(&export_q))?;
        result.set_item("visit_policy_count", export_action_ids.len())?;
        result.set_item(
            "root_prior_policy_action_ids_bytes",
            to_bytes(&root_prior_action_ids),
        )?;
        result.set_item(
            "root_prior_policy_weights_bytes",
            to_bytes_f32(&root_prior_weights),
        )?;
        result.set_item("root_prior_policy_count", root_prior_action_ids.len())?;
        result.set_item("root_value", root.value())?;
        result.set_item("visits", policy_total)?;
        let diag = PyDict::new(py);
        let tree = search.diagnostics();
        diag.set_item("node_count", tree.node_count)?;
        diag.set_item("active_edge_count", tree.active_edge_count)?;
        diag.set_item("root_active_edges", tree.root_active_edges)?;
        diag.set_item("root_hidden_priors", tree.root_hidden_priors)?;
        if let Some(stats) = eval_stats {
            diag.set_item("evaluation", eval_stats_dict(py, stats)?)?;
        }
        if let Some(cache_len) = cache_len {
            diag.set_item("cache_len", cache_len)?;
        }
        result.set_item("diagnostics", diag)?;
        results.append(result)?;
    }

    Ok(results.into_any().unbind())
}

fn eval_stats_dict<'py>(py: Python<'py>, stats: &EvaluationStats) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("requested_states", stats.requested_states)?;
    dict.set_item("cache_hits", stats.cache_hits)?;
    dict.set_item("duplicate_hits", stats.duplicate_hits)?;
    dict.set_item("unique_states", stats.unique_states)?;
    dict.set_item("evaluator_chunks", stats.evaluator_chunks)?;
    dict.set_item("encoded_states", stats.encoded_states)?;
    dict.set_item("encoded_nodes", stats.encoded_nodes)?;
    dict.set_item("encoding_seconds", stats.encoding_seconds)?;
    dict.set_item("evaluator_seconds", stats.evaluator_seconds)?;
    dict.set_item("parse_seconds", stats.parse_seconds)?;
    dict.set_item("cache_inserts", stats.cache_inserts)?;
    dict.set_item("cache_size_peak", stats.cache_size_peak)?;
    Ok(dict)
}

fn root_prior_policy(root: &RustNode) -> (Vec<PackedCoord>, Vec<f32>) {
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
    shaped: bool,
) -> Option<RootDirichletNoise> {
    let config = config?;
    Some(RootDirichletNoise {
        total_alpha: config.total_alpha,
        fraction: config.fraction,
        seed: seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        shaped,
    })
}

fn root_noise_exact(
    config: Option<RootNoiseConfig>,
    seed: u64,
    shaped: bool,
) -> Option<RootDirichletNoise> {
    let config = config?;
    Some(RootDirichletNoise {
        total_alpha: config.total_alpha,
        fraction: config.fraction,
        seed,
        shaped,
    })
}

/// Test-only surfaces for the M6 property gates (§5.4 LCB closed-form table
/// tests + moves-left utility sign/gate properties). Pure functions over the
/// same formulas the search uses.
pub fn debug_lcb_from_stats(
    stats: &[(u64, u32, u32, f32, f32)], // (action_id, delta, visits, value_sum, value_sq_sum)
    z: f32,
    min_visits: u32,
    visit_fraction: f32,
) -> Option<u64> {
    let max_delta = stats.iter().map(|s| s.1).max().unwrap_or(0);
    if max_delta == 0 {
        return None;
    }
    let threshold = (min_visits as f32).max(visit_fraction * max_delta as f32);
    let mut best: Option<(f32, u64)> = None;
    for &(action_id, delta, visits, value_sum, value_sq_sum) in stats {
        if (delta as f32) < threshold || visits == 0 {
            continue;
        }
        let n = visits as f32;
        let q = value_sum / n;
        let variance = (value_sq_sum / n - q * q).max(0.0);
        let lcb = q - z * variance.sqrt() / n.sqrt();
        let replace = match best {
            Some((current, current_id)) => lcb > current || (lcb == current && action_id < current_id),
            None => true,
        };
        if replace {
            best = Some((lcb, action_id));
        }
    }
    best.map(|(_, id)| id)
}

pub fn debug_ml_bonus(
    q: f32,
    m_edge: f32,
    m_node: f32,
    weight: f32,
    scale: f32,
    gate: f32,
    two_sided: bool,
) -> f32 {
    // s gates by the CHOOSER's perspective Q: +1 clearly winning (prefer fewer
    // moves left = faster win), -1 clearly losing when two-sided (prefer more
    // moves left = slower loss), 0 in the |Q| <= gate dead-zone (no sign
    // discontinuity at Q=0). Both signs add a POSITIVE bonus to the desired
    // child because tanh flips with (m_edge - m_node). Bounded by `weight`.
    let s = if q > gate {
        1.0
    } else if two_sided && q < -gate {
        -1.0
    } else {
        return 0.0;
    };
    -weight * s * ((m_edge - m_node) / scale).tanh()
}

pub fn mix_seed(base_seed: u64, game_key: u64, ply: u32, stream: u64) -> u64 {
    let mut value = base_seed ^ 0xA076_1D64_78BD_642F;
    value ^= game_key.wrapping_mul(0xE703_7ED1_A0B4_28DB);
    value ^= (ply as u64).wrapping_mul(0x8EBC_6AF0_9C88_C6E3);
    value ^= stream.wrapping_mul(0x5899_65CC_7537_4CC3);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn classify_root_move(root_state: &RustHexoState, action_id: PackedCoord) -> i8 {
    let me = root_state.current_player();
    let mut child = root_state.clone();
    let coord = unpack_coord(action_id);
    match apply_placement(&mut child, Placement { coord }) {
        Err(_) => 0,
        Ok(res) => {
            if let Some(outcome) = res.outcome {
                return if outcome.winner == me { 1 } else { -1 };
            }
            match threats::analyze(&child).verdict() {
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

fn tactical_guard_weights(
    root_state: &RustHexoState,
    action_ids: &[PackedCoord],
    weights: &[f32],
) -> Vec<f32> {
    let analysis = threats::analyze(root_state);
    if !analysis.own_win_now && analysis.opp_threat_count == 0 {
        return weights.to_vec();
    }
    let classes: Vec<i8> = action_ids
        .iter()
        .map(|&id| classify_root_move(root_state, id))
        .collect();
    let mut guarded = weights.to_vec();
    if classes.iter().any(|&c| c == 1) {
        for (i, &c) in classes.iter().enumerate() {
            if c != 1 {
                guarded[i] = 0.0;
            }
        }
    } else if classes.iter().any(|&c| c != -1) {
        for (i, &c) in classes.iter().enumerate() {
            if c == -1 {
                guarded[i] = 0.0;
            }
        }
    }
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
    let (action_ids, weights, _q, _total) = visit_policy(search.root(), baseline);
    let guarded = if search.tss_enabled {
        tactical_guard_weights(&search.root_state, &action_ids, &weights)
    } else {
        weights.clone()
    };
    let (selected, _override) =
        select_action_with_lcb(search, baseline, &action_ids, &guarded, temperature, seed)?;
    Ok(selected)
}

/// Action selection: temperature sampling on Full paths (dense semantics
/// verbatim) and, on greedy (T == 0) paths with the §5.4.1 divergence on,
/// LCB-of-Q selection among eligible children (fallback max-visits). The TSS
/// guard has already zeroed proven-losing weights; LCB only ever picks among
/// guard-positive actions.
fn select_action_with_lcb(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    action_ids: &[PackedCoord],
    guarded_weights: &[f32],
    temperature: f32,
    seed: u64,
) -> PyResult<(Option<PackedCoord>, bool)> {
    let dv = search.divergences;
    if temperature == 0.0 && dv.lcb_move_selection {
        let visit_pick = select_action_from_policy(action_ids, guarded_weights, 0.0, seed)?;
        let root = search.root();
        if let Some(lcb_id) = lcb_pick(root, baseline, &dv) {
            // Respect the tactical guard: never let LCB pick a zeroed action.
            let allowed = action_ids
                .iter()
                .zip(guarded_weights.iter())
                .any(|(&id, &w)| id == lcb_id && w > 0.0);
            if allowed {
                // §5.4.4 decisiveness tie-break on the PLAYED move: among moves
                // value-tied with the LCB leader, prefer the decisive one. Needs
                // the moves-left head (gated on moves_left_utility); inert + safe
                // (returns lcb_id) in the dead-zone or with no ml stats.
                let final_id = if dv.ml_final_pick && dv.moves_left_utility {
                    ml_final_pick(root, baseline, &dv, action_ids, guarded_weights)
                        .unwrap_or(lcb_id)
                } else {
                    lcb_id
                };
                let overrode = visit_pick.map(|v| v != final_id).unwrap_or(false);
                return Ok((Some(final_id), overrode));
            }
        }
        return Ok((visit_pick, false));
    }
    Ok((
        select_action_from_policy(action_ids, guarded_weights, temperature, seed)?,
        false,
    ))
}

fn visit_policy(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
) -> (Vec<PackedCoord>, Vec<f32>, Vec<f32>, u32) {
    // Compute each edge's delta visits ONCE (edge_delta_visits is a HashMap
    // lookup when baseline is Some) and reuse the cached deltas for both the
    // total and the per-edge weights. Value-identical to the prior two-pass
    // form: same sum, same per-edge weight numerators.
    let deltas: Vec<u32> = root
        .edges
        .iter()
        .map(|edge| edge_delta_visits(edge, baseline))
        .collect();
    let policy_total: u32 = deltas.iter().copied().sum();
    let mut policy_action_ids = Vec::with_capacity(root.edges.len());
    let mut policy_weights = Vec::with_capacity(root.edges.len());
    let mut policy_q = Vec::with_capacity(root.edges.len());
    for (edge, &visits) in root.edges.iter().zip(deltas.iter()) {
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
        policy_q.push(edge.value());
    }
    (policy_action_ids, policy_weights, policy_q, policy_total)
}

fn edge_delta_visits(edge: &RustEdge, baseline: Option<&HashMap<PackedCoord, u32>>) -> u32 {
    let before = baseline
        .and_then(|visits| visits.get(&edge.action_id).copied())
        .unwrap_or(0);
    edge.visits.saturating_sub(before)
}

fn pruned_visit_policy(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    forced_playout_k: f32,
    c_puct: f32,
) -> (Vec<PackedCoord>, Vec<f32>, Vec<f32>) {
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
        let (ids, weights, q, _total) = visit_policy(root, baseline);
        return (ids, weights, q);
    }
    let mut out_ids = Vec::with_capacity(edges.len());
    let mut weights = Vec::with_capacity(edges.len());
    let mut out_q = Vec::with_capacity(edges.len());
    for (index, edge) in edges.iter().enumerate() {
        if pruned[index] == 0 {
            continue;
        }
        out_ids.push(edge.action_id);
        weights.push(pruned[index] as f32 / total as f32);
        out_q.push(edge.value());
    }
    (out_ids, weights, out_q)
}

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
            let reduced = cumulative[index].saturating_sub(removed + 1);
            let u = q + priors[index] * explore / (1.0 + reduced as f32);
            if u > u_best {
                break;
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
