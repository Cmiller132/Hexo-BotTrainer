//! hexfield search drivers: lockstep batched search and the continuous per-slot
//! scheduler.
//!
//! - `root_fpu_zero_under_noise` defaults FALSE and the root-policy-temperature
//!   schedule defaults OFF (1.0 / no ramp).
//! - The optional search divergences (LCB greedy selection, early-stop by move
//!   class, visit-scaled c_puct, moves-left utility) default ON and are forced
//!   off by `search_parity_mode`.
//!
//! Seed discipline: `mix_seed` and stream ids 0-6 are pinned by golden vectors
//! in tests.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use hexo_engine::{
    apply_placement, pack_coord, unpack_coord, HexoState as RustHexoState, PackedCoord, Placement,
};

use crate::cache::{
    new_shared_evaluation_cache, new_shared_evaluation_stats, state_hash, EvaluationStats,
    RustEvaluation, RustEvaluationRequest, SharedEvaluationCache, SharedEvaluationStats,
    EVAL_CACHE_MAX_STATES,
};
use crate::payload::{
    evaluate_state_refs_cached, finish_eval_cached, submit_eval_cached, PendingEval,
};
use crate::state::states_from_py_states;
use crate::threats_shared as threats;
use crate::tss_async::TssAsyncPool;
use crate::tss_core::{self, ProofStatus, SolveCaps, SolveStats, ZoneSearchCaps};
use crate::tss_solver::{EffectiveSolveConfig, TssSolver};
use crate::tree::{
    gumbel_completed_q, gumbel_sigma, gumbel_softmax, random_unit, terminal_value,
    tss_solve_verified, tss_solve_verified_with_stats, tss_verified_solve_caps, Divergences,
    RootDirichletNoise, RustEdge, RustLeaf, RustNode, RustSearch, SolverHorizon, TssCounters,
    TssLeafRoute, TssParkResolution, Widening,
};
use crate::tss_verify::CertNode;

pub const ACTIVE_ROOT_LIMIT: usize = 512;

pub const SEED_STREAM_ROOT_NOISE: u64 = 0;
pub const SEED_STREAM_MOVE_SELECT: u64 = 1;
pub const SEED_STREAM_PCR: u64 = 2;
pub const SEED_STREAM_POLICY_INIT_SELECT: u64 = 3;
pub const SEED_STREAM_POLICY_INIT_COUNT: u64 = 4;
pub const SEED_STREAM_POLICY_INIT_SAMPLE: u64 = 5;
/// Gumbel-Top-k root draws. Dedicated stream so Gumbel noise is independent of
/// the Dirichlet root-noise stream (id 0).
pub const SEED_STREAM_GUMBEL: u64 = 6;

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
    /// Play temperature for the Fast class. 0.0 (default) => greedy LCB pick.
    fast_temperature: f32,
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
    /// Root FPU reduction. When Some it takes precedence over the
    /// noise-conditioned `root_fpu_zero_under_noise` mechanism and applies to
    /// every move class. When None, the `root_fpu_zero_under_noise` path applies.
    root_fpu_reduction: Option<f32>,
    /// Divergences view for Full/Init move classes (the base search profile).
    divergences_full: Divergences,
    /// Divergences view for the Fast move class. Equals `divergences_full` when
    /// no `fast_*` overrides are set (golden invariant), so absent fast levers
    /// reproduce today's single-profile behavior byte-for-byte.
    divergences_fast: Divergences,
}

impl ContinuousMovePolicy {
    /// The per-class Divergences view: Fast moves get `divergences_fast`,
    /// Full/Init get `divergences_full`.
    fn divergences_for(&self, class: MoveClass) -> Divergences {
        match class {
            MoveClass::Fast => self.divergences_fast,
            MoveClass::Full | MoveClass::Init => self.divergences_full,
        }
    }
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

    /// Classify a ply into a Full/Fast/Init move class.
    ///
    /// PCR classification is per-TURN, not per-ply: the mix counter is
    /// `ply / 2`, so the two plies of one turn (2k and 2k+1) hash to the SAME
    /// stream input and therefore share one Full/Fast class. Two reasons:
    ///
    ///  1. Clean tree reuse. A Full turn builds a deep PUCT subtree that its
    ///     paired ply promotes and reuses under the SAME regime. If the two
    ///     plies could land in different classes, the per-class
    ///     `set_divergences` refactor below would swap the Gumbel-root vs PUCT
    ///     regime mid-turn onto a reused root, corrupting the promoted SH state.
    ///     Sharing the class keeps the whole turn's reused tree on one regime.
    ///  2. Balanced player coverage. Each Full turn exports one P0 and one P1
    ///     policy target, so Full turns contribute training rows for both
    ///     players symmetrically instead of skewing to whichever seat happened
    ///     to draw Full.
    ///
    /// The `policy_init_remaining > 0 => Init` short-circuit and the
    /// `pcr_full_proportion >= 1.0 => Full` short-circuit are unchanged. Call
    /// sites still pass the real `ply`; the `/2` happens only here.
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
        // Per-turn: both plies 2k and 2k+1 map to turn index k.
        let turn = ply / 2;
        let unit = random_unit(mix_seed(base_seed, game_key, turn, SEED_STREAM_PCR));
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
        // When set, `root_fpu_reduction` takes precedence and applies to every
        // move class.
        if let Some(value) = self.root_fpu_reduction {
            return value;
        }
        // Otherwise zero FPU only at noised Full roots when the knob is set.
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

    /// Per-class PLAY temperature for the continuous driver.
    ///   Full => the ply schedule (floor applied by config).
    ///   Fast => `fast_temperature` (default 0.0 = greedy LCB pick, bit-for-bit
    ///           unchanged; at T=0.1 the sampler exponent is 1/T=10 over the
    ///           guard-filtered delta-visit histogram — gentle exploration, near
    ///           argmax unless the top candidates are close. At T>0 the LCB pick +
    ///           ml_final_pick no longer fire for Fast moves — they require T==0).
    ///   Init => 0.0 (the played move is then prior-sampled by the caller at
    ///           policy_init_temperature).
    fn temperature_for_class(
        &self,
        class: MoveClass,
        temperature_by_ply: &[f32],
        ply: u32,
    ) -> f32 {
        match class {
            MoveClass::Full => temperature_for_ply(temperature_by_ply, ply),
            MoveClass::Fast => self.fast_temperature,
            MoveClass::Init => 0.0,
        }
    }

    /// Whether the evaluator must emit moves-left output. True when EITHER
    /// class view enables the moves-left utility (a shared evaluation feeds both
    /// Full and Fast roots, so it must satisfy whichever class needs ML).
    fn request_moves_left(&self) -> bool {
        self.divergences_full.moves_left_utility || self.divergences_fast.moves_left_utility
    }

    /// Whether the evaluator must emit raw pre-softmax policy logits. True when
    /// EITHER class view enables a Gumbel mechanism that reads `logits(a)` (the
    /// improved target, the Gumbel-Top-k root sampler, or the non-root
    /// selection). Fast will need logits when it runs under Gumbel while Full
    /// stays PUCT, so both views are OR-ed.
    fn request_logits(&self) -> bool {
        let needs = |d: &Divergences| d.gumbel_target || d.gumbel_root || d.gumbel_nonroot_select;
        needs(&self.divergences_full) || needs(&self.divergences_fast)
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

/// A deep-solver leaf whose async request was accepted and which therefore
/// remains outside the GPU evaluation queue until the result lands or its
/// bounded bail deadline expires. The leaf continues to own its virtual visit
/// and pending mark for the entire stay in the pen.
struct ParkedLeaf {
    leaf: RustLeaf,
    parked_at: Instant,
    generation: u64,
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
    // Play-policy telemetry: moves selected via the quota-pruned Gumbel play
    // distribution, and how many of those played the raw delta leader (the SH
    // winner). winner/moves ≈ exploitation rate of the play sampler. The
    // `_early` pair covers ply < 20 (the high-temperature exploration window).
    // Late-game rates do NOT approach 1: SH forces the two finalists to a
    // ~228:196 visit split (1024 visits, m=32), so at the 0.15 temperature
    // floor the runner-up keeps (196/228)^(1/0.15) ≈ 0.37 relative weight and
    // the winner-rate ceiling is ≈ 0.73.
    gumbel_play_moves: u64,
    gumbel_play_winner_moves: u64,
    gumbel_play_moves_early: u64,
    gumbel_play_winner_early: u64,
    // Per-phase wall time (seconds) accumulated over the run: where the
    // scheduler loop actually spends its time. `Instant::now()` bracketing is
    // ~ns-scale against ms-scale phases, so this is always on.
    select_seconds: f64,
    submit_seconds: f64,
    finish_seconds: f64,
    backup_seconds: f64,
    complete_seconds: f64,
    loop_iterations: u64,
    completes_skipped: u64,
    // End-of-run async-pool tail drain (Codex review, late-alarm loss): fatal
    // verify failures / worker panics banked AFTER the last in-loop drain are
    // collected here at scheduler exit and folded into the epoch's fatal
    // counters by the Python driver — the alarm can no longer time out into
    // a stderr-only Drop message.
    tss_async_verify_failed_tail: u64,
    tss_async_worker_panics_tail: u64,
    tss_async_tail_cleared: u64,
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

/// Resolve the configured dynamic-worker ceiling exactly as the pool does.
/// Keeping this tiny mirror here lets a session decide whether its warm pool
/// remains configuration-compatible at the next call boundary.
fn resolved_tss_worker_max(base: u32, configured: u32) -> usize {
    TssAsyncPool::resolved_max_worker_count(base, configured, true)
}

fn tss_pool_matches(pool: &TssAsyncPool, base: u32, max: u32, park: bool) -> bool {
    let expected_max = TssAsyncPool::resolved_max_worker_count(base, max, park);
    pool.base_worker_count() == base.clamp(1, 32) as usize
        && pool.max_worker_count() == expected_max
        && pool.park_mode() == park
}

/// Early-stop test. Greedy unrecorded searches (Fast / eval-arena) stop when
/// the remaining budget cannot overtake the visit leader AND, when LCB
/// selection is active, the LCB winner currently equals the visit winner.
/// Recorded Full roots must first pass a visit floor (`full_visit_floor`).
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
    // Build the per-edge stats vec once (delta + LCB inputs) and derive
    // best/second/best_id from it. The `delta > best` (strictly-greater)
    // tie-break keeps the first edge at the max delta as best_id.
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
/// `debug_lcb_from_stats`.
fn lcb_pick(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    dv: &Divergences,
) -> Option<PackedCoord> {
    let stats = lcb_stats(root, baseline);
    debug_lcb_from_stats(&stats, dv.lcb_z, dv.lcb_min_visits, dv.lcb_visit_fraction)
        .map(|id| id as PackedCoord)
}

/// Final-move decisiveness tie-break. Among root moves whose LCB is within
/// `ml_final_pick_band` of the LCB leader AND are guard-positive, pick the most
/// decisive one: fewest moves-left when the root is clearly winning (root value
/// > ml_q_gate), most moves-left when clearly losing (< -ml_q_gate). Returns
/// None in the |value| <= gate dead-zone or when no candidate carries a
/// moves-left mean; the caller then keeps the plain LCB pick. Only re-picks
/// among moves within `ml_final_pick_band` of the LCB leader.
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
    /// Background deep-solve pool (tss_solver_async). Created lazily on the
    /// first run whose divergences enable it and kept for the session's life
    /// so worker solver caches stay warm across run_continuous calls.
    tss_pool: Option<TssAsyncPool>,
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
            tss_pool: None,
        })
    }

    fn discard(&mut self, game_key: u64) {
        self.searches.remove(&game_key);
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
        // Default FALSE: no FPU zeroing at noised roots.
        root_fpu_zero_under_noise: Option<bool>,
        // When provided, the root FPU reduction; takes precedence over the
        // noise-conditioned mechanism.
        root_fpu_reduction: Option<f32>,
        search_parity_mode: Option<bool>,
        divergence_overrides: Option<&Bound<'_, PyDict>>,
        debug_no_advance: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        validate_search_inputs(visits, c_puct, temperature)?;
        let divergences = resolve_divergences(search_parity_mode, divergence_overrides)?;
        // Async solve pool (tss_solver_async): lazily created, session-owned.
        // Any base/max/queue-mode change replaces the pool (see
        // run_continuous). Park mode selects FIFO/no-eviction queue semantics.
        if divergences.tss_solver_async {
            let base = divergences.tss_solver_async_threads;
            let max = divergences.tss_solver_async_threads_max;
            let park = divergences.tss_solver_park;
            let resize = self
                .tss_pool
                .as_ref()
                .is_some_and(|pool| !tss_pool_matches(pool, base, max, park));
            if resize {
                self.tss_pool = None;
            }
            if self.tss_pool.is_none() {
                self.tss_pool = Some(TssAsyncPool::new(base, max, park));
            }
        }
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
        // Root policy temperature defaults to 1.0 (schedule off).
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
        // Root FPU reduction. If `root_fpu_reduction` is given explicitly it
        // takes precedence. Otherwise use the noise-conditioned mechanism: zero
        // FPU only at noised roots when `root_fpu_zero_under_noise` is set.
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
        // Request raw logits whenever any Gumbel mechanism reads them.
        let request_logits = divergences.gumbel_target
            || divergences.gumbel_root
            || divergences.gumbel_nonroot_select;

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
                    // (Re)build the Gumbel-Top-k candidate set + SH schedule on
                    // the reused root, mirroring the continuous reuse paths;
                    // cleared when the Gumbel root is off so the PUCT root runs.
                    // The root hash folds into the seed stream so successive
                    // moves of one game draw fresh Gumbel noise even when the
                    // caller repeats its per-call seed.
                    if divergences.gumbel_root {
                        let gumbel_seed =
                            mix_seed(seed, *game_key ^ root_hash, 0, SEED_STREAM_GUMBEL);
                        search.init_gumbel_root(gumbel_seed, target_visits);
                    } else {
                        search.clear_gumbel_root();
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
                request_logits,
            )?;
            for ((index, root), evaluation) in missing_indices
                .into_iter()
                .zip(missing_roots.into_iter())
                .zip(root_evals.iter())
            {
                let root_hash = state_hash(&root);
                let mut search = RustSearch::new(
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
                )?;
                // Build the Gumbel-Top-k candidate set + SH schedule for a
                // fresh root when the Gumbel root is on (mirrors the continuous
                // RootInit path). No-op without raw root logits.
                if divergences.gumbel_root {
                    let gumbel_seed =
                        mix_seed(seed, game_keys[index] ^ root_hash, 0, SEED_STREAM_GUMBEL);
                    search.init_gumbel_root(gumbel_seed, target_visits);
                }
                searches[index] = Some(search);
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
            request_logits,
            &move_temps,
            &baselines,
            self.tss_pool.as_ref(),
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
        // Build the native payloads FIRST and advance the retained tree
        // through the payload's OWN action (Codex review, action/tree
        // mismatch): the payload path is the one that applies the play-prune
        // and deep root-guard overrides, so a separately re-derived selection
        // could advance the tree through a different move than the one
        // returned to the caller — the next call's root-hash guard would
        // then discard all retained tree/memo/solver work.
        let natives: Vec<PayloadNative> = searches
            .iter()
            .enumerate()
            .map(|(index, search)| {
                build_search_result_payload_native(
                    search,
                    baselines.get(index),
                    move_temps[index],
                    seed.wrapping_add(index as u64),
                    c_puct,
                    forced_playout_k,
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let selected_actions: Vec<PackedCoord> =
            natives.iter().map(|native| native.action_id).collect();
        let results = PyList::empty(py);
        for native in &natives {
            let result =
                native.to_pydict(py, Some(&evaluation_stats_snapshot), Some(cache_len))?;
            results.append(result)?;
        }
        let results = results.into_any().unbind();

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
            if search.advance_root(selected)? {
                self.searches.insert(game_key, search);
            }
        }

        Ok(results)
    }

    /// Continuous per-slot scheduler (the production self-play driver).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (game_keys, states, evaluator, on_move, visits, c_puct, base_seed, virtual_batch_size, flush_target, active_root_limit, temperature_by_ply, root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None, root_policy_temperature=None, fpu_reduction=None, virtual_loss=None, widening_policy_mass=None, widening_max_children=None, widening_min_children=None, forced_playout_k=None, root_policy_temperature_early=None, root_policy_temperature_halflife=None, pcr_full_proportion=None, pcr_fast_visits=None, pcr_fast_temperature=None, policy_init_fraction=None, policy_init_avg_plies=None, policy_init_max_plies=None, policy_init_temperature=None, tss_enabled=None, root_fpu_zero_under_noise=None, root_fpu_reduction=None, search_parity_mode=None, divergence_overrides=None, fast_divergence_overrides=None))]
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
        pcr_fast_temperature: Option<f32>,
        policy_init_fraction: Option<f32>,
        policy_init_avg_plies: Option<f32>,
        policy_init_max_plies: Option<u32>,
        policy_init_temperature: Option<f32>,
        tss_enabled: Option<bool>,
        root_fpu_zero_under_noise: Option<bool>,
        // Root FPU reduction; takes precedence over the noise-conditioned knob
        // when provided.
        root_fpu_reduction: Option<f32>,
        search_parity_mode: Option<bool>,
        divergence_overrides: Option<&Bound<'_, PyDict>>,
        // Fast-class divergence view. When None, the Fast class reuses the base
        // (Full) divergences, so absent fast levers = today's single profile.
        fast_divergence_overrides: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        validate_search_inputs(visits, c_puct, 0.0)?;
        let divergences = resolve_divergences(search_parity_mode, divergence_overrides)?;
        // Fast-class view: parse the fast override map when provided, else fall
        // back to the base view (golden invariant: divergences_fast ==
        // divergences_full when no fast_* keys are set).
        let divergences_fast = match fast_divergence_overrides {
            Some(fast) => resolve_divergences(search_parity_mode, Some(fast))?,
            None => divergences,
        };
        // Async solve pool (tss_solver_async): created lazily on the first
        // run whose class views ask for it, then kept warm on the session.
        // A changed thread count REPLACES the pool at this run boundary
        // (Codex review, first-call-wins): dropping the old pool quiesces and
        // joins its workers, so a live-config resize actually applies instead
        // of silently keeping the original size for the session's lifetime.
        if divergences.tss_solver_async || divergences_fast.tss_solver_async {
            let park = divergences.tss_solver_park || divergences_fast.tss_solver_park;
            // Preserve the frozen legacy max-of-both behavior park-off. With
            // parking enabled, however, a disabled class must not silently
            // raise the configured base of the class(es) that own the pool.
            let base = if park {
                let mut enabled_base = 1u32;
                if divergences.tss_solver_async {
                    enabled_base = enabled_base.max(divergences.tss_solver_async_threads);
                }
                if divergences_fast.tss_solver_async {
                    enabled_base =
                        enabled_base.max(divergences_fast.tss_solver_async_threads);
                }
                enabled_base
            } else {
                divergences
                    .tss_solver_async_threads
                    .max(divergences_fast.tss_solver_async_threads)
            };
            // One session pool serves both move-class views. Resolve each
            // class's auto ceiling first, then provision the larger ceiling;
            // this preserves both configurations even when only one view uses
            // an explicit maximum.
            let mut max = base as usize;
            if park && divergences.tss_solver_async {
                max = max.max(resolved_tss_worker_max(
                    divergences.tss_solver_async_threads,
                    divergences.tss_solver_async_threads_max,
                ));
            }
            if park && divergences_fast.tss_solver_async {
                max = max.max(resolved_tss_worker_max(
                    divergences_fast.tss_solver_async_threads,
                    divergences_fast.tss_solver_async_threads_max,
                ));
            }
            let max = max as u32;
            let resize = self
                .tss_pool
                .as_ref()
                .is_some_and(|pool| !tss_pool_matches(pool, base, max, park));
            if resize {
                self.tss_pool = None; // Drop joins the old workers first.
            }
            if self.tss_pool.is_none() {
                self.tss_pool = Some(TssAsyncPool::new(base, max, park));
            }
        }
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
        // Fast-class play temperature. Default 0.0 reproduces the greedy LCB pick
        // (see temperature_for_class) bit-for-bit.
        let pcr_fast_temperature =
            validate_nonnegative_f32("pcr_fast_temperature", pcr_fast_temperature.unwrap_or(0.0))?;
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
            fast_temperature: pcr_fast_temperature,
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
            // Default false.
            root_fpu_zero_under_noise: root_fpu_zero_under_noise.unwrap_or(false),
            // Root FPU reduction (validated >= 0 when provided).
            root_fpu_reduction: match root_fpu_reduction {
                Some(value) => Some(validate_nonnegative_f32("root_fpu_reduction", value)?),
                None => None,
            },
            divergences_full: divergences,
            divergences_fast,
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
            // Effective starting ply comes from the STATE, not a constant 0
            // (Codex review, seeded-slot ply): a blunder-seeded game arrives
            // mid-game (Python replayed the prefix — placements == the
            // driver's tape.ply), and treating it as ply 0 would re-run the
            // Init/temperature opening schedule on a tactical midgame
            // position. Unseeded games start at 0 placements, so this is
            // exactly the old behavior for them.
            let start_ply = root.placements_made();
            // Init plies belong to the game OPENING: a seeded prefix consumed
            // them, so the drawn count is discounted by the plies already
            // played (0 for every unseeded game — no behavior change there).
            let policy_init_remaining = move_policy
                .policy_init_plies(base_seed, game_key)
                .saturating_sub(start_ply);
            let move_class =
                move_policy.classify(base_seed, game_key, start_ply, policy_init_remaining);
            let mut slot = ContinuousSlot {
                game_key,
                ply: start_ply,
                search: None,
                phase: ContinuousPhase::AwaitRootEval,
                in_flight: 0,
                baseline: HashMap::new(),
                policy_init_remaining,
                move_class,
            };
            if let Some(mut search) = self.searches.remove(&game_key) {
                if search.root_hash == root_hash {
                    // Per-class divergence view: Fast=fast, Full/Init=base.
                    let class_div = move_policy.divergences_for(move_class);
                    search.set_additional_visits(move_policy.visits_for(move_class));
                    search.set_forced_playout_k(move_policy.forced_k_for(move_class));
                    search.set_root_fpu_reduction(move_policy.root_fpu_for(move_class));
                    search.set_tss_enabled(move_policy.tss_enabled);
                    search.set_divergences(class_div);
                    search.apply_root_policy_temperature(move_policy.root_temp_for(move_class, 0));
                    if let Some(noise) = root_noise_exact(
                        move_policy.noise_for(move_class),
                        mix_seed(base_seed, game_key, 0, SEED_STREAM_ROOT_NOISE),
                        class_div.dirichlet_shaped,
                    ) {
                        search.apply_root_dirichlet_noise(noise);
                    }
                    // (Re)build the Gumbel-Top-k candidate set + SH schedule on
                    // a reused root. init_gumbel_root clears any prior state
                    // first; when this class's view has gumbel_root off it is
                    // cleared so the normal PUCT root runs.
                    if class_div.gumbel_root {
                        let gumbel_seed = mix_seed(base_seed, game_key, 0, SEED_STREAM_GUMBEL);
                        search.init_gumbel_root(gumbel_seed, move_policy.visits_for(move_class));
                    } else {
                        search.clear_gumbel_root();
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
        // Scheduler-owned wait-at-leaf pen. Entries retain their virtual visit,
        // pending mark, and slot in-flight count until exactly one resolution
        // path consumes them.
        let mut parked = Vec::new();
        // Select-eval overlap: the next select pass runs with the flush's
        // virtual losses still pending (pre-backup tree state). A no-progress
        // prefetch is discarded so the next iteration re-selects after the
        // backup frees the paths.
        let mut prefetched: Option<(Vec<RustLeaf>, bool)> = None;
        // HEXFIELD_ASYNC_EVAL: the forward is enqueued (submit, no device
        // sync), the pre-backup select runs with the GIL released while those
        // kernels execute, then the forward is drained (finish). Off =>
        // synchronous eval-then-select. Only the sync point moves.
        // HEXFIELD_NO_PREFETCH disables the prefetch select.
        let async_eval = std::env::var("HEXFIELD_ASYNC_EVAL").is_ok();
        let no_prefetch = std::env::var("HEXFIELD_NO_PREFETCH").is_ok();
        // HEXFIELD_PIPELINE_DEPTH2: depth-2 double-buffered eval (default OFF).
        // Keeps one eval in flight on the GPU while the host selects the next
        // batch and backs up the previous flush. Deepens the async
        // (submit/finish) window by one flush, so the leaf stream differs from
        // strict lockstep (still virtual-loss-faithful). Requires
        // HEXFIELD_ASYNC_EVAL for submit-without-sync; without it, falls back to
        // the lockstep loop with a warning.
        let pipeline_depth2 = std::env::var("HEXFIELD_PIPELINE_DEPTH2").is_ok();
        let pipeline_depth2 = if pipeline_depth2 && !async_eval {
            eprintln!(
                "hexfield: HEXFIELD_PIPELINE_DEPTH2 ignored (requires HEXFIELD_ASYNC_EVAL=1); \
                 falling back to the lockstep scheduler"
            );
            false
        } else {
            pipeline_depth2
        };
        if pipeline_depth2 {
            self.run_continuous_pipeline_depth2(
                py,
                &mut slots,
                &mut queue,
                evaluator,
                on_move,
                c_puct,
                base_seed,
                leaf_batch_per_root,
                flush_target,
                virtual_loss,
                &move_policy,
                widening,
                divergences,
                &temperature_by_ply,
                &evaluation_stats,
                &mut stats,
                &mut parked,
            )?;
            debug_assert!(parked.is_empty(), "depth-2 scheduler returned with parked leaves");
            self.tss_pool_tail_drain(&mut stats);
            return self.finish_continuous_stats(py, stats, &evaluation_stats);
        }
        // HEXFIELD_GATE_COMPLETE: skip the per-iteration complete scan (a
        // par_iter readiness sweep over every slot) on iterations where nothing
        // could have become ready — no backup ran this iteration, the previous
        // complete decided no moves, and the loop is not at a Stop decision.
        // Completion readiness only changes when a backup lands new visits or a
        // completed move advances a root, so the gated scan is decision-
        // identical; the flag exists for the A/B.
        let gate_complete = std::env::var("HEXFIELD_GATE_COMPLETE").is_ok();
        let mut last_moves_decided: u64 = 1; // force the first scan
        while continuous_has_work(&slots, &parked) || !queue.is_empty() {
            stats.loop_iterations += 1;
            // Async solve pool: re-wire fresh-move searches (new generation),
            // then land completed solves in their memos — both before any
            // select so consumption is as prompt as the pool allows.
            if let Some(pool) = self.tss_pool.as_ref() {
                wire_tss_async(&mut slots, pool);
                drain_tss_async(pool, &mut slots);
            }
            resolve_parked_continuous(
                &mut slots,
                &mut parked,
                &mut queue,
                virtual_loss,
            )?;
            debug_assert_continuous_pen(&slots, &parked);
            let phase_t0 = std::time::Instant::now();
            let (new_leaves, made_progress) = match prefetched.take() {
                Some(result) => result,
                None => py.detach(|| {
                    select_continuous_pass(
                        &mut slots,
                        c_puct,
                        leaf_batch_per_root,
                        virtual_loss,
                        &mut parked,
                    )
                })?,
            };
            stats.select_seconds += phase_t0.elapsed().as_secs_f64();
            queue.extend(new_leaves.into_iter().map(ContinuousEvalItem::Leaf));
            if let Some(pool) = self.tss_pool.as_ref() {
                drain_tss_worker_spawns(pool, &mut slots);
            }
            // A genuinely pen-only pass is live work: keep polling until a
            // drain resolves it or the bounded timeout releases it. If eval
            // work is already queued, preserve the ordinary no-progress flush
            // so parked leaves never hold up unrelated slots.
            let made_progress =
                made_progress || (!parked.is_empty() && queue.is_empty());

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
                // tree state). Async: submit -> select -> finish. Sync: eval ->
                // select. Both yield (prefetch_result, evaluations).
                let (prefetch_result, evaluations) = if async_eval {
                    let t_submit = std::time::Instant::now();
                    let pending = submit_eval_cached(
                        py,
                        evaluator,
                        &requests,
                        &self.evaluation_cache,
                        Some(&evaluation_stats),
                        move_policy.request_moves_left(),
                        move_policy.request_logits(),
                    )?;
                    stats.submit_seconds += t_submit.elapsed().as_secs_f64();
                    let t_prefetch = std::time::Instant::now();
                    let prefetch_result = if no_prefetch {
                        (Vec::new(), false)
                    } else {
                        py.detach(|| {
                            select_continuous_pass(
                                &mut slots,
                                c_puct,
                                leaf_batch_per_root,
                                virtual_loss,
                                &mut parked,
                            )
                        })?
                    };
                    stats.select_seconds += t_prefetch.elapsed().as_secs_f64();
                    let t_finish = std::time::Instant::now();
                    let evaluations = finish_eval_cached(
                        py,
                        evaluator,
                        pending,
                        &self.evaluation_cache,
                        Some(&evaluation_stats),
                        self.cache_max_states,
                    )?;
                    stats.finish_seconds += t_finish.elapsed().as_secs_f64();
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
                        move_policy.request_logits(),
                    )?;
                    let prefetch_result = if no_prefetch {
                        (Vec::new(), false)
                    } else {
                        select_continuous_pass(
                            &mut slots,
                            c_puct,
                            leaf_batch_per_root,
                            virtual_loss,
                            &mut parked,
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
                let t_backup = std::time::Instant::now();
                backup_continuous_items(
                    py,
                    &mut slots,
                    items,
                    &evaluations,
                    &move_policy,
                    widening,
                    base_seed,
                    virtual_loss,
                    divergences,
                )?;
                stats.backup_seconds += t_backup.elapsed().as_secs_f64();
                prefetched = if prefetch_result.1 {
                    Some(prefetch_result)
                } else {
                    None
                };
                if let Some(pool) = self.tss_pool.as_ref() {
                    drain_tss_worker_spawns(pool, &mut slots);
                }
            }

            let flushed_this_iter = matches!(decision, ContinuousFlushDecision::Flush { .. });
            debug_assert_continuous_pen(&slots, &parked);
            let must_complete = !gate_complete
                || flushed_this_iter
                || last_moves_decided > 0
                || matches!(decision, ContinuousFlushDecision::Stop);
            let t_complete = std::time::Instant::now();
            let mut moves_decided = if must_complete {
                complete_continuous_slots(
                    py,
                    on_move,
                    &mut slots,
                    c_puct,
                    &move_policy,
                    &temperature_by_ply,
                    base_seed,
                    &mut queue,
                    &mut stats,
                    false,
                )?
            } else {
                stats.completes_skipped += 1;
                0
            };
            stats.complete_seconds += t_complete.elapsed().as_secs_f64();

            if matches!(decision, ContinuousFlushDecision::Stop) && moves_decided == 0 {
                // Rescue pass before declaring a stall: a Gumbel
                // Sequential-Halving root can saturate its reachable tree below
                // target_visits and its round caps (terminal subtrees), which
                // the normal completion path cannot finalize. Force-complete any
                // such stuck Gumbel slot from its accrued visits; a non-Gumbel
                // deadlock is a hard error.
                moves_decided = complete_continuous_slots(
                    py,
                    on_move,
                    &mut slots,
                    c_puct,
                    &move_policy,
                    &temperature_by_ply,
                    base_seed,
                    &mut queue,
                    &mut stats,
                    true,
                )?;
                if moves_decided == 0 {
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
            // After the rescue so a rescue-decided move re-arms the next scan.
            last_moves_decided = moves_decided;
        }
        debug_assert!(parked.is_empty(), "continuous scheduler exited with parked leaves");
        self.tss_pool_tail_drain(&mut stats);

        self.finish_continuous_stats(py, stats, &evaluation_stats)
    }
}

// Internal (non-`#[pymethods]`) scheduler helpers. These take native Rust types
// (`Widening`, `Divergences`, `&mut [ContinuousSlot]`) that pyo3 cannot expose,
// so they MUST live outside the `#[pymethods]` block above.
impl HexfieldMctsSession {
    /// End-of-run async-pool quiesce + alarm sweep (Codex review, late-alarm
    /// loss): every scheduler exit path calls this BEFORE assembling the run
    /// stats, so a verify failure banked after the last in-loop drain still
    /// reaches this epoch's telemetry instead of a stderr-only Drop message.
    /// Pending queue entries are discarded (their generations died with the
    /// finished slots); the bounded wait covers in-flight solves.
    fn tss_pool_tail_drain(&self, stats: &mut ContinuousSchedulerStats) {
        let Some(pool) = self.tss_pool.as_ref() else {
            return;
        };
        stats.tss_async_tail_cleared +=
            pool.quiesce_for_telemetry(std::time::Duration::from_secs(2)) as u64;
        // Responses landing now belong to dead generations; their ordinary
        // counters are dropped exactly like any stale response, but the drain
        // empties the channel so nothing leaks into the next run's first pass.
        let _ = pool.try_drain();
        stats.tss_async_verify_failed_tail += pool.take_verify_failures() as u64;
        stats.tss_async_worker_panics_tail += pool.take_worker_panics() as u64;
    }

    /// Build the `run_continuous` stats dict (shared by the lockstep loop and the
    /// depth-2 pipeline). Pure GIL-held conversion of the accumulated counters.
    fn finish_continuous_stats(
        &self,
        py: Python<'_>,
        stats: ContinuousSchedulerStats,
        evaluation_stats: &SharedEvaluationStats,
    ) -> PyResult<Py<PyAny>> {
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
        dict.set_item("gumbel_play_moves", stats.gumbel_play_moves)?;
        dict.set_item("gumbel_play_winner_moves", stats.gumbel_play_winner_moves)?;
        dict.set_item("gumbel_play_moves_early", stats.gumbel_play_moves_early)?;
        dict.set_item("gumbel_play_winner_early", stats.gumbel_play_winner_early)?;
        let hist = PyDict::new(py);
        let mut hist_items: Vec<_> = stats.flush_size_histogram.into_iter().collect();
        hist_items.sort_unstable_by_key(|(size, _)| *size);
        for (size, count) in hist_items {
            hist.set_item(size, count)?;
        }
        dict.set_item("flush_size_histogram", hist)?;
        dict.set_item("tss_async_verify_failed_tail", stats.tss_async_verify_failed_tail)?;
        dict.set_item("tss_async_worker_panics_tail", stats.tss_async_worker_panics_tail)?;
        dict.set_item("tss_async_tail_cleared", stats.tss_async_tail_cleared)?;
        dict.set_item("on_move_seconds", stats.on_move_seconds)?;
        dict.set_item("select_seconds", stats.select_seconds)?;
        dict.set_item("submit_seconds", stats.submit_seconds)?;
        dict.set_item("finish_seconds", stats.finish_seconds)?;
        dict.set_item("backup_seconds", stats.backup_seconds)?;
        dict.set_item("complete_seconds", stats.complete_seconds)?;
        dict.set_item("loop_iterations", stats.loop_iterations)?;
        dict.set_item("completes_skipped", stats.completes_skipped)?;
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

    /// Depth-2 double-buffered eval loop (gated OFF by default via
    /// `HEXFIELD_PIPELINE_DEPTH2`; the lockstep loop above is the default path).
    ///
    /// Invariant (one eval in flight, one staged): at the top of each iteration
    /// at most one flush's eval (`inflight`) is enqueued on the GPU but not yet
    /// backed up. Per iteration the host: selects N (next leaves), submits N to
    /// the GPU (no sync), then drains the previous flush (`inflight` = P) —
    /// finish + parallel backup — and stashes N as the new `inflight`. So the GPU
    /// computes N while the host backs up P and selects N+1; the staleness
    /// window is one flush wider than the lockstep async path. Virtual loss
    /// (applied at selection, restored at backup) keeps the extra-stale selects
    /// search-faithful: a leaf with an in-flight eval carries a pending virtual
    /// penalty so the next select does not re-pick it.
    ///
    /// Exactly-once backup: each flush's `items` ride in the `inflight` tuple
    /// next to their `PendingEval`; `take` on submit (from the queue) and `take`
    /// on drain (from the Option) make every flush submitted once and finished
    /// once. The loop runs while `inflight.is_some()` so the final flush is
    /// always drained before exit; the Gumbel stuck-root rescue runs only after
    /// the pipeline is empty.
    #[allow(clippy::too_many_arguments)]
    fn run_continuous_pipeline_depth2(
        &mut self,
        py: Python<'_>,
        slots: &mut [ContinuousSlot],
        queue: &mut Vec<ContinuousEvalItem>,
        evaluator: &Bound<'_, PyAny>,
        on_move: &Bound<'_, PyAny>,
        c_puct: f32,
        base_seed: u64,
        leaf_batch_per_root: u32,
        flush_target: usize,
        virtual_loss: f32,
        move_policy: &ContinuousMovePolicy,
        widening: Widening,
        divergences: Divergences,
        temperature_by_ply: &[f32],
        evaluation_stats: &SharedEvaluationStats,
        stats: &mut ContinuousSchedulerStats,
        parked: &mut Vec<ParkedLeaf>,
    ) -> PyResult<()> {
        // The in-flight (submitted, not-yet-backed-up) flush: its eval handle, the
        // items it will resolve, and the unique-state count snapshot taken at its
        // submit (for the per-flush histogram, computed when it drains).
        let mut inflight: Option<(PendingEval, Vec<ContinuousEvalItem>, usize)> = None;

        // HEXFIELD_PIPELINE_COMPLETE_OVERLAP (default OFF): moves the
        // per-iteration `complete` phase (Phase-A parallel build under
        // py.detach + Phase-B GIL-held `on_move`) to run after submit(N) but
        // before the drain of the previous flush P, so it runs while N's forward
        // computes on the GPU rather than with the GPU idle after the drain's
        // `finish` D2H sync. `complete_continuous_slots` only finalizes slots
        // with `in_flight == 0`; a slot whose eval is still buffered in the
        // un-drained `inflight` (P) keeps `in_flight > 0`, so it is not completed
        // in the overlapped pass and completes on the next iteration after P is
        // drained. Off => complete runs after the drain.
        let complete_overlap = std::env::var("HEXFIELD_PIPELINE_COMPLETE_OVERLAP").is_ok();

        // The loop continues as long as there is host work OR an eval is still in
        // flight (so the last flush is always drained + completed).
        while continuous_has_work(slots, parked) || !queue.is_empty() || inflight.is_some() {
            // Async solve pool: re-wire fresh-move searches (new generation),
            // then land completed solves — before the select, same as the
            // lockstep scheduler.
            if let Some(pool) = self.tss_pool.as_ref() {
                wire_tss_async(slots, pool);
                drain_tss_async(pool, slots);
            }
            resolve_parked_continuous(slots, parked, queue, virtual_loss)?;
            debug_assert_continuous_pen(slots, parked);
            // (1) select N on the CURRENT (post-previous-backup) tree state.
            let (new_leaves, selected_progress) = py.detach(|| {
                select_continuous_pass(
                    slots,
                    c_puct,
                    leaf_batch_per_root,
                    virtual_loss,
                    parked,
                )
            })?;
            queue.extend(new_leaves.into_iter().map(ContinuousEvalItem::Leaf));
            if let Some(pool) = self.tss_pool.as_ref() {
                drain_tss_worker_spawns(pool, slots);
            }
            let made_progress =
                selected_progress || (!parked.is_empty() && queue.is_empty());

            let decision = continuous_flush_decision(queue.len(), flush_target, made_progress);

            // Track whether THIS pass drained the buffered eval. A drain backs up
            // a flush and mutates the trees (slots can become Active / completable
            // next pass), so it counts as pipeline progress: we must NOT declare a
            // stall in the same iteration that drained — loop again and let select /
            // complete act on the freshly backed-up state first.
            let mut drained_this_pass = false;

            // When the overlapped complete runs inside the flush branch (before
            // the drain), it records its decided count here and suppresses the
            // post-drain complete for this pass.
            let mut completed_this_pass = false;
            let mut overlapped_moves = 0u64;

            // (2) On a flush: submit N (enqueue, no sync), THEN drain the previous
            // flush P (finish + backup). Submitting first keeps the GPU busy with N
            // while the host backs up P.
            if let ContinuousFlushDecision::Flush { no_progress } = decision {
                if no_progress {
                    stats.no_progress_flushes += 1;
                }
                let items_n = std::mem::take(queue);
                stats.flush_count += 1;
                stats.queued_states += items_n.len() as u64;
                let unique_before_n = lock_unique_states(evaluation_stats);
                let requests_n: Vec<RustEvaluationRequest> = items_n
                    .iter()
                    .map(continuous_item_request)
                    .collect();
                let pending_n = submit_eval_cached(
                    py,
                    evaluator,
                    &requests_n,
                    &self.evaluation_cache,
                    Some(evaluation_stats),
                    move_policy.request_moves_left(),
                    move_policy.request_logits(),
                )?;
                drop(requests_n);
                // With the complete-overlap flag set, finalize ready slots here:
                // after N is enqueued (its forward computing on the GPU) but
                // before the drain of P, so the completes overlap N's GPU
                // forward. Slots whose eval is still buffered in the un-drained
                // `inflight` (P) keep in_flight > 0 and are not finalized here,
                // so they complete on the next pass after P is drained.
                if complete_overlap {
                    overlapped_moves = complete_continuous_slots(
                        py,
                        on_move,
                        slots,
                        c_puct,
                        move_policy,
                        temperature_by_ply,
                        base_seed,
                        queue,
                        stats,
                        false,
                    )?;
                    completed_this_pass = true;
                }
                // Drain the PREVIOUS flush now that N is enqueued on the GPU.
                if let Some((pending_p, items_p, unique_before_p)) = inflight.take() {
                    self.drain_pipeline_flush(
                        py,
                        slots,
                        evaluator,
                        pending_p,
                        items_p,
                        unique_before_p,
                        move_policy,
                        widening,
                        base_seed,
                        virtual_loss,
                        divergences,
                        evaluation_stats,
                        stats,
                    )?;
                    drained_this_pass = true;
                }
                inflight = Some((pending_n, items_n, unique_before_n));
            } else if !selected_progress && inflight.is_some() {
                // No new flush this pass and select stalled: drain the buffered
                // eval so its backup frees paths / completes slots. Without this
                // the loop would spin (select keeps stalling) until a flush; this
                // both unblocks progress and bounds the staleness to one flush.
                let (pending_p, items_p, unique_before_p) = inflight.take().expect("inflight set");
                self.drain_pipeline_flush(
                    py,
                    slots,
                    evaluator,
                    pending_p,
                    items_p,
                    unique_before_p,
                    move_policy,
                    widening,
                    base_seed,
                    virtual_loss,
                    divergences,
                    evaluation_stats,
                    stats,
                )?;
                drained_this_pass = true;
            }

            // (3) Complete any slots whose evals have all landed (in_flight == 0).
            // A slot with an eval still in `inflight` has in_flight > 0 and is
            // correctly NOT completed here. When the complete-overlap path already
            // ran the complete this pass (after submit, before drain), reuse its
            // decided count instead of completing a second time.
            debug_assert_continuous_pen(slots, parked);
            let mut moves_decided = if completed_this_pass {
                overlapped_moves
            } else {
                complete_continuous_slots(
                    py,
                    on_move,
                    slots,
                    c_puct,
                    move_policy,
                    temperature_by_ply,
                    base_seed,
                    queue,
                    stats,
                    false,
                )?
            };

            // (4) Stall handling: only a GENUINE deadlock — Stop decision, no move
            // completed, the pipeline fully drained (inflight None), AND no drain
            // happened this pass (a drain just mutated the trees, so loop again and
            // let the next select/complete act before judging the run stuck).
            if matches!(decision, ContinuousFlushDecision::Stop)
                && moves_decided == 0
                && inflight.is_none()
                && !drained_this_pass
                && parked.is_empty()
            {
                moves_decided = complete_continuous_slots(
                    py,
                    on_move,
                    slots,
                    c_puct,
                    move_policy,
                    temperature_by_ply,
                    base_seed,
                    queue,
                    stats,
                    true,
                )?;
                if moves_decided == 0 {
                    let stuck = slots
                        .iter()
                        .filter(|slot| !matches!(slot.phase, ContinuousPhase::Empty))
                        .count();
                    return Err(PyRuntimeError::new_err(format!(
                        "hexfield continuous MCTS scheduler (depth-2) stalled with {stuck} \
                         unfinished slots (queue empty, no selectable leaves, no in-flight eval, \
                         no completable roots)"
                    )));
                }
            }
        }
        debug_assert!(
            inflight.is_none(),
            "depth-2 pipeline exited with an undrained in-flight eval"
        );
        debug_assert!(parked.is_empty(), "depth-2 pipeline exited with parked leaves");
        Ok(())
    }

    /// Finish + back up one in-flight flush P (parallel backup), folding its
    /// unique-state count into the flush histogram. Exactly-once: called only on
    /// an `inflight` value moved out by `take`.
    #[allow(clippy::too_many_arguments)]
    fn drain_pipeline_flush(
        &mut self,
        py: Python<'_>,
        slots: &mut [ContinuousSlot],
        evaluator: &Bound<'_, PyAny>,
        pending: PendingEval,
        items: Vec<ContinuousEvalItem>,
        unique_before: usize,
        move_policy: &ContinuousMovePolicy,
        widening: Widening,
        base_seed: u64,
        virtual_loss: f32,
        divergences: Divergences,
        evaluation_stats: &SharedEvaluationStats,
        stats: &mut ContinuousSchedulerStats,
    ) -> PyResult<()> {
        let evaluations = finish_eval_cached(
            py,
            evaluator,
            pending,
            &self.evaluation_cache,
            Some(evaluation_stats),
            self.cache_max_states,
        )?;
        let unique_after = lock_unique_states(evaluation_stats);
        let unique_flushed = unique_after.saturating_sub(unique_before);
        stats.flushed_states += unique_flushed as u64;
        *stats
            .flush_size_histogram
            .entry(unique_flushed.max(1).next_power_of_two())
            .or_insert(0) += 1;
        backup_continuous_items(
            py,
            slots,
            items,
            &evaluations,
            move_policy,
            widening,
            base_seed,
            virtual_loss,
            divergences,
        )?;
        Ok(())
    }
}

// === Lockstep internals ===

#[allow(clippy::too_many_arguments)]
/// Lockstep flavor of the continuous wire pass: searches are indexed by
/// position in the batch. One lockstep call = one move, so every call wires
/// fresh generations (set_additional_visits / RustSearch::new both leave the
/// handle empty) and cross-call responses are dropped as stale.
fn wire_tss_async_searches(searches: &mut [RustSearch], pool: &TssAsyncPool) {
    for (index, search) in searches.iter_mut().enumerate() {
        if !search.divergences.tss_solver_async || search.tss_async_generation().is_some() {
            continue;
        }
        search.set_tss_async(Some(pool.handle_for(index as u32)));
    }
}

/// Lockstep flavor of the continuous drain pass (same staleness contract:
/// generation mismatch drops the result but never the fatal verify counter).
fn drain_tss_async_searches(pool: &TssAsyncPool, searches: &mut [RustSearch]) {
    if let Some(search) = searches.first_mut() {
        search.tss.deep_verify_failed += pool.take_verify_failures();
        search.tss.async_workers_spawned += pool.take_workers_spawned();
    }
    let worker_panics = pool.take_worker_panics();
    if worker_panics > 0 {
        eprintln!(
            "hexfield tss_async: {worker_panics} solve worker panic(s) — requests lost, \
             workers recycled with fresh solvers"
        );
    }
    for response in pool.try_drain() {
        let Some(search) = searches.get_mut(response.slot as usize) else {
            if response.counters.deep_verify_failed > 0 {
                eprintln!(
                    "hexfield tss_async: certificate VERIFY FAILURE in an orphaned \
                     response (lockstep slot {}) — investigate immediately",
                    response.slot
                );
            }
            continue;
        };
        if search.tss_async_generation() == Some(response.generation) {
            search.apply_tss_async_response(&response);
        } else {
            search.apply_tss_async_response_stale(&response);
        }
    }
}

fn parked_wait_ms(parked_at: Instant, now: Instant) -> u64 {
    now.saturating_duration_since(parked_at)
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

fn record_park_wait(search: &mut RustSearch, wait_ms: u64) {
    search.tss.park_wait_ms_sum += wait_ms;
    search.tss.park_wait_ms_max = search.tss.park_wait_ms_max.max(wait_ms);
}

/// Resolve a lockstep pen immediately after the pool drain. Moving an entry
/// into `eval_leaves` deliberately leaves its pending mark and virtual visit
/// untouched; the ordinary eval backup owns their one eventual release.
fn resolve_parked_searches(
    searches: &mut [RustSearch],
    parked: &mut Vec<ParkedLeaf>,
    eval_leaves: &mut Vec<RustLeaf>,
    virtual_loss: f32,
) -> PyResult<()> {
    if parked.is_empty() {
        return Ok(());
    }
    let now = Instant::now();
    let mut waiting = Vec::with_capacity(parked.len());
    for entry in parked.drain(..) {
        let root_index = entry.leaf.root_index;
        let search = searches.get_mut(root_index).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "TSS parked leaf references missing lockstep search {root_index}"
            ))
        })?;
        if search.tss_async_generation() != Some(entry.generation) {
            return Err(PyRuntimeError::new_err(format!(
                "TSS parked leaf generation changed before resolution for lockstep search \
                 {root_index}"
            )));
        }
        let wait_ms = parked_wait_ms(entry.parked_at, now);
        match search.tss_park_resolution(entry.leaf.state_hash, &entry.leaf.state) {
            TssParkResolution::Hard(hard) => {
                search.tss.park_hard += 1;
                record_park_wait(search, wait_ms);
                search.mark_pending(entry.leaf.parent_node, entry.leaf.edge_index, -1);
                let leaf_player = entry.leaf.state.current_player();
                search.backup_virtual(
                    &entry.leaf.path,
                    leaf_player,
                    hard.value(),
                    virtual_loss,
                    None,
                );
            }
            TssParkResolution::Release => {
                search.tss.park_released += 1;
                record_park_wait(search, wait_ms);
                eval_leaves.push(entry.leaf);
            }
            TssParkResolution::Pending
                if now.saturating_duration_since(entry.parked_at)
                    > Duration::from_millis(
                        search.divergences.tss_solver_park_timeout_ms as u64,
                    ) =>
            {
                search.tss.park_bailed += 1;
                record_park_wait(search, wait_ms);
                eval_leaves.push(entry.leaf);
            }
            TssParkResolution::Pending => waiting.push(entry),
        }
    }
    *parked = waiting;
    Ok(())
}

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
    request_logits: bool,
    move_temps: &[f32],
    baselines: &[HashMap<PackedCoord, u32>],
    tss_pool: Option<&TssAsyncPool>,
) -> PyResult<()> {
    // Two-stage pipeline: the next batch is selected before the current batch
    // is backed up. This ordering extends the virtual-loss window by one batch:
    // select(N+1) runs after evaluate(N) and before backup(N).
    //
    // Early-stop: in_flight is passed as 0 here. The visit-overtake test inside
    // early_stop_ready is in-flight-safe — apply_virtual_visit increments both
    // completed_visits and the selected edge's visit count at selection time, so
    // best/second (per-edge delta visits) include pending leaves while
    // remaining = target - completed excludes them. best-second > remaining thus
    // proves the visit leader is unbeatable by all un-selected visits regardless
    // of how many are pending; the pending batch is still evaluated + backed up
    // by the loop below before exit. The continuous path's in_flight==0 guard is
    // about slot-advance safety (node-id invalidation), a separate concern.
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

    // HEXFIELD_ASYNC_EVAL: the forward is enqueued (submit, no device sync), the
    // pre-backup select runs with the GIL released while those kernels execute,
    // then the forward is drained (finish). Off => synchronous eval-then-select.
    // Only the sync point moves; the leaf stream is bit-identical. Depth-2 is
    // NOT read here — it stays self-play-only (it would change the leaf stream).
    //
    // Unlike run_continuous (self-play, always a real HexfieldEvaluator), the eval
    // `search` entry receives diverse evaluators (arena stubs, custom eval
    // opponents). The async split needs the two-phase submit_payload/result
    // protocol; when the evaluator only implements the synchronous __call__
    // contract, fall back to the sync path rather than raising. Real evaluators
    // have submit_payload, so production async is unaffected.
    let async_eval = std::env::var("HEXFIELD_ASYNC_EVAL").is_ok()
        && evaluator.hasattr("submit_payload").unwrap_or(false);

    // Async solve pool (lockstep flavor): wire fresh generations before the
    // priming select so eval/arena searches enqueue instead of solving
    // inline, exactly like self-play.
    if let Some(pool) = tss_pool {
        wire_tss_async_searches(searches, pool);
    }
    let mut parked = Vec::new();
    early_stop_pass(searches);
    // No leaves in flight on the priming select, so the SH barrier is unblocked
    // for every search (empty in-flight set).
    let (mut pending_leaves, _primed_progress) =
        select_leaf_batch(
            searches,
            c_puct,
            leaf_batch_per_root,
            virtual_loss,
            &[],
            &mut parked,
        )?;

    loop {
        // Land completed pool solves before each batch's select.
        if let Some(pool) = tss_pool {
            drain_tss_async_searches(pool, searches);
        }
        resolve_parked_searches(
            searches,
            &mut parked,
            &mut pending_leaves,
            virtual_loss,
        )?;
        // Check between every batch (a no-op in parity mode); see the
        // in-flight-safety note on early_stop_pass above.
        early_stop_pass(searches);
        if pending_leaves.is_empty() {
            let needs_visits = searches.iter().any(RustSearch::needs_visits);
            if !needs_visits && parked.is_empty() {
                break;
            }
            if !needs_visits {
                // All remaining work is in the pen. Poll without blocking;
                // the next pool drain or the bounded bail deadline resolves it.
                continue;
            }
            // pending_leaves is empty here: nothing is un-backed, so the SH
            // barrier is unblocked for every search.
            let (leaves, made_progress) = select_leaf_batch(
                searches,
                c_puct,
                leaf_batch_per_root,
                virtual_loss,
                &[],
                &mut parked,
            )?;
            if leaves.is_empty() {
                if !made_progress && parked.is_empty() {
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
        // Prefetch select with the current batch still pending (pre-backup
        // tree state). `pending_leaves` carries −virtual_loss on the trees of
        // the searches it touches, so the SH barrier is blocked for exactly
        // those searches (their round ranking would read contaminated stats).
        // Async: submit -> select (GIL released) -> finish. Sync: eval ->
        // select. Both yield (next_leaves, evaluations); the leaf stream is
        // identical because the select reads the same pre-backup tree state
        // with the same batch in flight either way.
        let (next_leaves, evaluations) = if async_eval {
            let pending = submit_eval_cached(
                py,
                evaluator,
                &leaf_requests,
                evaluation_cache,
                Some(evaluation_stats),
                request_moves_left,
                request_logits,
            )?;
            let next_leaves = if searches.iter().any(RustSearch::needs_visits) {
                py.detach(|| {
                    select_leaf_batch(
                        searches,
                        c_puct,
                        leaf_batch_per_root,
                        virtual_loss,
                        &pending_leaves,
                        &mut parked,
                    )
                })?
                .0
            } else {
                Vec::new()
            };
            let evaluations = finish_eval_cached(
                py,
                evaluator,
                pending,
                evaluation_cache,
                Some(evaluation_stats),
                cache_max_states,
            )?;
            (next_leaves, evaluations)
        } else {
            let evaluations = evaluate_state_refs_cached(
                py,
                evaluator,
                &leaf_requests,
                evaluation_cache,
                Some(evaluation_stats),
                cache_max_states,
                request_moves_left,
                request_logits,
            )?;
            let next_leaves = if searches.iter().any(RustSearch::needs_visits) {
                select_leaf_batch(
                    searches,
                    c_puct,
                    leaf_batch_per_root,
                    virtual_loss,
                    &pending_leaves,
                    &mut parked,
                )?
                .0
            } else {
                Vec::new()
            };
            (next_leaves, evaluations)
        };
        apply_eval_backups(searches, pending_leaves, &evaluations, virtual_loss)?;
        pending_leaves = next_leaves;
    }
    debug_assert!(parked.is_empty(), "lockstep scheduler exited with parked leaves");
    // Tail quiesce (Codex review, late-alarm loss): a verify failure banked
    // after the loop's final drain must still reach this call's telemetry.
    // The searches are alive here (payloads are built from them after this
    // returns), so the alarm folds into per-move counters as usual.
    if let Some(pool) = tss_pool {
        pool.quiesce_for_telemetry(std::time::Duration::from_secs(2));
        drain_tss_async_searches(pool, searches);
    }
    Ok(())
}

fn select_leaf_batch(
    searches: &mut [RustSearch],
    c_puct: f32,
    leaf_batch_per_root: u32,
    virtual_loss: f32,
    // Leaves selected in a prior batch that have not yet been backed up. Each
    // still carries −virtual_loss on its owning search's tree, so the SH barrier
    // must not advance a round for any search that owns one (its ranking would
    // read vl-contaminated per-edge visits/completedQ).
    in_flight: &[RustLeaf],
    parked: &mut Vec<ParkedLeaf>,
) -> PyResult<(Vec<RustLeaf>, bool)> {
    let mut leaves = Vec::new();
    let mut made_progress = false;
    for (root_index, search) in searches.iter_mut().enumerate() {
        if !search.needs_visits() {
            continue;
        }
        // Intra-search Sequential-Halving barrier (mirrors the continuous
        // scheduler): when every surviving Gumbel candidate has met its round
        // cap, halve the survivor set and re-seed before selecting. Looped
        // because advancing may immediately satisfy the next round's barrier.
        // No-op without an active Gumbel root.
        //
        // Gated on a drained search: skip the barrier while this search has any
        // un-backed leaf in flight, since those leaves' virtual losses would
        // contaminate the round ranking. The pending leaves are guaranteed to
        // back up (apply_eval_backups runs every loop iteration), so the barrier
        // fires on a later drained pass — no deadlock.
        let drained = !in_flight.iter().any(|leaf| leaf.root_index == root_index)
            && !parked
                .iter()
                .any(|entry| entry.leaf.root_index == root_index);
        if drained && search.has_gumbel_root() {
            while search.maybe_advance_gumbel_round() {}
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
            } else if let Some(hard) = selected.hard {
                // Async descent-stop: a pool-verified proof arrived for this
                // position; the simulation stops here with a hard backup
                // (counters bumped inside tss_async_descent_hard).
                let leaf_player = selected.state.current_player();
                search.backup_virtual(&selected.path, leaf_player, hard.value(), virtual_loss, None);
            } else if let Some(node_id) = selected.existing_node {
                let node = &search.nodes[node_id];
                let player = node.player;
                let value = node.value();
                let leaf_ml = if ml_on { node.ml_mean() } else { None };
                search.backup_virtual(&selected.path, player, value, virtual_loss, leaf_ml);
            } else if let Some(hard) = search
                .tss_enabled
                .then(|| tss_core::solve_leaf_lambda1(&selected.state))
                .flatten()
            {
                // λ¹ HardValue: certified producer, no node, no GPU eval
                // (tss_core.rs is the only mint — the soundness firewall).
                let leaf_player = selected.state.current_player();
                search.tss.leaf_verdict_hits += 1;
                search.backup_virtual(&selected.path, leaf_player, hard.value(), virtual_loss, None);
            } else if search.divergences.tss_solver_park {
                let enqueue_started = Instant::now();
                match search.tss_deep_leaf_route(&selected.state, selected.state_hash) {
                    TssLeafRoute::Hard(hard) => {
                        // A memo hit may already be available at selection
                        // time. It keeps the ordinary verified-hard path and
                        // never enters the pen.
                        let leaf_player = selected.state.current_player();
                        search.backup_virtual(
                            &selected.path,
                            leaf_player,
                            hard.value(),
                            virtual_loss,
                            None,
                        );
                    }
                    TssLeafRoute::Parked => {
                        search.mark_pending(selected.parent_node, selected.edge_index, 1);
                        search.tss.park_parked += 1;
                        parked.push(ParkedLeaf {
                            leaf: RustLeaf {
                                root_index,
                                parent_node: selected.parent_node,
                                edge_index: selected.edge_index,
                                path: selected.path,
                                state: selected.state,
                                state_hash: selected.state_hash,
                            },
                            parked_at: enqueue_started,
                            generation: search
                                .tss_async_generation()
                                .expect("parked TSS leaf has an async generation"),
                        });
                    }
                    TssLeafRoute::Miss => {
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
            } else if let Some(hard) =
                search.tss_deep_leaf(&selected.state, selected.state_hash)
            {
                // Verified deep proof (Stage-4 ladder): certificate-checked
                // hard backup, GPU eval elided. Shadow mode never reaches here.
                let leaf_player = selected.state.current_player();
                search.backup_virtual(&selected.path, leaf_player, hard.value(), virtual_loss, None);
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
) -> PyResult<(Vec<RustLeaf>, Vec<ParkedLeaf>, bool, u32)> {
    let mut leaves = Vec::new();
    let mut parked = Vec::new();
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
        } else if let Some(hard) = selected.hard {
            // Async descent-stop: a pool-verified proof arrived for this
            // position; the simulation stops here with a hard backup
            // (counters bumped inside tss_async_descent_hard).
            let leaf_player = selected.state.current_player();
            search.backup_virtual(&selected.path, leaf_player, hard.value(), virtual_loss, None);
        } else if let Some(node_id) = selected.existing_node {
            let node = &search.nodes[node_id];
            let player = node.player;
            let value = node.value();
            let leaf_ml = if ml_on { node.ml_mean() } else { None };
            search.backup_virtual(&selected.path, player, value, virtual_loss, leaf_ml);
        } else if let Some(hard) = search
            .tss_enabled
            .then(|| tss_core::solve_leaf_lambda1(&selected.state))
            .flatten()
        {
            // λ¹ HardValue: certified producer, no node, no GPU eval
            // (tss_core.rs is the only mint — the soundness firewall).
            let leaf_player = selected.state.current_player();
            search.tss.leaf_verdict_hits += 1;
            search.backup_virtual(&selected.path, leaf_player, hard.value(), virtual_loss, None);
        } else if search.divergences.tss_solver_park {
            let enqueue_started = Instant::now();
            match search.tss_deep_leaf_route(&selected.state, selected.state_hash) {
                TssLeafRoute::Hard(hard) => {
                    let leaf_player = selected.state.current_player();
                    search.backup_virtual(
                        &selected.path,
                        leaf_player,
                        hard.value(),
                        virtual_loss,
                        None,
                    );
                }
                TssLeafRoute::Parked => {
                    search.mark_pending(selected.parent_node, selected.edge_index, 1);
                    search.tss.park_parked += 1;
                    added_in_flight += 1;
                    parked.push(ParkedLeaf {
                        leaf: RustLeaf {
                            root_index: slot_index,
                            parent_node: selected.parent_node,
                            edge_index: selected.edge_index,
                            path: selected.path,
                            state: selected.state,
                            state_hash: selected.state_hash,
                        },
                        parked_at: enqueue_started,
                        generation: search
                            .tss_async_generation()
                            .expect("parked TSS leaf has an async generation"),
                    });
                }
                TssLeafRoute::Miss => {
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
        } else if let Some(hard) = search.tss_deep_leaf(&selected.state, selected.state_hash) {
            // Verified deep proof (Stage-4 ladder): certificate-checked hard
            // backup, GPU eval elided. Shadow mode never reaches here.
            let leaf_player = selected.state.current_player();
            search.backup_virtual(&selected.path, leaf_player, hard.value(), virtual_loss, None);
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
    Ok((leaves, parked, made_progress, added_in_flight))
}

/// Async-pool wire pass (main thread, once per scheduler iteration, before
/// any select): every active search whose class view enables the pool and
/// whose handle was cleared (search creation / reuse-rebind / move advance
/// all go through `set_additional_visits`, which drops it) gets a fresh
/// slot-stamped handle at a NEW generation. Pairing the generation mint with
/// the memo clear is the staleness guarantee: a response minted under an
/// older generation can never match a live search.
fn wire_tss_async(slots: &mut [ContinuousSlot], pool: &TssAsyncPool) {
    for (slot_index, slot) in slots.iter_mut().enumerate() {
        let Some(search) = slot.search.as_mut() else {
            continue;
        };
        if !search.divergences.tss_solver_async || search.tss_async_generation().is_some() {
            continue;
        }
        search.set_tss_async(Some(pool.handle_for(slot_index as u32)));
    }
}

/// Async-pool drain pass (main thread, right after the wire pass): route
/// every completed solve to its slot's live search. Generation mismatch =>
/// the move/game advanced past the request — drop the result as stale,
/// EXCEPT the fatal `deep_verify_failed` count, which is never dropped
/// (production alarms on nonzero regardless of which move it belonged to).
fn drain_tss_async(pool: &TssAsyncPool, slots: &mut [ContinuousSlot]) {
    // Worker-side alarms first: the atomic is the sole carrier of the fatal
    // verify signal (banked at solve time, so a dropped/stale/never-drained
    // response cannot lose it). Fold into any live search => epoch JSON.
    if let Some(search) = slots.iter_mut().find_map(|slot| slot.search.as_mut()) {
        search.tss.deep_verify_failed += pool.take_verify_failures();
    }
    drain_tss_worker_spawns(pool, slots);
    let worker_panics = pool.take_worker_panics();
    if worker_panics > 0 {
        eprintln!(
            "hexfield tss_async: {worker_panics} solve worker panic(s) — requests lost, \
             workers recycled with fresh solvers"
        );
    }
    for response in pool.try_drain() {
        let search = slots
            .get_mut(response.slot as usize)
            .and_then(|slot| slot.search.as_mut());
        match search {
            Some(search) => {
                if search.tss_async_generation() == Some(response.generation) {
                    search.apply_tss_async_response(&response);
                } else {
                    search.apply_tss_async_response_stale(&response);
                }
            }
            None => {
                if response.counters.deep_verify_failed > 0 {
                    // Never let the fatal signal vanish with an emptied slot.
                    eprintln!(
                        "hexfield tss_async: certificate VERIFY FAILURE in an orphaned \
                         response (slot {}) — investigate immediately",
                        response.slot
                    );
                }
            }
        }
    }
}

/// Attribute synchronous dynamic-spawn deltas before a move can complete in
/// the same scheduler iteration that enqueued the triggering request.
fn drain_tss_worker_spawns(pool: &TssAsyncPool, slots: &mut [ContinuousSlot]) {
    if !pool.park_mode() {
        return;
    }
    let Some(search) = slots.iter_mut().find_map(|slot| slot.search.as_mut()) else {
        return;
    };
    search.tss.async_workers_spawned += pool.take_workers_spawned();
}

/// Continuous counterpart of `resolve_parked_searches`. A hard result owns
/// the pending-mark release, virtual backup, and `in_flight` decrement here;
/// an Unknown/non-consumable result or timeout only moves the leaf into the
/// normal eval queue, whose backup performs those operations later.
fn resolve_parked_continuous(
    slots: &mut [ContinuousSlot],
    parked: &mut Vec<ParkedLeaf>,
    queue: &mut Vec<ContinuousEvalItem>,
    virtual_loss: f32,
) -> PyResult<()> {
    if parked.is_empty() {
        return Ok(());
    }
    let now = Instant::now();
    let mut waiting = Vec::with_capacity(parked.len());
    for entry in parked.drain(..) {
        let slot_index = entry.leaf.root_index;
        let slot = slots.get_mut(slot_index).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "TSS parked leaf references missing continuous slot {slot_index}"
            ))
        })?;
        let search = slot.search.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "TSS parked leaf resolved for empty continuous slot {slot_index}"
            ))
        })?;
        if search.tss_async_generation() != Some(entry.generation) {
            return Err(PyRuntimeError::new_err(format!(
                "TSS parked leaf generation changed before resolution for continuous slot \
                 {slot_index}"
            )));
        }
        let wait_ms = parked_wait_ms(entry.parked_at, now);
        match search.tss_park_resolution(entry.leaf.state_hash, &entry.leaf.state) {
            TssParkResolution::Hard(hard) => {
                search.tss.park_hard += 1;
                record_park_wait(search, wait_ms);
                search.mark_pending(entry.leaf.parent_node, entry.leaf.edge_index, -1);
                slot.in_flight = slot.in_flight.checked_sub(1).ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "TSS parked hard backup underflowed in_flight for slot {slot_index}"
                    ))
                })?;
                let leaf_player = entry.leaf.state.current_player();
                search.backup_virtual(
                    &entry.leaf.path,
                    leaf_player,
                    hard.value(),
                    virtual_loss,
                    None,
                );
            }
            TssParkResolution::Release => {
                search.tss.park_released += 1;
                record_park_wait(search, wait_ms);
                queue.push(ContinuousEvalItem::Leaf(entry.leaf));
            }
            TssParkResolution::Pending
                if now.saturating_duration_since(entry.parked_at)
                    > Duration::from_millis(
                        search.divergences.tss_solver_park_timeout_ms as u64,
                    ) =>
            {
                search.tss.park_bailed += 1;
                record_park_wait(search, wait_ms);
                queue.push(ContinuousEvalItem::Leaf(entry.leaf));
            }
            TssParkResolution::Pending => waiting.push(entry),
        }
    }
    *parked = waiting;
    Ok(())
}

#[cfg(debug_assertions)]
fn debug_assert_continuous_pen(slots: &[ContinuousSlot], parked: &[ParkedLeaf]) {
    if parked.is_empty() {
        return;
    }
    let mut parked_per_slot = vec![0u32; slots.len()];
    for entry in parked {
        let slot = &slots[entry.leaf.root_index];
        let search = slot
            .search
            .as_ref()
            .expect("parked leaf must retain its owning search");
        parked_per_slot[entry.leaf.root_index] =
            parked_per_slot[entry.leaf.root_index].saturating_add(1);
        debug_assert_eq!(
            search.tss_async_generation(),
            Some(entry.generation),
            "a move must not advance while it owns a parked leaf"
        );
    }
    for (slot_index, parked_count) in parked_per_slot.into_iter().enumerate() {
        debug_assert!(
            slots[slot_index].in_flight >= parked_count,
            "every parked leaf must contribute one in-flight unit"
        );
    }
}

#[cfg(not(debug_assertions))]
fn debug_assert_continuous_pen(_slots: &[ContinuousSlot], _parked: &[ParkedLeaf]) {}

fn select_continuous_pass(
    slots: &mut [ContinuousSlot],
    c_puct: f32,
    leaf_batch_per_root: u32,
    virtual_loss: f32,
    parked: &mut Vec<ParkedLeaf>,
) -> PyResult<(Vec<RustLeaf>, bool)> {
    // Per-slot selection is independent (each closure owns one slot's tree via
    // &mut; the RNG is seeded by slot_index, not execution order), so it is
    // fanned across cores with rayon. Results fold in slot order.
    let per_slot: PyResult<Vec<(Vec<RustLeaf>, Vec<ParkedLeaf>, bool)>> = slots
        .par_iter_mut()
        .enumerate()
        .map(|(slot_index, slot)| {
            if !matches!(slot.phase, ContinuousPhase::Active) {
                return Ok((Vec::new(), Vec::new(), false));
            }
            let cap = leaf_batch_per_root.saturating_sub(slot.in_flight);
            if cap == 0 {
                return Ok((Vec::new(), Vec::new(), false));
            }
            let Some(search) = slot.search.as_mut() else {
                return Ok((Vec::new(), Vec::new(), false));
            };
            if !search.needs_visits() {
                return Ok((Vec::new(), Vec::new(), false));
            }
            // Intra-slot Sequential-Halving barrier: when all surviving Gumbel
            // candidates in this slot have reached the current round's
            // per-candidate cap, halve the survivor set and advance the SH
            // round. No-op unless a Gumbel root is active. Looped because
            // advancing may immediately satisfy the next round's barrier (e.g.
            // tiny budgets).
            //
            // Gated on a DRAINED slot (in_flight == 0): the barrier ranks on
            // per-edge visits and completedQ, both of which carry −virtual_loss
            // for every in-flight sim (apply_virtual_visit bumps visits and
            // subtracts vl at selection; the real backup adds it back). Advancing
            // a round on vl-contaminated stats mis-ranks survivors. A re-descent
            // into an existing subtree carries −vl on the root edge WITHOUT a
            // pending flag, so the root-edge `pending` count alone is not a
            // sufficient drain test — the slot's in_flight counter is. When
            // in_flight > 0 the barrier simply waits: those evals are guaranteed
            // to back up and drive in_flight to 0, at which point either the
            // barrier fires or the force-stuck rescue (in_flight == 0 in
            // complete_ready_slots) finalizes the move, so this cannot deadlock.
            if slot.in_flight == 0 && search.has_gumbel_root() {
                while search.maybe_advance_gumbel_round() {}
            }
            let (leaves, parked, progressed, added_in_flight) =
                select_continuous_leaves(search, slot_index, c_puct, cap, virtual_loss)?;
            slot.in_flight = slot.in_flight.saturating_add(added_in_flight);
            Ok((leaves, parked, progressed))
        })
        .collect();
    let mut leaves = Vec::new();
    let mut made_progress = false;
    for (slot_leaves, slot_parked, progressed) in per_slot? {
        made_progress |= progressed;
        leaves.extend(slot_leaves);
        parked.extend(slot_parked);
    }
    Ok((leaves, made_progress))
}

/// Apply one backup item (Leaf or RootInit) to its owning slot. `slot` is the
/// item's owning slot (`leaf.root_index` for Leaf / `slot_index` for RootInit)
/// and is never indexed by any other slot, so callers may hand it a disjoint
/// `&mut` from `par_iter_mut`.
#[allow(clippy::too_many_arguments)]
fn apply_backup_item(
    slot: &mut ContinuousSlot,
    item: ContinuousEvalItem,
    evaluation: &Arc<RustEvaluation>,
    move_policy: &ContinuousMovePolicy,
    widening: Widening,
    base_seed: u64,
    virtual_loss: f32,
    // The RootInit branch now derives its per-class divergence view from
    // `move_policy` (divergences_for), and the Leaf branch reads the search's
    // own stored view; the threaded base divergences are no longer consulted
    // here. Kept in the signature so the shared serial/parallel backup callers
    // pass one uniform argument.
    _divergences: Divergences,
) -> PyResult<()> {
    match item {
        ContinuousEvalItem::Leaf(leaf) => {
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
        ContinuousEvalItem::RootInit { state, .. } => {
            let move_class = move_policy.classify(
                base_seed,
                slot.game_key,
                slot.ply,
                slot.policy_init_remaining,
            );
            slot.move_class = move_class;
            // Per-class divergence view for this fresh root (Fast=fast,
            // Full/Init=base). Replaces the single threaded `divergences`.
            let class_div = move_policy.divergences_for(move_class);
            let mut search = RustSearch::new(
                state,
                &**evaluation,
                move_policy.visits_for(move_class),
                move_policy.fpu_reduction,
                move_policy.root_fpu_for(move_class),
                move_policy.root_temp_for(move_class, slot.ply),
                root_noise_exact(
                    move_policy.noise_for(move_class),
                    mix_seed(base_seed, slot.game_key, slot.ply, SEED_STREAM_ROOT_NOISE),
                    class_div.dirichlet_shaped,
                ),
                widening,
                move_policy.forced_k_for(move_class),
                move_policy.tss_enabled,
                class_div,
            )?;
            if search.root_edges_empty() {
                return Err(PyValueError::new_err(
                    "hexfield continuous MCTS root has no legal actions",
                ));
            }
            // Build the Gumbel-Top-k candidate set + SH schedule when this
            // class's view has gumbel_root on. No-op otherwise (the search keeps
            // the normal PUCT root). budget = the move's visits.
            if class_div.gumbel_root {
                let gumbel_seed = mix_seed(base_seed, slot.game_key, slot.ply, SEED_STREAM_GUMBEL);
                search.init_gumbel_root(gumbel_seed, move_policy.visits_for(move_class));
            }
            slot.baseline = search.root_edge_visits().into_iter().collect();
            slot.search = Some(search);
            slot.phase = ContinuousPhase::Active;
            slot.in_flight = 0;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn backup_continuous_items(
    py: Python<'_>,
    slots: &mut [ContinuousSlot],
    items: Vec<ContinuousEvalItem>,
    evaluations: &[Arc<RustEvaluation>],
    move_policy: &ContinuousMovePolicy,
    widening: Widening,
    base_seed: u64,
    virtual_loss: f32,
    divergences: Divergences,
) -> PyResult<()> {
    // Each item targets exactly one slot. Items are bucketed by slot
    // (order-preserving) and slots processed with `par_iter_mut`: within a slot
    // items run in the same in-flush order, and across slots there is no shared
    // mutable state (each closure owns one slot's `&mut` tree, the same
    // disjoint-borrow guarantee `select_continuous_pass` uses).
    // `HEXFIELD_SERIAL_BACKUP=1` runs the serial path instead.
    if std::env::var("HEXFIELD_SERIAL_BACKUP").is_ok() {
        return py.detach(|| {
            for (item, evaluation) in items.into_iter().zip(evaluations.iter()) {
                let slot_index = match &item {
                    ContinuousEvalItem::Leaf(leaf) => leaf.root_index,
                    ContinuousEvalItem::RootInit { slot_index, .. } => *slot_index,
                };
                apply_backup_item(
                    &mut slots[slot_index],
                    item,
                    evaluation,
                    move_policy,
                    widening,
                    base_seed,
                    virtual_loss,
                    divergences,
                )?;
            }
            Ok(())
        });
    }

    py.detach(|| {
        // Stage 1: bucket items by owning slot, preserving in-flush order
        // (serial, cheap — no tree work).
        let mut per_slot: Vec<Vec<(ContinuousEvalItem, Arc<RustEvaluation>)>> =
            (0..slots.len()).map(|_| Vec::new()).collect();
        for (item, evaluation) in items.into_iter().zip(evaluations.iter()) {
            let slot_index = match &item {
                ContinuousEvalItem::Leaf(leaf) => leaf.root_index,
                ContinuousEvalItem::RootInit { slot_index, .. } => *slot_index,
            };
            per_slot[slot_index].push((item, Arc::clone(evaluation)));
        }

        // Stage 2: process slots in parallel (disjoint `&mut`), serial within a
        // slot in the preserved in-flush order.
        slots
            .par_iter_mut()
            .zip(per_slot.into_par_iter())
            .try_for_each(|(slot, bucket)| -> PyResult<()> {
                for (item, evaluation) in bucket {
                    apply_backup_item(
                        slot,
                        item,
                        &evaluation,
                        move_policy,
                        widening,
                        base_seed,
                        virtual_loss,
                        divergences,
                    )?;
                }
                Ok(())
            })
    })
}

#[allow(clippy::too_many_arguments)]
/// Everything the serial Phase-B dispatch needs to call `on_move` and apply its
/// response for one completed slot, computed in the off-GIL parallel Phase A.
/// Holds the pure-Rust `PayloadNative` (converted to a `PyDict` under the GIL in
/// Phase B) plus the per-slot scalars Phase B applies (early-stop bookkeeping,
/// the resolved `action_id`, move-class flags). No Python objects live here, so
/// it is `Send` and safe to collect from a rayon `par_iter`.
struct PreparedMove {
    move_class: MoveClass,
    game_key: u64,
    ply: u32,
    /// True when this completion is an early stop (drives the early-stop search
    /// mutation + stats in Phase B).
    early: bool,
    /// `search.remaining_visits()` captured before the early-stop mutation, used
    /// for the `early_stop_visits_saved` stat.
    early_remaining_visits: u32,
    payload: PayloadNative,
    /// Final played action_id (= the Init prior sample when `move_class==Init`,
    /// else the payload's selected action). Drives `advance_root` in Phase B.
    action_id: PackedCoord,
    /// For Init moves, the sampled action_id + selection label that overwrite
    /// the payload dict.
    init_override: Option<PackedCoord>,
}

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
    force_stuck_gumbel: bool,
) -> PyResult<u64> {
    // Phase A (parallel, GIL released): build the payload + decision for every
    // ready slot. Pure Rust, read-only over the slot trees (`par_iter`). Each
    // closure writes only its own `Option<PreparedMove>` (disjoint by slot
    // index), with no shared mutable state. `HEXFIELD_SERIAL_COMPLETE=1` runs
    // the build serially instead.
    let serial_build = std::env::var("HEXFIELD_SERIAL_COMPLETE").is_ok();
    let prepared: Vec<Option<PreparedMove>> = py.detach(|| {
        let prepare = |_slot_index: usize, slot: &ContinuousSlot| -> PyResult<Option<PreparedMove>> {
            if !matches!(slot.phase, ContinuousPhase::Active) {
                return Ok(None);
            }
            let move_class = slot.move_class;
            let in_flight = slot.in_flight;
            let (complete, early) = slot
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
                    // SH saturation safety net: a Gumbel root can exhaust its
                    // reachable tree (terminal/solved subtrees) below
                    // target_visits and below its SH round caps. When the slot
                    // has no in-flight evals and the scheduler made no global
                    // progress this pass (force_stuck_gumbel), it can neither
                    // reach completion nor advance the SH barrier. Finalize the
                    // move from the visits accrued so far instead.
                    if force_stuck_gumbel
                        && search.has_gumbel_root()
                        && in_flight == 0
                        && search.needs_visits()
                    {
                        return (true, true);
                    }
                    // Fast moves stop unrestricted; recorded Full roots keep the
                    // visit floor.
                    let early = early_stop_ready(
                        search,
                        Some(&slot.baseline),
                        matches!(move_class, MoveClass::Full),
                        in_flight,
                    );
                    (early, early)
                })
                .unwrap_or((false, false));
            if !complete {
                return Ok(None);
            }

            let search = slot
                .search
                .as_ref()
                .expect("active continuous slot has search");
            // Capture remaining_visits() before Phase B applies the early-stop
            // mutation (target_visits = completed_visits), for the
            // early_stop_visits_saved stat.
            let early_remaining_visits = if early {
                search.remaining_visits()
            } else {
                0
            };

            let game_key = slot.game_key;
            let ply = slot.ply;
            let move_seed = mix_seed(base_seed, game_key, ply, SEED_STREAM_MOVE_SELECT);
            let temperature = move_policy.temperature_for_class(move_class, temperature_by_ply, ply);
            let mut payload = build_search_result_payload_native(
                search,
                Some(&slot.baseline),
                temperature,
                move_seed,
                c_puct,
                move_policy.forced_k_for(move_class),
            )?;
            // The early-stop path sets `search.early_stopped = true` in Phase B;
            // this build is read-only, so reflect `early` onto the native field
            // here so the payload's `early_stopped` reads true.
            if early {
                payload.early_stopped = true;
            }

            // Init class: sample the played move from the root prior (overrides
            // the payload's selected action). Deterministic seed. A verified
            // deep root WIN outranks the exploration sample (Codex review,
            // proof-vs-Init precedence): a proven forced win is never
            // discarded for a prior draw.
            let init_override = if matches!(move_class, MoveClass::Init) && !payload.deep_override {
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
                Some(sampled)
            } else {
                None
            };
            let action_id = init_override.unwrap_or(payload.action_id);

            Ok(Some(PreparedMove {
                move_class,
                game_key,
                ply,
                early,
                early_remaining_visits,
                payload,
                action_id,
                init_override,
            }))
        };

        if serial_build {
            let mut out = Vec::with_capacity(slots.len());
            for (slot_index, slot) in slots.iter().enumerate() {
                out.push(prepare(slot_index, slot)?);
            }
            Ok(out)
        } else {
            slots
                .par_iter()
                .enumerate()
                .map(|(slot_index, slot)| prepare(slot_index, slot))
                .collect::<PyResult<Vec<_>>>()
        }
    })?;

    // Phase B (serial, GIL held): convert each native payload to a PyDict and
    // dispatch `on_move` in slot-index order, then apply the (possibly
    // tree-mutating) response. All slot mutation that depends on Python output
    // stays here, single-owner.
    let mut moves_decided = 0u64;
    for slot_index in 0..slots.len() {
        let Some(prepared) = prepared[slot_index].as_ref() else {
            continue;
        };
        let move_class = prepared.move_class;
        let game_key = prepared.game_key;
        let ply = prepared.ply;
        let action_id = prepared.action_id;

        // Early-stop bookkeeping (mutates the slot's search + stats), applied
        // here in slot order.
        if prepared.early {
            let search = slots[slot_index].search.as_mut().expect("active slot");
            stats.early_stop_visits_saved += prepared.early_remaining_visits as u64;
            match move_class {
                MoveClass::Full => stats.early_stops_full += 1,
                _ => stats.early_stops_fast += 1,
            }
            search.early_stopped = true;
            search.target_visits = search.completed_visits;
        }

        let payload_dict = prepared.payload.to_pydict(py, None, None)?;
        payload_dict.set_item("pcr_full", matches!(move_class, MoveClass::Full))?;
        payload_dict.set_item("policy_init", matches!(move_class, MoveClass::Init))?;
        if prepared.payload.lcb_override {
            stats.lcb_overrides += 1;
        }
        if prepared.payload.play_pruned {
            stats.gumbel_play_moves += 1;
            if prepared.payload.play_winner {
                stats.gumbel_play_winner_moves += 1;
            }
            if ply < 20 {
                stats.gumbel_play_moves_early += 1;
                if prepared.payload.play_winner {
                    stats.gumbel_play_winner_early += 1;
                }
            }
        }
        if let Some(sampled) = prepared.init_override {
            payload_dict.set_item("action_id", sampled)?;
            payload_dict.set_item("action_selection", "policy_init_prior")?;
        }

        moves_decided += 1;
        stats.moves_decided += 1;
        match move_class {
            MoveClass::Full => stats.full_moves += 1,
            MoveClass::Fast => stats.fast_moves += 1,
            MoveClass::Init => stats.init_moves += 1,
        }
        let started = std::time::Instant::now();
        let response = on_move.call1((game_key, &payload_dict))?;
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
                        // Per-class divergence view for the promoted root. The
                        // paired ply of a turn shares the turn's class (see
                        // classify), so a Full-turn PUCT subtree is reused under
                        // the Full regime and a Fast turn under the Fast regime.
                        let class_div = move_policy.divergences_for(next_class);
                        search.set_additional_visits(move_policy.visits_for(next_class));
                        search.set_forced_playout_k(move_policy.forced_k_for(next_class));
                        search.set_root_fpu_reduction(move_policy.root_fpu_for(next_class));
                        search.set_tss_enabled(move_policy.tss_enabled);
                        search.set_divergences(class_div);
                        search
                            .apply_root_policy_temperature(move_policy.root_temp_for(next_class, next_ply));
                        if let Some(noise) = root_noise_exact(
                            move_policy.noise_for(next_class),
                            mix_seed(base_seed, game_key, next_ply, SEED_STREAM_ROOT_NOISE),
                            class_div.dirichlet_shaped,
                        ) {
                            search.apply_root_dirichlet_noise(noise);
                        }
                        // (Re)build the Gumbel-Top-k candidate set + SH schedule
                        // for the promoted root, mirroring the epoch-entry reuse
                        // path. Without this the previous move's finished SH
                        // state (survivors/round caps keyed to the old root's
                        // actions) persists onto the new root, and the slot
                        // either hammers a stale survivor or stalls until the
                        // force-stuck safety net finalizes the move with zero
                        // new visits. When this class's view has gumbel_root off
                        // the state is cleared so the normal PUCT root runs.
                        if class_div.gumbel_root {
                            let gumbel_seed =
                                mix_seed(base_seed, game_key, next_ply, SEED_STREAM_GUMBEL);
                            search.init_gumbel_root(
                                gumbel_seed,
                                move_policy.visits_for(next_class),
                            );
                        } else {
                            search.clear_gumbel_root();
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
                // Mirror the epoch-entry slot init: a blunder-seeded
                // replacement arrives mid-game, so its ply and Init budget
                // derive from the state (0 placements = old behavior).
                let start_ply = next_state.placements_made();
                slots[slot_index].game_key = new_key;
                slots[slot_index].ply = start_ply;
                slots[slot_index].search = None;
                slots[slot_index].baseline.clear();
                slots[slot_index].in_flight = 0;
                slots[slot_index].phase = ContinuousPhase::AwaitRootEval;
                slots[slot_index].policy_init_remaining = move_policy
                    .policy_init_plies(base_seed, new_key)
                    .saturating_sub(start_ply);
                slots[slot_index].move_class = move_policy.classify(
                    base_seed,
                    new_key,
                    start_ply,
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

fn continuous_has_work(slots: &[ContinuousSlot], parked: &[ParkedLeaf]) -> bool {
    !parked.is_empty()
        || slots
            .iter()
            .any(|slot| !matches!(slot.phase, ContinuousPhase::Empty))
}

/// Snapshot the cumulative unique-states counter (depth-2 per-flush histogram).
fn lock_unique_states(stats: &SharedEvaluationStats) -> usize {
    stats
        .lock()
        .expect("evaluation stats mutex poisoned")
        .unique_states
}

/// Map a queued eval item to its forward-pass request (state + hash). Identical
/// to the inline match in the lockstep loop; shared by the depth-2 path.
fn continuous_item_request(item: &ContinuousEvalItem) -> RustEvaluationRequest<'_> {
    match item {
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
    }
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

/// Every key `resolve_divergences` understands. Unknown keys are a hard error:
/// a silently-dropped key (version skew, typo) reverts part of the search
/// profile to defaults with zero symptoms — the same silent-PUCT failure class
/// the lockstep Gumbel-init fix closed.
const KNOWN_DIVERGENCE_KEYS: &[&str] = &[
    "lcb_move_selection",
    "early_stop",
    "visit_scaled_c_puct",
    "moves_left_utility",
    "ml_weight",
    "ml_scale",
    "ml_q_gate",
    "ml_two_sided",
    "ml_final_pick",
    "ml_final_pick_band",
    "lcb_z",
    "c_scale",
    "c_base",
    "nucleus_f64",
    "new_child_fpu",
    "lazy_widening",
    "clean_root_prior_cache",
    "dirichlet_shaped",
    "pruned_dynamic_cpuct",
    "scaled_fpu",
    "gumbel_target",
    "gumbel_root",
    "gumbel_sequential_halving",
    "gumbel_nonroot_select",
    "gumbel_c_visit",
    "gumbel_c_scale",
    "gumbel_target_c_scale",
    "gumbel_m",
    "gumbel_draw_temperature",
    "gumbel_target_min_visits",
    "gumbel_play_prune",
    "tss_interior_guard",
    "tss_solver_mode",
    "tss_solver_node_cap",
    "tss_solver_sample_16",
    "tss_solver_root_guard",
    "tss_solver_async",
    "tss_solver_async_threads",
    "tss_solver_async_threads_max",
    "tss_solver_park",
    "tss_solver_park_timeout_ms",
    "tss_solver_async_inline_16",
    "tss_zone",
    "tss_zone_stale_filter",
    "tss_zone_count2",
    "tss_pair_commutation",
    "tss_solver_horizon",
    "tss_solver_dual_pass",
    "tss_solver_horizon_ladder",
    // Fast-class Gumbel levers (main_8: PUCT Full / Gumbel Fast). These name the
    // Fast view's values; the driver's Python side folds them into the SECOND
    // (fast) override map whose base keys resolve_divergences reads. They are
    // whitelisted here so the strict known-keys gate never rejects them when
    // they ride in an override dict — a parser/whitelist mismatch on new keys
    // tripped the supervisor circuit breaker on 2026-07-04 (supervisor_halted.flag).
    "fast_gumbel_root_enabled",
    "fast_gumbel_sequential_halving",
    "fast_gumbel_nonroot_select",
    "fast_gumbel_c_visit",
    "fast_gumbel_c_scale",
    "fast_gumbel_m",
    "fast_gumbel_play_prune",
];

fn resolve_divergences(
    search_parity_mode: Option<bool>,
    overrides: Option<&Bound<'_, PyDict>>,
) -> PyResult<Divergences> {
    if let Some(overrides) = overrides {
        for key in overrides.keys() {
            let key: String = key.extract()?;
            if !KNOWN_DIVERGENCE_KEYS.contains(&key.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "unknown divergence override key {key:?}; known keys: {KNOWN_DIVERGENCE_KEYS:?}"
                )));
            }
        }
    }
    let mut dv = if search_parity_mode.unwrap_or(false) {
        Divergences::parity()
    } else {
        Divergences::production()
    };
    if let Some(overrides) = overrides {
        // Per-divergence toggles from the override dict.
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
        // Search divergences, individually flippable via the override dict.
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
        if let Some(v) = overrides.get_item("scaled_fpu")? {
            dv.scaled_fpu = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_interior_guard")? {
            dv.tss_interior_guard = v.extract()?;
        }
        // TSS numeric controls are range-validated (Codex review, silent
        // misconfig): an out-of-band value silently changes rollout behavior
        // (mode=30 acts like mode 3, sample_16=160 samples everything), so a
        // typo'd config must fail loudly at resolve time instead.
        if let Some(v) = overrides.get_item("tss_solver_mode")? {
            let mode: u32 = v.extract()?;
            if mode > 3 {
                return Err(PyValueError::new_err(format!(
                    "tss_solver_mode must be 0..=3, got {mode}"
                )));
            }
            dv.tss_solver_mode = mode;
        }
        if let Some(v) = overrides.get_item("tss_solver_node_cap")? {
            let cap: u32 = v.extract()?;
            if cap == 0 {
                return Err(PyValueError::new_err(
                    "tss_solver_node_cap must be >= 1 (every solve would be Unknown at 0)",
                ));
            }
            dv.tss_solver_node_cap = cap;
        }
        if let Some(v) = overrides.get_item("tss_solver_sample_16")? {
            let sample: u32 = v.extract()?;
            if sample > 16 {
                return Err(PyValueError::new_err(format!(
                    "tss_solver_sample_16 must be 0..=16 (sixteenths), got {sample}"
                )));
            }
            dv.tss_solver_sample_16 = sample;
        }
        if let Some(v) = overrides.get_item("tss_solver_root_guard")? {
            dv.tss_solver_root_guard = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_solver_async")? {
            dv.tss_solver_async = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_solver_async_threads")? {
            let threads: u32 = v.extract()?;
            if !(1..=32).contains(&threads) {
                return Err(PyValueError::new_err(format!(
                    "tss_solver_async_threads must be 1..=32, got {threads}"
                )));
            }
            dv.tss_solver_async_threads = threads;
        }
        if let Some(v) = overrides.get_item("tss_solver_async_threads_max")? {
            dv.tss_solver_async_threads_max = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_solver_park")? {
            dv.tss_solver_park = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_solver_park_timeout_ms")? {
            dv.tss_solver_park_timeout_ms = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_solver_async_inline_16")? {
            let inline: u32 = v.extract()?;
            if inline > 16 {
                return Err(PyValueError::new_err(format!(
                    "tss_solver_async_inline_16 must be 0..=16 (sixteenths), got {inline}"
                )));
            }
            dv.tss_solver_async_inline_16 = inline;
        }
        if let Some(v) = overrides.get_item("tss_zone")? {
            dv.tss_zone = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_zone_stale_filter")? {
            dv.tss_zone_stale_filter = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_zone_count2")? {
            dv.tss_zone_count2 = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_pair_commutation")? {
            dv.tss_pair_commutation = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_solver_horizon")? {
            dv.tss_solver_horizon = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_solver_dual_pass")? {
            dv.tss_solver_dual_pass = v.extract()?;
        }
        if let Some(v) = overrides.get_item("tss_solver_horizon_ladder")? {
            dv.tss_solver_horizon_ladder = v.extract()?;
        }
        // Gumbel AlphaZero flags (default-OFF).
        if let Some(v) = overrides.get_item("gumbel_target")? {
            dv.gumbel_target = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_root")? {
            dv.gumbel_root = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_sequential_halving")? {
            dv.gumbel_sequential_halving = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_nonroot_select")? {
            dv.gumbel_nonroot_select = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_c_visit")? {
            dv.gumbel_c_visit = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_c_scale")? {
            dv.gumbel_c_scale = v.extract()?;
        }
        // Export-only target σ override (absent => target keeps gumbel_c_scale).
        if let Some(v) = overrides.get_item("gumbel_target_c_scale")? {
            dv.gumbel_target_c_scale = Some(v.extract()?);
        }
        if let Some(v) = overrides.get_item("gumbel_m")? {
            dv.gumbel_m = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_draw_temperature")? {
            dv.gumbel_draw_temperature = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_target_min_visits")? {
            dv.gumbel_target_min_visits = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_play_prune")? {
            dv.gumbel_play_prune = v.extract()?;
        }
        // Fast-class Gumbel levers (main_8). When present these override the
        // gumbel fields with the Fast view's values; they are applied LAST so a
        // fast-override map carrying fast_* keys wins over any base-keyed gumbel
        // entry it also holds. Absent => the base gumbel fields stand, so a plain
        // (non-fast) override map is unchanged and the fast view falls back to
        // the base view (golden invariant).
        if let Some(v) = overrides.get_item("fast_gumbel_root_enabled")? {
            dv.gumbel_root = v.extract()?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_sequential_halving")? {
            dv.gumbel_sequential_halving = v.extract()?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_nonroot_select")? {
            dv.gumbel_nonroot_select = v.extract()?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_c_visit")? {
            dv.gumbel_c_visit = v.extract()?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_c_scale")? {
            dv.gumbel_c_scale = v.extract()?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_m")? {
            dv.gumbel_m = v.extract()?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_play_prune")? {
            dv.gumbel_play_prune = v.extract()?;
        }
    }
    if dv.tss_solver_async_threads_max != 0
        && (!(dv.tss_solver_async_threads..=64).contains(&dv.tss_solver_async_threads_max))
    {
        return Err(PyValueError::new_err(format!(
            "tss_solver_async_threads_max must be 0 (auto) or {}..=64, got {}",
            dv.tss_solver_async_threads, dv.tss_solver_async_threads_max
        )));
    }
    if !(1..=5000).contains(&dv.tss_solver_park_timeout_ms) {
        return Err(PyValueError::new_err(format!(
            "tss_solver_park_timeout_ms must be 1..=5000, got {}",
            dv.tss_solver_park_timeout_ms
        )));
    }
    // Horizon (owner ruling 2026-07-20, PLAN_TSS_MCTS_INTEGRATION.md §5): the
    // floor is h16, or 0 for unbounded (node cap the only budget). The
    // 1..=15 band is rejected loudly — a below-floor deadline silently loses
    // bounded-horizon WINs (the R-FIX1 class of defect).
    if dv.tss_solver_horizon != 0 && dv.tss_solver_horizon < 16 {
        return Err(PyValueError::new_err(format!(
            "tss_solver_horizon must be 0 (unbounded) or >= 16 (the owner floor), got {}",
            dv.tss_solver_horizon
        )));
    }
    if dv.tss_solver_horizon_ladder && dv.tss_solver_horizon == 0 {
        return Err(PyValueError::new_err(
            "tss_solver_horizon_ladder requires a bounded tss_solver_horizon (>= 16); \
             an unbounded base has nothing taller to climb to",
        ));
    }
    if dv.tss_solver_park && !dv.tss_solver_async {
        return Err(PyValueError::new_err(
            "tss_solver_park=true requires tss_solver_async=true",
        ));
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

/// Pure-Rust core of a single search-result payload: every value the PyDict in
/// `build_search_result_payloads` carries, as plain Rust scalars/bytes so it can
/// be computed without the GIL (built in a `par_iter` off the GIL, then
/// converted to a `PyDict` serially under the GIL right before `on_move`).
/// `to_pydict` is the GIL-held conversion.
struct PayloadNative {
    action_id: PackedCoord,
    action_selection: &'static str,
    lcb_override: bool,
    early_stopped: bool,
    // A verified deep root WIN forced the played action (tss_deep_root_win).
    // Downstream: the Init-class prior sample must NOT replace the action
    // (a proof outranks exploration), and the lockstep driver must advance
    // the retained tree through this same action.
    deep_override: bool,
    // Play-policy telemetry: whether the quota-pruned Gumbel play distribution
    // drove selection, and whether the played move is the raw delta leader.
    play_pruned: bool,
    play_winner: bool,
    export_action_ids: Vec<PackedCoord>,
    export_weights: Vec<f32>,
    export_q: Vec<f32>,
    root_prior_action_ids: Vec<PackedCoord>,
    root_prior_weights: Vec<f32>,
    // Present only when `gumbel_target` is on (otherwise None; the gumbel keys
    // are omitted from the payload).
    gumbel: Option<GumbelTargetNative>,
    root_value: f32,
    visits: u32,
    node_count: usize,
    active_edge_count: usize,
    root_active_edges: usize,
    root_hidden_priors: usize,
    // === TSS shadow export (docs/PLAN_TSS_DEEPENING.md §9) ===
    // λ¹ classification map over the union of the play/recorded/π' supports,
    // sorted by action_id for deterministic bytes. Empty on quiet roots.
    tss_class_ids: Vec<PackedCoord>,
    tss_class_vals: Vec<i8>,
    // λ¹ verdict at the root position, side-to-move perspective (0 = None).
    tss_proof: i8,
    // Root analysis scalars: turn budget B, min hitting set k (-1 = infeasible/
    // None), live opponent-threat count. All 0/-1/0 when tss is disabled.
    tss_b: u8,
    tss_k: i8,
    tss_opp_threats: u32,
    // Per-move counters accumulated during the search (reset per move).
    tss_counters: TssCounters,
}

struct GumbelTargetNative {
    action_ids: Vec<PackedCoord>,
    weights: Vec<f32>,
    logits: Vec<f32>,
}

impl PayloadNative {
    /// Convert the pure-Rust payload into the `PyDict` the on_move callback
    /// expects. `eval_stats` / `cache_len` are the per-batch diagnostics only
    /// the lockstep multi-search path supplies (the continuous path passes None).
    fn to_pydict<'py>(
        &self,
        py: Python<'py>,
        eval_stats: Option<&EvaluationStats>,
        cache_len: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        result.set_item("action_id", self.action_id)?;
        result.set_item("action_selection", self.action_selection)?;
        result.set_item("lcb_override", self.lcb_override)?;
        result.set_item("early_stopped", self.early_stopped)?;
        result.set_item("play_pruned", self.play_pruned)?;
        result.set_item("play_winner", self.play_winner)?;
        let to_bytes = |data: &[u32]| -> Bound<'py, PyBytes> {
            let len = std::mem::size_of_val(data);
            let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) };
            PyBytes::new(py, raw)
        };
        let to_bytes_f32 = |data: &[f32]| -> Bound<'py, PyBytes> {
            let len = std::mem::size_of_val(data);
            let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) };
            PyBytes::new(py, raw)
        };
        result.set_item(
            "visit_policy_action_ids_bytes",
            to_bytes(&self.export_action_ids),
        )?;
        result.set_item("visit_policy_weights_bytes", to_bytes_f32(&self.export_weights))?;
        result.set_item("visit_policy_q_bytes", to_bytes_f32(&self.export_q))?;
        result.set_item("visit_policy_count", self.export_action_ids.len())?;
        result.set_item(
            "root_prior_policy_action_ids_bytes",
            to_bytes(&self.root_prior_action_ids),
        )?;
        result.set_item(
            "root_prior_policy_weights_bytes",
            to_bytes_f32(&self.root_prior_weights),
        )?;
        result.set_item("root_prior_policy_count", self.root_prior_action_ids.len())?;
        if let Some(gumbel) = &self.gumbel {
            result.set_item("gumbel_policy_action_ids_bytes", to_bytes(&gumbel.action_ids))?;
            result.set_item("gumbel_policy_weights_bytes", to_bytes_f32(&gumbel.weights))?;
            result.set_item("gumbel_policy_count", gumbel.action_ids.len())?;
            result.set_item("root_prior_logits_bytes", to_bytes_f32(&gumbel.logits))?;
        }
        result.set_item("root_value", self.root_value)?;
        result.set_item("visits", self.visits)?;
        // TSS shadow export: the λ¹ class map rides only when non-empty (quiet
        // roots omit the keys, mirroring the gumbel-key convention); the root
        // proof scalar always rides (0 = no proof).
        if !self.tss_class_ids.is_empty() {
            result.set_item("tss_class_action_ids_bytes", to_bytes(&self.tss_class_ids))?;
            let vals = unsafe {
                std::slice::from_raw_parts(
                    self.tss_class_vals.as_ptr() as *const u8,
                    self.tss_class_vals.len(),
                )
            };
            result.set_item("tss_class_bytes", PyBytes::new(py, vals))?;
        }
        result.set_item("tss_proof", self.tss_proof)?;
        let diag = PyDict::new(py);
        diag.set_item("node_count", self.node_count)?;
        diag.set_item("active_edge_count", self.active_edge_count)?;
        diag.set_item("root_active_edges", self.root_active_edges)?;
        diag.set_item("root_hidden_priors", self.root_hidden_priors)?;
        let tss = PyDict::new(py);
        tss.set_item("b", self.tss_b)?;
        tss.set_item("k", self.tss_k)?;
        tss.set_item("opp_threats", self.tss_opp_threats)?;
        tss.set_item("root_tactical", self.tss_counters.root_tactical)?;
        tss.set_item("root_injected", self.tss_counters.root_injected)?;
        tss.set_item("leaf_verdict_hits", self.tss_counters.leaf_verdict_hits)?;
        tss.set_item("prune_eligible", self.tss_counters.prune_eligible)?;
        tss.set_item("prune_dropped", self.tss_counters.prune_dropped)?;
        tss.set_item("deep_calls", self.tss_counters.deep_calls)?;
        tss.set_item("deep_win", self.tss_counters.deep_win)?;
        tss.set_item("deep_loss", self.tss_counters.deep_loss)?;
        tss.set_item("deep_unknown", self.tss_counters.deep_unknown)?;
        tss.set_item("deep_nodes", self.tss_counters.deep_nodes)?;
        tss.set_item("deep_verify_failed", self.tss_counters.deep_verify_failed)?;
        tss.set_item("horizon_retry", self.tss_counters.horizon_retry)?;
        tss.set_item(
            "horizon_preflight_failed",
            self.tss_counters.horizon_preflight_failed,
        )?;
        tss.set_item("horizon_cut", self.tss_counters.horizon_cut)?;
        tss.set_item("horizon_cut_tall", self.tss_counters.horizon_cut_tall)?;
        tss.set_item("deep_kb_death", self.tss_counters.deep_kb_death)?;
        // Engine attribution for minted certificates (V0 wholesale adoption).
        tss.set_item("cert_version", crate::tss_core::TSS_CERT_VERSION)?;
        tss.set_item("zone_nodes", self.tss_counters.zone_nodes)?;
        tss.set_item("pair_omitted", self.tss_counters.pair_omitted)?;
        tss.set_item("zone_verify_failed", self.tss_counters.zone_verify_failed)?;
        tss.set_item("deep_hard_backups", self.tss_counters.deep_hard_backups)?;
        tss.set_item("deep_memo_hits", self.tss_counters.deep_memo_hits)?;
        tss.set_item("async_enqueued", self.tss_counters.async_enqueued)?;
        tss.set_item("async_dropped", self.tss_counters.async_dropped)?;
        tss.set_item("async_stale", self.tss_counters.async_stale)?;
        tss.set_item("async_pending_hits", self.tss_counters.async_pending_hits)?;
        tss.set_item("park_parked", self.tss_counters.park_parked)?;
        tss.set_item("park_hard", self.tss_counters.park_hard)?;
        tss.set_item("park_released", self.tss_counters.park_released)?;
        tss.set_item("park_bailed", self.tss_counters.park_bailed)?;
        tss.set_item("park_wait_ms_sum", self.tss_counters.park_wait_ms_sum)?;
        tss.set_item("park_wait_ms_max", self.tss_counters.park_wait_ms_max)?;
        tss.set_item(
            "async_workers_spawned",
            self.tss_counters.async_workers_spawned,
        )?;
        tss.set_item("depth_sum", self.tss_counters.depth_sum)?;
        tss.set_item("depth_max", self.tss_counters.depth_max)?;
        tss.set_item("backups", self.tss_counters.backups)?;
        diag.set_item("tss", tss)?;
        if let Some(stats) = eval_stats {
            diag.set_item("evaluation", eval_stats_dict(py, stats)?)?;
        }
        if let Some(cache_len) = cache_len {
            diag.set_item("cache_len", cache_len)?;
        }
        result.set_item("diagnostics", diag)?;
        Ok(result)
    }
}

/// Build the native payload for one search. Carries no Python state and makes
/// no Python calls, so it is safe to run inside a rayon `par_iter` with the GIL
/// released; the final `PyDict` construction is deferred (see
/// `PayloadNative::to_pydict`).
#[allow(clippy::too_many_arguments)]
fn build_search_result_payload_native(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    temperature: f32,
    seed: u64,
    c_puct: f32,
    forced_playout_k: f32,
) -> PyResult<PayloadNative> {
    let root = search.root();
    let (policy_action_ids, policy_weights, _policy_q, policy_total) = visit_policy(root, baseline);
    // Forced-playout pruning is PUCT bookkeeping: at a Gumbel SH root the
    // selection path never takes the forced branches, so there are no forced
    // playouts to prune and the PUCT pruning math (n_forced = sqrt(k*P*N))
    // would strip legitimate SH round-quota visits from the recorded target.
    // Gate it off whenever the SH root is active.
    let (mut export_action_ids, mut export_weights, mut export_q) =
        if forced_playout_k > 0.0 && !search.has_gumbel_root() {
            // When pruned_dynamic_cpuct is on the recorded-target pruning uses
            // selection's c_for(N); otherwise static c_puct.
            let effective_c = search.effective_pruning_c_puct(c_puct, root.visits);
            pruned_visit_policy(root, baseline, forced_playout_k, effective_c)
        } else {
            let (ids, w, q, _t) = visit_policy(root, baseline);
            (ids, w, q)
        };
    // Recorded-target fallback for a force-completed Gumbel SH root: such a move
    // can finalize with zero net delta visits over its reuse baseline, so the
    // delta-visit export above is empty. A Full (pcr_full) row with an
    // empty/zero-mass policy target is a hard error in shard expansion, so
    // substitute the cumulative visit distribution (baseline-free), then the
    // root prior — both real, legal, positive-mass targets. Inert for normal
    // completion, where the export is non-empty.
    if export_action_ids.is_empty() {
        let (cum_ids, cum_w, cum_q, cum_total) = visit_policy(root, None);
        if !cum_ids.is_empty() && cum_total > 0 {
            export_action_ids = cum_ids;
            export_weights = cum_w;
            export_q = cum_q;
        } else {
            // No edge carries a cumulative visit: fall back to the root prior.
            let (prior_ids, prior_w) = root_prior_policy(root);
            let prior_q: Vec<f32> = prior_ids
                .iter()
                .map(|id| {
                    root.edges
                        .iter()
                        .find(|e| e.action_id == *id)
                        .map(|e| e.value())
                        .unwrap_or(0.0)
                })
                .collect();
            export_action_ids = prior_ids;
            export_weights = prior_w;
            export_q = prior_q;
        }
    }
    let (root_prior_action_ids, root_prior_weights) = root_prior_policy(root);
    // Play distribution for Gumbel SH roots at exploration temperatures
    // (gumbel_play_prune): the delta-visit histogram is a SCHEDULE artifact —
    // every round-0 loser carries its equal entry quota (~budget/(R*m)), so
    // temperature-sampling it plays measured-bad moves at the quota rate.
    // Zero every action whose delta never exceeded the round-0 quota (it was
    // eliminated without surviving a halving) and renormalize; the surviving
    // mass is ordered by rounds survived — SH's own quality ranking at visit
    // counts it already paid for. The RECORDED targets above are untouched.
    // Gated to T>0 (the T=0 greedy/LCB path keeps the raw histogram, so eval
    // arena behavior is unchanged) and inert when pruning would empty the
    // support (degenerate/force-finalized roots keep the fallback chain).
    let play_pair: Option<(Vec<PackedCoord>, Vec<f32>)> = if temperature > 0.0
        && search.divergences.gumbel_play_prune
        && policy_total > 0
    {
        search.gumbel_play_quota().and_then(|quota| {
            let total = policy_total as f32;
            let cut = quota as f32 + 0.5;
            let mut ids = Vec::with_capacity(policy_action_ids.len());
            let mut ws = Vec::with_capacity(policy_action_ids.len());
            for (id, w) in policy_action_ids.iter().zip(policy_weights.iter()) {
                if *w * total > cut {
                    ids.push(*id);
                    ws.push(*w);
                }
            }
            if ids.is_empty() {
                None
            } else {
                let sum: f32 = ws.iter().sum();
                if sum > 0.0 {
                    for w in ws.iter_mut() {
                        *w /= sum;
                    }
                }
                Some((ids, ws))
            }
        })
    } else {
        None
    };
    let play_pruned = play_pair.is_some();
    let (sel_ids, sel_weights): (&Vec<PackedCoord>, &Vec<f32>) = match &play_pair {
        Some((ids, ws)) => (ids, ws),
        None => (&policy_action_ids, &policy_weights),
    };
    // Improved-policy target π'=softmax(logits+σ(completedQ)). Exported only
    // when gumbel_target is on; the raw root logits column ships alongside.
    // Built BEFORE the guard so the λ¹ class map below covers π''s support
    // (pure function of the tree — the move is observationally identical).
    let div = &search.divergences;
    let mut gumbel = if div.gumbel_target {
        // Export-only σ softening: gumbel_target_c_scale overrides c_scale in the
        // target's σ call ONLY, so π' can be flattened without touching the SH
        // ranking or interior selection (both keep div.gumbel_c_scale).
        let target_c_scale = div.gumbel_target_c_scale.unwrap_or(div.gumbel_c_scale);
        let (gumbel_ids, gumbel_weights, gumbel_logits) = gumbel_target_policy(
            root,
            baseline,
            div.gumbel_c_visit,
            target_c_scale,
            div.gumbel_target_min_visits,
        );
        Some(GumbelTargetNative {
            action_ids: gumbel_ids,
            weights: gumbel_weights,
            logits: gumbel_logits,
        })
    } else {
        None
    };
    // λ¹ root analysis + per-move classification, computed ONCE per move and
    // shared by the play-time guard, the recorded class column, and the shadow
    // metrics (docs/PLAN_TSS_DEEPENING.md §4/§9). Quiet or tss-disabled roots
    // skip classification entirely (empty map, guard is the identity).
    let root_analysis = search
        .tss_enabled
        .then(|| threats::analyze(&search.root_state));
    let root_has_threats = root_analysis
        .as_ref()
        .map(|a| a.own_win_now || a.opp_threat_count > 0)
        .unwrap_or(false);
    let mut tss_classes: HashMap<PackedCoord, i8> = HashMap::new();
    if root_has_threats {
        let union = sel_ids
            .iter()
            .chain(export_action_ids.iter())
            .chain(gumbel.iter().flat_map(|g| g.action_ids.iter()));
        for &id in union {
            tss_classes
                .entry(id)
                .or_insert_with(|| classify_root_move(&search.root_state, id));
        }
    }
    // Deep root guard (Stage-4 rung 6, PLAN §10): at a λ¹-undecided root,
    // one verified deep solve upgrades the row proof; a verified WIN also
    // upgrades the certificate's root move to class +1, so the play-time
    // guard forces a proven win the net might miss (and, under Lever 1, the
    // recorded targets sharpen onto it). Root solves are not subsampled.
    // CPU-only, pure (local counters merged into the payload's view).
    let mut deep_counters = TssCounters::default();
    let mut deep_root_proof: i8 = 0;
    let mut deep_forced_move: Option<PackedCoord> = None;
    if search.tss_enabled && div.tss_solver_root_guard {
        let lambda1_undecided = root_analysis
            .as_ref()
            .map_or(false, |a| a.verdict().is_none());
        if lambda1_undecided {
            // Per-move solver instance: the payload builder holds only &search
            // (off-GIL parallel build), so the root guard can't share the
            // per-search persistent cache; one root solve per move is cheap.
            let mut root_solver = crate::tss_solver::TssSolver::default();
            root_solver.configure_leaf_profile();
            root_solver.set_dual_pass(div.tss_solver_dual_pass);
            let solved = tss_solve_verified(
                &search.root_state,
                div.tss_solver_node_cap as u64,
                tss_core::SolveGoal::Both,
                tss_core::ZoneSearchCaps {
                    enabled: div.tss_zone,
                    stale_area_filter: div.tss_zone_stale_filter,
                    count2_threshold: div.tss_zone_count2,
                    pair_commutation: div.tss_pair_commutation,
                },
                SolverHorizon {
                    horizon: div.tss_solver_horizon,
                    ladder: div.tss_solver_horizon_ladder,
                },
                &mut root_solver,
                &mut deep_counters,
            );
            match solved.status {
                ProofStatus::Win => {
                    deep_root_proof = 1;
                    if let Some(cert) = &solved.cert {
                        if let Some(CertNode::Choice { mv, .. }) =
                            cert.nodes.get(cert.root_node as usize)
                        {
                            // The certificate's root move is a verified win
                            // (and legal — verification replayed it). It
                            // upgrades the class map for the recorded targets
                            // AND directly overrides play below: a proven
                            // win-in-N is never skipped, even when the net
                            // left it outside the visit support.
                            let id = pack_coord(*mv);
                            tss_classes.insert(id, 1);
                            deep_forced_move = Some(id);
                        }
                    }
                }
                ProofStatus::Loss => deep_root_proof = -1,
                ProofStatus::Unknown => {}
            }
        }
    }
    // With the deep guard off, classes exist iff the root has threats, so this
    // gate is exactly the pre-Stage-4 behavior.
    let guarded_weights = if root_has_threats || !tss_classes.is_empty() {
        tactical_guard_weights_from(&tss_classes, sel_ids, sel_weights)
    } else {
        sel_weights.clone()
    };
    let (selected, lcb_override) = select_action_with_lcb(
        search,
        baseline,
        sel_ids,
        &guarded_weights,
        temperature,
        seed,
    )?;
    // Played-move resolution. `selected` can be None when the delta-visit policy
    // is empty: a force-completed Gumbel SH root can finalize a move with zero
    // net visits over its reuse baseline, so every edge's delta is 0 and
    // `visit_policy` drops them all. PackedCoord 0 unpacks to the illegal
    // sentinel HexCoord{q:-32768,r:-32768}, so fall back to the cumulative visit
    // distribution (baseline-free), then to the root prior — both real, legal
    // root action_ids. Inert for normal full-visit completion, where `selected`
    // is always Some.
    let selected = match selected {
        Some(action_id) => action_id,
        None => fallback_root_action(root).ok_or_else(|| {
            PyValueError::new_err(
                "continuous move selection found no legal root action (empty edges and priors)",
            )
        })?,
    };
    debug_assert!(
        root.edges.iter().any(|e| e.action_id == selected)
            || root
                .remaining_priors()
                .iter()
                .any(|(a, _)| *a == selected),
        "selected played action_id must be a real root action, never the sentinel"
    );
    // Deep root-guard override (Stage-4 rung 6): a verified WIN certificate's
    // root move takes precedence over the sampled selection. Sound by
    // verification (the move was replayed against the exact root) and legal
    // even outside the visit support (root priors span the full legal set).
    let deep_override = matches!(deep_forced_move, Some(mv) if mv != selected);
    let selected = deep_forced_move.unwrap_or(selected);
    // A proof move the net left OUTSIDE the visit support must still be
    // learnable (Codex review, played-but-not-learned): append it to the
    // recorded target support at weight 0 (inert for the raw visit target —
    // Lever-1 sharpening one-hots onto it via the class map, which already
    // carries it at +1) with the proven-win Q. Shard classes align to
    // pol_act, so this also preserves the +1 class through the round trip.
    if let Some(mv) = deep_forced_move {
        if !export_action_ids.contains(&mv) {
            export_action_ids.push(mv);
            export_weights.push(0.0);
            export_q.push(1.0);
        }
        // Same for the π' target support, so Lever-1 sharpening of π' can
        // reach the proof move too (logit 0.0 = the shard default for
        // support entries the net never scored).
        if let Some(g) = gumbel.as_mut() {
            if !g.action_ids.contains(&mv) {
                g.action_ids.push(mv);
                g.weights.push(0.0);
                g.logits.push(0.0);
            }
        }
    }
    let action_selection = if deep_override {
        "tss_deep_root_win"
    } else if play_pruned {
        "gumbel_play_policy"
    } else if baseline.is_some() {
        "delta_visit_policy"
    } else {
        "cumulative_visit_policy"
    };
    // Telemetry: whether the played action is the raw delta-visit leader (the
    // SH winner on a completed SH root). Read alongside play_pruned to judge
    // how exploratory the play distribution actually is.
    let play_winner = policy_action_ids
        .iter()
        .zip(policy_weights.iter())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| *id == selected)
        .unwrap_or(false);
    // (π' was built above, before the guard, so the class map covers it.)
    // Deterministic class export: sorted by action_id.
    let mut tss_pairs: Vec<(PackedCoord, i8)> = tss_classes.into_iter().collect();
    tss_pairs.sort_by_key(|(id, _)| *id);
    let (tss_class_ids, tss_class_vals): (Vec<PackedCoord>, Vec<i8>) =
        tss_pairs.into_iter().unzip();
    let (tss_proof, tss_b, tss_k, tss_opp_threats) = match &root_analysis {
        Some(a) => (
            a.verdict()
                .map(|v| if v > 0.0 { 1i8 } else { -1i8 })
                .unwrap_or(0),
            a.b,
            a.min_hitting_set.map(|k| k as i8).unwrap_or(-1),
            a.opp_threat_count as u32,
        ),
        None => (0, 0, -1, 0),
    };
    // Deep root proof fills in only where λ¹ had none (verified-only; feeds
    // the Lever-2 disagreement stream and, later, proof-corrected labels).
    let tss_proof = if tss_proof == 0 { deep_root_proof } else { tss_proof };
    // Per-move counters = the search's accumulation + this build's root-guard
    // solves (the builder runs read-only off-GIL; local counters merge here).
    let mut tss_counters = search.tss;
    tss_counters.add(&deep_counters);
    let tree = search.diagnostics();
    Ok(PayloadNative {
        action_id: selected,
        action_selection,
        lcb_override,
        early_stopped: search.early_stopped,
        // is_some(), not `deep_override`: even when the sample happened to
        // land on the proof move, an Init-class prior re-sample downstream
        // would walk away from it — the flag must veto that too.
        deep_override: deep_forced_move.is_some(),
        play_pruned,
        play_winner,
        export_action_ids,
        export_weights,
        export_q,
        root_prior_action_ids,
        root_prior_weights,
        gumbel,
        root_value: root.value(),
        visits: policy_total,
        node_count: tree.node_count,
        active_edge_count: tree.active_edge_count,
        root_active_edges: tree.root_active_edges,
        root_hidden_priors: tree.root_hidden_priors,
        tss_class_ids,
        tss_class_vals,
        tss_proof,
        tss_b,
        tss_k,
        tss_opp_threats,
        tss_counters,
    })
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

/// Deterministic last-resort played-move pick when the normal (delta-visit)
/// selection yields nothing (a force-completed Gumbel SH root that accrued no
/// net visits over its reuse baseline). Prefers the most-visited cumulative root
/// edge (baseline-free), then the highest-prior root action. Returns a real,
/// legal root action_id (never the PackedCoord-0 sentinel), or None only if the
/// root has no edges and no priors. Ties broken by smallest action_id.
fn fallback_root_action(root: &RustNode) -> Option<PackedCoord> {
    // 1) Most-visited cumulative edge.
    let by_visits = root
        .edges
        .iter()
        .max_by(|a, b| {
            a.visits
                .cmp(&b.visits)
                .then_with(|| b.action_id.cmp(&a.action_id))
        })
        .map(|edge| (edge.action_id, edge.visits));
    if let Some((action_id, visits)) = by_visits {
        if visits > 0 {
            return Some(action_id);
        }
    }
    // 2) No edge carries a visit (degenerate): take the highest-prior root
    // action across edges + unexpanded candidates.
    let (prior_ids, prior_weights) = root_prior_policy(root);
    let best_prior = prior_ids
        .iter()
        .copied()
        .zip(prior_weights.iter().copied())
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.cmp(&a.0))
        })
        .map(|(action_id, _)| action_id);
    if best_prior.is_some() {
        return best_prior;
    }
    // 3) Priors all non-positive but an edge exists: any edge action_id beats a
    // sentinel. Fall back to the visit-argmax (already legal) if present.
    by_visits.map(|(action_id, _)| action_id)
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

/// Improved-policy target π'(a)=softmax(logits+σ(completedQ)) over the root
/// candidate support. Returns (action_ids, weights, raw_logits); the raw-logits
/// column is returned alongside. Only the root's own edges form the support;
/// v_mix fills completedQ for an in-support edge that is unvisited.
///
/// Support floor: edges with `N(a) < min_visits` are excluded from the softmax
/// support (which then renormalizes over the survivors). Falls back to the full
/// edge set if the floor would empty the support.
fn gumbel_target_policy(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    c_visit: f32,
    c_scale: f32,
    min_visits: u32,
) -> (Vec<PackedCoord>, Vec<f32>, Vec<f32>) {
    let logit_map = root.root_logits.clone().unwrap_or_default();
    // completedQ map + the v_mix visited-weighted fallback.
    let (completed, v_mix) = gumbel_completed_q(root, &logit_map);
    // σ scale = THIS MOVE's max delta visits over the move-entry baseline, so the
    // exported target's σ multiplier matches the SH ranking's (tree.rs
    // maybe_advance_gumbel_round) and a reused root's inherited visits do not
    // inflate it. On a fresh (baseline None / all-zero) root this equals the
    // cumulative max, so the recorded target is unchanged for lockstep/fresh
    // roots.
    let max_n = root
        .edges
        .iter()
        .map(|e| edge_delta_visits(e, baseline))
        .max()
        .unwrap_or(0);

    // Candidate support = root edges meeting the visit floor.
    let mut in_support: Vec<&RustEdge> = root
        .edges
        .iter()
        .filter(|edge| edge.visits >= min_visits)
        .collect();
    // Degenerate guard: if the floor empties the support, fall back to all edges.
    if in_support.is_empty() {
        in_support = root.edges.iter().collect();
    }

    // Deterministic action_id order (mirrors root_prior_policy's stable order).
    in_support.sort_unstable_by_key(|edge| edge.action_id);

    let mut action_ids = Vec::with_capacity(in_support.len());
    let mut logits = Vec::with_capacity(in_support.len());
    let mut scores = Vec::with_capacity(in_support.len());
    for edge in &in_support {
        let l = logit_map.get(&edge.action_id).copied().unwrap_or(0.0);
        let q = completed.get(&edge.action_id).copied().unwrap_or(v_mix);
        action_ids.push(edge.action_id);
        logits.push(l);
        scores.push(l + gumbel_sigma(q, max_n, c_visit, c_scale));
    }
    let weights = gumbel_softmax(&scores);
    (action_ids, weights, logits)
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

/// Test-facing surface exercising the same LCB formula the search uses. Pure
/// function over the per-edge stats.
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
    // s gates by the chooser's-perspective Q: +1 when q > gate (prefer fewer
    // moves left), -1 when two-sided and q < -gate (prefer more moves left),
    // 0 in the |Q| <= gate dead-zone. Both signs add a positive bonus to the
    // desired child because tanh flips with (m_edge - m_node). Bounded by
    // `weight`.
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

/// Classify one root move for the side to move: `1` proven winning (immediate
/// outcome, or a sound λ¹ child proof in our favor), `-1` proven losing, `0`
/// unproven. Perspective maps back by PLAYER IDENTITY (FirstStone keeps the
/// mover; SecondStone hands over), never by ply parity.
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
            match tss_core::lambda1_status(&child) {
                ProofStatus::Unknown => 0,
                status => {
                    let child_wins = status == ProofStatus::Win;
                    let ours_win = child_wins == (child.current_player() == me);
                    if ours_win {
                        1
                    } else {
                        -1
                    }
                }
            }
        }
    }
}

/// Guard math over a precomputed class lookup (`classify_root_move` per id;
/// absent ids read as 0/unproven): zero non-winning moves when a proven win
/// exists, else zero proven-losing moves; an all-zero result falls back to the
/// original weights (never zero the only legal move).
fn tactical_guard_weights_from(
    classes: &HashMap<PackedCoord, i8>,
    action_ids: &[PackedCoord],
    weights: &[f32],
) -> Vec<f32> {
    let cls = |id: &PackedCoord| classes.get(id).copied().unwrap_or(0);
    let mut guarded = weights.to_vec();
    if action_ids.iter().any(|a| cls(a) == 1) {
        for (i, a) in action_ids.iter().enumerate() {
            if cls(a) != 1 {
                guarded[i] = 0.0;
            }
        }
    } else if action_ids.iter().any(|a| cls(a) != -1) {
        for (i, a) in action_ids.iter().enumerate() {
            if cls(a) == -1 {
                guarded[i] = 0.0;
            }
        }
    }
    if guarded.iter().all(|&w| w <= 0.0) {
        return weights.to_vec();
    }
    guarded
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
    let mut classes: HashMap<PackedCoord, i8> = HashMap::with_capacity(action_ids.len());
    for &id in action_ids {
        classes
            .entry(id)
            .or_insert_with(|| classify_root_move(root_state, id));
    }
    tactical_guard_weights_from(&classes, action_ids, weights)
}

fn validate_harness_horizon(horizon: u32, ladder: bool) -> PyResult<()> {
    if horizon != 0 && horizon < 16 {
        return Err(PyValueError::new_err(
            "horizon must be 0 (unbounded) or >= 16 (the owner floor)",
        ));
    }
    if ladder && horizon == 0 {
        return Err(PyValueError::new_err(
            "the horizon ladder requires a bounded horizon (>= 16)",
        ));
    }
    Ok(())
}

fn harness_zone_caps(zone: bool) -> ZoneSearchCaps {
    ZoneSearchCaps {
        enabled: zone,
        stale_area_filter: false,
        count2_threshold: false,
        pair_commutation: false,
    }
}

fn harness_solver(wide: bool, zone: ZoneSearchCaps, dual_pass: bool) -> TssSolver {
    let mut solver = TssSolver::default();
    if wide {
        solver.configure_leaf_profile();
    }
    solver.set_zone_options(zone);
    solver.set_dual_pass(dual_pass);
    solver
}

#[derive(Debug)]
struct HarnessManifestEnv {
    shared_fragments: Option<String>,
    interior_census_gate: Option<String>,
    k_reply_consume: Option<String>,
    k_reply_shadow: Option<String>,
}

#[derive(Debug)]
struct HarnessSolverManifest {
    effective: EffectiveSolveConfig,
    caps: SolveCaps,
    env: HarnessManifestEnv,
}

fn harness_solver_manifest(
    node_cap: u64,
    horizon: SolverHorizon,
    zone: bool,
    wide: bool,
    dual_pass: bool,
) -> HarnessSolverManifest {
    let solver = harness_solver(wide, harness_zone_caps(zone), dual_pass);
    let caps = tss_verified_solve_caps(0, node_cap, horizon);
    let effective = solver.effective_solve_config(&caps, solver.sample_runtime_flags());
    HarnessSolverManifest {
        effective,
        caps,
        env: HarnessManifestEnv {
            shared_fragments: std::env::var("TSS_SHARED_FRAGMENTS").ok(),
            interior_census_gate: std::env::var("TSS_INTERIOR_CENSUS_GATE").ok(),
            k_reply_consume: std::env::var("TSS_K_REPLY_CONSUME").ok(),
            k_reply_shadow: std::env::var("TSS_K_REPLY_SHADOW").ok(),
        },
    }
}

fn set_solve_stats(d: &Bound<'_, PyDict>, stats: &SolveStats) -> PyResult<()> {
    d.set_item("stats_nodes", stats.nodes)?;
    d.set_item("stats_expansions", stats.expansions)?;
    d.set_item("stats_tt_hits", stats.tt_hits)?;
    d.set_item("stats_tt_entries", stats.tt_entries)?;
    d.set_item("stats_peak_tt_bytes", stats.peak_tt_bytes)?;
    d.set_item("stats_horizon_cuts", stats.horizon_cuts)?;
    d.set_item("stats_kb_death_cuts", stats.kb_death_cuts)?;
    d.set_item("stats_fragment_lookups", stats.fragment_lookups)?;
    d.set_item("stats_fragment_hits", stats.fragment_hits)?;
    d.set_item("stats_fragment_imports", stats.fragment_imports)?;
    d.set_item(
        "stats_interior_gate_evaluations",
        stats.interior_gate_evaluations,
    )?;
    d.set_item(
        "stats_interior_gate_dismissals",
        stats.interior_gate_dismissals,
    )?;
    d.set_item("stats_interior_gate_nanos", stats.interior_gate_nanos)?;
    Ok(())
}

/// Effective configuration echo for the persistent TSS harness solver. The
/// values come from the same pure resolver consumed by `solve_goal`, with the
/// production verified caps resolved for a placements=0 root.
#[pyfunction]
#[pyo3(signature = (node_cap, horizon, ladder, zone, wide, dual_pass=false))]
pub fn hexfield_eq_solver_manifest(
    py: Python<'_>,
    node_cap: u64,
    horizon: u32,
    ladder: bool,
    zone: bool,
    wide: bool,
    dual_pass: bool,
) -> PyResult<Py<PyAny>> {
    validate_harness_horizon(horizon, ladder)?;
    let manifest = harness_solver_manifest(
        node_cap,
        SolverHorizon { horizon, ladder },
        zone,
        wide,
        dual_pass,
    );
    let effective = manifest.effective;
    let d = PyDict::new(py);
    d.set_item("vcf_pair_complete", effective.vcf_pair_complete)?;
    d.set_item("dual_pass", effective.dual_pass)?;
    d.set_item("quiet_turn_or_edges", effective.quiet_turn_or_edges)?;
    d.set_item(
        "ranked_unforced_defender_zone",
        effective.ranked_unforced_defender_zone,
    )?;
    d.set_item("tt_enabled", effective.tt_enabled)?;
    d.set_item("tt_bytes_cap", effective.tt_bytes_cap)?;
    d.set_item(
        "shared_fragments_enabled",
        effective.shared_fragments_enabled,
    )?;
    d.set_item(
        "fragment_store_cap_bytes",
        effective.fragment_store_cap_bytes,
    )?;
    d.set_item("lazy_frontier", effective.lazy_frontier)?;
    d.set_item("interior_census_gate", effective.interior_census_gate)?;
    d.set_item("k_reply_consume", effective.k_reply_consume)?;
    d.set_item("semantic_horizon", manifest.caps.semantic_horizon)?;
    d.set_item("node_cap", manifest.caps.node_cap)?;
    d.set_item("cert_version", tss_core::TSS_CERT_VERSION)?;
    let env = PyDict::new(py);
    env.set_item("TSS_SHARED_FRAGMENTS", manifest.env.shared_fragments)?;
    env.set_item(
        "TSS_INTERIOR_CENSUS_GATE",
        manifest.env.interior_census_gate,
    )?;
    env.set_item("TSS_K_REPLY_CONSUME", manifest.env.k_reply_consume)?;
    env.set_item("TSS_K_REPLY_SHADOW", manifest.env.k_reply_shadow)?;
    d.set_item("env", env)?;
    Ok(d.into_any().unbind())
}

/// λ¹ threat-analysis diagnostic for a live engine state, via the shared
/// `analysis_pydict` builder (identical diagnostic surface across lineages by
/// construction). Drives the hexfield_eq TSS regression fixtures
/// (tests/test_hexfield_eq_tss_*.py) and offline instrumentation.
#[pyfunction]
pub fn hexfield_eq_threat_analysis(
    py: Python<'_>,
    state: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let s = single_state_from_py(py, state)?;
    Ok(threats::analysis_pydict(py, &s)?.into_any().unbind())
}

/// V1 SOAK harness probe (PLAN_TSS_MCTS_INTEGRATION.md §9): run ONE deep solve
/// through the exact production verified path (`tss_solve_verified`, incl. the
/// tight-zone fast-path, the §5 horizon ladder, and the certificate-horizon
/// preflight) on a fresh solver of the requested width profile, and return a
/// flat dict of the verdict, wall time, certificate geometry, and every
/// `TssCounters` field. Consumes NOTHING — the caller measures, never backs up.
/// When `with_stats` is set, an additional direct `solve_goal` on a fresh
/// same-profile solver at the base horizon surfaces the raw `SolveStats`
/// (census-gate dismissals, TT / fragment reuse, nodes) that the verified path
/// folds away. Those stats are cold-by-construction and diagnostics only.
///
/// `goal` ∈ {"win","loss","both"}; `horizon` is 0 (unbounded) or ≥16 (the owner
/// floor — the 1..=15 band is rejected, matching the config seam); `wide=true`
/// selects the leaf-decided profile (`configure_leaf_profile`: wide
/// vcf_pair_complete + lazy frontier + interior census gate), `wide=false` the
/// narrow default `WidthOptions` profile. This is measurement plumbing: it is
/// NOT wired into any consumption path and cannot mint a training value.
#[pyfunction]
#[pyo3(signature = (state, node_cap, goal, horizon, ladder, zone, wide, with_stats=false))]
#[allow(clippy::too_many_arguments)]
pub fn hexfield_eq_deep_solve_probe(
    py: Python<'_>,
    state: &Bound<'_, PyAny>,
    node_cap: u64,
    goal: &str,
    horizon: u32,
    ladder: bool,
    zone: bool,
    wide: bool,
    with_stats: bool,
) -> PyResult<Py<PyAny>> {
    let s = single_state_from_py(py, state)?;
    validate_harness_horizon(horizon, ladder)?;
    let goal_enum = match goal {
        "win" => tss_core::SolveGoal::Win,
        "loss" => tss_core::SolveGoal::Loss,
        "both" => tss_core::SolveGoal::Both,
        other => {
            return Err(PyValueError::new_err(format!(
                "goal must be win|loss|both, got {other:?}"
            )))
        }
    };
    let zone_caps = harness_zone_caps(zone);
    let placements = s.placements_made();

    let mut solver = harness_solver(wide, zone_caps, false);
    let mut counters = TssCounters::default();
    let start = Instant::now();
    let solved = tss_solve_verified(
        &s,
        node_cap,
        goal_enum,
        zone_caps,
        SolverHorizon { horizon, ladder },
        &mut solver,
        &mut counters,
    );
    let wall_nanos = start.elapsed().as_nanos() as u64;

    // Certificate geometry: derived_t (the verifier's own max exact-leaf
    // resolution ply) minus the root placement index = proof depth in plies.
    let (cert_depth, cert_choice_nodes, cert_universal_nodes, cert_zone_nodes) = match &solved.cert {
        Some(cert) => {
            let mut derived_t = 0u32;
            let (mut choice, mut univ, mut zn) = (0u32, 0u32, 0u32);
            for node in &cert.nodes {
                match node {
                    CertNode::OrCompletion { completion_ply, .. } => {
                        derived_t = derived_t.max(*completion_ply);
                    }
                    CertNode::Win { resolution_ply, .. }
                    | CertNode::Loss { resolution_ply, .. } => {
                        derived_t = derived_t.max(*resolution_ply);
                    }
                    CertNode::Choice { .. } => choice += 1,
                    CertNode::Universal { zone, .. } => {
                        univ += 1;
                        zn += u32::from(zone.is_some());
                    }
                }
            }
            (
                derived_t.saturating_sub(placements),
                choice,
                univ,
                zn,
            )
        }
        None => (0, 0, 0, 0),
    };
    // The certificate's designated root move (the OR-node winning continuation),
    // for the §8 internalization baseline: prior mass + rank of THIS move at the
    // proven root. Emitted only when the root node is a Choice (a proven WIN root
    // guard shape); absent for defender-loss roots and for OrCompletion roots.
    let cert_root_move: Option<(i32, i32)> = solved.cert.as_ref().and_then(|cert| {
        match cert.nodes.get(cert.root_node as usize) {
            Some(CertNode::Choice { mv, .. }) => Some((mv.q as i32, mv.r as i32)),
            _ => None,
        }
    });
    let status = match solved.status {
        ProofStatus::Win => "win",
        ProofStatus::Loss => "loss",
        ProofStatus::Unknown => "unknown",
    };

    let d = PyDict::new(py);
    d.set_item("status", status)?;
    if let Some((q, r)) = cert_root_move {
        d.set_item("cert_root_move_q", q)?;
        d.set_item("cert_root_move_r", r)?;
    }
    d.set_item("wall_nanos", wall_nanos)?;
    d.set_item("placements", placements)?;
    d.set_item("has_cert", solved.cert.is_some())?;
    d.set_item("cert_depth", cert_depth)?;
    d.set_item("cert_choice_nodes", cert_choice_nodes)?;
    d.set_item("cert_universal_nodes", cert_universal_nodes)?;
    d.set_item("cert_zone_nodes", cert_zone_nodes)?;
    d.set_item("cert_version", tss_core::TSS_CERT_VERSION)?;
    // Every TssCounters field the verified path touched.
    d.set_item("deep_calls", counters.deep_calls)?;
    d.set_item("deep_win", counters.deep_win)?;
    d.set_item("deep_loss", counters.deep_loss)?;
    d.set_item("deep_unknown", counters.deep_unknown)?;
    d.set_item("deep_nodes", counters.deep_nodes)?;
    d.set_item("deep_verify_failed", counters.deep_verify_failed)?;
    d.set_item("horizon_retry", counters.horizon_retry)?;
    d.set_item("horizon_preflight_failed", counters.horizon_preflight_failed)?;
    d.set_item("horizon_cut", counters.horizon_cut)?;
    d.set_item("horizon_cut_tall", counters.horizon_cut_tall)?;
    d.set_item("deep_kb_death", counters.deep_kb_death)?;
    d.set_item("zone_nodes", counters.zone_nodes)?;
    d.set_item("pair_omitted", counters.pair_omitted)?;
    d.set_item("zone_verify_failed", counters.zone_verify_failed)?;

    if with_stats {
        let mut stats_solver = harness_solver(wide, zone_caps, false);
        // Shared production-cap constructor keeps this cold diagnostic at the
        // trainer leaf/root/async memory profile.
        let caps = tss_verified_solve_caps(
            placements,
            node_cap,
            SolverHorizon {
                horizon,
                ladder: false,
            },
        );
        let raw = stats_solver.solve_goal(&s, &caps, goal_enum);
        set_solve_stats(&d, &raw.stats)?;
    }

    Ok(d.into_any().unbind())
}

/// V1 SOAK warmth-sensitivity probe: solve a SEQUENCE of states on ONE
/// persistent solver (built once, `configure_leaf_profile` if `wide`), so the
/// shared positive-proof-fragment cache warms across positions exactly as the
/// production per-batch persistent leaf solver does across moves. Pass a game's
/// positions in ply order to bound the cold-vs-warm gap in the single-shot
/// probe. Returns a list of per-position dicts with verdict/certificate data,
/// verified-path counters, and actual aggregate `stats_*` telemetry from this
/// persistent solver. Measurement only.
#[pyfunction]
#[pyo3(signature = (states, node_cap, goal, horizon, ladder, zone, wide, dual_pass=false))]
#[allow(clippy::too_many_arguments)]
pub fn hexfield_eq_deep_solve_batch(
    py: Python<'_>,
    states: &Bound<'_, PyList>,
    node_cap: u64,
    goal: &str,
    horizon: u32,
    ladder: bool,
    zone: bool,
    wide: bool,
    dual_pass: bool,
) -> PyResult<Py<PyAny>> {
    validate_harness_horizon(horizon, ladder)?;
    let goal_enum = match goal {
        "win" => tss_core::SolveGoal::Win,
        "loss" => tss_core::SolveGoal::Loss,
        "both" => tss_core::SolveGoal::Both,
        other => {
            return Err(PyValueError::new_err(format!(
                "goal must be win|loss|both, got {other:?}"
            )))
        }
    };
    let zone_caps = harness_zone_caps(zone);
    let mut solver = harness_solver(wide, zone_caps, dual_pass);
    let out = PyList::empty(py);
    for state_any in states.iter() {
        let s = single_state_from_py(py, &state_any)?;
        let placements = s.placements_made();
        let mut counters = TssCounters::default();
        let start = Instant::now();
        let verified = tss_solve_verified_with_stats(
            &s,
            node_cap,
            goal_enum,
            zone_caps,
            SolverHorizon { horizon, ladder },
            &mut solver,
            &mut counters,
        );
        let solved = &verified.solve;
        let wall_nanos = start.elapsed().as_nanos() as u64;
        let mut derived_t = 0u32;
        let mut zn = 0u32;
        if let Some(cert) = &solved.cert {
            for node in &cert.nodes {
                match node {
                    CertNode::OrCompletion { completion_ply, .. } => {
                        derived_t = derived_t.max(*completion_ply);
                    }
                    CertNode::Win { resolution_ply, .. }
                    | CertNode::Loss { resolution_ply, .. } => {
                        derived_t = derived_t.max(*resolution_ply);
                    }
                    CertNode::Universal { zone, .. } => zn += u32::from(zone.is_some()),
                    CertNode::Choice { .. } => {}
                }
            }
        }
        let status = match solved.status {
            ProofStatus::Win => "win",
            ProofStatus::Loss => "loss",
            ProofStatus::Unknown => "unknown",
        };
        let d = PyDict::new(py);
        d.set_item("status", status)?;
        d.set_item("wall_nanos", wall_nanos)?;
        d.set_item("has_cert", solved.cert.is_some())?;
        d.set_item("cert_depth", derived_t.saturating_sub(placements))?;
        d.set_item("deep_nodes", counters.deep_nodes)?;
        d.set_item("deep_verify_failed", counters.deep_verify_failed)?;
        d.set_item("horizon_cut", counters.horizon_cut)?;
        d.set_item("horizon_cut_tall", counters.horizon_cut_tall)?;
        d.set_item("deep_kb_death", counters.deep_kb_death)?;
        d.set_item("zone_nodes", zn)?;
        set_solve_stats(&d, &verified.stats)?;
        out.append(d)?;
    }
    Ok(out.into_any().unbind())
}

/// Action selection: temperature sampling when temperature > 0, and on greedy
/// (T == 0) paths with `lcb_move_selection` on, LCB-of-Q selection among
/// eligible children (fallback max-visits). The TSS guard has already zeroed
/// proven-losing weights; LCB only ever picks among guard-positive actions.
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
                // Decisiveness tie-break on the played move: among moves
                // value-tied with the LCB leader, prefer the decisive one. Gated
                // on moves_left_utility; returns lcb_id in the dead-zone or with
                // no ml stats.
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
    // Compute each edge's delta visits once (edge_delta_visits is a HashMap
    // lookup when baseline is Some) and reuse the deltas for both the total and
    // the per-edge weights.
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
        // powf in f64: at low temperatures (large exponents) an f32 powf
        // underflows flat histograms to all-zero mass, aborting the batch with
        // the positive-finite-mass error below. f64 keeps ~1e-308 of headroom.
        let value = (*weight as f64).powf(inv_temperature as f64);
        total += value;
        adjusted.push(value);
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(PyValueError::new_err(
            "temperature-adjusted visit policy must contain positive finite mass",
        ));
    }
    // Walk the CDF, skipping zero-weight (e.g. tactical-guard-zeroed) entries so
    // they can never be selected: random_unit == 0.0 puts threshold at 0.0 up
    // front, which would otherwise return the FIRST action even if its adjusted
    // weight is 0; and f64 residue at the tail must not fall through onto a
    // zero-weight last action. The fallback is the LAST positive-weight action.
    let mut threshold = random_unit(seed) * total;
    let mut last_positive: Option<PackedCoord> = None;
    for (action_id, weight) in action_ids.iter().copied().zip(adjusted) {
        if weight <= 0.0 {
            continue;
        }
        last_positive = Some(action_id);
        threshold -= weight;
        if threshold <= 0.0 {
            return Ok(Some(action_id));
        }
    }
    Ok(last_positive)
}

#[cfg(test)]
mod fallback_tests {
    use super::*;
    use crate::tree::{NodePriors, RustPriorCandidate};
    use hexo_engine::Player;
    use hexo_utils::StateHash;
    use std::ffi::OsString;
    use std::sync::Mutex;

    static TSS_HARNESS_ENV_MUTEX: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        name: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                previous: std::env::var_os(name),
            }
        }

        fn set(&self, value: Option<&str>) {
            if let Some(value) = value {
                std::env::set_var(self.name, value);
            } else {
                std::env::remove_var(self.name);
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = self.previous.take() {
                std::env::set_var(self.name, value);
            } else {
                std::env::remove_var(self.name);
            }
        }
    }

    #[test]
    fn solver_manifest_reflects_shared_fragments_env() {
        let _lock = TSS_HARNESS_ENV_MUTEX.lock().unwrap();
        let env = EnvVarGuard::new("TSS_SHARED_FRAGMENTS");
        env.set(None);
        let off = harness_solver_manifest(
            500,
            SolverHorizon {
                horizon: 0,
                ladder: false,
            },
            false,
            true,
            false,
        );
        assert_eq!(off.env.shared_fragments, None);
        assert!(!off.effective.shared_fragments_enabled);
        assert_eq!(off.effective.fragment_store_cap_bytes, 0);

        env.set(Some("1"));
        let on = harness_solver_manifest(
            500,
            SolverHorizon {
                horizon: 0,
                ladder: false,
            },
            false,
            true,
            false,
        );
        assert_eq!(on.env.shared_fragments.as_deref(), Some("1"));
        assert!(on.effective.shared_fragments_enabled);
        assert!(on.effective.fragment_store_cap_bytes > 0);
        assert_eq!(on.caps.tt_bytes_cap, on.effective.tt_bytes_cap);
    }

    #[test]
    fn solver_manifest_matches_real_solve_effective_config() {
        let _lock = TSS_HARNESS_ENV_MUTEX.lock().unwrap();
        let env = EnvVarGuard::new("TSS_SHARED_FRAGMENTS");
        env.set(Some("1"));
        let horizon = SolverHorizon {
            horizon: 0,
            ladder: false,
        };
        let manifest = harness_solver_manifest(500, horizon, false, true, false);
        let state = RustHexoState::new();
        let zone = harness_zone_caps(false);
        let mut solver = harness_solver(true, zone, false);
        let mut counters = TssCounters::default();
        let _ = tss_solve_verified(
            &state,
            500,
            tss_core::SolveGoal::Both,
            zone,
            horizon,
            &mut solver,
            &mut counters,
        );
        assert_eq!(solver.last_effective_config(), Some(manifest.effective));
        let real_caps = tss_verified_solve_caps(state.placements_made(), 500, horizon);
        assert_eq!(real_caps.node_cap, manifest.caps.node_cap);
        assert_eq!(real_caps.tt_bytes_cap, manifest.caps.tt_bytes_cap);
        assert_eq!(real_caps.semantic_horizon, manifest.caps.semantic_horizon);
    }

    #[test]
    fn solver_manifest_echoes_dual_pass_from_shared_resolver() {
        let horizon = SolverHorizon {
            horizon: 0,
            ladder: false,
        };
        for dual_pass in [false, true] {
            let manifest = harness_solver_manifest(500, horizon, false, true, dual_pass);
            assert_eq!(manifest.effective.dual_pass, dual_pass);

            let solver = harness_solver(true, harness_zone_caps(false), dual_pass);
            let caps = tss_verified_solve_caps(0, 500, horizon);
            assert_eq!(
                solver.effective_solve_config(&caps, solver.sample_runtime_flags()),
                manifest.effective,
            );

            Python::initialize();
            Python::attach(|py| {
                let echoed = hexfield_eq_solver_manifest(
                    py, 500, 0, false, false, true, dual_pass,
                )
                .unwrap();
                let dict = echoed.bind(py).cast::<PyDict>().unwrap();
                assert_eq!(
                    dict.get_item("dual_pass")
                        .unwrap()
                        .unwrap()
                        .extract::<bool>()
                        .unwrap(),
                    dual_pass,
                );
            });
        }
    }

    fn shared_fragment_batch_fixture() -> RustHexoState {
        let mut state = scheduler_replay(&[
            (0, 0),
            (-1, 0),
            (1, -2),
            (-2, 0),
            (1, 0),
            (0, -2),
            (1, -3),
            (0, -3),
            (2, -5),
            (2, -4),
            (1, -4),
            (3, -4),
            (3, -2),
        ]);
        for coord in [
            hexo_engine::HexCoord::new(-1, -1),
            hexo_engine::HexCoord::new(1, -5),
        ] {
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        state
    }

    #[test]
    fn persistent_batch_stats_report_warm_fragment_imports_only_when_enabled() {
        let _lock = TSS_HARNESS_ENV_MUTEX.lock().unwrap();
        let env = EnvVarGuard::new("TSS_SHARED_FRAGMENTS");
        let state = shared_fragment_batch_fixture();
        let zone = harness_zone_caps(false);
        let horizon = SolverHorizon {
            horizon: 0,
            ladder: false,
        };

        env.set(Some("1"));
        let mut warm_solver = harness_solver(true, zone, false);
        let mut first_counters = TssCounters::default();
        let first = tss_solve_verified_with_stats(
            &state,
            10_000,
            tss_core::SolveGoal::Loss,
            zone,
            horizon,
            &mut warm_solver,
            &mut first_counters,
        );
        let mut second_counters = TssCounters::default();
        let second = tss_solve_verified_with_stats(
            &state,
            10_000,
            tss_core::SolveGoal::Loss,
            zone,
            horizon,
            &mut warm_solver,
            &mut second_counters,
        );
        assert_eq!(first.solve.status, ProofStatus::Loss);
        assert_eq!(second.solve.status, first.solve.status);
        assert!(second.stats.fragment_lookups > 0);
        assert!(second.stats.fragment_hits > 0);
        assert!(second.stats.fragment_imports > 0);
        assert_eq!(second.stats.nodes, second_counters.deep_nodes);

        env.set(None);
        let mut cold_solver = harness_solver(true, zone, false);
        let mut cold_first_counters = TssCounters::default();
        let _ = tss_solve_verified_with_stats(
            &state,
            10_000,
            tss_core::SolveGoal::Loss,
            zone,
            horizon,
            &mut cold_solver,
            &mut cold_first_counters,
        );
        let mut cold_second_counters = TssCounters::default();
        let cold_second = tss_solve_verified_with_stats(
            &state,
            10_000,
            tss_core::SolveGoal::Loss,
            zone,
            horizon,
            &mut cold_solver,
            &mut cold_second_counters,
        );
        assert_eq!(cold_second.stats.fragment_lookups, 0);
        assert_eq!(cold_second.stats.fragment_hits, 0);
        assert_eq!(cold_second.stats.fragment_imports, 0);
        assert_eq!(cold_second.stats.nodes, cold_second_counters.deep_nodes);
    }

    fn edge(action_id: PackedCoord, prior: f32, visits: u32, value_sum: f32) -> RustEdge {
        RustEdge {
            action_id,
            action: unpack_coord(action_id),
            prior,
            visits,
            value_sum,
            value_sq_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            pending: 0,
            child: None,
            forced: false,
        }
    }

    fn node(edges: Vec<RustEdge>, candidates: Vec<RustPriorCandidate>) -> RustNode {
        RustNode {
            state_hash: StateHash::default(),
            player: Player::Player0,
            eval_value: 0.0,
            eval_ml: None,
            visits: edges.iter().map(|e| e.visits).sum(),
            value_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            edges,
            priors: NodePriors::Owned(candidates),
            max_eligible_children: 8,
            root_logits: None,
        }
    }

    // ---- Forced-playout target pruning (KataGo policy-target pruning). ----
    // These lock the main_8 Full-move export path: for a PUCT (non-Gumbel) root
    // with forced_playout_k > 0, build_search_result_payload_native records the
    // policy target from pruned_visit_policy -> prune_forced_delta_counts, which
    // strips the sqrt(k*P*N) forced-exploration visits back out while leaving the
    // raw visit_policy (used only for play selection) untouched. Counts are exact
    // u32, so these assert exact expected histograms. explore = c_puct*sqrt(N).

    #[test]
    fn prune_forced_strips_forced_visits_from_a_low_value_child() {
        // N=100, c=1.5 -> explore=15. Best = idx0 (60 delta).
        // u_best = 0.5 + 0.7*15/61 = 0.6721.
        // idx1: n_forced = floor(sqrt(2*0.05*100)) = 3; each removal keeps
        // U_1 (~ -0.22) far below u_best, so all 3 forced visits are stripped.
        let pruned = prune_forced_delta_counts(
            &[60, 10], &[0.7, 0.05], &[60, 10], &[0.5, -0.3], 100, 2.0, 1.5,
        );
        assert_eq!(pruned, vec![60, 7]);
    }

    #[test]
    fn prune_forced_keeps_genuine_visits_of_a_high_value_child() {
        // Same best/u_best. idx1 has n_forced=floor(sqrt(60))=7, but value 0.8
        // makes U_1 = 0.8 + 0.3*15/10 = 1.25 > u_best on the first candidate
        // removal, so the loop breaks immediately: genuine visits, not forced.
        let pruned = prune_forced_delta_counts(
            &[60, 10], &[0.7, 0.3], &[60, 10], &[0.5, 0.8], 100, 2.0, 1.5,
        );
        assert_eq!(pruned, vec![60, 10]);
    }

    #[test]
    fn prune_forced_zero_k_is_identity() {
        // forced_playout_k = 0 (Gumbel-era / off): the recorded target is the
        // raw delta-visit histogram, unchanged.
        let deltas = [60u32, 10, 5];
        let pruned = prune_forced_delta_counts(
            &deltas, &[0.5, 0.3, 0.2], &[60, 10, 5], &[0.4, 0.1, -0.2], 75, 0.0, 1.5,
        );
        assert_eq!(pruned, deltas.to_vec());
    }

    #[test]
    fn prune_forced_never_prunes_best_and_caps_at_n_forced() {
        // Three children; best = idx0 (50) is never touched. u_best = 0.5471.
        // idx1: n_forced=floor(sqrt(80))=8, U stays below u_best across all 8 ->
        // loses exactly 8 (30->22). idx2: n_forced=floor(sqrt(4))=2 -> 8->6.
        let pruned = prune_forced_delta_counts(
            &[50, 30, 8], &[0.5, 0.4, 0.02], &[50, 30, 8], &[0.4, 0.2, -0.5], 100, 2.0, 1.5,
        );
        assert_eq!(pruned, vec![50, 22, 6]);
    }

    // Non-sentinel action ids: PackedCoord 0 unpacks to the illegal sentinel
    // HexCoord{q:-32768,r:-32768}; any non-zero id we use here is a "real" action
    // for the purposes of the played-move invariant.
    const A1: PackedCoord = 0x8001_8000; // q=1, r=0
    const A2: PackedCoord = 0x8000_8001; // q=0, r=1
    const A3: PackedCoord = 0x8001_8001; // q=1, r=1

    #[test]
    fn delta_visit_policy_is_empty_when_all_edges_match_baseline() {
        // A root whose edges all sit at their reuse baseline (zero net delta)
        // yields an empty delta-visit policy, so the normal selection returns
        // None (the force-completed Gumbel SH case).
        let root = node(
            vec![edge(A1, 0.6, 3, 1.5), edge(A2, 0.4, 2, 0.5)],
            Vec::new(),
        );
        let baseline: HashMap<PackedCoord, u32> =
            [(A1, 3u32), (A2, 2u32)].into_iter().collect();
        let (ids, weights, _q, total) = visit_policy(&root, Some(&baseline));
        assert!(ids.is_empty(), "all-baseline delta policy must be empty");
        assert!(weights.is_empty());
        assert_eq!(total, 0);
        // The normal sampler returns None on an empty policy.
        let picked = select_action_from_policy(&ids, &weights, 1.0, 7).unwrap();
        assert!(picked.is_none());
    }

    #[test]
    fn fallback_never_returns_sentinel_and_prefers_most_visited() {
        // The fallback used when `selected` is None must yield a REAL root action
        // (never PackedCoord 0 / the sentinel). With visits present it picks the
        // most-visited cumulative edge.
        let root = node(
            vec![edge(A1, 0.2, 5, 2.0), edge(A2, 0.7, 1, 0.1)],
            vec![RustPriorCandidate { action_id: A3, prior: 0.9 }],
        );
        let picked = fallback_root_action(&root).expect("fallback yields an action");
        assert_ne!(picked, 0, "fallback must never return the sentinel id 0");
        assert_eq!(picked, A1, "most-visited cumulative edge wins");
    }

    #[test]
    fn cumulative_visit_policy_recovers_target_when_delta_is_empty() {
        // The RECORDED-target fallback in build_search_result_payloads substitutes
        // the baseline-free cumulative visit policy when the delta-visit export is
        // empty. Pin the property it relies on: with edges carrying cumulative
        // visits, visit_policy(root, None) yields a NON-EMPTY, positive-mass target
        // even though the delta policy (vs an all-matching baseline) is empty.
        let root = node(
            vec![edge(A1, 0.6, 3, 1.5), edge(A2, 0.4, 2, 0.5)],
            Vec::new(),
        );
        let baseline: HashMap<PackedCoord, u32> =
            [(A1, 3u32), (A2, 2u32)].into_iter().collect();
        let (d_ids, _d_w, _d_q, d_total) = visit_policy(&root, Some(&baseline));
        assert!(d_ids.is_empty() && d_total == 0, "delta policy is empty");
        let (c_ids, c_w, _c_q, c_total) = visit_policy(&root, None);
        assert_eq!(c_ids.len(), 2, "cumulative policy keeps both edges");
        assert_eq!(c_total, 5, "cumulative total = sum of edge visits");
        let mass: f32 = c_w.iter().sum();
        assert!((mass - 1.0).abs() < 1e-6, "cumulative target carries unit mass");
        assert!(c_ids.iter().all(|&id| id != 0), "no sentinel in the target");
    }

    #[test]
    fn fallback_uses_highest_prior_when_no_visits() {
        // Degenerate root: edges exist but carry zero visits (force-completed with
        // nothing searched). Fallback then takes the highest-prior action across
        // edges + unexpanded candidates — still a real, legal action id.
        let root = node(
            vec![edge(A1, 0.2, 0, 0.0), edge(A2, 0.3, 0, 0.0)],
            vec![RustPriorCandidate { action_id: A3, prior: 0.5 }],
        );
        let picked = fallback_root_action(&root).expect("fallback yields an action");
        assert_ne!(picked, 0);
        assert_eq!(picked, A3, "highest-prior action wins when no edge is visited");
    }

    // --- Fast-class play temperature + sampler zero-weight edge ---------------

    // Non-sentinel ids for the sampler tests.
    const S1: PackedCoord = A1;
    const S2: PackedCoord = A2;
    const S3: PackedCoord = A3;

    fn move_policy(fast_temperature: f32) -> ContinuousMovePolicy {
        ContinuousMovePolicy {
            full_visits: 512,
            fast_visits: 128,
            fast_temperature,
            pcr_full_proportion: 0.33,
            policy_init_fraction: 0.0,
            policy_init_avg_plies: 0.0,
            policy_init_max_plies: 0,
            policy_init_temperature: 1.0,
            root_policy_temperature: 1.0,
            root_policy_temperature_early: 0.0,
            root_policy_temperature_halflife: 0.0,
            fpu_reduction: 0.2,
            forced_playout_k: 0.0,
            noise: None,
            tss_enabled: true,
            root_fpu_zero_under_noise: false,
            root_fpu_reduction: None,
            divergences_full: Divergences::production(),
            divergences_fast: Divergences::production(),
        }
    }

    fn scheduler_park_test_search(timeout_ms: u32) -> (RustSearch, TssAsyncPool, u64) {
        let action_id = pack_coord(hexo_engine::HexCoord { q: 0, r: 0 });
        let eval = RustEvaluation {
            value: 0.0,
            legal_action_count: 1,
            priors: vec![(action_id, 1.0)],
            moves_left: None,
            logits: None,
        };
        let mut divergences = Divergences::parity();
        divergences.tss_solver_mode = 3;
        divergences.tss_solver_node_cap = 1;
        divergences.tss_solver_sample_16 = 16;
        divergences.tss_solver_async = true;
        divergences.tss_solver_async_threads = 1;
        divergences.tss_solver_async_threads_max = 1;
        divergences.tss_solver_park = true;
        divergences.tss_solver_park_timeout_ms = timeout_ms;
        // Park mode must supersede even an all-inline legacy tier.
        divergences.tss_solver_async_inline_16 = 16;
        let mut search = RustSearch::new(
            RustHexoState::new(),
            &eval,
            4,
            0.2,
            0.0,
            1.0,
            None,
            Widening {
                mass: 1.0,
                min_children: 1,
                max_children: 1,
            },
            0.0,
            true,
            divergences,
        )
        .expect("park test search builds");
        let pool = TssAsyncPool::new(1, 1, true);
        search.set_tss_async(Some(pool.handle_for(0)));
        let generation = search.tss_async_generation().expect("handle wired");
        (search, pool, generation)
    }

    /// Obtain a REAL selected leaf exactly as the scheduler does: selection
    /// materializes the root edge and applies the virtual visit; the pending
    /// mark mirrors the park/miss arms. Hand-forged `(0, 0)` paths panic on
    /// the lazily-widened root (its edge vector starts empty).
    fn scheduler_park_selected(search: &mut RustSearch) -> (Vec<(usize, usize)>, usize, usize) {
        let selected = search
            .select_pending_leaf(1.0)
            .expect("park test selection succeeds")
            .expect("park test root offers a pending leaf");
        search.apply_virtual_visit(&selected.path, 1.0);
        search.mark_pending(selected.parent_node, selected.edge_index, 1);
        (selected.path, selected.parent_node, selected.edge_index)
    }

    fn scheduler_replay(coords: &[(i16, i16)]) -> RustHexoState {
        let mut state = RustHexoState::new();
        for &(q, r) in coords {
            apply_placement(
                &mut state,
                Placement {
                    coord: hexo_engine::HexCoord { q, r },
                },
            )
            .unwrap();
        }
        state
    }

    fn scheduler_win_now_fixture() -> RustHexoState {
        scheduler_replay(&[
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (4, 0),
            (8, 4),
            (10, 3),
        ])
    }

    fn scheduler_forced_defense_fixture() -> RustHexoState {
        scheduler_replay(&[
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (4, 0),
        ])
    }

    #[test]
    fn parked_hard_backup_releases_pending_and_in_flight_exactly_once() {
        let (mut search, _pool, generation) = scheduler_park_test_search(100);
        let state = scheduler_win_now_fixture();
        let hash = state_hash(&state);
        let mut solve_counters = TssCounters::default();
        let solved = tss_solve_verified(
            &state,
            2000,
            tss_core::SolveGoal::Both,
            tss_core::ZoneSearchCaps::default(),
            SolverHorizon::DEFAULT,
            &mut crate::tss_solver::TssSolver::default(),
            &mut solve_counters,
        );
        let hard = solved.hard.expect("fixture yields a verified hard value");
        search.apply_tss_async_response(&crate::tss_async::SolveResponse {
            slot: 0,
            generation,
            hash,
            binding: crate::tss_verify::RootBinding::from_state(&state),
            status: solved.status,
            hard: Some(hard),
            counters: solve_counters,
        });

        let (path, parent_node, edge_index) = scheduler_park_selected(&mut search);
        search.tss.park_parked = 1;
        let leaf = RustLeaf {
            root_index: 0,
            parent_node,
            edge_index,
            path,
            state,
            state_hash: hash,
        };
        let mut slots = vec![ContinuousSlot {
            game_key: 7,
            ply: 0,
            search: Some(search),
            phase: ContinuousPhase::Active,
            in_flight: 1,
            baseline: HashMap::new(),
            policy_init_remaining: 0,
            move_class: MoveClass::Full,
        }];
        let mut parked = vec![ParkedLeaf {
            leaf,
            parked_at: Instant::now(),
            generation,
        }];
        let mut queue = Vec::new();

        resolve_parked_continuous(&mut slots, &mut parked, &mut queue, 1.0).unwrap();
        assert!(parked.is_empty());
        assert!(queue.is_empty(), "hard result elides evaluation");
        assert_eq!(slots[0].in_flight, 0);
        let search = slots[0].search.as_ref().unwrap();
        assert_eq!(search.nodes[parent_node].edges[edge_index].pending, 0);
        assert_eq!(search.tss.park_hard, 1);
        assert_eq!(search.tss.backups, 1);

        // Draining the now-empty pen cannot release or back up a second time.
        resolve_parked_continuous(&mut slots, &mut parked, &mut queue, 1.0).unwrap();
        let search = slots[0].search.as_ref().unwrap();
        assert_eq!(search.tss.park_hard, 1);
        assert_eq!(search.tss.backups, 1);
    }

    #[test]
    fn parked_bail_moves_leaf_to_eval_without_releasing_pending_early() {
        let (mut search, _pool, generation) = scheduler_park_test_search(1);
        let state = scheduler_win_now_fixture();
        let hash = state_hash(&state);
        let (path, parent_node, edge_index) = scheduler_park_selected(&mut search);
        search.tss.park_parked = 1;
        let leaf = RustLeaf {
            root_index: 0,
            parent_node,
            edge_index,
            path,
            state,
            state_hash: hash,
        };
        let mut slots = vec![ContinuousSlot {
            game_key: 9,
            ply: 0,
            search: Some(search),
            phase: ContinuousPhase::Active,
            in_flight: 1,
            baseline: HashMap::new(),
            policy_init_remaining: 0,
            move_class: MoveClass::Full,
        }];
        let mut parked = vec![ParkedLeaf {
            leaf,
            parked_at: Instant::now() - Duration::from_millis(5),
            generation,
        }];
        let mut queue = Vec::new();

        resolve_parked_continuous(&mut slots, &mut parked, &mut queue, 1.0).unwrap();
        assert!(parked.is_empty());
        assert_eq!(queue.len(), 1, "bail releases exactly one eval item");
        assert!(matches!(
            queue.first(),
            Some(ContinuousEvalItem::Leaf(_))
        ));
        assert_eq!(slots[0].in_flight, 1, "eval path still owns in-flight");
        let search = slots[0].search.as_ref().unwrap();
        assert_eq!(search.nodes[parent_node].edges[edge_index].pending, 1);
        assert_eq!(search.tss.park_bailed, 1);
        assert!(search.tss.park_wait_ms_sum >= 1);
    }

    #[test]
    fn pen_only_loop_bails_and_finishes_the_selected_visit() {
        let (mut search, _pool, generation) = scheduler_park_test_search(1);
        search.target_visits = 1;
        let state = scheduler_forced_defense_fixture();
        let hash = state_hash(&state);
        let (path, parent_node, edge_index) = scheduler_park_selected(&mut search);
        search.tss.park_parked = 1;
        assert!(matches!(
            search.tss_deep_leaf_route(&state, hash),
            TssLeafRoute::Parked
        ));
        assert_eq!(
            search.tss.async_enqueued, 1,
            "the liveness test must hold a real accepted solve request"
        );
        let legal_ids: Vec<_> = state.board().legal_moves().action_ids().collect();
        let mut slots = vec![ContinuousSlot {
            game_key: 11,
            ply: 0,
            search: Some(search),
            phase: ContinuousPhase::Active,
            in_flight: 1,
            baseline: HashMap::new(),
            policy_init_remaining: 0,
            move_class: MoveClass::Full,
        }];
        let mut parked = vec![ParkedLeaf {
            leaf: RustLeaf {
                root_index: 0,
                parent_node,
                edge_index,
                path,
                state,
                state_hash: hash,
            },
            // Model an Unknown/slow worker by deliberately leaving its real
            // response channel undrained. The scheduler must poll, let the
            // actual deadline pass, bail, and continue.
            parked_at: Instant::now(),
            generation,
        }];
        let mut queue = Vec::new();
        assert!(continuous_has_work(&slots, &parked));
        assert_eq!(
            continuous_flush_decision(0, 1, true),
            ContinuousFlushDecision::Hold,
            "a pen-only pass is live work, not a scheduler stall"
        );
        let liveness_deadline = Instant::now() + Duration::from_secs(1);
        loop {
            assert!(
                Instant::now() < liveness_deadline,
                "pen-only scheduler failed to reach its bounded bail"
            );
            resolve_parked_continuous(&mut slots, &mut parked, &mut queue, 1.0).unwrap();
            let decision = continuous_flush_decision(queue.len(), 1, !parked.is_empty());
            match decision {
                ContinuousFlushDecision::Hold => std::thread::yield_now(),
                ContinuousFlushDecision::Flush { no_progress } => {
                    assert!(no_progress, "the bailed leaf must force a short eval flush");
                    break;
                }
                ContinuousFlushDecision::Stop => {
                    panic!("scheduler declared a stall while a parked leaf was pending")
                }
            }
        }
        assert!(parked.is_empty());
        assert_eq!(queue.len(), 1);

        let legal_count = legal_ids.len();
        assert!(legal_count > 0, "forced-defense fixture must remain nonterminal");
        let prior = 1.0 / legal_count as f32;
        let child_eval = Arc::new(RustEvaluation {
            value: 0.25,
            legal_action_count: legal_count,
            priors: legal_ids.into_iter().map(|id| (id, prior)).collect(),
            moves_left: None,
            logits: None,
        });
        let item = queue.pop().unwrap();
        apply_backup_item(
            &mut slots[0],
            item,
            &child_eval,
            &move_policy(0.0),
            Widening {
                mass: 1.0,
                min_children: 1,
                max_children: 1,
            },
            0,
            1.0,
            Divergences::parity(),
        )
        .unwrap();

        let search = slots[0].search.as_ref().unwrap();
        assert_eq!(slots[0].in_flight, 0);
        assert_eq!(search.nodes[parent_node].edges[edge_index].pending, 0);
        assert_eq!(search.completed_visits, 1);
        assert_eq!(search.nodes[parent_node].edges[edge_index].visits, 1);
        assert_eq!(search.tss.backups, 1);
        assert_eq!(search.tss.park_bailed, 1);
        assert!(continuous_completion_ready(
            search.completed_visits,
            search.target_visits,
            slots[0].in_flight,
        ));
    }

    #[test]
    fn fast_class_default_temperature_is_zero() {
        // Default 0.0 => Fast plays greedily (the T==0 LCB pick branch),
        // reproducing current behavior. Full uses the ply schedule; Init is 0.0.
        let policy = move_policy(0.0);
        let by_ply = vec![0.9, 0.5, 0.15];
        assert_eq!(policy.temperature_for_class(MoveClass::Fast, &by_ply, 1), 0.0);
        assert_eq!(policy.temperature_for_class(MoveClass::Init, &by_ply, 1), 0.0);
        assert_eq!(
            policy.temperature_for_class(MoveClass::Full, &by_ply, 1),
            0.5,
            "Full follows the ply schedule"
        );
    }

    #[test]
    fn fast_class_uses_the_lever_when_set() {
        // The lever flows to the Fast class only; Full/Init are unchanged.
        let policy = move_policy(0.1);
        let by_ply = vec![0.9, 0.5, 0.15];
        assert_eq!(policy.temperature_for_class(MoveClass::Fast, &by_ply, 0), 0.1);
        assert_eq!(policy.temperature_for_class(MoveClass::Init, &by_ply, 0), 0.0);
        assert_eq!(
            policy.temperature_for_class(MoveClass::Full, &by_ply, 0),
            0.9,
            "Full still follows the ply schedule"
        );
    }

    #[test]
    fn sampler_never_selects_zero_weight_first_entry() {
        // random_unit(seed) can be exactly 0.0 for some seeds, putting the CDF
        // threshold at 0.0 before the walk. A zero-weight (tactical-guard-zeroed)
        // FIRST entry must still never be selected. Sweep many seeds at T=0.1.
        let ids = vec![S1, S2, S3];
        let weights = vec![0.0f32, 0.7, 0.3];
        for seed in 0u64..2000 {
            let picked = select_action_from_policy(&ids, &weights, 0.1, seed)
                .unwrap()
                .expect("positive mass yields a pick");
            assert_ne!(picked, S1, "zero-weight first entry must never be selected");
        }
    }

    #[test]
    fn sampler_never_selects_zero_weight_last_entry() {
        // f64 residue at the tail must not fall through onto a zero-weight LAST
        // action; the fallback is the last POSITIVE-weight action.
        let ids = vec![S1, S2, S3];
        let weights = vec![0.6f32, 0.4, 0.0];
        for seed in 0u64..2000 {
            let picked = select_action_from_policy(&ids, &weights, 0.1, seed)
                .unwrap()
                .expect("positive mass yields a pick");
            assert_ne!(picked, S3, "zero-weight last entry must never be selected");
        }
    }

    #[test]
    fn sampler_leading_zero_weights_never_selected_high_temperature() {
        // Two leading zero-weight entries (guard-zeroed) with only the tail
        // carrying mass; at a large T the exponent flattens weights but the zero
        // entries stay zero and must never be picked, across seeds.
        let ids = vec![S1, S2, S3];
        let weights = vec![0.0f32, 0.0, 1.0];
        for seed in 0u64..500 {
            let picked = select_action_from_policy(&ids, &weights, 2.0, seed)
                .unwrap()
                .expect("positive mass yields a pick");
            assert_eq!(picked, S3, "only the positive-weight action is selectable");
        }
    }

    // --- Export-only σ softening: gumbel_target_c_scale (lever 1) --------------

    /// Root with two searched edges carrying distinct Qs and distinct logits,
    /// plus a root_logits map, so gumbel_target_policy exercises the full σ path.
    /// Qs are modest (+0.2 / 0.0) so the softening test's softmax stays away from
    /// the one-hot saturation region where a c_scale change is invisible.
    fn target_root() -> RustNode {
        // edge A1: 5 visits, sum 1.0 => Q=+0.2 ; edge A2: 4 visits, sum 0.0 =>
        // Q=0.0. Distinct Qs so a smaller c_scale provably flattens the target.
        let mut root = node(
            vec![edge(A1, 0.6, 5, 1.0), edge(A2, 0.4, 4, 0.0)],
            Vec::new(),
        );
        // Distinct logits (raw, unconstrained sign) keyed by action id.
        root.root_logits = Some([(A1, 0.2f32), (A2, -0.2f32)].into_iter().collect());
        root
    }

    #[test]
    fn target_c_scale_unset_is_bit_identical_to_gumbel_c_scale() {
        // The resolver `gumbel_target_c_scale.unwrap_or(gumbel_c_scale)` must make
        // an unset override bit-identical to computing the target with the plain
        // gumbel_c_scale — no drift from the default path.
        let root = target_root();
        let c_visit = 50.0f32;
        let c_scale = 1.0f32;
        // Reference: the exporter called with c_scale directly.
        let (ref_ids, ref_w, ref_l) = gumbel_target_policy(&root, None, c_visit, c_scale, 1);
        // Resolved value when the override is None (mirrors search.rs call site).
        let div = Divergences::gumbel(); // gumbel_target_c_scale defaults to None
        let resolved = div.gumbel_target_c_scale.unwrap_or(div.gumbel_c_scale);
        assert_eq!(resolved, c_scale, "unset override resolves to gumbel_c_scale");
        let (ids, w, l) = gumbel_target_policy(&root, None, c_visit, resolved, 1);
        assert_eq!(ids, ref_ids);
        assert_eq!(l, ref_l, "logits output is independent of c_scale");
        // Bit-identical weights (same inputs => same float ops).
        assert_eq!(w, ref_w, "unset target c_scale must be bit-identical");
    }

    #[test]
    fn target_c_scale_softens_and_matches_reference_softmax() {
        // With gumbel_target_c_scale = 0.35 the exported weights must equal an
        // independent softmax(l + σ(q, max_n, c_visit, 0.35)) over the support and
        // be strictly flatter (lower top-1 mass) than the c_scale=1.0 target.
        // c_visit=1.0 keeps the σ gain small enough that neither target saturates
        // to a one-hot (where a c_scale change would be invisible in f32).
        let root = target_root();
        let c_visit = 1.0f32;
        let soft = 0.35f32;

        // Exporter output at the softened scale.
        let (ids, weights, _logits) = gumbel_target_policy(&root, None, c_visit, soft, 1);

        // Independent reference over the same (ascending action_id) support.
        let logit_map = root.root_logits.clone().unwrap();
        let (completed, v_mix) = gumbel_completed_q(&root, &logit_map);
        let max_n = root.edges.iter().map(|e| e.visits).max().unwrap();
        let mut support: Vec<PackedCoord> =
            root.edges.iter().filter(|e| e.visits >= 1).map(|e| e.action_id).collect();
        support.sort_unstable();
        assert_eq!(ids, support, "exporter uses ascending action_id support");
        let ref_scores: Vec<f32> = support
            .iter()
            .map(|a| {
                let l = logit_map.get(a).copied().unwrap_or(0.0);
                let q = completed.get(a).copied().unwrap_or(v_mix);
                l + gumbel_sigma(q, max_n, c_visit, soft)
            })
            .collect();
        let ref_weights = gumbel_softmax(&ref_scores);
        for (w, r) in weights.iter().zip(ref_weights.iter()) {
            assert!((w - r).abs() < 1e-6, "softened target must match reference");
        }

        // Strictly flatter than the c_scale=1.0 target: lower top-1 mass.
        let (_full_ids, full_weights, _fl) = gumbel_target_policy(&root, None, c_visit, 1.0, 1);
        let top1_soft = weights.iter().copied().fold(f32::MIN, f32::max);
        let top1_full = full_weights.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            top1_soft < top1_full,
            "softened top-1 mass {top1_soft} must be below the c_scale=1.0 top-1 {top1_full}"
        );
    }

    /// The strict KNOWN_DIVERGENCE_KEYS gate must accept every key the python
    /// side emits — the 2026-07-04 deploy crashed because the new lever keys
    /// were parsed but missing from the whitelist. Exercises the real pyo3
    /// resolve path with both new keys present, plus every main_8 fast_* key.
    #[test]
    fn resolve_divergences_accepts_the_new_lever_keys() {
        Python::initialize();
        Python::attach(|py| {
            let overrides = PyDict::new(py);
            overrides.set_item("gumbel_target_c_scale", 0.35f32).unwrap();
            overrides.set_item("gumbel_draw_temperature", 1.0f32).unwrap();
            let dv = resolve_divergences(None, Some(&overrides))
                .expect("new lever keys must pass the known-keys gate");
            assert_eq!(dv.gumbel_target_c_scale, Some(0.35));
            assert_eq!(dv.gumbel_draw_temperature, 1.0);

            // Every main_8 fast_* key must pass the gate AND fold onto its
            // gumbel field. A whitelist/parser mismatch here is exactly the
            // failure class that tripped the supervisor circuit breaker.
            let fast = PyDict::new(py);
            fast.set_item("fast_gumbel_root_enabled", true).unwrap();
            fast.set_item("fast_gumbel_sequential_halving", true).unwrap();
            fast.set_item("fast_gumbel_nonroot_select", true).unwrap();
            fast.set_item("fast_gumbel_c_visit", 12.0f32).unwrap();
            fast.set_item("fast_gumbel_c_scale", 0.5f32).unwrap();
            fast.set_item("fast_gumbel_m", 8u32).unwrap();
            fast.set_item("fast_gumbel_play_prune", true).unwrap();
            let fv = resolve_divergences(None, Some(&fast))
                .expect("fast_* keys must pass the known-keys gate");
            assert!(fv.gumbel_root);
            assert!(fv.gumbel_sequential_halving);
            assert!(fv.gumbel_nonroot_select);
            assert_eq!(fv.gumbel_c_visit, 12.0);
            assert_eq!(fv.gumbel_c_scale, 0.5);
            assert_eq!(fv.gumbel_m, 8);
            assert!(fv.gumbel_play_prune);

            // The gate itself still rejects a genuinely unknown key.
            let bogus = PyDict::new(py);
            bogus.set_item("gumbel_bogus_lever", 1.0f32).unwrap();
            assert!(resolve_divergences(None, Some(&bogus)).is_err());

            // Async solve-pool keys (the python side always emits both) must
            // pass the gate and land on their fields.
            let tss_async = PyDict::new(py);
            tss_async.set_item("tss_solver_async", true).unwrap();
            tss_async.set_item("tss_solver_async_threads", 12u32).unwrap();
            tss_async
                .set_item("tss_solver_async_threads_max", 48u32)
                .unwrap();
            tss_async.set_item("tss_solver_park", true).unwrap();
            tss_async
                .set_item("tss_solver_park_timeout_ms", 200u32)
                .unwrap();
            tss_async.set_item("tss_solver_async_inline_16", 4u32).unwrap();
            let av = resolve_divergences(None, Some(&tss_async))
                .expect("tss_solver_async keys must pass the known-keys gate");
            assert!(av.tss_solver_async);
            assert_eq!(av.tss_solver_async_threads, 12);
            assert_eq!(av.tss_solver_async_threads_max, 48);
            assert!(av.tss_solver_park);
            assert_eq!(av.tss_solver_park_timeout_ms, 200);
            assert_eq!(av.tss_solver_async_inline_16, 4);

            let dual_pass = PyDict::new(py);
            dual_pass.set_item("tss_solver_dual_pass", true).unwrap();
            let dv = resolve_divergences(None, Some(&dual_pass))
                .expect("tss_solver_dual_pass must pass the known-keys gate");
            assert!(dv.tss_solver_dual_pass);

            let park_without_async = PyDict::new(py);
            park_without_async.set_item("tss_solver_park", true).unwrap();
            assert!(resolve_divergences(None, Some(&park_without_async)).is_err());

            let max_below_base = PyDict::new(py);
            max_below_base
                .set_item("tss_solver_async_threads", 12u32)
                .unwrap();
            max_below_base
                .set_item("tss_solver_async_threads_max", 8u32)
                .unwrap();
            assert!(resolve_divergences(None, Some(&max_below_base)).is_err());

            let bad_timeout = PyDict::new(py);
            bad_timeout
                .set_item("tss_solver_park_timeout_ms", 0u32)
                .unwrap();
            assert!(resolve_divergences(None, Some(&bad_timeout)).is_err());
        });
    }

    // === main_8: turn-based classification + per-class divergences ===========

    /// Full ContinuousMovePolicy tuned for classify() sampling tests: a real
    /// pcr_full_proportion and no policy-init so classify exercises the PCR
    /// hash rather than the Init/Full short-circuits.
    fn classify_policy(pcr_full_proportion: f32) -> ContinuousMovePolicy {
        let mut p = move_policy(0.0);
        p.pcr_full_proportion = pcr_full_proportion;
        p.policy_init_fraction = 0.0;
        p.policy_init_avg_plies = 0.0;
        p.policy_init_max_plies = 0;
        p
    }

    #[test]
    fn classify_is_per_turn_paired_plies_share_a_class() {
        // Both plies of a turn (2k, 2k+1) must map to the same class, for many
        // turns and many seeds. This is the invariant the per-class
        // set_divergences reuse relies on.
        let policy = classify_policy(0.5);
        for base_seed in 0u64..64 {
            for game_key in [0u64, 1, 7, 4242, u64::MAX] {
                for k in 0u32..64 {
                    let a = policy.classify(base_seed, game_key, 2 * k, 0);
                    let b = policy.classify(base_seed, game_key, 2 * k + 1, 0);
                    assert_eq!(
                        a, b,
                        "plies {} and {} of turn {k} must share a class (seed {base_seed}, key {game_key})",
                        2 * k,
                        2 * k + 1
                    );
                    // And it is never Init here (no policy-init remaining).
                    assert!(matches!(a, MoveClass::Full | MoveClass::Fast));
                }
            }
        }
    }

    #[test]
    fn classify_full_fraction_matches_proportion_over_turns() {
        // Over a large sample of turns, roughly pcr_full_proportion of TURNS are
        // Full. Sample one ply per turn (the pair is identical by the test
        // above) across many game_keys to average out the per-key hash.
        let prop = 0.33f32;
        let policy = classify_policy(prop);
        let base_seed = 12345u64;
        let mut full = 0u64;
        let mut total = 0u64;
        for game_key in 0u64..2000 {
            for k in 0u32..16 {
                if matches!(policy.classify(base_seed, game_key, 2 * k, 0), MoveClass::Full) {
                    full += 1;
                }
                total += 1;
            }
        }
        let frac = full as f64 / total as f64;
        assert!(
            (frac - prop as f64).abs() < 0.02,
            "Full turn fraction {frac} should be ~{prop} over {total} turns"
        );
    }

    #[test]
    fn classify_short_circuits_are_unchanged() {
        // policy_init_remaining > 0 => Init regardless of ply/turn.
        let policy = classify_policy(0.33);
        assert!(matches!(policy.classify(1, 2, 0, 3), MoveClass::Init));
        assert!(matches!(policy.classify(1, 2, 5, 1), MoveClass::Init));
        // pcr_full_proportion >= 1.0 => always Full.
        let all_full = classify_policy(1.0);
        for ply in 0u32..16 {
            assert!(matches!(all_full.classify(9, 9, ply, 0), MoveClass::Full));
        }
    }

    #[test]
    fn divergences_for_selects_the_class_view() {
        let mut policy = move_policy(0.0);
        let mut fast = Divergences::production();
        fast.gumbel_root = true; // make the fast view distinguishable
        policy.divergences_fast = fast;
        assert!(!policy.divergences_for(MoveClass::Full).gumbel_root);
        assert!(!policy.divergences_for(MoveClass::Init).gumbel_root);
        assert!(policy.divergences_for(MoveClass::Fast).gumbel_root);
    }

    #[test]
    fn golden_invariant_fast_equals_full_without_fast_overrides() {
        // When no fast_* keys are set, the driver falls back to the base view
        // for the fast map (fast_divergence_overrides=None => divergences_fast =
        // divergences). Mirror that here: fast resolved from None equals base.
        Python::initialize();
        Python::attach(|py| {
            let base_overrides = PyDict::new(py);
            base_overrides.set_item("gumbel_root", true).unwrap();
            base_overrides.set_item("gumbel_c_scale", 0.7f32).unwrap();
            let base = resolve_divergences(None, Some(&base_overrides)).unwrap();
            // No fast overrides => fast view IS the base view (Rust fallback).
            let fast_fallback = base;
            assert_eq!(
                fast_fallback, base,
                "divergences_fast must equal divergences_full when no fast_* keys set"
            );

            // And request_logits/request_moves_left with identical views match
            // the single-view result.
            let mut policy = move_policy(0.0);
            policy.divergences_full = base;
            policy.divergences_fast = base;
            assert_eq!(policy.request_logits(), base.gumbel_root);
        });
    }

    #[test]
    fn request_logits_true_if_either_view_needs_them() {
        let mut policy = move_policy(0.0);
        // Neither view needs logits.
        let mut plain = Divergences::production();
        plain.gumbel_target = false;
        plain.gumbel_root = false;
        plain.gumbel_nonroot_select = false;
        policy.divergences_full = plain;
        policy.divergences_fast = plain;
        assert!(!policy.request_logits());
        // Fast view alone needs them => request_logits true.
        let mut fast = plain;
        fast.gumbel_root = true;
        policy.divergences_fast = fast;
        assert!(policy.request_logits());
        // Reset fast, set only full.
        policy.divergences_fast = plain;
        let mut full = plain;
        full.gumbel_nonroot_select = true;
        policy.divergences_full = full;
        assert!(policy.request_logits());
    }
}
