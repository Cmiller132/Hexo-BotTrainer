//! tss_core.rs — typed Threat-Space Search results: the soundness seam between
//! proof producers and the search tree (docs/PLAN_TSS_DEEPENING.md §2).
//!
//! The tree is the poison channel: any hard ±1 reaching `backup_virtual`
//! propagates into the soft-policy / cell_q / stvalue training targets with no
//! head involvement. This module therefore types the seam: `HardValue` is the
//! only TSS value `backup_virtual` may receive, its field is private, and the
//! only constructors are the certified producers defined HERE:
//!
//!   1. `solve_leaf_lambda1` — the sound one-turn (λ¹) verdict, a verbatim
//!      wrapper of `threats::analyze().verdict()` (sound post-opening; see
//!      threats_shared.rs header and the design doc §1).
//!   2. `hard_value_from_verified` — deep proofs, minted only inside this
//!      module after an independent certificate verifier accepts the claim
//!      (Stage 4; the `DeepSolve` implementation itself can never mint one).
//!
//! Code outside this module cannot fabricate a `HardValue`; "deep results
//! degrade to net-eval until verified" is structural, not a runtime flag.

use hexo_engine::HexoState as RustHexoState;

use crate::threats_shared as threats;

/// Three-valued solve status. UNKNOWN must propagate — a capped / exhausted /
/// unproven solve is UNKNOWN, never a verdict (§2.4). `Loss` is a claim that
/// the SIDE TO MOVE at the solved state loses; for deep solvers that requires
/// the dual certificate (a proven opponent winning strategy, §2.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofStatus {
    Win,
    Loss,
    Unknown,
}

impl ProofStatus {
    /// The backup value for the side to move, when proven.
    #[inline]
    pub fn value(self) -> Option<f32> {
        match self {
            ProofStatus::Win => Some(1.0),
            ProofStatus::Loss => Some(-1.0),
            ProofStatus::Unknown => None,
        }
    }
}

/// A value certified to enter `backup_virtual` as a hard ±1 for the side to
/// move at the solved state. Sealed: the field is private and the only
/// constructors live in this module (the two certified producers above).
#[derive(Clone, Copy, Debug)]
pub struct HardValue(f32);

impl HardValue {
    /// The certified backup value (±1, side-to-move perspective).
    #[inline]
    pub fn value(self) -> f32 {
        self.0
    }

    #[inline]
    pub fn status(self) -> ProofStatus {
        if self.0 > 0.0 {
            ProofStatus::Win
        } else {
            ProofStatus::Loss
        }
    }
}

/// Certified producer #1 — the sound λ¹ verdict for the side to move.
/// Verbatim wrapper of `threats::analyze().verdict()`: `Some(+1)` proven win
/// within the turn budget, `Some(-1)` proven one-turn forced loss, `None`
/// (no proof) stays `None` — the net evaluates the leaf.
#[inline]
pub fn solve_leaf_lambda1(state: &RustHexoState) -> Option<HardValue> {
    threats::analyze(state).verdict().map(HardValue)
}

/// Typed status view of the λ¹ solve, for consumers that classify rather than
/// back up (the root guard / recorded-target classifier).
#[inline]
pub fn lambda1_status(state: &RustHexoState) -> ProofStatus {
    match threats::analyze(state).verdict() {
        Some(v) if v > 0.0 => ProofStatus::Win,
        Some(_) => ProofStatus::Loss,
        None => ProofStatus::Unknown,
    }
}

// === Deep-solver seam (Stage 3/4; frozen for the delegated build) ===========

/// Deterministic solve budget. No wall clock on any path that can mint a hard
/// value: a timed-out solve is UNKNOWN by construction because it never
/// completes a certificate (§2.6). Caps binding must yield UNKNOWN, never a
/// verdict.
#[derive(Clone, Copy, Debug)]
pub struct SolveCaps {
    /// Maximum solver node expansions for this solve.
    pub node_cap: u64,
    /// Hard ceiling on transposition-table + cache bytes (the WSL host kills
    /// unbounded growth; §11). The solver must account and stay under it.
    pub tt_bytes_cap: usize,
    /// Absolute placement index of the semantic proof deadline.  This is
    /// deliberately distinct from `node_cap` and the structural depth guard:
    /// zone obligations and typed leaf resolutions are statements about game
    /// plies, not about how much search work happened to be affordable.
    pub semantic_horizon: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZoneSearchCaps {
    pub enabled: bool,
    pub stale_area_filter: bool,
    pub count2_threshold: bool,
    pub pair_commutation: bool,
}

/// Uniform D11/T4 seed-band radius. L9' bounds the first protected occupation
/// chain by `8(B-1)`; `d` is the verifier-checked admissible local B wrapper.
/// Keeping this theorem arithmetic in the shared contract module preserves
/// finder/verifier separation while giving both sides one production value.
#[inline]
pub(crate) fn seed_band_radius(d: u32) -> i32 {
    i32::try_from(d.saturating_sub(1).saturating_mul(8)).unwrap_or(i32::MAX)
}

/// Which root-perspective hard result a caller wants the deep solver to seek.
///
/// This is deliberately separate from [`SolveCaps`] so existing callers using
/// its two-field literal remain source-compatible.  `DeepSolve::solve` keeps
/// the historical [`SolveGoal::Both`] behavior; reusable solver callers may
/// request one side explicitly and give that attempt the whole node budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveGoal {
    Win,
    Loss,
    Both,
}

/// Per-solve diagnostics (telemetry only — never consulted for soundness).
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ClosureDebtStats {
    pub(crate) pairs_evaluated: u64,
    pub(crate) pairs_accepted: u64,
    pub(crate) pairs_retained: u64,
    pub(crate) pairs_selected: u64,
    pub(crate) pairs_linked: u64,
    pub(crate) pairs_expanded: u64,
    pub(crate) winning_choice_nodes: u64,
    pub(crate) winning_rank_bins: [u64; 8],
    pub(crate) reveal_pair_evaluated: u64,
    pub(crate) reveal_pair_prefix: u64,
    pub(crate) pair_generation_nanos: u64,
    pub(crate) gate_build_nanos: u64,
    pub(crate) second_candidate_nanos: u64,
    pub(crate) pair_evaluation_nanos: u64,
    pub(crate) dedup_nanos: u64,
    pub(crate) avoidable_second_candidate_nanos: u64,
    pub(crate) avoidable_pair_evaluation_nanos: u64,
    pub(crate) avoidable_dedup_nanos: u64,
    /// R-OS3 paired reveal-prefix counterfactual. Order 0 is historical;
    /// order 1 is `(zone_bound, historical_ordinal)`.
    pub(crate) reveal_proven_pair_nodes: u64,
    pub(crate) reveal_rank_bins: [[u64; 8]; 2],
    pub(crate) reveal_evaluation_rank_bins: [[u64; 8]; 2],
    pub(crate) reveal_total_evaluated: [u64; 2],
    pub(crate) reveal_prefix_evaluated: [u64; 2],
    pub(crate) reveal_total_expanded: [u64; 2],
    pub(crate) reveal_avoidable_expanded: [u64; 2],
    pub(crate) reveal_avoidable_second_candidate_nanos: [u64; 2],
    pub(crate) reveal_avoidable_pair_evaluation_nanos: [u64; 2],
    pub(crate) reveal_avoidable_dedup_nanos: [u64; 2],
    /// Test-only time spent deriving and bucketing the offline key. This is
    /// reported separately and is never counted as avoidable classifier work.
    pub(crate) reveal_analysis_nanos: u64,
}

/// Test-only accounting for the prior-scale threshold hunt. These values are
/// observational only and are never consulted by the search.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ThresholdScaleStats {
    pub(crate) recursive_node_visits: u64,
    pub(crate) expanded_node_revisits: u64,
    pub(crate) threshold_cross_returns: u64,
    pub(crate) same_parent_reselections: u64,
    pub(crate) sibling_switches: u64,
    pub(crate) residencies: u64,
    pub(crate) residency_expansions: u64,
    /// Expansion deltas in bins 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33+.
    pub(crate) residency_expansion_bins: [u64; 8],
    /// First indexed-TT admission refusal as
    /// (expansion clock, retained arena entries, indexed bytes).
    pub(crate) first_admission_refusal: Option<(u64, u64, u64)>,
    /// Selection-score gap bins are: no sibling; selected better by 33+,
    /// 17-32, 9-16, 5-8, 3-4, 2, 1; tie; selected worse by 1, 2, 3-4, 5+.
    /// Each expansion is charged to its immediate parent's selection snapshot.
    pub(crate) choice_gap_expansions_pre_saturation: [u64; 13],
    pub(crate) choice_gap_expansions_post_saturation: [u64; 13],
    pub(crate) universal_gap_expansions_pre_saturation: [u64; 13],
    pub(crate) universal_gap_expansions_post_saturation: [u64; 13],
    /// Root (or otherwise parentless) expansions, split at first refusal.
    pub(crate) unclassified_expansions_pre_saturation: u64,
    pub(crate) unclassified_expansions_post_saturation: u64,
    /// PN and DN inherited-threshold sentinel hits and strict clamp events.
    pub(crate) sentinel_inherited_threshold_hits: [u64; 2],
    pub(crate) sentinel_inherited_threshold_clamps: [u64; 2],
    /// `value + delta` expressions that reached the sentinel, and those whose
    /// unclamped result was strictly above it.
    pub(crate) sentinel_increment_hits: u64,
    pub(crate) sentinel_increment_clamps: u64,
    /// Choice-DN and Universal-PN branch sums that reached the sentinel.
    pub(crate) sentinel_sum_hits: [u64; 2],
    /// Exclusive time in `WidePnSearch::work`, with expansion and recursive
    /// child time paused so every wall interval is counted at most once.
    pub(crate) descent_nanos: u64,
    /// Apply/undo time on descent edges; this is a subset of `descent_nanos`.
    pub(crate) state_apply_undo_nanos: u64,
}

#[cfg(test)]
impl ThresholdScaleStats {
    pub(crate) fn merge(&mut self, other: Self) {
        self.recursive_node_visits = self
            .recursive_node_visits
            .saturating_add(other.recursive_node_visits);
        self.expanded_node_revisits = self
            .expanded_node_revisits
            .saturating_add(other.expanded_node_revisits);
        self.threshold_cross_returns = self
            .threshold_cross_returns
            .saturating_add(other.threshold_cross_returns);
        self.same_parent_reselections = self
            .same_parent_reselections
            .saturating_add(other.same_parent_reselections);
        self.sibling_switches = self.sibling_switches.saturating_add(other.sibling_switches);
        self.residencies = self.residencies.saturating_add(other.residencies);
        self.residency_expansions = self
            .residency_expansions
            .saturating_add(other.residency_expansions);
        for (target, value) in self
            .residency_expansion_bins
            .iter_mut()
            .zip(other.residency_expansion_bins)
        {
            *target = target.saturating_add(value);
        }
        if self.first_admission_refusal.is_none() {
            self.first_admission_refusal = other.first_admission_refusal;
        }
        for (target_bins, source_bins) in [
            (
                &mut self.choice_gap_expansions_pre_saturation,
                other.choice_gap_expansions_pre_saturation,
            ),
            (
                &mut self.choice_gap_expansions_post_saturation,
                other.choice_gap_expansions_post_saturation,
            ),
            (
                &mut self.universal_gap_expansions_pre_saturation,
                other.universal_gap_expansions_pre_saturation,
            ),
            (
                &mut self.universal_gap_expansions_post_saturation,
                other.universal_gap_expansions_post_saturation,
            ),
        ] {
            for (target, value) in target_bins.iter_mut().zip(source_bins) {
                *target = target.saturating_add(value);
            }
        }
        self.unclassified_expansions_pre_saturation = self
            .unclassified_expansions_pre_saturation
            .saturating_add(other.unclassified_expansions_pre_saturation);
        self.unclassified_expansions_post_saturation = self
            .unclassified_expansions_post_saturation
            .saturating_add(other.unclassified_expansions_post_saturation);
        for (target, value) in self
            .sentinel_inherited_threshold_hits
            .iter_mut()
            .zip(other.sentinel_inherited_threshold_hits)
        {
            *target = target.saturating_add(value);
        }
        for (target, value) in self
            .sentinel_inherited_threshold_clamps
            .iter_mut()
            .zip(other.sentinel_inherited_threshold_clamps)
        {
            *target = target.saturating_add(value);
        }
        self.sentinel_increment_hits = self
            .sentinel_increment_hits
            .saturating_add(other.sentinel_increment_hits);
        self.sentinel_increment_clamps = self
            .sentinel_increment_clamps
            .saturating_add(other.sentinel_increment_clamps);
        for (target, value) in self
            .sentinel_sum_hits
            .iter_mut()
            .zip(other.sentinel_sum_hits)
        {
            *target = target.saturating_add(value);
        }
        self.descent_nanos = self.descent_nanos.saturating_add(other.descent_nanos);
        self.state_apply_undo_nanos = self
            .state_apply_undo_nanos
            .saturating_add(other.state_apply_undo_nanos);
    }
}

#[cfg(test)]
impl ClosureDebtStats {
    pub(crate) fn merge(&mut self, other: Self) {
        self.pairs_evaluated = self.pairs_evaluated.saturating_add(other.pairs_evaluated);
        self.pairs_accepted = self.pairs_accepted.saturating_add(other.pairs_accepted);
        self.pairs_retained = self.pairs_retained.saturating_add(other.pairs_retained);
        self.pairs_selected = self.pairs_selected.saturating_add(other.pairs_selected);
        self.pairs_linked = self.pairs_linked.saturating_add(other.pairs_linked);
        self.pairs_expanded = self.pairs_expanded.saturating_add(other.pairs_expanded);
        self.winning_choice_nodes = self
            .winning_choice_nodes
            .saturating_add(other.winning_choice_nodes);
        for (target, value) in self
            .winning_rank_bins
            .iter_mut()
            .zip(other.winning_rank_bins)
        {
            *target = target.saturating_add(value);
        }
        self.reveal_pair_evaluated = self
            .reveal_pair_evaluated
            .saturating_add(other.reveal_pair_evaluated);
        self.reveal_pair_prefix = self
            .reveal_pair_prefix
            .saturating_add(other.reveal_pair_prefix);
        self.pair_generation_nanos = self
            .pair_generation_nanos
            .saturating_add(other.pair_generation_nanos);
        self.gate_build_nanos = self.gate_build_nanos.saturating_add(other.gate_build_nanos);
        self.second_candidate_nanos = self
            .second_candidate_nanos
            .saturating_add(other.second_candidate_nanos);
        self.pair_evaluation_nanos = self
            .pair_evaluation_nanos
            .saturating_add(other.pair_evaluation_nanos);
        self.dedup_nanos = self.dedup_nanos.saturating_add(other.dedup_nanos);
        self.avoidable_second_candidate_nanos = self
            .avoidable_second_candidate_nanos
            .saturating_add(other.avoidable_second_candidate_nanos);
        self.avoidable_pair_evaluation_nanos = self
            .avoidable_pair_evaluation_nanos
            .saturating_add(other.avoidable_pair_evaluation_nanos);
        self.avoidable_dedup_nanos = self
            .avoidable_dedup_nanos
            .saturating_add(other.avoidable_dedup_nanos);
        self.reveal_proven_pair_nodes = self
            .reveal_proven_pair_nodes
            .saturating_add(other.reveal_proven_pair_nodes);
        for order in 0..2 {
            for bin in 0..8 {
                self.reveal_rank_bins[order][bin] = self.reveal_rank_bins[order][bin]
                    .saturating_add(other.reveal_rank_bins[order][bin]);
                self.reveal_evaluation_rank_bins[order][bin] = self.reveal_evaluation_rank_bins
                    [order][bin]
                    .saturating_add(other.reveal_evaluation_rank_bins[order][bin]);
            }
            self.reveal_total_evaluated[order] = self.reveal_total_evaluated[order]
                .saturating_add(other.reveal_total_evaluated[order]);
            self.reveal_prefix_evaluated[order] = self.reveal_prefix_evaluated[order]
                .saturating_add(other.reveal_prefix_evaluated[order]);
            self.reveal_total_expanded[order] = self.reveal_total_expanded[order]
                .saturating_add(other.reveal_total_expanded[order]);
            self.reveal_avoidable_expanded[order] = self.reveal_avoidable_expanded[order]
                .saturating_add(other.reveal_avoidable_expanded[order]);
            self.reveal_avoidable_second_candidate_nanos[order] = self
                .reveal_avoidable_second_candidate_nanos[order]
                .saturating_add(other.reveal_avoidable_second_candidate_nanos[order]);
            self.reveal_avoidable_pair_evaluation_nanos[order] = self
                .reveal_avoidable_pair_evaluation_nanos[order]
                .saturating_add(other.reveal_avoidable_pair_evaluation_nanos[order]);
            self.reveal_avoidable_dedup_nanos[order] = self.reveal_avoidable_dedup_nanos[order]
                .saturating_add(other.reveal_avoidable_dedup_nanos[order]);
        }
        self.reveal_analysis_nanos = self
            .reveal_analysis_nanos
            .saturating_add(other.reveal_analysis_nanos);
    }
}

/// Engine / certificate-schema version stamped into per-run telemetry so
/// minted certificates are attributable to the engine that produced them
/// (PLAN_TSS_MCTS_INTEGRATION.md §3, C1 one-engine principle). `1` = the
/// original narrow Stage-3 trainer solver; `2` = the campaign wide
/// `vcf_pair_complete` engine adopted wholesale in the V0 port (R-FIX1
/// zone-clock fix, lazy defender frontier, interior census gate, incremental
/// defender enumeration, cap-resume, extended-contract zones P0–P3). Bump on
/// any change to the minting engine or the certificate grammar.
pub const TSS_CERT_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, Default)]
pub struct SolveStats {
    pub nodes: u64,
    pub expansions: u64,
    pub tt_hits: u64,
    pub tt_entries: u64,
    pub peak_tt_bytes: u64,
    /// Lines the semantic-horizon deadline refused while still alive (a live
    /// descent, typed-leaf resolution, or completion past the deadline).
    /// Distinguishes depth-bound Unknowns from structural ones ahead of any
    /// horizon-ladder decision. Telemetry only: never a value, never a
    /// backup — a heuristic escalation trigger under contract rule 7.
    pub horizon_cuts: u64,
    /// Subset of `horizon_cuts` that fell at a defender-to-move node (the
    /// opponent still branching, i.e. before the fully-forced `k == B`
    /// boundary). Feeds `deep_kb_death` on the horizon-ladder tall pass: the
    /// signal that Group-2 zone consumption would matter.
    pub kb_death_cuts: u64,
    /// Direct-map slot replacements in the solve-local TT. These are cache
    /// evictions, not proof-semantic events.
    pub tt_evictions: u64,
    /// TT/index insertions refused because the caller's byte cap was full.
    pub tt_admission_rejections: u64,
    /// Exact-key positive-fragment queries made by the wide solver.
    pub fragment_lookups: u64,
    /// Queries that passed full-key, claimant, horizon, and depth checks.
    pub fragment_hits: u64,
    /// Shared fragment roots actually imported into the returned certificate.
    pub fragment_imports: u64,
    /// Resident entries after this solve (telemetry only).
    pub fragment_store_entries: u64,
    /// Resident byte-accounted fragment-store charge after this solve.
    pub fragment_store_bytes: u64,
    pub interior_gate_evaluations: u64,
    pub interior_gate_dismissals: u64,
    pub interior_gate_nanos: u64,
    #[cfg(test)]
    pub(crate) stage_refreshes: u64,
    #[cfg(test)]
    pub(crate) live_ge3_seed_scans: u64,
    #[cfg(test)]
    pub(crate) live_ge3_seed_nanos: u64,
    #[cfg(test)]
    pub(crate) closure_debt: ClosureDebtStats,
    #[cfg(test)]
    pub(crate) threshold_scale: ThresholdScaleStats,
}

impl SolveStats {
    /// Fold one solver attempt into a solve-level aggregate. Additive counters
    /// sum, high-water marks take their maximum, and resident-store gauges
    /// describe the most recently completed attempt.
    pub(crate) fn merge(&mut self, part: Self) {
        self.nodes = self.nodes.saturating_add(part.nodes);
        self.expansions = self.expansions.saturating_add(part.expansions);
        self.tt_hits = self.tt_hits.saturating_add(part.tt_hits);
        self.tt_entries = self.tt_entries.max(part.tt_entries);
        self.peak_tt_bytes = self.peak_tt_bytes.max(part.peak_tt_bytes);
        self.horizon_cuts = self.horizon_cuts.saturating_add(part.horizon_cuts);
        self.kb_death_cuts = self.kb_death_cuts.saturating_add(part.kb_death_cuts);
        self.tt_evictions = self.tt_evictions.saturating_add(part.tt_evictions);
        self.tt_admission_rejections = self
            .tt_admission_rejections
            .saturating_add(part.tt_admission_rejections);
        self.fragment_lookups = self.fragment_lookups.saturating_add(part.fragment_lookups);
        self.fragment_hits = self.fragment_hits.saturating_add(part.fragment_hits);
        self.fragment_imports = self.fragment_imports.saturating_add(part.fragment_imports);
        self.fragment_store_entries = part.fragment_store_entries;
        self.fragment_store_bytes = part.fragment_store_bytes;
        self.interior_gate_evaluations = self
            .interior_gate_evaluations
            .saturating_add(part.interior_gate_evaluations);
        self.interior_gate_dismissals = self
            .interior_gate_dismissals
            .saturating_add(part.interior_gate_dismissals);
        self.interior_gate_nanos = self
            .interior_gate_nanos
            .saturating_add(part.interior_gate_nanos);
        #[cfg(test)]
        {
            self.stage_refreshes = self.stage_refreshes.saturating_add(part.stage_refreshes);
            self.live_ge3_seed_scans = self
                .live_ge3_seed_scans
                .saturating_add(part.live_ge3_seed_scans);
            self.live_ge3_seed_nanos = self
                .live_ge3_seed_nanos
                .saturating_add(part.live_ge3_seed_nanos);
            self.closure_debt.merge(part.closure_debt);
            self.threshold_scale.merge(part.threshold_scale);
        }
    }
}

/// A deep solve's outcome: a typed status, an optional replayable certificate
/// (present for every Win/Loss claim), and diagnostics. The certificate type
/// is solver-defined; the search consumes only `status` — and only via
/// `hard_value_from_verified`, never directly.
pub struct DeepResult<C> {
    pub status: ProofStatus,
    pub cert: Option<C>,
    pub stats: SolveStats,
}

/// The deep-solver interface the Stage-3 delegated build implements
/// (docs/TSS_SOLVER_SPEC.md freezes the semantics: df-pn, exhaustive-with-
/// instant-dispatch AND nodes, threat-creating OR restriction, dual LOSS
/// certificates, UNKNOWN propagation, full-canonical-key cache equality).
pub trait DeepSolve {
    type Cert;
    fn solve(&mut self, state: &RustHexoState, caps: &SolveCaps) -> DeepResult<Self::Cert>;
}

/// The independent certificate verifier (§2.2): replays a certificate against
/// the state and accepts or rejects the claimed status. Implemented as its own
/// module sharing only engine primitives with the solver, so a solver bug is
/// not mirrored in its checker.
pub trait CertVerify {
    type Cert;
    fn verify(&self, state: &RustHexoState, cert: &Self::Cert, claimed: ProofStatus) -> bool;
}

/// Certified producer #2 — deep proofs, minted ONLY here and only after the
/// independent verifier accepts the certificate for this exact state. A
/// rejected or missing certificate yields `None` (the caller must degrade to
/// net-eval AND bump the fatal `verify_failed` telemetry counter).
///
/// The verifier parameter is the CONCRETE `TssVerifier` — not the `CertVerify`
/// trait — so no sibling module can mint a `HardValue` through an
/// always-accepting stand-in (Codex review, mint sealing). The generic
/// trait-driven variant survives as a test-only helper below.
pub fn hard_value_from_verified(
    verifier: &crate::tss_verify::TssVerifier,
    state: &RustHexoState,
    result: &DeepResult<crate::tss_verify::TssCertificate>,
) -> Option<HardValue> {
    hard_value_from_verify_impl(verifier, state, result)
}

/// Trait-generic mint used by `hard_value_from_verified` and (directly) by
/// tests exercising the accept/reject contract with stub verifiers. Private:
/// production callers cannot name it with a stub verifier.
fn hard_value_from_verify_impl<V, C>(
    verifier: &V,
    state: &RustHexoState,
    result: &DeepResult<C>,
) -> Option<HardValue>
where
    V: CertVerify<Cert = C>,
{
    let value = result.status.value()?;
    let cert = result.cert.as_ref()?;
    if verifier.verify(state, cert, result.status) {
        Some(HardValue(value))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{apply_placement, HexCoord, Placement};

    /// Deterministic xorshift for reproducible random playouts (no rand dep).
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// Play a random legal game for `plies` placements (rejection-sampled
    /// coordinates near the origin; apply_placement enforces legality),
    /// returning every intermediate non-terminal state.
    fn random_states(seed: u64, plies: usize) -> Vec<RustHexoState> {
        let mut rng = XorShift(seed | 1);
        let mut state = RustHexoState::new();
        let mut out = vec![state.clone()];
        for _ in 0..plies {
            let mut placed = false;
            for _try in 0..200 {
                let q = (rng.next() % 17) as i16 - 8;
                let r = (rng.next() % 17) as i16 - 8;
                let mut child = state.clone();
                match apply_placement(
                    &mut child,
                    Placement {
                        coord: HexCoord { q, r },
                    },
                ) {
                    Ok(res) => {
                        if res.outcome.is_some() {
                            return out; // terminal: stop, later states don't exist
                        }
                        state = child;
                        out.push(state.clone());
                        placed = true;
                        break;
                    }
                    Err(_) => continue,
                }
            }
            if !placed {
                break;
            }
        }
        out
    }

    /// The typed wrapper is verbatim: for every reachable state its HardValue
    /// equals the raw λ¹ verdict, and the status view agrees.
    #[test]
    fn lambda1_wrapper_is_verbatim() {
        let mut checked = 0usize;
        for seed in 1..40u64 {
            for state in random_states(seed * 0x9E37_79B9, 60) {
                let raw = threats::analyze(&state).verdict();
                let typed = solve_leaf_lambda1(&state).map(HardValue::value);
                assert_eq!(raw, typed);
                let status = lambda1_status(&state);
                match raw {
                    Some(v) if v > 0.0 => assert_eq!(status, ProofStatus::Win),
                    Some(_) => assert_eq!(status, ProofStatus::Loss),
                    None => assert_eq!(status, ProofStatus::Unknown),
                }
                if let Some(hv) = solve_leaf_lambda1(&state) {
                    assert_eq!(hv.status().value(), Some(hv.value()));
                    assert!(hv.value() == 1.0 || hv.value() == -1.0);
                }
                checked += 1;
            }
        }
        assert!(checked > 500, "random-state corpus too small: {checked}");
    }

    /// Lemma L1 (instant dispatch — the interior forced-move guard's soundness
    /// argument, PLAN_TSS_DEEPENING.md §0/§3): at any reachable state with
    /// verdict None, live opponent threats, and min_hitting_set == B, every
    /// legal move OUTSIDE tactical_cells() loses by the one-ply λ¹ argument.
    /// Exercised over the random corpus + the guarantee that the dropped move
    /// can never be an immediate win (verdict None excludes own count-4/5).
    #[test]
    fn lemma_l1_every_nontactical_move_at_k_eq_b_is_lost() {
        let mut forced_nodes = 0usize;
        let mut dropped_checked = 0usize;
        for seed in 1..120u64 {
            for state in random_states(seed.wrapping_mul(0xD134_2543_DE82_EF95), 70) {
                let a = threats::analyze(&state);
                if a.own_win_now || a.opp_threat_count == 0 {
                    continue;
                }
                if a.min_hitting_set != Some(a.b) {
                    continue;
                }
                forced_nodes += 1;
                let mover = state.current_player();
                let tactical: Vec<HexCoord> = threats::tactical_cells(&state);
                // Enumerate legal moves by rejection over the covering box:
                // random_states places within ±8, legality reaches 8 further.
                for q in -16..=16i16 {
                    for r in -16..=16i16 {
                        let coord = HexCoord { q, r };
                        if tactical.contains(&coord) {
                            continue;
                        }
                        let mut child = state.clone();
                        let Ok(res) = apply_placement(&mut child, Placement { coord }) else {
                            continue;
                        };
                        // verdict None at the parent ⇒ no own count-4/5 ⇒ a
                        // single placement can never complete our 6.
                        assert!(
                            res.outcome.is_none(),
                            "a non-tactical move ended the game at a verdict-None node"
                        );
                        let v = threats::analyze(&child)
                            .verdict()
                            .expect("L1: non-tactical child must be λ¹-decided");
                        let ours = if child.current_player() == mover {
                            v
                        } else {
                            -v
                        };
                        assert_eq!(
                            ours, -1.0,
                            "L1 violated: non-tactical move ({q},{r}) at k==B is not a \
                             proven loss (seed {seed})"
                        );
                        dropped_checked += 1;
                    }
                }
            }
        }
        assert!(
            forced_nodes > 20 && dropped_checked > 2000,
            "corpus too thin: {forced_nodes} forced nodes / {dropped_checked} dropped moves"
        );
    }

    /// ProofStatus::value is the exact backup mapping.
    #[test]
    fn proof_status_values() {
        assert_eq!(ProofStatus::Win.value(), Some(1.0));
        assert_eq!(ProofStatus::Loss.value(), Some(-1.0));
        assert_eq!(ProofStatus::Unknown.value(), None);
    }

    /// The deep producer refuses to mint without an accepted certificate:
    /// Unknown never mints; a rejecting verifier never mints; an accepting
    /// verifier mints the exact status value.
    #[test]
    fn deep_producer_gated_by_verifier() {
        struct Accept;
        struct Reject;
        impl CertVerify for Accept {
            type Cert = ();
            fn verify(&self, _s: &RustHexoState, _c: &(), _st: ProofStatus) -> bool {
                true
            }
        }
        impl CertVerify for Reject {
            type Cert = ();
            fn verify(&self, _s: &RustHexoState, _c: &(), _st: ProofStatus) -> bool {
                false
            }
        }
        let state = RustHexoState::new();
        let win = DeepResult {
            status: ProofStatus::Win,
            cert: Some(()),
            stats: SolveStats::default(),
        };
        let unknown = DeepResult::<()> {
            status: ProofStatus::Unknown,
            cert: None,
            stats: SolveStats::default(),
        };
        let certless_loss = DeepResult::<()> {
            status: ProofStatus::Loss,
            cert: None,
            stats: SolveStats::default(),
        };
        assert_eq!(
            hard_value_from_verify_impl(&Accept, &state, &win).map(HardValue::value),
            Some(1.0)
        );
        assert!(hard_value_from_verify_impl(&Reject, &state, &win).is_none());
        assert!(hard_value_from_verify_impl(&Accept, &state, &unknown).is_none());
        assert!(hard_value_from_verify_impl(&Accept, &state, &certless_loss).is_none());
    }
}
