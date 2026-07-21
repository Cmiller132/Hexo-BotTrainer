//! Proof-carrying forced-tree solver.
//!
//! The search fixes a `claimant` player and constructs a winning strategy for
//! that identity.  Nodes owned by the claimant are existential; nodes owned by
//! the other player are universal.  This is deliberately not negamax: a
//! `FirstStone` placement leaves the same player to place `SecondStone`.
//!
//! The implementation is a deterministic, proof-number-ordered depth-first
//! AND/OR proof constructor.  It is equivalent to the proof side of df-pn for
//! the three-valued interface used here: the most promising (lowest initial
//! proof-number) OR child is expanded first, while every child of an AND node
//! must produce a proof.  Failure or any resource exhaustion is `Unknown`; a
//! failed restricted attack is never interpreted as a proof for the opponent.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::Arc;

#[cfg(test)]
use std::cell::{Cell, RefCell};
#[cfg(test)]
use std::rc::Rc;
use std::time::Instant;

use hexo_engine::{
    apply_placement, hex_distance, Axis, HexCoord, HexoState as RustHexoState, Placement, Player,
    TurnPhase, WindowKey,
};

use crate::threats_shared as threats;
use crate::tss_core::{
    seed_band_radius, CertVerify, DeepResult, DeepSolve, ProofStatus, SolveCaps, SolveGoal,
    SolveStats, ZoneSearchCaps,
};
#[cfg(test)]
use crate::tss_core::{ClosureDebtStats, ThresholdScaleStats};
use crate::tss_verify::{
    CertCommutation, CertEdge, CertNode, CertNodeId, RootBinding, TssCertificate, TssVerifier,
    ZoneInfo, MAX_CERT_COMMUTATIONS, MAX_CERT_DEPTH, MAX_CERT_EDGES, MAX_CERT_NODES,
    MAX_CERT_ROOT_STONES, MAX_CERT_WITNESSES,
};

#[cfg(test)]
pub(crate) const ORDERING_STUDY_ORDERS: [&str; 7] = [
    "baseline",
    "zone_bound",
    "census_distance",
    "gate_adjacency",
    "d_stone",
    "census_zone_composite",
    "zone_gate_composite",
];

/// One compact, post-solve observation. Per-child features live only on the
/// cfg(test) child records; this is the offline result retained by the corpus
/// harness instead of emitting a potentially enormous child trace.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OrderingStudyRecord {
    pub(crate) depth: u32,
    pub(crate) generated_children: u32,
    pub(crate) pair_node: bool,
    pub(crate) ranks: [u32; ORDERING_STUDY_ORDERS.len()],
}

#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct OrderingStudyReport {
    pub(crate) records: Vec<OrderingStudyRecord>,
}

#[cfg(test)]
std::thread_local! {
    static ORDERING_STUDY_REPORT: RefCell<OrderingStudyReport> = RefCell::new(OrderingStudyReport::default());
}

#[cfg(test)]
pub(crate) fn begin_ordering_study_report() {
    ORDERING_STUDY_REPORT.with(|slot| slot.borrow_mut().records.clear());
}

#[cfg(test)]
pub(crate) fn take_ordering_study_report() -> OrderingStudyReport {
    ORDERING_STUDY_REPORT.with(|slot| std::mem::take(&mut *slot.borrow_mut()))
}

// Test-only NQ6 census/PN telemetry. The production solver has no field,
// branch, or callable entry point for this collector.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PnInitTelemetryMode {
    WidePn,
    NarrowCompat,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PnInitTelemetryOutcome {
    Proven,
    Refuted,
    Unknown,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct PnInitTelemetryNode {
    pub(crate) serial: u64,
    pub(crate) parent_serial: Option<u64>,
    pub(crate) engine_node: u64,
    pub(crate) mode: PnInitTelemetryMode,
    pub(crate) depth: u32,
    pub(crate) h_rem: Option<u32>,
    pub(crate) phase_code: u8,
    pub(crate) win_arm: bool,
    pub(crate) census: Option<u8>,
    pub(crate) lb_plies: Option<u8>,
    pub(crate) gate: bool,
    pub(crate) coordinate_safe: bool,
    pub(crate) census_scan_nanos: u64,
    pub(crate) live_ge4: u32,
    pub(crate) live_ge3: u32,
    pub(crate) disjoint_two_gap: u32,
    pub(crate) outcome: PnInitTelemetryOutcome,
    pub(crate) frozen_state: Option<String>,
}

#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct PnInitTelemetryReport {
    pub(crate) nodes: Vec<PnInitTelemetryNode>,
}

#[cfg(test)]
#[derive(Default)]
struct PnInitTelemetrySession {
    report: PnInitTelemetryReport,
    wide_last_event: HashMap<usize, u64>,
}

#[cfg(test)]
std::thread_local! {
    static PN_INIT_TELEMETRY: std::cell::RefCell<Option<PnInitTelemetrySession>> = const { std::cell::RefCell::new(None) };
    static PN_INIT_WIDE_STACK: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    static PN_INIT_NARROW_STACK: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn begin_pn_init_telemetry() {
    PN_INIT_TELEMETRY.with(|slot| {
        *slot.borrow_mut() = Some(PnInitTelemetrySession::default());
    });
    PN_INIT_WIDE_STACK.with(|stack| stack.borrow_mut().clear());
    PN_INIT_NARROW_STACK.with(|stack| stack.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn take_pn_init_telemetry_report() -> Option<PnInitTelemetryReport> {
    PN_INIT_WIDE_STACK.with(|stack| assert!(stack.borrow().is_empty()));
    PN_INIT_NARROW_STACK.with(|stack| assert!(stack.borrow().is_empty()));
    PN_INIT_TELEMETRY.with(|slot| slot.borrow_mut().take().map(|session| session.report))
}

fn interior_census_lb_plies(phase: TurnPhase, census: u8) -> Option<u8> {
    if census > 5 {
        return None;
    }
    let m = match phase {
        TurnPhase::FirstStone if census >= 4 => 6 - census,
        TurnPhase::FirstStone => (7 - census).min(6),
        TurnPhase::SecondStone { .. } if census >= 3 => 6 - census,
        TurnPhase::SecondStone { .. } => (7 - census).min(6),
        TurnPhase::Opening => return None,
    };
    let index = usize::from(m.saturating_sub(1));
    match phase {
        TurnPhase::FirstStone => [1, 2, 5, 6, 9, 10].get(index).copied(),
        TurnPhase::SecondStone { .. } => [1, 4, 5, 8, 9, 12].get(index).copied(),
        TurnPhase::Opening => None,
    }
}

fn interior_census_coordinate_safe(state: &RustHexoState, h_rem: i64) -> bool {
    const SAFE: i64 = 16_383;
    if h_rem < 0 {
        return false;
    }
    let Some(radius) = h_rem.checked_add(1).and_then(|x| x.checked_mul(8)) else {
        return false;
    };
    let Some(limit) = SAFE.checked_sub(radius) else {
        return false;
    };
    state.board().occupied_cells().iter().all(|coord| {
        let q = i64::from(coord.q);
        let r = i64::from(coord.r);
        q.checked_add(r)
            .and_then(|sum| sum.checked_neg())
            .and_then(|s| Some((q.checked_abs()?, r.checked_abs()?, s.checked_abs()?)))
            .is_some_and(|(q_abs, r_abs, s_abs)| q_abs <= limit && r_abs <= limit && s_abs <= limit)
    })
}

#[derive(Clone, Copy, Debug)]
struct InteriorCensusGateEvaluation {
    dismiss: bool,
    nanos: u64,
}

/// Evaluate Contract 8.1/8.2 for one interior claimant-owned bounded WIN arm.
/// `None` means the node is outside the proved/elected scope and no census was
/// scanned. A non-dismissing `Some` is still a measured live evaluation.
fn evaluate_interior_census_gate(
    state: &RustHexoState,
    claimant: Player,
    root_ply: u32,
    semantic_horizon: u32,
) -> Option<InteriorCensusGateEvaluation> {
    if state.is_terminal()
        || state.current_player() != claimant
        || state.placements_made() <= root_ply
        || !matches!(
            state.phase(),
            TurnPhase::FirstStone | TurnPhase::SecondStone { .. }
        )
    {
        return None;
    }

    // Contract 8.2 requires widened, checked absolute-to-relative arithmetic.
    let base_wide = i64::from(state.placements_made());
    let semantic_wide = i64::from(semantic_horizon);
    let h_rem = semantic_wide.checked_sub(base_wide)?;
    if !(0..=8).contains(&h_rem) || !interior_census_coordinate_safe(state, h_rem) {
        return None;
    }

    let started = Instant::now();
    let mut census = 0u8;
    let mut invariant_ok = true;
    for entry in state.board().windows().entries() {
        let ac = entry.count(claimant);
        let dc = entry.count(claimant.other());
        if ac > 5 || dc > 5 {
            invariant_ok = false;
        }
        if ac > 0 && dc == 0 {
            census = census.max(ac);
        }
    }
    let lb_plies = invariant_ok
        .then(|| interior_census_lb_plies(state.phase(), census))
        .flatten();
    let dismiss = lb_plies.is_some_and(|lb| i64::from(lb) > h_rem);
    let nanos = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    Some(InteriorCensusGateEvaluation { dismiss, nanos })
}

#[cfg(test)]
fn pn_init_frozen_state(state: &RustHexoState) -> String {
    let mut stones = state
        .board()
        .occupied_cells()
        .iter()
        .map(|&coord| {
            let owner = state.board().get(coord).expect("occupied cell has owner");
            (coord.q, coord.r, owner.index())
        })
        .collect::<Vec<_>>();
    stones.sort_unstable();
    format!(
        "placements={} player={} phase={:?} stones={stones:?}",
        state.placements_made(),
        state.current_player().index(),
        state.phase()
    )
}

#[cfg(test)]
fn pn_init_census_features(state: &RustHexoState, attacker: Player) -> (u8, u64, u32, u32, u32) {
    // Provenance: exact Contract-8.1 recipe copied from hunt-dtw-bounds at
    // reviewed theorem commit ffdd414ad5197444eef44af4f28da376a5d95507.
    let started = std::time::Instant::now();
    let mut census = 0u8;
    for entry in state.board().windows().entries() {
        let ac = entry.count(attacker);
        let dc = entry.count(attacker.other());
        if ac > 0 && dc == 0 {
            census = census.max(ac);
        }
    }
    let census_scan_nanos = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;

    let mut live_ge4 = 0u32;
    let mut live_ge3 = 0u32;
    let mut two_gap = Vec::new();
    for entry in state.board().windows().entries() {
        let ac = entry.count(attacker);
        let dc = entry.count(attacker.other());
        if ac == 0 || dc != 0 {
            continue;
        }
        live_ge3 = live_ge3.saturating_add(u32::from(ac >= 3));
        live_ge4 = live_ge4.saturating_add(u32::from(ac >= 4));
        if ac == 4 {
            let mut empties = entry.empty_cells();
            empties.sort_unstable_by_key(|coord| (coord.q, coord.r));
            if empties.len() == 2 {
                two_gap.push((entry.key(), empties[0], empties[1]));
            }
        }
    }
    two_gap.sort_unstable_by_key(|(key, _, _)| (key.axis.index(), key.start.q, key.start.r));
    let mut used = HashSet::new();
    let mut disjoint_two_gap = 0u32;
    for (_, left, right) in two_gap {
        if !used.contains(&left) && !used.contains(&right) {
            used.insert(left);
            used.insert(right);
            disjoint_two_gap = disjoint_two_gap.saturating_add(1);
        }
    }
    (
        census,
        census_scan_nanos,
        live_ge4,
        live_ge3,
        disjoint_two_gap,
    )
}

#[cfg(test)]
fn pn_init_record_node(
    state: &RustHexoState,
    claimant: Player,
    root_ply: u32,
    semantic_horizon: u32,
    engine_node: u64,
    mode: PnInitTelemetryMode,
    parent_serial: Option<u64>,
) -> Option<u64> {
    PN_INIT_TELEMETRY.with(|slot| {
        let mut slot = slot.borrow_mut();
        let session = slot.as_mut()?;
        let serial = session.report.nodes.len() as u64;
        let depth = state.placements_made().checked_sub(root_ply)?;
        let h_rem = semantic_horizon.checked_sub(state.placements_made());
        let win_arm = state.current_player() == claimant;
        let supported = matches!(
            state.phase(),
            TurnPhase::FirstStone | TurnPhase::SecondStone { .. }
        );
        let phase_code = match state.phase() {
            TurnPhase::Opening => 0,
            TurnPhase::FirstStone => 1,
            TurnPhase::SecondStone { .. } => 2,
        };
        let should_scan = supported && h_rem.is_some_and(|h| h <= 16);
        let (feature_census, feature_scan_nanos, live_ge4, live_ge3, disjoint_two_gap) =
            if should_scan {
                pn_init_census_features(state, claimant)
            } else {
                (0, 0, 0, 0, 0)
            };
        let census = win_arm.then_some(feature_census);
        let census_scan_nanos = if win_arm { feature_scan_nanos } else { 0 };
        let lb_plies = census.and_then(|c| interior_census_lb_plies(state.phase(), c));
        let coordinate_safe = h_rem
            .map(i64::from)
            .is_some_and(|h| interior_census_coordinate_safe(state, h));
        let gate = !state.is_terminal()
            && win_arm
            && supported
            && census.is_some_and(|c| c <= 5)
            && coordinate_safe
            && h_rem.is_some_and(|h| h <= 8 && lb_plies.is_some_and(|lb| u32::from(lb) > h));
        session.report.nodes.push(PnInitTelemetryNode {
            serial,
            parent_serial,
            engine_node,
            mode,
            depth,
            h_rem,
            phase_code,
            win_arm,
            census,
            lb_plies,
            gate,
            coordinate_safe,
            census_scan_nanos,
            live_ge4,
            live_ge3,
            disjoint_two_gap,
            outcome: PnInitTelemetryOutcome::Unknown,
            frozen_state: gate.then(|| pn_init_frozen_state(state)),
        });
        Some(serial)
    })
}

#[cfg(test)]
struct PnInitWideWorkGuard {
    active: bool,
}

#[cfg(test)]
impl PnInitWideWorkGuard {
    fn enter(id: usize) -> Self {
        let active = PN_INIT_TELEMETRY.with(|slot| slot.borrow().is_some());
        if active {
            PN_INIT_WIDE_STACK.with(|stack| stack.borrow_mut().push(id));
        }
        Self { active }
    }
}

#[cfg(test)]
impl Drop for PnInitWideWorkGuard {
    fn drop(&mut self) {
        if self.active {
            PN_INIT_WIDE_STACK.with(|stack| {
                stack
                    .borrow_mut()
                    .pop()
                    .expect("wide telemetry stack underflow");
            });
        }
    }
}

#[cfg(test)]
struct PnInitNarrowGuard {
    serial: Option<u64>,
}

#[cfg(test)]
impl PnInitNarrowGuard {
    fn enter(
        state: &RustHexoState,
        claimant: Player,
        root_ply: u32,
        semantic_horizon: u32,
    ) -> Self {
        let parent_serial = PN_INIT_NARROW_STACK.with(|stack| stack.borrow().last().copied());
        let engine_node = PN_INIT_TELEMETRY.with(|slot| {
            slot.borrow()
                .as_ref()
                .map(|session| session.report.nodes.len() as u64)
                .unwrap_or(0)
        });
        let serial = pn_init_record_node(
            state,
            claimant,
            root_ply,
            semantic_horizon,
            engine_node,
            PnInitTelemetryMode::NarrowCompat,
            parent_serial,
        );
        if let Some(serial) = serial {
            PN_INIT_NARROW_STACK.with(|stack| stack.borrow_mut().push(serial));
        }
        Self { serial }
    }

    fn finish(&mut self, proven: bool, hit_limit: bool) {
        let Some(serial) = self.serial.take() else {
            return;
        };
        PN_INIT_TELEMETRY.with(|slot| {
            let mut slot = slot.borrow_mut();
            let node =
                &mut slot.as_mut().expect("active narrow telemetry").report.nodes[serial as usize];
            node.outcome = if proven {
                PnInitTelemetryOutcome::Proven
            } else if hit_limit {
                PnInitTelemetryOutcome::Unknown
            } else {
                PnInitTelemetryOutcome::Refuted
            };
        });
        PN_INIT_NARROW_STACK.with(|stack| {
            let popped = stack.borrow_mut().pop();
            assert_eq!(popped, Some(serial));
        });
    }
}

#[cfg(test)]
impl Drop for PnInitNarrowGuard {
    fn drop(&mut self) {
        if let Some(serial) = self.serial.take() {
            PN_INIT_NARROW_STACK.with(|stack| {
                let popped = stack.borrow_mut().pop();
                assert_eq!(popped, Some(serial));
            });
        }
    }
}

/// A second, fixed guard in addition to `SolveCaps::node_cap`.  It bounds stack
/// depth even when a caller supplies an accidentally enormous node cap.
const MAX_SEARCH_DEPTH: usize = MAX_CERT_DEPTH;

/// Conservative allocator-header charge used by the explicit TT accounting.
const ALLOC_OVERHEAD: usize = 32;
/// The direct table never reserves more than this many inline slots.
const MAX_TT_SLOTS: usize = 1 << 20;
/// Expected bytes per slot.  Entries with larger position keys simply make the
/// table stop accepting replacements before the caller's byte cap is crossed.
const TARGET_BYTES_PER_TT_SLOT: usize = 256;
/// Shared entries own certificate fragments and are consequently wider than
/// solve-local entries.  The target only determines direct-table density; the
/// exact retained capacities below remain the authoritative byte accounting.
const TARGET_BYTES_PER_SHARED_TT_SLOT: usize = 512;
/// Internal positive fragments are promoted only while cheap to compact.  A
/// successful attempt root is offered regardless of these two tuning limits.
const MAX_PROMOTED_FRAGMENT_NODES: usize = 128;
const MAX_PROMOTED_FRAGMENT_EDGES: usize = 512;
/// Wide shared fragments may retain at most one eighth of the caller's TT cap.
/// Slots are allocated lazily after a solve, and the next solve subtracts only
/// bytes actually retained, so an empty/cold store leaves the historical wide
/// search cap byte-for-byte intact.
const WIDE_FRAGMENT_CAP_DIVISOR: usize = 8;
/// Bounded independently verified descendants collected from one attempt.
const MAX_WIDE_FRAGMENT_PROMOTIONS: usize = 64;
/// Fragment slots are wider than key-only PN-index entries.
const TARGET_BYTES_PER_PROVEN_FRAGMENT_SLOT: usize = 1024;
/// Solve-local TT sentinel for a fully explored restricted position with no
/// proof in the current wide/depth-bounded attempt. Certificate IDs can never
/// approach this value (`MAX_CERT_NODES` is 100k).
const LOCAL_TT_FAILED: CertNodeId = CertNodeId::MAX;

#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct QuotientTelemetryReport {
    pub retained_entries: u64,
    pub indexed_entries: u64,
    pub tt_hits: u64,
    pub d6_index_duplicates: u64,
    pub d6_index_denominator: u64,
    pub expanded_unique_positions: u64,
    pub d6_expanded_duplicates: u64,
    pub d6_canonicalization_calls: u64,
    pub d6_canonicalization_nanos: u64,
    pub horizon_queries: u64,
    pub horizon_exact_hits: u64,
    pub horizon_clock_misses: u64,
    pub horizon_monotone_hits: u64,
    pub horizon_position_clock_entries: u64,
    pub horizon_multi_clock_positions: u64,
    pub horizon_positions: u64,
    pub horizon_sound_wins: u64,
    pub horizon_sound_refutations: u64,
    pub horizon_staged_cutoffs_excluded: u64,
    pub commutation_eligible_nodes: u64,
    pub commutation_independent_nodes: u64,
    pub commutation_shared_window: u64,
    pub commutation_legality_coupling: u64,
    pub commutation_threat_response: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SharedFragmentStoreSnapshot {
    pub enabled: bool,
    pub entries: u64,
    pub bytes: u64,
    pub peak_bytes: u64,
    pub stored_nodes: u64,
    pub stored_edges: u64,
    pub admissions: u64,
    pub replacements: u64,
    pub refusals: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LeafSurfaceReuseSnapshot {
    pub(crate) shared_reconfigurations: u64,
    pub(crate) fragment_reconfigurations: u64,
    pub(crate) shared_slots: u64,
    pub(crate) fragment_slots: u64,
}

#[cfg(test)]
thread_local! {
    static LAST_QUOTIENT_REPORT: RefCell<Option<QuotientTelemetryReport>> = const { RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn take_quotient_telemetry_report() -> Option<QuotientTelemetryReport> {
    LAST_QUOTIENT_REPORT.with(|slot| slot.borrow_mut().take())
}

#[cfg(test)]
fn clear_quotient_telemetry_report() {
    LAST_QUOTIENT_REPORT.with(|slot| *slot.borrow_mut() = None);
}

/// Optional attacker-universe expansions.  The default is deliberately the
/// historical narrow generator so production callers retain byte-identical
/// search behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Round3Flag {
    #[default]
    Off,
    Shadow,
    Consume,
}

impl Round3Flag {
    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Consume => "consume",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WidthOptions {
    vcf_pair_complete: bool,
    quiet_turn_or_edges: Round3Flag,
    ranked_unforced_defender_zone: Round3Flag,
}

/// Test-only Q8 observation at one Consume quiet-fallback OR node. Production
/// keeps the flag-gated kernel, while records and their sink remain absent
/// from release builds.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct KReplyShadowRecord {
    pub full_quiet: usize,
    pub urgent: bool,
    pub k_reply: Option<usize>,
    pub proved_win: bool,
    pub winning_edge: Option<HexCoord>,
    pub winning_edge_in_k: Option<bool>,
    pub position: Option<RootBinding>,
    pub consumed: bool,
    pub consumed_matches_shadow: Option<bool>,
}

#[cfg(test)]
struct KReplyShadowTicket {
    record: usize,
    kernel: Vec<HexCoord>,
}

#[derive(Clone, Debug)]
pub(crate) struct KReplyKernel {
    pub eligible: bool,
    pub urgent: bool,
    pub cells: Vec<HexCoord>,
}

impl KReplyKernel {
    /// Exact retained view. The common nonurgent case borrows `Legal(P)`
    /// instead of allocating an identical vector.
    pub(crate) fn retained<'a>(&'a self, legal: &'a [HexCoord]) -> &'a [HexCoord] {
        if !self.eligible {
            &[]
        } else if self.urgent {
            &self.cells
        } else {
            legal
        }
    }
}

impl WidthOptions {
    pub(crate) fn vcf_pair_complete() -> Self {
        Self {
            vcf_pair_complete: true,
            quiet_turn_or_edges: Round3Flag::Off,
            ranked_unforced_defender_zone: Round3Flag::Off,
        }
    }

    #[cfg(test)]
    pub(crate) fn round3_shadow() -> Self {
        Self {
            vcf_pair_complete: true,
            quiet_turn_or_edges: Round3Flag::Shadow,
            ranked_unforced_defender_zone: Round3Flag::Shadow,
        }
    }

    pub(crate) fn round3_consume() -> Self {
        Self {
            vcf_pair_complete: true,
            quiet_turn_or_edges: Round3Flag::Consume,
            ranked_unforced_defender_zone: Round3Flag::Consume,
        }
    }

    fn consumes_quiet_turns(self) -> bool {
        self.quiet_turn_or_edges == Round3Flag::Consume
    }

    fn consumes_ranked_zone(self) -> bool {
        self.ranked_unforced_defender_zone == Round3Flag::Consume
    }
}

/// Process-environment switches sampled once at the public solve boundary.
/// Keeping the sample separate makes effective-configuration resolution a
/// pure operation that can also back the harness manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SolveRuntimeFlags {
    lazy_frontier: bool,
    interior_census_gate: bool,
    k_reply_consume: bool,
}

/// Fully resolved flags and memory caps used by one `solve_goal` invocation.
/// This is telemetry/configuration data only; the search never mutates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EffectiveSolveConfig {
    pub(crate) vcf_pair_complete: bool,
    pub(crate) dual_pass: bool,
    pub(crate) loss_reserve_nodes: u32,
    pub(crate) group2: bool,
    pub(crate) quiet_turn_or_edges: &'static str,
    pub(crate) ranked_unforced_defender_zone: &'static str,
    pub(crate) tt_enabled: bool,
    pub(crate) tt_bytes_cap: usize,
    pub(crate) shared_fragments_enabled: bool,
    pub(crate) fragment_store_cap_bytes: usize,
    pub(crate) lazy_frontier: bool,
    pub(crate) interior_census_gate: bool,
    pub(crate) k_reply_consume: bool,
    uses_wide_pn: bool,
    local_tt_cap: usize,
    shared_tt_cap: usize,
}

/// Reusable proof-carrying solver.  Its shared TT retains only complete,
/// self-contained positive proof fragments; solve-local arena IDs never cross
/// an attempt boundary.
#[derive(Debug)]
pub(crate) struct TssSolver {
    tt_enabled: bool,
    hash_mask: u64,
    shared_tt: SharedProofCache,
    /// Default-off T10/U22 wide proven-fragment path. Read once at construction.
    shared_fragments_enabled: bool,
    fragment_store: ProvenFragmentStore,
    zone: ZoneSearchCaps,
    width: WidthOptions,
    /// Reuse an undecided wide `Both` primal's unspent nodes for the dual
    /// claim. Default-off preserves the historical primal-only wide split.
    dual_pass: bool,
    /// Hold this many post-root nodes out of the wide `Both` primal for an
    /// opponent-WIN attempt. A positive reserve schedules that attempt even
    /// without the leftover policy; if the primal returns early, an enabled
    /// dual pass upgrades it to every actual leftover node. Zero preserves the
    /// current full-primal allocation.
    loss_reserve_nodes: u32,
    /// Default-off v1 Group-2 reduced-fanout selector (narrow zone path only;
    /// DESIGN_G2_CERT_EXTENSION.md §2.4, task flag `tss_solver_group2`). When
    /// on, eligible unforced defender nodes attempt the exact FHW closure and
    /// emit `UniversalGroup2V1`; any failure falls back to the legacy paths
    /// and, at the attempt boundary, to a clean group2-off re-solve so the
    /// flag can never decide fewer positions than off.
    group2: bool,
    /// Leaf-profile overrides (PLAN_TSS_MCTS_INTEGRATION.md §3). When `Some`,
    /// they replace the per-solve environment reads for the lazy defender
    /// frontier and the interior census gate, so the trainer leaf/root/async
    /// path runs the campaign engine at the leaf-decided config deterministically
    /// (not conditioned on process env). `None` preserves the historical env
    /// behavior for the offline corpus/hunt harnesses.
    force_lazy_frontier: Option<bool>,
    force_interior_census_gate: Option<bool>,
    #[cfg(test)]
    last_narrow_signatures: Vec<NarrowAttemptSignature>,
    #[cfg(test)]
    last_k_reply_shadow: Vec<KReplyShadowRecord>,
    #[cfg(test)]
    leaf_surface_shared_reconfigurations: u64,
    #[cfg(test)]
    leaf_surface_fragment_reconfigurations: u64,
    /// Test-only observation seam for cap-resume identity checks. Production
    /// builds neither retain nor expose unfinished proof numbers.
    #[cfg(test)]
    last_wide_root_numbers: Option<(u32, u32)>,
    #[cfg(test)]
    last_effective_config: Option<EffectiveSolveConfig>,
}

impl Default for TssSolver {
    fn default() -> Self {
        let shared_fragments_enabled =
            std::env::var("TSS_SHARED_FRAGMENTS").ok().as_deref() == Some("1");
        Self {
            tt_enabled: true,
            hash_mask: u64::MAX,
            shared_tt: SharedProofCache::new(0, u64::MAX),
            shared_fragments_enabled,
            fragment_store: ProvenFragmentStore::new(0, u64::MAX),
            zone: ZoneSearchCaps::default(),
            width: WidthOptions::default(),
            dual_pass: false,
            loss_reserve_nodes: 0,
            group2: false,
            force_lazy_frontier: None,
            force_interior_census_gate: None,
            #[cfg(test)]
            last_narrow_signatures: Vec::new(),
            #[cfg(test)]
            last_k_reply_shadow: Vec::new(),
            #[cfg(test)]
            leaf_surface_shared_reconfigurations: 0,
            #[cfg(test)]
            leaf_surface_fragment_reconfigurations: 0,
            #[cfg(test)]
            last_wide_root_numbers: None,
            #[cfg(test)]
            last_effective_config: None,
        }
    }
}

impl TssSolver {
    #[cfg(test)]
    pub(crate) fn leaf_surface_reuse_snapshot(&self) -> LeafSurfaceReuseSnapshot {
        LeafSurfaceReuseSnapshot {
            shared_reconfigurations: self.leaf_surface_shared_reconfigurations,
            fragment_reconfigurations: self.leaf_surface_fragment_reconfigurations,
            shared_slots: self.shared_tt.slots.len() as u64,
            fragment_slots: self.fragment_store.slots.len() as u64,
        }
    }

    #[cfg(test)]
    pub(crate) fn set_shared_fragments_for_test(&mut self, enabled: bool) {
        if self.shared_fragments_enabled != enabled {
            self.fragment_store.clear();
        }
        self.shared_fragments_enabled = enabled;
    }

    #[cfg(test)]
    pub(crate) fn last_wide_root_numbers(&self) -> Option<(u32, u32)> {
        self.last_wide_root_numbers
    }

    #[cfg(test)]
    pub(crate) fn last_effective_config(&self) -> Option<EffectiveSolveConfig> {
        self.last_effective_config
    }

    #[cfg(test)]
    pub(crate) fn shared_fragment_store_snapshot(&self) -> SharedFragmentStoreSnapshot {
        SharedFragmentStoreSnapshot {
            enabled: self.shared_fragments_enabled,
            entries: self.fragment_store.entry_count as u64,
            bytes: self.fragment_store.current_bytes as u64,
            peak_bytes: self.fragment_store.peak_bytes as u64,
            stored_nodes: self.fragment_store.stored_nodes as u64,
            stored_edges: self.fragment_store.stored_edges as u64,
            admissions: self.fragment_store.admissions,
            replacements: self.fragment_store.replacements,
            refusals: self.fragment_store.refusals,
        }
    }

    /// Set the zone/commutation options for subsequent solves. Changing the
    /// options DROPS the persistent positive-fragment cache: cached fragments
    /// are verified proofs either way, but their node-cost provenance belongs
    /// to the profile that built them — reusing them across an ON→OFF flip
    /// contaminates A/B node counts and conditional determinism (Codex
    /// review, profile isolation). Same-options calls keep the warm cache.
    pub(crate) fn set_zone_options(&mut self, zone: ZoneSearchCaps) {
        if self.zone != zone {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.zone = zone;
    }

    /// Set the attacker-width profile for subsequent solves.  As with zone
    /// options, changing profiles drops cached positive fragments so their
    /// node-cost provenance cannot leak between narrow and wide searches.
    pub(crate) fn set_width_options(&mut self, width: WidthOptions) {
        if self.width != width {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.width = width;
    }

    pub(crate) fn set_dual_pass(&mut self, dual_pass: bool) {
        if self.dual_pass != dual_pass {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.dual_pass = dual_pass;
    }

    pub(crate) fn set_loss_reserve_nodes(&mut self, loss_reserve_nodes: u32) {
        if self.loss_reserve_nodes != loss_reserve_nodes {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.loss_reserve_nodes = loss_reserve_nodes;
    }

    /// Enable/disable the v1 Group-2 selector. Changing the option drops the
    /// persistent caches (same profile-isolation rule as the other options:
    /// cached fragments' node-cost provenance must not leak across profiles).
    pub(crate) fn set_group2(&mut self, group2: bool) {
        if self.group2 != group2 {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.group2 = group2;
    }

    /// Externally selected verifier policy follows this solver option
    /// (design §5.1: trainer configuration, never certificate contents,
    /// chooses the policy).
    pub(crate) fn group2_enabled(&self) -> bool {
        self.group2
    }

    /// Configure this solver to the campaign leaf-decided profile
    /// (PLAN_TSS_MCTS_INTEGRATION.md §3, HUNT_REPORT_LEAF_SURFACE config D):
    /// wide `vcf_pair_complete` attacker width, the lazy defender frontier ON,
    /// the interior census gate ON, shared fragments OFF and k-reply OFF (the
    /// profile's measured no-value knobs, left at their env/default state). The
    /// lazy/gate forces make the trainer leaf/root/async path deterministic and
    /// independent of process environment. The 256 KiB per-solve TT and the
    /// node cap are supplied per solve by the caller (`SolveCaps`).
    pub(crate) fn configure_leaf_profile(&mut self) {
        self.set_width_options(WidthOptions::vcf_pair_complete());
        self.force_lazy_frontier = Some(true);
        self.force_interior_census_gate = Some(true);
    }

    /// Sample process-global runtime switches once. The returned value is an
    /// explicit input to `effective_solve_config`, keeping resolution itself
    /// deterministic and side-effect free.
    pub(crate) fn sample_runtime_flags(&self) -> SolveRuntimeFlags {
        SolveRuntimeFlags {
            lazy_frontier: std::env::var("TSS_LAZY_FRONTIER").ok().as_deref() == Some("1"),
            interior_census_gate: std::env::var_os("TSS_INTERIOR_CENSUS_GATE")
                .is_some_and(|value| value == "1"),
            k_reply_consume: matches!(std::env::var("TSS_K_REPLY_CONSUME").as_deref(), Ok("1")),
        }
    }

    /// Pure effective-configuration resolver shared by real solves and the
    /// harness manifest. `fragment_store.current_bytes` is projected through
    /// the same reconfiguration rule the solve applies immediately afterward.
    pub(crate) fn effective_solve_config(
        &self,
        caps: &SolveCaps,
        runtime: SolveRuntimeFlags,
    ) -> EffectiveSolveConfig {
        let tt_bytes_cap = if self.tt_enabled {
            caps.tt_bytes_cap
        } else {
            0
        };
        let uses_wide_pn = self.width.vcf_pair_complete
            && !(self.width.consumes_quiet_turns() && self.width.consumes_ranked_zone());
        let fragment_store_cap_bytes = if uses_wide_pn && self.shared_fragments_enabled {
            tt_bytes_cap / WIDE_FRAGMENT_CAP_DIVISOR
        } else {
            0
        };
        let fragment_store_bytes = if self.fragment_store.cap == fragment_store_cap_bytes
            && self.fragment_store.hash_mask == self.hash_mask
        {
            self.fragment_store.current_bytes
        } else {
            0
        };
        let (local_tt_cap, shared_tt_cap) = if uses_wide_pn && self.shared_fragments_enabled {
            (tt_bytes_cap.saturating_sub(fragment_store_bytes), 0)
        } else if self.width.vcf_pair_complete {
            (tt_bytes_cap, 0)
        } else {
            split_tt_cap(tt_bytes_cap)
        };
        EffectiveSolveConfig {
            vcf_pair_complete: self.width.vcf_pair_complete,
            dual_pass: self.dual_pass,
            loss_reserve_nodes: self.loss_reserve_nodes,
            group2: self.group2,
            quiet_turn_or_edges: self.width.quiet_turn_or_edges.name(),
            ranked_unforced_defender_zone: self.width.ranked_unforced_defender_zone.name(),
            tt_enabled: self.tt_enabled,
            tt_bytes_cap,
            shared_fragments_enabled: self.shared_fragments_enabled,
            fragment_store_cap_bytes,
            lazy_frontier: self.force_lazy_frontier.unwrap_or(runtime.lazy_frontier),
            interior_census_gate: self
                .force_interior_census_gate
                .unwrap_or(runtime.interior_census_gate),
            k_reply_consume: runtime.k_reply_consume,
            uses_wide_pn,
            local_tt_cap,
            shared_tt_cap,
        }
    }

    #[cfg(test)]
    pub(crate) fn k_reply_shadow(&self) -> &[KReplyShadowRecord] {
        &self.last_k_reply_shadow
    }

    #[cfg(test)]
    fn without_tt() -> Self {
        Self {
            tt_enabled: false,
            hash_mask: u64::MAX,
            shared_tt: SharedProofCache::new(0, u64::MAX),
            shared_fragments_enabled: false,
            fragment_store: ProvenFragmentStore::new(0, u64::MAX),
            zone: ZoneSearchCaps::default(),
            width: WidthOptions::default(),
            dual_pass: false,
            loss_reserve_nodes: 0,
            group2: false,
            force_lazy_frontier: None,
            force_interior_census_gate: None,
            last_narrow_signatures: Vec::new(),
            last_k_reply_shadow: Vec::new(),
            leaf_surface_shared_reconfigurations: 0,
            leaf_surface_fragment_reconfigurations: 0,
            last_wide_root_numbers: None,
            last_effective_config: None,
        }
    }

    /// Test hook: masking every hash to zero forces all positions into one
    /// bucket.  Full-key equality must still prevent a value-bearing false hit.
    #[cfg(test)]
    fn with_hash_mask(hash_mask: u64) -> Self {
        Self {
            tt_enabled: true,
            hash_mask,
            shared_tt: SharedProofCache::new(0, hash_mask),
            shared_fragments_enabled: false,
            fragment_store: ProvenFragmentStore::new(0, hash_mask),
            zone: ZoneSearchCaps::default(),
            width: WidthOptions::default(),
            dual_pass: false,
            loss_reserve_nodes: 0,
            group2: false,
            force_lazy_frontier: None,
            force_interior_census_gate: None,
            last_narrow_signatures: Vec::new(),
            last_k_reply_shadow: Vec::new(),
            leaf_surface_shared_reconfigurations: 0,
            leaf_surface_fragment_reconfigurations: 0,
            last_wide_root_numbers: None,
            last_effective_config: None,
        }
    }

    /// Solve only for the requested root-perspective side(s).  One-sided modes
    /// receive the entire remaining node budget; the legacy trait entry point
    /// below delegates to `Both`.
    pub(crate) fn solve_goal(
        &mut self,
        state: &RustHexoState,
        caps: &SolveCaps,
        goal: SolveGoal,
    ) -> DeepResult<TssCertificate> {
        #[cfg(test)]
        {
            self.last_narrow_signatures.clear();
            self.last_k_reply_shadow.clear();
            self.last_wide_root_numbers = None;
            self.last_effective_config = None;
            clear_quotient_telemetry_report();
        }

        let runtime = self.sample_runtime_flags();
        let effective = self.effective_solve_config(caps, runtime);
        #[cfg(test)]
        {
            self.last_effective_config = Some(effective);
        }
        #[cfg(test)]
        if self.fragment_store.cap != effective.fragment_store_cap_bytes
            || self.fragment_store.hash_mask != self.hash_mask
        {
            self.leaf_surface_fragment_reconfigurations = self
                .leaf_surface_fragment_reconfigurations
                .saturating_add(1);
        }
        self.fragment_store
            .reconfigure(effective.fragment_store_cap_bytes, self.hash_mask);
        debug_assert_eq!(
            effective.local_tt_cap,
            if effective.uses_wide_pn && effective.shared_fragments_enabled {
                effective
                    .tt_bytes_cap
                    .saturating_sub(self.fragment_store.current_bytes)
            } else if effective.vcf_pair_complete {
                effective.tt_bytes_cap
            } else {
                split_tt_cap(effective.tt_bytes_cap).0
            }
        );
        #[cfg(test)]
        if self.shared_tt.cap != effective.shared_tt_cap
            || self.shared_tt.hash_mask != self.hash_mask
        {
            self.leaf_surface_shared_reconfigurations =
                self.leaf_surface_shared_reconfigurations.saturating_add(1);
        }
        self.shared_tt
            .reconfigure(effective.shared_tt_cap, self.hash_mask);

        let initial_stats = SolveStats {
            peak_tt_bytes: self
                .shared_tt
                .current_bytes
                .saturating_add(self.fragment_store.current_bytes)
                as u64,
            fragment_store_entries: self.fragment_store.entry_count as u64,
            fragment_store_bytes: self.fragment_store.current_bytes as u64,
            ..SolveStats::default()
        };
        if caps.node_cap == 0
            || caps.semantic_horizon < state.placements_made()
            || state.board().len() > MAX_CERT_ROOT_STONES
        {
            return unknown(initial_stats);
        }

        // A root lambda-one/terminal result is both common and symmetric
        // between the primal and dual claims.  Count it as one examined node,
        // but filter it when the caller explicitly requested only the other
        // perspective.
        let mut stats = SolveStats {
            nodes: 1,
            ..initial_stats
        };
        if let Some((claimant, leaf)) = immediate_winner(state, self.width) {
            if node_resolution(&leaf) > caps.semantic_horizon {
                return unknown(stats);
            }
            let status = status_for_claimant(state.current_player(), claimant);
            if goal_accepts(goal, status) {
                let cert = TssCertificate {
                    root: RootBinding::from_state(state),
                    claimant,
                    root_node: 0,
                    nodes: vec![leaf],
                    semantic_horizon: caps.semantic_horizon,
                };
                return DeepResult {
                    status,
                    cert: Some(cert),
                    stats,
                };
            }
            return unknown(stats);
        }

        let remaining = caps.node_cap - 1;
        let (primal_cap, mut dual_cap) = match goal {
            SolveGoal::Win => (remaining, 0),
            SolveGoal::Loss => (0, remaining),
            // Pair-complete mode is deliberately a restricted VCF WIN search.
            // The default reserve is zero, preserving its full advertised
            // forcing-proof cap. A configured floor is an explicit policy
            // experiment and remains a positive opponent-claim search only;
            // its failure can establish no NO result.
            SolveGoal::Both if self.width.vcf_pair_complete => {
                wide_both_initial_caps(remaining, effective.loss_reserve_nodes)
            }
            SolveGoal::Both => ((remaining + 1) / 2, remaining / 2),
        };
        let root_player = state.current_player();

        if primal_cap > 0 {
            let attempt = self.prove_for(
                state,
                root_player,
                primal_cap,
                effective.local_tt_cap,
                caps.semantic_horizon,
                self.zone,
                self.width,
                effective.k_reply_consume,
                effective.interior_census_gate,
                effective.lazy_frontier,
                effective.group2,
            );
            #[cfg(test)]
            if let Some(signature) = attempt.tt_signature.as_ref() {
                self.last_narrow_signatures.push(signature.clone());
            }
            stats.merge(attempt.stats);
            if let Some(cert) = attempt.cert {
                return DeepResult {
                    status: ProofStatus::Win,
                    cert: Some(cert),
                    stats,
                };
            }
            if goal == SolveGoal::Both
                && effective.vcf_pair_complete
                && effective.dual_pass
            {
                dual_cap = caps.node_cap.saturating_sub(stats.nodes);
            }
        }

        if dual_cap > 0 {
            let attempt = self.prove_for(
                state,
                root_player.other(),
                dual_cap,
                effective.local_tt_cap,
                caps.semantic_horizon,
                self.zone,
                self.width,
                effective.k_reply_consume,
                effective.interior_census_gate,
                effective.lazy_frontier,
                effective.group2,
            );
            #[cfg(test)]
            if let Some(signature) = attempt.tt_signature.as_ref() {
                self.last_narrow_signatures.push(signature.clone());
            }
            stats.merge(attempt.stats);
            if let Some(cert) = attempt.cert {
                return DeepResult {
                    status: ProofStatus::Loss,
                    cert: Some(cert),
                    stats,
                };
            }
        }

        unknown(stats)
    }
}

impl DeepSolve for TssSolver {
    type Cert = TssCertificate;

    fn solve(&mut self, state: &RustHexoState, caps: &SolveCaps) -> DeepResult<Self::Cert> {
        self.solve_goal(state, caps, SolveGoal::Both)
    }
}

impl TssSolver {
    #[allow(clippy::too_many_arguments)]
    fn prove_for(
        &mut self,
        state: &RustHexoState,
        claimant: Player,
        node_cap: u64,
        local_tt_cap: usize,
        semantic_horizon: u32,
        zone: ZoneSearchCaps,
        width: WidthOptions,
        k_reply_consume: bool,
        interior_census_gate: bool,
        lazy_frontier: bool,
        group2: bool,
    ) -> AttemptResult {
        if !width.vcf_pair_complete
            || (width.consumes_quiet_turns() && width.consumes_ranked_zone())
        {
            let zone = if width.consumes_ranked_zone() {
                ZoneSearchCaps {
                    enabled: true,
                    stale_area_filter: false,
                    count2_threshold: true,
                    pair_commutation: false,
                }
            } else {
                zone
            };
            return WidePnSearch::prove_narrow_compat(
                state,
                claimant,
                node_cap,
                local_tt_cap,
                self.hash_mask,
                &mut self.shared_tt,
                semantic_horizon,
                zone,
                width,
                MAX_SEARCH_DEPTH,
                k_reply_consume,
                #[cfg(test)]
                (width.consumes_quiet_turns() && std::env::var_os("TSS_K_REPLY_SHADOW").is_some())
                    .then_some(&mut self.last_k_reply_shadow),
                interior_census_gate,
                group2,
            );
        }
        let depth_cap = wide_search_final_depth(state.placements_made(), semantic_horizon);

        self.prove_for_wide_pn_with_lazy_frontier(
            state,
            claimant,
            node_cap,
            local_tt_cap,
            semantic_horizon,
            depth_cap,
            width,
            interior_census_gate,
            lazy_frontier,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn prove_for_wide_pn(
        &mut self,
        state: &RustHexoState,
        claimant: Player,
        node_cap: u64,
        local_tt_cap: usize,
        semantic_horizon: u32,
        depth_cap: usize,
        width: WidthOptions,
        interior_census_gate: bool,
    ) -> AttemptResult {
        let runtime = self.sample_runtime_flags();
        let lazy_frontier = self.force_lazy_frontier.unwrap_or(runtime.lazy_frontier);
        self.prove_for_wide_pn_with_lazy_frontier(
            state,
            claimant,
            node_cap,
            local_tt_cap,
            semantic_horizon,
            depth_cap,
            width,
            interior_census_gate,
            lazy_frontier,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prove_for_wide_pn_with_lazy_frontier(
        &mut self,
        state: &RustHexoState,
        claimant: Player,
        node_cap: u64,
        local_tt_cap: usize,
        semantic_horizon: u32,
        depth_cap: usize,
        width: WidthOptions,
        interior_census_gate: bool,
        lazy_frontier: bool,
    ) -> AttemptResult {
        let fragments_enabled = self.shared_fragments_enabled;
        let shared_bytes = self
            .shared_tt
            .current_bytes
            .saturating_add(self.fragment_store.current_bytes);
        let fragment_store = fragments_enabled.then_some(&self.fragment_store);
        let mut search = WidePnSearch::new_with_width(
            claimant,
            state.placements_made(),
            node_cap,
            local_tt_cap,
            semantic_horizon,
            depth_cap,
            width,
            fragment_store,
        );
        search.interior_census_gate = interior_census_gate;
        search.lazy_frontier = lazy_frontier;
        let root = search.insert_root(state);
        search.run(state, root);
        #[cfg(test)]
        search.finalize_ordering_study();
        #[cfg(test)]
        {
            self.last_wide_root_numbers = Some((search.entries[root].pn, search.entries[root].dn));
        }
        #[cfg(test)]
        pn_init_finalize_wide(&search);
        #[cfg(test)]
        if std::env::var_os("TSS_TRACE_PN").is_some() {
            eprintln!(
                "WIDTH_PN pn={} dn={} expansions={} entries={}",
                search.entries[root].pn,
                search.entries[root].dn,
                search.expansions,
                search.entries.len(),
            );
            eprintln!(
                "WIDTH_PN_TT indexed_entries={} retained_entries={} tt_bytes={} tt_bytes_cap={} index_rejections={} first_rejection={:?} stage_refreshes={}",
                search.by_position.len(),
                search.entries.len(),
                search.current_bytes,
                search.tt_bytes_cap,
                search.tt_index_rejections,
                search.tt_first_rejection,
                search.stage_refreshes,
            );
            if let WidePnNode::Branch { children, .. } = &search.entries[root].node {
                for (rank, child) in children.iter().take(32).enumerate() {
                    let child_fields = search.format_trace_child(child);
                    eprintln!("WIDTH_PN_ROOT rank={rank} mv={:?} {child_fields}", child.mv);
                    if let Some(entry) = search.resolved_child_entry(child) {
                        if let WidePnNode::Branch {
                            children: replies, ..
                        } = &search.entries[entry].node
                        {
                            for (reply_rank, reply) in replies.iter().take(16).enumerate() {
                                let (reply_pn, reply_dn) = search.child_numbers(reply);
                                eprintln!(
                                    "WIDTH_PN_NESTED root_rank={rank} rank={reply_rank} mv={:?} pn={reply_pn} dn={reply_dn} linked={}",
                                    reply.mv,
                                    reply.entry.is_some()
                                );
                            }
                        }
                    }
                }
            }
        }
        let mut stats = SolveStats {
            nodes: search.expansions,
            expansions: search.expansions,
            tt_hits: search.tt_hits,
            tt_entries: (search.by_position.len() + self.shared_tt.entry_count()) as u64,
            peak_tt_bytes: shared_bytes.saturating_add(search.peak_bytes) as u64,
            horizon_cuts: search.horizon_cuts,
            kb_death_cuts: search.kb_death_cuts,
            #[cfg(test)]
            tt_admission_rejections: search.tt_index_rejections,
            fragment_lookups: search.fragment_lookups,
            fragment_hits: search.fragment_hits,
            fragment_store_entries: self.fragment_store.entry_count as u64,
            fragment_store_bytes: self.fragment_store.current_bytes as u64,
            interior_gate_evaluations: search.interior_gate_evaluations,
            interior_gate_dismissals: search.interior_gate_dismissals,
            interior_gate_nanos: search.interior_gate_nanos,
            #[cfg(test)]
            stage_refreshes: search.stage_refreshes,
            #[cfg(test)]
            live_ge3_seed_scans: search.live_ge3_seed_scans.get(),
            #[cfg(test)]
            live_ge3_seed_nanos: search.live_ge3_seed_nanos.get(),
            #[cfg(test)]
            closure_debt: *search.closure_stats.borrow(),
            #[cfg(test)]
            threshold_scale: *search.threshold_stats.borrow(),
            ..SolveStats::default()
        };
        let mut promotions = Vec::new();
        let cert = search.materialize(state, root).and_then(|materialized| {
            let fragment_imports = materialized.fragment_imports;
            let _dag_reuses = materialized.dag_reuses;
            let (nodes, root_node) =
                compact_certificate(&materialized.arena, materialized.root_node)?;
            let mut cert = TssCertificate {
                root: RootBinding::from_state(state),
                claimant,
                root_node,
                nodes,
                semantic_horizon,
            };
            if fragments_enabled {
                rebase_shared_fragment_labels(&mut cert, state)?;
                let claimed = status_for_claimant(state.current_player(), claimant);
                if !TssVerifier.verify(state, &cert, claimed) {
                    return None;
                }
                if let Some(proof) = CachedProof::from_compact(cert.nodes.clone(), cert.root_node) {
                    promotions.push((PositionKey::from_state(state), proof));
                }
            } else {
                rebase_zone_distances(&mut cert, state)?;
            }
            // Count only imports that survive compaction, dominant relabel,
            // and (when enabled) strict final-certificate verification.
            stats.fragment_imports = fragment_imports;
            Some(cert)
        });

        if fragments_enabled {
            let mut promoted_keys = promotions
                .iter()
                .map(|(key, _)| key.clone())
                .collect::<HashSet<_>>();
            for candidate in &search.proven_candidates {
                if promotions.len() >= MAX_WIDE_FRAGMENT_PROMOTIONS {
                    break;
                }
                let key = PositionKey::from_state(&candidate.state);
                if promoted_keys.contains(&key) {
                    continue;
                }
                let Some(materialized) = search.materialize(&candidate.state, candidate.id) else {
                    continue;
                };
                let Some((nodes, root_node)) = compact_certificate_limited(
                    &materialized.arena,
                    materialized.root_node,
                    MAX_PROMOTED_FRAGMENT_NODES,
                    MAX_PROMOTED_FRAGMENT_EDGES,
                ) else {
                    continue;
                };
                let mut cert = TssCertificate {
                    root: RootBinding::from_state(&candidate.state),
                    claimant,
                    root_node,
                    nodes,
                    semantic_horizon,
                };
                if rebase_shared_fragment_labels(&mut cert, &candidate.state).is_none() {
                    continue;
                }
                let claimed = status_for_claimant(candidate.state.current_player(), claimant);
                if !TssVerifier.verify(&candidate.state, &cert, claimed) {
                    continue;
                }
                if let Some(proof) = CachedProof::from_compact(cert.nodes, cert.root_node) {
                    promoted_keys.insert(key.clone());
                    promotions.push((key, proof));
                }
            }
        }
        #[cfg(test)]
        if let Some(telemetry) = search.quotient_telemetry.take() {
            let report = telemetry.finish(&search.entries, &search.by_position, search.tt_hits);
            LAST_QUOTIENT_REPORT.with(|slot| *slot.borrow_mut() = Some(report));
        }
        drop(search);
        // The solved attempt root was queued first; admit it last so a direct-
        // mapped collision with one of its descendants cannot erase the warm
        // repeat entry in the same promotion batch.
        promotions.reverse();
        for (key, proof) in promotions {
            self.fragment_store.insert(key, claimant, proof);
        }
        stats.fragment_store_entries = self.fragment_store.entry_count as u64;
        stats.fragment_store_bytes = self.fragment_store.current_bytes as u64;
        stats.peak_tt_bytes = stats
            .peak_tt_bytes
            .max(self.fragment_store.current_bytes as u64);
        AttemptResult {
            cert,
            stats,
            #[cfg(test)]
            tt_signature: None,
        }
    }
}

/// Test-only owner of one unfinished wide proof-number frontier. The session
/// is deliberately not reachable from production call paths: this campaign
/// must establish identity and value before any public API is proposed.
#[cfg(test)]
pub(crate) struct CapResumeSession {
    binding: CapResumeBinding,
    search: WidePnSearch<'static>,
    root: usize,
    stage_depth: usize,
    stage_initialized: bool,
    last_node_cap: u64,
    advances: u64,
    reentries: u64,
    valid: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct CapResumeBinding {
    root: RootBinding,
    claimant: Player,
    goal: SolveGoal,
    semantic_horizon: u32,
    width: WidthOptions,
    zone: ZoneSearchCaps,
    tt_enabled: bool,
    tt_bytes_cap: usize,
    hash_mask: u64,
    shared_fragments: bool,
    lazy_frontier: bool,
    lazy_key_validation: bool,
    interior_census_gate: bool,
    k_reply_consume: bool,
    quotient_telemetry: bool,
    max_depth_cap: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CapResumeError {
    Discarded,
    BindingMismatch,
    NonMonotoneNodeCap,
    UnsupportedProfile,
    VerifierRejected,
}

#[cfg(test)]
pub(crate) struct CapResumeAdvance {
    pub(crate) result: DeepResult<TssCertificate>,
    pub(crate) root_pn: u32,
    pub(crate) root_dn: u32,
    pub(crate) stage_depth: usize,
    pub(crate) advances: u64,
    pub(crate) reentries: u64,
}

#[cfg(test)]
impl CapResumeBinding {
    fn capture(
        solver: &TssSolver,
        state: &RustHexoState,
        caps: &SolveCaps,
        goal: SolveGoal,
    ) -> Result<Self, CapResumeError> {
        let uses_wide_pn = solver.width.vcf_pair_complete
            && !(solver.width.consumes_quiet_turns() && solver.width.consumes_ranked_zone());
        if !uses_wide_pn
            || solver.shared_fragments_enabled
            || caps.node_cap == 0
            || caps.semantic_horizon < state.placements_made()
            || state.board().len() > MAX_CERT_ROOT_STONES
            || immediate_winner(state, solver.width).is_some()
        {
            return Err(CapResumeError::UnsupportedProfile);
        }
        let claimant = match goal {
            SolveGoal::Win | SolveGoal::Both => state.current_player(),
            SolveGoal::Loss => state.current_player().other(),
        };
        Ok(Self {
            root: RootBinding::from_state(state),
            claimant,
            goal,
            semantic_horizon: caps.semantic_horizon,
            width: solver.width,
            zone: solver.zone,
            tt_enabled: solver.tt_enabled,
            tt_bytes_cap: caps.tt_bytes_cap,
            hash_mask: solver.hash_mask,
            shared_fragments: solver.shared_fragments_enabled,
            lazy_frontier: std::env::var("TSS_LAZY_FRONTIER").ok().as_deref() == Some("1"),
            lazy_key_validation: std::env::var_os("TSS_LAZY_FRONTIER_VALIDATE_KEYS").is_some(),
            interior_census_gate: std::env::var_os("TSS_INTERIOR_CENSUS_GATE")
                .is_some_and(|value| value == "1"),
            k_reply_consume: matches!(std::env::var("TSS_K_REPLY_CONSUME").as_deref(), Ok("1")),
            quotient_telemetry: std::env::var_os("TSS_TURN_QUOTIENT_TELEMETRY").is_some(),
            max_depth_cap: wide_search_final_depth(state.placements_made(), caps.semantic_horizon),
        })
    }
}

#[cfg(test)]
impl CapResumeSession {
    pub(crate) fn new(
        solver: &TssSolver,
        state: &RustHexoState,
        caps: &SolveCaps,
        goal: SolveGoal,
    ) -> Result<Self, CapResumeError> {
        let binding = CapResumeBinding::capture(solver, state, caps, goal)?;
        let local_tt_cap = if binding.tt_enabled {
            binding.tt_bytes_cap
        } else {
            0
        };
        let mut search = WidePnSearch::new_with_width(
            binding.claimant,
            state.placements_made(),
            0,
            local_tt_cap,
            binding.semantic_horizon,
            binding.max_depth_cap,
            binding.width,
            None,
        );
        debug_assert_eq!(search.lazy_frontier, binding.lazy_frontier);
        search.interior_census_gate = binding.interior_census_gate;
        let root = search.insert_root(state);
        Ok(Self {
            binding,
            search,
            root,
            stage_depth: 0,
            stage_initialized: false,
            last_node_cap: 0,
            advances: 0,
            reentries: 0,
            valid: true,
        })
    }

    /// Continue this exact query to a strictly larger total solve cap. The
    /// session is invalidated before doing fallible search/materialization work;
    /// consequently a caller that catches a panic cannot reuse partial state.
    pub(crate) fn advance_to_node_cap(
        &mut self,
        solver: &TssSolver,
        state: &RustHexoState,
        caps: &SolveCaps,
        goal: SolveGoal,
    ) -> Result<CapResumeAdvance, CapResumeError> {
        if !self.valid {
            return Err(CapResumeError::Discarded);
        }
        // Invalidate before even rebuilding the proposed binding. A panic in
        // binding capture, search, materialization, or verification therefore
        // leaves no reusable partial session for a catch_unwind caller.
        self.valid = false;
        let binding = match CapResumeBinding::capture(solver, state, caps, goal) {
            Ok(binding) => binding,
            Err(error) => return Err(error),
        };
        if binding != self.binding {
            return Err(CapResumeError::BindingMismatch);
        }
        if caps.node_cap <= self.last_node_cap {
            return Err(CapResumeError::NonMonotoneNodeCap);
        }

        self.search.node_cap = caps.node_cap - 1;
        if self.advances != 0 {
            self.reentries = self.reentries.saturating_add(1);
        }
        self.advances = self.advances.saturating_add(1);
        self.search.run_resumable(
            state,
            self.root,
            &mut self.stage_depth,
            &mut self.stage_initialized,
        );
        pn_init_finalize_wide(&self.search);

        let root_pn = self.search.entries[self.root].pn;
        let root_dn = self.search.entries[self.root].dn;
        let claimed = status_for_claimant(state.current_player(), self.binding.claimant);
        let cert = self
            .search
            .materialize(state, self.root)
            .and_then(|materialized| {
                let (nodes, root_node) =
                    compact_certificate(&materialized.arena, materialized.root_node)?;
                let mut cert = TssCertificate {
                    root: RootBinding::from_state(state),
                    claimant: self.binding.claimant,
                    root_node,
                    nodes,
                    semantic_horizon: self.binding.semantic_horizon,
                };
                rebase_zone_distances(&mut cert, state)?;
                Some(cert)
            });
        if cert
            .as_ref()
            .is_some_and(|certificate| !TssVerifier.verify(state, certificate, claimed))
        {
            return Err(CapResumeError::VerifierRejected);
        }
        let status = if cert.is_some() {
            claimed
        } else {
            ProofStatus::Unknown
        };
        let result = DeepResult {
            status,
            cert,
            stats: SolveStats {
                nodes: self.search.expansions.saturating_add(1),
                expansions: self.search.expansions,
                tt_hits: self.search.tt_hits,
                tt_entries: self.search.by_position.len() as u64,
                peak_tt_bytes: self.search.peak_bytes as u64,
                horizon_cuts: self.search.horizon_cuts,
                kb_death_cuts: self.search.kb_death_cuts,
                tt_admission_rejections: self.search.tt_index_rejections,
                fragment_lookups: self.search.fragment_lookups,
                fragment_hits: self.search.fragment_hits,
                interior_gate_evaluations: self.search.interior_gate_evaluations,
                interior_gate_dismissals: self.search.interior_gate_dismissals,
                interior_gate_nanos: self.search.interior_gate_nanos,
                stage_refreshes: self.search.stage_refreshes,
                live_ge3_seed_scans: self.search.live_ge3_seed_scans.get(),
                live_ge3_seed_nanos: self.search.live_ge3_seed_nanos.get(),
                closure_debt: *self.search.closure_stats.borrow(),
                threshold_scale: *self.search.threshold_stats.borrow(),
                ..SolveStats::default()
            },
        };
        self.last_node_cap = caps.node_cap;
        self.valid = true;
        Ok(CapResumeAdvance {
            result,
            root_pn,
            root_dn,
            stage_depth: self.stage_depth,
            advances: self.advances,
            reentries: self.reentries,
        })
    }
}

/// Imported zone fragments may have been built with a larger admissible
/// budget. Their searched set remains sound for the selected proof, but the
/// carried evidence must be relabelled from the assembled certificate itself:
/// exact D14 local budgets and the certificate's exact build horizon.
fn rebase_zone_distances(cert: &mut TssCertificate, root: &RustHexoState) -> Option<()> {
    let mut states = vec![None; cert.nodes.len()];
    let mut stack = vec![(cert.root_node, root.clone())];
    while let Some((id, state)) = stack.pop() {
        let slot = states.get_mut(id as usize)?;
        if let Some(seen) = slot.as_ref() {
            if PositionKey::from_state(seen) != PositionKey::from_state(&state) {
                return None;
            }
            continue;
        }
        *slot = Some(state.clone());
        match cert.nodes.get(id as usize)? {
            CertNode::Choice { mv, child } => {
                let mut next = state;
                let result = apply_placement(&mut next, Placement { coord: *mv }).ok()?;
                if result.outcome.is_some() {
                    return None;
                }
                stack.push((*child, next));
            }
            CertNode::Universal { edges, .. } => {
                for edge in edges {
                    let mut next = state.clone();
                    let result = apply_placement(&mut next, Placement { coord: edge.mv }).ok()?;
                    if result.outcome.is_some() {
                        return None;
                    }
                    stack.push((edge.child, next));
                }
            }
            CertNode::UniversalGroup2V1(node) => {
                for edge in &node.edges {
                    let mut next = state.clone();
                    let result = apply_placement(&mut next, Placement { coord: edge.mv }).ok()?;
                    if result.outcome.is_some() {
                        return None;
                    }
                    stack.push((edge.child, next));
                }
            }
            // Gates are never produced by this solver; fail closed.
            CertNode::FhwGateV1(_) => return None,
            _ => {}
        }
    }
    if states.iter().any(Option::is_none) {
        return None;
    }

    // `compact_certificate` emits nodes in postorder, so every ordinary child
    // precedes its parent. Reconstruct the same D14 recurrence as the
    // independent verifier: factual WIN leaves consume no defender budget,
    // LOSS leaves retain the current turn remainder, Choice passes the budget
    // through, and Universal adds one to the maximum child budget.
    let mut local_budgets = Vec::with_capacity(cert.nodes.len());
    for (index, node) in cert.nodes.iter().enumerate() {
        let local_budget = match node {
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => 0,
            CertNode::Loss { .. } => {
                let state = states.get(index)?.as_ref()?;
                u32::from(threats::placements_remaining(state))
            }
            CertNode::Choice { child, .. } => {
                let child = *child as usize;
                if child >= index {
                    return None;
                }
                *local_budgets.get(child)?
            }
            CertNode::Universal { edges, .. } => {
                let mut maximum = 0u32;
                for edge in edges {
                    let child = edge.child as usize;
                    if child >= index {
                        return None;
                    }
                    maximum = maximum.max(*local_budgets.get(child)?);
                }
                maximum.saturating_add(1)
            }
            CertNode::UniversalGroup2V1(node) => {
                let mut maximum = 0u32;
                for edge in &node.edges {
                    let child = edge.child as usize;
                    if child >= index {
                        return None;
                    }
                    maximum = maximum.max(*local_budgets.get(child)?);
                }
                maximum.saturating_add(1)
            }
            CertNode::FhwGateV1(_) => return None,
        };
        local_budgets.push(local_budget);
    }

    let build_horizon = cert.semantic_horizon;
    for (index, node) in cert.nodes.iter_mut().enumerate() {
        if let CertNode::Universal {
            zone: Some(zone), ..
        } = node
        {
            let Some(&local_budget) = local_budgets.get(index) else {
                return None;
            };
            zone.d = local_budget;
            zone.build_horizon = build_horizon;
        }
    }
    Some(())
}

/// T10/U18 relabelling for an assembled shared DAG. The current Rust grammar
/// serializes D14's local budget but not D15/D16 tables, so the representable
/// max-dominant join is the exact child-max recurrence below. Reachable
/// protected/core obligations remain the union of the final outgoing DAG and
/// are independently reconstructed by `TssVerifier`.
fn rebase_shared_fragment_labels(cert: &mut TssCertificate, root: &RustHexoState) -> Option<u64> {
    fn visit(
        cert: &TssCertificate,
        id: CertNodeId,
        state: &RustHexoState,
        memo: &mut [Option<(PositionKey, u32)>],
        visiting: &mut [bool],
        depth: usize,
    ) -> Option<u32> {
        if depth > MAX_CERT_DEPTH {
            return None;
        }
        let index = id as usize;
        let key = PositionKey::from_state(state);
        if let Some((seen, budget)) = memo.get(index)?.as_ref() {
            return (seen == &key).then_some(*budget);
        }
        if *visiting.get(index)? {
            return None;
        }
        visiting[index] = true;
        let budget = match cert.nodes.get(index)? {
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => 0,
            CertNode::Loss { .. } => u32::from(threats::placements_remaining(state)),
            CertNode::Choice { mv, child } => {
                let mut next = state.clone();
                let result = apply_placement(&mut next, Placement { coord: *mv }).ok()?;
                if result.outcome.is_some() {
                    return None;
                }
                visit(cert, *child, &next, memo, visiting, depth + 1)?
            }
            CertNode::Universal { edges, .. } => {
                let mut maximum = 0u32;
                for edge in edges {
                    let mut next = state.clone();
                    let result = apply_placement(&mut next, Placement { coord: edge.mv }).ok()?;
                    if result.outcome.is_some() {
                        return None;
                    }
                    maximum =
                        maximum.max(visit(cert, edge.child, &next, memo, visiting, depth + 1)?);
                }
                maximum.saturating_add(1)
            }
            // Shared-fragment relabelling never sees extension nodes (they
            // are excluded from fragment promotion); fail closed.
            CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
        };
        visiting[index] = false;
        memo[index] = Some((key, budget));
        Some(budget)
    }

    let mut memo = vec![None; cert.nodes.len()];
    let mut visiting = vec![false; cert.nodes.len()];
    visit(cert, cert.root_node, root, &mut memo, &mut visiting, 0)?;
    if memo.iter().any(Option::is_none) {
        return None;
    }
    let mut relabelled = 0u64;
    for (index, labelled) in memo.into_iter().enumerate() {
        let (_, budget) = labelled?;
        if let CertNode::Universal {
            zone: Some(zone), ..
        } = cert.nodes.get_mut(index)?
        {
            relabelled = relabelled.saturating_add(u64::from(
                zone.d != budget || zone.build_horizon != cert.semantic_horizon,
            ));
            zone.d = budget;
            zone.build_horizon = cert.semantic_horizon;
        }
    }
    Some(relabelled)
}

fn split_tt_cap(total: usize) -> (usize, usize) {
    let shared = total / 2;
    (total - shared, shared)
}

/// Split the post-root wide `Both` allowance while always leaving a nonempty
/// primal allowance when any post-root work is available. This rules out a
/// configuration value silently turning a `Both` solve into loss-only search.
fn wide_both_initial_caps(remaining: u64, loss_reserve_nodes: u32) -> (u64, u64) {
    let reserve = u64::from(loss_reserve_nodes).min(remaining.saturating_sub(1));
    (remaining - reserve, reserve)
}

fn goal_accepts(goal: SolveGoal, status: ProofStatus) -> bool {
    matches!(
        (goal, status),
        (SolveGoal::Both, ProofStatus::Win | ProofStatus::Loss)
            | (SolveGoal::Win, ProofStatus::Win)
            | (SolveGoal::Loss, ProofStatus::Loss)
    )
}

struct AttemptResult {
    cert: Option<TssCertificate>,
    stats: SolveStats,
    #[cfg(test)]
    tt_signature: Option<NarrowAttemptSignature>,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct NarrowAttemptSignature {
    nodes: u64,
    tt_hits: u64,
    hit_limit: bool,
    arena_len: usize,
    edge_count: usize,
    local_tt: String,
}

fn unknown<C>(stats: SolveStats) -> DeepResult<C> {
    DeepResult {
        status: ProofStatus::Unknown,
        cert: None,
        stats,
    }
}

fn status_for_claimant(root_player: Player, claimant: Player) -> ProofStatus {
    if root_player == claimant {
        ProofStatus::Win
    } else {
        ProofStatus::Loss
    }
}

/// Return the player proved to win at this node and the corresponding compact
/// leaf.  Lambda-one is intentionally unavailable at Opening (the shared
/// theorem is post-opening), although reachable Opening currently has no
/// threats and therefore cannot produce a verdict anyway.
fn immediate_winner(state: &RustHexoState, width: WidthOptions) -> Option<(Player, CertNode)> {
    if state.is_terminal() {
        return None;
    }
    if matches!(state.phase(), TurnPhase::Opening) {
        return None;
    }
    let analysis = threats::analyze(state);
    let winner = winner_from_analysis(state, &analysis)?;
    typed_lambda_leaf(state, winner, &analysis, width).map(|leaf| (winner, leaf))
}

fn window_key_order(key: WindowKey) -> (u8, i16, i16) {
    (key.axis.index(), key.start.q, key.start.r)
}

const L13_LOSS_WITNESS_CAP_B1: usize = 3;
const L13_LOSS_WITNESS_CAP_B2: usize = 5;

/// Whether no set of at most `budget` cells hits every member of `family`.
/// Connect-6 loss leaves only use budgets one and two; an unsupported budget
/// deliberately returns false so callers cannot emit an unproved witness.
fn family_hitting_exceeds_budget(family: &[Vec<HexCoord>], budget: u8) -> bool {
    if family.is_empty() {
        return false;
    }
    if family.iter().any(Vec::is_empty) {
        return true;
    }

    let mut universe = family.iter().flatten().copied().collect::<Vec<_>>();
    universe.sort_by_key(|coord| (coord.q, coord.r));
    universe.dedup();
    if budget >= 1
        && universe
            .iter()
            .any(|cell| family.iter().all(|set| set.contains(cell)))
    {
        return false;
    }
    if budget >= 2 {
        for left in 0..universe.len() {
            for right in (left + 1)..universe.len() {
                if family
                    .iter()
                    .all(|set| set.contains(&universe[left]) || set.contains(&universe[right]))
                {
                    return false;
                }
            }
        }
    }
    matches!(budget, 1 | 2)
}

/// L13 reverse deletion preserves the earliest canonical family members when
/// either choice is redundant and returns an inclusion-minimal obstruction.
fn inclusion_minimal_loss_obstruction(family: &[Vec<HexCoord>], budget: u8) -> Option<Vec<usize>> {
    let cap = match budget {
        1 => L13_LOSS_WITNESS_CAP_B1,
        2 => L13_LOSS_WITNESS_CAP_B2,
        _ => return None,
    };
    if !family_hitting_exceeds_budget(family, budget) {
        return None;
    }

    let mut kept = (0..family.len()).collect::<Vec<_>>();
    for candidate in (0..family.len()).rev() {
        let trial = kept
            .iter()
            .copied()
            .filter(|index| *index != candidate)
            .map(|index| family[index].clone())
            .collect::<Vec<_>>();
        if family_hitting_exceeds_budget(&trial, budget) {
            kept.retain(|index| *index != candidate);
        }
    }

    debug_assert!(kept.iter().all(|removed| {
        let trial = kept
            .iter()
            .copied()
            .filter(|index| index != removed)
            .map(|index| family[index].clone())
            .collect::<Vec<_>>();
        !family_hitting_exceeds_budget(&trial, budget)
    }));
    if kept.len() > cap {
        return None;
    }
    Some(kept)
}

/// Materialize the sparse obstruction as window identities.  The proved 3/5
/// bounds are checked rather than assumed: an unexpected violation fails
/// closed by declining to materialize a tactical LOSS leaf.
fn sparse_loss_witnesses(
    state: &RustHexoState,
    winner: Player,
    budget: u8,
) -> Option<Vec<WindowKey>> {
    let mut family = state
        .board()
        .windows()
        .live_threat_entries()
        .filter_map(|(owner, entry)| (owner == winner).then(|| (entry.key(), entry.empty_cells())))
        .collect::<Vec<_>>();
    family.sort_by_key(|(key, _)| window_key_order(*key));
    family.dedup_by_key(|(key, _)| *key);

    let full_sets = family
        .iter()
        .map(|(_, empties)| empties.clone())
        .collect::<Vec<_>>();
    let kept = inclusion_minimal_loss_obstruction(&full_sets, budget)?;
    Some(kept.into_iter().map(|index| family[index].0).collect())
}

fn typed_lambda_leaf(
    state: &RustHexoState,
    winner: Player,
    analysis: &threats::ThreatAnalysis,
    width: WidthOptions,
) -> Option<CertNode> {
    if winner == state.current_player() {
        let mut candidates = state
            .board()
            .windows()
            .entries()
            .filter(|entry| {
                entry.active_player() == Some(winner)
                    && (entry.count(winner) == 5 || (analysis.b == 2 && entry.count(winner) == 4))
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|entry| {
            (
                std::cmp::Reverse(entry.count(winner)),
                window_key_order(entry.key()),
            )
        });
        let witness = candidates.first().copied()?;
        let count = witness.count(winner);
        let extra = if count == 5 { 1 } else { 2 };
        Some(CertNode::Win {
            witness: witness.key(),
            count,
            budget: analysis.b,
            resolution_ply: state.placements_made().saturating_add(extra),
        })
    } else {
        let witnesses = if width.vcf_pair_complete {
            sparse_loss_witnesses(state, winner, analysis.b)?
        } else {
            let mut witnesses = state
                .board()
                .windows()
                .live_threat_entries()
                .filter_map(|(owner, entry)| (owner == winner).then_some(entry.key()))
                .collect::<Vec<_>>();
            witnesses.sort_by_key(|key| window_key_order(*key));
            witnesses.dedup();
            witnesses
        };
        (!witnesses.is_empty()).then_some(CertNode::Loss {
            witnesses,
            resolution_ply: state
                .placements_made()
                .saturating_add(u32::from(analysis.b))
                .saturating_add(2),
        })
    }
}

fn node_resolution(node: &CertNode) -> u32 {
    match node {
        CertNode::OrCompletion { completion_ply, .. } => *completion_ply,
        CertNode::Win { resolution_ply, .. } | CertNode::Loss { resolution_ply, .. } => {
            *resolution_ply
        }
        CertNode::UniversalGroup2V1(_) => 0,
        // R1: a gate's escape deadline participates in the derived resolution.
        CertNode::FhwGateV1(gate) => gate.proof.escape_resolution_ply,
        CertNode::Choice { .. } | CertNode::Universal { .. } => 0,
    }
}

fn winner_from_analysis(
    state: &RustHexoState,
    analysis: &threats::ThreatAnalysis,
) -> Option<Player> {
    if analysis.own_win_now {
        Some(state.current_player())
    } else if analysis.forced_loss() {
        Some(state.current_player().other())
    } else {
        None
    }
}

const PN_INFINITY: u32 = 1_000_000_000;

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ThresholdDelta {
    One,
    Two,
    Four,
    MeanSiblingPrior,
}

#[cfg(test)]
impl ThresholdDelta {
    fn from_env() -> Option<Self> {
        let value = std::env::var("TSS_THRESHOLD_DELTA").ok()?;
        Some(match value.as_str() {
            "1" => Self::One,
            "2" => Self::Two,
            "4" => Self::Four,
            "mean" => Self::MeanSiblingPrior,
            _ => panic!("TSS_THRESHOLD_DELTA must be one of 1, 2, 4, mean"),
        })
    }
}

/// Default-off live ordering arm for retained attacker-pair children. The
/// numeric values are the public `TSS_ZONE_ORDER` contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ZoneOrderMode {
    #[default]
    Off,
    ZoneBound,
    DStone,
}

impl ZoneOrderMode {
    fn from_env() -> Self {
        match std::env::var("TSS_ZONE_ORDER").ok().as_deref() {
            None | Some("") | Some("0") => Self::Off,
            Some("1") => Self::ZoneBound,
            Some("2") => Self::DStone,
            Some(_) => panic!("TSS_ZONE_ORDER must be one of 0, 1, 2"),
        }
    }

    fn enabled(self) -> bool {
        self != Self::Off
    }

    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::ZoneBound => "zone_bound",
            Self::DStone => "d_stone",
        }
    }
}

fn zone_order_band_from_env(mode: ZoneOrderMode) -> u32 {
    if !mode.enabled() {
        return 0;
    }
    std::env::var("TSS_ZONE_ORDER_BAND")
        .ok()
        .map(|value| {
            value
                .parse::<u32>()
                .expect("TSS_ZONE_ORDER_BAND must be a nonnegative integer")
        })
        .unwrap_or(0)
}

#[cfg(test)]
static WIDE_GEN_PAIR_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static WIDE_GEN_DEFENDER_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static WIDE_GEN_PRIOR_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static WIDE_EXPAND_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static WIDE_REFRESH_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static WIDE_INSERT_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static WIDE_ZONE_ORDER_CONTEXTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static WIDE_ZONE_ORDER_CONTEXT_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static WIDE_ZONE_ORDER_KEYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static WIDE_ZONE_ORDER_KEY_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Temporary hot-path attribution for the corpus harness: accumulated wall
/// nanos of the wide generators/priors, read via `wide_gen_profile`.
#[cfg(test)]
struct WideGenTimer<'a> {
    sink: &'a std::sync::atomic::AtomicU64,
    start: std::time::Instant,
}

#[cfg(test)]
impl<'a> WideGenTimer<'a> {
    fn start(sink: &'a std::sync::atomic::AtomicU64) -> Self {
        Self {
            sink,
            start: std::time::Instant::now(),
        }
    }
}

#[cfg(test)]
impl Drop for WideGenTimer<'_> {
    fn drop(&mut self) {
        let nanos = u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.sink
            .fetch_add(nanos, std::sync::atomic::Ordering::Relaxed);
    }
}

/// (pair_gen_ms, defender_gen_ms, regen_ms, expand_ms, refresh_ms, insert_ms)
/// accumulated since process start.
#[cfg(test)]
pub(crate) fn wide_gen_profile() -> (u64, u64, u64, u64, u64, u64) {
    (
        WIDE_GEN_PAIR_NANOS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        WIDE_GEN_DEFENDER_NANOS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        WIDE_GEN_PRIOR_NANOS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        WIDE_EXPAND_NANOS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        WIDE_REFRESH_NANOS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
        WIDE_INSERT_NANOS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
    )
}

/// (context builds, retained-child keys, context nanos, key nanos) accumulated
/// since process start. Only the live flag contributes; the offline R-OS1
/// observer is deliberately excluded.
#[cfg(test)]
pub(crate) fn zone_order_profile() -> (u64, u64, u64, u64) {
    (
        WIDE_ZONE_ORDER_CONTEXTS.load(std::sync::atomic::Ordering::Relaxed),
        WIDE_ZONE_ORDER_KEYS.load(std::sync::atomic::Ordering::Relaxed),
        WIDE_ZONE_ORDER_CONTEXT_NANOS.load(std::sync::atomic::Ordering::Relaxed),
        WIDE_ZONE_ORDER_KEY_NANOS.load(std::sync::atomic::Ordering::Relaxed),
    )
}

#[cfg(test)]
pub(crate) fn zone_order_config() -> (&'static str, u32) {
    let mode = ZoneOrderMode::from_env();
    (mode.name(), zone_order_band_from_env(mode))
}
/// Small conjunctions can exploit ordinary PN re-selection and shared TT work
/// without the multiplicative interleaving measured at the four-way AND
/// frontier. Latch visit-order commitment only once at least four distinct
/// linked proof obligations are live.
const MIN_COMMITTED_UNIVERSAL_OBLIGATIONS: usize = 4;
/// Each placement belongs to at most 18 length-six windows (six starts on
/// each of three axes), so a completed two-stone turn can create at most 36
/// distinct threats.  This geometry bound turns fork degree into a compact,
/// strictly monotone proof prior without a tuned scale.
const MAX_TURN_FORK_DEGREE: u32 = 36;

fn pn_from_fork_degree(fork_degree: usize) -> u32 {
    let fork_degree = u32::try_from(fork_degree)
        .unwrap_or(u32::MAX)
        .min(MAX_TURN_FORK_DEGREE);
    MAX_TURN_FORK_DEGREE + 1 - fork_degree
}

fn dn_from_tau(tau: Option<u8>) -> u32 {
    tau.map(u32::from).unwrap_or(1).max(1)
}

/// No proof can use more placements than either the caller's remaining
/// semantic horizon or the verifier's maximum replay depth.
fn wide_search_final_depth(root_ply: u32, semantic_horizon: u32) -> usize {
    usize::try_from(semantic_horizon.saturating_sub(root_ply))
        .unwrap_or(usize::MAX)
        .min(MAX_SEARCH_DEPTH)
}

/// Advance only to an exact, strictly deeper selected cutoff. Repeated or
/// regressive observations terminate fail-closed instead of spinning.
fn next_wide_stage_depth(
    current_depth: usize,
    encountered_depth: usize,
    final_depth: usize,
) -> Option<usize> {
    let next_depth = encountered_depth.min(final_depth);
    (next_depth > current_depth).then_some(next_depth)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WidePnPrior {
    pn: u32,
    dn: u32,
}

impl WidePnPrior {
    const UNIFORM: Self = Self { pn: 1, dn: 1 };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WidePnKind {
    Choice,
    Universal { implicit_dispatch: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WidePnMove {
    One(HexCoord),
    Pair(HexCoord, HexCoord),
    /// One complete, forced defender turn. This is emitted only by the
    /// wide pair-canonicalization path; unlike `Pair`, it materializes as an
    /// implicit Universal turn with a checked commutation witness.
    DefenderPair(HexCoord, HexCoord),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WidePnChildResult {
    Pending,
    ClaimantCompletion,
    ClaimantTactical,
    Refuted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WidePnStepOutcome {
    Progress,
    DepthCutoff { depth: usize, made_progress: bool },
    Stalled,
}

/// Cheap generation-time observations used only by the offline ordering
/// study. Smaller values are better except `gate_adjacency`, where larger is
/// better. `zone_bound` is the maximum claimant-support distance needed to
/// cover every placement in the child; `d_stone` is the corresponding minimum.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OrderingChildFeatures {
    zone_bound: u16,
    census_distance: u16,
    gate_adjacency: u16,
    d_stone: u16,
}

struct OrderingFeatureContext {
    claimant_stones: Vec<HexCoord>,
    #[cfg(test)]
    claimant_windows: Vec<(u8, Vec<HexCoord>)>,
    #[cfg(test)]
    defender_gate_cells: Vec<HexCoord>,
}

impl OrderingFeatureContext {
    fn from_state(state: &RustHexoState, claimant: Player, observe_study: bool) -> Self {
        let claimant_stones = state
            .board()
            .occupied_cells()
            .iter()
            .copied()
            .filter(|&coord| state.board().get(coord) == Some(claimant))
            .collect();
        #[cfg(test)]
        let mut claimant_windows = Vec::new();
        #[cfg(test)]
        let mut defender_gate_cells = Vec::new();
        #[cfg(test)]
        if observe_study {
            for entry in state.board().windows().entries() {
                let Some(owner) = entry.active_player() else {
                    continue;
                };
                let count = entry.count(owner);
                if owner == claimant && count > 0 {
                    claimant_windows.push((count, entry.empty_cells()));
                } else if owner == claimant.other() && count >= 4 {
                    defender_gate_cells.extend(entry.empty_cells());
                }
            }
            defender_gate_cells.sort_unstable_by_key(|coord| raw_coord_key(*coord));
            defender_gate_cells.dedup();
        }
        #[cfg(not(test))]
        let _ = observe_study;
        Self {
            claimant_stones,
            #[cfg(test)]
            claimant_windows,
            #[cfg(test)]
            defender_gate_cells,
        }
    }

    fn nearest_claimant_distance(&self, placed: HexCoord) -> u16 {
        self.claimant_stones
            .iter()
            .map(|&stone| hex_distance(placed, stone))
            .min()
            .and_then(|distance| u16::try_from(distance).ok())
            .unwrap_or(u16::MAX)
    }

    fn pair_key(&self, first: HexCoord, second: HexCoord, mode: ZoneOrderMode) -> u16 {
        let first = self.nearest_claimant_distance(first);
        let second = self.nearest_claimant_distance(second);
        match mode {
            ZoneOrderMode::Off => 0,
            ZoneOrderMode::ZoneBound => first.max(second),
            ZoneOrderMode::DStone => first.min(second),
        }
    }

    #[cfg(test)]
    fn cached_nearest_claimant_distance(
        &self,
        placed: HexCoord,
        cache: &mut HashMap<HexCoord, u16>,
    ) -> u16 {
        if let Some(&distance) = cache.get(&placed) {
            return distance;
        }
        let distance = self.nearest_claimant_distance(placed);
        cache.insert(placed, distance);
        distance
    }

    #[cfg(test)]
    fn features(&self, placements: &[HexCoord]) -> OrderingChildFeatures {
        let distances = placements
            .iter()
            .map(|&placed| {
                self.claimant_stones
                    .iter()
                    .map(|&stone| hex_distance(placed, stone))
                    .min()
                    .and_then(|distance| u16::try_from(distance).ok())
                    .unwrap_or(u16::MAX)
            })
            .collect::<Vec<_>>();
        let d_stone = distances.iter().copied().min().unwrap_or(u16::MAX);
        let zone_bound = distances.iter().copied().max().unwrap_or(u16::MAX);
        let post_census = self
            .claimant_windows
            .iter()
            .map(|(count, empties)| {
                count.saturating_add(
                    placements
                        .iter()
                        .filter(|placed| empties.contains(placed))
                        .count()
                        .try_into()
                        .unwrap_or(u8::MAX),
                )
            })
            .max()
            .unwrap_or(0)
            .min(6);
        let gate_adjacency = placements
            .iter()
            .filter(|&&placed| {
                self.defender_gate_cells
                    .iter()
                    .any(|&gate| hex_distance(placed, gate) <= 1)
            })
            .count()
            .try_into()
            .unwrap_or(u16::MAX);
        OrderingChildFeatures {
            zone_bound,
            census_distance: u16::from(6u8.saturating_sub(post_census)),
            gate_adjacency,
            d_stone,
        }
    }
}

#[derive(Clone, Debug)]
struct WidePnChild {
    mv: WidePnMove,
    result: WidePnChildResult,
    entry: Option<usize>,
    /// Exact key retained in lazy mode until the edge links an arena entry.
    /// Defender keys virtually represent the eager entry before selection;
    /// historical attacker-lazy keys remain selection-only.
    future_key: Option<WideFutureKey>,
    /// Static estimates used until the child position is linked. Completed
    /// attacker turns carry both their fork-derived PN and tau-derived DN so
    /// lazy linking cannot erase the principled ordering signal.
    prior: WidePnPrior,
    urgent_block: bool,
    /// Width class of the first placement in an atomic attacker pair.  Zero is
    /// also the neutral value for one-placement and defender children, so the
    /// root-only tier prior cannot perturb their established ordering.
    first_width_tier: u8,
    /// Live R-OS2 distance key. Zero is the inert default used by flag-off and
    /// every non-pair child; selection consults it only when the solve-local
    /// mode is enabled at an attacker Choice.
    zone_order_key: u16,
    #[cfg(test)]
    ordering: OrderingChildFeatures,
}

#[derive(Clone, Debug)]
enum WideFutureKey {
    /// Historical attacker lazy edge: the key participates only when selected.
    OnSelection(WidePositionKey),
    /// R-LF1 defender thunk: pre-selection reads virtually observe the eager
    /// entry represented by the deferred key.
    Virtual(WidePositionKey),
}

impl WideFutureKey {
    fn key(&self) -> &WidePositionKey {
        match self {
            Self::OnSelection(key) | Self::Virtual(key) => key,
        }
    }

    fn virtual_key(&self) -> Option<&WidePositionKey> {
        match self {
            Self::Virtual(key) => Some(key),
            Self::OnSelection(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WideChildObligation<'a> {
    Entry(usize),
    FutureKey(&'a WidePositionKey),
}

#[derive(Clone, Copy, Debug)]
struct WideDeferredPosition {
    depth: usize,
    prior: WidePnPrior,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ThresholdBandSelection {
    kind: WidePnKind,
    gap_bin: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
struct ClosurePairNodeProfile {
    evaluated: u64,
    second_candidate_nanos: u64,
    pair_evaluation_nanos: u64,
    dedup_nanos: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
struct RevealZoneWork {
    evaluated: u64,
    second_candidate_nanos: u64,
    pair_evaluation_nanos: u64,
    dedup_nanos: u64,
}

#[cfg(test)]
#[derive(Clone, Debug, Default)]
struct RevealPairNodeProfile {
    /// Indexed by the byte-exact `u16` zone-bound value. Empty intervening
    /// buckets are retained so prefix sums have no comparison ambiguity.
    zone: Vec<RevealZoneWork>,
}

#[cfg(test)]
impl RevealPairNodeProfile {
    fn work_mut(&mut self, zone_bound: u16) -> &mut RevealZoneWork {
        let index = usize::from(zone_bound);
        if self.zone.len() <= index {
            self.zone
                .resize(index.saturating_add(1), RevealZoneWork::default());
        }
        &mut self.zone[index]
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
struct ClosurePairChildProfile {
    evaluation_ordinal: u64,
    second_candidate_nanos: u64,
    pair_evaluation_nanos: u64,
    dedup_nanos: u64,
    zone_bound: u16,
    zone_evaluation_prefix: u64,
    zone_second_candidate_nanos: u64,
    zone_pair_evaluation_nanos: u64,
    zone_dedup_nanos: u64,
    selected: bool,
    linked: bool,
    expanded: bool,
}

#[cfg(test)]
struct ThresholdResidencyGuard {
    enabled: bool,
    stats: Rc<RefCell<ThresholdScaleStats>>,
    expansion_clock: Rc<Cell<u64>>,
    start_expansions: u64,
    active_since: Option<Instant>,
    exclusive_nanos: u64,
}

#[cfg(test)]
impl ThresholdResidencyGuard {
    fn disabled(stats: Rc<RefCell<ThresholdScaleStats>>, expansion_clock: Rc<Cell<u64>>) -> Self {
        Self {
            enabled: false,
            stats,
            expansion_clock,
            start_expansions: 0,
            active_since: None,
            exclusive_nanos: 0,
        }
    }

    fn enabled(stats: Rc<RefCell<ThresholdScaleStats>>, expansion_clock: Rc<Cell<u64>>) -> Self {
        let start_expansions = expansion_clock.get();
        Self {
            enabled: true,
            stats,
            expansion_clock,
            start_expansions,
            active_since: Some(Instant::now()),
            exclusive_nanos: 0,
        }
    }

    fn pause(&mut self) {
        if let Some(started) = self.active_since.take() {
            self.exclusive_nanos = self
                .exclusive_nanos
                .saturating_add(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        }
    }

    fn resume(&mut self) {
        if self.enabled {
            debug_assert!(self.active_since.is_none());
            self.active_since = Some(Instant::now());
        }
    }

    fn state_started(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    fn record_state_elapsed(&self, started: Option<Instant>) {
        let Some(started) = started else {
            return;
        };
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let mut stats = self.stats.borrow_mut();
        stats.state_apply_undo_nanos = stats.state_apply_undo_nanos.saturating_add(nanos);
    }

    fn record_threshold_cross(&self) {
        if self.enabled {
            let mut stats = self.stats.borrow_mut();
            stats.threshold_cross_returns = stats.threshold_cross_returns.saturating_add(1);
        }
    }
}

#[cfg(test)]
impl Drop for ThresholdResidencyGuard {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        self.pause();
        let expansions = self
            .expansion_clock
            .get()
            .saturating_sub(self.start_expansions);
        let bin = match expansions {
            0 => 0,
            1 => 1,
            2 => 2,
            3..=4 => 3,
            5..=8 => 4,
            9..=16 => 5,
            17..=32 => 6,
            _ => 7,
        };
        let mut stats = self.stats.borrow_mut();
        stats.residencies = stats.residencies.saturating_add(1);
        stats.residency_expansions = stats.residency_expansions.saturating_add(expansions);
        stats.residency_expansion_bins[bin] = stats.residency_expansion_bins[bin].saturating_add(1);
        stats.descent_nanos = stats.descent_nanos.saturating_add(self.exclusive_nanos);
    }
}

fn turn_start_defender_blocks(candidates: &[Candidate]) -> HashSet<HexCoord> {
    candidates
        .iter()
        .filter_map(|candidate| candidate.defender_block.then_some(candidate.coord))
        .collect()
}

fn wide_move_contains_defender_block(mv: WidePnMove, defender_blocks: &HashSet<HexCoord>) -> bool {
    match mv {
        WidePnMove::One(coord) => defender_blocks.contains(&coord),
        WidePnMove::Pair(first, second) | WidePnMove::DefenderPair(first, second) => {
            defender_blocks.contains(&first) || defender_blocks.contains(&second)
        }
    }
}

fn wide_choice_has_urgent_block(children: &[WidePnChild]) -> bool {
    children.iter().any(|child| child.urgent_block)
}

#[derive(Clone, Debug)]
enum WidePnNode {
    Unexpanded,
    ProvenLeaf(CertNode),
    /// Independently verified, exact-key positive proof retained by the
    /// solver-owned cross-solve store. The Arc is immutable for this run.
    ProvenFragment(Arc<ProvenFragment>),
    /// This restricted horizon did not reach a proof. Unlike a genuine
    /// refutation, the node is reopened when the retained search deepens.
    DepthCutoff,
    Refuted,
    Branch {
        kind: WidePnKind,
        children: Vec<WidePnChild>,
    },
}

#[cfg(test)]
fn wide_pn_node_tag(node: &WidePnNode) -> &'static str {
    match node {
        WidePnNode::Unexpanded => "unexpanded",
        WidePnNode::ProvenLeaf(_) => "proven_leaf",
        WidePnNode::ProvenFragment(_) => "proven_fragment",
        WidePnNode::DepthCutoff => "depth_cutoff",
        WidePnNode::Refuted => "refuted",
        WidePnNode::Branch {
            kind: WidePnKind::Choice,
            ..
        } => "choice",
        WidePnNode::Branch {
            kind: WidePnKind::Universal { .. },
            ..
        } => "universal",
    }
}

#[derive(Clone, Debug)]
struct WidePnEntry {
    pn: u32,
    dn: u32,
    /// Immutable initialization restored whenever a staged depth cutoff is
    /// reopened. Recompute may replace the live numbers with child aggregates,
    /// but it never destroys these state-derived priors.
    prior: WidePnPrior,
    node: WidePnNode,
    depth: usize,
    /// Wide-mode visit-order state for an AND node. Once an unresolved
    /// defender obligation is selected, keep driving that same child until it
    /// proves or refutes. This does not participate in PN/DN recomputation or
    /// certificate materialization.
    universal_obligation: Option<usize>,
}

#[derive(Clone, Debug)]
struct WideProvenCandidate {
    id: usize,
    state: RustHexoState,
}

/// Exact, compact key used only by the wide proof-number frontier. Coordinates
/// are zig-zag/varint encoded after sorting `(q,r,owner)` tuples, so equality is
/// collision-free while dense late-game boards do not duplicate a padded
/// `StoneKey` vector in every transposition entry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WidePositionKey {
    bytes: Box<[u8]>,
}

impl WidePositionKey {
    fn from_state(state: &RustHexoState) -> Self {
        let mut stones = state
            .board()
            .occupied_cells()
            .iter()
            .map(|&coord| {
                (
                    coord.q,
                    coord.r,
                    player_code(state.board().get(coord).expect("occupied cell has owner")),
                )
            })
            .collect::<Vec<_>>();
        stones.sort_unstable();
        let mut encoded = Vec::with_capacity(stones.len().saturating_mul(3).saturating_add(12));
        encoded.push(player_code(state.current_player()));
        push_wide_varint(&mut encoded, state.placements_made());
        match state.phase() {
            TurnPhase::Opening => encoded.push(0),
            TurnPhase::FirstStone => encoded.push(1),
            TurnPhase::SecondStone { first } => {
                encoded.push(2);
                push_wide_varint(&mut encoded, zigzag_i16(first.q));
                push_wide_varint(&mut encoded, zigzag_i16(first.r));
            }
        }
        match state.terminal() {
            None => encoded.push(0),
            Some(outcome) => {
                encoded.push(1 + player_code(outcome.winner));
                push_wide_varint(&mut encoded, outcome.placements);
            }
        }
        for (q, r, owner) in stones {
            push_wide_varint(
                &mut encoded,
                zigzag_i16(q).saturating_mul(2) | u32::from(owner),
            );
            push_wide_varint(&mut encoded, zigzag_i16(r));
        }
        Self {
            bytes: encoded.into_boxed_slice(),
        }
    }

    /// Exact nonterminal key after a legal two-stone turn. The wide attacker
    /// pair gate is deliberately stateless, so constructing the thunk key
    /// directly avoids cloning/applying the engine state for every retained
    /// pair.
    fn after_completed_pair(state: &RustHexoState, first: HexCoord, second: HexCoord) -> Self {
        debug_assert!(matches!(state.phase(), TurnPhase::FirstStone));
        let owner = player_code(state.current_player());
        let mut stones = state
            .board()
            .occupied_cells()
            .iter()
            .map(|&coord| {
                (
                    coord.q,
                    coord.r,
                    player_code(state.board().get(coord).expect("occupied cell has owner")),
                )
            })
            .collect::<Vec<_>>();
        stones.push((first.q, first.r, owner));
        stones.push((second.q, second.r, owner));
        stones.sort_unstable();

        let mut encoded = Vec::with_capacity(stones.len().saturating_mul(3).saturating_add(12));
        encoded.push(player_code(state.current_player().other()));
        push_wide_varint(&mut encoded, state.placements_made().saturating_add(2));
        encoded.push(1); // TurnPhase::FirstStone after a completed turn.
        encoded.push(0); // Retained Pending pairs are nonterminal.
        for (q, r, owner) in stones {
            push_wide_varint(
                &mut encoded,
                zigzag_i16(q).saturating_mul(2) | u32::from(owner),
            );
            push_wide_varint(&mut encoded, zigzag_i16(r));
        }
        Self {
            bytes: encoded.into_boxed_slice(),
        }
    }

    #[cfg(test)]
    fn from_position_key(key: &PositionKey) -> Self {
        let mut encoded = Vec::with_capacity(key.stones.len().saturating_mul(3).saturating_add(12));
        encoded.push(key.current_player);
        push_wide_varint(&mut encoded, key.placements_made);
        match key.phase {
            KeyPhase::Opening => encoded.push(0),
            KeyPhase::FirstStone => encoded.push(1),
            KeyPhase::SecondStone { q, r } => {
                encoded.push(2);
                push_wide_varint(&mut encoded, zigzag_i16(q));
                push_wide_varint(&mut encoded, zigzag_i16(r));
            }
        }
        match key.terminal {
            None => encoded.push(0),
            Some(terminal) => {
                encoded.push(1 + terminal.winner);
                push_wide_varint(&mut encoded, terminal.placements);
            }
        }
        for stone in &key.stones {
            push_wide_varint(
                &mut encoded,
                zigzag_i16(stone.q).saturating_mul(2) | u32::from(stone.owner),
            );
            push_wide_varint(&mut encoded, zigzag_i16(stone.r));
        }
        Self {
            bytes: encoded.into_boxed_slice(),
        }
    }

    fn heap_bytes(&self) -> usize {
        self.bytes.len()
    }

    #[cfg(test)]
    fn d6_canonical(&self) -> Self {
        fn take_varint(bytes: &[u8], cursor: &mut usize) -> u32 {
            let mut value = 0u32;
            let mut shift = 0u32;
            loop {
                let byte = bytes[*cursor];
                *cursor += 1;
                value |= u32::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    return value;
                }
                shift += 7;
                assert!(shift < 35, "wide-position varint overflow");
            }
        }
        fn unzigzag(value: u32) -> i16 {
            let signed = ((value >> 1) as i32) ^ -((value & 1) as i32);
            i16::try_from(signed).expect("wide-position coordinate is i16")
        }

        let mut cursor = 0usize;
        let current_player = self.bytes[cursor];
        cursor += 1;
        let placements_made = take_varint(&self.bytes, &mut cursor);
        let phase_tag = self.bytes[cursor];
        cursor += 1;
        let phase_first = if phase_tag == 2 {
            Some((
                unzigzag(take_varint(&self.bytes, &mut cursor)),
                unzigzag(take_varint(&self.bytes, &mut cursor)),
            ))
        } else {
            None
        };
        let terminal_tag = self.bytes[cursor];
        cursor += 1;
        let terminal_placements = if terminal_tag == 0 {
            None
        } else {
            Some(take_varint(&self.bytes, &mut cursor))
        };
        let mut stones = Vec::new();
        while cursor < self.bytes.len() {
            let packed_q = take_varint(&self.bytes, &mut cursor);
            let owner = (packed_q & 1) as u8;
            let q = unzigzag(packed_q >> 1);
            let r = unzigzag(take_varint(&self.bytes, &mut cursor));
            stones.push((HexCoord::new(q, r), owner));
        }

        let mut best: Option<Vec<u8>> = None;
        for symmetry in 0..12u8 {
            let mut transformed = stones
                .iter()
                .map(|&(coord, owner)| {
                    let (q, r) = d6_coord_i32(coord, symmetry);
                    (
                        i16::try_from(q).expect("D6 q remains i16"),
                        i16::try_from(r).expect("D6 r remains i16"),
                        owner,
                    )
                })
                .collect::<Vec<_>>();
            transformed.sort_unstable();
            let mut encoded = Vec::with_capacity(self.bytes.len());
            encoded.push(current_player);
            push_wide_varint(&mut encoded, placements_made);
            encoded.push(phase_tag);
            if let Some((q, r)) = phase_first {
                let (q, r) = d6_coord_i32(HexCoord::new(q, r), symmetry);
                push_wide_varint(&mut encoded, zigzag_i16(q as i16));
                push_wide_varint(&mut encoded, zigzag_i16(r as i16));
            }
            encoded.push(terminal_tag);
            if let Some(placements) = terminal_placements {
                push_wide_varint(&mut encoded, placements);
            }
            for (q, r, owner) in transformed {
                push_wide_varint(
                    &mut encoded,
                    zigzag_i16(q).saturating_mul(2) | u32::from(owner),
                );
                push_wide_varint(&mut encoded, zigzag_i16(r));
            }
            if best
                .as_ref()
                .is_none_or(|old| encoded.as_slice() < old.as_slice())
            {
                best = Some(encoded);
            }
        }
        Self {
            bytes: best.expect("D6 contains identity").into_boxed_slice(),
        }
    }

    /// The key of the NONTERMINAL claimant FirstStone position reached after
    /// two extra defender placements on `state`, built without touching the
    /// engine. Caller contract (asserted by the defender pair plan before
    /// use): `state` is a forced defender FirstStone node with no live
    /// defender >=4 window, so the pair cannot complete six and the child is
    /// exactly (claimant to move, FirstStone, non-terminal).
    fn for_defender_pair(state: &RustHexoState, claimant: Player, extra: &[HexCoord]) -> Self {
        let mut stones = state
            .board()
            .occupied_cells()
            .iter()
            .map(|&coord| {
                (
                    coord.q,
                    coord.r,
                    player_code(state.board().get(coord).expect("occupied cell has owner")),
                )
            })
            .collect::<Vec<_>>();
        let defender = claimant.other();
        for &coord in extra {
            stones.push((coord.q, coord.r, player_code(defender)));
        }
        stones.sort_unstable();
        let mut encoded = Vec::with_capacity(stones.len().saturating_mul(3).saturating_add(12));
        encoded.push(player_code(claimant));
        push_wide_varint(
            &mut encoded,
            state
                .placements_made()
                .saturating_add(u32::try_from(extra.len()).unwrap_or(0)),
        );
        encoded.push(1); // TurnPhase::FirstStone
        encoded.push(0); // non-terminal
        for (q, r, owner) in stones {
            push_wide_varint(
                &mut encoded,
                zigzag_i16(q).saturating_mul(2) | u32::from(owner),
            );
            push_wide_varint(&mut encoded, zigzag_i16(r));
        }
        Self {
            bytes: encoded.into_boxed_slice(),
        }
    }
}

#[derive(Clone, Debug)]
struct WideDefenderPair {
    first: HexCoord,
    second: HexCoord,
    final_key: WidePositionKey,
    final_prior: WidePnPrior,
}

#[derive(Clone, Debug)]
struct WideDirectedDefenderPair {
    first: HexCoord,
    second: HexCoord,
    final_key: WidePositionKey,
    /// Only the retained raw-low -> raw-high representative pays for the
    /// fork-derived prior. The reverse direction is used solely to validate
    /// exact final-position equality.
    retained_prior: Option<WidePnPrior>,
}

#[derive(Clone, Debug)]
struct WideDefenderPairPlan {
    /// The exact K2 kernel, in the ordinary canonical defender order.
    kernel: Vec<HexCoord>,
    /// One raw-coordinate-ordered representative for each symmetric pair.
    pairs: Vec<WideDefenderPair>,
}

/// Derive a complete defender turn at a forced B=2 boundary. The reduction is
/// deliberately all-or-nothing: every K2 first move must be nonterminal, must
/// reach another exact forced boundary with B=1, and every resulting directed
/// pair must have the reverse direction with the identical final position.
/// Any unsupported shape is reported to the caller, which falls back to the
/// ordinary ordered defender expansion rather than dropping a reply.
fn forced_defender_pair_plan(
    state: &mut RustHexoState,
    claimant: Player,
) -> Option<WideDefenderPairPlan> {
    if state.current_player() == claimant || !matches!(state.phase(), TurnPhase::FirstStone) {
        return None;
    }
    let root_analysis = threats::analyze(state);
    if root_analysis.b != 2
        || root_analysis.opp_threat_count == 0
        || root_analysis.own_win_now
        || root_analysis.min_hitting_set != Some(2)
    {
        return None;
    }

    let mut kernel = forced_defender_replies(
        state,
        claimant,
        root_analysis.b,
        WidthOptions::vcf_pair_complete(),
    );
    let root_frame = canonical_frame(state);
    kernel.sort_by_key(|coord| canonical_coord_key(root_frame, *coord));
    kernel.dedup();
    if kernel.is_empty() {
        return None;
    }
    let kernel_set = kernel.iter().copied().collect::<HashSet<_>>();

    // One fork scan per plan: a defender pair perturbs the claimant window
    // structure only by two-colouring hit windows, so the plan-root fork
    // degree is a faithful child prior (validated by node-count A/B against
    // the historical per-pair exact scan).
    let shared_fork_pn = pn_from_fork_degree(attacker_fork_degree(state, claimant));

    // Apply/undo on the caller's state instead of cloning the full engine
    // (board + window store) once per kernel cell. Every exit path restores
    // the exact turn-start state.
    let mut directed = Vec::new();
    for &first in &kernel {
        let Ok((first_result, first_delta)) = state.apply_with_delta(Placement { coord: first })
        else {
            return None;
        };
        if first_result.outcome.is_some()
            || state.current_player() == claimant
            || !matches!(state.phase(), TurnPhase::SecondStone { .. })
        {
            state.undo(first_delta);
            return None;
        }

        let analysis = threats::analyze(state);
        if analysis.b != 1
            || analysis.opp_threat_count == 0
            || analysis.own_win_now
            || analysis.min_hitting_set != Some(1)
        {
            state.undo(first_delta);
            return None;
        }
        let mut seconds = forced_defender_replies(
            state,
            claimant,
            analysis.b,
            WidthOptions::vcf_pair_complete(),
        );
        // The plan-root frame is itself D6-covariant, so reusing it keeps the
        // rotation-invariance property while skipping a 12-symmetry stone
        // canonicalization per kernel cell.
        seconds.sort_by_key(|coord| canonical_coord_key(root_frame, *coord));
        seconds.dedup();
        if seconds.is_empty() {
            state.undo(first_delta);
            return None;
        }

        for second in seconds {
            if second == first || !kernel_set.contains(&second) {
                state.undo(first_delta);
                return None;
            }
            // No live defender >=4 window exists at the plan root (checked
            // above via own_win_now), so the pair cannot complete six: the
            // child is exactly (claimant, FirstStone, non-terminal) and its
            // key is constructible without touching the engine. `second` is a
            // kernel threat-window empty, hence always a legal placement.
            let final_key = WidePositionKey::for_defender_pair(state, claimant, &[second]);
            let retained_prior =
                (raw_coord_key(first) < raw_coord_key(second)).then(|| WidePnPrior {
                    pn: shared_fork_pn,
                    dn: 1,
                });
            directed.push(WideDirectedDefenderPair {
                first,
                second,
                final_key,
                retained_prior,
            });
        }
        state.undo(first_delta);
    }

    let directed_index = directed
        .iter()
        .enumerate()
        .map(|(index, pair)| {
            (
                (raw_coord_key(pair.first), raw_coord_key(pair.second)),
                index,
            )
        })
        .collect::<HashMap<_, _>>();
    if directed_index.len() != directed.len() {
        return None;
    }
    for pair in &directed {
        let reverse = directed_index
            .get(&(raw_coord_key(pair.second), raw_coord_key(pair.first)))
            .and_then(|&index| directed.get(index))?;
        if reverse.final_key != pair.final_key {
            return None;
        }
    }

    let kernel_rank = kernel
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, coord)| (coord, rank))
        .collect::<HashMap<_, _>>();
    let mut pairs = directed
        .into_iter()
        .filter(|pair| raw_coord_key(pair.first) < raw_coord_key(pair.second))
        .map(|pair| {
            Some(WideDefenderPair {
                first: pair.first,
                second: pair.second,
                final_key: pair.final_key,
                final_prior: pair.retained_prior?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    pairs.sort_by_key(|pair| {
        let first_rank = kernel_rank[&pair.first];
        let second_rank = kernel_rank[&pair.second];
        (first_rank.min(second_rank), first_rank.max(second_rank))
    });
    if pairs.is_empty() {
        return None;
    }
    Some(WideDefenderPairPlan { kernel, pairs })
}

/// Conservative retained-byte charge for one exact-key TT index entry.  PN
/// nodes and their child vectors live in the node-capped search arena and are
/// deliberately excluded from the caller's TT/cache byte ceiling.
fn wide_position_index_bytes(key: &WidePositionKey) -> usize {
    key.heap_bytes()
        .saturating_add(size_of::<(WidePositionKey, usize)>())
        .saturating_add(ALLOC_OVERHEAD)
}

fn zigzag_i16(value: i16) -> u32 {
    let value = i32::from(value);
    u32::try_from((value << 1) ^ (value >> 31)).expect("i16 zig-zag is nonnegative")
}

fn push_wide_varint(out: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuotientSoundVerdict {
    Win,
    Refutation,
}

#[cfg(test)]
#[derive(Default)]
struct HorizonHistory {
    clocks: HashSet<usize>,
    sound_wins: Vec<usize>,
    sound_refutations: Vec<usize>,
}

#[cfg(test)]
struct QuotientTelemetry {
    report: QuotientTelemetryReport,
    entry_keys: Vec<(WidePositionKey, WidePositionKey)>,
    expanded_raw: HashSet<WidePositionKey>,
    expanded_canonical: HashMap<WidePositionKey, WidePositionKey>,
    horizon: HashMap<WidePositionKey, HorizonHistory>,
}

#[cfg(test)]
impl QuotientTelemetry {
    fn enabled() -> Option<Self> {
        std::env::var_os("TSS_TURN_QUOTIENT_TELEMETRY").map(|_| Self {
            report: QuotientTelemetryReport::default(),
            entry_keys: Vec::new(),
            expanded_raw: HashSet::new(),
            expanded_canonical: HashMap::new(),
            horizon: HashMap::new(),
        })
    }

    fn canonicalize(&mut self, raw: &WidePositionKey) -> WidePositionKey {
        let started = Instant::now();
        let canonical = raw.d6_canonical();
        self.report.d6_canonicalization_calls =
            self.report.d6_canonicalization_calls.saturating_add(1);
        self.report.d6_canonicalization_nanos = self
            .report
            .d6_canonicalization_nanos
            .saturating_add(started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
        canonical
    }

    fn observe_insert(&mut self, raw: &WidePositionKey) {
        let canonical = self.canonicalize(raw);
        self.entry_keys.push((raw.clone(), canonical));
    }

    fn observe_expand(&mut self, id: usize, state: &RustHexoState) {
        let Some((raw, canonical)) = self.entry_keys.get(id).cloned() else {
            return;
        };
        if !self.expanded_raw.insert(raw.clone()) {
            return;
        }
        self.report.expanded_unique_positions =
            self.report.expanded_unique_positions.saturating_add(1);
        if self
            .expanded_canonical
            .get(&canonical)
            .is_some_and(|first_raw| first_raw != &raw)
        {
            self.report.d6_expanded_duplicates =
                self.report.d6_expanded_duplicates.saturating_add(1);
        } else {
            self.expanded_canonical.insert(canonical, raw);
        }

        if let Some(interaction) = classify_last_two_turns(state) {
            self.report.commutation_eligible_nodes =
                self.report.commutation_eligible_nodes.saturating_add(1);
            if interaction == 0 {
                self.report.commutation_independent_nodes =
                    self.report.commutation_independent_nodes.saturating_add(1);
            }
            if interaction & 1 != 0 {
                self.report.commutation_shared_window =
                    self.report.commutation_shared_window.saturating_add(1);
            }
            if interaction & 2 != 0 {
                self.report.commutation_legality_coupling =
                    self.report.commutation_legality_coupling.saturating_add(1);
            }
            if interaction & 4 != 0 {
                self.report.commutation_threat_response =
                    self.report.commutation_threat_response.saturating_add(1);
            }
        }
    }

    fn observe_stage(
        &mut self,
        entries: &[WidePnEntry],
        by_position: &HashMap<WidePositionKey, usize>,
        stage_depth: usize,
    ) {
        let verdicts = sound_verdicts(entries, by_position);
        for (id, entry) in entries.iter().enumerate() {
            if entry.depth > stage_depth {
                continue;
            }
            let Some((raw, _)) = self.entry_keys.get(id) else {
                continue;
            };
            let clock = stage_depth - entry.depth;
            self.report.horizon_queries = self.report.horizon_queries.saturating_add(1);
            let history = self.horizon.entry(raw.clone()).or_default();
            if history.clocks.contains(&clock) {
                self.report.horizon_exact_hits = self.report.horizon_exact_hits.saturating_add(1);
            } else {
                self.report.horizon_clock_misses =
                    self.report.horizon_clock_misses.saturating_add(1);
                let monotone = history.sound_wins.iter().any(|&old| old <= clock)
                    || history.sound_refutations.iter().any(|&old| old >= clock);
                if monotone {
                    self.report.horizon_monotone_hits =
                        self.report.horizon_monotone_hits.saturating_add(1);
                }
                history.clocks.insert(clock);
            }
            match verdicts.get(id).copied().flatten() {
                Some(QuotientSoundVerdict::Win) => {
                    if !history.sound_wins.contains(&clock) {
                        history.sound_wins.push(clock);
                        self.report.horizon_sound_wins =
                            self.report.horizon_sound_wins.saturating_add(1);
                    }
                }
                Some(QuotientSoundVerdict::Refutation) => {
                    if !history.sound_refutations.contains(&clock) {
                        history.sound_refutations.push(clock);
                        self.report.horizon_sound_refutations =
                            self.report.horizon_sound_refutations.saturating_add(1);
                    }
                }
                None => {}
            }
            if matches!(entry.node, WidePnNode::DepthCutoff) {
                self.report.horizon_staged_cutoffs_excluded = self
                    .report
                    .horizon_staged_cutoffs_excluded
                    .saturating_add(1);
            }
        }

        self.report.indexed_entries = by_position.len() as u64;
    }

    fn finish(
        mut self,
        entries: &[WidePnEntry],
        by_position: &HashMap<WidePositionKey, usize>,
        tt_hits: u64,
    ) -> QuotientTelemetryReport {
        self.report.retained_entries = entries.len() as u64;
        self.report.indexed_entries = by_position.len() as u64;
        self.report.tt_hits = tt_hits;
        let mut canonical = HashSet::new();
        for &id in by_position.values() {
            if let Some((_, key)) = self.entry_keys.get(id) {
                canonical.insert(key.clone());
            }
        }
        self.report.d6_index_denominator = by_position.len() as u64;
        self.report.d6_index_duplicates =
            (by_position.len().saturating_sub(canonical.len())) as u64;
        self.report.horizon_positions = self.horizon.len() as u64;
        self.report.horizon_position_clock_entries = self
            .horizon
            .values()
            .map(|history| history.clocks.len() as u64)
            .sum();
        self.report.horizon_multi_clock_positions = self
            .horizon
            .values()
            .filter(|history| history.clocks.len() > 1)
            .count() as u64;
        self.report
    }
}

#[cfg(test)]
fn child_sound_verdict(
    child: &WidePnChild,
    verdicts: &[Option<QuotientSoundVerdict>],
    by_position: &HashMap<WidePositionKey, usize>,
) -> Option<QuotientSoundVerdict> {
    match child.result {
        WidePnChildResult::ClaimantCompletion | WidePnChildResult::ClaimantTactical => {
            Some(QuotientSoundVerdict::Win)
        }
        WidePnChildResult::Refuted => Some(QuotientSoundVerdict::Refutation),
        WidePnChildResult::Pending => child
            .entry
            .or_else(|| {
                child
                    .future_key
                    .as_ref()
                    .and_then(WideFutureKey::virtual_key)
                    .and_then(|key| by_position.get(key).copied())
            })
            .and_then(|id| verdicts.get(id).copied().flatten()),
    }
}

#[cfg(test)]
fn sound_verdicts(
    entries: &[WidePnEntry],
    by_position: &HashMap<WidePositionKey, usize>,
) -> Vec<Option<QuotientSoundVerdict>> {
    let mut verdicts = vec![None; entries.len()];
    let mut ids = (0..entries.len()).collect::<Vec<_>>();
    ids.sort_unstable_by_key(|&id| Reverse(entries[id].depth));
    for id in ids {
        verdicts[id] = match &entries[id].node {
            WidePnNode::ProvenLeaf(_) | WidePnNode::ProvenFragment(_) => {
                Some(QuotientSoundVerdict::Win)
            }
            WidePnNode::Refuted => Some(QuotientSoundVerdict::Refutation),
            WidePnNode::Unexpanded | WidePnNode::DepthCutoff => None,
            WidePnNode::Branch { kind, children } => match kind {
                WidePnKind::Choice => {
                    if children.iter().any(|child| {
                        child_sound_verdict(child, &verdicts, by_position)
                            == Some(QuotientSoundVerdict::Win)
                    }) {
                        Some(QuotientSoundVerdict::Win)
                    } else if children.iter().all(|child| {
                        child_sound_verdict(child, &verdicts, by_position)
                            == Some(QuotientSoundVerdict::Refutation)
                    }) {
                        Some(QuotientSoundVerdict::Refutation)
                    } else {
                        None
                    }
                }
                WidePnKind::Universal { .. } => {
                    if children.iter().all(|child| {
                        child_sound_verdict(child, &verdicts, by_position)
                            == Some(QuotientSoundVerdict::Win)
                    }) {
                        Some(QuotientSoundVerdict::Win)
                    } else if children.iter().any(|child| {
                        child_sound_verdict(child, &verdicts, by_position)
                            == Some(QuotientSoundVerdict::Refutation)
                    }) {
                        Some(QuotientSoundVerdict::Refutation)
                    } else {
                        None
                    }
                }
            },
        };
    }
    verdicts
}

#[cfg(test)]
#[derive(Clone)]
struct TelemetryTurn {
    player: Player,
    placements: Vec<HexCoord>,
}

#[cfg(test)]
fn last_two_complete_turns(state: &RustHexoState) -> Option<(TelemetryTurn, TelemetryTurn)> {
    let history = state.placement_history();
    let last = history.last()?;
    if !matches!(last.phase, TurnPhase::SecondStone { .. }) || history.len() < 4 {
        return None;
    }
    let last_first = history.get(history.len() - 2)?;
    if !matches!(last_first.phase, TurnPhase::FirstStone) || last_first.player != last.player {
        return None;
    }
    let previous_last = history.get(history.len() - 3)?;
    if !matches!(previous_last.phase, TurnPhase::SecondStone { .. }) {
        return None;
    }
    let previous_first = history.get(history.len() - 4)?;
    if !matches!(previous_first.phase, TurnPhase::FirstStone)
        || previous_first.player != previous_last.player
        || previous_last.player == last.player
    {
        return None;
    }
    Some((
        TelemetryTurn {
            player: previous_last.player,
            placements: vec![previous_first.coord, previous_last.coord],
        },
        TelemetryTurn {
            player: last.player,
            placements: vec![last_first.coord, last.coord],
        },
    ))
}

#[cfg(test)]
fn coords_share_window(left: HexCoord, right: HexCoord) -> bool {
    let dq = i32::from(left.q) - i32::from(right.q);
    let dr = i32::from(left.r) - i32::from(right.r);
    (dq == 0 && dr.abs() <= 5) || (dr == 0 && dq.abs() <= 5) || (dq == -dr && dq.abs() <= 5)
}

#[cfg(test)]
fn classify_last_two_turns(state: &RustHexoState) -> Option<u8> {
    let (first, second) = last_two_complete_turns(state)?;
    let mut flags = 0u8;
    if first.placements.iter().any(|&left| {
        second
            .placements
            .iter()
            .any(|&right| coords_share_window(left, right))
    }) {
        flags |= 1;
    }

    let excluded = first
        .placements
        .iter()
        .chain(second.placements.iter())
        .copied()
        .collect::<HashSet<_>>();
    let mut available = state
        .board()
        .occupied_cells()
        .iter()
        .copied()
        .filter(|coord| !excluded.contains(coord))
        .collect::<Vec<_>>();
    for &coord in &second.placements {
        if !available
            .iter()
            .any(|&support| hex_distance(coord, support) <= 8)
        {
            flags |= 2;
            break;
        }
        available.push(coord);
    }

    let first_hits_second = first.placements.iter().any(|&coord| {
        state.board().windows().entries().any(|entry| {
            if !entry.key().contains(coord) {
                return false;
            }
            let later = second
                .placements
                .iter()
                .filter(|&&placed| entry.key().contains(placed))
                .count() as u8;
            entry.count(second.player).saturating_sub(later) >= 3
        })
    });
    let second_hits_first = second.placements.iter().any(|&coord| {
        state
            .board()
            .windows()
            .entries()
            .any(|entry| entry.key().contains(coord) && entry.count(first.player) >= 3)
    });
    if first_hits_second || second_hits_first {
        flags |= 4;
    }
    Some(flags)
}

/// Wide VCF search keeps a persistent proof-number frontier.  Unlike the
/// quota-based DFS experiments, expanding a sibling never discards work in an
/// earlier forcing turn. Claimant pairs are represented as one OR edge, so
/// turn-forcing is structural rather than an after-the-fact recursive filter.
struct WidePnSearch<'store> {
    claimant: Player,
    root_ply: u32,
    node_cap: u64,
    tt_bytes_cap: usize,
    semantic_horizon: u32,
    depth_cap: usize,
    /// Final solve depth; staged deepening mutates `depth_cap` below.
    max_depth_cap: usize,
    width: WidthOptions,
    /// Solve-local, read-once R-OS2 ordering configuration. Off takes the
    /// historical selector branch without computing or consulting a key.
    zone_order_mode: ZoneOrderMode,
    zone_order_band: u32,
    /// Read once when this solve-local search is created. Default-off keeps the
    /// historical eager defender-frontier admission path byte-for-byte in the
    /// decision logic.
    lazy_frontier: bool,
    /// Still-live lines refused by the semantic-horizon deadline this search
    /// (SolveStats::horizon_cuts), and the defender-to-move subset
    /// (SolveStats::kb_death_cuts).
    horizon_cuts: u64,
    kb_death_cuts: u64,
    interior_census_gate: bool,
    interior_gate_evaluations: u64,
    interior_gate_dismissals: u64,
    interior_gate_nanos: u64,
    expansions: u64,
    tt_hits: u64,
    current_bytes: usize,
    peak_bytes: usize,
    /// Expansion ceiling for one `work` invocation. The production driver
    /// leaves it open (`u64::MAX`); the historical single-expansion `step`
    /// wrapper sets it to `expansions + 1` so the focused stepper tests keep
    /// their one-expansion-per-call contract.
    soft_expansion_cap: u64,
    #[cfg(test)]
    tt_index_rejections: u64,
    #[cfg(test)]
    tt_first_rejection: Option<(u64, usize)>,
    #[cfg(test)]
    stage_refreshes: u64,
    #[cfg(test)]
    live_ge3_seed: bool,
    #[cfg(test)]
    live_ge3_seed_scans: std::cell::Cell<u64>,
    #[cfg(test)]
    live_ge3_seed_nanos: std::cell::Cell<u64>,
    #[cfg(test)]
    closure_counters: bool,
    #[cfg(test)]
    reveal_prefix_study: bool,
    #[cfg(test)]
    ordering_study: bool,
    #[cfg(test)]
    closure_stats: RefCell<ClosureDebtStats>,
    #[cfg(test)]
    closure_pair_nodes: HashMap<usize, ClosurePairNodeProfile>,
    #[cfg(test)]
    closure_pair_children: HashMap<(usize, usize), ClosurePairChildProfile>,
    #[cfg(test)]
    closure_last_pair_node: std::cell::Cell<ClosurePairNodeProfile>,
    #[cfg(test)]
    closure_last_pair_children: RefCell<Vec<ClosurePairChildProfile>>,
    #[cfg(test)]
    reveal_pair_nodes: HashMap<usize, RevealPairNodeProfile>,
    #[cfg(test)]
    reveal_last_pair_node: RefCell<RevealPairNodeProfile>,
    #[cfg(test)]
    threshold_counters: bool,
    #[cfg(test)]
    threshold_delta: Option<ThresholdDelta>,
    #[cfg(test)]
    threshold_stats: Rc<RefCell<ThresholdScaleStats>>,
    #[cfg(test)]
    threshold_expansion_clock: Rc<Cell<u64>>,
    #[cfg(test)]
    threshold_entry_visits: Vec<u64>,
    #[cfg(test)]
    threshold_last_selected: Vec<Option<usize>>,
    #[cfg(test)]
    threshold_band_stack: Vec<ThresholdBandSelection>,
    #[cfg(test)]
    quotient_telemetry: Option<QuotientTelemetry>,
    entries: Vec<WidePnEntry>,
    by_position: HashMap<WidePositionKey, usize>,
    /// Exact prospective identity for lazy defender thunks. This preserves the
    /// first eager admission's prior/depth and lets a selected attacker thunk
    /// recover that transposed state without pre-linking an arena/TT entry.
    deferred_by_position: HashMap<WidePositionKey, WideDeferredPosition>,
    fragment_store: Option<&'store ProvenFragmentStore>,
    fragment_lookups: u64,
    fragment_hits: u64,
    proven_candidate_ids: HashSet<usize>,
    proven_candidates: Vec<WideProvenCandidate>,
}

#[cfg(test)]
fn pn_init_record_wide_expansion(search: &WidePnSearch<'_>, state: &RustHexoState, id: usize) {
    let parent_engine = PN_INIT_WIDE_STACK.with(|stack| {
        let stack = stack.borrow();
        stack
            .len()
            .checked_sub(2)
            .and_then(|index| stack.get(index).copied())
    });
    let parent_serial = PN_INIT_TELEMETRY.with(|slot| {
        let slot = slot.borrow();
        let session = slot.as_ref()?;
        parent_engine.and_then(|parent| session.wide_last_event.get(&parent).copied())
    });
    let serial = pn_init_record_node(
        state,
        search.claimant,
        search.root_ply,
        search.semantic_horizon,
        id as u64,
        PnInitTelemetryMode::WidePn,
        parent_serial,
    );
    if let Some(serial) = serial {
        PN_INIT_TELEMETRY.with(|slot| {
            slot.borrow_mut()
                .as_mut()
                .expect("active wide telemetry")
                .wide_last_event
                .insert(id, serial);
        });
    }
}

#[cfg(test)]
fn pn_init_finalize_wide(search: &WidePnSearch<'_>) {
    PN_INIT_TELEMETRY.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(session) = slot.as_mut() else {
            return;
        };
        for node in &mut session.report.nodes {
            if node.mode != PnInitTelemetryMode::WidePn {
                continue;
            }
            let Some(entry) = search.entries.get(node.engine_node as usize) else {
                continue;
            };
            node.outcome = if entry.pn == 0 {
                PnInitTelemetryOutcome::Proven
            } else if entry.dn == 0 && !matches!(entry.node, WidePnNode::DepthCutoff) {
                PnInitTelemetryOutcome::Refuted
            } else {
                PnInitTelemetryOutcome::Unknown
            };
        }
    });
}

impl<'store> WidePnSearch<'store> {
    /// C1 migration mode: preserve the narrow DFS's recursive expansion order
    /// while entering through the wide engine seam.  Reusing the PN frontier
    /// here would change node counts even when it found the same proof, which
    /// is outside the owner-approved identity contract.
    #[allow(clippy::too_many_arguments)]
    fn prove_narrow_compat(
        state: &RustHexoState,
        claimant: Player,
        node_cap: u64,
        local_tt_cap: usize,
        hash_mask: u64,
        shared_tt: &mut SharedProofCache,
        semantic_horizon: u32,
        zone: ZoneSearchCaps,
        width: WidthOptions,
        depth_cap: usize,
        k_reply_consume: bool,
        #[cfg(test)] k_reply_shadow: Option<&mut Vec<KReplyShadowRecord>>,
        interior_census_gate: bool,
        group2: bool,
    ) -> AttemptResult {
        debug_assert!(
            !width.vcf_pair_complete
                || (width.consumes_quiet_turns() && width.consumes_ranked_zone())
        );

        let mut work = state.clone();
        let entry_key = PositionKey::from_state(&work);
        let root_ply = state.placements_made();
        let mut search = NarrowCompatSearch::with_shared(
            node_cap,
            local_tt_cap,
            hash_mask,
            &mut *shared_tt,
            root_ply,
            semantic_horizon,
            zone,
            width,
            depth_cap,
            k_reply_consume,
            #[cfg(test)]
            k_reply_shadow,
            interior_census_gate,
            group2,
        );
        let proof = search.prove(&mut work, claimant, root_ply, None);

        debug_assert_eq!(entry_key, PositionKey::from_state(&work));

        let cert = proof.and_then(|root| {
            let (nodes, root_node) = compact_certificate(&search.arena, root)?;
            if search.can_admit_compact(&entry_key, &nodes) {
                if let Some(cached) = CachedProof::from_compact(nodes.clone(), root_node) {
                    search.insert_shared(entry_key.clone(), claimant, cached);
                }
            }
            let mut cert = TssCertificate {
                root: RootBinding::from_state(state),
                claimant,
                root_node,
                nodes,
                semantic_horizon,
            };
            if cert.nodes.iter().any(CertNode::is_group2_extension) {
                // v1 Group-2 finalization: canonical order, strict-tree
                // unfolding, derived scalars, and both digests — then a
                // strict self-verification under the extension policy. Any
                // failure drops the certificate; the clean re-solve below
                // restores flag-off behavior.
                let finalized =
                    crate::tss_verify_group2::finder_finalize_group2(state, &cert)?;
                let claimed = status_for_claimant(state.current_player(), claimant);
                if !crate::tss_core::CertVerify::verify(
                    &crate::tss_verify::Group2Verifier,
                    state,
                    &finalized,
                    claimed,
                ) {
                    return None;
                }
                cert = finalized;
            } else {
                rebase_zone_distances(&mut cert, state)?;
            }
            Some(cert)
        });
        let stats = SolveStats {
            nodes: search.nodes,
            expansions: search.nodes,
            tt_hits: search.tt_hits,
            tt_entries: search.tt_entry_count() as u64,
            peak_tt_bytes: search.peak_tt_bytes as u64,
            tt_evictions: search.tt.replacements,
            tt_admission_rejections: search.tt.refusals,
            interior_gate_evaluations: search.interior_gate_evaluations,
            interior_gate_dismissals: search.interior_gate_dismissals,
            interior_gate_nanos: search.interior_gate_nanos,
            horizon_cuts: search.horizon_cuts,
            kb_death_cuts: search.kb_death_cuts,
            ..SolveStats::default()
        };
        #[cfg(test)]
        let tt_signature = search.tt_behavior_signature();
        #[cfg(test)]
        if let Some(telemetry) = search.quotient_telemetry.take() {
            let report = telemetry.finish(&search.tt, search.tt_hits);
            LAST_QUOTIENT_REPORT.with(|slot| *slot.borrow_mut() = Some(report));
        }
        drop(search);
        if group2 && cert.is_none() {
            // Fail-safe re-solve with the selector off: the flag must never
            // decide fewer positions than flag-off. Costs are summed.
            let rerun = Self::prove_narrow_compat(
                state,
                claimant,
                node_cap,
                local_tt_cap,
                hash_mask,
                shared_tt,
                semantic_horizon,
                zone,
                width,
                depth_cap,
                k_reply_consume,
                #[cfg(test)]
                None,
                interior_census_gate,
                false,
            );
            let mut merged = stats;
            merged.merge(rerun.stats);
            return AttemptResult {
                cert: rerun.cert,
                stats: merged,
                #[cfg(test)]
                tt_signature: rerun.tt_signature,
            };
        }
        AttemptResult {
            cert,
            stats,
            #[cfg(test)]
            tt_signature: Some(tt_signature),
        }
    }

    fn new(
        claimant: Player,
        root_ply: u32,
        node_cap: u64,
        tt_bytes_cap: usize,
        semantic_horizon: u32,
        depth_cap: usize,
    ) -> Self {
        Self::new_with_width(
            claimant,
            root_ply,
            node_cap,
            tt_bytes_cap,
            semantic_horizon,
            depth_cap,
            WidthOptions::vcf_pair_complete(),
            None,
        )
    }

    fn new_with_width(
        claimant: Player,
        root_ply: u32,
        node_cap: u64,
        tt_bytes_cap: usize,
        semantic_horizon: u32,
        depth_cap: usize,
        width: WidthOptions,
        fragment_store: Option<&'store ProvenFragmentStore>,
    ) -> Self {
        let lazy_frontier = std::env::var("TSS_LAZY_FRONTIER").ok().as_deref() == Some("1");
        let zone_order_mode = ZoneOrderMode::from_env();
        let zone_order_band = zone_order_band_from_env(zone_order_mode);
        Self {
            claimant,
            root_ply,
            node_cap,
            tt_bytes_cap,
            semantic_horizon,
            depth_cap,
            max_depth_cap: depth_cap,
            width,
            zone_order_mode,
            zone_order_band,
            lazy_frontier,
            horizon_cuts: 0,
            kb_death_cuts: 0,
            interior_census_gate: false,
            interior_gate_evaluations: 0,
            interior_gate_dismissals: 0,
            interior_gate_nanos: 0,
            expansions: 0,
            tt_hits: 0,
            current_bytes: 0,
            peak_bytes: 0,
            soft_expansion_cap: u64::MAX,
            #[cfg(test)]
            tt_index_rejections: 0,
            #[cfg(test)]
            tt_first_rejection: None,
            #[cfg(test)]
            stage_refreshes: 0,
            #[cfg(test)]
            live_ge3_seed: std::env::var("TSS_LIVE_GE3_SEED").ok().as_deref() == Some("1"),
            #[cfg(test)]
            live_ge3_seed_scans: std::cell::Cell::new(0),
            #[cfg(test)]
            live_ge3_seed_nanos: std::cell::Cell::new(0),
            #[cfg(test)]
            closure_counters: std::env::var("TSS_CLOSURE_COUNTERS").ok().as_deref() == Some("1"),
            #[cfg(test)]
            reveal_prefix_study: std::env::var("TSS_REVEAL_PREFIX_STUDY").ok().as_deref()
                == Some("1"),
            #[cfg(test)]
            ordering_study: std::env::var("TSS_ORDERING_STUDY").ok().as_deref() == Some("1"),
            #[cfg(test)]
            closure_stats: RefCell::new(ClosureDebtStats::default()),
            #[cfg(test)]
            closure_pair_nodes: HashMap::new(),
            #[cfg(test)]
            closure_pair_children: HashMap::new(),
            #[cfg(test)]
            closure_last_pair_node: std::cell::Cell::new(ClosurePairNodeProfile::default()),
            #[cfg(test)]
            closure_last_pair_children: RefCell::new(Vec::new()),
            #[cfg(test)]
            reveal_pair_nodes: HashMap::new(),
            #[cfg(test)]
            reveal_last_pair_node: RefCell::new(RevealPairNodeProfile::default()),
            #[cfg(test)]
            threshold_counters: std::env::var("TSS_THRESHOLD_COUNTERS").ok().as_deref()
                == Some("1"),
            #[cfg(test)]
            threshold_delta: ThresholdDelta::from_env(),
            #[cfg(test)]
            threshold_stats: Rc::new(RefCell::new(ThresholdScaleStats::default())),
            #[cfg(test)]
            threshold_expansion_clock: Rc::new(Cell::new(0)),
            #[cfg(test)]
            threshold_entry_visits: Vec::new(),
            #[cfg(test)]
            threshold_last_selected: Vec::new(),
            #[cfg(test)]
            threshold_band_stack: Vec::new(),
            #[cfg(test)]
            quotient_telemetry: QuotientTelemetry::enabled(),
            entries: Vec::new(),
            by_position: HashMap::new(),
            deferred_by_position: HashMap::new(),
            fragment_store,
            fragment_lookups: 0,
            fragment_hits: 0,
            proven_candidate_ids: HashSet::new(),
            proven_candidates: Vec::new(),
        }
    }

    fn remember_proven_candidate(&mut self, state: &RustHexoState, id: usize) {
        if self.fragment_store.is_none()
            || self.proven_candidates.len() >= MAX_WIDE_FRAGMENT_PROMOTIONS
            || self.proven_candidate_ids.contains(&id)
            || !matches!(self.entries[id].node, WidePnNode::Branch { .. })
        {
            return;
        }
        self.proven_candidate_ids.insert(id);
        self.proven_candidates.push(WideProvenCandidate {
            id,
            state: state.clone(),
        });
    }

    fn insert_root(&mut self, state: &RustHexoState) -> usize {
        let prior = self.position_prior(state);
        self.insert_position(WidePositionKey::from_state(state), 0, prior)
    }

    fn insert_position(&mut self, key: WidePositionKey, depth: usize, prior: WidePnPrior) -> usize {
        #[cfg(test)]
        let _timer = WideGenTimer::start(&WIDE_INSERT_NANOS);
        if let Some(&id) = self.by_position.get(&key) {
            self.tt_hits = self.tt_hits.saturating_add(1);
            return id;
        }
        let deferred = self
            .lazy_frontier
            .then(|| self.deferred_by_position.remove(&key))
            .flatten();
        let (depth, prior) = deferred
            .map(|deferred| (deferred.depth, deferred.prior))
            .unwrap_or((depth, prior));

        // The retained PN frontier is the search arena, not the transposition
        // index.  A full (or disabled) TT must only stop indexing new keys;
        // refusing the arena entry would strand the selected Pending edge and
        // make a memory-profile choice alter frontier progress.
        let id = self.entries.len();
        #[cfg(test)]
        if let Some(telemetry) = self.quotient_telemetry.as_mut() {
            telemetry.observe_insert(&key);
        }
        self.entries.push(WidePnEntry {
            pn: prior.pn,
            dn: prior.dn,
            prior,
            node: WidePnNode::Unexpanded,
            depth,
            universal_obligation: None,
        });

        let added = wide_position_index_bytes(&key);
        if self.tt_bytes_cap > 0 && self.current_bytes.saturating_add(added) <= self.tt_bytes_cap {
            self.by_position.insert(key, id);
            self.current_bytes = self.current_bytes.saturating_add(added);
            self.peak_bytes = self.peak_bytes.max(self.current_bytes);
        } else {
            #[cfg(test)]
            if self.tt_bytes_cap > 0 {
                let first_rejection = self.tt_first_rejection.is_none();
                self.tt_index_rejections = self.tt_index_rejections.saturating_add(1);
                self.tt_first_rejection
                    .get_or_insert((self.expansions, self.entries.len()));
                if first_rejection && self.threshold_counters {
                    self.threshold_stats.borrow_mut().first_admission_refusal = Some((
                        self.expansions,
                        self.entries.len() as u64,
                        self.current_bytes as u64,
                    ));
                }
            }
        }
        id
    }

    fn defer_position(&mut self, key: &WidePositionKey, depth: usize, prior: WidePnPrior) {
        if self.by_position.contains_key(key) {
            return;
        }
        self.deferred_by_position
            .entry(key.clone())
            .or_insert(WideDeferredPosition { depth, prior });
    }

    fn position_prior(&self, state: &RustHexoState) -> WidePnPrior {
        #[cfg(test)]
        let _gen_timer = WideGenTimer::start(&WIDE_GEN_PRIOR_NANOS);
        if state.current_player() == self.claimant {
            let pn = pn_from_fork_degree(attacker_fork_degree(state, self.claimant));
            #[cfg(test)]
            let pn = if self.live_ge3_seed {
                self.live_ge3_seed_prior(state)
            } else {
                pn
            };
            WidePnPrior { pn, dn: 1 }
        } else {
            let analysis = threats::analyze(state);
            #[cfg(test)]
            let pn = if self.live_ge3_seed {
                self.live_ge3_seed_prior(state)
            } else {
                1
            };
            #[cfg(not(test))]
            let pn = 1;
            WidePnPrior {
                pn,
                dn: dn_from_tau(analysis.min_hitting_set),
            }
        }
    }

    #[cfg(test)]
    fn live_ge3_seed_prior(&self, state: &RustHexoState) -> u32 {
        let started = Instant::now();
        let live_ge3 = state
            .board()
            .windows()
            .entries()
            .filter(|entry| {
                entry.count(self.claimant) >= 3 && entry.count(self.claimant.other()) == 0
            })
            .count();
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.live_ge3_seed_scans
            .set(self.live_ge3_seed_scans.get().saturating_add(1));
        self.live_ge3_seed_nanos
            .set(self.live_ge3_seed_nanos.get().saturating_add(nanos));
        pn_from_fork_degree(live_ge3)
    }

    /// The immutable first-placement width class is a root bootstrap only.
    /// Persisting it below depth zero regresses otherwise closed positions by
    /// overriding accumulated proof-number evidence.
    fn prefer_width_tier_at_depth(&self, depth: usize) -> bool {
        depth == 0
    }

    fn completed_turn_prior(&self, state: &RustHexoState) -> WidePnPrior {
        debug_assert_ne!(state.current_player(), self.claimant);
        let analysis = threats::analyze(state);
        let pn = pn_from_fork_degree(analysis.opp_threat_count);
        #[cfg(test)]
        let pn = if self.live_ge3_seed {
            self.live_ge3_seed_prior(state)
        } else {
            pn
        };
        WidePnPrior {
            pn,
            dn: dn_from_tau(analysis.min_hitting_set),
        }
    }

    fn run(&mut self, root_state: &RustHexoState, root: usize) {
        let final_depth = self.depth_cap;
        let mut stage_depth = 0usize;

        // The selected PN path discovers the next useful horizon. Every stage
        // shares the caller's one global node cap; there are no scouting quotas.
        loop {
            self.depth_cap = stage_depth;
            self.reopen_depth_cutoffs(stage_depth);
            let is_final = stage_depth == final_depth;
            let selected_cutoff = self.run_until(root_state, root, self.node_cap, !is_final);
            // Transposed parents outside the active recursion also need to see
            // the selected cutoff (or proof) before the stage decision.
            self.refresh_all_bottom_up();
            #[cfg(test)]
            if let Some(telemetry) = self.quotient_telemetry.as_mut() {
                telemetry.observe_stage(&self.entries, &self.by_position, stage_depth);
            }
            #[cfg(test)]
            if self.trace_enabled() {
                let root_entry = &self.entries[root];
                eprintln!(
                    "WIDTH_PN_STAGE stage_depth={stage_depth} expansions={} selected_cutoff={selected_cutoff:?} root_pn={} root_dn={}",
                    self.expansions, root_entry.pn, root_entry.dn
                );
            }

            if self.entries[root].pn == 0 || self.expansions >= self.node_cap || is_final {
                break;
            }
            let Some(encountered_depth) = selected_cutoff else {
                break;
            };
            let Some(next_depth) =
                next_wide_stage_depth(stage_depth, encountered_depth, final_depth)
            else {
                break;
            };
            stage_depth = next_depth;
        }

        self.depth_cap = final_depth;
    }

    /// Test-only continuation driver. `stage_depth` and whether its cutoffs
    /// have already been reopened are session state, not call-local state.
    #[cfg(test)]
    fn run_resumable(
        &mut self,
        root_state: &RustHexoState,
        root: usize,
        stage_depth: &mut usize,
        stage_initialized: &mut bool,
    ) {
        let final_depth = self.max_depth_cap;
        loop {
            self.depth_cap = *stage_depth;
            if !*stage_initialized {
                self.reopen_depth_cutoffs(*stage_depth);
                *stage_initialized = true;
            }
            let is_final = *stage_depth == final_depth;
            let selected_cutoff = self.run_until(root_state, root, self.node_cap, !is_final);
            self.refresh_all_bottom_up();
            if let Some(telemetry) = self.quotient_telemetry.as_mut() {
                telemetry.observe_stage(&self.entries, &self.by_position, *stage_depth);
            }
            if self.trace_enabled() {
                let root_entry = &self.entries[root];
                eprintln!(
                    "WIDTH_PN_RESUME_STAGE stage_depth={} expansions={} selected_cutoff={selected_cutoff:?} root_pn={} root_dn={}",
                    *stage_depth, self.expansions, root_entry.pn, root_entry.dn
                );
            }

            if self.entries[root].pn == 0 || self.expansions >= self.node_cap || is_final {
                break;
            }
            let Some(encountered_depth) = selected_cutoff else {
                break;
            };
            let Some(next_depth) =
                next_wide_stage_depth(*stage_depth, encountered_depth, final_depth)
            else {
                break;
            };
            *stage_depth = next_depth;
            *stage_initialized = false;
        }
        self.depth_cap = final_depth;
    }

    fn run_until(
        &mut self,
        root_state: &RustHexoState,
        root: usize,
        expansion_cap: u64,
        deepen_after_selected_cutoff: bool,
    ) -> Option<usize> {
        let mut work = root_state.clone();
        while self.expansions < self.node_cap && self.expansions < expansion_cap {
            self.recompute(root);
            let Some(entry) = self.entries.get(root) else {
                break;
            };
            if entry.pn == 0 || entry.dn == 0 {
                break;
            }
            match self.work(&mut work, root, false, u32::MAX, u32::MAX) {
                WidePnStepOutcome::Progress => {}
                WidePnStepOutcome::DepthCutoff { depth, .. } if deepen_after_selected_cutoff => {
                    return Some(depth);
                }
                WidePnStepOutcome::DepthCutoff {
                    made_progress: true,
                    ..
                } => {}
                WidePnStepOutcome::DepthCutoff {
                    made_progress: false,
                    ..
                } => {
                    #[cfg(test)]
                    self.trace_selected_path(root_state, root, "cutoff_no_progress");
                    break;
                }
                WidePnStepOutcome::Stalled => {
                    #[cfg(test)]
                    self.trace_selected_path(root_state, root, "stalled");
                    break;
                }
            }
        }
        None
    }

    fn reopen_depth_cutoffs(&mut self, depth_cap: usize) {
        let reopened = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(id, entry)| {
                (entry.depth <= depth_cap && matches!(entry.node, WidePnNode::DepthCutoff))
                    .then_some(id)
            })
            .collect::<Vec<_>>();
        for &id in &reopened {
            self.entries[id].node = WidePnNode::Unexpanded;
        }
        if reopened.is_empty() {
            return;
        }

        // Edges always add placements, so a single deepest-first pass
        // propagates the reopened frontier through every (possibly
        // transposed) parent without retaining reverse-parent vectors in each
        // entry.  Refreshing only the cutoff entries leaves an ancestor's
        // cached dn=0 in place and can make the next depth stage stop before
        // doing any work.
        self.refresh_all_bottom_up();
    }

    fn refresh_all_bottom_up(&mut self) {
        #[cfg(test)]
        let _timer = WideGenTimer::start(&WIDE_REFRESH_NANOS);
        #[cfg(test)]
        {
            self.stage_refreshes = self.stage_refreshes.saturating_add(1);
        }
        let mut ids = (0..self.entries.len()).collect::<Vec<_>>();
        ids.sort_unstable_by_key(|&id| std::cmp::Reverse(self.entries[id].depth));
        for id in ids {
            self.recompute(id);
        }
    }

    fn step(&mut self, state: &mut RustHexoState, id: usize) -> WidePnStepOutcome {
        // Historical single-expansion stepper, preserved for the focused
        // stepper tests. The production driver calls `work` directly with
        // open thresholds and no soft cap.
        self.soft_expansion_cap = self.expansions.saturating_add(1);
        let outcome = self.work(state, id, false, u32::MAX, u32::MAX);
        self.soft_expansion_cap = u64::MAX;
        outcome
    }

    #[cfg(test)]
    fn start_threshold_residency(&mut self, id: usize) -> ThresholdResidencyGuard {
        if !self.threshold_counters {
            return ThresholdResidencyGuard::disabled(
                Rc::clone(&self.threshold_stats),
                Rc::clone(&self.threshold_expansion_clock),
            );
        }
        if self.threshold_entry_visits.len() <= id {
            self.threshold_entry_visits.resize(id + 1, 0);
        }
        let prior_visits = self.threshold_entry_visits[id];
        self.threshold_entry_visits[id] = prior_visits.saturating_add(1);
        {
            let mut stats = self.threshold_stats.borrow_mut();
            stats.recursive_node_visits = stats.recursive_node_visits.saturating_add(1);
            if prior_visits != 0 {
                stats.expanded_node_revisits = stats.expanded_node_revisits.saturating_add(1);
            }
        }
        ThresholdResidencyGuard::enabled(
            Rc::clone(&self.threshold_stats),
            Rc::clone(&self.threshold_expansion_clock),
        )
    }

    #[cfg(test)]
    fn record_threshold_selection(&mut self, parent: usize, child: usize) {
        if !self.threshold_counters {
            return;
        }
        if self.threshold_last_selected.len() <= parent {
            self.threshold_last_selected.resize(parent + 1, None);
        }
        if let Some(previous) = self.threshold_last_selected[parent] {
            let mut stats = self.threshold_stats.borrow_mut();
            stats.same_parent_reselections = stats.same_parent_reselections.saturating_add(1);
            if previous != child {
                stats.sibling_switches = stats.sibling_switches.saturating_add(1);
            }
        }
        self.threshold_last_selected[parent] = Some(child);
    }

    #[cfg(test)]
    fn threshold_band_selection(
        &self,
        kind: WidePnKind,
        children: &[WidePnChild],
        selected: usize,
    ) -> ThresholdBandSelection {
        let selected_score = match kind {
            WidePnKind::Choice => self.child_numbers(&children[selected]).0,
            WidePnKind::Universal { .. } => self.child_numbers(&children[selected]).1,
        };
        let second_best = children
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != selected)
            .map(|(_, child)| match kind {
                WidePnKind::Choice => self.child_numbers(child).0,
                WidePnKind::Universal { .. } => self.child_numbers(child).1,
            })
            .min();
        let gap_bin = match second_best {
            None => 0,
            Some(second) if selected_score < second => match second - selected_score {
                33.. => 1,
                17..=32 => 2,
                9..=16 => 3,
                5..=8 => 4,
                3..=4 => 5,
                2 => 6,
                1 => 7,
                _ => unreachable!(),
            },
            Some(second) if selected_score == second => 8,
            Some(second) => match selected_score - second {
                1 => 9,
                2 => 10,
                3..=4 => 11,
                _ => 12,
            },
        };
        ThresholdBandSelection { kind, gap_bin }
    }

    #[cfg(test)]
    fn selected_threshold_delta(
        &self,
        kind: WidePnKind,
        children: &[WidePnChild],
        selected: usize,
    ) -> u32 {
        match self.threshold_delta {
            None | Some(ThresholdDelta::One) => 1,
            Some(ThresholdDelta::Two) => 2,
            Some(ThresholdDelta::Four) => 4,
            Some(ThresholdDelta::MeanSiblingPrior) => {
                let mut sum = 0u64;
                let mut count = 0u64;
                for (index, child) in children.iter().enumerate() {
                    if index == selected {
                        continue;
                    }
                    let prior = match kind {
                        WidePnKind::Choice => child.prior.pn,
                        WidePnKind::Universal { .. } => child.prior.dn,
                    };
                    sum = sum.saturating_add(u64::from(prior));
                    count = count.saturating_add(1);
                }
                if count == 0 {
                    1
                } else {
                    u32::try_from(sum / count).unwrap_or(PN_INFINITY).max(1)
                }
            }
        }
    }

    #[cfg(test)]
    fn threshold_increment(&self, value: u32, delta: u32) -> u32 {
        let incremented = value.saturating_add(delta);
        if self.threshold_delta.is_some() {
            let result = incremented.min(PN_INFINITY);
            if self.threshold_counters && result == PN_INFINITY {
                let mut stats = self.threshold_stats.borrow_mut();
                stats.sentinel_increment_hits = stats.sentinel_increment_hits.saturating_add(1);
                if incremented > PN_INFINITY {
                    stats.sentinel_increment_clamps =
                        stats.sentinel_increment_clamps.saturating_add(1);
                }
            }
            result
        } else {
            incremented
        }
    }

    /// Threshold-bounded proof-number descent (df-pn scheduling). The node
    /// keeps driving its selected child while its own numbers stay below the
    /// caller-supplied thresholds, so consecutive expansions land at the
    /// frontier without re-descending from the root. Thresholds bound VISIT
    /// ORDER only: pn/dn recurrences, expansion, refutation marking, and
    /// certificate materialization are untouched, so proofs are unchanged.
    ///
    /// Child thresholds follow the standard df-pn recurrence (min against the
    /// second-best sibling plus one; budget subtraction on the conjunctive
    /// side), floored at the child's current number plus one so a
    /// policy-selected child (urgency, width tier, sequential probe,
    /// commitment) can always make local progress before control unwinds.
    fn work(
        &mut self,
        state: &mut RustHexoState,
        id: usize,
        inherited_commitment: bool,
        pn_threshold: u32,
        dn_threshold: u32,
    ) -> WidePnStepOutcome {
        #[cfg(test)]
        let _pn_init_work_guard = PnInitWideWorkGuard::enter(id);
        #[cfg(test)]
        let mut threshold_residency = self.start_threshold_residency(id);
        #[cfg(test)]
        if self.threshold_counters && self.threshold_delta.is_some() {
            let mut stats = self.threshold_stats.borrow_mut();
            for (index, threshold) in [pn_threshold, dn_threshold].into_iter().enumerate() {
                if threshold >= PN_INFINITY {
                    stats.sentinel_inherited_threshold_hits[index] =
                        stats.sentinel_inherited_threshold_hits[index].saturating_add(1);
                }
                if threshold > PN_INFINITY {
                    stats.sentinel_inherited_threshold_clamps[index] =
                        stats.sentinel_inherited_threshold_clamps[index].saturating_add(1);
                }
            }
        }
        #[cfg(test)]
        let pn_threshold = if self.threshold_delta.is_some() {
            pn_threshold.min(PN_INFINITY)
        } else {
            pn_threshold
        };
        #[cfg(test)]
        let dn_threshold = if self.threshold_delta.is_some() {
            dn_threshold.min(PN_INFINITY)
        } else {
            dn_threshold
        };
        let mut any_progress = false;
        let mut yielded_universal_children = Vec::new();
        loop {
            if matches!(self.entries[id].node, WidePnNode::DepthCutoff) {
                return WidePnStepOutcome::DepthCutoff {
                    depth: self.entries[id].depth,
                    made_progress: any_progress,
                };
            }
            if matches!(self.entries[id].node, WidePnNode::Unexpanded) {
                #[cfg(test)]
                threshold_residency.pause();
                let expansion_outcome = self.expand(state, id);
                #[cfg(test)]
                threshold_residency.resume();
                match expansion_outcome {
                    WidePnStepOutcome::Progress => {
                        any_progress = true;
                        if !matches!(self.entries[id].node, WidePnNode::Branch { .. }) {
                            return WidePnStepOutcome::Progress;
                        }
                    }
                    other => return other,
                }
            }
            self.recompute(id);
            if self.entries[id].pn == 0 || self.entries[id].dn == 0 {
                if self.entries[id].pn == 0 {
                    self.remember_proven_candidate(state, id);
                }
                return if any_progress {
                    WidePnStepOutcome::Progress
                } else {
                    WidePnStepOutcome::Stalled
                };
            }
            if self.entries[id].pn >= pn_threshold || self.entries[id].dn >= dn_threshold {
                // Thresholds crossed: the parent re-decides. Any expansion or
                // refutation made here already counts as progress.
                #[cfg(test)]
                threshold_residency.record_threshold_cross();
                return WidePnStepOutcome::Progress;
            }
            if self.expansions >= self.node_cap || self.expansions >= self.soft_expansion_cap {
                return if any_progress {
                    WidePnStepOutcome::Progress
                } else {
                    WidePnStepOutcome::Stalled
                };
            }

            let finish_partial_turn = matches!(state.phase(), TurnPhase::SecondStone { .. });
            let urgent_pair = matches!(state.phase(), TurnPhase::FirstStone)
                && matches!(
                    &self.entries[id].node,
                    WidePnNode::Branch {
                        kind: WidePnKind::Choice,
                        children,
                    } if wide_choice_has_urgent_block(children)
                );
            // Sequential probing is a root bootstrap for the two corpus shapes
            // that enter mid-turn or under an urgent block.  Applying it at
            // every descendant discards the proof-number evidence and
            // degenerates into depth-first search inside each forcing branch.
            let sequential_root_probe =
                self.entries[id].depth == 0 && (finish_partial_turn || urgent_pair);
            let prefer_width_tier = self.prefer_width_tier_at_depth(self.entries[id].depth);
            let commitment_domain = inherited_commitment
                || match &self.entries[id].node {
                    WidePnNode::Branch {
                        kind: WidePnKind::Universal { .. },
                        children,
                    } => self.universal_commitment_active(id, children),
                    _ => false,
                };

            let parent_before = (self.entries[id].pn, self.entries[id].dn);
            let selected = self.select_step_child_index_with_commitment(
                id,
                sequential_root_probe,
                prefer_width_tier,
                &yielded_universal_children,
                commitment_domain,
            );
            let Some(child_index) = selected else {
                return if any_progress {
                    WidePnStepOutcome::Progress
                } else {
                    WidePnStepOutcome::Stalled
                };
            };
            #[cfg(test)]
            self.record_closure_pair_selected(id, child_index);
            #[cfg(test)]
            self.record_threshold_selection(id, child_index);
            #[cfg(test)]
            let mut threshold_band_selection = None;
            let (kind, child, child_pn_threshold, child_dn_threshold, _root_children_unlinked) = {
                let WidePnNode::Branch { kind, children } = &self.entries[id].node else {
                    return WidePnStepOutcome::Stalled;
                };
                let (child_pn, child_dn) = self.child_numbers(&children[child_index]);
                #[cfg(test)]
                let threshold_delta = self.selected_threshold_delta(*kind, children, child_index);
                #[cfg(test)]
                if self.threshold_counters {
                    threshold_band_selection =
                        Some(self.threshold_band_selection(*kind, children, child_index));
                }
                let (child_pn_threshold, child_dn_threshold) = match kind {
                    WidePnKind::Choice => {
                        let mut second_pn = u32::MAX;
                        for (rank, other) in children.iter().enumerate() {
                            if rank != child_index {
                                second_pn = second_pn.min(self.child_numbers(other).0);
                            }
                        }
                        #[cfg(test)]
                        let second_pn_limit = self.threshold_increment(second_pn, threshold_delta);
                        #[cfg(not(test))]
                        let second_pn_limit = second_pn.saturating_add(1);
                        #[cfg(test)]
                        let child_pn_floor = self.threshold_increment(child_pn, 1);
                        #[cfg(not(test))]
                        let child_pn_floor = child_pn.saturating_add(1);
                        #[cfg(test)]
                        let child_dn_floor = self.threshold_increment(child_dn, 1);
                        #[cfg(not(test))]
                        let child_dn_floor = child_dn.saturating_add(1);
                        let pn_t = pn_threshold.min(second_pn_limit).max(child_pn_floor);
                        let dn_t = dn_threshold
                            .saturating_sub(self.entries[id].dn.saturating_sub(child_dn))
                            .max(child_dn_floor);
                        (pn_t, dn_t)
                    }
                    WidePnKind::Universal { .. } => {
                        let committed = self.entries[id].universal_obligation == Some(child_index);
                        #[cfg(test)]
                        let child_dn_floor = self.threshold_increment(child_dn, 1);
                        #[cfg(not(test))]
                        let child_dn_floor = child_dn.saturating_add(1);
                        let dn_t = if committed {
                            // Commitment domains drive the obligation to a
                            // verdict; sibling DN must not unseat it.
                            dn_threshold.max(child_dn_floor)
                        } else {
                            let mut second_dn = u32::MAX;
                            for (rank, other) in children.iter().enumerate() {
                                if rank != child_index {
                                    second_dn = second_dn.min(self.child_numbers(other).1);
                                }
                            }
                            #[cfg(test)]
                            let second_dn_limit =
                                self.threshold_increment(second_dn, threshold_delta);
                            #[cfg(not(test))]
                            let second_dn_limit = second_dn.saturating_add(1);
                            dn_threshold.min(second_dn_limit).max(child_dn_floor)
                        };
                        #[cfg(test)]
                        let child_pn_floor = self.threshold_increment(child_pn, 1);
                        #[cfg(not(test))]
                        let child_pn_floor = child_pn.saturating_add(1);
                        let pn_t = pn_threshold
                            .saturating_sub(self.entries[id].pn.saturating_sub(child_pn))
                            .max(child_pn_floor);
                        (pn_t, dn_t)
                    }
                };
                #[cfg(test)]
                if self.threshold_delta.is_some() {
                    debug_assert!(child_pn_threshold > child_pn);
                    debug_assert!(child_dn_threshold > child_dn);
                    debug_assert!(child_pn_threshold <= PN_INFINITY);
                    debug_assert!(child_dn_threshold <= PN_INFINITY);
                }
                (
                    *kind,
                    children[child_index].clone(),
                    child_pn_threshold,
                    child_dn_threshold,
                    children.iter().all(|child| child.entry.is_none()),
                )
            };
            #[cfg(test)]
            if self.entries[id].depth == 0 && _root_children_unlinked && self.trace_enabled() {
                eprintln!(
                    "WIDTH_PN_ROOT_SELECT sequential={sequential_root_probe} prefer_tier={prefer_width_tier} rank={child_index} mv={:?} first_tier={} prior_pn={} urgent={}",
                    child.mv,
                    child.first_width_tier,
                    child.prior.pn,
                    child.urgent_block,
                );
            }
            if child.result != WidePnChildResult::Pending {
                self.recompute(id);
                return if any_progress {
                    WidePnStepOutcome::Progress
                } else {
                    WidePnStepOutcome::Stalled
                };
            }

            let outcome = match child.mv {
                WidePnMove::One(coord) => {
                    // Historical attacker edges count first linking as local
                    // progress. A key-bearing defender thunk refines an eager
                    // edge whose arena link already existed, so admission must
                    // not add a progress event that eager never reported.
                    let linked = child.entry.is_none() && matches!(kind, WidePnKind::Choice);
                    #[cfg(test)]
                    let state_started = threshold_residency.state_started();
                    let applied = state.apply_with_delta(Placement { coord });
                    #[cfg(test)]
                    threshold_residency.record_state_elapsed(state_started);
                    let Ok((_result, delta)) = applied else {
                        self.set_child_refuted(id, child_index);
                        self.refresh(id);
                        any_progress = true;
                        continue;
                    };
                    let child_id = child.entry.unwrap_or_else(|| {
                        let depth =
                            usize::try_from(state.placements_made().saturating_sub(self.root_ply))
                                .unwrap_or(usize::MAX);
                        let key = child
                            .future_key
                            .as_ref()
                            .map(|future| future.key().clone())
                            .unwrap_or_else(|| WidePositionKey::from_state(state));
                        debug_assert_eq!(key, WidePositionKey::from_state(state));
                        #[cfg(test)]
                        if std::env::var_os("TSS_LAZY_FRONTIER_VALIDATE_KEYS").is_some() {
                            assert_eq!(key, WidePositionKey::from_state(state));
                        }
                        self.insert_position(key, depth, child.prior)
                    });
                    self.set_child_entry(id, child_index, child_id);
                    #[cfg(test)]
                    threshold_residency.pause();
                    #[cfg(test)]
                    if let Some(selection) = threshold_band_selection {
                        self.threshold_band_stack.push(selection);
                    }
                    let outcome = self.work(
                        state,
                        child_id,
                        commitment_domain,
                        child_pn_threshold,
                        child_dn_threshold,
                    );
                    #[cfg(test)]
                    if let Some(selection) = threshold_band_selection {
                        debug_assert_eq!(self.threshold_band_stack.pop(), Some(selection));
                    }
                    #[cfg(test)]
                    threshold_residency.resume();
                    #[cfg(test)]
                    let state_started = threshold_residency.state_started();
                    state.undo(delta);
                    #[cfg(test)]
                    threshold_residency.record_state_elapsed(state_started);
                    match outcome {
                        WidePnStepOutcome::DepthCutoff {
                            depth,
                            made_progress,
                        } => WidePnStepOutcome::DepthCutoff {
                            depth,
                            made_progress: made_progress || linked,
                        },
                        WidePnStepOutcome::Progress => WidePnStepOutcome::Progress,
                        WidePnStepOutcome::Stalled if linked => WidePnStepOutcome::Progress,
                        WidePnStepOutcome::Stalled => WidePnStepOutcome::Stalled,
                    }
                }
                WidePnMove::Pair(first, second) | WidePnMove::DefenderPair(first, second) => {
                    let linked = child.entry.is_none() && matches!(kind, WidePnKind::Choice);
                    #[cfg(test)]
                    let state_started = threshold_residency.state_started();
                    let first_applied = state.apply_with_delta(Placement { coord: first });
                    #[cfg(test)]
                    threshold_residency.record_state_elapsed(state_started);
                    let Ok((_first_result, first_delta)) = first_applied else {
                        self.set_child_refuted(id, child_index);
                        self.refresh(id);
                        any_progress = true;
                        continue;
                    };
                    #[cfg(test)]
                    let state_started = threshold_residency.state_started();
                    let second_applied = state.apply_with_delta(Placement { coord: second });
                    #[cfg(test)]
                    threshold_residency.record_state_elapsed(state_started);
                    let Ok((_second_result, second_delta)) = second_applied else {
                        #[cfg(test)]
                        let state_started = threshold_residency.state_started();
                        state.undo(first_delta);
                        #[cfg(test)]
                        threshold_residency.record_state_elapsed(state_started);
                        self.set_child_refuted(id, child_index);
                        self.refresh(id);
                        any_progress = true;
                        continue;
                    };
                    let child_id = child.entry.unwrap_or_else(|| {
                        let depth =
                            usize::try_from(state.placements_made().saturating_sub(self.root_ply))
                                .unwrap_or(usize::MAX);
                        let key = child
                            .future_key
                            .as_ref()
                            .map(|future| future.key().clone())
                            .unwrap_or_else(|| WidePositionKey::from_state(state));
                        debug_assert_eq!(key, WidePositionKey::from_state(state));
                        #[cfg(test)]
                        if std::env::var_os("TSS_LAZY_FRONTIER_VALIDATE_KEYS").is_some() {
                            assert_eq!(key, WidePositionKey::from_state(state));
                        }
                        self.insert_position(key, depth, child.prior)
                    });
                    self.set_child_entry(id, child_index, child_id);
                    #[cfg(test)]
                    if linked {
                        self.record_closure_pair_linked(id, child_index);
                    }
                    #[cfg(test)]
                    let closure_was_unexpanded = matches!(
                        self.entries.get(child_id).map(|entry| &entry.node),
                        Some(WidePnNode::Unexpanded)
                    );
                    #[cfg(test)]
                    threshold_residency.pause();
                    #[cfg(test)]
                    if let Some(selection) = threshold_band_selection {
                        self.threshold_band_stack.push(selection);
                    }
                    let outcome = self.work(
                        state,
                        child_id,
                        commitment_domain,
                        child_pn_threshold,
                        child_dn_threshold,
                    );
                    #[cfg(test)]
                    if let Some(selection) = threshold_band_selection {
                        debug_assert_eq!(self.threshold_band_stack.pop(), Some(selection));
                    }
                    #[cfg(test)]
                    threshold_residency.resume();
                    #[cfg(test)]
                    if closure_was_unexpanded
                        && !matches!(
                            self.entries.get(child_id).map(|entry| &entry.node),
                            Some(WidePnNode::Unexpanded)
                        )
                    {
                        self.record_closure_pair_expanded(id, child_index);
                    }
                    #[cfg(test)]
                    let state_started = threshold_residency.state_started();
                    state.undo(second_delta);
                    state.undo(first_delta);
                    #[cfg(test)]
                    threshold_residency.record_state_elapsed(state_started);
                    match outcome {
                        WidePnStepOutcome::DepthCutoff {
                            depth,
                            made_progress,
                        } => WidePnStepOutcome::DepthCutoff {
                            depth,
                            made_progress: made_progress || linked,
                        },
                        WidePnStepOutcome::Progress => WidePnStepOutcome::Progress,
                        WidePnStepOutcome::Stalled if linked => WidePnStepOutcome::Progress,
                        WidePnStepOutcome::Stalled => WidePnStepOutcome::Stalled,
                    }
                }
            };
            self.refresh(id);
            let parent_changed = parent_before != (self.entries[id].pn, self.entries[id].dn);
            let outcome = match outcome {
                WidePnStepOutcome::DepthCutoff {
                    depth,
                    made_progress,
                } => WidePnStepOutcome::DepthCutoff {
                    depth,
                    made_progress: made_progress || parent_changed,
                },
                WidePnStepOutcome::Progress => WidePnStepOutcome::Progress,
                WidePnStepOutcome::Stalled if parent_changed => WidePnStepOutcome::Progress,
                WidePnStepOutcome::Stalled => WidePnStepOutcome::Stalled,
            };
            match outcome {
                WidePnStepOutcome::DepthCutoff {
                    depth,
                    made_progress,
                } => {
                    // Depth cutoffs bubble to the stage driver unchanged so
                    // staged deepening keeps its advance-on-selected-cutoff
                    // semantics.
                    return WidePnStepOutcome::DepthCutoff {
                        depth,
                        made_progress: made_progress || any_progress,
                    };
                }
                WidePnStepOutcome::Progress => {
                    any_progress = true;
                }
                WidePnStepOutcome::Stalled => {
                    if matches!(kind, WidePnKind::Universal { .. })
                        && self.entries[id].universal_obligation == Some(child_index)
                        && self.expansions < self.node_cap
                    {
                        yielded_universal_children.push(child_index);
                        continue;
                    }
                    return WidePnStepOutcome::Stalled;
                }
            }
        }
    }

    fn set_child_entry(&mut self, parent: usize, child: usize, entry: usize) {
        if let WidePnNode::Branch { children, .. } = &mut self.entries[parent].node {
            children[child].entry = Some(entry);
            children[child].future_key = None;
        }
    }

    #[cfg(test)]
    fn record_closure_pair_selected(&mut self, parent: usize, child: usize) {
        let Some(profile) = self.closure_pair_children.get_mut(&(parent, child)) else {
            return;
        };
        if !profile.selected {
            profile.selected = true;
            let stats = self.closure_stats.get_mut();
            stats.pairs_selected = stats.pairs_selected.saturating_add(1);
        }
    }

    #[cfg(test)]
    fn record_closure_pair_linked(&mut self, parent: usize, child: usize) {
        let Some(profile) = self.closure_pair_children.get_mut(&(parent, child)) else {
            return;
        };
        if !profile.linked {
            profile.linked = true;
            let stats = self.closure_stats.get_mut();
            stats.pairs_linked = stats.pairs_linked.saturating_add(1);
        }
    }

    #[cfg(test)]
    fn record_closure_pair_expanded(&mut self, parent: usize, child: usize) {
        let Some(profile) = self.closure_pair_children.get_mut(&(parent, child)) else {
            return;
        };
        if !profile.expanded {
            profile.expanded = true;
            let stats = self.closure_stats.get_mut();
            stats.pairs_expanded = stats.pairs_expanded.saturating_add(1);
        }
    }

    fn set_child_refuted(&mut self, parent: usize, child: usize) {
        if let WidePnNode::Branch { children, .. } = &mut self.entries[parent].node {
            children[child].result = WidePnChildResult::Refuted;
            children[child].future_key = None;
        }
    }

    fn child_numbers(&self, child: &WidePnChild) -> (u32, u32) {
        match child.result {
            WidePnChildResult::ClaimantCompletion | WidePnChildResult::ClaimantTactical => {
                (0, PN_INFINITY)
            }
            WidePnChildResult::Refuted => (PN_INFINITY, 0),
            WidePnChildResult::Pending => self
                .resolved_child_entry(child)
                .and_then(|id| self.entries.get(id))
                .map(|entry| (entry.pn, entry.dn))
                .or_else(|| {
                    child
                        .future_key
                        .as_ref()
                        .and_then(WideFutureKey::virtual_key)
                        .and_then(|key| self.deferred_by_position.get(key))
                        .map(|deferred| (deferred.prior.pn, deferred.prior.dn))
                })
                .unwrap_or((child.prior.pn, child.prior.dn)),
        }
    }

    /// A thunk remains edge-local, but its exact key is also a virtual link to
    /// a transposition admitted through another parent. Every pre-selection
    /// read must observe that live entry just as an eagerly linked edge would.
    fn resolved_child_entry(&self, child: &WidePnChild) -> Option<usize> {
        child.entry.or_else(|| {
            child
                .future_key
                .as_ref()
                .and_then(WideFutureKey::virtual_key)
                .and_then(|key| self.by_position.get(key).copied())
        })
    }

    fn choice_order_pn(&self, child: &WidePnChild) -> u32 {
        self.child_numbers(child).0
    }

    #[cfg(test)]
    fn trace_pn_enabled() -> bool {
        std::env::var_os("TSS_TRACE_PN").is_some()
    }

    #[cfg(test)]
    fn trace_enabled(&self) -> bool {
        Self::trace_pn_enabled()
    }

    #[cfg(test)]
    fn format_trace_child(&self, child: &WidePnChild) -> String {
        let (pn, dn) = self.child_numbers(child);
        let (entry_id, entry_depth, entry_node, cutoff) = match self.resolved_child_entry(child) {
            Some(id) => match self.entries.get(id) {
                Some(entry) => (
                    id.to_string(),
                    entry.depth.to_string(),
                    wide_pn_node_tag(&entry.node),
                    matches!(entry.node, WidePnNode::DepthCutoff),
                ),
                None => (id.to_string(), "missing".to_owned(), "missing", false),
            },
            None => ("none".to_owned(), "none".to_owned(), "none", false),
        };
        format!(
            "pn={pn} dn={dn} prior_pn={} prior_dn={} result={:?} urgent={} first_tier={} entry={entry_id} entry_depth={entry_depth} entry_node={entry_node} cutoff={cutoff}",
            child.prior.pn,
            child.prior.dn,
            child.result,
            child.urgent_block,
            child.first_width_tier,
        )
    }

    #[cfg(test)]
    fn trace_selected_path(&self, root_state: &RustHexoState, root: usize, reason: &str) {
        if !self.trace_enabled() {
            return;
        }
        let Some(root_entry) = self.entries.get(root) else {
            eprintln!(
                "WIDTH_PN_STOP reason={reason} expansions={} depth_cap={} root=missing",
                self.expansions, self.depth_cap
            );
            return;
        };
        eprintln!(
            "WIDTH_PN_STOP reason={reason} expansions={} depth_cap={} root_pn={} root_dn={}",
            self.expansions, self.depth_cap, root_entry.pn, root_entry.dn
        );

        const PATH_LIMIT: usize = 64;
        let mut state = root_state.clone();
        let mut entry_id = root;
        let mut seen = HashSet::new();
        let mut inherited_commitment = false;
        for hop in 0..PATH_LIMIT {
            if !seen.insert(entry_id) {
                eprintln!("WIDTH_PN_PATH hop={hop} entry={entry_id} stop=cycle");
                return;
            }
            let Some(entry) = self.entries.get(entry_id) else {
                eprintln!("WIDTH_PN_PATH hop={hop} entry={entry_id} stop=missing_entry");
                return;
            };
            let WidePnNode::Branch { kind, children } = &entry.node else {
                eprintln!(
                    "WIDTH_PN_PATH hop={hop} entry={entry_id} depth={} node={} pn={} dn={} stop=non_branch",
                    entry.depth,
                    wide_pn_node_tag(&entry.node),
                    entry.pn,
                    entry.dn
                );
                return;
            };

            let finish_partial_turn = matches!(state.phase(), TurnPhase::SecondStone { .. });
            let urgent_pair = matches!(state.phase(), TurnPhase::FirstStone)
                && *kind == WidePnKind::Choice
                && wide_choice_has_urgent_block(children);
            let sequential_root_probe = entry.depth == 0 && (finish_partial_turn || urgent_pair);
            let commitment_domain = inherited_commitment
                || matches!(kind, WidePnKind::Universal { .. })
                    && self.universal_commitment_active(entry_id, children);
            let selected = match kind {
                WidePnKind::Choice => self.select_child_index_with_tier(
                    *kind,
                    children,
                    sequential_root_probe,
                    self.prefer_width_tier_at_depth(entry.depth),
                ),
                WidePnKind::Universal { .. } if commitment_domain => {
                    self.universal_obligation_index(entry_id, children, &[])
                }
                WidePnKind::Universal { .. } => self.select_child_index_with_tier(
                    *kind,
                    children,
                    sequential_root_probe,
                    self.prefer_width_tier_at_depth(entry.depth),
                ),
            };
            let Some(child_rank) = selected else {
                eprintln!(
                    "WIDTH_PN_PATH hop={hop} entry={entry_id} depth={} node={} pn={} dn={} stop=no_selectable_child",
                    entry.depth,
                    wide_pn_node_tag(&entry.node),
                    entry.pn,
                    entry.dn
                );
                return;
            };
            let child = &children[child_rank];
            let child_fields = self.format_trace_child(child);
            eprintln!(
                "WIDTH_PN_PATH hop={hop} entry={entry_id} depth={} node={} pn={} dn={} child_rank={child_rank} mv={:?} {child_fields}",
                entry.depth,
                wide_pn_node_tag(&entry.node),
                entry.pn,
                entry.dn,
                child.mv
            );

            if child.result != WidePnChildResult::Pending {
                return;
            }
            let Some(next_entry) = self.resolved_child_entry(child) else {
                return;
            };
            let applied = match child.mv {
                WidePnMove::One(coord) => apply_placement(&mut state, Placement { coord }).is_ok(),
                WidePnMove::Pair(first, second) | WidePnMove::DefenderPair(first, second) => {
                    apply_placement(&mut state, Placement { coord: first }).is_ok()
                        && apply_placement(&mut state, Placement { coord: second }).is_ok()
                }
            };
            if !applied {
                eprintln!(
                    "WIDTH_PN_PATH hop={hop} entry={entry_id} child_rank={child_rank} stop=illegal_replay"
                );
                return;
            }
            inherited_commitment = commitment_domain;
            entry_id = next_entry;
        }
        eprintln!("WIDTH_PN_PATH entry={entry_id} stop=path_limit limit={PATH_LIMIT}");
    }

    #[cfg(test)]
    fn select_child_index(
        &self,
        kind: WidePnKind,
        children: &[WidePnChild],
        sequential_root_probe: bool,
    ) -> Option<usize> {
        self.select_child_index_with_tier(kind, children, sequential_root_probe, false)
    }

    fn select_child_index_with_tier(
        &self,
        kind: WidePnKind,
        children: &[WidePnChild],
        sequential_root_probe: bool,
        prefer_width_tier: bool,
    ) -> Option<usize> {
        if kind != WidePnKind::Choice || !self.zone_order_mode.enabled() {
            return self.select_child_index_baseline(
                kind,
                children,
                sequential_root_probe,
                prefer_width_tier,
            );
        }

        let baseline = self.select_child_index_baseline(
            kind,
            children,
            sequential_root_probe,
            prefer_width_tier,
        )?;
        let baseline_child = &children[baseline];

        if sequential_root_probe {
            let baseline_class = (
                self.child_numbers(baseline_child).0 != 0,
                !baseline_child.urgent_block,
                if prefer_width_tier {
                    baseline_child.first_width_tier
                } else {
                    0
                },
                baseline_child.prior.pn,
            );
            return children
                .iter()
                .enumerate()
                .filter(|(_, child)| !self.child_is_genuinely_refuted(child))
                .filter(|(_, child)| {
                    (
                        self.child_numbers(child).0 != 0,
                        !child.urgent_block,
                        if prefer_width_tier {
                            child.first_width_tier
                        } else {
                            0
                        },
                        child.prior.pn,
                    ) == baseline_class
                })
                .min_by_key(|(rank, child)| (child.zone_order_key, *rank))
                .map(|(index, _)| index);
        }

        // Width tier and immutable fork prior are hard classes. Start with the
        // class selected by the historical policy, then admit only its current
        // PN tie/band. This cannot pull a child across any established class.
        let baseline_width = if prefer_width_tier {
            baseline_child.first_width_tier
        } else {
            0
        };
        let baseline_prior = baseline_child.prior.pn;
        let band_limit = self
            .choice_order_pn(baseline_child)
            .saturating_add(self.zone_order_band);
        children
            .iter()
            .enumerate()
            .filter(|(_, child)| !self.child_is_genuinely_refuted(child))
            .filter(|(_, child)| {
                (if prefer_width_tier {
                    child.first_width_tier
                } else {
                    0
                }) == baseline_width
                    && child.prior.pn == baseline_prior
                    && self.choice_order_pn(child) <= band_limit
            })
            .min_by_key(|(rank, child)| (child.zone_order_key, *rank))
            .map(|(index, _)| index)
    }

    /// Historical selector kept as a separate off-path so R-OS2 cannot alter
    /// default scheduling through a changed tuple or filter.
    fn select_child_index_baseline(
        &self,
        kind: WidePnKind,
        children: &[WidePnChild],
        sequential_root_probe: bool,
        prefer_width_tier: bool,
    ) -> Option<usize> {
        if kind == WidePnKind::Choice && sequential_root_probe {
            return children
                .iter()
                .enumerate()
                .filter(|(_, child)| !self.child_is_genuinely_refuted(child))
                .min_by_key(|(rank, child)| {
                    let tactical = self.child_numbers(child).0 == 0;
                    (
                        !tactical,
                        !child.urgent_block,
                        if prefer_width_tier {
                            child.first_width_tier
                        } else {
                            0
                        },
                        child.prior.pn,
                        *rank,
                    )
                })
                .map(|(index, _)| index);
        }
        if kind == WidePnKind::Choice && prefer_width_tier {
            // A completed proof remains more-proving than every unresolved
            // width class. The tier profile only orders live obligations; it
            // must not postpone an already terminal claimant child.
            if let Some((index, _)) = children.iter().enumerate().find(|(_, child)| {
                !self.child_is_genuinely_refuted(child) && self.choice_order_pn(child) == 0
            }) {
                return Some(index);
            }
        }
        children
            .iter()
            .enumerate()
            .filter(|(_, child)| match kind {
                // A finite sum can saturate at the same sentinel used for a
                // finished child.  Selection must use semantic resolution,
                // not the numeric tie, or an earlier finished child can make
                // an otherwise live frontier report `Stalled`.
                WidePnKind::Choice => !self.child_is_genuinely_refuted(child),
                WidePnKind::Universal { .. } => !self.child_is_genuinely_proven(child),
            })
            .min_by_key(|(_, child)| {
                match kind {
                    // Iterator::min_by_key retains the first equal key, so
                    // canonical generator order is the only normal tie-break.
                    WidePnKind::Choice if prefer_width_tier => (
                        u32::from(child.first_width_tier),
                        self.choice_order_pn(child),
                    ),
                    WidePnKind::Choice => (0, self.choice_order_pn(child)),
                    WidePnKind::Universal { .. } => (self.child_numbers(child).1, 0),
                }
            })
            .map(|(index, _)| index)
    }

    /// Return whether this AND node has the high linked fanout where DN
    /// re-selection compounds obligation interleaving. Exact TT convergence is
    /// counted once, and an unlinked proof obligation postpones commitment.
    /// Linked entries remain part of the node's structural fanout after they
    /// prove so a qualifying Universal stays sequential through its binary
    /// tail instead of changing policy mid-proof.
    fn has_commitment_fanout(&self, children: &[WidePnChild]) -> bool {
        let mut unique = Vec::with_capacity(MIN_COMMITTED_UNIVERSAL_OBLIGATIONS);
        for child in children {
            let WidePnChildResult::Pending = child.result else {
                continue;
            };
            let Some(identity) = self.child_obligation_identity(child) else {
                return false;
            };
            if unique.contains(&identity) {
                continue;
            }
            if unique.len() < MIN_COMMITTED_UNIVERSAL_OBLIGATIONS {
                unique.push(identity);
            }
        }
        unique.len() >= MIN_COMMITTED_UNIVERSAL_OBLIGATIONS
    }

    fn child_obligation_identity<'a>(
        &'a self,
        child: &'a WidePnChild,
    ) -> Option<WideChildObligation<'a>> {
        if let Some(entry) = self
            .resolved_child_entry(child)
            .filter(|&entry| self.entries.get(entry).is_some())
        {
            return Some(WideChildObligation::Entry(entry));
        }
        let key = child.future_key.as_ref()?.virtual_key()?;
        Some(
            self.by_position
                .get(key)
                .copied()
                .map(WideChildObligation::Entry)
                .unwrap_or(WideChildObligation::FutureKey(key)),
        )
    }

    fn same_child_obligation(&self, left: &WidePnChild, right: &WidePnChild) -> bool {
        match (
            self.child_obligation_identity(left),
            self.child_obligation_identity(right),
        ) {
            (Some(left), Some(right)) => left == right,
            _ => false,
        }
    }

    fn universal_commitment_active(&self, id: usize, children: &[WidePnChild]) -> bool {
        self.entries[id]
            .universal_obligation
            .and_then(|index| children.get(index))
            .is_some_and(|child| !self.child_is_genuinely_proven(child))
            || self.has_commitment_fanout(children)
    }

    /// Select one high-fanout Universal obligation without letting changing
    /// DN estimates interleave its siblings. The first selection is exactly
    /// the ordinary lowest-DN/generator-order choice; later selections retain
    /// it until it resolves. `yielded` contains true-stall failures already
    /// tried by the current descent and lets the existing stall path fail over
    /// once per distinct sibling instead of spinning on an unaffordable child.
    fn universal_obligation_index(
        &self,
        id: usize,
        children: &[WidePnChild],
        yielded: &[usize],
    ) -> Option<usize> {
        let selectable = |index: usize, child: &WidePnChild| {
            let yielded_same_entry = yielded.iter().any(|&yielded_index| {
                children
                    .get(yielded_index)
                    .is_some_and(|yielded_child| self.same_child_obligation(child, yielded_child))
            });
            !yielded.contains(&index)
                && !yielded_same_entry
                && !self.child_is_genuinely_proven(child)
        };
        if let Some(index) = self.entries[id].universal_obligation {
            if children
                .get(index)
                .is_some_and(|child| selectable(index, child))
            {
                return Some(index);
            }
        }
        children
            .iter()
            .enumerate()
            .filter(|(index, child)| selectable(*index, child))
            .min_by_key(|(_, child)| self.child_numbers(child).1)
            .map(|(index, _)| index)
    }

    fn select_step_child_index(
        &mut self,
        id: usize,
        sequential_root_probe: bool,
        prefer_width_tier: bool,
        yielded: &[usize],
    ) -> Option<usize> {
        self.select_step_child_index_with_commitment(
            id,
            sequential_root_probe,
            prefer_width_tier,
            yielded,
            false,
        )
    }

    fn select_step_child_index_with_commitment(
        &mut self,
        id: usize,
        sequential_root_probe: bool,
        prefer_width_tier: bool,
        yielded: &[usize],
        inherited_commitment: bool,
    ) -> Option<usize> {
        let (selected, universal_commitment) = {
            let WidePnNode::Branch { kind, children } = &self.entries[id].node else {
                return None;
            };
            match kind {
                WidePnKind::Choice => (
                    self.select_child_index_with_tier(
                        *kind,
                        children,
                        sequential_root_probe,
                        prefer_width_tier,
                    ),
                    None,
                ),
                WidePnKind::Universal { .. } => {
                    let commitment =
                        inherited_commitment || self.universal_commitment_active(id, children);
                    let selected = if commitment {
                        self.universal_obligation_index(id, children, yielded)
                    } else {
                        self.select_child_index_with_tier(
                            *kind,
                            children,
                            sequential_root_probe,
                            prefer_width_tier,
                        )
                    };
                    (selected, Some(commitment))
                }
            }
        };
        if let Some(commitment) = universal_commitment {
            self.entries[id].universal_obligation = if commitment { selected } else { None };
        }
        selected
    }

    /// A staged depth cutoff is unresolved, not a disproof. Sequential root
    /// probing must stay committed to that static top child so the caller can
    /// advance the horizon instead of silently moving to a lower-ranked turn.
    fn child_is_genuinely_refuted(&self, child: &WidePnChild) -> bool {
        match child.result {
            WidePnChildResult::Refuted => true,
            WidePnChildResult::Pending => self
                .resolved_child_entry(child)
                .and_then(|id| self.entries.get(id))
                .is_some_and(|entry| {
                    entry.dn == 0 && !matches!(entry.node, WidePnNode::DepthCutoff)
                }),
            WidePnChildResult::ClaimantCompletion | WidePnChildResult::ClaimantTactical => false,
        }
    }

    fn child_is_genuinely_proven(&self, child: &WidePnChild) -> bool {
        match child.result {
            WidePnChildResult::ClaimantCompletion | WidePnChildResult::ClaimantTactical => true,
            WidePnChildResult::Pending => self
                .resolved_child_entry(child)
                .and_then(|id| self.entries.get(id))
                .is_some_and(|entry| entry.pn == 0),
            WidePnChildResult::Refuted => false,
        }
    }

    fn recompute(&mut self, id: usize) -> bool {
        let previous = (self.entries[id].pn, self.entries[id].dn);
        let numbers = match &self.entries[id].node {
            WidePnNode::Unexpanded => {
                let prior = self.entries[id].prior;
                (prior.pn, prior.dn)
            }
            WidePnNode::ProvenLeaf(_) | WidePnNode::ProvenFragment(_) => (0, PN_INFINITY),
            WidePnNode::DepthCutoff | WidePnNode::Refuted => (PN_INFINITY, 0),
            WidePnNode::Branch { kind, children } => match kind {
                WidePnKind::Choice => {
                    let pn = children
                        .iter()
                        .map(|child| self.child_numbers(child).0)
                        .min()
                        .unwrap_or(PN_INFINITY);
                    let dn = children.iter().fold(0u32, |sum, child| {
                        sum.saturating_add(self.child_numbers(child).1)
                            .min(PN_INFINITY)
                    });
                    (pn, dn)
                }
                WidePnKind::Universal { .. } => {
                    let pn = children.iter().fold(0u32, |sum, child| {
                        sum.saturating_add(self.child_numbers(child).0)
                            .min(PN_INFINITY)
                    });
                    let dn = children
                        .iter()
                        .map(|child| self.child_numbers(child).1)
                        .min()
                        .unwrap_or(0);
                    (pn, dn)
                }
            },
        };
        #[cfg(test)]
        if self.threshold_counters {
            let sum_kind = match &self.entries[id].node {
                WidePnNode::Branch {
                    kind: WidePnKind::Choice,
                    ..
                } if numbers.1 == PN_INFINITY => Some(0),
                WidePnNode::Branch {
                    kind: WidePnKind::Universal { .. },
                    ..
                } if numbers.0 == PN_INFINITY => Some(1),
                _ => None,
            };
            if let Some(index) = sum_kind {
                let mut stats = self.threshold_stats.borrow_mut();
                stats.sentinel_sum_hits[index] = stats.sentinel_sum_hits[index].saturating_add(1);
            }
        }
        self.entries[id].pn = numbers.0;
        self.entries[id].dn = numbers.1;
        #[cfg(test)]
        if self.closure_counters && previous.0 != 0 && numbers.0 == 0 {
            self.record_closure_winning_rank(id);
        }
        previous != numbers
    }

    #[cfg(test)]
    fn record_closure_winning_rank(&mut self, id: usize) {
        let winning_index = match &self.entries[id].node {
            WidePnNode::Branch {
                kind: WidePnKind::Choice,
                children,
            } => children
                .iter()
                .position(|child| self.child_numbers(child).0 == 0),
            _ => None,
        };
        let Some(winning_index) = winning_index else {
            return;
        };
        let rank = winning_index.saturating_add(1);
        let bin = match rank {
            1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            5..=8 => 4,
            9..=16 => 5,
            17..=32 => 6,
            _ => 7,
        };
        let mut stats = self.closure_stats.borrow_mut();
        stats.winning_choice_nodes = stats.winning_choice_nodes.saturating_add(1);
        stats.winning_rank_bins[bin] = stats.winning_rank_bins[bin].saturating_add(1);
        let Some(node_profile) = self.closure_pair_nodes.get(&id).copied() else {
            return;
        };
        let Some(child_profile) = self
            .closure_pair_children
            .get(&(id, winning_index))
            .copied()
        else {
            return;
        };
        stats.reveal_pair_evaluated = stats
            .reveal_pair_evaluated
            .saturating_add(node_profile.evaluated);
        stats.reveal_pair_prefix = stats
            .reveal_pair_prefix
            .saturating_add(child_profile.evaluation_ordinal);
        stats.avoidable_second_candidate_nanos =
            stats.avoidable_second_candidate_nanos.saturating_add(
                node_profile
                    .second_candidate_nanos
                    .saturating_sub(child_profile.second_candidate_nanos),
            );
        stats.avoidable_pair_evaluation_nanos =
            stats.avoidable_pair_evaluation_nanos.saturating_add(
                node_profile
                    .pair_evaluation_nanos
                    .saturating_sub(child_profile.pair_evaluation_nanos),
            );
        stats.avoidable_dedup_nanos = stats.avoidable_dedup_nanos.saturating_add(
            node_profile
                .dedup_nanos
                .saturating_sub(child_profile.dedup_nanos),
        );
        if !self.reveal_prefix_study {
            return;
        }
        let Some(reveal_node) = self.reveal_pair_nodes.get(&id) else {
            return;
        };
        let WidePnNode::Branch { children, .. } = &self.entries[id].node else {
            return;
        };
        let rank_bin = |value: u64| match value {
            1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            5..=8 => 4,
            9..=16 => 5,
            17..=32 => 6,
            _ => 7,
        };
        let winner_zone = child_profile.zone_bound;
        let zone_rank = 1u64.saturating_add(
            children
                .iter()
                .enumerate()
                .filter(|(index, child)| {
                    (child.ordering.zone_bound, *index) < (winner_zone, winning_index)
                })
                .count()
                .try_into()
                .unwrap_or(u64::MAX),
        );
        let historical_rank = u64::try_from(rank).unwrap_or(u64::MAX);
        let total_zone_work =
            reveal_node
                .zone
                .iter()
                .copied()
                .fold(RevealZoneWork::default(), |mut total, work| {
                    total.evaluated = total.evaluated.saturating_add(work.evaluated);
                    total.second_candidate_nanos = total
                        .second_candidate_nanos
                        .saturating_add(work.second_candidate_nanos);
                    total.pair_evaluation_nanos = total
                        .pair_evaluation_nanos
                        .saturating_add(work.pair_evaluation_nanos);
                    total.dedup_nanos = total.dedup_nanos.saturating_add(work.dedup_nanos);
                    total
                });
        let lower_zone_work = reveal_node
            .zone
            .iter()
            .take(usize::from(winner_zone))
            .copied()
            .fold(RevealZoneWork::default(), |mut total, work| {
                total.evaluated = total.evaluated.saturating_add(work.evaluated);
                total.second_candidate_nanos = total
                    .second_candidate_nanos
                    .saturating_add(work.second_candidate_nanos);
                total.pair_evaluation_nanos = total
                    .pair_evaluation_nanos
                    .saturating_add(work.pair_evaluation_nanos);
                total.dedup_nanos = total.dedup_nanos.saturating_add(work.dedup_nanos);
                total
            });
        let zone_evaluation_prefix = lower_zone_work
            .evaluated
            .saturating_add(child_profile.zone_evaluation_prefix);
        let zone_second_prefix = lower_zone_work
            .second_candidate_nanos
            .saturating_add(child_profile.zone_second_candidate_nanos);
        let zone_evaluation_nanos_prefix = lower_zone_work
            .pair_evaluation_nanos
            .saturating_add(child_profile.zone_pair_evaluation_nanos);
        let zone_dedup_prefix = lower_zone_work
            .dedup_nanos
            .saturating_add(child_profile.zone_dedup_nanos);
        let mut expanded_total = 0u64;
        let mut historical_expanded_tail = 0u64;
        let mut zone_expanded_tail = 0u64;
        for (index, child) in children.iter().enumerate() {
            let Some(profile) = self.closure_pair_children.get(&(id, index)) else {
                continue;
            };
            if !profile.expanded {
                continue;
            }
            expanded_total = expanded_total.saturating_add(1);
            if index > winning_index {
                historical_expanded_tail = historical_expanded_tail.saturating_add(1);
            }
            if (child.ordering.zone_bound, index) > (winner_zone, winning_index) {
                zone_expanded_tail = zone_expanded_tail.saturating_add(1);
            }
        }
        stats.reveal_proven_pair_nodes = stats.reveal_proven_pair_nodes.saturating_add(1);
        stats.reveal_rank_bins[0][rank_bin(historical_rank)] =
            stats.reveal_rank_bins[0][rank_bin(historical_rank)].saturating_add(1);
        stats.reveal_rank_bins[1][rank_bin(zone_rank)] =
            stats.reveal_rank_bins[1][rank_bin(zone_rank)].saturating_add(1);
        stats.reveal_evaluation_rank_bins[0][rank_bin(child_profile.evaluation_ordinal)] = stats
            .reveal_evaluation_rank_bins[0][rank_bin(child_profile.evaluation_ordinal)]
        .saturating_add(1);
        stats.reveal_evaluation_rank_bins[1][rank_bin(zone_evaluation_prefix)] = stats
            .reveal_evaluation_rank_bins[1][rank_bin(zone_evaluation_prefix)]
        .saturating_add(1);
        for order in 0..2 {
            stats.reveal_total_evaluated[order] =
                stats.reveal_total_evaluated[order].saturating_add(node_profile.evaluated);
            stats.reveal_total_expanded[order] =
                stats.reveal_total_expanded[order].saturating_add(expanded_total);
        }
        stats.reveal_prefix_evaluated[0] =
            stats.reveal_prefix_evaluated[0].saturating_add(child_profile.evaluation_ordinal);
        stats.reveal_prefix_evaluated[1] =
            stats.reveal_prefix_evaluated[1].saturating_add(zone_evaluation_prefix);
        stats.reveal_avoidable_expanded[0] =
            stats.reveal_avoidable_expanded[0].saturating_add(historical_expanded_tail);
        stats.reveal_avoidable_expanded[1] =
            stats.reveal_avoidable_expanded[1].saturating_add(zone_expanded_tail);
        stats.reveal_avoidable_second_candidate_nanos[0] =
            stats.reveal_avoidable_second_candidate_nanos[0].saturating_add(
                node_profile
                    .second_candidate_nanos
                    .saturating_sub(child_profile.second_candidate_nanos),
            );
        stats.reveal_avoidable_pair_evaluation_nanos[0] =
            stats.reveal_avoidable_pair_evaluation_nanos[0].saturating_add(
                node_profile
                    .pair_evaluation_nanos
                    .saturating_sub(child_profile.pair_evaluation_nanos),
            );
        stats.reveal_avoidable_dedup_nanos[0] = stats.reveal_avoidable_dedup_nanos[0]
            .saturating_add(
                node_profile
                    .dedup_nanos
                    .saturating_sub(child_profile.dedup_nanos),
            );
        stats.reveal_avoidable_second_candidate_nanos[1] =
            stats.reveal_avoidable_second_candidate_nanos[1].saturating_add(
                total_zone_work
                    .second_candidate_nanos
                    .saturating_sub(zone_second_prefix),
            );
        stats.reveal_avoidable_pair_evaluation_nanos[1] =
            stats.reveal_avoidable_pair_evaluation_nanos[1].saturating_add(
                total_zone_work
                    .pair_evaluation_nanos
                    .saturating_sub(zone_evaluation_nanos_prefix),
            );
        stats.reveal_avoidable_dedup_nanos[1] = stats.reveal_avoidable_dedup_nanos[1]
            .saturating_add(
                total_zone_work
                    .dedup_nanos
                    .saturating_sub(zone_dedup_prefix),
            );
    }

    #[cfg(test)]
    fn finalize_ordering_study(&self) {
        if !self.ordering_study {
            return;
        }
        let mut records = Vec::new();
        for entry in &self.entries {
            let WidePnNode::Branch {
                kind: WidePnKind::Choice,
                children,
            } = &entry.node
            else {
                continue;
            };
            if entry.pn != 0 || children.is_empty() {
                continue;
            }
            let Some(winning_index) = children
                .iter()
                .position(|child| self.child_numbers(child).0 == 0)
            else {
                continue;
            };
            let key = |order: usize, index: usize, feature: OrderingChildFeatures| {
                let baseline = u32::try_from(index).unwrap_or(u32::MAX);
                let zone = u32::from(feature.zone_bound);
                let census = u32::from(feature.census_distance);
                let gate = u32::from(u16::MAX.saturating_sub(feature.gate_adjacency));
                let stone = u32::from(feature.d_stone);
                match order {
                    0 => [baseline, 0, 0, 0, 0],
                    1 => [zone, baseline, 0, 0, 0],
                    2 => [census, baseline, 0, 0, 0],
                    3 => [gate, baseline, 0, 0, 0],
                    4 => [stone, baseline, 0, 0, 0],
                    5 => [census, zone, stone, gate, baseline],
                    6 => [zone, gate, census, stone, baseline],
                    _ => unreachable!("ordering study key index"),
                }
            };
            let winning_feature = children[winning_index].ordering;
            let ranks = std::array::from_fn(|order| {
                let winning_key = key(order, winning_index, winning_feature);
                1u32.saturating_add(
                    children
                        .iter()
                        .enumerate()
                        .filter(|(index, child)| key(order, *index, child.ordering) < winning_key)
                        .count()
                        .try_into()
                        .unwrap_or(u32::MAX),
                )
            });
            records.push(OrderingStudyRecord {
                depth: u32::try_from(entry.depth).unwrap_or(u32::MAX),
                generated_children: u32::try_from(children.len()).unwrap_or(u32::MAX),
                pair_node: children
                    .iter()
                    .all(|child| matches!(child.mv, WidePnMove::Pair(_, _))),
                ranks,
            });
        }
        ORDERING_STUDY_REPORT.with(|slot| slot.borrow_mut().records.extend(records));
    }

    fn refresh(&mut self, id: usize) {
        self.recompute(id);
    }

    fn expand(&mut self, state: &mut RustHexoState, id: usize) -> WidePnStepOutcome {
        #[cfg(test)]
        let _timer = WideGenTimer::start(&WIDE_EXPAND_NANOS);
        if self.expansions >= self.node_cap {
            return WidePnStepOutcome::Stalled;
        }
        self.expansions += 1;
        #[cfg(test)]
        {
            if self.threshold_counters {
                self.threshold_expansion_clock.set(self.expansions);
                let post_saturation = self.tt_first_rejection.is_some();
                let selection = self.threshold_band_stack.last().copied();
                let mut stats = self.threshold_stats.borrow_mut();
                match (selection, post_saturation) {
                    (Some(selection), false) => match selection.kind {
                        WidePnKind::Choice => {
                            stats.choice_gap_expansions_pre_saturation[selection.gap_bin] = stats
                                .choice_gap_expansions_pre_saturation[selection.gap_bin]
                                .saturating_add(1);
                        }
                        WidePnKind::Universal { .. } => {
                            stats.universal_gap_expansions_pre_saturation[selection.gap_bin] =
                                stats.universal_gap_expansions_pre_saturation[selection.gap_bin]
                                    .saturating_add(1);
                        }
                    },
                    (Some(selection), true) => match selection.kind {
                        WidePnKind::Choice => {
                            stats.choice_gap_expansions_post_saturation[selection.gap_bin] = stats
                                .choice_gap_expansions_post_saturation[selection.gap_bin]
                                .saturating_add(1);
                        }
                        WidePnKind::Universal { .. } => {
                            stats.universal_gap_expansions_post_saturation[selection.gap_bin] =
                                stats.universal_gap_expansions_post_saturation[selection.gap_bin]
                                    .saturating_add(1);
                        }
                    },
                    (None, false) => {
                        stats.unclassified_expansions_pre_saturation = stats
                            .unclassified_expansions_pre_saturation
                            .saturating_add(1);
                    }
                    (None, true) => {
                        stats.unclassified_expansions_post_saturation = stats
                            .unclassified_expansions_post_saturation
                            .saturating_add(1);
                    }
                }
            }
            if let Some(telemetry) = self.quotient_telemetry.as_mut() {
                telemetry.observe_expand(id, state);
            }
            pn_init_record_wide_expansion(self, state, id);
        }
        let depth = usize::try_from(state.placements_made().saturating_sub(self.root_ply))
            .unwrap_or(usize::MAX);
        if depth > self.depth_cap {
            self.entries[id].node = WidePnNode::DepthCutoff;
            self.refresh(id);
            return WidePnStepOutcome::DepthCutoff {
                depth,
                made_progress: true,
            };
        }
        if state.placements_made() > self.semantic_horizon {
            // A still-live line the semantic deadline refused (depth-bound, not
            // structural): the horizon-ladder trigger. A defender-to-move node
            // is one where the opponent is still branching (k < B, before the
            // fully-forced boundary) — the Group-2 `deep_kb_death` signal.
            self.horizon_cuts = self.horizon_cuts.saturating_add(1);
            if state.current_player() != self.claimant {
                self.kb_death_cuts = self.kb_death_cuts.saturating_add(1);
            }
            self.entries[id].node = WidePnNode::Refuted;
            self.refresh(id);
            return WidePnStepOutcome::Progress;
        }
        if let Some(store) = self.fragment_store.filter(|store| store.entry_count != 0) {
            self.fragment_lookups = self.fragment_lookups.saturating_add(1);
            let key = PositionKey::from_state(state);
            if let Some(fragment) = store.lookup(&key, self.claimant) {
                let proof = &fragment.proof;
                let root_is_universal = matches!(
                    proof.nodes.get(proof.root_node as usize),
                    Some(CertNode::Universal { .. })
                );
                let compatible = proof.validate().is_some()
                    && proof.resolution_t <= self.semantic_horizon
                    && proof
                        .zone_build_t
                        .is_none_or(|build_t| self.semantic_horizon <= build_t)
                    && depth
                        .checked_add(proof.height)
                        .is_some_and(|height| height <= self.max_depth_cap)
                    // Parent commutation permissions are path-local. A cached
                    // Universal is consumed only at the solve root, whose
                    // verifier context is known to be empty.
                    && (depth == 0 || !root_is_universal);
                if compatible {
                    self.fragment_hits = self.fragment_hits.saturating_add(1);
                    self.entries[id].node = WidePnNode::ProvenFragment(fragment);
                    self.refresh(id);
                    return WidePnStepOutcome::Progress;
                }
            }
        }
        if let Some(outcome) = state.terminal() {
            self.entries[id].node = if outcome.winner == self.claimant {
                WidePnNode::Refuted
            } else {
                WidePnNode::Refuted
            };
            self.refresh(id);
            return WidePnStepOutcome::Progress;
        }
        if !matches!(state.phase(), TurnPhase::Opening) {
            let analysis = threats::analyze(state);
            if let Some(winner) = winner_from_analysis(state, &analysis) {
                if winner == self.claimant {
                    match typed_lambda_leaf(
                        state,
                        winner,
                        &analysis,
                        WidthOptions::vcf_pair_complete(),
                    ) {
                        Some(leaf) if node_resolution(&leaf) <= self.semantic_horizon => {
                            self.entries[id].node = WidePnNode::ProvenLeaf(leaf);
                        }
                        Some(_) => {
                            // A real claimant win whose resolution ply is past
                            // the deadline: a depth-bound refusal, not a
                            // structural one — a horizon cut.
                            self.horizon_cuts = self.horizon_cuts.saturating_add(1);
                            self.entries[id].node = WidePnNode::Refuted;
                        }
                        None => {
                            self.entries[id].node = WidePnNode::Refuted;
                        }
                    }
                } else {
                    self.entries[id].node = WidePnNode::Refuted;
                }
                self.refresh(id);
                return WidePnStepOutcome::Progress;
            }
        }

        if self.interior_census_gate && state.current_player() == self.claimant {
            if let Some(evaluation) = evaluate_interior_census_gate(
                state,
                self.claimant,
                self.root_ply,
                self.semantic_horizon,
            ) {
                self.interior_gate_evaluations = self.interior_gate_evaluations.saturating_add(1);
                self.interior_gate_nanos =
                    self.interior_gate_nanos.saturating_add(evaluation.nanos);
                if evaluation.dismiss {
                    self.interior_gate_dismissals = self.interior_gate_dismissals.saturating_add(1);
                    self.entries[id].node = WidePnNode::Refuted;
                    self.refresh(id);
                    return WidePnStepOutcome::Progress;
                }
            }
        }

        let (kind, mut children) = if state.current_player() == self.claimant {
            (WidePnKind::Choice, self.attack_children(state, depth))
        } else {
            let analysis = threats::analyze(state);
            let implicit_dispatch = !matches!(state.phase(), TurnPhase::Opening)
                && analysis.opp_threat_count > 0
                && !analysis.own_win_now
                && analysis.min_hitting_set == Some(analysis.b);
            if !implicit_dispatch {
                self.entries[id].node = WidePnNode::Refuted;
                self.refresh(id);
                return WidePnStepOutcome::Progress;
            }
            let children = self.defender_boundary_children(state, analysis.b);
            (WidePnKind::Universal { implicit_dispatch }, children)
        };
        #[cfg(test)]
        let closure_pair_profiles = if self.closure_counters
            && kind == WidePnKind::Choice
            && matches!(state.phase(), TurnPhase::FirstStone)
        {
            Some((
                self.closure_last_pair_node.get(),
                std::mem::take(&mut *self.closure_last_pair_children.borrow_mut()),
            ))
        } else {
            None
        };
        #[cfg(test)]
        let reveal_pair_profile = if self.reveal_prefix_study
            && kind == WidePnKind::Choice
            && matches!(state.phase(), TurnPhase::FirstStone)
        {
            Some(std::mem::take(
                &mut *self.reveal_last_pair_node.borrow_mut(),
            ))
        } else {
            None
        };
        children.shrink_to_fit();
        self.entries[id].node = if children.is_empty() {
            WidePnNode::Refuted
        } else {
            WidePnNode::Branch { kind, children }
        };
        #[cfg(test)]
        if let Some((node_profile, child_profiles)) = closure_pair_profiles {
            self.closure_pair_nodes.insert(id, node_profile);
            for (child_index, profile) in child_profiles.into_iter().enumerate() {
                self.closure_pair_children
                    .insert((id, child_index), profile);
            }
        }
        #[cfg(test)]
        if let Some(profile) = reveal_pair_profile {
            self.reveal_pair_nodes.insert(id, profile);
        }
        self.refresh(id);
        WidePnStepOutcome::Progress
    }

    fn attack_children(&self, state: &mut RustHexoState, depth: usize) -> Vec<WidePnChild> {
        match state.phase() {
            TurnPhase::FirstStone => self.attack_pair_children(state, depth),
            TurnPhase::SecondStone { first } => {
                self.attack_single_children(state, depth, Some(first))
            }
            TurnPhase::Opening => self.attack_single_children(state, depth, None),
        }
    }

    /// Enumerate complete attacker turns. A first stone is never admitted to
    /// the proof frontier by itself: either it wins immediately, or a retained
    /// pair must pass the new-threat and tight-dispatch forcing checks.
    /// Stateless replacement for the historical apply-and-analyze pair gate;
    /// see `WideTurnGate::evaluate_pair` for the classification contract.
    fn evaluate_wide_pair_at_gate(
        &self,
        gate: &WideTurnGate,
        first: HexCoord,
        second: HexCoord,
    ) -> Option<(WidePnChildResult, WidePnPrior)> {
        let (result, prior) = gate.evaluate_pair(first, second, self.semantic_horizon)?;
        #[cfg(test)]
        let prior = if self.live_ge3_seed {
            let started = Instant::now();
            let live_ge3 = gate.live_ge3_after_pair(first, second);
            let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.live_ge3_seed_scans
                .set(self.live_ge3_seed_scans.get().saturating_add(1));
            self.live_ge3_seed_nanos
                .set(self.live_ge3_seed_nanos.get().saturating_add(nanos));
            WidePnPrior {
                pn: pn_from_fork_degree(live_ge3),
                dn: prior.dn,
            }
        } else {
            prior
        };
        Some((result, prior))
    }

    fn attack_pair_children(&self, state: &mut RustHexoState, _depth: usize) -> Vec<WidePnChild> {
        #[cfg(test)]
        let _gen_timer = WideGenTimer::start(&WIDE_GEN_PAIR_NANOS);
        #[cfg(test)]
        let closure_started = self.closure_counters.then(Instant::now);
        #[cfg(test)]
        let gate_started = self.closure_counters.then(Instant::now);
        let gate = WideTurnGate::build(state, self.claimant);
        #[cfg(test)]
        let observe_ordering_study = self.ordering_study;
        #[cfg(test)]
        let observe_reveal_prefix = self.reveal_prefix_study;
        #[cfg(not(test))]
        let observe_ordering_study = false;
        #[cfg(not(test))]
        let observe_reveal_prefix = false;
        #[cfg(test)]
        let reveal_context_started = observe_reveal_prefix.then(Instant::now);
        let ordering_context = if self.zone_order_mode.enabled()
            || observe_ordering_study
            || observe_reveal_prefix
        {
            #[cfg(test)]
            let context_started = self.zone_order_mode.enabled().then(Instant::now);
            let context =
                OrderingFeatureContext::from_state(state, self.claimant, observe_ordering_study);
            #[cfg(test)]
            if let Some(started) = context_started {
                WIDE_ZONE_ORDER_CONTEXTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                WIDE_ZONE_ORDER_CONTEXT_NANOS.fetch_add(
                    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            Some(context)
        } else {
            None
        };
        #[cfg(test)]
        let mut reveal_analysis_nanos = reveal_context_started
            .map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        #[cfg(test)]
        let gate_build_nanos = gate_started
            .map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        #[cfg(test)]
        let mut pair_profile = ClosurePairNodeProfile::default();
        #[cfg(test)]
        let mut pair_child_profiles = Vec::new();
        #[cfg(test)]
        let mut reveal_pair_profile = RevealPairNodeProfile::default();
        #[cfg(test)]
        let mut reveal_distance_cache = HashMap::<HexCoord, u16>::new();
        #[cfg(test)]
        let mut reveal_zone_keys = Vec::<u16>::new();
        #[cfg(test)]
        let mut accepted = 0u64;
        #[cfg(test)]
        let mut retained = 0u64;
        let first_candidates = ordered_threat_creating_moves_with_width(
            state,
            self.claimant,
            WidthOptions::vcf_pair_complete(),
        );
        // Freeze urgency at the turn-start position. A block cell can disappear
        // from the second-stone candidate metadata after the other coordinate is
        // played, but the unordered pair still contains that original block.
        let defender_blocks = turn_start_defender_blocks(&first_candidates);
        let mut children = Vec::new();
        let mut seen_pairs = HashSet::new();
        // No claimant >=4 window exists here (win-now nodes leaf before
        // generation), so a lone first stone can never complete six: the
        // whole double loop is stateless — zero engine applies.
        let mut second_coords: Vec<HexCoord> = Vec::new();
        let mut second_seen: HashSet<HexCoord> = HashSet::new();
        for first_candidate in &first_candidates {
            let first_width_tier = wide_candidate_width_tier(first_candidate);
            let first = first_candidate.coord;
            #[cfg(test)]
            let mut reveal_second_elapsed = 0u64;
            {
                #[cfg(test)]
                let _regen_timer = WideGenTimer::start(&WIDE_GEN_PRIOR_NANOS);
                #[cfg(test)]
                let second_started = self.closure_counters.then(Instant::now);
                gate.second_candidates(
                    first,
                    &first_candidates,
                    &mut second_coords,
                    &mut second_seen,
                );
                #[cfg(test)]
                if let Some(started) = second_started {
                    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    reveal_second_elapsed = elapsed;
                    pair_profile.second_candidate_nanos =
                        pair_profile.second_candidate_nanos.saturating_add(elapsed);
                }
            }
            #[cfg(test)]
            if observe_reveal_prefix {
                let analysis_started = Instant::now();
                reveal_zone_keys.clear();
                let context = ordering_context
                    .as_ref()
                    .expect("reveal-prefix study builds a turn-start context");
                let first_distance =
                    context.cached_nearest_claimant_distance(first, &mut reveal_distance_cache);
                reveal_zone_keys.extend(second_coords.iter().map(|&second| {
                    first_distance.max(
                        context
                            .cached_nearest_claimant_distance(second, &mut reveal_distance_cache),
                    )
                }));
                let minimum_zone = reveal_zone_keys
                    .iter()
                    .copied()
                    .min()
                    .unwrap_or(first_distance);
                let work = reveal_pair_profile.work_mut(minimum_zone);
                work.second_candidate_nanos = work
                    .second_candidate_nanos
                    .saturating_add(reveal_second_elapsed);
                reveal_analysis_nanos = reveal_analysis_nanos.saturating_add(
                    u64::try_from(analysis_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                );
            }
            for (_second_index, &second) in second_coords.iter().enumerate() {
                #[cfg(test)]
                let reveal_zone_bound = if observe_reveal_prefix {
                    reveal_zone_keys[_second_index]
                } else {
                    0
                };
                // Stateless classification from the turn-start window
                // snapshot: no engine applies in the pair double loop.
                #[cfg(test)]
                let evaluation_started = self.closure_counters.then(Instant::now);
                let evaluated = self.evaluate_wide_pair_at_gate(&gate, first, second);
                #[cfg(test)]
                if let Some(started) = evaluation_started {
                    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    pair_profile.evaluated = pair_profile.evaluated.saturating_add(1);
                    pair_profile.pair_evaluation_nanos =
                        pair_profile.pair_evaluation_nanos.saturating_add(elapsed);
                    if observe_reveal_prefix {
                        let work = reveal_pair_profile.work_mut(reveal_zone_bound);
                        work.evaluated = work.evaluated.saturating_add(1);
                        work.pair_evaluation_nanos =
                            work.pair_evaluation_nanos.saturating_add(elapsed);
                    }
                }
                if let Some((result, prior)) = evaluated {
                    #[cfg(test)]
                    if self.closure_counters {
                        accepted = accepted.saturating_add(1);
                    }
                    // Deduplicate the two legal orders by their actual
                    // unordered coordinate pair. Candidate membership is not
                    // monotone: a defender-block coordinate can disappear
                    // after the other stone, so coordinate-order pruning can
                    // incorrectly discard the only generated ordering.
                    #[cfg(test)]
                    let dedup_started = self.closure_counters.then(Instant::now);
                    let inserted = {
                        let first_key = raw_coord_key(first);
                        let second_key = raw_coord_key(second);
                        let pair_key = if first_key <= second_key {
                            (first_key, second_key)
                        } else {
                            (second_key, first_key)
                        };
                        seen_pairs.insert(pair_key)
                    };
                    #[cfg(test)]
                    if let Some(started) = dedup_started {
                        let elapsed =
                            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                        pair_profile.dedup_nanos = pair_profile.dedup_nanos.saturating_add(elapsed);
                        if observe_reveal_prefix {
                            let work = reveal_pair_profile.work_mut(reveal_zone_bound);
                            work.dedup_nanos = work.dedup_nanos.saturating_add(elapsed);
                        }
                    }
                    if !inserted {
                        continue;
                    }
                    #[cfg(test)]
                    if self.closure_counters {
                        retained = retained.saturating_add(1);
                        pair_child_profiles.push(ClosurePairChildProfile {
                            evaluation_ordinal: pair_profile.evaluated,
                            second_candidate_nanos: pair_profile.second_candidate_nanos,
                            pair_evaluation_nanos: pair_profile.pair_evaluation_nanos,
                            dedup_nanos: pair_profile.dedup_nanos,
                            zone_bound: reveal_zone_bound,
                            zone_evaluation_prefix: reveal_pair_profile
                                .zone
                                .get(usize::from(reveal_zone_bound))
                                .map_or(0, |work| work.evaluated),
                            zone_second_candidate_nanos: reveal_pair_profile
                                .zone
                                .get(usize::from(reveal_zone_bound))
                                .map_or(0, |work| work.second_candidate_nanos),
                            zone_pair_evaluation_nanos: reveal_pair_profile
                                .zone
                                .get(usize::from(reveal_zone_bound))
                                .map_or(0, |work| work.pair_evaluation_nanos),
                            zone_dedup_nanos: reveal_pair_profile
                                .zone
                                .get(usize::from(reveal_zone_bound))
                                .map_or(0, |work| work.dedup_nanos),
                            ..ClosurePairChildProfile::default()
                        });
                    }
                    let mv = WidePnMove::Pair(first, second);
                    let zone_order_key = if self.zone_order_mode.enabled() {
                        #[cfg(test)]
                        let key_started = Instant::now();
                        let key = ordering_context
                            .as_ref()
                            .expect("live zone ordering builds a turn-start context")
                            .pair_key(first, second, self.zone_order_mode);
                        #[cfg(test)]
                        {
                            WIDE_ZONE_ORDER_KEYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            WIDE_ZONE_ORDER_KEY_NANOS.fetch_add(
                                u64::try_from(key_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }
                        key
                    } else {
                        0
                    };
                    children.push(WidePnChild {
                        mv,
                        result,
                        entry: None,
                        future_key: (self.lazy_frontier && result == WidePnChildResult::Pending)
                            .then(|| {
                                WideFutureKey::OnSelection(WidePositionKey::after_completed_pair(
                                    state, first, second,
                                ))
                            }),
                        prior,
                        urgent_block: wide_move_contains_defender_block(mv, &defender_blocks),
                        first_width_tier,
                        zone_order_key,
                        #[cfg(test)]
                        ordering: ordering_context
                            .as_ref()
                            .map(|context| {
                                if observe_ordering_study {
                                    context.features(&[first, second])
                                } else {
                                    OrderingChildFeatures {
                                        zone_bound: reveal_zone_bound,
                                        ..OrderingChildFeatures::default()
                                    }
                                }
                            })
                            .unwrap_or_default(),
                    });
                }
            }
        }
        #[cfg(test)]
        if self.closure_counters {
            let pair_generation_nanos = closure_started
                .map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
                .unwrap_or(0);
            let mut stats = self.closure_stats.borrow_mut();
            stats.pairs_evaluated = stats.pairs_evaluated.saturating_add(pair_profile.evaluated);
            stats.pairs_accepted = stats.pairs_accepted.saturating_add(accepted);
            stats.pairs_retained = stats.pairs_retained.saturating_add(retained);
            stats.pair_generation_nanos = stats
                .pair_generation_nanos
                .saturating_add(pair_generation_nanos);
            stats.gate_build_nanos = stats.gate_build_nanos.saturating_add(gate_build_nanos);
            stats.second_candidate_nanos = stats
                .second_candidate_nanos
                .saturating_add(pair_profile.second_candidate_nanos);
            stats.pair_evaluation_nanos = stats
                .pair_evaluation_nanos
                .saturating_add(pair_profile.pair_evaluation_nanos);
            stats.dedup_nanos = stats.dedup_nanos.saturating_add(pair_profile.dedup_nanos);
            drop(stats);
            self.closure_last_pair_node.set(pair_profile);
            *self.closure_last_pair_children.borrow_mut() = pair_child_profiles;
            if observe_reveal_prefix {
                let mut stats = self.closure_stats.borrow_mut();
                stats.reveal_analysis_nanos = stats
                    .reveal_analysis_nanos
                    .saturating_add(reveal_analysis_nanos);
                drop(stats);
                *self.reveal_last_pair_node.borrow_mut() = reveal_pair_profile;
            }
        }
        children
    }

    fn attack_single_children(
        &self,
        state: &mut RustHexoState,
        depth: usize,
        turn_first: Option<HexCoord>,
    ) -> Vec<WidePnChild> {
        let candidates = ordered_threat_creating_moves_with_width(
            state,
            self.claimant,
            WidthOptions::vcf_pair_complete(),
        );
        #[cfg(test)]
        let ordering_context = self
            .ordering_study
            .then(|| OrderingFeatureContext::from_state(state, self.claimant, true));
        let mut children = Vec::new();
        for candidate in candidates {
            #[cfg(test)]
            let ordering = ordering_context
                .as_ref()
                .map(|context| context.features(&[candidate.coord]))
                .unwrap_or_default();
            let Ok((result, delta)) = state.apply_with_delta(Placement {
                coord: candidate.coord,
            }) else {
                continue;
            };
            let completion_ply = self.root_ply.saturating_add(depth as u32).saturating_add(1);
            let (child_result, prior) = if let Some(outcome) = result.outcome {
                if outcome.winner == self.claimant && completion_ply <= self.semantic_horizon {
                    (
                        Some(WidePnChildResult::ClaimantCompletion),
                        WidePnPrior::UNIFORM,
                    )
                } else {
                    (None, WidePnPrior::UNIFORM)
                }
            } else if let Some(first) = turn_first {
                if immediate_winner(state, WidthOptions::vcf_pair_complete()).is_some_and(
                    |(winner, ref leaf)| {
                        winner == self.claimant && node_resolution(leaf) <= self.semantic_horizon
                    },
                ) {
                    (
                        Some(WidePnChildResult::ClaimantTactical),
                        WidePnPrior::UNIFORM,
                    )
                } else {
                    let forcing = (turn_created_claimant_threat(
                        state,
                        self.claimant,
                        first,
                        candidate.coord,
                    ) && turn_forces_small_defender_reply(state, self.claimant))
                    .then_some(WidePnChildResult::Pending);
                    let prior = forcing
                        .is_some()
                        .then(|| self.completed_turn_prior(state))
                        .unwrap_or(WidePnPrior::UNIFORM);
                    (forcing, prior)
                }
            } else {
                (
                    Some(WidePnChildResult::Pending),
                    if state.current_player() == self.claimant {
                        self.position_prior(state)
                    } else {
                        self.completed_turn_prior(state)
                    },
                )
            };
            let future_key = (self.lazy_frontier
                && child_result == Some(WidePnChildResult::Pending))
            .then(|| WideFutureKey::OnSelection(WidePositionKey::from_state(state)));
            state.undo(delta);
            if let Some(result) = child_result {
                children.push(WidePnChild {
                    mv: WidePnMove::One(candidate.coord),
                    result,
                    entry: None,
                    future_key,
                    prior,
                    urgent_block: candidate.defender_block,
                    first_width_tier: 0,
                    zone_order_key: 0,
                    #[cfg(test)]
                    ordering,
                });
            }
        }
        children
    }

    fn defender_children(
        &mut self,
        state: &mut RustHexoState,
        defender_budget: u8,
    ) -> Vec<WidePnChild> {
        let mut explicit = forced_defender_replies(
            state,
            self.claimant,
            defender_budget,
            WidthOptions::vcf_pair_complete(),
        );
        let frame = canonical_frame(state);
        explicit.sort_by_key(|coord| canonical_coord_key(frame, *coord));
        let mut children = Vec::with_capacity(explicit.len());
        for coord in explicit {
            let Ok((result, delta)) = state.apply_with_delta(Placement { coord }) else {
                continue;
            };
            let child_result = match result.outcome {
                Some(outcome) if outcome.winner == self.claimant => {
                    WidePnChildResult::ClaimantCompletion
                }
                Some(_) => WidePnChildResult::Refuted,
                None => WidePnChildResult::Pending,
            };
            let prior = (child_result == WidePnChildResult::Pending)
                .then(|| self.position_prior(state))
                .unwrap_or(WidePnPrior::UNIFORM);
            let (entry, future_key) = if child_result == WidePnChildResult::Pending {
                let depth = usize::try_from(state.placements_made().saturating_sub(self.root_ply))
                    .unwrap_or(usize::MAX);
                let key = WidePositionKey::from_state(state);
                if self.lazy_frontier {
                    self.defer_position(&key, depth, prior);
                    (None, Some(WideFutureKey::Virtual(key)))
                } else {
                    (Some(self.insert_position(key, depth, prior)), None)
                }
            } else {
                (None, None)
            };
            state.undo(delta);
            children.push(WidePnChild {
                mv: WidePnMove::One(coord),
                result: child_result,
                entry,
                future_key,
                prior,
                urgent_block: false,
                first_width_tier: 0,
                zone_order_key: 0,
                #[cfg(test)]
                ordering: OrderingChildFeatures::default(),
            });
        }
        children
    }

    fn defender_boundary_children(
        &mut self,
        state: &mut RustHexoState,
        defender_budget: u8,
    ) -> Vec<WidePnChild> {
        if defender_budget == 2 && matches!(state.phase(), TurnPhase::FirstStone) {
            if let Some(children) = self.defender_pair_children(state) {
                return children;
            }
        }
        self.defender_children(state, defender_budget)
    }

    fn defender_pair_children(&mut self, state: &mut RustHexoState) -> Option<Vec<WidePnChild>> {
        #[cfg(test)]
        let _gen_timer = WideGenTimer::start(&WIDE_GEN_DEFENDER_NANOS);
        let plan = forced_defender_pair_plan(state, self.claimant)?;
        let depth = usize::try_from(
            state
                .placements_made()
                .saturating_add(2)
                .saturating_sub(self.root_ply),
        )
        .unwrap_or(usize::MAX);
        Some(
            plan.pairs
                .into_iter()
                .map(|pair| {
                    #[cfg(test)]
                    let final_prior = if self.live_ge3_seed {
                        let (_, first_delta) = state
                            .apply_with_delta(Placement { coord: pair.first })
                            .expect("validated defender-pair first move");
                        let (_, second_delta) = state
                            .apply_with_delta(Placement { coord: pair.second })
                            .expect("validated defender-pair second move");
                        let prior = WidePnPrior {
                            pn: self.live_ge3_seed_prior(state),
                            dn: pair.final_prior.dn,
                        };
                        state.undo(second_delta);
                        state.undo(first_delta);
                        prior
                    } else {
                        pair.final_prior
                    };
                    #[cfg(not(test))]
                    let final_prior = pair.final_prior;
                    let (entry, future_key) = if self.lazy_frontier {
                        self.defer_position(&pair.final_key, depth, final_prior);
                        (None, Some(WideFutureKey::Virtual(pair.final_key)))
                    } else {
                        (
                            Some(self.insert_position(pair.final_key, depth, final_prior)),
                            None,
                        )
                    };
                    WidePnChild {
                        mv: WidePnMove::DefenderPair(pair.first, pair.second),
                        result: WidePnChildResult::Pending,
                        entry,
                        future_key,
                        prior: final_prior,
                        urgent_block: false,
                        first_width_tier: 0,
                        zone_order_key: 0,
                        #[cfg(test)]
                        ordering: OrderingChildFeatures::default(),
                    }
                })
                .collect(),
        )
    }

    fn materialize(&self, state: &RustHexoState, root: usize) -> Option<WideMaterializedProof> {
        if self.entries.get(root)?.pn != 0 {
            return None;
        }
        let mut work = state.clone();
        let mut builder = WideProofMaterializer {
            search: self,
            arena: Vec::new(),
            edge_count: 0,
            commutation_count: 0,
            witness_count: 0,
            fragment_imports: 0,
            dag_reuses: 0,
            memo: HashMap::new(),
        };
        let root_node = builder.build(&mut work, root)?;
        Some(WideMaterializedProof {
            arena: builder.arena,
            root_node,
            fragment_imports: builder.fragment_imports,
            dag_reuses: builder.dag_reuses,
        })
    }
}

struct WideMaterializedProof {
    arena: Vec<CertNode>,
    root_node: CertNodeId,
    fragment_imports: u64,
    dag_reuses: u64,
}

struct WideProofMaterializer<'search, 'store> {
    search: &'search WidePnSearch<'store>,
    arena: Vec<CertNode>,
    edge_count: usize,
    commutation_count: usize,
    witness_count: usize,
    fragment_imports: u64,
    dag_reuses: u64,
    memo: HashMap<PositionKey, CertNodeId>,
}

impl WideProofMaterializer<'_, '_> {
    fn build(&mut self, state: &mut RustHexoState, id: usize) -> Option<CertNodeId> {
        let key = PositionKey::from_state(state);
        if let Some(&node) = self.memo.get(&key) {
            self.dag_reuses = self.dag_reuses.saturating_add(1);
            return Some(node);
        }
        let entry = self.search.entries.get(id)?;
        if entry.pn != 0 {
            return None;
        }
        let node = match entry.node.clone() {
            WidePnNode::ProvenLeaf(leaf) => self.alloc(leaf, 0)?,
            WidePnNode::ProvenFragment(fragment) => {
                if fragment.claimant != self.search.claimant || fragment.key != key {
                    return None;
                }
                self.import_fragment(state, &fragment.proof)?
            }
            WidePnNode::Branch {
                kind: WidePnKind::Choice,
                children,
            } => {
                let child = children
                    .iter()
                    .find(|child| self.search.child_numbers(child).0 == 0)?
                    .clone();
                self.build_choice(state, &child)?
            }
            WidePnNode::Branch {
                kind: WidePnKind::Universal { implicit_dispatch },
                children,
            } => self.build_universal(state, implicit_dispatch, &children)?,
            WidePnNode::Unexpanded | WidePnNode::DepthCutoff | WidePnNode::Refuted => return None,
        };
        self.memo.insert(key, node);
        Some(node)
    }

    fn import_fragment(
        &mut self,
        state: &RustHexoState,
        proof: &CachedProof,
    ) -> Option<CertNodeId> {
        proof.validate()?;
        let depth =
            usize::try_from(state.placements_made().saturating_sub(self.search.root_ply)).ok()?;
        if proof.resolution_t > self.search.semantic_horizon
            || proof
                .zone_build_t
                .is_some_and(|build_t| self.search.semantic_horizon > build_t)
            || depth.checked_add(proof.height)? > self.search.max_depth_cap
            || self.arena.len().checked_add(proof.nodes.len())? > MAX_CERT_NODES
            || self.edge_count.checked_add(proof.explicit_edges)? > MAX_CERT_EDGES
            || self
                .commutation_count
                .checked_add(proof.commutation_count)?
                > MAX_CERT_COMMUTATIONS
            || self.witness_count.checked_add(proof.witness_count)? > MAX_CERT_WITNESSES
        {
            return None;
        }

        let base = self.arena.len();
        let final_len = base.checked_add(proof.nodes.len())?;
        u32::try_from(final_len).ok()?;
        let mut nodes = proof.nodes.clone();
        for node in &mut nodes {
            remap_node_ids_with_offset(node, base, final_len)?;
        }
        let root = offset_node_id(proof.root_node, base, final_len)?;
        self.arena.append(&mut nodes);
        self.edge_count += proof.explicit_edges;
        self.commutation_count += proof.commutation_count;
        self.witness_count += proof.witness_count;
        self.fragment_imports = self.fragment_imports.saturating_add(1);
        Some(root)
    }

    fn build_choice(
        &mut self,
        state: &mut RustHexoState,
        child: &WidePnChild,
    ) -> Option<CertNodeId> {
        match child.mv {
            WidePnMove::One(coord) => {
                let (result, delta) = state.apply_with_delta(Placement { coord }).ok()?;
                let node = match child.result {
                    WidePnChildResult::ClaimantCompletion => {
                        if result.outcome?.winner != self.search.claimant {
                            state.undo(delta);
                            return None;
                        }
                        let completion = wide_completion_node(
                            state,
                            self.search.claimant,
                            coord,
                            state.placements_made(),
                        );
                        state.undo(delta);
                        self.alloc(completion?, 0)?
                    }
                    WidePnChildResult::ClaimantTactical => {
                        if result.outcome.is_some() {
                            state.undo(delta);
                            return None;
                        }
                        let analysis = threats::analyze(state);
                        let leaf = typed_lambda_leaf(
                            state,
                            self.search.claimant,
                            &analysis,
                            WidthOptions::vcf_pair_complete(),
                        )
                        .filter(|leaf| node_resolution(leaf) <= self.search.semantic_horizon);
                        state.undo(delta);
                        let leaf = self.alloc(leaf?, 0)?;
                        self.alloc(
                            CertNode::Choice {
                                mv: coord,
                                child: leaf,
                            },
                            1,
                        )?
                    }
                    WidePnChildResult::Pending => {
                        let child_id = self.search.resolved_child_entry(child)?;
                        let proof = self.build(state, child_id);
                        state.undo(delta);
                        self.alloc(
                            CertNode::Choice {
                                mv: coord,
                                child: proof?,
                            },
                            1,
                        )?
                    }
                    WidePnChildResult::Refuted => {
                        state.undo(delta);
                        return None;
                    }
                };
                Some(node)
            }
            WidePnMove::Pair(first, second) => {
                let (first_result, first_delta) =
                    state.apply_with_delta(Placement { coord: first }).ok()?;
                if first_result.outcome.is_some() {
                    state.undo(first_delta);
                    return None;
                }
                let (second_result, second_delta) =
                    state.apply_with_delta(Placement { coord: second }).ok()?;
                let node = match child.result {
                    WidePnChildResult::ClaimantCompletion => {
                        if second_result.outcome?.winner != self.search.claimant {
                            state.undo(second_delta);
                            state.undo(first_delta);
                            return None;
                        }
                        let completion = wide_completion_node(
                            state,
                            self.search.claimant,
                            second,
                            state.placements_made(),
                        );
                        state.undo(second_delta);
                        state.undo(first_delta);
                        let completion = self.alloc(completion?, 0)?;
                        self.alloc(
                            CertNode::Choice {
                                mv: first,
                                child: completion,
                            },
                            1,
                        )?
                    }
                    WidePnChildResult::ClaimantTactical => {
                        if second_result.outcome.is_some() {
                            state.undo(second_delta);
                            state.undo(first_delta);
                            return None;
                        }
                        let analysis = threats::analyze(state);
                        let leaf = typed_lambda_leaf(
                            state,
                            self.search.claimant,
                            &analysis,
                            WidthOptions::vcf_pair_complete(),
                        )
                        .filter(|leaf| node_resolution(leaf) <= self.search.semantic_horizon);
                        state.undo(second_delta);
                        state.undo(first_delta);
                        let leaf = self.alloc(leaf?, 0)?;
                        let second_choice = self.alloc(
                            CertNode::Choice {
                                mv: second,
                                child: leaf,
                            },
                            1,
                        )?;
                        self.alloc(
                            CertNode::Choice {
                                mv: first,
                                child: second_choice,
                            },
                            1,
                        )?
                    }
                    WidePnChildResult::Pending => {
                        let proof = self.build(state, self.search.resolved_child_entry(child)?);
                        state.undo(second_delta);
                        state.undo(first_delta);
                        let second_choice = self.alloc(
                            CertNode::Choice {
                                mv: second,
                                child: proof?,
                            },
                            1,
                        )?;
                        self.alloc(
                            CertNode::Choice {
                                mv: first,
                                child: second_choice,
                            },
                            1,
                        )?
                    }
                    WidePnChildResult::Refuted => {
                        state.undo(second_delta);
                        state.undo(first_delta);
                        return None;
                    }
                };
                Some(node)
            }
            WidePnMove::DefenderPair(_, _) => None,
        }
    }

    fn build_universal(
        &mut self,
        state: &mut RustHexoState,
        implicit_dispatch: bool,
        children: &[WidePnChild],
    ) -> Option<CertNodeId> {
        if children
            .first()
            .is_some_and(|child| matches!(child.mv, WidePnMove::DefenderPair(_, _)))
        {
            if !implicit_dispatch
                || children
                    .iter()
                    .any(|child| !matches!(child.mv, WidePnMove::DefenderPair(_, _)))
            {
                return None;
            }
            return self.build_defender_pair_universal(state, children);
        }
        let mut edges = Vec::with_capacity(children.len());
        for child in children {
            if self.search.child_numbers(child).0 != 0 || child.result != WidePnChildResult::Pending
            {
                return None;
            }
            let WidePnMove::One(coord) = child.mv else {
                return None;
            };
            let (_result, delta) = state.apply_with_delta(Placement { coord }).ok()?;
            let proof = self.build(state, self.search.resolved_child_entry(child)?);
            state.undo(delta);
            edges.push(CertEdge {
                mv: coord,
                child: proof?,
            });
        }
        let edge_count = edges.len();
        self.alloc(
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone: None,
                commutations: Vec::new(),
            },
            edge_count,
        )
    }

    fn build_defender_pair_universal(
        &mut self,
        state: &mut RustHexoState,
        children: &[WidePnChild],
    ) -> Option<CertNodeId> {
        let plan = forced_defender_pair_plan(state, self.search.claimant)?;
        if plan.pairs.len() != children.len() {
            return None;
        }

        let mut child_by_pair = HashMap::with_capacity(children.len());
        for child in children {
            if child.result != WidePnChildResult::Pending || self.search.child_numbers(child).0 != 0
            {
                return None;
            }
            let WidePnMove::DefenderPair(first, second) = child.mv else {
                return None;
            };
            if raw_coord_key(first) >= raw_coord_key(second)
                || child_by_pair
                    .insert((raw_coord_key(first), raw_coord_key(second)), child)
                    .is_some()
            {
                return None;
            }
        }

        // Build each unique final-state proof exactly once. The reverse order
        // reaches the same state by construction and is represented below by
        // a checked CertCommutation rather than a second PN obligation.
        let mut proof_by_pair = HashMap::with_capacity(plan.pairs.len());
        for pair in &plan.pairs {
            let pair_key = (raw_coord_key(pair.first), raw_coord_key(pair.second));
            let child = *child_by_pair.get(&pair_key)?;
            let (first_result, first_delta) = state
                .apply_with_delta(Placement { coord: pair.first })
                .ok()?;
            if first_result.outcome.is_some() {
                state.undo(first_delta);
                return None;
            }
            let (second_result, second_delta) =
                match state.apply_with_delta(Placement { coord: pair.second }) {
                    Ok(applied) => applied,
                    Err(_) => {
                        state.undo(first_delta);
                        return None;
                    }
                };
            if second_result.outcome.is_some()
                || WidePositionKey::from_state(state) != pair.final_key
            {
                state.undo(second_delta);
                state.undo(first_delta);
                return None;
            }
            let Some(child_id) = self.search.resolved_child_entry(child) else {
                state.undo(second_delta);
                state.undo(first_delta);
                return None;
            };
            let proof = self.build(state, child_id);
            state.undo(second_delta);
            state.undo(first_delta);
            if proof_by_pair.insert(pair_key, proof?).is_some() {
                return None;
            }
        }

        // Retain the raw-low -> raw-high orientation explicitly. Every
        // raw-high -> raw-low orientation is omitted from that nested
        // Universal and justified by a root-level commutation record.
        let mut nested_by_first = HashMap::with_capacity(plan.kernel.len());
        for &first in &plan.kernel {
            let edges = plan
                .pairs
                .iter()
                .filter(|pair| pair.first == first)
                .map(|pair| {
                    Some(CertEdge {
                        mv: pair.second,
                        child: *proof_by_pair
                            .get(&(raw_coord_key(pair.first), raw_coord_key(pair.second)))?,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            let edge_count = edges.len();
            let node = self.alloc(
                CertNode::Universal {
                    edges,
                    implicit_dispatch: true,
                    zone: None,
                    commutations: Vec::new(),
                },
                edge_count,
            )?;
            if nested_by_first.insert(first, node).is_some() {
                return None;
            }
        }

        let edges = plan
            .kernel
            .iter()
            .map(|&mv| {
                Some(CertEdge {
                    mv,
                    child: *nested_by_first.get(&mv)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let commutations = plan
            .pairs
            .iter()
            .map(|pair| {
                Some(CertCommutation {
                    first: pair.second,
                    omitted_second: pair.first,
                    first_child: *nested_by_first.get(&pair.second)?,
                    mirror_child: *nested_by_first.get(&pair.first)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let edge_count = edges.len();
        self.alloc(
            CertNode::Universal {
                edges,
                implicit_dispatch: true,
                zone: None,
                commutations,
            },
            edge_count,
        )
    }

    fn alloc(&mut self, node: CertNode, added_edges: usize) -> Option<CertNodeId> {
        let added_commutations = match &node {
            CertNode::Universal { commutations, .. } => commutations.len(),
            _ => 0,
        };
        let added_witnesses = match &node {
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => 1,
            CertNode::Loss { witnesses, .. } => witnesses.len(),
            CertNode::Choice { .. }
            | CertNode::Universal { .. }
            | CertNode::UniversalGroup2V1(_) => 0,
            CertNode::FhwGateV1(gate) => gate.proof.threats.len(),
        };
        if self.arena.len() >= MAX_CERT_NODES
            || self.edge_count.saturating_add(added_edges) > MAX_CERT_EDGES
            || self.commutation_count.saturating_add(added_commutations) > MAX_CERT_COMMUTATIONS
            || (self.search.fragment_store.is_some()
                && self.witness_count.saturating_add(added_witnesses) > MAX_CERT_WITNESSES)
        {
            return None;
        }
        let id = u32::try_from(self.arena.len()).ok()?;
        self.arena.push(node);
        self.edge_count += added_edges;
        self.commutation_count += added_commutations;
        self.witness_count += added_witnesses;
        Some(id)
    }
}

fn wide_completion_node(
    state: &RustHexoState,
    claimant: Player,
    coord: HexCoord,
    completion_ply: u32,
) -> Option<CertNode> {
    let mut witnesses = state
        .board()
        .windows()
        .entries()
        .filter(|entry| {
            entry.key().contains(coord)
                && entry.count(claimant) == 6
                && entry.count(claimant.other()) == 0
        })
        .map(|entry| entry.key())
        .collect::<Vec<_>>();
    witnesses.sort_by_key(|key| window_key_order(*key));
    Some(CertNode::OrCompletion {
        mv: coord,
        witness: witnesses.first().copied()?,
        completion_ply,
    })
}

/// Narrow DFS state retained byte-for-byte as the compatibility backend
/// selected by `WidePnSearch::prove_narrow_compat`.
struct NarrowCompatSearch<'a> {
    node_cap: u64,
    nodes: u64,
    tt_hits: u64,
    hit_limit: bool,
    arena: Vec<CertNode>,
    edge_count: usize,
    tt: BoundedTt,
    shared_tt: Option<&'a mut SharedProofCache>,
    peak_tt_bytes: usize,
    /// Absolute placement index at the attempt root.  Structural depth is
    /// derived from the separately threaded ply clock.
    root_ply: u32,
    semantic_horizon: u32,
    clock_is_absolute: bool,
    zone: ZoneSearchCaps,
    width: WidthOptions,
    depth_cap: usize,
    /// Immutable solve-level opt-in. Environment lookup happens in
    /// `TssSolver::solve_goal`, never on the recursive search path.
    k_reply_consume: bool,
    #[cfg(test)]
    k_reply_shadow: Option<&'a mut Vec<KReplyShadowRecord>>,
    #[cfg(test)]
    quotient_telemetry: Option<NarrowQuotientTelemetry>,
    interior_census_gate: bool,
    interior_gate_evaluations: u64,
    interior_gate_dismissals: u64,
    interior_gate_nanos: u64,
    /// Still-live lines the semantic-horizon deadline refused, and the
    /// defender-to-move subset. Mirror of the wide-search counters
    /// (SolveStats::horizon_cuts / kb_death_cuts) for the narrow-compat path.
    horizon_cuts: u64,
    kb_death_cuts: u64,
    /// v1 Group-2 selector opt-in for this attempt.
    group2: bool,
    /// True once any node outside the narrow v1 class (implicit dispatch,
    /// legacy zone, commutation) has been allocated. Later Group-2 attempts
    /// are skipped: the assembled certificate could no longer validate as v1
    /// (class rules 2/3), so trying would only waste budget.
    emitted_dirty: bool,
}

#[cfg(test)]
struct NarrowQuotientTelemetry {
    report: QuotientTelemetryReport,
    expanded_raw: HashSet<WidePositionKey>,
    expanded_canonical: HashMap<WidePositionKey, WidePositionKey>,
}

#[cfg(test)]
impl NarrowQuotientTelemetry {
    fn enabled() -> Option<Self> {
        std::env::var_os("TSS_TURN_QUOTIENT_TELEMETRY").map(|_| Self {
            report: QuotientTelemetryReport::default(),
            expanded_raw: HashSet::new(),
            expanded_canonical: HashMap::new(),
        })
    }

    fn observe_expand(&mut self, state: &RustHexoState) {
        let raw = WidePositionKey::from_state(state);
        let started = Instant::now();
        let canonical = raw.d6_canonical();
        self.report.d6_canonicalization_calls =
            self.report.d6_canonicalization_calls.saturating_add(1);
        self.report.d6_canonicalization_nanos = self
            .report
            .d6_canonicalization_nanos
            .saturating_add(started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
        if !self.expanded_raw.insert(raw.clone()) {
            return;
        }
        self.report.expanded_unique_positions =
            self.report.expanded_unique_positions.saturating_add(1);
        if self
            .expanded_canonical
            .get(&canonical)
            .is_some_and(|first_raw| first_raw != &raw)
        {
            self.report.d6_expanded_duplicates =
                self.report.d6_expanded_duplicates.saturating_add(1);
        } else {
            self.expanded_canonical.insert(canonical, raw);
        }
        if let Some(interaction) = classify_last_two_turns(state) {
            self.report.commutation_eligible_nodes =
                self.report.commutation_eligible_nodes.saturating_add(1);
            if interaction == 0 {
                self.report.commutation_independent_nodes =
                    self.report.commutation_independent_nodes.saturating_add(1);
            }
            if interaction & 1 != 0 {
                self.report.commutation_shared_window =
                    self.report.commutation_shared_window.saturating_add(1);
            }
            if interaction & 2 != 0 {
                self.report.commutation_legality_coupling =
                    self.report.commutation_legality_coupling.saturating_add(1);
            }
            if interaction & 4 != 0 {
                self.report.commutation_threat_response =
                    self.report.commutation_threat_response.saturating_add(1);
            }
        }
    }

    fn finish(mut self, tt: &BoundedTt, tt_hits: u64) -> QuotientTelemetryReport {
        let entries = tt.slots.iter().flatten().collect::<Vec<_>>();
        let mut canonical = HashSet::new();
        for entry in &entries {
            canonical.insert(WidePositionKey::from_position_key(&entry.key).d6_canonical());
        }
        self.report.retained_entries = entries.len() as u64;
        self.report.indexed_entries = entries.len() as u64;
        self.report.tt_hits = tt_hits;
        self.report.d6_index_denominator = entries.len() as u64;
        self.report.d6_index_duplicates = entries.len().saturating_sub(canonical.len()) as u64;
        self.report.horizon_queries = self.report.expanded_unique_positions;
        self.report.horizon_clock_misses = self.report.expanded_unique_positions;
        self.report.horizon_positions = self.report.expanded_unique_positions;
        self.report.horizon_position_clock_entries = self.report.expanded_unique_positions;
        self.report
    }
}

#[derive(Clone, Debug)]
struct PairContext {
    first: HexCoord,
    turn_start_legal: Vec<HexCoord>,
}

/// Exact Q8 reply-survival kernel from the NQ2 proof. Urgency is deliberately
/// scoped to the theorem's nonterminal attacker SecondStone position. Defender
/// windows come from the engine's incrementally maintained exact mirror of all
/// active count-4+ windows; tests bind that mirror to a full `entries()` scan.
fn k_reply_eligible(state: &RustHexoState, claimant: Player) -> bool {
    state.terminal().is_none()
        && state.current_player() == claimant
        && matches!(state.phase(), TurnPhase::SecondStone { .. })
}

pub(crate) fn k_reply_kernel(
    state: &RustHexoState,
    claimant: Player,
    legal: &[HexCoord],
) -> KReplyKernel {
    let eligible = k_reply_eligible(state, claimant);
    let defender = claimant.other();
    let mut defender_windows = Vec::new();
    let mut win_now_windows = Vec::new();
    if eligible {
        // `live_threat_entries` is an exact, placement/undo-maintained mirror
        // of `entries().filter_map(threat_player)`. Iterate every member of
        // that complete active family, then apply Q8's exact owner/count cuts.
        for (owner, entry) in state.board().windows().live_threat_entries() {
            if owner == defender
                && entry.active_player() == Some(defender)
                && matches!(entry.count(defender), 4 | 5)
            {
                defender_windows.push(entry.key());
            } else if owner == claimant
                && entry.active_player() == Some(claimant)
                && entry.count(claimant) == 5
            {
                win_now_windows.push(entry.key());
            }
        }
    }
    let urgent = !defender_windows.is_empty();
    let cells = if !eligible {
        Vec::new()
    } else if urgent {
        legal
            .iter()
            .copied()
            .filter(|coord| {
                let wins_now = win_now_windows.iter().any(|window| window.contains(*coord));
                wins_now
                    || defender_windows
                        .iter()
                        .all(|window| window.contains(*coord))
            })
            .collect()
    } else {
        // Q8 defines BlockAll_D(P)=Legal(P) for the empty defender-window
        // family. `retained()` returns Legal(P) without copying it.
        Vec::new()
    };
    KReplyKernel {
        eligible,
        urgent,
        cells,
    }
}

impl NarrowCompatSearch<'static> {
    fn new(node_cap: u64, tt_bytes_cap: usize, hash_mask: u64) -> Self {
        let tt = BoundedTt::new(tt_bytes_cap, hash_mask);
        let peak_tt_bytes = tt.current_bytes;
        Self {
            node_cap,
            nodes: 0,
            tt_hits: 0,
            hit_limit: false,
            arena: Vec::new(),
            edge_count: 0,
            tt,
            shared_tt: None,
            peak_tt_bytes,
            root_ply: 0,
            semantic_horizon: u32::MAX,
            clock_is_absolute: false,
            zone: ZoneSearchCaps::default(),
            width: WidthOptions::default(),
            depth_cap: MAX_SEARCH_DEPTH,
            k_reply_consume: false,
            #[cfg(test)]
            k_reply_shadow: None,
            #[cfg(test)]
            quotient_telemetry: NarrowQuotientTelemetry::enabled(),
            interior_census_gate: false,
            interior_gate_evaluations: 0,
            interior_gate_dismissals: 0,
            interior_gate_nanos: 0,
            horizon_cuts: 0,
            kb_death_cuts: 0,
            group2: false,
            emitted_dirty: false,
        }
    }
}

impl<'a> NarrowCompatSearch<'a> {
    fn with_shared(
        node_cap: u64,
        tt_bytes_cap: usize,
        hash_mask: u64,
        shared_tt: &'a mut SharedProofCache,
        root_ply: u32,
        semantic_horizon: u32,
        zone: ZoneSearchCaps,
        width: WidthOptions,
        depth_cap: usize,
        k_reply_consume: bool,
        #[cfg(test)] k_reply_shadow: Option<&'a mut Vec<KReplyShadowRecord>>,
        interior_census_gate: bool,
        group2: bool,
    ) -> Self {
        let tt = BoundedTt::new(tt_bytes_cap, hash_mask);
        let peak_tt_bytes = tt.current_bytes.saturating_add(shared_tt.current_bytes);
        let shared_tt = (!shared_tt.slots.is_empty()).then_some(shared_tt);
        Self {
            node_cap,
            nodes: 0,
            tt_hits: 0,
            hit_limit: false,
            arena: Vec::new(),
            edge_count: 0,
            tt,
            shared_tt,
            peak_tt_bytes,
            root_ply,
            semantic_horizon,
            clock_is_absolute: true,
            zone,
            width,
            depth_cap,
            k_reply_consume,
            #[cfg(test)]
            k_reply_shadow,
            #[cfg(test)]
            quotient_telemetry: NarrowQuotientTelemetry::enabled(),
            interior_census_gate,
            interior_gate_evaluations: 0,
            interior_gate_dismissals: 0,
            interior_gate_nanos: 0,
            horizon_cuts: 0,
            kb_death_cuts: 0,
            group2,
            emitted_dirty: false,
        }
    }

    fn tt_entry_count(&self) -> usize {
        self.tt.entry_count()
            + self
                .shared_tt
                .as_ref()
                .map(|shared| shared.entry_count())
                .unwrap_or(0)
    }

    #[cfg(test)]
    fn tt_behavior_signature(&self) -> NarrowAttemptSignature {
        NarrowAttemptSignature {
            nodes: self.nodes,
            tt_hits: self.tt_hits,
            hit_limit: self.hit_limit,
            arena_len: self.arena.len(),
            edge_count: self.edge_count,
            local_tt: format!("{:?}", self.tt),
        }
    }

    #[cfg(test)]
    fn begin_k_reply_shadow(
        &mut self,
        full_legal_len: usize,
        enumerated: &[HexCoord],
        eligible: bool,
        urgent: bool,
        kernel: Option<&KReplyKernel>,
    ) -> Option<KReplyShadowTicket> {
        if self.k_reply_shadow.is_none() {
            return None;
        }
        let consumed = self.k_reply_consume && urgent;
        let kernel_cells = kernel
            .map(|kernel| kernel.cells.as_slice())
            .unwrap_or_default();
        let records = self
            .k_reply_shadow
            .as_deref_mut()
            .expect("checked Q8 shadow sink");
        let record = records.len();
        records.push(KReplyShadowRecord {
            full_quiet: full_legal_len,
            urgent,
            k_reply: urgent.then_some(kernel_cells.len()),
            proved_win: false,
            winning_edge: None,
            winning_edge_in_k: None,
            position: None,
            consumed,
            consumed_matches_shadow: consumed.then(|| enumerated == kernel_cells),
        });
        debug_assert!(!urgent || eligible);
        Some(KReplyShadowTicket {
            record,
            kernel: kernel_cells.to_vec(),
        })
    }

    #[cfg(test)]
    fn mark_k_reply_shadow_win(
        &mut self,
        ticket: &KReplyShadowTicket,
        state: &RustHexoState,
        edge: HexCoord,
    ) {
        let record = self
            .k_reply_shadow
            .as_deref_mut()
            .expect("Q8 shadow ticket requires sink")
            .get_mut(ticket.record)
            .expect("Q8 shadow ticket indexes its record");
        record.proved_win = true;
        record.winning_edge = Some(edge);
        record.winning_edge_in_k = record.urgent.then(|| ticket.kernel.contains(&edge));
        if record.winning_edge_in_k == Some(false) {
            record.position = Some(RootBinding::from_state(state));
        }
    }

    fn prove(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
        pair: Option<&PairContext>,
    ) -> Option<CertNodeId> {
        let depth = if self.clock_is_absolute {
            debug_assert_eq!(state.placements_made(), ply);
            usize::try_from(ply.checked_sub(self.root_ply)?).ok()?
        } else {
            ply as usize
        };
        if self.clock_is_absolute && ply > self.semantic_horizon {
            // Deadline refused a still-live line (depth-bound Unknown). A
            // defender-to-move node is pre-forced-boundary (k < B).
            self.horizon_cuts = self.horizon_cuts.saturating_add(1);
            if state.current_player() != claimant {
                self.kb_death_cuts = self.kb_death_cuts.saturating_add(1);
            }
            return None;
        }
        if depth > self.depth_cap {
            if !self.width.vcf_pair_complete {
                self.hit_limit = true;
            }
            return None;
        }
        if self.nodes >= self.node_cap {
            self.hit_limit = true;
            return None;
        }
        self.nodes += 1;
        #[cfg(test)]
        if let Some(telemetry) = self.quotient_telemetry.as_mut() {
            telemetry.observe_expand(state);
        }

        #[cfg(test)]
        let mut pn_init_guard =
            PnInitNarrowGuard::enter(state, claimant, self.root_ply, self.semantic_horizon);
        let pn_init_result = (|| {
            let key = PositionKey::from_state(state);
            if pair.is_none() {
                if let Some(node) = self.tt.lookup(&key, claimant) {
                    if node == LOCAL_TT_FAILED && self.width.vcf_pair_complete {
                        self.tt_hits += 1;
                        return None;
                    }
                    if (node as usize) < self.arena.len() {
                        self.tt_hits += 1;
                        return Some(node);
                    }
                }
                if let Some(node) = self.lookup_shared(&key, claimant, depth) {
                    self.tt.insert(key, claimant, node);
                    self.tt_hits += 1;
                    self.observe_tt_bytes();
                    return Some(node);
                }
            }

            if let Some(outcome) = state.terminal() {
                let _ = outcome;
                // A claimant completion is represented at its parent by the typed
                // OrCompletion leaf; defender-terminal edges are not certifiable.
                return None;
            }

            // Analyze each non-terminal node exactly once.  Universal dispatch
            // consumes this same immutable result instead of repeating the scan.
            let analysis = threats::analyze(state);
            if !matches!(state.phase(), TurnPhase::Opening) {
                if let Some(winner) = winner_from_analysis(state, &analysis) {
                    if winner != claimant {
                        return None;
                    }
                    let leaf = typed_lambda_leaf(state, winner, &analysis, self.width)?;
                    if node_resolution(&leaf) > self.semantic_horizon {
                        self.horizon_cuts = self.horizon_cuts.saturating_add(1);
                        return None;
                    }
                    let node = self.alloc_node(leaf, 0)?;
                    self.remember_proof(key, claimant, node);
                    return Some(node);
                }
            }

            let gate_dismissed = if self.interior_census_gate && state.current_player() == claimant
            {
                evaluate_interior_census_gate(state, claimant, self.root_ply, self.semantic_horizon)
                    .is_some_and(|evaluation| {
                        self.interior_gate_evaluations =
                            self.interior_gate_evaluations.saturating_add(1);
                        self.interior_gate_nanos =
                            self.interior_gate_nanos.saturating_add(evaluation.nanos);
                        if evaluation.dismiss {
                            self.interior_gate_dismissals =
                                self.interior_gate_dismissals.saturating_add(1);
                        }
                        evaluation.dismiss
                    })
            } else {
                false
            };

            let node = if state.current_player() == claimant {
                if gate_dismissed {
                    None
                } else {
                    self.prove_choice(state, claimant, ply, &analysis, pair)
                }
            } else {
                self.prove_universal(state, claimant, ply, &analysis, pair)
            };
            let Some(node) = node else {
                if self.width.vcf_pair_complete && !self.hit_limit && pair.is_none() {
                    self.tt.insert(key, claimant, LOCAL_TT_FAILED);
                    self.observe_tt_bytes();
                }
                return None;
            };
            if pair.is_none() {
                self.remember_proof(key, claimant, node);
            }
            Some(node)
        })();
        #[cfg(test)]
        pn_init_guard.finish(pn_init_result.is_some(), self.hit_limit);
        pn_init_result
    }

    fn prove_choice(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
        analysis: &threats::ThreatAnalysis,
        pair: Option<&PairContext>,
    ) -> Option<CertNodeId> {
        // Descending line count is the static proof-number initialization:
        // completions before four-builds before three-builds.  The coordinate
        // tie break makes the order independent of WindowStore hash iteration.
        let mut candidates = ordered_threat_creating_moves_with_width(state, claimant, self.width);
        if self.width.vcf_pair_complete {
            if let Some(pair) = pair {
                candidates.retain(|candidate| pair_candidate_allowed(candidate.coord, pair));
            }
        }
        let quiet_priority = candidates
            .iter()
            .enumerate()
            .map(|(rank, candidate)| (candidate.coord, rank))
            .collect::<HashMap<_, _>>();
        let turn_start_candidates = (self.width.vcf_pair_complete
            && pair.is_none()
            && matches!(state.phase(), TurnPhase::FirstStone)
            && threats::placements_remaining(state) == 2)
            .then(|| {
                let mut coords = candidates
                    .iter()
                    .map(|candidate| candidate.coord)
                    .collect::<Vec<_>>();
                coords.sort_by_key(|coord| raw_coord_key(*coord));
                coords
            });
        // Wide mode is a VCF search, not merely a wider unrestricted attack
        // search.  Capture the turn's first coordinate so the completed pair
        // can be rejected unless it created a new count-four (or stronger)
        // claimant window.  This also covers roots entered at SecondStone.
        let turn_first = if self.width.vcf_pair_complete {
            match state.phase() {
                TurnPhase::SecondStone { first } => Some(first),
                _ => None,
            }
        } else {
            None
        };
        for candidate in candidates {
            let Ok((result, delta)) = state.apply_with_delta(Placement {
                coord: candidate.coord,
            }) else {
                continue;
            };
            if result
                .outcome
                .is_some_and(|outcome| outcome.winner == claimant)
            {
                let mut witnesses = state
                    .board()
                    .windows()
                    .entries()
                    .filter(|entry| {
                        entry.key().contains(candidate.coord)
                            && entry.count(claimant) == 6
                            && entry.count(claimant.other()) == 0
                    })
                    .map(|entry| entry.key())
                    .collect::<Vec<_>>();
                witnesses.sort_by_key(|key| window_key_order(*key));
                let witness = witnesses.first().copied();
                state.undo(delta);
                let completion_ply = ply.checked_add(1)?;
                if completion_ply > self.semantic_horizon {
                    self.horizon_cuts = self.horizon_cuts.saturating_add(1);
                    return None;
                }
                return self.alloc_node(
                    CertNode::OrCompletion {
                        mv: candidate.coord,
                        witness: witness?,
                        completion_ply,
                    },
                    0,
                );
            }
            if let Some(first) = turn_first {
                let created = turn_created_claimant_threat(state, claimant, first, candidate.coord);
                if !created || !turn_forces_small_defender_reply(state, claimant) {
                    state.undo(delta);
                    continue;
                }
            }
            let pair_context = turn_start_candidates.as_ref().and_then(|turn_start_legal| {
                (matches!(state.phase(), TurnPhase::SecondStone { .. })).then(|| PairContext {
                    first: candidate.coord,
                    turn_start_legal: turn_start_legal.clone(),
                })
            });
            let child = self.prove(state, claimant, ply.checked_add(1)?, pair_context.as_ref());
            state.undo(delta);

            if let Some(child) = child {
                return self.alloc_node(
                    CertNode::Choice {
                        mv: candidate.coord,
                        child,
                    },
                    1,
                );
            }
            if self.hit_limit {
                return None;
            }
        }
        if self.width.consumes_quiet_turns() {
            let mut complete = Vec::new();
            state.write_legal_moves(&mut complete);
            #[cfg(test)]
            let full_legal_len = complete.len();
            let eligible = k_reply_eligible(state, claimant);
            // `analysis` was recomputed from this exact current/post-first
            // state immediately before dispatch to `prove_choice`. Because
            // current_player == claimant here, its opponent threat family is
            // exactly T_D(P); no second active-window walk is needed merely
            // to establish the overwhelmingly common nonurgent case.
            let urgent = eligible && analysis.opp_threat_count > 0;
            #[cfg(not(test))]
            let observe_k_reply = false;
            #[cfg(test)]
            let observe_k_reply = self.k_reply_shadow.is_some();
            let k_reply = (urgent && (self.k_reply_consume || observe_k_reply))
                .then(|| k_reply_kernel(state, claimant, &complete));
            if self.k_reply_consume && urgent {
                let kernel = k_reply
                    .as_ref()
                    .expect("urgent Q8 consumption computes its kernel");
                debug_assert!(kernel.eligible && kernel.urgent);
                complete.clone_from(&kernel.cells);
            }
            #[cfg(test)]
            let k_reply_ticket = observe_k_reply
                .then(|| {
                    self.begin_k_reply_shadow(
                        full_legal_len,
                        &complete,
                        eligible,
                        urgent,
                        k_reply.as_ref(),
                    )
                })
                .flatten();
            if let Some(pair) = pair {
                restrict_pair_candidates(&mut complete, pair);
            }
            let frame = canonical_frame(state);
            complete.sort_by_key(|coord| {
                (
                    quiet_priority.get(coord).copied().unwrap_or(usize::MAX),
                    canonical_coord_key(frame, *coord),
                )
            });
            for coord in complete {
                let Ok((result, delta)) = state.apply_with_delta(Placement { coord }) else {
                    continue;
                };
                let completion_ply = ply.checked_add(1)?;
                if result
                    .outcome
                    .is_some_and(|outcome| outcome.winner == claimant)
                {
                    let completion = (completion_ply <= self.semantic_horizon)
                        .then(|| wide_completion_node(state, claimant, coord, completion_ply))
                        .flatten();
                    state.undo(delta);
                    let node = self.alloc_node(completion?, 0);
                    #[cfg(test)]
                    if node.is_some() {
                        if let Some(ticket) = &k_reply_ticket {
                            self.mark_k_reply_shadow_win(ticket, state, coord);
                        }
                    }
                    return node;
                }
                if result.outcome.is_some() {
                    state.undo(delta);
                    continue;
                }
                let pair_context = turn_start_candidates.as_ref().and_then(|turn_start_legal| {
                    matches!(state.phase(), TurnPhase::SecondStone { .. }).then(|| PairContext {
                        first: coord,
                        turn_start_legal: turn_start_legal.clone(),
                    })
                });
                let child = self.prove(state, claimant, completion_ply, pair_context.as_ref());
                state.undo(delta);
                if let Some(child) = child {
                    let node = self.alloc_node(CertNode::Choice { mv: coord, child }, 1);
                    #[cfg(test)]
                    if node.is_some() {
                        if let Some(ticket) = &k_reply_ticket {
                            self.mark_k_reply_shadow_win(ticket, state, coord);
                        }
                    }
                    return node;
                }
                if self.hit_limit {
                    return None;
                }
            }
        }
        // Exhausting a restricted attacker set only says that this attack
        // generator found no proof.  It is deliberately not a disproof.
        None
    }

    fn prove_universal(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
        analysis: &threats::ThreatAnalysis,
        pair: Option<&PairContext>,
    ) -> Option<CertNodeId> {
        let implicit_dispatch = !matches!(state.phase(), TurnPhase::Opening)
            && analysis.opp_threat_count > 0
            && !analysis.own_win_now
            && analysis.min_hitting_set == Some(analysis.b);

        // A wide descendant defender is reachable only after a completed
        // forcing attacker turn.  Keep this invariant at the dispatcher as a
        // backstop so an opening/special-phase path can never reintroduce the
        // full-legal fallback that vcf_pair_complete is designed to exclude.
        if self.width.vcf_pair_complete && !implicit_dispatch && !self.width.consumes_ranked_zone()
        {
            return None;
        }

        // v1 Group-2 FHW forcing gate (design §3.3 / §5.3): at an eligible
        // FORCED (implicit-dispatch) defender node, run the structural FHW
        // closure, prove one representative subtree per FC-cover class, and emit
        // a reduced `FhwGateV1` in place of the full forced-reply Universal. Any
        // failure or Unknown child falls through to the unchanged implicit-
        // dispatch path below; the finalize-boundary self-verify (never emit a
        // cert the strict verifier rejects) plus the flag-off re-solve keep
        // "flag-on never decides fewer" structural.
        if self.group2
            && implicit_dispatch
            && !self.emitted_dirty
            && !matches!(state.phase(), TurnPhase::Opening)
        {
            if let Some(node) = self.prove_universal_fhw_gate(state, claimant, ply) {
                return Some(node);
            }
        }

        // v1 Group-2 selector (design §2.4, gate-free sub-class): at an
        // eligible unforced node, run the exact append-only FHW closure and
        // emit `UniversalGroup2V1`. Any failure falls through to the
        // unchanged legacy paths below; children proven during the attempt
        // stay memoized in the local TT, so the fallback re-proves them at
        // hit cost.
        if self.group2
            && !implicit_dispatch
            && !self.emitted_dirty
            && (self.zone.enabled || self.width.consumes_ranked_zone())
            && !matches!(state.phase(), TurnPhase::Opening)
            && group2_finder_preconditions(state, claimant, analysis)
        {
            if let Some(node) = self.prove_universal_group2(state, claimant, ply) {
                return Some(node);
            }
        }

        // At the proved L1 boundary U3 lets the verifier theorem-dismiss the
        // complement without enumerating it.  At spare nodes the default-off
        // U1 generator is consumable only because U2 re-derives the zone.
        let zone = (!implicit_dispatch
            && (self.zone.enabled || self.width.consumes_ranked_zone())
            && !matches!(state.phase(), TurnPhase::Opening))
        .then(|| {
            remaining_defender_placements_for_horizon(state, claimant, self.semantic_horizon).map(
                |d| ZoneInfo {
                    d,
                    build_horizon: self.semantic_horizon,
                },
            )
        })
        .flatten();
        let mut explicit = if implicit_dispatch {
            forced_defender_replies(state, claimant, analysis.b, self.width)
        } else if let Some(zone) = zone {
            zone_initial_candidates(state, claimant, zone.d, self.zone)
        } else {
            let mut all_legal = Vec::new();
            state.write_legal_moves(&mut all_legal);
            if all_legal.is_empty() {
                return None;
            }
            all_legal
        };
        if !implicit_dispatch && zone.is_none() {
            if let Some(pair) = pair {
                restrict_pair_candidates(&mut explicit, pair);
            }
        }

        if implicit_dispatch {
            let frame = canonical_frame(state);
            explicit.sort_by_key(|coord| canonical_coord_key(frame, *coord));
        } else {
            // At spare-budget nodes every legal move remains explicit, but
            // likely defenses are searched first so a refutation can stop the
            // lazy child loop before distant quiet moves are materialized.
            let hitting = hitting_universe(state, claimant);
            let frame = canonical_frame(state);
            explicit.sort_by_key(|coord| {
                let hits = hitting.contains(coord);
                (!hits, canonical_coord_key(frame, *coord))
            });
        }
        if explicit.is_empty() {
            return None;
        }

        let turn_start_legal = ((self.zone.pair_commutation
            || (self.width.vcf_pair_complete && implicit_dispatch))
            && pair.is_none()
            && matches!(state.phase(), TurnPhase::FirstStone)
            && threats::placements_remaining(state) == 2)
            .then(|| {
                let mut legal = Vec::new();
                state.write_legal_moves(&mut legal);
                legal.sort_by_key(|coord| raw_coord_key(*coord));
                legal
            });
        let mut edges = Vec::with_capacity(explicit.len());
        for &mv in &explicit {
            let Ok((result, delta)) = state.apply_with_delta(Placement { coord: mv }) else {
                return None;
            };
            let pair_context = turn_start_legal.as_ref().and_then(|legal| {
                (result.outcome.is_none() && matches!(state.phase(), TurnPhase::SecondStone { .. }))
                    .then(|| PairContext {
                        first: mv,
                        turn_start_legal: legal.clone(),
                    })
            });
            let child = self.prove(state, claimant, ply.checked_add(1)?, pair_context.as_ref());
            state.undo(delta);
            let child = child?; // Unknown poisons the universal claim.
            edges.push(CertEdge { mv, child });
        }

        if let Some(zone) = zone {
            loop {
                let required =
                    zone_certificate_extras(state, claimant, zone.d, &edges, &self.arena)?;
                let mut added = required
                    .into_iter()
                    .filter(|mv| !explicit.contains(mv))
                    .collect::<Vec<_>>();
                if added.is_empty() {
                    break;
                }
                let frame = canonical_frame(state);
                added.sort_by_key(|coord| canonical_coord_key(frame, *coord));
                for mv in added {
                    let Ok((_result, delta)) = state.apply_with_delta(Placement { coord: mv })
                    else {
                        return None;
                    };
                    let child = self.prove(state, claimant, ply.checked_add(1)?, None);
                    state.undo(delta);
                    let child = child?;
                    explicit.push(mv);
                    edges.push(CertEdge { mv, child });
                }
            }
        }

        let commutations = turn_start_legal
            .as_ref()
            .map(|legal| pair_commutations(legal, &edges, &self.arena))
            .unwrap_or_default();
        let explicit_edge_count = edges.len();

        self.alloc_node(
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            },
            explicit_edge_count,
        )
    }

    /// G2-Z1 append-only closure with the exact §3.4 required set: seed with
    /// the current hitting universe (or the least legal cell), prove children,
    /// recompute `Required_FHW` against the frozen children, and repeat until
    /// the explicit set covers it. Emits a placeholder-proof
    /// `UniversalGroup2V1`; scalars and digests are filled by
    /// `finder_finalize_group2` after compaction.
    fn prove_universal_group2(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
    ) -> Option<CertNodeId> {
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        legal.sort_by_key(|coord| raw_coord_key(*coord));
        if legal.is_empty() {
            return None;
        }
        let in_legal = |mv: HexCoord| {
            legal
                .binary_search_by_key(&raw_coord_key(mv), |c| raw_coord_key(*c))
                .is_ok()
        };
        let mut queue: Vec<HexCoord> = hitting_universe(state, claimant)
            .into_iter()
            .filter(|mv| in_legal(*mv))
            .collect();
        queue.sort_by_key(|coord| raw_coord_key(*coord));
        queue.dedup();
        if queue.is_empty() {
            queue.push(legal[0]);
        }
        let mut edges: Vec<CertEdge> = Vec::new();
        let mut proven: Vec<HexCoord> = Vec::new();
        // The required set is monotone in the frozen child set and bounded by
        // the finite legal set, so this loop terminates.
        loop {
            for mv in std::mem::take(&mut queue) {
                let Ok((result, delta)) = state.apply_with_delta(Placement { coord: mv }) else {
                    return None;
                };
                if result.outcome.is_some() {
                    state.undo(delta);
                    return None;
                }
                let child = self.prove(state, claimant, ply.checked_add(1)?, None);
                state.undo(delta);
                let child = child?;
                edges.push(CertEdge { mv, child });
                proven.push(mv);
            }
            let pairs: Vec<(HexCoord, CertNodeId)> =
                edges.iter().map(|edge| (edge.mv, edge.child)).collect();
            let required = crate::tss_verify_group2::finder_required_fhw(
                state,
                claimant,
                &pairs,
                &self.arena,
            )?;
            let mut missing: Vec<HexCoord> = required
                .into_iter()
                .filter(|mv| in_legal(*mv) && !proven.contains(mv))
                .collect();
            if missing.is_empty() {
                break;
            }
            missing.sort_by_key(|coord| raw_coord_key(*coord));
            missing.dedup();
            queue = missing;
        }
        edges.sort_by_key(|edge| raw_coord_key(edge.mv));
        let edge_count = edges.len();
        self.alloc_node(
            CertNode::UniversalGroup2V1(Box::new(crate::tss_verify::UniversalGroup2NodeV1 {
                edges,
                proof: crate::tss_verify::Group2ZoneV1 {
                    schema_version: 1,
                    authority: crate::tss_verify::Group2AuthorityV1::compiled(),
                    claimed_d14_budget: 0,
                    build_horizon: 0,
                    child_plan_sha256: [0u8; 32],
                    finder_summary_sha256: [0u8; 32],
                },
            })),
            edge_count,
        )
    }

    /// Structural FHW forcing-gate emission at a forced defender node. Builds
    /// the gate skeleton (H_Q, K, R ⊊ K via FC-cover, classified map), proves
    /// exactly one representative subtree per FC-cover class, and emits an
    /// `FhwGateV1` whose map carries EMPTY role/window rows (the finalizer fills
    /// them post-compaction, then a strict self-verify gates consumption). The
    /// fanout reduction is `|R|` proven children in place of `|K|` forced
    /// replies. Returns `None` (⇒ legacy fallback) on any closure failure,
    /// disabled edge class, or Unknown child.
    fn prove_universal_fhw_gate(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
    ) -> Option<CertNodeId> {
        // FrontierCovered emission is enabled: A1 is discharged end-to-end by
        // `fc_gate_certificate_reductive_reconstructs_and_verifies` (a positive
        // `R ⊊ K` FC certificate + its 12 D6 images verify). Exact-only gates
        // give ~0 fanout reduction, so FC is where the reduction lives.
        const GROUP2_FHW_ALLOW_FC: bool = true;

        let skel = crate::tss_verify_group2::finder_build_fhw_gate(
            state,
            claimant,
            self.semantic_horizon,
            GROUP2_FHW_ALLOW_FC,
        )?;
        // Prove each representative subtree; any Unknown poisons the gate.
        let mut rep_edges: Vec<CertEdge> = Vec::with_capacity(skel.representatives.len());
        for &s in &skel.representatives {
            let Ok((result, delta)) = state.apply_with_delta(Placement { coord: s }) else {
                return None;
            };
            if result.outcome.is_some() {
                state.undo(delta);
                return None;
            }
            let child = self.prove(state, claimant, ply.checked_add(1)?, None);
            state.undo(delta);
            let child = child?;
            rep_edges.push(CertEdge { mv: s, child });
        }
        rep_edges.sort_by_key(|edge| raw_coord_key(edge.mv));
        let map: Vec<crate::tss_verify::FhwMapV1> = skel
            .map
            .iter()
            .map(|(d, s, cls)| crate::tss_verify::FhwMapV1 {
                real_reply: *d,
                representative: *s,
                edge_class: *cls,
                roles: Vec::new(),
                windows: Vec::new(),
            })
            .collect();
        let edge_count = rep_edges.len();
        self.alloc_node(
            CertNode::FhwGateV1(Box::new(crate::tss_verify::FhwGateNodeV1 {
                representatives: rep_edges,
                proof: crate::tss_verify::FhwGateProofV1 {
                    schema_version: 1,
                    authority: crate::tss_verify::Group2AuthorityV1::compiled(),
                    threats: skel.threats,
                    escape_resolution_ply: skel.escape_resolution_ply,
                    map,
                },
            })),
            edge_count,
        )
    }

    fn alloc_node(&mut self, node: CertNode, added_edges: usize) -> Option<CertNodeId> {
        if self.arena.len() >= MAX_CERT_NODES
            || self.edge_count.saturating_add(added_edges) > MAX_CERT_EDGES
        {
            self.hit_limit = true;
            return None;
        }
        if let CertNode::Universal {
            implicit_dispatch,
            zone,
            commutations,
            ..
        } = &node
        {
            if *implicit_dispatch || zone.is_some() || !commutations.is_empty() {
                self.emitted_dirty = true;
            }
        }
        let id = u32::try_from(self.arena.len()).ok()?;
        self.arena.push(node);
        self.edge_count += added_edges;
        Some(id)
    }

    fn lookup_shared(
        &mut self,
        key: &PositionKey,
        claimant: Player,
        depth: usize,
    ) -> Option<CertNodeId> {
        let proof = self.shared_tt.as_ref()?.lookup_cloned(key, claimant)?;
        self.import_cached_proof(proof, depth)
    }

    fn remember_proof(&mut self, key: PositionKey, claimant: Player, node: CertNodeId) {
        self.tt.insert(key.clone(), claimant, node);
        // Persistent promotion is aimed at reusable forcing structure.  A
        // factual leaf is cheaper to re-establish than to compact, allocate,
        // and retain, while every non-leaf fragment still owns its leaves.
        // The solve root is offered separately after final compaction.
        let promotes_structure = matches!(
            self.arena.get(node as usize),
            Some(CertNode::Choice { .. } | CertNode::Universal { .. })
        );
        if promotes_structure
            && self
                .shared_tt
                .as_ref()
                .is_some_and(|shared| shared.could_admit_minimal(&key))
        {
            if let Some(proof) = CachedProof::from_arena_limited(
                &self.arena,
                node,
                MAX_PROMOTED_FRAGMENT_NODES,
                MAX_PROMOTED_FRAGMENT_EDGES,
            ) {
                self.insert_shared(key, claimant, proof);
            }
        }
        self.observe_tt_bytes();
    }

    fn insert_shared(&mut self, key: PositionKey, claimant: Player, proof: CachedProof) {
        if let Some(shared) = self.shared_tt.as_deref_mut() {
            shared.insert(key, claimant, proof);
        }
        self.observe_tt_bytes();
    }

    fn can_admit_compact(&self, key: &PositionKey, nodes: &[CertNode]) -> bool {
        self.shared_tt
            .as_ref()
            .is_some_and(|shared| shared.could_admit_compact(key, nodes))
    }

    /// Import is atomic with respect to the live arena: every structural and
    /// resource check happens against the owned clone before any node is
    /// appended.  A fragment that does not fit is merely a cache miss.
    fn import_cached_proof(&mut self, mut proof: CachedProof, depth: usize) -> Option<CertNodeId> {
        proof.validate()?;
        if proof.resolution_t > self.semantic_horizon
            || proof
                .zone_build_t
                .is_some_and(|build_t| self.semantic_horizon > build_t)
            || depth.checked_add(proof.height)? > MAX_SEARCH_DEPTH
            || self.arena.len().checked_add(proof.nodes.len())? > MAX_CERT_NODES
            || self.edge_count.checked_add(proof.explicit_edges)? > MAX_CERT_EDGES
        {
            return None;
        }

        let base = self.arena.len();
        let final_len = base.checked_add(proof.nodes.len())?;
        u32::try_from(final_len).ok()?;
        for node in &mut proof.nodes {
            remap_node_ids_with_offset(node, base, final_len)?;
        }
        let root = offset_node_id(proof.root_node, base, final_len)?;
        self.arena.append(&mut proof.nodes);
        self.edge_count += proof.explicit_edges;
        Some(root)
    }

    fn observe_tt_bytes(&mut self) {
        let shared = self
            .shared_tt
            .as_ref()
            .map(|cache| cache.current_bytes)
            .unwrap_or(0);
        self.peak_tt_bytes = self
            .peak_tt_bytes
            .max(self.tt.current_bytes.saturating_add(shared));
    }
}

fn arena_subtree_contains_zone(arena: &[CertNode], root: CertNodeId) -> bool {
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.get(id as usize) {
            Some(CertNode::Choice { child, .. }) => stack.push(*child),
            Some(CertNode::Universal { edges, zone, .. }) => {
                if zone.is_some() {
                    return true;
                }
                stack.extend(edges.iter().map(|edge| edge.child));
            }
            Some(_) => {}
            None => return true,
        }
    }
    false
}

fn pair_commutations(
    turn_start_legal: &[HexCoord],
    parent_edges: &[CertEdge],
    arena: &[CertNode],
) -> Vec<CertCommutation> {
    let mut result = Vec::new();
    for first_edge in parent_edges {
        let Some(CertNode::Universal {
            edges: first_replies,
            ..
        }) = arena.get(first_edge.child as usize)
        else {
            continue;
        };
        for &omitted_second in turn_start_legal {
            if raw_coord_key(omitted_second) >= raw_coord_key(first_edge.mv)
                || first_replies.iter().any(|edge| edge.mv == omitted_second)
            {
                continue;
            }
            let Some(mirror_edge) = parent_edges.iter().find(|edge| edge.mv == omitted_second)
            else {
                continue;
            };
            let Some(CertNode::Universal {
                edges: mirror_replies,
                ..
            }) = arena.get(mirror_edge.child as usize)
            else {
                continue;
            };
            if mirror_replies.iter().any(|edge| edge.mv == first_edge.mv) {
                result.push(CertCommutation {
                    first: first_edge.mv,
                    omitted_second,
                    first_child: first_edge.child,
                    mirror_child: mirror_edge.child,
                });
            }
        }
    }
    result.sort_by_key(|item| {
        (
            raw_coord_key(item.first),
            raw_coord_key(item.omitted_second),
        )
    });
    result
}

fn restrict_pair_candidates(candidates: &mut Vec<HexCoord>, pair: &PairContext) {
    candidates.retain(|mv| pair_candidate_allowed(*mv, pair));
}

fn pair_candidate_allowed(mv: HexCoord, pair: &PairContext) -> bool {
    raw_coord_key(mv) > raw_coord_key(pair.first) || !pair.turn_start_legal.contains(&mv)
}

/// Reconstruct membership in the pair-complete attacker universe immediately
/// before `first` was placed.  This lets the proof-number search canonicalize
/// ordinary `(a,b)/(b,a)` pairs without pruning a second coordinate that only
/// became a count-two candidate after `first`.
fn wide_candidate_was_legal_before_first(
    state: &RustHexoState,
    claimant: Player,
    first: HexCoord,
    candidate: HexCoord,
) -> bool {
    debug_assert_eq!(state.board().get(first), Some(claimant));
    debug_assert_eq!(state.board().get(candidate), None);
    for axis in Axis::ALL {
        for offset in 0..6i16 {
            let key = WindowKey {
                start: candidate - axis.vector().scale(offset),
                axis,
            };
            let Some(entry) = state.board().windows().entry(key) else {
                continue;
            };
            let first_in_window = key.contains(first);
            let prior_claimant = entry
                .count(claimant)
                .saturating_sub(u8::from(first_in_window));
            let prior_defender = entry.count(claimant.other());
            if (prior_defender == 0 && prior_claimant >= 2)
                || (prior_claimant == 0 && prior_defender >= 4)
            {
                return true;
            }
        }
    }
    false
}

/// True when the just-completed pair created a claimant count-four-or-stronger
/// window that was not already a threat at turn start.  Any changed window is
/// incident to one of the two placements, so at most 36 O(1) store lookups are
/// needed.  Subtracting both stones reconstructs the pre-turn count exactly.
fn turn_created_claimant_threat(
    state: &RustHexoState,
    claimant: Player,
    first: HexCoord,
    second: HexCoord,
) -> bool {
    let mut inspected = Vec::with_capacity(36);
    for placed in [first, second] {
        for axis in Axis::ALL {
            for offset in 0..6i16 {
                let key = WindowKey {
                    start: placed - axis.vector().scale(offset),
                    axis,
                };
                if inspected.contains(&key) {
                    continue;
                }
                inspected.push(key);
                let Some(entry) = state.board().windows().entry(key) else {
                    continue;
                };
                if entry.active_player() != Some(claimant) || entry.count(claimant) < 4 {
                    continue;
                }
                let prior_count = entry
                    .count(claimant)
                    .saturating_sub(u8::from(key.contains(first)))
                    .saturating_sub(u8::from(key.contains(second)));
                if prior_count < 4 {
                    return true;
                }
            }
        }
    }
    false
}

/// The addendum's forcing discipline requires more than merely leaving a live
/// threat: every ensuing defender placement must stay in the small, verifier-
/// justified hitting dispatcher.  A tactical claimant leaf is already done;
/// otherwise this is exactly the dispatch boundary used by `prove_universal`.
/// Rejecting looser turns only narrows the WIN search.
fn turn_forces_small_defender_reply(state: &RustHexoState, claimant: Player) -> bool {
    let analysis = threats::analyze(state);
    winner_from_analysis(state, &analysis) == Some(claimant)
        || (!matches!(state.phase(), TurnPhase::Opening)
            && analysis.opp_threat_count > 0
            && !analysis.own_win_now
            && analysis.min_hitting_set == Some(analysis.b))
}

#[derive(Clone)]
struct Candidate {
    coord: HexCoord,
    /// Maximum pre-placement stone count among active claimant windows that
    /// this coordinate extends.  Larger means a lower initial proof number.
    strength: u8,
    priority_class: u8,
    child_threats: usize,
    /// This move occupies an empty of an active defender count-four/five
    /// window.  Wide mode must retain such tempo-preserving blocks even when
    /// the cell is not yet in a claimant-owned window.
    defender_block: bool,
    /// Distinct count-two windows through this cell.  In pair-complete mode
    /// this is the primary ordering key within the newly admitted tier.
    pair_start_degree: usize,
    /// Nearest claimant stone, used only to break widened-tier ordering ties.
    own_proximity: i16,
    /// Count-three claimant windows this placement turns into live threats.
    /// Their pre-placement empties let SecondStone reply forcedness be derived
    /// without rescanning or mutating the engine state.
    created_threats: Vec<Vec<HexCoord>>,
}

/// The established wide ordering has exactly two attacker-width classes:
/// narrow candidates (including mandatory defender blocks) and count-two-only
/// pair builds.  Keep this derivation shared by generation and the root-tier
/// selector so the prior cannot invent a third classification.
fn wide_candidate_width_tier(candidate: &Candidate) -> u8 {
    match (candidate.defender_block, candidate.strength) {
        (true, _) | (_, 3..) => 0,
        (_, 2) => 1,
        _ => unreachable!("wide candidates are defender blocks or count>=2"),
    }
}

struct CandidateBatch {
    candidates: Vec<Candidate>,
    claimant_threats: Vec<Vec<HexCoord>>,
    defender_threats: Vec<Vec<HexCoord>>,
}

fn threat_creating_moves_with_threshold(
    state: &RustHexoState,
    claimant: Player,
    minimum_strength: u8,
) -> CandidateBatch {
    assert!(
        minimum_strength >= 2,
        "count-one/r3 attacker width is not supported"
    );
    let mut candidates: Vec<Candidate> = Vec::new();
    // Coordinate-keyed dedup index. Aggregation per encounter is identical to
    // the previous linear `find`; only the lookup cost changes (the final
    // deterministic sort below fixes the output order either way).
    let mut candidate_index: HashMap<HexCoord, usize> = HashMap::new();
    let mut claimant_threats = Vec::new();
    let mut defender_threats = Vec::new();
    for entry in state.board().windows().entries() {
        let Some(owner) = entry.active_player() else {
            continue;
        };
        if entry.count(owner) >= 4 {
            if owner == claimant {
                claimant_threats.push(entry.empty_cells());
            } else {
                defender_threats.push(entry.empty_cells());
            }
        }
        if owner != claimant {
            continue;
        }
        let strength = entry.count(claimant);
        if strength < minimum_strength {
            continue;
        }
        let empties = entry.empty_cells();
        for &coord in &empties {
            let created = (strength == 3).then(|| {
                empties
                    .iter()
                    .copied()
                    .filter(|empty| *empty != coord)
                    .collect::<Vec<_>>()
            });
            if let Some(&slot) = candidate_index.get(&coord) {
                let existing = &mut candidates[slot];
                existing.strength = existing.strength.max(strength);
                if strength == 2 {
                    existing.pair_start_degree += 1;
                }
                if let Some(created) = created {
                    existing.created_threats.push(created);
                }
            } else {
                candidate_index.insert(coord, candidates.len());
                candidates.push(Candidate {
                    coord,
                    strength,
                    priority_class: u8::MAX,
                    child_threats: 0,
                    defender_block: false,
                    pair_start_degree: usize::from(strength == 2),
                    own_proximity: i16::MAX,
                    created_threats: created.into_iter().collect(),
                });
            }
        }
    }
    if minimum_strength == 2 {
        for coord in defender_threats.iter().flatten().copied() {
            if let Some(&slot) = candidate_index.get(&coord) {
                candidates[slot].defender_block = true;
            } else {
                candidate_index.insert(coord, candidates.len());
                candidates.push(Candidate {
                    coord,
                    strength: 0,
                    priority_class: u8::MAX,
                    child_threats: 0,
                    defender_block: true,
                    pair_start_degree: 0,
                    own_proximity: i16::MAX,
                    created_threats: Vec::new(),
                });
            }
        }
    }
    candidates.sort_by_key(|item| (Reverse(item.strength), item.coord.q, item.coord.r));
    CandidateBatch {
        candidates,
        claimant_threats,
        defender_threats,
    }
}

/// Static proof-number initialization derived from WindowStore membership.
/// The candidate set is unchanged.  A count-four extension is an immediate
/// lambda-one proof after a one-stone remainder; otherwise same-turn builds
/// precede replies, and newly created threat-window count orders each class.
fn ordered_threat_creating_moves(state: &RustHexoState, claimant: Player) -> Vec<Candidate> {
    ordered_threat_creating_moves_with_width(state, claimant, WidthOptions::default())
}

fn ordered_threat_creating_moves_with_width(
    state: &RustHexoState,
    claimant: Player,
    width: WidthOptions,
) -> Vec<Candidate> {
    let CandidateBatch {
        mut candidates,
        claimant_threats,
        defender_threats,
    } = if width.vcf_pair_complete {
        threat_creating_moves_with_threshold(state, claimant, 2)
    } else {
        threat_creating_moves_with_threshold(state, claimant, 3)
    };
    // Hoisted once per generation: the claimant stone list only depends on the
    // position, not on the candidate being ranked.
    let claimant_stones: Vec<HexCoord> = if width.vcf_pair_complete {
        state
            .board()
            .occupied_cells()
            .iter()
            .copied()
            .filter(|coord| state.board().get(*coord) == Some(claimant))
            .collect()
    } else {
        Vec::new()
    };
    for candidate in &mut candidates {
        candidate.child_threats = claimant_threats.len() + candidate.created_threats.len();
        if width.vcf_pair_complete && candidate.strength <= 2 {
            candidate.own_proximity = claimant_stones
                .iter()
                .map(|&coord| hex_distance(candidate.coord, coord))
                .min()
                .unwrap_or(i16::MAX);
        }
        candidate.priority_class = if candidate.defender_block && candidate.strength < 4 {
            match state.phase() {
                TurnPhase::FirstStone => 1,
                TurnPhase::SecondStone { .. } => 2,
                TurnPhase::Opening => 3,
            }
        } else {
            match state.phase() {
                TurnPhase::FirstStone if candidate.strength >= 4 => 0,
                TurnPhase::FirstStone => 2,
                TurnPhase::SecondStone { .. } => {
                    post_turn_reply_priority(candidate, &claimant_threats, &defender_threats)
                }
                TurnPhase::Opening => 3,
            }
        };
    }
    if candidates.len() <= 1 {
        return candidates;
    }
    let frame = canonical_frame(state);
    if width.vcf_pair_complete {
        candidates.sort_by_key(|item| {
            let width_tier = wide_candidate_width_tier(item);
            let canonical = canonical_coord_key(frame, item.coord);
            (
                width_tier,
                if width_tier == 0 {
                    item.priority_class
                } else {
                    0
                },
                Reverse(if width_tier == 0 {
                    item.child_threats
                } else {
                    item.pair_start_degree
                }),
                Reverse(if width_tier == 0 { item.strength } else { 0 }),
                if width_tier == 0 && matches!(state.phase(), TurnPhase::SecondStone { .. }) {
                    item.pair_start_degree
                } else {
                    0
                },
                if width_tier == 0 {
                    0
                } else {
                    item.own_proximity
                },
                canonical.0,
                canonical.1,
            )
        });
    } else {
        candidates.sort_by_key(|item| {
            let canonical = canonical_coord_key(frame, item.coord);
            (
                item.priority_class,
                Reverse(item.child_threats),
                Reverse(item.strength),
                canonical.0,
                canonical.1,
            )
        });
    }
    candidates
}

/// One claimant-pure count>=2 window as seen from a candidate empty cell:
/// the immutable turn-start facts needed to evaluate any pair through it.
#[derive(Clone)]
struct WidePairWindow {
    key: WindowKey,
    strength: u8,
    empties: Vec<HexCoord>,
}

/// Turn-start snapshot for stateless wide pair classification. Valid only at
/// a claimant FirstStone Choice node, where `expand` has already proven that
/// no live claimant >=4 window exists (such nodes become win-now leaves
/// before any pair generation). Consequences used throughout: no pair can
/// complete six this turn (a window would need >=4 prior stones), and the
/// defender's post-pair threat family is exactly the windows through the two
/// placed stones that reach count >=4.
struct WideTurnGate {
    /// For each empty cell: the claimant-pure count>=2 windows holding it.
    windows_by_cell: HashMap<HexCoord, Vec<WidePairWindow>>,
    /// For each empty cell: the claimant-pure count-1 windows holding it.
    /// After a first stone in such a window its other empties join the
    /// count>=2 second-ply universe; stored separately so pair evaluation
    /// never scans them.
    weak_windows_by_cell: HashMap<HexCoord, Vec<WidePairWindow>>,
    /// Empties of every live defender >=4 window (the hit/block sets).
    defender_threats: Vec<Vec<HexCoord>>,
    #[cfg(test)]
    live_claimant_windows: Vec<WidePairWindow>,
    /// `placements_made` at turn start.
    start_placements: u32,
}

impl WideTurnGate {
    fn build(state: &RustHexoState, claimant: Player) -> Self {
        let mut windows_by_cell: HashMap<HexCoord, Vec<WidePairWindow>> = HashMap::new();
        let mut weak_windows_by_cell: HashMap<HexCoord, Vec<WidePairWindow>> = HashMap::new();
        let mut defender_threats = Vec::new();
        #[cfg(test)]
        let mut live_claimant_windows = Vec::new();
        for entry in state.board().windows().entries() {
            let Some(owner) = entry.active_player() else {
                continue;
            };
            let count = entry.count(owner);
            if owner == claimant {
                if count >= 1 {
                    let empties = entry.empty_cells();
                    let window = WidePairWindow {
                        key: entry.key(),
                        strength: count,
                        empties: empties.clone(),
                    };
                    #[cfg(test)]
                    live_claimant_windows.push(window.clone());
                    let sink = if count >= 2 {
                        &mut windows_by_cell
                    } else {
                        &mut weak_windows_by_cell
                    };
                    for &cell in &empties {
                        sink.entry(cell).or_default().push(window.clone());
                    }
                }
            } else if count >= 4 {
                defender_threats.push(entry.empty_cells());
            }
        }
        Self {
            windows_by_cell,
            weak_windows_by_cell,
            defender_threats,
            #[cfg(test)]
            live_claimant_windows,
            start_placements: state.placements_made(),
        }
    }

    #[cfg(test)]
    fn live_ge3_after_pair(&self, first: HexCoord, second: HexCoord) -> usize {
        self.live_claimant_windows
            .iter()
            .filter(|window| {
                window.strength
                    + u8::from(window.empties.contains(&first))
                    + u8::from(window.empties.contains(&second))
                    >= 3
            })
            .count()
    }

    /// The second-ply candidate coordinates after the claimant plays `first`,
    /// derived without touching the engine: the strongest continuations are
    /// the other empties of the count>=2 windows through `first` (they join
    /// the tight forcing tier — the round-2 width-sorter property), then the
    /// turn-start candidate list, then the empties of count-1 windows through
    /// `first` (which reach the count-2 build tier only via `first`). This is
    /// a slight SUPERSET of the historical post-apply regeneration (cells
    /// whose defender-block status died with `first` are retained); wider is
    /// WIN-sound and the forcing gate discards non-forcing pairs anyway.
    fn second_candidates(
        &self,
        first: HexCoord,
        turn_start: &[Candidate],
        out: &mut Vec<HexCoord>,
        seen: &mut HashSet<HexCoord>,
    ) {
        out.clear();
        seen.clear();
        seen.insert(first);
        if let Some(list) = self.windows_by_cell.get(&first) {
            let mut promoted: Vec<(u8, HexCoord)> = Vec::new();
            for window in list {
                for &cell in &window.empties {
                    if cell != first {
                        promoted.push((window.strength, cell));
                    }
                }
            }
            // Strongest promotions first; deterministic within a strength
            // class by raw coordinate order.
            promoted.sort_by_key(|&(strength, cell)| (Reverse(strength), raw_coord_key(cell)));
            for (_, cell) in promoted {
                if seen.insert(cell) {
                    out.push(cell);
                }
            }
        }
        for candidate in turn_start {
            if seen.insert(candidate.coord) {
                out.push(candidate.coord);
            }
        }
        if let Some(list) = self.weak_windows_by_cell.get(&first) {
            let mut fresh: Vec<HexCoord> = Vec::new();
            for window in list {
                for &cell in &window.empties {
                    if !seen.contains(&cell) {
                        fresh.push(cell);
                    }
                }
            }
            fresh.sort_by_key(|&cell| raw_coord_key(cell));
            for cell in fresh {
                if seen.insert(cell) {
                    out.push(cell);
                }
            }
        }
    }

    /// Classify the attacker turn (first, second) exactly as the reference
    /// apply-and-analyze path did, without touching the engine state:
    ///
    /// - `None`: the turn creates no claimant >=4 window, or the defender
    ///   keeps a win-now/spare-budget reply — the forcing discipline prunes
    ///   it (`turn_created_claimant_threat` / `turn_forces_small_defender_
    ///   reply` both replicated below).
    /// - `ClaimantTactical`: the defender is 1-ply forced-lost and the sparse
    ///   LOSS leaf materializes within the semantic horizon
    ///   (`immediate_winner` + `typed_lambda_leaf` equivalents).
    /// - `Pending` + prior: a forcing turn searched normally, with the
    ///   `completed_turn_prior` numbers derived from the same family.
    fn evaluate_pair(
        &self,
        first: HexCoord,
        second: HexCoord,
        semantic_horizon: u32,
    ) -> Option<(WidePnChildResult, WidePnPrior)> {
        // The claimant windows reaching >=4 once the pair is placed, with
        // their post-pair empties. A window through both stones is collected
        // once (from the `first` list; the `second` pass skips it).
        let mut family: Vec<(WindowKey, Vec<HexCoord>)> = Vec::new();
        if let Some(list) = self.windows_by_cell.get(&first) {
            for window in list {
                let joint = window.empties.contains(&second);
                if window.strength + 1 + u8::from(joint) >= 4 {
                    family.push((
                        window.key,
                        window
                            .empties
                            .iter()
                            .copied()
                            .filter(|&cell| cell != first && cell != second)
                            .collect(),
                    ));
                }
            }
        }
        if let Some(list) = self.windows_by_cell.get(&second) {
            for window in list {
                if window.empties.contains(&first) {
                    continue;
                }
                if window.strength + 1 >= 4 {
                    family.push((
                        window.key,
                        window
                            .empties
                            .iter()
                            .copied()
                            .filter(|&cell| cell != first && cell != second)
                            .collect(),
                    ));
                }
            }
        }
        if family.is_empty() {
            return None;
        }
        // Post-pair defender analysis at FirstStone (B = 2): any unhit live
        // defender >=4 window is win-now; the claimant family is the entire
        // opponent threat set.
        let defender_win_now = self
            .defender_threats
            .iter()
            .any(|set| !set.contains(&first) && !set.contains(&second));
        if defender_win_now {
            return None;
        }
        let mhs = wide_family_min_hitting_set(&family);
        let threat_count = family.len();
        if mhs.is_none() {
            // Defender 1-ply forced-lost: ClaimantTactical iff the sparse
            // LOSS leaf materializes (inclusion-minimal obstruction within
            // the L13 cap) inside the horizon; otherwise the turn is still
            // forcing and searched as Pending — both exactly the reference
            // classifier's branches.
            family.sort_by_key(|(key, _)| window_key_order(*key));
            let full_sets = family
                .iter()
                .map(|(_, empties)| empties.clone())
                .collect::<Vec<_>>();
            let resolution = self.start_placements.saturating_add(6);
            if resolution <= semantic_horizon
                && inclusion_minimal_loss_obstruction(&full_sets, 2)
                    .is_some_and(|kept| !kept.is_empty())
            {
                return Some((WidePnChildResult::ClaimantTactical, WidePnPrior::UNIFORM));
            }
            return Some((
                WidePnChildResult::Pending,
                WidePnPrior {
                    pn: pn_from_fork_degree(threat_count),
                    dn: dn_from_tau(None),
                },
            ));
        }
        if mhs == Some(2) {
            return Some((
                WidePnChildResult::Pending,
                WidePnPrior {
                    pn: pn_from_fork_degree(threat_count),
                    dn: dn_from_tau(Some(2)),
                },
            ));
        }
        None
    }
}

/// Exact replica of the shared threat-analysis minimum hitting set at the
/// defender budget of two, over the stateless post-pair family.
fn wide_family_min_hitting_set(family: &[(WindowKey, Vec<HexCoord>)]) -> Option<u8> {
    if family.is_empty() {
        return Some(0);
    }
    if family.iter().any(|(_, set)| set.is_empty()) {
        return None;
    }
    let mut universe: Vec<HexCoord> = Vec::new();
    for (_, set) in family {
        for &cell in set {
            if !universe.contains(&cell) {
                universe.push(cell);
            }
        }
    }
    for &cell in &universe {
        if family.iter().all(|(_, set)| set.contains(&cell)) {
            return Some(1);
        }
    }
    for left in 0..universe.len() {
        for right in (left + 1)..universe.len() {
            let (x, y) = (universe[left], universe[right]);
            if family
                .iter()
                .all(|(_, set)| set.contains(&x) || set.contains(&y))
            {
                return Some(2);
            }
        }
    }
    None
}

/// Static fork potential for an unexpanded attacker OR node. Count-three
/// extensions contribute the live threats they expose immediately; count-two
/// pair starts contribute their distinct continuation windows. The best
/// available degree is sufficient for an OR prior and is independent of hash
/// iteration because only the maximum is retained.
///
/// Single window pass. For a candidate cell `x` the wide generator derives
/// `child_threats = T + c3(x)` (T = live claimant >=4 windows, c3 = claimant
/// count-3 windows holding `x` as an empty) and `pair_start_degree = c2(x)`,
/// so the maximum over candidates is `max(T, max_x max(T + c3(x), c2(x)))`
/// whenever any candidate exists: cells appearing only in count>=4 claimant
/// windows or only as defender blocks contribute exactly `T`. Building and
/// sorting the full ranked candidate list for one scalar was the dominant
/// cost of every attacker-node prior.
fn attacker_fork_degree(state: &RustHexoState, claimant: Player) -> usize {
    let mut threat_count = 0usize;
    let mut any_candidate = false;
    let mut degrees: HashMap<HexCoord, (usize, usize)> = HashMap::new();
    for entry in state.board().windows().entries() {
        let Some(owner) = entry.active_player() else {
            continue;
        };
        let count = entry.count(owner);
        if owner == claimant {
            if count >= 4 {
                threat_count += 1;
            }
            if count >= 2 {
                any_candidate = true;
            }
            if count == 3 {
                for cell in entry.empty_cells() {
                    degrees.entry(cell).or_default().0 += 1;
                }
            } else if count == 2 {
                for cell in entry.empty_cells() {
                    degrees.entry(cell).or_default().1 += 1;
                }
            }
        } else if count >= 4 {
            // Defender-threat empties are wide-mode block candidates even
            // when no claimant window holds them.
            any_candidate = true;
        }
    }
    if !any_candidate {
        return 0;
    }
    degrees
        .values()
        .map(|&(c3, c2)| (threat_count + c3).max(c2))
        .max()
        .unwrap_or(0)
        .max(threat_count)
}

/// Reconstruct exactly the child threat-cost class after a claimant's
/// SecondStone at a reachable unresolved search node, without mutating the
/// engine state.  (`prove` removes terminal/lambda-one parents first, so a
/// strength-five completion and its off-path ordering distinctions cannot
/// reach this function.)  Window masks change only in windows containing
/// `coord`: claimant windows gain one bit and defender windows become blocked.
/// The returned classes match the former child `analyze` probe: immediate
/// claimant proof (0), fully forced two-hit reply (1), or a reply with
/// spare/counter-winning budget (3).
fn post_turn_reply_priority(
    candidate: &Candidate,
    claimant_threats: &[Vec<HexCoord>],
    defender_threats: &[Vec<HexCoord>],
) -> u8 {
    // The child is FirstStone (B=2), so any defender count-four/five not
    // blocked by this placement is win-now.  Child analysis gives it
    // precedence over claimant threats.
    if defender_threats
        .iter()
        .any(|empties| !empties.contains(&candidate.coord))
    {
        return 3;
    }
    match min_hitting_set_at_most_two(
        claimant_threats,
        &candidate.created_threats,
        candidate.coord,
    ) {
        None => 0,
        Some(2) => 1,
        Some(_) => 3,
    }
}

fn min_hitting_set_at_most_two(
    existing: &[Vec<HexCoord>],
    created: &[Vec<HexCoord>],
    placed: HexCoord,
) -> Option<u8> {
    if existing.is_empty() && created.is_empty() {
        return Some(0);
    }
    let sets = || existing.iter().chain(created.iter());
    if sets().any(|set| !set.iter().any(|coord| *coord != placed)) {
        return None;
    }
    let mut universe = Vec::new();
    for set in sets() {
        for &coord in set {
            if coord != placed && !universe.contains(&coord) {
                universe.push(coord);
            }
        }
    }
    if universe
        .iter()
        .any(|coord| sets().all(|set| set.contains(coord)))
    {
        return Some(1);
    }
    for left in 0..universe.len() {
        for right in (left + 1)..universe.len() {
            if sets().all(|set| set.contains(&universe[left]) || set.contains(&universe[right])) {
                return Some(2);
            }
        }
    }
    None
}

/// Union of empties of every live claimant threat.  At a defender node this is
/// the L1 hitting-cell universe, not a selected minimal hitting set.
fn hitting_universe(state: &RustHexoState, claimant: Player) -> Vec<HexCoord> {
    let mut cells = Vec::new();
    for (owner, entry) in state.board().windows().live_threat_entries() {
        if owner == claimant {
            cells.extend(entry.empty_cells());
        }
    }
    cells.sort_by_key(|coord| (coord.q, coord.r));
    cells.dedup();
    cells
}

fn forced_defender_replies(
    state: &RustHexoState,
    claimant: Player,
    defender_budget: u8,
    width: WidthOptions,
) -> Vec<HexCoord> {
    if width.vcf_pair_complete {
        extendable_hit_kernel(state, claimant, defender_budget)
    } else {
        hitting_universe(state, claimant)
    }
}

/// Cells that can occur in a size-`budget` transversal of the claimant's live
/// threat family. At the forced boundary `tau == budget`, every omitted cell
/// leaves the defender without an extendable defense, so T6 permits the wide
/// WIN search to restrict its explicit universal replies to this kernel.
///
/// Connect-6 reaches this boundary only with budgets one and two. The fallback
/// deliberately returns the full hitting universe for any future budget so an
/// unsupported phase can lose performance but never lose a necessary reply.
fn extendable_hit_kernel(state: &RustHexoState, claimant: Player, budget: u8) -> Vec<HexCoord> {
    let family = state
        .board()
        .windows()
        .live_threat_entries()
        .filter_map(|(owner, entry)| (owner == claimant).then(|| entry.empty_cells()))
        .collect::<Vec<_>>();
    extendable_hit_kernel_for_family(&family, budget)
}

fn extendable_hit_kernel_for_family(family: &[Vec<HexCoord>], budget: u8) -> Vec<HexCoord> {
    let mut universe = family.iter().flatten().copied().collect::<Vec<_>>();
    universe.sort_by_key(|coord| (coord.q, coord.r));
    universe.dedup();
    match budget {
        1 => universe
            .into_iter()
            .filter(|cell| family.iter().all(|threat| threat.contains(cell)))
            .collect(),
        2 => universe
            .iter()
            .copied()
            .filter(|cell| {
                universe.iter().copied().any(|mate| {
                    mate != *cell
                        && family
                            .iter()
                            .all(|threat| threat.contains(cell) || threat.contains(&mate))
                })
            })
            .collect(),
        _ => universe,
    }
}

fn remaining_defender_placements_for_horizon(
    state: &RustHexoState,
    claimant: Player,
    horizon: u32,
) -> Option<u32> {
    // Zone machinery only distinguishes budgets 0..=5 (>= 6 takes the full
    // legal set) and production horizons sit ~12 plies out, so any count past
    // this band signals a corrupted/degenerate horizon — bail (None => no
    // zone, full legal set) instead of walking a `u32::MAX` horizon.
    const DEFENDER_BUDGET_BAIL: u32 = 8;
    let mut ply = state.placements_made();
    if horizon < ply {
        return None;
    }
    let mut player = state.current_player();
    let mut phase = state.phase();
    let mut count = 0u32;
    while ply < horizon {
        if player != claimant {
            count = count.checked_add(1)?;
            if count > DEFENDER_BUDGET_BAIL {
                return None;
            }
        }
        match phase {
            TurnPhase::Opening => {
                player = player.other();
                phase = TurnPhase::FirstStone;
            }
            TurnPhase::FirstStone => {
                phase = TurnPhase::SecondStone {
                    first: HexCoord::ZERO,
                }
            }
            TurnPhase::SecondStone { .. } => {
                player = player.other();
                phase = TurnPhase::FirstStone;
            }
        }
        ply = ply.checked_add(1)?;
    }
    Some(count)
}

fn all_incident_windows_two_coloured(state: &RustHexoState, cell: HexCoord) -> bool {
    for axis in Axis::ALL {
        for offset in 0..6i16 {
            let key = WindowKey {
                start: cell - axis.vector().scale(offset),
                axis,
            };
            let Some(entry) = state
                .board()
                .windows()
                .entries()
                .find(|entry| entry.key() == key)
            else {
                return false;
            };
            if entry.count(Player::Player0) == 0 || entry.count(Player::Player1) == 0 {
                return false;
            }
        }
    }
    true
}

/// Finder-side mirror of the verifier's Group-2 class-rule 4 preconditions
/// (§2.3): defender to move at a nonterminal post-opening node, b in {1,2},
/// no mover win-now (conservative direct window upper bound AND the shared
/// analysis), and the exactly reconstructed k < b. This is a pre-check only:
/// the emitted certificate is still strictly re-verified.
fn group2_finder_preconditions(
    state: &RustHexoState,
    claimant: Player,
    analysis: &threats::ThreatAnalysis,
) -> bool {
    if state.is_terminal() || state.current_player() == claimant || analysis.own_win_now {
        return false;
    }
    let b = threats::placements_remaining(state);
    if !(1..=2).contains(&b) {
        return false;
    }
    let mover = state.current_player();
    let direct_win_upper = state.board().windows().entries().any(|entry| {
        entry.count(claimant) == 0 && entry.count(mover).saturating_add(b) >= 6
    });
    if direct_win_upper {
        return false;
    }
    // Exact k: 0 iff the claimant-threat family is empty; 1 iff every member
    // shares a common cell; else >= 2 (rejecting at both accepted budgets).
    let defender = claimant.other();
    let mut family: Vec<Vec<HexCoord>> = Vec::new();
    for entry in state.board().windows().entries() {
        if entry.count(defender) == 0 && entry.count(claimant) >= 4 {
            let empties = entry.empty_cells();
            if empties.is_empty() {
                return false;
            }
            family.push(empties);
        }
    }
    let k: u8 = if family.is_empty() {
        0
    } else {
        let mut common = family[0].clone();
        for member in &family[1..] {
            common.retain(|cell| member.contains(cell));
            if common.is_empty() {
                break;
            }
        }
        if common.is_empty() {
            2
        } else {
            1
        }
    };
    k < b
}

fn zone_initial_candidates(
    state: &RustHexoState,
    claimant: Player,
    d: u32,
    options: ZoneSearchCaps,
) -> Vec<HexCoord> {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| (coord.q, coord.r));
    if d >= 6 {
        return legal;
    }
    let defender = claimant.other();
    let mut out = hitting_universe(state, claimant);
    for entry in state.board().windows().entries() {
        let attacker_term = entry.active_player() == Some(claimant)
            && entry.count(claimant) >= if options.count2_threshold { 2 } else { 1 };
        let defender_term = entry.active_player() == Some(defender)
            && u32::from(entry.count(defender)) >= 6u32.saturating_sub(d);
        if attacker_term || defender_term {
            out.extend(entry.empty_cells());
        }
    }
    out.sort_by_key(|coord| (coord.q, coord.r));
    out.dedup();
    out.retain(|cell| {
        legal
            .binary_search_by_key(&(cell.q, cell.r), |c| (c.q, c.r))
            .is_ok()
    });
    if options.stale_area_filter {
        out.retain(|cell| !all_incident_windows_two_coloured(state, *cell));
    }
    let hitting = hitting_universe(state, claimant);
    let frame = canonical_frame(state);
    out.sort_by_key(|coord| (!hitting.contains(coord), canonical_coord_key(frame, *coord)));
    out
}

fn arena_core(arena: &[CertNode], root: CertNodeId, out: &mut Vec<HexCoord>) -> Option<()> {
    match arena.get(root as usize)? {
        CertNode::OrCompletion { mv, witness, .. } => {
            out.push(*mv);
            out.extend(witness.cells());
        }
        CertNode::Win { witness, .. } => out.extend(witness.cells()),
        CertNode::Loss { witnesses, .. } => {
            for witness in witnesses {
                out.extend(witness.cells());
            }
        }
        CertNode::Choice { mv, child } => {
            out.push(*mv);
            arena_core(arena, *child, out)?;
        }
        CertNode::Universal { edges, .. } => {
            for edge in edges {
                arena_core(arena, edge.child, out)?;
            }
        }
        CertNode::UniversalGroup2V1(node) => {
            for edge in &node.edges {
                arena_core(arena, edge.child, out)?;
            }
        }
        CertNode::FhwGateV1(gate) => {
            for edge in &gate.representatives {
                arena_core(arena, edge.child, out)?;
            }
        }
    }
    Some(())
}

fn zone_certificate_extras(
    state: &RustHexoState,
    claimant: Player,
    d: u32,
    edges: &[CertEdge],
    arena: &[CertNode],
) -> Option<Vec<HexCoord>> {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| (coord.q, coord.r));
    let mut protected = Vec::new();
    for edge in edges {
        arena_core(arena, edge.child, &mut protected)?;
    }
    let defender = claimant.other();
    for entry in state.board().windows().entries() {
        if entry.active_player() == Some(defender)
            && u32::from(entry.count(defender)).saturating_add(d) >= 6
        {
            protected.extend(entry.empty_cells());
        }
    }
    protected.sort_by_key(|coord| (coord.q, coord.r));
    protected.dedup();
    let stones = state.board().occupied_cells();
    let pending = protected
        .iter()
        .copied()
        .filter(|cell| {
            legal
                .binary_search_by_key(&(cell.q, cell.r), |c| (c.q, c.r))
                .is_err()
                && !stones.contains(cell)
        })
        .collect::<Vec<_>>();
    let mut required = protected
        .iter()
        .copied()
        .filter(|cell| {
            legal
                .binary_search_by_key(&(cell.q, cell.r), |c| (c.q, c.r))
                .is_ok()
        })
        .collect::<Vec<_>>();
    if !pending.is_empty() {
        let radius = seed_band_radius(d);
        required.extend(legal.iter().copied().filter(|cell| {
            pending
                .iter()
                .any(|target| i32::from(hex_distance(*cell, *target)) <= radius)
        }));
    }
    required.sort_by_key(|coord| (coord.q, coord.r));
    required.dedup();
    Some(required)
}

/// Step-1-only view of the revised T3/T4 zone.  This is deliberately derived
/// from a completed certificate, because D10's `Prot(N)` is the union of live
/// roles in reachable descendants and therefore is not finder-hint data.
/// The independent verifier gets its own implementation in the verify phase.
#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Round3ShadowReport {
    pub quiet_turns: usize,
    pub quiet_legal_edges: usize,
    pub zones: Vec<Round3ShadowZoneRecord>,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Round3ShadowZoneRecord {
    pub ply: u32,
    pub b: u8,
    pub k: Option<u8>,
    pub local_budget: u32,
    pub full_legal: usize,
    pub zone: Vec<HexCoord>,
    pub z_dir: usize,
    pub z_seed: usize,
    pub z_touch: usize,
    pub z_virgin: usize,
    pub represented_in_zone: usize,
    pub best_represented_rank: Option<usize>,
    pub worst_represented_rank: Option<usize>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct Round3ShadowSummary {
    local_budget: u32,
    protected: Vec<HexCoord>,
}

#[cfg(test)]
fn shadow_uniform_zone(
    state: &RustHexoState,
    claimant: Player,
    local_budget: u32,
    protected: &[HexCoord],
) -> (Vec<HexCoord>, usize, usize, usize, usize) {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| raw_coord_key(*coord));
    let stones = state.board().occupied_cells();

    let mut z_dir = protected
        .iter()
        .copied()
        .filter(|cell| {
            legal
                .binary_search_by_key(&raw_coord_key(*cell), |c| raw_coord_key(*c))
                .is_ok()
        })
        .collect::<Vec<_>>();
    z_dir.sort_by_key(|coord| raw_coord_key(*coord));
    z_dir.dedup();

    let pending = protected
        .iter()
        .copied()
        .filter(|cell| {
            legal
                .binary_search_by_key(&raw_coord_key(*cell), |c| raw_coord_key(*c))
                .is_err()
                && !stones.contains(cell)
        })
        .collect::<Vec<_>>();
    let radius = seed_band_radius(local_budget);
    let mut z_seed = if pending.is_empty() {
        Vec::new()
    } else {
        legal
            .iter()
            .copied()
            .filter(|cell| {
                pending
                    .iter()
                    .any(|target| i32::from(hex_distance(*cell, *target)) <= radius)
            })
            .collect::<Vec<_>>()
    };
    z_seed.sort_by_key(|coord| raw_coord_key(*coord));
    z_seed.dedup();

    let defender = claimant.other();
    let mut z_touch = Vec::new();
    for entry in state.board().windows().entries() {
        let count = entry.count(defender);
        if entry.active_player() == Some(defender)
            && count >= 1
            && u32::from(count).saturating_add(local_budget) >= 6
        {
            z_touch.extend(entry.empty_cells());
        }
    }
    z_touch.sort_by_key(|coord| raw_coord_key(*coord));
    z_touch.dedup();

    // A full legal set is a conservative uniform B-clock implementation of
    // Z_virgin once B>=6. T4 explicitly permits larger admissible upper bounds;
    // the common B<=5 case remains the exact empty virgin component.
    let z_virgin = if local_budget >= 6 {
        legal.clone()
    } else {
        Vec::new()
    };

    let sizes = (z_dir.len(), z_seed.len(), z_touch.len(), z_virgin.len());
    let mut zone = z_dir;
    zone.extend(z_seed);
    zone.extend(z_touch);
    zone.extend(z_virgin);
    zone.sort_by_key(|coord| raw_coord_key(*coord));
    zone.dedup();
    if zone.is_empty() {
        if let Some(&fallback) = legal.first() {
            zone.push(fallback);
        }
    }
    (zone, sizes.0, sizes.1, sizes.2, sizes.3)
}

#[cfg(test)]
pub(crate) fn round3_shadow_certificate(
    root: &RustHexoState,
    cert: &TssCertificate,
) -> Option<Round3ShadowReport> {
    if cert.root != RootBinding::from_state(root) {
        return None;
    }

    fn walk(
        cert: &TssCertificate,
        id: CertNodeId,
        state: &mut RustHexoState,
        report: &mut Round3ShadowReport,
        depth: usize,
    ) -> Option<Round3ShadowSummary> {
        if depth > MAX_CERT_DEPTH {
            return None;
        }
        let node = cert.nodes.get(id as usize)?;
        let mut protected = Vec::new();
        let local_budget = match node {
            CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
            CertNode::OrCompletion { mv, .. } => {
                protected.push(*mv);
                0
            }
            CertNode::Win { witness, .. } => {
                protected.extend(
                    state
                        .board()
                        .windows()
                        .entries()
                        .find(|entry| entry.key() == *witness)?
                        .empty_cells(),
                );
                0
            }
            CertNode::Loss { witnesses, .. } => {
                for witness in witnesses {
                    protected.extend(
                        state
                            .board()
                            .windows()
                            .entries()
                            .find(|entry| entry.key() == *witness)?
                            .empty_cells(),
                    );
                }
                u32::from(threats::placements_remaining(state))
            }
            CertNode::Choice { mv, child } => {
                let phase = state.phase();
                let mut legal = Vec::new();
                if matches!(phase, TurnPhase::SecondStone { .. }) {
                    state.write_legal_moves(&mut legal);
                }
                let (result, delta) = state.apply_with_delta(Placement { coord: *mv }).ok()?;
                if result.outcome.is_some() {
                    state.undo(delta);
                    return None;
                }
                if matches!(phase, TurnPhase::SecondStone { .. })
                    && !turn_forces_small_defender_reply(state, cert.claimant)
                {
                    report.quiet_turns = report.quiet_turns.saturating_add(1);
                    report.quiet_legal_edges = report.quiet_legal_edges.saturating_add(legal.len());
                }
                let child_summary = walk(cert, *child, state, report, depth + 1);
                state.undo(delta);
                let child_summary = child_summary?;
                protected.push(*mv);
                protected.extend(child_summary.protected);
                child_summary.local_budget
            }
            CertNode::Universal { edges, .. } => {
                let mut child_budget = 0u32;
                for edge in edges {
                    let (result, delta) =
                        state.apply_with_delta(Placement { coord: edge.mv }).ok()?;
                    if result.outcome.is_some() {
                        state.undo(delta);
                        return None;
                    }
                    let child_summary = walk(cert, edge.child, state, report, depth + 1);
                    state.undo(delta);
                    let child_summary = child_summary?;
                    child_budget = child_budget.max(child_summary.local_budget);
                    protected.extend(child_summary.protected);
                }
                let local_budget = child_budget.saturating_add(1);
                let analysis = threats::analyze(state);
                if analysis.min_hitting_set.is_none_or(|k| k < analysis.b) {
                    let mut full_legal_moves = Vec::new();
                    state.write_legal_moves(&mut full_legal_moves);
                    let (zone, z_dir, z_seed, z_touch, z_virgin) =
                        shadow_uniform_zone(state, cert.claimant, local_budget, &protected);
                    let hitting = hitting_universe(state, cert.claimant);
                    let frame = canonical_frame(state);
                    let mut ranked = zone.clone();
                    ranked.sort_by_key(|coord| {
                        (!hitting.contains(coord), canonical_coord_key(frame, *coord))
                    });
                    let mut ranks = edges
                        .iter()
                        .filter_map(|edge| ranked.iter().position(|cell| *cell == edge.mv))
                        .map(|rank| rank + 1)
                        .collect::<Vec<_>>();
                    ranks.sort_unstable();
                    report.zones.push(Round3ShadowZoneRecord {
                        ply: state.placements_made(),
                        b: analysis.b,
                        k: analysis.min_hitting_set,
                        local_budget,
                        full_legal: full_legal_moves.len(),
                        zone,
                        z_dir,
                        z_seed,
                        z_touch,
                        z_virgin,
                        represented_in_zone: ranks.len(),
                        best_represented_rank: ranks.first().copied(),
                        worst_represented_rank: ranks.last().copied(),
                    });
                }
                local_budget
            }
        };
        protected.sort_by_key(|coord| raw_coord_key(*coord));
        protected.dedup();
        Some(Round3ShadowSummary {
            local_budget,
            protected,
        })
    }

    let mut state = root.clone();
    let mut report = Round3ShadowReport::default();
    walk(cert, cert.root_node, &mut state, &mut report, 0)?;
    report
        .zones
        .sort_by_key(|zone| (zone.ply, zone.full_legal, zone.zone.len()));
    Some(report)
}

/// Choose the lexicographically least D6 image of the full semantic position.
/// Search ties are compared in this frame, so rotating/reflection-transforming
/// an input cannot change which proof-cost class is expanded first merely due
/// to raw `(q,r)` order.  The TT remains uncanonicalized and still uses exact
/// raw-position equality.
fn canonical_frame(state: &RustHexoState) -> u8 {
    let stone_count = state.board().occupied_cells().len();
    // One owner lookup per stone, not one per stone per symmetry.
    let stones: Vec<(HexCoord, u8)> = state
        .board()
        .occupied_cells()
        .iter()
        .map(|&coord| {
            (
                coord,
                player_code(state.board().get(coord).expect("occupied cell has owner")),
            )
        })
        .collect();
    let mut best_phase: Option<(u8, i32, i32)> = None;
    let mut best_stones = Vec::with_capacity(stone_count);
    let mut candidate_stones = Vec::with_capacity(stone_count);
    let mut best_symmetry = 0;
    for symmetry in 0..12u8 {
        let phase = match state.phase() {
            TurnPhase::Opening => (0, 0, 0),
            TurnPhase::FirstStone => (1, 0, 0),
            TurnPhase::SecondStone { first } => {
                let (q, r) = d6_coord_i32(first, symmetry);
                (2, q, r)
            }
        };
        candidate_stones.clear();
        candidate_stones.extend(stones.iter().map(|&(coord, owner)| {
            let (q, r) = d6_coord_i32(coord, symmetry);
            (q, r, owner)
        }));
        candidate_stones.sort_unstable();
        if best_phase
            .as_ref()
            .is_none_or(|best| (&phase, &candidate_stones) < (best, &best_stones))
        {
            best_phase = Some(phase);
            best_symmetry = symmetry;
            std::mem::swap(&mut best_stones, &mut candidate_stones);
        }
    }
    debug_assert!(best_phase.is_some(), "D6 contains identity");
    best_symmetry
}

fn canonical_coord_key(frame: u8, coord: HexCoord) -> (i32, i32) {
    d6_coord_i32(coord, frame)
}

fn raw_coord_key(coord: HexCoord) -> (i16, i16) {
    (coord.q, coord.r)
}

fn d6_coord_i32(coord: HexCoord, symmetry: u8) -> (i32, i32) {
    let mut q = i32::from(coord.q);
    let mut r = i32::from(coord.r);
    if symmetry >= 6 {
        r = -q - r;
    }
    for _ in 0..(symmetry % 6) {
        (q, r) = (-r, q + r);
    }
    (q, r)
}

// === Full-key transposition table ==========================================

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct KeyStone {
    q: i16,
    r: i16,
    owner: u8,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum KeyPhase {
    Opening,
    FirstStone,
    SecondStone { q: i16, r: i16 },
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct KeyTerminal {
    winner: u8,
    placements: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PositionKey {
    stones: Vec<KeyStone>,
    current_player: u8,
    phase: KeyPhase,
    placements_made: u32,
    terminal: Option<KeyTerminal>,
}

impl PositionKey {
    fn from_state(state: &RustHexoState) -> Self {
        let mut stones: Vec<KeyStone> = state
            .board()
            .occupied_cells()
            .iter()
            .map(|coord| KeyStone {
                q: coord.q,
                r: coord.r,
                owner: player_code(state.board().get(*coord).expect("occupied cell has owner")),
            })
            .collect();
        stones.sort_by_key(|stone| (stone.q, stone.r, stone.owner));
        let phase = match state.phase() {
            TurnPhase::Opening => KeyPhase::Opening,
            TurnPhase::FirstStone => KeyPhase::FirstStone,
            TurnPhase::SecondStone { first } => KeyPhase::SecondStone {
                q: first.q,
                r: first.r,
            },
        };
        let terminal = state.terminal().map(|outcome| KeyTerminal {
            winner: player_code(outcome.winner),
            placements: outcome.placements,
        });
        Self {
            stones,
            current_player: player_code(state.current_player()),
            phase,
            placements_made: state.placements_made(),
            terminal,
        }
    }

    fn stable_hash(&self) -> u64 {
        // FNV-1a is used only for bucket selection.  Equality below, never this
        // 64-bit value, authorizes a proof hit.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for &byte in bytes {
                *hash ^= u64::from(byte);
                *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        feed(&mut hash, &[self.current_player]);
        feed(&mut hash, &self.placements_made.to_le_bytes());
        match self.phase {
            KeyPhase::Opening => feed(&mut hash, &[0]),
            KeyPhase::FirstStone => feed(&mut hash, &[1]),
            KeyPhase::SecondStone { q, r } => {
                feed(&mut hash, &[2]);
                feed(&mut hash, &q.to_le_bytes());
                feed(&mut hash, &r.to_le_bytes());
            }
        }
        match self.terminal {
            None => feed(&mut hash, &[0]),
            Some(terminal) => {
                feed(&mut hash, &[1, terminal.winner]);
                feed(&mut hash, &terminal.placements.to_le_bytes());
            }
        }
        for stone in &self.stones {
            feed(&mut hash, &stone.q.to_le_bytes());
            feed(&mut hash, &stone.r.to_le_bytes());
            feed(&mut hash, &[stone.owner]);
        }
        hash
    }

    fn heap_bytes(&self) -> usize {
        self.stones
            .capacity()
            .saturating_mul(size_of::<KeyStone>())
            .saturating_add(ALLOC_OVERHEAD)
    }
}

fn player_code(player: Player) -> u8 {
    match player {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}

#[derive(Debug)]
struct TtEntry {
    hash: u64,
    key: PositionKey,
    claimant: Player,
    node: CertNodeId,
}

#[derive(Debug)]
struct BoundedTt {
    slots: Vec<Option<TtEntry>>,
    cap: usize,
    current_bytes: usize,
    peak_bytes: usize,
    hash_mask: u64,
    replacements: u64,
    refusals: u64,
}

impl BoundedTt {
    fn new(cap: usize, hash_mask: u64) -> Self {
        let slot_count = (cap / TARGET_BYTES_PER_TT_SLOT).min(MAX_TT_SLOTS);
        if slot_count == 0 {
            return Self {
                slots: Vec::new(),
                cap,
                current_bytes: 0,
                peak_bytes: 0,
                hash_mask,
                replacements: 0,
                refusals: 0,
            };
        }
        let mut slots = Vec::with_capacity(slot_count);
        slots.resize_with(slot_count, || None);
        let base = slots
            .capacity()
            .saturating_mul(size_of::<Option<TtEntry>>())
            .saturating_add(ALLOC_OVERHEAD);
        if base > cap {
            return Self {
                slots: Vec::new(),
                cap,
                current_bytes: 0,
                peak_bytes: 0,
                hash_mask,
                replacements: 0,
                refusals: 0,
            };
        }
        Self {
            slots,
            cap,
            current_bytes: base,
            peak_bytes: base,
            hash_mask,
            replacements: 0,
            refusals: 0,
        }
    }

    fn lookup(&self, key: &PositionKey, claimant: Player) -> Option<CertNodeId> {
        if self.slots.is_empty() {
            return None;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let entry = self.slots[index].as_ref()?;
        (entry.hash == hash && entry.claimant == claimant && entry.key == *key)
            .then_some(entry.node)
    }

    fn entry_count(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    fn insert(&mut self, key: PositionKey, claimant: Player, node: CertNodeId) {
        if self.slots.is_empty() {
            return;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old_heap = self.slots[index]
            .as_ref()
            .map(|entry| entry.key.heap_bytes())
            .unwrap_or(0);
        let new_heap = key.heap_bytes();
        let candidate_bytes = self
            .current_bytes
            .saturating_sub(old_heap)
            .saturating_add(new_heap);
        if candidate_bytes > self.cap {
            self.refusals = self.refusals.saturating_add(1);
            return;
        }
        if self.slots[index]
            .as_ref()
            .is_some_and(|old| old.hash != hash || old.claimant != claimant || old.key != key)
        {
            self.replacements = self.replacements.saturating_add(1);
        }
        self.slots[index] = Some(TtEntry {
            hash,
            key,
            claimant,
            node,
        });
        self.current_bytes = candidate_bytes;
        self.peak_bytes = self.peak_bytes.max(candidate_bytes);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CachedProof {
    nodes: Vec<CertNode>,
    root_node: CertNodeId,
    explicit_edges: usize,
    commutation_count: usize,
    witness_count: usize,
    /// Maximum number of certificate edges below `root_node`.
    height: usize,
    /// Maximum exact resolution label over all contained typed leaves.
    resolution_t: u32,
    /// Minimum zone-build deadline over all contained zoned components.
    zone_build_t: Option<u32>,
}

impl CachedProof {
    fn from_arena_limited(
        arena: &[CertNode],
        root: CertNodeId,
        max_nodes: usize,
        max_edges: usize,
    ) -> Option<Self> {
        let (nodes, root_node) = compact_certificate_limited(arena, root, max_nodes, max_edges)?;
        Self::from_compact(nodes, root_node)
    }

    fn from_compact(nodes: Vec<CertNode>, root_node: CertNodeId) -> Option<Self> {
        if nodes.is_empty() || root_node as usize >= nodes.len() {
            return None;
        }
        let mut heights = vec![0usize; nodes.len()];
        let mut explicit_edges = 0usize;
        let mut commutation_count = 0usize;
        let mut witness_count = 0usize;
        let mut resolution_t = 0u32;
        let mut zone_build_t: Option<u32> = None;
        for (index, node) in nodes.iter().enumerate() {
            resolution_t = resolution_t.max(node_resolution(node));
            match node {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } => {
                    witness_count = witness_count.checked_add(1)?;
                }
                CertNode::Loss { witnesses, .. } => {
                    witness_count = witness_count.checked_add(witnesses.len())?;
                }
                CertNode::Universal { commutations, .. } => {
                    commutation_count = commutation_count.checked_add(commutations.len())?;
                }
                CertNode::Choice { .. } => {}
                // Extension nodes are never admitted to the proof caches;
                // refusing here keeps every cached fragment legacy-shaped.
                CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
            }
            if let CertNode::Universal {
                zone: Some(zone), ..
            } = node
            {
                zone_build_t = Some(
                    zone_build_t.map_or(zone.build_horizon, |old| old.min(zone.build_horizon)),
                );
            }
            heights[index] = match node {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => 0,
                CertNode::Choice { child, .. } => {
                    let child = *child as usize;
                    if child >= index {
                        return None;
                    }
                    heights[child].checked_add(1)?
                }
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    explicit_edges = explicit_edges.checked_add(edges.len())?;
                    let mut height = 0usize;
                    for edge in edges {
                        let child = edge.child as usize;
                        if child >= index {
                            return None;
                        }
                        height = height.max(heights[child].checked_add(1)?);
                    }
                    for item in commutations {
                        for child in [item.first_child, item.mirror_child] {
                            let child = child as usize;
                            if child >= index {
                                return None;
                            }
                            height = height.max(heights[child].checked_add(1)?);
                        }
                    }
                    height
                }
                CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
            };
        }
        let height = heights[root_node as usize];
        let proof = Self {
            nodes,
            root_node,
            explicit_edges,
            commutation_count,
            witness_count,
            height,
            resolution_t,
            zone_build_t,
        };
        proof.validate()?;
        Some(proof)
    }

    fn validate(&self) -> Option<()> {
        if self.nodes.is_empty()
            || self.nodes.len() > MAX_CERT_NODES
            || self.root_node as usize >= self.nodes.len()
            || self.height > MAX_CERT_DEPTH
            || self.explicit_edges > MAX_CERT_EDGES
        {
            return None;
        }
        let rebuilt = Self::from_compact_unchecked_metadata(&self.nodes, self.root_node)?;
        (rebuilt
            == (
                self.explicit_edges,
                self.commutation_count,
                self.witness_count,
                self.height,
                self.resolution_t,
                self.zone_build_t,
            ))
            .then_some(())
    }

    fn from_compact_unchecked_metadata(
        nodes: &[CertNode],
        root_node: CertNodeId,
    ) -> Option<(usize, usize, usize, usize, u32, Option<u32>)> {
        let mut heights = vec![0usize; nodes.len()];
        let mut explicit_edges = 0usize;
        let mut commutation_count = 0usize;
        let mut witness_count = 0usize;
        let mut resolution_t = 0u32;
        let mut zone_build_t: Option<u32> = None;
        for (index, node) in nodes.iter().enumerate() {
            resolution_t = resolution_t.max(node_resolution(node));
            match node {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } => {
                    witness_count = witness_count.checked_add(1)?;
                }
                CertNode::Loss { witnesses, .. } => {
                    witness_count = witness_count.checked_add(witnesses.len())?;
                }
                CertNode::Universal { commutations, .. } => {
                    commutation_count = commutation_count.checked_add(commutations.len())?;
                }
                CertNode::Choice { .. } => {}
                // Extension nodes are never admitted to the proof caches;
                // refusing here keeps every cached fragment legacy-shaped.
                CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
            }
            if let CertNode::Universal {
                zone: Some(zone), ..
            } = node
            {
                zone_build_t = Some(
                    zone_build_t.map_or(zone.build_horizon, |old| old.min(zone.build_horizon)),
                );
            }
            heights[index] = match node {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => 0,
                CertNode::Choice { child, .. } => {
                    let child = *child as usize;
                    if child >= index {
                        return None;
                    }
                    heights[child].checked_add(1)?
                }
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    explicit_edges = explicit_edges.checked_add(edges.len())?;
                    let mut height = 0usize;
                    for edge in edges {
                        let child = edge.child as usize;
                        if child >= index {
                            return None;
                        }
                        height = height.max(heights[child].checked_add(1)?);
                    }
                    for item in commutations {
                        for child in [item.first_child, item.mirror_child] {
                            let child = child as usize;
                            if child >= index {
                                return None;
                            }
                            height = height.max(heights[child].checked_add(1)?);
                        }
                    }
                    height
                }
                CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
            };
        }
        Some((
            explicit_edges,
            commutation_count,
            witness_count,
            heights[root_node as usize],
            resolution_t,
            zone_build_t,
        ))
    }

    fn heap_bytes(&self) -> usize {
        let mut bytes = allocation_bytes(self.nodes.capacity(), size_of::<CertNode>());
        for node in &self.nodes {
            match node {
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    bytes = bytes
                        .saturating_add(allocation_bytes(edges.capacity(), size_of::<CertEdge>()));
                    bytes = bytes.saturating_add(allocation_bytes(
                        commutations.capacity(),
                        size_of::<CertCommutation>(),
                    ));
                }
                // Adaptive LOSS contracts own a witness vector too — omitting
                // it made the cap admission understate real heap (Codex
                // review, cache accounting).
                CertNode::Loss { witnesses, .. } => {
                    bytes = bytes.saturating_add(allocation_bytes(
                        witnesses.capacity(),
                        size_of::<WindowKey>(),
                    ));
                }
                // Complete boxed-v3 accounting (design §2.5): extension nodes
                // never enter the cache (from_compact refuses them), but the
                // charge is exhaustive so a future admission path cannot
                // silently understate heap.
                CertNode::UniversalGroup2V1(node) => {
                    bytes = bytes.saturating_add(group2_node_heap_bytes(node));
                }
                CertNode::FhwGateV1(gate) => {
                    bytes = bytes.saturating_add(fhw_gate_heap_bytes(gate));
                }
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Choice { .. } => {}
            }
        }
        bytes
    }
}

/// Exhaustive heap charge for one boxed `UniversalGroup2NodeV1`.
fn group2_node_heap_bytes(node: &crate::tss_verify::UniversalGroup2NodeV1) -> usize {
    allocation_bytes(1, size_of::<crate::tss_verify::UniversalGroup2NodeV1>())
        .saturating_add(allocation_bytes(node.edges.capacity(), size_of::<CertEdge>()))
        .saturating_add(allocation_bytes(node.proof.authority.defender_path.len(), 1))
        .saturating_add(allocation_bytes(node.proof.authority.fhw_path.len(), 1))
}

/// Exhaustive heap charge for one boxed `FhwGateNodeV1`.
fn fhw_gate_heap_bytes(gate: &crate::tss_verify::FhwGateNodeV1) -> usize {
    let mut bytes = allocation_bytes(1, size_of::<crate::tss_verify::FhwGateNodeV1>())
        .saturating_add(allocation_bytes(
            gate.representatives.capacity(),
            size_of::<CertEdge>(),
        ))
        .saturating_add(allocation_bytes(
            gate.proof.threats.capacity(),
            size_of::<WindowKey>(),
        ))
        .saturating_add(allocation_bytes(
            gate.proof.map.capacity(),
            size_of::<crate::tss_verify::FhwMapV1>(),
        ))
        .saturating_add(allocation_bytes(gate.proof.authority.defender_path.len(), 1))
        .saturating_add(allocation_bytes(gate.proof.authority.fhw_path.len(), 1));
    for entry in &gate.proof.map {
        bytes = bytes
            .saturating_add(allocation_bytes(
                entry.roles.capacity(),
                size_of::<crate::tss_verify::FhwRoleClaimV1>(),
            ))
            .saturating_add(allocation_bytes(
                entry.windows.capacity(),
                size_of::<crate::tss_verify::FhwWindowClaimV1>(),
            ));
    }
    bytes
}

#[derive(Debug, PartialEq, Eq)]
struct SharedTtEntry {
    hash: u64,
    key: PositionKey,
    claimant: Player,
    proof: CachedProof,
}

impl SharedTtEntry {
    fn heap_bytes(&self) -> usize {
        self.key
            .heap_bytes()
            .saturating_add(self.proof.heap_bytes())
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SharedProofCache {
    slots: Vec<Option<SharedTtEntry>>,
    cap: usize,
    current_bytes: usize,
    peak_bytes: usize,
    hash_mask: u64,
}

impl SharedProofCache {
    fn new(cap: usize, hash_mask: u64) -> Self {
        let slot_count = (cap / TARGET_BYTES_PER_SHARED_TT_SLOT).min(MAX_TT_SLOTS);
        if slot_count == 0 {
            return Self {
                slots: Vec::new(),
                cap,
                current_bytes: 0,
                peak_bytes: 0,
                hash_mask,
            };
        }
        let mut slots = Vec::with_capacity(slot_count);
        slots.resize_with(slot_count, || None);
        let base = allocation_bytes(slots.capacity(), size_of::<Option<SharedTtEntry>>());
        if base > cap {
            return Self {
                slots: Vec::new(),
                cap,
                current_bytes: 0,
                peak_bytes: 0,
                hash_mask,
            };
        }
        Self {
            slots,
            cap,
            current_bytes: base,
            peak_bytes: base,
            hash_mask,
        }
    }

    fn reconfigure(&mut self, cap: usize, hash_mask: u64) {
        if self.cap != cap || self.hash_mask != hash_mask {
            *self = Self::new(cap, hash_mask);
        } else {
            self.peak_bytes = self.current_bytes;
        }
    }

    /// Drop every retained fragment (profile isolation on option changes).
    fn clear(&mut self) {
        *self = Self::new(self.cap, self.hash_mask);
    }

    fn lookup_cloned(&self, key: &PositionKey, claimant: Player) -> Option<CachedProof> {
        if self.slots.is_empty() {
            return None;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let entry = self.slots[index].as_ref()?;
        (entry.hash == hash && entry.claimant == claimant && entry.key == *key)
            .then(|| entry.proof.clone())
    }

    fn entry_count(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    fn could_admit_minimal(&self, key: &PositionKey) -> bool {
        self.could_admit_heap(key, allocation_bytes(1, size_of::<CertNode>()))
    }

    fn could_admit_compact(&self, key: &PositionKey, nodes: &[CertNode]) -> bool {
        let mut proof_heap = allocation_bytes(nodes.len(), size_of::<CertNode>());
        for node in nodes {
            match node {
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    proof_heap = proof_heap
                        .saturating_add(allocation_bytes(edges.len(), size_of::<CertEdge>()));
                    proof_heap = proof_heap.saturating_add(allocation_bytes(
                        commutations.len(),
                        size_of::<CertCommutation>(),
                    ));
                }
                CertNode::Loss { witnesses, .. } => {
                    proof_heap = proof_heap
                        .saturating_add(allocation_bytes(witnesses.len(), size_of::<WindowKey>()));
                }
                CertNode::UniversalGroup2V1(node) => {
                    proof_heap = proof_heap.saturating_add(group2_node_heap_bytes(node));
                }
                CertNode::FhwGateV1(gate) => {
                    proof_heap = proof_heap.saturating_add(fhw_gate_heap_bytes(gate));
                }
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Choice { .. } => {}
            }
        }
        self.could_admit_heap(key, proof_heap)
    }

    fn could_admit_heap(&self, key: &PositionKey, proof_heap: usize) -> bool {
        if self.slots.is_empty() {
            return false;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old_heap = self.slots[index]
            .as_ref()
            .map(SharedTtEntry::heap_bytes)
            .unwrap_or(0);
        self.current_bytes
            .saturating_sub(old_heap)
            .saturating_add(key.heap_bytes())
            .saturating_add(proof_heap)
            <= self.cap
    }

    fn insert(&mut self, key: PositionKey, claimant: Player, proof: CachedProof) {
        if self.slots.is_empty() || proof.validate().is_none() {
            return;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old_heap = self.slots[index]
            .as_ref()
            .map(SharedTtEntry::heap_bytes)
            .unwrap_or(0);
        let entry = SharedTtEntry {
            hash,
            key,
            claimant,
            proof,
        };
        let candidate_bytes = self
            .current_bytes
            .saturating_sub(old_heap)
            .saturating_add(entry.heap_bytes());
        if candidate_bytes > self.cap {
            return;
        }
        self.slots[index] = Some(entry);
        self.current_bytes = candidate_bytes;
        self.peak_bytes = self.peak_bytes.max(candidate_bytes);
    }

    #[cfg(test)]
    fn recomputed_bytes(&self) -> usize {
        let base = allocation_bytes(self.slots.capacity(), size_of::<Option<SharedTtEntry>>());
        base.saturating_add(
            self.slots
                .iter()
                .flatten()
                .map(SharedTtEntry::heap_bytes)
                .sum::<usize>(),
        )
    }
}

/// One immutable positive proof proposition. `key` and `claimant` are part of
/// the owned Arc so every live wide-PN handle can recheck identity without
/// cloning either the key or certificate payload.
#[derive(Debug, PartialEq, Eq)]
struct ProvenFragment {
    key: PositionKey,
    claimant: Player,
    proof: CachedProof,
}

impl ProvenFragment {
    fn heap_bytes(&self) -> usize {
        self.key
            .heap_bytes()
            .saturating_add(self.proof.heap_bytes())
            .saturating_add(size_of::<Self>())
            .saturating_add(ALLOC_OVERHEAD)
    }
}

#[derive(Debug)]
struct ProvenFragmentEntry {
    hash: u64,
    fragment: Arc<ProvenFragment>,
}

#[derive(Debug)]
struct ProvenFragmentStore {
    slots: Vec<Option<ProvenFragmentEntry>>,
    cap: usize,
    current_bytes: usize,
    peak_bytes: usize,
    hash_mask: u64,
    entry_count: usize,
    stored_nodes: usize,
    stored_edges: usize,
    admissions: u64,
    replacements: u64,
    refusals: u64,
}

impl ProvenFragmentStore {
    fn new(cap: usize, hash_mask: u64) -> Self {
        Self {
            // A fresh official-corpus solver is cold. Reserving a full direct
            // table here would steal search TT without enabling a single hit.
            // Allocate it only when a verified proof is actually promoted.
            slots: Vec::new(),
            cap,
            current_bytes: 0,
            peak_bytes: 0,
            hash_mask,
            entry_count: 0,
            stored_nodes: 0,
            stored_edges: 0,
            admissions: 0,
            replacements: 0,
            refusals: 0,
        }
    }

    fn reconfigure(&mut self, cap: usize, hash_mask: u64) {
        if self.cap != cap || self.hash_mask != hash_mask {
            *self = Self::new(cap, hash_mask);
        } else {
            self.peak_bytes = self.current_bytes;
        }
    }

    fn clear(&mut self) {
        *self = Self::new(self.cap, self.hash_mask);
    }

    fn ensure_slots(&mut self) -> bool {
        if !self.slots.is_empty() {
            return true;
        }
        let slot_count = (self.cap / TARGET_BYTES_PER_PROVEN_FRAGMENT_SLOT).min(MAX_TT_SLOTS);
        if slot_count == 0 {
            return false;
        }
        let mut slots = Vec::with_capacity(slot_count);
        slots.resize_with(slot_count, || None);
        let base = allocation_bytes(slots.capacity(), size_of::<Option<ProvenFragmentEntry>>());
        if base > self.cap {
            return false;
        }
        self.slots = slots;
        self.current_bytes = base;
        self.peak_bytes = self.peak_bytes.max(base);
        true
    }

    fn lookup(&self, key: &PositionKey, claimant: Player) -> Option<Arc<ProvenFragment>> {
        if self.slots.is_empty() {
            return None;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let entry = self.slots[index].as_ref()?;
        (entry.hash == hash && entry.fragment.claimant == claimant && entry.fragment.key == *key)
            .then(|| Arc::clone(&entry.fragment))
    }

    fn insert(&mut self, key: PositionKey, claimant: Player, proof: CachedProof) -> bool {
        if proof.validate().is_none() || !self.ensure_slots() {
            self.refusals = self.refusals.saturating_add(1);
            return false;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old = self.slots[index].as_ref();

        // Alternative proof graphs are never structurally unioned. For the
        // identical proposition replace only when the new admissible horizon
        // interval contains the old one (or the interval is identical and the
        // payload is smaller). Resolution/build intervals can otherwise be
        // incomparable, so a lexicographic choice would silently discard
        // useful warm queries. This is cache policy, not a proof-label merge.
        if let Some(old) = old.filter(|old| {
            old.hash == hash && old.fragment.claimant == claimant && old.fragment.key == key
        }) {
            let old_build = old.fragment.proof.zone_build_t.unwrap_or(u32::MAX);
            let new_build = proof.zone_build_t.unwrap_or(u32::MAX);
            let interval_dominates =
                proof.resolution_t <= old.fragment.proof.resolution_t && new_build >= old_build;
            let interval_is_strict =
                proof.resolution_t < old.fragment.proof.resolution_t || new_build > old_build;
            let new_is_better = interval_dominates
                && (interval_is_strict || proof.heap_bytes() < old.fragment.proof.heap_bytes());
            if !new_is_better {
                self.refusals = self.refusals.saturating_add(1);
                return false;
            }
        }

        let fragment = Arc::new(ProvenFragment {
            key,
            claimant,
            proof,
        });
        let new_heap = fragment.heap_bytes();
        let old_heap = old.map(|entry| entry.fragment.heap_bytes()).unwrap_or(0);
        let candidate_bytes = self
            .current_bytes
            .saturating_sub(old_heap)
            .saturating_add(new_heap);
        if candidate_bytes > self.cap {
            self.refusals = self.refusals.saturating_add(1);
            return false;
        }

        if let Some(old) = old {
            self.stored_nodes = self
                .stored_nodes
                .saturating_sub(old.fragment.proof.nodes.len());
            self.stored_edges = self
                .stored_edges
                .saturating_sub(old.fragment.proof.explicit_edges);
            self.replacements = self.replacements.saturating_add(1);
        } else {
            self.entry_count = self.entry_count.saturating_add(1);
        }
        self.stored_nodes = self.stored_nodes.saturating_add(fragment.proof.nodes.len());
        self.stored_edges = self
            .stored_edges
            .saturating_add(fragment.proof.explicit_edges);
        self.slots[index] = Some(ProvenFragmentEntry { hash, fragment });
        self.current_bytes = candidate_bytes;
        self.peak_bytes = self.peak_bytes.max(candidate_bytes);
        self.admissions = self.admissions.saturating_add(1);
        true
    }

    #[cfg(test)]
    fn recomputed_bytes(&self) -> usize {
        allocation_bytes(
            self.slots.capacity(),
            size_of::<Option<ProvenFragmentEntry>>(),
        )
        .saturating_add(
            self.slots
                .iter()
                .flatten()
                .map(|entry| entry.fragment.heap_bytes())
                .sum::<usize>(),
        )
    }
}

fn allocation_bytes(capacity: usize, element_size: usize) -> usize {
    if capacity == 0 {
        0
    } else {
        capacity
            .saturating_mul(element_size)
            .saturating_add(ALLOC_OVERHEAD)
    }
}

fn offset_node_id(id: CertNodeId, base: usize, final_len: usize) -> Option<CertNodeId> {
    let index = id as usize;
    let mapped = base.checked_add(index)?;
    (mapped < final_len)
        .then(|| u32::try_from(mapped).ok())
        .flatten()
}

fn remap_node_ids_with_offset(node: &mut CertNode, base: usize, final_len: usize) -> Option<()> {
    match node {
        CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {}
        CertNode::Choice { child, .. } => {
            *child = offset_node_id(*child, base, final_len)?;
        }
        CertNode::Universal {
            edges,
            commutations,
            ..
        } => {
            for edge in edges {
                edge.child = offset_node_id(edge.child, base, final_len)?;
            }
            for item in commutations {
                item.first_child = offset_node_id(item.first_child, base, final_len)?;
                item.mirror_child = offset_node_id(item.mirror_child, base, final_len)?;
            }
        }
        CertNode::UniversalGroup2V1(node) => {
            for edge in &mut node.edges {
                edge.child = offset_node_id(edge.child, base, final_len)?;
            }
        }
        // Gate role rows carry node references of their own; this solver
        // never builds gates, so refuse rather than remap partially.
        CertNode::FhwGateV1(_) => return None,
    }
    Some(())
}

/// Remove abandoned OR branches from the certificate arena and remap every
/// reachable child.  The resulting certificate has no orphan nodes, which the
/// independent verifier requires.
pub(crate) fn compact_certificate(
    arena: &[CertNode],
    root: CertNodeId,
) -> Option<(Vec<CertNode>, CertNodeId)> {
    compact_certificate_limited(arena, root, MAX_CERT_NODES, MAX_CERT_EDGES)
}

fn compact_certificate_limited(
    arena: &[CertNode],
    root: CertNodeId,
    max_nodes: usize,
    max_edges: usize,
) -> Option<(Vec<CertNode>, CertNodeId)> {
    fn copy(
        old: CertNodeId,
        arena: &[CertNode],
        remap: &mut [Option<CertNodeId>],
        visiting: &mut [bool],
        out: &mut Vec<CertNode>,
        edge_count: &mut usize,
        max_nodes: usize,
        max_edges: usize,
    ) -> Option<CertNodeId> {
        let index = old as usize;
        if index >= arena.len() || visiting[index] {
            return None;
        }
        if let Some(mapped) = remap[index] {
            return Some(mapped);
        }
        visiting[index] = true;
        let mapped_node = match &arena[index] {
            CertNode::OrCompletion {
                mv,
                witness,
                completion_ply,
            } => CertNode::OrCompletion {
                mv: *mv,
                witness: *witness,
                completion_ply: *completion_ply,
            },
            CertNode::Win {
                witness,
                count,
                budget,
                resolution_ply,
            } => CertNode::Win {
                witness: *witness,
                count: *count,
                budget: *budget,
                resolution_ply: *resolution_ply,
            },
            CertNode::Loss {
                witnesses,
                resolution_ply,
            } => CertNode::Loss {
                witnesses: witnesses.clone(),
                resolution_ply: *resolution_ply,
            },
            CertNode::Choice { mv, child } => CertNode::Choice {
                mv: *mv,
                child: copy(
                    *child, arena, remap, visiting, out, edge_count, max_nodes, max_edges,
                )?,
            },
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            } => {
                *edge_count = edge_count.checked_add(edges.len())?;
                if *edge_count > max_edges {
                    return None;
                }
                let mut mapped_edges = Vec::with_capacity(edges.len());
                for edge in edges {
                    mapped_edges.push(CertEdge {
                        mv: edge.mv,
                        child: copy(
                            edge.child, arena, remap, visiting, out, edge_count, max_nodes,
                            max_edges,
                        )?,
                    });
                }
                let mut mapped_commutations = Vec::with_capacity(commutations.len());
                for item in commutations {
                    mapped_commutations.push(CertCommutation {
                        first: item.first,
                        omitted_second: item.omitted_second,
                        first_child: copy(
                            item.first_child,
                            arena,
                            remap,
                            visiting,
                            out,
                            edge_count,
                            max_nodes,
                            max_edges,
                        )?,
                        mirror_child: copy(
                            item.mirror_child,
                            arena,
                            remap,
                            visiting,
                            out,
                            edge_count,
                            max_nodes,
                            max_edges,
                        )?,
                    });
                }
                CertNode::Universal {
                    edges: mapped_edges,
                    implicit_dispatch: *implicit_dispatch,
                    zone: zone.clone(),
                    commutations: mapped_commutations,
                }
            }
            CertNode::UniversalGroup2V1(node) => {
                *edge_count = edge_count.checked_add(node.edges.len())?;
                if *edge_count > max_edges {
                    return None;
                }
                let mut mapped_edges = Vec::with_capacity(node.edges.len());
                for edge in &node.edges {
                    mapped_edges.push(CertEdge {
                        mv: edge.mv,
                        child: copy(
                            edge.child, arena, remap, visiting, out, edge_count, max_nodes,
                            max_edges,
                        )?,
                    });
                }
                CertNode::UniversalGroup2V1(Box::new(
                    crate::tss_verify::UniversalGroup2NodeV1 {
                        edges: mapped_edges,
                        proof: node.proof.clone(),
                    },
                ))
            }
            CertNode::FhwGateV1(gate) => {
                // The emitted gate map is a SKELETON: correct
                // real_reply/representative/edge_class per K but EMPTY role and
                // window lists (the finalizer fills the rows, whose node
                // references are assigned on the unfolded tree post-compaction).
                // So there are no arena IDs inside the map to remap here — only
                // the representative subtree children.
                if gate
                    .proof
                    .map
                    .iter()
                    .any(|m| !m.roles.is_empty() || !m.windows.is_empty())
                {
                    return None; // never compact an already-filled gate
                }
                *edge_count = edge_count.checked_add(gate.representatives.len())?;
                if *edge_count > max_edges {
                    return None;
                }
                let mut mapped_reps = Vec::with_capacity(gate.representatives.len());
                for edge in &gate.representatives {
                    mapped_reps.push(CertEdge {
                        mv: edge.mv,
                        child: copy(
                            edge.child, arena, remap, visiting, out, edge_count, max_nodes,
                            max_edges,
                        )?,
                    });
                }
                CertNode::FhwGateV1(Box::new(crate::tss_verify::FhwGateNodeV1 {
                    representatives: mapped_reps,
                    proof: gate.proof.clone(),
                }))
            }
        };
        visiting[index] = false;
        if out.len() >= max_nodes {
            return None;
        }
        let mapped = u32::try_from(out.len()).ok()?;
        out.push(mapped_node);
        remap[index] = Some(mapped);
        Some(mapped)
    }

    if arena.len() > MAX_CERT_NODES || max_nodes > MAX_CERT_NODES || max_edges > MAX_CERT_EDGES {
        return None;
    }
    let mut remap = vec![None; arena.len()];
    let mut visiting = vec![false; arena.len()];
    let mut nodes = Vec::new();
    let mut edge_count = 0usize;
    let root_node = copy(
        root,
        arena,
        &mut remap,
        &mut visiting,
        &mut nodes,
        &mut edge_count,
        max_nodes,
        max_edges,
    )?;
    Some((nodes, root_node))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tss_core::CertVerify;
    use crate::tss_reference;
    use crate::tss_verify::{
        d6_remap_certificate, d6_transform_coord, TssVerifier, D6_SYMMETRY_COUNT,
    };
    use hexo_engine::{apply_placement, WindowStore};

    fn replay(coords: &[(i16, i16)]) -> RustHexoState {
        let mut state = RustHexoState::new();
        for &(q, r) in coords {
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord { q, r },
                },
            )
            .unwrap();
        }
        state
    }

    #[test]
    fn interior_census_phase_tables_match_contract_8_1() {
        let first = [10, 10, 9, 6, 2, 1];
        let second = [12, 12, 9, 5, 4, 1];
        for census in 0..=5u8 {
            assert_eq!(
                interior_census_lb_plies(TurnPhase::FirstStone, census),
                Some(first[census as usize])
            );
            assert_eq!(
                interior_census_lb_plies(
                    TurnPhase::SecondStone {
                        first: HexCoord::ZERO,
                    },
                    census,
                ),
                Some(second[census as usize])
            );
        }
        assert_eq!(interior_census_lb_plies(TurnPhase::FirstStone, 6), None);
        assert_eq!(interior_census_lb_plies(TurnPhase::Opening, 0), None);
    }

    #[test]
    fn interior_census_secondstone_c3_uses_strict_ply_five_boundary() {
        let state = replay(&[
            (0, 0),
            (2, 1),
            (3, 1),
            (-1, 0),
            (0, -1),
            (4, 1),
            (1, 2),
            (-1, 1),
            (1, -1),
            (1, 3),
            (1, 4),
            (-2, 0),
            (0, -2),
            (2, 0),
            (3, -1),
            (-2, 1),
            (-1, -2),
            (4, -2),
        ]);
        assert!(matches!(state.phase(), TurnPhase::SecondStone { .. }));
        assert_eq!(
            pn_init_census_features(&state, state.current_player()).0,
            3,
            "binding reachable witness must retain exact census c=3"
        );
        let root_ply = state.placements_made() - 1;
        let through_four = evaluate_interior_census_gate(
            &state,
            state.current_player(),
            root_ply,
            state.placements_made() + 4,
        )
        .expect("eligible c=3 SecondStone evaluation");
        assert!(through_four.dismiss);
        let through_five = evaluate_interior_census_gate(
            &state,
            state.current_player(),
            root_ply,
            state.placements_made() + 5,
        )
        .expect("eligible c=3 SecondStone evaluation");
        assert!(!through_five.dismiss, "LB == h must not gate");
    }

    fn cache_test_leaf() -> CertNode {
        CertNode::Win {
            witness: WindowKey {
                start: HexCoord::ZERO,
                axis: hexo_engine::Axis::Q,
            },
            count: 5,
            budget: 2,
            resolution_ply: 1,
        }
    }

    fn wide_pn_test_entry(pn: u32, node: WidePnNode, depth: usize) -> WidePnEntry {
        WidePnEntry {
            pn,
            dn: 1,
            prior: WidePnPrior { pn, dn: 1 },
            node,
            depth,
            universal_obligation: None,
        }
    }

    /// Geometry-only scaffold for the proof program's adversarial positions.
    /// P2 materializes these ownership maps into matched-horizon solver roots;
    /// keeping the canonical stone sets here makes mutations independent of a
    /// particular legal replay ordering.
    struct ZoneFixtureSpec {
        stones: Vec<(HexCoord, Player)>,
        focus: HexCoord,
    }

    fn g1_junction_spec() -> ZoneFixtureSpec {
        let attacker = Player::Player0;
        let defender = attacker.other();
        let arms = [
            (3, 0),
            (4, 0),
            (5, 0),
            (-3, 0),
            (-4, 0),
            (-5, 0),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, -3),
            (0, -4),
            (0, -5),
        ];
        let pin = [(8, -2), (11, -5), (12, -6), (13, -7)];
        let caps = [(6, 0), (-6, 0), (0, 6), (0, -6)];
        let mut stones = arms
            .into_iter()
            .chain(pin)
            .map(|(q, r)| (HexCoord::new(q, r), attacker))
            .collect::<Vec<_>>();
        stones.extend(
            caps.into_iter()
                .map(|(q, r)| (HexCoord::new(q, r), defender)),
        );
        ZoneFixtureSpec {
            stones,
            focus: HexCoord::ZERO,
        }
    }

    fn g3_counterfork_spec() -> ZoneFixtureSpec {
        let attacker = Player::Player0;
        let defender = attacker.other();
        let defender_arms = [
            (8, 0),
            (9, 0),
            (10, 0),
            (5, 3),
            (5, 4),
            (5, 5),
            (8, -3),
            (9, -4),
            (10, -5),
            (5, 0),
        ];
        let attacker_scaffold = [(-9, 4), (-8, 4), (-6, 4), (-4, 4)];
        let mut stones = defender_arms
            .into_iter()
            .map(|(q, r)| (HexCoord::new(q, r), defender))
            .collect::<Vec<_>>();
        stones.extend(
            attacker_scaffold
                .into_iter()
                .map(|(q, r)| (HexCoord::new(q, r), attacker)),
        );
        ZoneFixtureSpec {
            stones,
            focus: HexCoord::new(5, 0),
        }
    }

    #[test]
    fn zone_adversary_geometry_scaffolds_match_python_reference() {
        let g1 = g1_junction_spec();
        let g1_store = WindowStore::from_placements(&g1.stones);
        let junction_routes = g1_store
            .entries()
            .filter(|entry| {
                entry.key().contains(g1.focus)
                    && entry.count(Player::Player0) == 3
                    && entry.count(Player::Player1) == 0
            })
            .count();
        assert_eq!(junction_routes, 4, "G1 must retain all four junction arms");
        let pins = g1_store
            .entries()
            .filter(|entry| {
                entry.count(Player::Player0) == 4
                    && entry.count(Player::Player1) == 0
                    && entry.empty_cells().len() == 2
            })
            .count();
        assert_eq!(pins, 1, "G1 pin is deliberately a single live window");

        let g3 = g3_counterfork_spec();
        let g3_store = WindowStore::from_placements(&g3.stones);
        let fork_windows = g3_store
            .entries()
            .filter(|entry| {
                entry.key().contains(g3.focus)
                    && entry.count(Player::Player1) == 4
                    && entry.count(Player::Player0) == 0
            })
            .map(|entry| entry.empty_cells())
            .collect::<Vec<_>>();
        assert_eq!(fork_windows.len(), 3);
        for left in 0..fork_windows.len() {
            for right in left + 1..fork_windows.len() {
                assert!(
                    fork_windows[left]
                        .iter()
                        .all(|cell| !fork_windows[right].contains(cell)),
                    "G3 threat empties must be pairwise disjoint"
                );
            }
        }
    }

    #[test]
    fn pair_generator_uses_frozen_parent_order_and_keeps_newly_legal_cells() {
        let lower = HexCoord::new(-2, 0);
        let first = HexCoord::new(0, 0);
        let higher = HexCoord::new(2, 0);
        let newly_legal_lower = HexCoord::new(-1, -1);
        let pair = PairContext {
            first,
            turn_start_legal: vec![lower, first, higher],
        };
        let mut candidates = vec![lower, higher, newly_legal_lower];
        restrict_pair_candidates(&mut candidates, &pair);
        assert!(!candidates.contains(&lower));
        assert!(candidates.contains(&higher));
        assert!(candidates.contains(&newly_legal_lower));
    }

    #[test]
    fn pair_complete_width_adds_every_count_two_cell_at_both_turn_plies() {
        let first = pair_width_first_stone_fixture();
        assert_eq!(first.current_player(), Player::Player0);
        assert_eq!(first.phase(), TurnPhase::FirstStone);
        let mut second = first.clone();
        apply_placement(
            &mut second,
            Placement {
                coord: HexCoord::new(2, 0),
            },
        )
        .unwrap();
        assert_eq!(
            second.phase(),
            TurnPhase::SecondStone {
                first: HexCoord::new(2, 0)
            }
        );

        let width = WidthOptions::vcf_pair_complete();
        let mut saw_narrow_candidate = false;
        for state in [&first, &second] {
            let narrow = ordered_threat_creating_moves(state, Player::Player0);
            let wide = ordered_threat_creating_moves_with_width(state, Player::Player0, width);
            saw_narrow_candidate |= !narrow.is_empty();

            let narrow_coords = narrow.iter().map(|item| item.coord).collect::<Vec<_>>();
            let wide_narrow_coords = wide
                .iter()
                .filter(|item| item.strength >= 3)
                .map(|item| item.coord)
                .collect::<Vec<_>>();
            assert_eq!(wide_narrow_coords, narrow_coords);
            assert!(wide.iter().any(|item| item.strength == 2));
            assert!(wide
                .iter()
                .skip(narrow.len())
                .all(|item| item.strength == 2));

            for entry in state.board().windows().entries().filter(|entry| {
                entry.active_player() == Some(Player::Player0) && entry.count(Player::Player0) == 2
            }) {
                for coord in entry.empty_cells() {
                    assert!(wide.iter().any(|item| item.coord == coord));
                }
            }
        }
        assert!(saw_narrow_candidate);
    }

    #[test]
    fn pair_complete_count_two_order_prefers_forks_then_proximity() {
        let state = pair_width_first_stone_fixture();
        let candidates = ordered_threat_creating_moves_with_width(
            &state,
            Player::Player0,
            WidthOptions::vcf_pair_complete(),
        );
        let frame = canonical_frame(&state);
        let pair_starts = candidates
            .iter()
            .filter(|item| item.strength == 2)
            .collect::<Vec<_>>();
        assert!(pair_starts.len() > 1);
        assert!(pair_starts.windows(2).all(|pair| {
            let key = |item: &Candidate| {
                let canonical = canonical_coord_key(frame, item.coord);
                (
                    Reverse(item.pair_start_degree),
                    item.own_proximity,
                    canonical.0,
                    canonical.1,
                )
            };
            key(pair[0]) <= key(pair[1])
        }));
    }

    #[test]
    fn wide_urgent_root_classification_is_permutation_invariant() {
        let ordinary = HexCoord::new(-3, 1);
        let block = HexCoord::new(4, -2);
        let candidate = |coord, defender_block| Candidate {
            coord,
            strength: 3,
            priority_class: 0,
            child_threats: 1,
            defender_block,
            pair_start_degree: 0,
            own_proximity: 0,
            created_threats: Vec::new(),
        };
        let mut candidates = vec![candidate(ordinary, false), candidate(block, true)];
        let defender_blocks = turn_start_defender_blocks(&candidates);
        candidates.reverse();
        assert_eq!(
            turn_start_defender_blocks(&candidates),
            defender_blocks,
            "turn-start urgency must not depend on candidate order"
        );

        let child = |mv| WidePnChild {
            mv,
            result: WidePnChildResult::Pending,
            entry: None,
            future_key: None,
            prior: WidePnPrior::UNIFORM,
            urgent_block: wide_move_contains_defender_block(mv, &defender_blocks),
            first_width_tier: 0,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };
        let mut children = vec![
            child(WidePnMove::One(ordinary)),
            child(WidePnMove::One(block)),
        ];
        assert!(wide_choice_has_urgent_block(&children));
        children.reverse();
        assert!(
            wide_choice_has_urgent_block(&children),
            "urgent-root classification must not depend on child order"
        );
    }

    #[test]
    fn wide_unordered_pair_dedup_cannot_change_urgent_flag() {
        let block = HexCoord::new(-2, 5);
        let ordinary = HexCoord::new(6, -1);
        let defender_blocks = HashSet::from([block]);
        let forward = WidePnMove::Pair(block, ordinary);
        let reverse = WidePnMove::Pair(ordinary, block);
        let pair_key = |first: HexCoord, second: HexCoord| {
            let first = raw_coord_key(first);
            let second = raw_coord_key(second);
            if first <= second {
                (first, second)
            } else {
                (second, first)
            }
        };

        assert_eq!(pair_key(block, ordinary), pair_key(ordinary, block));
        assert!(wide_move_contains_defender_block(forward, &defender_blocks));
        assert_eq!(
            wide_move_contains_defender_block(forward, &defender_blocks),
            wide_move_contains_defender_block(reverse, &defender_blocks),
            "either ordering retained by unordered-pair dedup must carry the same urgency"
        );
    }

    #[test]
    fn wide_pn_fork_priors_lead_choice_order_without_breaking_root_commitment() {
        let priors = (0..=MAX_TURN_FORK_DEGREE as usize)
            .map(pn_from_fork_degree)
            .collect::<Vec<_>>();
        assert!(priors.windows(2).all(|pair| pair[0] > pair[1]));
        assert_eq!(priors[0], MAX_TURN_FORK_DEGREE + 1);
        assert_eq!(priors[MAX_TURN_FORK_DEGREE as usize], 1);
        assert_eq!(pn_from_fork_degree(usize::MAX), 1);

        let search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        let child = |q, fork_degree| WidePnChild {
            mv: WidePnMove::One(HexCoord::new(q, 0)),
            result: WidePnChildResult::Pending,
            entry: None,
            future_key: None,
            prior: WidePnPrior {
                pn: pn_from_fork_degree(fork_degree),
                dn: 1,
            },
            urgent_block: false,
            first_width_tier: 0,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };
        let tied = [child(5, 3), child(-5, 3)];
        assert_eq!(
            search.select_child_index(WidePnKind::Choice, &tied, false),
            Some(0)
        );
        assert_eq!(
            search.select_child_index(WidePnKind::Choice, &tied, true),
            Some(0)
        );
        let mixed = [child(5, 1), child(-5, 4)];
        assert_eq!(
            search.select_child_index(WidePnKind::Choice, &mixed, false),
            Some(1)
        );
        assert_eq!(
            search.select_child_index(WidePnKind::Choice, &mixed, true),
            Some(1),
            "immutable fork prior must outrank generator rank"
        );

        let mut urgent = child(-5, 1);
        urgent.urgent_block = true;
        let urgent_after_fork = [child(5, 4), urgent];
        assert_eq!(
            search.select_child_index(WidePnKind::Choice, &urgent_after_fork, true),
            Some(1),
            "urgent root blocks remain ahead of non-urgent forks"
        );

        let mut tactical = child(-6, 0);
        tactical.result = WidePnChildResult::ClaimantTactical;
        let tactical_after_urgent = [urgent_after_fork[1].clone(), tactical];
        assert_eq!(
            search.select_child_index(WidePnKind::Choice, &tactical_after_urgent, true),
            Some(1),
            "a completed tactical leaf is always first"
        );

        let mut refuted = child(6, MAX_TURN_FORK_DEGREE as usize);
        refuted.result = WidePnChildResult::Refuted;
        let refuted_before_live = [refuted, child(-6, 0)];
        assert_eq!(
            search.select_child_index(WidePnKind::Choice, &refuted_before_live, true),
            Some(1),
            "a genuine refutation cannot retain sequential commitment"
        );
    }

    #[test]
    fn ordering_study_offline_pass_reranks_without_mutating_children() {
        begin_ordering_study_report();
        let state = RustHexoState::new();
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        search.ordering_study = true;
        let root = search.insert_root(&state);
        let child = |q, result, ordering| WidePnChild {
            mv: WidePnMove::Pair(HexCoord::new(q, 0), HexCoord::new(q + 1, 0)),
            result,
            entry: None,
            future_key: None,
            prior: WidePnPrior::UNIFORM,
            urgent_block: false,
            first_width_tier: 0,
            zone_order_key: 0,
            ordering,
        };
        let original = vec![
            child(
                0,
                WidePnChildResult::Pending,
                OrderingChildFeatures {
                    zone_bound: 2,
                    census_distance: 3,
                    gate_adjacency: 0,
                    d_stone: 5,
                },
            ),
            child(
                3,
                WidePnChildResult::Pending,
                OrderingChildFeatures {
                    zone_bound: 1,
                    census_distance: 2,
                    gate_adjacency: 1,
                    d_stone: 4,
                },
            ),
            child(
                6,
                WidePnChildResult::ClaimantTactical,
                OrderingChildFeatures {
                    zone_bound: 0,
                    census_distance: 1,
                    gate_adjacency: 2,
                    d_stone: 3,
                },
            ),
        ];
        search.entries[root].node = WidePnNode::Branch {
            kind: WidePnKind::Choice,
            children: original.clone(),
        };
        search.recompute(root);
        search.finalize_ordering_study();
        let report = take_ordering_study_report();
        assert_eq!(report.records.len(), 1);
        assert_eq!(report.records[0].ranks, [3, 1, 1, 1, 1, 1, 1]);
        let WidePnNode::Branch { children, .. } = &search.entries[root].node else {
            panic!("synthetic ordering root must remain a branch")
        };
        assert_eq!(children[0].mv, original[0].mv);
        assert_eq!(children[1].mv, original[1].mv);
        assert_eq!(children[2].mv, original[2].mv);
    }

    #[test]
    fn live_zone_order_respects_pn_band_and_hard_classes() {
        let context = OrderingFeatureContext {
            claimant_stones: vec![HexCoord::new(0, 0)],
            claimant_windows: Vec::new(),
            defender_gate_cells: Vec::new(),
        };
        assert_eq!(
            context.pair_key(
                HexCoord::new(1, 0),
                HexCoord::new(3, 0),
                ZoneOrderMode::ZoneBound,
            ),
            3
        );
        assert_eq!(
            context.pair_key(
                HexCoord::new(1, 0),
                HexCoord::new(3, 0),
                ZoneOrderMode::DStone,
            ),
            1
        );

        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        search.zone_order_mode = ZoneOrderMode::Off;
        let child = |q, prior_pn, zone_order_key| WidePnChild {
            mv: WidePnMove::Pair(HexCoord::new(q, 0), HexCoord::new(q + 1, 0)),
            result: WidePnChildResult::Pending,
            entry: None,
            future_key: None,
            prior: WidePnPrior {
                pn: prior_pn,
                dn: 1,
            },
            urgent_block: false,
            first_width_tier: 0,
            zone_order_key,
            ordering: OrderingChildFeatures::default(),
        };
        let tied = [child(0, 5, 4), child(3, 5, 1)];
        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &tied, false, false),
            Some(0),
            "flag off retains the historical generator tie break"
        );
        search.zone_order_mode = ZoneOrderMode::ZoneBound;
        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &tied, false, false),
            Some(1),
            "the live key breaks an exact PN/prior tie"
        );

        let mut urgent = tied[0].clone();
        urgent.urgent_block = true;
        assert_eq!(
            search.select_child_index_with_tier(
                WidePnKind::Choice,
                &[urgent, tied[1].clone()],
                true,
                false,
            ),
            Some(0),
            "zone distance cannot cross the urgent-root class"
        );
        let mut wider = tied[1].clone();
        wider.first_width_tier = 1;
        assert_eq!(
            search.select_child_index_with_tier(
                WidePnKind::Choice,
                &[tied[0].clone(), wider],
                false,
                true,
            ),
            Some(0),
            "zone distance cannot cross the width class"
        );
        let different_prior = [child(0, 5, 4), child(3, 6, 1)];
        search.zone_order_band = 10;
        assert_eq!(
            search
                .select_child_index_with_tier(WidePnKind::Choice, &different_prior, false, false,),
            Some(0),
            "the numeric band cannot cross an immutable fork-prior class"
        );

        let first_entry = search.entries.len();
        search
            .entries
            .push(wide_pn_test_entry(5, WidePnNode::Unexpanded, 2));
        let second_entry = search.entries.len();
        search
            .entries
            .push(wide_pn_test_entry(6, WidePnNode::Unexpanded, 2));
        let mut linked = tied.clone();
        linked[0].entry = Some(first_entry);
        linked[1].entry = Some(second_entry);
        search.zone_order_band = 0;
        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &linked, false, false),
            Some(0)
        );
        search.zone_order_band = 1;
        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &linked, false, false),
            Some(1),
            "band one admits the adjacent current-PN child inside one hard class"
        );
    }

    #[test]
    fn wide_pn_sequential_root_honors_width_tier_after_urgency() {
        let search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        assert!(search.prefer_width_tier_at_depth(0));
        assert!(!search.prefer_width_tier_at_depth(1));

        let child = |q, prior_pn, first_width_tier| WidePnChild {
            mv: WidePnMove::Pair(HexCoord::new(q, 0), HexCoord::new(q + 1, 0)),
            result: WidePnChildResult::Pending,
            entry: None,
            future_key: None,
            prior: WidePnPrior {
                pn: prior_pn,
                dn: 1,
            },
            urgent_block: false,
            first_width_tier,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };
        let tier_zero = child(0, 20, 0);
        let tier_one = child(3, 3, 1);
        let children = [tier_zero.clone(), tier_one.clone()];

        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &children, true, false),
            Some(1),
            "without the root prior, the lower immutable fork PN leads"
        );
        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &children, true, true),
            Some(0),
            "the sequential root path must prefer a tier-zero first placement"
        );

        let mut urgent_tier_one = tier_one.clone();
        urgent_tier_one.urgent_block = true;
        assert_eq!(
            search.select_child_index_with_tier(
                WidePnKind::Choice,
                &[tier_zero.clone(), urgent_tier_one],
                true,
                true,
            ),
            Some(1),
            "the established urgent-block bootstrap remains ahead of width tier"
        );

        let mut tactical_tier_one = tier_one.clone();
        tactical_tier_one.result = WidePnChildResult::ClaimantTactical;
        assert_eq!(
            search.select_child_index_with_tier(
                WidePnKind::Choice,
                &[tier_zero.clone(), tactical_tier_one],
                true,
                true,
            ),
            Some(1),
            "an already-proven child remains first on the sequential path"
        );

        let mut refuted_tier_zero = tier_zero;
        refuted_tier_zero.result = WidePnChildResult::Refuted;
        assert_eq!(
            search.select_child_index_with_tier(
                WidePnKind::Choice,
                &[refuted_tier_zero, tier_one],
                true,
                true,
            ),
            Some(1),
            "a refuted tier-zero child cannot retain sequential commitment"
        );
    }

    #[test]
    fn wide_pn_persistent_pair_tier_survives_linked_pn_changes() {
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        let tier_zero_entry = search.entries.len();
        search
            .entries
            .push(wide_pn_test_entry(50, WidePnNode::Unexpanded, 2));
        let tier_one_entry = search.entries.len();
        search
            .entries
            .push(wide_pn_test_entry(3, WidePnNode::Unexpanded, 2));
        let pair_child = |first_q, entry, first_width_tier| WidePnChild {
            mv: WidePnMove::Pair(HexCoord::new(first_q, 0), HexCoord::new(first_q + 1, 0)),
            result: WidePnChildResult::Pending,
            entry: Some(entry),
            future_key: None,
            prior: WidePnPrior::UNIFORM,
            urgent_block: false,
            first_width_tier,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };
        let children = [
            pair_child(0, tier_zero_entry, 0),
            pair_child(3, tier_one_entry, 1),
        ];

        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &children, false, false),
            Some(1),
            "profile-off selection remains strict minimum PN"
        );
        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &children, false, true),
            Some(0)
        );

        search.entries[tier_zero_entry].pn = 500;
        let stored = search
            .entries
            .iter()
            .map(|entry| (entry.pn, entry.dn))
            .collect::<Vec<_>>();
        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &children, false, true),
            Some(0),
            "linked PN changes cannot erase the immutable first-placement tier"
        );
        assert_eq!(
            search
                .entries
                .iter()
                .map(|entry| (entry.pn, entry.dn))
                .collect::<Vec<_>>(),
            stored,
            "ordering must not mutate stored PN/DN values"
        );

        let mut ordinary = children.clone();
        for (rank, child) in ordinary.iter_mut().enumerate() {
            child.mv = WidePnMove::One(HexCoord::new(rank as i16, 0));
            child.first_width_tier = 0;
        }
        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &ordinary, false, true),
            Some(1),
            "neutral one-placement children retain minimum-PN ordering"
        );

        let mut terminal = children.clone();
        terminal[1].result = WidePnChildResult::ClaimantTactical;
        terminal[1].entry = None;
        assert_eq!(
            search.select_child_index_with_tier(WidePnKind::Choice, &terminal, false, true),
            Some(1),
            "an already-proven child remains ahead of every unresolved tier"
        );
    }

    #[test]
    fn wide_pn_choice_skips_refuted_child_tied_with_saturated_live_child() {
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        let refuted = search.entries.len();
        search.entries.push(WidePnEntry {
            pn: PN_INFINITY,
            dn: 0,
            prior: WidePnPrior::UNIFORM,
            node: WidePnNode::Refuted,
            depth: 1,
            universal_obligation: None,
        });
        let child = |q, entry| WidePnChild {
            mv: WidePnMove::One(HexCoord::new(q, 0)),
            result: WidePnChildResult::Pending,
            entry,
            future_key: None,
            prior: WidePnPrior {
                pn: PN_INFINITY,
                dn: 1,
            },
            urgent_block: false,
            first_width_tier: 0,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };
        let children = [child(0, Some(refuted)), child(1, None), child(2, None)];

        assert!(search.child_is_genuinely_refuted(&children[0]));
        assert_eq!(search.child_numbers(&children[0]).0, PN_INFINITY);
        assert_eq!(search.child_numbers(&children[1]).0, PN_INFINITY);
        assert_eq!(
            search.select_child_index(WidePnKind::Choice, &children, false),
            Some(1),
            "the first unresolved child wins the saturated live tie"
        );
    }

    #[test]
    fn wide_pn_universal_skips_proven_child_tied_with_saturated_live_child() {
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        let proven = search.entries.len();
        search.entries.push(WidePnEntry {
            pn: 0,
            dn: PN_INFINITY,
            prior: WidePnPrior::UNIFORM,
            node: WidePnNode::ProvenLeaf(cache_test_leaf()),
            depth: 1,
            universal_obligation: None,
        });
        let child = |q, entry| WidePnChild {
            mv: WidePnMove::One(HexCoord::new(q, 0)),
            result: WidePnChildResult::Pending,
            entry,
            future_key: None,
            prior: WidePnPrior {
                pn: 1,
                dn: PN_INFINITY,
            },
            urgent_block: false,
            first_width_tier: 0,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };
        let children = [child(0, Some(proven)), child(1, None), child(2, None)];

        assert!(search.child_is_genuinely_proven(&children[0]));
        assert_eq!(search.child_numbers(&children[0]).1, PN_INFINITY);
        assert_eq!(search.child_numbers(&children[1]).1, PN_INFINITY);
        assert_eq!(
            search.select_child_index(
                WidePnKind::Universal {
                    implicit_dispatch: true,
                },
                &children,
                false,
            ),
            Some(1),
            "the first unresolved child wins the saturated live tie"
        );
    }

    #[test]
    fn wide_pn_universal_obligation_stays_committed_until_verdict() {
        let state = RustHexoState::new();
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        let root = search.insert_root(&state);
        let first = search.entries.len();
        search
            .entries
            .push(wide_pn_test_entry(1, WidePnNode::Unexpanded, 1));
        let second = search.entries.len();
        search
            .entries
            .push(wide_pn_test_entry(1, WidePnNode::Unexpanded, 1));
        let third = search.entries.len();
        search
            .entries
            .push(wide_pn_test_entry(1, WidePnNode::Unexpanded, 1));
        let fourth = search.entries.len();
        search
            .entries
            .push(wide_pn_test_entry(1, WidePnNode::Unexpanded, 1));
        search.entries[first].dn = 2;
        search.entries[second].dn = 5;
        search.entries[third].dn = 7;
        search.entries[fourth].dn = 9;
        let child = |q, entry| WidePnChild {
            mv: WidePnMove::One(HexCoord::new(q, 0)),
            result: WidePnChildResult::Pending,
            entry: Some(entry),
            future_key: None,
            prior: WidePnPrior::UNIFORM,
            urgent_block: false,
            first_width_tier: 0,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };
        search.entries[root].node = WidePnNode::Branch {
            kind: WidePnKind::Universal {
                implicit_dispatch: true,
            },
            children: vec![
                child(0, first),
                child(1, second),
                child(2, third),
                child(3, fourth),
            ],
        };

        assert_eq!(
            search.select_step_child_index(root, false, false, &[]),
            Some(0)
        );
        search.entries[first].dn = 50;
        search.entries[second].dn = 1;
        assert_eq!(
            search.select_step_child_index(root, false, false, &[]),
            Some(0),
            "changing DN estimates cannot interleave a committed obligation"
        );

        search.entries[first].node = WidePnNode::DepthCutoff;
        search.entries[first].pn = PN_INFINITY;
        search.entries[first].dn = 0;
        assert_eq!(
            search.select_step_child_index(root, false, false, &[]),
            Some(0),
            "a staged cutoff stays committed so the reopened stage resumes it"
        );

        search.entries[first].pn = 0;
        search.entries[first].dn = PN_INFINITY;
        search.entries[first].node = WidePnNode::ProvenLeaf(cache_test_leaf());
        assert_eq!(
            search.select_step_child_index(root, false, false, &[]),
            Some(1),
            "a verdict advances a structurally high-fanout Universal"
        );
        assert_eq!(search.entries[root].universal_obligation, Some(1));

        search.entries[second].dn = 50;
        search.entries[third].dn = 1;
        assert_eq!(
            search.select_step_child_index(root, false, false, &[]),
            Some(1),
            "the qualifying Universal remains committed through its binary tail"
        );
        search.entries[second].pn = 0;
        search.entries[second].dn = PN_INFINITY;
        search.entries[second].node = WidePnNode::ProvenLeaf(cache_test_leaf());
        assert_eq!(
            search.select_step_child_index(root, false, false, &[]),
            Some(2),
            "the next lowest-DN unresolved obligation becomes committed"
        );
    }

    #[test]
    fn wide_pn_binary_or_tt_converged_universal_keeps_pn_reselection() {
        let state = RustHexoState::new();
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        let root = search.insert_root(&state);
        let first = search.entries.len();
        search
            .entries
            .push(wide_pn_test_entry(1, WidePnNode::Unexpanded, 1));
        let second = search.entries.len();
        search
            .entries
            .push(wide_pn_test_entry(1, WidePnNode::Unexpanded, 1));
        search.entries[first].dn = 2;
        search.entries[second].dn = 5;
        let child = |q, entry| WidePnChild {
            mv: WidePnMove::One(HexCoord::new(q, 0)),
            result: WidePnChildResult::Pending,
            entry: Some(entry),
            future_key: None,
            prior: WidePnPrior::UNIFORM,
            urgent_block: false,
            first_width_tier: 0,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };
        search.entries[root].node = WidePnNode::Branch {
            kind: WidePnKind::Universal {
                implicit_dispatch: true,
            },
            children: vec![
                child(0, first),
                child(1, second),
                child(2, first),
                child(3, second),
            ],
        };

        assert_eq!(
            search.select_step_child_index(root, false, false, &[]),
            Some(0)
        );
        assert_eq!(search.entries[root].universal_obligation, None);
        search.entries[first].dn = 50;
        search.entries[second].dn = 1;
        assert_eq!(
            search.select_step_child_index(root, false, false, &[]),
            Some(1),
            "four edges converging to two TT entries remain a binary conjunction"
        );
        assert_eq!(search.entries[root].universal_obligation, None);

        assert_eq!(
            search.select_step_child_index_with_commitment(root, false, false, &[], true),
            Some(1),
            "an inherited high-fanout domain commits its descendant conjunction"
        );
        search.entries[first].dn = 1;
        search.entries[second].dn = 50;
        assert_eq!(
            search.select_step_child_index_with_commitment(root, false, false, &[], true),
            Some(1),
            "descendant DN changes cannot escape the inherited obligation domain"
        );
    }

    #[test]
    fn wide_pn_stalled_universal_obligation_yields_once_per_sibling() {
        let state = RustHexoState::new();
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        let root = search.insert_root(&state);
        let entries = (0..4)
            .map(|_| {
                let entry = search.entries.len();
                search
                    .entries
                    .push(wide_pn_test_entry(1, WidePnNode::Unexpanded, 1));
                entry
            })
            .collect::<Vec<_>>();
        let children = entries
            .into_iter()
            .enumerate()
            .map(|(q, entry)| WidePnChild {
                mv: WidePnMove::One(HexCoord::new(i16::try_from(q).unwrap(), 0)),
                result: WidePnChildResult::Pending,
                entry: Some(entry),
                future_key: None,
                prior: WidePnPrior::UNIFORM,
                urgent_block: false,
                first_width_tier: 0,
                zone_order_key: 0,
                ordering: OrderingChildFeatures::default(),
            })
            .collect();
        search.entries[root].node = WidePnNode::Branch {
            kind: WidePnKind::Universal {
                implicit_dispatch: true,
            },
            children,
        };

        assert_eq!(
            search.select_step_child_index(root, false, false, &[]),
            Some(0)
        );
        assert_eq!(
            search.select_step_child_index(root, false, false, &[0]),
            Some(1)
        );
        assert_eq!(
            search.select_step_child_index(root, false, false, &[0, 1]),
            Some(2)
        );
        assert_eq!(
            search.select_step_child_index(root, false, false, &[0, 1, 2]),
            Some(3)
        );
        assert_eq!(
            search.select_step_child_index(root, false, false, &[0, 1, 2, 3]),
            None,
            "all-stalled obligations terminate instead of cycling"
        );
        assert_eq!(search.entries[root].universal_obligation, None);
    }

    #[test]
    fn wide_pn_all_finished_parents_recompute_to_terminal() {
        let state = RustHexoState::new();
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        let root = search.insert_root(&state);
        let child = |q, result| WidePnChild {
            mv: WidePnMove::One(HexCoord::new(q, 0)),
            result,
            entry: None,
            future_key: None,
            prior: WidePnPrior::UNIFORM,
            urgent_block: false,
            first_width_tier: 0,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };

        let refuted = vec![
            child(0, WidePnChildResult::Refuted),
            child(1, WidePnChildResult::Refuted),
        ];
        search.entries[root].node = WidePnNode::Branch {
            kind: WidePnKind::Choice,
            children: refuted.clone(),
        };
        search.recompute(root);
        assert_eq!(
            (search.entries[root].pn, search.entries[root].dn),
            (PN_INFINITY, 0)
        );
        assert_eq!(
            search.select_child_index(WidePnKind::Choice, &refuted, false),
            None
        );
        let mut work = state.clone();
        assert_eq!(search.step(&mut work, root), WidePnStepOutcome::Stalled);

        let proven = vec![
            child(0, WidePnChildResult::ClaimantTactical),
            child(1, WidePnChildResult::ClaimantCompletion),
        ];
        search.entries[root].node = WidePnNode::Branch {
            kind: WidePnKind::Universal {
                implicit_dispatch: true,
            },
            children: proven.clone(),
        };
        search.recompute(root);
        assert_eq!(
            (search.entries[root].pn, search.entries[root].dn),
            (0, PN_INFINITY)
        );
        assert_eq!(
            search.select_child_index(
                WidePnKind::Universal {
                    implicit_dispatch: true,
                },
                &proven,
                false,
            ),
            None
        );
        assert_eq!(search.step(&mut work, root), WidePnStepOutcome::Stalled);
    }

    #[test]
    fn wide_staging_depth_and_advance_are_semantic_and_monotonic() {
        assert_eq!(wide_search_final_depth(40, 39), 0);
        assert_eq!(wide_search_final_depth(40, 40), 0);
        assert_eq!(wide_search_final_depth(40, 41), 1);
        assert_eq!(wide_search_final_depth(40, u32::MAX), MAX_SEARCH_DEPTH);

        assert_eq!(next_wide_stage_depth(0, 7, MAX_SEARCH_DEPTH), Some(7));
        assert_eq!(next_wide_stage_depth(7, 42, MAX_SEARCH_DEPTH), Some(42));
        assert_eq!(
            next_wide_stage_depth(42, MAX_SEARCH_DEPTH + 10, MAX_SEARCH_DEPTH),
            Some(MAX_SEARCH_DEPTH)
        );
        assert_eq!(
            next_wide_stage_depth(MAX_SEARCH_DEPTH, MAX_SEARCH_DEPTH + 1, MAX_SEARCH_DEPTH,),
            None,
            "the hard final depth terminates instead of reopening an inadmissible cutoff"
        );
        assert_eq!(next_wide_stage_depth(7, 7, 20), None);
        assert_eq!(next_wide_stage_depth(8, 7, 20), None);
    }

    #[test]
    fn wide_staging_final_cutoffs_explore_siblings_and_fail_closed() {
        let mut state = RustHexoState::new();
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::ZERO,
            },
        )
        .unwrap();
        let claimant = state.current_player();
        let mut search = WidePnSearch::new(claimant, state.placements_made(), 10, 0, 100, 0);
        let root = search.insert_root(&state);
        search.entries[root].node = WidePnNode::Branch {
            kind: WidePnKind::Choice,
            children: vec![
                WidePnChild {
                    mv: WidePnMove::One(HexCoord::new(3, 0)),
                    result: WidePnChildResult::Pending,
                    entry: None,
                    future_key: None,
                    prior: WidePnPrior { pn: 1, dn: 1 },
                    urgent_block: false,
                    first_width_tier: 0,
                    zone_order_key: 0,
                    ordering: OrderingChildFeatures::default(),
                },
                WidePnChild {
                    mv: WidePnMove::One(HexCoord::new(6, 0)),
                    result: WidePnChildResult::Pending,
                    entry: None,
                    future_key: None,
                    prior: WidePnPrior { pn: 2, dn: 1 },
                    urgent_block: false,
                    first_width_tier: 0,
                    zone_order_key: 0,
                    ordering: OrderingChildFeatures::default(),
                },
            ],
        };
        search.recompute(root);

        search.run(&state, root);

        assert_eq!(search.depth_cap, 0);
        assert_eq!(search.expansions, 2);
        let WidePnNode::Branch { children, .. } = &search.entries[root].node else {
            panic!("synthetic root remains branched")
        };
        assert!(children.iter().all(|child| {
            child
                .entry
                .and_then(|id| search.entries.get(id))
                .is_some_and(|entry| matches!(entry.node, WidePnNode::DepthCutoff))
        }));
        assert_ne!(search.entries[root].pn, 0);
        assert!(search.materialize(&state, root).is_none());
    }

    #[test]
    fn wide_pn_selected_descendant_cutoff_advances_and_reopens() {
        let state = RustHexoState::new();
        let claimant = state.current_player();
        let moves = [
            HexCoord::new(0, 0),
            HexCoord::new(3, 0),
            HexCoord::new(6, 0),
            HexCoord::new(9, 0),
            HexCoord::new(12, 0),
            HexCoord::new(15, 0),
            HexCoord::new(18, 0),
        ];
        let mut states = vec![state.clone()];
        for &coord in &moves {
            let mut next = states.last().expect("root state exists").clone();
            apply_placement(&mut next, Placement { coord }).unwrap();
            states.push(next);
        }

        let mut search = WidePnSearch::new(claimant, 0, 100, 0, 100, 6);
        let root = search.insert_root(&state);
        let mut ids = vec![root];
        for depth in 1..=6 {
            let prior = search.position_prior(&states[depth]);
            ids.push(search.insert_position(
                WidePositionKey::from_state(&states[depth]),
                depth,
                prior,
            ));
        }
        for depth in 0..=6 {
            let kind = if states[depth].current_player() == claimant {
                WidePnKind::Choice
            } else {
                WidePnKind::Universal {
                    implicit_dispatch: true,
                }
            };
            let prior = search.position_prior(&states[depth + 1]);
            let entry = if depth < 6 {
                Some(ids[depth + 1])
            } else {
                None
            };
            search.entries[ids[depth]].node = WidePnNode::Branch {
                kind,
                children: vec![WidePnChild {
                    mv: WidePnMove::One(moves[depth]),
                    result: WidePnChildResult::Pending,
                    entry,
                    future_key: None,
                    prior,
                    urgent_block: false,
                    first_width_tier: 0,
                    zone_order_key: 0,
                    ordering: OrderingChildFeatures::default(),
                }],
            };
        }
        for &id in ids.iter().rev() {
            search.recompute(id);
        }

        assert_eq!(search.run_until(&state, root, 100, true), Some(7));
        assert_eq!(search.expansions, 1);
        let cutoff = match &search.entries[ids[6]].node {
            WidePnNode::Branch { children, .. } => {
                children[0].entry.expect("last edge linked its child")
            }
            _ => panic!("synthetic path remains branched"),
        };
        assert_eq!(search.entries[cutoff].depth, 7);
        assert!(matches!(
            search.entries[cutoff].node,
            WidePnNode::DepthCutoff
        ));

        search.depth_cap = 7;
        search.reopen_depth_cutoffs(7);
        assert!(matches!(
            search.entries[cutoff].node,
            WidePnNode::Unexpanded
        ));
        assert!(search.entries[root].dn > 0);

        let root_key = PositionKey::from_state(&state);
        let mut replay = state.clone();
        assert_eq!(search.step(&mut replay, root), WidePnStepOutcome::Progress);
        assert_eq!(PositionKey::from_state(&replay), root_key);
    }

    #[test]
    fn wide_pn_completed_turn_child_carries_fork_and_tau_priors() {
        let mut state = pair_width_first_stone_fixture();
        let claimant = state.current_player();
        for coord in [HexCoord::new(2, 0), HexCoord::new(3, 0)] {
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        let analysis = threats::analyze(&state);
        assert_eq!(analysis.min_hitting_set, Some(2));
        assert_eq!(analysis.opp_threat_count, 3);

        let search = WidePnSearch::new(claimant, 0, 10, 0, 100, 10);
        let prior = search.completed_turn_prior(&state);
        assert_eq!(prior.pn, pn_from_fork_degree(analysis.opp_threat_count));
        assert_eq!(prior.dn, 2);
        assert_eq!(search.position_prior(&state).dn, 2);

        let lazy = WidePnChild {
            mv: WidePnMove::Pair(HexCoord::new(2, 0), HexCoord::new(3, 0)),
            result: WidePnChildResult::Pending,
            entry: None,
            future_key: None,
            prior,
            urgent_block: false,
            first_width_tier: 0,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };
        assert_eq!(search.child_numbers(&lazy), (prior.pn, prior.dn));
    }

    #[test]
    fn wide_pn_entry_prior_survives_recompute_and_depth_reopen() {
        let state = RustHexoState::new();
        let prior = WidePnPrior { pn: 7, dn: 2 };
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        let id = search.insert_position(WidePositionKey::from_state(&state), 3, prior);

        search.recompute(id);
        assert_eq!((search.entries[id].pn, search.entries[id].dn), (7, 2));
        search.entries[id].node = WidePnNode::DepthCutoff;
        search.recompute(id);
        assert_eq!(
            (search.entries[id].pn, search.entries[id].dn),
            (PN_INFINITY, 0)
        );

        search.reopen_depth_cutoffs(3);
        assert!(matches!(search.entries[id].node, WidePnNode::Unexpanded));
        assert_eq!((search.entries[id].pn, search.entries[id].dn), (7, 2));
        search.recompute(id);
        assert_eq!((search.entries[id].pn, search.entries[id].dn), (7, 2));
    }

    #[test]
    fn wide_lazy_frontier_distinguishes_virtual_and_selection_only_keys() {
        let state = RustHexoState::new();
        let key = WidePositionKey::from_state(&state);
        let first_prior = WidePnPrior { pn: 9, dn: 4 };
        let edge_prior = WidePnPrior { pn: 3, dn: 2 };
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 1 << 20, 100, 10);
        search.lazy_frontier = true;
        search.defer_position(&key, 4, first_prior);

        let child = |future_key| WidePnChild {
            mv: WidePnMove::One(HexCoord::new(1, 0)),
            result: WidePnChildResult::Pending,
            entry: None,
            future_key: Some(future_key),
            prior: edge_prior,
            urgent_block: false,
            first_width_tier: 0,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };
        let virtual_child = child(WideFutureKey::Virtual(key.clone()));
        let attacker_child = child(WideFutureKey::OnSelection(key.clone()));

        assert_eq!(search.child_numbers(&virtual_child), (9, 4));
        assert_eq!(search.child_numbers(&attacker_child), (3, 2));
        assert!(search.entries.is_empty());
        assert!(search.by_position.is_empty());

        let id = search.insert_position(key, 99, edge_prior);
        assert_eq!(search.entries[id].depth, 4);
        assert_eq!(search.entries[id].prior, first_prior);
        assert!(search.deferred_by_position.is_empty());
        assert_eq!(search.child_numbers(&virtual_child), (9, 4));
        assert_eq!(search.child_numbers(&attacker_child), (3, 2));
    }

    #[test]
    fn wide_pn_trace_child_format_is_compact_and_structural() {
        let state = RustHexoState::new();
        let prior = WidePnPrior { pn: 7, dn: 2 };
        let mut search = WidePnSearch::new(Player::Player0, 0, 10, 0, 100, 10);
        let entry = search.insert_position(WidePositionKey::from_state(&state), 7, prior);
        search.entries[entry].node = WidePnNode::DepthCutoff;
        search.recompute(entry);
        let child = WidePnChild {
            mv: WidePnMove::One(HexCoord::new(3, -2)),
            result: WidePnChildResult::Pending,
            entry: Some(entry),
            future_key: None,
            prior,
            urgent_block: true,
            first_width_tier: 0,
            zone_order_key: 0,
            ordering: OrderingChildFeatures::default(),
        };

        assert_eq!(
            search.format_trace_child(&child),
            format!(
                "pn={PN_INFINITY} dn=0 prior_pn=7 prior_dn=2 result=Pending urgent=true first_tier=0 entry={entry} entry_depth=7 entry_node=depth_cutoff cutoff=true"
            )
        );
    }

    #[test]
    fn wide_pn_zero_tt_cap_keeps_unindexed_frontier_progress() {
        let mut state = pair_width_first_stone_fixture();
        let claimant = state.current_player();
        let root_ply = state.placements_made();
        let mut search = WidePnSearch::new(claimant, root_ply, 16, 0, u32::MAX, 10);
        let root = search.insert_root(&state);
        assert!(search.by_position.is_empty());
        assert_eq!(search.current_bytes, 0);
        assert_eq!(search.peak_bytes, 0);

        let mv = HexCoord::new(2, 0);
        let mut child_state = state.clone();
        apply_placement(&mut child_state, Placement { coord: mv }).unwrap();
        let prior = search.position_prior(&child_state);
        search.entries[root].node = WidePnNode::Branch {
            kind: WidePnKind::Choice,
            children: vec![WidePnChild {
                mv: WidePnMove::One(mv),
                result: WidePnChildResult::Pending,
                entry: None,
                future_key: None,
                prior,
                urgent_block: false,
                first_width_tier: 0,
                zone_order_key: 0,
                ordering: OrderingChildFeatures::default(),
            }],
        };
        search.recompute(root);

        let root_key = PositionKey::from_state(&state);
        assert_eq!(search.step(&mut state, root), WidePnStepOutcome::Progress);
        assert_eq!(PositionKey::from_state(&state), root_key);
        assert!(search.expansions > 0);
        assert!(search.by_position.is_empty());
        assert_eq!(search.current_bytes, 0);
        assert_eq!(search.peak_bytes, 0);
    }

    #[test]
    fn wide_pn_full_tt_retains_indexed_hits_and_falls_back_unindexed() {
        let state = pair_width_first_stone_fixture();
        let claimant = state.current_player();
        let root_key = WidePositionKey::from_state(&state);
        let tt_cap = wide_position_index_bytes(&root_key);
        let mut search =
            WidePnSearch::new(claimant, state.placements_made(), 16, tt_cap, u32::MAX, 10);
        let root = search.insert_root(&state);
        assert_eq!(search.by_position.len(), 1);
        assert_eq!(search.current_bytes, tt_cap);
        assert_eq!(search.peak_bytes, tt_cap);

        let root_prior = search.position_prior(&state);
        let reused = search.insert_position(root_key, 0, root_prior);
        assert_eq!(reused, root);
        assert_eq!(search.entries.len(), 1);
        assert_eq!(search.tt_hits, 1);

        let mut child_state = state.clone();
        apply_placement(
            &mut child_state,
            Placement {
                coord: HexCoord::new(2, 0),
            },
        )
        .unwrap();
        let child_key = WidePositionKey::from_state(&child_state);
        let child_prior = search.position_prior(&child_state);
        let first = search.insert_position(child_key.clone(), 1, child_prior);
        let second = search.insert_position(child_key, 1, child_prior);
        assert_ne!(
            first, second,
            "unindexed positions are arena-local tree nodes"
        );
        assert_eq!(search.entries.len(), 3);
        assert_eq!(search.by_position.len(), 1);
        assert!(search.current_bytes <= search.tt_bytes_cap);
        assert!(search.peak_bytes <= search.tt_bytes_cap);
        assert_eq!(search.current_bytes, tt_cap);
        assert_eq!(search.peak_bytes, tt_cap);
    }

    #[test]
    fn wide_pn_saturated_tt_links_pending_attacker_child_without_stall() {
        let mut state = pair_width_first_stone_fixture();
        let claimant = state.current_player();
        let root_key = WidePositionKey::from_state(&state);
        let tt_cap = wide_position_index_bytes(&root_key);
        let mut search =
            WidePnSearch::new(claimant, state.placements_made(), 16, tt_cap, u32::MAX, 10);
        let root = search.insert_root(&state);

        let mv = HexCoord::new(2, 0);
        let mut child_state = state.clone();
        apply_placement(&mut child_state, Placement { coord: mv }).unwrap();
        let prior = search.position_prior(&child_state);
        search.entries[root].node = WidePnNode::Branch {
            kind: WidePnKind::Choice,
            children: vec![WidePnChild {
                mv: WidePnMove::One(mv),
                result: WidePnChildResult::Pending,
                entry: None,
                future_key: None,
                prior,
                urgent_block: false,
                first_width_tier: 0,
                zone_order_key: 0,
                ordering: OrderingChildFeatures::default(),
            }],
        };
        search.recompute(root);

        assert_eq!(search.step(&mut state, root), WidePnStepOutcome::Progress);
        let WidePnNode::Branch { children, .. } = &search.entries[root].node else {
            panic!("synthetic root must remain a branch")
        };
        assert_eq!(children[0].result, WidePnChildResult::Pending);
        assert!(children[0].entry.is_some());
        assert!(search.expansions > 0);
        assert_eq!(search.by_position.len(), 1);
        assert!(search.current_bytes <= search.tt_bytes_cap);
        assert!(search.peak_bytes <= search.tt_bytes_cap);
    }

    #[test]
    fn l13_sparse_obstructions_are_tight_on_triangle_and_c5() {
        let vertices = (0..5).map(|q| HexCoord::new(q, 0)).collect::<Vec<_>>();
        let triangle = vec![
            vec![vertices[0], vertices[1]],
            vec![vertices[1], vertices[2]],
            vec![vertices[2], vertices[0]],
        ];
        assert_eq!(
            inclusion_minimal_loss_obstruction(&triangle, 1),
            Some(vec![0, 1, 2])
        );

        let c5 = (0..5)
            .map(|index| vec![vertices[index], vertices[(index + 1) % 5]])
            .collect::<Vec<_>>();
        assert_eq!(
            inclusion_minimal_loss_obstruction(&c5, 2),
            Some(vec![0, 1, 2, 3, 4])
        );
        assert_eq!(inclusion_minimal_loss_obstruction(&c5, 3), None);
    }

    #[test]
    fn root_loss_certificates_keep_default_family_and_sparse_wide_family() {
        let mut second_stone = forced_loss_fixture();
        apply_placement(
            &mut second_stone,
            Placement {
                coord: HexCoord::new(-8, 0),
            },
        )
        .unwrap();

        for state in [forced_loss_fixture(), second_stone] {
            let analysis = threats::analyze(&state);
            assert!(analysis.forced_loss());
            let winner = state.current_player().other();
            let cap = match analysis.b {
                1 => L13_LOSS_WITNESS_CAP_B1,
                2 => L13_LOSS_WITNESS_CAP_B2,
                _ => panic!("unexpected defender budget"),
            };
            let mut full_family = state
                .board()
                .windows()
                .threats()
                .filter_map(|(owner, entry)| (owner == winner).then_some(entry.key()))
                .collect::<Vec<_>>();
            full_family.sort_by_key(|key| window_key_order(*key));
            full_family.dedup();

            let caps = SolveCaps {
                node_cap: 1,
                tt_bytes_cap: 0,
                semantic_horizon: u32::MAX,
            };
            let narrow = TssSolver::default().solve(&state, &caps);
            assert_eq!(narrow.status, ProofStatus::Loss);
            let narrow_cert = narrow.cert.expect("narrow root LOSS certificate");
            let CertNode::Loss {
                witnesses: narrow_witnesses,
                ..
            } = &narrow_cert.nodes[narrow_cert.root_node as usize]
            else {
                panic!("root lambda loss must materialize a LOSS leaf")
            };
            assert_eq!(narrow_witnesses, &full_family);
            assert!(TssVerifier.verify(&state, &narrow_cert, ProofStatus::Loss));

            let mut wide_solver = TssSolver::default();
            wide_solver.set_width_options(WidthOptions::vcf_pair_complete());
            let wide = wide_solver.solve(&state, &caps);
            assert_eq!(wide.status, ProofStatus::Loss);
            let wide_cert = wide.cert.expect("wide root LOSS certificate");
            let CertNode::Loss {
                witnesses: wide_witnesses,
                ..
            } = &wide_cert.nodes[wide_cert.root_node as usize]
            else {
                panic!("root lambda loss must materialize a LOSS leaf")
            };
            assert!(wide_witnesses.len() <= cap);
            assert!(wide_witnesses
                .iter()
                .all(|witness| full_family.contains(witness)));
            assert!(TssVerifier.verify(&state, &wide_cert, ProofStatus::Loss));

            let sparse_sets = wide_witnesses
                .iter()
                .map(|witness| {
                    state
                        .board()
                        .windows()
                        .entries()
                        .find(|entry| entry.key() == *witness)
                        .expect("certificate witness remains live")
                        .empty_cells()
                })
                .collect::<Vec<_>>();
            assert!(family_hitting_exceeds_budget(&sparse_sets, analysis.b));
            for removed in 0..sparse_sets.len() {
                let trial = sparse_sets
                    .iter()
                    .enumerate()
                    .filter_map(|(index, set)| (index != removed).then_some(set.clone()))
                    .collect::<Vec<_>>();
                assert!(!family_hitting_exceeds_budget(&trial, analysis.b));
            }
        }
    }

    #[test]
    fn extendable_hit_kernel_matches_k1_and_k2_algebra() {
        let a = HexCoord::new(0, 0);
        let b = HexCoord::new(1, 0);
        let c = HexCoord::new(2, 0);
        let z = HexCoord::new(3, 0);

        let k1_family = vec![vec![a, b], vec![a, c]];
        assert_eq!(extendable_hit_kernel_for_family(&k1_family, 1), vec![a]);

        let k2_family = vec![vec![a, z], vec![a, c], vec![b]];
        assert_eq!(extendable_hit_kernel_for_family(&k2_family, 2), vec![a, b]);
    }

    #[test]
    fn xsnfyll_forced_defender_uses_k2_only_in_wide_mode() {
        let mut state = xsnfyll_forced_defender_fixture();
        let claimant = Player::Player1;
        let analysis = threats::analyze(&state);
        assert_eq!(state.current_player(), claimant.other());
        assert_eq!(analysis.b, 2);
        assert_eq!(analysis.min_hitting_set, Some(analysis.b));

        let universe = vec![
            HexCoord::new(1, -7),
            HexCoord::new(1, -6),
            HexCoord::new(1, -1),
            HexCoord::new(3, -5),
            HexCoord::new(4, -6),
        ];
        let kernel = vec![
            HexCoord::new(1, -6),
            HexCoord::new(3, -5),
            HexCoord::new(4, -6),
        ];
        assert_eq!(hitting_universe(&state, claimant), universe);
        assert_eq!(extendable_hit_kernel(&state, claimant, analysis.b), kernel);
        assert_eq!(
            forced_defender_replies(&state, claimant, analysis.b, WidthOptions::default()),
            universe
        );
        assert_eq!(
            forced_defender_replies(
                &state,
                claimant,
                analysis.b,
                WidthOptions::vcf_pair_complete(),
            ),
            kernel
        );

        for omitted in universe.iter().filter(|cell| !kernel.contains(cell)) {
            let (_result, delta) = state
                .apply_with_delta(Placement { coord: *omitted })
                .unwrap();
            assert!(
                threats::analyze(&state).forced_loss(),
                "nonkernel reply {omitted:?} must leave no live defense"
            );
            state.undo(delta);
        }

        let mut search = WidePnSearch::new(claimant, state.placements_made(), 100, 0, 100, 10);
        let children = search.defender_children(&mut state, analysis.b);
        let mut child_moves = children
            .iter()
            .map(|child| match child.mv {
                WidePnMove::One(coord) => coord,
                WidePnMove::Pair(_, _) | WidePnMove::DefenderPair(_, _) => {
                    panic!("default defender child must be one placement")
                }
            })
            .collect::<Vec<_>>();
        child_moves.sort_by_key(|coord| (coord.q, coord.r));
        assert_eq!(child_moves, kernel);
    }

    #[test]
    fn forced_defender_pairs_collapse_symmetric_k2_orders() {
        let mut state = xsnfyll_forced_defender_fixture();
        let claimant = Player::Player1;
        let plan = forced_defender_pair_plan(&mut state, claimant)
            .expect("the exact K2 fixture has symmetric forced second replies");
        let kernel = extendable_hit_kernel(&state, claimant, 2);
        assert_eq!(
            plan.kernel.iter().copied().collect::<HashSet<_>>(),
            kernel.iter().copied().collect::<HashSet<_>>()
        );
        assert!(!plan.pairs.is_empty());
        for pair in &plan.pairs {
            assert!(raw_coord_key(pair.first) < raw_coord_key(pair.second));
            assert!(plan.kernel.contains(&pair.first));
            assert!(plan.kernel.contains(&pair.second));

            let mut forward = state.clone();
            apply_placement(&mut forward, Placement { coord: pair.first }).unwrap();
            apply_placement(&mut forward, Placement { coord: pair.second }).unwrap();
            let mut reverse = state.clone();
            apply_placement(&mut reverse, Placement { coord: pair.second }).unwrap();
            apply_placement(&mut reverse, Placement { coord: pair.first }).unwrap();
            assert_eq!(
                WidePositionKey::from_state(&forward),
                WidePositionKey::from_state(&reverse)
            );
            assert_eq!(WidePositionKey::from_state(&forward), pair.final_key);
        }

        let mut search = WidePnSearch::new(claimant, state.placements_made(), 100, 0, 100, 10);
        let children = search
            .defender_pair_children(&mut state)
            .expect("the symmetric fixture must retain atomic children");
        assert_eq!(children.len(), plan.pairs.len());
        assert!(children.iter().all(|child| {
            child.entry.is_some()
                && child.result == WidePnChildResult::Pending
                && matches!(child.mv, WidePnMove::DefenderPair(_, _))
        }));
    }

    #[test]
    fn defender_pair_profile_falls_back_to_ordered_children_when_plan_is_unsupported() {
        // Player1 has two independent one-cell wins at (-2,0) and (-1,0),
        // so Player0 must spend both replies there. Those two cells also finish
        // Player0's own horizontal six; after either first reply the atomic
        // planner must reject the turn because the defender is win-now.
        let mut state = replay(&[
            (0, 0),
            (-2, 1),
            (-1, 1),
            (1, 0),
            (2, 0),
            (-2, 2),
            (-1, 2),
            (3, 0),
            (-2, 6),
            (-2, 3),
            (-1, 3),
            (-1, 6),
            (8, 3),
            (-2, 4),
            (-1, 4),
            (6, -3),
            (8, -5),
            (-2, 5),
            (-1, 5),
        ]);
        let claimant = Player::Player1;
        let analysis = threats::analyze(&state);
        assert_eq!(state.current_player(), claimant.other());
        assert_eq!(state.phase(), TurnPhase::FirstStone);
        assert_eq!(analysis.b, 2);
        assert_eq!(analysis.min_hitting_set, Some(2));
        assert!(forced_defender_pair_plan(&mut state, claimant).is_none());

        let mut expected = forced_defender_replies(
            &state,
            claimant,
            analysis.b,
            WidthOptions::vcf_pair_complete(),
        );
        expected.sort_by_key(|coord| raw_coord_key(*coord));

        let mut search = WidePnSearch::new(claimant, state.placements_made(), 100, 0, 100, 10);
        let children = search.defender_boundary_children(&mut state, analysis.b);
        let mut actual = children
            .iter()
            .map(|child| match child.mv {
                WidePnMove::One(coord) => coord,
                WidePnMove::Pair(_, _) | WidePnMove::DefenderPair(_, _) => {
                    panic!("unsupported atomic shape must use ordinary defender moves")
                }
            })
            .collect::<Vec<_>>();
        actual.sort_by_key(|coord| raw_coord_key(*coord));
        assert!(!actual.is_empty());
        assert_eq!(actual, expected);
    }

    #[test]
    fn forced_defender_pairs_materialize_checked_commutations() {
        let mut state = xsnfyll_forced_defender_fixture();
        let claimant = Player::Player1;
        let plan = forced_defender_pair_plan(&mut state, claimant).unwrap();
        let mut search = WidePnSearch::new(
            claimant,
            state.placements_made(),
            10_000,
            64 << 20,
            u32::MAX,
            64,
        );
        let root = search.insert_root(&state);
        search.run(&state, root);
        assert_eq!(search.entries[root].pn, 0);
        let materialized = search
            .materialize(&state, root)
            .expect("collapsed defender proof must materialize");
        let cert = TssCertificate {
            root: RootBinding::from_state(&state),
            claimant,
            root_node: materialized.root_node,
            nodes: materialized.arena,
            semantic_horizon: u32::MAX,
        };
        let CertNode::Universal {
            edges,
            commutations,
            implicit_dispatch,
            ..
        } = &cert.nodes[cert.root_node as usize]
        else {
            panic!("pair plan must materialize as a Universal root")
        };
        assert!(*implicit_dispatch);
        assert_eq!(edges.len(), plan.kernel.len());
        assert_eq!(commutations.len(), plan.pairs.len());
        for item in commutations {
            assert!(raw_coord_key(item.omitted_second) < raw_coord_key(item.first));
            let CertNode::Universal {
                edges: first_replies,
                ..
            } = &cert.nodes[item.first_child as usize]
            else {
                panic!("first child must be the nested Universal")
            };
            let CertNode::Universal {
                edges: mirror_replies,
                ..
            } = &cert.nodes[item.mirror_child as usize]
            else {
                panic!("mirror child must be the nested Universal")
            };
            assert!(!first_replies
                .iter()
                .any(|edge| edge.mv == item.omitted_second));
            assert!(mirror_replies.iter().any(|edge| edge.mv == item.first));
        }
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(TssVerifier.verify_with_dispatch_oracle(&state, &cert, ProofStatus::Loss));
    }

    #[test]
    fn xsnfyll_kernel_certificate_accepts_supersets_and_rejects_missing_kernel() {
        let state = xsnfyll_forced_defender_fixture();
        let claimant = Player::Player1;
        let analysis = threats::analyze(&state);
        let universe = hitting_universe(&state, claimant);
        let kernel = extendable_hit_kernel(&state, claimant, analysis.b);
        let mut solver = TssSolver::default();
        let attempt = solver.prove_for_wide_pn(
            &state,
            claimant,
            10_000,
            64 << 20,
            u32::MAX,
            64,
            WidthOptions::vcf_pair_complete(),
            false,
        );
        let cert = attempt.cert.expect("xsnfyll continuation must prove");
        assert!(attempt.stats.nodes < 10_000);
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(TssVerifier.verify_with_dispatch_oracle(&state, &cert, ProofStatus::Loss));
        let CertNode::Universal { edges, .. } = &cert.nodes[cert.root_node as usize] else {
            panic!("forced defender certificate root must be universal")
        };
        let mut root_moves = edges.iter().map(|edge| edge.mv).collect::<Vec<_>>();
        root_moves.sort_by_key(|coord| (coord.q, coord.r));
        assert_eq!(root_moves, kernel);

        let mut full_universe = cert.clone();
        for mv in universe.iter().filter(|mv| !kernel.contains(mv)) {
            let mut child_state = state.clone();
            apply_placement(&mut child_state, Placement { coord: *mv }).unwrap();
            let child_analysis = threats::analyze(&child_state);
            let leaf = typed_lambda_leaf(
                &child_state,
                claimant,
                &child_analysis,
                WidthOptions::vcf_pair_complete(),
            )
            .expect("omitted nonkernel reply must have a lambda-one leaf");
            let child = u32::try_from(full_universe.nodes.len()).unwrap();
            full_universe.nodes.push(leaf);
            let CertNode::Universal { edges, .. } =
                &mut full_universe.nodes[full_universe.root_node as usize]
            else {
                unreachable!()
            };
            edges.push(CertEdge { mv: *mv, child });
        }
        assert!(TssVerifier.verify(&state, &full_universe, ProofStatus::Loss));
        assert!(TssVerifier.verify_with_dispatch_oracle(&state, &full_universe, ProofStatus::Loss));

        let mut missing_kernel = cert;
        let CertNode::Universal { edges, .. } =
            &mut missing_kernel.nodes[missing_kernel.root_node as usize]
        else {
            unreachable!()
        };
        edges.retain(|edge| edge.mv != kernel[0]);
        assert!(!TssVerifier.verify(&state, &missing_kernel, ProofStatus::Loss));
    }

    #[test]
    fn pair_complete_width_keeps_defender_threat_blocks() {
        let state = forced_defense_fixture();
        let claimant = state.current_player();
        let defender = claimant.other();
        let blocks = hitting_universe(&state, defender);
        assert!(!blocks.is_empty());
        let candidates = ordered_threat_creating_moves_with_width(
            &state,
            claimant,
            WidthOptions::vcf_pair_complete(),
        );
        for block in blocks {
            assert!(candidates
                .iter()
                .any(|candidate| candidate.coord == block && candidate.defender_block));
        }
    }

    #[test]
    fn pair_complete_turn_forcing_requires_a_new_post_pair_threat() {
        let mut second_stone = pair_width_first_stone_fixture();
        apply_placement(
            &mut second_stone,
            Placement {
                coord: HexCoord::new(2, 0),
            },
        )
        .unwrap();
        assert!(matches!(
            second_stone.phase(),
            TurnPhase::SecondStone { .. }
        ));
        let claimant = second_stone.current_player();
        let first = match second_stone.phase() {
            TurnPhase::SecondStone { first } => first,
            _ => unreachable!(),
        };

        let mut forcing = second_stone.clone();
        let forcing_second = HexCoord::new(3, 0);
        apply_placement(
            &mut forcing,
            Placement {
                coord: forcing_second,
            },
        )
        .unwrap();
        assert!(turn_created_claimant_threat(
            &forcing,
            claimant,
            first,
            forcing_second
        ));
        let forcing_analysis = threats::analyze(&forcing);
        assert_eq!(forcing_analysis.b, 2);
        assert!(!forcing_analysis.own_win_now);
        assert_eq!(forcing_analysis.opp_threat_count, 3);
        assert_eq!(forcing_analysis.min_hitting_set, Some(forcing_analysis.b));
        assert!(turn_forces_small_defender_reply(&forcing, claimant));

        let mut loose = second_stone.clone();
        let loose_second = HexCoord::new(5, 0);
        apply_placement(
            &mut loose,
            Placement {
                coord: loose_second,
            },
        )
        .unwrap();
        let loose_analysis = threats::analyze(&loose);
        assert_eq!(loose_analysis.b, 2);
        assert!(!loose_analysis.own_win_now);
        assert_eq!(loose_analysis.opp_threat_count, 1);
        assert_eq!(loose_analysis.min_hitting_set, Some(1));
        assert!(turn_created_claimant_threat(
            &loose,
            claimant,
            first,
            loose_second
        ));
        assert!(!turn_forces_small_defender_reply(&loose, claimant));

        let mut quiet = second_stone;
        let quiet_second = HexCoord::new(8, 5);
        apply_placement(
            &mut quiet,
            Placement {
                coord: quiet_second,
            },
        )
        .unwrap();
        assert!(!turn_created_claimant_threat(
            &quiet,
            claimant,
            first,
            quiet_second
        ));
        assert!(!turn_forces_small_defender_reply(&quiet, claimant));
    }

    #[test]
    fn zone_generator_is_deterministic_and_never_count_truncates() {
        let state = quiet_fixture();
        let claimant = state.current_player().other();
        let d = 1;
        let caps = ZoneSearchCaps {
            enabled: true,
            stale_area_filter: false,
            count2_threshold: false,
            pair_commutation: false,
        };
        let first = zone_initial_candidates(&state, claimant, d, caps);
        let second = zone_initial_candidates(&state, claimant, d, caps);
        assert_eq!(first, second);
        assert!(!first.is_empty());
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        assert!(first.len() <= legal.len());
        assert!(first.iter().all(|mv| legal.contains(mv)));

        let d6 = zone_initial_candidates(&state, claimant, 6, caps);
        assert_eq!(d6.len(), legal.len(), "D>=6 must use the full legal set");
        assert!(legal.iter().all(|mv| d6.contains(mv)));
    }

    fn forced_loss_fixture() -> RustHexoState {
        replay(&[
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (0, 4),
            (8, 4),
            (10, 3),
            (1, 4),
            (2, 4),
            (12, 2),
            (14, 1),
            (3, 4),
            (16, 0),
        ])
    }

    fn quiet_fixture() -> RustHexoState {
        replay(&[(0, 0), (0, 8), (2, 7)])
    }

    fn pair_width_first_stone_fixture() -> RustHexoState {
        replay(&[(0, 0), (0, 8), (2, 7), (1, 0), (4, 6), (6, 5), (8, 4)])
    }

    fn xsnfyll_forced_defender_fixture() -> RustHexoState {
        let mut state = replay(&[
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
        assert_eq!(state.current_player(), Player::Player1);
        assert_eq!(state.phase(), TurnPhase::FirstStone);
        for coord in [HexCoord::new(-1, -1), HexCoord::new(1, -5)] {
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        state
    }

    fn forced_defense_fixture() -> RustHexoState {
        replay(&[
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

    fn win_now_fixture() -> RustHexoState {
        let mut state = forced_defense_fixture();
        for coord in [HexCoord::new(8, 4), HexCoord::new(10, 3)] {
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        state
    }

    fn spare_tempo_fixture() -> RustHexoState {
        replay(&[
            (0, 0),
            (1, 4),
            (2, 4),
            (-2, -3),
            (-1, -3),
            (3, 4),
            (4, 1),
            (0, -3),
            (2, -3),
            (4, 2),
            (4, 3),
            (3, -3),
            (-3, 1),
            (1, 7),
            (2, 6),
            (1, -2),
            (-4, 2),
            (3, 5),
            (-1, 0),
            (2, -4),
            (-5, 2),
            (0, -1),
            (-2, 1),
            (2, -5),
            (-6, 3),
        ])
    }

    fn deep_universal_fixture() -> RustHexoState {
        replay(&[
            (0, 0),
            (-1, 0),
            (0, -1),
            (-2, -3),
            (-1, -3),
            (-2, 1),
            (-3, 1),
            (0, -3),
            (1, -3),
            (-4, 2),
            (2, -4),
            (1, 4),
            (2, 4),
            (-5, 2),
            (2, -5),
            (3, 4),
            (4, 1),
            (-6, 3),
            (3, -6),
            (4, 2),
            (4, 3),
            (-7, 3),
            (3, -7),
            (1, 7),
            (2, 6),
            (-1, 2),
            (2, -1),
            (3, 5),
        ])
    }

    fn deep_universal_certificate(state: &RustHexoState) -> TssCertificate {
        // The default proof ordering finds the shorter direct lambda-one move
        // (4,4).  Staple the also-valid longer forcing choice (2,-3) so the
        // harness exercises a real universal strategy subtree.
        let mut work = state.clone();
        let (_placed, delta) = work
            .apply_with_delta(Placement {
                coord: HexCoord::new(2, -3),
            })
            .unwrap();
        let mut context = NarrowCompatSearch::new(500_000, 8 << 20, u64::MAX);
        let child = context
            .prove(&mut work, Player::Player0, 1, None)
            .expect("validated forcing branch must prove");
        work.undo(delta);
        let root = context
            .alloc_node(
                CertNode::Choice {
                    mv: HexCoord::new(2, -3),
                    child,
                },
                1,
            )
            .unwrap();
        let (nodes, root_node) = compact_certificate(&context.arena, root).unwrap();
        TssCertificate {
            root: RootBinding::from_state(state),
            claimant: Player::Player0,
            root_node,
            nodes,
            semantic_horizon: u32::MAX,
        }
    }

    fn transformed_state(state: &RustHexoState, symmetry: u8) -> RustHexoState {
        let mut transformed = RustHexoState::new();
        for record in state.placement_history() {
            let coord = d6_transform_coord(record.coord, symmetry).unwrap();
            apply_placement(&mut transformed, Placement { coord }).unwrap();
        }
        transformed
    }

    fn assert_exact_state(left: &RustHexoState, right: &RustHexoState) {
        assert_eq!(left.current_player(), right.current_player());
        assert_eq!(left.phase(), right.phase());
        assert_eq!(left.placements_made(), right.placements_made());
        assert_eq!(left.terminal(), right.terminal());
        assert_eq!(left.last_turn(), right.last_turn());
        assert_eq!(left.placement_history(), right.placement_history());
        assert_eq!(
            left.board().occupied_cells(),
            right.board().occupied_cells()
        );
        for &coord in left.board().occupied_cells() {
            assert_eq!(left.board().get(coord), right.board().get(coord));
        }
        let mut left_legal = Vec::new();
        let mut right_legal = Vec::new();
        left.write_legal_action_ids(&mut left_legal);
        right.write_legal_action_ids(&mut right_legal);
        assert_eq!(left_legal, right_legal);
        fn windows(state: &RustHexoState) -> Vec<(u8, i16, i16, u8, u8)> {
            let mut entries: Vec<_> = state
                .board()
                .windows()
                .entries()
                .map(|entry| {
                    let key = entry.key();
                    (
                        key.axis.index(),
                        key.start.q,
                        key.start.r,
                        entry.mask(Player::Player0),
                        entry.mask(Player::Player1),
                    )
                })
                .collect();
            entries.sort_unstable();
            entries
        }
        assert_eq!(windows(left), windows(right));
    }

    #[test]
    fn root_lambda_loss_has_dual_certificate() {
        let state = forced_loss_fixture();
        let caps = SolveCaps {
            node_cap: 1,
            tt_bytes_cap: 0,
            semantic_horizon: u32::MAX,
        };
        let result = TssSolver::default().solve(&state, &caps);
        assert_eq!(result.status, ProofStatus::Loss);
        let cert = result.cert.unwrap();
        assert_eq!(cert.claimant, state.current_player().other());
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Loss));
    }

    #[test]
    fn full_key_rejects_forced_hash_collision() {
        let left = replay(&[(0, 0), (1, 0)]);
        let right = replay(&[(0, 0), (0, 1)]);
        let left_key = PositionKey::from_state(&left);
        let right_key = PositionKey::from_state(&right);
        assert_ne!(left_key, right_key);

        let mut tt = BoundedTt::new(4096, 0);
        tt.insert(left_key.clone(), Player::Player0, 7);
        assert_eq!(tt.lookup(&left_key, Player::Player0), Some(7));
        assert_eq!(tt.lookup(&right_key, Player::Player0), None);
        tt.insert(right_key.clone(), Player::Player0, 9);
        assert_eq!(tt.lookup(&left_key, Player::Player0), None);
        assert_eq!(tt.lookup(&right_key, Player::Player0), Some(9));
    }

    #[test]
    fn tt_allocation_never_exceeds_cap() {
        let states = [
            replay(&[(0, 0)]),
            replay(&[(0, 0), (1, 0)]),
            replay(&[(0, 0), (0, 1), (1, 0)]),
        ];
        for cap in [0, 1, 255, 256, 1024, 4096] {
            let mut tt = BoundedTt::new(cap, u64::MAX);
            let base = if tt.slots.is_empty() {
                0
            } else {
                tt.slots.capacity() * size_of::<Option<TtEntry>>() + ALLOC_OVERHEAD
            };
            assert_eq!(tt.current_bytes, base);
            assert_eq!(tt.peak_bytes, base);
            let mut expected_peak = base;
            for (node, state) in states.iter().enumerate() {
                tt.insert(
                    PositionKey::from_state(state),
                    state.current_player(),
                    node as u32,
                );
                let exact_accounted = base
                    + tt.slots
                        .iter()
                        .flatten()
                        .map(|entry| entry.key.heap_bytes())
                        .sum::<usize>();
                expected_peak = expected_peak.max(exact_accounted);
                assert_eq!(tt.current_bytes, exact_accounted);
                assert_eq!(tt.peak_bytes, expected_peak);
                assert!(tt.current_bytes <= cap);
                assert!(tt.peak_bytes <= cap);
            }
        }
    }

    #[test]
    fn zero_node_cap_is_unknown_and_certless() {
        let state = forced_loss_fixture();
        let result = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 0,
                tt_bytes_cap: usize::MAX,
                semantic_horizon: u32::MAX,
            },
        );
        assert_eq!(result.status, ProofStatus::Unknown);
        assert!(result.cert.is_none());
        assert_eq!(result.stats.nodes, 0);
    }

    #[test]
    fn semantic_horizon_is_an_absolute_placement_deadline() {
        let state = deep_universal_fixture();
        let root_ply = state.placements_made();
        let result = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 500_000,
                tt_bytes_cap: 0,
                semantic_horizon: root_ply,
            },
        );
        assert_eq!(result.status, ProofStatus::Unknown);
        assert!(result.cert.is_none());

        let expired = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 500_000,
                tt_bytes_cap: 0,
                semantic_horizon: root_ply - 1,
            },
        );
        assert_eq!(expired.status, ProofStatus::Unknown);
        assert_eq!(expired.stats.nodes, 0);
    }

    #[test]
    fn solver_configurations_are_deterministic_on_hard_leaf() {
        let state = forced_loss_fixture();
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 4096,
            semantic_horizon: u32::MAX,
        };
        let a = TssSolver::without_tt().solve(&state, &caps);
        let b = TssSolver::with_hash_mask(0).solve(&state, &caps);
        let c = TssSolver::default().solve(&state, &caps);
        assert_eq!(a.status, b.status);
        assert_eq!(b.status, c.status);
        assert_eq!(a.cert, b.cert);
        assert_eq!(b.cert, c.cert);
    }

    #[test]
    fn deep_win_contains_verified_universal_coverage() {
        let state = deep_universal_fixture();
        let result = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 500_000,
                tt_bytes_cap: 8 << 20,
                semantic_horizon: u32::MAX,
            },
        );
        assert_eq!(result.status, ProofStatus::Win);
        let cert = deep_universal_certificate(&state);
        assert!(cert
            .nodes
            .iter()
            .any(|node| matches!(node, CertNode::Universal { .. })));
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Win));
    }

    #[test]
    fn curated_deep_branch_zone_or_dispatch_is_reference_consistent() {
        let state = deep_universal_fixture();
        let base = deep_universal_certificate(&state);
        let (exact_t, _) = crate::tss_verify::certificate_horizon_preflight(&base).unwrap();

        fn collect_states(
            cert: &TssCertificate,
            id: CertNodeId,
            state: &RustHexoState,
            out: &mut [Option<RustHexoState>],
        ) {
            if out[id as usize].is_some() {
                return;
            }
            out[id as usize] = Some(state.clone());
            match &cert.nodes[id as usize] {
                CertNode::Choice { mv, child } => {
                    let mut next = state.clone();
                    apply_placement(&mut next, Placement { coord: *mv }).unwrap();
                    collect_states(cert, *child, &next, out);
                }
                CertNode::Universal { edges, .. } => {
                    for edge in edges {
                        let mut next = state.clone();
                        apply_placement(&mut next, Placement { coord: edge.mv }).unwrap();
                        collect_states(cert, edge.child, &next, out);
                    }
                }
                _ => {}
            }
        }
        let mut states = vec![None; base.nodes.len()];
        collect_states(&base, base.root_node, &state, &mut states);

        let mut accepted = None;
        for index in 0..base.nodes.len() {
            let Some(node_state) = states[index].as_ref() else {
                continue;
            };
            let CertNode::Universal {
                edges,
                implicit_dispatch: false,
                ..
            } = &base.nodes[index]
            else {
                continue;
            };
            let Some(d) =
                remaining_defender_placements_for_horizon(node_state, base.claimant, exact_t)
            else {
                continue;
            };
            for drop_index in 0..edges.len() {
                let mut nodes = base.nodes.clone();
                let CertNode::Universal { edges, zone, .. } = &mut nodes[index] else {
                    unreachable!()
                };
                *zone = Some(ZoneInfo {
                    d,
                    build_horizon: exact_t,
                });
                edges.remove(drop_index);
                let Some((nodes, root_node)) = compact_certificate(&nodes, base.root_node) else {
                    continue;
                };
                let candidate = TssCertificate {
                    root: base.root.clone(),
                    claimant: base.claimant,
                    root_node,
                    nodes,
                    semantic_horizon: exact_t,
                };
                if TssVerifier.verify(&state, &candidate, ProofStatus::Win) {
                    accepted = Some(candidate);
                    break;
                }
            }
            if accepted.is_some() {
                break;
            }
        }
        let cert = if let Some(cert) = accepted {
            cert
        } else {
            // This curated forcing line's final core can cover every explicit
            // reply. It still exercises the complete zone verifier; omission
            // behavior is covered by the generator-set and mutation tests.
            let mut nodes = base.nodes.clone();
            let Some((index, d)) = states.iter().enumerate().find_map(|(index, replay)| {
                let replay = replay.as_ref()?;
                matches!(
                    base.nodes[index],
                    CertNode::Universal {
                        implicit_dispatch: false,
                        ..
                    }
                )
                .then(|| {
                    remaining_defender_placements_for_horizon(replay, base.claimant, exact_t)
                        .map(|d| (index, d))
                })
                .flatten()
            }) else {
                // This line is dispatch-only; the U3 paired oracle is its
                // applicable certificate gate.
                return;
            };
            let CertNode::Universal { zone, .. } = &mut nodes[index] else {
                unreachable!()
            };
            *zone = Some(ZoneInfo {
                d,
                build_horizon: exact_t,
            });
            TssCertificate {
                root: base.root.clone(),
                claimant: base.claimant,
                root_node: base.root_node,
                nodes,
                semantic_horizon: exact_t,
            }
        };
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Win));
        assert_eq!(
            tss_reference::solve(&state, exact_t - state.placements_made()).status,
            ProofStatus::Win
        );
    }

    #[test]
    fn curated_differential_and_every_hard_certificate_verifies() {
        let fixtures = [
            (quiet_fixture(), ProofStatus::Unknown, 1u64, None),
            (forced_defense_fixture(), ProofStatus::Unknown, 1, None),
            (win_now_fixture(), ProofStatus::Win, 1, Some(2)),
            // The exact Python forced-loss fixture is retained here, while a
            // zero cap exercises the permitted Unknown differential path.  Its
            // hard dual certificate is tested above; the independently
            // exhaustive hard comparison uses its SecondStone child below so
            // default debug CI does not enumerate ~10^6 two-stone defenses.
            (forced_loss_fixture(), ProofStatus::Unknown, 0, None),
            (spare_tempo_fixture(), ProofStatus::Unknown, 1, None),
        ];
        for (state, expected, node_cap, reference_depth) in fixtures {
            let caps = SolveCaps {
                node_cap,
                tt_bytes_cap: 0,
                semantic_horizon: u32::MAX,
            };
            let result = TssSolver::default().solve(&state, &caps);
            assert_eq!(result.status, expected);
            if result.status == ProofStatus::Unknown {
                assert!(result.cert.is_none());
                continue;
            }
            let cert = result.cert.as_ref().expect("hard result needs cert");
            assert!(TssVerifier.verify(&state, cert, result.status));
            let reference = tss_reference::solve(&state, reference_depth.unwrap());
            assert_eq!(
                reference.status, result.status,
                "reference disagreement after {} nodes",
                reference.nodes
            );
        }
    }

    #[test]
    fn differential_forced_loss_after_first_defender_stone() {
        let mut state = forced_loss_fixture();
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::new(-8, 0),
            },
        )
        .unwrap();
        assert!(matches!(state.phase(), TurnPhase::SecondStone { .. }));
        assert!(threats::analyze(&state).forced_loss());
        let result = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 1,
                tt_bytes_cap: 0,
                semantic_horizon: u32::MAX,
            },
        );
        assert_eq!(result.status, ProofStatus::Loss);
        assert!(TssVerifier.verify(&state, result.cert.as_ref().unwrap(), result.status));
        let reference = tss_reference::solve(&state, 3);
        assert_eq!(reference.status, ProofStatus::Loss);
    }

    #[test]
    fn seeded_random_differential_covers_all_phases_and_dense_positions() {
        #[derive(Clone, Copy)]
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

        let mut corpus = vec![win_now_fixture()]; // threat-dense endgame anchor
        let mut saw_phase = [false; 3];
        for seed in 1..=6u64 {
            let mut rng = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut state = RustHexoState::new();
            for ply in 0..20usize {
                let phase_index = match state.phase() {
                    TurnPhase::Opening => 0,
                    TurnPhase::FirstStone => 1,
                    TurnPhase::SecondStone { .. } => 2,
                };
                saw_phase[phase_index] = true;
                if ply < 3 || ply % 5 == seed as usize % 5 {
                    corpus.push(state.clone());
                }
                let mut legal = Vec::new();
                state.write_legal_moves(&mut legal);
                if legal.is_empty() {
                    break;
                }
                let coord = legal[(rng.next() as usize) % legal.len()];
                let result = apply_placement(&mut state, Placement { coord }).unwrap();
                if result.outcome.is_some() {
                    break;
                }
            }
        }
        assert!(saw_phase.into_iter().all(|seen| seen));
        assert!(corpus
            .iter()
            .any(|state| state.board().windows().has_threats()));

        let caps = SolveCaps {
            node_cap: 1,
            tt_bytes_cap: 1024,
            semantic_horizon: u32::MAX,
        };
        let mut hard = 0usize;
        for state in &corpus {
            let result = TssSolver::default().solve(state, &caps);
            let Some(cert) = result.cert.as_ref() else {
                assert_eq!(result.status, ProofStatus::Unknown);
                continue;
            };
            hard += 1;
            assert!(TssVerifier.verify(state, cert, result.status));
            let b = threats::placements_remaining(state) as u32;
            let depth = if result.status == ProofStatus::Win {
                b
            } else {
                b + 2
            };
            assert_eq!(tss_reference::solve(state, depth).status, result.status);
        }
        assert!(
            hard >= 1,
            "dense anchor must produce at least one hard proof"
        );

        // Repeat the one-sided differential with U1 enabled at a matched
        // semantic horizon. Unknown remains an allowed restricted-search
        // outcome; every hard claim must verify and match the exhaustive
        // reference at the exact derived deadline.
        let mut zoned_hard = 0usize;
        for state in &corpus {
            let mut solver = TssSolver::default();
            solver.set_zone_options(ZoneSearchCaps {
                enabled: true,
                stale_area_filter: false,
                count2_threshold: false,
                pair_commutation: false,
            });
            let mut zone_caps = SolveCaps {
                node_cap: 16,
                tt_bytes_cap: 4096,
                semantic_horizon: state.placements_made().saturating_add(4),
            };
            let mut result = solver.solve(&state, &zone_caps);
            if let Some(cert) = result.cert.as_ref() {
                if let Some((exact_t, true)) =
                    crate::tss_verify::certificate_horizon_preflight(cert)
                {
                    if exact_t != zone_caps.semantic_horizon {
                        zone_caps.semantic_horizon = exact_t;
                        result = solver.solve(state, &zone_caps);
                    }
                }
            }
            let Some(cert) = result.cert.as_ref() else {
                assert_eq!(result.status, ProofStatus::Unknown);
                continue;
            };
            zoned_hard += 1;
            assert!(TssVerifier.verify(state, cert, result.status));
            let (exact_t, _) = crate::tss_verify::certificate_horizon_preflight(cert).unwrap();
            assert_eq!(
                tss_reference::solve(state, exact_t - state.placements_made()).status,
                result.status,
            );
        }
        assert!(zoned_hard >= 1);
    }

    #[test]
    fn dispatch_theorem_and_per_move_staple_oracles_agree() {
        let caps = SolveCaps {
            node_cap: 2_000,
            tt_bytes_cap: 0,
            semantic_horizon: u32::MAX,
        };
        let mut corpus = Vec::new();
        let mut random = 0x5a17_1e5u64;
        for _ in 0..16 {
            let mut state = RustHexoState::new();
            for ply in 0..18 {
                if ply >= 8 && ply % 3 == 0 {
                    corpus.push(state.clone());
                }
                let mut legal = Vec::new();
                state.write_legal_moves(&mut legal);
                if legal.is_empty() {
                    break;
                }
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let coord = legal[(random as usize) % legal.len()];
                if apply_placement(&mut state, Placement { coord })
                    .unwrap()
                    .outcome
                    .is_some()
                {
                    break;
                }
            }
        }
        corpus.extend([
            forced_defense_fixture(),
            deep_universal_fixture(),
            spare_tempo_fixture(),
        ]);
        let mut checked = 0usize;
        for state in corpus {
            let result = TssSolver::default().solve(&state, &caps);
            let Some(cert) = result.cert.as_ref() else {
                continue;
            };
            let theorem = TssVerifier.verify(&state, cert, result.status);
            let per_move = TssVerifier.verify_with_dispatch_oracle(&state, cert, result.status);
            assert_eq!(theorem, per_move, "paired dispatch oracle divergence");
            assert!(theorem, "solver-produced hard certificate must verify");
            checked += 1;
        }
        assert!(checked > 0, "paired corpus must contain hard certificates");
    }

    #[test]
    fn spare_stone_counter_threat_cannot_be_implicitly_dispatched() {
        let state = spare_tempo_fixture();
        let root = threats::analyze(&state);
        assert_eq!(state.phase(), TurnPhase::FirstStone);
        assert_eq!(root.b, 2);
        assert_eq!(root.min_hitting_set, Some(1));
        assert!(root.opp_threat_count > 0);
        assert!(!root.own_win_now);

        // Y is outside the sole hitting cell G, yet it creates a counter-fork;
        // lambda-one correctly leaves the child unresolved instead of
        // dispatching it as a loss.
        let counter = HexCoord::new(4, 4);
        let guard = HexCoord::new(1, -3);
        assert!(
            ordered_threat_creating_moves(&mut state.clone(), Player::Player1)
                .iter()
                .any(|candidate| candidate.coord == counter)
        );
        let mut line = state.clone();
        apply_placement(&mut line, Placement { coord: counter }).unwrap();
        let after_counter = threats::analyze(&line);
        assert_eq!(line.phase(), TurnPhase::SecondStone { first: counter });
        assert_eq!(after_counter.verdict(), None);
        assert_eq!(after_counter.min_hitting_set, Some(1));
        apply_placement(&mut line, Placement { coord: guard }).unwrap();
        assert!(threats::analyze(&line).forced_loss());

        // A malicious P0 strategy certificate that advertises dispatch at the
        // k<B root is rejected before any child can be trusted.
        let bogus = TssCertificate {
            root: RootBinding::from_state(&state),
            claimant: Player::Player0,
            root_node: 0,
            nodes: vec![CertNode::Universal {
                edges: Vec::new(),
                implicit_dispatch: true,
                zone: None,
                commutations: Vec::new(),
            }],
            semantic_horizon: u32::MAX,
        };
        assert!(!TssVerifier.verify(&state, &bogus, ProofStatus::Loss));
    }

    #[test]
    fn certificate_mutations_are_rejected() {
        let state = deep_universal_fixture();
        let cert = deep_universal_certificate(&state);
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Win));

        let mut corrupt_move = cert.clone();
        let root = corrupt_move.root_node as usize;
        let CertNode::Choice { mv, .. } = &mut corrupt_move.nodes[root] else {
            panic!("deep cert root must be Choice");
        };
        *mv = HexCoord::ZERO; // occupied at the root
        assert!(!TssVerifier.verify(&state, &corrupt_move, ProofStatus::Win));

        let mut dropped_child = cert.clone();
        let universal = dropped_child
            .nodes
            .iter_mut()
            .find_map(|node| match node {
                CertNode::Universal { edges, .. } if !edges.is_empty() => Some(edges),
                _ => None,
            })
            .expect("fixture must contain a non-empty universal node");
        universal.remove(0);
        assert!(!TssVerifier.verify(&state, &dropped_child, ProofStatus::Win));

        assert!(!TssVerifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(!TssVerifier.verify(&quiet_fixture(), &cert, ProofStatus::Win));

        let mut wrong_witness = cert.clone();
        wrong_witness.root.phase = TurnPhase::SecondStone {
            first: HexCoord::new(99, 99),
        };
        assert!(!TssVerifier.verify(&state, &wrong_witness, ProofStatus::Win));

        let mut cycle = cert.clone();
        let root = cycle.root_node;
        let CertNode::Choice { child, .. } = &mut cycle.nodes[root as usize] else {
            unreachable!();
        };
        *child = root;
        assert!(!TssVerifier.verify(&state, &cycle, ProofStatus::Win));
    }

    #[test]
    fn d6_status_and_certificate_replay_all_twelve_symmetries() {
        let state = deep_universal_fixture();
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 64 << 10,
            semantic_horizon: u32::MAX,
        };
        let base = TssSolver::default().solve(&state, &caps);
        assert_eq!(base.status, ProofStatus::Win);
        let deep_cert = deep_universal_certificate(&state);

        for symmetry in 0..D6_SYMMETRY_COUNT {
            let transformed = transformed_state(&state, symmetry);
            let result = TssSolver::default().solve(&transformed, &caps);
            assert_eq!(result.status, base.status, "symmetry={symmetry}");
            let cert = result.cert.as_ref().unwrap();
            assert!(TssVerifier.verify(&transformed, cert, result.status));

            let remapped = d6_remap_certificate(&deep_cert, symmetry).unwrap();
            assert!(TssVerifier.verify(&transformed, &remapped, ProofStatus::Win));
        }
    }

    #[test]
    fn tt_disabled_tiny_large_and_interleaved_solves_match() {
        let state = deep_universal_fixture();
        let caps_for = |bytes| SolveCaps {
            node_cap: 64,
            tt_bytes_cap: bytes,
            semantic_horizon: u32::MAX,
        };
        let off = TssSolver::without_tt().solve(&state, &caps_for(0));
        let tiny = TssSolver::default().solve(&state, &caps_for(256));
        let large = TssSolver::default().solve(&state, &caps_for(8 << 20));
        assert_eq!(off.status, ProofStatus::Win);
        assert_eq!(off.status, tiny.status);
        assert_eq!(tiny.status, large.status);
        for result in [&off, &tiny, &large] {
            assert!(TssVerifier.verify(&state, result.cert.as_ref().unwrap(), result.status));
        }
        assert_eq!(off.stats.peak_tt_bytes, 0);
        assert!(tiny.stats.peak_tt_bytes <= 256);
        assert!(large.stats.peak_tt_bytes <= (8 << 20));

        let mut solver = TssSolver::with_hash_mask(0);
        let first = solver.solve(&state, &caps_for(4096));
        let _other = solver.solve(&forced_loss_fixture(), &caps_for(4096));
        let repeated = solver.solve(&state, &caps_for(4096));
        assert_eq!(first.status, repeated.status);
        for result in [&first, &repeated] {
            assert!(TssVerifier.verify(&state, result.cert.as_ref().unwrap(), result.status));
        }
        // Cache history may change discovery cost, but never hard-verdict
        // validity.  A fixed initial cache state remains deterministic below.
        assert!(first.stats.peak_tt_bytes <= 4096);
        assert!(repeated.stats.peak_tt_bytes <= 4096);
    }

    #[test]
    fn shared_tt_warm_and_cold_verdicts_verify() {
        let state = deep_universal_fixture();
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 8 << 20,
            semantic_horizon: u32::MAX,
        };
        let cold = TssSolver::default().solve(&state, &caps);
        let mut warm_solver = TssSolver::default();
        let first = warm_solver.solve(&state, &caps);
        let warm = warm_solver.solve(&state, &caps);
        for result in [&cold, &first, &warm] {
            assert_eq!(result.status, ProofStatus::Win);
            assert!(TssVerifier.verify(&state, result.cert.as_ref().unwrap(), result.status));
            assert!(result.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64);
        }
        assert!(warm.stats.tt_hits > 0);
        assert!(warm.stats.nodes < first.stats.nodes);
    }

    #[test]
    fn wide_shared_fragments_warm_exact_root_and_verify() {
        let state = xsnfyll_forced_defender_fixture();
        let caps = SolveCaps {
            node_cap: 10_000,
            tt_bytes_cap: 64 << 20,
            semantic_horizon: u32::MAX,
        };
        let mut cold_solver = TssSolver::default();
        cold_solver.set_width_options(WidthOptions::vcf_pair_complete());
        let cold = cold_solver.solve_goal(&state, &caps, SolveGoal::Loss);

        let mut warm_solver = TssSolver::default();
        warm_solver.set_shared_fragments_for_test(true);
        warm_solver.set_width_options(WidthOptions::vcf_pair_complete());
        let first = warm_solver.solve_goal(&state, &caps, SolveGoal::Loss);
        let warm = warm_solver.solve_goal(&state, &caps, SolveGoal::Loss);

        for result in [&cold, &first, &warm] {
            assert_eq!(result.status, ProofStatus::Loss);
            assert!(TssVerifier.verify(&state, result.cert.as_ref().unwrap(), result.status));
            assert!(result.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64);
        }
        assert_eq!(cold.status, first.status);
        assert_eq!(first.status, warm.status);
        assert_eq!(cold.stats.nodes, first.stats.nodes);
        assert_eq!(cold.stats.tt_hits, first.stats.tt_hits);
        assert_eq!(first.stats.fragment_lookups, 0);
        assert!(first.stats.fragment_store_entries > 0);
        assert_eq!(warm.stats.fragment_hits, 1);
        assert_eq!(warm.stats.fragment_imports, 1);
        assert!(warm.stats.nodes < first.stats.nodes);
        let snapshot = warm_solver.shared_fragment_store_snapshot();
        assert!(snapshot.enabled);
        assert!(snapshot.entries > 0);
        assert!(snapshot.stored_nodes > 0);
        assert_eq!(
            snapshot.bytes as usize,
            warm_solver.fragment_store.recomputed_bytes()
        );
    }

    #[test]
    fn wide_shared_fragments_forced_collision_never_cross_contaminates() {
        let a = xsnfyll_forced_defender_fixture();
        let b = transformed_state(&a, 1);
        let caps = SolveCaps {
            node_cap: 10_000,
            tt_bytes_cap: 64 << 20,
            semantic_horizon: u32::MAX,
        };
        let mut solver = TssSolver::with_hash_mask(0);
        solver.set_shared_fragments_for_test(true);
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        let first_a = solver.solve_goal(&a, &caps, SolveGoal::Loss);
        let first_b = solver.solve_goal(&b, &caps, SolveGoal::Loss);
        let second_a = solver.solve_goal(&a, &caps, SolveGoal::Loss);
        for (state, result) in [(&a, &first_a), (&b, &first_b), (&a, &second_a)] {
            assert_eq!(result.status, ProofStatus::Loss);
            assert!(TssVerifier.verify(state, result.cert.as_ref().unwrap(), result.status));
            assert!(result.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64);
        }
        // B may evict A's direct-mapped entry, but it can never answer A.
        assert_eq!(first_b.stats.fragment_hits, 0);
        assert_eq!(second_a.status, first_a.status);
    }

    #[test]
    fn shared_fragment_store_full_key_claimant_and_accounting() {
        let state = deep_universal_fixture();
        let other = transformed_state(&state, 1);
        let proof = CachedProof::from_compact(vec![cache_test_leaf()], 0).unwrap();
        let mut store = ProvenFragmentStore::new(64 << 10, 0);
        let key = PositionKey::from_state(&state);
        assert!(store.insert(key.clone(), Player::Player0, proof));
        assert!(store.lookup(&key, Player::Player0).is_some());
        assert!(store.lookup(&key, Player::Player1).is_none());
        assert!(store
            .lookup(&PositionKey::from_state(&other), Player::Player0)
            .is_none());
        assert_eq!(store.current_bytes, store.recomputed_bytes());
        assert!(store.current_bytes <= store.cap);
        assert!(store.peak_bytes <= store.cap);
    }

    #[test]
    fn shared_tt_survives_origin_arena_drop() {
        let state = quiet_fixture();
        let key = PositionKey::from_state(&state);
        let mut cache = SharedProofCache::new(4096, u64::MAX);
        {
            let arena = vec![cache_test_leaf()];
            let proof = CachedProof::from_arena_limited(&arena, 0, 4, 4).unwrap();
            cache.insert(key.clone(), Player::Player0, proof);
        }
        let proof = cache.lookup_cloned(&key, Player::Player0).unwrap();
        let mut context = NarrowCompatSearch::new(4, 0, u64::MAX);
        assert_eq!(context.import_cached_proof(proof, 0), Some(0));
        assert!(matches!(context.arena.as_slice(), [CertNode::Win { .. }]));
    }

    #[test]
    fn shared_tt_parent_proof_reuses_descendant() {
        let parent = deep_universal_fixture();
        let claimant = parent.current_player();
        let mut descendant_root = parent.clone();
        descendant_root
            .apply_with_delta(Placement {
                coord: HexCoord::new(2, -3),
            })
            .unwrap();
        assert_ne!(descendant_root.current_player(), claimant);

        let generous = SolveCaps {
            node_cap: 500_000,
            tt_bytes_cap: 8 << 20,
            semantic_horizon: u32::MAX,
        };
        let (local_cap, shared_cap) = split_tt_cap(generous.tt_bytes_cap);
        let mut solver = TssSolver::default();
        solver.shared_tt.reconfigure(shared_cap, u64::MAX);
        let reply = {
            let mut work = descendant_root.clone();
            let mut context = NarrowCompatSearch::with_shared(
                generous.node_cap,
                local_cap,
                u64::MAX,
                &mut solver.shared_tt,
                descendant_root.placements_made(),
                generous.semantic_horizon,
                ZoneSearchCaps::default(),
                WidthOptions::default(),
                MAX_SEARCH_DEPTH,
                false,
                None,
                false,
                false,
            );
            let root = context
                .prove(&mut work, claimant, descendant_root.placements_made(), None)
                .unwrap();
            let CertNode::Universal { edges, .. } = &context.arena[root as usize] else {
                panic!("forcing descendant root must be universal");
            };
            edges
                .iter()
                .find(|edge| {
                    matches!(
                        context.arena[edge.child as usize],
                        CertNode::Choice { .. } | CertNode::Universal { .. }
                    )
                })
                .expect("fixture needs a cached non-leaf grandchild")
                .mv
        };

        // Re-root one level below the narrow compatibility root. This key was
        // promoted while proving an internal universal edge, so an exact-root
        // result memo cannot satisfy the lookup.
        let mut descendant = descendant_root.clone();
        descendant
            .apply_with_delta(Placement { coord: reply })
            .unwrap();
        let descendant_key = PositionKey::from_state(&descendant);
        assert!(solver
            .shared_tt
            .lookup_cloned(&descendant_key, claimant)
            .is_some());

        let expected = status_for_claimant(descendant.current_player(), claimant);
        let goal = match expected {
            ProofStatus::Win => SolveGoal::Win,
            ProofStatus::Loss => SolveGoal::Loss,
            ProofStatus::Unknown => unreachable!(),
        };
        let tiny = SolveCaps {
            node_cap: 2,
            tt_bytes_cap: generous.tt_bytes_cap,
            semantic_horizon: u32::MAX,
        };
        let cold = TssSolver::default().solve_goal(&descendant, &tiny, goal);
        assert_eq!(cold.status, ProofStatus::Unknown);
        let result = solver.solve_goal(&descendant, &tiny, goal);
        assert_eq!(result.status, expected);
        assert!(result.stats.tt_hits > 0);
        assert!(TssVerifier.verify(&descendant, result.cert.as_ref().unwrap(), result.status));
    }

    #[test]
    fn shared_tt_forced_collision_a_b_a_stays_valid() {
        let a = deep_universal_fixture();
        let b = transformed_state(&a, 1);
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 8 << 20,
            semantic_horizon: u32::MAX,
        };
        let mut solver = TssSolver::with_hash_mask(0);
        let first_a = solver.solve(&a, &caps);
        let a_key = PositionKey::from_state(&a);
        let b_key = PositionKey::from_state(&b);
        let claimant = a.current_player();
        assert!(solver.shared_tt.lookup_cloned(&a_key, claimant).is_some());
        let first_b = solver.solve(&b, &caps);
        assert!(solver.shared_tt.lookup_cloned(&a_key, claimant).is_none());
        assert!(solver.shared_tt.lookup_cloned(&b_key, claimant).is_some());
        let second_a = solver.solve(&a, &caps);
        for (state, result) in [(&a, &first_a), (&b, &first_b), (&a, &second_a)] {
            assert_eq!(result.status, ProofStatus::Win);
            assert!(TssVerifier.verify(state, result.cert.as_ref().unwrap(), result.status));
        }
        assert!(first_a.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64);
        assert!(first_b.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64);
        assert!(second_a.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64);
    }

    #[test]
    fn shared_tt_claimant_isolation() {
        let state = quiet_fixture();
        let key = PositionKey::from_state(&state);
        let proof = CachedProof::from_compact(vec![cache_test_leaf()], 0).unwrap();
        let mut cache = SharedProofCache::new(4096, 0);
        cache.insert(key.clone(), Player::Player0, proof);
        assert!(cache.lookup_cloned(&key, Player::Player0).is_some());
        assert!(cache.lookup_cloned(&key, Player::Player1).is_none());
    }

    #[test]
    fn width_profile_change_drops_shared_fragments_but_same_profile_keeps_them() {
        let state = quiet_fixture();
        let key = PositionKey::from_state(&state);
        let proof = CachedProof::from_compact(vec![cache_test_leaf()], 0).unwrap();
        let mut solver = TssSolver::default();
        solver.shared_tt.reconfigure(4096, u64::MAX);
        solver.shared_tt.insert(key.clone(), Player::Player0, proof);
        assert!(solver.shared_tt.current_bytes > 0);

        solver.set_width_options(WidthOptions::default());
        assert!(solver.shared_tt.current_bytes > 0);
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        assert!(solver
            .shared_tt
            .lookup_cloned(&key, Player::Player0)
            .is_none());
        assert!(solver.shared_tt.slots.iter().all(Option::is_none));
        assert_eq!(
            solver.shared_tt.current_bytes,
            solver.shared_tt.recomputed_bytes()
        );
    }

    #[test]
    fn shared_tt_sustained_churn_respects_cap() {
        let mut cache = SharedProofCache::new(4096, u64::MAX);
        let proof = CachedProof::from_compact(vec![cache_test_leaf()], 0).unwrap();
        let base = deep_universal_fixture();
        for round in 0..200usize {
            let state = transformed_state(&base, (round % D6_SYMMETRY_COUNT as usize) as u8);
            cache.insert(
                PositionKey::from_state(&state),
                state.current_player(),
                proof.clone(),
            );
            assert_eq!(cache.current_bytes, cache.recomputed_bytes());
            assert!(cache.current_bytes <= cache.cap);
            assert!(cache.peak_bytes <= cache.cap);
        }
    }

    #[test]
    fn shared_tt_large_tiny_zero_reconfiguration() {
        let state = deep_universal_fixture();
        let mut solver = TssSolver::default();
        let large = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 8 << 20,
            semantic_horizon: u32::MAX,
        };
        assert_eq!(solver.solve(&state, &large).status, ProofStatus::Win);
        assert!(solver.shared_tt.current_bytes > 0);

        let tiny = SolveCaps {
            node_cap: 0,
            tt_bytes_cap: 1024,
            semantic_horizon: u32::MAX,
        };
        let tiny_result = solver.solve(&state, &tiny);
        assert_eq!(tiny_result.status, ProofStatus::Unknown);
        assert_eq!(solver.shared_tt.cap, split_tt_cap(tiny.tt_bytes_cap).1);
        assert!(solver.shared_tt.current_bytes <= solver.shared_tt.cap);
        assert!(tiny_result.stats.peak_tt_bytes <= tiny.tt_bytes_cap as u64);

        let zero = SolveCaps {
            node_cap: 0,
            tt_bytes_cap: 0,
            semantic_horizon: u32::MAX,
        };
        let zero_result = solver.solve(&state, &zero);
        assert_eq!(zero_result.status, ProofStatus::Unknown);
        assert!(solver.shared_tt.slots.is_empty());
        assert_eq!(solver.shared_tt.current_bytes, 0);
        assert_eq!(zero_result.stats.peak_tt_bytes, 0);
    }

    #[test]
    fn shared_tt_import_preflight_is_atomic() {
        let chain = CachedProof::from_compact(
            vec![
                cache_test_leaf(),
                CertNode::Choice {
                    mv: HexCoord::new(1, 0),
                    child: 0,
                },
                CertNode::Choice {
                    mv: HexCoord::new(2, 0),
                    child: 1,
                },
            ],
            2,
        )
        .unwrap();

        let mut depth_context = NarrowCompatSearch::new(4, 0, u64::MAX);
        assert!(depth_context
            .import_cached_proof(chain.clone(), MAX_SEARCH_DEPTH - 1)
            .is_none());
        assert!(depth_context.arena.is_empty());

        let mut node_context = NarrowCompatSearch::new(4, 0, u64::MAX);
        node_context.arena = vec![cache_test_leaf(); MAX_CERT_NODES - 2];
        let before_nodes = node_context.arena.len();
        assert!(node_context.import_cached_proof(chain, 0).is_none());
        assert_eq!(node_context.arena.len(), before_nodes);

        let edge_proof = CachedProof::from_compact(
            vec![
                cache_test_leaf(),
                CertNode::Universal {
                    edges: vec![CertEdge {
                        mv: HexCoord::new(1, 0),
                        child: 0,
                    }],
                    implicit_dispatch: false,
                    zone: None,
                    commutations: Vec::new(),
                },
            ],
            1,
        )
        .unwrap();
        let mut edge_context = NarrowCompatSearch::new(4, 0, u64::MAX);
        edge_context.edge_count = MAX_CERT_EDGES;
        assert!(edge_context.import_cached_proof(edge_proof, 0).is_none());
        assert!(edge_context.arena.is_empty());
        assert_eq!(edge_context.edge_count, MAX_CERT_EDGES);
    }

    #[test]
    fn zone_cache_composition_refuses_slow_sibling_horizon() {
        let mut quick = cache_test_leaf();
        if let CertNode::Win { resolution_ply, .. } = &mut quick {
            *resolution_ply = 4;
        }
        let mut slow = cache_test_leaf();
        if let CertNode::Win { resolution_ply, .. } = &mut slow {
            *resolution_ply = 8;
        }
        // The quick branch was zoned at T=6. Flattening it beside a slower
        // T=8 sibling must retain min(zone-build)=6 and max(resolution)=8.
        let composite = CachedProof::from_compact(
            vec![
                quick,
                CertNode::Universal {
                    edges: vec![CertEdge {
                        mv: HexCoord::new(1, 0),
                        child: 0,
                    }],
                    implicit_dispatch: false,
                    zone: Some(ZoneInfo {
                        d: 1,
                        build_horizon: 6,
                    }),
                    commutations: Vec::new(),
                },
                slow,
                CertNode::Universal {
                    edges: vec![
                        CertEdge {
                            mv: HexCoord::new(2, 0),
                            child: 1,
                        },
                        CertEdge {
                            mv: HexCoord::new(3, 0),
                            child: 2,
                        },
                    ],
                    implicit_dispatch: false,
                    zone: None,
                    commutations: Vec::new(),
                },
            ],
            3,
        )
        .unwrap();
        assert_eq!(composite.resolution_t, 8);
        assert_eq!(composite.zone_build_t, Some(6));

        let mut rejected = NarrowCompatSearch::new(32, 0, u64::MAX);
        rejected.semantic_horizon = 8;
        assert!(rejected.import_cached_proof(composite.clone(), 0).is_none());
        assert!(rejected.arena.is_empty(), "import preflight must be atomic");

        let mut accepted = NarrowCompatSearch::new(32, 0, u64::MAX);
        accepted.semantic_horizon = 6;
        // Resolution 8 is independently too late, so a containing proof can
        // never smuggle this malformed composite through at either horizon.
        assert!(accepted.import_cached_proof(composite, 0).is_none());
    }

    #[test]
    fn shared_tt_never_caches_unknown() {
        let state = deep_universal_fixture();
        let caps = SolveCaps {
            node_cap: 1,
            tt_bytes_cap: 64 << 10,
            semantic_horizon: u32::MAX,
        };
        let mut solver = TssSolver::default();
        let result = solver.solve(&state, &caps);
        assert_eq!(result.status, ProofStatus::Unknown);
        let key = PositionKey::from_state(&state);
        for claimant in [state.current_player(), state.current_player().other()] {
            assert!(solver.shared_tt.lookup_cloned(&key, claimant).is_none());
        }
    }

    #[test]
    fn shared_tt_conditional_determinism() {
        let a = deep_universal_fixture();
        let b = transformed_state(&a, 2);
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 64 << 10,
            semantic_horizon: u32::MAX,
        };
        let run = |solver: &mut TssSolver| {
            let first = solver.solve(&a, &caps);
            let second = solver.solve(&b, &caps);
            (
                first.status,
                first.cert,
                first.stats.nodes,
                first.stats.tt_hits,
                second.status,
                second.cert,
                second.stats.nodes,
                second.stats.tt_hits,
            )
        };
        let mut left = TssSolver::default();
        let mut right = TssSolver::default();
        assert_eq!(run(&mut left), run(&mut right));
        assert_eq!(left.shared_tt, right.shared_tt);
    }

    #[test]
    fn solve_goal_filters_root_facts() {
        let loss = forced_loss_fixture();
        let win = win_now_fixture();
        let caps = SolveCaps {
            node_cap: 1,
            tt_bytes_cap: 4096,
            semantic_horizon: u32::MAX,
        };
        let loss_filtered = TssSolver::default().solve_goal(&loss, &caps, SolveGoal::Win);
        assert_eq!(loss_filtered.status, ProofStatus::Unknown);
        assert!(loss_filtered.cert.is_none());
        let loss_kept = TssSolver::default().solve_goal(&loss, &caps, SolveGoal::Loss);
        assert_eq!(loss_kept.status, ProofStatus::Loss);
        assert!(TssVerifier.verify(&loss, loss_kept.cert.as_ref().unwrap(), loss_kept.status));
        assert_eq!(
            TssSolver::default()
                .solve_goal(&win, &caps, SolveGoal::Loss)
                .status,
            ProofStatus::Unknown
        );
    }

    #[test]
    fn solve_goal_one_sided_gets_full_budget() {
        let state = deep_universal_fixture();
        let caps = SolveCaps {
            node_cap: 3,
            tt_bytes_cap: 4096,
            semantic_horizon: u32::MAX,
        };
        let both = TssSolver::default().solve_goal(&state, &caps, SolveGoal::Both);
        let win = TssSolver::default().solve_goal(&state, &caps, SolveGoal::Win);
        assert_eq!(both.status, ProofStatus::Unknown);
        assert_eq!(win.status, ProofStatus::Win);
        assert!(TssVerifier.verify(&state, win.cert.as_ref().unwrap(), win.status));
    }

    fn wide_solver_with_dual_pass(dual_pass: bool) -> TssSolver {
        let mut solver = TssSolver::default();
        // Keep policy tests independent of the process-global warmth env used
        // by parallel campaign tests elsewhere in the crate.
        solver.set_shared_fragments_for_test(false);
        solver.configure_leaf_profile();
        solver.set_dual_pass(dual_pass);
        solver
    }

    fn wide_solver_with_loss_budget(dual_pass: bool, loss_reserve_nodes: u32) -> TssSolver {
        let mut solver = wide_solver_with_dual_pass(dual_pass);
        solver.set_loss_reserve_nodes(loss_reserve_nodes);
        solver
    }

    fn assert_deep_result_identical(
        actual: &DeepResult<TssCertificate>,
        expected: &DeepResult<TssCertificate>,
    ) {
        assert_eq!(actual.status, expected.status);
        assert_eq!(actual.cert, expected.cert);
        assert_eq!(format!("{:?}", actual.stats), format!("{:?}", expected.stats));
    }

    #[test]
    fn wide_both_dual_pass_recovers_a_cheap_verified_loss_within_budget() {
        let caps = SolveCaps {
            node_cap: 500,
            tt_bytes_cap: 256 << 10,
            semantic_horizon: u32::MAX,
        };

        // The existing forced-loss fixture is already a root lambda-one fact,
        // so today's flag-off `Both` path decides it before the budget split.
        // Pin that current behavior explicitly.
        let root_fact = forced_loss_fixture();
        let root_off = wide_solver_with_dual_pass(false)
            .solve_goal(&root_fact, &caps, SolveGoal::Both);
        let root_on = wide_solver_with_dual_pass(true)
            .solve_goal(&root_fact, &caps, SolveGoal::Both);
        assert_eq!(root_off.status, ProofStatus::Loss);
        assert_eq!(root_off.status, root_on.status);
        assert_eq!(root_off.stats.nodes, root_on.stats.nodes);
        assert!(TssVerifier.verify(
            &root_fact,
            root_off.cert.as_ref().unwrap(),
            root_off.status,
        ));

        // This existing non-lambda-one forced-defender fixture has a cheap
        // opponent-WIN proof, while the wide primal leaves budget unused.
        let state = xsnfyll_forced_defender_fixture();
        let off = wide_solver_with_dual_pass(false).solve_goal(&state, &caps, SolveGoal::Both);
        assert_eq!(off.status, ProofStatus::Unknown);
        assert!(off.cert.is_none());

        let on = wide_solver_with_dual_pass(true).solve_goal(&state, &caps, SolveGoal::Both);
        assert_eq!(on.status, ProofStatus::Loss);
        assert!(on.stats.nodes > off.stats.nodes, "both attempts must run");
        assert!(on.stats.nodes <= caps.node_cap);
        assert!(TssVerifier.verify(
            &state,
            on.cert.as_ref().expect("dual loss carries a certificate"),
            on.status,
        ));
    }

    #[test]
    fn wide_dual_pass_full_budget_primal_matches_flag_off() {
        let state = quiet_fixture();
        let caps = SolveCaps {
            node_cap: 2,
            tt_bytes_cap: 0,
            semantic_horizon: u32::MAX,
        };
        let off = wide_solver_with_dual_pass(false).solve_goal(&state, &caps, SolveGoal::Both);
        let on = wide_solver_with_dual_pass(true).solve_goal(&state, &caps, SolveGoal::Both);
        assert_eq!(off.status, on.status);
        assert_eq!(off.cert, on.cert);
        assert_eq!(off.stats.nodes, caps.node_cap);
        assert_eq!(on.stats.nodes, caps.node_cap);
    }

    #[test]
    fn wide_dual_pass_known_win_has_flag_parity() {
        let state = win_now_fixture();
        let caps = SolveCaps {
            node_cap: 500,
            tt_bytes_cap: 256 << 10,
            semantic_horizon: u32::MAX,
        };
        let off = wide_solver_with_dual_pass(false).solve_goal(&state, &caps, SolveGoal::Both);
        let on = wide_solver_with_dual_pass(true).solve_goal(&state, &caps, SolveGoal::Both);
        assert_eq!(off.status, ProofStatus::Win);
        assert_eq!(off.status, on.status);
        assert_eq!(off.cert, on.cert);
        assert_eq!(off.stats.nodes, on.stats.nodes);
    }

    #[test]
    fn wide_dual_pass_flag_off_keeps_legacy_primal_only_split() {
        let caps = SolveCaps {
            node_cap: 32,
            tt_bytes_cap: 256 << 10,
            semantic_horizon: u32::MAX,
        };
        for state in [quiet_fixture(), pair_width_first_stone_fixture()] {
            let both = wide_solver_with_dual_pass(false)
                .solve_goal(&state, &caps, SolveGoal::Both);
            let win = wide_solver_with_dual_pass(false).solve_goal(&state, &caps, SolveGoal::Win);
            assert_eq!(both.status, win.status);
            assert_eq!(both.cert, win.cert);
            assert_eq!(both.stats.nodes, win.stats.nodes);
        }
    }

    #[test]
    fn wide_loss_reserve_zero_is_bit_identical_with_dual_off_and_on() {
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 256 << 10,
            semantic_horizon: u32::MAX,
        };
        for dual_pass in [false, true] {
            for state in [
                quiet_fixture(),
                xsnfyll_forced_defender_fixture(),
                win_now_fixture(),
            ] {
                let implicit_zero = wide_solver_with_dual_pass(dual_pass)
                    .solve_goal(&state, &caps, SolveGoal::Both);
                let explicit_zero = wide_solver_with_loss_budget(dual_pass, 0)
                    .solve_goal(&state, &caps, SolveGoal::Both);
                assert_deep_result_identical(&implicit_zero, &explicit_zero);
            }
        }
    }

    #[test]
    fn wide_loss_reserve_preserves_an_early_exit_dual_loss() {
        let state = xsnfyll_forced_defender_fixture();
        let caps = SolveCaps {
            node_cap: 500,
            tt_bytes_cap: 256 << 10,
            semantic_horizon: u32::MAX,
        };
        let current = wide_solver_with_loss_budget(true, 0)
            .solve_goal(&state, &caps, SolveGoal::Both);
        let reserved = wide_solver_with_loss_budget(true, 64)
            .solve_goal(&state, &caps, SolveGoal::Both);
        assert_eq!(current.status, ProofStatus::Loss);
        assert_deep_result_identical(&reserved, &current);
        assert!(TssVerifier.verify(
            &state,
            reserved.cert.as_ref().expect("reserved loss carries a certificate"),
            reserved.status,
        ));
    }

    #[test]
    fn wide_loss_reserve_schedules_its_floor_without_leftover_policy() {
        let state = xsnfyll_forced_defender_fixture();
        let caps = SolveCaps {
            node_cap: 500,
            tt_bytes_cap: 256 << 10,
            semantic_horizon: u32::MAX,
        };
        let current = wide_solver_with_loss_budget(false, 0)
            .solve_goal(&state, &caps, SolveGoal::Both);
        let reserved = wide_solver_with_loss_budget(false, 64)
            .solve_goal(&state, &caps, SolveGoal::Both);
        assert_eq!(current.status, ProofStatus::Unknown);
        assert_eq!(reserved.status, ProofStatus::Loss);
        assert!(TssVerifier.verify(
            &state,
            reserved.cert.as_ref().expect("reserved loss carries a certificate"),
            reserved.status,
        ));
        assert!(reserved.stats.nodes <= caps.node_cap);
    }

    #[test]
    fn wide_loss_reserve_never_skips_a_nonempty_primal_allowance() {
        assert_eq!(wide_both_initial_caps(0, u32::MAX), (0, 0));
        assert_eq!(wide_both_initial_caps(1, u32::MAX), (1, 0));
        assert_eq!(wide_both_initial_caps(499, u32::MAX), (1, 498));
        assert_eq!(wide_both_initial_caps(499, 32), (467, 32));
    }

    #[test]
    fn wide_loss_reserve_keeps_combined_attempts_inside_the_original_cap() {
        let state = quiet_fixture();
        let caps = SolveCaps {
            node_cap: 3,
            tt_bytes_cap: 0,
            semantic_horizon: u32::MAX,
        };
        let current = wide_solver_with_loss_budget(true, 0)
            .solve_goal(&state, &caps, SolveGoal::Both);
        let reserved = wide_solver_with_loss_budget(true, 1)
            .solve_goal(&state, &caps, SolveGoal::Both);
        assert_eq!(current.status, ProofStatus::Unknown);
        assert_eq!(reserved.status, ProofStatus::Unknown);
        assert_eq!(current.stats.nodes, caps.node_cap);
        assert_eq!(reserved.stats.nodes, caps.node_cap);
        assert!(reserved.cert.is_none());
    }

    #[test]
    fn loss_reserve_is_inert_outside_wide_both() {
        let caps = SolveCaps {
            node_cap: 32,
            tt_bytes_cap: 256 << 10,
            semantic_horizon: u32::MAX,
        };
        for goal in [SolveGoal::Win, SolveGoal::Loss] {
            let state = quiet_fixture();
            let off = wide_solver_with_loss_budget(true, 0).solve_goal(&state, &caps, goal);
            let on = wide_solver_with_loss_budget(true, 16).solve_goal(&state, &caps, goal);
            assert_deep_result_identical(&on, &off);
        }

        let state = quiet_fixture();
        let off = TssSolver::default().solve_goal(&state, &caps, SolveGoal::Both);
        let mut on_solver = TssSolver::default();
        on_solver.set_loss_reserve_nodes(16);
        let on = on_solver.solve_goal(&state, &caps, SolveGoal::Both);
        assert_deep_result_identical(&on, &off);
    }

    #[test]
    fn quiet_no_hitting_universal_is_d6_cap_stable() {
        let state = quiet_fixture();
        let caps = SolveCaps {
            node_cap: 4,
            tt_bytes_cap: 0,
            semantic_horizon: u32::MAX,
        };
        let expected = TssSolver::default()
            .solve_goal(&state, &caps, SolveGoal::Loss)
            .status;
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let transformed = transformed_state(&state, symmetry);
            let status = TssSolver::default()
                .solve_goal(&transformed, &caps, SolveGoal::Loss)
                .status;
            assert_eq!(status, expected, "symmetry={symmetry}");
        }
    }

    #[test]
    fn make_unmake_round_trips_on_proof_and_cap_exit() {
        let original = deep_universal_fixture();
        for cap in [2, 500_000] {
            let mut work = original.clone();
            let before = work.clone();
            let mut context = NarrowCompatSearch::new(cap, 64 << 10, u64::MAX);
            let _ = context.prove(&mut work, Player::Player0, 0, None);
            assert_exact_state(&work, &before);
        }
    }

    #[test]
    fn node_cap_only_moves_results_toward_unknown() {
        let state = deep_universal_fixture();
        let small = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 3,
                tt_bytes_cap: 4096,
                semantic_horizon: u32::MAX,
            },
        );
        let enough = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 5,
                tt_bytes_cap: 4096,
                semantic_horizon: u32::MAX,
            },
        );
        assert_eq!(small.status, ProofStatus::Unknown);
        assert!(small.cert.is_none());
        assert_eq!(enough.status, ProofStatus::Win);
        assert!(TssVerifier.verify(&state, enough.cert.as_ref().unwrap(), enough.status));
        assert!(small.stats.nodes <= 3);
        assert!(enough.stats.nodes <= 5);
        assert!(small.stats.peak_tt_bytes <= 4096);
        assert!(enough.stats.peak_tt_bytes <= 4096);
    }

    #[test]
    #[ignore = "R1b sharpness fixture; run explicitly"]
    fn hunt_r1b_chain_sharpness() {
        for b in 2..=5u32 {
            let seed = HexCoord::new(8, 0);
            let target = HexCoord::new(8 * i16::try_from(b).unwrap(), 0);
            let binding_distance = i32::from(hex_distance(seed, target));
            assert_eq!(binding_distance, 8 * (i32::try_from(b).unwrap() - 1));
            assert!(binding_distance <= seed_band_radius(b));
            assert!(binding_distance > seed_band_radius(b - 1));
        }
    }

    #[test]
    #[ignore = "R1b production seed-band cross-check; run explicitly"]
    fn hunt_seed_band_matches_production() {
        let mut state = RustHexoState::new();
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::ZERO,
            },
        )
        .unwrap();
        let claimant = state.current_player();
        let target = HexCoord::new(32, 0);
        let witness = WindowKey {
            start: HexCoord::ZERO,
            axis: Axis::Q,
        };
        let arena = vec![
            CertNode::Win {
                witness,
                count: 5,
                budget: 2,
                resolution_ply: 3,
            },
            CertNode::Choice {
                mv: target,
                child: 0,
            },
        ];
        let edges = vec![CertEdge {
            mv: target,
            child: 1,
        }];
        let required = zone_certificate_extras(&state, claimant, 4, &edges, &arena).unwrap();

        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        let mut protected = witness.cells().to_vec();
        protected.push(target);
        let mut expected = protected
            .iter()
            .copied()
            .filter(|cell| legal.contains(cell))
            .collect::<Vec<_>>();
        expected.extend(
            legal
                .iter()
                .copied()
                .filter(|cell| i32::from(hex_distance(*cell, target)) <= seed_band_radius(4)),
        );
        expected.sort_by_key(|coord| raw_coord_key(*coord));
        expected.dedup();
        assert_eq!(required, expected);
        assert!(required.contains(&HexCoord::new(8, 0)));
        assert!(!required.contains(&HexCoord::new(0, 1)));
    }

    fn narrow_identity_pair(hash_mask: u64, zone: ZoneSearchCaps) -> (TssSolver, TssSolver) {
        let mut legacy = TssSolver::with_hash_mask(hash_mask);
        let mut migrated = TssSolver::with_hash_mask(hash_mask);
        legacy.set_zone_options(zone);
        migrated.set_zone_options(zone);
        (legacy, migrated)
    }

    /// The certificate has no public wire codec.  This test-only canonical
    /// encoder covers every field explicitly so identity is checked as bytes in
    /// addition to the type's structural equality.
    fn narrow_certificate_bytes(cert: &Option<TssCertificate>) -> Vec<u8> {
        fn put_u32(out: &mut Vec<u8>, value: u32) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        fn put_len(out: &mut Vec<u8>, value: usize) {
            out.extend_from_slice(&(value as u64).to_le_bytes());
        }
        fn put_player(out: &mut Vec<u8>, player: Player) {
            out.push(player_code(player));
        }
        fn put_coord(out: &mut Vec<u8>, coord: HexCoord) {
            out.extend_from_slice(&coord.q.to_le_bytes());
            out.extend_from_slice(&coord.r.to_le_bytes());
        }
        fn put_window(out: &mut Vec<u8>, window: WindowKey) {
            put_coord(out, window.start);
            out.push(window.axis.index());
        }

        let mut out = Vec::new();
        let Some(cert) = cert else {
            out.push(0);
            return out;
        };
        out.push(1);
        put_len(&mut out, cert.root.occupancy.len());
        for coord in &cert.root.occupancy {
            put_coord(&mut out, *coord);
        }
        put_len(&mut out, cert.root.owners.len());
        for owner in &cert.root.owners {
            put_player(&mut out, *owner);
        }
        put_player(&mut out, cert.root.current_player);
        match cert.root.phase {
            TurnPhase::Opening => out.push(0),
            TurnPhase::FirstStone => out.push(1),
            TurnPhase::SecondStone { first } => {
                out.push(2);
                put_coord(&mut out, first);
            }
        }
        put_u32(&mut out, cert.root.placements_made);
        match cert.root.terminal {
            None => out.push(0),
            Some(outcome) => {
                out.push(1);
                put_player(&mut out, outcome.winner);
                put_u32(&mut out, outcome.placements);
            }
        }
        put_player(&mut out, cert.claimant);
        put_u32(&mut out, cert.root_node);
        put_len(&mut out, cert.nodes.len());
        for node in &cert.nodes {
            match node {
                // §2.5: the legacy canonical helper stays legacy-only; the v3
                // transcript (tags 5/6) is a separate identity encoding.
                CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => {
                    panic!("legacy canonical encoder is legacy-only")
                }
                CertNode::OrCompletion {
                    mv,
                    witness,
                    completion_ply,
                } => {
                    out.push(0);
                    put_coord(&mut out, *mv);
                    put_window(&mut out, *witness);
                    put_u32(&mut out, *completion_ply);
                }
                CertNode::Win {
                    witness,
                    count,
                    budget,
                    resolution_ply,
                } => {
                    out.push(1);
                    put_window(&mut out, *witness);
                    out.push(*count);
                    out.push(*budget);
                    put_u32(&mut out, *resolution_ply);
                }
                CertNode::Loss {
                    witnesses,
                    resolution_ply,
                } => {
                    out.push(2);
                    put_len(&mut out, witnesses.len());
                    for witness in witnesses {
                        put_window(&mut out, *witness);
                    }
                    put_u32(&mut out, *resolution_ply);
                }
                CertNode::Choice { mv, child } => {
                    out.push(3);
                    put_coord(&mut out, *mv);
                    put_u32(&mut out, *child);
                }
                CertNode::Universal {
                    edges,
                    implicit_dispatch,
                    zone,
                    commutations,
                } => {
                    out.push(4);
                    put_len(&mut out, edges.len());
                    for edge in edges {
                        put_coord(&mut out, edge.mv);
                        put_u32(&mut out, edge.child);
                    }
                    out.push(u8::from(*implicit_dispatch));
                    match zone {
                        None => out.push(0),
                        Some(zone) => {
                            out.push(1);
                            put_u32(&mut out, zone.d);
                            put_u32(&mut out, zone.build_horizon);
                        }
                    }
                    put_len(&mut out, commutations.len());
                    for item in commutations {
                        put_coord(&mut out, item.first);
                        put_coord(&mut out, item.omitted_second);
                        put_u32(&mut out, item.first_child);
                        put_u32(&mut out, item.mirror_child);
                    }
                }
            }
        }
        put_u32(&mut out, cert.semantic_horizon);
        out
    }

    fn narrow_replay_string(state: &RustHexoState) -> String {
        state
            .placement_history()
            .iter()
            .map(|record| format!("({}, {})", record.coord.q, record.coord.r))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn assert_narrow_identity(
        label: &str,
        state: &RustHexoState,
        caps: &SolveCaps,
        goal: SolveGoal,
        legacy: &mut TssSolver,
        migrated: &mut TssSolver,
    ) -> (SolveStats, SolveStats) {
        let replay = narrow_replay_string(state);
        let old = legacy.solve_goal(state, caps, goal);
        let new = migrated.solve_goal(state, caps, goal);
        let context = format!(
            "case={label} goal={goal:?} node_cap={} tt_bytes_cap={} horizon={} replay=[{replay}]",
            caps.node_cap, caps.tt_bytes_cap, caps.semantic_horizon
        );

        assert_eq!(old.status, new.status, "status mismatch: {context}");
        assert_eq!(old.stats.nodes, new.stats.nodes, "node mismatch: {context}");
        assert_eq!(
            old.stats.tt_hits, new.stats.tt_hits,
            "TT-hit mismatch: {context}"
        );
        assert_eq!(
            old.stats.peak_tt_bytes, new.stats.peak_tt_bytes,
            "TT-byte mismatch: {context}"
        );
        assert_eq!(old.cert, new.cert, "certificate mismatch: {context}");
        assert_eq!(
            narrow_certificate_bytes(&old.cert),
            narrow_certificate_bytes(&new.cert),
            "certificate-byte mismatch: {context}"
        );
        assert_eq!(
            legacy.last_narrow_signatures, migrated.last_narrow_signatures,
            "solve-local TT behavior mismatch: {context}"
        );
        assert_eq!(
            legacy.shared_tt, migrated.shared_tt,
            "persistent TT behavior mismatch: {context}"
        );
        if let Some(cert) = old.cert.as_ref() {
            assert!(
                TssVerifier.verify(state, cert, old.status),
                "legacy certificate rejected: {context}"
            );
            assert!(
                TssVerifier.verify_with_dispatch_oracle(state, cert, old.status),
                "legacy certificate rejected by dispatch oracle: {context}"
            );
        } else {
            assert_eq!(
                old.status,
                ProofStatus::Unknown,
                "certless hard result: {context}"
            );
        }
        (old.stats, new.stats)
    }

    #[derive(Clone, Copy)]
    struct NarrowIdentityRng(u64);

    impl NarrowIdentityRng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    #[test]
    #[ignore = "round-5 C1 full fixture and 512-position identity gate"]
    fn tss_round5_narrow_compat_identity() {
        let production_caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 256 << 10,
            semantic_horizon: u32::MAX,
        };

        // Deterministic certificates, goal filtering, semantic deadlines, and
        // all three phases from the default narrow fixture family.
        let fixtures = [
            ("quiet", quiet_fixture()),
            ("forced-defense", forced_defense_fixture()),
            ("win-now", win_now_fixture()),
            ("forced-loss", forced_loss_fixture()),
            ("spare-tempo", spare_tempo_fixture()),
            ("deep-universal", deep_universal_fixture()),
        ];
        let (mut legacy, mut migrated) = narrow_identity_pair(u64::MAX, ZoneSearchCaps::default());
        for (label, state) in &fixtures {
            for goal in [SolveGoal::Win, SolveGoal::Loss, SolveGoal::Both] {
                assert_narrow_identity(
                    label,
                    state,
                    &production_caps,
                    goal,
                    &mut legacy,
                    &mut migrated,
                );
            }
        }
        let expired = SolveCaps {
            semantic_horizon: deep_universal_fixture().placements_made().saturating_sub(1),
            ..production_caps
        };
        assert_narrow_identity(
            "expired-horizon",
            &deep_universal_fixture(),
            &expired,
            SolveGoal::Win,
            &mut legacy,
            &mut migrated,
        );

        // A warm second solve must reproduce the exact imported-fragment path,
        // including its lower node count and positive TT hit.
        let warm_caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 8 << 20,
            semantic_horizon: u32::MAX,
        };
        let warm_state = deep_universal_fixture();
        let (mut warm_legacy, mut warm_migrated) =
            narrow_identity_pair(u64::MAX, ZoneSearchCaps::default());
        let (cold, _) = assert_narrow_identity(
            "persistent-cold",
            &warm_state,
            &warm_caps,
            SolveGoal::Win,
            &mut warm_legacy,
            &mut warm_migrated,
        );
        let (warm, _) = assert_narrow_identity(
            "persistent-warm",
            &warm_state,
            &warm_caps,
            SolveGoal::Win,
            &mut warm_legacy,
            &mut warm_migrated,
        );
        assert!(warm.tt_hits > 0, "warm solve must import a shared fragment");
        assert!(warm.nodes < cold.nodes, "warm solve must reduce node count");

        // Forced bucket collisions exercise full-key local and shared TT
        // equality.  Repeating each solve also covers cache warmth under the
        // collision mask.
        let (mut collision_legacy, mut collision_migrated) =
            narrow_identity_pair(0, ZoneSearchCaps::default());
        for (label, state) in &fixtures {
            for repeat in 0..2 {
                assert_narrow_identity(
                    &format!("collision-{label}-{repeat}"),
                    state,
                    &production_caps,
                    SolveGoal::Win,
                    &mut collision_legacy,
                    &mut collision_migrated,
                );
            }
        }

        // Zone-enabled narrow certificates and all D6 images retain exact
        // bytes, while the independent verifier rechecks each hard claim.
        let zone = ZoneSearchCaps {
            enabled: true,
            stale_area_filter: false,
            count2_threshold: false,
            pair_commutation: false,
        };
        let (mut zone_legacy, mut zone_migrated) = narrow_identity_pair(u64::MAX, zone);
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let state = transformed_state(&deep_universal_fixture(), symmetry);
            assert_narrow_identity(
                &format!("zone-d6-{symmetry}"),
                &state,
                &production_caps,
                SolveGoal::Win,
                &mut zone_legacy,
                &mut zone_migrated,
            );
        }

        // Fixed-seed randomized sweep: at least 512 positions, all phases,
        // both one-sided goals, cap exits, zero/small TT, and production TT.
        let mut corpus = Vec::new();
        let mut phases = [false; 3];
        let mut seed = 1u64;
        while corpus.len() < 512 {
            let mut rng = NarrowIdentityRng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut state = RustHexoState::new();
            for snapshot in 0..32usize {
                let phase = match state.phase() {
                    TurnPhase::Opening => 0,
                    TurnPhase::FirstStone => 1,
                    TurnPhase::SecondStone { .. } => 2,
                };
                phases[phase] = true;
                corpus.push((seed, snapshot, state.clone()));
                if corpus.len() >= 512 {
                    break;
                }
                let mut legal = Vec::new();
                state.write_legal_moves(&mut legal);
                if legal.is_empty() {
                    break;
                }
                let coord = legal[(rng.next() as usize) % legal.len()];
                let result = apply_placement(&mut state, Placement { coord }).unwrap();
                if result.outcome.is_some() {
                    break;
                }
            }
            seed = seed.saturating_add(1);
        }
        assert!(phases.into_iter().all(|seen| seen));
        assert!(corpus.len() >= 512);

        let random_caps = [
            SolveCaps {
                node_cap: 0,
                tt_bytes_cap: 0,
                semantic_horizon: u32::MAX,
            },
            SolveCaps {
                node_cap: 1,
                tt_bytes_cap: 1024,
                semantic_horizon: u32::MAX,
            },
            SolveCaps {
                node_cap: 32,
                tt_bytes_cap: 256 << 10,
                semantic_horizon: u32::MAX,
            },
            SolveCaps {
                node_cap: 64,
                tt_bytes_cap: 256 << 10,
                semantic_horizon: u32::MAX,
            },
        ];
        let (mut random_legacy, mut random_migrated) =
            narrow_identity_pair(u64::MAX, ZoneSearchCaps::default());
        for (index, (seed, snapshot, state)) in corpus.iter().enumerate() {
            let mut caps = random_caps[index % random_caps.len()];
            if index % 7 == 0 {
                caps.semantic_horizon = state.placements_made().saturating_add(3);
            }
            for goal in [SolveGoal::Win, SolveGoal::Loss] {
                assert_narrow_identity(
                    &format!("random-seed-{seed}-snapshot-{snapshot}"),
                    state,
                    &caps,
                    goal,
                    &mut random_legacy,
                    &mut random_migrated,
                );
            }
        }
    }
}
