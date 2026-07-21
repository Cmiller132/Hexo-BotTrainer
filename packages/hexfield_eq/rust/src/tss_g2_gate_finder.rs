//! Finder-side FHW closure builder for the v1 Group-2 `FhwGateV1` gate class
//! (.codex-g2-resolve/DESIGN_G2_CERT_EXTENSION.md §5 + §3.3, and
//! DESIGN_AMENDMENT_R1_R2.md).
//!
//! SCOPE (the largest SOUND subset deliverable by this lane; see
//! `.gate/G2_FINDER_CLOSURE_REPORT.md`):
//!
//!   * The **structural closure** — H_Q, F_Q, exact bounded transversal
//!     (b in {1,2}), kernel K, representatives R, retraction phi, and the
//!     per-`(d,s)` edge classification (Exact / FrontierCovered /
//!     NonFrontierCovered). Every one of these is derivable purely from the
//!     gate position `P_Q` and a single representative ghost placement
//!     `G = P_Q + s`; NO proven representative subtree is required, so it is
//!     computable and self-checkable in-lane. This is the reductive core that
//!     answers "do reductive gates (R subsetneq K) exist at production-shaped
//!     nodes" (the 40.5%-ceiling question).
//!
//!   * The **row classifiers** — `classify_role` (3 `FhwRoleRowV1` leaves) and
//!     `classify_window` (9 `FhwKappaRowV1` leaves) — pure functions of board
//!     geometry plus the two subtree clock inputs `k = f_cut(C_s,rho)` and
//!     `q = Q_cut(C_s,W)`. These MIRROR the design §3.3 tables exactly (they
//!     are what the future accept path will recompute), so feeding them real
//!     ghost geometry plus a specified `q`/`k` produces a positive fixture that
//!     is derivably true modulo the clock provenance.
//!
//! DELIBERATE BOUNDARY (documented, not papered over): the clock scalars
//! `f_cut(C_s,rho)`/`Q_cut(C_s,W)` come from a proven representative subtree
//! `C_s`, which only the native-PN Open/Closed closure (design §5, out of this
//! lane's scope) can produce for a production node. Here they are inputs:
//! supplied by a constructed subtree in fixtures, and NOT emitted for corpus
//! nodes (where the structural reduction is measured, not a fully-rowed cert).
//!
//! FAIL-CLOSED: every builder path and classifier returns `None`/an error on
//! any self-check failure, mandatory-guard failure, or arithmetic overflow —
//! never a permissive fallback. A gate is emitted only when every self-check
//! passes.
//!
//! This module is `#[cfg(test)]`-only: it is not compiled into the production
//! binary, so the flag-off / golden-digest bit-identity discipline holds by
//! construction (there is no new production code path). It never imports or
//! mutates the verifier accept path.

use std::collections::HashSet;

use hexo_engine::{
    apply_placement, hex_distance, Axis, HexCoord, HexoState as RustHexoState, Placement, Player,
    TurnPhase, WindowKey,
};

use crate::threats_shared;
use crate::tss_verify::{
    FhwEdgeClassV1, FhwKappaRowV1, FhwRoleRowV1, GuardResultV1,
};

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Inclusive radius-`radius` axial ball around `center`. `ball(c, 8)` is the
/// 217-cell B_8 ball used everywhere by FHW-T3-R.
fn ball(center: HexCoord, radius: i16) -> Vec<HexCoord> {
    let mut out = Vec::new();
    for dq in -radius..=radius {
        let r_min = (-radius).max(-dq - radius);
        let r_max = radius.min(-dq + radius);
        for dr in r_min..=r_max {
            out.push(HexCoord {
                q: center.q + dq,
                r: center.r + dr,
            });
        }
    }
    out
}

fn coord_key(c: HexCoord) -> (i16, i16) {
    (c.q, c.r)
}

/// The ghost state `G = P_Q + s`: the gate position with the single
/// representative reply `s` applied. Used for `Lambda(G)`, `GI(G)`, and FC/RC/WC.
#[derive(Clone)]
struct Ghost {
    state: RustHexoState,
    /// Materialized `Lambda(G) = union over occupied x of B_8(x)`.
    lambda: HashSet<(i16, i16)>,
}

impl Ghost {
    fn new(gate: &RustHexoState, s: HexCoord) -> Option<Self> {
        let mut state = gate.clone();
        // `s in K subseteq Legal`, so this placement is engine-legal. A
        // terminal result is still a valid board for the geometry below.
        apply_placement(&mut state, Placement { coord: s }).ok()?;
        let mut lambda: HashSet<(i16, i16)> = HashSet::new();
        for &occ in state.board().occupied_cells() {
            for cell in ball(occ, 8) {
                lambda.insert(coord_key(cell));
            }
        }
        Some(Self { state, lambda })
    }

    fn in_lambda(&self, cell: HexCoord) -> bool {
        self.lambda.contains(&coord_key(cell))
    }

    /// `GI(G)(z)`: `z` is neither occupied nor legal in `G` (design §3.2).
    fn is_ghost_illegal(&self, z: HexCoord) -> bool {
        self.state.board().get(z).is_none() && !self.state.board().legal_moves().contains(z)
    }
}

/// FC predicate (design §3.3): `d == s` or every one of the 217 cells of
/// `B_8(d)` lies in `Lambda(G)`.
fn frontier_covered(d: HexCoord, s: HexCoord, ghost: &Ghost) -> bool {
    if d == s {
        return true;
    }
    ball(d, 8).into_iter().all(|z| ghost.in_lambda(z))
}

/// RC predicate (design §3.3): `GI(G) intersect B_8(d) intersect B_{8(k-1)}(y)`
/// is empty; the last ball is empty when `k == 0`.
fn rc_pass(d: HexCoord, y: HexCoord, k: u32, ghost: &Ghost) -> bool {
    if k == 0 {
        return true; // empty inner ball => intersection empty
    }
    let inner_radius = 8i32.checked_mul(i32::try_from(k - 1).unwrap_or(i32::MAX)).unwrap_or(i32::MAX);
    let inner_radius = i16::try_from(inner_radius).unwrap_or(i16::MAX);
    let d_ball: HashSet<(i16, i16)> = ball(d, 8).into_iter().map(coord_key).collect();
    // Intersection empty iff no cell of B_8(d) is simultaneously ghost-illegal
    // and within B_{8(k-1)}(y).
    !ball(y, inner_radius)
        .into_iter()
        .any(|z| d_ball.contains(&coord_key(z)) && ghost.is_ghost_illegal(z))
}

/// WC predicate (design §3.3): `GI(G) intersect B_8(d) intersect B_{8(q-6)}(W)`
/// is empty. Only queried on the non-FC / all-empty / nonincident / `q>=6`
/// branch.
fn wc_pass(d: HexCoord, window: WindowKey, q: u32, ghost: &Ghost) -> bool {
    let radius = 8i32
        .checked_mul(i32::try_from(q.saturating_sub(6)).unwrap_or(i32::MAX))
        .unwrap_or(i32::MAX);
    let radius = i16::try_from(radius).unwrap_or(i16::MAX);
    let d_ball: HashSet<(i16, i16)> = ball(d, 8).into_iter().map(coord_key).collect();
    // Union of B_{radius}(w) over the six window cells.
    for w in window.cells() {
        for z in ball(w, radius) {
            if d_ball.contains(&coord_key(z)) && ghost.is_ghost_illegal(z) {
                return false;
            }
        }
    }
    true
}

fn window_distance(cell: HexCoord, key: WindowKey) -> u32 {
    key.cells()
        .iter()
        .map(|w| i32::from(hex_distance(cell, *w)).unsigned_abs())
        .min()
        .unwrap_or(u32::MAX)
}

// ---------------------------------------------------------------------------
// Threat family and exact bounded transversal
// ---------------------------------------------------------------------------

/// A single attacker (claimant) threat: its window key and its empties `E(U)`.
#[derive(Clone, Debug)]
struct Threat {
    window: WindowKey,
    empties: Vec<HexCoord>,
}

/// H_Q: the current attacker (claimant) threat family at `state`, canonical
/// sorted-unique by `(axis, start.q, start.r)`. Each key is validated as a real
/// A-threat: attacker count >= 4 and zero defender stones (attacker-alive).
fn attacker_threats(state: &RustHexoState, claimant: Player) -> Vec<Threat> {
    let mut threats: Vec<Threat> = Vec::new();
    for entry in state.board().windows().threat_entries(claimant) {
        // `threat_entries(claimant)` yields active windows with count(claimant)
        // >= 4; active means zero opponent stones, so attacker-alive holds.
        if entry.count(claimant) < 4 {
            continue;
        }
        let empties = entry.empty_cells();
        if empties.is_empty() {
            continue; // count 6 == already won; not a live b-threat
        }
        threats.push(Threat {
            window: entry.key(),
            empties,
        });
    }
    threats.sort_by_key(|t| (t.window.axis.index(), t.window.start.q, t.window.start.r));
    threats.dedup_by_key(|t| (t.window.axis.index(), t.window.start.q, t.window.start.r));
    threats
}

/// Exact transversal number of `family` (each set = a threat's empties),
/// capped at `cap`. Returns the least `k <= cap` hitting every set, or
/// `cap + 1` when more than `cap` are needed. An empty family has transversal
/// number 0. A set with no empties is treated as unhittable (returns `cap + 1`).
fn transversal_number(family: &[Vec<HexCoord>], cap: u8) -> u8 {
    if family.is_empty() {
        return 0;
    }
    if family.iter().any(Vec::is_empty) {
        return cap.saturating_add(1);
    }
    let mut universe: Vec<HexCoord> = Vec::new();
    for set in family {
        for &c in set {
            if !universe.contains(&c) {
                universe.push(c);
            }
        }
    }
    // k = 1: a single cell common to every set.
    if cap >= 1 && universe.iter().any(|c| family.iter().all(|s| s.contains(c))) {
        return 1;
    }
    // k = 2: any pair covering every set.
    if cap >= 2 {
        for i in 0..universe.len() {
            for j in (i + 1)..universe.len() {
                let (a, b) = (universe[i], universe[j]);
                if family.iter().all(|s| s.contains(&a) || s.contains(&b)) {
                    return 2;
                }
            }
        }
    }
    cap.saturating_add(1)
}

// ---------------------------------------------------------------------------
// Row classifiers (design §3.3 tables) — pure, faithful to the accept-path
// recompute. `k = f_cut(C_s,rho)` and `q = Q_cut(C_s,W)` are inputs.
// ---------------------------------------------------------------------------

/// Classify a ghost-role row for real reply `d`, representative `s`, and a live
/// role carried by `y` with `k = f_cut(C_s,rho)`. Returns `(row, epsilon)` or
/// `None` when a mandatory condition fails (carrier not avoided, or D22-N radius
/// fails on a charged non-FC ghost-illegal role) — fail-closed.
fn classify_role(
    edge_class: FhwEdgeClassV1,
    d: HexCoord,
    y: HexCoord,
    k: u32,
    ghost: &Ghost,
) -> Option<(FhwRoleRowV1, u8)> {
    // Every row requires `d` to avoid the carrier cell.
    if d == y {
        return None;
    }
    match edge_class {
        FhwEdgeClassV1::Exact | FhwEdgeClassV1::FrontierCovered => {
            Some((FhwRoleRowV1::ExactOrFcZero, 0))
        }
        FhwEdgeClassV1::NonFrontierCovered => {
            let ghost_illegal = ghost.is_ghost_illegal(y);
            if ghost_illegal {
                if rc_pass(d, y, k, ghost) {
                    Some((FhwRoleRowV1::NonFcRcZero, 0))
                } else {
                    // epsilon = 1; mandatory D22-N: dist(d,y) > 8k.
                    let radius = 8u32.checked_mul(k)?;
                    if u32::from(hex_distance(d, y).unsigned_abs()) > radius {
                        Some((FhwRoleRowV1::NonFcCharged, 1))
                    } else {
                        None // mandatory guard fails => reject the gate
                    }
                }
            } else {
                // A non-FC ghost-legal role is kept on the conservative charged
                // row; D22-N is required only for a ghost-illegal role.
                Some((FhwRoleRowV1::NonFcCharged, 1))
            }
        }
    }
}

/// Geometry summary of one demand window `W` at the gate position `P_Q`.
struct WindowGeom {
    d_alive: bool,
    touched: bool,
    all_empty: bool,
    cnt_d: u32,
}

/// Derive the window geometry at the gate: `D_alive` (defender can still fill it
/// -> no claimant stone), `touched` (>=1 defender stone), `all_empty`, and
/// `cnt_D`.
fn window_geom(gate: &RustHexoState, claimant: Player, window: WindowKey) -> WindowGeom {
    let defender = claimant.other();
    let mut claimant_ct = 0u32;
    let mut defender_ct = 0u32;
    for cell in window.cells() {
        match gate.board().get(cell) {
            Some(p) if p == claimant => claimant_ct += 1,
            Some(p) if p == defender => defender_ct += 1,
            _ => {}
        }
    }
    let d_alive = claimant_ct == 0;
    WindowGeom {
        d_alive,
        touched: d_alive && defender_ct >= 1,
        all_empty: claimant_ct == 0 && defender_ct == 0,
        cnt_d: defender_ct,
    }
}

/// Classify a `(d, s, W)` window row with `q = Q_cut(C_s,W)`. Mirrors design
/// §3.3's ordered, mutually exclusive table exactly. Returns
/// `(row, kappa, guard)` or `None` when a mandatory retained guard fails —
/// fail-closed (a failed guard rejects even if a finder wrote `Pass`).
fn classify_window(
    edge_class: FhwEdgeClassV1,
    d: HexCoord,
    window: WindowKey,
    q: u32,
    geom: &WindowGeom,
    ghost: &Ghost,
) -> Option<(FhwKappaRowV1, u8, GuardResultV1)> {
    use FhwKappaRowV1 as Row;
    use GuardResultV1 as Guard;

    if !geom.d_alive {
        return Some((Row::NonDAlive, 0, Guard::NotApplicable));
    }
    let d_in = window.contains(d);
    let exact_or_fc = matches!(
        edge_class,
        FhwEdgeClassV1::Exact | FhwEdgeClassV1::FrontierCovered
    );

    if exact_or_fc {
        if !d_in {
            return Some((Row::ExactOrFcNonIncident, 0, Guard::NotApplicable));
        }
        // d in W: direct increment; retained guard must pass.
        let guard_ok = if geom.touched {
            geom.cnt_d.checked_add(1)?.checked_add(q)? < 6
        } else {
            // all-empty incident window
            1u32.checked_add(q)? < 6
        };
        return guard_ok.then_some((Row::ExactOrFcDirect, 1, Guard::Pass));
    }

    // non-FC
    if geom.touched {
        if !d_in {
            return Some((Row::NonFcTouchedNonIncident, 0, Guard::NotApplicable));
        }
        let guard_ok = geom.cnt_d.checked_add(1)?.checked_add(q)? < 6;
        return guard_ok.then_some((Row::NonFcTouchedDirect, 1, Guard::Pass));
    }

    // non-FC, all-empty
    if d_in {
        let guard_ok = 1u32.checked_add(q)? < 6;
        return guard_ok.then_some((Row::NonFcEmptyDirect, 1, Guard::Pass));
    }
    // non-FC, all-empty, d not in W
    if q < 6 {
        return Some((Row::NonFcEmptyNonIncidentQlt6, 0, Guard::NotApplicable));
    }
    if wc_pass(d, window, q, ghost) {
        return Some((Row::NonFcEmptyNonIncidentWcPass, 0, Guard::Pass));
    }
    // WC fails: N-virgin mandatory guard dist(d,W) > 8(1 + q - 6).
    let virgin_radius = 8u32.checked_mul(1u32.checked_add(q)?.checked_sub(6)?)?;
    (window_distance(d, window) > virgin_radius).then_some((Row::NonFcEmptyNonIncidentWcFail, 1, Guard::Pass))
}

// ---------------------------------------------------------------------------
// Structural closure builder
// ---------------------------------------------------------------------------

/// Why a gate closure attempt failed (histogram bins for the corpus firing
/// measurement). Ordered roughly by how early the check runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClosureFail {
    NotDefenderToMove,
    Opening,
    Terminal,
    OwnWinNow,
    BudgetOutOfRange,
    NoThreats,
    ThreatCountOutOfRange,
    TransversalNotB,
    KernelEmpty,
    KernelReplyTerminal,
    EscapeHorizonExceeded,
}

/// One derived `(d, s)` edge of the structural gate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateEdge {
    pub real_reply: HexCoord,
    pub representative: HexCoord,
    pub edge_class: FhwEdgeClassV1,
}

/// A successfully closed structural gate (no proven subtree; role/window rows
/// are NOT emitted here — see module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct GateBuild {
    pub b: u8,
    pub threats: Vec<WindowKey>,
    pub kernel: Vec<HexCoord>,
    pub representatives: Vec<HexCoord>,
    pub edges: Vec<GateEdge>,
    pub escape_resolution_ply: u32,
    pub legal_count: usize,
}

impl GateBuild {
    /// Kernel reduction: |K| / |Legal|.
    pub fn kernel_ratio(&self) -> f64 {
        self.kernel.len() as f64 / self.legal_count.max(1) as f64
    }
    /// Representative reduction: |R| / |K|.
    pub fn representative_ratio(&self) -> f64 {
        self.representatives.len() as f64 / self.kernel.len().max(1) as f64
    }
    /// True when the gate is genuinely reductive: R subsetneq K.
    pub fn is_reductive(&self) -> bool {
        self.representatives.len() < self.kernel.len()
    }
}

/// Attempt the structural FHW closure at `gate` for `claimant`, with the
/// certificate's `semantic_horizon`. Returns the closed gate or the reason it
/// could not close. Every condition is a fail-closed self-check.
pub fn try_build_gate(
    gate: &RustHexoState,
    claimant: Player,
    semantic_horizon: u32,
) -> Result<GateBuild, ClosureFail> {
    // --- Eligibility (design §3.3 preamble + R2) ---
    if gate.current_player() != claimant.other() {
        return Err(ClosureFail::NotDefenderToMove);
    }
    if matches!(gate.phase(), TurnPhase::Opening) {
        return Err(ClosureFail::Opening);
    }
    if gate.is_terminal() {
        return Err(ClosureFail::Terminal);
    }
    let analysis = threats_shared::analyze(gate);
    if analysis.own_win_now {
        return Err(ClosureFail::OwnWinNow);
    }
    let b = match gate.phase() {
        TurnPhase::FirstStone => 2u8,
        TurnPhase::SecondStone { .. } => 1u8,
        TurnPhase::Opening => return Err(ClosureFail::Opening),
    };
    if b == 0 || b > 2 {
        return Err(ClosureFail::BudgetOutOfRange);
    }

    // --- H_Q, F_Q ---
    let threat_list = attacker_threats(gate, claimant);
    if threat_list.is_empty() {
        return Err(ClosureFail::NoThreats);
    }
    // b=1 => exactly one named threat; b=2 => one through three (v1 |K|<=6).
    let ok_count = match b {
        1 => threat_list.len() == 1,
        2 => (1..=3).contains(&threat_list.len()),
        _ => false,
    };
    if !ok_count {
        return Err(ClosureFail::ThreatCountOutOfRange);
    }
    let family: Vec<Vec<HexCoord>> = threat_list.iter().map(|t| t.empties.clone()).collect();

    // --- exact transversal == b ---
    if transversal_number(&family, 2) != b {
        return Err(ClosureFail::TransversalNotB);
    }

    // --- kernel K = { d in Legal : transversal(F_Q \ d) <= b-1 } ---
    let mut legal = Vec::new();
    gate.write_legal_moves(&mut legal);
    legal.sort_by_key(|c| coord_key(*c));
    let legal_count = legal.len();
    let mut kernel: Vec<HexCoord> = Vec::new();
    for &d in &legal {
        let residual: Vec<Vec<HexCoord>> = family
            .iter()
            .filter(|set| !set.contains(&d))
            .cloned()
            .collect();
        if transversal_number(&residual, 2) <= b.saturating_sub(1) {
            kernel.push(d);
        }
    }
    if kernel.is_empty() {
        return Err(ClosureFail::KernelEmpty);
    }
    // Every kernel reply applied must be nonterminal.
    for &d in &kernel {
        let mut probe = gate.clone();
        match apply_placement(&mut probe, Placement { coord: d }) {
            Ok(result) => {
                if result.outcome.is_some() {
                    return Err(ClosureFail::KernelReplyTerminal);
                }
            }
            Err(_) => return Err(ClosureFail::KernelReplyTerminal),
        }
    }

    // --- R and phi: minimal FC-cover (Exact when d==s) ---
    // Greedy: each still-uncovered d that becomes a representative covers every
    // d' with FC(d', d). phi(d') = the representative covering it (itself when
    // d' is a representative). This yields R subseteq K with genuine reduction
    // where the FC frontier is wide.
    let mut ghosts: Vec<(HexCoord, Ghost)> = Vec::new();
    for &s in &kernel {
        if let Some(g) = Ghost::new(gate, s) {
            ghosts.push((s, g));
        }
    }
    let ghost_of = |s: HexCoord| ghosts.iter().find(|(c, _)| *c == s).map(|(_, g)| g);

    let mut uncovered: HashSet<(i16, i16)> = kernel.iter().map(|c| coord_key(*c)).collect();
    let mut representatives: Vec<HexCoord> = Vec::new();
    let mut phi: Vec<(HexCoord, HexCoord)> = Vec::new(); // (d, s)
    // Deterministic order: kernel is already sorted.
    for &s in &kernel {
        if !uncovered.contains(&coord_key(s)) {
            continue;
        }
        let Some(g) = ghost_of(s) else { continue };
        representatives.push(s);
        // Cover every uncovered d' with FC(d', s).
        let newly: Vec<HexCoord> = kernel
            .iter()
            .copied()
            .filter(|&dprime| {
                uncovered.contains(&coord_key(dprime)) && frontier_covered(dprime, s, g)
            })
            .collect();
        for dprime in newly {
            uncovered.remove(&coord_key(dprime));
            phi.push((dprime, s));
        }
    }
    // FC need not cover the whole kernel; any residue is Exact (d == s). Every
    // still-uncovered d becomes its own representative.
    for &d in &kernel {
        if uncovered.remove(&coord_key(d)) {
            representatives.push(d);
            phi.push((d, d));
        }
    }
    representatives.sort_by_key(|c| coord_key(*c));
    representatives.dedup();
    phi.sort_by_key(|(d, _)| coord_key(*d));

    // Build the classified edge list.
    let mut edges: Vec<GateEdge> = Vec::with_capacity(phi.len());
    for &(d, s) in &phi {
        let edge_class = if d == s {
            FhwEdgeClassV1::Exact
        } else if ghost_of(s).is_some_and(|g| frontier_covered(d, s, g)) {
            FhwEdgeClassV1::FrontierCovered
        } else {
            FhwEdgeClassV1::NonFrontierCovered
        };
        edges.push(GateEdge {
            real_reply: d,
            representative: s,
            edge_class,
        });
    }

    // --- escape deadline (R1): p(Q) + b + 2, must fit the horizon ---
    let escape_resolution_ply = gate
        .placements_made()
        .checked_add(u32::from(b))
        .and_then(|v| v.checked_add(2))
        .ok_or(ClosureFail::EscapeHorizonExceeded)?;
    if escape_resolution_ply > semantic_horizon {
        return Err(ClosureFail::EscapeHorizonExceeded);
    }

    let build = GateBuild {
        b,
        threats: threat_list.iter().map(|t| t.window).collect(),
        kernel,
        representatives,
        edges,
        escape_resolution_ply,
        legal_count,
    };

    // --- structural self-checks (fail-closed) ---
    self_check_structural(gate, claimant, &build)?;

    Ok(build)
}

/// Recompute the theorem-side structural conditions for an emitted gate and
/// reject on any mismatch. These are the correctness evidence in the absence of
/// the accept path (design §3.3). Failure maps to a closure error.
fn self_check_structural(
    gate: &RustHexoState,
    claimant: Player,
    build: &GateBuild,
) -> Result<(), ClosureFail> {
    // R2: post-opening, defender-to-move, nonterminal, not own-win-now.
    if matches!(gate.phase(), TurnPhase::Opening) {
        return Err(ClosureFail::Opening);
    }
    if gate.current_player() != claimant.other() {
        return Err(ClosureFail::NotDefenderToMove);
    }
    if gate.is_terminal() {
        return Err(ClosureFail::Terminal);
    }
    // b in {1,2}.
    if build.b == 0 || build.b > 2 {
        return Err(ClosureFail::BudgetOutOfRange);
    }
    // Recompute the family from the emitted threats and re-derive == b.
    let mut family: Vec<Vec<HexCoord>> = Vec::new();
    for &w in &build.threats {
        let empties: Vec<HexCoord> = w
            .cells()
            .into_iter()
            .filter(|c| gate.board().get(*c).is_none())
            .collect();
        // Each emitted threat must be a real attacker-alive >=4 window.
        let claimant_ct = w
            .cells()
            .iter()
            .filter(|c| gate.board().get(**c) == Some(claimant))
            .count();
        let defender_ct = w
            .cells()
            .iter()
            .filter(|c| gate.board().get(**c) == Some(claimant.other()))
            .count();
        if claimant_ct < 4 || defender_ct != 0 || empties.is_empty() {
            return Err(ClosureFail::NoThreats);
        }
        family.push(empties);
    }
    if transversal_number(&family, 2) != build.b {
        return Err(ClosureFail::TransversalNotB);
    }
    // Recompute K independently and require the emitted kernel to match exactly.
    let mut legal = Vec::new();
    gate.write_legal_moves(&mut legal);
    let mut recomputed: Vec<HexCoord> = legal
        .iter()
        .copied()
        .filter(|&d| {
            let residual: Vec<Vec<HexCoord>> = family
                .iter()
                .filter(|set| !set.contains(&d))
                .cloned()
                .collect();
            transversal_number(&residual, 2) <= build.b.saturating_sub(1)
        })
        .collect();
    recomputed.sort_by_key(|c| coord_key(*c));
    let mut emitted = build.kernel.clone();
    emitted.sort_by_key(|c| coord_key(*c));
    if recomputed != emitted {
        return Err(ClosureFail::KernelEmpty);
    }
    // phi: every edge's real_reply is in K, representative is in R, phi(s)=s for
    // representatives, and the edge_class is consistent with the geometry.
    let kernel_set: HashSet<(i16, i16)> = build.kernel.iter().map(|c| coord_key(*c)).collect();
    let rep_set: HashSet<(i16, i16)> = build.representatives.iter().map(|c| coord_key(*c)).collect();
    // Representatives must be a subset of K.
    if !build.representatives.iter().all(|s| kernel_set.contains(&coord_key(*s))) {
        return Err(ClosureFail::KernelEmpty);
    }
    // The edge domain must equal K exactly (one edge per real reply).
    let edge_domain: HashSet<(i16, i16)> =
        build.edges.iter().map(|e| coord_key(e.real_reply)).collect();
    if edge_domain != kernel_set || edge_domain.len() != build.edges.len() {
        return Err(ClosureFail::KernelEmpty);
    }
    for edge in &build.edges {
        if !rep_set.contains(&coord_key(edge.representative)) {
            return Err(ClosureFail::KernelEmpty);
        }
        // Recompute the edge class from geometry and require agreement.
        let recomputed_class = if edge.real_reply == edge.representative {
            FhwEdgeClassV1::Exact
        } else {
            let ghost = Ghost::new(gate, edge.representative).ok_or(ClosureFail::KernelEmpty)?;
            if frontier_covered(edge.real_reply, edge.representative, &ghost) {
                FhwEdgeClassV1::FrontierCovered
            } else {
                FhwEdgeClassV1::NonFrontierCovered
            }
        };
        if recomputed_class != edge.edge_class {
            return Err(ClosureFail::KernelEmpty);
        }
    }
    // phi(s) = s: every representative must appear as an Exact self-edge.
    for &s in &build.representatives {
        let self_edge = build
            .edges
            .iter()
            .find(|e| e.real_reply == s)
            .ok_or(ClosureFail::KernelEmpty)?;
        if self_edge.representative != s || self_edge.edge_class != FhwEdgeClassV1::Exact {
            return Err(ClosureFail::KernelEmpty);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Corpus shadow firing measurement
// ---------------------------------------------------------------------------

/// A captured real closed gate from a production-shaped defender node, for
/// serialization as a positive fixture.
#[derive(Clone, Debug)]
pub struct GateExample {
    /// Board occupancy `(coord, owner)` at the gate position `P_Q`.
    pub occupancy: Vec<(HexCoord, Player)>,
    /// The gate side-to-move (defender).
    pub side_to_move: Player,
    /// The claimant (attacker) the gate proves a win for.
    pub claimant: Player,
    pub placements_made: u32,
    pub build: GateBuild,
}

/// Aggregated firing measurement over a set of positions.
#[derive(Clone, Debug, Default)]
pub struct FiringStats {
    /// Up to `MAX_EXAMPLES` captured real closed gates for fixture emission.
    pub examples: Vec<GateExample>,
    /// Defender-to-move nodes visited by the bounded forcing walk.
    pub defender_nodes_seen: u64,
    /// Nodes that were implicit-dispatch eligible (own threats needing exactly
    /// `b` hits, no own win, post-opening) — the gate-eligible sites.
    pub eligible_nodes: u64,
    /// Gates that closed all structural self-checks.
    pub gates_closed: u64,
    /// Of the closed gates, those with R subsetneq K (genuinely reductive).
    pub reductive_gates: u64,
    /// Per-reason failure histogram over eligible nodes that did not close.
    pub failures: Vec<(ClosureFail, u64)>,
    /// Sum of |K|, |R|, |Legal| over closed gates (for average reduction).
    pub sum_kernel: u64,
    pub sum_representatives: u64,
    pub sum_legal: u64,
    /// Best observed reductions (smallest ratios).
    pub best_kernel_ratio: f64,
    pub best_representative_ratio: f64,
}

/// Maximum captured example gates per measurement (fixture material).
pub const MAX_EXAMPLES: usize = 12;

impl FiringStats {
    fn record_fail(&mut self, reason: ClosureFail) {
        if let Some(entry) = self.failures.iter_mut().find(|(r, _)| *r == reason) {
            entry.1 += 1;
        } else {
            self.failures.push((reason, 1));
        }
    }
}

/// Bounded forcing-tree walk from `root`, invoking the structural closure at
/// every defender-to-move node. Attacker nodes follow the tactical (forcing)
/// set; defender nodes recurse through their hitting-set replies. `node_cap`
/// bounds total nodes visited; `depth_cap` bounds ply depth. This reaches the
/// same forcing defender nodes the production selector sees, without importing
/// or perturbing `tss_solver`.
pub fn measure_position(
    root: &RustHexoState,
    claimant: Player,
    semantic_horizon: u32,
    node_cap: u64,
    depth_cap: u32,
    stats: &mut FiringStats,
) {
    let mut visited = 0u64;
    walk(
        root,
        claimant,
        semantic_horizon,
        0,
        depth_cap,
        node_cap,
        &mut visited,
        stats,
    );
}

#[allow(clippy::too_many_arguments)]
fn walk(
    state: &RustHexoState,
    claimant: Player,
    semantic_horizon: u32,
    depth: u32,
    depth_cap: u32,
    node_cap: u64,
    visited: &mut u64,
    stats: &mut FiringStats,
) {
    if depth > depth_cap || *visited >= node_cap || state.is_terminal() {
        return;
    }
    *visited += 1;

    if state.current_player() == claimant {
        // Attacker to move: follow the forcing (tactical) set to build threats.
        let mut moves = threats_shared::tactical_cells(state);
        if moves.is_empty() {
            // No forcing move: sample a few legal moves to keep advancing turns.
            let mut legal = Vec::new();
            state.write_legal_moves(&mut legal);
            legal.sort_by_key(|c| coord_key(*c));
            moves = legal.into_iter().take(4).collect();
        }
        moves.sort_by_key(|c| coord_key(*c));
        moves.dedup();
        for mv in moves {
            if *visited >= node_cap {
                return;
            }
            let mut next = state.clone();
            if let Ok(result) = apply_placement(&mut next, Placement { coord: mv }) {
                if result.outcome.is_none() {
                    walk(
                        &next,
                        claimant,
                        semantic_horizon,
                        depth + 1,
                        depth_cap,
                        node_cap,
                        visited,
                        stats,
                    );
                }
            }
        }
        return;
    }

    // Defender to move.
    stats.defender_nodes_seen += 1;
    let analysis = threats_shared::analyze(state);
    let eligible = !matches!(state.phase(), TurnPhase::Opening)
        && analysis.opp_threat_count > 0
        && !analysis.own_win_now
        && analysis.min_hitting_set == Some(analysis.b);
    if eligible {
        stats.eligible_nodes += 1;
        match try_build_gate(state, claimant, semantic_horizon) {
            Ok(build) => {
                stats.gates_closed += 1;
                stats.sum_kernel += build.kernel.len() as u64;
                stats.sum_representatives += build.representatives.len() as u64;
                stats.sum_legal += build.legal_count as u64;
                if build.is_reductive() {
                    stats.reductive_gates += 1;
                }
                // Capture a diverse set of real closed gates: prefer reductive
                // gates with FrontierCovered edges (the interesting class).
                let has_fc = build
                    .edges
                    .iter()
                    .any(|e| e.edge_class == FhwEdgeClassV1::FrontierCovered);
                let want = build.is_reductive() && has_fc;
                if stats.examples.len() < MAX_EXAMPLES && (want || stats.examples.len() < 4) {
                    let occupancy: Vec<(HexCoord, Player)> = state
                        .board()
                        .occupied_cells()
                        .iter()
                        .filter_map(|&c| state.board().get(c).map(|p| (c, p)))
                        .collect();
                    stats.examples.push(GateExample {
                        occupancy,
                        side_to_move: state.current_player(),
                        claimant,
                        placements_made: state.placements_made(),
                        build: build.clone(),
                    });
                }
                let kr = build.kernel_ratio();
                let rr = build.representative_ratio();
                if stats.gates_closed == 1 {
                    stats.best_kernel_ratio = kr;
                    stats.best_representative_ratio = rr;
                } else {
                    stats.best_kernel_ratio = stats.best_kernel_ratio.min(kr);
                    stats.best_representative_ratio = stats.best_representative_ratio.min(rr);
                }
            }
            Err(reason) => stats.record_fail(reason),
        }
    }

    // Recurse through defender hitting-set replies (the moves that answer the
    // attacker threats) to reach deeper forcing defender nodes.
    let mut replies = threats_shared::tactical_cells(state);
    replies.sort_by_key(|c| coord_key(*c));
    replies.dedup();
    for mv in replies.into_iter().take(6) {
        if *visited >= node_cap {
            return;
        }
        let mut next = state.clone();
        if let Ok(result) = apply_placement(&mut next, Placement { coord: mv }) {
            if result.outcome.is_none() {
                walk(
                    &next,
                    claimant,
                    semantic_horizon,
                    depth + 1,
                    depth_cap,
                    node_cap,
                    visited,
                    stats,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;
