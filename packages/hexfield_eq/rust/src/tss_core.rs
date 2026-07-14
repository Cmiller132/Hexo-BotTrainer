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

#[derive(Clone, Copy, Debug, Default)]
pub struct ZoneSearchCaps {
    pub enabled: bool,
    pub stale_area_filter: bool,
    pub count2_threshold: bool,
    pub pair_commutation: bool,
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
#[derive(Clone, Copy, Debug, Default)]
pub struct SolveStats {
    pub nodes: u64,
    pub tt_hits: u64,
    pub peak_tt_bytes: u64,
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
/// net-eval AND bump the fatal `verify_failed` telemetry counter). Unused
/// until Stage 4 wires the consumption ladder.
pub fn hard_value_from_verified<V, C>(
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
            hard_value_from_verified(&Accept, &state, &win).map(HardValue::value),
            Some(1.0)
        );
        assert!(hard_value_from_verified(&Reject, &state, &win).is_none());
        assert!(hard_value_from_verified(&Accept, &state, &unknown).is_none());
        assert!(hard_value_from_verified(&Accept, &state, &certless_loss).is_none());
    }
}
