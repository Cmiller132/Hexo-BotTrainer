//! Empirical sharpness hunt for zone-radius open problems R1b and R2.
//!
//! Normative source: docs/PROOF_TSS_DEFENDER_ZONES.md §5 (D10-D16, L9', L12),
//! §12 items 5-6, §12a; and docs/_TIGHTNESS_FRONTIER_REPORT.md §2.2 (R1b) /
//! §3.1 (R2). This module is DATA-producing (fixture-or-shrink), not a proof.
//!
//! What the shipped verifier actually computes (tss_verify.rs ~L1031-1071,
//! tss_solver.rs `zone_certificate_extras` ~L4981):
//!   * completion guard  : active_player == defender && cnt_D(W) + d >= 6
//!                         => search every empty cell of W  (touched windows;
//!                         virgin windows have active_player == None so never
//!                         fire here, and d >= 6 short-circuits to the full
//!                         legal set — so `Z_virgin` is ABSORBED, never the
//!                         `8(E^D-6)` radius form).
//!   * seed band (Z5)    : for each ghost-illegal, non-stone protected target
//!                         `y`, search every legal cell within radius `8*d`.
//!
//! The proof's uniform wrapper (D11 Z5', with r_N(y) := B(N)) uses `8(B-1)`.
//! The implementation uses `8*d` (= 8*B): one relay MORE than the proof
//! minimum. R1b asks whether `8(B-1)` is sharp; R2 asks whether the full union
//! `Z_virgin` can shrink below `8(E^D-6)`.
//!
//! Faithfulness: `seed_band_required` below reproduces the production block
//! line-for-line with a parameterized radius multiplier `m` (production is
//! `m == d`). `hunt_seed_band_matches_production` cross-checks it against the
//! real `zone_certificate_extras` on real states; `hunt_legality_matches_engine`
//! cross-checks the self-computed legality against the engine.
//!
//! Run: (CARGO_TARGET_DIR=.target-hunt)
//!   cargo test -p hexfield_eq --release hunt_ -- --ignored --nocapture --test-threads=1

#![cfg(test)]

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use hexo_engine::{
    apply_placement, hex_distance, HexCoord, HexoState as RustHexoState, Placement, Player,
    TurnPhase, WindowStore,
};

use crate::tss_core::{CertVerify, ProofStatus, SolveCaps, SolveGoal, ZoneSearchCaps};
use crate::tss_solver::{ScopedRelayDelta, TssSolver};
use crate::tss_verify::{certificate_horizon_preflight, CertNode, TssVerifier};

const LEGAL_RADIUS: i16 = 8;

// ===========================================================================
// Ownership-map positions (construction freedom the legal-replay API denies).
// ===========================================================================

type Owned = (HexCoord, Player);

/// Every cell within `radius` hex-steps of `center` (axial diamond).
fn cells_within(center: HexCoord, radius: i16) -> Vec<HexCoord> {
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

/// Engine legality, recomputed from a raw stone set: a cell is legal iff it is
/// empty and within `LEGAL_RADIUS` of some stone (legal.rs::update_for_placement,
/// rules.rs::is_legal_placement). Cross-checked against the real HexoState in
/// `hunt_legality_matches_engine`.
fn legal_cells(stones: &[Owned]) -> BTreeSet<(i16, i16)> {
    let occupied: BTreeSet<(i16, i16)> = stones.iter().map(|(c, _)| (c.q, c.r)).collect();
    let mut legal = BTreeSet::new();
    for (stone, _) in stones {
        for c in cells_within(*stone, LEGAL_RADIUS) {
            let key = (c.q, c.r);
            if !occupied.contains(&key) {
                legal.insert(key);
            }
        }
    }
    legal
}

fn is_legal(legal: &BTreeSet<(i16, i16)>, c: HexCoord) -> bool {
    legal.contains(&(c.q, c.r))
}

// ===========================================================================
// Faithful, radius-parameterized seed band (mirrors zone_certificate_extras).
// ===========================================================================

#[derive(Debug, Clone)]
struct SeedContext {
    /// Protected obligation set (D10 roles + completion-window empties).
    protected: Vec<HexCoord>,
    /// Ghost-illegal, non-stone protected cells: the seed-band targets.
    pending: Vec<HexCoord>,
    legal: BTreeSet<(i16, i16)>,
}

/// Build the protected set exactly as production does: caller-supplied D10
/// obligation cells, plus the defender-completion guard
/// `active_player == defender && cnt_D + d >= 6 => all empties`.
fn seed_context(
    stones: &[Owned],
    claimant: Player,
    d: u32,
    obligations: &[HexCoord],
) -> SeedContext {
    let defender = claimant.other();
    let store = WindowStore::from_placements(stones);
    let mut protected: Vec<HexCoord> = obligations.to_vec();
    for entry in store.entries() {
        if entry.active_player() == Some(defender)
            && u32::from(entry.count(defender)).saturating_add(d) >= 6
        {
            protected.extend(entry.empty_cells());
        }
    }
    protected.sort_by_key(|c| (c.q, c.r));
    protected.dedup();

    let legal = legal_cells(stones);
    let occupied: BTreeSet<(i16, i16)> = stones.iter().map(|(c, _)| (c.q, c.r)).collect();
    let pending: Vec<HexCoord> = protected
        .iter()
        .copied()
        .filter(|c| !is_legal(&legal, *c) && !occupied.contains(&(c.q, c.r)))
        .collect();

    SeedContext {
        protected,
        pending,
        legal,
    }
}

/// The seed-band `required` set at radius multiplier `m` (radius = 8*m).
/// Production is `m == d`. Mirrors zone_certificate_extras' pending+radius
/// block: (protected ∩ legal) ∪ {legal cells within 8*m of a pending target}.
fn seed_band_required(ctx: &SeedContext, m: u32) -> Vec<HexCoord> {
    let radius = i32::try_from(m.saturating_mul(8)).unwrap_or(i32::MAX);
    let legal_coords: Vec<HexCoord> = ctx.legal.iter().map(|&(q, r)| HexCoord { q, r }).collect();
    let mut required: Vec<HexCoord> = ctx
        .protected
        .iter()
        .copied()
        .filter(|c| is_legal(&ctx.legal, *c))
        .collect();
    if !ctx.pending.is_empty() {
        required.extend(legal_coords.iter().copied().filter(|cell| {
            ctx.pending
                .iter()
                .any(|t| i32::from(hex_distance(*cell, *t)) <= radius)
        }));
    }
    required.sort_by_key(|c| (c.q, c.r));
    required.dedup();
    required
}

// ===========================================================================
// L9' reachability: the soundness mechanism behind the seed band.
// ===========================================================================
//
// A dismissed seed threatens soundness iff, starting from it, the DEFENDER can
// build a legal relay chain that occupies a ghost-illegal protected cell `y`
// (first protected occupation, L9'). Each defender stone extends legality by 8,
// so the chain to occupy `y` has `min_placements(y)` cells and the first (a
// currently-legal cell) sits at distance <= 8*(min_placements-1) from `y`.
//
// `reach_seed_distance(y, B)` = the largest hex distance from a currently-legal
// "chain-start" seed to `y`, over defender chains of length <= B that occupy
// `y`. The seed band's radius must be >= this value to remain sound. The proof
// bounds it by 8(B-1).

/// Max distance from a currently-legal chain-start seed to `y` over defender
/// chains of length <= B that occupy `y`, via a single BFS rooted at `y` over
/// the undirected "within-8" graph on empty cells. `dist_to_y[c]` = fewest
/// relay hops from `c` to `y`; a legal cell `s` can be the FIRST stone of a
/// length-<=B occupation of `y` iff `dist_to_y[s] + 1 <= B`. Returns
/// (farthest such seed distance, min placements to occupy `y`, farthest seed),
/// or None if `y` is not occupiable in <= B defender placements.
fn reach_seed_distance(
    stones: &[Owned],
    legal: &BTreeSet<(i16, i16)>,
    y: HexCoord,
    budget_b: u32,
) -> Option<(i16, u32, HexCoord)> {
    let region_radius = (8 * budget_b as i16) + LEGAL_RADIUS;
    let occupied: BTreeSet<(i16, i16)> = stones.iter().map(|(c, _)| (c.q, c.r)).collect();
    let region: Vec<HexCoord> = cells_within(y, region_radius)
        .into_iter()
        .filter(|c| !occupied.contains(&(c.q, c.r)))
        .collect();
    let region_set: HashSet<(i16, i16)> = region.iter().map(|c| (c.q, c.r)).collect();

    // BFS rooted at y over empty cells (undirected within-8 adjacency).
    let mut dist_to_y: HashMap<(i16, i16), u32> = HashMap::new();
    dist_to_y.insert((y.q, y.r), 0);
    let mut queue = VecDeque::new();
    queue.push_back(y);
    while let Some(cur) = queue.pop_front() {
        let cur_d = dist_to_y[&(cur.q, cur.r)];
        // Only cells within B-1 hops of y can be a chain-start seed, and only a
        // legal cell within B-1 hops decides occupiability. Discover hops up to
        // B-1 (expand up to B-2) and stop — everything needed is then present.
        if cur_d + 1 >= budget_b {
            continue;
        }
        for cand in cells_within(cur, LEGAL_RADIUS) {
            let key = (cand.q, cand.r);
            if !region_set.contains(&key) {
                continue;
            }
            if !dist_to_y.contains_key(&key) {
                dist_to_y.insert(key, cur_d + 1);
                queue.push_back(cand);
            }
        }
    }

    // min placements to occupy y = 1 + min hops from a legal cell to y.
    let min_placements = dist_to_y
        .iter()
        .filter(|((q, r), _)| is_legal(legal, HexCoord { q: *q, r: *r }))
        .map(|(_, &h)| h + 1)
        .min()?;
    if min_placements > budget_b {
        return None;
    }
    // Farthest legal seed whose chain to y is <= B-1 hops (=> total <= B stones).
    let mut best: Option<(i16, HexCoord)> = None;
    for c in &region {
        if !is_legal(legal, *c) {
            continue;
        }
        if let Some(&hops) = dist_to_y.get(&(c.q, c.r)) {
            if hops + 1 <= budget_b {
                let dd = hex_distance(*c, y);
                if best.map_or(true, |(bd, _)| dd > bd) {
                    best = Some((dd, *c));
                }
            }
        }
    }
    best.map(|(dd, s)| (dd, min_placements, s))
}

// ===========================================================================
// R1b synthetic family: the L9' chain that attains reach = 8(B-1).
// ===========================================================================
//
// docs/_TIGHTNESS_FRONTIER_REPORT.md §2.1 (rank-r chain), realized as a board.
// Anchor at origin makes seed s=(8,0) legal; the protected target y=(8B,0) is
// initially illegal; the defender walks s->(16,0)->...->y in exactly B stones.

/// Attacker anchor + defender chain realizing budget/rank B. Returns
/// (stones, seed, target). No stone lies within 8 of `y` initially, so `y` is
/// a ghost-illegal protected target; the only legal cell inside 8(B-1) of `y`
/// is the chain start `s`.
fn r1b_chain_family(budget_b: u32) -> (Vec<Owned>, HexCoord, HexCoord) {
    let attacker = Player::Player0;
    let anchor = HexCoord { q: 0, r: 0 };
    let seed = HexCoord { q: 8, r: 0 };
    let y = HexCoord {
        q: 8 * budget_b as i16,
        r: 0,
    };
    // Anchor only. A single attacker stone at origin legalizes the disk of
    // radius 8; the nearest point of that disk to `y` is `seed` at 8(B-1).
    let stones = vec![(anchor, attacker)];
    (stones, seed, y)
}

// ===========================================================================
// Tests: cross-checks first, then the datasets.
// ===========================================================================

/// Self-computed legality must equal the engine's on real replayed states.
#[test]
fn hunt_legality_matches_engine() {
    // Deterministic legal games; compare legal sets each ply.
    let seeds: [u64; 6] = [1, 2, 3, 5, 8, 13];
    let mut checked = 0usize;
    for seed in seeds {
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut state = RustHexoState::new();
        let mut owned: Vec<Owned> = Vec::new();
        for _ply in 0..40 {
            // Compare before this placement.
            let mut engine_legal = Vec::new();
            state.write_legal_moves(&mut engine_legal);
            if !owned.is_empty() {
                let mine = legal_cells(&owned);
                let engine: BTreeSet<(i16, i16)> =
                    engine_legal.iter().map(|c| (c.q, c.r)).collect();
                assert_eq!(mine, engine, "legality mismatch at seed {seed}");
                checked += 1;
            }
            // Advance by a random legal move.
            if engine_legal.is_empty() {
                break;
            }
            let mv = engine_legal[(next() as usize) % engine_legal.len()];
            let player = state.current_player();
            match apply_placement(&mut state, Placement { coord: mv }) {
                Ok(res) => {
                    owned.push((mv, player));
                    if res.outcome.is_some() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }
    assert!(checked > 100, "too few legality comparisons: {checked}");
    println!("HUNT legality_cross_check comparisons={checked} status=OK");
}

/// The radius-parameterized seed band at `m == d` reproduces the production
/// `zone_certificate_extras` on real states with a hand-built arena.
#[test]
fn hunt_seed_band_matches_production() {
    use crate::tss_solver::zone_certificate_extras;
    use crate::tss_verify::{CertEdge, CertNode};
    use hexo_engine::{Axis, WindowKey};

    // A quiet real position with a modest legal set.
    let mut state = RustHexoState::new();
    for &(q, r) in &[(0, 0), (0, 8), (2, 7), (1, 0), (4, 6)] {
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord { q, r },
            },
        )
        .unwrap();
    }
    let claimant = state.current_player();

    // Ownership map matching the real state, for our probe.
    let owned: Vec<Owned> = state
        .board()
        .occupied_cells()
        .iter()
        .map(|&c| (c, state.board().get(c).expect("owner")))
        .collect();

    // Hand-built arena: Choice(mv=obligation) -> Win leaf. `arena_core` collects
    // the Choice mv and the Win witness cells as the protected core.
    let obligation = HexCoord { q: 20, r: -10 }; // far, currently illegal
    let win_leaf = CertNode::Win {
        witness: WindowKey {
            start: HexCoord { q: 0, r: 0 },
            axis: Axis::Q,
        },
        count: 5,
        budget: 2,
        resolution_ply: state.placements_made() + 2,
    };
    let choice = CertNode::Choice {
        mv: obligation,
        child: 0,
    };
    let arena = vec![win_leaf, choice];
    let edges = vec![CertEdge {
        mv: obligation,
        child: 1,
    }];

    for d in 1..=5u32 {
        let production = zone_certificate_extras(&state, claimant, d, &edges, &arena)
            .expect("production extras");
        // Reproduce production's obligation set: arena_core = {obligation} ∪ Win
        // witness cells. Our probe takes obligations directly.
        let mut obligations = vec![obligation];
        obligations.extend(
            WindowKey {
                start: HexCoord { q: 0, r: 0 },
                axis: Axis::Q,
            }
            .cells(),
        );
        let ctx = seed_context(&owned, claimant, d, &obligations);
        let mut mine = seed_band_required(&ctx, d);
        let mut prod = production.clone();
        mine.sort_by_key(|c| (c.q, c.r));
        prod.sort_by_key(|c| (c.q, c.r));
        assert_eq!(mine, prod, "seed band mismatch at d={d}");
    }
    println!("HUNT seed_band_production_cross_check status=OK d=1..5");
}

// ===========================================================================
// R1b DATASET 1 — synthetic sharpness fixture: reach attains 8(B-1).
// ===========================================================================

/// For every budget B, the L9' chain family attains reach = 8(B-1): the
/// binding seed sits at exactly 8(B-1) from the ghost-illegal protected target
/// and is occupiable as the first stone of a B-stone chain. The seed band keeps
/// it at the proof wrapper radius 8(B-1), keeps it at the implementation radius
/// 8B (with a full relay of slack), and SHEDS it at 8(B-2). Shedding it leaves
/// the defender's first protected occupation of `y` unguarded => the smaller
/// uniform radius is unsound (candidate PIN of R1b at the coverage level).
#[test]
#[ignore = "hunt dataset; run explicitly with --nocapture"]
fn hunt_r1b_chain_sharpness() {
    let attacker = Player::Player0;
    println!("HUNT R1B_FIXTURE begin  (radius unit = 8 cells)");
    println!(
        "  B | reach | min_pl | 8(B-1) | seed@8B | seed@8(B-1) | seed@8(B-2) | |req|@8B/8(B-1)/8(B-2)"
    );
    for b in 2..=5u32 {
        let (stones, seed, y) = r1b_chain_family(b);
        let legal = legal_cells(&stones);
        // Reachability: y is occupiable in exactly B defender placements, and
        // the farthest legal chain-start is `seed` at 8(B-1).
        let reach = reach_seed_distance(&stones, &legal, y, b).expect("y reachable within B");
        let (reach_dist, min_pl, reach_seed) = reach;
        assert_eq!(
            reach_dist as i32,
            8 * (b as i32 - 1),
            "reach must equal 8(B-1)"
        );
        assert_eq!(min_pl, b, "min placements to occupy y must equal B");
        assert_eq!(reach_seed, seed, "binding seed must be the chain start");
        // y not occupiable within B-1 placements.
        assert!(
            reach_seed_distance(&stones, &legal, y, b - 1).is_none(),
            "y must NOT be occupiable within B-1 placements"
        );

        // Seed band at d = B, targeting y as the sole ghost-illegal obligation.
        let ctx = seed_context(&stones, attacker, b, &[y]);
        assert!(
            ctx.pending.iter().any(|c| *c == y),
            "y must be a ghost-illegal (pending) seed target"
        );
        let seed_in = |m: u32| seed_band_required(&ctx, m).iter().any(|c| *c == seed);
        let n = |m: u32| seed_band_required(&ctx, m).len();

        let in_impl = seed_in(b); // 8B
        let in_wrap = seed_in(b - 1); // 8(B-1)  proof wrapper
        let in_shed = seed_in(b.saturating_sub(2)); // 8(B-2)
        println!(
            "  {b} |  {:>3}  |   {b}    |  {:>3}   |   {}   |     {}      |     {}      | {}/{}/{}",
            reach_dist,
            8 * (b - 1),
            in_impl,
            in_wrap,
            in_shed,
            n(b),
            n(b - 1),
            n(b.saturating_sub(2)),
        );
        assert!(in_impl, "implementation radius 8B must keep the seed");
        assert!(in_wrap, "proof wrapper radius 8(B-1) must keep the seed");
        assert!(
            !in_shed,
            "radius 8(B-2) must SHED the seed (else 8(B-1) not sharp at B={b})"
        );
    }
    println!(
        "HUNT R1B_FIXTURE verdict=SHARP-AT-8(B-1)  impl_uses=8B(one relay of slack)  status=OK"
    );
}

#[test]
#[ignore = "absolute-pin construction; run explicitly with --nocapture"]
fn hunt_r1b_absolute_pin() {
    fn replay(coords: &[(i16, i16)]) -> RustHexoState {
        let mut state = RustHexoState::new();
        for &(q, r) in coords {
            apply_placement(&mut state, Placement { coord: HexCoord::new(q, r) }).unwrap();
        }
        state
    }

    fn solve_at(
        state: &RustHexoState,
        goal: SolveGoal,
        semantic_horizon: u32,
        delta: u32,
    ) -> crate::tss_core::DeepResult<crate::tss_verify::TssCertificate> {
        let _relay = ScopedRelayDelta::new(delta);
        let mut solver = TssSolver::default();
        solver.set_zone_options(ZoneSearchCaps {
            enabled: true,
            ..ZoneSearchCaps::default()
        });
        solver.solve_goal(
            state,
            &SolveCaps {
                node_cap: 100_000,
                tt_bytes_cap: 64 << 20,
                semantic_horizon,
            },
            goal,
        )
    }

    // Attempt 1: the repository's curated deep forcing line.  Entering at its
    // documented defender node does produce a real deep certificate, but every
    // Universal is forced dispatch, so prove_universal never attaches a zone
    // and the seed-band radius is not consulted.
    let deep_coords = [
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
    ];
    let mut deep = replay(&deep_coords);
    apply_placement(
        &mut deep,
        Placement {
            coord: HexCoord::new(2, -3),
        },
    )
    .unwrap();
    assert_eq!(deep.phase(), TurnPhase::FirstStone);
    let analysis = crate::threats_shared::analyze(&deep);
    assert_eq!(analysis.min_hitting_set, Some(analysis.b));
    assert!(analysis.opp_threat_count > 0 && !analysis.own_win_now);
    let deep_t = deep.placements_made() + 8;
    let deep_results = (0..=2)
        .map(|delta| solve_at(&deep, SolveGoal::Loss, deep_t, delta))
        .collect::<Vec<_>>();
    for result in &deep_results {
        assert_eq!(result.status, ProofStatus::Loss);
        let cert = result.cert.as_ref().expect("deep forcing certificate");
        assert!(cert.nodes.iter().any(|node| matches!(node, CertNode::Universal { .. })));
        assert!(cert.nodes.iter().all(|node| !matches!(
            node,
            CertNode::Universal {
                implicit_dispatch: false,
                ..
            }
        )));
        assert_eq!(certificate_horizon_preflight(cert), Some((deep_t, false)));
    }
    assert_eq!(deep_results[0].cert, deep_results[1].cert);
    assert_eq!(deep_results[1].cert, deep_results[2].cert);
    let deep_cert = deep_results[2].cert.as_ref().unwrap();
    for delta in 0..=2 {
        let _relay = ScopedRelayDelta::new(delta);
        assert!(TssVerifier.verify_with_dispatch_oracle(
            &deep,
            deep_cert,
            ProofStatus::Loss
        ));
    }
    println!(
        "HUNT R1B_ABSOLUTE attempt=1 status=BLOCKED blocker=prove_universal:implicit_dispatch root_ply={} derived_t={} cert_nodes={} zones=0 deltas=identical",
        deep.placements_made(),
        deep_t,
        deep_cert.nodes.len()
    );

    // Attempt 2: the compact one-turn win family.  The finder emits a typed
    // leaf with no Universal.  Independently, every empty witness cell is
    // already legal (within a length-six window of four/five claimant stones),
    // so verify_zone_node's `pending` filter would be empty even if stapled.
    let one_turn = replay(&[
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
    ]);
    let one_t = one_turn.placements_made() + 2;
    let one_results = (0..=2)
        .map(|delta| solve_at(&one_turn, SolveGoal::Win, one_t, delta))
        .collect::<Vec<_>>();
    for result in &one_results {
        assert_eq!(result.status, ProofStatus::Win);
        assert!(result.cert.as_ref().is_some_and(|cert| {
            cert.nodes
                .iter()
                .all(|node| !matches!(node, CertNode::Universal { .. }))
        }));
    }
    assert_eq!(one_results[0].cert, one_results[1].cert);
    assert_eq!(one_results[1].cert, one_results[2].cert);
    let one_cert = one_results[2].cert.as_ref().unwrap();
    let witness = one_cert
        .nodes
        .iter()
        .find_map(|node| match node {
            CertNode::Win { witness, .. } => Some(*witness),
            _ => None,
        })
        .expect("one-turn Win witness");
    let mut one_legal = Vec::new();
    one_turn.write_legal_moves(&mut one_legal);
    let pending = witness
        .cells()
        .into_iter()
        .filter(|cell| {
            one_turn.board().get(*cell).is_none() && !one_legal.contains(cell)
        })
        .collect::<Vec<_>>();
    assert!(pending.is_empty());
    println!(
        "HUNT R1B_ABSOLUTE attempt=2 status=BLOCKED blocker=verify_zone_node:pending-empty witness={witness:?} resolution={} deltas=identical",
        one_t
    );

    // Attempt 3: literal B=4 sharp chain on a real legal root.  The geometry is
    // exact: s=(8,0) is the only root-legal cell within 24=8(B-1) of y=(32,0),
    // while delta=2 would retain only radius 16.  The four defender placements
    // are legal in the real engine (two harmless attacker fillers separate the
    // defender turns).  But the finder cannot create an arena_core obligation:
    // its narrow attacker generator admits only extensions of an existing
    // count-three window, and this frontier has one claimant stone.
    const B: u32 = 4;
    let chain = replay(&[(0, 0)]);
    let seed = HexCoord::new(8, 0);
    let target = HexCoord::new(32, 0);
    let relay = [seed, HexCoord::new(16, 0), HexCoord::new(24, 0), target];
    let mut chain_legal = Vec::new();
    chain.write_legal_moves(&mut chain_legal);
    assert!(chain_legal.contains(&seed));
    assert!(!chain_legal.contains(&target));
    let sharp_seeds = chain_legal
        .iter()
        .copied()
        .filter(|cell| hex_distance(*cell, target) <= 8 * (B as i16 - 1))
        .collect::<Vec<_>>();
    assert_eq!(sharp_seeds, vec![seed]);
    assert!(hex_distance(seed, target) > 8 * (B as i16 - 2));

    let mut real_line = chain.clone();
    for coord in [
        relay[0],
        relay[1],
        HexCoord::new(0, -1),
        HexCoord::new(0, -2),
        relay[2],
        relay[3],
    ] {
        apply_placement(&mut real_line, Placement { coord }).unwrap();
    }
    assert_eq!(real_line.board().get(target), Some(Player::Player1));

    let chain_t = chain.placements_made() + 8;
    let chain_results = (0..=2)
        .map(|delta| solve_at(&chain, SolveGoal::Loss, chain_t, delta))
        .collect::<Vec<_>>();
    assert!(chain_results.iter().all(|result| {
        result.status == ProofStatus::Unknown && result.cert.is_none()
    }));
    println!(
        "HUNT R1B_ABSOLUTE attempt=3 status=BLOCKED blocker=threat_creating_moves:minimum_strength=3 B={B} seed={seed:?} target={target:?} relay={relay:?} shipped_radius={} sharp_radius={} weakened_radius={} statuses=UNKNOWN/UNKNOWN/UNKNOWN",
        8 * B,
        8 * (B - 1),
        8 * (B - 2)
    );
    println!(
        "HUNT R1B_ABSOLUTE outcome=BLOCKED attempts=3 exact_check=threat_creating_moves(line3914)->prove_choice(line3350); count3-scaffold makes the witness legal, while forcing it makes the defender node implicit_dispatch"
    );

    // Keep the public verifier trait import exercised as a second independent
    // check that the one-turn certificate is valid under the production path.
    for delta in 0..=2 {
        let _relay = ScopedRelayDelta::new(delta);
        assert!(TssVerifier.verify(&one_turn, one_cert, ProofStatus::Win));
    }
}

// ===========================================================================
// R1b DATASET 2 — reach envelope over diverse realistic positions.
// ===========================================================================

/// Deterministic xorshift stream (no rand dependency).
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

/// A random legal position: `plies` placements near the origin. Returns the
/// ownership map and the live state.
fn random_owned(seed: u64, plies: usize) -> Vec<Owned> {
    let mut rng = XorShift(seed | 1);
    let mut state = RustHexoState::new();
    let mut owned: Vec<Owned> = Vec::new();
    for _ in 0..plies {
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        if legal.is_empty() {
            break;
        }
        // Uniform random legal move. Legality keeps play compact (every move is
        // within 8 of an existing stone), so windows and dense frontiers form.
        let mv = legal[(rng.next() as usize) % legal.len()];
        let player = state.current_player();
        match apply_placement(&mut state, Placement { coord: mv }) {
            Ok(res) => {
                owned.push((mv, player));
                if res.outcome.is_some() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    owned
}

/// Illegal, non-stone cells within radius `8*budget` of some stone: the cells a
/// <=budget defender chain could conceivably occupy (candidate ghost-illegal
/// obligation targets).
fn frontier_illegal_targets(stones: &[Owned], budget: u32) -> Vec<HexCoord> {
    let occupied: BTreeSet<(i16, i16)> = stones.iter().map(|(c, _)| (c.q, c.r)).collect();
    let legal = legal_cells(stones);
    let mut set: BTreeSet<(i16, i16)> = BTreeSet::new();
    let reach = 8 * budget as i16;
    for (s, _) in stones {
        for c in cells_within(*s, reach) {
            let key = (c.q, c.r);
            if !occupied.contains(&key) && !legal.contains(&key) {
                set.insert(key);
            }
        }
    }
    set.into_iter().map(|(q, r)| HexCoord { q, r }).collect()
}

/// Over many diverse real positions, measure the minimal SOUND uniform seed
/// radius per (position, budget): max over ghost-illegal frontier targets `y`
/// of reach(y,B). Report its distribution against the proof wrapper 8(B-1) and
/// the shipped implementation radius 8B. Establishes (a) reach never exceeds
/// 8(B-1) [relay-structure bound, so 8B always carries >=1 removable relay],
/// and (b) how often 8(B-1) is even approached by realistic geometry.
#[test]
#[ignore = "hunt dataset; run explicitly with --nocapture"]
fn hunt_r1b_reach_envelope() {
    let mut positions: Vec<Vec<Owned>> = Vec::new();
    // Diverse random legal games of varied length.
    for seed in 1..=90u64 {
        for plies in [12usize, 24, 40, 60] {
            let owned = random_owned(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), plies);
            if owned.len() >= 6 {
                positions.push(owned);
            }
        }
    }
    // Perturbations of the linear chain family (adversarial geometry).
    for b in 2..=5u32 {
        let (stones, _, _) = r1b_chain_family(b);
        positions.push(stones);
    }
    // Precompute legality once per position.
    let legal_per_pos: Vec<BTreeSet<(i16, i16)>> =
        positions.iter().map(|s| legal_cells(s)).collect();
    println!(
        "HUNT R1B_ENVELOPE positions={} (random legal games + chain family)",
        positions.len()
    );
    // B ranges over realistic per-turn defender budgets (a Hexo turn is 2 stones).
    println!("  B | positions_with_target | max_reach | at_8(B-1) | frac_ge_half | ever_gt_8(B-1) | ever_ge_8B");
    for b in 1..=2u32 {
        let bound = 8 * (b as i32 - 1); // 8(B-1)
        let impl_r = 8 * b as i32; // 8B
        let mut with_target = 0usize;
        let mut max_reach = 0i32;
        let mut attain_bound = 0usize;
        let mut ge_half = 0usize;
        let mut ever_gt_bound = false;
        let mut ever_ge_impl = false;
        for (stones, legal) in positions.iter().zip(&legal_per_pos) {
            let targets = frontier_illegal_targets(stones, b);
            // Deterministic subsample: cap targets per position to bound runtime
            // while keeping thousands of measurements across the corpus.
            const TARGET_CAP: usize = 80;
            let stride = (targets.len() / TARGET_CAP).max(1);
            let mut pos_max = -1i32;
            for y in targets.iter().step_by(stride).copied() {
                if let Some((dist, _min_pl, _seed)) = reach_seed_distance(stones, legal, y, b) {
                    pos_max = pos_max.max(dist as i32);
                }
            }
            if pos_max < 0 {
                continue;
            }
            with_target += 1;
            max_reach = max_reach.max(pos_max);
            if pos_max >= bound && bound > 0 {
                attain_bound += 1;
            }
            if bound > 0 && pos_max * 2 >= bound {
                ge_half += 1;
            }
            if pos_max > bound {
                ever_gt_bound = true;
            }
            if pos_max >= impl_r {
                ever_ge_impl = true;
            }
        }
        println!(
            "  {b} |         {:>5}          |    {:>3}    |   {:>4}    |     {:>4}     |     {}      |    {}",
            with_target, max_reach, attain_bound, ge_half, ever_gt_bound, ever_ge_impl
        );
        assert!(
            !ever_gt_bound,
            "SOUNDNESS-SURPRISE: reach exceeded 8(B-1) at B={b} (would break the proof bound)"
        );
        assert!(
            !ever_ge_impl,
            "reach reached the implementation radius 8B at B={b} (no removable slack)"
        );
    }
    println!("HUNT R1B_ENVELOPE verdict=reach<=8(B-1)<8B always; 8B carries a removable relay");
}

// ===========================================================================
// R2 — virgin-window completion radius 8(E^D-6).
// ===========================================================================
//
// L12 (split completion safety): before the attacker enters an all-empty
// window W, the zone must prevent the real defender from COMPLETING W. The
// defender fills W's 6 cells; if W is far, it first relays to W. Filling W in
// E^D placements costs (E^D-6) relays + 6 fills, so a dismissed seed at
// distance up to 8(E^D-6) from W must be searched. `Z_virgin` uses that radius.
//
// FULL-UNION open question (docs §3.1): `Z_virgin` is a union over ALL all-empty
// windows; a seed lies in 18 incident windows. If any incident window has
// exposure >= 6, it selects the seed at distance 0, potentially letting the
// union shrink below 8(E^D-6). We probe whether the construction's relay seeds
// are independently covered by other incident all-empty windows.

/// All-empty (virgin) window along the Q axis: {(i,0): 0..=5}.
fn q_window(start: HexCoord) -> [HexCoord; 6] {
    let mut w = [HexCoord::default(); 6];
    for (i, cell) in w.iter_mut().enumerate() {
        *cell = HexCoord {
            q: start.q + i as i16,
            r: start.r,
        };
    }
    w
}

/// Distance from a cell to a window (min over its cells).
fn dist_to_window(c: HexCoord, w: &[HexCoord]) -> i16 {
    w.iter()
        .map(|&cell| hex_distance(c, cell))
        .min()
        .unwrap_or(i16::MAX)
}

/// Analog of `reach_seed_distance` for COMPLETING an all-empty window `w`
/// (occupy all 6 cells). Returns (farthest legal chain-start distance to W,
/// min defender placements to complete W, farthest seed) for chains of total
/// length <= `budget_e`; None if W is not completable within the budget.
fn complete_window_reach(
    stones: &[Owned],
    w: &[HexCoord],
    budget_e: u32,
) -> Option<(i16, u32, HexCoord)> {
    let legal = legal_cells(stones);
    let occupied: BTreeSet<(i16, i16)> = stones.iter().map(|(c, _)| (c.q, c.r)).collect();
    // W must be all-empty to be a virgin completion target.
    if w.iter().any(|c| occupied.contains(&(c.q, c.r))) {
        return None;
    }
    let region_radius = (8 * budget_e as i16) + LEGAL_RADIUS;
    let center = w[0];
    let region: Vec<HexCoord> = cells_within(center, region_radius + 8)
        .into_iter()
        .filter(|c| !occupied.contains(&(c.q, c.r)))
        .collect();
    let region_set: HashSet<(i16, i16)> = region.iter().map(|c| (c.q, c.r)).collect();

    // BFS rooted at W's cells (all at hop 0) over empty cells.
    let mut dist_to_w: HashMap<(i16, i16), u32> = HashMap::new();
    let mut queue = VecDeque::new();
    for cell in w {
        dist_to_w.insert((cell.q, cell.r), 0);
        queue.push_back(*cell);
    }
    while let Some(cur) = queue.pop_front() {
        let cur_d = dist_to_w[&(cur.q, cur.r)];
        // Completing W from a legal seed at hop h costs h+6 placements, so only
        // legal cells within (E-6) hops of W matter. Discover hops up to E-6.
        if cur_d + 6 >= budget_e {
            continue;
        }
        for cand in cells_within(cur, LEGAL_RADIUS) {
            let key = (cand.q, cand.r);
            if !region_set.contains(&key) {
                continue;
            }
            if !dist_to_w.contains_key(&key) {
                dist_to_w.insert(key, cur_d + 1);
                queue.push_back(cand);
            }
        }
    }
    // Completing W from legal seed s costs dist_to_w[s] + 6 placements.
    let min_complete = dist_to_w
        .iter()
        .filter(|((q, r), _)| is_legal(&legal, HexCoord { q: *q, r: *r }))
        .map(|(_, &h)| h + 6)
        .min()?;
    if min_complete > budget_e {
        return None;
    }
    // Farthest legal seed whose completion of W fits the budget:
    // dist_to_w[s] + 6 <= E  <=>  dist_to_w[s] <= E-6.
    let cap = budget_e.saturating_sub(6);
    let mut best: Option<(i16, HexCoord)> = None;
    for c in &region {
        if !is_legal(&legal, *c) {
            continue;
        }
        if let Some(&h) = dist_to_w.get(&(c.q, c.r)) {
            if h <= cap {
                let dd = dist_to_window(*c, w);
                if best.map_or(true, |(bd, _)| dd > bd) {
                    best = Some((dd, *c));
                }
            }
        }
    }
    best.map(|(dd, s)| (dd, min_complete, s))
}

/// §3.1 virgin family: all-empty target window W on the Q axis, a relay chain
/// of `k = E-6` distance-8 steps (direction v = (8,-4)) reaching W, and an
/// attacker support stone one step from the far seed p_0 (makes p_0 legal
/// without touching W). Returns (stones, W cells, binding seed p_0).
fn r2_virgin_family(e: u32) -> (Vec<Owned>, [HexCoord; 6], HexCoord) {
    let attacker = Player::Player0;
    let k = e.saturating_sub(6) as i16;
    let w = q_window(HexCoord { q: 0, r: 0 });
    let v = HexCoord { q: 8, r: -4 }; // hex_distance(v,0) == 8
                                      // p_0 = -k*v is the far seed.
    let p0 = HexCoord {
        q: -k * v.q,
        r: -k * v.r,
    };
    // Support stone legalizes p_0 WITHOUT legalizing W (else the defender could
    // fill W directly, collapsing the relay chain). Step FURTHER from W along
    // the -v ray: support = p_0 + (-2,1) sits at distance 8k+2 from W's nearest
    // cell (> 8 for all k>=1) yet within distance 2 of p_0. For k=0 (p_0 is the
    // W cell (0,0)), use (-1,0) per docs §3.1 so W stays all-empty.
    let support = if k == 0 {
        HexCoord { q: -1, r: 0 }
    } else {
        HexCoord {
            q: p0.q - 2,
            r: p0.r + 1,
        }
    };
    let stones = vec![(support, attacker)];
    (stones, w, p0)
}

/// Every all-empty window incident to a cell (3 axes x 6 offsets = 18), given
/// the current stone set. Returns each as its 6 cells.
fn incident_virgin_windows(stones: &[Owned], cell: HexCoord) -> Vec<[HexCoord; 6]> {
    use hexo_engine::Axis;
    let occupied: BTreeSet<(i16, i16)> = stones.iter().map(|(c, _)| (c.q, c.r)).collect();
    let mut out = Vec::new();
    for axis in Axis::ALL {
        let vec = axis.vector();
        for offset in 0..6i16 {
            let start = HexCoord {
                q: cell.q - vec.q * offset,
                r: cell.r - vec.r * offset,
            };
            let mut window = [HexCoord::default(); 6];
            for (i, wc) in window.iter_mut().enumerate() {
                *wc = HexCoord {
                    q: start.q + vec.q * i as i16,
                    r: start.r + vec.r * i as i16,
                };
            }
            if window.iter().all(|c| !occupied.contains(&(c.q, c.r))) {
                out.push(window);
            }
        }
    }
    out
}

/// R2 structural finding: the shipped verifier ABSORBS `Z_virgin`. The
/// completion guard (`active_player == defender && cnt_D + d >= 6`) never fires
/// on an all-empty window (active_player == None), and the seed band only
/// targets ghost-illegal D10 obligations, never virgin windows. Confirm that a
/// high-exposure all-empty window contributes NOTHING to the shipped zone at
/// d < 6.
#[test]
#[ignore = "hunt dataset; run explicitly with --nocapture"]
fn hunt_r2_virgin_absorption() {
    let attacker = Player::Player0;
    // A position whose windows around W are all-empty (virgin).
    let (stones, w, _seed) = r2_virgin_family(8);
    let store = WindowStore::from_placements(&stones);
    // No window is defender-active near W (all empty), so the completion guard
    // yields no protected cells at any d < 6.
    for d in 1..=5u32 {
        // Empty obligation set: only the completion guard could add virgin cells.
        let ctx = seed_context(&stones, attacker, d, &[]);
        let virgin_contrib = ctx
            .protected
            .iter()
            .filter(|c| w.iter().any(|wc| wc == *c))
            .count();
        assert_eq!(
            virgin_contrib, 0,
            "virgin window W leaked into the shipped protected set at d={d}"
        );
    }
    // Directly: no window overlapping W is defender-active.
    let defender = attacker.other();
    let any_defender_active_on_w = store.entries().any(|e| {
        e.active_player() == Some(defender)
            && e.empty_cells().iter().any(|c| w.iter().any(|wc| wc == c))
    });
    assert!(
        !any_defender_active_on_w,
        "unexpected defender-active window over the virgin W"
    );
    println!(
        "HUNT R2_ABSORPTION verdict=Z_virgin ABSORBED by shipped verifier \
         (completion guard needs active_player==defender; virgin => None). \
         d>=6 short-circuits to full legal set. status=OK"
    );
}

/// R2 completion-reach: the §3.1 construction attains the fixed-window radius
/// 8(E^D-6). The defender can complete the all-empty window W from a seed at
/// exactly 8(E-6), in exactly E placements; a smaller virgin radius sheds it.
#[test]
#[ignore = "hunt dataset; run explicitly with --nocapture"]
fn hunt_r2_completion_reach() {
    println!("HUNT R2_COMPLETION begin (fixed-window virgin radius)");
    println!("  E | k=E-6 | reach | 8(E-6) | min_complete | seed==p0");
    for e in 6..=11u32 {
        let (stones, w, p0) = r2_virgin_family(e);
        let reach = complete_window_reach(&stones, &w, e);
        let (dist, min_c, seed) = reach.expect("W completable within E");
        let k = e - 6;
        assert_eq!(dist as i32, 8 * k as i32, "reach must equal 8(E-6)");
        assert_eq!(min_c, e, "min placements to complete W must equal E");
        // p0 is the (a) farthest binding seed at exactly 8(E-6).
        let seed_is_p0 = dist_to_window(seed, &w) == dist_to_window(p0, &w);
        println!(
            "  {e} |   {k}   |  {:>3}  |  {:>3}   |     {min_c}      |  {}",
            dist,
            8 * k,
            seed_is_p0
        );
        assert!(
            complete_window_reach(&stones, &w, e - 1).is_none(),
            "W must NOT be completable within E-1 placements"
        );
    }
    println!("HUNT R2_COMPLETION verdict=fixed-window radius 8(E-6) ATTAINED (matches docs §3.1)");
}

/// R2 FULL-UNION probe (obstruction surface). `Z_virgin` unions over ALL
/// all-empty windows; the target W's binding seed p_0 lies in up to 18 incident
/// windows. If ANY of them is a legitimate `Z_virgin` member — i.e. all-empty
/// AND exposure `E^D >= 6` in the same certificate — it selects p_0 at distance
/// 0 and W's long-radius coverage of p_0 is redundant (the union self-covers,
/// so it could shrink). The construction (docs §3.1) must instead force every
/// other incident window to have exposure < 6 or be non-D-alive.
///
/// CRUCIAL: whether an incident all-empty window has exposure `>= 6` is a
/// D16 recurrence quantity fixed by the proof tree (when the attacker enters
/// that window), NOT a static board property. So static geometry alone CANNOT
/// decide self-coverage. This probe reports the geometric OBSTRUCTION SURFACE:
/// how many distinct all-empty windows are incident to p_0 (each a potential
/// self-cover if the certificate happens to give it exposure >= 6). Resolving
/// the union's sharpness needs certificate-level exposure labels — the blocker.
#[test]
#[ignore = "hunt dataset; run explicitly with --nocapture"]
fn hunt_r2_full_union() {
    use hexo_engine::Axis;
    println!("HUNT R2_UNION begin (obstruction surface for the binding seed p_0)");
    println!("  E | incident_all_empty_windows@p0 | of_those_axis(Q/R/QR) | note");
    for e in 7..=11u32 {
        let (stones, _w, p0) = r2_virgin_family(e);
        let incident = incident_virgin_windows(&stones, p0);
        // Break down by axis for transparency.
        let occupied: BTreeSet<(i16, i16)> = stones.iter().map(|(c, _)| (c.q, c.r)).collect();
        let mut per_axis = [0usize; 3];
        for (ai, axis) in Axis::ALL.iter().enumerate() {
            let vec = axis.vector();
            for offset in 0..6i16 {
                let start = HexCoord {
                    q: p0.q - vec.q * offset,
                    r: p0.r - vec.r * offset,
                };
                let win: Vec<HexCoord> = (0..6i16)
                    .map(|i| HexCoord {
                        q: start.q + vec.q * i,
                        r: start.r + vec.r * i,
                    })
                    .collect();
                if win.iter().all(|c| !occupied.contains(&(c.q, c.r))) {
                    per_axis[ai] += 1;
                }
            }
        }
        println!(
            "  {e} |              {:>2}               |       {}/{}/{}        | all incident windows are all-empty (isolated seed)",
            incident.len(),
            per_axis[0],
            per_axis[1],
            per_axis[2],
        );
    }
    println!(
        "HUNT R2_UNION verdict=INCONCLUSIVE(static): p_0 is incident to 18 all-empty windows, \
         each a potential self-cover IFF the certificate gives it exposure>=6. Exposure is a \
         D16 recurrence label (attacker-entry timing), not a static board property, so the \
         full-union sharpness CANNOT be settled by geometry alone. BLOCKER=certificate exposure \
         labels. NOTE: the shipped verifier ABSORBS Z_virgin (see R2_ABSORPTION), so the union \
         radius is not even used in the shipped code path."
    );
}
