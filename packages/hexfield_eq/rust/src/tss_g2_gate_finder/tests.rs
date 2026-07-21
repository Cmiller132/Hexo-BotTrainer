//! Tests for the finder-side FHW gate closure builder:
//!   * exact bounded transversal + FC/RC/WC geometry unit tests,
//!   * exhaustive `FhwKappaRowV1`/`FhwRoleRowV1` leaf realization (the positive
//!     classifier fixtures the future accept path will recompute),
//!   * the four `(d in W, s in W)` incidence pairs,
//!   * structural closure self-check behavior,
//!   * the corpus firing measurement, and
//!   * fixture + manifest emission to `tests/fixtures/g2_gates/`.

use super::*;
use hexo_engine::apply_placement;

// ---------------------------------------------------------------------------
// Small constructed states
// ---------------------------------------------------------------------------

/// A legal state with a compact stone cluster near the origin. Cells near the
/// cluster are legal; cells far away (> LEGAL_RADIUS from every stone) are
/// ghost-illegal. Used as the source position for `Ghost` geometry tests.
fn cluster_state() -> RustHexoState {
    let mut state = RustHexoState::new();
    // Opening (P0), then alternating turns kept inside the legal radius.
    let moves = [
        (0, 0),   // P0 opening
        (1, 0),   // P1 first
        (2, 0),   // P1 second
        (3, 0),   // P0 first
        (0, 1),   // P0 second
        (1, 1),   // P1 first
        (2, 1),   // P1 second
        (3, 1),   // P0 first
        (0, 2),   // P0 second
    ];
    for (q, r) in moves {
        apply_placement(&mut state, Placement { coord: HexCoord::new(q, r) })
            .unwrap_or_else(|e| panic!("cluster move ({q},{r}) illegal: {e:?}"));
    }
    state
}

/// A ghost built from the cluster state by applying one nearby representative.
fn cluster_ghost() -> Ghost {
    let state = cluster_state();
    let s = HexCoord::new(4, 0); // legal, near the cluster
    Ghost::new(&state, s).expect("ghost construction")
}

fn geom(d_alive: bool, touched: bool, all_empty: bool, cnt_d: u32) -> WindowGeom {
    WindowGeom {
        d_alive,
        touched,
        all_empty,
        cnt_d,
    }
}

// ---------------------------------------------------------------------------
// Transversal + geometry primitives
// ---------------------------------------------------------------------------

#[test]
fn transversal_number_exact_small_families() {
    let empty: Vec<Vec<HexCoord>> = vec![];
    assert_eq!(transversal_number(&empty, 3), 0);

    // Single set of two empties -> tau = 1 (either cell hits it).
    let one = vec![vec![HexCoord::new(0, 0), HexCoord::new(1, 0)]];
    assert_eq!(transversal_number(&one, 3), 1);

    // Two sets sharing a common cell -> tau = 1.
    let common = vec![
        vec![HexCoord::new(0, 0), HexCoord::new(1, 0)],
        vec![HexCoord::new(0, 0), HexCoord::new(5, 5)],
    ];
    assert_eq!(transversal_number(&common, 3), 1);

    // Two disjoint sets -> tau = 2.
    let disjoint = vec![
        vec![HexCoord::new(0, 0), HexCoord::new(1, 0)],
        vec![HexCoord::new(5, 5), HexCoord::new(6, 5)],
    ];
    assert_eq!(transversal_number(&disjoint, 3), 2);

    // Three pairwise-disjoint sets need > 2 hits. The function is exact only up
    // to 2 and returns `cap + 1` above that (all the builder ever needs, since
    // b <= 2). With cap = 2 that is 3, meaning "> 2".
    let three = vec![
        vec![HexCoord::new(0, 0)],
        vec![HexCoord::new(5, 5)],
        vec![HexCoord::new(9, 0)],
    ];
    assert_eq!(transversal_number(&three, 2), 3);
    // A two-hittable family is reported exactly under the same cap.
    let two = vec![
        vec![HexCoord::new(0, 0)],
        vec![HexCoord::new(5, 5)],
    ];
    assert_eq!(transversal_number(&two, 2), 2);
}

#[test]
fn frontier_covered_is_exact_when_equal_and_geometric_otherwise() {
    let ghost = cluster_ghost();
    let d = HexCoord::new(1, 0);
    // d == s is always Exact/FC.
    assert!(frontier_covered(d, d, &ghost));
    // A far cell's B_8 ball cannot be entirely inside Lambda(cluster) -> not FC.
    let far = HexCoord::new(40, 0);
    let s = HexCoord::new(4, 0);
    assert!(!frontier_covered(far, s, &ghost));
}

#[test]
fn ghost_illegal_predicate_matches_far_empty_cells() {
    let ghost = cluster_ghost();
    // Far, empty, illegal cell.
    assert!(ghost.is_ghost_illegal(HexCoord::new(40, 0)));
    // Near, legal cell is not ghost-illegal.
    assert!(!ghost.is_ghost_illegal(HexCoord::new(4, 1)));
    // Occupied cell is not ghost-illegal (it is occupied, not illegal-empty).
    assert!(!ghost.is_ghost_illegal(HexCoord::new(0, 0)));
}

// ---------------------------------------------------------------------------
// FhwKappaRowV1: exhaustive leaf realization
// ---------------------------------------------------------------------------

fn win(start: (i16, i16), axis: Axis) -> WindowKey {
    WindowKey {
        start: HexCoord::new(start.0, start.1),
        axis,
    }
}

#[test]
fn kappa_leaf_non_d_alive() {
    let ghost = cluster_ghost();
    let w = win((0, 0), Axis::Q);
    let g = geom(false, false, false, 0); // window carries a claimant stone
    let out = classify_window(FhwEdgeClassV1::Exact, HexCoord::new(0, 0), w, 3, &g, &ghost);
    assert_eq!(out, Some((FhwKappaRowV1::NonDAlive, 0, GuardResultV1::NotApplicable)));
}

#[test]
fn kappa_leaf_exact_or_fc_nonincident() {
    let ghost = cluster_ghost();
    let w = win((10, 0), Axis::Q); // cells (10,0)..(15,0)
    let d = HexCoord::new(0, 0); // not in W
    let g = geom(true, false, true, 0);
    let out = classify_window(FhwEdgeClassV1::FrontierCovered, d, w, 4, &g, &ghost);
    assert_eq!(
        out,
        Some((FhwKappaRowV1::ExactOrFcNonIncident, 0, GuardResultV1::NotApplicable))
    );
}

#[test]
fn kappa_leaf_exact_or_fc_direct_touched_and_all_empty() {
    let ghost = cluster_ghost();
    let w = win((10, 0), Axis::Q);
    let d = HexCoord::new(12, 0); // in W

    // Touched variant: cnt_d + 1 + q < 6.
    let touched = geom(true, true, false, 2);
    let out = classify_window(FhwEdgeClassV1::Exact, d, w, 2, &touched, &ghost);
    assert_eq!(out, Some((FhwKappaRowV1::ExactOrFcDirect, 1, GuardResultV1::Pass)));
    // Guard failure (cnt_d + 1 + q = 6) rejects.
    let fail = classify_window(FhwEdgeClassV1::Exact, d, w, 3, &touched, &ghost);
    assert_eq!(fail, None);

    // All-empty variant: 1 + q < 6.
    let empty = geom(true, false, true, 0);
    let out = classify_window(FhwEdgeClassV1::FrontierCovered, d, w, 4, &empty, &ghost);
    assert_eq!(out, Some((FhwKappaRowV1::ExactOrFcDirect, 1, GuardResultV1::Pass)));
    let fail = classify_window(FhwEdgeClassV1::FrontierCovered, d, w, 5, &empty, &ghost);
    assert_eq!(fail, None);
}

#[test]
fn kappa_leaf_non_fc_touched_nonincident_and_direct() {
    let ghost = cluster_ghost();
    let w = win((10, 0), Axis::Q);
    let touched = geom(true, true, false, 2);

    // Non-incident: d not in W.
    let out = classify_window(
        FhwEdgeClassV1::NonFrontierCovered,
        HexCoord::new(0, 0),
        w,
        3,
        &touched,
        &ghost,
    );
    assert_eq!(
        out,
        Some((FhwKappaRowV1::NonFcTouchedNonIncident, 0, GuardResultV1::NotApplicable))
    );

    // Incident: d in W, guard cnt_d + 1 + q < 6.
    let out = classify_window(
        FhwEdgeClassV1::NonFrontierCovered,
        HexCoord::new(12, 0),
        w,
        2,
        &touched,
        &ghost,
    );
    assert_eq!(out, Some((FhwKappaRowV1::NonFcTouchedDirect, 1, GuardResultV1::Pass)));
}

#[test]
fn kappa_leaf_non_fc_empty_direct() {
    let ghost = cluster_ghost();
    let w = win((10, 0), Axis::Q);
    let empty = geom(true, false, true, 0);
    let out = classify_window(
        FhwEdgeClassV1::NonFrontierCovered,
        HexCoord::new(12, 0),
        w,
        3,
        &empty,
        &ghost,
    );
    assert_eq!(out, Some((FhwKappaRowV1::NonFcEmptyDirect, 1, GuardResultV1::Pass)));
}

#[test]
fn kappa_leaf_non_fc_empty_nonincident_qlt6() {
    let ghost = cluster_ghost();
    let w = win((10, 0), Axis::Q);
    let empty = geom(true, false, true, 0);
    let out = classify_window(
        FhwEdgeClassV1::NonFrontierCovered,
        HexCoord::new(0, 0),
        w,
        5,
        &empty,
        &ghost,
    );
    assert_eq!(
        out,
        Some((FhwKappaRowV1::NonFcEmptyNonIncidentQlt6, 0, GuardResultV1::NotApplicable))
    );
}

#[test]
fn kappa_leaf_non_fc_empty_nonincident_wc_pass() {
    // q = 6 => WC ball radius 0 => only the six window cells are tested. With an
    // all-empty window whose cells are legal in the ghost (near the cluster),
    // none are ghost-illegal, so WC is empty and passes.
    let state = cluster_state();
    let ghost = Ghost::new(&state, HexCoord::new(4, 0)).unwrap();
    let w = win((4, 2), Axis::Q); // cells (4,2)..(9,2): empty, near cluster => legal
    // Confirm the window cells are not ghost-illegal.
    for c in w.cells() {
        assert!(!ghost.is_ghost_illegal(c), "cell {c:?} unexpectedly ghost-illegal");
    }
    let d = HexCoord::new(2, 2); // not in W, near cluster
    assert!(!w.contains(d));
    let empty = geom(true, false, true, 0);
    let out = classify_window(FhwEdgeClassV1::NonFrontierCovered, d, w, 6, &empty, &ghost);
    assert_eq!(
        out,
        Some((FhwKappaRowV1::NonFcEmptyNonIncidentWcPass, 0, GuardResultV1::Pass))
    );
}

#[test]
fn kappa_leaf_non_fc_empty_nonincident_wc_fail_is_geometrically_unrealizable_with_passing_guard() {
    // FINDING (documented in the report): on the non-FC / all-empty /
    // nonincident / q>=6 branch, WC fails iff there is a ghost-illegal cell z
    // with dist(z,d)<=8 and dist(z,W)<=8(q-6). Then dist(d,W) <= 8 + 8(q-6) =
    // 8(q-5), which CONTRADICTS the mandatory N-virgin guard dist(d,W) >
    // 8(1+q-6) = 8(q-5). So whenever WC fails, the guard fails and the gate
    // REJECTS; the NonFcEmptyNonIncidentWcFail leaf is never emitted with a
    // passing guard. We assert the rejection here (the realized fixture is the
    // rejection, not a passing row).
    let state = cluster_state();
    // Ghost with a wide ghost-illegal frontier: apply a representative but keep
    // a far region illegal.
    let ghost = Ghost::new(&state, HexCoord::new(4, 0)).unwrap();
    let q = 7u32; // WC radius 8, N-virgin threshold 16
    let d = HexCoord::new(10, 0);
    // A window near enough to d that a ghost-illegal cell could fall in both
    // balls; by the triangle inequality this forces dist(d,W) <= 16, so the
    // guard dist(d,W) > 16 cannot hold. Pick W adjacent so WC can fail.
    let w = win((11, 0), Axis::Q); // dist(d,W) = 1
    let empty = geom(true, false, true, 0);
    let out = classify_window(FhwEdgeClassV1::NonFrontierCovered, d, w, q, &empty, &ghost);
    // Either WC passes (WcPass) or, if it fails, the guard fails => None.
    assert!(
        matches!(
            out,
            None | Some((FhwKappaRowV1::NonFcEmptyNonIncidentWcPass, _, _))
        ),
        "WcFail must not be emitted with a passing guard; got {out:?}"
    );
}

// ---------------------------------------------------------------------------
// FhwRoleRowV1: leaf realization
// ---------------------------------------------------------------------------

#[test]
fn role_leaf_exact_or_fc_zero() {
    let ghost = cluster_ghost();
    let d = HexCoord::new(1, 0);
    let y = HexCoord::new(5, 0);
    assert_eq!(
        classify_role(FhwEdgeClassV1::Exact, d, y, 3, &ghost),
        Some((FhwRoleRowV1::ExactOrFcZero, 0))
    );
    assert_eq!(
        classify_role(FhwEdgeClassV1::FrontierCovered, d, y, 3, &ghost),
        Some((FhwRoleRowV1::ExactOrFcZero, 0))
    );
    // Carrier avoidance: d == y rejects on every class.
    assert_eq!(classify_role(FhwEdgeClassV1::Exact, d, d, 3, &ghost), None);
}

#[test]
fn role_leaf_non_fc_rc_zero() {
    // Non-FC, ghost-illegal carrier, k = 0 => inner ball empty => RC passes.
    let ghost = cluster_ghost();
    let y = HexCoord::new(40, 0); // ghost-illegal
    assert!(ghost.is_ghost_illegal(y));
    let d = HexCoord::new(1, 0);
    assert_eq!(
        classify_role(FhwEdgeClassV1::NonFrontierCovered, d, y, 0, &ghost),
        Some((FhwRoleRowV1::NonFcRcZero, 0))
    );
}

#[test]
fn role_leaf_non_fc_charged_via_ghost_legal_carrier() {
    // Non-FC, ghost-LEGAL carrier => conservative charged row, epsilon 1, no
    // D22-N required.
    let ghost = cluster_ghost();
    let y = HexCoord::new(4, 1); // legal near the cluster => not ghost-illegal
    assert!(!ghost.is_ghost_illegal(y));
    let d = HexCoord::new(1, 0);
    assert_eq!(
        classify_role(FhwEdgeClassV1::NonFrontierCovered, d, y, 2, &ghost),
        Some((FhwRoleRowV1::NonFcCharged, 1))
    );
}

#[test]
fn role_charged_via_ghost_illegal_rc_fail_is_geometrically_unrealizable() {
    // FINDING (documented): RC fails iff a ghost-illegal z has dist(z,d)<=8 and
    // dist(z,y)<=8(k-1), forcing dist(d,y) <= 8k, which CONTRADICTS the
    // mandatory D22-N guard dist(d,y) > 8k. So the charged row via a
    // ghost-illegal RC-fail carrier is never emitted; it either resolves to
    // NonFcRcZero (RC passes) or rejects. Assert the classifier never returns a
    // charged row while claiming a ghost-illegal RC-fail with passing D22-N.
    let ghost = cluster_ghost();
    let y = HexCoord::new(40, 0); // ghost-illegal
    let d = HexCoord::new(35, 0); // within 8 of y (dist 5)
    let k = 1u32; // inner ball radius 0 => RC tests GI ∩ B_8(d) ∩ {y}
    let out = classify_role(FhwEdgeClassV1::NonFrontierCovered, d, y, k, &ghost);
    // y is ghost-illegal and within B_8(d), inner ball {y} => RC fails; D22-N
    // dist(d,y)=5 > 8*1=8 is false => reject.
    assert_eq!(out, None);
}

// ---------------------------------------------------------------------------
// The four (d in W, s in W) incidence pairs
// ---------------------------------------------------------------------------

#[test]
fn all_four_incidence_pairs_are_constructible() {
    let w = win((10, 0), Axis::Q); // cells (10,0)..(15,0)
    let inside_a = HexCoord::new(11, 0);
    let inside_b = HexCoord::new(13, 0);
    let outside_a = HexCoord::new(0, 0);
    let outside_b = HexCoord::new(1, 0);
    let cases = [
        (outside_a, outside_b, (false, false)),
        (outside_a, inside_b, (false, true)),
        (inside_a, outside_b, (true, false)),
        (inside_a, inside_b, (true, true)),
    ];
    for (d, s, (exp_d, exp_s)) in cases {
        assert_eq!(w.contains(d), exp_d, "d_in mismatch for {d:?}");
        assert_eq!(w.contains(s), exp_s, "s_in mismatch for {s:?}");
    }
}

// ---------------------------------------------------------------------------
// Structural closure builder — behavior on eligibility
// ---------------------------------------------------------------------------

#[test]
fn closure_rejects_non_defender_and_opening() {
    // Fresh state: Opening, P0 to move. With claimant = P0 the mover is the
    // claimant (not the defender) => NotDefenderToMove. With claimant = P1 the
    // defender is to move but the phase is Opening => Opening.
    let state = RustHexoState::new();
    assert_eq!(
        try_build_gate(&state, Player::Player0, 64),
        Err(ClosureFail::NotDefenderToMove)
    );
    assert_eq!(
        try_build_gate(&state, Player::Player1, 64),
        Err(ClosureFail::Opening)
    );
}

#[test]
fn closure_rejects_when_no_attacker_threats() {
    // Quiet post-opening defender-to-move position with no attacker threats.
    let state = cluster_state();
    // In cluster_state the side to move has no >=4 attacker window for the
    // opponent; any defender-to-move framing yields NoThreats or an earlier
    // eligibility failure — never a spurious gate.
    let claimant = state.current_player().other();
    let result = try_build_gate(&state, claimant, 64);
    assert!(
        matches!(
            result,
            Err(ClosureFail::NoThreats)
                | Err(ClosureFail::NotDefenderToMove)
                | Err(ClosureFail::ThreatCountOutOfRange)
                | Err(ClosureFail::TransversalNotB)
        ),
        "unexpected closure on a quiet position: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Corpus firing measurement
// ---------------------------------------------------------------------------

#[test]
fn corpus_firing_measurement() {
    let corpus = crate::tss_corpus::load_corpus();
    let mut total = FiringStats::default();
    let mut per_position: Vec<(String, FiringStats)> = Vec::new();
    let mut captured: Vec<(String, GateExample)> = Vec::new();

    for pos in &corpus {
        let claimant = if pos.expect_win {
            pos.state.current_player()
        } else {
            pos.state.current_player().other()
        };
        let horizon = pos.state.placements_made() + 40;
        let mut stats = FiringStats::default();
        // Bounded forcing walk: enough to reach forcing defender nodes without
        // blowing up the serialized suite runtime.
        measure_position(&pos.state, claimant, horizon, 6_000, 10, &mut stats);
        for ex in &stats.examples {
            if captured.len() < MAX_EXAMPLES {
                captured.push((pos.id.clone(), ex.clone()));
            }
        }
        // Fold into the total.
        total.defender_nodes_seen += stats.defender_nodes_seen;
        total.eligible_nodes += stats.eligible_nodes;
        total.gates_closed += stats.gates_closed;
        total.reductive_gates += stats.reductive_gates;
        total.sum_kernel += stats.sum_kernel;
        total.sum_representatives += stats.sum_representatives;
        total.sum_legal += stats.sum_legal;
        if stats.gates_closed > 0 {
            if total.gates_closed == stats.gates_closed {
                // first contributor
                total.best_kernel_ratio = stats.best_kernel_ratio;
                total.best_representative_ratio = stats.best_representative_ratio;
            } else {
                total.best_kernel_ratio = total.best_kernel_ratio.min(stats.best_kernel_ratio);
                total.best_representative_ratio =
                    total.best_representative_ratio.min(stats.best_representative_ratio);
            }
        }
        for (reason, count) in &stats.failures {
            for _ in 0..*count {
                total.record_fail(*reason);
            }
        }
        per_position.push((pos.id.clone(), stats));
    }

    eprintln!("=== G2 FHW gate closure firing measurement (19-position forcing corpus) ===");
    eprintln!(
        "defender_nodes_seen={} eligible_nodes={} gates_closed={} reductive_gates={}",
        total.defender_nodes_seen, total.eligible_nodes, total.gates_closed, total.reductive_gates
    );
    if total.gates_closed > 0 {
        eprintln!(
            "avg |K|={:.2} avg |R|={:.2} avg |Legal|={:.2} best_kernel_ratio={:.4} best_rep_ratio={:.4}",
            total.sum_kernel as f64 / total.gates_closed as f64,
            total.sum_representatives as f64 / total.gates_closed as f64,
            total.sum_legal as f64 / total.gates_closed as f64,
            total.best_kernel_ratio,
            total.best_representative_ratio,
        );
    }
    let mut failures = total.failures.clone();
    failures.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    eprintln!("closure-failure histogram (over eligible nodes that did not close):");
    for (reason, count) in &failures {
        eprintln!("  {reason:?}: {count}");
    }
    for (id, stats) in &per_position {
        if stats.eligible_nodes > 0 {
            eprintln!(
                "  pos={id} defender_nodes={} eligible={} closed={} reductive={}",
                stats.defender_nodes_seen, stats.eligible_nodes, stats.gates_closed, stats.reductive_gates
            );
        }
    }

    // The measurement must actually traverse the corpus.
    assert!(total.defender_nodes_seen > 0, "walk reached no defender nodes");
    // The reductive prize exists at production-shaped nodes: assert we closed at
    // least one genuinely reductive gate (R subsetneq K) on the corpus. This is
    // the load-bearing evidence for the 40.5%-ceiling question.
    assert!(
        total.reductive_gates > 0,
        "no reductive gate closed on the corpus — the reductive prize would be absent"
    );
    eprintln!("captured {} example gates for fixtures", captured.len());

    // Emit the measurement into the fixture manifest as ground-truth firing
    // numbers, and serialize the captured real gates as positive fixtures.
    write_firing_manifest(&total, &per_position);
    write_gate_fixtures(&captured);
}

/// Serialize captured real closed gates (production-shaped positions) as
/// positive fixtures: exact stone occupancy of `P_Q`, the threat family H_Q,
/// the kernel K, representatives R, and every `(d -> s : class)` edge. Each is a
/// self-check-passing structural gate the future accept path will recompute.
fn write_gate_fixtures(captured: &[(String, GateExample)]) {
    let dir = fixture_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let mut out = String::new();
    out.push_str("# G2 FHW structural gate fixtures (real production-shaped nodes)\n");
    out.push_str("# Each GATE block is a self-check-passing structural closure\n");
    out.push_str("# (design §3.3): eligibility, exact transversal == b, kernel K,\n");
    out.push_str("# representatives R, retraction phi, and per-edge class. Role/window\n");
    out.push_str("# rows are NOT emitted here (they need a proven representative subtree;\n");
    out.push_str("# see the report's documented boundary).\n\n");
    for (idx, (corpus_id, ex)) in captured.iter().enumerate() {
        let cls = |c: FhwEdgeClassV1| match c {
            FhwEdgeClassV1::Exact => "Exact",
            FhwEdgeClassV1::FrontierCovered => "FC",
            FhwEdgeClassV1::NonFrontierCovered => "NonFC",
        };
        out.push_str(&format!(
            "GATE id={idx} corpus={corpus_id} b={} claimant=P{} side_to_move=P{} placements_made={} legal={} |K|={} |R|={} reductive={} escape_ply={}\n",
            ex.build.b,
            player_idx(ex.claimant),
            player_idx(ex.side_to_move),
            ex.placements_made,
            ex.build.legal_count,
            ex.build.kernel.len(),
            ex.build.representatives.len(),
            ex.build.is_reductive(),
            ex.build.escape_resolution_ply,
        ));
        for (c, p) in &ex.occupancy {
            out.push_str(&format!("  stone {} {} P{}\n", c.q, c.r, player_idx(*p)));
        }
        for w in &ex.build.threats {
            out.push_str(&format!("  threat axis={} start {} {}\n", w.axis.index(), w.start.q, w.start.r));
        }
        for k in &ex.build.kernel {
            out.push_str(&format!("  kernel {} {}\n", k.q, k.r));
        }
        for r in &ex.build.representatives {
            out.push_str(&format!("  rep {} {}\n", r.q, r.r));
        }
        for e in &ex.build.edges {
            out.push_str(&format!(
                "  edge d {} {} -> s {} {} class={}\n",
                e.real_reply.q, e.real_reply.r, e.representative.q, e.representative.r, cls(e.edge_class)
            ));
        }
        out.push('\n');
    }
    let _ = std::fs::write(dir.join("structural_gates.txt"), out);
}

fn player_idx(p: Player) -> u8 {
    match p {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}

// ---------------------------------------------------------------------------
// Fixture + manifest emission
// ---------------------------------------------------------------------------

fn fixture_dir() -> std::path::PathBuf {
    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("tests");
    dir.push("fixtures");
    dir.push("g2_gates");
    dir
}

fn write_firing_manifest(total: &FiringStats, per_position: &[(String, FiringStats)]) {
    let dir = fixture_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let mut out = String::new();
    out.push_str("# G2 FHW gate closure — corpus firing measurement\n");
    out.push_str("# Generated by tss_g2_gate_finder::tests::corpus_firing_measurement.\n");
    out.push_str(&format!(
        "defender_nodes_seen {}\neligible_nodes {}\ngates_closed {}\nreductive_gates {}\n",
        total.defender_nodes_seen, total.eligible_nodes, total.gates_closed, total.reductive_gates
    ));
    if total.gates_closed > 0 {
        out.push_str(&format!(
            "avg_kernel {:.4}\navg_representatives {:.4}\navg_legal {:.4}\nbest_kernel_ratio {:.4}\nbest_representative_ratio {:.4}\n",
            total.sum_kernel as f64 / total.gates_closed as f64,
            total.sum_representatives as f64 / total.gates_closed as f64,
            total.sum_legal as f64 / total.gates_closed as f64,
            total.best_kernel_ratio,
            total.best_representative_ratio,
        ));
    }
    out.push_str("# closure-failure histogram\n");
    let mut failures = total.failures.clone();
    failures.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (reason, count) in &failures {
        out.push_str(&format!("fail {reason:?} {count}\n"));
    }
    out.push_str("# per-position (only positions with eligible nodes)\n");
    for (id, stats) in per_position {
        if stats.eligible_nodes > 0 {
            out.push_str(&format!(
                "pos {id} defender_nodes {} eligible {} closed {} reductive {}\n",
                stats.defender_nodes_seen, stats.eligible_nodes, stats.gates_closed, stats.reductive_gates
            ));
        }
    }
    let _ = std::fs::write(dir.join("firing_measurement.txt"), out);
}

#[test]
fn emit_classifier_fixture_manifest() {
    // Serialize the realized positive-fixture inventory (the classifier leaves,
    // role rows, and incidence pairs) with their per-fixture properties. These
    // are the positive fixtures the future accept path will recompute.
    let dir = fixture_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let manifest = r#"# G2 FHW gate positive-fixture manifest (v1)
# schema: one record per line: <class> <name> <property>=<value> ...
#
# All rows are the exact classifier outputs the accept path will recompute
# (design §3.3 tables). "q" = Q_cut(C_s,W), "k" = f_cut(C_s,rho): in fixtures
# these are supplied inputs (a proven representative subtree provides them for a
# real gate; that provenance is the documented out-of-lane boundary).

kappa NonDAlive realized=yes edge=exact d_in=any note=window_carries_claimant_stone
kappa ExactOrFcNonIncident realized=yes edge=fc d_in=false q=4 kappa=0 guard=NotApplicable
kappa ExactOrFcDirect realized=yes edge=exact d_in=true touched=true q=2 cnt_d=2 kappa=1 guard=Pass
kappa ExactOrFcDirect realized=yes edge=fc d_in=true all_empty=true q=4 kappa=1 guard=Pass
kappa NonFcTouchedNonIncident realized=yes edge=nonfc touched=true d_in=false q=3 kappa=0 guard=NotApplicable
kappa NonFcTouchedDirect realized=yes edge=nonfc touched=true d_in=true q=2 cnt_d=2 kappa=1 guard=Pass
kappa NonFcEmptyDirect realized=yes edge=nonfc all_empty=true d_in=true q=3 kappa=1 guard=Pass
kappa NonFcEmptyNonIncidentQlt6 realized=yes edge=nonfc all_empty=true d_in=false q=5 kappa=0 guard=NotApplicable
kappa NonFcEmptyNonIncidentWcPass realized=yes edge=nonfc all_empty=true d_in=false q=6 wc=pass kappa=0 guard=Pass
kappa NonFcEmptyNonIncidentWcFail realized=no reason=triangle_inequality_contradiction detail=WC_fail_witness_forces_dist(d,W)<=8(q-5)_but_N-virgin_guard_requires_>8(q-5)_so_only_rejection_is_reachable

role ExactOrFcZero realized=yes edge=exact epsilon=0
role ExactOrFcZero realized=yes edge=fc epsilon=0
role NonFcRcZero realized=yes edge=nonfc ghost_illegal=yes k=0 rc=pass epsilon=0
role NonFcCharged realized=yes edge=nonfc ghost_legal=yes epsilon=1 note=conservative_charged_no_D22N_required
role NonFcCharged-via-ghost-illegal realized=no reason=triangle_inequality_contradiction detail=RC_fail_witness_forces_dist(d,y)<=8k_but_D22N_guard_requires_>8k

incidence dd_out_ss_out realized=yes d_in=false s_in=false
incidence dd_out_ss_in realized=yes d_in=false s_in=true
incidence dd_in_ss_out realized=yes d_in=true s_in=false
incidence dd_in_ss_in realized=yes d_in=true s_in=true
"#;
    let _ = std::fs::write(dir.join("classifier_fixtures.txt"), manifest);
}
