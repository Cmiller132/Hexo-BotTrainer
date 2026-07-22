use crate::tss_core::{ProofStatus, SolveCaps, SolveGoal};
use crate::tss_corpus::load_corpus;
use crate::tss_refute_leaf_cert::root_plus_empty_set_baseline_bytes;
use crate::tss_refute_leaf_cert::ReachableRootV1;
use crate::tss_refute_leaf_cert::{
    encode_artifact, root_header_from_engine, root_semantic_preimage_v1, root_semantic_sha256,
    sha256, LeafArtifactV1, LeafCountsV1,
};
use crate::tss_refute_produce::{
    produce_refute_leaf_exact_v1_after_search, ProduceResultV1, RefuteLeafModeV1, SearchProfileV1,
};
use crate::tss_refute_produce::{
    producer_counts_for_test, producer_counts_without_earlier_for_test,
};
use crate::tss_refute_verify::OfflinePolicyV1;
use crate::tss_refute_verify::{verify_refute_leaf_exact_v1, VerifyRejectionV1};
use crate::tss_solver::{TssSolver, WidthOptions};
use hexo_engine::{apply_placement, HexoState, Placement, TurnPhase};

fn replay(coords: &[(i16, i16)]) -> HexoState {
    let mut s = HexoState::new();
    for &(q, r) in coords {
        apply_placement(
            &mut s,
            Placement {
                coord: hexo_engine::HexCoord { q, r },
            },
        )
        .unwrap();
    }
    s
}
fn corpus_prefix(id: &str, n: usize) -> HexoState {
    let p = load_corpus().into_iter().find(|p| p.id == id).unwrap();
    let coords = p
        .state
        .placement_history()
        .iter()
        .take(n)
        .map(|r| (r.coord.q, r.coord.r))
        .collect::<Vec<_>>();
    replay(&coords)
}
fn forged(state: &HexoState, counts: LeafCountsV1) -> Vec<u8> {
    let root = root_header_from_engine(state).unwrap();
    let digest = root_semantic_sha256(&root);
    encode_artifact(&LeafArtifactV1 {
        root,
        root_semantic_sha256: digest,
        counts,
    })
}
fn emit(state: &HexoState) -> crate::tss_refute_produce::ProducedRefuteLeafV1 {
    let token = ReachableRootV1::from_trusted_engine_state(state).unwrap();
    let result = produce_refute_leaf_exact_v1_after_search(
        RefuteLeafModeV1::Emit,
        state,
        &token,
        OfflinePolicyV1::default(),
        SearchProfileV1::completed_natural_width_exhaust(),
        1,
        200,
    );
    let ProduceResultV1::Emitted(p) = result else {
        panic!("not emitted: {result:?}")
    };
    p
}

#[test]
#[ignore = "discovery/economics harness; serialized release-only"]
fn refute_leaf_corpus_discovery() {
    for position in load_corpus() {
        match producer_counts_for_test(&position.state, OfflinePolicyV1::default()) {
            Ok((counts, completion, tactical, tight)) => println!(
                "REFUTE_DISCOVERY id={} t={} q={} classes={} fail={}/{}/{}/{} positive={}/{}/{} eligible={}",
                position.id,
                counts.t_count,
                counts.q_count,
                counts.quotient_class_count,
                counts.fail_no_new,
                counts.fail_defender_first,
                counts.fail_loose_0,
                counts.fail_loose_1,
                completion,
                tactical,
                tight,
                completion == 0 && tactical == 0 && tight == 0,
            ),
            Err(error) => println!("REFUTE_DISCOVERY id={} error={error}", position.id),
        }
    }
}

#[test]
#[ignore = "fixture discovery"]
fn focused_q2_discovery() {
    let state = replay(&[
        (0, 0),
        (0, 1),
        (-1, 5),
        (1, 0),
        (4, 0),
        (1, 4),
        (4, 2),
        (5, 0),
        (-2, -2),
    ]);
    println!(
        "FOCUSED_Q2 {:?}",
        producer_counts_for_test(&state, OfflinePolicyV1::default())
    );
}

#[test]
fn focused_q2_produce_verify_round_trip() {
    let state = replay(&[
        (0, 0),
        (0, 1),
        (-1, 5),
        (1, 0),
        (4, 0),
        (1, 4),
        (4, 2),
        (5, 0),
        (-2, -2),
    ]);
    let token = ReachableRootV1::from_trusted_engine_state(&state).unwrap();
    let result = produce_refute_leaf_exact_v1_after_search(
        RefuteLeafModeV1::Emit,
        &state,
        &token,
        OfflinePolicyV1::default(),
        SearchProfileV1::completed_natural_width_exhaust(),
        1,
        200,
    );
    let ProduceResultV1::Emitted(produced) = result else {
        panic!("not emitted: {result:?}")
    };
    assert_eq!(produced.artifact.counts.q_count, 2);
    assert_eq!(produced.artifact.counts.quotient_class_count, 1);
    assert_eq!(produced.artifact.counts.fail_no_new, 2);
}

#[test]
fn third_oracle_golden_vectors_match_codec_producer_and_verifier() {
    let cases=[
  (replay(&[(0,0),(5,0),(5,-1)]),"485852464c56313a524f4f542d53454d414e5449433a563100010001000100010001000300000000000500ffff0105000000010001030000000000","1e0ee42712858b73e46fcfe603a6400bf29676ccb5d5921fbdb52225b26d6167",(0,0,0,[0,0,0,0])),
  (replay(&[(0,0),(0,1),(-1,5),(1,0),(4,0),(1,4),(4,2),(5,0),(-2,-2)]),"485852464c56313a524f4f542d53454d414e5449433a5631000100010001000100010009fefffeff00ffff05000100000000000000010001010000000001000400010400000000040002000105000000000101090000000001","499d226e46bd418ab44e42819229b09b8ed47f31047856a059445586c73e5b0a",(2,2,1,[2,0,0,0])),
 ];
    for (state, preimage, digest, expected) in cases {
        let produced = emit(&state);
        assert_eq!(
            hex(&root_semantic_preimage_v1(&produced.artifact.root)),
            preimage
        );
        assert_eq!(hex(&produced.artifact.root_semantic_sha256), digest);
        let c = produced.artifact.counts;
        assert_eq!(
            (
                c.t_count,
                c.q_count,
                c.quotient_class_count,
                [
                    c.fail_no_new,
                    c.fail_defender_first,
                    c.fail_loose_0,
                    c.fail_loose_1
                ]
            ),
            expected
        );
        let token = ReachableRootV1::from_trusted_engine_state(&state).unwrap();
        let checked = verify_refute_leaf_exact_v1(
            &state,
            &token,
            &produced.bytes,
            OfflinePolicyV1::default(),
        )
        .unwrap();
        assert_eq!(checked.artifact.counts, c);
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn three_real_corpus_width_exhaust_leaves_round_trip() {
    for (id, n) in [
        ("0hz3hty", 5usize),
        ("0l4291i_live", 5usize),
        ("8is963b", 7usize),
    ] {
        let state = corpus_prefix(id, n);
        let cap = 200;
        let caps = SolveCaps {
            node_cap: cap,
            tt_bytes_cap: 256 << 10,
            semantic_horizon: u32::MAX,
        };
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        let search = solver.solve_goal(&state, &caps, SolveGoal::Win);
        assert_eq!(search.status, ProofStatus::Unknown);
        assert!(search.stats.expansions < cap);
        assert!(solver
            .last_wide_root_numbers()
            .is_some_and(|(_, dn)| dn == 0));
        let token = ReachableRootV1::from_trusted_engine_state(&state).unwrap();
        let result = produce_refute_leaf_exact_v1_after_search(
            RefuteLeafModeV1::Emit,
            &state,
            &token,
            OfflinePolicyV1::default(),
            SearchProfileV1::completed_natural_width_exhaust(),
            search.stats.expansions,
            cap,
        );
        assert!(
            matches!(result, ProduceResultV1::Emitted(_)),
            "{id}/{n}: {result:?}"
        );
    }
}

#[test]
fn mandatory_r2_1_earlier_constructor_and_typed_gate_fixtures() {
    // Existing reachable corpus roots exercise the hostile earlier constructors.
    let forced = corpus_prefix("8is963b", 91);
    let (_, completion, tactical, tight) =
        producer_counts_without_earlier_for_test(&forced, OfflinePolicyV1::default()).unwrap();
    assert_eq!((completion, tactical, tight), (0, 0, 0));
    let own = corpus_prefix("8is963b", 13);
    for (state, reason) in [(&forced, "ForcedLoss_A"), (&own, "OwnWinNow_A")] {
        let token = ReachableRootV1::from_trusted_engine_state(state).unwrap();
        let result = produce_refute_leaf_exact_v1_after_search(
            RefuteLeafModeV1::Emit,
            state,
            &token,
            OfflinePolicyV1::default(),
            SearchProfileV1::completed_natural_width_exhaust(),
            1,
            200,
        );
        assert!(
            matches!(result,ProduceResultV1::Ineligible(crate::tss_refute_produce::LeafIneligibilityV1::NotRefuteLeafExactSemantic(r)) if r==reason)
        );
        let bytes = forged(state, LeafCountsV1::default());
        assert!(
            matches!(verify_refute_leaf_exact_v1(state,&token,&bytes,OfflinePolicyV1::default()),Err(VerifyRejectionV1::Semantic(r)) if r==reason)
        );
    }
    // A terminal claimant root cannot enter the cut.
    let terminal = replay(&[
        (0, 0),
        (0, 2),
        (2, 2),
        (1, 0),
        (2, 0),
        (4, 2),
        (6, 2),
        (3, 0),
        (4, 0),
        (8, 2),
        (10, 2),
        (5, 0),
    ]);
    assert!(terminal.is_terminal());
    let token = ReachableRootV1::from_trusted_engine_state(&terminal).unwrap();
    let result = produce_refute_leaf_exact_v1_after_search(
        RefuteLeafModeV1::Emit,
        &terminal,
        &token,
        OfflinePolicyV1::default(),
        SearchProfileV1::completed_natural_width_exhaust(),
        1,
        200,
    );
    assert!(matches!(
        result,
        ProduceResultV1::Ineligible(
            crate::tss_refute_produce::LeafIneligibilityV1::NotRefuteLeafExactSemantic(
                "ClaimantTerminal"
            )
        )
    ));
    // The semantically eligible Q2 root rejects wrong profile and equality cap.
    let eligible = replay(&[
        (0, 0),
        (0, 1),
        (-1, 5),
        (1, 0),
        (4, 0),
        (1, 4),
        (4, 2),
        (5, 0),
        (-2, -2),
    ]);
    let token = ReachableRootV1::from_trusted_engine_state(&eligible).unwrap();
    let wrong = produce_refute_leaf_exact_v1_after_search(
        RefuteLeafModeV1::Emit,
        &eligible,
        &token,
        OfflinePolicyV1::default(),
        SearchProfileV1::Other,
        1,
        200,
    );
    assert!(matches!(
        wrong,
        ProduceResultV1::Ineligible(
            crate::tss_refute_produce::LeafIneligibilityV1::IneligibleLeafProfile
        )
    ));
    let cap = produce_refute_leaf_exact_v1_after_search(
        RefuteLeafModeV1::Emit,
        &eligible,
        &token,
        OfflinePolicyV1::default(),
        SearchProfileV1::completed_natural_width_exhaust(),
        200,
        200,
    );
    assert!(matches!(
        cap,
        ProduceResultV1::Ineligible(
            crate::tss_refute_produce::LeafIneligibilityV1::IneligibleNodeCap
        )
    ));
    let off = produce_refute_leaf_exact_v1_after_search(
        RefuteLeafModeV1::Off,
        &eligible,
        &token,
        OfflinePolicyV1 {
            q_count: 0,
            ..OfflinePolicyV1::default()
        },
        SearchProfileV1::Other,
        u64::MAX,
        0,
    );
    assert_eq!(off, ProduceResultV1::Disabled);
}

fn resign_payload(bytes: &mut [u8]) {
    let len = bytes.len();
    let payload_len = bytes[len - 33 - 8] as usize;
    let start = len - 32 - payload_len;
    let sum = sha256(&bytes[start..len - 32]);
    bytes[len - 32..].copy_from_slice(&sum);
}

#[test]
fn strict_codec_and_semantic_mutations_all_reject() {
    let state = replay(&[
        (0, 0),
        (0, 1),
        (-1, 5),
        (1, 0),
        (4, 0),
        (1, 4),
        (4, 2),
        (5, 0),
        (-2, -2),
    ]);
    let token = ReachableRootV1::from_trusted_engine_state(&state).unwrap();
    let good = emit(&state).bytes;
    let reject = |b: &[u8]| {
        assert!(verify_refute_leaf_exact_v1(&state, &token, b, OfflinePolicyV1::default()).is_err())
    };
    let mut x = good.clone();
    x[0] ^= 1;
    reject(&x);
    let mut x = good.clone();
    x[8] = 2;
    reject(&x);
    let mut x = good.clone();
    x.push(0);
    reject(&x);
    let mut x = good.clone();
    *x.last_mut().unwrap() ^= 1;
    reject(&x);
    let mut x = good.clone();
    x[18] = 0x89;
    x.insert(19, 0);
    reject(&x); // redundant root-count uvar
    let mut x = good.clone();
    x[23] ^= 1;
    reject(&x); // owner/root binding
    let mut x = good.clone();
    x[65] = 2;
    reject(&x);
    let mut x = good.clone();
    x[70] = 1;
    reject(&x);
    let mut x = good.clone();
    x[71] ^= 1;
    reject(&x);
    let mut x = good.clone();
    x[72] ^= 1;
    reject(&x);
    let payload = good.len() - 40;
    for index in 0..8 {
        let mut x = good.clone();
        x[payload + index] ^= if index == 0 { 1 } else { 3 };
        resign_payload(&mut x);
        reject(&x);
    }
    let mut swapped = good.clone();
    for k in 0..5 {
        swapped.swap(19 + k, 24 + k)
    }
    reject(&swapped);
    let mut low = OfflinePolicyV1::default();
    low.q_count = 1;
    assert!(matches!(
        verify_refute_leaf_exact_v1(&state, &token, &good, low),
        Err(VerifyRejectionV1::UnsupportedPolicyBudget("Q"))
    ));
    let other = replay(&[(0, 0), (5, 0), (5, -1)]);
    let other_token = ReachableRootV1::from_trusted_engine_state(&other).unwrap();
    assert!(
        verify_refute_leaf_exact_v1(&other, &other_token, &good, OfflinePolicyV1::default())
            .is_err()
    );
}

#[test]
fn one_sided_counter_faults_and_weak_promotion_are_caught() {
    let state = replay(&[
        (0, 0),
        (0, 1),
        (-1, 5),
        (1, 0),
        (4, 0),
        (1, 4),
        (4, 2),
        (5, 0),
        (-2, -2),
    ]);
    let token = ReachableRootV1::from_trusted_engine_state(&state).unwrap();
    let good = emit(&state).bytes;
    let payload = good.len() - 40;
    for (index, value) in [(2usize, 1u8), (3, 2), (4, 1)] {
        let mut x = good.clone();
        x[payload + index] = value;
        resign_payload(&mut x);
        assert!(
            verify_refute_leaf_exact_v1(&state, &token, &x, OfflinePolicyV1::default()).is_err()
        );
    }
    // Changing one claimant stone creates count-one-through-a G1 partners. A
    // producer with only that promotion omitted reports the old Q2 telemetry;
    // the independent verifier rejects it.
    let weak = replay(&[
        (0, 0),
        (2, 1),
        (-1, 5),
        (1, 0),
        (4, 0),
        (1, 4),
        (4, 2),
        (5, 0),
        (-2, -2),
    ]);
    let Ok((counts, _, _, _)) = producer_counts_for_test(&weak, OfflinePolicyV1::default()) else {
        panic!("weak fixture")
    };
    assert!(counts.q_count > 2);
    let weak_token = ReachableRootV1::from_trusted_engine_state(&weak).unwrap();
    let bad = forged(
        &weak,
        LeafCountsV1 {
            t_count: 2,
            q_count: 2,
            quotient_class_count: 1,
            fail_no_new: 2,
            ..LeafCountsV1::default()
        },
    );
    assert!(
        verify_refute_leaf_exact_v1(&weak, &weak_token, &bad, OfflinePolicyV1::default()).is_err()
    );
}

#[test]
fn verifier_source_firewall_denylist() {
    let source = include_str!("tss_refute_verify.rs");
    for forbidden in [
        "use crate::tss_solver",
        "use crate::tss_verify::",
        "threats_shared::",
        ".windows()",
        "WindowStore",
    ] {
        assert!(
            !source.contains(forbidden),
            "forbidden verifier dependency: {forbidden}"
        );
    }
}

fn d6(c: (i16, i16), g: usize) -> (i16, i16) {
    let (q, r) = (c.0 as i32, c.1 as i32);
    let base = if g < 6 { (q, r) } else { (r, q) };
    let (q, r) = base;
    let out = match g % 6 {
        0 => (q, r),
        1 => (-r, q + r),
        2 => (-q - r, q),
        3 => (-q, -r),
        4 => (r, -q - r),
        _ => (q + r, -q),
    };
    (out.0 as i16, out.1 as i16)
}

#[test]
fn d6_closure_round_trips_all_images_and_binds_raw_root() {
    let history = [
        (0, 0),
        (0, 1),
        (-1, 5),
        (1, 0),
        (4, 0),
        (1, 4),
        (4, 2),
        (5, 0),
        (-2, -2),
    ];
    let original = emit(&replay(&history));
    for g in 0..12 {
        let transformed = history.iter().map(|&c| d6(c, g)).collect::<Vec<_>>();
        let state = replay(&transformed);
        let produced = emit(&state);
        assert_eq!(produced.artifact.counts, original.artifact.counts);
        if transformed != history {
            let token = ReachableRootV1::from_trusted_engine_state(&state).unwrap();
            assert!(verify_refute_leaf_exact_v1(
                &state,
                &token,
                &original.bytes,
                OfflinePolicyV1::default()
            )
            .is_err());
        }
    }
}

#[test]
fn hostile_external_budgets_fail_before_unbounded_work() {
    let state = replay(&[
        (0, 0),
        (0, 1),
        (-1, 5),
        (1, 0),
        (4, 0),
        (1, 4),
        (4, 2),
        (5, 0),
        (-2, -2),
    ]);
    let token = ReachableRootV1::from_trusted_engine_state(&state).unwrap();
    let bytes = emit(&state).bytes;
    let mut p = OfflinePolicyV1::default();
    p.windows = 1;
    assert!(matches!(
        verify_refute_leaf_exact_v1(&state, &token, &bytes, p),
        Err(VerifyRejectionV1::UnsupportedPolicyBudget("windows"))
    ));
    let mut p = OfflinePolicyV1::default();
    p.wire_bytes = (bytes.len() - 1) as u64;
    assert!(matches!(
        verify_refute_leaf_exact_v1(&state, &token, &bytes, p),
        Err(VerifyRejectionV1::UnsupportedPolicyBudget("wire bytes"))
    ));
    let mut p = OfflinePolicyV1::default();
    p.state_bytes = 1;
    assert!(matches!(
        verify_refute_leaf_exact_v1(&state, &token, &bytes, p),
        Err(VerifyRejectionV1::UnsupportedPolicyBudget(
            "root allocation"
        ))
    ));
}

fn percentile(v: &[u128], fraction: f64) -> u128 {
    let mut x = v.to_vec();
    x.sort_unstable();
    x[((x.len() - 1) as f64 * fraction).ceil() as usize]
}
#[test]
#[ignore = "economics measurement; serialized release-only"]
fn refute_leaf_economics_measurement() {
    let roots = [
        ("0hz3hty/p5", corpus_prefix("0hz3hty", 5)),
        ("0l4291i_live/p5", corpus_prefix("0l4291i_live", 5)),
        ("8is963b/p7", corpus_prefix("8is963b", 7)),
    ];
    let mut aggregate_emit = Vec::new();
    let mut aggregate_verify = Vec::new();
    let mut aggregate_search = Vec::new();
    let mut aggregate_bytes = 0usize;
    let mut aggregate_baseline = 0usize;
    for (id, state) in roots {
        let token = ReachableRootV1::from_trusted_engine_state(&state).unwrap();
        let mut emit_ns = Vec::new();
        let mut verify_ns = Vec::new();
        let mut search_ns = Vec::new();
        let mut bytes = 0;
        let mut baseline = 0;
        let mut work = crate::tss_refute_verify::RefuteWorkV1::default();
        for _ in 0..30 {
            let started = std::time::Instant::now();
            let result = produce_refute_leaf_exact_v1_after_search(
                RefuteLeafModeV1::Emit,
                &state,
                &token,
                OfflinePolicyV1::default(),
                SearchProfileV1::completed_natural_width_exhaust(),
                1,
                200,
            );
            emit_ns.push(started.elapsed().as_nanos());
            let ProduceResultV1::Emitted(p) = result else {
                panic!("econ emit")
            };
            bytes = p.bytes.len();
            baseline = root_plus_empty_set_baseline_bytes(&p.artifact.root);
            let started = std::time::Instant::now();
            let checked =
                verify_refute_leaf_exact_v1(&state, &token, &p.bytes, OfflinePolicyV1::default())
                    .unwrap();
            verify_ns.push(started.elapsed().as_nanos());
            work = checked.work;
            let caps = SolveCaps {
                node_cap: 200,
                tt_bytes_cap: 256 << 10,
                semantic_horizon: u32::MAX,
            };
            let mut solver = TssSolver::default();
            solver.set_width_options(WidthOptions::vcf_pair_complete());
            let started = std::time::Instant::now();
            let search = solver.solve_goal(&state, &caps, SolveGoal::Win);
            search_ns.push(started.elapsed().as_nanos());
            assert_eq!(search.status, ProofStatus::Unknown);
        }
        println!("REFUTE_ECON id={id} reps=30 bytes={bytes} baseline={baseline} ratio={:.6} emit_median_us={:.3} emit_p95_us={:.3} emit_max_us={:.3} verify_median_us={:.3} verify_p95_us={:.3} verify_max_us={:.3} search_median_us={:.3} search_p95_us={:.3} search_max_us={:.3} work_windows={} work_q={} work_threat={} work_pair={} work_transversal={} state_bytes={} heap_bytes={}",bytes as f64/baseline as f64,percentile(&emit_ns,0.5)as f64/1e3,percentile(&emit_ns,0.95)as f64/1e3,*emit_ns.iter().max().unwrap()as f64/1e3,percentile(&verify_ns,0.5)as f64/1e3,percentile(&verify_ns,0.95)as f64/1e3,*verify_ns.iter().max().unwrap()as f64/1e3,percentile(&search_ns,0.5)as f64/1e3,percentile(&search_ns,0.95)as f64/1e3,*search_ns.iter().max().unwrap()as f64/1e3,work.windows,work.q,work.threat_memberships,work.pair_ops,work.transversal_ops,work.retained_state_bytes,work.estimated_heap_bytes);
        aggregate_emit.extend(emit_ns);
        aggregate_verify.extend(verify_ns);
        aggregate_search.extend(search_ns);
        aggregate_bytes += bytes;
        aggregate_baseline += baseline;
    }
    println!("REFUTE_ECON_AGG roots=3 reps_per_root=30 bytes={} baseline={} ratio={:.6} emit_total_ms={:.3} verify_total_ms={:.3} search_total_ms={:.3} emit_p95_us={:.3} emit_max_us={:.3} verify_p95_us={:.3} verify_max_us={:.3} search_p95_us={:.3} search_max_us={:.3}",aggregate_bytes,aggregate_baseline,aggregate_bytes as f64/aggregate_baseline as f64,aggregate_emit.iter().sum::<u128>()as f64/1e6,aggregate_verify.iter().sum::<u128>()as f64/1e6,aggregate_search.iter().sum::<u128>()as f64/1e6,percentile(&aggregate_emit,0.95)as f64/1e3,*aggregate_emit.iter().max().unwrap()as f64/1e3,percentile(&aggregate_verify,0.95)as f64/1e3,*aggregate_verify.iter().max().unwrap()as f64/1e3,percentile(&aggregate_search,0.95)as f64/1e3,*aggregate_search.iter().max().unwrap()as f64/1e3);
}

#[test]
#[ignore = "fixture discovery"]
fn focused_q1_seed_discovery() {
    for extra in [(-2, -2), (2, 0), (3, 0), (6, 0), (-1, 0)] {
        let state = replay(&[
            (0, 0),
            (0, 1),
            (-1, 5),
            (1, 0),
            (4, 0),
            (1, 4),
            (4, 2),
            (5, 0),
            extra,
        ]);
        println!(
            "FOCUSED_Q1 extra={extra:?} {:?}",
            producer_counts_for_test(&state, OfflinePolicyV1::default())
        );
    }
}

#[test]
#[ignore = "discovery/economics harness; serialized release-only"]
fn refute_leaf_prefix_discovery() {
    let mut unique = std::collections::BTreeSet::new();
    for position in load_corpus() {
        let history = position
            .state
            .placement_history()
            .iter()
            .map(|r| r.coord)
            .collect::<Vec<_>>();
        let mut state = HexoState::new();
        for (index, coord) in history.into_iter().enumerate() {
            apply_placement(&mut state, Placement { coord }).unwrap();
            if state.phase() != TurnPhase::FirstStone || state.is_terminal() {
                continue;
            }
            let Ok((counts, c, x, t)) =
                producer_counts_for_test(&state, OfflinePolicyV1::default())
            else {
                continue;
            };
            if c != 0 || x != 0 || t != 0 {
                continue;
            }
            let key = state
                .snapshot()
                .placements()
                .iter()
                .map(|c| (c.q, c.r))
                .collect::<Vec<_>>();
            if !unique.insert(key) {
                continue;
            }
            let cap = 200u64;
            let caps = SolveCaps {
                node_cap: cap,
                tt_bytes_cap: 256 << 10,
                semantic_horizon: u32::MAX,
            };
            let mut solver = TssSolver::default();
            solver.set_width_options(WidthOptions::vcf_pair_complete());
            let result = solver.solve_goal(&state, &caps, SolveGoal::Win);
            let numbers = solver.last_wide_root_numbers();
            println!("REFUTE_PREFIX source={} prefix={} t={} q={} classes={} status={:?} expansions={} root_numbers={:?} natural={}",position.id,index+1,counts.t_count,counts.q_count,counts.quotient_class_count,result.status,result.stats.expansions,numbers,result.status==ProofStatus::Unknown&&result.stats.expansions<cap&&numbers.is_some_and(|(_,dn)|dn==0));
            if unique.len() >= 20 {
                return;
            }
        }
    }
}

#[test]
#[ignore = "fixture discovery"]
fn earlier_constructor_discovery() {
    for position in load_corpus() {
        let history = position
            .state
            .placement_history()
            .iter()
            .map(|r| r.coord)
            .collect::<Vec<_>>();
        let mut state = HexoState::new();
        for (index, coord) in history.into_iter().enumerate() {
            if apply_placement(&mut state, Placement { coord }).is_err() {
                break;
            }
            if state.phase() != TurnPhase::FirstStone {
                continue;
            }
            if let Err(error) = producer_counts_for_test(&state, OfflinePolicyV1::default()) {
                if error.contains("OwnWinNow_A") || error.contains("ForcedLoss_A") {
                    println!(
                        "EARLIER source={} prefix={} error={}",
                        position.id,
                        index + 1,
                        error
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "fixture discovery"]
fn empty_admitted_earlier_constructor_discovery() {
    for position in load_corpus() {
        let history = position
            .state
            .placement_history()
            .iter()
            .map(|r| r.coord)
            .collect::<Vec<_>>();
        let mut state = HexoState::new();
        for (index, coord) in history.into_iter().enumerate() {
            if apply_placement(&mut state, Placement { coord }).is_err() {
                break;
            }
            if state.phase() != TurnPhase::FirstStone {
                continue;
            }
            let enforced = producer_counts_for_test(&state, OfflinePolicyV1::default());
            let earlier = matches!(&enforced, Err(e) if e.contains("OwnWinNow_A") || e.contains("ForcedLoss_A"));
            if !earlier {
                continue;
            }
            if let Ok((counts, completion, tactical, tight)) =
                producer_counts_without_earlier_for_test(&state, OfflinePolicyV1::default())
            {
                if completion == 0 && tactical == 0 && tight == 0 {
                    println!(
                        "EMPTY_EARLIER source={} prefix={} enforced={:?} t={} q={}",
                        position.id,
                        index + 1,
                        enforced,
                        counts.t_count,
                        counts.q_count
                    );
                }
            }
        }
    }
}
