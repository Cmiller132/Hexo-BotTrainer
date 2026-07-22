# SOLVER_NOTES — current measured state of the TSS solver

Reset 2026-07-22 (full history: `docs/archive/SOLVER_NOTES_2026-07-21_full.md`
and git). Convention going forward: append dated entries per finding; labels
MEASURED (verified data) / CODE-FACT (cite lines) / HYPOTHESIS / RETRACTED
(strike, don't delete, from this reset onward). Entry point and laws:
`docs/HANDOFF.md`.

## 1. Production profile (main_4 line; trainer currently stopped)

`tss_solver_mode=3` (WIN+LOSS), node cap 750, J2near OFF (owner reversal
2026-07-21 late: the same-evening cap-1000/J2near-on point was judged not
worth the wall — J2near adds zero decided rows at caps ≤1,000; cap 750 is
the measured strictly-better post-fold point, +89 decisions under the old
wall — see the §5 entries), 256 KiB TT, unbounded horizon
(`semantic_horizon=u32::MAX`), wide (`vcf_pair_complete`; the first-class
`tss_solver_j2near` key stays wired default-off over
`configure_leaf_profile`: wide + lazy frontier + interior census gate
[inert when unbounded]), `dual_pass=true`, `all_leaves=true` (park
5000 ms emergency-only), async 12/24, root+interior guards. G2 / ordering
hints / loss_reserve / fragments / zones OFF.

## 2. Measured facts the current work stands on

- **Generation dominates wall — now 1.72x cheaper; residual NAMED.**
  Winning proof path ≈ 0.0003%. Candidate-gen rounds 1+2 folded
  (`2a1bdf97`, bit-identical). **Exhaustive cap-750 decomposition
  (2026-07-21, `e08d9da0` on claude/j2near-profile;
  REPORT_TSS_BOTTLENECK_750.md; buckets sum to 100.00%, node identity
  730,143 exact, 2.03% profiling overhead):** state make/unmake
  **24.10%**, window analysis/gate build **21.43%**, attacker generation
  proper **20.90%**, defender-pair plan construction **15.77%**, PN
  select/backprop/stage bookkeeping **7.71%**, second-cand regen 6.13%,
  certificate 1.31%, TT 0.31%, setup 0.15%, other 1.46%. The old 32.85%
  unattributed residual = state make/unmake + PN bookkeeping (31.80%).
  Combined attacker+window+second-cand path = 48.46%. **Outcome
  attribution: 641 UNKNOWN-at-cap rows consume 65.46% of battery wall**
  (perfect-early-stop ceiling 30.08 s of 45.96 s); frozen-pn subset (259
  rows, pn identical 500→750) costs 9.07% in the final increment alone.
  Defender-plan is the only bucket whose share AND per-node cost grow
  with cap (caveat: cap-500 instrumented pass had 15.1% overhead, so
  cross-cap ratios are indicative). Deep F19 = attacker 35.97%, defender
  30.56%, TT 2.01% (older split). Historic pre-fold shape (A_OR_GEN
  60.5%, D_FORCED_GEN 20.4%; P7 1.42x `2c262e10`) superseded.
- **TT at cap 500 ≈ overhead but cheap** (hit/entry ≈ 0.01; measured
  probe/insert wall share 0.433% — below any removal payoff, TT-min
  KILLED in r2). Deep memory resident = WidePnSearch arena +
  `by_position` (no eviction; admission-rejection only) — TT replacement
  policy cannot affect deep solves (`7c4c04f1`).
- **Grind class** = 73.5% of Unknown wall; at 50k: ~23% provable WIN
  (p50 ~1.7k nodes), ~39% width-exhaust (~2k self-terminate), ~38%
  cap-bound.
- **Loss side**: dual_pass (leftover-budget second pass) = +288 dev
  losses, throughput-neutral; 58 atlas losses need ≥512 dedicated nodes —
  unreachable at cap 500 under any allocation.
- **Ordering family dead**: miss cost 1.8%; ep90 policy priors and threat
  statics both measurably harm df-pn. Bar = proof-participation signals
  only (probe-seeding qualifies; lane in flight).
- **Horizon closed**: unbounded+cap beats h16 on coverage (+26%) and
  strength (+60 Elo h2h); extra wins sit at cert depth 17–22.
- **Reference standing** (quiet 2 GiB gate + matched host): 14/14 corpus
  WINs + 2 LOSSes certified, 0 false claims; 0l4291i = WIN at 512 MiB /
  1,913,955 nodes (the old "memory ceiling" was a 1M rung-cap artifact)
  vs pdspn 256 MB / 1,058 seeded nodes. Real gaps: informed-node
  efficiency (~1,800x on 0l), easy-win latency (~2x), no certified
  refutations (planned).
- **Width boundary**: 3 atlas wins provably outside `vcf_pair_complete`;
  mechanism = free second stone after a forcing first stone (J2near
  candidate, ~1.04x accepted-child multiplier on eligible roots,
  0/248 grinds eligible; A/B mandatory).
- **Rank-two defender boundary**: threat families rank ≤2 ⇒ ≤4 minimum
  cover pairs (exhaustive 33,861-family model check + 229 real pairs,
  0 violations). Stateless plan construction = the D_FORCED_GEN lever.
- **Shallow exact layer**: exact h=2 (win-in-own-turn) and h≤4
  (unanswerable-family forced loss) predicates match engine diagnostics
  on 6,294 roots, 0 mismatches; h2 fires 102 / h4 fires 146 there. Lean
  formalization in flight; h≤6/h≤8 extension lane in flight.
- **Board is unbounded** (sparse axial Z²) ⇒ no global fill deadline
  exists; the parametric deadline-dismissal ladder is NO-GO (`72f68ced`).

## 3. Traps (verified the hard way)

- Cargo from the wrong cwd resolves the main checkout's old crate
  (~68 tests) — always `cd` into the worktree in the same command; expect
  ~257 tests (python feature) / ~172 (plain); serialize (`--test-threads=1`).
- Env-gated features (`TSS_SHARED_FRAGMENTS`, `TSS_CAP_RESUME`,
  `TSS_THRESHOLD_DELTA`, `TSS_TT_REPLACEMENT`, `TSS_K_REPLY_CONSUME`,
  census gate) follow the process environment — every arm must enumerate
  and assert its env gates (harness does this since `41b0d23d`).
- Probe/stats paths that construct fresh solvers describe COLD state by
  construction; batch APIs may omit stats keys entirely — check emission
  before trusting a zero.
- Bench must record the engine binary sha; debug-vs-release .so confusion
  invalidated a round once.
- The `:5979` terminal-Refuted arm is UNREACHABLE via normal producers;
  a debug_assert tripwire + regression test guard it (`8271f696`).

## 4. Open probes / in-flight (details: HANDOFF §7)

LIVE: j2near (free-tempo widening impl + witness gate + matched-cap A/B) ·
horizon-h10 (research-first per owner ruling: translation-quotient theorem
attempt + h≤8 bite on all rows + port spec [shelf doc]). DONE, awaiting
owner/next-step: refute-cert v1 design (GO-conditional; hostile review
round is the required next gate) · triage phase 1 (classification; Phase B
sub-root telemetry optional). Queue: sibling-transplant shadow · G2
R-item amendments + fixed step-zero screen (deep/labeling scope only) ·
CapResumeSession promotion · GPU bench close-out.

## 5. Iteration log (from this reset)

- 2026-07-22: document reset to current-facts form (owner-directed
  de-bloat); full history archived.
- 2026-07-21 (quiet-host reference round). MEASURED: official 2 GiB gate
  rerun — 14/14 WINs + 2 LOSSes certified, 0 failures, 416 s total
  (load fingerprint: one background Lean lake build); 0l4291i WIN at
  1,879,612 nodes / 1.73 GB peak TT / 172 s in-rung (~274 s with
  fresh-ladder rungs). Strix quiet battery (120 s / 50 M / 256 MB;
  `%TEMP%\hexo-strix-clone\battery_*.csv`): pdspn 19/19 (0l: 1,058
  level-1 nodes, 260.9 s), idtt 16/19, dfpn 16/19, their deployed config
  5/19. **REFRAME:** 0l is wall-PARITY (274 s vs 261 s) despite the
  1,777x node ratio — pdspn level-1 nodes each run bounded probes, so
  the "~1,800x informed-node gap" is node-accounting, not speed. Real
  remaining gaps: 94gnnol-class disproof (their No 20.7 s vs our 1M-node
  cap-bound Unknown), idtt easy-win latency 2–7x (fresh-ladder tax +
  in-wall cert+verify), and no refutation artifacts (mvp2lvc
  width-exhausts at 17,957 nodes / 1.8 s ≈ their No walls, uncertified).
- 2026-07-21 (candidate-gen fold, `63b34cbb`/`2a1bdf97`). MEASURED:
  rounds 1+2 bit-identical — production battery solve wall 49.96 s →
  29.0 s median (~1.72x), deep F19 ~1.58x; window-generation memo at
  32,768 direct-mapped slots, 46.2% hits (58.3% on F19). Kills by
  measurement: TT-min profile (0.433% tax < 3% bar), per-solve setup
  reuse (0.186% < 10% bar), 2-way associativity, defender-side memo.
  CODE-FACT: rung-1 stateless rank-two defender plans now carry the
  kernel-checked licensing theorem `forcedB2PairQuotient`
  (tss-lean `f4315e6`, R-SH3). Gates: suite 218/0/39; 6,443-row identity
  digest `a8c6f3ca3ba55827` + SHA `02CD…FDB` (independently reproduced
  in r2 step 0); Stage-0 WSL pytest golden 33 passed.
- 2026-07-21 (triage phase 1 — Unknown-type classification; artifacts
  `%TEMP%\triage\`). MEASURED on 248 labeled grinds (57 provable / 97
  width-exhaust / 94 cap-bound), 11-cap ladder 100–5,000 at 256 KiB:
  root-endpoint pn/dn does NOT separate classes at N≤500 (precision ≈
  base rate). Classes separate by termination at 2–5k: provable 44/57
  prove ≤5k (median 1,293 nodes), exhaust 64/97 self-exit (median
  1,294), cap-bound 0/94 move — root pn frozen at 34 from cap 100→5,000.
  Stagnation feature: real signal (69% of cap-bound frozen) but 50–58%
  precision with 20–27 provable casualties — root granularity
  insufficient; Phase B = sub-root telemetry if pursued. Owner rulings:
  classification now, re-allocation deferred; horizon work
  research-first (no incremental-horizon/ladder consumption builds).
- 2026-07-21. MEASURED: production cert-depth distribution
  (dualpass_adoption records, 1,212 certs): ≤8 = 42.5%, ≤12 = 59.3%
  (plies-vs-placements units unverified). Shelved sizing datum for any
  future horizon-consumption decision.
- 2026-07-21 (J2near fold, `94466ead`/`f5a5c5f0`; REPORT_J2NEAR.md).
  MEASURED: free-tempo second-stone widening landed DEFAULT-OFF. All 3
  atlas witnesses flip UNKNOWN → strictly verified WIN (predicted
  candidates present; child counts 19→39, 19→39, 8→12); five verified
  non-seed upgrades (4 human @cap 2k, 1 grind @cap 50k); zero verifier
  failures. Default-on BLOCKED by preregistered gates: one cap-500 human
  WIN → UNKNOWN (widened branching consumed its budget), puzzle cap-2k
  cohort median wall 1.074 > 1.05, per-row p95 wall > 1.20. Disposition:
  `vcf_pair_j2near` / `TSS_VCF_J2NEAR` = targeted recovery tier; a future
  default-on attempt needs an ordering/budget-reserve policy that
  restores the downgraded proof + fresh preregistered A/B. Flag-off:
  6,443/6,443 archived identity; merged-tree gates 220/0/43 suite +
  digest a8c6f3ca exact + witness node counts lane-identical.
- 2026-07-21 (cap-headroom Pareto, `cac5ef4a`; REPORT_J2NEAR_CAP.md).
  MEASURED: wall-matched grid {500,640,750,860,1000}×{J2near off,on},
  6,443 rows, 3 alternating reps. **Cap 750 / J2near-off adopted**
  (`hexfield_eq_main_4.toml` retuned 500→750): archive-verdict superset
  (all 1,212 preserved), **+89 strictly verified decisions** (57 W/32 L),
  median solve wall 45.956 s [44.689–46.730] < old 49.96 s. Strict
  frontier also contains cap 640 (+61 @ 38.9 s). J2near-on dominated at
  every tested cap: downgrade row `human_41e2eece..._p11` restored only
  at ≥860 (over wall), its five known upgrades need ≥1,111 nodes (beyond
  cap 1,000) — retest J2near only when headroom passes ~cap 1,100. Zero
  W/L flips, zero verifier failures, all ten arms.
- 2026-07-21 (J2near production wiring + cap 1000, `80e3ea18`;
  REPORT_J2NEAR_WIRE.md). **Cap 1000 / J2near-ON adopted** — owner
  completeness ruling (training target = most complete verified solution
  set; wall relaxed), superseding the same-day cap-750/J2near-off
  disposition. `tss_solver_j2near` is now a first-class rollout key
  (python `SelfplayConfig` → divergence map → Rust whitelist →
  `Divergences::solver_j2near_enabled()` = key && mode>0 → inline /
  root-guard / async-worker solvers via `set_leaf_j2near`); the
  `TSS_VCF_J2NEAR` env path DELETED (single-path hygiene; harnesses use
  `WidthOptions` directly). Safety at cap 1000 already MEASURED in the
  cap-headroom grid: both arms 1,362 decided (+150 vs old archive), zero
  downgrades/flips; on-arm wall 60.484 s [59.021–60.705]. CODE-FACT: at
  cap 1000 the on-arm decides no row the off-arm doesn't — J2near-unique
  proofs need 1,111–1,783 nodes; the cap curve above 1000 is being
  measured on claude/j2near-profile. Gates: 4-case seam test, j2near unit
  set, python-feature suite 221/0/43 + release lib 136/0/42, all repeated
  in an independent orchestrator rerun from a cold build.
- 2026-07-21 late (owner REVERSAL of the same-evening completeness
  ruling): **cap back to 750, J2near OFF** ("it seems to not be worth
  it") — consistent with the measured data: J2near-on decides zero extra
  rows at any cap ≤1,000 and cap 1,000 costs 60.5 s vs 45.96 s at 750 for
  +61 rows. The `tss_solver_j2near` key and wiring REMAIN (default-off,
  retest lever); only the TOML policy reverted. Bottleneck diagnostic
  relaunched at the cap-750/off production point.
- 2026-07-21 (bottleneck diagnostic, `e08d9da0` on claude/j2near-profile;
  REPORT_TSS_BOTTLENECK_750.md). MEASURED: exhaustive cap-750 phase
  decomposition — details folded into §2. Ranked attacks: (1) fuse
  attacker+window+second-cand path (48.46%; −25% ≈ 5.6 s ≈ cap ~839),
  (2) state make/unmake (24.10%; −50% ≈ 5.5 s ≈ cap ~838), (3) guarded
  cap-bound early abandon (frozen-pn subset alone ≈ 4.2 s ≈ cap ~817;
  perfect-oracle ceiling 30.1 s ≈ cap ~1,231 — requires sub-root
  stagnation evidence + no-regression gate per triage findings). Runner-up
  defender-plan (15.77%, the with-cap-growing bucket). TT is a non-target
  (0.31%). Attack 1 is live on claude/attacker-gen-r3.
- 2026-07-21 (triage Phase B, `be4bd34a` on claude/triage-b;
  REPORT_TRIAGE_PHASE_B.md). MEASURED: sub-root trajectory telemetry
  (cfg-test, 34,616 snapshots over the 248 labeled grinds @cap 5k).
  Preregistered bar passes but only as a POCKET detector: 100% precision
  / 13.8% recall at N=100 (0 provable casualties); bar-qualified recall
  ≤20.2% through N=500, 33.0% at N=1500; broad-recall rules ~50%
  precision with 24–30 provable casualties. Thresholds in-sample —
  held-out validation required before any use. Classification family
  effectively closed; reallocation remains owner-deferred.
- 2026-07-21 (refute-cert v1 review, `d4af5aef` on claude/refute-design).
  Hostile review verdict SOUND-WITH-REQUIRED-CHANGES (R1–R8, 8 NCEs, 46
  attacks); amendment lane launched same evening. Universal-polarity core
  held.
- 2026-07-21 (horizon research, `e1180970` on claude/deadline-ladder;
  REPORT_HORIZON_H10.md). **h=10 frontier CLOSED** (proof-ready): a
  four-stone two-cover lemma (10 shapes exhausted) makes the remote
  empty-board race a position-independent LOSS constant, so
  WinWithin10 = WinWithin8 ∨ finite restricted search over the anchored
  universe + one-window halo. 78/78 tested eligible certs caught (45
  untested — Python runtime boundary, floors not rates); 2 genuinely new
  depth-10 wins with witness pairs. All-phase h≤8 bite on 6,443 rows:
  human 5.77% exact WIN + 7.39% exact LOSS; self-play 3.10%/0.52%;
  grinds 0/0. RESEARCH-FIRST per owner ruling — port spec is a shelf
  doc (PORT_SPEC_HORIZON_H8.md); no consumption builds. Next open wall:
  h=12 (both players unanchored).
