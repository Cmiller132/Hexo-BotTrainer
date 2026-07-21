# SOLVER_NOTES — living notes on the TSS deep solver

Working document for the solver-improvement iteration loop. Append a dated
entry per finding; **never delete a retracted claim — strike it and say why**.
Claims here are working knowledge, not gate results; anything promoted to a
decision goes through the usual battery + report + handoff.

Status legend: MEASURED (verified data), CODE-FACT (read from source, cite
line), HYPOTHESIS (unverified), RETRACTED, OPEN-PROBE.

**Mission (owner ruling 2026-07-20):** maximize *verified* win/loss coverage
at positions — find as many of the wins and losses that exist as possible, no
shortcuts, every improvement proven correct. **Adoption metric = win/loss
coverage on fixed position sets** (wall/yield are secondary diagnostics).
Envelope (cap 500 / park 150 ms) is tunable if data supports. Adaptive
budgets: PARKED (worth doing, not now).

---

## 1. Ground truth: the production profile (main_3 `_resume_config.toml`)

- `tss_enabled=true`, `tss_solver_mode=3` (WIN+LOSS consumption),
  `tss_solver_root_guard=true`, `tss_interior_guard=true`
- `tss_solver_node_cap=500`, `tss_solver_park=true` @ 150 ms,
  `tss_solver_async=true` (8 threads, inline16=4)
- `tss_zone=false` (retired ep35), width = wholesale wide
  (`TssSolverSlot::default()` → `configure_leaf_profile`)
- Horizon: **OWNER RULING 2026-07-20 — bounded horizon DROPPED**; unbounded +
  node cap is the profile going forward (V2 h2h = confirmation gate, running).

`configure_leaf_profile` (tss_solver.rs:833) forces: wide width
(`vcf_pair_complete`), lazy frontier ON, interior census gate ON.

## 2. MEASURED facts (V1 battery, ep90, 3,255 positions × arms, all verified,
   `deep_verify_failed=0`, 0 NO→WIN)

- Unbounded+cap vs h16-flat: +26% proven WINs (189 vs 150) at identical p50
  1.9 ms; p90 278 ms vs 76 ms. The 39 extra wins sit at cert depth 17–22.
- Unknown wall: 73.5% of unbounded-arm wall in 248 cap-bound grinds
  (nodes ≥ 500); frontier-exhausted Unknowns = 18.2%.
- Grind class structure: 99.7% of grind wall has `opp_threats ≤ 1` (median 0);
  94% cold; `net_value` median 0.02; placements p50 = 57; wall 372–1,496 ms
  each ⇒ in production every grind blows the 150 ms park window and occupies
  an async worker 4–9× the park budget for zero verdicts.
- Proven-WIN contrast: 45.5% hot, `net_value` p50 0.55, `opp_threats` p90 = 3.
- TT economics at cap 500: hit/entry ≈ 0.01 (~6 hits per ~490 entries per
  solve) in every class. The TT is near-pure overhead at production budgets.
- LOSS-goal probe: p50 16–22 µs, ~2.9% yield. WIN-goal is the expensive one.
- Calibration: net sign-disagrees with 11.6% of proven WINs; policy top-1
  matches the certified move 56.6% (prior mass 0.47).
- Human OOD: 15.0% WIN yield (2.6× selfplay).
- Ladder: dead as built — `horizon_cut` fires 0× in the wide profile.

## 3. RETRACTIONS and measurement artifacts (2026-07-20)

- ~~"Warmth ≡ cold; fragment machinery dead"~~ **RETRACTED.** The fragment
  store is env-gated: `TSS_SHARED_FRAGMENTS=1` read once in
  `TssSolver::default()` (tss_solver.rs:735); cap forced to 0 otherwise
  (tss_solver.rs:922). Neither the V1 warmth driver nor ANY trainer path sets
  it (grep: only Rust test harnesses reference it). The V1 "warm" arm was
  structurally cold; the 3,255/3,255 agreement was cold-vs-cold. Correct
  statement: **warmth has never been evaluated, in the battery or in
  production.** → OPEN-PROBE P3.
- **Instrumentation gap:** `hexfield_eq_deep_solve_batch` emits NO `stats_*`
  keys at all (its record dict ends at `zone_nodes`, search.rs:4997-5008) —
  only the single-shot probe's `with_stats` path emits them, and that path
  re-solves on a SEPARATE fresh `stats_solver` (search.rs:4889), so its
  fragment counters describe a cold solver BY CONSTRUCTION. Any
  "fragment_lookups=0 in the warmth arm" reading was summing a missing key's
  default. Two independent artifact layers on the same conclusion.
- `interior_gate_evaluations=0` across all 37k solves is NOT "gate never
  useful" — it is an applicability fact, see §4. (These counters come from
  the cold `stats_solver` re-solve, per the gap above — applicability
  analysis in §4 stands on the code, not on these counters.)
- `horizon_cut=0` in wide arms does NOT mean depth never binds in h16 (h16
  found 39 fewer wins, so it binds). HYPOTHESIS: wide enforces depth via
  `max_depth_cap` → `WidePnNode::DepthCutoff` (tss_solver.rs:~5930) before
  the semantic-horizon check at :5937 can count a `horizon_cut`. Verify
  before quoting.

## 4. CODE-FACT: why the interior census gate never fires — and what the
   horizon drop does to it

`evaluate_interior_census_gate` (tss_solver.rs:213) returns None unless ALL of:
- claimant to move, non-terminal, below root, phase ∈ {FirstStone, SecondStone};
- **`h_rem = semantic_horizon − placements_made ∈ [0, 8]`** — the census
  arithmetic only counts against a deadline ≤ 8 plies out;
- `interior_census_coordinate_safe(state, h_rem)` (no boundary effects).

Consequences:
- Unbounded horizon ⇒ `h_rem` never ∈ [0,8] ⇒ **the gate is structurally dead
  in the chosen production profile.** The horizon drop (correct on its own
  evidence) kills the only census dismissal we have as-built.
- Any census fail-fast (A5) must therefore FIRST solve the deadline problem:
  a sound local deadline in an unbounded search. Candidate sources, all
  unproven: board-fill arithmetic (game is finite — placements bound any
  win), completion-pattern reach bounds, remaining-budget arguments (unsound
  as stated — budget is not a game-semantic deadline). This is a design
  problem, not an application of R-CF3 as it stands.
- Owner's skepticism (2026-07-20) on A5 is therefore CORRECT in mechanism:
  wins that exist are mostly shallow, deadlines that certify are short, and
  the existing short-deadline gate finds zero applicable nodes at production
  shapes. The 73.5%-of-wall grind prize is real; the mechanism to claim it
  does not exist yet.

## 5. Suspicious sites (check before trusting)

- tss_solver.rs:5979-5984 — terminal node maps to `WidePnNode::Refuted` in
  BOTH match arms, including `outcome.winner == claimant`. If reachable,
  provable wins are silently dropped (completeness leak, soundness intact —
  verifier still guards verdicts). Likely unreachable (claimant wins detected
  as completions earlier); confirm with a targeted test or a debug counter.
- Probe drivers construct `TssSolver::default()` ⇒ every env-gated feature
  silently follows the harness environment, not production intent. Any future
  arm MUST enumerate the env-gated flags it believes are on and assert them.
  Known env gates: `TSS_SHARED_FRAGMENTS`, `TSS_INTERIOR_CENSUS_GATE`
  (overridden by leaf profile), `TSS_K_REPLY_CONSUME` (read per solve,
  tss_solver.rs:906).
- **Harness instrument caveats (2026-07-20 shakedown, all now gated or
  documented):**
  - The batch sweep is SERIAL (one thread, one persistent solver,
    `hexfield_eq_deep_solve_batch` plain loop) — the bench is fully
    production-threaded, the coverage sweep is not. Deliberate for now
    (verdicts don't need concurrency), but cap-raising campaigns will want
    a rayon parallel batch with per-thread solvers. Queued cargo item.
  - Verdicts are deterministic given *solver-cache state* (tree.rs
    docstring), and the persistent solver carries TT across positions in a
    batch — a verdict near the cap boundary can depend on batch
    composition/order. Full-set runs always solve in pinned sorted order
    (comparable); quick-tier SAMPLES may disagree with full runs on
    boundary positions. Compare like with like.
  - The bench builds config from the production toml whose defaults are NOT
    the adapter's (engine default h16!) — the first baseline's bench
    silently measured h16 under an unbounded-claiming manifest. Fixed:
    runner passes the FULL translated arm config, the bench echoes its
    resolved `effective_tss`, and `gate_bench_identity` (hard) compares
    echo-to-echo. Never pass only the user's config delta to a subprocess
    with different defaults.
  - Canaries must PIN every config axis they don't test (goal=win for the
    win-fixture canaries) — a goal=loss arm spuriously failed the horizon
    canary before the pin.

## 6. Probe backlog (status; owner rulings 2026-07-20 inline)

- **P1 TT economics** — TT-off/TT-tiny A/B at cap 500: verdict parity +
  per-solve wall. Owner caution: must not weaken deep/critical solves — so
  run the A/B at cap 500 AND at a raised cap to measure TT value as f(cap)
  before touching anything. Needs cargo lane. OPEN-PROBE.
- **P2 Fail-fast / deadline design — grind anatomy MEASURED (Lane C
  2026-07-20, 50k-node two-pass on all 248 grinds, 151s wall):** the grind
  class splits three ways — **57 (23%) are provable WINS** (p50 1,721
  nodes, p90 10,852: most need only ~4x the production cap), **96 (39%)
  width-exhaust** (the wide proof space empties at p50 1,963 nodes — they
  self-terminate just past cap 500, so the fail-fast prize for them is the
  gap between 500 and their exhaustion point, not the full park window),
  and **95 (38%) still cap-bound at 50k** (TT-saturation suspect at the
  256KiB harness TT — big-TT cargo lane revisits). Implication: a cap-2000
  production raise would convert roughly half the crackable grinds;
  deadline-style cuts only pay on the 95 deep tail. OPEN-PROBE, now
  data-driven.
- **P3 Warmth** — MEASURED + CLOSED at cap 500 (2026-07-20, full 6,510-solve
  paired rerun, raws/soak_warmth_frag_s{0,1}.jsonl vs cold control):
  mechanism WORKS (35 positions saved nodes; 1,281 nodes saved in the
  unbounded arm, 0 added; best single save 339→134) but **zero verdict
  flips** — savings concentrate in already-proven wins (fragments come from
  wins, so they help re-prove neighbors, not crack Unknowns). Under the
  coverage metric: no gain at cap 500; production flip NOT justified. Wall
  deltas from this run are load-confounded (ran beside V2) — node counts are
  the causal signal. Reopen only as part of P1's value-vs-cap sweep.
- **P4 Both-goal probing — MEASURED 2026-07-20, prize quantified.** The
  structure (tss_solver.rs solve_goal budget split, ~line 1113): under the
  wide profile `SolveGoal::Both` gives the WIN attempt the FULL budget and
  the dedicated loss attempt ZERO. Production mode 3 therefore detects only
  the losses the primal win-side search proves incidentally (immediate /
  forced, 1–41 nodes in the V1 fixtures). Measured on identical quick-tier
  samples (harness, cap 500): human 338 positions — goal=both surfaces
  **15 losses**, goal=loss (full budget) proves **42**; selfplay 343 —
  both 0, loss-goal 7. **~64% of provable losses are budget-starved away.**
  Also: `goal=win` FILTERS loss facts at the root
  (solve_goal_filters_root_facts) — a win-goal sweep shows loss=0 by
  construction. Candidate fixes to A/B in the harness: (a) two-pass
  win-then-loss at leaves (costs a second budget only when win is Unknown —
  and Unknowns are exactly the expensive class, so measure the wall hit);
  (b) a nonzero dual split; (c) loss-probe-first (losses are cheap: V1
  16–22µs) then win. Soundness: loss = opponent-win proof under restricted
  width = pure strengthening, verifier-checked; only NO-results are
  width-unsound. OPEN — first-class harness campaign arm.
  **First campaign ran 2026-07-20 (quick tiers, human sample n=338, paired
  vs anchor):** cap ladder under Both — 1000/2000/4000 give W 50/54/54,
  L stuck at 15 at EVERY cap (dual starved regardless of budget, as
  predicted); two_pass@500 = 48W+42L (p=1.5e-08), two_pass@2000 =
  54W+44L (p=5.8e-11, +56% decided vs anchor). Economics (nodes/decided):
  **two_pass@500 = 215, BETTER than the anchor's 243** — loss passes on
  unknowns are cheap and yield; cap raises cost 315-611 nodes/decision.
  Ranking: two_pass@500 strictly dominates every both-cap arm; cap 2000 is
  a second-order add-on. Production adoption of two_pass = a leaf-hook
  protocol change (Rust; the P4 build item). Ground-truth scale of the
  miss: the production-parity arm failed **70 cheap certified atlas
  losses** (113–387 dedicated-loss nodes) in the puzzle dev split.
  **Mechanism sharpened by the puzzle_v2 gate iteration:** when the primal
  win search width-exhausts, Both returns Unknown with the ENTIRE
  remaining budget unused — 15 human positions whose win pass died at 2
  nodes and whose loss proof costs 5–44 nodes came back unknown-at-cost-2.
  The two-pass fix is nearly FREE exactly there (498/500 budget idle).
  Consequence for ground truth: loss labels can never be must_solve for
  arbitrary-goal arms (v3 mint: must_solve = wins ≤400 only; the loss-side
  obligation lives in the loss_detection canary for claiming arms).
- **P6 (NEW, from the residue map `e11c393d`) — candidate-generation
  engineering.** The wall-time residue instrument (3 bit-identical reps,
  100% wall accounted, 0.83% overhead) found: F19 deep solves spend 82%
  of wall in candidate GENERATION machinery (attacker 45.9% + forced
  defender 36.0%); human-160 production-like positions spend 40.5% in
  UNFORCED defender generation (16.7M events — the exact class the G2
  selector addresses; ceiling for that build) and 39.4% in unresolved
  attacker work; the winning proof path costs ~0.0003% — nearly all
  solver cost is enumeration around proofs, not proofs. Ordering-miss
  direct cost = 1.8% (kills the ordering family from the cost side too).
  CAVEAT: human cohort ran at 50k-node budget, not cap 500 — proportions
  at production cap will differ; harness remains the adoption
  instrument. USES: (a) G2 sizing ceiling ~40%; (b) generation-machinery
  memoization/incremental-build = top soundness-free efficiency round
  for the offline/labeling path; (c) rerun the (feature-gated, reusable)
  instrument after G2 to VERIFY the block shrank. OPEN-PROBE (top of the
  efficiency stack).
- **P5 Solver-internal efficiency — ordering MECHANISM BUILT, first hint
  source REJECTED (2026-07-20 late, R-ORDER-PRIOR `5cff787c`).** Stable
  reorder-only hints (candidate-set invariant, cold-isolated, verifier
  untouched) are now a permanent engine capability. Measured with the ep90
  net's own root priors on 2,631 selfplay dev positions: **cap 500 = 12 up
  / 26 down; cap 2000 = 3 up / 39 down — raw game-policy priors MISDIRECT
  df-pn** (they down-weight the forcing/sacrificial lines proofs ride;
  harm grows with depth). Nodes-to-proof DID improve on the shared decided
  set (5.2k saved vs 1.4k added at cap 500) — ordering signal is real, the
  policy is the wrong oracle. Next hint-source candidates, in order:
  proof-participation statistics (residue map, in flight), threat-
  proximity statics, refutation-first defender ordering (strongest reply
  first at AND nodes). OPEN — mechanism ready, oracle wanted.
  **Threat-proximity statics also REJECTED (same night): +19/−99 on human
  dev at cap 500 — worse than policy priors.** Reading: wide-pn's native
  generation order already encodes forcing structure; EVERY generic
  reorder tried so far destroys it. The bar for an oracle is now high —
  proof-participation data or nothing.
- **Dual-pass ADOPTION RUN (dev+holdout): ALL PASS, no overfitting**
  (human: dev 14.5%W/12.3%L vs holdout 14.2%W/12.1%L; selfplay similar).
  Box checked for the production flag flip.
- **Cap×dual-pass Pareto (quick samples): cap 250 FAILS the must-solve
  floor** (2 atlas wins lost — the ground-truth gate correctly rejects
  it; the envelope floor is between 250 and 500). 500→2000 grows human
  sample coverage 90→98 and puzzle 13→42/48; 2000→4000 adds ~2 — the
  knee is at ~2000. Standard-tier promotion candidate = cap 2000 +
  dual_pass, pending bench throughput at that cap.
- **WIDTH-INCOMPLETENESS WITNESSES FOUND (atlas-deep probe at 100k):**
  of the 5 certified-WIN atlas rows unprovable at the 20k labeling cap,
  2 crack at 22k/31k nodes (deep but reachable) — but **3 width-exhaust
  at 21–43 nodes**: the wide profile PROVES no vcf_pair_complete win
  exists, yet the atlas certificate says WIN. These are the first
  concrete witnesses that the width misses real wins at ANY budget.
  Implication: some fraction of the 96 width-exhausted grinds may be
  true wins too — the width-completeness study (owner passed earlier)
  now has hard evidence instead of a hypothesis.
- **Promotion package COMPLETE (same-binary paired benches, order .so):**
  cap500+dual_pass = 416.5 moves/min, 19% park bail, window shows
  4,714 W + **5,337 L** (losses OUTNUMBER wins at production — the loss
  stream the engine never had). cap2000+dual_pass = 346.25 (−16.9%),
  park bail **97.6%** (max wait 5.1s): at cap 2000 the solver stops
  steering play in the 150ms window and becomes a background labeler.
  **RECOMMENDATION: production flip = cap 500 + tss_solver_dual_pass=true
  (free, +288 dev losses, live loss stream); cap 2000 = OFFLINE/labeling
  operating point only. The in-game deep-solve lever is now the park
  envelope (150ms window sweep), not the cap.**
- 2026-07-21 (late-night gates, consolidated from sibling branches —
  this file is the CANONICAL notes copy; branch topology below):
  - **R-LOSS-SIDE gated `a0df3c60` (branch claude/loss-side):** the 58
    cap-500-missed atlas losses cost ≥512 dedicated nodes (median 1,064,
    max 19,536) — unreachable at cap 500 under ANY allocation; loss-
    probe-first rejected pre-implementation (non-resumable preprobe =
    predictably regressive); leftover-only dual pass CONFIRMED optimal
    (reserve-32 == reserve-0 measured); default-zero loss_reserve_nodes
    knob + DESIGN_TSS_LOSS_SIDE.md shipped. 197/197 independent.
  - **R-CP10 gated+pushed `32064e8` (tss-lean cp1/lean-grammar):**
    accept⇒won capstones kernel-PROVEN for base/D17 × tree/DAG
    (standard axioms only); negative side = NoContractWin, honestly
    scoped. Open: FH/T6 classes, Rust-byte correspondence.
  - **G2 v1 built+gated `2a6fc0db` (branch claude/g2-cert, from
    loss-side):** verifier extension = 471 pure additions (LegacyOnly
    default rejects new variants; isolated tss_verify_group2.rs);
    gate-free class fully verified incl. R1/R2 + hostile-review
    counterexample battery; FhwGateV1 rejects wholesale (v2 work);
    selector demo 19 edges vs 886 legal (~46x, verified); never-
    decides-less structural; 209/209 independent; harness group2-on
    ALL PASS verdict-parity + truthful canary. Selector does NOT fire
    on wide-profile fixture proofs (implicit-dispatch class = v2
    gates); the residue 40% ceiling awaits v2. Review point: finder/
    verifier share derivation helpers — independent reimpl follow-up.
  - **Branch topology (merge order for consolidation):** claude/v1-soak
    → claude/twopass-leaf → {claude/order-prior (THIS copy of notes),
    claude/loss-side → claude/g2-cert}. Merging order-prior +
    g2-cert lineages reunites all solver work; harness_runs archives
    are per-branch.
- **Batch-order dependence: MEASURED ZERO** (human dev n=2,185, forward
  vs reversed batch at cap 500: 0 verdict changes, 0 contradictions).
  The documented TT-carryover caveat affects node counts at most, not
  verdicts; sample-vs-full comparisons are safe at this cap. Also clears
  hinted-vs-baseline A/Bs of the cold-cache confound (hinted solves run
  cold by design; carryover demonstrably doesn't move verdicts).
- ~~Verdict injection (Lever-2)~~ — OWNER RULING: production consumption
  already improves search/policy/value (steered visits + outcomes are the
  training targets); direct target-swapping is NOT a solver lane. Dropped
  from this backlog.

## 7. Iteration log

- 2026-07-20: Document created. V1 gated (e062922b); unknown-wall + grind
  characterization added (6518acc6); warmth retraction + census-gate
  applicability analysis (§3, §4); horizon dropped by owner ruling; V2 h2h
  running (unbounded vs h16 under production consumption).
- 2026-07-20 (later): Owner rulings folded in (mission = verified coverage on
  fixed sets, no shortcuts; adaptive budgets parked; envelope tunable).
  Batch-API stats gap found (§3). Warmth smoke: fragment store engages under
  env (19→2 nodes on a repeated-structure WIN), full rerun launched.
- 2026-07-20 (evening): **V2 h2h LANDED — unbounded beats h16 at strength**:
  150–106 over 256 paired production games at ep90, 58.6% decided
  [52.5, 64.5], seat-balanced, pentanomial p≈0.002 (docs/V2_H2H_REPORT.md).
  Horizon question closed on both axes (V1 coverage + V2 strength);
  horizon=0 normative. Harness: SET-HUMAN-V1 frozen (2,720 positions,
  pin 5784defe; V1 raws = 320 positions × 10 arms, NOT 2,720 positions —
  anchor subset asserted at mint). Bench venv wiring fixed (GPU venv .pth
  points at a stale checkout; bench now self-resolves hexfield_eq from the
  worktree + infrastructure packages from the main checkout). Smoke-tier
  0-decision readout = torch.compile eating the 20s window (0s warmup by
  design) — the full profile's 60s warmup absorbs it; smoke validates
  wiring, not throughput.
- 2026-07-20 (shakedown, owner-directed): deliberate issue hunt over the
  first baseline. FOUND+FIXED: bench arm-identity bug (h16 benched under an
  unbounded manifest — now translated config + effective_tss echo +
  gate_bench_identity, selftest 19/19); canary goal-pinning bug; harness
  default flipped goal win→both (win FILTERS loss facts; both = production
  parity). FOUND+MEASURED: mode-3 loss starvation (§6 P4: both surfaces
  15/42 provable human losses; goal=loss full-budget proves 42; ~64%
  starved). DOCUMENTED: serial sweep vs threaded bench; TT carryover order
  dependence. loss canary live (canaries_v2, 3 agreed V1 losses).
  Baseline re-run as baseline_production_v2 (both-goal, bench unbounded).
- 2026-07-20 (anchor): **baseline_production_v2 = the standing baseline**
  (harness_runs/20260720_211755_baseline_production_v2, ALL GATES PASS incl.
  bench_identity echo-to-echo). Coverage dev-split: selfplay 150 W + 10 L /
  2,631; human 317 W + 105 L / 2,185 (loss coverage now measured — was
  structurally 0 before the goal flip). Bench: 136.0 moves/min, 544
  decisions, vf=0 — genuinely-unbounded costs ~2% throughput vs the flawed
  h16 bench (139.0), the +60 Elo side of the V2 trade. Bench-window note:
  unbounded deep_calls 11,950 vs h16's 20,712 at similar moves/min —
  unbounded spends more budget per call (grind class), park absorbs it.

## 8. 2026-07-21 overnight session — main_4 deployment + G2 v2 chain (this
   copy on claude/main4-integration is now the CANONICAL notes; the
   order-prior copy is frozen at a874e4c7)

- **main_4 LAUNCHED** (~03:53Z) from claude/main4-integration: weights-only
  warm start from main_3 ep90, cosine LR 2e-4→2e-5/150+floor, solver =
  cap 500 + dual_pass + unbounded + wide + park 300 ms + async 8 + guards.
  Epochs 1-20: deep_verify_failed=0 throughout; ~10-15k proven wins +
  15-24k proven losses consumed per epoch (loss stream live, loss backups
  outweigh win backups ≈3.3:1). Launch traps fixed: manifest at
  packages/hexfield_eq/Cargo.toml (not rust/); WSL needs rustup cargo
  1.95 (~/.cargo/bin — distro 1.75 fails lockfile v4); infra packages
  (hexo_utils/_engine/_models) resolve from the resume-run-crash-fdef2b
  worktree (built .so live only there) via INFRA_TREE split in the
  supervisor (4a30c8f7).
- **PARK SWEEP (a66b707a, order-prior): 300 ms adopted.** Bench (release
  .so, sha-pinned, 7 arms): bail 12.7%→0.0% at 300, decided verdicts/min
  +2.8%, throughput flat 75-300, narrower strictly bad. **LIVE-LOAD
  CAVEAT (MEASURED, epochs 1-20): under full trainer load bail runs
  76-85%, avg wait ~750 ms — the bench's 0% was selfplay-only; the train
  pass changes solve latency.** Async pool saturation is the bigger
  signal: async_dropped ≈55% of enqueued, stale ≈58%, workers_spawned=4
  (of 8 configured). OPEN TUNING ITEM: async capacity first (workers/
  queue), window second; decide with one paired measurement, not blind.
- **VALUE-SIGNAL AUDIT (6d8adc2f):** consumption path WIN/LOSS symmetric
  end-to-end (leaf mint, consume gate, guards, hard_z). New counters
  deep_win_backups/deep_loss_backups split the consumed stream at the
  single chokepoint. Flagged-not-changed: root-loss scalar (no move to
  force), win-only zero-mass recovery.
- **TSS→TRAINING TELEMETRY (epoch 20 snapshot):** 16,283 moves; proof
  rows 4,396 (27%), sharpened 3,391 (20.8%), win_rows 2,390 +
  loss_only_rows 1,560 (the brand-new class); win_retained_mass 0.265
  pre-sharpen vs gumbel_win_retained 0.751 post — the solver modification
  of policy targets is HEAVY. Search side: 671,599 leaf verdict hits,
  263k hard backups (6.4% of 4.1M), 392k memo hits, root injection 1.2%,
  forced-defense 18.6% of moves, proof_disagreements 55 (tiny).
- **INTERNALIZATION (instrument 8e8ce68d on v1-soak; SET pin 523e0783,
  683 certified positions):** main_3 curve = value-only learning (WIN
  sign-disagree 28.6%@ep25 → 11.3%@ep90 → 17-19% late decay; policy
  top-1 FLAT ~55% all run; loss values perfect from ep25). **main_4
  BREAKS THE POLICY PLATEAU: ep10 top-1 65.1% / mass 0.548 / WIN
  sign-disagree 3.0%; ep20 66.8% / 0.563 / 3.6%** (vs ep90-init ~59.5%/
  0.48/14.0 on the selfplay cohort). Caveats: fresh-optimizer + LR reset
  is a confound for the speed of the jump (not the plateau-break itself —
  main_3 trained with TSS sharpen and stayed flat); set is main_3-anchor
  derived (in-distribution for the solver).
- **G2 v2 CHAIN — CLOSED THIS PHASE, DO-NOT-ADOPT at production config
  (claude/g2-cert through 4f87b893):** accept path (Exact/FC) hostile-
  reviewed SOUND (0 critical, 2 hostile tests added); FC positive fixture
  discharged A1; production emission + wide-PN port BUILT (representative-
  only child materialization solves the DefenderPair grammar mismatch =
  design §5 "largest unknown"); dual-materialize + strict self-verify
  firewall; flag-off bit-identical. A/B ×2 (6,462 positions): parity
  PERFECT, vf=0 both arms, **firing=0** — the frozen verifier's no-mixing
  class rule (compact H_Q bound; 214/918 ThreatCountOutOfRange) blocks
  whole-cert gating on real wide proofs. **Adoption boundary = verifier
  amendment admitting mixed legacy+gate certificates**
  (G2V2_WIDEPN_REPORT.md §5) → requires a fresh hostile-review round
  (trusted-base change). Node-level applicability is real (525/704
  reductive closures on the corpus); cert-level composition is the wall.
  Suite 250/0/37. G2 stays OFF in main_4.
- **Branch topology now:** claude/main4-integration = consolidate-main +
  order-prior + g2-cert@7d6e5f1b + main_4 config/scripts (THE production
  branch, main_4 runs from it). claude/g2-cert continued past the merge
  point (accept path → wide-PN port, 4f87b893) — fold into a future
  integration round together with the amendment decision.

- **2026-07-21 maintenance retune MEASURED (fresh ep21 vs ep18-20, clean
  paired-adjacent epochs):** async 8/12→12/24 + park 300→600 ms →
  **2.5x proof yield** (wins 15k→38.3k, losses 22.6k→51.8k per epoch;
  proofs/move 2.3→5.5), drops 35%→3.4%, bail 76-85%→43%, vf=0, at
  −16% selfplay pace (18.8→15.8 pos/s). KEPT — signal density beats
  pace for the internalization mission. Residual: bail 43% at avg wait
  805 ms (hard-park tail); wider windows = diminishing returns, revisit
  only with cause. Ops lesson (twice-burned): halt-flag races the
  relaunch loop — deterministic stop = kill SUPERVISOR pid first, then
  child; never remove the flag while the supervisor lives (orphan
  dual-trainer incident; ep21 wiped + rerun clean on owner order).

- **2026-07-21 G2 mixed-cert amendment GATED `af3a9a88` (owner-authorized
  trusted-base change): FIRES AND IS SOUND, still DO-NOT-ADOPT.** Firing
  0 → 36-106 gate positions across 6,462 (count non-deterministic from
  parallel materialization order; soundness invariant); cross-verify
  142/142 clean across both runs; coverage and nodes/decision byte-
  identical ON vs OFF — the search never consumes the cert-level
  reduction (never-decides-less is structural). THE G2 LADDER IS NOW:
  theory PROVEN → verifier SOUND → emission FIRES → **consume-mode
  (design §5.1) = the one remaining step to any harness gain**, and the
  realistic prize sits at the deep/offline operating point (cap 2000+/
  50k labeling, 40% residue), not cap-500 in-game (gateable class ~1%
  of positions). Seam hostile review + consume-mode design = the next
  trusted-base round (Codex Jul 24 or owner-authorized). tss_verify.rs
  byte-identical through the whole campaign.

- **2026-07-21 P7 pair-generation prefilter round GATED `2c262e10`
  (owner-ordered speedup lane; run STOPPED at ep23 for the work):
  1.42x solver wall, BIT-IDENTICAL.** Attribution first (new
  instruments, both committed): production-shape residue (cap 500,
  256 KiB TT, unbounded, vcf_pair_complete — tss-vcf-width
  `1b4afec2`) put A_OR_GEN at 60.5% / D_FORCED_GEN 20.4% / TT+verify
  <1% of solve wall; closure-debt sub-profile put 43.8% of pair-gen in
  2.66M evaluate_pair calls at 81 ns with 2.1% acceptance. Three exact
  cuts (skips are provably None evaluations; children byte-identical):
  family-size prefilter (empty AND single-window families — a lone
  post-pair window always keeps an empty so mhs==1; per-second counts
  k1+joint+c3 vs the size-2 bound, empty allow set skips the whole
  first incl. second-generation), defender tier (unhit defender >=4
  window ⇒ second must lie in the <=2-cell intersection of unhit
  empties), and gate storage (windows stored once + per-cell index
  lists — the old build cloned window+empties per empty cell; per-pair
  first-list lookup hoisted per first). Evaluated pairs 2.66M→224k
  (42% accept). GATES: suite 217/0 (python feature), golden digest
  PASS, identity battery over ALL 6,443 frozen-set positions —
  statuses and node counts byte-identical, vf=0, wall 41.1s→28.9s
  (1.42x), max solve 138→61 ms. Latency reframe (quiet + 12-way
  contended batteries, `_bench_solver_wall.py`): cap-500 solves are
  NOT intrinsically slow (quiet p90 30 ms, 12-way contention only
  1.1-1.7x inflation, TT 0.3% of wall — NOT cache-bound); live
  hundreds-of-ms = queue wait, already retuned. Next efficiency rungs:
  second_candidates HashSet churn (~8% wall), first-candidate
  enumeration residual (~15%), defender-side D_FORCED_GEN (~20%),
  cross-node generation memoization (P6 proper).
- **2026-07-21 ALL-LEAVES MEASUREMENT (feasibility for solving quiet
  leaves too): the threat gate leaves real coverage on the table.**
  Of 5,975 human+selfplay set positions, 4,823 (81%) are quiet under
  the leaf gate; production solves on the quiet cohort yield **465
  verified WINs (9.6%), 0 losses**, vf=0, at mean 3.95 ms (p50
  0.14 ms — no-candidate quiet solves self-terminate, the solver is
  its own filter; p90 17 ms grind tail). CPU math: all-leaves ≈ 3.4x
  current solve CPU at ~5x request rate — feasible with the 1.42x
  engine + worker headroom ONLY with a park split: quiet leaves must
  NOT park (net eval immediately, solve lands in memo for later
  descent-stops), threat-parked solves keep priority. Needs a
  two-tier enqueue (priority flag) — design item, not yet built.

- **2026-07-21 ALL-LEAVES STRICT MODE BUILT+BENCH-GATED (owner-ordered:
  "all leaves evaluated by the solver and only then sent to the gpu",
  single-tier queue, no routine bail).** New flag `tss_solver_all_leaves`
  (default OFF; tree.rs drops the has_threats gate on both deep-leaf
  routes; plumbed config.py -> overrides -> Divergences; suite 217/0
  flag-off). Strict ordering = park mode + all_leaves + park timeout at
  the 5000 ms validation ceiling (emergency valve for pool failure —
  solves are node-capped so waiting is bounded work; the pipeline hides
  solve latency behind other simulations' GPU batches). Bench A/B at
  production shape (256g/256v, ep90, 12/24 workers, new P7 engine):
  enqueues 75,387 -> 298,540 (3.96x, quiet solves self-terminate),
  **drops 0 / bails 0 / vf 0**, mean park wait 2.8 ms; throughput 426.5
  -> 384.0 moves/min (**-10%**); per decision: WIN backups 10.0 -> 29.7
  (**3x**), total proof backups 30.7 -> 42.0 (+37%), loss backups
  20.6 -> 12.2 (down — quiet-leaf proofs are win-heavy and the extra
  early wins re-shape search; watch the loss stream on live epochs).
  Cost precedent: the ep20 retune was KEPT at -16% pace for 2.5x yield,
  so -10% for 3x wins + strict solver-first semantics is inside the
  owner's stated trade. **main_4 config FLIPPED (all_leaves=true, park
  5000) — takes effect at relaunch; validate first epochs: park_bailed
  ==0, async_dropped==0, deep_win/loss_backups, moves/min, vf==0.**
  Bench-only caveat: no concurrent train pass; live CPU contention is
  the reason the emergency valve exists. Curiosity (non-blocking):
  park_wait_ms_max 6,084 > the 5,000 deadline with bails=0 — drain-pass
  granularity at move tail; bail is liveness-only, not soundness.
