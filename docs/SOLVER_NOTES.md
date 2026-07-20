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

## 6. Probe backlog (status; owner rulings 2026-07-20 inline)

- **P1 TT economics** — TT-off/TT-tiny A/B at cap 500: verdict parity +
  per-solve wall. Owner caution: must not weaken deep/critical solves — so
  run the A/B at cap 500 AND at a raised cap to measure TT value as f(cap)
  before touching anything. Needs cargo lane. OPEN-PROBE.
- **P2 Fail-fast / deadline design** — precondition for any A5-style cut in
  the unbounded profile (§4). Quantify first: instrument which deadline
  source (board-fill, pattern reach) would have applied on the 248 grinds.
  Analysis can start from raws + a probe build. OPEN-PROBE (owner: worth
  quantifying, temper expectations).
- **P3 Warmth** — MEASURED + CLOSED at cap 500 (2026-07-20, full 6,510-solve
  paired rerun, raws/soak_warmth_frag_s{0,1}.jsonl vs cold control):
  mechanism WORKS (35 positions saved nodes; 1,281 nodes saved in the
  unbounded arm, 0 added; best single save 339→134) but **zero verdict
  flips** — savings concentrate in already-proven wins (fragments come from
  wins, so they help re-prove neighbors, not crack Unknowns). Under the
  coverage metric: no gain at cap 500; production flip NOT justified. Wall
  deltas from this run are load-confounded (ran beside V2) — node counts are
  the causal signal. Reopen only as part of P1's value-vs-cap sweep.
- **P4 Both-goal probing** — owner: WANTS win+loss detection. LOSS probes are
  ~10³× cheaper than WIN grinds; design question is where mode 3 currently
  asks the LOSS question vs where it could (every gated leaf?). Map the
  production call sites first. OPEN-PROBE.
- **P5 Solver-internal efficiency** — move ordering / pruning inside wide-pn
  so the same 500 nodes prove more. No specific lever identified yet; the
  win-vs-grind contrast (hot 45.5% vs 6%, threats) suggests ordering signal
  exists. OPEN-PROBE.
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
