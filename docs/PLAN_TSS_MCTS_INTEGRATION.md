# PLAN — TSS × MCTS Integration v3 (PRIMARY)

> **Provenance.** v3, 2026-07-17. **This is the primary and normative plan for
> solver-in-trainer integration.** It supersedes `PLAN_TSS_DEEPENING.md` (v2)
> as plan-of-record; v2 remains the design history of the built Stage 0–4
> stack, and its §2 soundness contract is carried into §2 here **verbatim and
> unchanged**. Origin: owner review session 2026-07-17 folding the solver
> campaign's landed results (leaf-surface campaign / config D, R-FIX1, R-LF2,
> R-CR1, C1 one-engine, R-KT1, R-TS1) into the trainer program.
>
> Build substrate: the v2 stack (Stages 0–4, branch `claude/tss-v2-build`,
> all flags default-off) and `docs/TSS_RUNBOOK.md` rung mechanics remain the
> deployment vehicle. This doc redirects **what** deploys, in what order,
> with what budgets and horizons. Nothing here weakens the soundness contract.

---

## 0. Document registry (deprecations)

| Doc | Status |
|---|---|
| **PLAN_TSS_MCTS_INTEGRATION.md** (this) | **PRIMARY / normative plan** |
| PLAN_TSS_DEEPENING.md (v2) | SUPERSEDED as plan; design history + build record. §2 contract copied here. |
| TSS_RUNBOOK.md | LIVE — rung/flag/soak mechanics for deployment |
| TSS_SOLVER_SPEC.md / TSS_SOLVER_PROOF.md | LIVE — build records of the trainer Stage-3 solver+verifier |
| PLAN_TSS_MOVESET_ZONES.md, PLAN_TSS_SOLVER_UPGRADES.md, PROOF_TSS_DEFENDER_ZONES.md | Solver-campaign docs; normative copies live on the solver branches — trainer copies are reference-only snapshots |
| TSS_PARK_SPEC.md, TSS_SOLVER_OPT_SPEC.md, TSS_SOLVER_PROFILE.md, TSS_ZONE_IMPL_BRIEF.md, TSS_ZONE_IMPL_STATUS.md | HISTORICAL |

## 1. What changed since v2 (why a new plan)

The solver campaign delivered an engine whose leaf-profile economics
invalidate v2's cost assumptions:

- **Leaf-surface campaign (config D = wide PN + `TSS_LAZY_FRONTIER` +
  `TSS_INTERIOR_CENSUS_GATE`, 256 KiB TT, MCTS-realistic harness):**
  cap 500 → **13.00% verdict rate vs the narrow baseline's 7.00% at cap
  8,000** (the baseline is what the trainer integrates today); h16 doubles
  the narrow rate (6.33% → 13.33% at cap 2k) at −65.6% wall; at h8 the
  census gate dismissed all 692 evaluated interiors, work is ~flat
  (~1,852 nodes regardless of cap), p90 wall −93.7%. 806/806 certificates
  verifier-accepted; zero WIN/LOSS contradictions; persistent-reuse guard
  PASS. Fragments and K_reply: OFF at this profile (measured no-value).
- **R-FIX1:** the bounded-horizon zone-clock defect (finder stamped
  caller-deadline budgets; verifier rejected genuinely winning zoned
  finite-horizon certificates) is FIXED. **Mandatory** for any leaf/root
  deployment — pre-fix engines silently lose bounded-horizon WINs.
- **v2's "vise" kill-criterion is answered in the good direction** — a
  cap small enough for throughput (500) proves plenty. v2's subsample
  fractions, cost guesses, and h8 default are stale.
- **Solver campaign closure results** (R-KT1 width-taxonomy null, R-TS1
  threshold null): the engine adopted here is final-shaped; no pending
  solver rewrite blocks integration.

**Owner rulings encoded by this doc (2026-07-17):**

1. Certificate-mined one-hot **policy** targets: **REJECTED** (§7).
2. Per-cell solver-class head as candidate **replacement for cell_q** (§7).
3. **Root channel = 100% coverage (decided)**; leaf channel targets
   **maximal coverage — aspirationally 100% ungated** (§4, §6).
4. Horizon to be **raised/reworked**; mechanism open, options in §5.
5. `has_threats` gate **retired**; external gating demoted to a
   fallback/escalation role — the engine's interior census gate is the
   primary filter (§6).
6. Expanded metric set; internalization curve is the program metric (§8).

## 2. Soundness contract (carried verbatim from v2 §2 — unchanged, binding)

| Signal | Search use | Training use |
|---|---|---|
| λ¹ proof (`verdict()` ±1) | hard ±1 backup; guard; interior pruning at `k == B` | Lever 1 target sharpening; Lever 2 label |
| Deep proof, **certificate verified pre-backup** | hard ±1 backup + eval elision (containment ladder) | Lever 2 label + per-action outcomes |
| UNKNOWN / capped / heuristic (pn-ratios etc.) | move *ordering* + unforced injection only — never a value | excluded from hard labels (aux/soft tier only, §7) |

1. **Typed seam.** `ProofStatus ∈ {Win, Loss, Unknown}`; only two
   `HardValue` producers (λ¹; verified deep certificate).
2. **Verify every deep hard result before backup — including cache hits.**
   Verification failure → downgrade to Unknown + fatal telemetry counter
   (must stay 0).
3. **A hard LOSS requires the dual certificate** — a proven opponent
   winning strategy whose universal nodes exhaust our legal moves. "My
   attack failed" is UNKNOWN, never LOSS.
4. **UNKNOWN never collapses to a scalar.** Cap/budget exhaustion poisons
   the parent AND/OR result to UNKNOWN.
5. **Full-key equality on every value-bearing cache hit**, including the
   D6 outer cache. The neural `StateHash` is never used for proofs.
6. **Determinism where targets are made.** Self-play hard paths use node
   caps only; a completed verified certificate found under a serve
   deadline is sound — converting a timeout into a verdict is not.
7. **Soft signals bias, they don't poison.** Ordering/injection shifts
   visits and therefore targets; safe because a heuristic cannot inject a
   false hard label.
8. **Pruning = proof-stapled dropping only.** `firstK` truncation of any
   forced set is forbidden everywhere. Geometric sets are uncapped and
   D6-covariant.

## 3. Engine adoption (Phase-3)

- **The leaf/root solver is the campaign engine at the leaf-decided
  config:** wide `vcf_pair_complete` width, lazy=1, gate=1, fragments=0,
  k_reply=0, 256 KiB TT, WIN goal (LOSS side: §3-OPEN below), relative
  horizon per §5, cap 500 at leaves. Flags recorded verbatim in
  HUNT_REPORT_LEAF_SURFACE.md.
- **R-FIX1 included** (finder stamps the verifier's exact D14 budgets).
  Acceptance check: the frozen compact-h16 regression certificate must
  verify.
- **OPEN-ENG (needs its own sizing pass): port scope.** The trainer's
  built Stage-3 solver and the campaign engine are separate builds. The
  one-engine principle (C1) argues for adopting the campaign engine as
  the single mint, with `tss_cert_version` tracking it through the
  schema. Decide: full engine adoption vs backporting the three levers
  (wide width + lazy + gate) into the trainer solver. Verifier stays
  independent either way.
- **Persistent solver per batch + cross-move TT:** validated (13 ms-cliff
  guard PASS in every config). v2 §11 memory hard-caps stand — 256 KiB TT
  per solver, per-search memo freed at move end, certificate buffers
  streamed; the 29 GB WSL ceiling is an acceptance criterion.
- **§3-OPEN: LOSS side.** The leaf campaign measured WIN-goal only, so
  the −1 half of the value signal is structurally underserved at leaves.
  Root channel: dual-seat solve from the start (once per move — cheap).
  Leaf channel: opponent-goal solve gated on opponent-threat features
  (§6); measure before committing.

## 4. Two channels, two budgets (DECIDED)

**Root channel — the target-maker.** Deep solve at every self-play root
position (both seats), cross-move TT reuse. Because training rows *are*
root positions, this channel annotates **100% of rows directly**: Lever-1
policy masks, Lever-2 proof-corrected values, per-cell class labels (§7),
the disagreement stream, certificate-horizon moves_left (§7). Cost: one
solve per played move — trivial next to a search. **Deploys first.**

**Leaf channel — the search-improver.** Solver at leaf expansion; verified
hard backups + eval elision. Affects targets only indirectly (visit
shifts). **Coverage doctrine (owner 2026-07-17): the target is 100% of
leaf expansions, ungated.** Feasibility rests on three properties, one
measured and two designed:

- The interior census gate is an *internal* early-out — quiet positions
  exit the solve almost immediately with a certified no-win-within-h, so
  cost concentrates on genuinely hot leaves (h8 measurement: all 692
  interiors dismissed, ~1,852 nodes flat, p90 wall −93.7%).
- Solves run on CPU in the GPU's shadow (async/pipeline lane, v2 §7.3);
  add the select-phase wall timer before assuming absorption.
- Every fired verdict *refunds* a GPU eval (elision), and the refund is
  largest exactly where solves fire most (threat-dense endgame, the S²
  ~13× batch-collapse zone) — net cost can go negative there.

R0 must produce the per-solve wall distribution (quiet vs hot, h8 vs h16)
and the shadow-absorption measurement. If 100% is unaffordable at the
chosen horizon, fall back to §6 consequence-gated partial coverage rather
than lowering the horizon.

Budgets, flags, and metrics are kept separate per channel. Serve-time root
guard unchanged (v2 rung 6 semantics: after WIN certificates are trusted).

## 5. Horizon policy (OPEN — owner wants taller; pick by measurement)

| Option | Shape | Notes |
|---|---|---|
| A | Flat h16, both channels | Simplest; h16 measured at 2× verdict rate, −65.6% wall vs narrow |
| B (**lean**) | Split: root h16–h32, leaf h8/h16 by A/B | Root is once-per-move — it can afford tall horizons; nobody has measured how tall pays. Leaf choice from the §6 soak. |
| C | Adaptive: h8 → re-solve h16 on Unknown + hot features | NQ8 refuted escalation ladders at the deep offline profile; leaf economics (256 KiB, cap 500) are a different regime — measure, don't assume either way. |

Decision rides the R0 shadow soak (§9): root h-ladder (h16/h24/h32)
verdict-rate-vs-ms curve; leaf h8-vs-h16 under the escalation features.
**Lean update (2026-07-17, coverage doctrine):** root = tall ladder (B);
leaf = base-h8 for all + h16 escalation on hot features (C's mechanism,
now the natural shape under 100% coverage). The NQ8 ladder refutation was
measured at the deep offline profile, not this regime — still measure.

## 6. Solve gating — retired as a filter, retained as escalation (DECIDED)

**`has_threats` is retired:** it requires a live ≥4 window to already
exist, but deep wins *begin with threat-creating moves* — an h16 win from
a quiet position is exactly what λ¹ cannot see and the net has not
learned, i.e. the highest-value solve class, and the old gate excluded
all of it.

Under the §4 coverage doctrine there is **no external solve/don't-solve
filter — the engine's interior census gate is the filter**, and it is
certified: a proven "no win within h from here" is the cheapest possible
early-out, computed inside the solve itself. The former gating features
survive in two demoted roles:

- **Horizon escalation (ties to §5):** every leaf gets the base-horizon
  solve; leaves scoring hot on cheap features (alive-window counts with
  count ≥2/≥3, `min_hitting_set`, census distance, contested root race,
  |net value| bands) get the taller re-solve. Escalation-not-exclusion
  means a mis-scored feature costs depth, never coverage or soundness.
- **Fallback partial coverage:** if R0 shows 100% is unaffordable at the
  chosen horizon, gate by *consequence* features only (visit weight of
  the leaf's root ancestor, top-2 root Q gap, net-vs-search
  disagreement) — provability filtering is already handled internally by
  the census gate.

**Protocol (counter-first, unchanged):** the R0 soak solves ungated; per
solve, log features + verdict + *would-it-have-flipped* (leaf: backup
sign vs net eval; root: move choice). Thresholds — now escalation
thresholds — maximize consequential verdicts per CPU-ms. Escalation
biases where compute goes — soundness-neutral by contract rule 7.

## 7. Targets

- **Lever 1 (guard-consistent policy masks) and Lever 2 (proof-corrected
  values): carried unchanged** from v2 §§4–5, now fed by root-channel
  classes at every row (coverage was the missing ingredient, not
  semantics).
- **Certificate mining for policy targets: REJECTED (owner 2026-07-17).**
  Rationale recorded: (i) interior-guard pruning already routes self-play
  through forced lines on-policy, producing full visit-distribution
  targets at those positions — richer than synthetic rows; (ii)
  certificates designate exactly ONE winning move per OR node (D9
  grammar), so mined policy targets would one-hot-encode the solver's
  arbitrary choice among winning moves — the "a set, never a single PV"
  violation. **PARKED variant** (revisit only if `proof_disagreements`
  stalls): value-only interior mining — interior ±1 labels are exact
  regardless of the designated move — with its own `target_regime` and a
  pre-registered distribution-shift kill criterion.
- **Tactical class head (CANDIDATE — replaces cell_q).** Per-cell 3-class
  target (proven-win / proven-loss / unknown) from root classification +
  root deep solves; optional position-level "forced win within h exists"
  bit as companion. Facts: cell_q today = per-cell binned value on MCTS
  export, `q_head_weight = 0.1` (main_3), train-only. Design conditions:
  class-imbalance handling (mask/weight so the head cannot win by
  predicting Unknown everywhere); labels are **aux-tier** under contract
  rule 7 (engine-trusted Unknowns allowed; the solver campaign's
  completeness theorem — agenda 3.1 — would upgrade them to proof-backed);
  the head doubles as the internalization instrument (§8.1). **OPEN:
  replace vs add** — verify nothing consumes cell_q at serve, then A/B
  under one `target_regime`.
- **moves_left at proven rows (NEW, cheap):** derive the target from the
  certificate horizon instead of behavioral game length; per-head masks
  exist in the v5 schema.

## 8. Metrics (v2 §9 carried; new set below, in priority order)

1. **Internalization curve (THE program metric):** at every root solve,
   the prior's mass on — and rank of — the certificate move, tracked over
   epochs. Rising = the net is absorbing the forced tree; flat = renting
   tactics from the solver forever (changes the value of every other
   lever).
2. **Consequential-verdict rate:** verdicts that flipped a backup vs the
   net eval, or changed the root move choice.
3. **Root-channel coverage:** % rows with a non-Unknown root class; %
   rows where the Lever-1 mask moved mass (KL raw‖masked).
4. **CPU-ms per consequential verdict, by gate feature** (feeds §6
   thresholds).
5. **Class-head accuracy on proven cells** (once the §7 head exists).
6. Carried unchanged: verify-failure counter (**must stay 0**),
   proof-vs-outcome disagreement stream, UNKNOWN rate under production
   caps, `opp_coverage`, fan-out / forced-line-depth histograms,
   select-phase wall timer.

## 9. Deployment (rungs; mechanics per TSS_RUNBOOK.md — draft order, OPEN)

- **R0 — engine port + shadow soak.** Adopt the §3 engine; solve + verify
  + log **ungated at every leaf on a measured slice — the 100%-coverage
  affordability trial** — consume nothing. Collects: per-solve wall
  distribution (quiet vs hot, h8 vs h16), shadow-absorption measurement,
  §6 features and flip outcomes, §5 horizon curves, §8 baselines.
  (Replaces v2 Stage-4 rung 1 with a wider mandate.)
- **R1 — root-channel consumption.** Lever-1 masks + Lever-2 labels from
  root classes (deep + λ¹), both seats. One lever per deployment step
  still applies: masks and labels are separate rungs (attribution), masks
  first.
- **R2 — leaf hard-LOSS canary** (eval-elision only; probe avoided lines
  actively — false LOSSes are silent).
- **R3 — leaf hard-WIN canary** with certificate-forced audits.
- **R4 — gated leaf rollout** at §6 thresholds; leaf horizon per §5.
- **R5 — cell_q → class-head swap** (own `target_regime`, own rung).
- **R6 — serve-time deep root guard** (owner: include; after R3 trust).

Health gate per rung: main_3's regular eval cadence (pool + Strix +
SealBot h2h), revert by flag + checkpoint, `target_regime` tags on every
target-semantics change.

## 10. Open decisions (owner)

1. **Port scope** (§3 OPEN-ENG): adopt campaign engine wholesale vs
   backport the three levers into the trainer solver.
2. **100%-coverage affordability verdict** (§4): pre-register the
   affordability bar (pos/s regression ceiling + absorption threshold)
   before R0; fallback = §6 consequence-gated partial coverage.
3. **Horizon heights** (§5): root ladder height; leaf base + escalation
   pair — from R0 curves.
4. **Leaf LOSS-side** solving (§3-OPEN): dual-seat at all leaves vs
   opponent-threat-gated. Note: the census early-out is WIN-goal
   machinery — LOSS-side solves lack the cheap internal filter today
   (solver-campaign agenda 1.2 LOSS-side censuses would supply it), so
   dual-seat-everywhere has worse economics than WIN-side until then.
5. **Class head: replace cell_q or add alongside** (§7).
6. Parked value-only certificate mining: stays parked unless
   `proof_disagreements` stalls (§7).
