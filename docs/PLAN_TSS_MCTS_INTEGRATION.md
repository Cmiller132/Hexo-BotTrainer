# PLAN — TSS × MCTS Integration v3.1 (PRIMARY)

> **Provenance.** v3.1, 2026-07-20 (v3 authored 2026-07-17). **This is the
> primary and normative plan for solver-in-trainer integration.** It
> supersedes `PLAN_TSS_DEEPENING.md` (v2) as plan-of-record; v2 remains the
> design history of the built Stage 0–4 stack, and its §2 soundness contract
> is carried into §2 here **verbatim and unchanged**. v3.1 additionally
> ABSORBS `PLAN_TSS_HORIZON_LADDER.md` (its ep54 gate result and ladder
> design are now §5; the standalone doc is deleted) and folds the 2026-07-20
> owner rulings: horizon ≥h16 or unbounded-with-node-cap, checkpoint-based
> validation at main_3 **ep90** (best-eval, owner-designated; the run was
> deliberately stopped at ep111), and the post-campaign engine facts
> (C_rel leaf-zone route CLOSED; ordering/threshold/taxonomy lever families
> closed; the campaign engine is final-shaped on `claude/tss-vcf-width`
> @ `ad606d0e`).
>
> Build substrate: the v2 stack (Stages 0–4, in this tree, all flags
> default-off — deployed through the park rung in main_3 up to ep111) and
> `docs/TSS_RUNBOOK.md` mechanics remain the deployment vehicle. This doc
> directs **what** deploys, in what order, with what budgets and horizons.
> Nothing here weakens the soundness contract.

---

## 0. Document registry

| Doc | Status |
|---|---|
| **PLAN_TSS_MCTS_INTEGRATION.md** (this) | **PRIMARY / normative plan** |
| TSS_RUNBOOK.md | LIVE — flags, profiles, rung/soak mechanics, build commands |
| PLAN_TSS_DEEPENING.md (v2) | Design history of the built Stage 0–4 stack; §2 contract source |
| TSS_SOLVER_SPEC.md / TSS_SOLVER_PROOF.md | LIVE — build records of the trainer Stage-3 solver+verifier |
| PROOF_TSS_DEFENDER_ZONES.md | NORMATIVE zone proof (rounds 1–6 review-hardened, 2,006-line version — solver-branch copies are STALE snapshots) |
| PLAN_TSS_SOLVER_UPGRADES.md | Solver-campaign lever register U1–U18 (reference; campaign closed) |
| PLAN_TSS_MOVESET_ZONES.md | Zone-campaign design history (reference) |
| PLAN_TSS_HORIZON_LADDER.md | ABSORBED into §5 (deleted) |
| TSS_PARK_SPEC.md, TSS_SOLVER_OPT_SPEC.md, TSS_SOLVER_PROFILE.md, TSS_ZONE_IMPL_BRIEF.md, TSS_ZONE_IMPL_STATUS.md | DELETED 2026-07-20 (historical; recoverable from git; PARK_SPEC's frozen build commands live in TSS_RUNBOOK.md) |

## 1. What changed since v2 — and since v3

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
- **R-FIX1:** the bounded-horizon zone-clock defect is FIXED. **Mandatory**
  for any leaf/root deployment — pre-fix engines silently lose
  bounded-horizon WINs.
- **v2's "vise" kill-criterion is answered in the good direction** — a
  cap small enough for throughput (500) proves plenty.
- **Campaign closure (through 07-19):** R-KT1 width-taxonomy null, R-TS1
  threshold null, ordering/scheduling family closed with zero residue,
  **C_rel leaf-zone deployment route CLOSED** (zone-relevance proven
  impossible — the engine port carries NO C_rel dependency), R-Z10 zones
  for deep certs partially upheld (G2-Z1) with FHW-T3 withdrawn pending
  repair (Group-2 zone consumption stays theory-gated and OUT of this
  plan). The engine adopted here is final-shaped; no pending solver
  rewrite blocks integration.
- **Deployment reality (new in v3.1):** main_3 ran the full built rung
  ladder (mode=3 + guard + sharpen + root_guard + async + park, cap 500)
  from ~ep25 to its deliberate stop at ep111. The ep54 `horizon_cut`
  measurement (§5) is from that deployment. Validation now happens
  offline against the **ep90 checkpoint**, not a live run.

**Owner rulings encoded by this doc (2026-07-17 + 2026-07-20):**

1. Certificate-mined one-hot **policy** targets: **REJECTED** (§7).
2. Per-cell solver-class head as candidate **replacement for cell_q** (§7).
3. **Root channel = 100% coverage (decided)**; leaf channel targets
   **maximal coverage — aspirationally 100% ungated** (§4, §6).
4. **Horizon: at least h16, or unbounded with just a node cap (07-20).**
   The v3 "lean" leaf base-h8 is superseded; §5 has the new policy.
5. `has_threats` gate **retired**; the engine's interior census gate is
   the primary filter (§6).
6. Expanded metric set; internalization curve is the program metric (§8).
7. **Validation at the ep90 checkpoint (07-20):** fixed-budget h2h + 
   throughput/park health + a `tss_zone` arm, before any live-run rung.

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

## 3. Engine adoption

- **The leaf/root solver is the campaign engine at the leaf-decided
  config:** wide `vcf_pair_complete` width, lazy=1, gate=1, fragments=0,
  k_reply=0, 256 KiB TT, WIN goal (LOSS side: §3-OPEN below), horizon per
  §5, cap 500 at leaves. Flags recorded verbatim in
  HUNT_REPORT_LEAF_SURFACE.md (solver branch).
- **Where the engine lives:** `claude/tss-vcf-width` @ `ad606d0e`
  (normative tip; includes R-FIX1, incremental defender enumeration,
  lazy frontier, interior census gate, cap-resume, and the extended-
  contract zones P0–P3). The trainer tree (this branch) has ONLY the
  older narrow Stage-3 solver — the port is real work, not a flag flip.
- **Port scope (OPEN-ENG, needs owner sign-off): orchestrator
  recommendation = adopt the campaign engine WHOLESALE** as the single
  mint (the C1 one-engine principle), with `tss_cert_version` tracking it
  through the schema. The alternative (backporting the three levers into
  the trainer solver) creates a permanent two-engine maintenance seam.
  The independent verifier stays untouched either way.
- **Persistent solver per batch + cross-move TT:** validated (13 ms-cliff
  guard PASS in every config). v2 §11 memory hard-caps stand — 256 KiB TT
  per solver, per-search memo freed at move end, certificate buffers
  streamed; the 29 GB WSL ceiling is an acceptance criterion.
- **§3-OPEN: LOSS side.** The leaf campaign measured WIN-goal only, so
  the −1 half of the value signal is structurally underserved at leaves.
  Root channel: dual-seat solve from the start (once per move — cheap).
  Leaf channel: opponent-goal solve gated on opponent-threat features
  (§6); measure before committing. Note the census early-out is WIN-goal
  machinery — LOSS-side solves lack the cheap internal filter until a
  LOSS-side census exists.

## 4. Two channels, two budgets (DECIDED)

**Root channel — the target-maker.** Deep solve at every self-play root
position (both seats), cross-move TT reuse. Because training rows *are*
root positions, this channel annotates **100% of rows directly**: Lever-1
policy masks, Lever-2 proof-corrected values, per-cell class labels (§7),
the disagreement stream, certificate-horizon moves_left (§7). Cost: one
solve per played move — trivial next to a search. **Deploys first.**

**Leaf channel — the search-improver.** Solver at leaf expansion; verified
hard backups + eval elision. Affects targets only indirectly (visit
shifts). **Coverage doctrine: the target is 100% of leaf expansions,
ungated.** Feasibility rests on:

- The interior census gate is an *internal* early-out — quiet positions
  exit the solve almost immediately with a certified no-win-within-h, so
  cost concentrates on genuinely hot leaves (h8 measurement: all 692
  interiors dismissed, ~1,852 nodes flat, p90 wall −93.7%).
- Solves run on CPU in the GPU's shadow (async/park lane — already
  deployed and healthy in main_3 through ep111); the select-phase wall
  timer confirms absorption.
- Every fired verdict *refunds* a GPU eval (elision), and the refund is
  largest exactly where solves fire most (threat-dense endgame, the S²
  ~13× batch-collapse zone) — net cost can go negative there.

R0 must produce the per-solve wall distribution (quiet vs hot, per
horizon arm) and the shadow-absorption measurement. If 100% is
unaffordable at the chosen horizon, fall back to §6 consequence-gated
partial coverage rather than lowering the horizon below the owner floor.

Budgets, flags, and metrics are kept separate per channel. Serve-time root
guard unchanged (deployed rung 8 semantics).

## 5. Horizon policy (owner-ruled 07-20; ladder mechanism absorbed from PLAN_TSS_HORIZON_LADDER)

**The ruling: every solve channel runs at h16 minimum, or unbounded
(`semantic_horizon = u32::MAX`) with the node cap as the only budget.**
The official 2 GiB profile and the atlas already run unbounded+cap; the
trainer's +12 guillotine is the anomaly this section removes.

**The evidence (ep54 clean epoch, `horizon_cut` counter):** of 850,517
deep solves, 6.83% decided; of the Unknowns, **38.25% were depth-cut**
(at least one still-live line refused by the +12 deadline) vs 61.75%
structural. Cap A/B at 500/2000/8000 left decided sets identical — depth,
not width, is the open frontier at the leaf loop. 38.25% is an upper
bound on conversion, but even 2–5% = +10–26% proofs/epoch, concentrated
in the deepest tactics the net can least see.

**Mechanism (the ladder, from the absorbed doc):** re-solve on the same
solver instance at a taller deadline ONLY when the base pass is Unknown
with `horizon_cuts > 0`. The shared TT retains verified positive
fragments, so the proven prefix replays from cache and the budget is
spent on the new plies. Soundness needs NO new theory: every completed
production cert is a forced chain (implicit dispatch at k==B), which is
depth-independent. Downstream (verifier, consumption, park/async) is
untouched; an overrunning tall pass bails to a plain eval.

**Per-channel policy:**

| Channel | Base | Escalation | Notes |
|---|---|---|---|
| Root (target-maker) | **unbounded + node cap** | — | once per move; can afford it; certificate horizon still recorded for moves_left |
| Leaf (search-improver) | **h16** (owner floor) | ladder to h32 / unbounded+cap on `horizon_cuts>0` + hot features | R0 measures h16-flat vs h16→h32 vs unbounded+cap: verdict yield, wall, park health |
| Root guard (serve) | unbounded + node cap | — | inherits root channel |

**Gate counters:** `horizon_cut` (base pass), `horizon_cut_24`-style
counter at the tall rung, `deep_kb_death` (tall pass died at a k<B
defender node — the signal that Group-2 zone consumption would matter;
do NOT build zones ahead of that number, and FHW-T3's repair gates the
theory anyway). **Kill criterion: depth-conversion < ~1% of cut solves →
flat h16, ladder off** (counters remain free telemetry).

**Relief valves** if tall passes crowd capacity: subsample tall passes,
raise `tss_solver_async_threads_max` toward the measured 12–16 worker
ceiling, or lower the leaf node cap — never lower the horizon floor.

## 6. Solve gating — retired as a filter, retained as escalation (DECIDED)

**`has_threats` is retired:** it requires a live ≥4 window to already
exist, but deep wins *begin with threat-creating moves* — a deep win from
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
  semantics). Lever 1 is already deployed (sharpen rung, ep52+); Lever 2
  train-read swap is still unbuilt (rung 9 of the ladder — build at its
  rung).
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
  rule 7; the head doubles as the internalization instrument (§8).
  **OPEN: replace vs add** — verify nothing consumes cell_q at serve,
  then A/B under one `target_regime`.
- **moves_left at proven rows (cheap):** derive the target from the
  certificate horizon instead of behavioral game length; per-head masks
  exist in the v5 schema.

## 8. Metrics (v2 §9 carried; new set below, in priority order)

1. **Internalization curve (THE program metric):** at every root solve,
   the prior's mass on — and rank of — the certificate move, tracked over
   epochs. Rising = the net is absorbing the forced tree; flat = renting
   tactics from the solver forever.
2. **Consequential-verdict rate:** verdicts that flipped a backup vs the
   net eval, or changed the root move choice.
3. **Root-channel coverage:** % rows with a non-Unknown root class; %
   rows where the Lever-1 mask moved mass (KL raw‖masked).
4. **CPU-ms per consequential verdict, by gate feature** (feeds §6
   thresholds).
5. **Depth-frontier counters:** `horizon_cut` per rung, tall-pass
   conversion rate, `deep_kb_death` (§5 gates).
6. **Class-head accuracy on proven cells** (once the §7 head exists).
7. Carried unchanged: verify-failure counter (**must stay 0**),
   proof-vs-outcome disagreement stream, UNKNOWN rate under production
   caps, `opp_coverage`, fan-out / forced-line-depth histograms,
   select-phase wall timer, park health (`park_bailed≈0`).

## 9. Deployment (rungs; mechanics per TSS_RUNBOOK.md)

**Validation phase (now, GPU free, ep90 checkpoint — owner battery 07-20):**

- **V0 — engine port** (§3 scope decision first). Acceptance: the frozen
  compact-h16 regression certificate verifies (R-FIX1 check); cargo +
  pytest suites green; flag-off golden digest bit-identical.
- **V1 — enriched offline soak at ep90** (owner 07-20: maximum useful
  information). Solve + verify + log ungated on a measured slice,
  consume nothing. Arms: h16-flat / h16→h32 ladder / unbounded+cap,
  each ± `tss_zone`. Collect:
  - **Yield & polarity:** WIN-goal at every leaf; dual-seat
    (defender-goal LOSS) on a 1-in-N paired subsample — WIN and LOSS
    counts per arm, plus LOSS-side cost (no census early-out; feeds
    §10.4). Paired narrow-vs-wide on identical leaf sets → verdict
    superiority table (wide-only / narrow-only / both / neither).
  - **Depth:** certificate-depth and forced-chain histograms per arm;
    `horizon_cut` / `horizon_cut_tall` / tall-pass conversion rate;
    `deep_kb_death`; deepest verified win; yields by game-phase
    (stone-count band).
  - **MCTS impact:** would-it-flip rate (backup sign vs ep90 net eval;
    root-move changes); net-vs-proof calibration (net value
    distribution on proven-WIN/proven-LOSS leaves — sign-disagreement
    = tactical-headroom measure); §8 internalization BASELINE (prior
    mass + rank of certificate move at proven roots); eval-elision
    refund count + shadow absorption (select-phase wall) + pos/s
    delta.
  - **Engine economics:** per-solve wall p50/p90/p99 quiet-vs-hot per
    arm; nodes/solve; census-gate interior dismissal rate; TT/memo/
    fragment reuse; park/async health under soak load.
  - **Tactical anchors:** MCTS+solver plays the 19-position forcing
    corpus + spare corpus at move-time budgets (finds the certified
    win? time-to-verdict) — ties trainer strength to the campaign's
    puzzle gate. `zone_nodes` must go nonzero under the ladder
    (zones were inert only at flat +12) — measure the zone delta.
  - Raws + SHA manifest; `deep_verify_failed==0` MUST throughout.
- **V2 — fixed-budget h2h at ep90:** new engine + chosen horizon config
  vs the current narrow engine, matched budget, pentanomial driver;
  plus a `tss_zone=true` arm (the horizon ladder makes zones live where
  the flat +12 measured them inert — see the runbook flag note).
- **V3 — throughput/park-health projection:** pos/s, epoch-time estimate,
  `deep_verify_failed==0`, `park_bailed≈0` — the go/no-go for the
  training relaunch.

**Training phase (at the Phase-3 relaunch, one lever per boundary):**

- **R1 — root-channel consumption.** Lever-1 masks + Lever-2 labels from
  root classes (deep + λ¹), both seats; masks and labels are separate
  rungs (attribution), masks first.
- **R2 — leaf hard-LOSS canary** (eval-elision only; probe avoided lines
  actively — false LOSSes are silent).
- **R3 — leaf hard-WIN canary** with certificate-forced audits.
- **R4 — full leaf rollout** at §5/§6 config.
- **R5 — cell_q → class-head swap** (own `target_regime`, own rung).
- **R6 — serve-time deep root guard** (after R3 trust).

Health gate per rung: the regular eval cadence (pool + Strix + SealBot
h2h), revert by flag + checkpoint, `target_regime` tags on every
target-semantics change.

## 10. Open decisions (owner)

1. **Port scope: DECIDED (owner 2026-07-20) — WHOLESALE adoption** of the
   campaign engine as the single mint (C1 one-engine principle);
   `tss_cert_version` tracks it through the schema. V0 unblocked.
2. **Affordability bar for 100% leaf coverage:** pre-register the pos/s
   regression ceiling + absorption threshold before V1.
3. **Exact leaf horizon shape** (§5): h16-flat vs h16→h32 ladder vs
   unbounded+cap — from the V1/V2 measurements (the floor is ruled; the
   shape is empirical).
4. **Leaf LOSS-side** solving (§3-OPEN): dual-seat at all leaves vs
   opponent-threat-gated.
5. **Class head: replace cell_q or add alongside** (§7).
6. Parked value-only certificate mining: stays parked unless
   `proof_disagreements` stalls (§7).
