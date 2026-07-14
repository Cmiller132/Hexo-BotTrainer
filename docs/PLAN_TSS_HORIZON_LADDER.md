# PLAN: TSS Horizon Ladder — depth-adaptive deep solving

Status: PROPOSED (gate passed ep54; build not started).
Companion docs: `PLAN_TSS_DEEPENING.md` (the TSS program), `TSS_RUNBOOK.md`
(rung mechanics), zones paper v2 + round-6 proof doc (`E:\hexo-bot` branch
`paper/tss-zones`, commit d3e9d8f).

## 1. What the horizon is

Every deep solve carries two independent limits:

- **`semantic_horizon` — a deadline in game plies.** Fixed at `leaf ply + 12`
  placements (three turns per side; hardcoded in `tree.rs
  tss_solve_verified`). Every line of a proof — including the λ¹ finishing
  sequence at its leaves — must resolve by that absolute ply. Lines still
  alive at the deadline are refused (solver sites: descent past deadline,
  typed-leaf resolution past deadline, completion past deadline).
- **`tss_solver_node_cap` = 500 — a budget in search effort.** Bounds how
  much AND/OR tree may be built chasing that deadline. Binds on *width*
  (defender fan-out), not depth.

The horizon is the only artificial depth limit in the system: the solver's
attacker generator (threat-creating moves only) and defender handling
(hitting universe when fully forced, full legal otherwise) already make
forcedness — not ply count — the natural controller of how deep a line can
go. Forced chains are near-linear in nodes; free defender nodes exhaust the
budget immediately. The +12 guillotine truncates forcing chains that the
budget could otherwise afford to follow.

## 2. The measurement that motivates this (ep54, clean 256-game epoch)

`horizon_cut` counter (commit 6e404bad): Unknown solves whose search had at
least one still-live line refused by the deadline.

| | value |
|---|---|
| deep solves | 850,517 |
| decided (WIN+LOSS) | 58,075 (6.83%) |
| Unknown | 792,442 |
| **horizon_cut (depth-bound)** | **303,097 = 38.25% of Unknowns** |
| structural (defender escaped) | 61.75% of Unknowns |

Context that pins depth as the *only* open frontier at the leaf loop:
- Cap A/B (500/2000/8000, horizon 12): decided sets identical — more nodes
  convert nothing.
- Pool health at cap 500: bail 1–4%, dropped 0 — no throughput constraint.
- So failures are 62% provably-not-depth-fixable and 38% "we stopped looking."

38% is an **upper bound**, not a conversion estimate: a cut line may still
die at ply 13. Even 2–5% conversion = +10–26% proofs/epoch, concentrated in
the deepest tactics — the proofs the net can least see, feeding hard
backups (~9.7% of all MCTS backups), Lever-1 win rows, and the cross-move
memo.

## 3. The ladder design

Pass 1 (unchanged): solve at +12, cap 500. ~65% of solves end here exactly
as today — decided or structurally Unknown. Zero added cost; today's entire
yield preserved.

Pass 2 (new, conditional): **only if pass 1 is Unknown AND `horizon_cuts >
0`** — re-solve at +24 (six turns per side), same solver instance. The
shared TT retains verified positive fragments, so the proven prefix replays
from cache and the budget is spent on plies 13–24.

Everything downstream is untouched: same independent verifier (cert depth
bound is 256; the deadline is cert metadata), same consumption path, same
park/async plumbing (an overrunning second pass bails to a plain eval; the
proof still lands in the memo). Deterministic given the same inputs, so
memo coherence and A/B methodology survive.

Soundness requires **no new theory**: every completed production cert is a
forced chain (implicit dispatch at k==B), which is depth-independent. The
zones-paper machinery is NOT needed for this rung (see §6).

Why +24 rather than unbounded: out-reaches essentially any real forcing
sequence, keeps the corruption-guard meaningful, bounds worst-case chain
wandering; laddering again later is one decision if the counters justify it.

## 4. Cost envelope

- Second passes ≈ 303k/epoch at warm-TT prices → worst case +35–90% node
  demand over the current ~175k nodes/s.
- Measured capacity: ~300k nodes/s at 12 workers (~25k/worker; ep41 retune:
  24 workers collapse to 7.9k/worker — ceiling is 12–16 workers).
- Relief valves if demand crowds capacity: subsample second passes
  (e.g. every 2nd/4th cut solve) or `threads_max` 12→14–16.
- Failure mode is graceful: pool saturation shows up as bail%/coverage loss,
  never as wall-clock stall (the select loop never blocks on the pen).

## 5. Plan of record

1. **DONE — gate.** `horizon_cut` counter deployed at the ep53 boundary;
   ep54 clean epoch measured 38.25%.
2. **Build.** (a) `tss_solver_horizon` as a validated TOML knob (default 12);
   (b) default-off `tss_solver_horizon_ladder` flag implementing pass 2 at
   2× horizon; (c) two phase-2 gate counters: `horizon_cut_24` (still cut at
   +24) and `deep_kb_death` (second pass died at a k<B defender node).
   Verify: cargo suite + flag-off golden digest bit-identity (same harness
   as the counter, commit 6e404bad pattern).
3. **Offline A/B.** Threat-dense corpus (82-position set + sampled
   production horizon-cut positions): conversion % of cut solves at +24 and
   nodes per conversion. **Kill criterion: conversion < ~1% → stop**; the
   counter remains free telemetry.
4. **Production rung.** One-epoch probe at a boundary (config flip +
   boundary kill, supervisor auto-resumes). Watch vs baseline: proofs/epoch
   (~58–67k), nodes/solve (~175), park bail% (1–4%), epoch time (~810–980s),
   `deep_verify_failed` == 0 (must). Revert = flag off.
   Sequencing: after Lever-1 has ≥1 eval point (ep55 eval) so rungs stay
   attributable — one lever per boundary, per program discipline.
5. **Read the next gate.** `horizon_cut_24` high → consider +36 (cheap).
   `deep_kb_death` high → phase 2 below. Both low → depth frontier
   exhausted; bank the win and stop.

## 6. Beyond the ladder — how the solver can improve further

Ranked by evidence, each behind its own gate:

- **Phase 2 — ranked zones at k<B for deep certs** (zones paper v2:
  local-clock T3, ranked zone T4 = Z_dir∪Z_seed∪Z_touch∪Z_virgin with
  per-cell deadlines and the 8(B−1) band, branch-indexed substitution T9,
  DAG certs T10). Paper's L10 explains why production zones were inert at
  +12: certificates whose early attacker placements are threat-creating
  need no virgin seed bands — zones only become *necessary* for deep certs
  passing through genuine-choice defender nodes. The ladder makes that
  regime reachable; `deep_kb_death` measures whether it matters. Do not
  build ahead of that number.
- **Attacker width** (separate project; gate already half-measured): the
  OR-node generator only sees count≥3 windows — connect-6 pair-builds
  through count-2 windows are structurally invisible (external VCF corpus:
  0/14 real forcing wins provable at any cap/horizon). Fixing this is the
  r3-universe posture (proof doc §10) and explodes OR branching — a
  big-budget rung for root-guard/serve or offline harvest, NOT the 500-node
  leaf loop.
- **Constant-factor cleanups, opportunistic**: T6 kernel K_b (refines the
  hitting universe at forced nodes), L13 sparse LOSS witnesses (≤6),
  DAG-shared certs. Sound and cheap; none move the strategic needle
  (verification is not the bottleneck).

## 7. What the revised paper changes, in one paragraph

For the ladder itself: nothing material — forced chains extend soundly at
any depth under machinery already live. Its value is (a) the local-clock
form of T3, which licenses per-path deadlines so a deep branch doesn't
inflate the accounting of the whole cert, and (b) making phase 2 *possible
at all*: without the ranked zone, a deep cert crossing one genuine-choice
defender node means enumerating ~250 legal replies — instant budget death;
with it, roughly tens of cells. The paper converts phase 2 from unsound-or-
hopeless to affordable-if-justified; the ladder's counters decide the
justification.
