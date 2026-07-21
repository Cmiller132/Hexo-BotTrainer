# V2 — Horizon-Shape Head-to-Head (unbounded vs flat h16)

**Date:** 2026-07-20 · **Gate:** task #14 · **Raw:** `raws/v2_h2h_result.json` · **Driver:** `scripts/_v1_soak/run_v2_h2h.py`

## Question

Does the V1 coverage finding (unbounded horizon = +26% verified wins at identical
p50 latency) translate into **playing strength** under the full production stack?

## Setup

- 256 games, 256 visits, ep90 pinned checkpoint on BOTH arms, paired openings
  (8 plies, temp 1.0, 128 mirrored pairs), seed base 20260720, cuda.
- Both arms run the wholesale-wide production profile (mode 3 WIN+LOSS,
  interior guard, root guard, cap 500, async 8/inline16=4, park 150ms, zone off).
- The ONLY divergence: Arm A `tss_solver_horizon=0` (unbounded + node cap),
  Arm B `tss_solver_horizon=16` flat (engine default at the time). Ladder off
  on both (retired by owner ruling before this match).

## Result

| | games | wins | decided winrate | CI95 |
|---|---|---|---|---|
| **A — unbounded** | 256 | **150** | **0.586** | [0.525, 0.645] |
| B — h16 flat | 256 | 106 | 0.414 | |

- 256/256 completed, 0 truncated, 0 budget-aborted, 0 draws.
- Seat-balanced: A won 74/128 as P0 and 76/128 as P1 — no first-move confound.
- Pentanomial (128 full pairs): A took both games of a pair 38×, split 74×,
  lost both 16×. Pair-winrate mean 0.586, SE 0.028 → z ≈ 3.09, **p ≈ 0.002**.
  The CI excludes 0.5 comfortably.

## Reading

The horizon question is now closed on BOTH axes:

1. **Coverage (V1):** unbounded finds 189 vs 150 verified wins at identical
   p50 wall; the 39 extra wins sit at cert depth 17–22, exactly the region a
   flat h16 truncates.
2. **Strength (V2, this match):** those deeper certificates are worth real
   games — ~59% h2h, +60 Elo-equivalent, at production settings where the
   solver improves search, policy targets, and value targets simultaneously.

The pre-registered concern — that the unbounded arm's fatter grind tail
(73.5% of unknown wall in 248 cap-bound grinds) would cost enough throughput
to erase the coverage edge — did not materialize: the park envelope (150ms)
bounds per-move exposure, so tail cost shows up as background thread
occupancy, not move-time regression.

## Ruling folded in

The owner's 07-20 ruling ("drop the horizon — the slow tail is node-bound,
not depth-bound") predates this result and is now empirically backed from
both directions. **Production profile = unbounded horizon + node cap 500 +
park 150ms. The horizon ladder is retired.** `tss_solver_horizon=0` is the
normative setting; h16 survives only as a historical comparison arm.

## Follow-ups this enables

- The freed GPU goes to: harness bench smoke → first standard-tier harness
  baseline run → V3 throughput go/no-go.
- The grind tail remains the top economics target (fail-fast research,
  cap×TT×warmth×goal Pareto sweep) — but as an efficiency question, not a
  correctness one.
