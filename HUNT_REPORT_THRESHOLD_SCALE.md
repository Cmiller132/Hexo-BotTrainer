# HUNT REPORT — R-TS1/R-TS1b: prior-scale-aware df-pn thresholds (ideation candidate 2)

**VERDICT: CLOSED NULL at Phase 2. The `+1` threshold increment is optimal
for this engine; every coarser increment loses, one catastrophically.
No flag ships; the Phase-1 counter instrumentation is retained
(`TSS_THRESHOLD_COUNTERS`, cfg(test), default-off).**

Provenance: R-TS1 (327k tokens) died on a gpt-5.6-sol capacity outage
mid-implementation; R-TS1b (continuation, 219k tokens) completed ALL
measurement — Phase-1 counters, full Phase-2 A/B, leaf matrix, strict
verification, byte-identity subset, production release build — then died
on a second capacity outage while writing this report. This document was
authored by the orchestrator from the session narratives; **every number
below was independently verified against the retained raw logs** listed
at the end.

## The conjecture (IDEATION_FINAL.md §3, candidate 2)
Proof numbers initialize from fork degree (1..37) and disproof from tau,
but descent still gives the selected child a `second+1` threshold. The
proof-number literature (1+ε) predicts heuristic initial values larger
than 1 make +1 increments too small → excessive internal re-traversal.

## Phase 1 — counters (official 1 GiB lazy+gate profile, +1 baseline)
- Wall with counters on: **499.85 s** (uncontaminated retained baseline
  495.94 s; counter overhead ≈ 0.8%). EXIT 0, all expect-flags asserted.
- Total descent/state time outside expansion: **13.93% of wall**
  (absolute ceiling for any scheduling change).
- Proportional attribution to *revisits*: **34.79 s = 7.01% of wall**.
  Clearing the 5% promote bar would require avoiding ~71% of
  revisit-associated traversal. Judged plausible given the observed
  threshold-cross/sibling-switch counts (e.g. 0l4291i_live @1M:
  1.83M visits / 834k revisits / 1.74M threshold crosses / 1.28M sibling
  switches) → **Phase 2 authorized**. Honest note: this was a ceiling
  argument, not a claimed win.

## Phase 2 — A/B (TSS_THRESHOLD_DELTA, scheduling-only, one global parameter)
Official 1 GiB lazy+gate deep profile:

| arm | wall | verdict |
|---|---|---|
| +1 baseline | 499.85 s | reference |
| delta 2 | **927.59 s (+85.6%)** | catastrophic loss |
| delta 4 | not run (leaf-disqualified) | — |
| mean sibling prior | not run (leaf-disqualified) | — |

Delta-2 root cause: **coarser thresholds destroy frontier accuracy on
the hardest row**. `0l4291i_live` expanded **6,054,588** nodes and
filled the full 1 GiB TT (peak 1,073,741,810 bytes) vs 1,879,611
expansions / ~549 MiB under +1; that row alone went 199.0 s → 627.7 s.
Larger increments let the search stay in a subtree longer (fewer
revisits — the thing Phase 1 measured), but the price is worse global
best-first arbitration: ~3.2× the expansions, TT saturation, and double
the wall. The +1 discipline IS the frontier accuracy; the revisits it
causes are cheaper than the misallocated expansions coarser thresholds
buy. All expected hard results still passed; no corpus failures.

Phase-3 leaf matrix (cap 500, h8+h16, all four arms,
THRESHOLD_LEAF_AB_RAW.log):
- h8: all arms identical — 16 verdicts, 1,852 expansions each.
- h16: +1 = 39 verdicts / 6,649 exp; delta 2 = 39 / 6,726 (worse);
  **delta 4 = 38 / 7,106 (LOSES one hard verdict)**; **mean = 38 / 7,135
  (LOSES one hard verdict)**. Zero contradictions anywhere; all 228 hard
  results across the eight cells strict-verified.
- Delta 4 and mean fail the binding "no material Phase-3 leaf
  regression" conjunction outright, so their deep runs cannot promote
  and were correctly not spent. No fixed delta wins → **ε ladder
  (1/8, 1/4, 1/2) correctly never opened.**

## Why this null is informative (re-test doctrine)
The 1+ε literature concern does not bind this engine: its TT-backed
iterative deepening re-derives interior state cheaply, so revisit cost
(7.01% attributed) is far below the misallocation cost of any coarser
schedule (+85.6% measured). Re-arm condition: only if a future format
change makes revisit cost dominant (e.g. much more expensive per-node
state reconstruction) — NOT from bottleneck movement alone. This closes
the last unposed algorithmic lever from IDEATION_FINAL.md §3.

## Integrity
- Byte-identity fast subset with all flags off: 3.35 s, EXIT 0
  (THRESHOLD_DEFAULT_OFF_RAW.log).
- Production `cargo build --release`: green
  (THRESHOLD_PRODUCTION_BUILD_RAW.log, EXIT 0).
- All instrumentation cfg(test)-gated, default-off (73 cfg(test) sites
  in the diff); strict verifier untouched (no verifier file modified).
- RAM gates recorded in every raw header (relaxed 07-17 protocol; all
  runs launched at ≥17 GB availability / ≥14 GB free-physical).

## Retained raws + regeneration
Raws (worktree root): THRESHOLD_BUILD_RAW.log,
THRESHOLD_COUNTER_SMOKE_RAW.log, THRESHOLD_COUNTER_FULL_RAW.log (+1
official, counters), THRESHOLD_DELTA1_SMOKE_RAW.log,
THRESHOLD_FULL_D1_RAW.log / THRESHOLD_FULL_D2_RAW.log (first-session
partials), THRESHOLD_DELTA2_FULL_RAW.log (delta-2 official),
THRESHOLD_LEAF_AB_RAW.log (8-cell leaf matrix),
THRESHOLD_DEFAULT_OFF_RAW.log, THRESHOLD_PRODUCTION_BUILD_RAW.log.

Regenerate (PowerShell, CARGO_TARGET_DIR=.target-codex, one cargo
host-wide, gate-class RAM gate first): set
TSS_BACKWALK_TT_BYTES=1073741824, TSS_LAZY_FRONTIER=1,
TSS_INTERIOR_CENSUS_GATE=1, TSS_THRESHOLD_COUNTERS=1, optional
TSS_THRESHOLD_DELTA=2|4|mean, matching TSS_CORPUS_EXPECT_* (lazy=1,
gate=1, threshold_counters=1, threshold_delta as set, others=0), then
`cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq
tss_corpus_check -- --ignored --test-threads=1 --nocapture`.
