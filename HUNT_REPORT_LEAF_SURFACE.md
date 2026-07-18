# Phase-3 trainer-leaf surface campaign

Date: 2026-07-17  
Consolidation landing: `b45b9bf00393ccfc5aa34d0a73917745eebbf189`

## Workload model (frozen before measurement)

This campaign models one process receiving batches of nearby MCTS leaves. The
input is the established human corpus at
`E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl`.
The loader retains decisive, legally replayable games. It Fisher-Yates shuffles
the game indices with the established xorshift stream seeded by
`0x9E3779B97F4A7C15` and takes the first 50 games. For each selected game, the
continuing RNG stream chooses one legal start prefix from placement 8 through
the last prefix that leaves room for the full window. The batch is exactly six
nonterminal states: that parent prefix and its next five human-corpus placement
successors. Every state is solved; the harness does not filter by player or
turn phase. Thus the sample is 50 batches and 300 nearby solves per matrix cell,
and naturally contains both `FirstStone` and `SecondStone` positions.

Each configuration sees byte-for-byte the same states in the same order. A
solver is constructed once per game batch, configuration, node cap, and
horizon arm, then reused for all six solves. It is not shared across caps or
horizons. This isolates warm within-batch reuse without leaking proofs between
matrix cells. The caps are 500, 2,000, and 8,000 nodes; the TT/cache ceiling is
exactly 256 KiB (`262144` bytes); the goal is one-sided `SolveGoal::Win`; and
the absolute semantic deadline is the state prefix plus either 8 or 16
placements. There is no wall-clock cutoff on a solve.

The configurations are:

| ID | Engine and levers |
|---|---|
| A | narrow compatibility engine, all four consolidated flags off |
| B | wide PN (`vcf_pair_complete`), all flags off |
| C | B + `TSS_LAZY_FRONTIER=1` |
| D | C + `TSS_INTERIOR_CENSUS_GATE=1` |
| E | D + `TSS_SHARED_FRAGMENTS=1`, retained across the six-state batch |
| F | E + `TSS_K_REPLY_CONSUME=1` |

The consolidation routes K-reply only through the separate round-3
narrow-compat quiet fallback, not through wide PN. Therefore F is first run as
the literal requested E+K configuration and must be an exact no-op. One cheap,
separately labelled diagnostic cohort (cap 2,000, relative horizon 8) compares
round-3 consume with K off/on and shadow telemetry enabled; it measures actual
fallback fires, urgent triggers, retained reply sizes, and trigger wall cost
without mislabelling a different engine as E.

For every solve, a hard status counts only if its certificate is accepted by
the independent `TssVerifier`. The harness stops immediately on a rejected
certificate or a WIN/LOSS contradiction. Along the ordered A→B→C→D→E→F
chain it also requires monotonicity: a previously hard verdict may not become
UNKNOWN and a hard status may not change. UNKNOWN→hard is allowed. Reported
wall values are per-solve median/p90 and the distribution/aggregate of the 50
six-solve batch totals. TT pressure is peak accounted bytes divided by 256 KiB;
TT slot replacements and cap admission rejections are reported separately.
Fragment hits/imports and store replacement/refusal counters are reported for
E. Interior evaluations/dismissals and scan time are reported for D/E/F.

The persistent-reuse gate records table reconfiguration counts after every
solve. Narrow must allocate/configure its shared half-cap only on the first
solve of a batch; wide must not acquire an eager shared table; and the fragment
partition may be configured only on E/F's first solve. Any later per-solve
reconfiguration is a failure. First-position versus successor wall is retained
as a secondary check for the historical 13 ms fresh-table cliff; timing alone
is not used as the structural assertion.

## Results

The final release run completed in 63.41 s. It produced 31 reported cells
(A-E at every cap/horizon plus the literal F cohort), verifier-accepted all 806
hard certificates including the separate K probe, found no contradiction, and
passed the ordered monotonicity and persistent-reuse assertions.

### Verdict rate and wall

Every hard verdict was WIN, as expected for one-sided `SolveGoal::Win`; no LOSS
was emitted. `wall` is per-solve median/p90 in milliseconds, `batch` is the
median/p90 six-solve batch total, and `total` is the sum of solve wall over all
300 solves. Rates have denominator 300.

#### Relative horizon 8

| config | cap | verdicts / rate | wall med / p90 ms | batch med / p90 ms | total ms |
|---|---:|---:|---:|---:|---:|
| A | 500 | 15 / 5.00% | 0.026 / 6.860 | 5.965 / 26.318 | 445.526 |
| B | 500 | 16 / 5.33% | 0.154 / 5.486 | 3.592 / 59.251 | 891.673 |
| C | 500 | 16 / 5.33% | 0.154 / 5.726 | 3.650 / 63.179 | 939.254 |
| D | 500 | 16 / 5.33% | 0.141 / 0.432 | 0.955 / 3.792 | 76.513 |
| E | 500 | 16 / 5.33% | 0.135 / 0.450 | 0.930 / 3.835 | 76.273 |
| A | 2,000 | 15 / 5.00% | 0.026 / 9.662 | 6.668 / 47.243 | 909.936 |
| B | 2,000 | 16 / 5.33% | 0.158 / 5.655 | 3.578 / 136.458 | 1,486.084 |
| C | 2,000 | 16 / 5.33% | 0.165 / 5.742 | 3.912 / 139.417 | 1,653.530 |
| D | 2,000 | 16 / 5.33% | 0.199 / 0.654 | 1.345 / 5.086 | 109.238 |
| E | 2,000 | 16 / 5.33% | 0.201 / 0.645 | 1.339 / 5.529 | 110.777 |
| F literal | 2,000 | 16 / 5.33% | 0.177 / 0.583 | 1.331 / 5.591 | 104.337 |
| A | 8,000 | 16 / 5.33% | 0.033 / 15.092 | 9.599 / 162.599 | 2,601.506 |
| B | 8,000 | 16 / 5.33% | 0.212 / 8.174 | 5.709 / 195.799 | 4,330.778 |
| C | 8,000 | 16 / 5.33% | 0.167 / 5.894 | 4.418 / 222.073 | 3,362.522 |
| D | 8,000 | 16 / 5.33% | 0.135 / 0.432 | 0.893 / 3.723 | 78.011 |
| E | 8,000 | 16 / 5.33% | 0.179 / 0.532 | 1.202 / 5.797 | 104.578 |

#### Relative horizon 16

| config | cap | verdicts / rate | wall med / p90 ms | batch med / p90 ms | total ms |
|---|---:|---:|---:|---:|---:|
| A | 500 | 16 / 5.33% | 0.034 / 12.409 | 9.710 / 45.192 | 761.133 |
| B | 500 | 38 / 12.67% | 0.212 / 11.134 | 7.013 / 121.296 | 1,430.186 |
| C | 500 | 38 / 12.67% | 0.217 / 10.864 | 7.306 / 123.980 | 1,467.144 |
| D | 500 | 39 / 13.00% | 0.212 / 5.904 | 4.353 / 41.871 | 704.706 |
| E | 500 | 39 / 13.00% | 0.218 / 6.866 | 4.401 / 34.292 | 761.910 |
| A | 2,000 | 19 / 6.33% | 0.032 / 36.388 | 14.471 / 131.039 | 2,216.006 |
| B | 2,000 | 38 / 12.67% | 0.180 / 8.213 | 5.883 / 246.905 | 2,479.353 |
| C | 2,000 | 38 / 12.67% | 0.165 / 9.480 | 5.853 / 267.044 | 2,837.710 |
| D | 2,000 | 40 / 13.33% | 0.220 / 5.958 | 4.629 / 44.238 | 763.423 |
| E | 2,000 | 40 / 13.33% | 0.219 / 7.267 | 3.767 / 36.061 | 802.245 |
| A | 8,000 | 21 / 7.00% | 0.035 / 41.319 | 17.484 / 406.008 | 6,106.497 |
| B | 8,000 | 39 / 13.00% | 0.170 / 9.942 | 4.796 / 443.909 | 5,181.412 |
| C | 8,000 | 39 / 13.00% | 0.198 / 9.424 | 6.463 / 618.187 | 5,782.041 |
| D | 8,000 | 40 / 13.33% | 0.155 / 3.838 | 3.085 / 28.273 | 557.019 |
| E | 8,000 | 40 / 13.33% | 0.155 / 4.835 | 3.552 / 25.004 | 560.404 |

The sub-millisecond median is not the economic decision by itself: D is about
0.10-0.19 ms slower at the median than narrow, but removes the multi-millisecond
capped tail. Against A at horizon 8, D changes verdict rate by +0.33 pp / +0.33
pp / 0 pp and aggregate wall by -82.8% / -88.0% / -97.0% at caps 500 / 2,000 /
8,000. Its p90 falls 93.7% / 93.2% / 97.1%. At horizon 16, D changes verdict
rate by +7.67 pp / +7.00 pp / +6.33 pp (2.44x / 2.11x / 1.90x A) and aggregate
wall by -7.4% / -65.6% / -90.9%; its p90 falls 52.4% / 83.6% / 90.7%.

### TT pressure and eviction/admission behavior

Each value is `peak accounted pressure / direct-map evictions / cap admission
rejections`, aggregated over the 300 solves. Wide PN has no replacement
evictions; when full it retains its arena but stops indexing new positions.

| horizon | config | cap 500 | cap 2,000 | cap 8,000 |
|---:|---|---:|---:|---:|
| 8 | A | 27.7% / 1 / 0 | 27.7% / 1 / 0 | 27.7% / 3 / 0 |
| 8 | B | 100.0% / 0 / 1,222 | 100.0% / 0 / 15,443 | 100.0% / 0 / 50,507 |
| 8 | C | 75.7% / 0 / 0 | 100.0% / 0 / 1,461 | 100.0% / 0 / 25,844 |
| 8 | D | 12.3% / 0 / 0 | 12.3% / 0 / 0 | 12.3% / 0 / 0 |
| 8 | E | 12.3% / 0 / 0 | 12.3% / 0 / 0 | 12.3% / 0 / 0 |
| 16 | A | 89.4% / 158 / 0 | 100.0% / 973 / 56 | 100.0% / 3,643 / 959 |
| 16 | B | 100.0% / 0 / 1,005 | 100.0% / 0 / 20,567 | 100.0% / 0 / 62,303 |
| 16 | C | 75.6% / 0 / 0 | 100.0% / 0 / 2,941 | 100.0% / 0 / 31,754 |
| 16 | D | 75.4% / 0 / 0 | 99.9% / 0 / 153 | 99.9% / 0 / 153 |
| 16 | E | 75.4% / 0 / 0 | 99.9% / 0 / 153 | 99.9% / 0 / 153 |

Lazy frontier materially reduces index pressure: versus B its admission
rejections fall 90.5% at h8/cap2k and 48.8% at h8/cap8k, and it avoids all
rejections in both cap-500 arms. It does not improve verdict rate or wall in
this sample. The interior gate is what makes the leaf workload dramatically
cheaper: at h8 D expands 1,852 nodes at every nominal cap versus C's 11,605 /
21,909 / 46,294.

### Gate and fragment telemetry

At h8 the gate evaluated and dismissed 692 interior nodes in every D/E cell;
the total census scan cost was only 0.57-0.77 ms over 300 solves. At h16 it
evaluated/dismissed 1,505/1,013 nodes at cap 500 and 1,677/1,148 at caps 2,000
and 8,000, with 2.00-2.58 ms total scan cost. The gate therefore spends only a
few milliseconds per whole 300-solve cohort to remove the expensive tail.

E had seven fragment lookups, zero hits, and zero imports in every h8 cell; the
store reached two entries / 2,054 bytes. At h16 it had 875 lookups, 22 hits,
and 22 surviving imports (2.51% lookup hit rate). It reduced D's 7,098
expansions to 6,861 at cap 2,000/8,000, but added no hard verdict. The store
reached 32,530 bytes against its 32,768-byte one-eighth partition, 22 resident
entries, 259 admissions, 77 replacements, and 11 refusals. Wall versus D was
within noise at h8, then +0.1% / +2.3% / +3.7% total at h16 caps 500 / 2,000 /
8,000. This workload does show safe warm reuse, but not enough value to spend a
fixed 32 KiB of the 256 KiB leaf budget yet. In the final run E's total wall
versus D ranged from -0.3% to +34.1% at h8 and +0.6% to +8.1% at h16; the
timing spread reinforces that the small node reuse did not become a robust wall
win.

### K-reply and the literal F configuration

Literal F at cap 2,000/h8 was deterministic E behavior: identical verdicts,
nodes, TT, gate, and fragment counters, with zero K shadow records. This is not
a negative K measurement; it confirms the consolidated routing fact that wide
PN never enters the narrow quiet fallback.

The separate round-3 narrow-compat probe did exercise K on all 300 states.
Off/on verdicts were identical (14/300), with every hard certificate accepted.
The on traversal saw 756 fallback fires and 398 urgent consumptions (52.6%);
the median retained set collapsed from 459 legal moves to two. Total wall fell
from 7,338.662 to 6,227.782 ms (-15.14%), median from 21.801 to 17.196 ms, and
p90 from 44.352 to 41.562 ms. Thus leaf urgency is dense and the existing
precheck avoids the deep-profile loss here. However, that route is roughly 85x
E's 0.201 ms median at the matched cap/horizon and is not E plus K. K should
not be asserted in the wide-PN Phase-3 configuration unless it receives a new
route-specific proof and integration seam.

### Persistent-reuse / 13 ms cliff gate

The structural gate passed in all 31 cells. A configured its 256-slot shared
half-cap exactly once per six-solve batch and never rebuilt it on a successor.
B-D configured no eager shared table. E/F configured the 32-slot fragment
partition exactly once per batch and never rebuilt it on a successor. At the
actual 256 KiB budget A's first-position median was 12.3-20.4 microseconds, not
the historical 13 milliseconds observed when a fresh 128 MiB half-table was
zeroed. Successor medians were 32.0-49.8 microseconds because the positions,
not allocations, were harder. The persistent production pattern remains
load-bearing and no per-solve zeroing regression was observed.

## Phase-3 recommendation

Enable **configuration D** for trainer leaves: pair-complete wide PN with lazy
frontier and the interior census gate. For the native relative-horizon-8 query,
use **node cap 500**: it produced all 16 D verdicts found at 2,000 and 8,000,
while delivering 5.33 hard verdicts per 100 solves and 76.513 ms aggregate wall
over 300 solves. Versus narrow A/cap500 this is +0.33 pp verdict rate (+6.7%
relative), -82.8% aggregate wall, and -93.7% p90 wall. Retain cap 2,000 only as
an optional horizon-16 rung: it finds 40/300 (13.33%) versus 39/300 at cap500,
and still costs 65.6% less total wall than narrow A/cap2,000.

Assert:

```text
width = WidthOptions::vcf_pair_complete()
TSS_LAZY_FRONTIER=1
TSS_INTERIOR_CENSUS_GATE=1
TSS_SHARED_FRAGMENTS=0
TSS_K_REPLY_CONSUME=0
goal = SolveGoal::Win
relative_horizon = 8
node_cap = 500
tt_bytes_cap = 262144
```

Keep one persistent `TssSolver` per real leaf batch/worker. Do not reconstruct
it for every solve.

Production integration still owes:

- a leaf-call-site width/profile switch and exact relative-horizon plumbing;
- retention of the interior gate's checked coordinate-safety guard at arbitrary
  trainer positions;
- rollout telemetry for verifier failures, verdict rate, queue/park tail, gate
  dismissals, and the 256 KiB pressure signal;
- a fragment-store size/admission policy study at 256 KiB before enabling E
  (the current one-eighth partition was 99.3% full, replaced 77 entries, and
  yielded no extra verdict here);
- a route-specific K-reply design if wide PN is ever meant to consume it; the
  current flag is a literal no-op on D/E, while round-3 narrow-compat is far too
slow despite a favorable leaf trigger result;
- confirmation on live MCTS-produced leaves, because human-successor windows
  are a faithful locality model but not the trainer's exact policy-induced
  state distribution.

## Regeneration

From this worktree only, with no other Cargo process and free RAM above 9 GiB:

```powershell
$free = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($free -le 9) { throw "free RAM must be above 9 GiB; got $free" }
$env:CARGO_TARGET_DIR='.target-codex'
cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq `
  leaf_surface_campaign -- --ignored --test-threads=1 --nocapture
```

Raw output is `LEAF_SURFACE_RAW.txt`. The test-only harness is
`packages/hexfield_eq/rust/src/tss_leaf_surface_hunt.rs` and is registered only
under `#[cfg(test)]` in `lib.rs`.
