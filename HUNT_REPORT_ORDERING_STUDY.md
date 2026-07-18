# R-OS1 — rank-of-winning-child ordering study

Date: 2026-07-18  
Branch / starting HEAD: `claude/tss-vcf-width` / `224a3682`  
Scope: measurement only; no production ordering was wired

## Verdict

**PROMISING, with a material but tail-risky signal. A live wiring A/B is
justified for the exact maximum claimant-support distance (`zone_bound`),
preferably first as a risk-controlled tie/band key. Nothing promotes from this
offline round.**

Across 26,710 proven attacker pair nodes from eight officially solved WIN
rows, the current generation order puts the winning child at rank 1 only
14.96% of the time, has median rank 6, and reaches it by rank 4 only 36.91% of
the time. Thus agenda §1.5's premise is real: winning children are not already
concentrated at rank 1–2.

The best prefix key is `zone_bound`, the smaller of which is better. It moves
the median from 6 to 4, rank-2 CDF from 20.22% to 33.67% (+13.45 pp), and
rank-4 CDF from 36.91% to 58.55% (+21.65 pp). This is enough offline headroom
to justify a live expansion A/B and it materially re-arms a retained-child
reveal prefix of width 2–4.

The qualification is important: the same key worsens mean rank from 8.699 to
10.505 (+20.8%), rank-16 CDF from 87.57% to 82.50%, and the 33+ tail from
1,030/26,710 (3.86%) to 2,723/26,710 (10.19%). It is not safe to infer a wall
win from the prefix gain. `d_stone` is the safer secondary signal: mean rank
improves 4.33% (8.699 to 8.323), top-8 gains 4.07 pp, and the 33+ tail shrinks,
but it has no top-4 gain. A pure global zone sort may trade many cheap wins for
a damaging tail; the live round must measure expansions and wall, not merely
rank-1 hits.

## Prior evidence and code path read

The design was based on:

- `HUNT_REPORT_THRESHOLD_SCALE.md`: scheduling has only a 13.93% non-expansion
  ceiling, while coarser df-pn thresholds caused a 3.2x hard-row expansion
  blow-up. This made tail behavior a first-class guard here.
- R-CD1 (`HUNT_REPORT_CLOSURE_DEBTS.md` and commit `b7e9f36c`): the sound eager
  tail is 7.83% of pair-generation wall under the current order, below the
  11.5% decision bar; its coarse winning-rank histogram already suggested
  depth but could not test alternative keys.
- R-PC1 (report at commit `0415fcec`): pair/planner constant-factor rewrites
  won 30.21% without changing any search count, and pair generation remained
  a primary cost center. The named R-CD1/R-PC1 register text is not folded into
  `docs/PLAN_TSS_SOLVER_UPGRADES.md` at this branch tip, so the canonical
  reports/commits were read alongside that plan's pair-width, prior, zone, and
  census sections.
- `packages/hexfield_eq/rust/src/tss_solver.rs`: the frozen first-candidate
  order, stateless second-candidate enumeration, classifier, unordered-pair
  dedup, retained child order, immutable fork/tau prior, root width tier, and
  dynamic minimum-PN Choice selection were all traced before adding counters.

Consequently this report measures **generation-order rank**, exactly the rank
used by R-CD1's retained-child bins. It does not pretend that generation rank
is the full dynamic df-pn selection score.

## Instrumentation and available features

All additions are `cfg(test)`. `TSS_ORDERING_STUDY=1` is read once per wide
search and defaults off. At generation time each test-only attacker child
stores four small observations; after the search finishes, an offline pass
visits proven Choice entries, identifies the first generation-order child with
PN=0, and computes seven stable counterfactual ranks. The search never reads a
counterfactual rank. The corpus harness retains only one compact record per
proven node: depth, generated-child count, pair/non-pair, and the seven ranks.
It emits aggregate histograms rather than child traces.

Available features, all computed from the turn-start state and child
coordinates:

1. `zone_bound`: maximum, across the one or two child placements, of the
   placement's exact hex distance to its nearest existing claimant stone.
   This is the conservative distance needed to cover the whole child from
   claimant support and is the useful gradient behind a seed-band bound.
2. `census_distance`: `6 - c_after`, where `c_after` is the maximum claimant-
   pure window census after adding the child coordinates, clamped at six.
   The existing WindowStore entries make this a cheap observational scan.
3. `gate_adjacency`: number of child placements (0–2) at hex distance at most
   one from an empty cell in a turn-start defender count-4/5 window—the live
   small-dispatch gate. Larger is ordered first.
4. `d_stone`: minimum, across the child placements, of nearest claimant-stone
   distance. The generator already computes the corresponding one-coordinate
   proximity for widened first candidates; the study generalizes it to the
   complete child.

The single-key orders use each feature as the primary key and retain baseline
generation order for ties. Two composites were admitted:

- `census_zone_composite` = census distance, zone bound, d_stone, descending
  gate adjacency, baseline;
- `zone_gate_composite` = zone bound, descending gate adjacency, census
  distance, d_stone, baseline.

They test threat-maturity-first and locality-first forms without searching a
large key space. Neither composite dominates the single zone key.

## Binding corpus run

Environment: `TSS_BACKWALK_TT_BYTES=1073741824`,
`TSS_LAZY_FRONTIER=1`, `TSS_INTERIOR_CENSUS_GATE=1`,
`TSS_ORDERING_STUDY=1`; other solver levers off. Each row used the standard
10k/100k/1M/20M ladder and stopped at WIN. The MSVC release test used one test
thread and `.target-codex`.

| row | winning cap | expansions | pair records |
|---|---:|---:|---:|
| `0hz3hty` | 10k | 2,411 | 76 |
| `0l4291i_live` | 20M | 1,879,611 | 18,842 |
| `acly7kb` | 10k | 74 | 13 |
| `g2xx6wl` | 10k | 4,106 | 96 |
| `jnzzmcm` | 10k | 9,797 | 550 |
| `lz60mfb` | 1M | 109,895 | 5,656 |
| `zrugh2x` | 100k | 41,733 | 1,080 |
| `hayes_20260712_turn16` | 100k | 11,663 | 397 |

All eight solved WIN. All 15 ladder rows are identical to
`CLOSURE_COUNTER_FULL_OFF_RAW.log` on status, nodes, expansions, TT entries,
TT hits, and peak TT bytes. In particular, `0l4291i_live` did not change at
tip: WIN at the 20M cap with 1,879,611 expansions and peak TT 549,161,606 B.
The instrumented selected-row test took 457.10 s (456.680 s summed solve
wall); its wall is telemetry cost, not an A/B result.

## Aggregate rank results

Pair Choice nodes only; CDF columns are percentages.

| order | rank 1 | rank ≤2 | rank ≤4 | rank ≤8 | rank ≤16 | median | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 14.96 | 20.22 | 36.91 | 70.09 | 87.57 | 6 | 8.699 |
| zone bound | 16.13 | **33.67** | **58.55** | 72.68 | 82.50 | **4** | 10.505 |
| census distance | 16.00 | 21.26 | 35.99 | 62.12 | 87.68 | 7 | 9.124 |
| gate adjacency | 15.19 | 19.77 | 36.65 | 69.98 | 87.57 | 6 | 8.712 |
| d_stone | 15.59 | 20.55 | 36.91 | **74.16** | **88.38** | 6 | **8.323** |
| census→zone composite | **17.65** | 31.71 | 46.36 | **72.73** | 85.11 | 5 | 10.544 |
| zone→gate composite | 16.11 | 30.73 | 52.46 | **72.73** | 82.90 | **4** | 10.589 |

The exact baseline histogram in bins `1,2,3,4,5–8,9–16,17–32,33+` is
`[3997, 1405, 3020, 1436, 8862, 4670, 2290, 1030]`. Median generated child
count is 26 and mean is 48.119, so rank movement is economically meaningful
rather than a collection of binary nodes.

### Depth bands

The useful zone signal is overwhelmingly deep.

| depth band | nodes | order | rank 1 | rank ≤4 | rank ≤8 | median | mean |
|---|---:|---|---:|---:|---:|---:|---:|
| 0–7 | 40 | baseline | 25.00 | 47.50 | 52.50 | 5 | 10.125 |
| 0–7 | 40 | zone bound | 25.00 | 47.50 | 55.00 | 5 | 10.075 |
| 8–15 | 387 | baseline | 47.55 | 58.40 | 76.74 | 3 | 5.974 |
| 8–15 | 387 | zone bound | 39.02 | 62.27 | 71.83 | 2 | 7.752 |
| 16+ | 26,283 | baseline | 14.47 | 36.57 | 70.01 | 6 | 8.737 |
| 16+ | 26,283 | zone bound | 15.78 | **58.52** | 72.72 | **4** | 10.546 |
| 16+ | 26,283 | d_stone | 15.11 | 36.57 | **74.09** | 6 | **8.355** |

Breadth is heterogeneous, not a hard-row-only fairy tale: zone bound changes
`g2xx6wl` median/mean from 10/22.14 to 2/8.63 and improves `zrugh2x` mean
8.58 to 7.15, but hurts `hayes_20260712_turn16` mean 4.13 to 14.73 and
`lz60mfb` mean 12.29 to 14.67. That is why the verdict is promising rather
than promoted.

## Expansion and lazy-reveal interpretation

**Expansion promise:** yes, offline but not quantified as a wall percentage.
The zone key puts an additional 21.65% of proven pair nodes' winners in the
top four and removes two ranks from the median. That clears the requested
"materially forward" criterion. Dynamic PN/DN, root width tier, TT
transpositions, and the worse 33+ tail prevent converting this directly into
an expected expansion percentage. The live A/B must use exact status/count
identity guards and should stop on a hard-row expansion regression.

**Lazy reveal:** re-armed for a retained-child prefix: prefix-2 coverage rises
13.45 pp and prefix-4 rises 21.65 pp. This round does not establish a new
sound classifier-work ceiling, because rejected evaluated pairs are not
children and therefore are absent from these rank records. A cursor-aware live
round must show that the cheap coordinate key can be applied before stateless
pair classification and must remeasure R-CD1's evaluation-ordinal tail. The
present result reopens that measurement; it does not overwrite the old 7.83%
ceiling by arithmetic.

## Recommended live round

Run a separate, default-off scheduling A/B—still no verifier changes—with:

1. exact `zone_bound` as the measured primary candidate because it owns the
   top-2/top-4 gains;
2. a risk-controlled form that preserves existing width/urgency/fork-prior
   classes and uses zone distance only inside an equivalence band;
3. `d_stone` as the conservative control because it improves mean/top-8 and
   shrinks the tail;
4. binding gates on per-row expansions and wall, especially
   `0l4291i_live`, `lz60mfb`, and `hayes_20260712_turn16`, plus a fresh R-CD1
   sound reveal-tail counter.

No ordering is wired in this worktree.

## Integrity and retained raws

- `ORDERING_BUILD_RAW.log`: final synthetic offline-rank unit, PASS.
- `ORDERING_SMOKE_RAW.log`: developmental smoke (before the zone key was
  sharpened from a relay bucket to exact distance); not binding evidence.
- `ORDERING_DEFAULT_OFF_RAW.log`: default-off identity subset, PASS with the
  retained `mvp2lvc` and `xsnfyll` counts.
- `ORDERING_OFFICIAL_RAW.log`: binding eight-row ladder and all aggregate
  histograms, EXIT 0.
- `ORDERING_PRODUCTION_BUILD_RAW.log`: non-test MSVC release build, PASS.
- `ORDERING_RAM_GATES_RAW.log`: gate reading before every Cargo invocation;
  every launch exceeded 10 GiB available and 5 GiB free physical. One initial
  unit capture wrapper stopped on PowerShell's stderr policy before producing
  a test result; the immediately repeated and final unit runs passed.

A workspace-wide `cargo fmt --all -- --check` also reported pre-existing
format drift in unrelated crates. The two touched Rust files pass a direct
`rustfmt --edition 2021 --check`, and `git diff --check` is clean.

All raw logs are UTF-8. `packages/hexfield_eq/rust/src/tss_verify.rs` is
untouched. Only `tss_solver.rs`, `tss_corpus.rs`, this report, and raw logs are
part of R-OS1; no commit was created.

## Regeneration

After verifying no other Cargo process exists and recording a fresh RAM gate
of at least 10 GiB available / 5 GiB free physical:

```powershell
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-codex'
$env:TSS_BACKWALK_TT_BYTES = '1073741824'
$env:TSS_LAZY_FRONTIER = '1'
$env:TSS_INTERIOR_CENSUS_GATE = '1'
$env:TSS_ORDERING_STUDY = '1'
$env:TSS_CORPUS_ID = @(
    '0hz3hty','0l4291i_live','acly7kb','g2xx6wl',
    'jnzzmcm','lz60mfb','zrugh2x','hayes_20260712_turn16'
) -join ','
$env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS = '0'
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER = '1'
$env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME = '0'
$env:TSS_CORPUS_EXPECT_CAP_RESUME = '0'
$env:TSS_CORPUS_EXPECT_LIVE_GE3_SEED = '0'
$env:TSS_CORPUS_EXPECT_CLOSURE_COUNTERS = '0'
$env:TSS_CORPUS_EXPECT_ORDERING_STUDY = '1'
$env:TSS_CORPUS_EXPECT_THRESHOLD_COUNTERS = '0'
$env:TSS_CORPUS_EXPECT_THRESHOLD_DELTA = 'off'

cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture
```
