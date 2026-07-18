# R-OS2 — live zone/d-stone band-key ordering A/B

Date: 2026-07-18  
Branch / starting HEAD: `claude/tss-vcf-width` / `bb7efa44`  
Scope: default-off live ordering implementation plus official three-arm A/B; no commit created

## Verdict

**NULL for `zone_bound`; REGRESSION for the `d_stone` control. Keep the
implementation default-off.**

All three arms returned the same status at every rung, all eight official rows
solved WIN at the same rung, and every returned certificate passed the strict
verifier. The `zone_bound`, band-1 arm did not clear the promotion bar: summed
winning-row expansions increased from 2,059,290 to 2,065,595 (+0.31%), while
summed winning-row wall decreased from 284.948 s to 283.153 s (-0.63%). The
hardest row, `0l4291i_live`, increased 1,879,611 to 1,889,544 expansions
(+0.53%), despite a small wall decrease. This is a mixed/no-material-win result,
not a safe promotion.

The `d_stone` control is a regression. Summed winning-row expansions increased
to 2,181,545 (+5.94%) and wall increased to 289.068 s (+1.45%). Its 20.05%
expansion win on `zrugh2x` was dominated by the 7.04% hard-row expansion loss on
`0l4291i_live`.

A reveal-prefix follow-on remains **narrowly justified as a default-off
measurement**, not as a production promotion. R-OS1's prefix-2/prefix-4 rank
signal targets classifier work rather than df-pn visit order, and the live key
itself is cheap. The present scheduling NULL means that follow-on must retain
its own expansion/tail gates and remeasure the R-CD1 reveal ceiling; it must not
infer a search win from R-OS1 ranks or be coupled to this flag.

## Default-off implementation

The solve-local configuration is read once when a `WidePnSearch` is created:

- absent, empty, or `TSS_ZONE_ORDER=0`: historical ordering, with
  `TSS_ZONE_ORDER_BAND` ignored and no distance context/key computation;
- `TSS_ZONE_ORDER=1`: exact `zone_bound`, the maximum claimant-support distance
  of the two retained child coordinates;
- `TSS_ZONE_ORDER=2`: exact `d_stone`, the minimum of the same two distances;
- `TSS_ZONE_ORDER_BAND=N`: nonnegative current-PN band, default `0`. The binding
  A/B used `N=1` for both live arms.

The off path calls a separately retained copy of the historical selector. The
live path first asks that selector which child/class would win. It then allows
the distance key to act only inside the following hard boundaries:

1. terminal/refuted semantic filtering is unchanged;
2. the sequential root path preserves tactical, urgent-block, root width-tier,
   and immutable fork-prior classes exactly;
3. the normal Choice path stays in the historical winner's width tier and
   immutable fork-prior class, and admits only current PN values through
   `historical_best_pn + band`;
4. generator rank remains the stable tie-break after the distance key;
5. Universal/defender selection is untouched.

Only retained attacker `Pair` children receive a live key. The key is computed
after stateless classification and unordered-pair dedup, so it cannot change
the generation set or classifier acceptance. One turn-start context collects
the claimant stones; each retained pair scans those stones for the nearest
distance of its two coordinates. Thus the added work is one occupied-cell scan
per attacker pair node plus `2 * claimant_stones` hex-distance evaluations per
retained child. This is cheap in aggregate but is honestly on a pair-generation
hot path when enabled. Sound/lazy reveal behavior is unchanged.

`packages/hexfield_eq/rust/src/tss_verify.rs` is untouched.

## Band preregistration smoke

Before the binding A/B, bands 0 and 1 were compared on four easy official WIN
rows (`0hz3hty`, `acly7kb`, `g2xx6wl`, and
`hayes_20260712_turn16`). Exact ties (`band=0`) changed aggregate expansions
18,254 to 18,240 (-0.08%). `band=1` changed them to 18,001 (-1.39%), including
Hayes 11,663 to 11,402. Band 1 was then fixed for both binding live arms; no
other band was run on the full corpus.

## Binding official A/B

Environment for every arm:

```text
TSS_BACKWALK_TT_BYTES=1073741824
TSS_LAZY_FRONTIER=1
TSS_INTERIOR_CENSUS_GATE=1
```

All other solver levers were off. The standard 10k/100k/1M/20M ladder stopped
at WIN. Values below are winning-rung `expansions / wall_ms`; parenthesized
changes are relative to baseline. Wall is a single run and easy subsecond rows
show substantial host noise, so expansions and the hard rows carry more weight.

| row | baseline | zone bound, band 1 | d_stone, band 1 |
|---|---:|---:|---:|
| `0hz3hty` | 2,411 / 135.0 | 2,365 / 136.6 (-1.91%, +1.19%) | 2,420 / 139.1 (+0.37%, +3.04%) |
| `0l4291i_live` | 1,879,611 / 263,217.7 | 1,889,544 / 261,404.5 (+0.53%, -0.69%) | 2,012,012 / 267,289.8 (+7.04%, +1.55%) |
| `acly7kb` | 74 / 9.3 | 74 / 11.7 (0.00%, +25.81%) | 74 / 14.8 (0.00%, +59.14%) |
| `g2xx6wl` | 4,106 / 591.6 | 4,160 / 864.6 (+1.32%, +46.15%) | 4,262 / 930.9 (+3.80%, +57.35%) |
| `jnzzmcm` | 9,797 / 948.1 | 9,758 / 1,224.0 (-0.40%, +29.10%) | 9,784 / 1,412.7 (-0.13%, +49.00%) |
| `lz60mfb` | 109,895 / 13,226.3 | 107,001 / 12,216.5 (-2.63%, -7.63%) | 108,149 / 13,559.9 (-1.59%, +2.52%) |
| `zrugh2x` | 41,733 / 5,029.4 | 41,291 / 6,011.6 (-1.06%, +19.53%) | 33,365 / 3,840.1 (-20.05%, -23.65%) |
| `hayes_20260712_turn16` | 11,663 / 1,791.0 | 11,402 / 1,283.7 (-2.24%, -28.32%) | 11,479 / 1,880.3 (-1.58%, +4.99%) |
| **winning-row sum** | **2,059,290 / 284,948.4** | **2,065,595 / 283,153.2 (+0.31%, -0.63%)** | **2,181,545 / 289,067.6 (+5.94%, +1.45%)** |

The full 15-step ladder wall (including capped UNKNOWN attempts) was 458.872 s
baseline, 430.870 s zone (-6.10%), and 464.046 s d_stone (+1.13%). The zone
full-ladder decrease is dominated by fixed-cap attempt throughput and is not
treated as a promotion signal in a single-run A/B.

The three tail-risk rows were mixed:

- `0l4291i_live`: zone +0.53% expansions; d_stone +7.04% (dominant control tail);
- `lz60mfb`: zone -2.63%; d_stone -1.59%;
- `hayes_20260712_turn16`: zone -2.24%; d_stone -1.58%.

Status/rung sequences were identical in all arms: `0l4291i_live` was UNKNOWN at
10k/100k/1M then WIN at 20M; `lz60mfb` was UNKNOWN at 10k/100k then WIN at 1M;
`zrugh2x` and Hayes were UNKNOWN at 10k then WIN at 100k; the other four rows
won at 10k. The corpus harness strictly verified every non-UNKNOWN certificate.

## Exact key-computation cost

The corpus harness separately timed only live claimant-context construction and
distance-key scans. It excluded stateless classification, dedup, and the R-OS1
offline observer.

| arm | contexts | retained-pair keys | context ms | key-scan ms | total ms | amortized ns/key | % full-ladder wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| flag off | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000% |
| zone bound | 469,497 | 5,437,128 | 378.824 | 958.336 | 1,337.160 | 245.9 | 0.310% |
| d_stone | 449,280 | 5,589,037 | 410.435 | 1,093.717 | 1,504.152 | 269.1 | 0.324% |

The two live arms use the same distance loop; their count/time differences come
from their different search trajectories. Test-only per-key timers/atomics add
some profiling overhead to flag-on wall and are absent from the production
build. The table reports the timed computation itself, not an inferred delta
from noisy end-to-end wall.

## Default-off identity and validation

The standard identity subset was captured before implementation and rerun with
the flag absent after implementation under the official deep profile:

| row/rung | pre-change expansions | flag-off expansions |
|---|---:|---:|
| `0hz3hty@10k` | 2,411 | 2,411 |
| `acly7kb@10k` | 74 | 74 |
| `mvp2lvc@10k` | 9,999 | 9,999 |
| `mvp2lvc@100k` | 17,956 | 17,956 |
| `mvp2lvc@1M` | 17,956 | 17,956 |
| `xsnfyll@10k` | 81 | 81 |

After removing wall fields, the complete `CORPUS` lines were identical:
status, nodes, expansions, TT entries/hits/bytes, stage refreshes, gate fields,
and seed fields all matched. Flag-off telemetry was exactly zero.

Validation retained in the raws:

- focused release unit for PN band plus urgency/width/fork-prior boundaries: PASS;
- pre/post default-off four-row identity: PASS;
- three official full-corpus arms: PASS, with strict verification on every certificate;
- non-test MSVC release build: PASS;
- direct `rustfmt --edition 2021` on both touched Rust files and
  `git diff --check`: PASS.

Every Cargo invocation used `--target x86_64-pc-windows-msvc` and
`CARGO_TARGET_DIR=.target-codex`, after a fresh gate requiring at least 10 GiB
available, 5 GiB free physical, and zero existing cargo processes. Every gate
passed. PowerShell rendered native Cargo stderr as `NativeCommandError` metadata
in the captured logs; the retained `EXIT_CODE=0` and test/build summaries are
authoritative.

## Retained raw logs

- `ORDERING_WIRE_PRECHANGE_IDENTITY_RAW.log`
- `ORDERING_WIRE_BUILD_RAW.log`
- `ORDERING_WIRE_FLAG_OFF_IDENTITY_RAW.log`
- `ORDERING_WIRE_BAND0_SMOKE_RAW.log`
- `ORDERING_WIRE_BAND1_SMOKE_RAW.log`
- `ORDERING_WIRE_OFFICIAL_BASELINE_RAW.log`
- `ORDERING_WIRE_OFFICIAL_ZONE_RAW.log`
- `ORDERING_WIRE_OFFICIAL_DSTONE_RAW.log`
- `ORDERING_WIRE_PRODUCTION_BUILD_RAW.log`
- `ORDERING_WIRE_RAM_GATES_RAW.log`

All raw logs are UTF-8 and were written incrementally. No verifier, generation
set, sound-reveal, or lazy-reveal change is part of R-OS2.
