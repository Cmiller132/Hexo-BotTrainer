# R-RESIDUE-IMPL wall-time residue map

Status: **OPEN**. Manifest SHA-256:
`CBE228E0573F6FD8FDEC3268CDD8E32E8658DC087CC2A3ED3FB896F7F70494E5`.
Source: `claude/tss-vcf-width` at `ec73f263`, schema 1, Rust 1.95.0,
`x86_64-pc-windows-msvc`, one test thread, High performance power policy.

## Validity and corpus semantics

Three clean process repetitions completed for the official F19, S2, and
frozen 160-position human cohorts. All 588 JSONL jobs passed the exclusive
partition and aggregate cross-checks with signed and absolute residual exactly
zero. Status, nodes, expansions, TT hits/entries, certificate node/edge shape,
strict-verifier result, rung stopping, and instrumentation event counts were
identical across repetitions. All 90 hard-result jobs across the raw protocol
rows were accepted by the strict verifier.

The human positions file was read only. All 160 positions replayed legally in
all three runs, so `skipped=0`. Efficiency results below use F19/S2 only;
human results are prevalence/economics only. F19 used `vcf_pair_complete`, the
10k/100k/1M/20M ladder, 2 GiB TT, and stopped NO controls at 1M. S2 used the
canonical 1M/512 MiB profile. Human used `round3_consume`, 50k, 256 MiB, and
relative horizon 10.

## Instrumentation overhead

The accepted matched A/A gate used seven complete F19 pairs with alternating
order and three S2 pairs. It checked exact semantic/certificate identity on
every pair.

| profile | disabled median ms | enabled median ms | overhead | events | ns/event | p95 row regression |
|---|---:|---:|---:|---:|---:|---:|
| F19 | 479809.700 | 483794.452 | **0.830486%** | 83,563,534 | 47.685 | 1.134687% |
| S2 | 0.315 | 0.292 | **-7.428571%** | 20 | noise-limited | -7.654836% |

F19 meets the 1% target and both profiles meet the 2% hard budget. The p95
row regression is below 5%. The timer uses Windows
`QueryPerformanceCounter`: raw monotonic ticks are accumulated directly by
the exclusive stack and each of the 17 totals is converted to nanoseconds once
at job end. No category subtraction, normalization, or post-hoc scaling is
used. The seven individual F19 samples are retained in
`.codex-residue/overhead-rep0.jsonl`.

## Aggregate results

The authoritative generated
[`aggregate-tables.md`](.codex-residue/aggregate-tables.md) contains all 17
categories for F19 protocol wall, F19 final attempts, S2, and human 160. It
reports median sums, wall share, logical-row-median p90/p95/max and maximum row
ID, disposition, estimate method, and eliminability upper bound.

### F19 protocol wall (34 jobs/repetition; median 483,872.317 ms)

| category | median sum ms | wall % | p95 row % | disposition/estimate |
|---|---:|---:|---:|---|
| D_FORCED_GEN | 174223.582 | 36.006107 | 38.140333 | implemented; ceiling only |
| A_OR_GEN | 222188.374 | 45.918802 | 67.960161 | implemented; ceiling only |
| A_OR_WINNER_PATH | 4497.675 | 0.929517 | 5.455545 | productive ceiling |
| A_OR_ORDERING_MISS | 8870.335 | 1.833197 | 12.041170 | OPEN; direct measured wall |
| A_OR_UNRESOLVED | 58772.403 | 12.146263 | 33.032769 | OPEN; `NOT_MEASURED` |
| TT_PROBE | 2930.948 | 0.605728 | 0.627870 | OPEN; `NOT_MEASURED` |
| TT_STORE | 4426.020 | 0.914708 | 1.025891 | OPEN; `NOT_MEASURED` |
| SEARCH_BOOKKEEPING | 4682.664 | 0.967748 | 5.715745 | OPEN; `NOT_MEASURED` |
| CERT_BUILD / CERT_VERIFY | 3282.659 | 0.678414 | — | verified/mandatory |
| OTHER_MEASURED | 0.379 | 0.000078 | **55.208333** | direct; row-tail failure |

F19 final-attempt wall is 335,452.817 ms. Its largest open measured central
estimate is ordering miss at 1.771006%; unresolved search is a 12.578226%
ceiling with no counterfactual central estimate. The F19 Other maximum is
56.281407% on `8is963b@10k`, an immediate-result micro-row.

### S2 (2 jobs/repetition; median 0.480 ms)

| category | median sum ms | wall % | max row % |
|---|---:|---:|---:|
| A_OR_GEN | 0.375 | 78.187500 | 78.736842 |
| TT_PROBE | 0.002 | 0.354167 | 0.397661 |
| TT_STORE | 0.006 | 1.208333 | 1.286550 |
| SEARCH_BOOKKEEPING | 0.065 | 13.520833 | 15.619048 |
| CERT_BUILD | 0.008 | 1.645833 | 1.732321 |
| OTHER_MEASURED | 0.023 | **4.729167** | **4.904695** |

S2 is an explicit closure residual: direct Other exceeds both the 1%
aggregate and 2% per-row criteria. These controls terminate in under 0.5 ms,
so session/result setup dominates their wall.

### Human 160 (160 jobs/repetition; median 156,709.042 ms)

| category | median sum ms | wall % | p95 row % | interpretation |
|---|---:|---:|---:|---|
| D_FORCED_GEN | 271.706 | 0.173383 | 0.717362 | forced prevalence |
| D_UNFORCED_UNCLASSIFIED_GEN | 63398.570 | **40.456230** | 53.113044 | audit debt; 16,688,646 events |
| A_OR_GEN | 173.878 | 0.110956 | 0.225844 | generation economics |
| A_OR_WINNER_PATH | 0.475 | 0.000303 | 0.005371 | productive |
| A_OR_UNRESOLVED | 61775.218 | 39.420328 | 56.323300 | unresolved ceiling |
| TT_PROBE / TT_STORE | 3471.317 | 2.215135 | — | local ceilings |
| SEARCH_BOOKKEEPING | 26318.861 | 16.794731 | 21.220173 | scheduler ceiling |
| CERT_VERIFY | 0.480 | 0.000307 | 0.198047 | strict accepted replay |
| OTHER_MEASURED | 1315.947 | 0.839739 | **3.808505** | aggregate passes; tail fails |

The human Other maximum is 64.259928% on
`human_b5bfa0cdcfe9b56b_p21`. FHW-eligible and classified non-FHW generation
are exactly zero because this revision has no reviewed frozen FHW selector.
Missing class information is correctly unclassified, never silently treated
as ineligible. Therefore no FHW consume-simulation estimate or FHW closure
claim exists. Each human repetition yielded 9 WIN, 5 LOSS, and 146 UNKNOWN.

## Orthogonal tag table

| horizon/cap tag | attempts | cuts | conversions | wall ms | wall % | deep_kb_death | verified yield |
|---|---:|---:|---:|---:|---:|---:|---:|
| F19/unbounded protocol | 34/rep | 0 | 0 | 483872.317 | 100 | 0 | final: 14 WIN + 2 LOSS + 3 UNKNOWN |
| S2/base | 2/rep | 0 | 0 | 0.480 | 100 | 0 | 0 hard certificates |
| human/base_rel10 | 160/rep | 0 | 0 | 156709.042 | 100 | 0 | 9 WIN + 5 LOSS + 146 UNKNOWN |

## Seam refinements and deviations

- The current wide engine is recursive df-pn. Choice-child exclusive time is
  captured at the recursive `work` seam, temporarily keyed by Choice edge,
  and relabeled only after final proof numbers are known. Nested named scopes
  pause the edge frame.
- Narrow compatibility Choice work uses an observation-only temporary tracker;
  incomplete/refuted restricted searches are unresolved, never ordering miss.
- TT primitives deduplicate same-category nesting. Key construction outside a
  timed primitive remains search bookkeeping rather than being reassigned by
  subtraction.
- Cap-resume orchestration is timed separately from continued search; temporary
  Choice edges are finalized at both fresh and resumed search exits.
- The frozen campaign runner lives in the cargo-enabled `tss_residue` test lane
  rather than changing legacy corpus printers. This preserves default-off
  production output while measuring solve, certificate build, and strict
  verification inside one job wall.
- No trainer projection was pooled into these aggregates: the mission required
  official forcing/spare efficiency and the separate frozen human economics
  cohort. This deviation from the broader spec table is explicit.

## Closure refusal

> **OPEN.** On profile
> `CBE228E0573F6FD8FDEC3268CDD8E32E8658DC087CC2A3ED3FB896F7F70494E5`,
> three clean repetitions of F19/S2 and human 160 accounted for 100.000000%
> of measured wall with direct aggregate `OTHER_MEASURED` of 0.000078%,
> 4.729167%, and 0.839739%, respectively, zero cross-check error, and F19
> instrumentation overhead 0.830486%. Closure is refused because F19, S2,
> and human contain rows above the 2% Other criterion; S2 aggregate Other is
> above 1%; human `D_UNFORCED_UNCLASSIFIED_GEN=40.456230%` with 16,688,646
> unclassified events; no FHW consume-simulation estimate exists; and several
> OPEN categories have ceilings but no measured central counterfactual.
> Consequently neither `L` nor `R_open` is published as a closure bound.

## Raw artifacts and tests

The nine `f19-rep[0-2].jsonl`, `s2-rep[0-2].jsonl`, and
`human160-rep[0-2].jsonl` files contain every machine-readable per-job row.
Generated [`per-job.md`](.codex-residue/per-job.md) contains the required 588
Markdown rows with milliseconds and `% job_wall` in every timing cell. The
aggregate detail, overhead raw, generators, manifest, report, and nine raws are
bound by `.codex-residue/SHA256SUMS.txt`.

Both complete release suites passed with and without `tss-residue`: 118 passed,
40 ignored, zero failed in each configuration. The required focused accounting,
parser/closure-refusal, verification-wall, cap-resume, three cohort campaigns,
and full overhead gate also passed. The verifier diff is ten added timing-only
lines and contains no acceptance-rule change.
