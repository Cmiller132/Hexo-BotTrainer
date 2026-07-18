# HUNT REPORT — R-T1.1 frontier-band response census

**E1 verdict: SPLIT.** On `0l4291i_live`, delta 2 reproduced the retained
6,054,588-expansion solve. Its first indexed-TT admission refusal was
timestamped at expansion 3,586,288. Relative to the unchanged +1 solve at
1,879,611 expansions, the 4,174,977 extra expansions decompose exactly into:

- **1,706,677 before first refusal: 40.878716% of the excess**;
- **2,468,300 after first refusal: 59.121284% of the excess**.

The pre-refusal excess is concentrated in the competitive frontier: 1,419,680
of those expansions (83.183871%) are attached to selections that began tied
with the second-best sibling. Saturation is therefore not a prerequisite for
the catastrophe, but most of the measured excess occurs after admission is
already saturated. This temporal census supports both halves of E1; it does
not by itself turn the post-refusal portion into a measured count of D1
duplicate semantic keys.

## Scope and provenance

- Worktree: `E:\Hexo-BotTrainer-hexgt\.claude\worktrees\tss-vcf-width`
- Branch: `claude/tss-vcf-width`
- Measured HEAD: `71fd05e49c18652f7aaf21a0abc4291f9bc8563c`
- Row: only `0l4291i_live`
- Ladder: the official fresh-solve 10k, 100k, 1M, 20M corpus ladder; results
  below use the solving 20M rung.
- Both arms exported `TSS_BACKWALK_TT_BYTES=1073741824`,
  `TSS_LAZY_FRONTIER=1`, `TSS_INTERIOR_CENSUS_GATE=1`,
  `TSS_INCR_DEFENDER=1`, and `TSS_THRESHOLD_COUNTERS=1`. The +1 arm left
  `TSS_THRESHOLD_DELTA` unset (`threshold_delta=off`); the other set it to 2.
  The current Rust tree contains no reader for `TSS_INCR_DEFENDER`; it was
  exported as required but is inert at this tip.
- Strict certificate verification remained in the existing corpus harness and
  passed for both WIN results. No verifier file was changed.

## Counter definition

All new state and observations are `cfg(test)`-gated and default off with
`TSS_THRESHOLD_COUNTERS`:

1. The already-retained first-refusal event in `insert_position` now also
   records `(expansion clock, retained arena entries, indexed bytes)`.
2. Each actual expansion is charged to its immediate parent's selection-time
   gap. Choice uses selected PN minus second-best PN; Universal uses the exact
   dual, selected DN minus second-best DN. The 13 bins are `no sibling`,
   selected better by `33+`, `17-32`, `9-16`, `5-8`, `3-4`, `2`, `1`, `tie`,
   and selected worse by `1`, `2`, `3-4`, `5+`. The root is explicitly
   unclassified. Histograms are split according to whether that expansion
   began before or after the first refusal.
3. Sentinel accounting separately records inherited PN/DN threshold hits and
   strict clamp events, `value + delta` sentinel hits and strict clamps, and
   Choice-DN/Universal-PN branch-sum hits. Sum hits count recurrence
   evaluations, not unique nodes or expansions.

Operationally, a refusal that occurs while expansion `e` is materializing its
frontier is timestamped `e`; that expansion began in the pre-refusal segment,
and expansion `e+1` is the first post-refusal expansion.

## Run results

| arm | status | expansions | indexed entries | peak indexed bytes | first refusal |
|---|---:|---:|---:|---:|---:|
| production +1 (`None`) | WIN | 1,879,611 | 1,879,574 | 549,161,606 | none |
| delta 2 | WIN | 6,054,588 | 3,586,248 | 1,073,741,810 | `(3,586,288, 3,586,249, 1,073,741,810)` |

The delta-2 index stopped 14 bytes below the 1,073,741,824-byte cap because
the next key did not fit. The marker's arena-entry count includes the newly
retained but refused edge-local entry; the indexed-entry total does not.

Both expansion totals, indexed totals, and peak bytes reproduce R-TS1 exactly.
The final-rung row times were 259.5 s (+1) and 814.2 s (delta 2), versus the
older 199.0 s and 627.7 s observations. The launch snapshots showed 0% and 3%
CPU load and no competing cargo process, but the census adds hot-path work and
runtime contention was not continuously sampled. Timing is therefore reported
only as contamination context and is not used in the verdict.

## Exact excess attribution

Let `B=1,879,611`, `S=3,586,288`, and `D=6,054,588`. Since +1 never refused an
admission, the temporal decomposition is:

```text
total excess       = D - B       = 4,174,977
pre-refusal excess = S - B       = 1,706,677 = 40.878716%
post-refusal work  = D - S       = 2,468,300 = 59.121284%
```

The histograms conserve expansions exactly:

- +1: 1,879,610 classified + 1 root = 1,879,611;
- delta-2 pre: 3,586,287 classified + 1 root = 3,586,288;
- delta-2 post: 2,468,300 classified = 2,468,300.

Thus no expansion is lost between the phase marker and the band census.

## Selection-time band histogram

Choice PN and Universal DN counts are combined below because they are the
min-selected competitive quantities in the dual recurrences. “Pre delta” is
`delta-2 pre` minus the complete +1 run, so the column sums to the exact
1,706,677 pre-refusal excess. Percentages are shares of the corresponding
classified segment (the single root is excluded).

| selected score versus second best | +1 count | +1 % | delta-2 pre | pre % | pre delta | delta-2 post | post % |
|---|---:|---:|---:|---:|---:|---:|---:|
| no sibling | 53,042 | 2.822% | 112,713 | 3.143% | +59,671 | 104,143 | 4.219% |
| better by 33+ | 71,340 | 3.795% | 140,415 | 3.915% | +69,075 | 131,216 | 5.316% |
| better by 17–32 | 51,055 | 2.716% | 103,377 | 2.883% | +52,322 | 68,347 | 2.769% |
| better by 9–16 | 551 | 0.029% | 2,347 | 0.065% | +1,796 | 141 | 0.006% |
| better by 5–8 | 845 | 0.045% | 3,518 | 0.098% | +2,673 | 152 | 0.006% |
| better by 3–4 | 151 | 0.008% | 1,069 | 0.030% | +918 | 29 | 0.001% |
| better by 2 | 1,474 | 0.078% | 4,594 | 0.128% | +3,120 | 1,663 | 0.067% |
| better by 1 | 110,832 | 5.897% | 208,254 | 5.807% | +97,422 | 129,764 | 5.257% |
| **tie** | **1,590,320** | **84.609%** | **3,010,000** | **83.931%** | **+1,419,680** | **2,032,845** | **82.358%** |
| worse by 1, 2, 3–4, or 5+ | 0 | 0% | 0 | 0% | 0 | 0 | 0% |
| **total classified** | **1,879,610** | **100%** | **3,586,287** | **100%** | **+1,706,677** | **2,468,300** | **100%** |

The zero “selected worse” bins are expected for this trace: policy never
selected a child numerically worse than its second-best sibling at a selection
boundary. Delta 2's extra unit is instead spent inside calls launched from a
tie or from an already-better child. Tie-launched calls alone explain 83.184%
of the pre-refusal excess and 82.696% of the total excess after including their
post-refusal delta. This is strong evidence for a high-mass competitive band,
not a literal observation of the abstract T2 gadget.

## Sentinel control (review Findings 1 and 6)

| event counter | +1 (`None`) | delta 2 |
|---|---:|---:|
| inherited PN threshold hits | 0 | 42 |
| inherited DN threshold hits | 0 | 42 |
| inherited PN strict clamps | 0 | 42 |
| inherited DN strict clamps | 0 | 42 |
| threshold-increment hits | 0 | 128,957 |
| threshold-increment strict clamps | 0 | 128,957 |
| Choice DN sum hits | 104,916 | 190,167 |
| Universal PN sum hits | 337,172 | 999,482 |

The missing Finding-6 control is closed with a nonzero answer: **delta 2 did
reach and actively clamp at `PN_INFINITY`**. In particular, all 84 inherited
threshold hits and all 128,957 increment hits were strict reductions from a
value above the finite sentinel, while live branch sums also reached the
sentinel in both arms. Therefore the off-versus-2 run cannot be described as a
literal one-expression A/B. The band evidence remains real, but causal wording
must preserve the finite-sentinel caveat.

## E1 decision

**SPLIT**, with exact fractions **40.878716% pre-refusal / 59.121284%
post-refusal**.

- Clause (a) is supported: 1,706,677 excess expansions already exist before
  any admission refusal, and 83.184% of that excess is tied-band work at the
  selection boundary. Admission saturation cannot be the original cause.
- Clause (b) is temporally supported: 2,468,300 additional expansions occur
  after loss of eligibility for new indexed reuse, the majority (59.121%) of
  all excess. This census does not distinguish D1 duplicate copies from work
  that the widened schedule would also have performed with an unlimited index,
  so “D1 materially caused all post-refusal work” remains a conjectural causal
  gloss rather than a measured fact.
- The sentinel counters prevent exclusive attribution of the pre-refusal
  schedule difference to `second_best + 2`; clamping was active in delta 2.

The theory document needs a **measurement/status erratum (or addendum), not a
formal-theorem erratum**: ledger E1 should change from undivided CONJECTURE to
the measured SPLIT result above, and §13's “missing delta-2 sentinel-hit
counter” note should be replaced by these nonzero counts. T2 and D1 remain
unchanged as formal results, and the production D1 causal composition remains
SKETCH.

## Validation and retained raws

- Release test build: EXIT 0.
- Counter smoke: EXIT 0; histogram conservation checked at runtime output.
- +1 hard row: EXIT 0; strict-verified WIN.
- Delta-2 hard row: EXIT 0; strict-verified WIN.
- Production release build for `x86_64-pc-windows-msvc`: EXIT 0.
- Every cargo command used `CARGO_TARGET_DIR=.target-codex`, and every command
  was preceded by the required availability >=10 GiB / free-physical >=5 GiB
  gate. All gates passed.
- Raw logs are UTF-8 without BOM or NUL bytes (avoiding PowerShell's UTF-16
  redirection pitfall):

| raw | SHA-256 |
|---|---|
| `FRONTIER_CENSUS_BUILD_RAW.log` | `AB5703BC949CEF0E1DFCEB89316D6B6775EF9138C804DE5CEFFF19DD2E4AF6B7` |
| `FRONTIER_CENSUS_SMOKE_RAW.log` | `AC451CD41265B6515AB902D2E5B64D5E204882A232EAEAEFF89DDA69B9131D09` |
| `FRONTIER_CENSUS_PLUS1_RAW.log` | `3F460C7098F0DCC9D8F95A1DC51DAC58BB80878878380E3D774B6B12BB31B428` |
| `FRONTIER_CENSUS_DELTA2_RAW.log` | `0F2A80C82CFBD20E0D032E64042DD7DCF1A78B45D3AE23C982B20F99FE232F9C` |

No commit was created.
