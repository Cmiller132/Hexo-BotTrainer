# G2 consume fixed step-zero report

Date: 2026-07-21  
Measured tree: `f01557ca463a021c4eff7090af499cdd06f4b85f` plus the uncommitted,
`cfg(test)`-only instrument described below  
Decision protocol: amended `DESIGN_G2_CONSUME.md` section 7.2

## Result

**MEASURED — NOT KILL; STOP FOR OWNER DECISION.** The fixed current-trace
root-support ceiling is **9.8323%** at Labeling-2k and **24.4847%** on the
Atlas-50k/grind frame. The preregistered rule kills Consume v1 only when
`U_nodes < 10%` at both operating points. Labeling is below 10%, but the deep
grind point is above it. This screen therefore does not kill Consume v1.

**CODE-FACT — no build authorization follows.** This was a one-sided sizing
screen. It cannot promote, authorize, or implement Consume. The independent
producer, FullControl baseline, Open/Closed state machine, and Consume
scheduler remain unbuilt and unauthorized.

| Operating point | Roots | Roots in `E` | All nodes | Nodes in `E` | `U_nodes` | 10% relation |
|---|---:|---:|---:|---:|---:|---|
| Labeling-2k | 6,462 | 4,266 | 1,378,773 | 135,565 | **9.8323%** | below |
| Atlas-50k/grind | 248 | 129 | 4,339,322 | 1,062,472 | **24.4847%** | above |

The exact deterministic formula was:

```text
E = roots reaching at least one exact-positive or indeterminate
    unforced-defender occurrence

U_nodes = sum(SolveStats.nodes for roots in E)
          / sum(SolveStats.nodes for all roots)
```

Charging the whole root makes this an intentionally absurd “eligible roots
become free” ceiling; nested or repeated sites cannot be double-counted.

**CODE-FACT.** This engine has no FullControl baseline and rejects an unforced
native-wide defender node immediately. Therefore this fixed lane measurement
observes `E` on the current reachable trace; it does not certify that a future
FullControl scheduler could reach no additional eligible roots. That limitation
can only enlarge the future `E` relative to this trace. Because the measured
verdict is NOT KILL, this result is not being used to kill a build from a
possibly narrow trace. The amended post-FullControl shadow remains mandatory
if the owner authorizes that next stage.

## Cohort frames

### Labeling-2k

**MEASURED.** This run used all 6,462 frozen rows named by the brief, as a
visible discovery frame:

| Dataset | Roots | SHA-256 |
|---|---:|---|
| `scripts/tss_harness/sets/selfplay_v1.jsonl` | 3,255 | `d8b4256408dfdabf71a90d3653962160bcc05ec66bba580dd6379149d998b708` |
| `scripts/tss_harness/sets/human_v1.jsonl` | 2,720 | `5784defe2531db55360e9860ddddc9b89b148547b16a0c970ff7d83f407c66b6` |
| `scripts/tss_harness/sets/puzzle_v3.jsonl` | 468 | `12b79c6ea132b8d0caa3c2a9108d5830039cd407b2e774670b59a144ea3495e7` |
| `packages/hexfield_eq/rust/corpus/forcing_corpus_moves.txt` | 19 | `89f16724483756ec8e41ba4a03009747ebb4760473a1f4bda75121e1c261f047` |

**CODE-FACT.** Because this screen exposed every row, none of these rows may
later serve as a blind adoption holdout under amended R2. A new untouched or
escrowed adoption holdout is required.

### Atlas-50k/grind

**MEASURED.** This was the exact 248-row `source="grind"` selection in
`raws/lanec_labels.jsonl` (SHA-256
`48bd13ab76d477feffd3067fd18bca41f0e9e30707a505bdc437c9dafc6ecb95`),
resolved to frozen moves in `selfplay_v1`. These are the earlier cap-500 grind
roots and form 41 source-game clusters in this selected frame.

**CODE-FACT.** This is not a complete production Atlas, a random Atlas sample,
or a weighted production estimate. The 24.4847% result is exact for the
248-root grind frame and is scope-limited to it. It cannot satisfy the amended
R2 adoption requirement or close GAP C23.

## Execution profile

**MEASURED.** Both cells used `goal=both`, wide leaf profile, dual pass on,
`loss_reserve=0`, Group-2 off, unbounded semantic horizon, fresh solver/cache
construction per root, and one test thread. Labeling used cap 2,000 and 256 KiB
TT; grind used cap 50,000 and 256 MiB TT. Commands were run from this worktree
with:

```text
CARGO_TARGET_DIR=<worktree>/.target-g2c
cargo test --release -p hexfield_eq --lib \
  --target x86_64-pc-windows-msvc g2_step_zero_measurement -- \
  --ignored --test-threads=1 --nocapture
```

Available physical memory immediately before the cells was 13.74 GiB and
13.94 GiB respectively. Test runtimes were 167.79 s and 706.99 s.

## Classification discipline

**CODE-FACT.** The hook is at the native-wide post-opening defender dispatch,
before the existing unforced refusal. Forced `tau=b` dispatches increment only
the separate `forced_fhw` counter. Non-forced dispatches are checked against
the frozen ordinary `UniversalGroup2V1` producer predicate.

**CODE-FACT.** A predicate positive is logged as exact eligible. A predicate
negative is forcibly logged as indeterminate, not negative, because the
existing shared-helper classifier has not met amended R4's independent
kill-grade negative contract. Thus classifier uncertainty can only enlarge
`E`. Neither run happened to produce an indeterminate occurrence. The screen
observed:

| Operating point | Exact eligible unforced occurrences | Indeterminate unforced | Forced FHW occurrences |
|---|---:|---:|---:|
| Labeling-2k | 4,266 | 0 | 886,675 |
| Atlas-50k/grind | 129 | 0 | 3,221,503 |

Each `E` root reached one exact eligible occurrence in the current trace. The
large forced counts are reported for definition discipline and are excluded
from unforced eligibility; they do not enter `E` by themselves.

**CODE-FACT.** This instrumentation is gated entirely by `cfg(test)`: the new
module, solver hook, observation state, corpus parsing, and JSON writer are
absent from non-test builds. Only test dev-dependencies were added. No Consume
behavior was implemented, no default flag path changed, and `tss_verify.rs`
was not edited.

## Distributions

Quantiles below are empirical nearest-rank quantiles over individual roots.
Cluster-share quantiles first aggregate all rows in a source game/family, then
take the unweighted nearest-rank quantile of that cluster's `E_nodes / nodes`.
These distributions describe the frozen frames; they are not confidence
intervals or production-weighted Atlas estimates.

### Root node totals

| Frame / roots | min | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Labeling, all | 1 | 3 | 3 | 41 | 761 | 2,000 | 2,000 | 2,000 |
| Labeling, `E` | 3 | 3 | 3 | 3 | 27 | 95 | 822 | 1,964 |
| Grind, all | 525 | 1,347 | 5,308 | 41,653 | 50,000 | 50,000 | 50,000 | 50,000 |
| Grind, `E` | 530 | 1,279 | 2,847 | 10,946 | 28,459 | 35,661 | 43,115 | 46,857 |

### Game/family clustering

| Frame | Clusters | roots/cluster min | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| Labeling | 719 | 1 | 8 | 10 | 86 | 88 |
| Grind | 41 | 1 | 6 | 11 | 20 | 20 |

| Frame cluster `U_nodes` | min | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Labeling | 0% | 0% | 0.2593% | 16.7051% | 91.1392% | 100% | 100% | 100% |
| Grind | 0% | 2.3698% | 19.7615% | 73.9670% | 100% | 100% | 100% | 100% |

The broad cluster distributions show why individual-root resampling would be
misleading and why amended R2 uses game/family clusters.

### Labeling composition

| Dataset | Roots | `E` roots | All nodes | `E` nodes | Dataset `U_nodes` |
|---|---:|---:|---:|---:|---:|
| selfplay_v1 | 3,255 | 2,652 | 562,632 | 90,244 | 16.0396% |
| human_v1 | 2,720 | 1,610 | 429,323 | 44,987 | 10.4786% |
| puzzle_v3 | 468 | 3 | 360,953 | 107 | 0.0296% |
| forcing corpus | 19 | 1 | 25,865 | 227 | 0.8776% |

Statuses were 880 Win, 603 Loss, and 4,979 Unknown at Labeling-2k; the grind
frame produced 59 Win and 189 Unknown. Status is descriptive only and does not
alter the fixed all-root denominator.

## Raw artifacts

| Artifact | SHA-256 |
|---|---|
| `.gate/g2-step-zero/labeling-2k.jsonl` | `5229056f134bc9f030104bf788facf9ab44c37f9d1d9f7f5074fd58a22c7d0aa` |
| `.gate/g2-step-zero/labeling-2k-cargo.log` | `f4b8e09bba1c4bde750438015f183ccde1323901b8c11e396773d24f69b27ac8` |
| `.gate/g2-step-zero/atlas-50k.jsonl` | `9ee09047f7c3a1a4e27647f04123288fb93db5360abc9fd1d298a85306e71604` |
| `.gate/g2-step-zero/atlas-50k-cargo.log` | `ea2cbe39ca41dfd9e72e940ae57a4087cad9ed8ec85f896fcde691725fe4ea35` |

Each JSONL row contains profile, dataset, root id, source cluster, cap, TT
bytes, exact node count, status, wall time, exact/indeterminate unforced site
counts, forced-site count, and `root_in_e`.

## Engine-lineage transfer

**CODE-FACT.** The measured worktree is the `fcea3c69` engine lineage and does
not contain candidate-generation fold `63b34cbb` as an ancestor. The later
production fold reported a 1.72x generation-wall improvement with a
bit-identical battery and node-count identity. Therefore these node-share
ratios transfer to that fold; wall times in this report do not.

## Final disposition

**MEASURED — the fixed lane step-zero verdict is `NOT KILL / STOP`.** Labeling-2k
alone falls below the 10% target, but Atlas-50k/grind leaves a measured
24.4847% conservative node ceiling. Per the preregistered rule, work stops at
the measured ceiling. Whether to authorize any subsequent FullControlShadow-PC
or Consume build is solely an owner decision.

**HYPOTHESIS.** The deep ceiling may or may not convert into realizable savings:
it grants every affected root zero cost and does not measure closure,
verification, fallback, or end-to-end batch economics. Only the amended
plan-complete kill screen and, if separately authorized, held-out causal A/B
could narrow that uncertainty.
