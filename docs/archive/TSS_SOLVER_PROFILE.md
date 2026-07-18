# TSS solver pre-optimization profile

This is the required pre-change profile for `docs/TSS_SOLVER_OPT_SPEC.md`
section 2.1. It was captured on 2026-07-13 before any solver behavior was
changed, at commit `9fbf061b` plus the benchmark-only corpus extension described
below.

## Method and corpus

- Host: Windows 10.0.26200, AMD64 Family 25 Model 97, 32 logical processors.
- Toolchain: `rustc 1.95.0 (59807616e 2026-04-14)`, release profile.
- Baseline command:
  `cargo test --release -p hexfield_eq tss_bench_report -- --ignored --nocapture`.
- Synthetic corpus: the original fourteen stones-on-board buckets, five
  positions per bucket (70 positions total).
- Curated corpus: four D6 images apiece of the exact Python
  `FORCED_DEFENSE`, `DEEP_WIN`, and `FORCED_LOSS` histories (12 positions).
- Bucket throughput uses `node_cap=100`. The extended-corpus latency pass uses
  `node_cap=2000`. Both use `tt_bytes_cap=65536`.
- The phase profile used temporary `cfg(test)` span markers in the solver. The
  markers contained no clock and only called timing guards implemented in the
  ignored benchmark module. Exclusive wall time and call counts were collected
  in the benchmark, then all temporary hooks were removed. No timing code
  remains in a solve path.

The Windows kernel sampler was unavailable to this non-elevated worktree, so
exclusive bench-owned spans were used instead. They cover the actual recursive
solve, not isolated microbenchmarks. The machine-readable raw outputs from this
run were `target/tss-baseline-extended.txt` and
`target/tss_preopt_profile.txt` (build artifacts, intentionally untracked).

## Per-node cost breakdown

The aggregate rows below are the actual cap-100 calls. Percentages are exclusive
CPU wall time within the measured solver phases; `us/node` divides each phase by
the solver's reported expanded-node count.

| Corpus / family | Nodes | Phase | Calls/node | Exclusive | us/node |
|---|---:|---|---:|---:|---:|
| Extended (82 solves) | 1,075 | omitted-move L1 staple sweep | 0.248 | 91.425% | 2,726.944 |
|  |  | immediate lambda-one analysis | 314.492 | 7.916% | 236.125 |
|  |  | recursive child apply/undo | 1.619 | 0.240% | 7.164 |
|  |  | certificate / other | 0.076 | 0.130% | 3.888 |
|  |  | OR eager ordering | 0.352 | 0.116% | 3.461 |
|  |  | canonical frame | 0.785 | 0.102% | 3.046 |
|  |  | universal legal enumeration + parent analysis | 0.433 | 0.050% | 1.497 |
|  |  | position key + TT | 1.180 | 0.020% | 0.594 |
| `FORCED_DEFENSE` D6x4 | 24 | omitted-move L1 staple sweep | 0.333 | 91.974% | 1,661.342 |
|  |  | immediate lambda-one analysis | 150.000 | 7.528% | 135.983 |
| `DEEP_WIN` D6x4 | 16 | OR eager ordering | 0.500 | 76.839% | 49.825 |
|  |  | immediate lambda-one analysis | 7.000 | 13.610% | 8.825 |
|  |  | canonical frame | 0.500 | 3.027% | 1.963 |
|  |  | recursive child apply/undo | 1.000 | 3.865% | 2.506 |
| `FORCED_LOSS` D6x4 | 4 | immediate lambda-one analysis | 1.000 | 81.564% | 3.650 |
|  |  | certificate / other | 1.000 | 18.436% | 0.825 |

The slow synthetic buckets show the same signature:

| Stones | Total us/node | Staple-sweep share |
|---:|---:|---:|
| 7 | 1,373.9 | 92.36% |
| 8 | 1,963.0 | 93.19% |
| 12 | 2,511.5 | 92.07% |
| 15 | 4,539.9 | 92.20% |
| 16 | 3,804.7 | 91.13% |
| 24 | 8,590.7 | 90.63% |

## Findings against the diagnosed hotspots

1. **Universal expansion: confirmed, with a sharper localization.** The cost is
   not ordinary legal enumeration or the parent analysis (0.050% aggregate).
   It is the search-side loop which applies and lambda-one-analyzes every move
   that the L1 certificate will omit. That loop alone is 91.425% aggregate and
   performs hundreds of analyses per reported node. The independent verifier
   repeats the required per-move check later.
2. **Eager OR ordering: confirmed.** It is hidden by the universal sweep in the
   aggregate, but consumes 76.839% (49.825 us/node) on the exact `DEEP_WIN`
   family, where no universal staple sweep runs. Every candidate is applied and
   analyzed before the first candidate is searched.
3. **Per-solve TT: structurally confirmed, not a cold per-node CPU hotspot.** A
   fresh TT is constructed inside every primal and dual attempt, so no proof can
   survive a solve boundary. Key plus local-TT work is only 0.020% of the cold
   profile. Sharing is therefore a cross-solve discovery/reuse lever rather than
   the cause of the current per-node wall time. A persistent implementation must
   never retain the current solve-local `CertNodeId` values.
4. **Unconditional dual attempt: confirmed by control flow.** With both sides
   requested, the root remainder is always split and the dual runs after an
   unsuccessful primal. This does not dominate a single phase row, but it spends
   up to half the expansion budget on a result a one-sided caller will discard.

## Pre-optimization wall-clock baseline

These are the retained extended-harness cap-100 rows. `PASS` means at least
20,000 nodes/s.

| Bucket | Stones | Positions | Nodes | Nodes/s | Gate |
|---|---:|---:|---:|---:|:---:|
| synthetic | 3 | 5 | 25 | 87,504.4 | PASS |
| synthetic | 4 | 5 | 20 | 159,109.0 | PASS |
| synthetic | 7 | 5 | 71 | 705.6 | FAIL |
| synthetic | 8 | 5 | 210 | 510.9 | FAIL |
| synthetic | 11 | 5 | 21 | 48,287.0 | PASS |
| synthetic | 12 | 5 | 150 | 384.7 | FAIL |
| synthetic | 15 | 5 | 113 | 218.1 | FAIL |
| synthetic | 16 | 5 | 207 | 265.1 | FAIL |
| synthetic | 19 | 5 | 21 | 45,180.7 | PASS |
| synthetic | 20 | 5 | 19 | 68,691.3 | PASS |
| synthetic | 23 | 5 | 21 | 29,093.9 | PASS |
| synthetic | 24 | 5 | 113 | 136.0 | FAIL |
| synthetic | 27 | 5 | 21 | 22,309.6 | PASS |
| synthetic | 28 | 5 | 19 | 23,491.6 | PASS |
| `FORCED_DEFENSE` | 9 | 4 | 24 | 565.7 | FAIL |
| `DEEP_WIN` | 15 | 4 | 16 | 15,081.5 | FAIL |
| `FORCED_LOSS` | 17 | 4 | 4 | 233,918.1 | PASS |

The cap-2000 pass over all 82 positions expanded 12,510 nodes in 158.978 s
(78.7 nodes/s). Median solve time was 0.1585 ms, p95 was 12,168.6472 ms, and
the maximum was 26,428.7016 ms. Thus the median gate happened to pass because
most positions short-circuit, while eight of the seventeen throughput rows
failed and the threat/full-expansion tail was unusable.
