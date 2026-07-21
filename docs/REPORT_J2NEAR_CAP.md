# J2near cap-headroom report

Date: 2026-07-21  
Lane: `claude/j2near-cap`  
Recommendation: **raise the production node cap from 500 to 750 and keep
J2near off**

## Executive result

**MEASURED.** Cap 750 with J2near off is strictly better than the old
production point. It preserves all 1,212 archive-decided rows with the same
verdict, adds 89 strictly verified decisions (57 WIN and 32 LOSS), and takes a
45.956 s median solve wall over the 6,443-row battery. The three solve-wall
observations are 46.730, 45.956, and 44.689 s, all below the old 49.96 s wall.
It consumes 730,143 nodes per battery, or 62,941 ns/node at the median wall.

**MEASURED.** No J2near-on arm is a strictly-better production point in the
tested grid. At caps 500, 640, and 750 it changes the archived verified WIN
`human_41e2eecefcb26883_p11` to UNKNOWN. Cap 860 restores that row with a
strictly verified 754-node proof and has no archive downgrade or verdict flip,
but its 53.456 s median is over budget. Cap 1,000 is also decision-safe but
takes 60.484 s.

**HYPOTHESIS.** Adopt cap 750 with J2near off. It is the highest-decision tested
arm that meets both the archive-superset requirement and the old-wall budget.
Keep J2near available but default-off; this grid finds no reason to spend the
production budget on its extra branching.

## Measurement contract

**CODE-FACT.** The ignored Rust runner
`tss_j2near_ab::tss_j2near_matched_ab` uses the production solver shape:
dual-pass enabled, a 256 KiB transposition table, and an unbounded
`u32::MAX` semantic horizon. The lane added `TSS_J2NEAR_CAPS` for a comma-
separated cap grid and `TSS_J2NEAR_OUTPUT_DIR` for raw-output placement. Solver
defaults and production configuration were not changed; `tss_verify.rs` was
not edited.

**MEASURED.** The run used caps 500, 640, 750, 860, and 1,000 over the complete
`human_v1`, `selfplay_v1`, and `puzzle_v3` sets (6,443 rows). At every cap the
runner alternated whole-battery arm order across three repetitions: off/on,
on/off, off/on. It asserted identical status, node count, certificate-verification
state, and verification-failure count for every row across all three runs.
Thus the requested two-run determinism spot check was exceeded for every arm.
The test passed in 1,381.73 s with 1 test passed and 0 failed.

**CODE-FACT.** Arm wall is the sum of the per-row interval around `solve()`;
set loading, strict certificate verification, and build time are excluded.
The strict verifier is invoked immediately after every solve. This is solve
wall, not outer test-process elapsed time.

**MEASURED.** Free physical RAM was 13.79 GiB before the build, above the
required 8 GiB. The run was a release MSVC build with the worktree-local
`.cargo-target`, `RUST_MIN_STACK=33554432`, and `--test-threads=1`. Other Cargo
processes appeared during the run. The alternating paired execution is the
contention defense; the table reports all spreads, deterministic total nodes,
and ns/node rather than hiding the host variation.

## Full arm and Pareto table

The baseline is the archived old production decided set: 700 WIN + 512 LOSS =
1,212. `Up`, `Down`, and `Flip` compare every row directly with that archive.
Wall is median `[min-max]` across the three battery repetitions. Nodes are for
one complete battery. `P` denotes the ordinary two-dimensional frontier on
more decided rows and lower median wall; it does not waive archive safety.
`Better` applies the stricter lane gate: archive verdict superset, at least one
new decision, no W/L flip, and median wall at or below 49.96 s.

| Cap | J2near | W / L | Decided | Up | Down | Flip | Solve wall s, median `[min-max]` | Nodes | ns/node | P | Better |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 500 | off | 700 / 512 | 1,212 | 0 | 0 | 0 | 28.193 `[28.071-28.354]` | 556,452 | 50,666 | yes | no: no new row |
| 500 | on | 699 / 512 | 1,211 | 0 | 1 | 0 | 28.765 `[28.727-29.231]` | 559,469 | 51,414 | no | no: downgrade |
| 640 | off | 737 / 536 | 1,273 | 61 | 0 | 0 | 38.907 `[34.189-39.922]` | 656,991 | 59,220 | yes | **yes** |
| 640 | on | 736 / 536 | 1,272 | 61 | 1 | 0 | 36.062 `[35.150-38.102]` | 661,259 | 54,535 | yes | no: downgrade |
| 750 | off | 757 / 544 | 1,301 | 89 | 0 | 0 | 45.956 `[44.689-46.730]` | 730,143 | 62,941 | yes | **yes** |
| 750 | on | 754 / 544 | 1,298 | 87 | 1 | 0 | 48.286 `[42.763-50.828]` | 735,451 | 65,655 | no | no: downgrade |
| 860 | off | 781 / 554 | 1,335 | 123 | 0 | 0 | 52.675 `[51.695-53.186]` | 797,794 | 66,026 | yes | no: wall |
| 860 | on | 780 / 554 | 1,334 | 122 | 0 | 0 | 53.456 `[52.823-55.125]` | 803,970 | 66,490 | no | no: wall |
| 1,000 | off | 796 / 566 | 1,362 | 150 | 0 | 0 | 58.722 `[57.705-59.566]` | 879,234 | 66,787 | yes | no: wall |
| 1,000 | on | 796 / 566 | 1,362 | 150 | 0 | 0 | 60.484 `[59.021-60.705]` | 886,436 | 68,233 | no | no: wall |

**MEASURED.** The strictly-better frontier versus the old point contains two
tested configurations:

| Cap | J2near | New verified decisions | Median wall | Worst observed wall | Margin to old wall |
|---:|:---:|---:|---:|---:|---:|
| 640 | off | +61 | 38.907 s | 39.922 s | 11.053 s median |
| 750 | off | +89 | 45.956 s | 46.730 s | 4.004 s median |

Cap 640 is the lower-wall frontier point; cap 750 buys 28 more verified
decisions while remaining below the old wall in every repetition. Cap 750 is
therefore the production recommendation.

**MEASURED.** As a direct check on the paired design, the median of the three
within-repetition J2near-on/off wall ratios was 1.019x, 0.979x, 1.051x, 1.015x,
and 1.030x at caps 500, 640, 750, 860, and 1,000 respectively. The cap-640
reversal and the wider cap-750 spread expose the timing variation that the
paired design was intended to retain; the deterministic node totals above are
the load-independent comparison.

## Per-row safety and J2near checks

**MEASURED.** There were zero WIN/LOSS flips in all ten arms. Every newly
decided row had a certificate accepted by the strict verifier. There were zero
verification failures and zero unverified decided results in every arm and in
all three repetitions.

**MEASURED.** The only archive downgrade was
`human_41e2eecefcb26883_p11`, and only in the J2near-on arms below cap 860:

| Cap | J2near off | J2near on |
|---:|---:|---:|
| 500 | verified WIN, 454 nodes | UNKNOWN, 500 nodes |
| 640 | verified WIN, 454 nodes | UNKNOWN, 640 nodes |
| 750 | verified WIN, 454 nodes | UNKNOWN, 750 nodes |
| 860 | verified WIN, 454 nodes | verified WIN, 754 nodes |
| 1,000 | verified WIN, 454 nodes | verified WIN, 754 nodes |

This directly answers the requested restoration check: it is **not** restored
at cap 640 or 750; it is restored at cap 860 and remains restored at cap 1,000.

**MEASURED.** None of the five known cap-2,000 J2near upgrades appears by cap
1,000. All five remain UNKNOWN in both arms throughout this grid:

- `human_330765c103651880_p11`
- `human_3bfd5e45945dcdb5_p43`
- `human_a32ed96ded852131_p13`
- `human_ee0378062be6e03f_p18`
- `atlas_oa-6fda812864c6d19a`

This is consistent with the earlier matched-cap measurements, where their
J2near proofs required 1,111 to 1,783 nodes. It also means the J2near tier adds
no decided row over the off arm at any tested cap. At cap 1,000 the arms have
identical per-row decisions, while J2near uses 7,202 more nodes and 1.762 s more
median solve wall.

## Recommended production point

**HYPOTHESIS.** Configure production for node cap **750**, dual-pass on,
unbounded horizon, 256 KiB TT, and J2near **off**. Relative to the old point,
this gives:

- the exact same verdict for all 1,212 previously decided rows;
- 89 additional strictly verified decisions, reaching 1,301 total;
- zero downgrades, zero W/L flips, and zero verifier failures;
- 45.956 s median solve wall, 4.004 s below the old 49.96 s;
- a full observed spread of 44.689-46.730 s; and
- 730,143 deterministic nodes per battery at 62,941 median ns/node.

**HYPOTHESIS.** Do not enable J2near at the new production cap. Raising the cap
from 750 to 860 makes it archive-safe, but neither the on nor off arm at 860
fits the measured wall budget. If future candidate-generation speedups create
additional headroom, retest J2near at caps high enough to reach the previously
measured 1,111-1,783-node upgrades.

## Raw evidence and validation

**MEASURED.** Machine-readable and console evidence is retained under
`.gate/j2near-cap/`:

- `matched.jsonl`: 32,215 per-position rows, with both arms and three wall
  samples per arm;
- `summary.json`: archive diffs, status counts, wall samples, nodes, and
  verifier summaries per arm; and
- `measurement.log`: RAM gate, build output, cap checkpoints, and the passing
  Rust test result.

The exact measurement invocation was:

```text
TSS_J2NEAR_CAPS=500,640,750,860,1000
TSS_J2NEAR_REPETITIONS=3
TSS_J2NEAR_OUTPUT_DIR=<worktree>/.gate/j2near-cap
CARGO_TARGET_DIR=<worktree>/.cargo-target
RUST_MIN_STACK=33554432
cargo test -p hexfield_eq --lib --release \
  --target x86_64-pc-windows-msvc \
  tss_j2near_ab::tss_j2near_matched_ab -- \
  --ignored --exact --test-threads=1 --nocapture
```
