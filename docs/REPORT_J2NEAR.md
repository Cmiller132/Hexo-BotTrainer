# J2near free-tempo widening report

Date: 2026-07-21  
Branch/worktree: `claude/j2near`  
Recommendation: **keep flagged and default-off**

## Executive result

**MEASURED.** All three preregistered atlas witnesses changed from UNKNOWN to
strictly verified WIN at the 100,000-node, unbounded-horizon, 256 MiB-TT gate.
The predicted missing stones were present in the widened candidate set, and no
solver certificate verification failed in any witness, frozen-cohort, grind, or
broader-population run.

**MEASURED.** The mechanism is not killed: in addition to the seeded witnesses,
it produced four verified human UNKNOWN-to-WIN upgrades at cap 2,000 and one
verified self-play/grind UNKNOWN-to-WIN upgrade at cap 50,000.

**MEASURED.** Default-on adoption is blocked. One previously verified human WIN
became UNKNOWN at the production cap of 500, and the preregistered per-row p95
wall-ratio threshold of 1.20 was exceeded. The eligible-root p90 branching
threshold was not exceeded.

## Implementation

**CODE-FACT.** `WidthOptions` now carries `free_tempo_j2near`. The existing
`vcf_pair_complete` profile remains false, the new `vcf_pair_j2near` profile is
true, and `TSS_VCF_J2NEAR=1` enables the tier for the existing wide profile. The
effective setting is included in the Python solver manifest and harness feature
declaration. The batch runner refuses an unsupported bench overlay rather than
silently measuring the wrong arm.

**CODE-FACT.** At a fresh pair root, `WideTurnGate` reconstructs the post-first
count-four family from count-three windows through the first stone. J2near is
considered only if no live defender count-four-or-greater window remains and the
claimant family has hitting number at least two. It counts post-first claimant-
pure count-one window membership by axis and appends only cells outside the
existing exact second-stone universe that have support at least four on at least
two axes.

**CODE-FACT.** At a `SecondStone` root, the analogous test uses the already
played first stone's forcing state and appends qualifying cells outside the
regenerated ordinary candidate set. Both seams route additions through the
existing pair classifier, defender generation, deduplication, lazy frontier,
and certificate materialization. No new proof node or pruning rule was added.
`tss_verify.rs` is unchanged.

**CODE-FACT.** The option is default-off. When false, both new widening calls
are branch-excluded. The cap-500 flag-off arm reproduced all 6,443 archived
`20260720_231040_dualpass_adoption` `(status, node cost)` records exactly. The
archive contains no standalone candidate-digest field; structural flag-off
candidate identity is additionally guarded by exact root-child tests, including
the three witness shapes.

## Witness gate

All rows used cap 100,000, semantic horizon `u32::MAX`, and a 256 MiB TT.

| Witness | Required candidate | Root children off -> on | Flag off | Flag on | Certificate / D6 diagnostic |
|---|---|---:|---:|---:|---|
| `oa-0153903c5a863630` | `(0,-1),(-1,2)` | 19 -> 39 | UNKNOWN, 42 nodes | WIN, 9,378 nodes | 523 nodes; strict pass; mask `0x081` |
| `oa-773ca1a59e95f4e1` | `(3,-3),(3,-2)` | 19 -> 39 | UNKNOWN, 42 nodes | WIN, 9,194 nodes | 523 nodes; strict pass; mask `0x081` |
| `oa-6fda812864c6d19a` | SecondStone seed `(0,-2)` | 8 -> 12 | UNKNOWN, 20 nodes | WIN, 1,458 nodes | 522 nodes; strict pass; mask `0x081` |

**MEASURED.** Candidate tests showed each required move only in the enabled
candidate set, with exactly the predicted child-count changes. Canonical strict
verification passed for all three certificates with zero failures. The opening-
atlas D6 diagnostic reproduced its expected `0x081` acceptance mask (2 of 12
atlas images); this is the established atlas-remap diagnostic, not 10 canonical
certificate rejections.

## Branching census

Eligibility and accepted-child statistics are root properties and therefore do
not depend on the matched solve cap. The census is over usable roots; multiplier
statistics include eligible roots whose J2near tier is empty.

| Cohort | Rows / usable | Eligible | J2near nonempty | Added accepted children on eligible roots | Child multiplier on eligible roots |
|---|---:|---:|---:|---|---|
| Witnesses | 3 / 3 | 3 | 3 | 20, 20, 4 | 2.053, 2.053, 1.500 |
| `puzzle_v3` | 468 / 463 | 21 | 15 | mean 4.43; p50 2; p90 14; max 20 | mean 1.191; p50 1.067; p90 1.519; max 2.053 |
| `human_v1` | 2,720 / 2,701 | 100 | 38 | mean 1.22; p50 0; p90 4; max 13 | mean 1.039; p50 1.000; p90 1.057; max 2.250 |
| `selfplay_v1` | 3,255 / 3,124 | 4 | 1 | total 1 | mean 1.010; p50 1.000; p90 1.000; max 1.040 |
| 248 grind roots | 248 / 248 | 0 | 0 | 0 | 1.000 |

**MEASURED.** Every real-cohort eligible-root p90 multiplier is below the 2.0
default-on branching block threshold. The human maximum of 2.25 is above two,
but the preregistered gate is p90, not maximum.

## Matched-cap decision results

Every arm used the same release build and unbounded semantic horizon. Frozen
cap-500 and cap-2,000 runs used a 256 KiB TT. The grind used cap 50,000 and a
256 MiB TT. All decision and node results were deterministic across three runs;
arm order alternated by repetition. `Decided` is verified WIN plus verified LOSS.

| Cap / cohort | Off status counts U/W/L | On status counts U/W/L | Decided off -> on | Upgrades | Downgrades | Verifier failures |
|---|---:|---:|---:|---:|---:|---:|
| 500 `human_v1` | 1993 / 393 / 334 | 1994 / 392 / 334 | 727 -> 726 | 0 | 1 | 0 |
| 500 `puzzle_v3` | 240 / 118 / 110 | same | 228 -> 228 | 0 | 0 | 0 |
| 500 `selfplay_v1` | 2998 / 189 / 68 | same | 257 -> 257 | 0 | 0 | 0 |
| 500 all frozen | 5231 / 700 / 512 | 5232 / 699 / 512 | 1212 -> 1211 | 0 | 1 | 0 |
| 2,000 `human_v1` | 1924 / 445 / 351 | 1920 / 449 / 351 | 796 -> 800 | 4 | 0 | 0 |
| 2,000 `puzzle_v3` | 84 / 210 / 174 | 83 / 211 / 174 | 384 -> 385 | 1 | 0 | 0 |
| 2,000 `selfplay_v1` | 2958 / 221 / 76 | same | 297 -> 297 | 0 | 0 | 0 |
| 2,000 all frozen | 4966 / 876 / 601 | 4961 / 881 / 601 | 1477 -> 1482 | 5 | 0 | 0 |
| 50,000 grind | 189 / 59 / 0 | 188 / 60 / 0 | 59 -> 60 | 1 | 0 | 0 |

### Per-row changes

**MEASURED — downgrade.** At cap 500,
`human_41e2eecefcb26883_p11` changed from verified WIN at 454 nodes to UNKNOWN
at the 500-node cap. This independently blocks default-on adoption.

**MEASURED — upgrades.** At cap 2,000, the following changed from UNKNOWN to
verified WIN:

- `human_330765c103651880_p11`: 2,000 -> 1,783 nodes
- `human_3bfd5e45945dcdb5_p43`: 338 -> 1,111 nodes
- `human_a32ed96ded852131_p13`: 2,000 -> 1,343 nodes
- `human_ee0378062be6e03f_p18`: 22 -> 1,286 nodes
- `atlas_oa-6fda812864c6d19a`: 21 -> 1,458 nodes

At cap 50,000, `sp_41_p37` changed from UNKNOWN at 10,596 nodes to verified
WIN at 9,656 nodes. There were no WIN/LOSS flips.

## Matched-cap cost results

For each row, wall ratio is `median(three on runs) / median(three off runs)`.
The p50 and p95 columns are calculated over decision-identical rows. `Cohort
wall` is the median of three paired whole-cohort total-wall ratios. Node deltas
are `on - off`, also only on decision-identical rows.

| Cap / cohort | Identical rows | Wall p50 | Wall p95 | Cohort wall | Node delta median | Node delta mean | Node-ratio p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 500 `human_v1` | 2,719 | 1.010x | 1.561x | 1.023x | 0 | +0.543 | 1.000x |
| 500 `puzzle_v3` | 468 | 0.999x | 1.586x | 0.969x | 0 | +3.122 | 1.000x |
| 500 `selfplay_v1` | 3,255 | 1.007x | 1.570x | 1.008x | 0 | +0.010 | 1.000x |
| 500 all frozen | 6,442 | 1.007x | 1.569x | 1.014x | 0 | +0.461 | 1.000x |
| 2,000 `human_v1` | 2,716 | 0.938x | 1.529x | 0.979x | 0 | +1.511 | 1.000x |
| 2,000 `puzzle_v3` | 467 | 1.031x | 1.597x | 1.074x | 0 | +10.375 | 1.000x |
| 2,000 `selfplay_v1` | 3,255 | 0.970x | 1.522x | 0.926x | 0 | +0.139 | 1.000x |
| 2,000 all frozen | 6,438 | 0.974x | 1.532x | 0.963x | 0 | +1.460 | 1.000x |
| 50,000 grind | 247 | 0.920x | 1.296x | 0.994x | 0 | +207.316 | 1.001x |

**MEASURED.** Whole-cohort medians are generally close to parity and all
decision-identical node-delta medians are zero. Nevertheless, every per-row p95
wall ratio exceeds the preregistered 1.20 threshold. Very short solves make the
tail timing ratios noisy, but the threshold is binding and is reported without
post-hoc relaxation.

Machine-readable measurement rows are in `.scratch/j2near_ab/matched.jsonl`,
`.scratch/j2near_ab/grind.jsonl`, and
`.scratch/j2near_ab/broader_puzzle_unknown_100k.jsonl`.

## Broader certified-miss check

**MEASURED.** The full `puzzle_v3` population, not a sample, was screened
flag-off at cap 100,000 with an unbounded horizon and 256 MiB TT. Exactly 3 of
468 rows remained UNKNOWN: the three seeded atlas witnesses. Flag-on verified
all three as WIN, so this particular population contributed **0 wins beyond the
seeded three**.

**MEASURED.** The separate matched-cap certified misses do provide wins beyond
the seeds: the four human IDs at cap 2,000 and `sp_41_p37` at cap 50,000. Thus
the preregistered broader-win kill condition is not triggered, although the
full-cap puzzle result shows that the extra capability is sparse.

## Validation and gate evaluation

**MEASURED.** Validation completed as follows:

- Focused candidate/profile tests: 2 passed, 0 failed, 1 expensive witness test ignored.
- Release witness/D6 gate: 1 passed; all 3 witness cases verified.
- Python-feature Rust suite: 219 passed, 0 failed, 42 ignored; doc tests passed.
- Frozen flag-off identity: 6,443 of 6,443 archived status/node records matched.
- Frozen matched A/B: 12,886 rows (two caps), three repetitions per arm.
- Grind matched A/B: 248 rows, three repetitions per arm.
- Full-population broader check: 468 screened, 3 UNKNOWN rerun flag-on, test passed.
- `cargo check -p hexfield_eq --tests`: passed.
- `git diff --check`: passed; `tss_verify.rs` has no diff.

**MEASURED — known host block.** The stage-0 Python golden was not run because
this Windows host has Python 3.14 without `pytest` or `maturin`, and no installed
WSL distribution. This was a known block in the lane brief. The required
serialized `--features python` Rust suite passed instead.

| Preregistered criterion | Result |
|---|---|
| Any witness UNKNOWN after candidate appears | Pass: 3/3 verified WIN |
| Any canonical verifier failure | Pass: zero |
| D6 atlas diagnostic rejection | Pass: expected `0x081` mask reproduced |
| No verified win beyond seeded three | Pass: five non-seed matched-cap upgrades |
| Eligible-root p90 child multiplier > 2 | Pass: maximum cohort p90 1.519 |
| Decision-identical median wall > 1.05 | **Block: puzzle cap-2k paired cohort-total median is 1.074** |
| Decision-identical p95 wall > 1.20 | **Block: all reported cohorts exceed 1.20** |
| Any flag-off-decided row becomes UNKNOWN | **Block: one cap-500 human WIN downgrade** |
| Flag-off identity | Pass: 6,443/6,443 archived status/node identities; new tier branch-excluded off |

## Recommendation

**HYPOTHESIS.** Keep `vcf_pair_j2near` available as a named experimental or
targeted recovery profile, but leave `vcf_pair_complete` and the environment
default off. The widening is sound and demonstrates real capability beyond its
seed witnesses, so deleting it would discard useful search reach. It is not a
safe production default at the present cap because additional branching can
consume the budget before an existing proof is reached, as the cap-500 human
downgrade demonstrates. A future default-on attempt would need an ordering or
budget-reserve policy that restores the downgraded proof, followed by a fresh
preregistered A/B that clears both wall gates.
