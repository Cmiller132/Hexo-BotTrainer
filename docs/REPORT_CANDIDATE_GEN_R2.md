# Candidate-generation efficiency round 2

Date: 2026-07-21  
Branch/worktree: `claude/candidate-gen`  
Baseline: uncommitted round-1 tree documented in `REPORT_CANDIDATE_GEN.md`

## Disposition

**CODE-FACT.** One bit-identical production change is retained: the exact
attacker window-generation memo remains direct-mapped and grows from 512 to
32,768 slots. Allocation remains lazy (only a search that performs a memoized
window lookup allocates the slots). `tss_verify.rs` was not edited.

**MEASURED.** The final serialized Rust suite with the Python feature is green:
218 passed, 0 failed, 39 ignored. The final frozen battery is exactly the
round-1 baseline: 6,443 positions, 556,452 nodes, FNV digest
`a8c6f3ca3ba55827`, and emitted-row SHA-256
`02CD63718E0D06F83853B523C40F7057626A7A3113264235C3CECB162482CFDB`.

**MEASURED.** The memo sweep selected 32,768 direct-mapped slots. Its
instrumented three-run medians were 30.469 s battery / 29.005 s solve with a
46.208% median hit rate. The corresponding 512-slot medians were 37.644 s /
35.917 s and 25.089%. Host load changed during the sweep, so these same-binary
medians and their spreads are reported, but node counts and exact identity are
the primary acceptance evidence. A final hard-coded repeat was 30.370 s /
28.911 s at 46.206% hits.

## Step 0: independent round-1 reproduction

The commands were copied verbatim from the round-1 report and run before any
round-2 source change.

| Gate | Expected | Measured | Result |
|---|---:|---:|---|
| Serialized `--features python` suite | 218/0/39 | 218/0/39 | PASS |
| Frozen positions | 6,443 | 6,443 | PASS |
| Nodes | 556,452 | 556,452 | PASS |
| FNV digest | `a8c6f3ca3ba55827` | `a8c6f3ca3ba55827` | PASS |
| Identity SHA-256 | `02CD...FDB` | `02CD...FDB` | PASS |

Raw evidence: `step0_full_suite.log`, `step0_battery.log`,
`step0.identity.tsv`, and `step0_identity_sha256.log` under
`.gate/candidate-gen-r2/`. The pre-build host check found 13.86 GB available
physical memory and an active `lake` process, which was left untouched
(`step0_ram_check.log`).

## Step 1: fresh post-rung profile

Timers are `cfg(test)` instrumentation. `expand` is inclusive of generation,
so its sub-blocks must not be added to it. Percentages use measured solve wall,
not Cargo/build wall.

### Production shape: cap 500 / 256 KiB TT

Solve wall was 35.337 s for the instrumented 512-slot baseline.

| Block | Wall (ms) | % of solve wall | Scope |
|---|---:|---:|---|
| Wide expansion | 23,727 | 67.15% | inclusive |
| Attacker pair generation | 17,195 | 48.66% | inside expansion |
| Defender generation | 5,507 | 15.58% | inside expansion |
| Second-candidate regeneration | 1,997 | 5.65% | inside attacker generation |
| PN refresh | 628 | 1.78% | separately timed |
| TT probe/insert path | 163 | 0.46% | separately timed |
| Outside inclusive expansion | 11,610 | 32.85% | solve residual |

Raw evidence: `step1_production_profile.log`.

### Deep F19: 256 MiB TT, maximum rung 100,000

The 28 emitted corpus attempts sum to 34,333.2 ms solve wall. The harness
finishes in 34.45 s and reaches the same two expected fixed-cap failures as
round 1 (`0l4291i_live` and `lz60mfb`).

| Block | Wall (ms) | % of solve wall | Scope |
|---|---:|---:|---|
| Wide expansion | 23,290 | 67.83% | inclusive |
| Attacker pair generation | 12,351 | 35.97% | inside expansion |
| Defender generation | 10,493 | 30.56% | inside expansion |
| Second-candidate regeneration | 1,576 | 4.59% | inside attacker generation |
| PN refresh | 677 | 1.97% | separately timed |
| TT probe/insert path | 690 | 2.01% | separately timed |
| Outside inclusive expansion | 11,043 | 32.17% | solve residual |

Raw evidence: `step1_f19_profile_100k.log`. The earlier
`step1_f19_profile.log` accidentally omitted `TSS_CORPUS_MAX_CAP=100000` and
therefore entered the 1M rung; the command timeout stopped it. It is retained
as an audit log and excluded from every result.

## Step 2: memo scaling sweep

Every arm used the same release test binary and preserved 6,443 positions,
556,452 nodes, and digest `a8c6f3ca3ba55827`. Walls are medians of three;
parentheses show the full three-run spread.

| Slots / ways | Median hit rate | Battery median (spread), s | Solve median (spread), s | Decision |
|---|---:|---:|---:|---|
| 512 / 1 | 25.089% | 37.644 (37.106-38.135) | 35.917 (35.337-36.376) | baseline |
| 2,048 / 1 | 27.356% | 37.898 (37.813-38.224) | 36.098 (36.040-36.480) | KILL: no wall gain |
| 8,192 / 1 | 38.196% | 37.841 (34.591-38.143) | 36.111 (32.952-36.345) | KILL: median no gain |
| 32,768 / 1 | 46.208% | 30.469 (30.350-31.059) | 29.005 (28.891-29.568) | RETAIN |
| 32,768 / 2 | 65.375% | 31.678 (30.296-31.988) | 30.160 (28.852-30.490) | KILL: associativity overhead |

**CODE-FACT.** The retained exact key is unchanged: `(WindowKey, player-0
mask, player-1 mask)`. Collisions replace one slot and cannot create false
hits. The existing debug hit oracle and mask-delta regression remain active.

**MEASURED.** A defender-side prototype routed live-threat empty extraction
through the same memo. It added 879,149 lookups per battery, remained exact,
but memo-on median solve wall was 38.754 s (30.043-40.523) versus 35.079 s
(34.163-39.554) off. Defender-generation median was 6,119 ms on versus
5,607 ms off. The conversion/allocation work remained, so the prototype was
removed. Raw evidence: `step2_defender_memo_{on,off}_rep*.log`.

Raw direct/set-associative evidence:
`step2_memo_{512x1,2048x1,8192x1,32768x1,32768x2}_rep*.log` (the first
512 observation is `step1_production_profile.log`).

## Step 3: per-solve setup share

**MEASURED.** The frozen battery already uses the persistent `TssSolver` batch
shape used by the trainer. Across it, wide-search construction plus root setup
was 53 ms; search execution was 27,476 ms; total solve wall was 28,432 ms.
Setup was therefore 0.186% of leaf solve wall, far below the brief's 10%
prototype trigger. Persistent arena/scratch reuse was not implemented.

Raw evidence: `step3_setup_profile.log`. The timers retained in source are
`cfg(test)` only.

## Step 4: TT tax at cap 500

**MEASURED.** On the retained tree, the inclusive TT probe/insert timer was
123 ms out of 28,432 ms solve wall, or 0.433%. This is below the 3% threshold.
No `TSS_TT_MIN_PROFILE` behavior-change flag was added, and status-flip/node
evaluation was therefore not triggered. Flag-off identity is the ordinary
final identity gate.

Raw evidence: `step3_setup_profile.log` (`insert_ms=123`).

## Final deep observation

With the retained 32,768-slot direct memo, the F19 attempt rows sum to
28,947.0 ms and the harness wall is 29.04 s. Final generation memo hits are
7,429,017 / 12,743,545 = 58.296%. The same two fixed-cap expected-WIN rows
remain UNKNOWN; there are no new failures. This single final observation is
not promoted to a median wall claim.

Raw evidence: `final_f19.log`.

## Gate matrix

| Boundary | Serialized suite | Frozen identity | Stage-0 Python golden |
|---|---|---|---|
| Step 0 round-1 reproduction | PASS: 218/0/39 | PASS: exact | BLOCKED on this host per brief |
| Memo sweep arms | not rerun per arm | PASS each arm: exact | BLOCKED on this host per brief |
| Final source | PASS: 218/0/39 | PASS: exact digest + SHA-256 | BLOCKED on this host per brief |

The orchestrator-owned Stage-0 gate was not run, as explicitly directed in the
round-2 brief. Final raw evidence is `final_full_suite.log`,
`final_battery.log`, `final.identity.tsv`, and
`final_identity_sha256.log` under `.gate/candidate-gen-r2/`.

## Reproduction commands

```powershell
$env:CARGO_TARGET_DIR = (Join-Path (Get-Location) '.cargo-target')
cargo test -p hexfield_eq --features python --target x86_64-pc-windows-msvc -- --test-threads=1

$env:TSS_IDENTITY_OUT = (Join-Path (Get-Location) '.gate/candidate-gen-r2/final.identity.tsv')
cargo test -p hexfield_eq --features python --release --target x86_64-pc-windows-msvc tss_frozen_identity_battery -- --ignored --test-threads=1 --nocapture

$env:TSS_BACKWALK_TT_BYTES = '268435456'
$env:TSS_CORPUS_MAX_CAP = '100000'
cargo test -p hexfield_eq --features python --release --target x86_64-pc-windows-msvc tss_corpus_check -- --ignored --test-threads=1 --nocapture
```
