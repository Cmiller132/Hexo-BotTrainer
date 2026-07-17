# G2R9 shared proven fragments: campaign report

Date: 2026-07-17
Design base: `hunt/turn-quotient` at `86a6418c`; completion checkpoint
`8615726d`
Rollout flag: `TSS_SHARED_FRAGMENTS=1` (default off)
Status: **COMPLETED UNDER THE AMENDED ORCHESTRATOR VERDICT CONTRACT;
DEFAULT-OFF; BOTH OFFICIAL GATES PASS; REDUCED 20M RESULT BOUNDED AND
UNCLAIMED.**

The original stopped-campaign report is preserved below as history. Its
identity-only warm ruling was superseded by the G2R9b amendment; the completion
record begins at "G2R9b completion under the amended verdict contract."

## Original G2R9 stop verdict

The design contract is consistent with proved Lean T10 at
`69adffc7dd3cb1b33d56242c5d219b1a7d969224`. Exact-key positive fragments are
independently verified before admission, final DAG labels are reconstructed
with max-dominant child recurrences, final obligations come from the assembled
graph, and the unchanged strict verifier replays the complete returned
certificate. No review found a silent-wrong-verdict route.

The requested campaign nevertheless has an explicit stronger gate: flag-off
and flag-on verdicts must be identical at the fixed caps, and *any* flip is a
mandatory stop. The eager/lazy-off campaign reached
`human_014_g1531_p95` and produced `UNKNOWN` with fragments off versus a
strict-verifier-accepted `WIN` on the warm fragments-on solve. This is a useful
proof found within the same cap, not a verifier failure, but it is still a
verdict flip under the owner's stated rule. The lazy composition lane,
reduced-budget campaign, mutation control, and both official 2 GiB gates were
therefore not run.

## Design and implementation boundary

The normative design memo is `BUILD_SHARED_FRAGMENTS.md`. The implemented
store is limited to the wide PN engine and retains only complete positive
proofs. It stores no `UNKNOWN`, cap exit, partial branch, `DepthCutoff`, or
negative/refutation result.

Key properties exercised before the stop:

- `TSS_SHARED_FRAGMENTS` is read once by `TssSolver::default`; unset/off is the
  default.
- Full `PositionKey` plus claimant equality authorizes hits; the 64-bit hash
  only selects a direct-mapped slot.
- Positive horizon transfer is
  `resolution_t <= query_horizon`; zoned payloads additionally require
  `query_horizon <= zone_build_t`, rebase, and final replay.
- Cached `Universal` roots are restricted to solve depth zero so an unknown
  path-local commutation context cannot be erased.
- The store is immutable during search. Live proof payloads are `Arc`-pinned;
  admission/replacement happens only after search drop.
- Retained fragments are capped at one eighth of the caller TT ceiling, slots
  allocate lazily, and a warm solve subtracts only actual retained bytes. A
  cold empty store leaves the historical local TT cap unchanged.
- Imported arenas are remapped atomically and checked for depth, node, edge,
  commutation, witness, resolution, and build-horizon bounds.
- The complete flag-on certificate is relabelled and strictly verified before
  it can be returned. Production `HardValue` minting remains sealed behind
  `hard_value_from_verified`.

## Completed regression evidence

| check | result | evidence |
|---|---:|---|
| Rust formatting / diff whitespace | PASS | `rustfmt --edition 2021`; `git diff --check` |
| Debug library suite | PASS | 99 passed, 0 failed, 22 ignored; 39.88 s |
| Release library compile | PASS | `cargo test --release -p hexfield_eq --lib --no-run` |
| Focused fragment tests | PASS | 4 passed, 0 failed; warm exact root, forced collision, full-key/claimant/accounting, profile reset |
| Adversarial soundness review | PASS | no remaining silent-wrong-verdict blocker |

The full eager campaign ran for 198.67 s before its mandatory stop. Cases are
ordered as 38 forcing rows (all 19 at 10k and 100k), one compact fixture, then
100 human roots. Therefore 53 cases completed flag-off/on cold and warm
identity before the failing human case; the failing case itself retained cold
identity and failed only the warm A/B comparison. Every hard certificate seen
by the harness before comparison was accepted by `TssVerifier`.

## Mandatory-stop evidence

Single-root reproduction, 512 MiB TT, 10k nodes, lazy frontier off:

| mode | phase | verdict | expansions | wall | lookups | hits | imports |
|---|---|---:|---:|---:|---:|---:|---:|
| fragments off | cold | UNKNOWN | 10,000 | 917.264 ms | 0 | 0 | 0 |
| fragments off | warm | UNKNOWN | 10,000 | 921.480 ms | 0 | 0 | 0 |
| fragments on | cold | UNKNOWN | 10,000 | 929.967 ms | 0 | 0 | 0 |
| fragments on | warm | **WIN** | 3,770 | 289.466 ms | 3,763 | 2 | 2 |

The warm fragments-on certificate was strict-verifier accepted before the
identity assertion. Relative to flag-off warm, expansions fell 62.3% and wall
time fell 68.6%. The final retained store contained 69 entries / 1,216,402
bytes, representing 1,127 certificate nodes and 512 explicit edges; it had 69
admissions, zero replacements, and zero refusals.

Exact stop marker:

```text
SF_STOP verdict flip: cohort=human_100_cap10000
id=human_014_g1531_p95 cap=10000 horizon=4294967295 lazy=off
comparison=off-warm-vs-on-warm left=UNKNOWN right=WIN
```

This finding is deliberately not relabelled as a campaign pass. A verified
new WIN is the intended efficiency mechanism, but the work order required
verdict identity rather than merely absence of contradictory hard verdicts.

## Original checkpoint: requested headline gates not completed

| requested measurement/gate | result |
|---|---|
| 139-root eager soundness + warm campaign | **STOPPED** at case 54/139 on the flip above |
| 139-root fragments + lazy-frontier composition | **NOT RUN after stop** |
| different-root mutation control | **NOT REACHED before stop** |
| 0l reduced TT, 512 MiB | **NOT RUN after stop** |
| 0l reduced TT, 1 GiB | **NOT RUN after stop** |
| official 2 GiB fragments-only corpus gate | **NOT RUN after stop** |
| official 2 GiB fragments + lazy corpus gate | **NOT RUN after stop** |

No claim is made about reduced-budget closure or `CORPUS_DONE failures=0` for
this build.

## Files

- `BUILD_SHARED_FRAGMENTS.md` — design and soundness contract, written before
  engine edits.
- `packages/hexfield_eq/rust/src/tss_core.rs` — public fragment telemetry in
  `SolveStats`.
- `packages/hexfield_eq/rust/src/tss_solver.rs` — default-off store, lookup,
  import, max-dominant relabel, strict promotion, accounting, and focused tests.
- `packages/hexfield_eq/rust/src/tss_turn_quotient_hunt.rs` — ignored 139-root,
  warm, mutation, reduced-budget, and single-case diagnostic harnesses.
- `packages/hexfield_eq/rust/src/tss_corpus.rs` — explicit strict verification
  of every returned corpus certificate plus fragment telemetry.
- `HUNT_REPORT_SHARED_FRAGMENTS.md` — this stop report.

No commit was made.

## Original checkpoint regeneration

Check free RAM before every Cargo invocation and keep commands serialized with
one test thread:

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:CARGO_TARGET_DIR='.target-hunt'
cargo test -p hexfield_eq --lib -- --test-threads=1
```

Exact mandatory-stop reproduction:

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_SHARED_FRAGMENT_TT_BYTES='536870912'
$env:TSS_SHARED_FRAGMENT_LAZY_MODE='off'
$env:TSS_SHARED_FRAGMENT_CASE_ID='human_014_g1531_p95'
$env:TSS_TURN_QUOTIENT_HUMAN_CORPUS='E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl'
cargo test --release -p hexfield_eq shared_fragment_soundness_and_warm_campaign -- --ignored --test-threads=1 --nocapture
```

The full campaign command below was documented for a future owner decision at
the stopped checkpoint and was not green under that original ruling:

```powershell
Remove-Item Env:TSS_SHARED_FRAGMENT_CASE_ID -ErrorAction SilentlyContinue
$env:TSS_SHARED_FRAGMENT_LAZY_MODE='off'
cargo test --release -p hexfield_eq shared_fragment_soundness_and_warm_campaign -- --ignored --test-threads=1 --nocapture
```

## G2R9b completion under the amended verdict contract

Completion date: 2026-07-17
Checkpoint: `hunt/turn-quotient` at `8615726d` plus the uncommitted harness,
gate, memo, and report changes listed below. No commit was made.

### Amended ruling and final verdict

The orchestrator retained exact verdict identity for every cold comparison and
for every flag-off repeat against its cold baseline. It amended only warm
flag-on comparisons: `UNKNOWN -> WIN` or `UNKNOWN -> LOSS` is permitted when
the new hard verdict carries a strict-verifier-accepted certificate. Losing a
hard verdict, `WIN <-> LOSS`, or producing an unverified hard verdict remains a
mandatory stop. A different-root warm mutation remains under cold identity,
and forcing-NO rows may never return `WIN`.

The rationale is soundness-preserving. `UNKNOWN` is a resource verdict, not a
game-theoretic result. A fragment import supplies only independently verified
sub-proofs; T10 licenses their finished DAG composition, and the unchanged
strict verifier re-accepts the caller's complete certificate before the sealed
single mint returns a hard value. A warm `UNKNOWN ->` verified-hard transition
therefore recovers capacity at the fixed budget, the same kind of effect as a
larger node cap. The single-mint architecture is unchanged.

Under that ruling the build is **GREEN, DEFAULT-OFF**. Both 139-root lanes,
both different-root mutation controls, all completed reduced-TT comparisons,
and both official 2 GiB gates passed their applicable contracts. The only warm
verdict change was the licensed, verified improvement that originally caused
the historical stop.

### Soundness campaign

Each lane covered 139 roots: 19 forcing positions at 10k, the same 19 at 100k,
`double_fork_compact`, and 100 deterministic human roots. There were zero
`SF_STOP` markers. Every hard result carried a certificate accepted by
`TssVerifier`. The ten forcing-NO cases per lane (five rows at two caps) covered
40 phase/mode solves per lane and produced zero `WIN` results.

| lane | roots | cold off/on | flag-off cold/warm | warm contract | hard certs | forcing-NO | mutation |
|---|---:|---|---|---|---|---|---|
| eager | 139 | PASS | PASS | PASS, 1 improvement | PASS | PASS | PASS |
| fragments + lazy | 139 | PASS | PASS | PASS, 1 improvement | PASS | PASS | PASS |

The status census was identical in both lanes. Off-cold, off-warm, and on-cold
each produced 45 `WIN`, 7 `LOSS`, and 87 `UNKNOWN`; on-warm produced 46 `WIN`,
7 `LOSS`, and 86 `UNKNOWN`.

The different-root control seeded `0hz3hty` (`WIN`), then solved the different
root `8is963b`. Seeded fragments-on, fresh fragments-on, and fresh fragments-off
all returned `LOSS` in both eager and lazy modes. The mutated solve used one
expansion, had zero fragment lookups/hits/imports, and the seeded store held 64
entries / 1,184,498 bytes.

### Complete monotone-improvement census

There were two mode-specific observations of one unique root/rung signature:

| lane | root | rung | off verdict | on verdict | strict verifier | off exp. | on exp. | exp. saved |
|---|---|---:|---|---|---|---:|---:|---:|
| eager | `human_014_g1531_p95` | 10,000 | UNKNOWN | WIN | PASS | 10,000 | 3,770 | 6,230 |
| lazy | `human_014_g1531_p95` | 10,000 | UNKNOWN | WIN | PASS | 10,000 | 3,770 | 6,230 |

The eager row fell from 910.294 ms to 285.589 ms (-68.627%); the lazy row fell
from 978.258 ms to 307.098 ms (-68.608%). Each mode made 3,763 lookups, 2 hits,
and 2 imports on that warm solve; the final store was 69 entries / 1,216,402
bytes. No other verdict changed. The two census markers each report
`count=1 expansions_saved=6230`; across the two independently exercised
composition modes that is 12,460 observed expansion savings, not two unique
game positions.

### Headline cold and warm work

`delta` is fragments-on minus fragments-off. Store sums are sums of the 139
per-root snapshots, not simultaneous resident memory.

| lane | phase | off exp. | on exp. | exp. delta | off wall | on wall | wall delta |
|---|---|---:|---:|---:|---:|---:|---:|
| eager | cold | 609,698 | 609,698 | 0.000% | 61,888.692 ms | 62,389.116 ms | +0.809% |
| eager | warm | 609,698 | 484,973 | -20.457% | 61,906.976 ms | 51,707.399 ms | -16.476% |
| lazy | cold | 609,698 | 609,698 | 0.000% | 67,294.787 ms | 67,960.132 ms | +0.989% |
| lazy | warm | 609,698 | 484,973 | -20.457% | 67,441.105 ms | 56,433.321 ms | -16.322% |

Warm fragments-on saved 124,725 aggregate expansions in each lane. Its
aggregate fragment profile was identical in the two lanes: 424,391 lookups,
199 hits (ratio 0.000469 = 0.0469%), 39 imports, snapshot sums of 2,908 entries
and 63,968,538 bytes, and a per-root maximum of 128 entries / 1,392,206 bytes.
Cold fragments-on made no lookups or imports; its snapshot sums were 2,091
entries / 62,178,948 bytes, with a maximum of 64 / 1,392,206 bytes.

### Reduced-TT `0l4291i_live` bottleneck metric

Only one official corpus ID begins with `0l`: `0l4291i_live` (expected WIN).
Completed measurements used eager frontier mode and the historical 1M-node TT
saturation scale at exactly 512 MiB and 1 GiB. Cold is one fresh 1M solve per
flag state. Progressive warm is the 10k -> 100k -> 1M ladder in one solver per
flag state; its 100k and 1M flag-off results were also recomputed fresh and
matched exactly.

| profile | TT | off at 1M | on at 1M | exp. delta | wall delta | on hits/lookups/imports | final store |
|---|---:|---|---|---:|---:|---|---:|
| cold | 512 MiB | UNKNOWN, 1,000,000 | UNKNOWN, 1,000,000 | 0.000% | +0.757% | 0 / 0 / 0 | 64 / 1,192,520 B |
| cold | 1 GiB | UNKNOWN, 1,000,000 | UNKNOWN, 1,000,000 | 0.000% | +0.774% | 0 / 0 / 0 | 64 / 2,241,096 B |
| progressive warm | 512 MiB | UNKNOWN, 1,000,000 | UNKNOWN, 1,000,000 | 0.000% | -3.520% | 10 / 999,962 / 0 | 170 / 1,444,744 B |
| progressive warm | 1 GiB | UNKNOWN, 1,000,000 | UNKNOWN, 1,000,000 | 0.000% | -0.114% | 10 / 999,962 / 0 | 170 / 2,493,320 B |

Across the whole 10k/100k/1M progressive ladder, both sides used 1,110,000
expansions. At 512 MiB, wall was 113,629.444 ms off versus 110,232.722 ms on
(-2.989%); at 1 GiB it was 113,652.797 ms versus 113,890.647 ms (+0.209%). Each
on ladder made 1,099,926 lookups, 13 hits (ratio 0.000012), and zero imports.
All four completed campaigns returned PASS; both warm campaigns made two fresh
flag-off baseline comparisons, and their monotone-improvement counts were zero.

The honest closure answer is **no closure through the completed 1M cap at
either reduced TT budget, cold or progressive warm**. No completed result above
1M is claimed. An initial combined 512 MiB cold attempt at the 20M rung reached
the outer 604-second non-gate limit before either side emitted a completed row;
the orphaned process tree was stopped and its partial log is not treated as
`UNKNOWN`. The 1 GiB 20M comparison was not started after that bound was
demonstrated. This limitation follows the required 10-minute non-gate bound;
it leaves later-rung reduced-budget closure unmeasured rather than silently
relabeling a timeout as a solver verdict.

### Official 2 GiB gates

Both gates started above 11 GiB free RAM and used one process with
`--test-threads=1`. Optional gate-expectation variables asserted the requested
feature modes inside the test before corpus work began.

| gate | asserted mode | result | test time | `0l` closure | `0l` closing peak TT |
|---|---|---|---:|---|---:|
| fragments | fragments on, lazy off | `CORPUS_DONE failures=0` | 447.88 s | WIN at 1,879,612 exp. | 1,729,265,069 B |
| fragments + lazy | fragments on, lazy on | `CORPUS_DONE failures=0` | 486.52 s | WIN at 1,879,612 exp. | 549,161,606 B |

Each gate exercised 34 ladder rows across all 19 IDs, returned 1 passed / 0
failed, and had no verifier, assertion, or corpus-failure marker. The official
corpus constructs a fresh solver for every rung, so fragment state is not
carried between rungs and gate lookups/hits/imports are zero. Per-solve stores
were nevertheless admitted, reaching a maximum of 64 entries / 6,027,694
bytes in both gates. These gates are cold composition/admission evidence; the
139-root and reduced campaigns supply the warm-import evidence.

### Completion changes and evidence files

- `BUILD_SHARED_FRAGMENTS.md` records the amended cold/warm verdict contract
  and its resource-verdict rationale.
- `packages/hexfield_eq/rust/src/tss_turn_quotient_hunt.rs` asserts the amended
  relation exhaustively, records strict verification in comparisons, emits the
  complete improvement census, and strengthens reduced-rung baselines/labels.
- `packages/hexfield_eq/rust/src/tss_corpus.rs` requires every hard gate verdict
  to carry a certificate and asserts/logs the expected fragment/lazy modes.
- `HUNT_REPORT_SHARED_FRAGMENTS.md` preserves the original stop history and
  appends this completion.
- The production implementation remains at the checkpoint in `tss_solver.rs`
  and `tss_core.rs`; the completion did not change its proof or mint logic.

Raw logs are under `.codex-hunt/`:

- `g2r9b-eager-campaign.log`, `g2r9b-lazy-campaign.log`;
- `g2r9b-reduced-512-cold-1m.log`, `g2r9b-reduced-512-warm-1m.log`;
- `g2r9b-reduced-1g-cold-1m.log`, `g2r9b-reduced-1g-warm-1m.log`;
- `g2r9b-official-fragments.log`, `g2r9b-official-fragments-lazy.log`.

Final regression evidence: `rustfmt --edition 2021` and `git diff --check`
passed; the debug library suite passed 99 tests with 0 failures and 22 ignored
in 40.03 s. The two release 139-root tests, four completed reduced-TT tests,
and two official gates each returned 1 passed / 0 failed.

### Completion regeneration

Check RAM immediately before every Cargo invocation and run commands one at a
time:

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_SHARED_FRAGMENT_TT_BYTES='536870912'
$env:TSS_TURN_QUOTIENT_HUMAN_CORPUS='E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl'
Remove-Item Env:TSS_SHARED_FRAGMENT_CASE_ID -ErrorAction SilentlyContinue
$env:TSS_SHARED_FRAGMENT_LAZY_MODE='off'
cargo test --release -p hexfield_eq shared_fragment_soundness_and_warm_campaign -- --ignored --test-threads=1 --nocapture

Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:TSS_SHARED_FRAGMENT_LAZY_MODE='on'
cargo test --release -p hexfield_eq shared_fragment_soundness_and_warm_campaign -- --ignored --test-threads=1 --nocapture
```

Reduced-TT commands use one budget and one profile per invocation. Set
`TSS_SHARED_FRAGMENT_REDUCED_TT_BYTES` to `536870912` and then `1073741824`:

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:TSS_SHARED_FRAGMENT_HEAVY_IDS='0l4291i_live'
$env:TSS_SHARED_FRAGMENT_LAZY_MODE='off'
$env:TSS_SHARED_FRAGMENT_REDUCED_LADDER='1000000'
cargo test --release -p hexfield_eq shared_fragment_reduced_tt_campaign -- --ignored --test-threads=1 --nocapture

Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:TSS_SHARED_FRAGMENT_REDUCED_LADDER='10000,100000,1000000'
cargo test --release -p hexfield_eq shared_fragment_reduced_tt_campaign -- --ignored --test-threads=1 --nocapture
```

Official gates require more than 11 GiB free RAM and clear all corpus selectors:

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_BACKWALK_TT_BYTES='2147483648'
$env:TSS_SHARED_FRAGMENTS='1'
$env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS='1'
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER='0'
Remove-Item Env:TSS_LAZY_FRONTIER -ErrorAction SilentlyContinue
Remove-Item Env:TSS_CORPUS_FILE,Env:TSS_CORPUS_ID,Env:TSS_CORPUS_MAX_CAP -ErrorAction SilentlyContinue
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture

Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:TSS_LAZY_FRONTIER='1'
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER='1'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture
```
