# G2R9 shared proven fragments: stopped campaign report

Date: 2026-07-17  
Branch/base: `hunt/turn-quotient` at `86a6418c`  
Rollout flag: `TSS_SHARED_FRAGMENTS=1` (default off)  
Status: **IMPLEMENTED, DEFAULT-OFF, SOUNDNESS DESIGN REVIEWED; ROLLOUT GATE
STOPPED ON VERDICT NON-IDENTITY. NO OFFICIAL GATE CLAIM.**

## Verdict

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

## Requested headline gates not completed

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

## Regeneration

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

The full campaign command below is documented for a future owner decision; it
must not be represented as green at this revision:

```powershell
Remove-Item Env:TSS_SHARED_FRAGMENT_CASE_ID -ErrorAction SilentlyContinue
$env:TSS_SHARED_FRAGMENT_LAZY_MODE='off'
cargo test --release -p hexfield_eq shared_fragment_soundness_and_warm_campaign -- --ignored --test-threads=1 --nocapture
```

