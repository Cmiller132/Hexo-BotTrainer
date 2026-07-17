# R-RS1 — opening-root stabilizer quotient sizing

Date: 2026-07-17

Worktree: `hunt-root-stabilizer`

Audited base: `a49e8abd97cd49ffb2c653e23e62d51c8103cc38`

## Verdict

**PARTIAL STOP; DO NOT PROMOTE.** The implementation and bounded soundness
campaign are clean, but the binding 10k/100k sizing matrix did not complete.
The completed all-transform diagnostic at cap 128 is a null: the
game-frequency-weighted wall delta is **-0.2388%** (a regression), the
family-weighted delta is **+0.2413%**, and expansions are identical. This is
well below the 5% bar, but it is not substituted for the required 10k rung.
Accordingly this report makes no formal PROMOTE claim and does not claim that
the official STOP bar was executed at its prescribed cap.

The reason for stopping is measured, not speculative. One top-family,
one-transform, two-arm 10k cell failed to finish inside a 604 s outer command
window. Its orphaned test process was still making progress at 1,076.8 CPU-s,
but its output channel was gone and it had emitted no completed row. It was
stopped and no datum from that attempt is used. At that rate the complete
top-10 × 12-transform × 2-arm × 2-cap matrix is a many-hour campaign. The
assignment explicitly permits an honest partial stop at a clean boundary;
the completed boundary here is census + top-10 shadow + cap-128 twelve-way
differential A/B + focused soundness tests.

## Corpus census

The Rust harness independently reproduces the binding 6,902-game census.

| Root-family stabilizer | Families | Games | Game share |
|---:|---:|---:|---:|
| 1 | 201 | 2,559 | 37.0762% |
| 2 | 53 | 4,088 | 59.2292% |
| 4 | 8 | 255 | 3.6946% |
| **Nontrivial** | **61** | **4,343** | **62.9238%** |

The corpus-wide generic root-child-removal ceiling is **32.3855%**. The top
10 families cover 4,212 games (**61.0258%**). Their measured complete-edge
universes have a family-weighted child removal of **18.5788%** and a
game-frequency-weighted child removal of **36.0354%**. Those child-count
figures are ceilings, not wall gains.

## Top-10 shadow root telemetry

This shadow run uses the atlas-capable quiet-turn profile, the official 1 GiB
environment, and cap 128. `below_*` is attributed to the selected root orbit;
every row visited exactly one orbit and spent all 126 below-root expansions
there. Unvisited orbits therefore have exactly zero expansions and zero wall.

| Rank | Family | Games | Stab | Raw children | Orbits | Removed | Fixed | Root gen ms | Below exp. | Below wall ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `(-1,0);(0,1)` | 1,337 | 2 | 51,752 | 26,030 | 25,722 | 308 | 594.116 | 126 | 18.821 |
| 2 | `(-1,0);(-1,1)` | 1,157 | 2 | 49,452 | 24,808 | 24,644 | 164 | 560.514 | 126 | 18.473 |
| 3 | `(-2,0);(-2,2)` | 523 | 2 | 57,507 | 28,846 | 28,661 | 185 | 650.877 | 126 | 18.275 |
| 4 | `(-3,0);(-1,1)` | 258 | 1 | 59,707 | 59,707 | 0 | 59,707 | 678.171 | 126 | 18.482 |
| 5 | `(-2,1);(0,-1)` | 175 | 1 | 54,428 | 54,428 | 0 | 54,428 | 665.648 | 126 | 18.606 |
| 6 | `(-2,0);(0,2)` | 169 | 2 | 62,263 | 31,294 | 30,969 | 325 | 735.371 | 126 | 18.809 |
| 7 | `(-9,1);(-8,0)` | 165 | 1 | 93,885 | 93,885 | 0 | 93,885 | 1,135.657 | 126 | 18.721 |
| 8 | `(-2,1);(-1,0)` | 149 | 1 | 51,752 | 51,752 | 0 | 51,752 | 693.902 | 126 | 24.664 |
| 9 | `(-2,0);(0,1)` | 146 | 1 | 56,878 | 56,878 | 0 | 56,878 | 845.667 | 126 | 24.948 |
| 10 | `(-2,0);(-1,1)` | 133 | 1 | 54,428 | 54,428 | 0 | 54,428 | 810.431 | 126 | 25.185 |

The key sizing observation is that quotientable siblings were not reached at
the bounded rung: the search remained inside one retained orbit. Removing
25k–31k unvisited sibling edges from the four symmetric top-10 families did
not remove any expansion at cap 128.

## Twelve-transform A/B at cap 128

Each family row is the mean of all 12 D6 transforms. Arms were alternated by
transform to reduce fixed order bias. All 240 arms returned `UNKNOWN`; there
were no hard certificates at this cap, so the strict-verifier obligation was
vacuous for these rows. The implementation materializes any future proof as
ordinary nested `Choice` nodes, and the harness rejects a hard row unless the
unchanged `TssVerifier` accepts it. No hard row was reached in this partial
round.

| Rank | Stab | Removed | Baseline ms | Quotient ms | Wall delta | Baseline exp. | Quotient exp. | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 25,722 | 763.800 | 765.116 | -0.1722% | 1,524 | 1,524 | UNKNOWN |
| 2 | 2 | 24,644 | 785.522 | 794.750 | -1.1747% | 1,524 | 1,524 | UNKNOWN |
| 3 | 2 | 28,661 | 910.965 | 908.177 | +0.3060% | 1,524 | 1,524 | UNKNOWN |
| 4 | 1 | 0 | 942.710 | 963.333 | -2.1876% | 1,524 | 1,524 | UNKNOWN |
| 5 | 1 | 0 | 857.820 | 864.851 | -0.8195% | 1,524 | 1,524 | UNKNOWN |
| 6 | 2 | 30,969 | 981.023 | 935.095 | +4.6816% | 1,524 | 1,524 | UNKNOWN |
| 7 | 1 | 0 | 1,315.908 | 1,313.511 | +0.1821% | 1,524 | 1,524 | UNKNOWN |
| 8 | 1 | 0 | 675.591 | 667.951 | +1.1309% | 1,524 | 1,524 | UNKNOWN |
| 9 | 1 | 0 | 739.088 | 750.634 | -1.5622% | 1,524 | 1,524 | UNKNOWN |
| 10 | 1 | 0 | 719.628 | 707.664 | +1.6626% | 1,524 | 1,524 | UNKNOWN |

| Aggregate | Baseline wall | Quotient wall | Delta | Expansions before/after |
|---|---:|---:|---:|---:|
| Family weighted | 8,692.056 ms | 8,671.081 ms | **+0.2413%** | 15,240 / 15,240 |
| Game-frequency weighted | 3,487,020.693 weighted-ms | 3,495,347.851 weighted-ms | **-0.2388%** | 15,240 / 15,240 |

This bounded result is a measured null, not evidence for promotion. It is
also consistent with the orbit telemetry: all useful work was below one
retained representative, so child-list removal did not change the node
budget.

## 10k/100k sizing status

| Requested cell | Status | Usable data |
|---|---|---|
| Top 10 × 10k × 12 transforms × A/B | Not completed | No |
| Top 10 × 100k × 12 transforms × A/B | Not started (10k prerequisite did not complete) | No |
| Top family × 10k × one transform × A/B smoke | Timed out before first completed row; orphan stopped | No |

No cap was extended. No result from the incomplete 10k process is included in
an aggregate. The formal >=5% PROMOTE bar is therefore not met, and the branch
must remain default-off.

## Shadow implementation and soundness

The implementation is confined to `cfg(test)` code plus a `cfg(test)` hunt
module registration. Consumption additionally requires
`TSS_ROOT_STABILIZER_CONSUME=1`; flag-off production has no quotient field or
branch.

- Eligibility is exact: placement clock 3, `FirstStone`, nonterminal, current
  player equals claimant, Choice root, atlas quiet-turn profile.
- The stabilizer compares the owner-labelled sorted occupancy, current player,
  claimant, phase (including a transformed pending first stone), placement
  clock, terminal value, semantic horizon, depth/profile, and the explicit
  static Hexo rules/profile identity.
- The 12 candidate transforms are filtered by exact complete-binding equality,
  and the resulting set is checked for identity and group closure.
- Root edges are complete two-application semantics. A pair becomes unordered
  only when both application orders are generated legally and both are
  nonterminal; frontier-growth-only pairs remain ordered. Orbit construction
  transforms that exact edge key.
- Each orbit retains one actual raw edge. If it proves, the materializer emits
  two ordinary nested `Choice` nodes with the retained concrete moves. No
  symmetry assertion enters the certificate.
- The quotient never runs at a Universal node. Unexpected node shape,
  transform overflow, missing image, duplicate/overlapping orbit, subgroup
  failure, or injected inconsistency falls back to the full edge list.
- Focused tests compare all 12 transforms and inject an inconsistency; the
  injected arm has identical status, nodes, and expansions to the full-list
  shadow arm.
- `packages/hexfield_eq/rust/src/tss_verify.rs` is untouched (`git diff` is
  empty).

## Files

- `packages/hexfield_eq/rust/src/tss_solver.rs` — test-only binding,
  stabilizer/orbit construction, telemetry, fail-closed root A/B, and normal
  nested-Choice certificate seam.
- `packages/hexfield_eq/rust/src/tss_root_stabilizer_hunt.rs` — corpus census,
  ranked A-0 roots, twelve-transform differential, strict verification, shadow
  and sizing campaigns.
- `packages/hexfield_eq/rust/src/lib.rs` — `cfg(test)` module registration.
- `ROOT_STABILIZER_UNIT_RAW.log` — focused soundness test transcript.
- `ROOT_STABILIZER_SHADOW_RAW.log` — exact census and top-10 shadow telemetry.
- `ROOT_STABILIZER_CAP128_AB_RAW.log` — all 240 bounded A/B arms and aggregates.
- `ROOT_STABILIZER_10K_TIMEOUT_RAW.log` — explicitly unusable incomplete-cell
  record.
- `ROOT_STABILIZER_RAM_GATES.log` — free, standby, availability, Cargo count,
  thresholds, and timestamps.
- `HUNT_REPORT_ROOT_STABILIZER.md` — this report.

Pre-existing `.codex-rs/` was preserved. `.target-rs/` is the required local
build output and is not a deliverable. No commit was created.

## Exact regeneration commands

Run from the worktree root in PowerShell. These commands enforce the required
MSVC target, `.target-rs`, one test thread, one host-wide Cargo process, and
the relaxed 07-17 RAM gates.

```powershell
function Wait-RootStabilizerCargoGate([double]$RequiredAvailabilityGiB,
                                      [double]$RequiredFreeGiB,
                                      [string]$Purpose) {
    while (@(Get-Process cargo -ErrorAction SilentlyContinue).Count -gt 0) {
        Get-Process cargo | Select-Object Id, StartTime, CPU
        Start-Sleep -Seconds 300
    }
    $rsOs = Get-CimInstance Win32_OperatingSystem
    $rsFreeBytes = [double]$rsOs.FreePhysicalMemory * 1KB
    $rsStandbyBytes = [double](Get-Counter `
        '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue
    $rsAvailableBytes = $rsFreeBytes + $rsStandbyBytes
    [pscustomobject]@{
        Purpose = $Purpose
        Timestamp = Get-Date -Format o
        FreeGiB = $rsFreeBytes / 1GB
        StandbyGiB = $rsStandbyBytes / 1GB
        AvailabilityGiB = $rsAvailableBytes / 1GB
        RequiredAvailabilityGiB = $RequiredAvailabilityGiB
        RequiredFreeGiB = $RequiredFreeGiB
        CargoProcesses = 0
    } | Format-List | Out-File ROOT_STABILIZER_RAM_GATES.log -Append
    if ($rsAvailableBytes -lt $RequiredAvailabilityGiB * 1GB -or
        $rsFreeBytes -lt $RequiredFreeGiB * 1GB) {
        throw 'R-RS1 RAM gate failed'
    }
}

function Set-RootStabilizerOfficialEnv {
    Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
        ForEach-Object { Remove-Item "Env:$($_.Name)" }
    $env:CARGO_TARGET_DIR = '.target-rs'
    $env:TSS_BACKWALK_TT_BYTES = '1073741824'
    $env:TSS_LAZY_FRONTIER = '1'
    $env:TSS_INTERIOR_CENSUS_GATE = '1'
    $env:TSS_INCR_DEFENDER = '1'
}
```

Focused soundness units:

```powershell
Wait-RootStabilizerCargoGate 10 5 'root_stabilizer_units'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-rs'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq root_stabilizer_ -- `
    --test-threads=1 --nocapture *> ROOT_STABILIZER_UNIT_RAW.log
```

Top-10 shadow census used in this report:

```powershell
Wait-RootStabilizerCargoGate 12 6 'root_stabilizer_shadow_1gib'
Set-RootStabilizerOfficialEnv
$env:TSS_ROOT_STABILIZER_SHADOW_CAP = '128'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq root_stabilizer_atlas_shadow_campaign -- `
    --ignored --test-threads=1 --nocapture *> ROOT_STABILIZER_SHADOW_RAW.log
```

Bounded twelve-transform A/B used in this report:

```powershell
Wait-RootStabilizerCargoGate 12 6 'root_stabilizer_cap128_ab_1gib'
Set-RootStabilizerOfficialEnv
$env:TSS_ROOT_STABILIZER_FAMILIES = '10'
$env:TSS_ROOT_STABILIZER_TRANSFORMS = '12'
$env:TSS_ROOT_STABILIZER_CAPS = '128'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_root_stabilizer_atlas_campaign -- `
    --ignored --test-threads=1 --nocapture *> ROOT_STABILIZER_CAP128_AB_RAW.log
```

Binding official sizing command (not completed in this round). The defaults
are top 10, all 12 transforms, and caps `10000,100000`; do not raise caps until
the 10k/100k cells finish. Allow a multi-hour wall budget and retain the raw.

```powershell
Wait-RootStabilizerCargoGate 12 6 'root_stabilizer_official_10k_100k_1gib'
Set-RootStabilizerOfficialEnv
$env:TSS_ROOT_STABILIZER_FAMILIES = '10'
$env:TSS_ROOT_STABILIZER_TRANSFORMS = '12'
$env:TSS_ROOT_STABILIZER_CAPS = '10000,100000'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_root_stabilizer_atlas_campaign -- `
    --ignored --test-threads=1 --nocapture *> ROOT_STABILIZER_ATLAS_RAW.log
```
