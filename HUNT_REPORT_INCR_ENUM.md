# R-IE1 — incremental defender enumeration

Date: 2026-07-17  
Branch/base: `hunt/incr-enum` / `d5a2b5fd94ef5ff75ecbb4b08e80da53e8636832`  
Scope: cfg(test), default-off hunt; `94gnnol` excluded by owner ruling

## Verdict

**PROMOTE by the stated round bars, but keep the implementation test-only and
default-off pending a memory-layout round.** Phase 1 found that exact parent
state carries essentially the entire defender input: 2,910,072 of 2,910,351
calls (99.9904%) had a parent fingerprint, all 2,910,072 matched the batch
family/kernel, and all 11,002,776 one-stone residuals matched the bounded local
patch. There were zero parent or residual mismatches.

The conservative carryable functional time was **34.488 s**, or **9.33% of
full wall** and **24.30% of the counter-on defender bucket**. This clears the
Phase-1 decision gates of 5% full wall / 15% defender wall, so Phase 2 was
authorized.

The final consuming A/B reduced the official 18-position test wall from
**353.99 s to 326.94 s (-7.64%)** and summed solve wall from **353.638 s to
326.596 s (-7.65%)**. All 31 rows were identical on status, nodes, expansions,
TT entries/hits/peak bytes, stage refreshes, and gate/seed counters. The h8 and
h16 cap-500 leaf cells also had exact search/stat identity and no regression:
**77.285 -> 71.613 ms (-7.34%)** and **497.616 -> 473.346 ms (-4.88%)**.

The caveat is memory. The fixed inline parent snapshots add **370,553,216
bytes (353.39 MiB) of accounted peak heap payload** on the deep profile while
TT peak stays identical. That is not a stated promotion disqualifier, but it is
too large to recommend production wiring without a follow-up compact sidecar
or selected-edge reconstruction design. The production build does not contain
the implementation.

## Phase 1 — counters and decision

### Counter design

`TSS_INCR_ENUM_COUNTERS=1` enables cfg(test)-only observation of the existing
batch `forced_defender_pair_plan`:

- the accepted attacker parent records a stable fingerprint of the exact
  post-pair T6 threat family (window keys and empty cells) and emitted K2
  kernel;
- the defender child independently rebuilds the batch family/kernel and
  compares its fingerprint;
- for every root kernel cell, the counter patches the root family by deleting
  exactly the windows hit by that cell, derives the K1 residual kernel, and
  compares it to the independently rebuilt post-make batch result;
- timing is split into root analysis/enumeration, canonical frame, fork scan,
  make/unmake, residual analysis/enumeration, final keys, fingerprinting, and
  residual overhead;
- input shape histograms and a run-wide fingerprint XOR price the later
  pair-classification micro-round.

The fingerprint is measurement telemetry, not a soundness premise. Phase 2's
shadow gate compares full structures field-for-field.

### Official 1 GiB result

Configuration: 18 in-scope corpus positions, 1 GiB TT, lazy frontier on,
interior census gate on, all other solver levers off, one MSVC test thread.
The run passed all 14 WIN and four in-scope NO positions.

| Metric | Result |
|---|---:|
| Ladder rows | 31 |
| Test / summed solve wall | 369.52 s / 369.181 s |
| Nodes / expansions | 3,397,362 / 3,397,331 |
| Defender calls / successful plans | 2,910,351 / 2,910,351 |
| Parent fingerprints available | 2,910,072 (99.9904%) |
| Parent exact / mismatch | 2,910,072 / **0** |
| Residual patches exact / mismatch | 11,002,776 / **0** |
| Counter-on defender bucket | 141.915 s |
| Profiled planner time | 135.703 s |
| Fingerprint overhead inside profile | 2.727 s |
| Fingerprint XOR | `9e9bcaa2f1a631ea` |

Input shapes confirm R-KT1's tiny-width finding. Per root call the means were
2.512 live windows, 4.951 window-cell incidences, 4.118 distinct input cells,
and 3.781 kernel cells. Per residual the means were 1.250 windows, 2.472
incidences, 2.198 distinct cells, and 1.771 K1 cells; the local patch deleted
1.219 windows on average. Observed root windows were 2–6, distinct cells 2–7+,
and kernel cells 2–4. Residual kernels were always one or two cells.

### Cost distribution and gate arithmetic

| Planner component | Time | Profiled planner share | Carryable? |
|---|---:|---:|---|
| Root threat analysis | 0.693 s | 0.51% | yes |
| Root exact T6 enumeration | 1.341 s | 0.99% | yes |
| Canonical frame | 40.730 s | 30.01% | no |
| Fork-prior scan | 15.138 s | 11.16% | no |
| Enumeration-only make/unmake | 28.114 s | 20.72% | yes |
| Residual threat analysis | 2.129 s | 1.57% | yes |
| Residual exact enumeration | 2.211 s | 1.63% | yes |
| Final-key construction | 34.460 s | 25.39% | no |
| Fingerprinting | 2.727 s | 2.01% | instrumentation only |
| Other assembly/timer overhead | 8.160 s | 6.01% | not credited |

The conservative numerator is
`0.692904 + 1.341146 + 28.113886 + 2.129185 + 2.211240 = 34.488361 s`.
It is 9.333% of the 369.52 s counter run, 24.302% of its 141.915 s defender
bucket, and 26.461% of R-KT1's 130.336 s uninstrumented defender bucket.
**Decision: proceed to Phase 2.**

## Phase 2 — implementation and soundness

One cfg(test)-only env controls the implementation:

- absent/`0`: historical batch planner;
- `TSS_INCR_DEFENDER=shadow`: build incremental and batch plans, abort on the
  first structural mismatch, and consume the batch plan;
- `TSS_INCR_DEFENDER=1`: consume the incremental plan, with exact batch
  fallback when no bounded parent snapshot is available.

The parent gate fuses tau=2 classification with pair-incidence collection and
stores a fixed inline snapshot. Its explicit capacity is eight live threat
windows, two empties per live >=4 window, and eight kernel cells. The official
maxima were six/four. A shape outside the bound is represented by no snapshot
and therefore takes the unchanged batch path; nothing is truncated.

At the defender child, a first reply deletes the snapshot windows containing
that cell. The remaining family directly yields the exact K1 seconds. The
planner retains the historical canonical-frame ordering and fork-prior scan,
and builds the final position key from the root plus both defender cells,
without touching the engine.

### Hard shadow-equality gate

`INCR_DEFENDER_SHADOW_FULL_COMPACT_RAW.log` is the binding final-form audit:

| Metric | Result |
|---|---:|
| Incremental calls | 2,910,072 |
| Full plan equalities | **2,910,072** |
| Mismatches | **0** |
| Safe batch fallbacks | 279 |
| Nodes / expansions | 3,397,362 / 3,397,331 |
| All corpus expectations | PASS |
| Test wall (not an A/B arm) | 451.16 s |

Equality covers `Some`/`None` shape, canonical kernel order, atomic pair order,
both coordinates, exact final keys, and PN/DN priors. The abort specimen
contains the full placement history, parent snapshot, batch plan, and
incremental plan. No specimen fired.

### Official deep A/B

| Metric | Batch off | Incremental on | Delta |
|---|---:|---:|---:|
| Test wall | 353.99 s | **326.94 s** | **-27.05 s (-7.64%)** |
| Summed solve wall | 353.638 s | **326.596 s** | **-27.042 s (-7.65%)** |
| Nodes | 3,397,362 | 3,397,362 | 0 |
| Expansions | 3,397,331 | 3,397,331 | 0 |
| Peak TT bytes | 549,161,606 | 549,161,606 | 0 |
| Pair generation | 158.682 s | 162.649 s | +3.967 s (+2.50%) |
| Defender planning | 130.900 s | **98.970 s** | -31.930 s (-24.39%) |
| Expansion inclusive | 293.472 s | 265.492 s | -27.980 s (-9.53%) |
| Parent maintenance | 0 | 0.610 s | +0.610 s |
| Incremental plan work | 0 | 92.856 s | +92.856 s |
| Safe batch fallbacks | 0 | 279 | +279 |
| Accounted snapshot peak | 0 | 370,553,216 B | +353.39 MiB |

A parser compared all 31 row records and found zero differences across status,
expected class, nodes, expansions, TT entries/hits/cap/peak, stage refreshes,
gate evaluations/dismissals/time, and seed counters. The only permitted delta,
wall, changed.

### Phase-3 leaf cells

Configuration D from `HUNT_REPORT_LEAF_SURFACE.md`: wide pair-complete, lazy,
interior gate, shared fragments/K-reply off, cap 500, 256 KiB TT, 300 solves
per cell. Every hard result was strict-verified.

| Horizon | Batch wall | Incremental wall | Delta | Verdicts | Expansions | Peak snapshot |
|---:|---:|---:|---:|---:|---:|---:|
| h8 | 77.285 ms | **71.613 ms** | -7.34% | 16 / 16 | 1,852 / 1,852 | 19,312 B |
| h16 | 497.616 ms | **473.346 ms** | -4.88% | 39 / 39 | 6,649 / 6,649 | 194,072 B |

Nodes, expansions, TT hits, maximum TT entries, peak TT bytes, admission
rejections, stage refreshes, statuses, and verified-hard counts were identical
in each cell. There was no leaf regression.

## Integrity and recommendation

- `TSS_INCR_ENUM_COUNTERS` and `TSS_INCR_DEFENDER` are cfg(test), default-off.
- The final non-test MSVC release build passed. None of the counter,
  snapshot, mode, shadow, or consumer code is compiled into production.
- `tss_verify.rs` was untouched; every hard leaf/corpus result continued to
  pass the strict verifier.
- The fresh default-off full profile reproduces the R-KT1 search totals and
  its `mvp2lvc,xsnfyll` fast-subset rows exactly. Flag-off emits zero
  incremental calls and zero snapshot bytes.
- The deep performance and leaf bars pass. The round therefore records a
  **PROMOTE** result, but production consumption should wait for a dedicated
  memory-layout round. A selected-edge bounded reconstruction was measured in
  development with zero retained payload, but its maintenance cost was too
  high in the smoke; it was not selected or retained in the final path.

## Files changed

- `HUNT_REPORT_INCR_ENUM.md` — this report, decisions, raw index, commands.
- `packages/hexfield_eq/rust/src/tss_core.rs` — cfg(test) counter and
  incremental-maintenance statistics.
- `packages/hexfield_eq/rust/src/tss_solver.rs` — Phase-1 instrumentation,
  fixed snapshot, shadow equality, and consuming planner.
- `packages/hexfield_eq/rust/src/tss_corpus.rs` — env assertions and row/final
  telemetry.
- `packages/hexfield_eq/rust/src/tss_leaf_surface_hunt.rs` — h8/h16 cap-500
  A/B cell.

## Retained raw logs

Binding evidence:

- `INCR_ENUM_PHASE1_FULL_RAW.log` — official Phase-1 counter profile.
- `INCR_DEFENDER_SHADOW_FULL_COMPACT_RAW.log` — final-form corpus-wide shadow
  equality gate.
- `INCR_DEFENDER_AB_OFF_FULL_RAW.log` / `INCR_DEFENDER_AB_ON_FULL_RAW.log` —
  same-build official deep A/B.
- `INCR_DEFENDER_LEAF_AB_RAW.log` — h8/h16 cap-500 leaf A/B.
- `INCR_ENUM_PRODUCTION_BUILD_RAW.log` — passing non-test release build.
- `INCR_ENUM_RAM_GATES.log` — every free/standby/availability launch reading.

Development provenance retained:

- `INCR_ENUM_BUILD_RAW.log`, `INCR_ENUM_UNIT_RAW.log`,
  `INCR_ENUM_FINAL_UNIT_RAW.log`,
  `INCR_ENUM_COUNTER_SMOKE_RAW.log`;
- `INCR_DEFENDER_BUILD_RAW.log`, `INCR_DEFENDER_COMPACT_UNIT_RAW.log`;
- `INCR_DEFENDER_SHADOW_SMOKE_RAW.log`,
  `INCR_DEFENDER_SHADOW_SMOKE_FUSED_RAW.log`;
- `INCR_DEFENDER_SHADOW_FULL_RAW.log` (pre-compact heap-backed shadow);
- `INCR_DEFENDER_SELECTED_DELTA_SMOKE_RAW.log` (zero-retention alternative).

The first official Phase-1 launch waited two mandated five-minute intervals
for other-lane Cargo processes; its test wall is 369.52 s, while the tool
elapsed time includes that wait. All gate-class launches had at least 21.67
GiB availability and 13.89 GiB free physical. All ordinary launches exceeded
20.68/13.91 GiB.

## Exact regeneration commands

Run from the worktree root. Every Cargo command below uses `.target-ie`, the
MSVC target, and one test thread. `Wait-IncrCargoGate` implements the required
host-wide five-minute wait and logs both RAM components.

```powershell
function Wait-IncrCargoGate([double]$RequiredAvailabilityGiB,
                            [double]$RequiredFreeGiB,
                            [string]$Purpose) {
    while (@(Get-Process cargo -ErrorAction SilentlyContinue).Count -gt 0) {
        Get-Process cargo | Select-Object Id, StartTime, CPU
        Start-Sleep -Seconds 300
    }
    $ieOs = Get-CimInstance Win32_OperatingSystem
    $ieFreeBytes = [double]$ieOs.FreePhysicalMemory * 1KB
    $ieStandbyBytes = [double](Get-Counter `
        '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue
    $ieAvailableBytes = $ieFreeBytes + $ieStandbyBytes
    [pscustomobject]@{
        Purpose = $Purpose
        Timestamp = Get-Date -Format o
        FreeGiB = $ieFreeBytes / 1GB
        StandbyGiB = $ieStandbyBytes / 1GB
        AvailabilityGiB = $ieAvailableBytes / 1GB
        RequiredAvailabilityGiB = $RequiredAvailabilityGiB
        RequiredFreeGiB = $RequiredFreeGiB
    } | Format-List | Out-File INCR_ENUM_RAM_GATES.log -Append
    if ($ieAvailableBytes -lt $RequiredAvailabilityGiB * 1GB -or
        $ieFreeBytes -lt $RequiredFreeGiB * 1GB) {
        throw 'R-IE1 RAM gate failed'
    }
}

function Set-IncrOfficialEnv([string]$Mode = 'off',
                            [bool]$Counters = $false) {
    Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
        ForEach-Object { Remove-Item "Env:$($_.Name)" }
    $env:CARGO_TARGET_DIR = '.target-ie'
    $env:TSS_BACKWALK_TT_BYTES = '1073741824'
    $env:TSS_LAZY_FRONTIER = '1'
    $env:TSS_INTERIOR_CENSUS_GATE = '1'
    if ($Counters) { $env:TSS_INCR_ENUM_COUNTERS = '1' }
    if ($Mode -ne 'off') { $env:TSS_INCR_DEFENDER = $Mode }
    $env:TSS_CORPUS_ID = @(
        '0hz3hty','0l4291i_live','8is963b','acly7kb','dy3dg99','g2xx6wl',
        'hu01jk4','jh7yo7y','jnzzmcm','l9mxn59','lz60mfb','mvp2lvc',
        'xsnfyll','zrugh2x','strongloss_a_prefix6','strongloss_b_prefix8',
        'hayes_20260712_turn16','hayes_20260712_placement31'
    ) -join ','
    $env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS = '0'
    $env:TSS_CORPUS_EXPECT_LAZY_FRONTIER = '1'
    $env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE = '1'
    $env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME = '0'
    $env:TSS_CORPUS_EXPECT_CAP_RESUME = '0'
    $env:TSS_CORPUS_EXPECT_LIVE_GE3_SEED = '0'
    $env:TSS_CORPUS_EXPECT_CLOSURE_COUNTERS = '0'
    $env:TSS_CORPUS_EXPECT_THRESHOLD_COUNTERS = '0'
    $env:TSS_CORPUS_EXPECT_THRESHOLD_DELTA = 'off'
    $env:TSS_CORPUS_EXPECT_INCR_ENUM_COUNTERS = if ($Counters) { '1' } else { '0' }
    $env:TSS_CORPUS_EXPECT_INCR_DEFENDER = $Mode
}
```

Focused counter/unit test:

```powershell
Wait-IncrCargoGate 10 5 'counter_unit'
$env:CARGO_TARGET_DIR = '.target-ie'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq incremental_defender_counter -- `
    --test-threads=1 --nocapture *> INCR_ENUM_UNIT_RAW.log
```

Phase-1 official counters:

```powershell
Wait-IncrCargoGate 12 6 'phase1_official_1gib'
Set-IncrOfficialEnv 'off' $true
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture *> INCR_ENUM_PHASE1_FULL_RAW.log
```

Final compact shadow equality:

```powershell
Wait-IncrCargoGate 12 6 'phase2_shadow_full_compact_1gib'
Set-IncrOfficialEnv 'shadow' $false
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture *> INCR_DEFENDER_SHADOW_FULL_COMPACT_RAW.log
```

Official deep A/B:

```powershell
Wait-IncrCargoGate 12 6 'phase2_ab_off_1gib'
Set-IncrOfficialEnv 'off' $false
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture *> INCR_DEFENDER_AB_OFF_FULL_RAW.log

Wait-IncrCargoGate 12 6 'phase2_ab_on_1gib'
Set-IncrOfficialEnv '1' $false
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture *> INCR_DEFENDER_AB_ON_FULL_RAW.log
```

Phase-3 leaf A/B (the test itself runs off then on for h8/h16 cap 500):

```powershell
Wait-IncrCargoGate 10 5 'phase2_leaf_ab_h8_h16_cap500'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-ie'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq incr_defender_leaf_ab -- `
    --ignored --test-threads=1 --nocapture *> INCR_DEFENDER_LEAF_AB_RAW.log
```

Non-test production build:

```powershell
Wait-IncrCargoGate 10 5 'production_release_build'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-ie'
cargo build --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq *> INCR_ENUM_PRODUCTION_BUILD_RAW.log
```
