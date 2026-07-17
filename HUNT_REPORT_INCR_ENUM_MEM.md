# R-IE2 - incremental enumeration memory round

Date: 2026-07-17  
Branch/HEAD baseline: `hunt/incr-enum` / `2f46925eafc79b5f38747c1d6086dd5f025b4de0`  
Binding baseline: `HUNT_REPORT_INCR_ENUM.md`; `94gnnol` remains excluded

## Verdict

**PASS every R-IE2 bar. Selected-edge reconstruction is the production-layout
candidate.** The R-IE1 fixed snapshot is no longer retained in a lazy edge or
arena node. When df-pn selects an attacker pair whose defender child is still
unexpanded, R-IE2 applies the pair as usual, reconstructs the exact tiny T6
family/kernel from that selected child position, and carries the fixed bounded
value only through the recursive call stack. It is restored on unwind. An
already-expanded/transposed child needs no snapshot; a defender expansion with
no active selected attacker-pair frame, or an out-of-bound shape, takes the
unchanged exact batch path.

The mandatory full shadow run compared **2,910,349 incremental plans
field-for-field**, with **2,910,349 equal, zero mismatches, and two exact batch
fallbacks**. All 2,910,627 reconstruction attempts succeeded. The two
fallbacks are the cap-10,000 and cap-100,000 rows of
`hayes_20260712_placement31`; those defender expansions have no active selected
attacker-pair frame. They are not lazy, stage-refresh, or reconstruction-shape
failures.

The official same-build deep A/B improved test wall from **360.50 s to 328.99
s (-8.74%)** and summed solve wall from **360.149 s to 328.639 s (-8.75%)**.
All 31 rows are search-stat identical. Against the retained R-IE1 batch raw,
the new consuming arm is **-7.06%** official wall; against the retained R-IE1
consuming raw it is 2.05 s (+0.63%) slower in absolute wall on a host whose new
batch arm was 6.51 s slower. The same-build percentage win is larger than
R-IE1's -7.64%.

**Hard memory line: peak accounted incremental snapshot heap payload is 0
bytes, PASS <= 32 MiB.** R-IE1 charged 370,553,216 bytes (353.39 MiB). R-IE2
has no snapshot `Arc`, box, arena record, lazy-edge payload, or TT payload.
The bounded snapshot is inline and path-local; reconstruction scratch is
short-lived and one selected edge at a time. Peak TT bytes remain exactly
549,161,606.

The implementation remains cfg(test), default-off. A non-test MSVC release
build passed and contains none of the mode, snapshot, reconstruction, shadow,
or telemetry path. `tss_verify.rs` is untouched.

## Design decision

### Selected path-local reconstruction (chosen)

The key observation is that the only consumer is the first expansion of the
post-attacker-pair defender child. At selection, the normal descent has already
applied both stored edge coordinates, so the engine itself is the exact source
for the child T6 family. R-IE2 scans that family once, computes K2 once, copies
the bounded family/kernel into `IncrDefenderSnapshot`, and invokes the proven
R-IE1 residual patcher.

The active value is a solver slot whose previous value lives in the recursive
call frame. No node, future key, deferred frontier, arena entry, or TT entry
owns it. Stage re-entry reconstructs again if a linked child is still
unexpanded. A transposed child that is already expanded skips reconstruction
because no defender enumeration will consume it.

Measured full-shadow coverage:

| Item | Count |
|---|---:|
| Defender planner invocations | 2,910,351 |
| Incremental consumers | 2,910,349 |
| Exact batch fallbacks | 2 |
| Reconstruction attempts / successes | 2,910,627 / 2,910,627 |
| Shadow equalities / mismatches | 2,910,349 / **0** |
| Reconstruction maintenance | 1.323 s |
| Retained snapshot heap payload | **0 B** |

The 278 successful reconstructions not consumed by the defender planner are
selected unexpanded child descents that resolve or stop before that planner
site. Their measured cost is included.

### Compact sidecar (not selected)

The observed shapes make a variable-length sidecar feasible, but even a
perfect encoding still creates one retained record per pending node and needs
lifetime/offset management across lazy linking. The selected reconstruction
path achieved zero retained bytes and exceeded the wall bar, so an arena could
not improve the binding memory result and would add implementation surface.

### Pure parent-path snapshot carry (not sufficient alone)

The pair gate runs when an attacker node is expanded, while a lazy edge can be
selected much later. Therefore the original gate-local family is not generally
on the current descent stack at selection. Retaining it until selection is the
R-IE1 memory problem. Reconstructing only the selected edge closes that gap;
the two non-selected-parent consumers safely batch-fallback.

## Soundness ladder

`INCR_ENUM_MEM_SHADOW_FULL_RAW.log` is the binding final-form audit:

| Metric | Result |
|---|---:|
| Corpus expectations | 14 WIN + 4 in-scope NO, PASS |
| Ladder rows | 31 |
| Nodes / expansions | 3,397,362 / 3,397,331 |
| Incremental calls | 2,910,349 |
| Shadow calls / equalities | 2,910,349 / 2,910,349 |
| Mismatches | **0** |
| Exact batch fallbacks | 2 |
| Reconstruction success | 2,910,627 / 2,910,627 |
| Snapshot / peak snapshot payload | 0 B / **0 B** |
| Test wall (not a consuming arm) | 443.48 s |

Equality covers `Some`/`None`, canonical kernel order, atomic pair order, both
coordinates, exact final keys, and PN/DN priors. The abort diagnostic still
prints placement history, reconstructed snapshot, batch plan, and incremental
plan. A parser compared the 31 shadow rows with R-IE1's binding compact-shadow
raw across status, expected class, nodes, expansions, TT entries/hits/cap/peak,
stage refreshes, gate counts, and seed counts: zero differing rows.

## Official deep A/B

| Metric | R-IE2 off | R-IE2 on | Delta |
|---|---:|---:|---:|
| Test wall | 360.50 s | **328.99 s** | **-31.51 s (-8.74%)** |
| Summed solve wall | 360.149 s | **328.639 s** | **-31.510 s (-8.75%)** |
| Nodes | 3,397,362 | 3,397,362 | 0 |
| Expansions | 3,397,331 | 3,397,331 | 0 |
| Peak TT bytes | 549,161,606 | 549,161,606 | 0 |
| Pair generation | 160.831 s | 160.500 s | -0.331 s |
| Defender planning | 133.416 s | **100.518 s** | -32.898 s |
| Expansion inclusive | 298.142 s | **264.856 s** | -33.286 s |
| Reconstruction maintenance | 0 | 1.331 s | +1.331 s |
| Incremental plan work | 0 | 94.031 s | +94.031 s |
| Incremental calls / fallbacks | 0 / 0 | 2,910,349 / 2 | - |
| Peak accounted snapshot payload | 0 | **0** | 0 |

A parser compared all 31 off/on row records and found zero differences across
status, expected class, nodes, expansions, TT entries/hits/cap/peak, stage
refreshes, gate evaluations/dismissals, and seed counters.

### Comparison with both R-IE1 arms

| Consuming wall | Official test | Summed rows | Snapshot peak |
|---|---:|---:|---:|
| R-IE1 batch off | 353.99 s | 353.638 s | 0 B |
| R-IE1 fixed-snapshot on | 326.94 s | 326.596 s | 370,553,216 B |
| **R-IE2 selected reconstruction on** | **328.99 s** | **328.639 s** | **0 B** |

R-IE2 on versus R-IE1 batch is -25.00 s (-7.06%) official and -24.999 s
(-7.07%) summed. R-IE2 on versus R-IE1 fixed-snapshot on is +2.05 s (+0.63%)
official and +2.043 s (+0.63%) summed. Since the new same-build off arm also
moved +6.51 s, the binding A/B comparison is the same-build -8.74% result.

## Leaf cells

Configuration D, cap 500, 256 KiB TT, 300 solves per cell. Every hard result
was strict-verified.

| Horizon | R-IE2 off | R-IE2 on | Delta | Verdicts | Expansions | Snapshot peak |
|---:|---:|---:|---:|---:|---:|---:|
| h8 | 77.639 ms | **70.619 ms** | **-9.04%** | 16 / 16 | 1,852 / 1,852 | 0 B |
| h16 | 505.890 ms | **462.296 ms** | **-8.62%** | 39 / 39 | 6,649 / 6,649 | 0 B |

Nodes, expansions, TT hits, maximum TT entries, peak TT bytes, admission
rejections, stage refreshes, statuses, and verified-hard counts are identical.
Both cells improve, so the no-leaf-regression bar passes.

## Binding bar verdicts

| Bar | Verdict |
|---|---|
| Full corpus shadow before consuming timing | **PASS** - 2,910,349/2,910,349 equal |
| Consume at least -5% vs batch | **PASS** - same-build -8.74%; R-IE1 batch -7.06% |
| All 31 search rows identical | **PASS** - zero differing rows |
| No leaf regression | **PASS** - h8 -9.04%, h16 -8.62% |
| Peak accounted payload <= 32 MiB | **PASS** - 0 B retained snapshot heap |
| TT peak unchanged | **PASS** - 549,161,606 B both arms |
| cfg(test), default-off, production-free | **PASS** - non-test release build green |
| Strict verifier untouched | **PASS** |

## Files changed

- `HUNT_REPORT_INCR_ENUM_MEM.md` - this report, raw index, and commands.
- `packages/hexfield_eq/rust/src/tss_core.rs` - reconstruction telemetry.
- `packages/hexfield_eq/rust/src/tss_solver.rs` - selected-edge path-local
  reconstruction; fixed snapshots removed from lazy edges/nodes.
- `packages/hexfield_eq/rust/src/tss_corpus.rs` - reconstruction telemetry in
  row and final output.

## Retained raw logs

Binding evidence:

- `INCR_ENUM_MEM_SHADOW_FULL_RAW.log` - full field-for-field shadow gate.
- `INCR_ENUM_MEM_AB_OFF_FULL_RAW.log` / `INCR_ENUM_MEM_AB_ON_FULL_RAW.log` -
  official same-build deep A/B.
- `INCR_ENUM_MEM_LEAF_AB_RAW.log` - h8/h16 cap-500 leaf A/B.
- `INCR_ENUM_MEM_PRODUCTION_BUILD_RAW.log` - passing non-test release build.
- `INCR_ENUM_MEM_RAM_GATES.log` - all launch RAM readings.

Development evidence:

- `INCR_ENUM_MEM_UNIT_RAW.log` - focused incremental planner unit.
- `INCR_ENUM_MEM_SHADOW_SMOKE_RAW.log` - selected reconstruction smoke shadow.

All seven launches found no host-wide Cargo process at their gate. Ordinary
launches had at least 21.38 GiB availability / 13.92 GiB free physical;
gate-class 1 GiB launches had at least 21.31 / 13.86 GiB. The trainer was down
and the 07-17 relaxed availability rule applied.

## Exact regeneration commands

Run from the worktree root. Every Cargo command uses `.target-ie`, the MSVC
target, and one test thread.

```powershell
function Wait-IeMemCargoGate([double]$RequiredAvailabilityGiB,
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
    } | Format-List | Out-File INCR_ENUM_MEM_RAM_GATES.log -Append
    if ($ieAvailableBytes -lt $RequiredAvailabilityGiB * 1GB -or
        $ieFreeBytes -lt $RequiredFreeGiB * 1GB) {
        throw 'R-IE2 RAM gate failed'
    }
}

function Set-IeMemOfficialEnv([string]$Mode = 'off') {
    Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
        ForEach-Object { Remove-Item "Env:$($_.Name)" }
    $env:CARGO_TARGET_DIR = '.target-ie'
    $env:TSS_BACKWALK_TT_BYTES = '1073741824'
    $env:TSS_LAZY_FRONTIER = '1'
    $env:TSS_INTERIOR_CENSUS_GATE = '1'
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
    $env:TSS_CORPUS_EXPECT_INCR_ENUM_COUNTERS = '0'
    $env:TSS_CORPUS_EXPECT_INCR_DEFENDER = $Mode
}
```

Focused unit:

```powershell
Wait-IeMemCargoGate 10 5 'selected_reconstruction_unit'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-ie'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq incremental_defender_counter -- `
    --test-threads=1 --nocapture *> INCR_ENUM_MEM_UNIT_RAW.log
```

Full shadow gate (must pass before either consuming arm):

```powershell
Wait-IeMemCargoGate 12 6 'selected_reconstruction_shadow_full_1gib'
Set-IeMemOfficialEnv 'shadow'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture *> INCR_ENUM_MEM_SHADOW_FULL_RAW.log
```

Official deep A/B:

```powershell
Wait-IeMemCargoGate 12 6 'selected_reconstruction_ab_off_full_1gib'
Set-IeMemOfficialEnv 'off'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture *> INCR_ENUM_MEM_AB_OFF_FULL_RAW.log

Wait-IeMemCargoGate 12 6 'selected_reconstruction_ab_on_full_1gib'
Set-IeMemOfficialEnv '1'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture *> INCR_ENUM_MEM_AB_ON_FULL_RAW.log
```

Leaf A/B:

```powershell
Wait-IeMemCargoGate 10 5 'selected_reconstruction_leaf_ab_h8_h16_cap500'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-ie'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq incr_defender_leaf_ab -- `
    --ignored --test-threads=1 --nocapture *> INCR_ENUM_MEM_LEAF_AB_RAW.log
```

Non-test production build:

```powershell
Wait-IeMemCargoGate 10 5 'selected_reconstruction_production_release_build'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-ie'
cargo build --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq *> INCR_ENUM_MEM_PRODUCTION_BUILD_RAW.log
```
