# R-PC1 - pair/planner micro-optimization round

Date: 2026-07-17  
Branch / baseline HEAD: `hunt/incr-enum` / `bcf2cc70cd1400e63bdad714fd5afe743e50a139`  
Binding baselines: `HUNT_REPORT_INCR_ENUM.md`,
`HUNT_REPORT_INCR_ENUM_MEM.md`, and their retained raws  
Scope: `94gnnol` excluded by owner ruling

## Verdict

**PROMOTE.** Three semantics-identical constant-factor rewrites reduce the
official R-IE2 consuming test wall from **328.99 s to 229.62 s**
(-99.37 s, **-30.21%**) and summed solve wall from **328.639 s to
229.263 s** (-99.376 s, **-30.24%**). This exceeds the 3% promotion bar by a
wide margin. The optimized full run has all **31/31 rows identical** to the
retained R-IE2 consuming raw on status, expected class, nodes, expansions, TT
entries/hits/cap/peak, stage refreshes, gate counts, and seed counts.

The selected h8 and h16 leaf cells also preserve exact search/stat identity
and improve wall by 11.46% and 21.06%. Peak TT remains 549,161,606 bytes and
the R-IE2 path-local incremental snapshot remains zero retained heap bytes.

All implementation changes are mode **(a)**: direct production rewrites with
the same exact ordering, membership, and key bytes. No semantic-risk mode (b)
was retained or added. The historical full-sort canonical implementation is
retained only as a `cfg(test)` equality oracle. `tss_verify.rs` is untouched.

## Profile-first baseline and selected frontier

Before editing, the same current-HEAD `mvp2lvc,xsnfyll` cap-10k cohort was run
with both existing counter sets. It reproduced the R-IE decomposition:

| Measured component | Baseline smoke |
|---|---:|
| Pair generation | 1,107.743 ms |
| Pair gate build | 134.043 ms |
| Second candidates | 163.552 ms |
| Pair evaluation | 507.405 ms |
| Pair dedup | 4.185 ms |
| Defender profiled total | 205.590 ms |
| Canonical frame | 46.077 ms |
| Fork scan | 22.025 ms |
| Final keys | 40.761 ms |

The retained campaign-wide pre-round pair profile had already placed the
corresponding full costs at 34.534 s gate build, 43.457 s second candidates,
138.675 s evaluation, and 0.866 s dedup over 1.803 billion evaluations. This
made hash lookup/layout, rather than a new reveal rule, the measured target.
R-CD1's sound-reveal ceiling remained closed and was not reopened.

One implementation frontier was selected after the smoke:

1. use the workspace's fast hash tables only at hot maps/sets whose iteration
   order is not observable;
2. reject D6 frames by phase and first sorted stone before doing a full sort;
3. scan and sort defender root occupancy once, then merge two sorted extras
   into each final key.

No riskier pair cursor, state hash memo, or retained canonical-frame cache was
needed. The selected smoke improved enough to justify the binding ladder.

## Per-optimization attribution

### 1. Hot pair/planner maps and sets - mode (a)

The pair gate's coordinate-to-window maps, second-candidate and unordered-pair
sets, defender planner membership/rank indexes, and fork-degree accumulator
now use `AHashMap` / `AHashSet`. These sites only perform keyed lookup,
membership insertion, duplicate detection, or a maximum over values; none
emits hash iteration order. The crate therefore adds the already-pinned
workspace `ahash` dependency.

On the identical counter smoke:

| Pair subcomponent | Before | After | Delta |
|---|---:|---:|---:|
| Gate build | 134.043 ms | 116.828 ms | -12.84% |
| Second candidates | 163.552 ms | 91.166 ms | -44.26% |
| Pair evaluation | 507.405 ms | 393.478 ms | -22.45% |
| Dedup | 4.185 ms | 2.739 ms | -34.55% |
| Total pair generation | 1,107.743 ms | 870.515 ms | -21.42% |

In the counter-free official consuming run, the aggregate pair bucket moves
from **160.500 s to 116.107 s** (-44.393 s, **-27.66%**). The same rewrite
also contributes to the separately measured fork scan below.

### 2. Exact canonical-frame contender pruning - mode (a)

The historical implementation transformed and sorted the entire occupied
stone vector for all 12 D6 images. The new implementation first computes all
phase keys and the minimum transformed stone tuple for each image. Any image
whose phase or first sorted tuple is larger cannot win the same lexicographic
ordering. Only exact contenders receive a full transformed sort; ties still
retain the lowest symmetry number.

The `cfg(test)` oracle keeps the historical twelve-sort implementation. A
focused test compares the new result to that oracle at every placement prefix
under all 12 transforms of the forced-defender fixture.

On the official 2,910,351-call component profile, canonical time moves from
**40.730 s to 10.372 s** (-30.358 s, **-74.54%**).

### 3. One sorted root for exact defender keys - mode (a)

The historical key constructor rescanned owner lookups and resorted the same
occupied board for every directed pair. `WideDefenderKeyBuilder` now captures
and sorts root `(q,r,owner)` tuples once per plan. Each exact key sorts its two
defender extras and merge-encodes them with the root tuples using the same
header, zig-zag values, varints, and final boxed byte slice.

The existing forced-pair unit applies both pair orders and compares the
builder result byte-for-byte to `WidePositionKey::from_state`; the incremental
planner unit also retains full batch/incremental plan equality.

On the official component profile, final-key time moves from **34.460 s to
9.085 s** (-25.375 s, **-73.64%**).

### Fork scan and combined planner result

The fork algorithm is unchanged; its local degree map uses the same
order-insensitive fast hash layout. Official fork time moves from **15.138 s
to 12.055 s** (-3.083 s, **-20.37%**).

The full instrumented batch planner total moves from 135.703 s to 80.517 s
(-40.67%) with identical call, shape, patch, and fingerprint counts. In the
binding R-IE2 consuming arm, incremental plan work moves from **94.031 s to
38.880 s** (-58.65%) and the inclusive defender bucket from **100.518 s to
45.479 s** (-54.76%). Reconstruction maintenance is essentially flat/slightly
better, 1.331 s to 1.289 s.

## Official deep result

Configuration: all 18 in-scope corpus positions, 1 GiB TT, lazy frontier on,
interior census gate on, R-IE2 consuming mode on, all other solver levers off,
one MSVC test thread.

| Metric | R-IE2 baseline | R-PC1 | Delta |
|---|---:|---:|---:|
| Test wall | 328.99 s | **229.62 s** | **-30.21%** |
| Summed solve wall | 328.639 s | **229.263 s** | **-30.24%** |
| Pair generation | 160.500 s | **116.107 s** | **-27.66%** |
| Defender planning | 100.518 s | **45.479 s** | **-54.76%** |
| Incremental plan work | 94.031 s | **38.880 s** | **-58.65%** |
| Expansion inclusive | 264.856 s | **165.416 s** | **-37.55%** |
| Nodes / expansions | 3,397,362 / 3,397,331 | identical | 0 / 0 |
| Peak TT bytes | 549,161,606 | 549,161,606 | 0 |
| Incremental calls / fallbacks | 2,910,349 / 2 | identical | 0 / 0 |
| Reconstruction success | 2,910,627 / 2,910,627 | identical | 0 / 0 |
| Peak snapshot payload | 0 B | 0 B | 0 |

`PAIRCLASS_ROW_IDENTITY_RAW.log` records 31/31 identical consuming rows. The
component-profile batch rows are independently 31/31 identical to the R-IE2
off raw. The counter cardinalities and fingerprint remain exactly
2,910,351 calls, 11,002,776 residual patches, zero mismatches, and
`9e9bcaa2f1a631ea`.

## Fast and leaf identity

The counter-free consuming fast subset (`mvp2lvc,xsnfyll`, caps through 100k)
has 3/3 exact rows and improves test wall from 3.04 s to 2.31 s (-24.01%).

Configuration D leaf cells use cap 500, 256 KiB TT, 300 solves per cell,
shared fragments off, and K-reply off. Every returned hard result passed the
unchanged strict verifier.

| Horizon | R-IE2 on | R-PC1 on | Delta | Verdicts | Expansions |
|---:|---:|---:|---:|---:|---:|
| h8 | 70.619 ms | **62.528 ms** | **-11.46%** | 16 / 16 | 1,852 / 1,852 |
| h16 | 462.296 ms | **364.938 ms** | **-21.06%** | 39 / 39 | 6,649 / 6,649 |

Both leaf rows are identical to R-IE2 on nodes, expansions, TT hits, maximum
TT entries, peak TT bytes, verified-hard count, status count, and zero peak
snapshot bytes. There is no leaf regression.

## Integrity, gates, and files

- All Cargo invocations used `CARGO_TARGET_DIR=.target-ie`, target
  `x86_64-pc-windows-msvc`, and `--test-threads=1` for tests.
- Every launch observed zero host-wide Cargo processes. The mandatory loop
  would wait 300 seconds if one appeared.
- Gate-class launches had at least 21.97 GiB availability and 13.74 GiB free
  physical. Ordinary successful launches had at least 21.93 / 13.69 GiB.
- The non-test MSVC release build passed.
- `packages/hexfield_eq/rust/src/tss_verify.rs` has no diff.
- No git commit was created.

Changed implementation/report files:

- `Cargo.lock` - adds `ahash` to the `hexfield_eq` dependency list.
- `packages/hexfield_eq/Cargo.toml` - selects the workspace dependency.
- `packages/hexfield_eq/rust/src/tss_solver.rs` - three mode-(a) rewrites and
  the `cfg(test)` canonical equality oracle/test.
- `HUNT_REPORT_PAIRCLASS.md` - this report and regeneration commands.

Pre-existing untracked `.codex-ie/prompt-pairclass.txt` and `.target-ie/` were
preserved.

## Retained raw logs

Binding evidence:

- `PAIRCLASS_OPT_FULL_RAW.log` - optimized official consuming profile.
- `PAIRCLASS_COMPONENT_FULL_RAW.log` - official post-change component profile.
- `PAIRCLASS_LEAF_RAW.log` - h8/h16 leaf A/B and strict verification.
- `PAIRCLASS_ROW_IDENTITY_RAW.log` - encoding-aware fast/full/component/leaf
  row audits.
- `PAIRCLASS_METRICS_RAW.log` - extracted walls, final buckets, counters, and
  identity summary.
- `PAIRCLASS_PRODUCTION_BUILD_RAW.log` - non-test release build.
- `PAIRCLASS_RAM_GATES.log` - launch readings.

Development and fast evidence:

- `PAIRCLASS_BASELINE_COUNTER_SMOKE_RAW.log` /
  `PAIRCLASS_OPT_COUNTER_SMOKE_RAW.log` - same-cohort component A/B.
- `PAIRCLASS_BASELINE_FAST_RAW.log` / `PAIRCLASS_OPT_FAST_RAW.log` -
  counter-free fast identity A/B.
- `PAIRCLASS_UNIT_RAW.log` - canonical oracle equality.
- `PAIRCLASS_DEFENDER_UNIT_RAW.log` - exact key and incremental plan equality.

## Exact regeneration commands

Run from the worktree root. All functions remove old `TSS_*` values before
setting the complete intended environment.

```powershell
function Wait-PairclassCargoGate([double]$RequiredAvailabilityGiB,
                                 [double]$RequiredFreeGiB,
                                 [string]$Purpose) {
    while (@(Get-Process cargo -ErrorAction SilentlyContinue).Count -gt 0) {
        Get-Process cargo | Select-Object Id, StartTime, CPU
        Start-Sleep -Seconds 300
    }
    $pcOs = Get-CimInstance Win32_OperatingSystem
    $pcFreeBytes = [double]$pcOs.FreePhysicalMemory * 1KB
    $pcStandbyBytes = [double](Get-Counter `
        '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue
    $pcAvailableBytes = $pcFreeBytes + $pcStandbyBytes
    [pscustomobject]@{
        Purpose = $Purpose
        Timestamp = Get-Date -Format o
        FreeGiB = $pcFreeBytes / 1GB
        StandbyGiB = $pcStandbyBytes / 1GB
        AvailabilityGiB = $pcAvailableBytes / 1GB
        RequiredAvailabilityGiB = $RequiredAvailabilityGiB
        RequiredFreeGiB = $RequiredFreeGiB
        CargoProcesses = 0
    } | Format-List | Out-File PAIRCLASS_RAM_GATES.log -Append
    if ($pcAvailableBytes -lt $RequiredAvailabilityGiB * 1GB -or
        $pcFreeBytes -lt $RequiredFreeGiB * 1GB) {
        throw 'R-PC1 RAM gate failed'
    }
}

function Set-PairclassOfficialEnv([string]$Mode = '1',
                                  [bool]$IncrCounters = $false,
                                  [bool]$ClosureCounters = $false) {
    Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
        ForEach-Object { Remove-Item "Env:$($_.Name)" }
    $env:CARGO_TARGET_DIR = '.target-ie'
    $env:TSS_BACKWALK_TT_BYTES = '1073741824'
    $env:TSS_LAZY_FRONTIER = '1'
    $env:TSS_INTERIOR_CENSUS_GATE = '1'
    if ($Mode -ne 'off') { $env:TSS_INCR_DEFENDER = $Mode }
    if ($IncrCounters) { $env:TSS_INCR_ENUM_COUNTERS = '1' }
    if ($ClosureCounters) { $env:TSS_CLOSURE_COUNTERS = '1' }
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
    $env:TSS_CORPUS_EXPECT_CLOSURE_COUNTERS = if ($ClosureCounters) { '1' } else { '0' }
    $env:TSS_CORPUS_EXPECT_THRESHOLD_COUNTERS = '0'
    $env:TSS_CORPUS_EXPECT_THRESHOLD_DELTA = 'off'
    $env:TSS_CORPUS_EXPECT_INCR_ENUM_COUNTERS = if ($IncrCounters) { '1' } else { '0' }
    $env:TSS_CORPUS_EXPECT_INCR_DEFENDER = $Mode
}
```

Focused equality units:

```powershell
Wait-PairclassCargoGate 10 5 'canonical_oracle_unit'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-ie'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq canonical_frame_contender_pruning -- `
    --test-threads=1 --nocapture *> PAIRCLASS_UNIT_RAW.log

Wait-PairclassCargoGate 10 5 'defender_key_unit'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-ie'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq incremental_defender_counter -- `
    --test-threads=1 --nocapture *> PAIRCLASS_DEFENDER_UNIT_RAW.log
```

Official consuming run:

```powershell
Wait-PairclassCargoGate 12 6 'optimized_official_1gib_consume'
Set-PairclassOfficialEnv '1' $false $false
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture *> PAIRCLASS_OPT_FULL_RAW.log
```

Counter-free fast subset:

```powershell
Wait-PairclassCargoGate 10 5 'optimized_fast_subset'
Set-PairclassOfficialEnv '1' $false $false
$env:TSS_CORPUS_ID = 'mvp2lvc,xsnfyll'
$env:TSS_CORPUS_MAX_CAP = '100000'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture *> PAIRCLASS_OPT_FAST_RAW.log
```

Official component profile (batch planner, counters on):

```powershell
Wait-PairclassCargoGate 12 6 'optimized_component_profile_1gib'
Set-PairclassOfficialEnv 'off' $true $false
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture *> PAIRCLASS_COMPONENT_FULL_RAW.log
```

Leaf cells:

```powershell
Wait-PairclassCargoGate 10 5 'optimized_leaf_h8_h16'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-ie'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq incr_defender_leaf_ab -- `
    --ignored --test-threads=1 --nocapture *> PAIRCLASS_LEAF_RAW.log
```

Non-test production build:

```powershell
Wait-PairclassCargoGate 10 5 'production_release_build'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR = '.target-ie'
cargo build --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq *> PAIRCLASS_PRODUCTION_BUILD_RAW.log
```

The row audit decodes PowerShell UTF-16 raws when needed, extracts `CORPUS`
records, and compares the field list documented in the identity sections
against `INCR_ENUM_MEM_AB_ON_FULL_RAW.log` and
`INCR_ENUM_MEM_LEAF_AB_RAW.log`.
