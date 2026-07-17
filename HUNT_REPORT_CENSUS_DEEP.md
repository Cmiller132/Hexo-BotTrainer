# R-CF1 — deadness census family, deep-profile hunt

Date: 2026-07-17  
Branch/worktree: `hunt/census-deep` / `hunt-census-deep`  
Starting HEAD: `d5a2b5fd`  
Scope: measurement and conjecture design only; **no consumption wiring and no
strict-verifier change**.

## Executive verdict

The landed R-IG1 gate is inert for one quantitative reason: the official
profile sets `semantic_horizon=u32::MAX`, while the proven single-window DTW
family has a maximum finite lower bound of 12. At every one of the 484,270
claimant-owned interior evaluation points, remaining semantic horizon was at
least 4,294,967,130; all 484,270 therefore missed by the `257_plus` bucket.
The census shape was not the problem: 112,278 points (23.1850%) already had
`c<=2`.

This is a structural dry verdict for every **raw scalar, finite DTW-bound**
extension at the unbounded contract. Raising `c`, adding another finite phase
row, or polishing the current comparison cannot make that family fire. It is
not a dry verdict for the whole census idea: finite stage deadlines, exact
fixed-family certificates, defender restore, and recursive strategy
certificates have different theorem shapes. Those are the candidates carried
into the shadow round.

The official shadow then produced a useful but narrower win:

- four new finite-deadline semantic families fired at depth with zero
  deadline counterexamples: defender restore (224,761), deadline ES (168,400),
  ES plus an exact ordered pre-block witness (256,386), and ES plus a
  disjoint-triple witness (171,408);
- the exact forcing-grammar census attractor classified 86,331 nodes with zero
  PN=0 counterexamples, but it remains a post-solve restricted-grammar closure,
  not a semantic pruning theorem;
- **no candidate from this round proves permanent no-WIN at the unbounded
  official contract.** The semantic survivors are deadline certificates and
  could only support a deadline-aware defer/reopen mechanism after proof. No
  production wiring was added.

The ordered Lean queue is: (1) deadline-family completeness plus base ES,
(2) ordered ES pre-block, (3) exact defender restore, (4) the disjoint-triple
invariant. This dependency order differs from raw fire-rate order, where
pre-block > restore > triple > base ES.

## Binding context and proof boundary

The implementation and candidate ladder follow these non-negotiable facts.

- R-IG1 is a WIN-goal, `h<=8` theorem. It scans all
  `WindowStore::entries()`, uses exact opponent-free aliveness and the zero
  fallback, and retains its coordinate guard and checked arithmetic.
- The reachable SecondStone `c=3` forced WIN is a hard refutation of the naive
  strengthening. No candidate infers deadness from max-c alone.
- ES fixed-family blocking is proved; unrestricted global `Phi<1` forever is
  not. Window births and remote quiet turns invalidate that shortcut.
- T3/T4 zones restrict defender replies, not claimant movement. Small zones
  are not attacker confinement.
- A strict certificate `Choice` accepts arbitrary legal moves. A statement
  about WideTurnGate is only a forcing-grammar theorem unless separately
  lifted to semantic game deadness.
- Every shadow predicate is `cfg(test)`, exact-flag default-off. Production
  code has no census-deep module or fields. No result is consumed.

The theorem-precise queue, including rejected variants, is in
`CENSUS_CANDIDATES.md`.

## Phase A — why R-IG1 is inert

### Instrumentation

With `TSS_CENSUS_DEEP_COUNTERS=1` exactly, both wide and narrow post-tactical
gate seams record:

- the ordered first failing precondition;
- orthogonal owner/interior/phase predicates;
- exact semantic-horizon and `LB-h` near-miss buckets;
- full-store claimant/opponent census histograms by backend, phase, and depth;
- exact raw ES integer screens;
- captured stage remainder as a separate scheduling diagnostic;
- scan count and wall.

The scan occurs counterfactually at every supported interior seam, including
defender-owned points, so owner failure does not hide the loss-side/exposure
surface. Counters do not affect search decisions.

### Official profile

The retained Phase-A run is the 18 in-scope corpus positions (all official
positions except owner-held `94gnnol`), fresh solver per rung,
`TSS_BACKWALK_TT_BYTES=1073741824`, lazy frontier on, landed interior gate on,
cap-resume off, MSVC release target, and one test thread. It passed all 31
ladder rows with zero corpus failures in 362.15 s.

| quantity | observed |
|---|---:|
| ladder rows | 31 |
| summed nodes / expansions | 3,397,362 / 3,397,331 |
| post-tactical gate seams | 3,394,650 |
| supported interior full-store scans | 3,394,621 |
| claimant-owned interior | 484,270 |
| defender-owned interior | 2,910,351 |
| root points | 29 |
| full-scan wall / mean | 8,569.343 ms / 2.524 us |
| invariant failures / h=8 coordinate failures | 0 / 0 |

The node/expansion totals exactly match the prior fresh, owner-scoped deep
trajectory; the counters changed observation wall, not search trajectory.

### Ordered failure census

| first failure | count | share of all seams |
|---|---:|---:|
| defender-owned gate shape | 2,910,351 | 85.73% |
| root (not interior) | 29 | <0.001% |
| semantic horizon greater than 8 | 484,270 | 14.27% |
| eligible | 0 | 0% |

Every point had a supported phase. Every semantic horizon landed in
`257_plus`; exact remaining horizons ranged from 4,294,967,130 through
4,294,967,286. Consequently `current_evaluations=0` and
`current_dismissals=0`.

### Census and near-miss distribution

Among the 484,270 claimant-owned interior points:

| exact claimant census | count | share |
|---|---:|---:|
| `c=1` | 42 | 0.008673% |
| `c=2` | 112,236 | 23.176327% |
| `c=3` | 371,992 | 76.815000% |
| `c=0,4,5` | 0 | 0% |

Thus the current `c<=2` shape held at 112,278 points (23.185000%). All
484,270 exact DTW comparisons were nevertheless `miss_257_plus`. The maximum
table value (SecondStone `c=0`, 12) is still roughly 4.295 billion placements
short of the official remainder.

Every observed supported node was FirstStone; there were no post-tactical
SecondStone or Opening rows in this wide profile. The depth distribution shows
that the diagnosis is not a shallow-only artifact:

| root-relative depth band | all scanned points | claimant-owned | `c=1` | `c=2` | `c=3` | bounded-stage DTW screen |
|---|---:|---:|---:|---:|---:|---:|
| 0–16 | 101,493 | 44,094 | 42 | 21,040 | 23,012 | 22,438 |
| 17–32 | 281,176 | 67,391 | 0 | 23,469 | 43,922 | 13,296 |
| 33–48 | 964,895 | 182,222 | 0 | 39,487 | 142,735 | 1,904 |
| 49+ | 2,047,057 | 190,563 | 0 | 28,240 | 162,323 | 5,276 |

At every band the unbounded semantic-horizon predicate fails before the
census comparison is allowed to decide. The bounded-stage column is a
separate counterfactual deadline screen, not a landed-gate firing.

Raw global ES `Phi<1` fired zero times for claimant and opponent profiles at
every recorded depth. That is consistent with the proof document’s narrow
domain and is not evidence for a global forever gate.

As a **bounded-stage scheduling** diagnostic, the existing DTW table was
strictly beyond the captured stage remainder at 42,914 claimant nodes
(8.861585%). This is not a permanent refutation: it can only justify a
deadline `DepthCutoff`, and the solver may already materialize a positive
typed leaf whose exact resolution is later than that deadline. Phase C
therefore compares materialized absolute proof resolution with the captured
deadline rather than calling every later PN=0 a counterexample.

### Diagnosis

The gate is not starved of low censuses. It is starved of a finite semantic
horizon. A scalar per-window completion lower bound is necessarily finite at
every finite `c`, so no member of that theorem family can dismiss an
`u32::MAX`-horizon subtree. The viable theorem shapes are instead:

1. finite deadline certificates that can replace work by `DepthCutoff` and
   reopen later;
2. a genuinely invariant finite strategy certificate (fixed-family cover,
   exposure-closed finite family, or a coinductive invariant);
3. search-grammar attractors, stated only at their actual restricted contract.

## Phase B — candidates designed

The top semantic shadows are:

- `STAGE_DTW`: the landed theorem under a captured finite stage deadline;
- `DEFENDER_RESTORE4`: exact `tau=b` service kills every current c>=4
  claimant window and shifts the c<=3 DTW bound to 8/7 placements;
- `DEADLINE_ES`: discard only windows which provably cannot finish by the
  deadline, then apply exact fixed-family ES thresholds;
- `DEADLINE_ES_PREBLOCK`: exhibit and replay the defender’s remaining ordered
  pre-block placements, then require residual `Psi<1/3`;
- `DEADLINE_ES_TRIPLE`: cover every deadline-relevant labelled residual edge
  by pairwise disjoint triples and use a two-response invariant.

The exact finite residual-gap minimax (`FF_GAP`) and D16 exposure closure are
specified but lower in the queue. The former is a capped exact oracle for the
c=3 overlap cases; the latter is the route beyond five claimant placements,
but depends on not-yet-serialized exposure labels and stated proof rows.

The forcing-grammar auxiliary shadows are exact pair-service at c<=1/2/3 and
its finite ranked AND/OR attractor. They are kept because they can expose real
deep subtree mass and have a clean induction proof, but they are explicitly
not semantic no-WIN claims.

## Phase C — adversarial shadow

The implemented audit follows each completed solve’s reachable resolved PN
DAG with exact `apply_with_delta`/`undo`, a visited entry bitmap, and no outcome
input to candidate formation. It retains:

- unique predicate fires;
- ancestor-dominated would-prune roots;
- the union of recorded expansion events under those roots (an honest upper
  bound, not a claimed realized saving, because transposed descendants may
  also be reachable elsewhere and bounded cuts reopen);
- exact materialized proof resolution for PN=0 bounded candidates;
- every true counterexample with corpus id, cap, entry, depth, phase, profile,
  and the complete sorted stone position;
- evaluation and total audit wall plus traversal errors.

For bounded candidates, `resolution<=deadline` is a counterexample and a
later proof is reported separately as `late_wins`. For forcing-grammar
candidates, every PN=0 fire is a counterexample.

The final audit also closes three weaknesses found in hostile review before
the official run:

- roots are excluded;
- `DEFENDER_RESTORE4` enumerates and replays exact ordered legal one- or
  two-placement sequences and full-scans the successor to verify `c_A<=3`;
- PN=0 deadline validation uses a memoized exact resolution DP
  (`Choice=min`, `Universal=max`), rather than the first materialized proof.
  Every retained counterexample would additionally include the legal
  root-relative replay path and canonical sorted occupancy. No specimen was
  emitted because no counterexample occurred.

The captured search depth boundary is used only to choose a hypothetical
absolute placement deadline. It is **not** claimed to be the solver's current
stage semantics: atomic completions and typed tactical leaves can resolve
beyond an expansion depth cap. Such cases are the separately reported
`late_wins` and require a new deadline-aware scheduling contract before any
consumer exists.

### Official result

The same 18-position, 31-row 1 GiB profile passed in 418.49 s. Search
trajectory was exactly identical to Phase A: 3,397,362 nodes and 3,397,331
expansions, with an exact zero-row diff over status, expected status, nodes,
expansions, TT entries/hits, peak TT bytes, stage refreshes, and landed-gate
counts. The audit visited 29 completed wide arenas, reported zero traversal
errors, zero counterexamples, and zero unresolved PN=0 deadline relations.

Semantic deadline candidates:

| candidate | fires | applicable fire rate | dominated roots | expansion-mass upper bound | refuted / unknown / late WIN | counterexamples | eval wall | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `STAGE_DTW` control | 42,914 | 8.861585% of claimant nodes | 1,316 | 3,357,140 | 9,712 / 30,432 / 2,770 | 0 | 7,773.717 ms | survives deadline shadow |
| `DEFENDER_RESTORE4` | 224,761 | 7.722814% of defender nodes | 411 | 3,363,404 | 15,242 / 208,311 / 1,208 | 0 | 5,445.713 ms | **survivor** |
| `DEADLINE_ES` | 168,400 | 4.960789% of all interior nodes | 592 | 3,363,045 | 15,279 / 150,105 / 3,016 | 0 | 1,359.464 ms | **survivor** |
| `DEADLINE_ES_PREBLOCK` | 256,386 | 7.552714% of all interior nodes | 411 | 3,363,404 | 22,270 / 230,370 / 3,746 | 0 | 5,786.590 ms | **survivor** |
| `DEADLINE_ES_TRIPLE` | 171,408 | 5.049400% of all interior nodes | 666 | 3,363,119 | 17,487 / 150,901 / 3,020 | 0 | 2,981.007 ms | survivor; 82 capped discoveries |

The expansion-mass column is intentionally labelled an upper bound. These
predicates dominate early shallow deadlines, so their final-DAG descendant
union includes work that a bounded cut would later reopen. It is evidence of
where the predicates sit, not a speedup forecast. Honest realized value needs
a proved deadline-aware replay experiment.

`DEADLINE_ES` is the best cost/proof anchor: its aggregate measured evaluation
was 1.359 s (about 0.400 us per interior evaluation). Pre-block adds 87,986
fires over base ES for 4.427 s additional aggregate evaluation. Triples add
only 3,008 fires over base, cost an additional 1.622 s, and had 82 capped
searches, so they remain behind pre-block despite surviving.

Forcing-grammar auxiliary candidates:

| candidate | fires | would-prune roots | expansion-mass upper bound | PN=0 counterexamples | eval wall | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `PAIR_SERVICE_C1` | 42 | 42 | 42 | 0 | 14,506.756 ms shared | exact but negligible |
| `PAIR_SERVICE_C2` | 12,165 | 12,165 | 12,190 | 0 | 14,506.756 ms shared | almost all seed-only |
| `PAIR_SERVICE_C3` | 20,304 | 20,304 | 20,347 | 0 | 14,506.756 ms shared | almost all seed-only |
| `DEFENDER_REPLY_LIFT` | 36,336 | 36,217 | 57,806 | 0 | 81.726 ms postorder bookkeeping | restricted grammar |
| `TWO_CYCLE_LIFT` | 9,332 | 9,332 | 34,626 | 0 | 81.726 ms postorder bookkeeping | restricted grammar |
| `CENSUS_ATTRACTOR` | 86,331 | 33,990 | 138,303 | 0 | 81.726 ms postorder bookkeeping | post-solve closure |

Pair service is too expensive for its observed prize: the exact quadratic
classifier consumed 14.507 s and its c<=3 fires contained only 43 expansions
beyond one per seed. The attractor covers 4.0709% of official expansions, but
that number is circular as an online-prune estimate because the predicate is
computed from already solved descendants. Its proper research role is a
ranked proof/cache closure; a prospective shallow evaluator is required before
claiming runtime savings.

## Safety, identity, and overhead

- Both environment switches require exact string `1`:
  `TSS_CENSUS_DEEP_COUNTERS=1` and `TSS_CENSUS_DEEP_SHADOW=1`.
- The entire module and all solver fields/calls are `cfg(test)`.
- Counters and shadow predicates never write PN/DN, branch lists, caches,
  certificates, or verifier state.
- No production consumption flag exists.

The complete shadow audit cost 55,381.870 ms. Phase-C summed per-row solve
wall was 418,137.0 ms versus 361,806.5 ms in Phase A, a 56,330.5 ms / 15.569%
instrumentation delta. Candidate times overlap because scans and postorder
facts are shared; they must not be summed.

The final fast default-off profile (`mvp2lvc,xsnfyll`, cap 10k) passed with
exactly the same two trajectory rows as the final shadow smoke and emitted no
`CENSUS_DEEP_*` report. `CENSUS_DEEP_IDENTITY_RAW.log` also records a zero-row
diff between all 31 Phase-A and Phase-C trajectories. A non-test MSVC release
build passed, proving the `cfg(test)` module and fields are absent from the
production configuration.

The host PowerShell capture layer renders Cargo's normal stderr progress as a
`NativeCommandError`, so tool wrappers display exit code 1. The authoritative
Cargo lines retained in every log are passing (`test result: ok` or
`Finished release profile`); no Cargo test or build failed.

## Files and retained evidence

Source/document deliverables:

- `packages/hexfield_eq/rust/src/tss_census_deep.rs`
- `packages/hexfield_eq/rust/src/tss_solver.rs`
- `packages/hexfield_eq/rust/src/tss_corpus.rs`
- `packages/hexfield_eq/rust/src/lib.rs`
- `HUNT_REPORT_CENSUS_DEEP.md`
- `CENSUS_CANDIDATES.md`

Retained logs:

- `CENSUS_DEEP_UNIT_RAW.log` and `CENSUS_DEEP_UNIT_RAW_2.log`
  (superseded development unit runs)
- `CENSUS_DEEP_UNIT_RAW_3.log` (final unit run)
- `CENSUS_DEEP_PHASE_A_SMOKE_RAW.log`
- `CENSUS_DEEP_PHASE_A_RAW.log`
- `CENSUS_DEEP_PHASE_C_SMOKE_RAW.log` (superseded development smoke)
- `CENSUS_DEEP_PHASE_C_SMOKE_RAW_2.log` (final corrected smoke)
- `CENSUS_DEEP_PHASE_C_RAW.log` (official shadow)
- `CENSUS_DEEP_DEFAULT_OFF_RAW.log`
- `CENSUS_DEEP_IDENTITY_RAW.log`
- `CENSUS_DEEP_PRODUCTION_BUILD_RAW.log`
- `CENSUS_DEEP_RAM_GATES.log`
- `CENSUS_DEEP_SHA256.txt` (SHA-256 manifest for every retained log)

## Exact regeneration commands

Run from the worktree root. Before **every** Cargo command, wait in five-minute
increments while any host-wide Cargo process exists. Ordinary Cargo requires
availability >=10 GiB and free physical >=5 GiB; a 1 GiB gate profile requires
availability >=12 GiB and free physical >=6 GiB. The commands below recompute
availability as free physical plus the three standby-cache counters and print
the reading into the retained log.

```powershell
$ErrorActionPreference = 'Stop'

function Wait-CargoAndRam([switch]$Gate1GiB) {
    while (Get-Process cargo -ErrorAction SilentlyContinue) {
        Write-Host 'another cargo lane is active; waiting 300 seconds'
        Start-Sleep -Seconds 300
    }
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGiB = ($os.FreePhysicalMemory * 1KB) / 1GB
    $standby = (Get-Counter `
        '\Memory\Standby Cache Reserve Bytes', `
        '\Memory\Standby Cache Normal Priority Bytes', `
        '\Memory\Standby Cache Core Bytes').CounterSamples |
        Measure-Object CookedValue -Sum
    $availabilityGiB = $freeGiB + ($standby.Sum / 1GB)
    $class = if ($Gate1GiB) { 'gate_1g' } else { 'ordinary' }
    $reading = "RAM_GATE timestamp=$(Get-Date -Format o) class=$class cargo_count=0 " +
        "free_physical_gib=$([math]::Round($freeGiB,3)) " +
        "availability_gib=$([math]::Round($availabilityGiB,3))"
    $reading | Tee-Object -FilePath CENSUS_DEEP_RAM_GATES.log -Append
    $minAvailability = if ($Gate1GiB) { 12 } else { 10 }
    $minFree = if ($Gate1GiB) { 6 } else { 5 }
    if ($availabilityGiB -lt $minAvailability -or $freeGiB -lt $minFree) {
        throw "RAM gate failed"
    }
}

function Clear-TssEnvironment {
    Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
        ForEach-Object { Remove-Item "Env:$($_.Name)" }
}

$env:CARGO_TARGET_DIR = '.target-cf'
$inScopeIds = '0hz3hty,0l4291i_live,8is963b,acly7kb,dy3dg99,' +
    'g2xx6wl,hu01jk4,jh7yo7y,jnzzmcm,l9mxn59,lz60mfb,mvp2lvc,' +
    'xsnfyll,zrugh2x,strongloss_a_prefix6,strongloss_b_prefix8,' +
    'hayes_20260712_turn16,hayes_20260712_placement31'
```

Focused unit tests:

```powershell
Clear-TssEnvironment
Wait-CargoAndRam
$env:CARGO_TARGET_DIR = '.target-cf'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq census_deep -- --test-threads=1 --nocapture `
    2>&1 | Tee-Object CENSUS_DEEP_UNIT_RAW_3.log
```

Common official 1 GiB profile setup:

```powershell
Clear-TssEnvironment
$env:CARGO_TARGET_DIR = '.target-cf'
$env:TSS_BACKWALK_TT_BYTES = '1073741824'
$env:TSS_LAZY_FRONTIER = '1'
$env:TSS_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS = '0'
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER = '1'
$env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME = '0'
$env:TSS_CORPUS_EXPECT_CAP_RESUME = '0'
$env:TSS_CORPUS_ID = $inScopeIds
```

Phase A counters-only profile:

```powershell
$env:TSS_CENSUS_DEEP_COUNTERS = '1'
Remove-Item Env:TSS_CENSUS_DEEP_SHADOW -ErrorAction SilentlyContinue
$env:TSS_CORPUS_EXPECT_CENSUS_DEEP_COUNTERS = '1'
$env:TSS_CORPUS_EXPECT_CENSUS_DEEP_SHADOW = '0'
Wait-CargoAndRam -Gate1GiB
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture `
    2>&1 | Tee-Object CENSUS_DEEP_PHASE_A_RAW.log
```

Phase C official shadow (rerun the common setup first if starting a new
PowerShell session):

```powershell
$env:TSS_CENSUS_DEEP_COUNTERS = '1'
$env:TSS_CENSUS_DEEP_SHADOW = '1'
$env:TSS_CORPUS_EXPECT_CENSUS_DEEP_COUNTERS = '1'
$env:TSS_CORPUS_EXPECT_CENSUS_DEEP_SHADOW = '1'
Wait-CargoAndRam -Gate1GiB
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture `
    2>&1 | Tee-Object CENSUS_DEEP_PHASE_C_RAW.log
```

Fast default-off identity:

```powershell
Clear-TssEnvironment
$env:CARGO_TARGET_DIR = '.target-cf'
$env:TSS_BACKWALK_TT_BYTES = '1073741824'
$env:TSS_LAZY_FRONTIER = '1'
$env:TSS_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS = '0'
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER = '1'
$env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME = '0'
$env:TSS_CORPUS_EXPECT_CAP_RESUME = '0'
$env:TSS_CORPUS_EXPECT_CENSUS_DEEP_COUNTERS = '0'
$env:TSS_CORPUS_EXPECT_CENSUS_DEEP_SHADOW = '0'
$env:TSS_CORPUS_ID = 'mvp2lvc,xsnfyll'
$env:TSS_CORPUS_MAX_CAP = '10000'
Wait-CargoAndRam -Gate1GiB
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture `
    2>&1 | Tee-Object CENSUS_DEEP_DEFAULT_OFF_RAW.log
```

Production configuration build:

```powershell
Clear-TssEnvironment
Wait-CargoAndRam
$env:CARGO_TARGET_DIR = '.target-cf'
cargo build --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq 2>&1 | Tee-Object CENSUS_DEEP_PRODUCTION_BUILD_RAW.log
```
