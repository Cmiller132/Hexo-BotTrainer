# R-KT1 forced-reply kernel taxonomy hunt report

Date: 2026-07-17  
Branch/base: `hunt/kernel-taxonomy` / `369e6969`  
Scope: hunt + shadow, no Lean, no consumption, `94gnnol` excluded by owner ruling

## Verdict

**Measured null: no new class qualifies as a certified AND-node width cut or
for the Lean proof queue.** The in-scope 18-position 1 GiB lazy+gate profile
passed with 2,910,351 classified reachable forced AND contexts, zero
load-bearing counterexamples, and zero traversal errors. However, every fire
was a defender-`FirstStone` node that the existing engine had already
compressed into unordered atomic `DefenderPair` children. The observed
`F2_COVERk_*` sets are P3 first-placement projections; they do not remove any
atomic pair obligation and therefore have no current AND-child saving.

No defender-`SecondStone` AND context fired. In particular, the only honest
new complete-reply proposal, `S1_DEAD_SPOKE_C4` (exact P2, current width
`2 -> 1`), has zero corpus support at the measured seam. The already-proven
urgent Q8 `K_reply` class remains valid and default-off, but its production
hook is a claimant `Choice` fallback outside the official wide PN path. This
round found no generalization eligible for promotion.

## Binding-source and seam audit

The design grid and rationales are in `TAXONOMY_KERNEL_REPLY.md`. Two source
facts control interpretation of the measurements:

1. The requested normative `docs/PROOF_TSS_DEFENDER_ZONES.md` contains
   D1--D13, T3--T8, and P1--P3, but not the D19--D21/L15 text cited by the
   agenda and upgrade plan. D19 checkpoint identity, `Q_N^D`, and adaptive
   escape phase are certificate metadata not present in `HexoState`; their
   taxonomy rows are `NO-CONJECTURE` rather than invented position classes.
2. The official wide engine consumes exact T6 before building forced AND
   children. At budget one its reply-cell width is at most two. At budget two
   its exact cell union is at most four and its complete unordered-pair width
   is at most six. The live seam therefore cannot exhibit a new
   `<=37 -> <=k` post-T6 cut.

The profile's full legal widths were 333--956, but those are not the solver's
enumerated forced-reply widths. The actual observed reply-cell widths were
2--4 and atomic child widths were 1--4. Reporting all three avoids crediting a
new conjecture for work T6 already removed.

## Shadow implementation

The implementation is compiled only under `cfg(test)` and is inert unless
`TSS_KERNEL_TAXONOMY_SHADOW=1` exactly:

- after each wide solve, it traverses the final reachable PN DAG with exact
  make/unmake and a compact entry-ID visited bitmap;
- every `Universal` is classified by phase, urgency composition
  (`C4`/`C5`/`MIXED`/`EMPTY`), and child representation;
- `SecondStone` checks the exact P2 dead-spoke predicate before proposing a
  singleton representative;
- `FirstStone` reconstructs the complete-pair incidence graph and finds a
  size-at-most-four vertex cover by a fixed-parameter `2^4` search;
- a child claimant proof means the defense fails; a genuine restricted-TSS
  exhaustion, with `DepthCutoff` excluded, means the defense refutes the
  claimant search; everything else is `Unknown`;
- a counterexample is emitted for **each** out-of-kernel reply that refutes
  while every kernel reply is proved to fail, retaining the full root binding,
  reply statuses, pair edges, corpus ID, and cap;
- `Unknown` never counts as a counterexample or as safe evidence.

“Refutes” above is solver-relative restricted-TSS exhaustion, not a certified
opponent win. Positive claimant children remain hard proofs. This is the
strongest classification available from the wide PN graph without adding a
new dual certificate path; the report keeps resolved-safe and inconclusive
counts separate.

The strict verifier was untouched. There is no width restriction, candidate
filter, search-order change, or consumption branch behind the flag. With the
flag off, the only test-build operation is one environment-variable check
after the search completes; non-test production does not compile the module
or hook.

## Full 18-position profile

Configuration: 1 GiB TT, lazy frontier on, interior census gate on, shared
fragments off, Q8 consume off, cap resume off, live-ge3 seed off, closure
counters off, MSVC target, one test thread. `TSS_CORPUS_ID` explicitly named
all corpus IDs except `94gnnol`. The launch gate was 15.298 GiB free physical
+ 3.111 GiB standby = 18.409 GiB availability, above the required 6/12 GiB.

The test passed all 14 WIN and four in-scope NO positions:

| Profile metric | Result |
|---|---:|
| Ladder rows | 31 |
| Test wall | 371.53 s |
| Summed solve wall | 371.189 s |
| Nodes / expansions | 3,397,362 / 3,397,331 |
| Peak TT bytes | 549,161,606 |
| Pair generation | 157.181 s |
| Defender enumeration | 130.336 s |
| Prior/regen | 30.304 s |
| Expansion inclusive | 291.306 s |
| Stage refresh | 9.564 s |
| Direct insertion | 2.255 s |
| Shadow audits | 29 (two immediate roots did not enter wide PN) |
| Shadow audit time, included in solve wall | 20.262 s |
| Classified AND contexts | 2,910,351 |
| Resolved safe / inconclusive | 195,098 (6.704%) / 2,715,253 (93.296%) |
| Counterexamples / traversal errors | **0 / 0** |

The retained fresh all-19 cap-resume-off baseline in
`HUNT_REPORT_CAP_RESUME.md` is not a direct wall comparison because this run
omits the owner-excluded `94gnnol` row and adds shadow audit work. Its search
and generation counters remain the official reference provenance; no
performance claim is made from the cross-scope wall delta.

## Per-class shadow results

`legal` is the full engine legal count. `reply` is the distinct T6 first-cell
width. `child` is the actual atomic AND width. `K` is the proposed projected
first-cell set. Child share is the exact share of observed atomic defender
obligations; the structural work proxy `|V|+2|E|` approximates the pair-plan
first-cell and directed-pair loops and is not a timing attribution.

| Class | Fires | Safe | CEX | Inconclusive | legal range | reply -> K | child range | child share | work-proxy share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `F2_COVER2_C4` | 2,357,334 | 109,225 | **0** | 2,248,109 | 333--956 | 4 -> 2 (0.500 retained) | 3--4 | 89.5237% | 88.1434% |
| `F2_COVER1_C4` | 400,135 | 56,419 | **0** | 343,716 | 334--948 | 2--3 -> 1 (0.342 aggregate retained) | 1--2 | 7.8875% | 8.8735% |
| `F2_COVER1_MIXED` | 152,325 | 29,252 | **0** | 123,073 | 351--917 | 2--3 -> 1 (0.377 aggregate retained) | 1--2 | 2.5831% | 2.9758% |
| `F2_COVER1_C5` | 557 | 202 | **0** | 355 | 537--761 | 2 -> 1 (0.500 retained) | 1 | 0.0057% | 0.0073% |
| All `S1_*` classes | **0** | 0 | 0 | 0 | -- | -- | -- | 0% | 0% |
| All `NO_CONJECTURE`/unsupported buckets | **0** | 0 | 0 | 0 | -- | -- | -- | 0% | 0% |

The aggregate projection retains 5,267,685 of 11,002,776 reply vertices
(0.478760), but retains **all 9,740,262 atomic pair children**. Consequently
the apparent 52.1% reply-cell collapse has zero current AND-child value.

The absence of F2 counterexamples is an extraction/status invariant check,
not an independent theorem discovery: because `C` covers every pair edge, a
refuting edge incident to an outside vertex is also incident to a retained
vertex. P3 then gives the identical completed-turn state. Future consumption
would still need either to keep all edges, as today, or prove a different
complete-pair domination lemma.

## Default-off identity and production gate

The non-test release build passed with `.target-kernel`, proving the test-only
module and hook are absent from production.

The official fast subset (`mvp2lvc,xsnfyll`, max cap 100k) was run once with
the shadow flag absent and once with it set. Status, nodes, expansions, TT
entries/hits, and peak bytes were identical on every row:

| ID / cap | Status | nodes / expansions | peak bytes |
|---|---|---:|---:|
| `mvp2lvc` / 10k | UNKNOWN | 10,000 / 9,999 | 1,948,056 |
| `mvp2lvc` / 100k | UNKNOWN | 17,957 / 17,956 | 3,928,520 |
| `xsnfyll` / 10k | WIN | 82 / 81 | 7,156 |

Flag-on added only a post-solve audit: 17,665 contexts, zero counterexamples,
zero traversal errors, 154.502 ms audit time. Flag-off output contained no
`KERNEL_TAXONOMY_*` lines.

## Counterexamples and narrower subclasses

No `KERNEL_TAXONOMY_COUNTEREXAMPLE` specimen was emitted. There is therefore
no refuted class or counterexample-induced narrower subclass this round.

Zero-fire rows are not survivors. In particular, `S1_DEAD_SPOKE_C4` retains
its honest P2 rationale but has no corpus-wide economic evidence because
complete-turn pair compression bypasses intermediate `SecondStone` AND
nodes. D19/D21/touched/virgin adaptive-escape rows remain `NO-CONJECTURE`
until producer and verifier carry their certificate metadata.

## Proof queue

The ordered promotion-eligible Lean queue is **empty**:

1. `F2_COVER2_C4` would rank first by observed mass, but is rejected from the
   queue because it is a first-placement projection and removes zero atomic
   children.
2. `F2_COVER1_C4`, `F2_COVER1_MIXED`, and `F2_COVER1_C5` fail the same
   complete-reply/economic criterion.
3. `S1_DEAD_SPOKE_C4` is a genuine complete-reply `2 -> 1` proposal with a
   cheap incremental trigger, but zero fires fails the material-width gate.

The only certified kernel story left standing is the existing T6/Q8/P2/P3
substrate. The Q8 consume flag should remain off for the already-recorded
G2R8 economics; this round supplies no reason to reopen it.

## Files changed

- `TAXONOMY_KERNEL_REPLY.md` -- class grid, rationales, O(1) trigger designs,
  source/seam ruling, and promotion criteria.
- `HUNT_REPORT_KERNEL_TAXONOMY.md` -- measurements, verdicts, proof queue, raw
  log index, and regeneration commands.
- `packages/hexfield_eq/rust/src/tss_kernel_taxonomy.rs` -- new test-only
  classifier, P2/P3 kernel construction, load-bearing evaluator, specimens,
  and telemetry.
- `packages/hexfield_eq/rust/src/lib.rs` -- test-only module registration.
- `packages/hexfield_eq/rust/src/tss_solver.rs` -- flag-gated post-solve wide
  PN DAG audit; no production path.
- `packages/hexfield_eq/rust/src/tss_corpus.rs` -- flag-gated context/reset and
  final report output.

## Retained raw logs

- `KERNEL_TAXONOMY_UNIT_RAW.log` -- launch-capture false start plus passing
  3/3 focused tests.
- `KERNEL_TAXONOMY_PRODUCTION_BUILD_RAW.log` -- passing non-test release
  build.
- `KERNEL_TAXONOMY_FAST_OFF_RAW.log` -- fast default-off identity baseline.
- `KERNEL_TAXONOMY_FAST_ON_RAW.log` -- matching flag-on fast subset and shadow
  summary.
- `KERNEL_TAXONOMY_FULL_PROFILE_RAW.log` -- complete 18-position profile,
  class summaries, complete width histograms, and any specimens (none).
- `KERNEL_TAXONOMY_RAM_GATES.log` -- consolidated free/standby/availability
  readings for every Cargo invocation.

## Exact measurement regeneration

Run from the worktree root. The helper implements the required host-wide
five-minute Cargo wait and both RAM readings; pass `(10,5)` for ordinary Cargo
and `(12,6)` for the gate-class 1 GiB profile.

These commands reproduce the measurements. To create a fresh named raw capture,
append `2>&1 | Tee-Object -FilePath <matching-log-name>` to its Cargo command and
pipe each `Wait-KernelCargoGate` result to
`Tee-Object -Append KERNEL_TAXONOMY_RAM_GATES.log`. The retained unit log also
contains the documented PowerShell capture false start, so it is evidence from
this run rather than a byte-for-byte reproducible artifact.

```powershell
function Wait-KernelCargoGate([double]$RequiredAvailabilityGiB,
                              [double]$RequiredFreeGiB) {
    while (@(Get-Process cargo -ErrorAction SilentlyContinue).Count -gt 0) {
        Get-Process cargo | Select-Object Id, StartTime, CPU
        Start-Sleep -Seconds 300
    }
    $ktOs = Get-CimInstance Win32_OperatingSystem
    $ktFreeBytes = [double]$ktOs.FreePhysicalMemory * 1KB
    $ktStandbyBytes = [double](Get-Counter `
        '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue
    $ktAvailableBytes = $ktFreeBytes + $ktStandbyBytes
    [pscustomobject]@{
        Timestamp = Get-Date -Format o
        FreeGiB = $ktFreeBytes / 1GB
        StandbyGiB = $ktStandbyBytes / 1GB
        AvailabilityGiB = $ktAvailableBytes / 1GB
    }
    if ($ktAvailableBytes -lt $RequiredAvailabilityGiB * 1GB -or
        $ktFreeBytes -lt $RequiredFreeGiB * 1GB) {
        throw 'R-KT1 RAM gate failed'
    }
}

function Set-KernelOfficialEnv {
    Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
        ForEach-Object { Remove-Item "Env:$($_.Name)" }
    $env:CARGO_TARGET_DIR = '.target-kernel'
    $env:TSS_BACKWALK_TT_BYTES = '1073741824'
    $env:TSS_LAZY_FRONTIER = '1'
    $env:TSS_INTERIOR_CENSUS_GATE = '1'
    $env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS = '0'
    $env:TSS_CORPUS_EXPECT_LAZY_FRONTIER = '1'
    $env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE = '1'
    $env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME = '0'
    $env:TSS_CORPUS_EXPECT_CAP_RESUME = '0'
    $env:TSS_CORPUS_EXPECT_LIVE_GE3_SEED = '0'
    $env:TSS_CORPUS_EXPECT_CLOSURE_COUNTERS = '0'
}
```

Focused tests and production build:

```powershell
Wait-KernelCargoGate 10 5
$env:CARGO_TARGET_DIR = '.target-kernel'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq kernel_taxonomy -- --test-threads=1 --nocapture

Wait-KernelCargoGate 10 5
$env:CARGO_TARGET_DIR = '.target-kernel'
cargo build --release --target x86_64-pc-windows-msvc -p hexfield_eq
```

Fast flag-off/flag-on identity pair:

```powershell
Wait-KernelCargoGate 10 5
Set-KernelOfficialEnv
$env:TSS_CORPUS_ID = 'mvp2lvc,xsnfyll'
$env:TSS_CORPUS_MAX_CAP = '100000'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture

Wait-KernelCargoGate 10 5
Set-KernelOfficialEnv
$env:TSS_CORPUS_ID = 'mvp2lvc,xsnfyll'
$env:TSS_CORPUS_MAX_CAP = '100000'
$env:TSS_KERNEL_TAXONOMY_SHADOW = '1'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture
```

Owner-scoped full 18-position profile:

```powershell
Wait-KernelCargoGate 12 6
Set-KernelOfficialEnv
$env:TSS_KERNEL_TAXONOMY_SHADOW = '1'
$env:TSS_CORPUS_ID = @(
    '0hz3hty','0l4291i_live','8is963b','acly7kb','dy3dg99','g2xx6wl',
    'hu01jk4','jh7yo7y','jnzzmcm','l9mxn59','lz60mfb','mvp2lvc',
    'xsnfyll','zrugh2x','strongloss_a_prefix6','strongloss_b_prefix8',
    'hayes_20260712_turn16','hayes_20260712_placement31'
) -join ','
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture
```
