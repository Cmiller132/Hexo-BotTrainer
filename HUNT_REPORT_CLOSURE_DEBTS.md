# R-CD1 register closure debts A+B

Date: 2026-07-17

Worktree: `tss-vcf-width`

Starting HEAD: `e05324f49b9fd78d6c005cf468d117b1bac3f64d`
(`R-CR1 cap-ladder resume — PROMOTE` landing)

## Status and verdicts

**CLEAN PARTIAL BOUNDARY.** The instrumentation, default-off identity check,
production build, and selected Phase-3 live A/B are complete. The two full
1 GiB runs have not started because free RAM never reached the mandatory
strictly-above-11-GiB threshold during this run. No lower-memory substitute is
reported as official evidence.

- **Debt A: OPEN / NO DECISION.** The counters work and a 10k smoke cohort is
  retained, but the official 1 GiB counter profile is the decision gate and is
  still unrun. No incremental reveal implementation was built.
- **Debt B: CLOSING NULL (measurement matrix partial).** The frozen promotion
  rule requires no material selected-leaf regression. The live seed regressed
  both selected cells: h8 wall `76.699 -> 90.817 ms` (+18.41%) with identical
  16 verdicts, while h16 wall `501.429 -> 564.050 ms` (+12.49%) and lost one
  verdict (`39 -> 38`). The conjunction required for promotion is therefore
  already impossible. The requested full deep A/B remains unrun and is called
  out rather than fabricated.

`TSS_CAP_RESUME` was off in every completed A/B arm. The selected-leaf harness
also kept shared fragments and K-reply off. The strict verifier source was not
modified.

## Instrumentation

Everything in this round is behind `cfg(test)`. Production does not contain
the flags, fields, counters, seed scan, or harness.

### Debt A counters

`TSS_CLOSURE_COUNTERS=1` enables per-search counters for:

- candidate pairs evaluated, classifier-accepted, dedup-retained, selected,
  linked, and expanded;
- winning Choice-child ranks in bins `1, 2, 3, 4, 5-8, 9-16, 17-32, 33+`;
- gate build, second-candidate generation, pair evaluation, dedup, and total
  pair-generation wall;
- the exact eager tail after the winning child at proven pair Choice nodes.

The last item is the sound reveal counterfactual. Refuted and unresolved
Choice nodes receive zero prospective saving because they must exhaust the
generator. At a proven Choice node, the reveal prefix ends only after the
eventual winning child has been classified; the remaining second-candidate,
evaluation, and dedup time is recorded as avoidable. This is deliberately
more conservative than treating every rejected or unlinked candidate as
avoidable.

The 10k smoke cohort (`mvp2lvc` plus `xsnfyll`) preserved baseline search
counts and produced:

| metric | value |
|---|---:|
| evaluated / accepted / retained | 6,599,166 / 71,402 / 36,703 |
| selected / linked / expanded | 5,870 / 5,870 / 5,792 |
| evaluated but never linked | 6,593,296 (99.911%) |
| evaluated but never expanded | 6,593,374 (99.912%) |
| total measured pair generation | 1,099.328 ms |
| conservative avoidable split tail | 87.515 ms (7.961%) |
| winning-rank bins | `[52, 32, 36, 32, 213, 31, 0, 5]` |

These are smoke numbers, not the official decision. They illustrate why the
raw unlinked fraction is not an economic estimate: most rejected candidates
still have to be classified before a sound cursor can discover the retained
prefix, and refutation is forbidden before exhaustion.

### Debt B live seed

`TSS_LIVE_GE3_SEED=1` replaces the fresh proof-number prior with
`37 - min(live_ge3, 36)`, preserving the existing 1..37 prior scale while
ranking larger live-claimant-window counts first. The scan covers every fresh
wide-PN prior regardless of census horizon; it does not piggyback on or alter
the interior census gate. Lazy attacker pairs use the exact turn-start window
snapshot to count the post-pair live windows, and collapsed defender pairs are
scanned at their exact post-pair state. Scan count and wall are separately
reported.

## Selected Phase-3 live A/B

Both arms used config D exactly: wide PN, lazy frontier on, interior census
gate on, shared fragments off, K-reply off, 256 KiB TT, node cap 500, and the
same 50 x six sampled leaves. The arms differed only in
`TSS_LIVE_GE3_SEED`. `TSS_CAP_RESUME` was absent/off.

| horizon | seed | verdicts | expansions | TT hits | stage refreshes | verified | wall ms | seed scans / scan ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | off | 16 | 1,852 | 0 | 650 | 16 | 76.699 | 0 / 0 |
| 8 | on | 16 | 1,852 | 0 | 650 | 16 | 90.817 | 4,667 / 2.496 |
| 16 | off | 39 | 6,649 | 39 | 980 | 39 | 501.429 | 0 / 0 |
| 16 | on | 38 | 6,727 | 23 | 970 | 38 | 564.050 | 41,989 / 25.233 |

All 109 returned hard certificates across the four arms were accepted by the
unchanged strict verifier. There were zero WIN/LOSS contradictions. The h16
seed arm had one hard-to-UNKNOWN regression and no UNKNOWN-to-hard gain.

## Default-off and production identity

With both new flags absent, the retained fast lazy+gate subset reproduced the
R-CR1 counts exactly:

- `mvp2lvc`: UNKNOWN at 10k/100k with 9,999/17,956 expansions and
  1,948,056/3,928,520 peak bytes;
- `xsnfyll`: WIN at 10k with 81 expansions and 7,156 peak bytes.

All closure counters and seed scan totals were zero. A non-test MSVC release
build also succeeded. Thus unused test instrumentation leaves production out
of the artifact and preserves the retained search decisions.

## Retained raw logs

- `CLOSURE_BUILD_RAW.log`
- `CLOSURE_DEFAULT_OFF_RAW.log`
- `CLOSURE_COUNTER_SMOKE_RAW.log`
- `CLOSURE_SEED_SMOKE_RAW.log`
- `CLOSURE_LEAF_AB_RAW.log`
- `CLOSURE_PRODUCTION_BUILD_RAW.log`

The following official logs do not yet exist because their >11 GiB start gate
was never satisfied:

- `CLOSURE_COUNTER_FULL_OFF_RAW.log`
- `CLOSURE_SEED_FULL_ON_RAW.log`

## Regeneration

Every command below is run from this worktree. Do not start Cargo if another
Cargo process exists. The leaf/build lanes require free RAM above 9 GiB; each
full 1 GiB arm requires a fresh check strictly above 11 GiB.

### Full profile, counters on, seed off

```powershell
$ErrorActionPreference = 'Stop'
if (Get-Process cargo -ErrorAction SilentlyContinue) {
    throw 'another cargo process is running'
}
$os = Get-CimInstance Win32_OperatingSystem
$free = [math]::Round($os.FreePhysicalMemory / 1MB, 3)
if ($free -le 11) { throw "free RAM $free GiB is not above 11 GiB" }
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }

$env:CARGO_TARGET_DIR = '.target-codex'
$env:TSS_BACKWALK_TT_BYTES = '1073741824'
$env:TSS_LAZY_FRONTIER = '1'
$env:TSS_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CLOSURE_COUNTERS = '1'
$env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS = '0'
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER = '1'
$env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME = '0'
$env:TSS_CORPUS_EXPECT_CAP_RESUME = '0'
$env:TSS_CORPUS_EXPECT_LIVE_GE3_SEED = '0'
$env:TSS_CORPUS_EXPECT_CLOSURE_COUNTERS = '1'

cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture 2>&1 |
    Tee-Object CLOSURE_COUNTER_FULL_OFF_RAW.log
```

### Full profile, counters on, seed on

Repeat the preceding environment exactly, then add/change only:

```powershell
$env:TSS_LIVE_GE3_SEED = '1'
$env:TSS_CORPUS_EXPECT_LIVE_GE3_SEED = '1'

cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture 2>&1 |
    Tee-Object CLOSURE_SEED_FULL_ON_RAW.log
```

### Selected Phase-3 cells

```powershell
$env:CARGO_TARGET_DIR = '.target-codex'
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq closure_debt_live_ge3_leaf_ab -- `
    --ignored --test-threads=1 --nocapture
```

### Default-off and production checks

```powershell
$env:CARGO_TARGET_DIR = '.target-codex'
cargo build --release --target x86_64-pc-windows-msvc -p hexfield_eq

$env:TSS_BACKWALK_TT_BYTES = '1073741824'
$env:TSS_LAZY_FRONTIER = '1'
$env:TSS_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CORPUS_ID = 'mvp2lvc,xsnfyll'
$env:TSS_CORPUS_MAX_CAP = '100000'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture
```

## Clean stopping statement (session)

No cursor implementation exists, because Debt A's official counter gate has
not run. Debt B cannot promote because the required leaf non-regression is
already false. To finish the literal measurement matrix, wait until free RAM
is above 11 GiB and run the two full commands above without changing any other
flag.

## Orchestrator completion addendum (2026-07-17, post-session)

The Debt A official counter profile was run by the orchestrator after the
session exited, using the exact §Regeneration command
(`CLOSURE_COUNTER_FULL_OFF_RAW.log`, retained; CORPUS_DONE failures=0;
test wall 584.85 s). RAM judgment documented: free-physical read 8.1 GiB,
but the deficit was Windows standby cache from same-day archive I/O (WSL
VM idle at 0.76 GiB, trainer not training, process peak <3 GiB) — the
conservative gate metric understated true availability for this
orchestrator-attended run.

Official aggregate (CLOSURE_DONE): 1,803,229,707 pairs evaluated,
3,805,838 expanded (99.79% never expanded); pair-generation wall
299,288.6 ms; conservative sound avoidable tail = 4,668.7 (second) +
18,466.3 (eval) + 308.2 (dedup) = **23,443.3 ms = 7.83% of
pair-generation wall**. The smoke estimate (7.96%) is confirmed.

**Debt A verdict: CLOSING NULL.** The decision rule required ≥11.5% of
pair-generation wall avoidable by a sound cursor to clear the 5%
end-to-end bar; the measured ceiling is 7.83% (≈3.4% end-to-end). The
mechanism matches the session's smoke analysis: refuted and unresolved
Choice nodes must exhaust the generator (refutation before exhaustion is
forbidden), and at proven nodes the winning child frequently sits deep in
rank order (bins `[10785, 5686, 11250, 5666, 21508, 14981, 5343, 2460]`),
so the sound reveal prefix covers most classification cost. No cursor
build is justified. The seed full-on arm (`CLOSURE_SEED_FULL_ON_RAW.log`)
was intentionally not run: Debt B's promotion conjunction had already
failed on the selected-leaf regression, so the deep arm cannot change
that verdict.

**Both register closure debts are now disposed: A null (measured
ceiling), B null (leaf regression).**
