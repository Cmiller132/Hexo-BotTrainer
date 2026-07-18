# R-LF2 - lazy frontier and the reduced-TT memory wall

Status: **MEASURED; 1 GiB FULL GATE GREEN WITH LAZY ON; 512 MiB `0l`
CLOSES IN BOTH MODES, BUT 512 MiB WAS FILTERED AND IS NOT A FULL-GATE RUN.**
No commit was made.

## Answer

The exact smallest budget in the requested matrix with an executed **full
19-row** lazy-on gate ending in `CORPUS_DONE failures=0` is **1 GiB**. The
1 GiB and 2 GiB lazy-on runs have identical verdicts, closing rungs,
expansions, and `peak_tt_bytes` on every rung; only wall time varies. In
particular, `0l4291i_live` closes at 1,879,612 expansions and
549,161,606 charged TT bytes in both runs. Thus 1 GiB is uncapped for the
official corpus under lazy admission and the official profile can be reduced
from 2 GiB to 1 GiB when `TSS_LAZY_FRONTIER=1` is asserted.

At 512 MiB the work order deliberately ran only the heavy `0l4291i_live` row,
not the full gate. That row closes in both modes. Lazy on changes the
cap-pressured traversal from 18,133,032 to 1,913,955 expansions (-89.44%) and
the final-rung wall from 1,660.210 s to 198.033 s (-88.07%, 8.38x faster).
This is strong evidence that lazy admission moves the practical memory wall,
but it is not an executed full-512-MiB `CORPUS_DONE` claim and it does not
establish the absolute minimum budget below 1 GiB.

Fragments were explicitly off throughout every new run. Every returned hard
verdict carried a certificate and the gate's independent `TssVerifier` check
accepted it. No expected-NO row became WIN, no WIN row became UNKNOWN at its
final ladder rung, and no certificate/soundness stop condition fired.

## Matrix

All new gate-class runs began with more than 11 GiB free physical RAM, used
`CARGO_TARGET_DIR=.target-hunt`, one Cargo process, and
`--test-threads=1`. `TSS_CORPUS_EXPECT_SHARED_FRAGMENTS=0` and the matching
lazy expectation were asserted inside the gate. Recorded preflights were
13.03 GiB (512/on), 13.00 GiB (512/off), 18.34 GiB (1 GiB/on), 16.99 GiB
(1 GiB/off), and 16.56 GiB (2 GiB/on).

| TT budget | lazy off | lazy on | scope |
|---:|---|---|---|
| 512 MiB | `0l` PASS, WIN@20M; test 1,773.99 s | `0l` PASS, WIN@20M; test 318.10 s | filtered `0l4291i_live` only; **not full** |
| 1 GiB | `CORPUS_DONE failures=0`; 836.51 s | `CORPUS_DONE failures=0`; 491.27 s | full 19 rows, 34 rung solves |
| 2 GiB | `CORPUS_DONE failures=0`; 436.80 s historical flags-off baseline | `CORPUS_DONE failures=0`; 492.02 s | full 19 rows, 34 rung solves |

The 2 GiB off cell is the owner-authorized known baseline rather than a rerun:
`tss-vcf-width/.codex-round9b-gate/final-matrix-19-9b.log` and its `GATE.md`
in the read-only sibling worktree, at
`ac3f455f`, record the fragments-off/default-lazy-off full gate, 14/14
certified WIN rows, 5/5 non-WIN controls, and `CORPUS_DONE failures=0` in
436.80 s. The default-off structural row data remain byte-identical on this
branch when uncapped; the new 1 GiB off run also reproduces every baseline
row's verdict, rung, expansions, and peak bytes except the cap-pressured `0l`
traversal.

Wall times are not compared across the historical 2 GiB baseline and the new
runs as a controlled performance experiment: they ran at different times and
the host is shared. The paired 1 GiB runs are the useful gate-level timing
comparison: lazy on reduced test wall by 41.27%.

## Per-row final result

Each cell is `status@closing-or-final-rung; expansions; peak_tt_bytes; wall`.
For expected-NO rows, `UNKNOWN@1M` is an accepted non-WIN final result. The
2 GiB off column is the historical flags-off baseline described above.

| id | expect | 1 GiB off | 1 GiB on | 2 GiB off baseline | 2 GiB on |
|---|---|---|---|---|---|
| 0hz3hty | WIN | WIN@10k; 2,412; 825,414 B; 0.126 s | WIN@10k; 2,412; 284,940 B; 0.136 s | WIN@10k; 2,412; 825,414 B; 0.125 s | WIN@10k; 2,412; 284,940 B; 0.137 s |
| 0l4291i_live | WIN | WIN@20M; 6,505,681; 1,073,741,631 B; 568.651 s | WIN@20M; 1,879,612; 549,161,606 B; 195.744 s | WIN@20M; 1,879,612; 1,729,265,069 B; 177.652 s | WIN@20M; 1,879,612; 549,161,606 B; 195.529 s |
| 8is963b | NO | LOSS@10k; 1; 0 B; 0.000 s | LOSS@10k; 1; 0 B; 0.000 s | LOSS@10k; 1; 0 B; 0.000 s | LOSS@10k; 1; 0 B; 0.000 s |
| 94gnnol | NO | UNKNOWN@1M; 1,000,000; 859,172,995 B; 109.688 s | UNKNOWN@1M; 1,000,000; 293,096,179 B; 125.919 s | UNKNOWN@1M; 1,000,000; 859,172,995 B; 105.324 s | UNKNOWN@1M; 1,000,000; 293,096,179 B; 125.530 s |
| acly7kb | WIN | WIN@10k; 75; 51,950 B; 0.009 s | WIN@10k; 75; 18,072 B; 0.009 s | WIN@10k; 75; 51,950 B; 0.008 s | WIN@10k; 75; 18,072 B; 0.009 s |
| dy3dg99 | NO | LOSS@10k; 1; 0 B; 0.000 s | LOSS@10k; 1; 0 B; 0.000 s | LOSS@10k; 1; 0 B; 0.000 s | LOSS@10k; 1; 0 B; 0.000 s |
| g2xx6wl | WIN | WIN@10k; 4,107; 5,391,595 B; 0.528 s | WIN@10k; 4,107; 1,488,570 B; 0.553 s | WIN@10k; 4,107; 5,391,595 B; 0.511 s | WIN@10k; 4,107; 1,488,570 B; 0.562 s |
| hu01jk4 | WIN | WIN@10k; 380; 277,892 B; 0.089 s | WIN@10k; 380; 143,141 B; 0.087 s | WIN@10k; 380; 277,892 B; 0.088 s | WIN@10k; 380; 143,141 B; 0.100 s |
| jh7yo7y | WIN | WIN@10k; 2,119; 879,142 B; 0.220 s | WIN@10k; 2,119; 333,256 B; 0.232 s | WIN@10k; 2,119; 879,142 B; 0.220 s | WIN@10k; 2,119; 333,256 B; 0.228 s |
| jnzzmcm | WIN | WIN@10k; 9,798; 9,149,610 B; 0.802 s | WIN@10k; 9,798; 2,224,728 B; 0.893 s | WIN@10k; 9,798; 9,149,610 B; 0.764 s | WIN@10k; 9,798; 2,224,728 B; 0.917 s |
| l9mxn59 | NO | UNKNOWN@1M; 226; 57,436 B; 0.014 s | UNKNOWN@1M; 226; 27,206 B; 0.015 s | UNKNOWN@1M; 226; 57,436 B; 0.014 s | UNKNOWN@1M; 226; 27,206 B; 0.015 s |
| lz60mfb | WIN | WIN@1M; 109,896; 90,366,042 B; 9.061 s | WIN@1M; 109,896; 25,072,700 B; 9.634 s | WIN@1M; 109,896; 90,366,042 B; 8.707 s | WIN@1M; 109,896; 25,072,700 B; 9.626 s |
| mvp2lvc | NO | UNKNOWN@1M; 17,957; 10,580,910 B; 1.749 s | UNKNOWN@1M; 17,957; 3,928,520 B; 1.949 s | UNKNOWN@1M; 17,957; 10,580,910 B; 1.674 s | UNKNOWN@1M; 17,957; 3,928,520 B; 1.956 s |
| xsnfyll | WIN | WIN@10k; 82; 14,688 B; 0.005 s | WIN@10k; 82; 7,156 B; 0.005 s | WIN@10k; 82; 14,688 B; 0.004 s | WIN@10k; 82; 7,156 B; 0.005 s |
| zrugh2x | WIN | WIN@100k; 41,734; 19,286,202 B; 3.775 s | WIN@100k; 41,734; 8,062,494 B; 4.031 s | WIN@100k; 41,734; 19,286,202 B; 3.647 s | WIN@100k; 41,734; 8,062,494 B; 4.030 s |
| strongloss_a_prefix6 | WIN | WIN@100k; 16,126; 3,637,168 B; 1.002 s | WIN@100k; 16,126; 1,580,150 B; 1.094 s | WIN@100k; 16,126; 3,637,168 B; 0.918 s | WIN@100k; 16,126; 1,580,150 B; 1.064 s |
| strongloss_b_prefix8 | WIN | WIN@10k; 1,099; 294,642 B; 0.064 s | WIN@10k; 1,099; 104,364 B; 0.066 s | WIN@10k; 1,099; 294,642 B; 0.060 s | WIN@10k; 1,099; 104,364 B; 0.066 s |
| hayes_20260712_turn16 | WIN | WIN@100k; 11,664; 4,722,266 B; 1.131 s | WIN@100k; 11,664; 1,840,070 B; 1.215 s | WIN@100k; 11,664; 4,722,266 B; 1.080 s | WIN@100k; 11,664; 1,840,070 B; 1.197 s |
| hayes_20260712_placement31 | WIN | WIN@100k; 11,664; 4,722,270 B; 1.123 s | WIN@100k; 11,664; 1,840,074 B; 1.262 s | WIN@100k; 11,664; 4,722,270 B; 1.085 s | WIN@100k; 11,664; 1,840,074 B; 1.197 s |

## The 512 MiB `0l` closure story

There is exactly one official corpus ID beginning with `0l`:
`0l4291i_live`. Both filtered runs used the standard ladder and fresh solver
per rung.

| lazy | rung | status | expansions | peak_tt_bytes | wall |
|---|---:|---|---:|---:|---:|
| off | 10k | UNKNOWN | 10,000 | 5,815,630 B | 1.228 s |
| off | 100k | UNKNOWN | 100,000 | 83,757,114 B | 11.537 s |
| off | 1M | UNKNOWN | 1,000,000 | 536,870,720 B | 100.867 s |
| off | 20M | WIN | 18,133,032 | 536,870,720 B | 1,660.210 s |
| on | 10k | UNKNOWN | 10,000 | 2,178,996 B | 1.298 s |
| on | 100k | UNKNOWN | 100,000 | 26,405,485 B | 12.505 s |
| on | 1M | UNKNOWN | 1,000,000 | 288,513,572 B | 106.112 s |
| on | 20M | WIN | 1,913,955 | 536,870,711 B | 198.033 s |

The uncapped 2 GiB off/on charged peaks are 1,729,265,069 and 549,161,606
bytes, a 68.24% reduction (3.149x ratio). At 1 GiB, eager admission saturates
and needs 6,505,681 expansions; lazy admission stays uncapped and needs the
same 1,879,612 expansions as at 2 GiB. At 512 MiB both modes saturate, so the
refinement lemma does not promise identical traversal. Lazy still closes near
the uncapped work count (+34,343 expansions, +1.83%), whereas eager needs
9.47x as many expansions. Both returned WIN certificates verified.

Accordingly, these are capacity/work facts under the cap-aware corollary. They
do not extend the uncapped-index exact-equivalence theorem through the cap
refusal.

## Recommendation and the 256 KiB leaf budget

**Recommend changing the official deep-solve profile to 1 GiB only together
with an asserted `TSS_LAZY_FRONTIER=1`.** That combination has a direct full
gate, remains comfortably above the measured 549,161,606-byte corpus peak,
and exactly reproduces the 2 GiB lazy-on structural results. Keep the feature
default-off until the owner separately chooses to change the production
default. Do not claim 512 MiB as the official full-gate profile from this
study: its only new evidence is the filtered bottleneck row.

The 256 KiB trainer-leaf budget does **not** inherit a demonstrated
proportional benefit today. Production trainer solves use the default narrow
profile, which splits 256 KiB between the solve-local TT and persistent cache;
R-LF1 changes only the wide PN engine, so lazy frontier is inactive on that
narrow path. If trainer leaves are later switched to pair-complete wide PN,
the LF1 admission reductions of 62.6%-67.3% correspond arithmetically to about
2.7x-3.1x as many indexed admissions at a fixed charged-TT budget (the
uncapped `0l` peak ratio here is 3.149x). That is only a first-order estimate:
at 256 KiB fixed allocations, early cap refusal, future-key storage, deferred
frontier metadata, and certificate/cache bytes matter, while
`peak_tt_bytes` excludes some of that frontier memory. A dedicated 256 KiB
wide leaf campaign is required before claiming any solve-rate or verdict
gain.

## Gate/filter handling and evidence

No gate source edit was needed. `tss_corpus_check` already had the additive,
default-off, validated `TSS_CORPUS_ID` filter (including comma-separated IDs
and an unknown-ID assertion). The unfiltered path is therefore untouched.
The current `CORPUS_MODE` line does not echo this pre-existing selector; the
work order required an echo only if a filter had to be added.

New raw logs:

- `.codex-hunt/lf2-512m-lazy-off-0l.log`
- `.codex-hunt/lf2-512m-lazy-on-0l.log`
- `.codex-hunt/lf2-1g-lazy-off-full.log`
- `.codex-hunt/lf2-1g-lazy-on-full.log`
- `.codex-hunt/lf2-2g-lazy-on-full.log`

The only tracked deliverable added by R-LF2 is this report. Existing LF1
implementation and proof files were not changed.

## Regeneration

Run one command at a time. Before a full or filtered gate-class run, require
strictly more than 11 GiB free RAM.

```powershell
$env:CARGO_TARGET_DIR='.target-hunt'
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }

$env:TSS_BACKWALK_TT_BYTES='536870912'
$env:TSS_CORPUS_ID='0l4291i_live'
$env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS='0'
Remove-Item Env:TSS_SHARED_FRAGMENTS -ErrorAction SilentlyContinue
$env:TSS_LAZY_FRONTIER='1'
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER='1'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture

Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
Remove-Item Env:TSS_LAZY_FRONTIER -ErrorAction SilentlyContinue
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER='0'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture
```

For each full budget, remove the selector and run off/on separately. The
required new full runs used 1 GiB off/on and 2 GiB on; 2 GiB off is the cited
historical baseline.

```powershell
Remove-Item Env:TSS_CORPUS_ID -ErrorAction SilentlyContinue
$env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS='0'
Remove-Item Env:TSS_SHARED_FRAGMENTS -ErrorAction SilentlyContinue

Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:TSS_BACKWALK_TT_BYTES='1073741824'
Remove-Item Env:TSS_LAZY_FRONTIER -ErrorAction SilentlyContinue
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER='0'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture

Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:TSS_LAZY_FRONTIER='1'
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER='1'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture

Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:TSS_BACKWALK_TT_BYTES='2147483648'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture
```
