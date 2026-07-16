# Round 8 final-gate checkpoint

> **Status:** the consolidated engine and every split gate are green. The one
> required clean all-19 corpus replay is pending an orchestrator run because
> its measured components exceed the binding two-hour synchronous-shell
> window. Do not treat this file as a completed final gate until the pending
> section is replaced with the returned all-19 result.

## Outcome

- The 512 MiB TT saturation hypothesis was confirmed by engine telemetry.
  At the 1M 0l cap, the first rejected index insert occurred at expansion
  603,584 with 1,831,466 retained entries. The run ended with 3,550,627
  retained entries but only 1,831,465 indexed, 1,719,162 rejected inserts,
  and 115 bytes of reported TT headroom.
- A test-harness-only `TSS_BACKWALK_TT_BYTES` override now feeds the existing
  `SolveCaps::tt_bytes_cap` resource cap. Its default remains exactly 512 MiB;
  production code and narrow defaults are unchanged.
- With a 2 GiB test TT, full `0l4291i_live` is **WIN in 2,335,295 certified
  nodes**, with 1,492,036 TT hits and peak TT 2,063,694,498 bytes. All
  6,880,208 retained positions were indexed and there were zero rejected
  inserts. The consolidated engine reproduced those node and hit counts
  exactly.
- Fix A was therefore neither needed nor implemented. All losing experimental
  scaffolds were deleted; Fix B's commitment-domain scheduling is the single
  unconditional wide-mode implementation.

## Final resource-profile decision

The clean all-19 replay must set
`TSS_BACKWALK_TT_BYTES=2147483648`. This is a legitimate resource profile,
not an engine-semantics switch:

- the variable exists only in the `#[cfg(test)]` corpus/backwalk module;
- it changes only the byte capacity supplied through the existing
  `SolveCaps` API;
- candidate generation, ordering, certification, and the corpus node ladder
  remain unchanged;
- unset behavior remains the original 512 MiB default; and
- 512 MiB was directly proven to stop indexing 0l's working set, while 2 GiB
  reproduces the banked certificate without a rejected insert.

Thus “default settings” below means default engine behavior and the built-in
10k -> 100k -> 1M -> 20M ladder, with the sole documented 2 GiB test-harness
resource override.

## Consolidated split-gate evidence

- Full unit/doc suite: 95 passed, 0 failed, 3 ignored; warning-free
  (`final-unit.log`).
- Twelve-entry WIN matrix: all green (`final-matrix-12-2g.log`).
- `lz60mfb`: WIN at the 1M rung in 122,132 nodes / 12,165 hits
  (`final-lz-2g.log`).
- Prefix-14 Universal regression: LOSS in 389,569 nodes / 164,841 hits,
  below the approximately 400k ceiling (`final-prefix14-2g.log`).
- Consolidated full 0l exact gate: WIN in 2,335,295 nodes / 1,492,036 hits
  (`final-0l-4m-2g.log`).
- Five `expect=NO` entries: zero WIN results and `CORPUS_DONE failures=0`
  (`final-no-5-2g.log`).
- Narrow default mode: 101 normalized rows, zero differences from
  `.codex-round5/narrow-b0.sig`, SHA-256
  `0098C8BFC6382156979FFE2C022E780EF34D53ABE37477E77A939C052470C4F2`
  (`final-narrow-default.log`, `final-narrow-default.sig`).

## Banked 19-entry matrix before the clean replay

Every row below is from the consolidated 2 GiB engine, but from split or
isolated gates. The official all-19 replay must reproduce them in one test
process. In particular, 0l's isolated 4M cap is not a built-in ladder rung, so
the clean corpus gate will report its 2,335,295-node proof on the 20M rung.

| # | Corpus ID | Banked rung | Status | Nodes | TT hits | Source |
|---:|---|---:|---|---:|---:|---|
| 1 | `0hz3hty` | 10k | WIN | 2,319 | 2,268 | 12-entry |
| 2 | `0l4291i_live` | 4M exact | WIN | 2,335,295 | 1,492,036 | isolated |
| 3 | `8is963b` | 10k | LOSS | 1 | 0 | five-NO |
| 4 | `94gnnol` | 1M | UNKNOWN | 1,000,000 | 612,111 | five-NO |
| 5 | `acly7kb` | 10k | WIN | 75 | 0 | 12-entry |
| 6 | `dy3dg99` | 10k | LOSS | 1 | 0 | five-NO |
| 7 | `g2xx6wl` | 10k | WIN | 4,244 | 1,995 | 12-entry |
| 8 | `hu01jk4` | 10k | WIN | 380 | 0 | 12-entry |
| 9 | `jh7yo7y` | 10k | WIN | 2,018 | 337 | 12-entry |
| 10 | `jnzzmcm` | 100k | WIN | 13,646 | 5,068 | 12-entry |
| 11 | `l9mxn59` | 1M | UNKNOWN | 235 | 18 | five-NO |
| 12 | `lz60mfb` | 1M | WIN | 122,132 | 12,165 | isolated |
| 13 | `mvp2lvc` | 1M | UNKNOWN | 19,207 | 1,847 | five-NO |
| 14 | `xsnfyll` | 10k | WIN | 81 | 1 | 12-entry |
| 15 | `zrugh2x` | 100k | WIN | 39,739 | 11,841 | 12-entry |
| 16 | `strongloss_a_prefix6` | 100k | WIN | 16,245 | 7,699 | 12-entry |
| 17 | `strongloss_b_prefix8` | 10k | WIN | 682 | 151 | 12-entry |
| 18 | `hayes_20260712_turn16` | 100k | WIN | 13,524 | 3,201 | 12-entry |
| 19 | `hayes_20260712_placement31` | 100k | WIN | 13,524 | 3,201 | 12-entry |

## Pending official all-19 replay

The full replay cannot safely run in this agent shell. The 0l 1M miss and
winning rung alone measured about 40 + 121 minutes; `94gnnol`'s NO ladder adds
about 59 minutes, before the other 17 entries. Run this synchronously with the
host RAM guard active:

```powershell
Set-Location 'E:\Hexo-BotTrainer-hexgt\.claude\worktrees\tss-vcf-width'

$ErrorActionPreference = 'Stop'
Get-ChildItem Env:TSS_* -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -LiteralPath ("Env:" + $_.Name) }

$env:CARGO_TARGET_DIR = 'E:\Hexo-BotTrainer-hexgt\.claude\worktrees\tss-vcf-width\.target-codex'
$env:CARGO_BUILD_JOBS = '4'
$env:TSS_BACKWALK_TT_BYTES = '2147483648'

cmd.exe /d /c "cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --nocapture 2>&1" |
    Tee-Object -FilePath '.codex-round8\final-matrix-19-2g.log'
$cargoExit = $LASTEXITCODE
if ($cargoExit -ne 0) {
    throw "Final all-19 gate failed with cargo exit code $cargoExit"
}
```

Completion requires `.codex-round8/final-matrix-19-2g.log` to contain all 19
IDs in corpus order, `CORPUS_DONE failures=0`, an `ok` Rust test result, all 14
WIN entries certified somewhere on the fixed ladder, and zero
`status=WIN expect=NO` rows. Replace this pending section with the official
per-entry rung/node matrix after the orchestrator returns the log.
