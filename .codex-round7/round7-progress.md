# Round 7 progress

Build base: `e29e18b8e4955a7baa629d31da68de4892ce83af`

## Step 0: binding handoff and harness audit

- Read `docs/TSS_VCF_WIDTH_BRIEF.md` (all 352 lines) and
  `.codex-round6/round6-progress.md` (all 92 lines).
- ADDENDUM 2 and ADDENDUM 3 remain binding except for the explicit Round 7
  supersessions: diagnosis is complete, the real 20M close is authorized, and
  the committed Round 6 localization/results replace older historical values.
- Confirmed branch `claude/tss-vcf-width` and HEAD `e29e18b8`.
- Existing untracked `.codex-round5/`, `.codex-round6/prompt.txt`,
  `.codex-round7/prompt.txt`, and `.target-codex/` were present at start and
  are being preserved.
- Exact hard-child harness inputs: `TSS_CORPUS_ID=0l4291i_live`,
  `TSS_BACKWALK_PREFIX=14`, `TSS_BACKWALK_EXTRA=9,-2;12,-1`, with
  `TSS_BACKWALK_CAP` set per rung. The harness computes an attacker-to-move
  `SolveGoal::Win` after those two extra defender placements.
- Operational limits adopted: reuse `.target-codex/`, `CARGO_BUILD_JOBS=4`,
  one solve at a time, no git commits, and the specified 20M memory kill
  thresholds.

## Step 1 checkpoint: default-profile sanity spot-check

- Incremental command: `cargo build --release -p hexfield_eq` with
  `CARGO_TARGET_DIR=.target-codex` and `CARGO_BUILD_JOBS=4`.
- Build succeeded in 4.04 s.
- Spot-check command selected `jh7yo7y,xsnfyll` with
  `TSS_CORPUS_MAX_CAP=10000`; no engine A/B environment variables were set.
- `jh7yo7y`: **WIN**, 1,681 nodes, 194 TT hits.
- `xsnfyll`: **WIN**, 77 nodes, 1 TT hit.
- `CORPUS_DONE failures=0`. These exactly match the binding expected values,
  so the committed experimental scaffolding is inert at defaults.

## Step 2a: isolated hard child at 100k

- Default engine; only observational `TSS_TRACE_PN=1` was added.
- Inputs: `0l4291i_live`, prefix 14, extras `9,-2;12,-1`, cap 100,000.
- Result: **UNKNOWN**, 100,000 nodes, 33,840 TT hits, 546.0 s.
- Final root values: `pn=34`, `dn=3,610`, 99,999 PN expansions,
  247,173 retained entries. The staged trace reached depth 60.
- The ignored helper's nonzero test exit is expected because exact mode
  asserts WIN; the measurement itself completed normally.

## Step 2 checkpoint: isolated hard child banked at the 1M rung

- Same default engine and exact hard-child inputs; cap 1,000,000. Output was
  logged to `.codex-round7/hard-child-1m.log`.
- Result: **WIN**, 225,924 nodes, 103,415 TT hits, 1,272.81 s.
- The staged PN search completed at depth 62 with 225,923 expansions,
  539,217 retained entries, `pn=0`, `dn=1,000,000,000`.
- Frontier experiment counters remained fully inert: 0 attempts, 0 proofs,
  0 probe expansions.
- This proves the sole off-reference defender-pair child within the 1M rung
  and authorizes the ordered full `0l4291i_live` 20M-rung attempt.
- Sandbox mechanics note: ordinary hidden `Start-Process`, `cmd start`, and
  ShellExecute children are killed as soon as the launching tool shell exits.
  The shell is inside a Windows job with `KILL_ON_JOB_CLOSE` and no permitted
  breakaway; direct breakaway, WMI, Task Scheduler, and WSL broker attempts
  were unavailable. A genuinely detached solve therefore requires an
  orchestrator/external launch in this environment.

