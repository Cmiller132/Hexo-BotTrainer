# Group 2, round 4 — consolidation

Date: 2026-07-16 (America/New_York)

Tip audited: `5536b2bb75fa6b0e7b6bfbc79dddc8f0aa9a0f77` (the docs commit on top of
round-3 implementation commit `bfd03ca9`). No git commit is permitted or
attempted in this round.

## Checkpoints

### C0 — binding brief and dependency audit complete

- Read `docs/PLAN_TSS_SOLVER_UPGRADES.md` §I.3, its IN-FLIGHT register, and
  `.codex-group2/round3-progress.md` before editing code.
- Initial tracked worktree state was clean. Existing untracked prompt, PID,
  native executable, target, and `.codex-round5/` artifacts are user-owned
  and remain untouched.

#### C1 production dependency map

`TssSolver::default()` is not a superseded offline-only wrapper. It is the
live trainer solver:

1. `RustSearch` owns `TssSolverSlot`, whose default and clone construct
   `TssSolver::default()` (`tree.rs`). Inline gated leaves call
   `tss_solve_verified` with that persistent solver.
2. The payload root guard constructs `TssSolver::default()` in `search.rs`
   and calls the same verified solve seam.
3. Every async worker constructs a persistent `TssSolver::default()` in
   `tss_async.rs` and calls the same seam; a caught worker panic also resets
   to a fresh default solver.
4. `tss_solve_verified` supplies the production 256 KiB cap and calls
   `solve_goal`. `TssSolver::default()` selects `WidthOptions::default()`.
5. `solve_goal` therefore applies `split_tt_cap` (half solve-local, half
   persistent positive-fragment cache) and `prove_for` dispatches to
   `prove_for_at_depth`, not `WidePnSearch`.
6. `prove_for_at_depth` builds `SearchContext`; claimant Choice nodes call
   `ordered_threat_creating_moves_with_width`, whose default branch calls the
   historical count>=3 `threat_creating_moves` generator.
7. The default Rust suite exercises this same route, including deterministic
   certificates, TT collision/equivalence, persistent-cache, goal-filter, and
   verifier checks. The Python Stage-0 golden digest pins the outer flag-off
   trainer stream.

Disposition: **C1 is not executable beyond audit in this round.** There is no
separate root/offline narrow entry point: the private narrow DFS/generator is
the production implementation and is also used by round-3 quiet+zone consume.
`tss_reference.rs` and `tss_reference_fast.rs` remain unchanged and test-only.

Concrete completion proposal: implement a `WidePnSearch` narrow-options mode
without changing `TssSolver::default()` routing; prove exact status, node
count, certificate bytes, local/shared-TT behavior, cache-warm behavior, and
Stage-0 golden digest identity over the full default and narrow fixture set;
only then flip the default dispatcher in a separately gated migration and
delete `SearchContext`/the count>=3 generator after a second identity run.

Checkpoint boundary: dependency map complete. No C1 code was deleted.

### C1 — C2 deletion and C3 scaffold inventory complete

- Deleted `WideRacer` completely: `WidePnSearch.racer`, the
  `TSS_WIDE_AB_RACER` cfg(test) environment hook, the ambiguous-choice probe
  branch, racer Zobrist/memo DFS, constants, and helper.
- History audit found that round-8b commit `c37e0799` already deleted the
  losing round-6 DAG/graph-PN, bounded frontier-probe, and associated A/B
  variants. No residual `TSS_WIDE_AB_DAG_PN`, `TSS_WIDE_AB_GRAPH_PN`,
  `DagFrontier`, `WideMacroDfs`, or frontier-probe implementation remains.
- Retained TT collision hooks, stage-refresh telemetry, and `TSS_TRACE_PN`:
  they are default-suite correctness checks or gate diagnostics, not losing
  alternate search paths.
- Retained `tss_corpus_backward_walk`, round-2 mining/exact-branch helpers,
  and round-3 shadow/verify/consume harnesses because progress memos contain
  their exact regeneration commands. Retained all `hunt_*` tests, forcing and
  spare corpora, `.codex-round9b-gate/`, and every progress memo as required.

Checkpoint boundary: C2 is complete. C3 found no additional safely deletable
residual after round-8b's prior consolidation; documented/load-bearing
harnesses remain.

### C2 — C4 profile and ladder documentation consolidated

- `docs/TSS_RUNBOOK.md` is now the single operational source for: 512 MiB
  ordinary offline default; 2 GiB official deep profile selected by
  `TSS_BACKWALK_TT_BYTES=2147483648`; 256 KiB trainer per-solve cap; forcing
  ladder 10k→100k→1M→20M with NO rows stopping at 1M; and spare-corpus
  acceptance semantics.
- `tss_corpus.rs` points to that story and gives the serialized official gate
  command. The master plan's stale "round 3 running / Step 4 in progress" and
  deletion claims were reconciled with the landed code and C1 audit.
- `double_fork_compact` is verifier-accepted evidence but is not currently a
  row in the two-NO spare corpus. Adding `WIN_VERIFIED` would require a new
  parsed expectation class and acceptance behavior, which is outside this
  deletion/comment-only round. Proposal: add that provenance-bearing status
  in a later corpus-semantics change, or add `WIN_PENDING` only if the corpus
  format is explicitly ruled to treat strict-verifier acceptance as its
  documented oracle provenance.

Checkpoint boundary: source edits are limited to deletion plus comments/docs.
C5 gates pending.

## C1–C5 disposition summary

| Item | Disposition |
|---|---|
| C1 | Partial/audit only; production dependency blocks deletion. |
| C2 | Complete; racer and all hooks/code deleted. |
| C3 | Complete audit; earlier round-8b deletion already removed losing variants; remaining candidates are required tests, diagnostics, or documented regeneration helpers. |
| C4 | Complete; one profile story and rung ladder in `TSS_RUNBOOK.md`; spare semantics unchanged. |
| C5 | Complete; every headline was re-derived after the consolidation edits. |

## Tip headline table

| Headline | Consolidated-tip result | Exact regeneration command |
|---|---|---|
| Default release suite | **PASS: 95 passed / 0 failed / 17 ignored**, 2.87 s test time | `$env:CARGO_TARGET_DIR='.target-codex'; cargo test --release -p hexfield_eq -- --test-threads=1` |
| Narrow shadow/default identity | **PASS**, including historical narrow WIN/2,884 and ordinary-wide vs shadow status/node/certificate equality | `$env:CARGO_TARGET_DIR='.target-codex'; cargo test --release -p hexfield_eq tss_round3_shadow_spare_coverage -- --ignored --test-threads=1 --nocapture` |
| Stage-0 golden digest | **PASS: 1/0**, 4.58 s in the documented Python 3.12 WSL environment | WSL command below |
| Verifier mutation gate | **PASS: baseline accepted; all seven mutations rejected** | `$env:CARGO_TARGET_DIR='.target-codex'; cargo test --release -p hexfield_eq tss_round3_verifier_mutations -- --ignored --test-threads=1 --nocapture` |
| R1b sharpness fixture | **PASS: 1/0** | `$env:CARGO_TARGET_DIR='.target-codex'; cargo test --release -p hexfield_eq hunt_r1b_chain_sharpness -- --ignored --test-threads=1 --nocapture` |
| R1b production cross-check | **PASS: 1/0** | `$env:CARGO_TARGET_DIR='.target-codex'; cargo test --release -p hexfield_eq hunt_seed_band_matches_production -- --ignored --test-threads=1 --nocapture` |
| `double_fork_compact` consume witness | **WIN / 409 nodes / 51 TT hits / 67,177,998 peak TT bytes / 24 ms / strict verifier ACCEPTED** at 10k | `$env:CARGO_TARGET_DIR='.target-codex'; $env:TSS_R3_CAP='10000'; $env:TSS_BACKWALK_TT_BYTES='536870912'; cargo test --release -p hexfield_eq tss_round3_consume_witness -- --ignored --test-threads=1 --nocapture` |
| Official all-19 gate | **PASS, `CORPUS_DONE failures=0`**, 14/14 WIN, 5/5 NO non-WIN, 444.90 s test time (452.3 s including rebuild) | `$env:CARGO_TARGET_DIR='.target-codex'; $env:TSS_BACKWALK_TT_BYTES='2147483648'; cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture` |

### C3 — C5 consolidated-tip gates complete

- Every build/solve was serialized and preceded by a host free-RAM reading
  above 8 GiB (observed range 12.58–13.97 GiB during verification).
- Default release remained exactly **95/0**; ignored count remained 17 because
  no documented harness was deleted.
- The explicit narrow identity harness passed, and the Stage-0 golden digest
  matched the frozen parent-build digest. A non-normative Windows CPython 3.14
  attempt access-violated inside the extension; the test was rerun in its
  documented Linux CPython 3.12 `hexfield-dev` environment and passed. This
  was an environment issue, not a solver status disagreement.
- R1b chain sharpness and production seed-band cross-check both passed.
- `double_fork_compact` re-derived the exact round-3 headline at the first
  rung: WIN/409, strict verifier accepted.
- The official single-process all-19 gate passed at the exact 2 GiB profile;
  there was no WIN-vs-LOSS disagreement at matched semantics.

Checkpoint boundary: all required headline gates are green.

## Exact regeneration commands

```powershell
$free = Get-CimInstance Win32_OperatingSystem | ForEach-Object { $_.FreePhysicalMemory / 1MB }
if ($free -le 8) { throw 'Free RAM must exceed 8 GiB before build/solve' }
$env:CARGO_TARGET_DIR='.target-codex'
cargo test --release -p hexfield_eq -- --test-threads=1

cargo test --release -p hexfield_eq tss_round3_shadow_spare_coverage -- --ignored --test-threads=1 --nocapture
cargo test --release -p hexfield_eq tss_round3_verifier_mutations -- --ignored --test-threads=1 --nocapture
cargo test --release -p hexfield_eq hunt_r1b_chain_sharpness -- --ignored --test-threads=1 --nocapture
cargo test --release -p hexfield_eq hunt_seed_band_matches_production -- --ignored --test-threads=1 --nocapture

$env:TSS_R3_CAP='10000'
$env:TSS_BACKWALK_TT_BYTES='536870912'
cargo test --release -p hexfield_eq tss_round3_consume_witness -- --ignored --test-threads=1 --nocapture

$env:TSS_BACKWALK_TT_BYTES='2147483648'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture
```

Stage-0 golden digest, in its documented environment:

```powershell
$free = Get-CimInstance Win32_OperatingSystem | ForEach-Object { $_.FreePhysicalMemory / 1MB }
if ($free -le 8) { throw 'Free RAM must exceed 8 GiB before build/solve' }
wsl -e bash -lc 'set -euo pipefail; source ~/.cargo/env; source /root/.venvs/hexfield-dev/bin/activate; cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/tss-vcf-width; export CARGO_TARGET_DIR=.target-codex; maturin develop --release -m packages/hexfield_eq/Cargo.toml; PYTHONPATH=packages/hexfield_eq/python:packages/hexo_runner/python:packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python python -m pytest tests/test_hexfield_eq_tss_shadow.py::test_stage0_digest_matches_golden -q'
```

## Final audit

- `rustfmt --edition 2021` was run only on the two edited Rust files.
- `git diff --check` passes.
- Tracked diff: `tss_solver.rs` is racer deletion only (1 restored direct
  expression line, 307 deleted lines); `tss_corpus.rs` is comments only;
  `PLAN_TSS_SOLVER_UPGRADES.md` and `TSS_RUNBOOK.md` are documentation only.
  This memo is new. There are no behavioral additions.
- Grep confirms no `WideRacer`, `TSS_WIDE_AB_RACER`, racer constants/helpers,
  old DAG/graph-PN hooks, or bounded frontier-probe scaffolds remain in Rust.
- Existing untracked user/build artifacts remain untouched. `HEAD` remains
  `5536b2bb75fa6b0e7b6bfbc79dddc8f0aa9a0f77`; no commit was attempted.
- Final free physical RAM reading: 13.16 GiB.

Final disposition: **complete and green subject to the explicit C1 owner
decision**. C1 cannot be deleted without replacing the live production leaf
solver under a separately proven byte-identity migration; every executable
consolidation item and every required tip gate is complete.
