# Group 2, round 6 — C1 migration flip/deletion (step 2 of 2)

Date: 2026-07-16 (America/New_York)

Base confirmed before edits: `ace1f5b2bfbba0c3ebf35f5fb80bf42963284a9f`
(round-5 narrow compatibility route plus identity harness). No git commit is
permitted or attempted. Existing untracked prompt, PID, executable, target,
and round-5 artifact paths are user-owned and remain untouched.

## Checkpoints

### C0 — work order and operational boundary confirmed

Read `.codex-group2/round5-progress.md` in full. Its Round-6 flip/deletion
checklist is the binding work order. Before any build or long run, free RAM
must exceed 10 GiB; native Cargo runs use `CARGO_TARGET_DIR=.target-codex`,
`--target x86_64-pc-windows-msvc`, and `--test-threads=1`. Stage-0 uses the
documented WSL command. Builds/solves remain serialized.

Initial free physical RAM: **13.25 GiB**. Tracked worktree state was clean.

Checkpoint boundary: no source edited and no build started.

### C1 — identity before flip green

Free RAM immediately before the run: **12.80 GiB**.

```powershell
$free = Get-CimInstance Win32_OperatingSystem | ForEach-Object { $_.FreePhysicalMemory / 1MB }
if ($free -le 10) { throw 'Free RAM must exceed 10 GiB before identity gate' }
$env:CARGO_TARGET_DIR='.target-codex'
cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_round5_narrow_compat_identity -- --ignored --test-threads=1 --nocapture
```

Result: **PASS: 1 passed / 0 failed**, 0.10 s test time. No identity
dimension mismatch and no WIN-vs-LOSS disagreement occurred.

Checkpoint boundary: the untouched `ace1f5b2` source is identity-green;
deletion remains forbidden until the separately reviewed flip and second green
identity run.

### C2 — default narrow dispatcher flipped; second identity green

The separately reviewed flip is one production line in `TssSolver::default()`:
its `width` changes from `WidthOptions::default()` to
`WidthOptions::narrow_compat()`. `solve_goal` caps, `split_tt_cap`, `prove_for`
arguments, and every caller remain byte-for-byte unchanged at this boundary.
The test-only `with_hash_mask` constructor remains legacy-default so the full
round-5 harness still compares the two routes rather than comparing a route to
itself.

Free RAM immediately before the second identity run: **12.89 GiB**. Command:
the C1 identity command above, unchanged.

Result: **PASS: 1 passed / 0 failed**, 0.10 s test time. No identity
dimension mismatch and no WIN-vs-LOSS disagreement occurred.

Checkpoint boundary: the production default flip is identity-green. The hard
gate permits legacy deletion.

### C3 — legacy route deleted; third identity green

- Removed the `SearchContext` alias and the duplicate `prove_for_at_depth`
  dispatcher route. Default narrow and round-3 quiet/zone consume cases now
  enter `WidePnSearch::prove_narrow_compat`; consume retains its existing zone
  override and all original `prove_for` arguments.
- Removed the now-obsolete migration bit and constructor. `WidthOptions` again
  expresses only semantic width/round-3 options, while `prove_for` owns the
  narrow compatibility dispatch.
- Removed the historical `threat_creating_moves` count>=3 wrapper. The
  generalized `threat_creating_moves_with_threshold` remains; ordinary narrow
  calls it directly with threshold 3 and pair-complete calls it with threshold
  2.
- Direct backend tests now name `NarrowCompatSearch` rather than retaining the
  deleted alias. The round-5 harness and its TT-signature hooks remain live.
- Grep is clean for `SearchContext`, `narrow_engine_migration`,
  `narrow_compat(`, `prove_for_at_depth`, and `fn threat_creating_moves(`.
- `rustfmt --edition 2021` and its `--check` form pass on the sole edited Rust
  file; `git diff --check` passes.

Free RAM immediately before the third identity run: **12.95 GiB**. Command:
the C1 identity command above, unchanged.

Result: **PASS: 1 passed / 0 failed**, 0.11 s test time. No identity
dimension mismatch and no WIN-vs-LOSS disagreement occurred.

Checkpoint boundary: deletion is identity-green; final evidence gates may
proceed.

### C4 — default release suite green

Free RAM immediately before the run: **12.98 GiB**.

```powershell
$env:CARGO_TARGET_DIR='.target-codex'
cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq -- --test-threads=1
```

Result: **PASS: 95 passed / 0 failed / 18 ignored**, 2.84 s test time.
This includes the round-3 quiet/zone production tests; the round-5 identity,
mutation, witness, and corpus gates remain intentionally ignored here and are
run explicitly below.

### C5 — Stage-0 WSL golden digest green

Free RAM immediately before the WSL build: **12.88 GiB**.

```powershell
wsl -e bash -lc 'set -euo pipefail; source ~/.cargo/env; source /root/.venvs/hexfield-dev/bin/activate; cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/tss-vcf-width; export CARGO_TARGET_DIR=.target-codex; maturin develop --release -m packages/hexfield_eq/Cargo.toml; PYTHONPATH=packages/hexfield_eq/python:packages/hexo_runner/python:packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python python -m pytest tests/test_hexfield_eq_tss_shadow.py::test_stage0_digest_matches_golden -q'
```

Result: **PASS: 1 passed / 0 failed**, 4.33 s pytest time. The extension
rebuilt and installed successfully in the documented CPython 3.12
`hexfield-dev` environment.

### C6 — verifier mutation gate green

Free RAM immediately before the run: **12.52 GiB**.

```powershell
$env:CARGO_TARGET_DIR='.target-codex'
cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_round3_verifier_mutations -- --ignored --test-threads=1 --nocapture
```

Result: **PASS: baseline WIN accepted; all seven mutations rejected**
(`omitted_zone_cell`, `omitted_defender_edge`, `dropped_quiet_edge`,
`wrong_budget`, `wrong_horizon`, `forged_leaf`, and
`out_of_zone_substitution`). Test result 1/0 in 0.07 s.

### C7 — `double_fork_compact` consume witness green

Free RAM immediately before the run: **12.78 GiB**.

```powershell
$env:CARGO_TARGET_DIR='.target-codex'
$env:TSS_R3_CAP='10000'
$env:TSS_BACKWALK_TT_BYTES='536870912'
cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_round3_consume_witness -- --ignored --test-threads=1 --nocapture
```

Result: **PASS: WIN / 409 nodes / 51 TT hits / 67,177,998 peak TT
bytes / 37 ms / strict verifier accepted**. Test result 1/0. This explicitly
proves the round-3 consume path still reaches the compatibility backend through
the same `prove_for` seam after legacy-route deletion.

### C8 — official serialized all-19 gate green

Free RAM immediately before the run: **12.79 GiB**.

```powershell
$env:CARGO_TARGET_DIR='.target-codex'
$env:TSS_BACKWALK_TT_BYTES='2147483648'
cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture
```

Result: **PASS: `CORPUS_DONE failures=0`**, 446.81 s test time. All 14
expected WIN rows closed at their official ladder rungs, and all five NO rows
remained Loss or Unknown. No verifier rejection and no WIN-vs-LOSS
disagreement occurred.

## Regeneration log

Commands and results are appended at each checklist boundary.

## Caller audit

| Site | Construction and cap path | Dispatch/cap audit | Changed? |
|---|---|---|---|
| `TssSolverSlot` (`tree.rs`) | Slot default and cold-cache clone both construct `TssSolver::default()`; `RustSearch` owns the persistent slot. Inline solves pass `divergences.tss_solver_node_cap` to `tss_solve_verified`. | Default construction now reaches the compatibility route through `prove_for`. Node cap, goal selection, zone options, and persistent ownership are unchanged. | **No caller change** |
| Verified-solve cap seam (`tree.rs`) | `tss_solve_verified` builds `SolveCaps` from the caller node cap, fixed `RustSearch::TSS_SOLVER_TT_BYTES`, and the existing +12 semantic horizon; the optional tight zone attempt still uses half the caller node cap and the same TT cap. | No cap or verifier/mint change accompanied the dispatch flip. | **No caller change** |
| Payload root guard (`search.rs`) | Creates a per-move `TssSolver::default()` and passes `div.tss_solver_node_cap`, `SolveGoal::Both`, and the existing zone settings to `tss_solve_verified`. | It inherits the new default route only; guard gating and caps are unchanged. | **No caller change** |
| Async persistent worker (`tss_async.rs`) | Each worker creates one persistent `TssSolver::default()` and forwards `request.node_cap`, goal, and zone unchanged to `tss_solve_verified`. | It inherits the new default route only; persistence and request caps are unchanged. | **No caller change** |
| Async panic reset (`tss_async.rs`) | A caught panic replaces the suspect worker solver with a fresh `TssSolver::default()`. | Reset behavior is unchanged and now reconstructs the same compatibility default as initial worker creation. | **No caller change** |
| TT split (`tss_solver.rs`) | Narrow mode still calls `split_tt_cap(effective_tt_cap)` before `prove_for`; pair-complete mode still keeps `(effective_tt_cap, 0)`. | The split and every `prove_for` cap argument are unchanged. | **No cap change** |

`git diff --exit-code HEAD -- tree.rs search.rs tss_async.rs` returns 0.
The only tracked file changed is `tss_solver.rs` (20 insertions, 127
deletions). Thus no caller-specific dispatcher flag or cap edit accompanied
the flip.

## Final audit

- `rustfmt --edition 2021` was run only on the edited Rust file;
  `rustfmt --edition 2021 --check packages/hexfield_eq/rust/src/tss_solver.rs`
  passes.
- `git diff --check` passes.
- Grep is clean across `packages/hexfield_eq/rust/src` for the deleted
  `SearchContext`, migration flag/constructor, duplicate legacy route, and
  historical count>=3 wrapper.
- The generalized generator remains referenced at thresholds 2 and 3. The
  identity harness's `last_narrow_signatures`, `NarrowAttemptSignature`, and
  `tt_behavior_signature` hooks all remain referenced; deletion left no
  orphaned test hook.
- Tracked implementation changes are confined to `tss_solver.rs`; this
  progress memo is new. Existing untracked user/build artifacts remain
  untouched.
- Final HEAD remains `ace1f5b2bfbba0c3ebf35f5fb80bf42963284a9f`; no commit
  was attempted.
- Final free physical RAM reading: **11.83 GiB**.

## Final headline

| Gate | Result |
|---|---|
| Identity 1 — pre-flip | **PASS: 1/0** |
| Identity 2 — post-flip | **PASS: 1/0** |
| Identity 3 — post-deletion | **PASS: 1/0** |
| Default release suite | **PASS: 95/0/18** |
| Stage-0 golden digest | **PASS: 1/0** |
| Verifier mutation gate | **PASS: baseline accepted; 7/7 mutations rejected** |
| `double_fork_compact` consume witness | **PASS: WIN/409, verified** |
| Official serialized all-19, 2 GiB | **PASS: failures=0** |

Final disposition: **complete and green for step 2 of 2**. The default narrow
solver now dispatches through `WidePnSearch::prove_narrow_compat`; the legacy
alias/route and wrapper are deleted; all identity, verifier, witness, trainer,
and official corpus evidence is unchanged.
