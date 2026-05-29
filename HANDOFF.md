# Handoff — dense_cnn Model 1 (scratch_64) MCTS memory work + crash hunt

This is a session handoff for continuing the dense_cnn Model 1 training work. It
assumes the reader has read `Model 1 goal.md` (the staged goals) and
`CLAUDE.md` (build/test/run conventions). Branch: **`rust-rebuild`**.

## Where this sits in the Model 1 goals

We are in **Goal #4 territory** (train many epochs until the model holds its own
vs SealBot best-50ms), running the small **64-channel / 4-block** config
(`configs/dense_cnn_model1_scratch_64.toml`). Goal #2's calibration and the
KataGo-style selfplay/shuffler/training split are in place. The immediate work
is *keeping a long run alive* — which is currently blocked by an intermittent
native crash (see "Open issue" below).

## What was done this session

### 1. MCTS memory & churn reduction (Change 1 — KEEP)
Implemented the approved "Dense CNN MCTS — memory & churn reduction" plan in
`packages/hexo_models/dense_cnn/rust/src/{mcts_eval.rs,mcts_tree.rs,mcts.rs}`:
- Eval-cache priors are now shared via **`Arc<RustEvaluation>`** (not `Rc` —
  `RustSearch` must be `Send` for `searches.par_iter_mut()` under rayon).
- Node priors are a `NodePriors` enum: `Shared(Arc<RustEvaluation>)` for interior
  nodes (no per-node prior copy) and `Owned(Vec<RustPriorCandidate>)` for roots
  (mutable for temperature + Dirichlet noise). `ensure_root_owned()` converts a
  reused/promoted `Shared` root to `Owned` before noise.
- Cache priors are normalized + sorted descending in `finalize_model_priors`.
- `root_prior_policy` exports the full distribution from either variant, so the
  training data (`root_prior_policy` → policy-surprise weights + `rootPolicy`
  target) stays **byte-identical**. This was the hard constraint and it holds.
- **Result: real memory win.** Tree prior memory ~6 MB (shared, counted once)
  vs ~1–1.6 GB of duplicated per-node priors before. Throughput-neutral.

### 2. Change 2 (eval-cache cap) — REVERTED, and the reason matters
A prior session lowered `MODEL1_EVAL_CACHE_MAX_STATES` to 131,072 believing it
caused a throughput regression, then reverted it (scratch_64 → 262144, main +
defaults + test → 1,048,576). **The revert is in place but the diagnosis behind
it was wrong.** Direct per-epoch diagnostics show:

| Cap     | real epoch-9 pos/s | unique states | recomputes (inserts−unique) |
|---------|--------------------|---------------|-----------------------------|
| 131072  | 11.48              | 1,416,161     | **0**                       |
| 262144  | 11.31              | 1,395,037     | **0**                       |

`eval_cache_inserts == eval_unique_states` in every block → **eviction never
forces a recompute** (tree reuse holds live states in the shared Arc nodes; only
dead states get evicted). There was never any cache thrashing. The "30 → 13
pos/s regression" was a **phantom**: ~30–34 was the *calibration probe* (2048
fresh-root positions, full GPU parallelism), while real-epoch selfplay is
~11 pos/s **regardless of cap or Change 1** — it's bounded by the GPU evaluator
running ~1.4M unique forward passes/epoch. The two numbers were never the same
metric.

**Implication:** the cache cap is throughput-neutral. Re-applying Change 2
(131072) would reclaim ~340 MB of cache RAM for free, but it's not urgent.
Decision was left to the user.

### 3. Change 3 (thread_local encode scratch) — KEEP
`evaluate_model1_states_chunk` reuses a `thread_local` `(Vec<f32>, Vec<i64>)`
scratch instead of re-allocating ~89 MB planes per chunk.

### Tests + build
Rust unit tests (3 new in `mcts_tree.rs`) + a Python two-move tree-reuse
regression test (`tests/test_dense_cnn_performance.py`) pass. Native extension is
built via `maturin build` (no venv on this machine — see build note below) and
the `.pyd` extracted into `packages/hexo_models/python/hexo_models/`.

## OPEN ISSUE (blocking) — intermittent native crash during selfplay

The trainer **self-terminates natively** roughly once per 1–2 epochs during
selfplay. Two confirmed occurrences:
- Run A (PID 47300): at the epoch 9→10 boundary.
- Run B (PID 23156): mid-epoch-11.

Signature: **no Python traceback, no Windows Error Reporting event, no crash
dump, no resource pressure** (watchdog `status: ok`, ~17 GB free RAM, GPU idle at
the time) and **not** a watchdog kill. Both crashed builds include Change 1, and
there is **no no-Change-1 baseline**, so Change 1 is a suspect but unproven — a
code audit of the new hot path (`materialize_next_candidate` is guarded by a
prior `peek_next_candidate`; `can_widen` bounds `edges.len() < max_eligible ≤
priors.len()`; `ensure_root_owned`/`remaining_priors` use `.min(len)`) found
**no unguarded panic**. The fault could be pre-existing (inference/engine).

### Instrumented run currently live
Run C is up with diagnostics enabled to make the next crash talk:
- **Trainer PID: 10672**, logs `runs/dense_cnn_model1_scratch_64/diagnostics/trainer.20260528_231854.{out,err}.log`
- Env set at launch (inherited via `Start-Process`):
  `PYTHONFAULTHANDLER=1` (dumps Python+native frame on SIGSEGV/SIGABRT to stderr),
  `PYTHONUNBUFFERED=1`, `RUST_BACKTRACE=full`.
- Restarts at **epoch 9** (config uses `[checkpoint] initialize_from` = bootstrap
  epoch 8, which always reloads the bootstrap → start at epoch 9). Latest saved
  checkpoint on disk is `epoch_000010.pt`.

### Next steps when it crashes
1. Read the tail of the `.err.log` — `PYTHONFAULTHANDLER` should print
   `Fatal Python error` / `Current thread ...` with the active Python frame
   (tells you whether it died in inference forward, the Rust MCTS step, or the
   engine). A Rust unwinding panic prints `thread '...' panicked at ...` +
   `stack backtrace:`.
2. If there is **no** faulthandler dump either, it's a hard crash (stack
   overflow / CUDA driver abort) — escalate to a real minidump by setting
   `HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\python.exe`
   (`DumpType=2`) and inspect the faulting module.
3. **Do NOT keep blindly relaunching.** Root-cause first. If Change 1 is
   implicated, the fix is in `mcts_tree.rs` / `mcts.rs`; rebuild per the note
   below.

## Monitoring routine (session-based — re-arm each session)

The routine is **session-only by design** (dies when the Claude session ends; it
is NOT a durable/disk-backed cron). To re-establish it in a new session:

1. **Recurring 30-min check-in** — create a session cron (`CronCreate`, NOT
   durable) firing at off-minutes, e.g. `17,47 * * * *`, with a prompt that:
   checks PID 10672 liveness, the latest checkpoint, current epoch + selfplay
   game count, and scans the `.err.log` for fault signatures; on crash, dumps and
   analyzes the captured stderr fault/backtrace and STOPS (no auto-relaunch).
2. **Real-time crash monitor** — a persistent `Monitor` that polls
   `powershell.exe Get-Process -Id 10672` for liveness, greps the `.err.log` for
   `Fatal Python error|Current thread|panicked|stack backtrace|SIGSEGV|SIGABRT`,
   flags new checkpoints, ~3-min heartbeat, and on exit dumps the last ~40 stderr
   lines. This catches the crash in real time (the cron is just the periodic
   check-in cadence on top).

If the PID has changed (relaunch), update both to the new PID and log stamp.

## Key facts / commands

- **Shell quirks:** the Bash tool is **git-bash** — repo is `/e/Hexo-BotTrainer`
  (NOT `/mnt/e/...`, which is WSL and does not exist here). Windows-native
  `C:\Python314\python.exe` cannot open `/e/...` paths — use `E:/...` in
  `python -c` file opens (stdin pipes are fine).
- **Build (no venv):**
  `C:\Python314\python.exe -m maturin build --release -m packages\hexo_models\Cargo.toml --features python`
  then extract `hexo_models/_rust.cp314-win_amd64.pyd` from
  `target/wheels/hexo_models-0.1.0-cp314-cp314-win_amd64.whl` into
  `packages/hexo_models/python/hexo_models/`. `maturin develop` FAILS (no venv).
- **PYTHONPATH for tests/runs:** the five package `python/` dirs
  (`hexo_models, hexo_train, hexo_runner, hexo_engine, hexo_utils`).
- **Launcher:** `scripts\start_model1_training.ps1 -ConfigPath configs\dense_cnn_model1_scratch_64.toml -SealBotPath E:\SealBot`
  (starts watchdog + trainer; inherits current shell env, so set the diagnostic
  env vars in the same PowerShell command).
- **Progress source of truth:** `runs/.../diagnostics/events.jsonl` (stage_started
  / stage_finished per epoch) and the per-game `selfplay/epoch_NNNNNN_game_*.npz`
  shards. The `.out.log` is empty (app logs to events.jsonl + stderr, not stdout).
- **Working state at handoff:** PID 10672 alive, redoing epoch 9, ~45 games in;
  SealBot eval shows `wins: 0` vs best-50ms (expected this early — watch game
  length per Goal #4, not just wins).
