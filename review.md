# Exhaustive Code Review Reverification

Review date: 2026-06-01

## Scope

This is a second, read-only verification pass over the earlier `review.md` findings. Source, tests, configs, and scripts were inspected, but no code/config/test changes were made by this pass. This file is the only intended edit.

Current repository state at the time of this update:

- Tracked files reviewed: 451.
- Non-ignored untracked files reviewed: `analysis/_deep_model_analysis.py`, `analysis/_deep_sample_review.py`, `analysis/_dma_results.txt`, `analysis/_eval_game_peek.py`, `analysis/_partial_spill_check.py`, `analysis/_quality_extract.py`, `analysis/_verify_readvalue.py`, `analysis/_wait_to_epoch10.sh`, and `review.md`.
- Current tracked modification outside this report: `configs/dense_cnn_model1_target_96x8.toml`, where `resume_from` moved from `epoch_000002.pt` to `epoch_000007.pt`.
- Five read-only subagents rechecked independent slices: Rust/core, model/training, runner/frontend, config/script/docs/artifacts, and analysis/tests.

Status summary:

- Fully fixed prior findings: 0.
- Partially fixed prior findings: F-02, F-10, F-11, F-14, F-18.
- Changed but still open prior findings: F-17, F-32.
- Still open prior findings: all other original findings.
- New findings from current untracked files: F-34, F-35.

## Verification Commands

- `cargo test --workspace`: passed. Rust tests still pass: 21 engine tests and 8 utility tests.
- `python -m pytest tests -q`: failed with 9 failures and 171 passes. The failures still come from the stale Dense CNN native extension signature mismatch: `Model1MctsSession.search() takes from 7 to 17 positional arguments but 19 were given`.
- `python -m pytest -q`: failed during collection. `analysis/throughput_understanding/_trt_failloud_test.py` is still collected and still executes TensorRT/CUDA setup at import time.
- Native extension inspection still shows `packages/hexo_models/python/hexo_models/_rust.cp314-win_amd64.pyd` exposes `Model1MctsSession.search(... widening_min_children=None)` without `forced_playout_k` or `move_temperatures`.

## Current Status Table

| ID | Severity | Status | Short result |
|---|---:|---|---|
| F-01 | High | Open | Hexformer still finalizes sample-budget-truncated games with `winner=None`. |
| F-02 | High | Partially fixed | Batch now aggregates per-game aborts, but setup failures still escape cleanup/result contract. |
| F-03 | High | Open | Stale ignored Dense CNN native extension still breaks tests and no ABI freshness gate exists. |
| F-04 | High | Open | 64x4 wipe script still deletes run dir while current config resumes from epoch 40. |
| F-05 | High | Open | Supervisors still mutate tracked configs in place. |
| F-06 | Medium | Open | Top-level pytest still collects import-time TensorRT script. |
| F-07 | Medium | Open | Engine coordinate math still uses unchecked `i16` arithmetic. |
| F-08 | Medium | Open | `.hxr` reader still allocates from unbounded declared lengths. |
| F-09 | Medium | Open | Hexformer Rust evaluator still zero-fills malformed byte outputs. |
| F-10 | Medium | Partially fixed | Runtime comments improved, but config comments still claim fallback while default is fail-loud. |
| F-11 | Medium | Partially fixed | Compact shard path now has D6 guard, but raw expansion remains unsafe for direct callers. |
| F-12 | Medium | Open | Frontend still hides `.ckpt` and non-epoch `.pt` artifacts. |
| F-13 | Medium | Open | `checkpoint.save_name` is still path-joined without filename validation. |
| F-14 | Medium | Partially fixed | Dense CNN current path has override; generic opt-out model path still falls through. |
| F-15 | Medium | Open | Frontend still times out generically on pre-state abort results. |
| F-16 | Medium | Open | SealBot startup timeout still can leak subprocess. |
| F-17 | Medium | Changed/Open | 96x8 resume path advanced to epoch 7; contradiction remains. |
| F-18 | Medium | Partially fixed | Some wait scripts exit nonzero, but several failure monitors still exit success. |
| F-19 | Medium | Open | `_smoke_optimized_test.toml` still says eval off but requires SealBot eval. |
| F-20 | Medium | Open | Resource watchdog still matches/kills broad first training process. |
| F-21 | Medium | Open | 71 files under ignored `runs/` are still tracked. |
| F-22 | Medium | Open | README still links missing `docs/structure/*.md` files. |
| F-23 | Medium | Open | MCTS microbench still records `roots=256` for 128 games. |
| F-24 | Medium | Open | Routine prompt still points at superseded `scratch_64` run. |
| F-25 | Medium | Open | `hexo-models` still omits direct `numpy` and `hexo-utils` dependencies. |
| F-26 | Low | Open | `begin_game(..., scenario=None)` is still rejected. |
| F-27 | Low | Open | Initial UI still shows manual-vs-SealBot while backend starts manual-vs-manual. |
| F-28 | Low | Open | `.claude/launch.json` still hard-codes local paths and binds `0.0.0.0`. |
| F-29 | Low | Open | `_status_now.txt` remains tracked volatile status. |
| F-30 | Low | Open | Aligned MCTS diff still crashes on zero matching positions. |
| F-31 | Low | Open | `_deep_sample_review.py` still hard-codes local run and crashes on empty input. |
| F-32 | Low | Changed/Open | Heavy top-level repro now allocates 4,000,000 rows per dtype/peak. |
| F-33 | Low | Open | Value-clamp test still asserts source text instead of behavior. |
| F-34 | Low | New/Open | `_wait_to_epoch10.sh` monitors 96x8 but uses global `pgrep -f train_model`. |
| F-35 | Low | New/Open | New untracked analysis scripts are hard-coded to one local run and execute at top level. |

## Detailed Findings

### F-01 High: Hexformer self-play can label unfinished sample-budget games as draws

Status: Open.

Evidence:

- `packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/selfplay.py:35` sets `target_samples`.
- `packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/selfplay.py:106` still stops a game when `samples_added + len(pending) < target_samples` becomes false.
- `packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/selfplay.py:148` still finalizes pending samples unconditionally.
- `packages/hexo_models/hexformer_ar/rust/src/sample_gen.rs:413`, `:431`, and `:441` still map `winner=None` into draw-like value, lookahead, and distance targets.

Training impact: unfinished positions can still be taught as value zero/draw-like positions. This directly corrupts Hexformer value targets.

### F-02 High: Batch setup failures escape the runner result contract and skip cleanup

Status: Partially fixed.

Verified improvement:

- `packages/hexo_runner/python/hexo_runner/modes/batch.py:60`, `:64`, and `:75` now aggregate per-game results/aborts in batch output.
- `packages/hexo_runner/python/hexo_runner/modes/batch.py:132` and `:136` still close the record file and players once the protected loop is entered.

Still open:

- `packages/hexo_runner/python/hexo_runner/modes/batch.py:107` creates players before the `try/finally`.
- `packages/hexo_runner/python/hexo_runner/modes/batch.py:113` runs `player.setup_worker()` before the `try/finally`.
- `packages/hexo_runner/python/hexo_runner/modes/batch.py:58` still lets `pool.map()` worker exceptions escape rather than converting them into a structured `BatchResult`.

Training impact: in-game abort reporting is better, but setup/create failures can still leak partially initialized players and abort a batch job outside the runner result contract.

### F-03 High: Current tests are broken by an ignored stale Dense CNN native extension

Status: Open.

Evidence:

- Python still forwards `forced_playout_k` and `move_temperatures` at `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/rust_bridge.py:63` and `:92`.
- Rust source still expects those parameters at `packages/hexo_models/dense_cnn/rust/src/mcts.rs:85` and `:106`.
- `_dense_cnn_module()` only checks module presence at `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/rust_bridge.py:115`; there is still no ABI/signature freshness gate.
- `packages/hexo_models/dense_cnn/rust/src/lib.rs:27` capabilities still do not include a concrete ABI/signature version.
- Local verification still loads `packages/hexo_models/python/hexo_models/_rust.cp314-win_amd64.pyd`, whose search signature lacks the two newer parameters.

Training impact: MCTS calibration/self-play can fail depending on the local ignored build artifact. The test suite still fails for this reason.

### F-04 High: The 64x4 wipe script can delete valuable checkpoints

Status: Open.

Evidence:

- `scripts/_wipe_and_relaunch_64x4.sh:13` still says there are no checkpoints worth keeping.
- `scripts/_wipe_and_relaunch_64x4.sh:14` still runs `rm -rf "$RD"`.
- `configs/dense_cnn_model1_target_64x4.toml:131` still resumes from `epoch_000040.pt`.

Training impact: a normal operator action can erase active 64x4 checkpoint history and run data.

### F-05 High: Supervisors mutate tracked configs with local resume paths

Status: Open.

Evidence:

- `scripts/supervise_target_64x4_wsl.sh:10` defaults `CONFIG` to the tracked config.
- `scripts/supervise_target_64x4_wsl.sh:50` writes that config with `cfg.write_text(t)`.
- `scripts/supervise_target_64x4_wsl.sh:69` calls `set_resume`.
- The same pattern remains in the 96x8 WSL supervisor and PowerShell supervisors.

Training impact: tracked configs still become local resume-state files. Reproducing a "fresh" run remains unreliable.

### F-06 Medium: Top-level pytest collection executes a TensorRT script at import time

Status: Open.

Evidence:

- `analysis/throughput_understanding/_trt_failloud_test.py` still matches pytest default `*_test.py` discovery.
- There is still no root pytest config/conftest limiting collection to `tests/`.
- The file still loads a checkpoint at `analysis/throughput_understanding/_trt_failloud_test.py:12`, moves the model to CUDA at `:13`, and calls TensorRT build at `:15`, `:19`, and `:24`.
- `python -m pytest -q` still fails during collection with `TRTAdoptError`.

Training impact: the top-level test command still cannot be used as a reliable safety check before running training.

### F-07 Medium: Coordinate math can overflow near the advertised infinite-board edge

Status: Open.

Evidence:

- `packages/hexo_engine/rust/src/coord.rs:13` still stores coordinate components as `i16`.
- `packages/hexo_engine/rust/src/coord.rs:28`, `:35`, `:47`, `:77`, and `:92` still perform direct `i16` arithmetic.
- No direct overflow test was found for these coordinate helpers.

Training impact: normal games likely never reach the edge, but pathological long games can still panic in debug builds or wrap in release builds, corrupting legal move/window updates.

### F-08 Medium: Malformed `.hxr` files can trigger unbounded allocation before validation

Status: Open.

Evidence:

- `packages/hexo_utils/rust/src/records.rs:514` reads unbounded `player_count`.
- `packages/hexo_utils/rust/src/records.rs:515` allocates `Vec::with_capacity(player_count)`.
- `packages/hexo_utils/rust/src/records.rs:555` reads unbounded game payload length.
- `packages/hexo_utils/rust/src/records.rs:556` allocates `vec![0; length]`.
- `packages/hexo_utils/rust/src/records.rs:590` reads unbounded action count.
- `packages/hexo_utils/rust/src/records.rs:591` allocates that capacity.
- `packages/hexo_utils/rust/src/records.rs:739` only rejects lengths that do not fit `usize`.

Training impact: malformed training/eval record files can still force huge allocations before returning a clean parse error.

### F-09 Medium: Hexformer Rust MCTS evaluator zero-fills malformed byte outputs

Status: Open.

Evidence:

- `packages/hexo_models/hexformer_ar/rust/src/mcts_eval.rs:151` still uses `read_f32(...).unwrap_or(0.0)` for `values_bytes`.
- `packages/hexo_models/hexformer_ar/rust/src/mcts_eval.rs:187` still uses `unwrap_or(0)` for candidate bytes.
- `packages/hexo_models/hexformer_ar/rust/src/mcts_eval.rs:218` still uses `read_f32(...).unwrap_or(0.0)` for `priors_bytes`.
- `packages/hexo_models/hexformer_ar/rust/src/mcts_eval.rs:239` still returns `Option<f32>` instead of an exact-length error.

Training impact: malformed evaluator output can still become fake zero values/priors, distorting MCTS decisions and generated visit-policy targets.

### F-10 Medium: Dense CNN TensorRT fallback comments contradict default behavior

Status: Partially fixed.

Verified improvement:

- Runtime comments now describe fail-loud behavior in `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/inference.py:121`.
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/trt_backend.py:292` also documents the explicit fallback flag.

Still open:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/config.py:168` still has stale fallback-oriented wording.
- `configs/dense_cnn_model1_target_96x8.toml:120` and `:123` still claim automatic/clean fallback.
- `configs/dense_cnn_model1_target_96x6.toml:118` and nearby comments still claim automatic fallback.
- Actual default remains fail-loud at `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/config.py:180` and `trt_backend.py:310`.

Training impact: configs can still fail on non-TRT hosts even where comments tell operators fallback is expected.

### F-11 Medium: Raw Dense CNN D6 expansion remains unsafe for direct callers

Status: Partially fixed.

Verified improvement:

- The production compact-shard path now documents and checks D6 spillover at `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/compact_io.py:276`.
- It falls back to identity through `symmetry_drops_support()` at `compact_io.py:290`.
- Tests explicitly verify the compact-shard guard behavior in `tests/test_dense_cnn_compact_io.py:286`, `:319`, and `:344`.

Still open:

- Raw direct expansion still drops out-of-crop facts in `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/input.py:78`, `:159`, and `:168`.

Training impact: the main compact-shard training path is protected, but direct callers and analysis scripts can still silently truncate transformed targets.

### F-12 Medium: Frontend hides default `.ckpt` checkpoint artifacts

Status: Open.

Evidence:

- `packages/hexo_frontend/python/hexo_frontend/web.py:45` allows only `.json`, `.jsonl`, `.png`, and `.hxr`.
- `packages/hexo_frontend/python/hexo_frontend/web.py:839` drops files outside that suffix set.
- `packages/hexo_frontend/python/hexo_frontend/web.py:1207` scans epoch history only as `epoch_*.pt`.

Training impact: default `.ckpt` files and non-epoch `.pt` artifacts can exist but remain invisible in the dashboard.

### F-13 Medium: `checkpoint.save_name` allows path traversal within output handling

Status: Open.

Evidence:

- `packages/hexo_train/python/hexo_train/config.py:223` still accepts raw `save_name`.
- `packages/hexo_train/python/hexo_train/checkpoints.py:76` passes it directly to final checkpoint save.
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/checkpoints.py:86` joins it into `ctx.checkpoint_dir / f"{name}.pt"`.
- `packages/hexo_train/python/hexo_train/artifacts.py:34` does the same for `.ckpt`.

Training impact: a bad config can still write final checkpoints outside the intended checkpoint directory.

### F-14 Medium: `uses_shared_sample_store=False` can still fall through to shared sample indexing

Status: Partially fixed.

Verified improvement:

- Dense CNN opts out at `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/plugin.py:75` and has its own selector at `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/trainer.py:113`.
- Generic init/finalization skip shared store setup when `uses_shared_sample_store` is false at `packages/hexo_train/python/hexo_train/pipeline.py:120` and `packages/hexo_train/python/hexo_train/epoch/samples.py:60`.

Still open:

- `packages/hexo_train/python/hexo_train/epoch/samples.py:88` only checks for a trainer override.
- `packages/hexo_train/python/hexo_train/epoch/samples.py:94` still falls through to `refresh_sample_index(components.shared.sample_store)`.

Training impact: current Dense CNN is covered, but another opt-out model without a selector will still fail later in sample selection.

### F-15 Medium: Frontend reports a generic timeout when a match aborts before first state

Status: Open.

Evidence:

- `packages/hexo_frontend/python/hexo_frontend/web.py:198` still waits for the initial state during reset.
- Runner aborts are structured as `GameResult` in `packages/hexo_runner/python/hexo_runner/loop.py:137` and `:174`.
- The frontend stores the result at `packages/hexo_frontend/python/hexo_frontend/web.py:322`.
- `_wait_for_state_locked()` still only checks `_python_state` and `_error` at `web.py:395`, then raises the generic timeout at `web.py:398`.

Training impact: this mainly affects manual/eval UX and debugging, not model weights directly. Pre-state runner failures can still hide their real cause.

### F-16 Medium: SealBot startup timeout can leak the subprocess

Status: Open.

Evidence:

- `packages/hexo_runner/python/hexo_runner/adapters/sealbot.py:181` starts the process.
- `sealbot.py:190` starts reader threads.
- `sealbot.py:194` calls `_read_response`.
- `sealbot.py:243` can raise `TimeoutError`.
- `sealbot.py:195` only calls `close()` after a returned non-ok response.

Training impact: evaluation/bootstrap jobs can still leave orphan SealBot workers after startup timeouts.

### F-17 Medium: "fresh/cold start" configs contradict executable values

Status: Changed/Open.

Evidence:

- `configs/dense_cnn_model1_target_96x8.toml` is currently modified.
- The diff changes `resume_from` from `epoch_000002.pt` to `epoch_000007.pt`.
- Working tree has `resume_from` at `configs/dense_cnn_model1_target_96x8.toml:149`, followed by "Fresh run: no resume_from" at `:150`.
- The same contradiction remains in `configs/dense_cnn_model1_target_96x6.toml:164` versus `:165`.
- The same contradiction remains in `configs/dense_cnn_model1_target_64x4.toml:131` versus `:132-135`.

Training impact: the issue is not fixed; the 96x8 config now proves the tracked config is still being used as live resume state.

### F-18 Medium: Several operational scripts report failures but exit success

Status: Partially fixed.

Verified improvement:

- Some newer wait scripts now exit nonzero, for example `scripts/_wait_epoch1_boundary.sh:32-33` and `scripts/_wait_k2_games.sh:22`.

Still open:

- `scripts/_mem_breakdown.sh:6` prints `TRAINER NOT ALIVE` then exits 0.
- `scripts/watch_scratch64_crash.sh:85` treats process death as capture-and-exit, and `:81` exits 0.
- Prior examples such as `scripts/_smoke_tempdecay.sh:24-28`, `_bootstrap_64x4.sh:15-21`, `_bootstrap_96x8.sh:15-21`, `_rebuild_hexo_models.sh:10-11`, and `_rebuild_hexo_models_clean.sh:10-11` still print/capture status without reliably exiting with it.

Training impact: automation can still mark failed or unhealthy runs as successful in some script paths.

### F-19 Medium: `_smoke_optimized_test.toml` contradicts itself about SealBot evaluation

Status: Open.

Evidence:

- `configs/_smoke_optimized_test.toml:3-4` still says eval is off and SealBot is unavailable under WSL.
- `configs/_smoke_optimized_test.toml:66-72` still configures evaluation with `require_sealbot = true`.
- `configs/_smoke_optimized_test.toml:84-85` still says the smoke includes SealBot eval.

Training impact: the smoke config can still fail in the scenario its header claims it supports.

### F-20 Medium: Resource watchdog can kill the wrong training process

Status: Open.

Evidence:

- `scripts/watch_model1_resources.ps1:45` still finds a trainer through a broad process query.
- `scripts/watch_model1_resources.ps1:49` matches any `hexo_train.cli.train_model` or `dense_cnn_model1.toml`.
- `scripts/watch_model1_resources.ps1:51` selects the first match.
- `scripts/watch_model1_resources.ps1:105` kills that PID.

Training impact: one watchdog can still terminate an unrelated training job.

### F-21 Medium: Generated `runs/` artifacts are tracked despite being ignored

Status: Open.

Evidence:

- `.gitignore:12` still ignores `runs/`.
- `git ls-files runs` still returns 71 tracked files, including a `.pt` checkpoint, `.hxr` files, logs, PNGs, and diagnostics.

Training impact: source control still contains local experiment state and stale diagnostics, making it harder to distinguish canonical code from run artifacts.

### F-22 Medium: README links to source-of-truth docs that do not exist

Status: Open.

Evidence:

- `README.md:7-12` still links six `docs/structure/*.md` files.
- `docs/` is absent and `git ls-files docs` is empty.

Training impact: onboarding and architecture review remain brittle, which increases the chance of operational mistakes.

### F-23 Medium: MCTS sims-scaling benchmark reports the wrong root count

Status: Open.

Evidence:

- `analysis/mcts_microbench.py:129` creates `fresh_games(128, ...)`.
- `analysis/mcts_microbench.py:148` still records `"roots": 256`.
- `analysis/mcts_microbench.py:88` uses `active_root_limit=max(256, len(pg))`, but that is not the number of roots searched.

Training impact: performance reports can still overstate root parallelism by 2x.

### F-24 Medium: Recurring operational prompt points at the superseded run

Status: Open.

Evidence:

- `ROUTINE_PROMPT.md:1-2` still describes `scratch_64` on branch `rust-rebuild`.
- `ROUTINE_PROMPT.md:11` still points at `scripts\supervise_scratch64.ps1`.
- `HANDOFF.md:1` says the current handoff is `target_64x4`.
- `HANDOFF.md:6-7` says the old `rust-rebuild`/`scratch_64` handoff is superseded.

Training impact: a scheduled watcher can still monitor or relaunch the wrong run.

### F-25 Medium: `hexo-models` package metadata omits direct runtime dependencies

Status: Open.

Evidence:

- `packages/hexo_models/pyproject.toml:10-15` still lists only `torch`, `hexo-engine`, `hexo-runner`, and `hexo-train`.
- Direct imports remain in `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/compact_io.py:25`, `dense_cnn/trainer.py:17`, `hexformer_ar/selfplay.py:14`, and `hexformer_ar/samples.py:14`.

Training impact: standalone package installs can still miss `numpy` or `hexo-utils`.

### F-26 Low: `begin_game(..., scenario=None)` is rejected despite guidance saying to use it

Status: Open.

Evidence:

- `packages/hexo_utils/rust/src/pybridge.rs:278` still accepts `**kwargs`.
- `packages/hexo_utils/rust/src/pybridge.rs:285-289` still rejects whenever `scenario` is present without checking whether the value is `None`.

Training impact: low direct training impact. It remains an API contract trap for optional kwargs forwarding.

### F-27 Low: Initial frontend match config disagrees with backend default match

Status: Open.

Evidence:

- Backend starts manual/manual at `packages/hexo_frontend/python/hexo_frontend/web.py:165-172`.
- Empty config normalizes to manual/manual at `web.py:379`.
- Frontend default config is P0 manual, P1 `sealbot-current` at `packages/hexo_frontend/python/hexo_frontend/static/app.js:47`.
- HTML selects P1 SealBot current at `packages/hexo_frontend/python/hexo_frontend/static/index.html:87`.

Training impact: debugging/eval UI can show the wrong initial player setup.

### F-28 Low: `.claude/launch.json` is machine-specific and binds to all interfaces

Status: Open.

Evidence:

- `.claude/launch.json:10` still contains `E:\Hexo-BotTrainer`, `C:\Python314\python.exe`, `E:/SealBot`, and `--host 0.0.0.0`.

Training impact: local dashboard launch remains nonportable and externally exposed by default.

### F-29 Low: `_status_now.txt` is a tracked volatile status snapshot

Status: Open.

Evidence:

- `_status_now.txt` is still tracked.
- `_status_now.txt:1` still contains `3477 MiB, 12282 MiB, 0 %`.

Training impact: stale status can mislead operational review and creates avoidable repository churn.

### F-30 Low: Aligned MCTS diff crashes on zero matching positions

Status: Open.

Evidence:

- `analysis/mcts_aligned_diff.py:22` can leave `tvs` empty.
- `analysis/mcts_aligned_diff.py:33` guards only mean calculation.
- `analysis/mcts_aligned_diff.py:37-38` still index `tvs` unconditionally.

Training impact: analysis tooling still fails with `IndexError` instead of producing a useful zero-comparison diagnostic.

### F-31 Low: Untracked deep sample review is hard-coded to one local run and crashes on empty input

Status: Open.

Evidence:

- `analysis/_deep_sample_review.py:25` still hard-codes `/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8`.
- `analysis/_deep_sample_review.py:121-124` still calls `min`, `max`, and `median` on accumulators without a non-empty guard.

Training impact: a review script intended to validate samples can still crash or silently inspect the wrong run.

### F-32 Low: Untracked read-value repro script runs heavy allocation at top level

Status: Changed/Open.

Evidence:

- `analysis/_verify_readvalue.py:19` now sets `n = 4_000_000` per dtype/peak/device, not 3,000,000.
- `analysis/_verify_readvalue.py:20-21` still allocates large tensors at top level.
- There is still no `if __name__ == "__main__":` guard.

Training impact: accidental execution can consume significant memory/compute. This is a tooling reliability issue, not a model-training data issue.

### F-33 Low: Untracked value-clamp test asserts source text instead of behavior

Status: Open.

Evidence:

- `tests/test_dense_cnn_value_clamp.py:16` imports `inspect`.
- `tests/test_dense_cnn_value_clamp.py:38` calls `inspect.getsource`.
- `tests/test_dense_cnn_value_clamp.py:39-40` asserts literal strings.

Training impact: the test can pass for the wrong reason or fail on equivalent correct implementations. It does not prove evaluator bytes are clamped at the Rust boundary.

### F-34 Low: `_wait_to_epoch10.sh` uses a global trainer process check

Status: New/Open.

Evidence:

- `analysis/_wait_to_epoch10.sh:12` scopes the script to the 96x8 run directory.
- `analysis/_wait_to_epoch10.sh:26` checks liveness using global `pgrep -f train_model`.

Training impact: any unrelated trainer process can suppress a DEAD alert for the watched 96x8 run, making the monitor report false health.

### F-35 Low: New untracked analysis scripts are local-run, top-level tools

Status: New/Open.

Evidence:

- `analysis/_eval_game_peek.py:9` hard-codes `/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8`.
- `analysis/_eval_game_peek.py:56` executes peeks at top level.
- `analysis/_quality_extract.py:7` hard-codes the same local run diagnostics path.
- `analysis/_quality_extract.py:20` starts top-level file scanning.

Training impact: if these are committed as reusable tools, they will inspect one machine-specific run and execute on import. `analysis/_deep_model_analysis.py` is better guarded with an empty-input guard and `main` guard, so it is not included in this finding.

## Verified Positive Changes

- Dense CNN compact-shard expansion now has a D6 support-drop guard and tests for partial policy, stone, and opponent-policy spill cases.
- Dense CNN inference now clamps decoded value output before bytes are sent toward Rust: `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/inference.py:302`.
- Batch mode now aggregates per-game aborts/results more explicitly, though setup failures are still outside the protected section.
- Dense CNN's current non-shared sample-store model path has a trainer-owned selector; the remaining issue is the generic framework fallback.
- Some newer wait scripts now exit nonzero on timeout/failure.

## Rejected or De-scoped Suspicious Items

- `analysis/_deep_model_analysis.py` is local-run-specific, but it has a no-position guard and `main` guard, so it is not grouped with the top-level execution findings.
- `analysis/_dma_results.txt` is a zero-byte untracked artifact. It is repository hygiene noise, but not a substantive code finding.
- The earlier false positives remain rejected: PyCapsule state API lifetime, terminal phase/current-player behavior, legal move incremental removal, Dense CNN f16/f32 ABI, dense checkpoint strictness, Hexformer sparse edge rebasing, frontend path traversal checks, and `scripts/_run_all_tests.sh` / `_run_dense_cnn_tests.sh` pipefail behavior.

## Updated Remediation Order

1. Fix F-01, the Hexformer sample-budget truncation target corruption.
2. Fix F-03, the Dense CNN native ABI freshness/rebuild guard, so `python -m pytest tests -q` can pass.
3. Fix F-02 and F-16, runner setup cleanup/result conversion and SealBot startup cleanup.
4. Stop supervisor config mutation and remove transient `resume_from` from tracked configs: F-05 and F-17.
5. Add guardrails to destructive scripts and process watchdogs: F-04, F-18, F-20, F-34.
6. Fix pytest discovery/import-time execution: F-06 plus the top-level analysis tools in F-35.
7. Remove tracked generated run artifacts and stale status files: F-21 and F-29.
8. Clean up medium/low framework, dashboard, packaging, and benchmark issues: F-07 through F-15, F-19, F-22 through F-33.
