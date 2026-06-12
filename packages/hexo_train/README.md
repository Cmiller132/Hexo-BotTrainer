# hexo_train

Model-neutral, config-driven training orchestration for Hexo models. Given a
TOML config, it discovers a model plugin (via the `hexo_train.models` entry
point group or an explicit module path), then runs a fixed self-play training
lifecycle: initialize run artifacts -> load/initialize checkpoint -> calibrate
performance -> N epochs of (selfplay -> finalize -> sample window -> D6
symmetry -> train passes -> epoch checkpoint -> optional eval) -> final
checkpoint -> diagnostics.

**Status: ACTIVE.** The live `dense_cnn_restnet` main_4 run is launched as
`python -m hexo_train.cli.train_model configs/dense_cnn_restnet_main_4.toml`
by the WSL supervisor scripts. All four model lineages (dense_cnn_restnet,
hexo_models/dense_cnn, hexo_models/hexgt, hexgnn) register plugins against
this package. About a third of the package (the shared sample store, default
target helpers, placeholder checkpoint paths) is early scaffolding that every
real plugin opts out of -- see "Gotchas" below.

## Module table

All paths relative to `packages/hexo_train/python/hexo_train/`.

| File | Role |
| --- | --- |
| `cli/train_model.py` | Thin argparse CLI -> `TrainingPipeline().run(config_path)`. The single public command (`python -m hexo_train.cli.train_model` / `hexo-train-model` console script). |
| `pipeline.py` | `TrainingPipeline`: the run "map". Fixed step sequence (initialize_run, load_checkpoint, calibrate_performance, run_epochs, publish_final_model, write_diagnostics), each wrapped in `_run_step` diagnostics. Teardown calls `trainer.close()` if present (restnet's expansion process pool). |
| `config.py` | TOML/YAML loading and normalization into frozen dataclasses (`ModelConfig`, `RunConfig`, `LoopConfig`, `SelfPlayConfig`, `SamplesConfig`, `TrainConfig`, `CheckpointConfig`, `TrainingConfig`). Rejects the removed `model_specific`/`stages` fields; resolves paths relative to the config dir. |
| `registry.py` | Plugin discovery: explicit module (`[model].module`), explicit entry point, or name lookup through the `hexo_train.models` entry point group. Defines the (minimal) `ModelPlugin` Protocol. |
| `context.py` | `RunContext`: creates `output/`, `checkpoints/`, `diagnostics/`, `samples/` dirs; holds the `DiagnosticsWriter`, `outputs` dict, `epoch_outputs` list; `ctx.section()` raw-config escape hatch. |
| `components.py` | Dependency container: `SharedComponents` (mutable run state: sample store/window/symmetries/checkpoint_state), `ComponentOverrides` (what a plugin returns), `ModelComponents`; `build_model_components` merges defaults + overrides. Mostly `Any`-typed. |
| `defaults.py` | `build_shared_components`: default target helpers (from `hexo_utils.samples`), `D6SymmetrySelector`, `CheckpointStore`, game spec from `[shared.game]`. |
| `checkpoints.py` | When checkpoints load/save: `load_or_initialize_checkpoint` (`resume_from`/`initialize_from` -> plugin loader), `save_epoch_checkpoint`/`save_final_checkpoint` (plugin saver or placeholder), per-epoch pointer publish; updates `shared.checkpoint_state` for the next selfplay. |
| `artifacts.py` | Durable run files: `write_run_manifest` (`manifest.json` -- read by the dashboard for lineage/arch), `publish_selfplay_checkpoint_pointer` (`selfplay_checkpoint.txt`), `write_final_diagnostics` (`run.completed.json`), placeholder `CheckpointStore`. |
| `diagnostics.py` | `DiagnosticsWriter`: append-only `diagnostics/events.jsonl` + per-stage `<step>.json`. The dashboard live-status view tails `events.jsonl`. |
| `symmetry.py` | Training-owned deterministic D6 augmentation selection (blake2b of `seed:epoch:sample-id`); `D6SymmetrySelector`, `SampleSymmetrySelection`. |
| `epoch/loop.py` | `run_epochs`/`run_epoch`: the fixed per-epoch order above; `_start_epoch` resumes from the loader's `{"status": "loaded", "epoch": N}` state (this is how main_4 fast-forwards past seeded epochs). |
| `epoch/selfplay.py` | `generate_selfplay` dispatch: `plugin.generate_selfplay()` (all real plugins) > `plugin.build_selfplay_request()` (transitional, no implementers) > placeholder payload. |
| `epoch/samples.py` | `prepare_sample_store` (shared store; skipped by all real plugins), `finalize_samples` (plugin finalizer hook), `select_training_samples` (delegates to `trainer.select_training_samples` when present -- restnet's KataGo shuffle path -- else the shared `hexo_utils` window). |
| `epoch/symmetry.py` | `select_epoch_symmetries`: applies the D6 selector to the current sample window, stores the selection on shared state. |
| `epoch/training.py` | `train_passes`: calls `trainer.train_passes(passes, sample_window, sample_symmetries, ...)` or returns skipped. |
| `__init__.py` | Re-exports config dataclasses, `RunContext`, `TrainingPipeline`, `load_model_plugin`, D6 selector types. |

## Connections to other packages

Imports OUT (what this package uses):

- `hexo_utils.samples` (target helpers, sample store/index/window -- the
  shared-store path only), `hexo_utils.encoding` (`D6_SIZE`, `D6Symmetry`).
- Declares `hexo-engine` and `hexo-runner` as deps but never imports them
  directly; game execution is reached only through `plugin.generate_selfplay`.

Imports IN (who uses this package):

- Model plugins import `hexo_train.components.ComponentOverrides` and register
  under the `hexo_train.models` entry point group:
  - `dense_cnn_restnet` (`packages/dense_cnn_restnet/pyproject.toml`) -- ACTIVE
  - `dense_cnn`, `hexgt` (`packages/hexo_models/pyproject.toml`) -- legacy/halted
  - `hexgnn` (`packages/hexgnn/pyproject.toml`) -- parked
- Plugins may also be loaded by module path (`[model].module = "dense_cnn_restnet.plugin"`
  in `configs/dense_cnn_restnet_main_*.toml`), bypassing entry points.

Duck-typed plugin/trainer contract (convention, not types):

- Plugin hooks: `build_model`, `training_component_overrides`,
  `generate_selfplay`, optional `evaluate_epoch`, optional
  `calibrate_performance`, optional `finalize_samples`.
- Trainer hooks: `select_training_samples`, `train_passes`, optional `close()`.
- Checkpoint loader returns `{"status": "loaded", "epoch": N}` to drive
  `epoch/loop._start_epoch` resume (load-bearing for main_4's epoch
  fast-forward from ckpt5).

File-format contracts (no Python import):

- `manifest.json` (from `artifacts.write_run_manifest`) -- read by
  `hexo_frontend/web.py` and `debug_infer.py` for lineage/arch detection.
- `diagnostics/events.jsonl` -- tailed by the dashboard's live training status.
- Plugins write `diagnostics/dense_cnn.selfplay.epoch_*.json` etc. through
  `ctx.diagnostics.write_json`; the dashboard and health scripts
  (`scripts/_wf_r4_health.py`, `_wf_r4_m4_gates.py`) read them.
- `selfplay_checkpoint.txt` / `data/checkpoints/*_latest.txt` pointer files
  (only when `update_checkpoint_pointer = true`; all restnet `main_*` configs
  set it false).

## How the restnet selfplay -> replay -> train loop maps onto this package

Per epoch (epoch/loop.py order), with the `dense_cnn_restnet` plugin:

1. **Selfplay** -- `plugin.generate_selfplay` runs
   `dense_cnn_restnet/selfplay.py` (continuous scheduler over the shared Rust
   MCTS), which writes per-game compact NPZ shards + JSON sidecars under
   `<run>/selfplay/` and live/epoch diagnostics JSON. hexo_train sees only the
   summary dict.
2. **Replay/sample window** -- `epoch/samples.select_training_samples`
   delegates to the restnet trainer's `select_training_samples`, which builds
   a KataGo-style shuffle over the mtime-ordered NPZ shard window
   (`dense_cnn_restnet/replay.py`). The shared `hexo_utils` sample store is
   skipped entirely (`uses_shared_sample_store=False`).
3. **Train** -- `epoch/training.train_passes` calls the restnet trainer's
   `train_passes` (parallel shard expansion with per-row D6, AMP optimizer
   steps).
4. **Checkpoint + eval** -- `checkpoints.save_epoch_checkpoint` via the
   plugin saver; `plugin.evaluate_epoch` runs SealBot evaluation games.

**Frozen-win override (main_4, high level):** main_3 collapsed because the
radius-20 input crop can freeze a 6-in-a-row "standing win" outside the crop
rim, so the net never sees or plays it. The fix lives entirely inside the
restnet plugin's selfplay (`dense_cnn_restnet/win_tracker.py` +
`selfplay.py`): an incremental tracker spots standing-win cells during
self-play and overrides the move with the winning placement (engine-verified
on a cloned state before playing). hexo_train is unaware of it beyond the
`frozen_win_overrides` counters in the epoch diagnostics it records.

## Entry points / how it gets exercised

| Entry | Notes |
| --- | --- |
| `python -m hexo_train.cli.train_model <config>` / `hexo-train-model` | The sole public command. |
| `scripts/_dc_restnet_supervise_main1.sh` | ACTIVE supervisor for the restnet main_2/3/4 lineage (generic despite the "main1" name; CONFIG/RUNDIR env overrides). Launched detached by `scripts/_wf_r4_launch_main4.sh`. |
| `scripts/_dc_supervise_main1.sh`, `_dc_launch_main1.sh`, `_rl_supervise.sh`, `_rl_supervise_hexgnn.sh` | Older per-model supervisors using the same CLI (legacy lineages). |
| `scripts/start_model1_training.ps1`, `scripts/run_model1_wsl_smoke.sh` | Windows/WSL launchers from the model1 era. |
| `tests/test_training_pipeline_simplification.py` | The package's dedicated test: config normalization, registry, full FakePlugin pipeline run, resume, D6 determinism. |
| `tests/test_dense_cnn_pipeline.py`, `test_dense_cnn_performance.py`, `test_dense_cnn_pool_lifecycle.py`, `test_hexgt_scaffold.py` | Drive `TrainingPipeline`/registry against real plugins (authoritative only in the WSL venv). |

## Gotchas

- **Shared sample store is live-but-bypassed scaffolding.** All four real
  plugins set `uses_shared_sample_store=False`, so `prepare_sample_store` and
  the default `select_training_samples`/`finalize_samples` branches only run
  for the FakePlugin tests. Do not assume it reflects production behavior.
- **Vestigial D6 work on the active lineage.** `epoch/symmetry.py` computes a
  per-sample symmetry tuple every epoch, but the restnet trainer consumes only
  `sample_symmetries.seed` and re-draws its own per-row symmetries. The
  `symmetry_count` diagnostic is misleading for restnet runs.
- **The real plugin contract is convention, not types.** `components.py` is
  nearly all `Any`; `registry.ModelPlugin` covers only 2 of the ~7 hooks
  actually used. The loader's `{"status": "loaded", "epoch": N}` resume shape
  (epoch/loop.py `_start_epoch`) is an implicit dict contract that main_4
  depends on.
- **`ctx.section()` escape hatch.** Most real configuration (the big
  `[model.config]` blocks, `[samples]`, `[shared.game]`) bypasses the typed
  config boundary; `config.py` validation covers only the orchestration
  skeleton.
- **Placeholder branches look like production code.** The "checkpoint loading
  is not implemented yet" / placeholder-saver paths in `checkpoints.py` and
  `artifacts.CheckpointStore.write_placeholder` only fire for a plugin with no
  loader/saver, which no real plugin is.
- **Pointer publishing is duplicated** between `artifacts.py` (final) and
  `checkpoints.py` (per-epoch) and is only used by legacy dense_cnn/hexgt
  configs.
- **YAML config support has no caller.** `config.py` accepts YAML (hence the
  PyYAML dep), but every config in `configs/` and every launcher emits TOML.
- Stray committed `__pycache__` directories exist under `python/hexo_train/`
  in the working tree.
