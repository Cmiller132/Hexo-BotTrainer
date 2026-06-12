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
this package.

## Design: defaults plus plugin overrides

The package separates orchestration (owned here) from model semantics (owned
by plugins). `hexo_train` ships model-neutral defaults -- a shared
`hexo_utils.samples` store/index/window path, target helpers, a deterministic
D6 selector, and placeholder checkpoint writers -- and each plugin returns
`ComponentOverrides` to replace the pieces it owns. All four registered
plugins supply their own trainer, checkpoint loader/saver, and replay storage
(`uses_shared_sample_store=False`), so on production runs the epoch ordering,
diagnostics, and artifact layout come from this package while storage,
training, and checkpoint IO come from the plugin. The default implementations
are exercised by the package's FakePlugin pipeline tests.

## Module table

All paths relative to `packages/hexo_train/python/hexo_train/`.

| File | Role |
| --- | --- |
| `cli/train_model.py` | Thin argparse CLI -> `TrainingPipeline().run(config_path)`. The single public command (`python -m hexo_train.cli.train_model` / `hexo-train-model` console script). |
| `pipeline.py` | `TrainingPipeline`: the run "map". Fixed step sequence (initialize_run, load_checkpoint, calibrate_performance, run_epochs, publish_final_model, write_diagnostics), each wrapped in `_run_step` diagnostics. Teardown calls `trainer.close()` when present (e.g. restnet's expansion process pool). |
| `config.py` | TOML/YAML loading and normalization into frozen dataclasses (`ModelConfig`, `RunConfig`, `LoopConfig`, `SelfPlayConfig`, `SamplesConfig`, `TrainConfig`, `CheckpointConfig`, `TrainingConfig`). Rejects the removed `model_specific`/`stages` fields; resolves paths relative to the config dir. Typed sections cover the orchestration skeleton; `[model.config]` passes through opaquely to the plugin's own config module, and model-neutral extras like `[shared.game]` stay reachable via `TrainingConfig.raw` / `ctx.section()`. Every config in `configs/` is TOML. |
| `registry.py` | Plugin discovery: explicit module (`[model].module`), explicit entry point, or name lookup through the `hexo_train.models` entry point group. Defines the `ModelPlugin` Protocol covering the two construction hooks. |
| `context.py` | `RunContext`: creates `output/`, `checkpoints/`, `diagnostics/`, `samples/` dirs; holds the `DiagnosticsWriter`, `outputs` dict, `epoch_outputs` list; `ctx.section()` exposes raw config sections. |
| `components.py` | Dependency container: `SharedComponents` (mutable run state: sample store/window/symmetries/checkpoint_state), `ComponentOverrides` (what a plugin returns), `ModelComponents`; `build_model_components` merges defaults + overrides. Fields are intentionally loosely typed (`Any`); the contract is duck-typed. |
| `defaults.py` | `build_shared_components`: default target helpers (from `hexo_utils.samples`), `D6SymmetrySelector`, `CheckpointStore`, game spec from `[shared.game]`. |
| `checkpoints.py` | When checkpoints load/save: `load_or_initialize_checkpoint` (`resume_from`/`initialize_from` -> plugin loader), `save_epoch_checkpoint`/`save_final_checkpoint` (plugin saver, or a placeholder metadata file for plugins without one), per-epoch pointer publish; updates `shared.checkpoint_state` for the next selfplay. |
| `artifacts.py` | Durable run files: `write_run_manifest` (`manifest.json` -- read by the dashboard for lineage/arch), `publish_selfplay_checkpoint_pointer` (`selfplay_checkpoint.txt`), `write_final_diagnostics` (`run.completed.json`), `CheckpointStore` path/placeholder helper. |
| `diagnostics.py` | `DiagnosticsWriter`: append-only `diagnostics/events.jsonl` + per-stage `<step>.json`. The dashboard live-status view tails `events.jsonl`. |
| `symmetry.py` | Training-owned deterministic D6 augmentation selection (blake2b of `seed:epoch:sample-id`); `D6SymmetrySelector`, `SampleSymmetrySelection`. |
| `epoch/loop.py` | `run_epochs`/`run_epoch`: the fixed per-epoch order above; `_start_epoch` resumes from the loader's `{"status": "loaded", "epoch": N}` state (how main_4 fast-forwarded past seeded epochs). |
| `epoch/selfplay.py` | `generate_selfplay` dispatch: `plugin.generate_selfplay()` (implemented by all registered plugins) > `plugin.build_selfplay_request()` > placeholder payload; the result is stored on `shared.selfplay_result`. |
| `epoch/samples.py` | Sample window per epoch: `finalize_samples` (plugin `sample_finalizer` hook), then `select_training_samples` -- delegates to `trainer.select_training_samples` when the trainer provides it (the restnet KataGo-style shuffle over NPZ shards), else builds the shared `hexo_utils` index/window. `prepare_sample_store` opens the shared store at run start for plugins that use it. |
| `epoch/symmetry.py` | `select_epoch_symmetries`: applies the D6 selector to the current sample window and stores the `SampleSymmetrySelection` on shared state. A trainer may consume the full per-sample tuple or just `selection.seed`; the restnet trainer uses the seed and draws its own per-row D6 symmetries during shard expansion. |
| `epoch/training.py` | `train_passes`: calls `trainer.train_passes(passes, sample_window, sample_symmetries, ...)` or returns skipped. |
| `__init__.py` | Re-exports config dataclasses, `RunContext`, `TrainingPipeline`, `load_model_plugin`, D6 selector types. |

## Connections to other packages

Imports OUT (what this package uses):

- `hexo_utils.samples` (target helpers, sample store/index/window -- the
  shared-store path), `hexo_utils.encoding` (`D6_SIZE`, `D6Symmetry`).
- Declares `hexo-engine` and `hexo-runner` as deps; game execution is reached
  through `plugin.generate_selfplay`, which drives them inside the plugin --
  this package itself never imports the engine or runner.

Imports IN (who uses this package):

- Model plugins import `hexo_train.components.ComponentOverrides` and register
  under the `hexo_train.models` entry point group:
  - `dense_cnn_restnet` (`packages/dense_cnn_restnet/pyproject.toml`) -- ACTIVE
  - `dense_cnn`, `hexgt` (`packages/hexo_models/pyproject.toml`) -- legacy/halted
  - `hexgnn` (`packages/hexgnn/pyproject.toml`) -- parked
- Plugins may also be loaded by module path (`[model].module = "dense_cnn_restnet.plugin"`
  in `configs/dense_cnn_restnet_main_*.toml`), bypassing entry points.

Plugin/trainer contract (duck-typed; hooks dispatched via hasattr checks in
`pipeline.py` and `epoch/*.py`):

- Plugin hooks: `build_model`, `training_component_overrides`,
  `generate_selfplay`, optional `evaluate_epoch` and `calibrate_performance`;
  an optional `sample_finalizer` component (`.finalize()`).
- Trainer hooks: `select_training_samples`, `train_passes`, optional `close()`.
- Checkpoint loader/saver: `loader.load(ref, ...)` returns the state dict
  stored on `shared.checkpoint_state`; `{"status": "loaded", "epoch": N}`
  drives `epoch/loop._start_epoch` to resume at epoch N+1 (load-bearing for
  main_4's fast-forward from ckpt5 -- keep both sides of the dict shape in
  sync). `saver.save(name=...)` returns the checkpoint path.

File-format contracts (no Python import):

- `manifest.json` (from `artifacts.write_run_manifest`) -- read by
  `hexo_frontend/web.py` and `debug_infer.py` for lineage/arch detection.
- `diagnostics/events.jsonl` -- tailed by the dashboard's live training status.
- Plugins write `diagnostics/dense_cnn.selfplay.epoch_*.json` etc. through
  `ctx.diagnostics.write_json`; the dashboard and health scripts
  (`scripts/_wf_r4_health.py`, `_wf_r4_m4_gates.py`) read them.
- `selfplay_checkpoint.txt` / `data/checkpoints/*_latest.txt` pointer files,
  written per-epoch and at final publish when `update_checkpoint_pointer =
  true` (legacy dense_cnn/hexgt configs; all restnet `main_*` configs set it
  false).

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
   (`dense_cnn_restnet/replay.py`); the plugin owns its replay storage
   (`uses_shared_sample_store=False`).
3. **Train** -- `epoch/training.train_passes` calls the restnet trainer's
   `train_passes` (parallel shard expansion with per-row D6, AMP optimizer
   steps).
4. **Checkpoint + eval** -- `checkpoints.save_epoch_checkpoint` via the
   plugin saver; `plugin.evaluate_epoch` runs SealBot evaluation games.

**Frozen-win override (main_4, high level):** the restnet plugin's selfplay
includes an incremental standing-win tracker
(`dense_cnn_restnet/win_tracker.py` + `selfplay.py`) that overrides the chosen
move with the winning placement when a 6-in-a-row standing win sits outside
the net's radius-20 input crop (engine-verified on a cloned state before
playing). It lives entirely inside the plugin; hexo_train sees only the
`frozen_win_overrides` counters in the epoch diagnostics it records.

## Entry points / how it gets exercised

| Entry | Notes |
| --- | --- |
| `python -m hexo_train.cli.train_model <config>` / `hexo-train-model` | The sole public command. |
| `scripts/_dc_restnet_supervise_main1.sh` | ACTIVE supervisor for the restnet main_2/3/4 lineage (the "main1" name is historical; CONFIG/RUNDIR env overrides select the run). Launched detached by `scripts/_wf_r4_launch_main4.sh`. |
| `scripts/_dc_supervise_main1.sh`, `_dc_launch_main1.sh`, `_rl_supervise.sh`, `_rl_supervise_hexgnn.sh` | Per-model supervisors for the legacy lineages, using the same CLI. |
| `scripts/start_model1_training.ps1`, `scripts/run_model1_wsl_smoke.sh` | Windows/WSL launchers from the model1 era. |
| `tests/test_training_pipeline_simplification.py` | The package's dedicated test: config normalization, registry, full FakePlugin pipeline run, resume, D6 determinism. |
| `tests/test_dense_cnn_pipeline.py`, `test_dense_cnn_performance.py`, `test_dense_cnn_pool_lifecycle.py`, `test_hexgt_scaffold.py` | Drive `TrainingPipeline`/registry against real plugins (run under the WSL venv). |
