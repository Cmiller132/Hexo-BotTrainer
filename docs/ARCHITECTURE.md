# Hexo-BotTrainer Architecture: Cross-Package Data Flow

Status date: 2026-07-09. The live training now runs the **hexfield** lineages
from this working tree under WSL (`/mnt/e/Hexo-BotTrainer-hexgt`): the `hexfield`
`main_9` self-play run and the `hexfield_eq` D6-equivariant rewrite (configs
`hexfield_main_9.toml` / `hexfield_eq_main_1.toml`, supervised by
`scripts/systemd/hexfield-*.service`). Run artifacts live on a *different* mount
root (`/mnt/e/Hexo-BotTrainer/runs/...`, i.e. `E:\Hexo-BotTrainer\runs`).

**Scope of this document.** Sections 2-7 trace the end-to-end data flow of the
`dense_cnn_restnet` lineage in detail. That lineage is now **parked** (no live
run), but its source remains in-tree — it still backs the dashboard's legacy-
checkpoint debug worker — and it shares the same `hexo_train` / `hexo_engine` /
`hexo_runner` / `hexo_utils` orchestration and cross-package contracts the
hexfield lineages train on, so the flow here is the reference model for how any
lineage trains. The hexfield lineages differ mainly in the model package
(`packages/hexfield*`, each with its own Rust cdylib `hexfield*._rust`) and their
supervisor scripts (`scripts/_hexfield_supervise_main1.sh` /
`_hexfield_eq_supervise_main1.sh`).

## 1. Package status

| Package | Path | Status |
|---|---|---|
| `dense_cnn_restnet` | `packages/dense_cnn_restnet` | **PARKED** model lineage (the one this document traces; no live run). Pure Python/PyTorch; ships no Rust of its own. Still loaded by the dashboard debug worker and as the legacy-shard oracle adapter. |
| `hexo_train` | `packages/hexo_train` | **ACTIVE** orchestration harness (the CLI that drives every lineage — dense_cnn_restnet and the hexfield lineages). Its shared sample-store / placeholder-checkpoint paths are early scaffolding that every real plugin bypasses. |
| `hexo_engine` | `packages/hexo_engine` | **ACTIVE** authoritative rules engine (Rust + PyO3 `hexo_engine._rust`). Used by everything. |
| `hexo_runner` | `packages/hexo_runner` | **ACTIVE** core (player contracts, match loop, `.hxr` record facade, SealBot adapter). Its `batch`/`evaluation` modes are bypassed in practice. |
| `hexo_utils` | `packages/hexo_utils` | **ACTIVE** for the `.hxr` codec and Rust `state_hash`; the JSON sample store (`samples/`) is unused scaffolding. |
| `hexo_frontend` | `packages/hexo_frontend` | **ACTIVE** dashboard (:8080 in WSL). Match/History/Debug screens; carries large uncommitted v2 changes. |
| `hexo_models/dense_cnn` | `packages/hexo_models/dense_cnn` | Split: Python half **LEGACY** ("Model 1", superseded by restnet) but still loadable for old checkpoints/dashboard; **Rust half** — dense_cnn_restnet drives `hexo_models._rust.dense_cnn` (encoding, MCTS, `run_continuous`); parked alongside that lineage. |
| `hexo_models/hexgt` | `packages/hexo_models/hexgt` | **LEGACY/HALTED** ("Model 2/3" GNN+transformer; run halted at epoch 40, 2026-06-05). Still load-bearing for the dashboard debug worker and as the hexgnn fork's ancestor. |
| `hexgnn` | `packages/hexgnn` | **PARKED/LEGACY** GNN experiment. Its Rust crate is still compiled into every `hexo_models` native build. |

Key inversion to know: the ACTIVE lineage's MCTS/encoder hot path lives in the
nominally-legacy `packages/hexo_models/dense_cnn/rust/src/` tree, compiled into one
crate (`packages/hexo_models/rust/src/lib.rs` `#[path]`-includes dense_cnn, hexgt, and
hexgnn Rust) and exposed as the single PyO3 module `hexo_models._rust`.

## 2. Training loop, end to end (active path)

Entry: `python -m hexo_train.cli.train_model <dense_cnn_restnet config>.toml`,
launched detached in WSL by a per-lineage supervisor script that resumes from the
latest checkpoint and rebuilds `_resume_config.toml` on each relaunch — the same
pattern the live hexfield supervisors (`scripts/_hexfield_supervise_main1.sh` /
`_hexfield_eq_supervise_main1.sh`) use today.

### 2.1 Config parse
- `hexo_train/python/hexo_train/config.py` loads the TOML and normalizes the
  orchestration sections (`[run]`, `[loop]`, `[selfplay]`, `[train]`, `[checkpoint]`)
  into frozen dataclasses. `[model].module = "dense_cnn_restnet.plugin"` (or the
  `hexo_train.models` entry-point name) selects the plugin via
  `hexo_train/registry.py`.
- The big `[model.config.*]` block is passed through raw and parsed by
  `dense_cnn_restnet/config.py:parse_model1_config` (strict unknown-key rejection;
  home of PCR, policy-init, soft_z, length-decay, frozen_win_override, temperature
  schemes).

### 2.2 Pipeline / driver
- `hexo_train/pipeline.py:TrainingPipeline.run` executes the fixed step sequence:
  initialize run dirs + `manifest.json` (`hexo_train/artifacts.py`) -> load checkpoint
  (`hexo_train/checkpoints.py` -> plugin loader) -> `calibrate_performance`
  (`dense_cnn_restnet/performance.py:calibrate_dense_cnn`) -> per-epoch loop
  (`hexo_train/epoch/loop.py:run_epochs`) -> final checkpoint + `run.completed.json`.
- Resume contract: the plugin loader returning `{"status": "loaded", "epoch": N}`
  makes `epoch/loop.py:_start_epoch` fast-forward (how main_4 resumed from ckpt5 at
  epoch 6).
- Every step is wrapped by `RunContext.diagnostics` (`hexo_train/diagnostics.py`),
  which appends to `diagnostics/events.jsonl` and writes per-stage JSON.

### 2.3 Self-play (scheduler -> Rust MCTS -> GPU evaluator)
- `hexo_train/epoch/selfplay.py` calls `plugin.generate_selfplay` ->
  `dense_cnn_restnet/selfplay.py:generate_selfplay_epoch` (selfplay.py:396), which
  dispatches to `_generate_selfplay_epoch_continuous` (the live scheduler, per-slot
  game replacement) or `_generate_selfplay_epoch_lockstep`.
- MCTS itself is native: `dense_cnn_restnet/mcts.py` wraps `Model1MctsSession` from
  `hexo_models._rust.dense_cnn` (source: `packages/hexo_models/dense_cnn/rust/src/mcts.rs`;
  `run` = epoch-batched, `run_continuous` at mcts.rs:670 = the continuous scheduler
  with PCR/policy-init knobs, used only by restnet). Tree mechanics in `mcts_tree.rs`,
  TSS threat logic in the shared `packages/hexo_models/rust/src/threats_shared.rs`.
- State intake: the Rust crate clones live `hexo_engine.HexoState` objects through the
  versioned C-ABI capsule `hexo_engine._rust.state_api` (v2, `pybridge.rs`).
- GPU evaluator callback (the inner loop's byte protocol): Rust `mcts_eval.rs` batches
  deduplicated leaves, encodes them to (N, 13, 41, 41) f32 planes (`encoding.rs`,
  radius-20 hex-disk crop — see the known crop limitation below) and calls Python
  `dense_cnn_restnet/inference.py:DenseCNNInference.evaluate_model1_payload`
  (inference.py:337), which returns exact-length `values_bytes`/`priors_bytes` that
  Rust parses strictly. Optional FP16 / `torch.compile` (`compile_backend.py`) /
  TensorRT (`trt_backend.py`) forwards plug in here.
- Per-move levers applied in Python: temperature schemes
  (`selfplay.py:_move_temperature`), PCR full/fast coin, policy-init openings, and the
  main_4 frozen-win override fed by `dense_cnn_restnet/win_tracker.py` (each override
  verified on a cloned engine state before play).
- Game truth comes from `hexo_engine` (`new_game`/`apply_action`/`terminal`); per-game
  length EMA persists in `selfplay/length_ema.json` (selfplay.py:356-377).

### 2.4 Self-play outputs: .hxr records + npz shards
- Each game is written as a `.hxr` binary record via
  `hexo_runner.records.HexoRecordFile` (selfplay.py:553/1305) — the codec itself is
  Rust, `packages/hexo_utils/rust/src/records.rs`, re-exported through
  `hexo_utils.records` -> `hexo_runner.records`.
- Training rows: `dense_cnn_restnet/samples.py` builds compact rows from Rust facts
  (`model1_sample_from_state`) and `finalize_game_samples` attaches z/soft-Z value,
  opp-policy, STV EMA, and moves-left targets. `dense_cnn_restnet/replay.py` applies
  policy-surprise frequency weighting and length-decay row drops, then
  `compact_io.py:write_compact_shard` writes one columnar `.npz` shard + `.json`
  sidecar per game under `<run>/selfplay/`.
- An epoch summary is written to `diagnostics/dense_cnn.selfplay.epoch_NNNNNN.json`,
  and a 2s live-progress file to `diagnostics/dense_cnn.selfplay.live.json`
  (selfplay.py:44).

### 2.5 Replay window -> training
- `hexo_train/epoch/samples.py:select_training_samples` delegates to
  `dense_cnn_restnet/trainer.py:DenseCNNTrainer.select_training_samples`, which calls
  `replay.py:build_katago_shuffle` (replay.py:317): mtime-ordered KataGo-style window
  with taper, md5 train/val split, output under `<run>/shuffleddata/<generation>/`
  (scratch in `<run>/shufflescratch/`).
- `hexo_train/epoch/training.py` -> `DenseCNNTrainer.train_passes`: a spawn process
  pool expands compact shards to dense tensors (`compact_io.expand_shard_to_arrays`
  -> `input.py`, applying per-row D6 symmetry from `d6.py`), then AMP optimizer steps
  with `losses.py:model1_loss` (policy + 65-bin value + opp-policy + STV +
  moves-left). Note: the hexo_train-level per-sample D6 selector
  (`hexo_train/symmetry.py`) runs but restnet only consumes its seed.

### 2.6 Checkpoints
- `hexo_train/checkpoints.py` invokes
  `dense_cnn_restnet/checkpoints.py:DenseCNNCheckpointSaver` each epoch. Payload:
  `{"model": "dense_cnn_restnet", "model_state", "optimizer_state", "train_state",
  "epoch", "metadata"}` written to `<run>/checkpoints/epoch_NNNNNN.pt`, with optional
  `.txt` pointer indirection. Loading fail-louds on shape mismatch;
  `initialize_from` = weights-only semantics (how the HF prefit from
  `scripts/bootstrap_dense_cnn_restnet_hf.py` warm-starts a run).

### 2.7 SealBot evaluation
- `plugin.evaluate_epoch` -> `dense_cnn_restnet/evaluation.py`: on the `eval_every`
  cadence, plays the current model against the external SealBot C++ minimax bot via
  `hexo_runner.adapters.sealbot.SealBotPlayer` (which spawns
  `hexo_runner/adapters/_sealbot_worker.py` as a JSON-line subprocess over the
  external checkout at `$SEALBOT_PATH`, default `E:\SealBot` / `/mnt/e/SealBot`).
  Model moves use cross-game leaf batching through one MCTS session. Outputs:
  `.hxr` records under `<run>/evaluation/epoch_NNNNNN/` (evaluation.py:111) and
  `diagnostics/dense_cnn.evaluation.epoch_NNNNNN.json` (evaluation.py:173).

### 2.8 Diagnostics -> dashboard -> browser
- The dashboard (`packages/hexo_frontend`, stdlib `ThreadingHTTPServer`, no framework)
  runs in WSL on :8080 with cwd at the *run mount* root (`scripts/_dashboard_launch.sh`),
  scanning `runs/<name>/` read-only. It imports no training code on the HTTP path.
- `web.py` History routes (all under `/api/training/`): `runs`, `run`, `live` (tails
  `events.jsonl` + `dense_cnn.selfplay.live.json`, 2.5s browser poll), `epoch`,
  `history-page` / `history-count` (paged `.hxr` game lists), `artifacts-page`,
  `file`, `history` (legacy full replay).
- The browser side is one SPA (`static/index.html` + `static/app.js` +
  `static/styles.css`, manual `?v=` cache-bust token in three places) with three
  screens addressed by URL hash: `#match`, `#history`, `#debug`.

## 3. ASCII diagram

```
configs/<dense_cnn_restnet run>.toml
        |
        v
hexo_train.cli.train_model -> TrainingPipeline (hexo_train/pipeline.py)
        |  plugin discovery: hexo_train/registry.py -> dense_cnn_restnet/plugin.py
        v
  per-epoch loop (hexo_train/epoch/loop.py)
        |
        |-- selfplay: dense_cnn_restnet/selfplay.py (continuous scheduler)
        |       |                                        ^
        |       v        byte protocol (planes ->        | values_bytes/
        |   hexo_models._rust.dense_cnn                  | priors_bytes
        |   Model1MctsSession.run_continuous  ---------> dense_cnn_restnet/
        |   (mcts.rs / mcts_tree.rs / encoding.rs)       inference.py (GPU)
        |       |  state clone via hexo_engine._rust.state_api capsule
        |       v
        |   game truth: hexo_engine (Rust rules engine)
        |       |
        |       +--> <run>/selfplay/*.hxr        (hexo_runner.records / hexo_utils Rust codec)
        |       +--> <run>/selfplay/*.npz + .json (samples.py + replay.py + compact_io.py)
        |       +--> diagnostics/dense_cnn.selfplay.{live,epoch_N}.json
        |
        |-- replay window: trainer.select_training_samples
        |       -> replay.build_katago_shuffle -> <run>/shuffleddata/<gen>/
        |
        |-- train: DenseCNNTrainer.train_passes (D6 expand pool + AMP, losses.py)
        |
        |-- checkpoint: checkpoints.py -> <run>/checkpoints/epoch_NNNNNN.pt
        |
        +-- eval: evaluation.py vs SealBot (hexo_runner sealbot adapter subprocess)
                -> <run>/evaluation/epoch_N/*.hxr
                -> diagnostics/dense_cnn.evaluation.epoch_N.json

<run dir>  <----- read-only scan -----  hexo_frontend/web.py (:8080, WSL)
                                            |            |
                              /api/training/*      /api/debug/* --> debug_service.py
                                    |                                  | NDJSON stdin/stdout
                                    v                                  v
                          browser SPA (app.js)              debug_worker.py (CPU torch
                          #match  #history  #debug          subprocess) -> debug_infer.py
```

## 4. Run directory layout

Run dirs are created by `hexo_train/context.py` + the packages; example:
`E:\Hexo-BotTrainer\runs\<run_name>\` (a `dense_cnn_restnet` or `hexfield` run).

| Artifact | Path in run dir | Writer | Reader |
|---|---|---|---|
| Run manifest (lineage, arch, config subset) | `manifest.json` | `hexo_train/artifacts.py` | dashboard `web.py`, `debug_infer.py` |
| Event log (step start/finish) | `diagnostics/events.jsonl` | `hexo_train/diagnostics.py` | dashboard live status |
| Self-play epoch summary | `diagnostics/dense_cnn.selfplay.epoch_NNNNNN.json` | `selfplay.py` | dashboard, health/gate scripts |
| Self-play live progress (2s) | `diagnostics/dense_cnn.selfplay.live.json` | `selfplay.py` | dashboard `/api/training/live` |
| Eval epoch diagnostics | `diagnostics/dense_cnn.evaluation.epoch_NNNNNN.json` | `evaluation.py` | dashboard, health scripts |
| Per-stage step JSON | `diagnostics/<step>.json` | `hexo_train/diagnostics.py` | dashboard |
| Self-play game records | `selfplay/*.hxr` | `selfplay.py` via `hexo_runner.records` | dashboard history pages, health scripts |
| Compact training shards | `selfplay/*.npz` + `.json` sidecars | `replay.py`/`compact_io.py` | shuffler, trainer, debug `record_row` |
| Game-length EMA | `selfplay/length_ema.json` | `selfplay.py` | next epoch's adaptive temperature |
| Shuffled replay window | `shuffleddata/<generation>/{train,val}/` + `train.json` | `replay.build_katago_shuffle` | `trainer.py` |
| Shuffle scratch | `shufflescratch/` | trainer | (transient) |
| Checkpoints | `checkpoints/epoch_NNNNNN.pt` (+ optional `.txt` pointers) | `checkpoints.py` | resume, dashboard debug worker, Arena bots |
| Eval game records | `evaluation/epoch_NNNNNN/*.hxr` | `evaluation.py` | dashboard |
| Final marker | `diagnostics/run.completed.json` | `hexo_train/artifacts.py` | humans/scripts |
| Supervisor state | `supervisor.lock`, `supervisor_halted.flag`, `driver.pid`, `_resume_config.toml` | shell supervisor scripts | supervisor / bounce scripts |
| (unused by restnet) shared sample store | `samples/` | `hexo_train` default path | nothing (all plugins set `uses_shared_sample_store=False`) |

## 5. Debug-worker subprocess protocol

The Debug screen never runs torch inside the HTTP server. Chain:

```
browser #debug -> web.py /api/debug/* -> debug_service.DebugWorker (singleton, lock)
    -> spawns `python -m hexo_frontend.debug_worker` as a child process
       (on Windows hosts: via wsl.exe + /root/.venvs/hexgt-build python,
        CUDA_VISIBLE_DEVICES="" so it never touches the training GPU)
    -> debug_worker dispatches to debug_infer.py (lineage-aware CPU inference)
```

- Wire format: newline-delimited JSON, one request per stdin line, one response per
  stdout line. Request `{"id": int, "op": ..., ...}`; response `{"id", "ok",
  "result"|"error"}`. stdout is protocol-only; diagnostics go to stderr.
- Ops (`debug_worker.py`): `ping`, `info`, `analyze`, `search`, `reeval`,
  `search_tree`, `record_row`, `game_eval`. The worker keeps a 3-model LRU checkpoint
  cache and translates `E:\...` <-> `/mnt/e/...` paths.
- `debug_service.py` adds timeouts, auto-restart on transport failure, an LRU result
  cache, and an error split: `DebugWorkerError`/timeout -> HTTP 500 (retryable) vs
  `DebugRequestError` -> 400 (deterministic). Env overrides:
  `HEXO_DEBUG_WORKER_CMD`, `HEXO_DEBUG_WSL_PYTHON`, `HEXO_DEBUG_USE_WSL`,
  `HEXO_DEBUG_RUN_ROOT`.
- `debug_infer.py` sniffs checkpoint lineage (hexgt graph vs `dense_cnn_restnet` vs
  plain `hexo_models.dense_cnn` from `payload["model"]` / run `manifest.json`),
  rebuilds the network, replays action-id sequences through `hexo_engine`, and
  returns a uniform schema (priors, binned value, aux heads, MCTS, PUCT debug tree,
  npz row decode, per-ply game-eval sweeps).

## 6. Match / Arena path

The Match screen plays real games through the production runner:

- `web.py:ManualMatchController` (threaded) bridges browser clicks to
  `hexo_runner.modes.match.run_match` -> `hexo_runner/loop.py:run_match_loop`, which
  owns the single authoritative `HexoState` via `hexo_runner/engine.py:HexoEngineAdapter`
  and writes a `.hxr` per game.
- Player kinds implement the `hexo_runner/player.py:RunnerPlayer` protocol: human
  (browser), SealBot (`adapters/sealbot.py`, discovered by
  `discover_sealbot_adapters` and served at `/api/adapters`), and checkpoint bots
  (`web.py:_CheckpointBotPlayer`, which decides moves by calling the *debug worker*
  `search` op — Arena bots run on CPU, not the training GPU).
- Routes: `/api/new`, `/api/move`, `/api/state` (poll; `dashboard.py` shapes the
  engine state + threat-window payload), `/api/match/stop`; series semantics are
  handled in the controller.

## 7. Known cross-cutting contracts and caveats

- **Tensor contract** (N, 13, 41, 41), 65 value bins: triplicated by hand between
  `dense_cnn_restnet/constants.py`, `hexo_models/dense_cnn/.../constants.py`, and
  `rust/src/constants.rs`; only boundary tests keep them aligned.
- **Action-id packing** `(q+2^15)<<16 | (r+2^15)`: implemented in Rust
  (`hexo_engine/rust/src/legal.rs`), Python (`hexo_engine/python/hexo_engine/types.py`),
  and again client-side in `app.js` — persisted in shards, records, and URLs, so it
  must never diverge.
- **Radius-20 crop limitation**: `encoding.rs` intentionally excludes out-of-crop
  legal moves from policy/MCTS; this froze out-of-rim wins and was the root cause of
  the main_3 collapse; main_4's
  `win_tracker.py` frozen-win override is the mitigation.
- **Rebuilds**: Rust edits are inert until
  `scripts/_rebuild_hexo_models_hexgt.sh` (maturin, WSL `hexgt-build` venv) is run;
  rebuilding for restnet also changes legacy dense_cnn search semantics (shared crate).
- **Tests**: flat `tests/` tree; authoritative only in the WSL `hexgt-build` venv.
  GPU/torch tests self-skip elsewhere.
- Each run's authoritative journal is the header of its config TOML in `configs/`
  (e.g. `configs/hexfield_main_9.toml`, `configs/hexfield_eq_main_1.toml`) — the
  config header doubles as the run's evidence dossier.
