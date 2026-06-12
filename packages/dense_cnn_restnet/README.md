# dense_cnn_restnet

**Status: ACTIVE.** This is the model lineage powering the live `main_4` training run
(`configs/dense_cnn_restnet_main_4.toml`). It is a faithful ResTNet (interleaved
residual + transformer trunk, arXiv:2410.05347) fork of the legacy
`packages/hexo_models/dense_cnn` ("Model 1") lineage.

The package is **pure Python / PyTorch and carries no Rust of its own**. It reuses
the already-built native accelerator `hexo_models._rust.dense_cnn` read-only
(featurizer, batched MCTS sessions including `run_continuous`, sample facts, the
`(N, 13, 41, 41)` byte contract). Installing or editing this package never rebuilds
the native module; Rust changes require `scripts/_rebuild_hexo_models_hexgt.sh`.

It plugs into the generic `hexo_train` pipeline as the `dense_cnn_restnet` plugin
(entry point in `pyproject.toml`, or `module = "dense_cnn_restnet.plugin"` in the
config) and owns config parsing, the network, self-play generation, KataGo-style
NPZ replay/shuffle, the training loop, SealBot evaluation, checkpoint IO, and
performance calibration.

## Module table

All paths relative to `python/dense_cnn_restnet/`.

| File | Role |
|---|---|
| `plugin.py` | Composition boundary for `hexo_train.registry`: `build_model`, AdamW with decay/no-decay param split, `ComponentOverrides` (trainer + checkpoint IO), delegates selfplay/eval/calibration. `get_plugin()` is the entry point. |
| `config.py` | TOML boundary: `parse_model1_config` builds frozen dataclasses (architecture/training/samples/selfplay/evaluation/performance) with unknown-key rejection. Home of every tuning lever: PCR, policy-init, soft_z, length decay, `frozen_win_override`, temperature schemes. |
| `constants.py` | Python side of the tensor contract (`BOARD_SIZE=41`, 13 planes, 65 value bins, `MOVES_LEFT_CAP=512`, plane indices); mirrors the shared `hexo_models/dense_cnn/rust/src/constants.rs`. |
| `architecture.py` | `RestnetNetwork`: hex-masked conv stem, residual/transformer interleaved trunk per `blocks_type`, relative-position MHSA, heads_v3 dual `ValueReduction` (main value vs aux stvalue/moves_left). `optimized_restnet_for_inference` does inference-time folding. |
| `inference.py` | `DenseCNNInference`: the only Torch evaluator used by production MCTS. `evaluate_model1_payload` is the strict byte callback Rust calls (returns `values_bytes`/`priors_bytes`). Bucket padding, FP16, optional TensorRT / torch.compile adoption. |
| `mcts.py` | Thin Python wrapper over the native `Model1MctsSession`: `BatchedMctsSession.run` / `run_continuous`, byte-backed `CompactVisitPolicy` / `SearchResult` decoding. |
| `rust_bridge.py` | Import/call boundary to `hexo_models._rust.dense_cnn` (`model1_batch_inputs`, `Model1MctsSession`, `model1_sample_from_state`); readable error if the native module is absent. |
| `selfplay.py` | ~1600-line core: `generate_selfplay_epoch` dispatches lockstep vs continuous schedulers; per-move temperature schemes, PCR full/fast coin, policy-init openings, frozen-win override, length-EMA persistence, live-progress JSON, per-game NPZ shard writes. |
| `win_tracker.py` | `IncrementalWinTracker`: O(1)-per-stone 6-in-a-row standing-win-cell bookkeeping; feeds selfplay's frozen-win override. New for main_4. |
| `samples.py` | Compact `Model1SampleData` rows; `sample_from_state` (Rust facts + search targets), `finalize_game_samples` (z / soft-Z value, opp-policy, STV EMA, moves_left), `expand_sample` to dense tensors. |
| `replay.py` | KataGo-style replay: policy-surprise frequency weighting, length-decay row drops, per-game NPZ shard writes with JSON sidecars, `build_katago_shuffle` window/taper/md5-split, `DenseTrainState` bucket bookkeeping. |
| `compact_io.py` | Columnar NPZ (de)serialization of compact shards; `expand_shard_to_arrays` does train-read expansion under per-row D6 symmetry. Shared format with selfplay writer, shuffler, trainer, HF prefit script. |
| `trainer.py` | `DenseCNNTrainer`: `select_training_samples` (shuffle + train-bucket + no-repeat-files window), `train_passes` (spawn-pool shard expansion, AMP steps, per-component loss reporting, validation). |
| `losses.py` | 65-bin value distribution helpers, soft/segmented cross-entropy, `model1_loss` combining policy/value/opp_policy/stvalue/moves_left. |
| `input.py` | Expands compact facts into the 13-plane 41x41 crop tensors (D6 applied first); mirrors the Rust encoder. |
| `d6.py` | Axial-coordinate D6 symmetry transforms and packed action-id transforms. |
| `geometry.py` | Crop geometry: hex-disk membership, coord <-> row/col/flat projection, `hex_distance`, crop center. |
| `evaluation.py` | Per-epoch SealBot eval with cross-game leaf batching through one MCTS session; `.hxr` records + `dense_cnn.evaluation.epoch_*.json` diagnostics. |
| `checkpoints.py` | Checkpoint loader/saver: `.pt` payload `{model: "dense_cnn_restnet", model_state, optimizer_state, train_state, epoch, metadata}`; fail-loud on shape mismatch at resume; weights-only semantics for `initialize_from`; `.txt` pointer indirection. |
| `performance.py` | `calibrate_dense_cnn` probes inference/training/selfplay batch sizes (both schedulers) plus MCTS diagnostics summarizers imported by selfplay. |
| `compile_backend.py` | Optional torch.compile drop-in for `forward_policy_value`, bucketed and correctness-gated vs eager. |
| `trt_backend.py` | Optional TensorRT FP16 forward (self-invoking subprocess engine build, correctness gate); self-play only. |
| `player.py` | hexo_runner player adapter -- **dead code candidate**: no live caller of the restnet copy (live users import `hexo_models.dense_cnn.player`). |
| `debug_artifacts.py` | Optional PNG game renderer -- **dead code candidate**: no caller of the restnet copy anywhere. |

## The selfplay -> replay -> train loop

Each epoch driven by `hexo_train.epoch.loop`:

1. **Selfplay** (`selfplay.generate_selfplay_epoch`): runs many concurrent games
   through one persistent Rust MCTS session, in either the *lockstep* or
   *continuous* scheduler (config `selfplay.scheduler`). The Rust search calls
   back into `DenseCNNInference.evaluate_model1_payload` with raw plane bytes;
   Python returns value/prior bytes. Finished games are finalized
   (`samples.finalize_game_samples`: z/soft-Z value targets, opp-policy, STV,
   moves_left) and written as one compact NPZ shard + JSON sidecar per game under
   `<run_dir>/selfplay/` (`replay.write_selfplay_npz`). The adaptive-temperature
   game-length EMA persists in `selfplay/length_ema.json`.
2. **Window selection** (`trainer.select_training_samples` ->
   `replay.build_katago_shuffle`): a KataGo-style growing/tapered window over the
   most recent shards (mtime-ordered -- why seeding scripts use `cp -p`), with
   policy-surprise frequency weighting, length-decay row drops, and an md5
   train/validation split.
3. **Training** (`trainer.train_passes`): a spawn process pool expands compact
   shards to dense tensors (`compact_io.expand_shard_to_arrays`, fresh per-row
   D6 symmetry each epoch), AMP optimizer steps with `losses.model1_loss`,
   per-component loss diagnostics, validation pass.
4. **Checkpoint + eval**: `checkpoints.py` saves `checkpoints/epoch_NNN.pt`;
   `evaluation.evaluate_epoch` plays SealBot gating games on the `eval_every`
   cadence, writing `.hxr` records and diagnostics JSON.

Diagnostics filenames are still `dense_cnn.selfplay.live.json`,
`dense_cnn.selfplay.epoch_*.json`, `dense_cnn.evaluation.epoch_*.json` (naming
debt from the fork); the dashboard and health scripts consume them.

## The frozen-win override (main_4, high level)

Root cause of the main_3 collapse: the network sees only a radius-20 hex-disk
crop around the stone centroid, and the Rust encoder *intentionally excludes
out-of-crop legal moves* from policy/MCTS. A game can reach a state where a
player has a "standing win" (an empty cell completing 6-in-a-row) that lies
outside the crop -- invisible to both players -- and the game freezes instead of
ending, poisoning value targets.

Mitigation (`selfplay.frozen_win_override` + `win_tracker.py`): during selfplay,
an `IncrementalWinTracker` per game maintains every mover's standing-win cells
incrementally. Before committing a search move, `_frozen_win_override_action`
checks: if the mover has standing wins and **all** of them are out-of-crop (the
frozen-game signature), it picks the min-packed-id winning cell, **verifies it on
a cloned engine state** (must be legal, terminal, correct winner), and plays it
instead of the search move. In-crop standing wins are left to the search. Failed
verifications fall through to the search move and are counted
(`frozen_win_override_failures`); the override never raises. Override counts are
reported per epoch and watched by `scripts/_wf_r4_m4_gates.py`.

## Connections to other packages

**Inbound dependencies (what this package imports):**

| Dependency | Used for |
|---|---|
| `hexo_models._rust.dense_cnn` (via `rust_bridge.py`) | Featurizer, MCTS sessions (`run` + `run_continuous`), sample facts. Read-only; the Rust source lives in `packages/hexo_models/dense_cnn/rust/`, built by maturin into the `hexo_models` wheel. |
| `hexo_engine` | Game truth: `new_game`, `apply_action`, `terminal`, `clone_state`, `engine_metadata`, coord packing. |
| `hexo_runner` | `.hxr` game records (`hexo_runner.records.HexoRecordFile`, `AbortRecord`); `hexo_runner.adapters.sealbot.SealBotPlayer` as the eval opponent. |
| `hexo_train` | `ComponentOverrides` plugin contract; the generic pipeline calls `build_model` / `training_component_overrides` / `generate_selfplay` / `evaluate_epoch` / `calibrate_performance`. |

**Outbound consumers (who reads this package or its artifacts):**

- `hexo_frontend/debug_infer.py` imports `architecture`, `inference`, `losses`,
  `mcts`, `rust_bridge` to serve the dashboard Debug screen and Match-Arena
  checkpoint bots against checkpoints tagged `payload["model"] == "dense_cnn_restnet"`.
- The :8080 dashboard (`hexo_frontend/web.py`) and health scripts
  (`scripts/_wf_r4_health.py`, `_wf_r4_m4_gates.py`) consume the diagnostics
  JSON contract written via `ctx.diagnostics`.
- `scripts/bootstrap_dense_cnn_restnet_hf.py` (HF behavioral-cloning prefit) and
  `scripts/_restnet_migrate_heads_v2.py` (head-layout migration) read/write the
  compact NPZ shard format and checkpoint payloads.

**Key protocols / shared formats:**

- *Evaluator byte protocol*: Rust MCTS -> `evaluate_model1_payload(plane bytes,
  legal flat rows)` -> exact-length `values_bytes`/`priors_bytes` back. Both
  sides validate strictly.
- *Compact NPZ shards + JSON sidecars* (`compact_io`/`replay`): selfplay ->
  shuffle -> trainer, also read by the prefit script. Must stay byte-compatible
  with the shared Rust encoder.
- *Checkpoint `.pt` payload* described above, with `.txt` pointer indirection.
- *Tensor contract* `(N, 13, 41, 41)`: `constants.py`/`geometry.py`/`input.py`
  must stay manually in sync with `hexo_models/dense_cnn/rust/src/{constants,encoding}.rs`.

## Entry points / how it gets exercised

| Entry | What |
|---|---|
| `python -m hexo_train.cli.train_model configs/dense_cnn_restnet_main_4.toml` | The live main_4 run (continuous scheduler, frozen_win_override, length decay). `main1`/`main_2`/`main_3` configs are earlier runs of the same lineage. |
| `scripts/_wf_r4_launch_main4.sh`, `_wf_r4_bounce_main4.sh`, `_dc_restnet_supervise_main1.sh` (generic supervisor despite the name), `_restnet_bounce.sh`, `_restnet_*` poll/verify monitors | WSL launch/supervise/babysit tooling. |
| `scripts/bootstrap_dense_cnn_restnet_hf.py` | HF prefit that warm-starts checkpoints through `DenseCNNTrainer`/replay. |
| hexo_frontend dashboard (:8080) | Debug-screen inference, Arena checkpoint bots, history/live views built on this package's diagnostics and checkpoints. |
| pytest | `tests/test_dense_cnn_restnet*.py` (core, attention, crop, heads_v2, fp16, pcr_policy_init, policy_mask), `tests/test_dense_cnn_continuous_scheduler.py`, `tests/test_dense_cnn_compile_backend.py`, `tests/test_restnet_win_tracker.py`, `tests/test_restnet_frozen_win_smoke.py`, `tests/test_restnet_length_decay.py`. Authoritative only in the WSL `hexgt-build` venv. |
| Analysis/bench scripts | `scripts/_wf_r4_h2h_arena.py`, `_continuous_ab_gate.py`, `_kv_gather_bench.py`, `_forward_lever_bench.py`, `_restnet_crop_coverage.py`, grid probes -- import the package directly. |

## Gotchas

- **Lineage-fork duplication.** Most of this package is a copy of
  `hexo_models/dense_cnn` with the same `Model1` naming. Fixes land in one
  lineage and not the other (`player.py` / `debug_artifacts.py` here are dead
  symptoms). The shared Rust module is the only physically shared layer -- and
  rebuilding it for restnet also changes the parent dense_cnn lineage's search.
- **Naming debt.** Everything internal is still `Model1` / `model1_*` /
  `dense_cnn.*` (including diagnostics filenames) inside a package named
  `dense_cnn_restnet`.
- **Scheduler changes the RNG stream.** PCR/policy-init determinism is
  implemented twice: a Python splitmix64 coin for the lockstep scheduler and a
  native Rust `mix_seed` for the continuous one. Switching scheduler changes the
  full/fast schedule for the same seed; the contract lives only in comments.
- **Four stacked temperature schemes** (linear decay, anchor schedule, adaptive
  halflife EMA, opening floor) plus the KataGo root-policy early ramp, with
  precedence encoded in `_move_temperature`'s branch order in `selfplay.py`.
  Easy to misconfigure; read `config.py`'s prose before touching any of them.
- **`residual_blocks` is silently ignored.** `Model1ArchitectureConfig` parses
  it (for dense_cnn config symmetry) but `build_model` does not use it, despite
  the package's otherwise fail-fast unknown-key philosophy.
- **`length_ema.json` is sticky.** Selfplay seeds the game-length EMA from
  `temperature_length_prior` only when the file is absent; the main_4 launcher
  hard-aborts if a stale one is present.
- **Resume vs initialize is a heuristic.** `checkpoints.py` infers weights-only
  vs full-resume by string-comparing the ref against
  `ctx.config.checkpoint.initialize_from`; fragile if the pipeline's ref
  resolution changes.
- **Cross-module private imports.** `selfplay.py` imports underscore-named
  helpers from `performance.py` and `mcts.py`; keep them in sync if refactoring.
- **Run dirs live on a different mount** than the repo
  (`/mnt/e/Hexo-BotTrainer/runs` vs `/mnt/e/Hexo-BotTrainer-hexgt`); ops scripts
  hardcode both.
