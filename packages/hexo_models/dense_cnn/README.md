# Dense CNN Model 1 Code Guide

This directory owns the production dense CNN model family: PyTorch model code,
native Rust MCTS, game-driven self-play, KataGo-style NPZ replay/shuffling,
row-budgeted training, checkpoint integration, and the plugin boundary used by
`hexo_train`.

The code is split into two halves:

- Python owns PyTorch, strict config parsing, plugin wiring, self-play control,
  NPZ replay/shuffling, training, checkpoints, and diagnostic adapters.
- Rust owns live `hexo_engine.HexoState` intake, dense tensor encoding, batched
  MCTS tree search, evaluator payload validation, and compact sample
  generation/finalization.

## Production Path

1. `hexo_train` loads `DenseCNNPlugin` from `python/.../plugin.py`.
2. The plugin parses `configs/dense_cnn_model1.toml`.
3. The plugin builds `Model1Network`, `DenseCNNTrainer`, checkpoint IO,
   self-play, evaluation, and NPZ replay components.
4. Self-play starts exactly the configured game count and searches every
   playable active-game move with the persistent Rust MCTS session until the
   game is terminal or reaches `max_actions`.
5. Python records compact pre-decision samples from live engine states.
6. At game end, Rust finalizes value, opponent-policy, and lookahead targets.
7. Python computes KataGo-style policy-surprise frequency weights for the game
   and physically materializes repeated rows.
8. Self-play writes per-game NPZ shards under `runs/.../selfplay/`.
9. The dense shuffler scans recent self-play NPZ shards, builds a KataGo-style
   power-law training window, shuffles rows, and writes
   `shuffleddata/<generation>/train/data*.npz`.
10. Training consumes the latest shuffled train directory using
    `train_samples_per_epoch` and train-bucket accounting.
11. Checkpoints save only model state, optimizer state, epoch, and dense train
    state. Legacy `sample_buffer` payloads are rejected instead of loaded.

There is no production fallback to the removed in-memory `SampleBuffer`,
sample-budget self-play, rollout tails, Python mirror state, raw mapping replay
entries, or silent repair of invalid inputs.

## Python Modules

| File | Role |
| --- | --- |
| `architecture.py` | PyTorch Model 1 network: hex-masked convolutions, residual trunk, policy, value, opponent-policy, and lookahead heads. |
| `checkpoints.py` | Model, optimizer, epoch, and train-state checkpoint IO. Legacy replay buffers are rejected explicitly. |
| `config.py` | Strict TOML-facing config dataclasses and validation. Removed keys fail fast. |
| `inference.py` | PyTorch inference adapter and strict Rust MCTS evaluator callback. |
| `input.py` | Expansion of compact sample facts into dense input planes, policy targets, and legal masks. |
| `losses.py` | Policy, value-bin, opponent-policy, lookahead, and combined Model 1 loss functions. |
| `mcts.py` | Python wrapper around the required Rust `BatchedMctsSession`, including root-prior policy payloads. |
| `performance.py` | Calibration probes for inference, self-play, MCTS virtual batching, and training batch size. |
| `plugin.py` | Training plugin consumed by `hexo_train`. |
| `replay.py` | KataGo-style dense CNN replay: policy-surprise materialization, self-play NPZ writing, shuffling, train windows, and train-bucket state. |
| `samples.py` | Compact sample dataclasses, compression helpers, Rust sample bridge, D6 expansion, tensor stacking, and target schema decoding. |
| `selfplay.py` | Game-driven all-MCTS self-play loop using live engine states and persistent native MCTS sessions. |
| `trainer.py` | Optimizer-backed training over shuffled NPZ rows. |

## Rust Modules

| File | Role |
| --- | --- |
| `encoding.rs` | Native Model 1 tensor encoding from `HexoState`, including crop, legal plane, recency, hot cells, and optional half-precision MCTS payloads. |
| `mcts_eval.rs` | Batched evaluator adapter, exact evaluation cache, strict byte-payload parser, and candidate-prior validation. |
| `mcts_tree.rs` | PUCT tree search, progressive widening, hidden-prior staging, tactical candidates, virtual visits, root promotion, and in-crop-only legal action universe. |
| `mcts.rs` | PyO3 `Model1MctsSession` boundary, batched root search, diagnostics, and result payloads. |
| `sample_gen.rs` | Native compact sample generation and game-end target finalization. |
| `state.rs` | Live `hexo_engine.HexoState` intake through the engine capsule API. |

## Self-Play And Replay

Dense CNN self-play is game-driven. Each epoch requests
`[selfplay].games_per_epoch` games, keeps at most `active_games` in flight, and
searches every playable nonterminal active position with MCTS. Finished games
write `.hxr` records and one NPZ shard containing effective training rows.

Policy surprise follows KataGo's frequency-weight idea for completed full-search
game samples:

- Compute `KL(target || root_prior)` for every searched move.
- Redistribute sample frequency with
  `0.5 + 0.5 * game_length * kl_i / sum_kl`.
- If the game has zero total surprise, every sample has weight `1.0`.
- Materialize `floor(weight)` copies plus one deterministic stochastic
  fractional copy.
- Do not apply per-sample loss scaling or importance correction.

The NPZ row schema is fixed:

- `inputNCHW`
- `policyTargetsNCHW`
- `oppPolicyTargetsNCHW`
- `rootPolicyNCHW`
- `legalMaskNCHW`
- `valueTargetsN`
- `lookaheadTargetsNC`
- `lookaheadMasksNC`
- `metadataInputNC`

Each self-play shard has a JSON sidecar with row counts, epoch, game id, schema
version, and policy-surprise summaries.

## Shuffling And Training

`replay.py` mirrors KataGo's selfplay/shuffler/training split inside the local
dense CNN loop:

1. Scan self-play NPZ shards and count rows from sidecars or NPZ headers.
2. Sort shards by modification time.
3. Select the newest power-law replay window using `shuffle_min_rows`,
   `shuffle_expand_window_per_row`, `shuffle_taper_window_exponent`, and
   `shuffle_taper_window_scale`.
4. Apply `shuffle_keep_target_rows` through a keep probability.
5. Split files by deterministic MD5 ranges: train-only by default, or
   train/validation when `validation_fraction > 0`.
6. Use scratch worker groups to write temporary shards, then merge them into
   batch-size-aligned files under `shuffleddata/<generation>/train/` and
   optional `val/`.
7. Let `DenseCNNTrainer` consume exactly `train_samples_per_epoch` rows when the
   train bucket has enough budget.

The dense trainer ignores generic `passes_per_epoch` as a replay budget. It
reports the requested generic pass count for diagnostics, but dense training is
row-budgeted.

Train state includes:

- `global_step_samples`
- `total_num_data_rows`
- `window_start_data_row_idx`
- `train_bucket_level`
- `train_bucket_level_at_row`
- `train_steps_since_last_reload`
- `data_files_used`
- `old_train_data_dirs`
- `latest_shuffle_dir`

New shuffled rows add `new_rows * max_train_bucket_per_new_data`, capped by
`max(max_train_bucket_size, train_samples_per_epoch)`. Training consumes row
budget before optimizer work, and row-count regressions do not mint free budget.

## MCTS Legality

Dense-cnn MCTS intentionally uses only engine-legal moves represented by the
current dense crop. The model cannot produce policy targets for out-of-crop
actions, so out-of-crop legal moves receive no hidden prior, are not lazily
materialized, are not tactical candidates, and cannot be selected by MCTS.

Progressive widening remains active for the in-crop legal universe. Hidden prior
mass still applies to in-crop legal actions omitted by candidate-limited top-k
priors.

## Boundary Rules

- Config parsing rejects unknown keys and invalid values.
- Rust MCTS rejects invalid search settings and mismatched state/key batches.
- Python evaluator callbacks reject malformed byte payloads before tensor views.
- Rust evaluator parsing rejects illegal candidates, duplicate priors, missing
  bytes, wrong lengths, invalid values, and zero prior mass.
- Dense sample schema is versioned across Python and Rust.
- Loss functions reject invalid targets before normalization.

When changing Model 1 representation, update both halves together:

- Plane indices: `python/.../constants.py` and `rust/src/constants.rs`.
- Crop projection: `python/.../geometry.py`, `python/.../input.py`, and
  `rust/src/encoding.rs`.
- Sample schema: `python/.../samples.py` and `rust/src/sample_gen.rs`.
- MCTS evaluator payload: `python/.../inference.py` and `rust/src/mcts_eval.rs`.
- Replay/training schema: `python/.../replay.py`, `python/.../trainer.py`, and
  dense CNN tests.
