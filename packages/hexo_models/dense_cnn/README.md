# Dense CNN Model 1 Code Guide

This directory owns the production dense CNN model family: PyTorch model code,
native Rust MCTS, game-driven self-play, KataGo-style NPZ replay/shuffling,
row-budgeted training, checkpoint integration, and the plugin boundary used by
`hexo_train`.

The code is split into two halves:

- Python owns PyTorch, config parsing, plugin wiring, self-play control, sample
  finalization, NPZ replay/shuffling, training, checkpoints, and the neural
  evaluator callback used by MCTS.
- Rust owns live `hexo_engine.HexoState` intake, dense tensor encoding, batched
  PUCT MCTS tree search, and state-derived sample facts.

## Production Path

1. `hexo_train` loads `DenseCNNPlugin` from `python/.../plugin.py`.
2. The plugin parses `configs/dense_cnn_model1.toml`.
3. The plugin builds `Model1Network`, `DenseCNNTrainer`, checkpoint IO,
   self-play, evaluation, and NPZ replay components.
4. Self-play starts exactly the configured game count and searches every
   playable active-game move with the persistent Rust MCTS session until the
   game is terminal or reaches `max_actions`.
5. Python records compact pre-decision sample facts from live engine states and
   attaches the MCTS visit policy and root prior.
6. At game end, Python finalizes value, opponent-policy, and short-term value
   targets from the game's decision sequence (pure arithmetic, no engine state).
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

There is no production fallback to an in-memory `SampleBuffer`, sample-budget
self-play, rollout tails, progressive widening, or a candidate-limited evaluator.

## Python Modules

| File | Role |
| --- | --- |
| `architecture.py` | PyTorch Model 1 network: hex-masked convolutions, residual trunk, policy, value, opponent-policy, and short-term value heads. |
| `checkpoints.py` | Model, optimizer, epoch, and train-state checkpoint IO. Legacy replay buffers are rejected explicitly. |
| `config.py` | TOML-facing config dataclasses. Rejects unknown keys per section and coerces types; no per-scalar range validation. |
| `inference.py` | PyTorch inference adapter and the single-mode Rust MCTS evaluator callback (full legal priors). |
| `input.py` | Expansion of compact sample facts into dense input planes, policy targets, and legal masks. |
| `losses.py` | Policy, value-bin, opponent-policy, short-term value, and combined Model 1 loss functions. |
| `mcts.py` | Python wrapper around the required Rust `Model1MctsSession`. |
| `performance.py` | Calibration probes for inference, self-play, MCTS virtual batching, and training batch size. |
| `plugin.py` | Training plugin consumed by `hexo_train`. |
| `replay.py` | KataGo-style dense CNN replay: policy-surprise materialization, self-play NPZ writing, shuffling, train windows, and train-bucket state. |
| `samples.py` | Compact sample dataclass, Python finalization (value/opponent-policy/short-term value), D6 expansion, and tensor stacking. |
| `selfplay.py` | Game-driven all-MCTS self-play loop using live engine states and the persistent native MCTS session. |
| `trainer.py` | Optimizer-backed training over shuffled NPZ rows. |

## Rust Modules

| File | Role |
| --- | --- |
| `encoding.rs` | Native Model 1 f32 tensor encoding from `HexoState`: crop, legal plane, recency, hot cells, and per-row legal crop flats. |
| `mcts_eval.rs` | Batched evaluator adapter, exact evaluation cache, and strict byte-payload parsing for the single full-prior evaluator mode. |
| `mcts_tree.rs` | PUCT tree search over all legal candidates: lazy-staged edges, FPU, virtual visits, root-policy temperature, total-alpha Dirichlet, root promotion. |
| `mcts.rs` | PyO3 `Model1MctsSession` boundary, batched root search, diagnostics, and result payloads. |
| `sample_gen.rs` | State-derived compact sample facts from a live `HexoState`. |
| `state.rs` | Live `hexo_engine.HexoState` intake through the engine capsule API. |

## Self-Play And Replay

Dense CNN self-play is game-driven. Each epoch requests
`[selfplay].games_per_epoch` games, keeps at most `active_games` in flight, and
searches every playable nonterminal active position with MCTS over all legal
in-crop moves. Finished games write `.hxr` records and one NPZ shard.

Finalization is pure Python over the game's `(player, root_value)` sequence:

- **Value:** `+1` if the decision's player won, `-1` if the other player won,
  `0` for a draw or a `max_actions` truncation.
- **Opponent policy:** the next opposing decision's MCTS visit policy.
- **Short-term value:** a perspective-corrected exponential moving average of
  future MCTS root values. Horizon `m` uses decay `m / (m + 1)` (mean look-ahead
  distance `m`); a horizon is emitted whenever at least one future decision
  exists.

Policy surprise follows KataGo's frequency-weight idea for completed game
samples:

- Compute `KL(target || root_prior)` for every searched move.
- Redistribute frequency with `uniform_fraction + (1 - uniform_fraction) *
  game_length * kl_i / sum_kl`, clamped to `policy_surprise_max_weight`.
- If the game has zero total surprise, every sample has weight `1.0`.
- Materialize `floor(weight)` copies plus one deterministic stochastic
  fractional copy.

The NPZ row schema is fixed:

- `inputNCHW`
- `policyTargetsNCHW`
- `oppPolicyTargetsNCHW`
- `rootPolicyNCHW`
- `legalMaskNCHW`
- `valueTargetsN`
- `shortTermValueTargetsNC`
- `shortTermValueMasksNC`
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
   `shuffle_taper_window_scale` (the real KataGo `shuffle.py` taper window).
4. Apply `shuffle_keep_target_rows` through a keep probability.
5. Split files by deterministic MD5 ranges: train-only by default, or
   train/validation when `validation_fraction > 0`.
6. Use scratch worker groups to write temporary shards, then merge them into
   batch-size-aligned files under `shuffleddata/<generation>/train/` and
   optional `val/`. (The two-phase disk shuffle is required because a full
   window of dense input planes does not fit in RAM.)
7. Let `DenseCNNTrainer` consume exactly `train_samples_per_epoch` rows when the
   train bucket has enough budget.

New shuffled rows add `new_rows * max_train_bucket_per_new_data`, capped by
`max(max_train_bucket_size, train_samples_per_epoch)`. Training consumes row
budget before optimizer work, and row-count regressions do not mint free budget.

## MCTS

Dense-cnn MCTS uses only engine-legal moves represented by the current dense
crop. The evaluator returns exactly one prior per in-crop legal move, so every
legal candidate is staged at a node and an edge is materialized lazily when PUCT
selects it. There is no progressive widening, no candidate cap, and no hidden
prior mass; out-of-crop legal moves are simply not part of the policy contract.

At the root, the model prior is softened by `root_policy_temperature` and then
mixed with Dirichlet noise whose total concentration `root_dirichlet_total_alpha`
is spread across the legal moves (per-action `alpha = total_alpha / legal_count`).

## Boundary Rules

- Config parsing rejects unknown keys per section.
- Rust MCTS rejects invalid search settings and mismatched state/key batches.
- The Python evaluator callback and Rust parser reject malformed byte payloads,
  wrong lengths, non-finite values, duplicate priors, and zero prior mass.

When changing Model 1 representation, update both halves together:

- Plane indices: `python/.../constants.py` and `rust/src/constants.rs`.
- Crop projection: `python/.../geometry.py`, `python/.../input.py`, and
  `rust/src/encoding.rs`.
- Sample facts: `python/.../samples.py` and `rust/src/sample_gen.rs`.
- MCTS evaluator payload: `python/.../inference.py` and `rust/src/mcts_eval.rs`.
- Replay/training schema: `python/.../replay.py`, `python/.../trainer.py`, and
  the dense CNN tests.
