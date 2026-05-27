# Dense CNN Pipeline Guide

This document explains how the current Hexo codebase is laid out and how the
`dense_cnn` model path works end to end. It focuses on the production Model 1
path under `packages/hexo_models/dense_cnn`, including MCTS, runner
interaction, inference, self-play sample generation, training, checkpointing,
and evaluation.

## 1. Codebase Layout

At a high level, this repository is split into model-neutral infrastructure and
model-owned implementations.

| Area | Purpose |
| --- | --- |
| `configs/` | Training configs. `configs/dense_cnn_model1.toml` is the main dense CNN config. |
| `packages/hexo_engine/` | Canonical Hexo rules, state transitions, legal moves, terminal detection, and Python bindings over Rust. |
| `packages/hexo_runner/` | Generic headless game runner. It owns game loops, player contracts, match/batch modes, SealBot adapter, and `.hxr` replay writing. |
| `packages/hexo_utils/` | Shared utilities: records, sample-store mechanics, symmetry contracts, and state hashing. |
| `packages/hexo_train/` | Config-driven training orchestration. It loads model plugins and owns the fixed training lifecycle. |
| `packages/hexo_models/` | Production model families. `dense_cnn` and `hexformer_ar` live side by side here. This package also builds `hexo_models._rust`. |
| `packages/hexo_frontend/` | Local browser/dashboard tooling over engine/runner artifacts. |
| `data/` | Shared durable pointers and sample/replay/checkpoint placeholders. |
| `runs/` | Run outputs: checkpoints, diagnostics, self-play records, evaluation records, dashboard artifacts. |
| `tests/` | Contract tests for engine, runner, training pipeline, dense CNN, Rust bridges, and frontend artifacts. |

The Rust workspace is defined in root `Cargo.toml` and currently includes:

- `packages/hexo_engine`
- `packages/hexo_models`
- `packages/hexo_utils`

Python packages are installed independently. Rust-backed Python packages use
`maturin`; pure Python packages use `hatchling`.

## 2. Core Ownership Boundaries

The main rule of the codebase is: shared packages orchestrate, model packages
interpret.

`hexo_engine` owns game truth:

- The authoritative `HexoState`.
- Legal action generation.
- Applying one single-stone placement.
- Turn phase progression.
- Terminal/winner detection.
- Replayable placement history.

`hexo_runner` owns generic game execution:

- Player lifecycle: `setup_worker`, `start_game`, `decide`,
  `observe_transition`, `finish_game`, `close`.
- One authoritative primary game state per match.
- Cloned states passed to players, so players cannot corrupt the primary state.
- `.hxr` records for completed or aborted games.
- Match and local multiprocessing batch modes.

`hexo_train` owns training order:

- Load config.
- Build run directories and diagnostics.
- Load a model plugin.
- Build model and model-specific components.
- Load or initialize checkpoint.
- Optionally calibrate performance.
- Run epochs.
- Save/publish checkpoints.
- Write final diagnostics.

`hexo_models.dense_cnn` owns dense CNN semantics:

- Model architecture.
- Input planes and crop encoding.
- Dense policy/value/auxiliary heads.
- Losses.
- Inference.
- MCTS bridge.
- Self-play implementation.
- Sample schema.
- Replay buffer.
- Training loop.
- Checkpoint payload.
- SealBot evaluation hook.

`hexo_utils` owns reusable mechanics that models may use:

- Generic sample store and sample windows.
- Reusable records.
- D6 symmetry contracts.
- Shared state hashing helpers.

Dense CNN uses some shared utilities, but its production self-play and training
samples are mostly model-owned.

## 3. Hexo Engine Model

The engine is the ground truth for game rules.

Important files:

- `packages/hexo_engine/rust/src/state.rs`
- `packages/hexo_engine/rust/src/rules.rs`
- `packages/hexo_engine/python/hexo_engine/api.py`
- `packages/hexo_engine/python/hexo_engine/types.py`

Hexo turns are autoregressive:

1. `Opening`: player0 places `(0, 0)`.
2. `FirstStone`: current player places the first stone of a normal turn.
3. `SecondStone`: the same player places the second stone, then turn passes.

Every action exposed to Python is one placement:

```text
PlacementAction(AxialCoord(q, r))
```

This matters for MCTS. MCTS does not choose two-stone pairs. Every tree edge is
one single placement. A normal Hexo turn is searched as:

```text
FirstStone -> SecondStone -> opponent FirstStone
```

The engine checks for a six-in-line win after every single placement. If the
first stone of a two-stone turn wins, the second stone is never played.

Action IDs are packed coordinates:

```text
u32 = ((q + 2^15) << 16) | (r + 2^15)
```

Python helpers:

- `pack_coord_id(coord)`
- `unpack_coord_id(action_id)`
- `legal_action_ids(state)`
- `apply_action(state, PlacementAction(...))`
- `terminal(state)`
- `to_python_state(state)`

## 4. Training Entry Point

The standard dense CNN command from the README is:

```powershell
$env:SEALBOT_PATH = "C:\path\to\SealBot"
python -m hexo_train.cli.train_model .\configs\dense_cnn_model1.toml
```

The CLI is intentionally thin:

```text
hexo_train.cli.train_model.main
  -> TrainingPipeline().run(config_path)
```

Important files:

- `packages/hexo_train/python/hexo_train/cli/train_model.py`
- `packages/hexo_train/python/hexo_train/pipeline.py`
- `packages/hexo_train/python/hexo_train/config.py`
- `configs/dense_cnn_model1.toml`

`hexo_train.config` normalizes the TOML into typed sections:

- `model`
- `run`
- `loop`
- `selfplay`
- `samples`
- `train`
- `checkpoint`

Model-owned nested settings remain opaque to `hexo_train` and are passed through
as `config.model.config`.

## 5. Dense CNN Config

`configs/dense_cnn_model1.toml` selects the dense CNN plugin:

```toml
[model]
name = "dense_cnn"
module = "hexo_models.dense_cnn.plugin"
```

Key baseline settings:

| Section | Setting | Meaning |
| --- | --- | --- |
| `model.config.architecture` | `input_channels = 13` | Number of input planes. |
| `model.config.architecture` | `channels = 96` | Trunk width. |
| `model.config.architecture` | `residual_blocks = 6` | Number of gated residual blocks. |
| `model.config.architecture` | `crop_size = 41` | Dense square crop side length. |
| `model.config.architecture` | `lookahead_horizons = [1, 4, 8]` | Auxiliary future-value targets. |
| `model.config.selfplay` | `samples_per_epoch = 4096` | Number of searched positions recorded per epoch. |
| `model.config.selfplay` | `search_visits = 128` | Exact MCTS simulations per searched position. |
| `model.config.selfplay` | `active_games = 2048` | Target active self-play games in the dense loop. |
| `model.config.samples` | `capacity = 200000` | Replay buffer capacity floor. |
| `model.config.samples` | `train_sample_count = 4096` | Samples drawn for each epoch's training window. |
| `model.config.evaluation` | `games_per_epoch = 64` | SealBot evaluation games after each epoch. |
| `model.config.evaluation` | `require_sealbot = true` | Fail fast if SealBot is unavailable. |
| `model.config.performance` | `target_selfplay_positions_per_second = 128.0` | Calibration target. |
| `loop` | `epochs = 30` | Total training epochs. |
| `checkpoint` | `resume_from = "../data/checkpoints/dense_cnn_model1_latest.txt"` | Resume pointer. |

The top-level `[selfplay] games_per_epoch` and model-level
`[model.config.selfplay] samples_per_epoch` are related but not identical:

- `games_per_epoch` bounds how many games the epoch may start.
- `samples_per_epoch` is the number of searched positions to record as
  training samples.
- Dense CNN searches only until the sample budget is covered, then may finish
  active games with policy rollouts.

## 6. Plugin Wiring

Important file:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/plugin.py`

The plugin object is `DenseCNNPlugin`.

`hexo_train.registry.load_model_plugin()` imports
`hexo_models.dense_cnn.plugin`, then reads `get_plugin()` or `plugin`.

The plugin provides these hooks:

| Hook | Called by | Purpose |
| --- | --- | --- |
| `build_model(game_spec, config)` | `hexo_train.components.build_model_components` | Create `Model1Network`. |
| `training_component_overrides(...)` | `build_model_components` | Supply trainer, optimizer, sample finalizer, checkpoint IO, symmetry selector. |
| `generate_selfplay(ctx, components, epoch, games_per_epoch)` | `hexo_train.epoch.selfplay.generate_selfplay` | Run dense CNN self-play. |
| `evaluate_epoch(ctx, components, epoch)` | `hexo_train.epoch.loop.evaluate_epoch` | Run SealBot evaluation. |
| `calibrate_performance(ctx, components)` | `TrainingPipeline._calibrate_performance` | Benchmark and choose batch settings. |

The plugin builds:

- `Model1Network`
- `SampleBuffer`
- `torch.optim.AdamW`
- `DenseCNNTrainer`
- `DenseCNNSampleFinalizer`
- `DenseCNNCheckpointLoader`
- `DenseCNNCheckpointSaver`
- `DenseCNNRandomExpansionSymmetrySelector`

The symmetry selector reports that dense CNN does not preselect one fixed D6
transform per sample window. Instead, the trainer samples a fresh random D6
transform each time each compact sample is expanded.

## 7. Top-Level Training Pipeline

Important file:

- `packages/hexo_train/python/hexo_train/pipeline.py`

The lifecycle is:

```text
TrainingPipeline.run(config_path)
  -> load_training_config(config_path)
  -> RunContext.from_config(config)
  -> load_model_plugin(config.model)
  -> build_shared_components(ctx)
  -> build_model_components(plugin, ctx, shared)
  -> initialize_run
  -> load_checkpoint
  -> calibrate_performance
  -> run_epochs
  -> publish_final_model
  -> write_diagnostics
```

RunContext creates:

```text
<output_dir>/
  checkpoints/
  diagnostics/
  samples/
```

For the dense config, the default output directory resolves to:

```text
runs/dense_cnn_model1/
```

The pipeline wraps each top-level step in diagnostics:

- Writes stage start/finish events.
- Stores results in `ctx.outputs`.
- Writes one JSON summary per stage in `diagnostics/`.
- Re-raises failures after recording the failure.

## 8. Epoch Order

Important file:

- `packages/hexo_train/python/hexo_train/epoch/loop.py`

Each epoch runs in this fixed order:

```text
run_epoch
  -> generate_selfplay
  -> finalize_samples
  -> select_training_samples
  -> select_epoch_symmetries
  -> train_passes
  -> save_epoch_checkpoint
  -> evaluate_epoch
```

For dense CNN, the model plugin owns most of these actions:

- `generate_selfplay` is dense CNN self-play.
- `finalize_samples` reports dense CNN's already-finalized replay buffer.
- `select_training_samples` is overridden by `DenseCNNTrainer`.
- `select_epoch_symmetries` returns metadata only; training picks random D6
  per expansion.
- `train_passes` is overridden by `DenseCNNTrainer`.
- `save_epoch_checkpoint` delegates to `DenseCNNCheckpointSaver`.
- `evaluate_epoch` runs dense CNN versus SealBot through `hexo_runner`.

## 9. Dense CNN Architecture

Important files:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/architecture.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/constants.py`

The model is `Model1Network`.

Input shape:

```text
batch x 13 x 41 x 41
```

Main parts:

- `HexConv2d`: a 3x3 convolution whose invalid square-grid hex corners are
  masked out.
- `GatedResBlock`: residual block where the main branch is multiplied by a
  learned sigmoid gate.
- `PolicyHead`: dense policy logits over all `41 * 41 = 1681` crop cells.
- `ValueBinnedHead`: 65-bin value distribution over `[-1, 1]`.
- `opp_policy_head`: auxiliary prediction of the next opponent MCTS policy.
- `lookahead_heads`: auxiliary binned values for configured horizons.

Default production config:

```text
13 input planes
96 trunk channels
6 gated residual blocks
41x41 crop
65 value bins
policy + value + opp_policy + lookahead heads
```

`forward(x)` returns:

```python
{
    "policy": Tensor[batch, 1681],
    "value": Tensor[batch, 65],
    "opp_policy": Tensor[batch, 1681],
    "lookahead_1": Tensor[batch, 65],
    "lookahead_4": Tensor[batch, 65],
    "lookahead_8": Tensor[batch, 65],
}
```

`forward_policy_value(x)` is the inference/search path and returns only:

```python
{
    "policy": Tensor[batch, 1681],
    "value": Tensor[batch, 65],
}
```

## 10. Dense CNN Input Planes

Important files:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/input.py`
- `packages/hexo_models/dense_cnn/rust/src/lib.rs`

Dense CNN input is a model-owned 41x41 crop around the average occupied board
coordinate. The Python and Rust encoders are expected to match exactly.

Plane list:

| Index | Name | Meaning |
| --- | --- | --- |
| 0 | `OWN_STONES` | Stones belonging to the current player. |
| 1 | `OPPONENT_STONES` | Stones belonging to the opponent. |
| 2 | `EMPTY` | Empty cells in the crop. |
| 3 | `LEGAL` | Legal single-placement actions in the crop. |
| 4 | `SECOND_PLACEMENT` | Whole-plane marker for `SecondStone` phase. |
| 5 | `FIRST_STONE` | The first stone of the current two-stone turn. |
| 6 | `PLAYER_COLOUR` | Whole-plane marker when current player is player0. |
| 7 | `OWN_RECENCY` | Recency-weighted own placement history. |
| 8 | `OPPONENT_RECENCY` | Recency-weighted opponent placement history. |
| 9 | `OPPONENT_HOT` | Empty cells in opponent threat windows. |
| 10 | `OWN_HOT` | Empty cells in own threat windows. |
| 11 | `CENTER_DISTANCE` | Normalized hex distance from crop center. |
| 12 | `OPPONENT_LAST_TURN` | Coordinates from opponent's last logical turn. |

The Rust encoder is used for fast batched inference and MCTS. The Python
encoder is used only when expanding compact training samples for collation.

## 11. Inference

Important file:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/inference.py`

The central class is `DenseCNNInference`.

It owns:

- Moving the model to CPU or CUDA.
- Enabling AMP on CUDA if configured.
- Switching the model to eval mode.
- Building input tensors.
- Running batched forward passes.
- Decoding binned values.
- Masking policy logits down to legal actions.
- Returning legal-action priors.

### State Inference

```text
DenseCNNInference.infer_states(states)
  -> rust_bridge.model1_batch_inputs(states)
  -> hexo_models._rust.dense_cnn.model1_batch_inputs(states)
  -> Rust clones live engine states through the engine capsule
  -> Rust encodes planes, legal action IDs, legal flat indices
  -> Python creates torch tensor from bytes
  -> model forward
  -> decode value
  -> softmax policy logits only over legal flat indices
```

### MCTS Callback Inference

Rust MCTS does not call Python once per leaf. It batches leaf states and calls:

```text
DenseCNNInference.evaluate_model1_payload(payload)
```

Payload contains:

- Flat float32 input bytes.
- Shape `(batch, 13, 41, 41)`.
- Legal flat indices as bytes.
- Row offsets describing which legal indices belong to each state.

The callback:

1. Creates a torch tensor from the input bytes.
2. Runs `forward_policy_value`.
3. Decodes value logits into scalar values in `[-1, 1]`.
4. Gathers policy logits only at legal flat indices.
5. Runs a per-row softmax.
6. Returns raw bytes:

```python
{
    "values_bytes": ...,
    "priors_bytes": ...,
}
```

This keeps Python/Rust communication compact and lets the GPU see large leaf
batches.

## 12. Dense CNN MCTS Boundary

Important files:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/mcts.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/rust_bridge.py`
- `packages/hexo_models/dense_cnn/rust/src/lib.rs`

Dense CNN requires the model-local Rust accelerator for MCTS. There is no
Python fallback in this path.

Python entry points:

```python
run_mcts(root_state, inference, visits, c_puct=1.5, temperature=1.0, seed=None)
run_batched_mcts(root_states, inference, visits, c_puct=1.5, temperature=1.0, seed=None, virtual_batch_size=None)
```

`run_mcts` is just a single-root wrapper over `run_batched_mcts`.

`run_batched_mcts` sends this into Rust:

- Live root `hexo_engine.HexoState` objects.
- Target visit count.
- PUCT exploration constant.
- Root temperature.
- Seed.
- Evaluator callback: `inference.evaluate_model1_payload`.
- Virtual leaf batch size.

It returns `SearchResult`:

```python
SearchResult(
    action_id=int,
    visit_policy={action_id: normalized_visit_weight},
    root_value=float,
    visits=int,
)
```

## 13. How Dense CNN MCTS Works

The actual dense CNN search implementation lives under
`packages/hexo_models/dense_cnn/rust/src/`.

### Step 1: Clone Root States

Python passes live engine state objects to `hexo_models._rust.dense_cnn`.
Rust clones each root through the generic `hexo_engine._rust.state_api_capsule`
and then mutates only search-local copies.

### Step 2: Evaluate Roots

Before simulations start, Rust evaluates every root:

```text
evaluate_model1_states_cached
  -> encode_model1_state for each root
  -> call Python evaluator with batched tensor bytes
  -> receive scalar values and legal priors
```

The root node is initialized with:

- `player`: current player at that state.
- `visits = 1`
- `value_sum = root network value`
- one edge per legal action.

Each edge stores:

- `action_id`
- unpacked `HexCoord`
- normalized prior
- visit count
- value sum
- pending count
- optional child node ID

### Step 3: Select Leaves With PUCT

For each root, Rust repeatedly selects paths using:

```text
score = Q + prior * c_puct * sqrt(parent_visits) / (1 + edge_visits)
```

where:

- `Q = edge.value_sum / edge.visits`, or `0` when unvisited.
- `prior` comes from the neural policy softmax over legal actions.
- `c_puct` defaults to `1.5`.

Tie-breaking prefers:

1. Higher score.
2. Lower visit count.
3. Lower action ID.

Each selected edge applies one engine placement to a cloned state while
descending. Because every edge is a single placement, the tree naturally
represents opening, first-stone, and second-stone states.

### Step 4: Virtual Leaf Batching

Dense CNN MCTS is batched across roots and across leaves.

The loop keeps a `completed` visit count for each root. For each root, it
selects up to:

```text
min(virtual_batch_size, target_visits - completed[root])
```

pending leaves before calling the neural evaluator.

When a path is selected:

- `apply_virtual_visit` immediately increments node and edge visits along the
  path.
- The root's completed count is incremented.
- If the selected leaf needs neural evaluation, the edge is marked `pending`.
- Pending unexpanded edges are skipped by later selection so the same leaf is
  not selected twice in the same virtual batch.

This is why dense CNN can ask Rust to search many active games and many leaves
per game while still producing exactly the configured visit count per searched
root.

### Step 5: Evaluate or Resolve Leaf

There are three leaf cases:

1. Terminal leaf: use exact terminal value, `+1` for winner and `-1` for loser.
2. Existing child: reuse the child node's current mean value.
3. New non-terminal leaf: batch it for neural evaluation, create a child node,
   attach it to the parent edge, then back up the child value.

Leaf evaluation is cached by a model-visible state hash derived from the cloned
engine state. Placement order is part of the identity because dense CNN recency
planes make board-equivalent histories distinct.

### Step 6: Backup Values

Model values are always from the evaluated state's current-player perspective.
MCTS stores node values from each node's player-to-act perspective.

Backup therefore flips the sign only when needed:

```text
if node.player == leaf_player:
    value = leaf_value
else:
    value = -leaf_value
```

This is important for Hexo's two-stone turns. You should not blindly flip value
after every edge. During `FirstStone -> SecondStone`, the same player is still
to act, so the perspective does not change.

### Step 7: Produce Root Policy and Action

At the end, Rust builds:

- `visit_policy`: normalized root edge visits.
- `root_value`: root mean value.
- `visits`: sum of root edge visits.
- `action_id`: selected root action.

Action selection:

- If `temperature <= 1e-6`, choose the max-visit action with deterministic
  tie-breaking.
- Otherwise sample from visit counts raised to `1 / temperature`.

Self-play uses the configured temperature, currently `1.0`.
Evaluation through `DenseCNNPlayer` uses `temperature = 0.0`.

## 14. Dense CNN MCTS

Dense CNN's production path calls the model-local Rust accelerator:

```text
hexo_models.dense_cnn.mcts
  -> hexo_models.dense_cnn.rust_bridge
  -> hexo_models._rust.dense_cnn.model1_batched_mcts
```

The reason is practical: dense CNN needs model-specific 13-plane/41x41
encoding, packed tensor-byte communication, batched leaf evaluation, and sample
generation that all match its architecture exactly.

## 15. Dense CNN Self-Play

Important file:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/selfplay.py`

Dense CNN self-play is model-owned and currently does not use the generic
runner loop for each move. It uses `hexo_engine` directly and writes `.hxr`
records itself.

The reason is throughput: self-play keeps many active games resident and sends
their current states to the Rust batched MCTS bridge.

High-level flow:

```text
generate_selfplay_epoch
  -> build DenseCNNInference
  -> create output selfplay/epoch_000001.hxr
  -> maintain active game list
  -> search positions until sample budget is covered
  -> apply selected actions through hexo_engine
  -> finish remaining game tails with policy rollouts
  -> write replay records
  -> finalize pending samples after each game
  -> append compressed samples to SampleBuffer
  -> write diagnostics and optional previews
```

Each active game dictionary contains:

```python
{
    "game_id": str,
    "seed": int,
    "state": engine.HexoState,
    "pending": list,
    "actions": list[int],
}
```

Self-play starts new games while:

- active game count is below `active_games` or calibrated batch size;
- max games has not been reached;
- sample budget is not already covered by finalized plus pending samples.

For playable games:

1. Some games are searched with MCTS.
2. Extra playable games after the sample budget is covered use model policy
   rollout actions, not MCTS.

The model policy rollout path:

```text
_policy_rollout_actions
  -> inference.infer_states(playable states)
  -> sample one action from legal_priors
```

This finishes games so terminal outcomes are known without spending MCTS on
positions that will not become training samples.

## 16. Self-Play Sample Generation

Important files:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/samples.py`
- `packages/hexo_models/dense_cnn/rust/src/sample_generation.rs`

Samples are created from the live engine state before the selected action is
applied. Python does not scrape generic state or rebuild history rows.

When MCTS returns a searched decision, self-play does:

```text
sample_from_state(
    state,
    game_id=...,
    turn_index=...,
    policy=search.visit_policy,
    value=search.root_value,
    metadata={...}
)
```

That state is cloned through the engine capsule. Rust emits compact model facts:

- `game_id`
- `turn_index`
- `current_player`
- `phase`
- `center`
- `stones`
- `legal_action_ids`
- `placement_history`
- `first_stone`
- `own_hot`
- `opponent_hot`
- `opponent_last_turn`
- `policy`
- `opp_policy`
- `value`
- `lookahead`
- `metadata`

During the game, the sample is stored as pending:

```python
(sample.current_player, sample, search.root_value)
```

It cannot receive its final value target until the game outcome is known.

## 17. Finalizing Samples

After a game completes or truncates, dense CNN finalizes its pending samples:

```text
finalize_game_samples(pending, winner, horizons, truncated)
  -> hexo_models._rust.dense_cnn.model1_finalize_game_samples(...)
```

Finalization does three important things.

First, it sets final value:

```text
winner == sample_player -> +1
winner != sample_player -> -1
no winner/truncated -> 0
```

Second, it sets `opp_policy` from the next future pending decision made by the
opponent. If no future opponent MCTS policy exists, the field remains empty and
metadata records:

```text
opp_policy_source = "none"
```

During training expansion, an empty auxiliary policy remains an empty target,
so it contributes no auxiliary policy loss instead of fabricating a policy.

Third, it sets lookahead values for configured horizons, for example
`1`, `4`, and `8`. A lookahead target uses a future root value adjusted into
the current sample player's perspective. If the future sample is the last one
and the game has a winner, it uses the final outcome value.

Finalized samples are appended to the dense CNN `SampleBuffer`.

## 18. Dense CNN Replay Buffer

Important file:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/samples.py`

Dense CNN uses an in-memory compressed replay buffer:

```python
SampleBuffer(capacity=200_000, recency_halflife=50_000.0, compression_level=6)
```

Each entry is:

```python
CompressedSample(
    payload=zlib_compressed_json,
    uncompressed_bytes=int,
)
```

The buffer:

- Enforces a capacity floor of 200,000.
- Compresses each `Model1SampleData` as sorted compact JSON plus zlib.
- Drops oldest samples when over capacity.
- Samples without replacement.
- Uses recency weighting:

```text
weight(age) = exp(-ln(2) * age / recency_halflife)
```

The buffer is checkpointed with the model, so resume can restore both model
weights and replay history.

One subtle point: `hexo_train` still opens a generic `hexo_utils.samples`
sample store during initialization. Dense CNN's trainer overrides sample
selection and trains from its model-owned `SampleBuffer`, not from generic
sample-store chunks.

## 19. Expanding Samples for Training

Important files:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/samples.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/input.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/d6.py`

Training does not store dense tensors in the replay buffer. It stores compact
facts, then expands them on demand:

```text
expand_sample(sample, symmetry=random D6)
  -> build_input_planes(...)
  -> dense_policy_target(...)
  -> dense opp_policy target
  -> legal_mask_flat(...)
  -> scalar value target
  -> lookahead scalar targets
```

Output keys:

```python
{
    "input": Tensor[13, 41, 41],
    "policy": Tensor[1681],
    "opp_policy": Tensor[1681],
    "legal_mask": Tensor[1681],
    "value": scalar Tensor,
    "lookahead_1": scalar Tensor,
    ...
}
```

Policy targets are dense 1681-cell vectors. If the stored policy is empty,
`dense_policy_target` falls back to a uniform distribution over legal action
IDs that map into the crop.

D6 symmetry is applied before tensor construction. The trainer picks a fresh
random transform per sample expansion.

## 20. Training

Important files:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/trainer.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/losses.py`

Dense CNN training is driven by `DenseCNNTrainer`.

### Sample Selection

`select_training_samples` draws from the model buffer:

```text
requested = ctx.config.samples.train_sample_count
records = buffer.sample(requested, seed=run_seed + epoch)
components.shared.sample_window = DenseSampleWindow(...)
```

It returns diagnostics:

- total buffer count;
- selected window size;
- requested count;
- recency halflife.

### Training Passes

`train_passes` receives:

- `passes`
- `sample_window`
- `sample_symmetries`
- `ctx`
- `components`
- `epoch`

Then it:

1. Switches model to train mode.
2. Iterates records in batches.
3. Decodes each compressed sample.
4. Samples random D6 augmentation.
5. Expands sample to dense tensors.
6. Stacks tensors into a batch.
7. Runs model forward.
8. Computes `model1_loss`.
9. Uses AMP scaler when CUDA AMP is enabled.
10. Clips gradients if configured.
11. Steps AdamW.
12. Writes optional policy-target previews.

### Losses

`model1_loss` combines:

| Component | Target | Default weight |
| --- | --- | --- |
| `policy` | MCTS visit distribution over dense crop cells | `1.0` |
| `value` | 65-bin soft target from final scalar value | `1.0` |
| `opp_policy` | Future opponent MCTS policy when available | `0.25` |
| `lookahead_*` | 65-bin future-value targets | `0.25` |

Values are represented with 65 bins over `[-1, 1]`. `decode_binned_value`
computes the expected scalar value from the softmax distribution.

## 21. Checkpointing

Important files:

- `packages/hexo_train/python/hexo_train/checkpoints.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/checkpoints.py`

`hexo_train` decides when to load and save. Dense CNN decides what a checkpoint
contains.

Dense CNN checkpoint payload:

```python
{
    "model": "hexo_models.dense_cnn",
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "sample_buffer": buffer.state_dict(),
    "epoch": int | None,
    "metadata": {
        "run": run_name,
        "sample_count": buffer.sample_count,
    },
}
```

Checkpoint saves:

- Epoch checkpoint: `runs/.../checkpoints/epoch_000001.pt`
- Final checkpoint: `runs/.../checkpoints/latest.pt`, or whatever
  `[checkpoint] save_name` specifies.

Pointer files are plain text files containing the checkpoint path. The dense
config uses:

```text
data/checkpoints/dense_cnn_model1_latest.txt
```

On load, a pointer file may point to a relative or absolute checkpoint path.
If the pointer does not exist yet, dense CNN initializes from scratch.

## 22. Performance Calibration

Important file:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/performance.py`

If enabled, calibration runs before epochs.

It benchmarks:

- Inference batch sizes.
- Training batch sizes.
- Self-play batch sizes.
- MCTS virtual leaf batch sizes.

It restores the original model and optimizer state afterward, so calibration
does not train the model.

The selected values are written back onto the trainer:

- `trainer.inference_batch_size`
- `trainer.selfplay_batch_size`
- `trainer.mcts_virtual_batch_size`
- `trainer.training_batch_size`

Calibration success requires:

- Measured self-play positions per second meets the configured target.
- Search uses the exact configured visit count.
- All searches report exact visits.

The production config pins MCTS visit candidates to `[128]`, so calibration
tunes batching without silently reducing search strength.

## 23. Runner Interaction

Dense CNN interacts with `hexo_runner` in two distinct ways.

### Self-Play Path

Dense CNN self-play does not currently call `hexo_runner.loop.run_match_loop`.

Instead:

```text
dense_cnn.selfplay.generate_selfplay_epoch
  -> engine.new_game
  -> dense_cnn.mcts.run_batched_mcts
  -> engine.apply_action
  -> HexoRecordFile.create(...).begin_game(...).record_action(...)
```

It still writes `.hxr` records using the same record layer:

```text
hexo_runner.records.HexoRecordFile
```

So self-play records remain replayable and compatible with runner/frontend
record tooling, even though the move loop is custom.

### Evaluation/Match Path

Evaluation uses the generic runner.

Important files:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/evaluation.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/player.py`
- `packages/hexo_runner/python/hexo_runner/loop.py`
- `packages/hexo_runner/python/hexo_runner/modes/match.py`
- `packages/hexo_runner/python/hexo_runner/adapters/sealbot.py`

Evaluation flow:

```text
dense_cnn.evaluation.evaluate_epoch
  -> validate SealBot config
  -> for each evaluation game:
       create DenseCNNPlayer
       create SealBotPlayer
       alternate player0/player1 roles
       run_match(GameSpec(...), players, output_dir)
```

The generic runner flow:

```text
run_match
  -> HexoRecordFile.create(...)
  -> run_match_loop
       -> engine.new_game
       -> player.start_game for both players
       -> while not terminal:
            current = engine.current_player(primary_state)
            cloned_state = engine.clone_state(primary_state)
            decision = active_player.decide(cloned_state)
            engine.apply_action(primary_state, decision.action)
            record action
            notify both players with cloned observer states
       -> finish completed or aborted record
       -> player.finish_game
       -> player.close
```

`DenseCNNPlayer.decide` does:

```text
run_mcts(state, inference, visits=config.selfplay.search_visits, temperature=0.0)
  -> PlacementAction(unpack_coord_id(search.action_id))
  -> DecisionResult(action, diagnostics={root_value, visits, model})
```

The runner only receives a normal `DecisionResult`; it does not know how dense
CNN searched.

## 24. SealBot Evaluation

Important files:

- `packages/hexo_runner/python/hexo_runner/adapters/sealbot.py`
- `packages/hexo_runner/python/hexo_runner/adapters/_sealbot_worker.py`

SealBot is an external opponent. The adapter starts a subprocess because the
`current` and `best` pybind variants export the same module/class names and
cannot both be imported safely in one Python process.

Config comes from:

- `SEALBOT_PATH`
- `sealbot_variant`
- `sealbot_time_limit`
- `require_sealbot`

`SealBotPlayer.decide` converts the engine's Python state mirror into a compact
payload:

- current player;
- phase;
- moves left in turn;
- placement count;
- terminal winner;
- stones.

SealBot may return a two-move turn. The adapter buffers those moves and returns
one `PlacementAction` per runner decision, matching the engine's single-stone
action model.

If `require_sealbot = true`, missing SealBot evaluation raises and fails the
training run instead of silently skipping Goal 4 evaluation.

## 25. Run Artifacts

A dense CNN run writes several types of artifacts.

Typical structure:

```text
runs/dense_cnn_model1/
  manifest.json
  checkpoints/
    epoch_000001.pt
    latest.pt
  diagnostics/
    config.normalized.json
    initialize_run.json
    load_checkpoint.json
    calibrate_performance.json
    dense_cnn.performance_calibration.json
    epoch_000001.json
    dense_cnn.selfplay.epoch_000001.json
    dense_cnn.game_history.epoch_000001.json
    dense_cnn.policy_targets.epoch_000001.json
    dense_cnn.evaluation.epoch_000001.json
    events.jsonl
    dense_cnn_previews/
  selfplay/
    epoch_000001.hxr
  evaluation/
    epoch_000001/
      eval-000001-0000.hxr
      ...
  samples/
    manifest.json
    chunks/
```

Note again that `samples/` is the shared sample-store directory. Dense CNN's
actual training replay data is currently stored in the dense CNN checkpointed
`SampleBuffer`.

## 26. Dense CNN End-to-End Call Graph

This is the most compact mental model of the whole path:

```text
CLI
  -> TrainingPipeline.run(config)
     -> load_model_plugin("hexo_models.dense_cnn.plugin")
     -> DenseCNNPlugin.build_model
        -> Model1Network
     -> DenseCNNPlugin.training_component_overrides
        -> SampleBuffer
        -> AdamW
        -> DenseCNNTrainer
        -> checkpoint loader/saver
     -> load_or_initialize_checkpoint
     -> DenseCNNPlugin.calibrate_performance
     -> run_epochs
        -> generate_selfplay
           -> DenseCNNInference
           -> dense_cnn.selfplay active game loop
              -> run_batched_mcts
                 -> hexo_models._rust.dense_cnn.model1_batched_mcts
                    -> clone live engine states through the capsule
                    -> encode_model1_state
                    -> Python evaluator callback
                    -> PUCT search with virtual batches
                    -> visit policy + selected action
              -> sample_from_state
              -> engine.apply_action
              -> write .hxr record
              -> finalize_game_samples
              -> SampleBuffer.extend
        -> finalize_samples
           -> DenseCNNSampleFinalizer reports buffer state
        -> select_training_samples
           -> SampleBuffer.sample recency-weighted window
        -> select_epoch_symmetries
           -> metadata only
        -> train_passes
           -> expand compressed samples with random D6
           -> model forward
           -> model1_loss
           -> optimizer step
        -> save_epoch_checkpoint
           -> model_state + optimizer_state + sample_buffer
        -> evaluate_epoch
           -> run_match(DenseCNNPlayer vs SealBotPlayer)
              -> DenseCNNPlayer.decide
                 -> run_mcts single root
                    -> same Rust batched MCTS bridge
     -> save_final_checkpoint
     -> publish checkpoint pointer
     -> final diagnostics
```

## 27. Where To Read First

For dense CNN, read these files in this order:

1. `configs/dense_cnn_model1.toml`
2. `packages/hexo_train/python/hexo_train/pipeline.py`
3. `packages/hexo_train/python/hexo_train/epoch/loop.py`
4. `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/plugin.py`
5. `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/selfplay.py`
6. `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/mcts.py`
7. `packages/hexo_models/dense_cnn/rust/src/lib.rs`
8. `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/inference.py`
9. `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/samples.py`
10. `packages/hexo_models/dense_cnn/rust/src/sample_generation.rs`
11. `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/trainer.py`
12. `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/evaluation.py`
13. `packages/hexo_runner/python/hexo_runner/loop.py`
14. `packages/hexo_engine/rust/src/state.rs`

The tests are also useful because they encode many intended contracts:

- `tests/test_dense_cnn_pipeline.py`
- `tests/test_dense_cnn_performance.py`
- `tests/test_dense_cnn_sample_generation.py`
- `tests/test_hexo_runner_match_mode.py`
- `tests/test_hexo_engine_rust_bridge.py`

## 28. Common Points Of Confusion

### Does dense CNN self-play use the runner?

Not for the high-throughput self-play loop. It uses `hexo_engine` directly,
calls dense CNN MCTS directly, and writes `.hxr` records itself.

Dense CNN does use the generic runner for evaluation/matches through
`DenseCNNPlayer`.

### Is one MCTS edge a whole turn?

No. One edge is one placement. A two-stone turn is represented by two edges.

### Does MCTS flip value after every edge?

No. It flips value only when the player-to-act perspective changes. This is
essential because the same player acts in both `FirstStone` and `SecondStone`.

### Why pass live state objects into Rust?

The production path is direct state handoff. Python passes live
`hexo_engine.HexoState` objects, model-owned Rust clones them through the
generic engine capsule, and search/sample generation mutate only those local
copies. This keeps the engine generic while avoiding history replay.

### Where are dense CNN samples stored?

During training they live in `SampleBuffer` as compressed JSON blobs in memory
and inside checkpoints. The generic `samples/` directory is opened by the
shared pipeline, but dense CNN training currently samples from its own replay
buffer.

### What is `opp_policy`?

It is an auxiliary target for the next opponent MCTS policy seen later in the
same game. If no future opponent MCTS decision is available, training falls
back to uniform legal policy.

### What is `lookahead`?

It is an auxiliary value target at future sample offsets such as 1, 4, and 8.
The future value is converted into the current sample player's perspective.

### Why is MCTS/sample generation Rust-only?

The production search and self-play sample path is model-owned Rust. Python
keeps PyTorch execution, training orchestration, checkpointing, and tensor
collation, but it no longer contains alternate MCTS or history-rebuild sample
generation logic.

### Why does calibration require exact visits?

The baseline goal is fixed-strength search: exactly 128 MCTS simulations per
searched position. Calibration can tune batching, but it must not make the run
look faster by reducing search work.
