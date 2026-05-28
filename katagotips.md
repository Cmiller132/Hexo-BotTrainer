Yes. I would rewrite the loop around a **KataGo-style synchronous cycle**:

```text
self-play games to terminal
→ write finalized compact samples
→ build/refresh shuffled training window
→ train up to a train-per-new-data budget
→ checkpoint/export current model
→ repeat
```

The key change is to stop making the dense CNN loop target “N MCTS samples” and then roll out the rest of games cheaply. That rule is efficient, but it is also the main source of odd learning behavior.

KataGo’s one-machine synchronous mode still has distinct self-play, shuffle, train, export, and optional gatekeeper phases; the script simply runs those phases sequentially and stops each phase after bounded work. The shuffler and train-bucket limits are not incidental details; they are central to how training data is randomized and how much the model is allowed to train per amount of new data. ([GitHub][1]) ([GitHub][2])

Below is the rewrite I would propose.

---

# Proposal: rewrite dense CNN training into a synchronous KataGo-like cycle

## Core goals

The rewrite should do four things:

1. **Remove `mcts_until_sample_budget_then_model_policy_rollout`.**
2. **Make self-play game-count based, not sample-budget based.**
3. **Separate replay/shuffle/training-budget logic from self-play.**
4. **Train more like KataGo: shuffled recent window, train-per-new-data budget, policy-surprise weighting, and short-term value targets.**

The dense CNN can still run sequentially on one consumer machine. “More like KataGo” does **not** mean adding async workers or distributed infrastructure. It means making the data lifecycle look like:

```text
generate complete games
store data
shuffle/window data
train a controlled amount
publish next model
```

rather than:

```text
generate exactly N searched positions
finish tails differently
train a tiny random slice
checkpoint replay buffer inside model
```

---

# Current issues to fix

## 1. Self-play is sample-budget-driven instead of game-driven

Current dense CNN self-play sets `target_samples = config.selfplay.samples_per_epoch`, then starts active games only while it needs more samples. Once enough pending samples exist, extra playable games go through policy rollout instead of MCTS. This is recorded as `"mcts_until_sample_budget_then_model_policy_rollout"`.    

That creates two different game regimes inside one self-play epoch:

```text
searched prefix: MCTS policy/value samples
tail: model-policy rollout only for terminal result
```

This makes the value target depend on cheaper rollout behavior after the sampled part of the game. It also underrepresents later-game positions.

### Rewrite

Replace the sample-budget loop with a game-count loop:

```text
for epoch/cycle:
    start up to games_per_epoch games
    keep active games batched
    for every non-terminal active game:
        run MCTS
        record the position sample
        apply the selected action
    when game terminal:
        finalize every sample in that game
        append finalized samples to replay store
```

There should be no `rollout_games` branch. If a game is active, every decision is MCTS-searched until terminal or `max_actions`.

### File impact

Rewrite:

```text
packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/selfplay.py
```

Replace this shape:

```python
remaining_samples = target_samples - samples_added - pending_count
search_games = playable[:remaining_samples]
rollout_games = playable[remaining_samples:]
```

with:

```python
search_games = playable
rollout_games = []
```

but not as a one-line patch. The surrounding loop should be rewritten around `completed_games < games_per_epoch`, not `samples_added < target_samples`.

### New self-play loop sketch

```python
def generate_selfplay_epoch(ctx, components, epoch, games_per_epoch):
    trainer = components.model.trainer
    config = trainer.config

    inference = DenseCNNInference(...)
    mcts_session = new_mcts_session(...)
    active = []
    completed_games = 0
    next_game_index = 0
    finalized_samples = []

    with HexoRecordFile.create(record_path, engine.engine_metadata(), players) as record_file:
        while completed_games < games_per_epoch or active:
            while (
                len(active) < config.selfplay.max_active_games
                and next_game_index < games_per_epoch
            ):
                active.append(new_active_game(epoch, next_game_index))
                next_game_index += 1

            playable = [
                game for game in active
                if engine.terminal(game.state) is None
                and len(game.actions) < config.selfplay.max_actions
            ]

            if playable:
                searches = mcts_session.run(
                    game_keys=[g.search_key for g in playable],
                    root_states=[g.state for g in playable],
                    inference=inference,
                    visits=config.selfplay.search_visits,
                    temperature=temperature_for_position(...),
                    root_policy_temperature=root_policy_temperature_for_position(...),
                    root_dirichlet_total_alpha=config.selfplay.root_dirichlet_total_alpha,
                    root_dirichlet_noise_fraction=config.selfplay.root_dirichlet_noise_fraction,
                )

                for game, search in zip(playable, searches):
                    sample = sample_from_state(
                        game.state,
                        game_id=game.game_id,
                        turn_index=len(game.actions),
                        policy=search.visit_policy,
                        value=search.root_value,
                        root_prior=search.root_prior_policy,
                        policy_surprise=search.policy_surprise,
                        metadata={...},
                    )
                    game.pending.append((sample.current_player, sample, search.root_value))
                    action = engine.PlacementAction(unpack_coord_id(search.action_id))
                    engine.apply_action(game.state, action)
                    game.actions.append(search.action_id)

            for game in finished_games(active):
                finalized = finalize_game_samples(
                    game.pending,
                    winner=winner,
                    shortterm_value_lambdas=config.architecture.shortterm_value_lambdas,
                    truncated=truncated,
                )
                append_samples_to_store(finalized)
                finalized_samples.extend(finalized)
                write_hxr_record(game)
                active.remove(game)
                completed_games += 1
                mcts_session.discard(game.search_key)

    return {
        "games": completed_games,
        "samples_generated": len(finalized_samples),
        "record_path": str(record_path),
        ...
    }
```

This is simpler conceptually. It removes:

```text
target_samples
remaining_samples
search_games vs rollout_games
min_mcts_samples_per_game
sample_depth_active_limit
completion_rollout
```

The hardware knob becomes **how many games and visits per cycle**, not “how many positions before changing game-completion policy.”

---

# 2. Replace `samples_per_epoch` with `games_per_epoch`

Current config has both top-level `[selfplay] games_per_epoch = 4096` and model-level `[model.config.selfplay] samples_per_epoch = 65536`. The guide even says they are related but not identical. 

That split is confusing and drives the unusual loop.

### Rewrite

Delete:

```toml
[model.config.selfplay]
samples_per_epoch = 65536
min_mcts_samples_per_game = 32
```

Use only:

```toml
[selfplay]
games_per_epoch = 256
```

or, if you want all dense settings under the model config:

```toml
[model.config.selfplay]
games_per_epoch = 256
```

but do not keep both top-level and model-level concepts.

Recommended config shape:

```toml
[model.config.selfplay]
games_per_epoch = 256
max_active_games = 256
search_visits = 128
max_actions = 1024
temperature_initial = 1.0
temperature_final = 0.25
temperature_decay_actions = 128
root_policy_temperature_initial = 1.25
root_policy_temperature_final = 1.10
root_policy_temperature_decay_actions = 128
root_dirichlet_total_alpha = 10.83
root_dirichlet_noise_fraction = 0.25
```

Why `root_dirichlet_total_alpha` instead of fixed per-action alpha? KataGo describes using a total alpha divided across legal moves rather than blindly using the same alpha per move across different move counts. ([GitHub][3])

The exact numbers can be tuned, but the conceptual API should be:

```text
games per cycle
active games for batching
MCTS visits
root exploration settings
temperature schedule
```

not:

```text
desired samples
minimum samples per active game
policy rollout tail mode
```

---

# 3. Remove progressive widening as a training behavior rule

Current config contains:

```toml
progressive_widening_initial_actions = 8
progressive_widening_child_initial_actions = 4
progressive_widening_candidate_actions = 128
progressive_widening_growth_interval = 256.0
progressive_widening_growth_base = 1.3
hidden_prior_mass = 0.05
```

These are strong arbitrary behavioral rules. They may be useful for performance, but they are not the cleanest learning behavior. They decide which moves are even eligible early in search based on a schedule unrelated to the model’s learning target. 

You asked to ignore out-of-crop moves. Under that assumption, the cleanest rewrite is:

```text
MCTS candidate set = all legal actions represented by the dense policy.
```

Implementation can still be lazy for memory, but the behavior should be equivalent to all legal candidates being available from the start.

### Rewrite

Keep compact/lazy edge materialization internally, but remove visit-growth unpruning. In Rust terms:

* Keep `RustNode.unexpanded_priors`.
* Keep materializing an edge only when selected.
* But remove `ProgressiveWideningConfig` from training behavior.
* At selection time, compare existing edges against the best unmaterialized prior candidate.
* Since all unvisited hidden candidates have the same visit count and value default, the highest-prior hidden candidate is enough for exact PUCT selection among hidden moves.

### File impact

Rewrite or simplify:

```text
packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs
packages/hexo_models/dense_cnn/rust/src/mcts.rs
packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/mcts.py
packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/config.py
```

Remove from public config:

```text
progressive_widening_initial_actions
progressive_widening_child_initial_actions
progressive_widening_candidate_actions
progressive_widening_growth_interval
progressive_widening_growth_base
hidden_prior_mass
```

Replace with:

```toml
[model.config.selfplay]
max_policy_candidates = 0  # 0 means no cap; all in-crop legal candidates
```

If consumer hardware needs a cap later, make it explicit:

```toml
max_policy_candidates = 1024
```

But do not combine a cap with a visit-dependent growth rule unless measurements show it is necessary.

---

# 4. Add root policy softmax temperature

KataGo applies a root-only policy softmax temperature above 1 to counter premature policy collapse in nearly equal moves. The docs describe a root temperature around 1.25 early, decaying toward 1.1 later, before adding Dirichlet noise or using the root prior in search. ([GitHub][3])

Hexo currently has root Dirichlet noise, but no root prior softening step.

### Rewrite

Add root-prior temperature to MCTS root creation:

```text
raw policy logits/probs
→ root policy softmax temperature
→ Dirichlet noise
→ MCTS prior
```

For non-root nodes, use normal policy priors.

### Config

```toml
[model.config.selfplay]
root_policy_temperature_initial = 1.25
root_policy_temperature_final = 1.10
root_policy_temperature_decay_actions = 128
```

For a position with `placement_count`:

```python
def root_policy_temperature(placement_count):
    t0 = config.root_policy_temperature_initial
    t1 = config.root_policy_temperature_final
    decay = config.root_policy_temperature_decay_actions
    return t1 + (t0 - t1) * exp(-placement_count / decay)
```

### File impact

Add the value to:

```text
DenseCNNTrainer
DenseCNN self-play call into MCTS
Python mcts.py signature
Rust Model1MctsSession.search
node_from_evaluation for root nodes
```

---

# 5. Add policy-surprise weighting

KataGo overweights samples where the MCTS policy target is surprising relative to the neural policy prior. The docs describe redistributing frequency weight so part of the mass is uniform and part is proportional to KL divergence from prior to target; surprising positions are seen more often rather than having their gradients downscaled. ([GitHub][3])

This is more important than it may look. Without it, the model spends too much training on “ordinary” positions where the net already agrees with search, and not enough on the positions where MCTS discovered that the net was wrong.

### Rewrite

Every `SearchResult` should include:

```python
root_prior_policy: Sequence[tuple[action_id, prior]]
visit_policy: Sequence[tuple[action_id, target]]
policy_surprise: float  # KL(target || prior)
sample_weight: float
```

Compute:

```python
kl = sum(target[a] * log((target[a] + eps) / (prior[a] + eps)) for a in target)

sample_weight = (
    policy_surprise_baseline
    + policy_surprise_scale * normalized_kl
)
```

A KataGo-like default:

```toml
[model.config.samples]
policy_surprise_enabled = true
policy_surprise_uniform_fraction = 0.5
policy_surprise_max_weight = 8.0
```

Implementation detail: do **not** physically duplicate samples. Store `sample_weight` in `Model1SampleData.metadata` and use it in replay sampling:

```text
effective_sampling_weight =
    recency_weight
    * sample_weight
```

This is simpler than writing duplicate records and still gives the same practical effect: surprising samples appear more often in training.

### File impact

Modify:

```text
packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/mcts.py
packages/hexo_models/dense_cnn/rust/src/mcts.rs
packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/samples.py
packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/trainer.py
```

Current `SearchResult` only has action ID, visit policy, root value, visits, and diagnostics.  It should become:

```python
@dataclass(frozen=True, slots=True)
class SearchResult:
    action_id: int
    visit_policy: Sequence[tuple[int, float]]
    root_prior_policy: Sequence[tuple[int, float]]
    root_value: float
    visits: int
    policy_surprise: float
    sample_weight: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
```

---

# 6. Replace fixed lookahead targets with exponential short-term value targets

Current dense CNN has `lookahead_horizons = [1, 4, 8]`, and finalization fills targets from future root values at those specific future sample offsets.  

That is inspired by KataGo, but less like it. KataGo trains auxiliary value targets that exponentially average future MCTS values over several horizons, which gives lower-variance feedback than only final outcome or a few exact offsets. ([GitHub][3])

### Rewrite

Replace:

```toml
lookahead_horizons = [1, 4, 8]
```

with:

```toml
shortterm_value_mean_actions = [4, 12, 32]
```

For each mean horizon `m`, compute:

```python
lambda = m / (m + 1)
target_t = weighted_average(
    future_root_values[t:],
    weights=(1-lambda) * lambda**k
)
```

Perspective must still be adjusted to the sample player, exactly as current finalization already handles perspective.

### Sample schema

Replace:

```python
lookahead: tuple[tuple[int, float], ...]
```

with either:

```python
shortterm_value: tuple[tuple[int, float], ...]
```

or keep the field name for compatibility but change semantics:

```python
lookahead = (
    (4, exp_avg_future_value_mean_4),
    (12, exp_avg_future_value_mean_12),
    (32, exp_avg_future_value_mean_32),
)
```

I would rename it to avoid confusion:

```python
shortterm_value_targets: tuple[tuple[int, float], ...]
```

### Model head names

Current model returns:

```python
lookahead_1
lookahead_4
lookahead_8
```

Change to:

```python
shortterm_value_4
shortterm_value_12
shortterm_value_32
```

or keep the head prefix as `lookahead_4`, but document that it is now exponential.

### File impact

Modify:

```text
architecture.py
config.py
samples.py
losses.py
trainer.py
rust sample finalization
```

Current `Model1Network` builds `lookahead_heads` from configured horizons.  That mechanism can stay; only the target semantics need to change.

---

# 7. Create a real replay/shuffle/training budget layer

Current dense CNN training samples directly from the in-memory `SampleBuffer`. The buffer is checkpointed with the model, samples are recency weighted, and the trainer draws `train_sample_count` samples per epoch.  

This is much simpler than KataGo, but too simple. The current config generates 65,536 samples and trains only 4,096 rows per epoch.  

That is the wrong ratio for learning. The self-play search is expensive, but most labels barely influence the model.

KataGo’s synchronous script has explicit `NUM_TRAIN_SAMPLES_PER_EPOCH`, `MAX_TRAIN_PER_DATA`, shuffle keep rows, min rows, and max train samples per cycle. ([GitHub][2])

### Rewrite

Introduce a dense CNN replay manager:

```text
DenseReplayStore
DenseReplayIndex
DenseShuffleWindow
DenseTrainBucket
```

This can be implemented inside `hexo_models.dense_cnn.samples` or as new files:

```text
dense_cnn/replay.py
dense_cnn/shuffle.py
dense_cnn/train_budget.py
```

### New data flow

```text
self-play produces finalized samples
→ append samples to durable replay chunks
→ update replay manifest
→ build shuffled training window from recent samples
→ train until train bucket is empty
```

### Do not checkpoint the entire replay buffer with the model

Current checkpoints include `sample_buffer`. 

Rewrite checkpoints to contain:

```python
{
    "model_state": ...,
    "optimizer_state": ...,
    "epoch": ...,
    "train_state": {
        "total_generated_weight": ...,
        "total_trained_weight": ...,
        "shuffle_seed": ...,
        "last_replay_manifest": ...,
    },
}
```

Replay samples should live in:

```text
runs/dense_cnn_model1/replay/
  manifest.json
  chunks/
    epoch_000001_game_000000.json.zlib
    epoch_000001_game_000001.json.zlib
    ...
```

This makes checkpoints smaller and makes training data durable across restarts.

### Minimal first version

To keep this practical:

* Keep `CompressedSample`.
* Store compressed samples in chunk files.
* Manifest tracks `sample_count`, `sample_weight_sum`, `epoch`, `game_id`, `chunk_path`.
* `DenseReplayIndex` loads only metadata until training needs rows.
* `DenseShuffleWindow` stores selected sample references, not decoded tensors.

### Training budget formula

Track:

```python
total_generated_weight += sum(sample.sample_weight for new samples)
allowed_train_rows = max_train_samples_per_new_sample * total_generated_weight
remaining_train_rows = allowed_train_rows - total_trained_rows
train_rows_this_epoch = min(max_train_samples_per_epoch, remaining_train_rows)
```

Config:

```toml
[model.config.samples]
replay_capacity_rows = 1000000
shuffle_window_rows = 300000
min_rows_to_train = 32768
policy_surprise_enabled = true
recency_halflife = 200000.0

[model.config.training]
batch_size = 128
max_train_samples_per_epoch = 131072
max_train_samples_per_new_sample = 2.0
```

For a small consumer GPU, `max_train_samples_per_new_sample = 1.0` or `2.0` is a reasonable starting point. The important part is that the ratio is explicit.

### Trainer rewrite

Replace current selection:

```python
requested = ctx.config.samples.train_sample_count or self.config.samples.train_sample_count
records = self.buffer.sample(requested, seed=...)
```

with:

```python
budget = train_bucket.remaining_budget()
if replay_index.sample_count < min_rows_to_train:
    return skipped("not enough replay rows")

window = replay_store.build_shuffle_window(
    rows=shuffle_window_rows,
    seed=epoch_seed,
    weight_fn=recency * sample_weight,
)

records = window.draw_without_replacement(
    count=min(max_train_samples_per_epoch, budget),
    seed=epoch_seed,
)
```

Then training consumes exactly those rows and increments:

```python
train_state.total_trained_rows += len(records)
```

---

# 8. Make epochs mean “cycles,” not “one tiny train pass”

The current epoch order is good structurally:

```text
generate_selfplay
finalize_samples
select_training_samples
select_epoch_symmetries
train_passes
save_epoch_checkpoint
evaluate_epoch
```

That order is implemented in `hexo_train.epoch.loop`. 

I would keep the generic epoch skeleton but reinterpret each epoch as a **KataGo-style synchronous cycle**:

```text
cycle N:
    self-play N games to terminal
    append replay chunks
    refresh shuffle window
    train up to bucket budget
    save checkpoint
    evaluate if configured
```

### Suggested rename

Code can keep `epoch` for compatibility, but diagnostics should use:

```text
cycle
selfplay_games
samples_generated
train_rows_allowed
train_rows_used
train_to_new_data_ratio
```

This will make the loop easier to understand.

---

# 9. Simplify dense CNN config

Current dense config has too many interdependent knobs. Some are necessary, but the important behavior should be visible.

## Proposed config

```toml
[model]
name = "dense_cnn"
module = "hexo_models.dense_cnn.plugin"

[model.config]
device = "cuda"

[model.config.architecture]
input_channels = 13
channels = 64
residual_blocks = 4
crop_size = 41
dropout = 0.0
shortterm_value_mean_actions = [4, 12, 32]

[model.config.selfplay]
games_per_epoch = 256
max_active_games = 256
search_visits = 128
max_actions = 1024

# Move selection.
temperature_initial = 1.0
temperature_final = 0.25
temperature_decay_actions = 128

# Root exploration.
root_policy_temperature_initial = 1.25
root_policy_temperature_final = 1.10
root_policy_temperature_decay_actions = 128
root_dirichlet_total_alpha = 10.83
root_dirichlet_noise_fraction = 0.25

# Engineering limits only.
mcts_session_cache_max_states = 1048576
mcts_active_root_limit = 256
max_policy_candidates = 0  # 0 = all in-crop legal candidates

[model.config.samples]
replay_capacity_rows = 1000000
shuffle_window_rows = 300000
min_rows_to_train = 32768
recency_halflife = 200000.0
compression_level = 6

policy_surprise_enabled = true
policy_surprise_uniform_fraction = 0.5
policy_surprise_max_weight = 8.0

[model.config.training]
batch_size = 128
learning_rate = 0.001
weight_decay = 0.0001
policy_weight = 1.0
value_weight = 1.0
opp_policy_weight = 0.25
shortterm_value_weight = 0.25
amp = true
max_grad_norm = 1.0

# KataGo-like train bucket.
max_train_samples_per_epoch = 131072
max_train_samples_per_new_sample = 2.0

[model.config.evaluation]
games_per_epoch = 64
sealbot_variant = "best"
sealbot_time_limit = 0.05
max_actions = 1024
require_sealbot = true

[model.config.performance]
calibrate = true
inference_batch_candidates = [128, 256, 512, 1024]
selfplay_batch_candidates = [128, 256]
training_batch_candidates = [64, 128, 192, 256]
mcts_visit_candidates = [128]
mcts_virtual_batch_candidates = [4, 8, 16]

[run]
name = "dense_cnn_model1"
output_dir = "../runs/dense_cnn_model1"
seed = 1

[loop]
epochs = 30

[checkpoint]
resume_from = "../data/checkpoints/dense_cnn_model1_latest.txt"
save_name = "latest"
```

Removed:

```text
samples_per_epoch
min_mcts_samples_per_game
completion rollout mode
progressive widening growth knobs
hidden_prior_mass
duplicated top-level selfplay/sample counts
```

Added:

```text
games_per_epoch
max_active_games
root policy temperature
total-alpha Dirichlet noise
shuffle window
train bucket
policy surprise weighting
short-term value target horizons
```

---

# 10. Detailed file-by-file rewrite plan

## A. `dense_cnn/config.py`

### Current issue

`Model1SelfPlayConfig` mixes learning behavior, performance behavior, and odd sample-budget behavior. 

### Rewrite

Replace:

```python
samples_per_epoch
active_games
min_mcts_samples_per_game
progressive_widening_*
hidden_prior_mass
```

with:

```python
games_per_epoch: int
max_active_games: int
search_visits: int
max_actions: int

temperature_initial: float
temperature_final: float
temperature_decay_actions: int

root_policy_temperature_initial: float
root_policy_temperature_final: float
root_policy_temperature_decay_actions: int

root_dirichlet_total_alpha: float
root_dirichlet_noise_fraction: float

max_policy_candidates: int
mcts_session_cache_max_states: int
mcts_active_root_limit: int
```

Add sample/training budget config:

```python
class Model1SampleConfig:
    replay_capacity_rows: int
    shuffle_window_rows: int
    min_rows_to_train: int
    recency_halflife: float
    policy_surprise_enabled: bool
    policy_surprise_uniform_fraction: float
    policy_surprise_max_weight: float

class Model1TrainingConfig:
    ...
    max_train_samples_per_epoch: int
    max_train_samples_per_new_sample: float
```

---

## B. `dense_cnn/selfplay.py`

### Current issue

This is the main behavior problem. It currently runs MCTS only until sample budget is covered, then policy-rolls out tails. 

### Rewrite

Make it game-driven:

```python
while completed_games < games_per_epoch or active:
    fill_active_games()
    search_all_playable_games()
    apply_all_selected_actions()
    finalize_finished_games()
```

Delete:

```text
target_samples
samples_added < target_samples loop condition
sample_depth_active_limit
min_mcts_samples_per_game
remaining_samples
rollout_games
_policy_rollout_actions
completion_rollout diagnostics
```

Keep:

```text
batched MCTS
tree reuse session
.hxr record writing
pending sample finalization
debug previews
performance diagnostics
```

### New behavior

Every recorded position comes from the same process:

```text
MCTS search → MCTS policy target → MCTS root value → selected action
```

Every terminal outcome is produced by a game where every move was searched.

---

## C. `dense_cnn/mcts.py` and Rust MCTS

### Current issue

The API exposes progressive widening and hidden priors as core behavior. 

### Rewrite

Python API:

```python
run_mcts(
    root_state,
    inference,
    visits,
    temperature,
    root_policy_temperature,
    root_dirichlet_total_alpha,
    root_dirichlet_noise_fraction,
    max_policy_candidates=0,
)
```

Remove public arguments:

```text
progressive_widening_initial_actions
progressive_widening_child_initial_actions
progressive_widening_candidate_actions
progressive_widening_growth_interval
progressive_widening_growth_base
hidden_prior_mass
```

Rust:

* Root node gets all legal candidate priors.
* Root prior is temperature-softened.
* Dirichlet alpha per action is:

```text
alpha_per_action = root_dirichlet_total_alpha / legal_candidate_count
```

* Existing lazy edge materialization remains as an optimization.
* No visit-based progressive widening schedule.
* Return root prior policy and KL surprise.

### New result payload

```python
{
    "action_id": ...,
    "visit_policy": ...,
    "root_prior_policy": ...,
    "root_value": ...,
    "policy_surprise": ...,
    "sample_weight": ...,
    "visits": ...,
}
```

---

## D. `dense_cnn/samples.py`

### Current issue

Samples lack root prior, policy surprise, sample frequency weight, and proper short-term exponential value targets. Current `Model1SampleData` stores policy, opp policy, value, lookahead, and metadata. 

### Rewrite schema

```python
@dataclass(frozen=True, slots=True)
class Model1SampleData:
    game_id: str
    turn_index: int
    current_player: str
    phase: str
    center: tuple[int, int]
    stones: tuple[tuple[int, int, str], ...]
    legal_action_ids: tuple[int, ...]
    placement_history: ...
    first_stone: ...
    own_hot: ...
    opponent_hot: ...
    opponent_last_turn: ...

    policy: tuple[tuple[int, float], ...]
    root_prior_policy: tuple[tuple[int, float], ...]
    opp_policy: tuple[tuple[int, float], ...]

    value: float
    shortterm_value: tuple[tuple[int, float], ...]

    policy_surprise: float
    sample_weight: float
    metadata: Mapping[str, Any]
```

### Finalization rewrite

Current finalization delegates outcome logic to Rust.  Keep that, but change the target semantics:

```text
final value = final game result from sample player perspective
shortterm values = exponential averages of future MCTS root values
opp policy = next opponent MCTS policy
```

Because every move is now searched, short-term targets and opponent-policy targets should be more consistently available.

---

## E. New `dense_cnn/replay.py`

### Purpose

Replace checkpointed in-memory `SampleBuffer` as the main data source.

### API

```python
class DenseReplayStore:
    def append_game_samples(game_id, samples) -> ReplayAppendResult
    def load_index() -> DenseReplayIndex
    def prune_to_capacity(capacity_rows) -> None

class DenseReplayIndex:
    sample_count: int
    sample_weight_sum: float
    chunks: tuple[ReplayChunkInfo, ...]

class DenseShuffleWindow:
    entries: tuple[ReplayEntry, ...]
    seed: int
    sample_count: int

class DenseTrainBucket:
    total_generated_weight: float
    total_trained_rows: int
    max_train_samples_per_new_sample: float

    def remaining_rows(self) -> int:
        return floor(total_generated_weight * max_train_samples_per_new_sample) - total_trained_rows
```

### Why this matters

The model checkpoint should not be the data store. The data store should survive model reloads independently, and the training window should be reshuffled explicitly.

---

## F. `dense_cnn/trainer.py`

### Current issue

`select_training_samples` samples a small count from the in-memory buffer, and `train_passes` trains once over that window.  

### Rewrite

Replace `DenseSampleWindow(records=...)` with replay references:

```python
@dataclass(slots=True)
class DenseSampleWindow:
    entries: tuple[ReplayEntry, ...]
    seed: int
    epoch: int
    sample_count: int
    metadata: Mapping[str, Any]
```

`select_training_samples`:

```python
def select_training_samples(...):
    replay = components.model.extra["replay_store"]
    index = replay.load_index()

    if index.sample_count < config.samples.min_rows_to_train:
        return skipped(...)

    budget_rows = train_state.remaining_rows()
    if budget_rows <= 0:
        return skipped("train bucket limited")

    rows = min(config.training.max_train_samples_per_epoch, budget_rows)

    window = replay.build_shuffle_window(
        rows=config.samples.shuffle_window_rows,
        seed=epoch_seed,
        weighting="recency_x_policy_surprise",
    )

    train_entries = window.draw(rows, seed=epoch_seed)

    components.shared.sample_window = DenseSampleWindow(...)
```

`train_passes`:

```python
for entry_batch in batches(sample_window.entries):
    compressed_samples = replay.read_entries(entry_batch)
    expanded = [
        expand_sample(sample, symmetry=random_d6())
        for sample in compressed_samples
    ]
    ...
```

After training:

```python
train_state.total_trained_rows += rows_trained
```

Return diagnostics:

```python
{
    "rows_trained": rows_trained,
    "generated_weight_total": train_state.total_generated_weight,
    "trained_rows_total": train_state.total_trained_rows,
    "train_to_data_ratio": trained_rows_total / generated_weight_total,
    "bucket_remaining": train_state.remaining_rows(),
}
```

---

## G. `dense_cnn/checkpoints.py`

### Current issue

The checkpoint stores the full `sample_buffer`. 

### Rewrite

Checkpoint should store model/training state, not replay rows:

```python
payload = {
    "model": "hexo_models.dense_cnn",
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
    "train_state": {
        "total_generated_weight": ...,
        "total_generated_samples": ...,
        "total_trained_rows": ...,
        "replay_manifest_path": ...,
        "shuffle_generation": ...,
    },
    "metadata": {...},
}
```

If you want portability, add an optional separate replay archive command later. Do not make every model checkpoint contain replay history by default.

---

## H. `dense_cnn/losses.py`

### Current issue

Losses are fine structurally, but names and masks should change from fixed lookahead to short-term value. 

### Rewrite

```python
for key, output in outputs.items():
    if key.startswith("shortterm_value_") and key in batch:
        components[key] = binned_value_loss(
            output,
            batch[key],
            mask=batch.get(f"{key}_mask"),
        )
        total += shortterm_value_weight * components[key]
```

Policy loss should optionally use sample weights:

```python
components["policy"] = weighted_soft_cross_entropy(
    outputs["policy"],
    batch["policy"],
    weight=batch.get("sample_weight"),
)
```

Same for value and auxiliary heads.

This makes policy-surprise weighting affect gradient frequency/weighting. If replay sampling already samples surprising rows more often, you can keep loss weights unscaled. I would start with **sampling frequency only**, because that is closer to KataGo’s described policy-surprise method. ([GitHub][3])

---

# 11. New diagnostics to require

Every cycle should write diagnostics that make learning behavior obvious.

## Self-play diagnostics

```json
{
  "games_completed": 256,
  "games_truncated": 0,
  "mcts_positions": 42318,
  "mcts_simulations": 5416704,
  "mean_game_length": 165.3,
  "search_visits": 128,
  "no_policy_rollout_tails": true,
  "root_policy_temperature_mean": 1.17,
  "root_dirichlet_total_alpha": 10.83,
  "policy_surprise_mean": 0.42,
  "sample_weight_mean": 1.18,
  "sample_weight_p95": 3.7
}
```

## Replay diagnostics

```json
{
  "replay_rows": 314218,
  "replay_weight_sum": 371002.4,
  "new_rows": 42318,
  "new_weight_sum": 49871.2,
  "capacity_rows": 1000000,
  "oldest_epoch": 1,
  "newest_epoch": 5
}
```

## Training diagnostics

```json
{
  "rows_selected": 99742,
  "rows_trained": 99742,
  "batch_size": 128,
  "steps": 780,
  "generated_weight_total": 371002.4,
  "trained_rows_total": 591220,
  "train_to_generated_weight_ratio": 1.59,
  "bucket_remaining_rows": 150784,
  "shuffle_window_rows": 300000,
  "mean_sample_age": 2.1
}
```

These diagnostics replace the current less-informative `samples_added`/`buffer_count` style. Current self-play diagnostics already include searched positions, simulations, positions per second, active games, and completion rollout.  The rewrite should preserve throughput diagnostics but add data-lifecycle diagnostics.

---

# 12. Migration plan

## Phase 1: remove rollout tails and sample budget

This is the highest-impact simplification.

Do:

* Replace sample-budget self-play with game-count self-play.
* Search every active playable position with MCTS.
* Record every searched position.
* Finalize samples after terminal.
* Keep the current `SampleBuffer` temporarily.
* Set `samples_per_epoch` unused/deprecated.

Do not yet add replay store or policy surprise if you want a safe first step.

Expected result:

```text
same general pipeline
less biased game data
more terminal/late-game data
simpler self-play semantics
```

## Phase 2: add train-per-new-data budget

Do:

* Track `total_generated_samples`.
* Track `total_trained_rows`.
* Add `max_train_samples_per_new_sample`.
* Train until budget, not just `train_sample_count`.

Even if replay is still in memory, this fixes the worst data ratio issue.

Current config trains only 4,096 samples after generating up to 65,536 searched samples.   The rewrite should make the ratio explicit and visible.

## Phase 3: move replay out of checkpoint

Do:

* Add durable replay chunks.
* Add manifest/index.
* Checkpoint train counters and manifest pointer.
* Stop serializing all samples inside model checkpoints.

Expected result:

```text
smaller checkpoints
better resume behavior
real training windows
easier debugging of generated data
```

## Phase 4: add policy-surprise weighting

Do:

* Return root prior policy from MCTS.
* Compute KL from MCTS target to root prior.
* Store `policy_surprise` and `sample_weight`.
* Sample replay rows by `recency × sample_weight`.

This is one of the most KataGo-like changes that directly affects learning from blind spots. KataGo describes this as one of the larger improvements in its training. ([GitHub][3])

## Phase 5: replace fixed lookahead with short-term exponential targets

Do:

* Keep auxiliary value heads.
* Change targets to exponential future-MCTS averages.
* Rename diagnostics and config.
* Use all searched future positions, now available because tails are searched too.

Expected result:

```text
less noisy value training
targets closer to KataGo’s short-term value machinery
better use of full-game MCTS traces
```

## Phase 6: simplify MCTS exploration knobs

Do:

* Remove public progressive-widening config.
* Use all in-crop legal policy candidates.
* Keep lazy edge materialization only as implementation optimization.
* Add root policy temperature and total-alpha root Dirichlet noise.

Expected result:

```text
fewer arbitrary exploration rules
more stable policy learning
cleaner MCTS target semantics
```

---

# 13. What the new loop should look like end-to-end

```text
TrainingPipeline.run
  → load config
  → build dense CNN model/trainer/replay store
  → load checkpoint train_state
  → calibrate batch sizes

  for cycle in 1..N:
      selfplay:
          start games_per_epoch games
          run all games to terminal/truncation
          MCTS every decision
          record root prior, visit policy, root value
          finalize final value + short-term value targets
          append replay chunks

      replay:
          refresh manifest/index
          prune old rows if over capacity
          compute generated sample/weight counters

      train:
          compute train bucket:
              allowed = max_train_per_new_sample * generated_weight_total
              budget = allowed - trained_rows_total
          if enough replay and budget > 0:
              build shuffled recent training window
              sample rows by recency × policy-surprise weight
              train up to min(max_train_samples_per_epoch, budget)

      checkpoint:
          save model + optimizer + train_state
          publish pointer

      evaluation:
          run DenseCNNPlayer vs SealBot if configured
```

---

# 14. Expected learning behavior changes

The rewrite should change learning behavior in these ways:

| Current behavior                             | Proposed behavior                              | Expected effect                            |
| -------------------------------------------- | ---------------------------------------------- | ------------------------------------------ |
| Searched prefixes + rollout tails            | All self-play moves searched                   | Cleaner value targets, more late-game data |
| Sample budget drives game loop               | Game count drives self-play                    | Simpler and less biased data generation    |
| 4,096 train rows after 65,536 generated rows | Explicit train-per-new-data budget             | Better use of expensive MCTS labels        |
| Replay buffer checkpointed with model        | Durable replay chunks + manifest               | More robust resume and shuffle behavior    |
| Fixed `[1,4,8]` lookahead targets            | Exponential short-term value targets           | Lower-variance auxiliary value learning    |
| Uniform-ish recency replay                   | Recency × policy-surprise replay               | Faster learning from MCTS corrections      |
| Many arbitrary MCTS widening knobs           | All legal in-crop candidates + root temp/noise | Fewer hidden behavioral rules              |
| Duplicated self-play settings                | One self-play cycle config                     | Easier tuning on consumer hardware         |

---

# 15. The most important concrete deletions

If you only do one pass, delete these concepts first:

```text
samples_per_epoch
min_mcts_samples_per_game
sample_depth_active_limit
remaining_samples
rollout_games
_policy_rollout_actions
completion_rollout = "mcts_until_sample_budget_then_model_policy_rollout"
```

Those are the pieces making the current loop least KataGo-like and most behaviorally unusual.

---

# 16. The most important concrete additions

Add these instead:

```text
games_per_epoch
max_active_games
train_bucket:
    max_train_samples_per_new_sample
    max_train_samples_per_epoch
    total_generated_weight
    total_trained_rows

replay/shuffle:
    replay_capacity_rows
    shuffle_window_rows
    min_rows_to_train

sample metadata:
    root_prior_policy
    policy_surprise
    sample_weight
    shortterm_value_targets

MCTS exploration:
    root_policy_temperature schedule
    root_dirichlet_total_alpha
```

This gives you a loop that is still single-machine and sequential, but much closer to the part of KataGo that matters for learning: **complete MCTS self-play games, shuffled replay, controlled train/data ratio, and increased focus on surprising positions.**

[1]: https://raw.githubusercontent.com/lightvector/KataGo/master/SelfplayTraining.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/lightvector/KataGo/master/python/selfplay/synchronous_loop.sh "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/lightvector/KataGo/master/docs/KataGoMethods.md "raw.githubusercontent.com"
