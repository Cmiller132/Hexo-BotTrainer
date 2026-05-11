# Finalized Project Overview: Hexo RL Prototype

This project is a **single-machine reinforcement learning prototype** for Hexo. The design uses a **Rust engine** for fast rule/state logic, a separate Rust/Python `models_common` layer for MCTS, encoding, replay, and model-facing utilities, and a **Python skeleton** for training, checkpointing, and experiment control.

The key rule assumption is that Hexo is played on an unlimited hex grid, Player 0 opens at `(0, 0)`, later turns place two stones of the same player, each stone must be empty and within 8 hex steps of existing stones, and a player wins immediately by making six connected stones in a straight line. Threats are six-cell windows with at least four own stones and zero opponent stones. 

The important design correction is:

```text
Use autoregressive MCTS.

A two-stone Hexo turn is represented as:
  first placement -> second placement -> switch player
```

This avoids pair-action explosion and keeps both the search and neural policy simple.

---

# 1. Project Goals

The prototype should:

```text
1. Implement a fast Rust Hexo engine.
2. Use autoregressive MCTS over single stone placements.
3. Generate self-play games on CPU threads.
4. Use Python/PyTorch for model inference and training.
5. Keep model code separate from game/search code.
6. Run comfortably on one consumer PC:
   Ryzen 7950X
   RTX 4070 Ti
   32 GB RAM
```

The project should **not** initially include:

```text
model gating
distributed actors
multi-GPU support
large multi-game framework
dense pair-action policy heads
large model zoo
fully asynchronous training and self-play
```

Keep it small, correct, and expandable.

---

# 2. Final Repository Layout

```text
hexo-rl/
Cargo.toml
packages/
  game_engine/
    Cargo.toml
    src/
      lib.rs
      game/

  models_common/
    Cargo.toml
    pyproject.toml
    src/                  # Rust MCTS, encoding, replay, PyO3
    python/models_common/ # Python model API, replay, inference helpers

  game_runner/
    pyproject.toml
    python/game_runner/   # CLI, config, training loop

  hexo_resnet/
    pyproject.toml
    python/hexo_resnet/   # ResNet model plugin

configs/
data/
tests/
```

The implemented layout keeps `game_engine` and `models_common` as separate Rust crates. The engine owns authoritative rules and state transitions; models-common owns search, encoding, replay samples, self-play helpers, and the Python bridge.

---
# 3. Rust Engine Overview

The Rust engine crate is responsible for:

```text
game state
legal move generation
incremental window tracking
win detection
threat detection
```

The Rust `models_common` crate owns autoregressive MCTS, search-owned cloned positions, training sample generation, state encoding, self-play helpers, and Python bindings. The engine should stay deterministic, heavily tested, and independent from model code.

---

# 4. Hex Coordinates

Use axial coordinates:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HexCoord {
    pub q: i16,
    pub r: i16,
}
```

The third cube coordinate can be derived:

```text
s = -q - r
```

Hex distance:

```rust
pub fn hex_distance(a: HexCoord, b: HexCoord) -> i16 {
    let dq = a.q - b.q;
    let dr = a.r - b.r;
    let ds = -dq - dr;
    dq.abs().max(dr.abs()).max(ds.abs())
}
```

The three line axes are:

```text
(1, 0)
(0, 1)
(1, -1)
```

These are enough for six-in-line detection.

---

# 5. Board Representation

Use a sparse board. Hexo is treated as unlimited, so do not use a fixed full board.

```rust
pub enum Stone {
    Player0,
    Player1,
}

pub struct Board {
    pub stones: HashMap<HexCoord, Stone>,
    pub occupied: Vec<HexCoord>,
}
```

For performance, later replace `HashMap` with `rustc_hash::FxHashMap` or `ahash::AHashMap`.

Basic board methods:

```rust
impl Board {
    pub fn is_empty(&self, coord: HexCoord) -> bool;
    pub fn get(&self, coord: HexCoord) -> Option<Stone>;
    pub fn place(&mut self, coord: HexCoord, stone: Stone) -> Result<(), MoveError>;
    pub fn occupied_cells(&self) -> &[HexCoord];
}
```

---

# 6. Turn Phase Model

This is the core design.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnPhase {
    Opening,
    FirstStone,
    SecondStone { first: HexCoord },
}
```

The full game state:

```rust
pub struct HexoState {
    pub board: Board,
    pub current_player: Player,
    pub phase: TurnPhase,
    pub placements_made: u32,
    pub terminal: Option<GameOutcome>,
    pub last_turn: Option<MoveRecord>,
    pub zobrist_hash: u64,
}
```

A normal Hexo turn is not represented as one pair action. It is represented as two sequential placement decisions.

```text
Opening:
  Player 0 places at (0, 0)

Normal turn:
  current player places first stone
  current player places second stone
  then player switches
```

Important rule:

```text
Check for a win after every single stone placement.
```

If the first stone of a two-stone turn completes six in a row, the game ends immediately and the second stone is never placed.

---

# 7. Applying Moves

The engine should expose one simple action:

```rust
pub struct Placement {
    pub coord: HexCoord,
}
```

The state transition handles the turn phase.

```rust
pub fn apply_placement(
    state: &mut HexoState,
    placement: Placement,
) -> Result<ApplyResult, MoveError> {
    match state.phase {
        TurnPhase::Opening => {
            // Only legal placement is (0, 0).
            // Place Player 0 stone.
            // Check win, though opening cannot normally win.
            // Switch to Player 1, FirstStone.
        }

        TurnPhase::FirstStone => {
            // Validate placement.
            // Place stone for current_player.
            // Check immediate win.
            // If not terminal:
            //   phase = SecondStone { first: placement.coord }
            //   current_player remains the same.
        }

        TurnPhase::SecondStone { first } => {
            // Validate placement.
            // Must not reuse first.
            // Place stone for current_player.
            // Check immediate win.
            // If not terminal:
            //   switch current_player.
            //   phase = FirstStone.
        }
    }
}
```

This keeps the game engine simple and aligns directly with autoregressive MCTS.

---

# 8. Legal Move Generation

Legal placement rules:

```text
Opening:
  only (0, 0)

After opening:
  coord must be empty
  coord must be within 8 hex steps of at least one occupied stone
```

Simple prototype implementation:

```rust
pub fn legal_placements(state: &HexoState, out: &mut Vec<HexCoord>) {
    out.clear();

    if state.terminal.is_some() {
        return;
    }

    match state.phase {
        TurnPhase::Opening => {
            out.push(HexCoord { q: 0, r: 0 });
        }

        TurnPhase::FirstStone | TurnPhase::SecondStone { .. } => {
            // Iterate around all occupied cells within radius 8.
            // Add empty cells to a deduplicated set.
        }
    }
}
```

For the prototype, generate candidates by scanning radius-8 neighborhoods around occupied cells and deduplicating with a hash set.

Later optimization:

```text
Maintain an incremental frontier set.
Update it after every placement.
```

But do not over-optimize before correctness.

---

# 9. Win Detection

Only check lines passing through the newly placed stone.

For each of the three axes:

```text
count stones forward
count stones backward
total = 1 + forward + backward
win if total >= 6
```

Pseudo-code:

```rust
const DIRECTIONS: [HexCoord; 3] = [
    HexCoord { q: 1, r: 0 },
    HexCoord { q: 0, r: 1 },
    HexCoord { q: 1, r: -1 },
];

pub fn is_winning_placement(
    board: &Board,
    coord: HexCoord,
    player: Player,
) -> bool {
    for dir in DIRECTIONS {
        let forward = count_in_direction(board, coord, dir, player);
        let backward = count_in_direction(board, coord, -dir, player);

        if 1 + forward + backward >= 6 {
            return true;
        }
    }

    false
}
```

This is fast and avoids scanning the whole board.

---

# 10. Threat Detection

Threat detection can be added early but should not block the first working version.

A threat is:

```text
a six-cell line window
with at least four stones for one player
and zero stones for the opponent
```

Use threats for:

```text
debugging
metrics
optional input planes
later tactical move ordering
```

Do not make the first MCTS depend heavily on handcrafted threat logic. Start with legal moves, policy priors, and value backup. Add tactical enhancements after the base loop works.

---

# 11. Autoregressive MCTS

MCTS searches one placement at a time.

A search node represents:

```text
board state
current player
turn phase
```

The legal actions at a node are:

```text
Opening:
  one legal action: center

FirstStone:
  all legal first placements

SecondStone:
  all legal second placements after the first stone
```

The MCTS does **not** know about pair actions.

---

## 11.1 Tree Structures

```rust
pub struct Node {
    pub state_hash: u64,
    pub player_to_act: Player,
    pub phase: TurnPhase,
    pub visits: u32,
    pub value_sum: f32,
    pub edges: Vec<Edge>,
    pub expanded: bool,
}

pub struct Edge {
    pub action: HexCoord,
    pub prior: f32,
    pub visits: u32,
    pub value_sum: f32,
    pub child: Option<NodeId>,
}
```

Each edge corresponds to a single placement.

---

## 11.2 PUCT Selection

Use a standard PUCT score:

```text
score = Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
```

Prototype defaults:

```yaml
c_puct: 1.5
```

Use legal move masking before normalizing priors.

---

## 11.3 Value Backup

This is the most important MCTS detail.

Because `FirstStone` and `SecondStone` belong to the same player, **do not flip the value sign just because a stone was placed**.

Flip perspective only when the player to act changes.

The simplest robust method:

```rust
fn value_for_node(
    leaf_value: f32,
    leaf_player: Player,
    node_player: Player,
) -> f32 {
    if leaf_player == node_player {
        leaf_value
    } else {
        -leaf_value
    }
}
```

At each visited node:

```text
backup value from that node's player-to-act perspective
```

This naturally handles:

```text
FirstStone -> SecondStone:
  same player, no sign flip

SecondStone -> opponent FirstStone:
  player changes, sign flips
```

---

## 11.4 Search Output

At the root, MCTS returns:

```rust
pub struct SearchResult {
    pub selected_action: HexCoord,
    pub visit_policy: Vec<(HexCoord, u32)>,
    pub root_value: f32,
}
```

The visit policy becomes the training target for that individual placement decision.

A normal two-stone turn produces two training samples:

```text
sample before first stone
sample before second stone
```

---

# 12. Rust Evaluator Interface

Rust should not know anything about PyTorch or model architecture.

Define an abstract evaluator:

```rust
pub trait Evaluator {
    fn evaluate_batch(
        &mut self,
        states: &[EncodedState],
    ) -> Vec<NetworkOutput>;
}
```

Network output:

```rust
pub struct NetworkOutput {
    pub policy_logits: Vec<f32>,
    pub value: f32,
}
```

For early testing, implement:

```text
UniformEvaluator:
  uniform prior over legal moves
  value = 0.0
```

This lets you test MCTS and self-play before Python integration.

---

# 13. Encoding States for the Model

The Rust encoder converts `HexoState` into a fixed crop.

Start with:

```text
crop size: 31x31 for dev
crop size: 37x37 for main
```

Center the crop around the boardâ€™s occupied bounding region or recent move. For the first version, use a simple deterministic crop centered on the occupied bounding box.

Recommended input planes:

```text
0. current player stones
1. opponent stones
2. legal cells
3. first stone this turn
4. last own stone 1
5. last own stone 2
6. last opponent stone 1
7. last opponent stone 2
8. phase is opening
9. phase is first stone
10. phase is second stone
11. valid crop mask
```

The model predicts:

```text
policy logits over crop cells
scalar value from current player's perspective
```

The Rust side must also provide a mapping:

```text
crop index -> HexCoord
HexCoord -> crop index
```

Only legal cells inside the crop should receive policy probability. If a legal cell is outside the crop, either enlarge the crop or exclude it for the prototype. The simpler prototype choice is:

```text
Use a sufficiently large crop and assert that legal candidates fit.
```

Later, support dynamic crops or sparse candidate features.

---

# 14. Replay Sample Format

Each placement decision creates one sample.

```rust
pub struct ReplaySample {
    pub game: String,
    pub rules_version: u32,
    pub state: EncodedState,
    pub current_player: Player,
    pub phase: TurnPhaseLabel,
    pub legal_actions: Vec<HexCoord>,
    pub policy_target: Vec<(HexCoord, f32)>,
    pub value_target: f32,
    pub placements_made: u32,
}
```

For policy target:

```text
policy_target[coord] = visit_count(coord) / total_root_visits
```

For value target:

```text
+1.0 if current_player eventually wins
-1.0 if current_player eventually loses
```

There is no normal draw rule in this prototype.

Recommended storage:

```text
data/selfplay/cycle_000001/games_000000.zst
data/selfplay/cycle_000001/games_000001.zst
data/replay/replay_latest.npz
```

For the first version, simple `.jsonl.zst` or `.msgpack.zst` is fine. Optimize later.

---

# 15. Self-Play Engine

Rust self-play loop:

```rust
pub fn play_selfplay_game<E: Evaluator>(
    game_config: &GameConfig,
    mcts_config: &MctsConfig,
    evaluator: &mut E,
) -> SelfplayGame {
    let mut state = HexoState::new();
    let mut samples = Vec::new();

    while !state.is_terminal() {
        let search = run_mcts(&state, evaluator, mcts_config);

        samples.push(make_pending_sample(
            &state,
            &search.visit_policy,
            search.root_value,
        ));

        apply_placement(&mut state, Placement {
            coord: search.selected_action,
        }).unwrap();
    }

    attach_final_values(samples, state.outcome());

    SelfplayGame {
        samples,
        outcome: state.outcome(),
    }
}
```

Use multiple CPU actors:

```text
8 actors for dev
12 actors for main
```

Do not run training and self-play simultaneously in the first prototype.

---

# 16. Python Skeleton Overview

Python is responsible for:

```text
loading config
loading model plugin
running batched inference
starting Rust self-play
reading replay data
training model
saving checkpoints
logging metrics
```

The Python runner should be thin. `game_engine` should own game rules, while `models_common` should own model-facing MCTS, encoding, replay, and inference utilities.

---

# 17. Python Model API

Use a stable model plugin interface:

```python
# packages/models_common/python/models_common/model_api.py

from typing import Protocol, Mapping, Any
import torch


class ModelPlugin(Protocol):
    name: str

    def build_model(
        self,
        game_spec: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> torch.nn.Module:
        ...

    def forward_inference(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        """
        Returns:
          policy_logits: [B, H, W]
          value: [B]
        """
        ...

    def loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        ...

    def augment_batch(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        ...
```

The Rust models-common bridge only sees policy/value outputs. It does not know which model package produced them.

---

# 18. Python Inference Server

The inference module batches Rust evaluation requests and runs them on the GPU.

```python
# packages/models_common/python/models_common/inference.py

class InferenceServer:
    def __init__(self, model, plugin, device, batch_size: int):
        self.model = model
        self.plugin = plugin
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def evaluate(self, encoded_states):
        batch = collate_encoded_states(encoded_states)
        batch = move_to_device(batch, self.device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.plugin.forward_inference(self.model, batch)

        return convert_outputs_for_rust(outputs)
```

For the first working version, this can be synchronous:

```text
Rust sends batch
Python evaluates batch
Rust continues
```

Later, this can become a proper queue-based server.

---

# 19. Python Training Loop

Simple loop:

```python
def train_one_cycle(config):
    model, plugin = load_model(config)
    replay = load_replay(config.replay.path)

    optimizer = make_optimizer(model, config.training)

    for step in range(config.training.steps_per_cycle):
        batch = replay.sample_batch(config.training.batch_size)
        batch = plugin.augment_batch(batch)

        outputs = plugin.forward_inference(model, batch)
        loss = plugin.loss(outputs, batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip_norm,
        )
        optimizer.step()

    save_checkpoint(model, config.checkpointing.latest_path)
```

No gating. No tournament. The latest model is always used for the next self-play cycle.

---

# 20. Main Training Driver

```python
# packages/game_runner/python/game_runner/cli.py

def loop(config_path: str):
    config = load_config(config_path)

    for cycle in range(config.loop.cycles):
        run_selfplay_cycle(config, cycle)
        update_replay(config, cycle)
        train_one_cycle(config)
        save_cycle_checkpoint(config, cycle)
```

Training loop:

```text
latest checkpoint
    â†“
self-play
    â†“
replay update
    â†“
train
    â†“
new latest checkpoint
    â†“
repeat
```

---

# 21. CLI Commands

Recommended initial commands:

```bash
hexo-rl test-engine
hexo-rl random-game
hexo-rl selfplay configs/dev.yaml
hexo-rl train configs/dev.yaml
hexo-rl loop configs/dev.yaml
hexo-rl inspect-replay data/selfplay/cycle_000001
```

Optional later commands:

```bash
hexo-rl eval checkpoints/a.pt checkpoints/b.pt
hexo-rl benchmark-mcts configs/main.yaml
hexo-rl benchmark-inference configs/main.yaml
```

---

# 22. Configuration Files

## `configs/dev.yaml`

```yaml
game:
  name: hexo
  crop_size: 31
  rules_version: 1
  max_placements: 300

model:
  package: hexo_resnet
  variant: small
  channels: 64
  residual_blocks: 6
  precision: fp16

selfplay:
  games_per_cycle: 200
  actors: 8
  mcts_visits: 64
  temperature_placements: 24
  dirichlet_noise: true

mcts:
  c_puct: 1.5
  root_noise_alpha: 0.3
  root_noise_frac: 0.25

inference:
  batch_size: 64
  device: cuda

training:
  batch_size: 256
  steps_per_cycle: 500
  replay_window_samples: 100000
  learning_rate: 0.0003
  weight_decay: 0.0001
  amp: true
  grad_clip_norm: 1.0

checkpointing:
  latest_path: data/checkpoints/latest.pt
  keep_last: 10

loop:
  cycles: 1000
```

## `configs/main.yaml`

```yaml
game:
  name: hexo
  crop_size: 37
  rules_version: 1
  max_placements: 500

model:
  package: hexo_resnet
  variant: base
  channels: 96
  residual_blocks: 8
  precision: fp16

selfplay:
  games_per_cycle: 1000
  actors: 12
  mcts_visits: 128
  temperature_placements: 32
  dirichlet_noise: true

mcts:
  c_puct: 1.5
  root_noise_alpha: 0.3
  root_noise_frac: 0.25

inference:
  batch_size: 96
  device: cuda

training:
  batch_size: 256
  steps_per_cycle: 1500
  replay_window_samples: 500000
  learning_rate: 0.0002
  weight_decay: 0.0001
  amp: true
  grad_clip_norm: 1.0

checkpointing:
  latest_path: data/checkpoints/latest.pt
  keep_last: 20

loop:
  cycles: 1000
```

---

# 23. Basic Setup Instructions

## 23.1 Create the Rust crate

```bash
mkdir hexo-rl
cd hexo-rl

mkdir rust
cd rust
cargo init --lib
```

Suggested Rust dependencies:

```toml
[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
zstd = "0.13"
rand = "0.8"
rayon = "1.10"
crossbeam-channel = "0.5"
ahash = "0.8"
thiserror = "1"
```

For the earliest version, you can leave out PyO3 until the Rust engine and MCTS tests pass.

---

## 23.2 Create the Python package

From the project root:

```bash
mkdir -p packages/game_runner/python/game_runner
mkdir -p packages/models_common/python/models_common
mkdir -p packages/hexo_resnet/python/hexo_resnet
touch packages/game_runner/python/game_runner/__init__.py
touch packages/game_runner/python/game_runner/cli.py
touch packages/game_runner/python/game_runner/config.py
touch packages/game_runner/python/game_runner/train.py
touch packages/game_runner/python/game_runner/selfplay.py
touch packages/game_runner/python/game_runner/replay.py
touch packages/game_runner/python/game_runner/metrics.py
touch packages/models_common/python/models_common/model_api.py
touch packages/models_common/python/models_common/inference.py
touch packages/models_common/python/models_common/replay.py
touch packages/models_common/python/models_common/rust_bridge.py
```

Suggested Python dependencies:

```toml
[project]
name = "game-runner"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "torch",
  "numpy",
  "pydantic",
  "pyyaml",
  "tqdm",
  "rich",
  "maturin",
]
```

---

## 23.3 Build Rust Python bindings

Once `pybridge.rs` exists:

```bash
maturin develop --manifest-path packages/models_common/Cargo.toml --features python
```

Then test:

```bash
python -c "import game_runner, models_common; print('ok')"
```

---

## 23.4 Run Rust tests

```bash
cd .
cargo test
```

Run this constantly during engine development. The Hexo engine should be proven correct before adding neural training.

---

# 24. Rust Test Checklist

Add unit tests for:

```text
opening move must be exactly (0, 0)
opening move switches to Player 1
normal first placement keeps the same player
normal first placement changes phase to SecondStone
normal second placement switches player
second placement cannot reuse the first placement
stones cannot be placed on occupied cells
legal cells must be within radius 8 of existing stones
six-in-line win works on q axis
six-in-line win works on r axis
six-in-line win works on q-r axis
win after first stone ends the game immediately
win after second stone ends the game immediately
terminal states have no legal moves
```

Add MCTS tests for:

```text
MCTS only selects legal placements
MCTS handles Opening phase
MCTS handles FirstStone phase
MCTS handles SecondStone phase
value backup does not flip between FirstStone and SecondStone
value backup flips after player switch
uniform evaluator produces non-empty visit policy
```

---

# 25. Initial Implementation Order

Build in this order:

```text
1. HexCoord and distance
2. Board
3. HexoState and TurnPhase
4. apply_placement
5. legal_placements
6. win detection
7. random self-play
8. replay sample structure
9. dummy evaluator
10. autoregressive MCTS
11. state encoder
12. PyO3 bridge
13. Python inference skeleton
14. Python training skeleton
15. ResNet model package
16. full self-play/train loop
```

Do not start with the neural model. Start with game correctness.

---

# 26. Minimal Python Model Package

The first model can be a small residual CNN:

```text
input: [B, C, H, W]
policy output: [B, H, W]
value output: [B]
```

Structure:

```text
conv input
6 residual blocks
policy head
value head
```

The policy head predicts one placement, not a two-stone pair.

```python
class HexoNet(torch.nn.Module):
    def __init__(self, in_channels: int, channels: int, blocks: int):
        super().__init__()
        self.stem = ConvBlock(in_channels, channels)
        self.blocks = torch.nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)]
        )
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(self, x, valid_mask=None):
        h = self.blocks(self.stem(x))
        policy_logits = self.policy_head(h)

        if valid_mask is not None:
            policy_logits = policy_logits.masked_fill(valid_mask == 0, -1e9)

        value = self.value_head(h, valid_mask)
        return {
            "policy_logits": policy_logits,
            "value": value,
        }
```

---

# 27. Replay and Training Target Shape

For each sample:

```text
state_tensor: [C, H, W]
legal_mask: [H, W]
policy_target: [H, W]
value_target: scalar
```

Loss:

```text
policy_loss = cross_entropy between MCTS visit policy and model policy
value_loss = MSE or BCE-style scalar value loss
total_loss = policy_loss + value_loss + weight_decay
```

Use:

```text
value target +1 if current player wins
value target -1 if current player loses
```

---

# 28. Hardware Defaults

For your machine, start here:

```text
CPU actors: 8
MCTS visits: 64
crop size: 31
model: 64 channels, 6 residual blocks
training batch size: 256
inference batch size: 64
self-play games per cycle: 200
training steps per cycle: 500
```

After correctness:

```text
CPU actors: 12
MCTS visits: 128
crop size: 37
model: 96 channels, 8 residual blocks
self-play games per cycle: 1000
training steps per cycle: 1500
```

Avoid training and self-play at the same time at first. Alternate them:

```text
self-play phase
training phase
self-play phase
training phase
```

This is easier to debug and avoids GPU contention.

---

# 29. Final Architecture Summary

```text
Rust engine:
  owns Hexo rules, state transitions, legal moves, win checks, MCTS, and self-play

Python runner:
  owns config, model loading, GPU inference, replay handling, training, and checkpointing

Model package:
  owns neural network architecture, losses, and data augmentation
```

The core loop:

```text
latest checkpoint
    â†“
Rust self-play using autoregressive MCTS
    â†“
placement-level replay samples
    â†“
Python training
    â†“
updated checkpoint
    â†“
repeat
```

The most important implementation rule:

```text
Hexo's two-stone turn is not a pair action.

It is:
  first placement state
  second placement state
  then player switch
```

That gives you a simple, correct prototype that can later grow into a larger multi-game, multi-model RL system without overbuilding the first version.
