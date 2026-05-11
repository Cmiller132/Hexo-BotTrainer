# Hexo RL Prototype

Single-machine reinforcement learning prototype for Hexo. The repo is organized
by package: `game_engine` owns rules and state transitions, `models_common`
owns model-facing MCTS/encoding/replay/inference helpers, `game_runner` owns
config and training orchestration, and `hexo_resnet` is the first model plugin.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .\packages\models_common
python -m pip install -e .\packages\game_runner
python -m pip install -e .\packages\hexo_resnet
```

The root Cargo workspace has two crates: `game_engine` for rules/state and
`models_common` for search, encoding, replay, self-play helpers, and the
private PyO3 module `models_common._rust`. Build the optional bridge once Rust
tooling is installed:

```bash
maturin develop --manifest-path .\packages\models_common\Cargo.toml --features python
```

## Commands

```bash
hexo-rl test-engine
hexo-rl random-game
hexo-rl selfplay configs/dev.yaml
hexo-rl train configs/dev.yaml
hexo-rl loop configs/dev.yaml --cycles 1
hexo-rl inspect-replay data/selfplay/cycle_000001
```
