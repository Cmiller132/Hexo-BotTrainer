# Hexo RL Prototype

Single-machine reinforcement learning prototype for Hexo. The Rust engine owns
rules and state transitions; `models_common` owns model-facing MCTS, encoding,
replay, inference helpers, and the optional Rust/Python bridge. Python `hexo_rl`
owns config, training orchestration, and checkpoints.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .\models_common
python -m pip install -e .\python
python -m pip install -e .\models\hexo_resnet
```

The Rust workspace has two crates: `hexo_engine` for rules/state and
`hexo_models_common` for search, encoding, replay, self-play helpers, and the
PyO3 module named `models_common_rust`. Build the bridge from `models_common/`
once Rust tooling is installed:

```bash
maturin develop --manifest-path ..\rust\crates\hexo_models_common\Cargo.toml --features python
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
