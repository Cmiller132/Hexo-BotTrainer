# Hexo RL Prototype

Single-machine reinforcement learning prototype for Hexo. The Rust engine owns
rules, MCTS, and self-play; Python owns config, model loading, replay handling,
inference, training, and checkpoints.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .\python
python -m pip install -e .\models\hexo_resnet
```

The Rust crate includes a minimal PyO3 bridge named `hexo_rl_rust` for uniform
self-play smoke runs. Build it from `python/` once Rust tooling is installed:

```bash
maturin develop --manifest-path ..\rust\Cargo.toml --features python
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
