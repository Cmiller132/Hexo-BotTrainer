# Hexo RL Prototype

Single-machine reinforcement-learning prototype for Hexo.

The current design documentation is intentionally consolidated:

- [Project Structure](docs/structure/PROJECT_STRUCTURE.md)
- [hexo-engine](docs/structure/HEXO_ENGINE.md)
- [hexo-runner](docs/structure/HEXO_RUNNER.md)
- [hexo-utils](docs/structure/HEXO_UTILS.md)
- [hexo-train](docs/structure/HEXO_TRAIN.md)
- [hexo-model-*](docs/structure/HEXO_MODEL.md)

The code is still in a prototype/design stage, so several package APIs are
scaffolding. The docs above are the source of truth for package ownership and
the intended layout.

## Packages

- `hexo_engine`: canonical rules, state transitions, terminal detection, and
  replayable state history.
- `hexo_runner`: headless game execution, player lifecycle, and detached game
  records.
- `hexo_utils`: reusable encoding, search, symmetry, and sample-buffer
  mechanics.
- `hexo_train`: self-play epoch orchestration, config loading, checkpoints, and
  run artifacts.
- `hexo_models`: standalone production model families. Model 1 lives in the
  compartmentalized `hexo_models.dense_cnn` package.
- `hexo_model_resnet`: first model package and training plugin scaffold.
- `hexo_frontend`: local browser tools and dashboards.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .\packages\hexo_engine
python -m pip install -e .\packages\hexo_utils
python -m pip install -e .\packages\hexo_runner
python -m pip install -e .\packages\hexo_train
python -m pip install -e .\packages\hexo_models
python -m pip install -e .\packages\hexo_model_resnet
python -m pip install -e .\packages\hexo_frontend
```

Rust-backed packages keep their Rust code inside the owning package directory.
Build a package bridge with maturin when that package's Python bindings are
needed.

## Dense CNN Model 1

The production Model 1 implementation lives under
`hexo_models.dense_cnn`, separate from other model families. A baseline
training config is available at `configs/dense_cnn_model1.toml` and follows
the 4096 self-play sample / 4096 training sample epoch path with 200k compressed
sample capacity and 64 SealBot-best 50 ms evaluation games per epoch.

Dense CNN self-play uses the Rust Model 1 encoder and Rust batched MCTS bridge.
Calibration keeps the search count fixed at exactly 128 MCTS simulations per
searched position and tunes active self-play batch size plus virtual leaf batch
size. The current baseline keeps the 96-channel, 6-block model and targets at
least 128 searched positions per second with 128 simulations.

The production config requires a SealBot checkout for epoch evaluation:

```powershell
$env:SEALBOT_PATH = "C:\path\to\SealBot"
python -m hexo_train.cli.train_model .\configs\dense_cnn_model1.toml
```

When `require_sealbot = true`, training fails fast if SealBot best 50 ms cannot
be launched; this prevents a run from looking complete while Goal 4 evaluation
was skipped.
