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
- `hexo_model_resnet`: first model package and training plugin scaffold.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .\packages\hexo_engine
python -m pip install -e .\packages\hexo_utils
python -m pip install -e .\packages\hexo_runner
python -m pip install -e .\packages\hexo_train
python -m pip install -e .\packages\hexo_model_resnet
```

Rust-backed packages keep their Rust code inside the owning package directory.
Build a package bridge with maturin when that package's Python bindings are
needed.
