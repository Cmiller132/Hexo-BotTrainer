# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Single-machine reinforcement-learning prototype for the game Hexo. It trains
self-play models (AlphaZero/KataGo-style) on one workstation (Ryzen 7950X + a
single CUDA GPU). The production target is **Model 1**, a dense CNN living in
`hexo_models.dense_cnn`. The staged goals (working model → fast → debuggable →
beats SealBot 50ms → cleanup → keep improving) are in `Model 1 goal.md`. KataGo
is the explicit design baseline — see `katagotips.md` and
`deep-research-report.md`.

## Repository layout

Python lives in `packages/<pkg>/python/<pkg>/`. Rust (where present) lives in
`packages/<pkg>/rust/`. The six installable Python packages:

- `hexo_engine` (Rust+PyO3): canonical rules, state transitions, terminal
  detection, replayable history. Python boundary in `hexo_engine/api.py`.
- `hexo_utils` (Rust+PyO3): replay records, symmetry, sample-buffer mechanics.
- `hexo_runner` (pure Python): headless game execution, player lifecycle,
  `.hxr` game records.
- `hexo_train` (pure Python): config-driven self-play training orchestration.
- `hexo_models` (Rust+PyO3): production model families. Two plugins registered
  via the `hexo_train.models` entry-point group: `dense_cnn` (Model 1) and
  `hexformer_ar`. A third `hexo_model_resnet` package dir exists separately.
- `hexo_frontend` (pure Python): local browser dashboards and debug tools.

Board constants are fixed: `BOARD_SIZE == 41`, `INPUT_CHANNELS == 13`. The
`ActionId` transport is `u32_i16_pair`, bounded to i16 coordinate components on
a sparse infinite board within the engine coordinate range.

## Build and setup

Rust-backed packages use **maturin**. Pure-Python packages install editable.
The Rust workspace (root `Cargo.toml`) has only three members: `hexo_engine`,
`hexo_models`, `hexo_utils`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .\packages\hexo_engine
python -m pip install -e .\packages\hexo_utils
python -m pip install -e .\packages\hexo_runner
python -m pip install -e .\packages\hexo_train
python -m pip install -e .\packages\hexo_models
python -m pip install -e .\packages\hexo_frontend
```

After changing Rust code, rebuild the affected package's native module, e.g.:

```powershell
maturin develop -m packages\hexo_models\Cargo.toml --features python
```

**Critical Rust detail:** the dense_cnn and hexformer_ar Rust crates are NOT
workspace members. They are compiled *into* `hexo_models._rust` via `#[path]`
includes in `packages/hexo_models/rust/src/lib.rs`. Native calls reach Python
through `hexo_models._rust.dense_cnn` (wrapped by
`dense_cnn/python/.../rust_bridge.py`). So editing
`packages/hexo_models/dense_cnn/rust/src/*.rs` requires rebuilding the
`hexo_models` package, not a separate crate.

## Tests

There is no root `pyproject.toml`, `pytest.ini`, or `conftest.py`. Tests in
`tests/` import the installed editable packages, so the packages must be
installed (or `PYTHONPATH` pointed at the `python/` dirs) first.

```powershell
python -m pytest tests                                  # full suite
python -m pytest tests\test_dense_cnn_pipeline.py       # one file
python -m pytest tests\test_dense_cnn_pipeline.py::test_name   # one test
```

Dense CNN test files are grouped by concern:
`test_dense_cnn_pipeline.py`, `test_dense_cnn_replay_schema.py`,
`test_dense_cnn_sample_generation.py`, `test_dense_cnn_performance.py`,
`test_dense_cnn_debug_artifacts.py`.

## Running training

The single public entry point is the config-driven CLI; the CLI is thin and all
lifecycle logic lives in `hexo_train.pipeline.TrainingPipeline`:

```powershell
python -m hexo_train.cli.train_model configs\dense_cnn_model1.toml
```

Production launcher (sets `PYTHONPATH` to the local worktree packages so a long
run never imports a stale installed `hexo_models` wheel, starts a resource
watchdog, and requires a SealBot checkout for epoch evaluation):

```powershell
.\scripts\start_model1_training.ps1 -SealBotPath "C:\path\to\SealBot"
```

When `[evaluation].require_sealbot = true`, training fails fast if SealBot
best-50ms cannot launch. `scripts\run_model1_wsl_smoke.sh` resumes a checkpoint
for a one-epoch CUDA smoke test under WSL. `runs/` (gitignored) holds all run
artifacts: `selfplay/` shards, `shuffleddata/<gen>/`, `checkpoints/`,
`diagnostics/`.

## Architecture: how a training run flows

1. `TrainingPipeline.run(config)` normalizes the TOML, builds a `RunContext`,
   and loads the model plugin via `hexo_train.registry.load_model_plugin`
   (resolved by explicit module path, entry-point name, or model name).
2. The plugin (`DenseCNNPlugin`) builds the network, trainer, checkpoint IO,
   self-play, evaluation, and NPZ replay components.
3. Fixed lifecycle steps run, each wrapped by `_run_step` for consistent
   diagnostics: initialize → load/initialize checkpoint → calibrate
   performance → run epochs → publish final checkpoint → write diagnostics.

`hexo_train` is deliberately a "map" — it never decodes tensors, computes
losses, or serializes model-specific checkpoint contents. Model packages own
all of that behind the `ModelPlugin` protocol (`hexo_train/registry.py`).

## Dense CNN (Model 1) specifics

The authoritative deep-dive is
[`packages/hexo_models/dense_cnn/README.md`](packages/hexo_models/dense_cnn/README.md).
Key points:

- **Python/Rust split:** Python owns PyTorch, config parsing, plugin wiring,
  self-play control, sample finalization, NPZ replay/shuffling, training,
  checkpoints, and the MCTS evaluator callback. Rust owns live `HexoState`
  intake, dense tensor encoding, batched PUCT MCTS, and state-derived sample
  facts.
- **Self-play is game-driven**, not sample-budgeted: request
  `games_per_epoch` complete games, keep `active_games` in flight, search every
  playable nonterminal position with a persistent native MCTS session until
  terminal or `max_actions`. There are no rollout tails and no in-memory
  `SampleBuffer` — checkpoints with legacy `sample_buffer` payloads are
  rejected, not migrated.
- **MCTS uses policy-nucleus (top-p) widening** to bound branching: each node
  materializes only its top-prior moves up to `widening_policy_mass` cumulative
  prior, clamped to `[widening_min_children, widening_max_children]`. Computed
  once at expansion — no visit-based growth. (This replaced progressive
  widening.)
- **Replay mirrors KataGo's selfplay/shuffler/training split**: per-game NPZ
  shards → power-law replay window → two-phase on-disk shuffle (a full window of
  dense planes does not fit in RAM) → `shuffleddata/<gen>/train/`. The NPZ row
  schema is fixed (see the README's schema list).
- **D6 augmentation** is applied at training time when compact samples are
  expanded into dense targets (`d6.py`, `input.py`); it must stay symmetric and
  well-tested to avoid subtly poisoning the model.

**When changing the Model 1 representation, update both language halves
together** (the README lists exact file pairs):
- Plane indices: `python/.../constants.py` ↔ `rust/src/constants.rs`
- Crop projection: `python/.../geometry.py`, `input.py` ↔ `rust/src/encoding.rs`
- Sample facts: `python/.../samples.py` ↔ `rust/src/sample_gen.rs`
- MCTS payload: `python/.../inference.py` ↔ `rust/src/mcts_eval.rs`
- Replay/training schema: `python/.../replay.py`, `trainer.py`, and the tests

## Conventions

- Config parsing rejects unknown keys per section and coerces types (no
  per-scalar range validation). Rust MCTS rejects invalid search settings and
  mismatched state/key batches. Both halves reject malformed byte payloads,
  wrong lengths, non-finite values, duplicate priors, and zero prior mass.
- Calibration keeps MCTS simulations fixed (128) and tunes batch sizes; the
  baseline 64-channel/4-block model targets ≥128 searched positions/sec.
- Per the project goals: prefer fixing performance at the root (rewrite hot
  paths in Rust) over surface tweaks, and delete dead experiments rather than
  accreting legacy wrappers.
