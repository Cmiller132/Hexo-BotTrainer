# Project Structure

## Goal

Hexo-RL is organized around six clear project families:

- `hexo-engine`: canonical game rules and state authority.
- `hexo-runner`: game execution and orchestration.
- `hexo-utils`: reusable mechanisms shared across packages.
- `hexo-train`: config-driven training orchestration.
- `hexo-model-*`: model families and learned decision systems.
- `hexo-frontend`: browser-facing local tools and dashboards.

Each package family owns one kind of responsibility. The layout should make it
easy to add new models, runners, and tooling without moving rule authority or
mixing training assumptions into the game engine.

## Dependency Direction

The intended package dependencies are:

```text
hexo-engine
  <- hexo-utils
  <- hexo-runner
  <- hexo-model-*

hexo-runner
  <- hexo-model-*        # only for runner player contracts

hexo-train
  <- hexo-model-*        # loaded dynamically as training plugins

hexo-frontend
  <- hexo-runner         # UI clients call runner-facing APIs, never the reverse
```

In practice:

- `hexo-engine` stands alone and owns game truth.
- `hexo-utils` may depend on engine contracts for reusable mechanisms.
- `hexo-runner` consumes engine and utility contracts, hosts players, and
  applies their actions through the engine.
- `hexo-train` consumes engine, runner, and utility contracts to build training
  runs, load model plugins, and write run outputs.
- `hexo-model-*` packages consume engine, utility, and runner player contracts
  so model-backed players report the same identity and decision shapes as every
  other participant. They also consume the small `hexo-train` plugin contract
  when they expose training components.
- `hexo-frontend` owns browser UI, static assets, and lightweight local web
  servers. It may depend on runner APIs, but runner packages must not import it.

The runner should discover or receive model-backed players through adapters and
plugins. It should not import or hard-code concrete model architectures.

## Current And Target Package Layout

```text
Cargo.toml                    # workspace listing package-local Rust manifests

packages/
  hexo_engine/
    pyproject.toml
    Cargo.toml
    python/
      hexo_engine/
        __init__.py
        api.py
        types.py
        errors.py
        py.typed
    rust/
      src/
        lib.rs
        board.rs
        coord.rs
        rules.rs
        state.rs
        tactics.rs
        identity.rs
        snapshot.rs
        error.rs
        pybridge.rs

  hexo_utils/
    pyproject.toml
    Cargo.toml
    python/
      hexo_utils/
        __init__.py
        py.typed
        rust_bridge.py
        encoding/
          __init__.py
          crop.py
          masks.py
          symmetry.py
        search/
          __init__.py
          mcts.py
        samples/
          __init__.py
          buffer.py
          records.py
          targets.py
    rust/
      src/
        lib.rs
        encoder.rs
        mcts.rs
        position.rs
        pybridge.rs
        samples.rs
        mcts/
          evaluator.rs
          search.rs
          tree.rs

  hexo_runner/
    pyproject.toml
    python/
      hexo_runner/
        __init__.py
        cli.py
        config.py
        player.py
        session.py
        loop.py
        py.typed
        records/
          __init__.py
          events.py
          record.py
          results.py
        modes/
          __init__.py
          match.py
          batch.py
          evaluation.py
          selfplay.py

  hexo_frontend/
    pyproject.toml
    python/
      hexo_frontend/
        __init__.py
        dashboard.py
        static/
          app.js
          index.html
          styles.css
        web.py
        py.typed

  hexo_train/
    pyproject.toml
    python/
      hexo_train/
        __init__.py
        artifacts.py
        checkpoints.py
        components.py
        config.py
        context.py
        defaults.py
        diagnostics.py
        pipeline.py
        registry.py
        symmetry.py
        py.typed
        cli/
          __init__.py
          train_model.py
        epoch/
          __init__.py
          loop.py
          samples.py
          selfplay.py
          symmetry.py
          training.py

  hexo_model_resnet/
    pyproject.toml
    python/
      hexo_model_resnet/
        __init__.py
        architecture.py
        augment.py
        checkpoints.py
        config.py
        decode.py
        diagnostics.py
        inference.py
        input.py
        losses.py
        player.py
        plugin.py
        py.typed
        samples.py
        trainer.py
        training.py
    Cargo.toml                 # optional, only if this model has Rust code
    rust/                      # optional, only if this model has Rust code
      src/

  hexo_model_*/
    pyproject.toml
    Cargo.toml                 # optional, only if this model has Rust code
    python/
      hexo_model_*/
    rust/                      # optional, only if this model has Rust code
      src/

docs/
  structure/
    PROJECT_STRUCTURE.md
    HEXO_ENGINE.md
    HEXO_RUNNER.md
    HEXO_UTILS.md
    HEXO_TRAIN.md
    HEXO_MODEL.md

data/
  checkpoints/
    .gitkeep                   # runtime checkpoint output location
  replay/
    .gitkeep                   # detached runner replay output location
  selfplay/
    .gitkeep                   # self-play output location

tests/
  test_training_pipeline_simplification.py
  engine/                     # target grouping as coverage expands
  utils/
  runner/
  models/
  integration/
```

This layout reflects both the current project tree and the near-term target.
Optional model-local Rust directories are part of the final package rule, but
`hexo_model_resnet` is currently Python-only.

## Documentation Set

The structure docs are deliberately small and package-oriented:

- `PROJECT_STRUCTURE.md`: repository layout, package ownership, and dependency
  direction.
- `HEXO_ENGINE.md`: rule/state authority.
- `HEXO_RUNNER.md`: game execution and records.
- `HEXO_UTILS.md`: reusable mechanisms.
- `HEXO_TRAIN.md`: self-play epoch training orchestration.
- `HEXO_MODEL.md`: model package responsibilities.
- `HEXO_FRONTEND.md`: browser-facing local tools and dashboards.

Avoid adding separate design-note files for training, review notes, or package
plans. Fold durable information into the package doc that owns it.

## Package Layout Rule

Python and Rust code live together inside the package that owns the behavior.
There is no separate top-level `crates/` source tree. A repository-level Cargo
workspace may list package-local manifests, but Rust source stays under the
owning package.

Packages that expose Rust to Python use PyO3 through maturin. This keeps the
Python wheel and Rust extension in one package while preserving a narrow,
typed bridge surface.

For example, `packages/hexo_engine` owns both:

- the host-facing Python API under `python/hexo_engine`;
- the Rust authority implementation under `rust/src`.

The same rule applies to utilities and model packages when they need Rust code.
If a package is Python-only, it simply omits `Cargo.toml` and `rust/`.

## Runtime Flow

```text
runner asks engine for state context
runner asks a player for an action
player may use model and utility code internally
runner submits action to engine
engine validates and applies the action
runner emits events and writes durable game records
model packages write trainable samples during self-play
```

## Record And Sample Layers

Record and sample data are layered by ownership:

- core game records: position trail, accepted actions, state snapshots,
  players, seeds, terminal result, and run outcome;
- training samples: model-owned samples written during self-play, with
  legal-action ordering, policy/search outputs, value targets once finalized,
  and optional references back to detached game records;
- sampled symmetries: deterministic D6 transforms chosen by `hexo_train` per
  training sample and applied by model-owned mappers;
- model extensions: model-owned payloads for anything beyond the common sample
  helpers;
- model training examples: model-specific tensors, masks, targets, and weights.

Core game records are detached neutral facts for analysis, audit, and
recordkeeping. Training samples are model-owned: by default, a model writes and
trains only on its own self-play samples, targets, masks, and weights.

Within `hexo_utils.samples`, the mechanical sample buffer files are deliberately
consolidated into `buffer.py` until real chunk IO, durable indexes, and sampling
logic become substantial enough to split. Record/schema definitions remain in
`records.py`, and common target construction remains in `targets.py`.

## Design Rules

- Rules authority belongs only to the engine.
- The runner does not know model tensor layouts.
- Models do not validate moves independently of the engine.
- Utilities provide mechanisms, not policy decisions.
- Model packages own architecture-specific representations and training logic.
- Package boundaries should be explicit enough that one model family can be
  replaced or retired without touching unrelated systems.
