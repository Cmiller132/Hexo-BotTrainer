# Project Structure

## Goal

Hexo-RL is organized around four clear project families:

- `hexo-engine`: canonical game rules and state authority.
- `hexo-runner`: game execution and orchestration.
- `hexo-utils`: reusable mechanisms shared across packages.
- `hexo-model-*`: model families and learned decision systems.

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
```

In practice:

- `hexo-engine` stands alone and owns game truth.
- `hexo-utils` may depend on engine contracts for reusable mechanisms.
- `hexo-runner` consumes engine and utility contracts, hosts players, and
  applies their actions through the engine.
- `hexo-model-*` packages consume engine, utility, and runner player contracts
  so model-backed players report the same identity and decision shapes as every
  other participant.

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
          schema.py
          records.py
          sampling.py
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

  hexo_model_resnet/
    pyproject.toml
    python/
      hexo_model_resnet/
        __init__.py
        architecture.py
        augment.py
        checkpoints.py
        config.py
        diagnostics.py
        inference.py
        input.py
        losses.py
        player.py
        plugin.py
        py.typed
        training.py
    Cargo.toml                 # optional, only if this model has Rust code
    rust/                      # optional, only if this model has Rust code
      src/

  hexo_model_*/
    pyproject.toml
    Cargo.toml                 # optional, only if this model has Rust code
    python/
      hexo_model_*/
    rust/
      src/

docs/
  structure/
    TRAINING_INFO.md

tests/
  engine/
  utils/
  runner/
  models/
  integration/
```

This layout reflects both the current project tree and the near-term target.
Optional model-local Rust directories are part of the final package rule, but
`hexo_model_resnet` is currently Python-only.

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
- sampled symmetries: deterministic D6 transforms chosen per training sample
  and applied by model-owned mappers;
- model extensions: model-owned payloads for anything beyond the common sample
  helpers;
- model training examples: model-specific tensors, masks, targets, and weights.

Core game records are detached neutral facts for analysis, audit, and
recordkeeping. Training samples are model-owned: by default, a model writes and
trains only on its own self-play samples, targets, masks, and weights.

## Design Rules

- Rules authority belongs only to the engine.
- The runner does not know model tensor layouts.
- Models do not validate moves independently of the engine.
- Utilities provide mechanisms, not policy decisions.
- Model packages own architecture-specific representations and training logic.
- Package boundaries should be explicit enough that one model family can be
  replaced or retired without touching unrelated systems.
