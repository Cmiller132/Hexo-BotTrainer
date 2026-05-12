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

The intended dependency direction is:

```text
hexo-engine
   ^
   |
hexo-utils
   ^
   |
hexo-model-*
   ^
   |
hexo-runner
```

In practice:

- `hexo-engine` stands alone and owns game truth.
- `hexo-utils` may depend on engine contracts for reusable mechanisms.
- `hexo-model-*` packages consume engine and utility contracts.
- `hexo-runner` hosts players and applies their actions through the engine.

The runner should discover or receive model-backed players through adapters. It
should not hard-code model architectures.

## Desired Package Layout

```text
packages/
  hexo_engine/
    pyproject.toml
    python/hexo_engine/

  hexo_utils/
    pyproject.toml
    python/hexo_utils/

  hexo_runner/
    pyproject.toml
    python/hexo_runner/

  hexo_model_resnet/
    pyproject.toml
    python/hexo_model_resnet/

  hexo_model_*/
    pyproject.toml
    python/hexo_model_*/

crates/
  hexgame_engine/
  hexgame_utils/
  hexgame_model_*/          # optional per model family

docs/
  structure/

tests/
  engine/
  utils/
  runner/
  models/
  integration/
```

## Rust Layout

The Rust crates mirror the same authority boundaries:

- `hexgame-engine`: rules, state, legal actions, terminal detection, tactics,
  identity, and replayable state.
- `hexgame-utils`: shared encoder/search/replay/runtime helpers that depend on
  the engine but do not own rules.
- `hexgame-model-*`: optional model-specific Rust code for representation,
  search, or high-volume data paths.

Python packages expose and compose these Rust crates through narrow APIs.

## Runtime Flow

```text
runner asks engine for state context
runner asks a player for an action
player may use model and utility code internally
runner submits action to engine
engine validates and applies the action
runner records events, replay, timings, and diagnostics
model packages turn records into training examples
```

## Replay Layers

Replay and training data are layered by ownership:

- engine history: accepted actions, state snapshots, terminal result,
  rules/version metadata;
- runner metadata: players, seeds, budgets, timings, execution outcome;
- model diagnostics: policy, value, search, or architecture-specific metadata;
- model training records: model-specific examples, masks, targets, and weights.

This allows multiple model families to train from the same games without
forcing them to share target semantics.

## Design Rules

- Rules authority belongs only to the engine.
- The runner does not know model tensor layouts.
- Models do not validate moves independently of the engine.
- Utilities provide mechanisms, not policy decisions.
- Model packages own architecture-specific representations and training logic.
- Package boundaries should be explicit enough that one model family can be
  replaced or retired without touching unrelated systems.
