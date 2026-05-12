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
    Cargo.toml
    python/
      hexo_engine/
    rust/
      src/

  hexo_utils/
    pyproject.toml
    Cargo.toml                 # when shared Rust helpers are needed
    python/
      hexo_utils/
    rust/
      src/

  hexo_runner/
    pyproject.toml
    python/
      hexo_runner/

  hexo_model_resnet/
    pyproject.toml
    Cargo.toml                 # optional, only if this model has Rust code
    python/
      hexo_model_resnet/
    rust/
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

tests/
  engine/
  utils/
  runner/
  models/
  integration/
```

## Package Layout Rule

Python and Rust code live together inside the package that owns the behavior.
There is no separate top-level `crates/` source tree. A repository-level Cargo
workspace may list package-local manifests, but Rust source stays under the
owning package.

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
runner records events, replay, and diagnostics
model packages turn records into training examples
```

## Replay Layers

Replay and training data are layered by ownership:

- engine history: accepted actions, state snapshots, terminal result,
  rules/version metadata;
- runner metadata: players, seeds, and execution outcome;
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
