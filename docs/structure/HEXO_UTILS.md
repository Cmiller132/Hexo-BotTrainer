# HEXO_UTILS

## Purpose

`hexo-utils` holds reusable mechanisms shared by model packages, runner modes,
training pipelines, and tests.

It reduces duplicated infrastructure without becoming a second engine, a model
policy layer, or a catch-all package.

## Owns

- Shared encoding helpers when they are not model-specific.
- Generic MCTS or tree-search machinery.
- Classical search baselines.
- Replay serialization helpers.
- Schema and version helpers.
- Training adapter framework mechanics.
- Batching and queue helpers.
- Resource profile helpers.
- Telemetry helpers.
- Test harness and mutation helpers.
- Deterministic seeding utilities.

## Does Not Own

- Game legality.
- Terminal state authority.
- Model architecture.
- Model-specific target meaning.
- Runner lifecycle.
- Policy decisions.
- Reward interpretation.

## Package Layout

```text
packages/hexo_utils/
  pyproject.toml
  python/hexo_utils/
    __init__.py
    encoding/
      __init__.py
      crop.py
      masks.py
    search/
      __init__.py
      mcts.py
      alphabeta.py
      tree.py
    replay/
      __init__.py
      schema.py
      records.py
      sampling.py
    runtime/
      __init__.py
      resources.py
      queues.py
      batching.py
    telemetry/
      __init__.py
      events.py
      metrics.py
    testing/
      __init__.py
      fixtures.py
      mutators.py

crates/hexgame_utils/
  Cargo.toml
  src/
    lib.rs
    encoder.rs
    mcts.rs
    search.rs
    replay.rs
    testing.rs
```

`hexgame-utils` may depend on `hexgame-engine`, but it must not duplicate
engine rules.

## Utility Rule

A helper belongs in `hexo-utils` when:

- it has more than one plausible consumer,
- it has a stable contract,
- it does not make model-specific policy choices,
- it does not reinterpret game legality or terminal state.

Otherwise it belongs in the engine, runner, or a model package.

## Core Areas

`encoding`: shared board/crop/mask encoders for model families that agree on a
common representation.

`search`: reusable search machinery, tree statistics, PUCT/MCTS helpers, and
classical baselines.

`replay`: schemas, record helpers, sampling mechanics, and validation tools.

`runtime`: batching, queues, resource profiles, and backpressure primitives.

`telemetry`: counters, timers, event helpers, and report fragments.

`testing`: fixtures, mutators, round-trip checks, and stress-test helpers.

## Contract Flow

```text
engine supplies canonical state and legal actions
utils supplies reusable mechanisms
model supplies policy interpretation and training meaning
runner supplies orchestration
```

Utilities should remain opt-in. A model package can use shared helpers when
the semantics match, or keep custom code when its representation needs it.
