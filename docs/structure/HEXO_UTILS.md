# HEXO_UTILS

## Purpose

`hexo-utils` holds a small set of reusable mechanisms shared by model packages,
runner modes, and training pipelines.

It reduces duplicated infrastructure without becoming a second engine, a model
policy layer, or a catch-all package.

## Owns

- Shared encoding helpers when they are not model-specific.
- Shared MCTS search machinery.
- Replay serialization helpers.
- Schema and version helpers.

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
  Cargo.toml                 # when Rust helpers are needed
  python/
    hexo_utils/
      __init__.py
      encoding/
        __init__.py
        crop.py
        masks.py
      search/
        __init__.py
        mcts.py
      replay/
        __init__.py
        schema.py
        records.py
        sampling.py
  rust/
    src/
      lib.rs
      encoder.rs
      mcts.rs
      replay.rs
```

Any Rust utility code lives inside `packages/hexo_utils` with the Python
utility package. Utility Rust may depend on engine contracts, but it must not
duplicate engine rules.

## Utility Rule

A helper belongs in `hexo-utils` when:

- it has more than one plausible consumer,
- it has a stable contract,
- it does not make model-specific policy choices,
- it does not reinterpret game legality or terminal state.

Otherwise it belongs in the engine, runner, or a model package.

## Core Areas

`encoding`: optional shared board/crop/mask helpers. Crops may be square,
circular, multi-window, or bypassed entirely by model packages that consume the
whole board or a custom representation. Masks translate engine legal actions
into model-facing shapes; threat masks may only strip engine legal actions
using engine tactical facts.

`search`: reusable MCTS machinery and supporting search statistics.

`replay`: schemas, compact canonical records, common policy logits over legal
actions, model-owned extension attachment, sampling mechanics, and validation
tools.

## Contract Flow

```text
engine supplies canonical state and legal actions
utils supplies reusable mechanisms
model supplies policy interpretation and training meaning
runner supplies orchestration
```

Utilities should remain opt-in. A model package can use shared helpers when
the semantics match, or keep custom code when its representation needs it.
