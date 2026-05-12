# HEXO_MODEL

## Purpose

`hexo-model-*` is a family of model packages. Each package owns one model
architecture family and the code needed to use that architecture for inference,
training, and diagnostics.

Models consume game truth from the engine. They do not define legality,
terminal state, or runner lifecycle.

## Owns

- Network architecture.
- Model configuration.
- State-to-input conversion for that architecture.
- Inference adapter.
- Player adapter for the runner.
- Policy/value decoding.
- Training target construction.
- Losses, augmentation, and batching rules.
- Checkpoint semantics.
- Model-specific diagnostics.
- Optional model-specific search when needed.

## Does Not Own

- Canonical game rules.
- Legal move authority.
- Terminal detection.
- Runner orchestration.
- Shared replay storage policy.
- Cross-model experiment scheduling.
- Generic utility mechanisms shared across model families.

## Package Layout

```text
packages/hexo_model_resnet/
  pyproject.toml
  python/hexo_model_resnet/
    __init__.py
    architecture.py
    config.py
    input.py
    inference.py
    player.py
    training.py
    losses.py
    augment.py
    diagnostics.py
    checkpoints.py

packages/hexo_model_*/
  pyproject.toml
  python/hexo_model_*/

crates/hexgame_model_*/       # optional per model family
```

Most model packages are Python packages. A model-owned Rust crate is reserved
for representation-coupled search, high-volume preprocessing, or data
structures that need Rust performance.

## Interfaces

From engine:

- canonical state or snapshot,
- legal actions,
- tactical summaries,
- state/action identities,
- terminal result and replay history.

To runner:

- a player adapter,
- a decision response with selected action,
- optional opaque diagnostics.

To utilities:

- shared encoders, MCTS, replay helpers, batching, resource profiles, and
  telemetry when they match the model's assumptions.

## Model Player Flow

```text
receive decision request
convert engine state + legal actions into model input
run inference and optional search
select legal action
return action + diagnostics
```

The runner sees only the action and diagnostics. Tensor layouts, logits,
candidate ranking, and search internals stay inside the model package.

## Training Data

Model packages decide how runner records become model-specific examples:

- filter replay records,
- rebuild or load model inputs,
- construct policy and value targets,
- apply masks and sample weights,
- collate batches,
- compute losses,
- save checkpoints and diagnostics.

The same engine and runner records can feed multiple model families with
different target semantics.
