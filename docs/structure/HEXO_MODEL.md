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
- Model-specific replay extensions beyond the shared policy-logit record.
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
  Cargo.toml                 # optional, only if this model has Rust code
  python/
    hexo_model_resnet/
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
  rust/
    src/
      lib.rs

packages/hexo_model_*/
  pyproject.toml
  Cargo.toml                 # optional, only if this model has Rust code
  python/
    hexo_model_*/
  rust/
    src/
```

Most model packages are Python-first. When a model owns Rust code for
representation-coupled search, high-volume preprocessing, or data structures,
that Rust code lives inside the same `packages/hexo_model_*` directory.

## Interfaces

From engine:

- canonical state or snapshot,
- legal actions,
- tactical summaries,
- state/action identities,
- terminal result and replay history.

To runner:

- a player adapter,
- a self-reported `identity` field matching the runner contract,
- a decision response with selected action,
- optional opaque diagnostics.

To utilities:

- shared encoders, MCTS, and replay helpers when they match the model's
  assumptions.

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
Model-backed players should conform to the runner contract directly rather than
requiring runner-specific special cases.

For self-play, model packages expose an `InferenceAdapter` that returns common
policy logits over the engine-provided legal actions plus any model-owned
extension records the model wants to persist.

## Training Data

Model packages decide how runner records become model-specific examples:

- filter replay records,
- rebuild or load model inputs,
- use the default legal-action policy/value target when it fits,
- construct or transform model-specific targets when it does not,
- parse model-owned replay extensions when needed,
- apply the sampled D6 symmetry consistently to inputs, legal masks, policy
  targets, and any model-owned extensions,
- apply masks and sample weights,
- collate batches,
- compute losses,
- save checkpoints and diagnostics.

The same engine and runner records can feed multiple model families with
different target semantics.
