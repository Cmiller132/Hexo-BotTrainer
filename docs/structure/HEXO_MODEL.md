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
- Self-play training sample writing.
- Losses, augmentation, and batching rules.
- Checkpoint semantics.
- Model-specific diagnostics.
- Model-specific sample payloads beyond the shared policy/value helpers.
- Optional model-specific search when needed.

## Does Not Own

- Canonical game rules.
- Legal move authority.
- Terminal detection.
- Runner orchestration.
- Shared sample buffer storage policy.
- Cross-model experiment scheduling.
- Generic utility mechanisms shared across model families.

## Package Layout

```text
packages/hexo_model_resnet/
  pyproject.toml
  python/
    hexo_model_resnet/
      __init__.py
      architecture.py
      config.py
      decode.py
      input.py
      inference.py
      player.py
      training.py
      losses.py
      augment.py
      diagnostics.py
      checkpoints.py
      plugin.py
      samples.py
      trainer.py
      py.typed
  Cargo.toml                 # optional, only if this model has Rust code
  rust/                      # optional, only if this model has Rust code
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
- terminal result and replayable history.

To runner:

- a player adapter,
- a self-reported `identity` field matching the runner contract,
- a decision response with selected action,
- optional opaque diagnostics.

To utilities:

- shared encoders, MCTS, and sample helpers when they match the model's
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

For self-play, model packages expose an `InferenceAdapter` and training sample
writer. The adapter can return policy/search outputs over engine-provided legal
actions; the model writer stores the trainable data it needs while the game is
running, then finalizes value/result targets when the game ends.

## Training Data

Model packages decide how their own self-play decisions become model-specific
training samples:

- capture model inputs or enough references to rebuild them,
- store policy/search outputs produced during self-play,
- use the default legal-action policy/value target when it fits,
- construct or transform model-specific targets when it does not,
- attach model-owned sample payloads when needed,
- finalize value targets after the terminal result is known,
- apply the sampled D6 symmetry consistently to inputs, legal masks, policy
  targets, and any model-owned extensions,
- apply masks and sample weights,
- collate batches,
- compute losses,
- save checkpoints and diagnostics.

By default, a model trains only on its own self-play samples. The saved policy
output, value semantics, extension payloads, and target shapes are model-owned
data. Engine and runner records remain detached, model-neutral facts for
analysis, audit, and recordkeeping; they are not the normal source of training
sample construction.

## ResNet Position Sample

For `hexo_model_resnet`, one sample represents one trainable position from
self-play. A first version should store:

- identity: sample id, model id, checkpoint id, game id, turn index, player;
- state input: encoded board planes or a reference to an encoded input chunk;
- legal policy space: legal action ids in logit order and the selected action;
- policy target: model/search logits or visit-derived policy over those legal
  actions;
- value target: finalized result from the sample player's perspective once the
  game ends;
- masks: legal mask, optional threat-filtered legal mask, and any crop/action
  mapping needed by the model;
- symmetry: the D6 symmetry chosen at sample read time or stored if the sample
  was pre-augmented;
- weights: sample weight and optional per-head weights;
- provenance: RNG seed, self-play mode metadata, and optional detached runner
  record reference for debugging.

The shared samples package may provide the legal-action policy/value container,
buffer writing, indexing, and sampling mechanics. ResNet owns the exact tensor
layout, crop behavior, auxiliary fields, and loss interpretation.

## Model Training Plugin

Each model package exposes a training plugin. The plugin accepts shared
defaults when they match and overrides the model-owned pieces:

- sample finalization;
- sample decoding;
- trainer or train-step behavior;
- checkpoint contents;
- optional stage handlers.

For ResNet, `samples.py` owns pending-sample finalization, `decode.py` owns
sample-to-tensor conversion, `trainer.py` owns train steps and metrics, and
`checkpoints.py` owns model checkpoint contents.
