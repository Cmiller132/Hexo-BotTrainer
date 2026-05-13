# HEXO_TRAIN

## Purpose

`hexo-train` is the shared training orchestration package. It owns the
config-driven lifecycle for training runs: loading config, loading a model
plugin, building shared components, running configured stages, and writing run
outputs.

It is intentionally separate from `hexo-utils`. Utilities provide reusable
mechanisms; training decides lifecycle, ordering, outputs, checkpoints, and
diagnostics.

## Owns

- Training CLI entry points.
- YAML/TOML training config loading.
- Model plugin discovery and loading.
- Shared/default training component construction.
- Stage orchestration.
- Run output directories.
- Checkpoint layout.
- Diagnostics and run manifests.
- Coordination between shared sample buffers and model-owned training code.

## Does Not Own

- Game rules or legal move authority.
- Runner game-loop behavior.
- Model architecture.
- Model tensor layout.
- Model-specific sample decoding.
- Policy/value target semantics.
- Loss functions.
- Optimizer details unless supplied by a model plugin.
- Self-play game records.

## Package Layout

```text
packages/hexo_train/
  pyproject.toml
  python/
    hexo_train/
      __init__.py
      config.py
      context.py
      pipeline.py
      registry.py
      components.py
      defaults.py
      diagnostics.py
      py.typed
      cli/
        __init__.py
        train_model.py
      stages/
        __init__.py
        artifacts.py
        checkpoint.py
        samples.py
        selfplay.py
        training.py
```

## CLI

The public entry point is:

```text
hexo-train-model path/to/train.toml
hexo-train-model path/to/train.yaml
```

`train_model.py` should stay thin:

```text
parse config path
TrainingPipeline().run(config_path)
return process status
```

Command-line parsing should not contain training policy. The pipeline owns the
run lifecycle.

## Config

Training config should describe:

- run identity and output directory,
- model plugin name or module,
- shared game/sample settings,
- model-specific config,
- optional stage subset to run in the canonical order.

The config loader normalizes YAML/TOML into one shared `TrainingConfig` shape.
It should validate required fields early so failures happen before long-running
jobs start.

## Plugin Loading

`registry.py` loads the requested model plugin dynamically. A model package may
be discovered by entry point or by explicit module during development.

The plugin boundary should allow:

```text
build model
build model-specific training components
run model-owned stages
decode model-owned samples
write model-owned checkpoints or metadata
```

`hexo-train` calls these hooks but does not interpret model tensors or targets.

## Components

Shared components are model-neutral:

- output directory,
- checkpoint directory,
- diagnostics writer,
- sample source description,
- engine/game spec,
- checkpoint store;
- default scalar value target helper;
- default legal-action policy target helper.

Model components are plugin-owned:

- model instance,
- trainer,
- optimizer,
- sample decoder,
- loss behavior,
- extra model-specific handles.

The pipeline builds shared components first, then gives those to the model
plugin so the model can construct whatever it needs.

## Defaults And Overrides

The key abstraction is a model training plugin. The shared package builds a
default component set, then the model plugin returns only the pieces it wants
to replace.

```text
defaults = build shared defaults
overrides = model_plugin.training_component_overrides(defaults, config, shared)
components = defaults with overrides applied
```

Example default:

```text
ScalarValueTargetHelper:
    winner == sample perspective -> +1.0
    winner != sample perspective -> -1.0
    draw or no result           ->  0.0
```

A normal policy/value model can use that default directly. A model with richer
value targets, such as value distributions or multiple outcome heads, replaces
only the value target helper. The same pattern applies to legal policy targets,
sample decoding, trainers, optimizers, checkpoint writers, and stage handlers.

## Pipeline Flow

```text
load and validate config
create run context
load model training plugin
build default and model-specific components
load or initialize checkpoint
prepare sample store
optionally generate self-play samples
finalize pending samples
refresh sample index
build sample window
train configured steps
save checkpoint
optionally update self-play checkpoint pointer
write diagnostics
```

`hexo-train` owns this order. Config may select a subset for development or
debugging, but it should not define a different order. Each stage calls shared
default logic unless the model plugin provides a specific override.

## Training Data Boundary

The normal training path is:

```text
self-play creates model-owned trainable samples
hexo_utils.samples stores and serves sample chunks
hexo_train orchestrates training stages
model plugin decodes samples into tensors
model plugin computes losses and updates weights
hexo_train writes run outputs and diagnostics
```

Runner game records remain detached from training. They are useful for
analysis, audit, and debugging, but they are not the normal source of training
targets.

## Dependency Direction

`hexo-train` may depend on:

- `hexo-engine` for game/spec contracts,
- `hexo-utils` for sample buffers and shared mechanisms,
- `hexo-runner` for self-play orchestration contracts,
- `hexo-model-*` only through dynamic plugin loading.

Concrete model packages should not be hard-coded into `hexo-train`.

## Design Rules

- Keep lifecycle orchestration in `hexo-train`.
- Keep reusable mechanics in `hexo-utils`.
- Keep game execution in `hexo-runner`.
- Keep rule authority in `hexo-engine`.
- Keep tensor layouts, targets, losses, and checkpoint meaning in model
  packages.
- Prefer explicit plugin hooks over model-specific conditionals in the training
  package.
