# HEXO_TRAIN

## Purpose

`hexo-train` is the shared self-play training orchestration package. It owns
the fixed training lifecycle: loading config, loading a model plugin, building
shared components, running self-play epochs, publishing checkpoints, and writing
run outputs.

One epoch means:

```text
self-play generation
finalize samples
select training samples
select sample symmetries
train configured passes
save epoch checkpoint
```

## Owns

- Training CLI entry points.
- YAML/TOML training config loading.
- Model plugin discovery and loading.
- Shared/default training component construction.
- Self-play epoch orchestration.
- Training-time D6 symmetry selection.
- Run output directories.
- Checkpoint layout and checkpoint pointer publishing.
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
```

## File Responsibilities

| File | Role |
| --- | --- |
| `pyproject.toml` | Python package metadata and `hexo-train-model` console entry point. |
| `python/hexo_train/__init__.py` | Public exports for config, context, pipeline, plugin loading, and symmetry selection. |
| `artifacts.py` | Run artifact helpers: checkpoint store, manifests, final diagnostics, and checkpoint pointers. |
| `checkpoints.py` | Checkpoint load/save orchestration and placeholder checkpoint fallback. |
| `components.py` | Shared/model component dataclasses and plugin override assembly. |
| `config.py` | TOML/YAML config loading, validation, and normalized config dataclasses. |
| `context.py` | Run directory layout, diagnostics object, raw sections, and recorded step outputs. |
| `defaults.py` | Shared default component construction. |
| `diagnostics.py` | Stage timing/status records and diagnostics file writing. |
| `pipeline.py` | Top-level run lifecycle from config load through final artifacts. |
| `registry.py` | Model plugin discovery by entry point or explicit module. |
| `symmetry.py` | Reusable deterministic D6 selector used by training epochs. |
| `py.typed` | Marker that the package ships type information. |
| `cli/__init__.py` | CLI package marker. |
| `cli/train_model.py` | Thin training CLI wrapper around `TrainingPipeline`. |
| `epoch/__init__.py` | Epoch package marker. |
| `epoch/loop.py` | `run_epochs()` and `run_epoch()` orchestration. |
| `epoch/selfplay.py` | Self-play request/generation step for one epoch. |
| `epoch/samples.py` | Sample store preparation, finalization, index refresh, and training window selection. |
| `epoch/symmetry.py` | Per-epoch application of the reusable D6 selector. |
| `epoch/training.py` | Delegation to `trainer.train_passes(...)` for selected samples and symmetries. |

## CLI

The public entry point is:

```text
hexo-train-model path/to/train.toml
hexo-train-model path/to/train.yaml
```

`train_model.py` stays thin:

```text
parse config path
TrainingPipeline().run(config_path)
return process status
```

Command-line parsing should not contain training policy. The pipeline owns the
run lifecycle.

## Config

Training config describes:

- run identity and output directory;
- model plugin name, module, entry point, and model-owned settings under
  `model.config`;
- `run.seed`;
- `loop.epochs`;
- `selfplay.games_per_epoch`;
- `selfplay.update_checkpoint_pointer`;
- `selfplay.checkpoint_pointer`;
- `samples.train_sample_count`;
- sample store settings currently read from `samples.path` and `samples.mode`;
- `train.passes_per_epoch`;
- optional shared settings under `shared`;
- checkpoint resume/save settings.

The config loader rejects generic `stages`. Self-play epoch training is the
only implemented path initially.

## Plugin Loading

`registry.py` loads the requested model plugin dynamically. A model package may
be discovered by entry point or by explicit module during development.

The plugin boundary should allow:

```text
build model
build model-specific training components
generate self-play requests or games
finalize model-owned samples
train over selected samples for configured passes
write model-owned checkpoints or metadata
```

`hexo-train` calls direct orchestration hooks such as self-play, finalization,
training, and checkpoint IO. It may store model-owned helpers such as a sample
decoder on the model components, but the current pipeline does not call the
decoder directly; the trainer owns how selected samples become tensors.

## Components

Run context is model-neutral and owns:

- output directory;
- checkpoint directory;
- diagnostics writer;
- sample directory;
- raw config sections;
- step and epoch outputs.

Default training components are model-neutral helpers:

- default scalar value target helper from `hexo_utils.samples`;
- default legal-action policy target helper from `hexo_utils.samples`;
- default D6 symmetry selector.

Shared components are mutable run handles used by the epoch loop:

- sample source description;
- engine/game spec;
- default helpers;
- sample store, index, and selected window;
- sample symmetries;
- checkpoint state;
- self-play result.

Model components are plugin-owned:

- model instance;
- trainer;
- optimizer;
- sample decoder;
- sample finalizer;
- checkpoint loader/saver;
- extra model-specific handles.

The pipeline builds shared components first, then gives those to the model
plugin so the model can construct whatever it needs.

## Pipeline Flow

```text
load and validate config
create run context
load model training plugin
build default and model-specific components
initialize run artifacts and sample store
load or initialize checkpoint
for each epoch:
    generate self-play
    finalize samples
    refresh sample index and select train_sample_count samples
    select deterministic D6 symmetries for the window
    train passes_per_epoch over the selected samples
    save epoch checkpoint
save final checkpoint
optionally update self-play checkpoint pointer
write diagnostics
```

The checkpoint saved at the end of an epoch is the model state used for the next
epoch's self-play.

## Training Data Boundary

Training follows a simple self-play loop:

```text
self-play writes trainable samples
sample buffer stores and shuffles samples
training reads from the sample buffer
new checkpoints feed future self-play
```

The normal package boundary is:

```text
self-play creates model-owned trainable samples
hexo_utils.samples stores and serves sample chunks
hexo_train orchestrates self-play epochs
model plugin decodes samples into tensors
model plugin applies selected D6 symmetries to tensors/targets
model plugin computes losses and updates weights
hexo_train writes run outputs and diagnostics
```

Runner game records remain detached from training. They are useful for
analysis, audit, and debugging, but they are not the normal source of training
targets.

For each model decision during self-play:

1. Runner asks the model-backed player for an action.
2. The model encodes the position.
3. Inference or search produces legal-action policy data and a selected action.
4. The model sample writer stores a pending sample.
5. Runner applies the action through the engine.
6. After self-play returns, `hexo_train` calls the model finalizer.
7. Finalized samples are appended to the model's sample buffer.

One epoch generates fresh games, finalizes result-dependent samples, selects
`samples.train_sample_count` samples, then trains over those samples for
`train.passes_per_epoch` passes.

## Dependency Direction

`hexo-train` may depend on:

- `hexo-engine` for game/spec contracts;
- `hexo-utils` for sample buffers and shared mechanisms;
- `hexo-runner` for self-play orchestration contracts;
- `hexo-model-*` only through dynamic plugin loading.

Concrete model packages should not be hard-coded into `hexo-train`.

## Design Rules

- Keep lifecycle orchestration in `hexo-train`.
- Keep reusable mechanics in `hexo-utils`.
- Keep game execution in `hexo-runner`.
- Keep rule authority in `hexo-engine`.
- Keep tensor layouts, targets, losses, and checkpoint meaning in model
  packages.
- Keep D6 selection timing in `hexo-train`; keep D6 application in model
  packages.
- Prefer explicit plugin hooks over model-specific conditionals in the training
  package.
