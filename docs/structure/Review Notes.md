# Review Notes

Use this file as a simple pass-by-pass checklist while reviewing the project. Each Python or Rust file gets the same note shape:

- Questions: things to ask or clarify.
- Comments: observations, decisions, or approvals.
- Concerns: risks, mismatches, or cleanup items.

## Top Level

### `patterns.rs`

Role: Top-level Rust pattern reference or prototype file.

Questions:
- 

Comments:
- 

Concerns:
- 

## `packages/hexo_engine`

### `packages/hexo_engine/python/hexo_engine/__init__.py`

Role: Python package exports for the engine API.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/python/hexo_engine/api.py`

Role: Python-facing wrapper for the Rust engine.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/python/hexo_engine/errors.py`

Role: Python engine error types and error translation boundary.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/python/hexo_engine/types.py`

Role: Python engine data types used by callers and wrappers.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/rust/src/lib.rs`

Role: Rust crate entry point and public module exports.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/rust/src/board.rs`

Role: Board representation and board-level operations.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/rust/src/coord.rs`

Role: Coordinate representation, conversion, and validation.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/rust/src/error.rs`

Role: Rust engine error types.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/rust/src/identity.rs`

Role: Player, placement, and identity naming.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/rust/src/pybridge.rs`

Role: PyO3 bridge from Rust engine types into Python.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/rust/src/rules.rs`

Role: Legal move, terminal state, and rule enforcement logic.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/rust/src/snapshot.rs`

Role: Stable state snapshots for external callers and records.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/rust/src/state.rs`

Role: Mutable game state and state transitions.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_engine/rust/src/tactics.rs`

Role: Tactical helpers used by rules/search/model inputs.

Questions:
- 

Comments:
- 

Concerns:
- 

## `packages/hexo_utils`

### `packages/hexo_utils/python/hexo_utils/__init__.py`

Role: Python package exports for shared utility mechanisms.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/rust_bridge.py`

Role: Python import boundary for Rust-backed utility functions.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/encoding/__init__.py`

Role: Encoding package exports.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/encoding/crop.py`

Role: Crop and position-window helpers for model inputs.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/encoding/masks.py`

Role: Shared legal/threat mask helper shapes.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/encoding/symmetry.py`

Role: D6 symmetry helpers for positions and targets.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/samples/__init__.py`

Role: Sample utility package exports.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/samples/index.py`

Role: Neutral sample index construction and refresh helpers.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/samples/records.py`

Role: Neutral per-position sample record shapes.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/samples/sampling.py`

Role: Shared sampling helpers for sample windows or batches.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/samples/schema.py`

Role: Shared sample schema descriptors.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/samples/store.py`

Role: Sample store directory and manifest mechanics.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/samples/targets.py`

Role: Shared default target helper shapes.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/samples/window.py`

Role: Training sample window selection mechanics.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/samples/writer.py`

Role: Sample append/write mechanics.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/search/__init__.py`

Role: Search package exports.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/python/hexo_utils/search/mcts.py`

Role: Python import/config boundary for Rust MCTS.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/rust/src/lib.rs`

Role: Rust utility crate entry point and module exports.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/rust/src/encoder.rs`

Role: Rust position encoding helpers.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/rust/src/mcts.rs`

Role: Rust MCTS module entry point.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/rust/src/mcts/evaluator.rs`

Role: Evaluation interface used by MCTS.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/rust/src/mcts/search.rs`

Role: MCTS search loop.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/rust/src/mcts/tree.rs`

Role: MCTS tree and node storage.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/rust/src/position.rs`

Role: Rust position view shared by encoding/search utilities.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/rust/src/pybridge.rs`

Role: PyO3 bridge for Rust utilities.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_utils/rust/src/samples.rs`

Role: Rust sample helpers, if sample hot paths move out of Python.

Questions:
- 

Comments:
- 

Concerns:
- 

## `packages/hexo_runner`

### `packages/hexo_runner/python/hexo_runner/__init__.py`

Role: Runner package exports.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/cli.py`

Role: Runner command line entry points.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/config.py`

Role: Runner configuration shapes.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/loop.py`

Role: Shared game loop orchestration.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/player.py`

Role: Player and inference adapter interfaces.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/session.py`

Role: Runner session setup and lifecycle state.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/modes/__init__.py`

Role: Runner mode package exports.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/modes/batch.py`

Role: Batch execution mode.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/modes/evaluation.py`

Role: Evaluation game mode.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/modes/match.py`

Role: Match game mode.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/modes/selfplay.py`

Role: Self-play game mode used by training or standalone runs.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/records/__init__.py`

Role: Runner record package exports.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/records/events.py`

Role: Event records emitted during games.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/records/record.py`

Role: Complete detached game record shape.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_runner/python/hexo_runner/records/results.py`

Role: Game result summaries.

Questions:
- 

Comments:
- 

Concerns:
- 

## `packages/hexo_train`

### `packages/hexo_train/python/hexo_train/__init__.py`

Role: Training package exports.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/components.py`

Role: Shared/default/model-specific training component contracts.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/config.py`

Role: YAML/TOML training config loading and validation.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/context.py`

Role: Per-run paths, diagnostics, and stage output state.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/defaults.py`

Role: Shared default training helpers and component factory.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/diagnostics.py`

Role: Training diagnostics writer.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/pipeline.py`

Role: Canonical stage-based training orchestration.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/registry.py`

Role: Dynamic model training plugin loading.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/cli/__init__.py`

Role: Training CLI package marker.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/cli/train_model.py`

Role: `train_model.py` CLI entry point.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/stages/__init__.py`

Role: Training stage package marker.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/stages/artifacts.py`

Role: Final run artifact and diagnostics stage.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/stages/checkpoint.py`

Role: Checkpoint load/save and self-play checkpoint pointer stages.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/stages/samples.py`

Role: Sample store, finalization, index, and window stages.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/stages/selfplay.py`

Role: Optional self-play sample generation stage.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_train/python/hexo_train/stages/training.py`

Role: Configured training-step stage.

Questions:
- 

Comments:
- 

Concerns:
- 

## `packages/hexo_model_resnet`

### `packages/hexo_model_resnet/python/hexo_model_resnet/__init__.py`

Role: ResNet model package exports.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/architecture.py`

Role: ResNet model architecture.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/augment.py`

Role: Model-specific training augmentation.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/checkpoints.py`

Role: ResNet checkpoint artifact handling.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/config.py`

Role: ResNet-specific model/training config.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/decode.py`

Role: ResNet sample decoding from stored samples into tensors.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/diagnostics.py`

Role: ResNet-specific training/inference diagnostics.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/inference.py`

Role: ResNet inference adapter behavior.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/input.py`

Role: ResNet input tensor construction.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/losses.py`

Role: ResNet loss functions.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/player.py`

Role: ResNet-backed player integration.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/plugin.py`

Role: ResNet model/training plugin boundary.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/samples.py`

Role: ResNet self-play sample finalization.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/trainer.py`

Role: ResNet trainer component used by `hexo_train`.

Questions:
- 

Comments:
- 

Concerns:
- 

### `packages/hexo_model_resnet/python/hexo_model_resnet/training.py`

Role: ResNet training-loop details and optimizer behavior.

Questions:
- 

Comments:
- 

Concerns:
- 
