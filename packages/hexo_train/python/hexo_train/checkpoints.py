"""Checkpoint load/save helpers for self-play training.

`hexo_train` owns when checkpoints are loaded and saved. The selected model
plugin owns what a checkpoint contains. These helpers therefore delegate to
model-provided loader/saver objects when they exist, and otherwise write small
placeholder metadata files so the pipeline remains executable during early
development.
"""

from __future__ import annotations

from typing import Any

from .components import TrainingComponents
from .context import RunContext


def load_or_initialize_checkpoint(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Load an existing checkpoint or describe a fresh initialization.

    This runs once before epoch 1. The resulting state is stored on
    `components.shared.checkpoint_state` so self-play generation can see the
    current model reference.
    """

    checkpoint_ref = (
        ctx.config.checkpoint.resume_from
        or ctx.config.checkpoint.initialize_from
    )
    loader = components.model.checkpoint_loader
    if loader is not None:
        state = loader.load(checkpoint_ref, ctx=ctx, components=components)
    else:
        state = {
            "status": "initialized" if checkpoint_ref is None else "referenced",
            "checkpoint_ref": str(checkpoint_ref) if checkpoint_ref else None,
            "note": "Model checkpoint loading is not implemented yet.",
        }

    components.shared.checkpoint_state = state
    return state


def save_epoch_checkpoint(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Save the checkpoint that will feed the next epoch's self-play."""

    return _save_checkpoint(
        ctx,
        components,
        name=f"epoch_{epoch:06d}",
        metadata={"epoch": epoch, "kind": "epoch"},
    )


def save_final_checkpoint(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Save the final model checkpoint for this run.

    Epoch checkpoints are named by epoch number. The final checkpoint uses the
    user-configured `checkpoint.save_name` so external tools have a stable name.
    """

    return _save_checkpoint(
        ctx,
        components,
        name=ctx.config.checkpoint.save_name,
        metadata={"kind": "final"},
    )


def _save_checkpoint(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Shared save path for epoch and final checkpoints.

    The helper updates `components.shared.checkpoint_state` after every save.
    That is the small but important handoff from training to the next self-play
    generation call.
    """

    saver = components.model.checkpoint_saver
    if saver is not None:
        path = saver.save(name=name, ctx=ctx, components=components)
    else:
        path = components.shared.defaults.checkpoint_store.write_placeholder(
            name,
            {
                "model": ctx.config.model.name,
                "note": "Placeholder checkpoint; model saver not wired yet.",
                **metadata,
            },
        )

    result = {"checkpoint_path": str(path), "name": name, **metadata}
    components.shared.checkpoint_state = result
    return result
