"""Checkpoint lifecycle stages.

`hexo_train` owns when checkpoints are loaded and saved. Model packages own
the checkpoint contents because tensor names, optimizer state, and architecture
metadata are model-specific.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def load_or_initialize_checkpoint(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Load an existing checkpoint or describe a fresh initialization."""

    checkpoint_config = ctx.section("checkpoint")
    checkpoint_ref = checkpoint_config.get("resume_from") or checkpoint_config.get(
        "initialize_from"
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


def save_checkpoint(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Save model-owned checkpoint contents into the run checkpoint directory."""

    checkpoint_config = ctx.section("checkpoint")
    name = str(checkpoint_config.get("save_name", "latest"))

    saver = components.model.checkpoint_saver
    if saver is not None:
        path = saver.save(name=name, ctx=ctx, components=components)
    else:
        path = components.shared.defaults.checkpoint_store.write_placeholder(
            name,
            {
                "model": ctx.config.model.name,
                "note": "Placeholder checkpoint; model saver not wired yet.",
            },
        )

    return {"checkpoint_path": str(path)}


def update_selfplay_checkpoint_pointer(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Optionally publish the latest checkpoint for future self-play workers."""

    selfplay_config = ctx.section("selfplay")
    if not bool(selfplay_config.get("update_checkpoint_pointer", False)):
        return {"status": "skipped", "reason": "selfplay pointer update disabled"}

    checkpoint_result = ctx.stage_outputs.get("save_checkpoint", {})
    checkpoint_path = checkpoint_result.get("checkpoint_path")
    pointer_path = Path(selfplay_config.get("checkpoint_pointer", ctx.output_dir / "selfplay_checkpoint.txt"))
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(str(checkpoint_path or ""), encoding="utf-8")
    return {"status": "updated", "pointer_path": str(pointer_path)}
