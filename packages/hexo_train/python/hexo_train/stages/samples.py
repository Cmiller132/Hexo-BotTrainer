"""Sample-store lifecycle stages.

`hexo_utils.samples` should eventually provide the real store, index, and
window implementations. These stages define where that shared mechanics layer
plugs into the training lifecycle.
"""

from __future__ import annotations

from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def prepare_sample_store(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Open or create the model-owned sample store for this run."""

    from hexo_utils.samples import open_sample_store

    sample_config = ctx.section("samples")
    sample_store = open_sample_store(
        sample_config.get("path", ctx.samples_dir),
        mode=str(sample_config.get("mode", "append")),
        metadata={"run": ctx.config.run.name},
    )
    components.shared.sample_store = sample_store
    return {"path": str(sample_store.path), "mode": sample_store.mode}


def finalize_pending_samples(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Let the model finalize result-dependent samples after self-play."""

    finalizer = components.model.sample_finalizer
    if finalizer is not None:
        return finalizer.finalize(ctx=ctx, components=components)
    return {
        "status": "skipped",
        "reason": "model sample finalizer not wired yet",
    }


def refresh_sample_index(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Refresh the searchable index over finalized sample chunks."""

    from hexo_utils.samples import refresh_sample_index as refresh_index

    sample_index = refresh_index(components.shared.sample_store)
    components.shared.sample_index = sample_index
    return {
        "sample_count": sample_index.sample_count,
        "store": str(sample_index.store.path),
        "metadata": dict(sample_index.metadata),
    }


def build_sample_window(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Choose the training sample window used by the train stage."""

    from hexo_utils.samples import build_sample_window as build_window

    train_config = ctx.section("train")
    raw_window_size = train_config.get("sample_window_size")
    window_size = int(raw_window_size) if raw_window_size is not None else None
    sample_window = build_window(
        components.shared.sample_index,
        window_size=window_size,
        seed=ctx.config.run.seed,
    )
    components.shared.sample_window = sample_window
    return {
        "window_size": sample_window.window_size,
        "seed": sample_window.seed,
        "metadata": dict(sample_window.metadata),
    }
