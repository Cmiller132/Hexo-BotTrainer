"""Checkpoint IO for Hexformer AR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


class HexformerCheckpointLoader:
    def load(self, checkpoint_ref: object | None, *, ctx: Any, components: Any) -> dict[str, Any]:
        if checkpoint_ref is None:
            return {"status": "initialized", "checkpoint_ref": None}
        path = _resolve_checkpoint_ref(Path(str(checkpoint_ref)))
        if path is None:
            return {
                "status": "initialized",
                "checkpoint_ref": str(checkpoint_ref),
                "reason": "checkpoint pointer is not published yet",
            }
        payload = torch.load(path, map_location="cpu")
        components.model.model.load_state_dict(payload["model_state"])
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None and components.model.optimizer is not None:
            components.model.optimizer.load_state_dict(optimizer_state)
        trainer = getattr(components.model, "trainer", None)
        buffer = getattr(trainer, "buffer", None)
        if buffer is not None and payload.get("sample_buffer") is not None:
            buffer.load_state_dict(payload["sample_buffer"])
        if trainer is not None and payload.get("trainer_state") is not None and hasattr(trainer, "load_state_dict"):
            trainer.load_state_dict(payload["trainer_state"])
        return {
            "status": "loaded",
            "checkpoint_ref": str(path),
            "epoch": payload.get("epoch"),
            "metadata": payload.get("metadata", {}),
            "sample_count": getattr(buffer, "sample_count", None),
        }


class HexformerCheckpointSaver:
    def save(self, *, name: str, ctx: Any, components: Any) -> Path:
        path = ctx.checkpoint_dir / f"{name}.pt"
        trainer = getattr(components.model, "trainer", None)
        buffer = getattr(trainer, "buffer", None)
        payload = {
            "model": "hexo_models.hexformer_ar",
            "model_state": components.model.model.state_dict(),
            "optimizer_state": (
                components.model.optimizer.state_dict()
                if components.model.optimizer is not None
                else None
            ),
            "sample_buffer": buffer.state_dict() if buffer is not None else None,
            "trainer_state": trainer.state_dict() if trainer is not None and hasattr(trainer, "state_dict") else None,
            "epoch": _epoch_from_name(name) or _latest_epoch(ctx, components),
            "metadata": {
                "run": ctx.config.run.name,
                "sample_count": getattr(buffer, "sample_count", None),
            },
        }
        torch.save(payload, path)
        return path


def _resolve_checkpoint_ref(path: Path) -> Path | None:
    resolved = path.expanduser()
    if resolved.suffix.lower() == ".txt":
        if not resolved.exists():
            return None
        target = resolved.read_text(encoding="utf-8").strip()
        if not target:
            return None
        target_path = Path(target).expanduser()
        if not target_path.is_absolute():
            target_path = resolved.parent / target_path
        return target_path
    return resolved


def _epoch_from_name(name: str) -> int | None:
    if not name.startswith("epoch_"):
        return None
    try:
        return int(name.removeprefix("epoch_"))
    except ValueError:
        return None


def _latest_epoch(ctx: Any, components: Any) -> int | None:
    epoch_outputs = getattr(ctx, "epoch_outputs", ())
    if epoch_outputs:
        return int(getattr(epoch_outputs[-1], "epoch"))
    checkpoint_state = getattr(getattr(components, "shared", None), "checkpoint_state", None)
    if isinstance(checkpoint_state, Mapping) and checkpoint_state.get("epoch") is not None:
        return int(checkpoint_state["epoch"])
    return None
