"""Checkpoint IO for the dense CNN model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


class DenseCNNCheckpointLoader:
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
        if not path.exists():
            return {
                "status": "initialized",
                "checkpoint_ref": str(path),
                "reason": "checkpoint target is missing",
            }
        payload = torch.load(path, map_location="cpu")
        sample_buffer = payload.get("sample_buffer")
        trainer = getattr(components.model, "trainer", None)
        buffer = getattr(trainer, "buffer", None)
        sample_buffer_load_stats = None
        if sample_buffer is not None and buffer is not None and hasattr(buffer, "load_state_dict"):
            sample_buffer_load_stats = _sample_buffer_load_stats(buffer.load_state_dict(sample_buffer), buffer)

        model_state = payload.get("model_state")
        incompatibilities = _state_dict_incompatibilities(components.model.model.state_dict(), model_state)
        if incompatibilities:
            return {
                "status": "initialized",
                "checkpoint_ref": str(path),
                "reason": "checkpoint model_state is incompatible with current dense_cnn architecture",
                "incompatible_tensors": incompatibilities,
                "checkpoint_epoch": payload.get("epoch"),
                "metadata": payload.get("metadata", {}),
                "sample_count": getattr(buffer, "sample_count", None),
                "sample_buffer_load_stats": sample_buffer_load_stats,
                "sample_buffer_loaded_count": _sample_buffer_loaded_count(sample_buffer_load_stats),
                "sample_buffer_filtered_count": _sample_buffer_filtered_count(sample_buffer_load_stats),
            }

        components.model.model.load_state_dict(model_state)
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None and components.model.optimizer is not None:
            components.model.optimizer.load_state_dict(optimizer_state)
        return {
            "status": "loaded",
            "checkpoint_ref": str(path),
            "epoch": payload.get("epoch"),
            "metadata": payload.get("metadata", {}),
            "sample_count": getattr(buffer, "sample_count", None),
            "sample_buffer_load_stats": sample_buffer_load_stats,
            "sample_buffer_loaded_count": _sample_buffer_loaded_count(sample_buffer_load_stats),
            "sample_buffer_filtered_count": _sample_buffer_filtered_count(sample_buffer_load_stats),
        }


class DenseCNNCheckpointSaver:
    def save(self, *, name: str, ctx: Any, components: Any) -> Path:
        path = ctx.checkpoint_dir / f"{name}.pt"
        trainer = getattr(components.model, "trainer", None)
        buffer = getattr(trainer, "buffer", None)
        payload = {
            "model": "hexo_models.dense_cnn",
            "model_state": components.model.model.state_dict(),
            "optimizer_state": (
                components.model.optimizer.state_dict()
                if components.model.optimizer is not None
                else None
            ),
            "sample_buffer": (
                buffer.state_dict()
                if buffer is not None and hasattr(buffer, "state_dict")
                else None
            ),
            "epoch": _epoch_from_name(name) or _latest_epoch(ctx, components),
            "metadata": _checkpoint_metadata(ctx=ctx, components=components, buffer=buffer),
        }
        torch.save(payload, path)
        return path


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
    if isinstance(checkpoint_state, dict) and checkpoint_state.get("epoch") is not None:
        return int(checkpoint_state["epoch"])
    return None


def _checkpoint_metadata(*, ctx: Any, components: Any, buffer: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    checkpoint_state = getattr(getattr(components, "shared", None), "checkpoint_state", None)
    if isinstance(checkpoint_state, Mapping):
        parent_metadata = checkpoint_state.get("metadata")
        if isinstance(parent_metadata, Mapping):
            metadata.update(dict(parent_metadata))
        checkpoint_ref = checkpoint_state.get("checkpoint_ref") or checkpoint_state.get("checkpoint_path")
        if checkpoint_ref is not None:
            metadata["parent_checkpoint"] = str(checkpoint_ref)

    metadata["run"] = ctx.config.run.name
    metadata["sample_count"] = getattr(buffer, "sample_count", None)
    return metadata


def _state_dict_incompatibilities(
    expected: Mapping[str, torch.Tensor],
    candidate: object,
    *,
    limit: int = 12,
) -> list[dict[str, object]]:
    if not isinstance(candidate, Mapping):
        return [{"key": "model_state", "expected": "mapping", "actual": type(candidate).__name__}]

    issues: list[dict[str, object]] = []
    expected_keys = set(expected)
    candidate_keys = set(str(key) for key in candidate)
    for key in sorted(expected_keys - candidate_keys):
        issues.append({"key": key, "expected": tuple(expected[key].shape), "actual": "missing"})
        if len(issues) >= limit:
            return issues
    for key in sorted(candidate_keys - expected_keys):
        value = candidate.get(key)
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        issues.append({"key": key, "expected": "missing", "actual": shape})
        if len(issues) >= limit:
            return issues
    for key in sorted(expected_keys & candidate_keys):
        value = candidate.get(key)
        if not isinstance(value, torch.Tensor):
            issues.append({"key": key, "expected": tuple(expected[key].shape), "actual": type(value).__name__})
        elif tuple(value.shape) != tuple(expected[key].shape):
            issues.append({"key": key, "expected": tuple(expected[key].shape), "actual": tuple(value.shape)})
        if len(issues) >= limit:
            return issues
    return issues


def _sample_buffer_load_stats(load_result: object, buffer: object) -> dict[str, object] | None:
    if isinstance(load_result, Mapping):
        return dict(load_result)
    stats = getattr(buffer, "last_load_stats", None)
    if isinstance(stats, Mapping):
        return dict(stats)
    return None


def _sample_buffer_loaded_count(stats: Mapping[str, object] | None) -> int | None:
    return _optional_int(stats.get("loaded")) if stats is not None else None


def _sample_buffer_filtered_count(stats: Mapping[str, object] | None) -> int | None:
    return _optional_int(stats.get("filtered")) if stats is not None else None


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_checkpoint_ref(path: Path) -> Path | None:
    resolved = path.expanduser()
    if resolved.suffix.lower() == ".txt" and not resolved.exists():
        return None
    if resolved.suffix.lower() == ".txt" and resolved.exists():
        target = resolved.read_text(encoding="utf-8-sig").strip()
        if not target:
            return None
        target_path = Path(target).expanduser()
        if not target_path.is_absolute():
            target_path = resolved.parent / target_path
        return target_path
    return resolved
