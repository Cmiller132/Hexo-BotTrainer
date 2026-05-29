"""Checkpoint IO for dense CNN training state.

The generic training pipeline owns when checkpoints are loaded and saved. This
module owns what dense_cnn needs to persist inside those checkpoints: model
weights, optimizer state, and KataGo-style train-bucket state.

Loading is strict about model-weight compatibility and the current replay
schema. Legacy checkpoints that still contain the removed in-memory replay
buffer are rejected so dense CNN has only one supported training path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


class DenseCNNCheckpointLoader:
    """Load dense_cnn checkpoint payloads into generic pipeline components."""

    def load(self, checkpoint_ref: object | None, *, ctx: Any, components: Any) -> dict[str, Any]:
        """Load model, optimizer, and dense_cnn train state if usable."""

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
        trainer = getattr(components.model, "trainer", None)
        if payload.get("sample_buffer") is not None:
            return {
                "status": "initialized",
                "checkpoint_ref": str(path),
                "reason": "legacy dense_cnn sample_buffer checkpoints are unsupported",
                "checkpoint_epoch": payload.get("epoch"),
                "metadata": payload.get("metadata", {}),
            }

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
            }

        components.model.model.load_state_dict(model_state)
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None and components.model.optimizer is not None:
            components.model.optimizer.load_state_dict(optimizer_state)
        train_state = payload.get("train_state")
        if trainer is not None and hasattr(trainer, "load_train_state") and isinstance(train_state, Mapping):
            trainer.load_train_state(train_state)
        return {
            "status": "loaded",
            "checkpoint_ref": str(path),
            "epoch": payload.get("epoch"),
            "metadata": payload.get("metadata", {}),
            "train_state": train_state if isinstance(train_state, Mapping) else None,
        }


class DenseCNNCheckpointSaver:
    """Save dense_cnn checkpoint payloads from generic pipeline components."""

    def save(self, *, name: str, ctx: Any, components: Any) -> Path:
        """Persist model, optimizer, and train-bucket state for one checkpoint."""

        path = ctx.checkpoint_dir / f"{name}.pt"
        trainer = getattr(components.model, "trainer", None)
        train_state = getattr(trainer, "train_state", None)
        payload = {
            "model": "hexo_models.dense_cnn",
            "model_state": components.model.model.state_dict(),
            "optimizer_state": (
                components.model.optimizer.state_dict()
                if components.model.optimizer is not None
                else None
            ),
            "train_state": train_state.to_dict() if hasattr(train_state, "to_dict") else None,
            "epoch": _epoch_from_name(name) or _latest_epoch(ctx, components),
            "metadata": {
                "run": ctx.config.run.name,
                "sample_count": getattr(trainer, "sample_count", None),
            },
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
    """Return a bounded list of tensor/key mismatches before loading weights."""

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
