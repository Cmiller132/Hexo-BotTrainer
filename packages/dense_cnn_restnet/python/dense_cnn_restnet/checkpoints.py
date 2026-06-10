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
            if not _is_initialize_only(checkpoint_ref, ctx):
                # A RESUME pointing at an incompatible checkpoint must never fall
                # through to a fresh initialization — that silently restarts the
                # run from random weights on top of its own data. Typical cause:
                # the architecture changed without running the checkpoint
                # migration (scripts/_restnet_migrate_heads_v2.py).
                raise RuntimeError(
                    "dense_cnn_restnet resume checkpoint is incompatible with the current "
                    f"architecture ({path}); run the checkpoint migration before relaunching. "
                    f"First mismatches: {incompatibilities[:4]}"
                )
            return {
                "status": "initialized",
                "checkpoint_ref": str(path),
                "reason": "checkpoint model_state is incompatible with current dense_cnn architecture",
                "incompatible_tensors": incompatibilities,
                "checkpoint_epoch": payload.get("epoch"),
                "metadata": payload.get("metadata", {}),
            }

        components.model.model.load_state_dict(model_state)
        # `initialize_from` is a WEIGHTS-ONLY warm start: the new run gets a fresh
        # optimizer (no stale moments/step counts from a different training phase,
        # e.g. the BC prefit's Adam state) and fresh train-bucket state. Only a
        # true `resume_from` restores optimizer and train state. The pipeline
        # collapses both fields into one ref, so the intent is recovered here from
        # the run config: resume_from set -> resume semantics, else initialize.
        weights_only = _is_initialize_only(checkpoint_ref, ctx)
        train_state = payload.get("train_state")
        if not weights_only:
            optimizer_state = payload.get("optimizer_state")
            if optimizer_state is not None and components.model.optimizer is not None:
                components.model.optimizer.load_state_dict(optimizer_state)
            if trainer is not None and hasattr(trainer, "load_train_state") and isinstance(train_state, Mapping):
                trainer.load_train_state(train_state)
        return {
            "status": "loaded",
            "checkpoint_ref": str(path),
            "weights_only": weights_only,
            "epoch": payload.get("epoch"),
            "metadata": payload.get("metadata", {}),
            "train_state": train_state if (not weights_only and isinstance(train_state, Mapping)) else None,
        }


class DenseCNNCheckpointSaver:
    """Save dense_cnn checkpoint payloads from generic pipeline components."""

    def save(self, *, name: str, ctx: Any, components: Any) -> Path:
        """Persist model, optimizer, and train-bucket state for one checkpoint."""

        path = ctx.checkpoint_dir / f"{name}.pt"
        trainer = getattr(components.model, "trainer", None)
        train_state = getattr(trainer, "train_state", None)
        payload = {
            "model": "dense_cnn_restnet",
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


def _is_initialize_only(checkpoint_ref: object, ctx: Any) -> bool:
    """Whether `checkpoint_ref` reached us via `initialize_from` (weights only).

    `hexo_train.checkpoints.load_or_initialize_checkpoint` resolves
    `resume_from or initialize_from` before calling the loader, so when
    `resume_from` is set the ref IS a resume (full restore). Only when
    `resume_from` is unset and the ref matches `initialize_from` is this a
    warm start. Defaults to full-restore on any config-shape surprise.
    """

    checkpoint = getattr(getattr(ctx, "config", None), "checkpoint", None)
    if checkpoint is None:
        return False
    if getattr(checkpoint, "resume_from", None) is not None:
        return False
    initialize_from = getattr(checkpoint, "initialize_from", None)
    return initialize_from is not None and str(checkpoint_ref) == str(initialize_from)


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
