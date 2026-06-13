"""Strict checkpoint IO for the hexo_train pipeline (spec: no silent partial
loads — bidirectional key equality, mismatch raises)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from .model import HexfieldNet


def save_checkpoint(path: Path, *, model: HexfieldNet, optimizer, epoch: int, extra: dict | None = None) -> Path:
    payload = {
        "meta": {"lineage": "hexfield", "epoch": int(epoch), **(extra or {})},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_into(model: HexfieldNet, payload: dict, *, optimizer=None) -> dict:
    state = payload["model"]
    expected = set(model.state_dict().keys())
    got = set(state.keys())
    if expected != got:
        missing = sorted(expected - got)[:5]
        unexpected = sorted(got - expected)[:5]
        raise ValueError(
            f"hexfield checkpoint key mismatch: missing={missing} unexpected={unexpected}"
        )
    model.load_state_dict(state, strict=True)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("meta", {})


class HexfieldCheckpointLoader:
    """hexo_train contract: load(ref, ctx, components) -> state dict.

    resume_from -> {"status": "loaded", "epoch": N} (epoch fast-forward);
    initialize_from -> weights-only warm start (e.g. the BC prefit);
    None -> fresh random init.
    """

    def load(self, checkpoint_ref, *, ctx, components) -> dict[str, Any]:
        model = components.model.model
        optimizer = components.model.optimizer
        if checkpoint_ref is None:
            return {"status": "initialized", "note": "fresh init"}
        path = Path(checkpoint_ref)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        # BC prefit checkpoints store the raw prefit dict; pipeline epochs
        # store the {meta, model, optimizer} shape.
        if "meta" in payload:
            resume = ctx.config.checkpoint.resume_from is not None
            meta = load_into(model, payload, optimizer=optimizer if resume else None)
            if resume:
                return {"status": "loaded", "epoch": int(meta.get("epoch", 0)), "path": str(path)}
            return {"status": "initialized_from", "path": str(path)}
        # prefit shape
        model.load_state_dict(payload["model"], strict=True)
        return {"status": "initialized_from", "path": str(path), "source": "bc_prefit"}


class HexfieldCheckpointSaver:
    """hexo_train contract: save(name, ctx, components) -> path."""

    def save(self, *, name: str, ctx, components) -> Path:
        epoch = 0
        match = re.search(r"epoch_(\d+)", name)
        if match:
            epoch = int(match.group(1))
        return save_checkpoint(
            ctx.checkpoint_dir / f"{name}.pt",
            model=components.model.model,
            optimizer=components.model.optimizer,
            epoch=epoch,
            extra={"run": ctx.config.run.name},
        )
