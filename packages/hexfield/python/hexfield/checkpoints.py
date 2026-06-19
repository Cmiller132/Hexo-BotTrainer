"""Strict checkpoint IO for the hexo_train pipeline: no silent partial loads —
bidirectional key equality, mismatch raises."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from .model import HexfieldNet
from .train_state import HexfieldTrainState


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
        # Invariant: optimizer state is loaded on CPU (map_location) but must move
        # to the model's device, or AdamW.step() mixes devices and crashes.
        dev = next(model.parameters()).device
        for st in optimizer.state.values():
            for key, val in st.items():
                if isinstance(val, torch.Tensor):
                    st[key] = val.to(dev)
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
            # The optimizer was just built (plugin.py) with the LIVE config lr, but
            # load_into's optimizer.load_state_dict restores the checkpoint's SAVED
            # lr into the param groups, silently overriding an lr config change on
            # resume. Capture the config lr BEFORE the load and re-apply it after,
            # so an lr edit takes effect on the next relaunch (Adam moment buffers
            # are kept; only the step size changes). Captured from the optimizer
            # itself — ctx.config here is hexo_train's TrainingConfig (no .training).
            config_lr = (
                [g["lr"] for g in optimizer.param_groups]
                if (resume and optimizer is not None)
                else None
            )
            meta = load_into(model, payload, optimizer=optimizer if resume else None)
            if resume:
                if config_lr is not None:
                    for group, lr in zip(optimizer.param_groups, config_lr):
                        group["lr"] = lr
                # Restore the KataGo-style train-bucket governor ONLY on a true
                # resume. A missing key -> from_dict(None) -> fresh state, so
                # old-format checkpoints resume cleanly. Never restore on the
                # initialize_from warm-start branch below: a BC-prefit warm start
                # must begin with a fresh governor, not inherit a stale bucket.
                trainer = getattr(components.model, "trainer", None)
                if trainer is not None:
                    trainer.train_state = HexfieldTrainState.from_dict(meta.get("train_state"))
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
        # Persist the KataGo-style train-bucket governor state inside the
        # checkpoint meta. Guard with getattr so a trainer without a train_state
        # (e.g. tests) does not crash the save.
        trainer = getattr(components.model, "trainer", None)
        extra = {
            "run": ctx.config.run.name,
            **(
                {"train_state": trainer.train_state.to_dict()}
                if getattr(trainer, "train_state", None) is not None
                else {}
            ),
        }
        return save_checkpoint(
            ctx.checkpoint_dir / f"{name}.pt",
            model=components.model.model,
            optimizer=components.model.optimizer,
            epoch=epoch,
            extra=extra,
        )
