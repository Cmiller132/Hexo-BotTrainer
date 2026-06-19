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


def warm_start_into(model: HexfieldNet, state: dict) -> dict:
    """Tolerant weights-only warm start (initialize_from / BC-prefit ONLY).

    Loads every checkpoint key that matches the model's state dict and EXACTLY
    matches its shape; leaves any model param that is MISSING from the checkpoint
    at its freshly-constructed value (``_init_weights``). This is the warm-start
    path for main_4: the soft_policy_conv/_head params (and any future new head)
    do not exist in the v3 BC prefit, so a strict load would raise. The new head
    therefore trains FRESH from its trunc_normal(0.02)/zeros init — exactly the
    owner-mandated zero/fresh-init choice, documented here.

    Precedent: the per-block ``bias_tables.*`` migration in model.py likewise
    tolerates an older single-table checkpoint. This relaxation is applied ONLY
    on the warm-start (initialize_from) path; the strict ``resume`` path
    (``load_into``) is intentionally left untouched so a true training resume
    still enforces bidirectional key equality.

    Returns a summary dict {loaded, missing, unexpected, shape_mismatch} for the
    caller to log. Shape-mismatched and unexpected (checkpoint-only) keys are
    DROPPED, not loaded — a deliberate, reported decision rather than a silent
    strict-load failure.
    """

    model_sd = model.state_dict()
    to_load: dict[str, torch.Tensor] = {}
    shape_mismatch: list[str] = []
    for key, val in state.items():
        if key not in model_sd:
            continue  # unexpected / checkpoint-only key -> dropped
        if model_sd[key].shape != val.shape:
            shape_mismatch.append(key)
            continue
        to_load[key] = val
    missing = sorted(set(model_sd) - set(to_load))
    unexpected = sorted(set(state) - set(model_sd))
    # strict=False: missing keys keep their _init_weights value; the matched
    # subset is copied in. assign=False so dtype/device follow the live params.
    model.load_state_dict(to_load, strict=False)
    return {
        "loaded": len(to_load),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }


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
            if not resume:
                # initialize_from warm start (meta-shape checkpoint): tolerant
                # load so a v3-lineage checkpoint without the new soft_policy head
                # warm-starts cleanly (the head trains fresh). Optimizer state is
                # intentionally NOT restored on warm start.
                summary = warm_start_into(model, payload["model"])
                return {
                    "status": "initialized_from",
                    "path": str(path),
                    "warm_start": summary,
                }
            # resume == True here (the not-resume warm start returned above).
            meta = load_into(model, payload, optimizer=optimizer)
            if config_lr is not None:
                for group, lr in zip(optimizer.param_groups, config_lr):
                    group["lr"] = lr
            # Restore the KataGo-style train-bucket governor ONLY on a true
            # resume (PLAN §6/M1). A missing key -> from_dict(None) -> fresh
            # state, so old-format checkpoints resume cleanly. The warm-start
            # branch above never reaches here, so a BC-prefit warm start always
            # begins with a fresh governor, not a stale bucket from another run.
            trainer = getattr(components.model, "trainer", None)
            if trainer is not None:
                trainer.train_state = HexfieldTrainState.from_dict(meta.get("train_state"))
            return {"status": "loaded", "epoch": int(meta.get("epoch", 0)), "path": str(path)}
        # prefit shape (raw BC-prefit dict): tolerant warm start so a v3 prefit
        # without the new soft_policy head warm-starts cleanly (head trains fresh).
        summary = warm_start_into(model, payload["model"])
        return {
            "status": "initialized_from",
            "path": str(path),
            "source": "bc_prefit",
            "warm_start": summary,
        }


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
