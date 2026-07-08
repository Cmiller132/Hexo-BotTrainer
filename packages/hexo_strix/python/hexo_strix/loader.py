"""Load a hexo-strix checkpoint into the ported HeXONet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .model import HeXONet, build_from_model_config


@dataclass(frozen=True)
class StrixCheckpoint:
    model: HeXONet
    model_config: dict
    game_config: dict
    train_steps: int | None
    path: Path


def load_strix_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> StrixCheckpoint:
    """Load ``checkpoint_*.pt`` into an eval-ready HeXONet.

    The embedded ``model_config``/``game_config`` are authoritative. Strips the
    ``_orig_mod.`` torch.compile prefix and loads strictly so any architecture
    mismatch surfaces immediately.
    """
    path = Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model_config = dict(ck["model_config"])
    game_config = dict(ck["game_config"])
    model = build_from_model_config(model_config)
    raw = ck["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in raw.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return StrixCheckpoint(
        model=model,
        model_config=model_config,
        game_config=game_config,
        train_steps=ck.get("train_steps"),
        path=path,
    )
