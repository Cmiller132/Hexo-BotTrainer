#!/usr/bin/env python3
"""Fold a trained ray-tap ``both`` checkpoint into dense31 Design A.

For every equipped trunk conv, each distance shell receives the original
direction block with the learned alpha applied as an input-column scale.  The
operation is exact in real arithmetic and removes alpha from the target state
dict.  Optimizer state is cleared because the parameter shapes changed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def fold_raytap_conv(w_base: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Expand ``(7,12,Co,Ci)`` + ``(5,Ci)`` to ``(31,12,Co,Ci)``."""

    if w_base.ndim != 4 or w_base.shape[0] != 7:
        raise ValueError(f"expected 7-tap w_base, got {tuple(w_base.shape)}")
    if alpha.ndim != 2 or alpha.shape[0] != 5:
        raise ValueError(f"expected alpha shape (5,Ci), got {tuple(alpha.shape)}")
    if w_base.shape[-1] != alpha.shape[-1]:
        raise ValueError(
            f"w_base orbit-in {w_base.shape[-1]} != alpha width {alpha.shape[-1]}"
        )
    out = w_base.new_empty((31, *w_base.shape[1:]))
    out[0] = w_base[0]
    for k in range(5):
        for d in range(6):
            # W @ diag(alpha): column-scale orbit_in, for every free slot.
            out[1 + 6 * k + d] = w_base[1 + d] * alpha[k].view(1, 1, -1)
    return out


def convert_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a dense31 state dict from a ray-tap ``both`` state dict."""

    out = dict(state)
    alpha_keys = sorted(
        k
        for k in state
        if k.startswith("conv_blocks.")
        and (k.endswith(".conv1.alpha") or k.endswith(".conv2.alpha"))
    )
    if not alpha_keys:
        raise ValueError("state dict has no equipped ray-tap alpha parameters")

    conv_prefixes = sorted({k[: -len(".alpha")] for k in alpha_keys})
    first = {p for p in conv_prefixes if p.endswith(".conv1")}
    second = {p for p in conv_prefixes if p.endswith(".conv2")}
    if not first or len(first) != len(second):
        raise ValueError(
            "dense31 surgery requires raytap='both': every trunk block must "
            "carry conv1.alpha and conv2.alpha"
        )
    first_blocks = {p.rsplit(".", 1)[0] for p in first}
    second_blocks = {p.rsplit(".", 1)[0] for p in second}
    if first_blocks != second_blocks:
        raise ValueError("conv1/conv2 equipped block sets differ")

    for prefix in conv_prefixes:
        wk = prefix + ".w_base"
        ak = prefix + ".alpha"
        if wk not in state:
            raise ValueError(f"missing {wk}")
        out[wk] = fold_raytap_conv(state[wk], state[ak])
        del out[ak]
    return out


def convert_checkpoint(payload: dict) -> dict:
    """Convert either a normal checkpoint payload or a raw state dict."""

    if "model" not in payload:
        return convert_state_dict(payload)
    out = dict(payload)
    out["model"] = convert_state_dict(payload["model"])
    meta = dict(payload.get("meta") or {})
    source_mode = meta.get("raytap")
    if source_mode is not None and str(source_mode) != "both":
        raise ValueError(f"checkpoint meta raytap={source_mode!r}, expected 'both'")
    meta["raytap"] = "dense31"
    meta["dense31_surgery"] = "all-distance-alpha-fold-v1"
    out["meta"] = meta
    if "optimizer" in out:
        out["optimizer"] = None
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="ray-tap checkpoint or state dict")
    parser.add_argument("output", type=Path, help="dense31 output checkpoint")
    parser.add_argument("--force", action="store_true", help="replace output if it exists")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"output exists (pass --force): {args.output}")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    converted = convert_checkpoint(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    torch.save(converted, tmp)
    os.replace(tmp, args.output)
    model_state = converted.get("model", converted)
    n31 = sum(
        1
        for k, v in model_state.items()
        if k.endswith(".w_base") and getattr(v, "shape", (0,))[0] == 31
    )
    print(f"wrote {args.output} ({n31} dense31 trunk convs; optimizer cleared)")


if __name__ == "__main__":
    main()
