"""Pinpoint the pure-FP16 overflow that makes the TRT engine emit NaN.

Runs the folded inference model in PURE fp16 (model.half(), no autocast — exactly
what the strongly-typed TRT fp16 engine does) on REAL game positions, with a
forward hook on every leaf module recording max|output| and any inf/NaN. The
layer whose activations approach the fp16 max (65504) / first goes inf is the
overflow site to keep in fp32 for the mixed-precision engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sys.path.insert(0, str(REPO / "packages" / p / "python"))

import torch
from torch import nn

CONFIG = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"


def build():
    import tomllib
    from hexo_models.dense_cnn.architecture import optimized_model1_for_inference
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin
    section = tomllib.load(open(CONFIG, "rb"))["model"]["config"]
    m = DenseCNNPlugin().build_model(game_spec={}, config=section)
    m.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state"], strict=True)
    return optimized_model1_for_inference(m).eval()


def real_inputs(n):
    import random
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id
    from hexo_models.dense_cnn import rust_bridge
    rng = random.Random(99)
    states = []
    gi = 0
    while len(states) < n:
        st = engine.new_game(seed=600_000 + gi); gi += 1
        for _ in range(rng.randint(6, 130)):
            if engine.terminal(st) is not None:
                break
            a = engine.legal_action_ids(st)
            if not a:
                break
            engine.apply_action(st, engine.PlacementAction(unpack_coord_id(rng.choice(a))))
        if engine.terminal(st) is None:
            states.append(engine.clone_state(st))
    payload = rust_bridge.model1_batch_inputs(states[:n])
    shape = tuple(int(x) for x in payload["shape"])
    arr = np.frombuffer(bytes(payload["inputs"]), dtype=np.float32).reshape(shape).copy()
    return torch.from_numpy(arr)


def main():
    model = build().to("cuda").half()  # PURE fp16
    x = real_inputs(256).to("cuda").half()

    stats = []
    hooks = []
    def mk(name, mod):
        def hook(m, inp, out):
            t = out if isinstance(out, torch.Tensor) else None
            if t is None:
                return
            f = t.float()
            stats.append((name, type(m).__name__, float(f.abs().max().item()),
                          bool(torch.isnan(f).any()), bool(torch.isinf(f).any())))
        return hook
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:  # leaf
            hooks.append(mod.register_forward_hook(mk(name, mod)))

    with torch.inference_mode():
        out = model.forward_policy_value(x)
    pol = out["policy"].float(); val = out["value"].float()
    print(f"FP16_MAX=65504. policy: max|.|={pol.abs().max():.1f} nan={torch.isnan(pol).any().item()} "
          f"inf={torch.isinf(pol).any().item()}", flush=True)
    print(f"value(logits): max|.|={val.abs().max():.1f} nan={torch.isnan(val).any().item()} "
          f"inf={torch.isinf(val).any().item()}", flush=True)
    print("\n[per-layer max|activation| in pure fp16, sorted desc] (>5000 = overflow-prone):", flush=True)
    stats.sort(key=lambda s: s[2], reverse=True)
    for name, typ, mx, isnan, isinf in stats:
        flag = " <-- NaN" if isnan else (" <-- INF" if isinf else (" <-- HOT" if mx > 5000 else ""))
        if mx > 1000 or isnan or isinf:
            print(f"  {mx:12.1f}  {typ:14s} {name}{flag}", flush=True)
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
