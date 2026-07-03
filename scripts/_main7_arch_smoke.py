"""Smoke-test the layout-driven trunk (HEXFIELD_TRUNK / HEXFIELD_ATTENTION_HEADS).

Run twice (constants are read at import):
  mode=main6  HEXFIELD_CHANNELS=128 (defaults otherwise)
      -> net must have 8 conv blocks / 3 attn blocks and load the live
         epoch checkpoint with strict=True (backward compat proof).
  mode=main7  HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=3
      HEXFIELD_TRUNK=CCACCACCACCACCA
      -> constructs, CPU forward runs, shapes sane; prints param count.
"""

import os
import sys

import torch

mode = sys.argv[1]
ckpt = sys.argv[2] if len(sys.argv) > 2 else None

from hexfield.model import HexfieldNet  # noqa: E402
from hexfield import constants as K  # noqa: E402

net = HexfieldNet()
n_c, n_a = len(net.conv_blocks), len(net.attn_blocks)
n_params = sum(p.numel() for p in net.parameters())
print(
    f"mode={mode} layout={K.TRUNK_LAYOUT} c={K.CHANNELS} "
    f"heads={K.ATTENTION_HEADS} d={K.HEAD_DIM} "
    f"conv_blocks={n_c} attn_blocks={n_a} bias_tables={len(net.bias_tables)} "
    f"params={n_params/1e6:.2f}M"
)

if mode == "main6":
    assert (n_c, n_a) == (8, 3), (n_c, n_a)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        missing, unexpected = net.load_state_dict(sd, strict=True), None
        print("checkpoint strict load: OK")
elif mode == "main7":
    assert (n_c, n_a) == (10, 5), (n_c, n_a)
    assert K.HEAD_DIM == 64, K.HEAD_DIM
    # Tiny CPU forward through the serve entry (materialized bias path).
    B, N = 2, 61
    feats = torch.randn(B, N, K.NUM_FEATURES)
    nbr = torch.full((B, N, 6), N, dtype=torch.long)  # all-missing -> pad row
    mask = torch.ones(B, N, dtype=torch.bool)
    coords = torch.zeros(B, N, 2, dtype=torch.long)
    # unique-ish coords inside the disk so rel_bias indexing is exercised
    q = torch.arange(N) % 9 - 4
    r = torch.arange(N) // 9 - 3
    coords[:, :, 0] = q
    coords[:, :, 1] = r
    with torch.no_grad():
        out = net.forward_policy_value(feats, nbr, mask, coords)
    shapes = {k: tuple(v.shape) for k, v in out.items()}
    print(f"forward ok: {shapes}")
else:
    raise SystemExit(f"unknown mode {mode}")
print("SMOKE PASS")
