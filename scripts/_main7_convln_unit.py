"""Unit: hex_conv_ln (+fp8 variant) vs fp32 reference at c=192 (BN=256 tile
path), plus a full main_7-stack forward_policy_value smoke at c=192.

Env (set by the .sh runner): HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=3
HEXFIELD_TRUNK=CCACCACCACCACCA + all serve flags.
"""

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python")
from hexfield._triton_conv import hex_conv_ln, hex_conv_ln_fp8  # noqa: E402

torch.manual_seed(0)
dev = "cuda"

B, NPAD, C = 4, 192, 192
N_VALID = [180, 96, 192, 33]

x = torch.randn(B, NPAD, C, device=dev, dtype=torch.float16) * 0.5
gidx = torch.randint(0, NPAD, (B, NPAD, 7), device=dev, dtype=torch.long)
mask = torch.zeros(B, NPAD, device=dev, dtype=torch.bool)
for i, nv in enumerate(N_VALID):
    mask[i, :nv] = True
    gidx[i, :, 1:] = torch.where(
        torch.rand(NPAD, 6, device=dev) < 0.15, NPAD, gidx[i, :, 1:]
    )  # some missing-neighbour sentinels
w = torch.randn(7, C, C, device=dev, dtype=torch.float32) * (1.0 / (7 * C) ** 0.5)
b = torch.randn(C, device=dev, dtype=torch.float32) * 0.02
lnw = 1.0 + torch.randn(C, device=dev, dtype=torch.float32) * 0.1
lnb = torch.randn(C, device=dev, dtype=torch.float32) * 0.05
EPS = 1e-5


def reference(relu: bool):
    x32 = x.float()
    x_ext = torch.cat([x32, x32.new_zeros(B, 1, C)], dim=1)
    flat = gidx.reshape(B, NPAD * 7, 1).expand(-1, -1, C)
    g = x_ext.gather(1, flat).reshape(B, NPAD, 7 * C)
    out = g @ w.reshape(7 * C, C) + b
    out = F.layer_norm(out, (C,), lnw, lnb, EPS)
    if relu:
        out = F.relu(out)
    return out * mask.unsqueeze(-1).float()


ok = True
with torch.no_grad():
    for relu in (True, False):
        got = hex_conv_ln(x, gidx, mask, w, b, lnw, lnb, EPS, relu)
        ref = reference(relu)
        d = (got.float() - ref).abs().max().item()
        status = "PASS" if d < 3e-2 else "FAIL"
        ok &= d < 3e-2
        print(f"hex_conv_ln  c={C} relu={relu}: max|diff|={d:.2e} {status}")
        got8 = hex_conv_ln_fp8(x, gidx, mask, w, b, lnw, lnb, EPS, relu)
        d8 = (got8.float() - ref).abs().max().item()
        fin = torch.isfinite(got8).all().item()
        status8 = "PASS" if (d8 < 0.5 and fin) else "FAIL"
        ok &= d8 < 0.5 and fin
        print(f"hex_conv_ln_fp8 relu={relu}: max|diff|={d8:.2e} finite={fin} {status8}")

# Full main_7 serve-stack forward at c=192 (routing smoke; random net).
from hexfield.model import HexfieldNet  # noqa: E402
from hexfield import constants as K  # noqa: E402

assert K.CHANNELS == 192 and K.ATTENTION_HEADS == 3, "run via the .sh (env!)"
net = HexfieldNet().to(dev).eval().half()
N = 384
feats = torch.randn(2, N, K.NUM_FEATURES, device=dev, dtype=torch.float16)
nbr = torch.full((2, N, 6), N, device=dev, dtype=torch.long)
m2 = torch.zeros(2, N, device=dev, dtype=torch.bool)
m2[0, :N], m2[1, : N // 2] = True, True
coords = torch.zeros(2, N, 2, device=dev, dtype=torch.long)
coords[:, :, 0] = torch.arange(N, device=dev) % 21 - 10
coords[:, :, 1] = torch.arange(N, device=dev) // 21 - 9
with torch.no_grad():
    out = net.forward_policy_value(feats, nbr, m2, coords, request_moves_left=True)
fin = all(torch.isfinite(v).all().item() for v in out.values())
print(f"main_7 stack forward: keys={sorted(out.keys())} finite={fin}")
ok &= fin
print("CONVLN UNIT " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
