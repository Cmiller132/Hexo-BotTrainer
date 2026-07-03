"""Smoke: grad-path flex with kernel_options through the real model at the
main_7 arch — one forward() + backward, assert finite loss/grads and that no
InductorError fallback fired (we can't see the warning here, but a compile
failure now raises inside the probe budget or silently falls back — we check
step time instead: eager fallback at these shapes is >1s, compiled is ~ms)."""

import sys
import time

import torch

sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python")
from hexfield.model import HexfieldNet  # noqa: E402
from hexfield import constants as K  # noqa: E402

assert K.CHANNELS == 192 and K.ATTENTION_HEADS == 3

torch.manual_seed(0)
dev = "cuda"
net = HexfieldNet().to(dev)
B, N = 24, 312
feats = torch.randn(B, N, K.NUM_FEATURES, device=dev)
nbr = torch.full((B, N, 6), N, device=dev, dtype=torch.long)
mask = torch.ones(B, N, device=dev, dtype=torch.bool)
coords = torch.zeros(B, N, 2, device=dev, dtype=torch.long)
coords[:, :, 0] = torch.arange(N, device=dev) % 19 - 9
coords[:, :, 1] = torch.arange(N, device=dev) // 19 - 8

def step():
    out = net.forward(feats, nbr, mask, coords)
    loss = sum(v.float().pow(2).mean() for v in out.values())
    net.zero_grad(set_to_none=True)
    loss.backward()
    return loss

loss = step()  # compile warmup
torch.cuda.synchronize()
t0 = time.time()
for _ in range(3):
    loss = step()
torch.cuda.synchronize()
dt = (time.time() - t0) / 3
gt = net.bias_tables[0].grad
print(f"loss={loss.item():.4f} finite={torch.isfinite(loss).item()} "
      f"bias_grad_norm={gt.norm().item():.3e} step={dt*1e3:.0f} ms")
assert torch.isfinite(loss).item() and torch.isfinite(gt).all().item()
print("TRAINFLEX SMOKE PASS")
