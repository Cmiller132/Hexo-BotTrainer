"""Tile sweep for hex_conv_ln at c=192 serve shapes: kernel-only timing vs the
plain-conv + eager-LN pair it replaces. Env HEXFIELD_CONVLN_* set per child
process by the .sh driver (constants read at import)."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python")
from hexfield._triton_conv import hex_conv, hex_conv_ln  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
C = 192
EPS = 1e-5
tag = (
    f"BM={os.environ.get('HEXFIELD_CONVLN_BM', '32')} "
    f"warps={os.environ.get('HEXFIELD_CONVLN_WARPS', '8')} "
    f"stages={os.environ.get('HEXFIELD_CONVLN_STAGES', '2')}"
)


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


for B, NPAD in [(247, 384), (140, 512), (90, 640)]:
    x = torch.randn(B, NPAD, C, device=dev, dtype=torch.float16) * 0.5
    gidx = torch.randint(0, NPAD, (B, NPAD, 7), device=dev, dtype=torch.long)
    mask = torch.ones(B, NPAD, device=dev, dtype=torch.bool)
    w = torch.randn(7, C, C, device=dev, dtype=torch.float32) * 0.02
    b = torch.zeros(C, device=dev, dtype=torch.float32)
    lnw = torch.ones(C, device=dev, dtype=torch.float32)
    lnb = torch.zeros(C, device=dev, dtype=torch.float32)
    with torch.no_grad():
        t_fused = timeit(lambda: hex_conv_ln(x, gidx, mask, w, b, lnw, lnb, EPS, True))
        m = mask.unsqueeze(-1)

        def unfused():
            y = hex_conv(x, gidx, mask, w, b)
            return F.relu(F.layer_norm(y, (C,), lnw.half(), lnb.half(), EPS)) * m

        t_un = timeit(unfused)
    print(
        f"{tag}  B={B} Npad={NPAD}: fused {t_fused:6.3f} ms  "
        f"unfused {t_un:6.3f} ms  ratio {t_fused/t_un:5.2f}"
    )
