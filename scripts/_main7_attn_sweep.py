"""Tile sweep for attn_pair at c=192/h=3/d=64 serve shapes (kernel only)."""

import os
import sys

import torch

sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python")
from hexfield._triton_attn import attn_pair  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
H, D = 3, 64
tag = (
    f"BM={os.environ.get('HEXFIELD_ATTN_BM', '64')} "
    f"BN={os.environ.get('HEXFIELD_ATTN_BN', '64')} "
    f"warps={os.environ.get('HEXFIELD_ATTN_WARPS', '4')} "
    f"stages={os.environ.get('HEXFIELD_ATTN_STAGES', '3')}"
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


for B, S in [(247, 392), (140, 520), (90, 648)]:
    q = torch.randn(B, H, S, D, device=dev, dtype=torch.float16) * 0.5
    k, v = q.clone(), q.clone()
    pair = torch.randint(0, 237, (B, S, S), device=dev, dtype=torch.uint8)
    tab = torch.randn(238, H, device=dev, dtype=torch.float16) * 0.2
    seq = torch.full((B,), S - 24, device=dev, dtype=torch.int32)
    with torch.no_grad():
        t = timeit(lambda: attn_pair(q, k, v, pair, tab, seq))
    print(f"{tag}  B={B} S={S}: {t:6.3f} ms")
