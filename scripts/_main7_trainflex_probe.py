"""Probe: does train-path flex_attention at c=192/d=64 compile on Ada with
explicit kernel_options block sizes? (Default configs want 147456B shared
memory vs the 101376B hardware limit -> InductorError -> eager fallback and
~9-10 s/step training. Smaller blocks should fit.)

Runs a grad-enabled flex call at training shapes for a few block configs and
reports compile success + rough step time. Contended GPU OK (relative only).
"""

import sys
import time

import torch
from torch.nn.attention.flex_attention import flex_attention

torch.manual_seed(0)
dev = "cuda"
B, H, S, D = 16, 3, 392, 64

q = torch.randn(B, H, S, D, device=dev, dtype=torch.float16, requires_grad=True)
k = torch.randn(B, H, S, D, device=dev, dtype=torch.float16, requires_grad=True)
v = torch.randn(B, H, S, D, device=dev, dtype=torch.float16, requires_grad=True)
table = torch.randn(238, H, device=dev, dtype=torch.float32, requires_grad=True)
pair = torch.randint(0, 238, (B, S, S), device=dev, dtype=torch.uint8)


def score_mod(score, b, h, q_idx, kv_idx):
    row = pair[b, q_idx, kv_idx].to(torch.int32)
    return score + table[row, h].to(score.dtype)


def trial(name, kopts):
    fn = torch.compile(flex_attention, dynamic=False)
    try:
        t0 = time.time()
        out = fn(q, k, v, score_mod=score_mod, kernel_options=kopts)
        out.sum().backward()
        torch.cuda.synchronize()
        t_compile = time.time() - t0
        for g in (q, k, v, table):
            g.grad = None
        t0 = time.time()
        for _ in range(5):
            out = fn(q, k, v, score_mod=score_mod, kernel_options=kopts)
            out.sum().backward()
        torch.cuda.synchronize()
        t_step = (time.time() - t0) / 5
        print(f"{name}: OK  compile {t_compile:.1f}s  fwd+bwd {t_step*1e3:.1f} ms")
        return True
    except Exception as e:
        print(f"{name}: FAIL  {type(e).__name__}: {str(e)[:160]}")
        return False


import warnings

warnings.filterwarnings("ignore")

ok_any = False
ok_any |= trial("default(None)", None)
ok_any |= trial(
    "fwd64/32+bwd32",
    {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_M1": 32, "BLOCK_N1": 32,
     "BLOCK_M2": 32, "BLOCK_N2": 32},
)
ok_any |= trial(
    "all32",
    {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 32, "BLOCK_N1": 32,
     "BLOCK_M2": 32, "BLOCK_N2": 32},
)
ok_any |= trial(
    "fwd64+bwd64/32",
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 64, "BLOCK_N1": 32,
     "BLOCK_M2": 32, "BLOCK_N2": 64},
)
sys.exit(0 if ok_any else 1)
