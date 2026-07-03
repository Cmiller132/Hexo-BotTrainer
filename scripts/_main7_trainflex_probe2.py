"""Probe 2: the LIVE train-flex score_mod (coords + LUT + region selects, the
_FlexBias style) at big microbucket shapes, with and without the small-block
kernel_options. This is the config that hit 'No valid triton configs' live."""

import sys
import time
import warnings

import torch
from torch.nn.attention.flex_attention import flex_attention

warnings.filterwarnings("ignore")
torch.manual_seed(0)
dev = "cuda"
H, D = 3, 64
NT = 8
M = 9
W = 2 * M + 1

KOPTS = {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_M1": 32, "BLOCK_N1": 32,
         "BLOCK_M2": 32, "BLOCK_N2": 32}


def make(b, s):
    q = torch.randn(b, H, s, D, device=dev, dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, H, s, D, device=dev, dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, H, s, D, device=dev, dtype=torch.float16, requires_grad=True)
    n = s - NT
    coords = torch.randint(-12, 13, (b, n, 2), device=dev, dtype=torch.long)
    mask = torch.rand(b, n, device=dev) > 0.1
    table = torch.randn(237, H, device=dev, dtype=torch.float32, requires_grad=True)
    lut = torch.randint(0, 233, (W * W,), device=dev, dtype=torch.long)
    return q, k, v, coords, mask, table, lut


def heavy_score_mod(coords, mask, table, lut):
    def score_mod(score, b, h, q_idx, kv_idx):
        qc = torch.clamp(q_idx - NT, min=0)
        kc = torch.clamp(kv_idx - NT, min=0)
        dq = coords[b, kc, 0] - coords[b, qc, 0]
        dr = coords[b, kc, 1] - coords[b, qc, 1]
        qi = torch.clamp(dq, -M, M) + M
        ri = torch.clamp(dr, -M, M) + M
        cell_idx = lut[qi * W + ri]
        q_tok = q_idx < NT
        k_tok = kv_idx < NT
        row = torch.where(
            q_tok & k_tok, torch.full_like(cell_idx, 236),
            torch.where(q_tok & ~k_tok, torch.full_like(cell_idx, 234),
                        torch.where(~q_tok & k_tok, torch.full_like(cell_idx, 235),
                                    cell_idx)))
        biased = score + table[row, h].to(score.dtype)
        is_pad = (kv_idx >= NT) & ~mask[b, kc]
        return torch.where(is_pad, biased + (-3.0e4), biased)

    return score_mod


def trial(name, b, s, kopts):
    q, k, v, coords, mask, table, lut = make(b, s)
    sm = heavy_score_mod(coords, mask, table, lut)
    fn = torch.compile(flex_attention, dynamic=False)
    try:
        t0 = time.time()
        out = fn(q, k, v, score_mod=sm, kernel_options=kopts)
        out.sum().backward()
        torch.cuda.synchronize()
        tc = time.time() - t0
        t0 = time.time()
        for _ in range(5):
            out = fn(q, k, v, score_mod=sm, kernel_options=kopts)
            out.sum().backward()
        torch.cuda.synchronize()
        ts = (time.time() - t0) / 5
        print(f"{name} B={b} S={s}: OK compile {tc:.1f}s fwd+bwd {ts*1e3:.1f} ms")
        return True
    except Exception as e:
        print(f"{name} B={b} S={s}: FAIL {type(e).__name__}: {str(e)[:140]}")
        return False


r = True
r &= trial("heavy+default", 32, 392, None)
r &= trial("heavy+kopts  ", 32, 392, KOPTS)
r &= trial("heavy+kopts  ", 12, 648, KOPTS)
r &= trial("heavy+kopts  ", 48, 264, KOPTS)
sys.exit(0 if r else 1)
