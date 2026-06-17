"""Feasibility probe: can flex_attention express the hexfield rel-pos bias on torch 2.12?

Builds the bias as a score_mod (captured coords + LUT + learned table gather) plus a
mask_mod (skip padded keys), runs flex_attention, and checks PARITY against the current
materialized-bias SDPA path on the SAME q/k/v. If parity is fp16-close, the production
integration is viable. Also times both at a large shape.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from hexfield.constants import (NUM_TOKENS, ATTENTION_HEADS, HEAD_DIM, BIAS_RING_MAX,
    BIAS_TOKEN_TOKEN_ROW, BIAS_TOKEN_CELL_ROW, BIAS_CELL_TOKEN_ROW)
from hexfield.model import HexfieldNet

DEV = torch.device("cuda")
M = HexfieldNet().to(DEV).eval()
TABLE = M.bias_table.detach().to(torch.float16)          # (ROWS, heads)
LUT = M._cell_bias_lut.to(DEV)                            # (cw*cw,) long
CM = M._cell_bias_M; CW = 2 * CM + 1
flex_c = torch.compile(flex_attention, dynamic=False)


def run(b_sz, npad, seed=0, do_time=False):
    g = torch.Generator().manual_seed(seed)
    coords = torch.randint(-22, 23, (b_sz, npad, 2), generator=g).to(DEV).long()
    n = torch.randint(npad // 2, npad + 1, (b_sz,), generator=g).to(DEV)   # real cells per row
    s = NUM_TOKENS + npad
    mask = (torch.arange(npad, device=DEV)[None, :] < n[:, None])          # (b, npad) real cells
    q = torch.randn(b_sz, ATTENTION_HEADS, s, HEAD_DIM, generator=g, dtype=torch.float16).to(DEV)
    k = torch.randn(b_sz, ATTENTION_HEADS, s, HEAD_DIM, generator=g, dtype=torch.float16).to(DEV)
    v = torch.randn(b_sz, ATTENTION_HEADS, s, HEAD_DIM, generator=g, dtype=torch.float16).to(DEV)

    # ---- reference: current materialized-bias SDPA ----
    ref_bias = M.build_attn_bias(coords, mask).to(torch.float16)           # (b,heads,s,s)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=ref_bias)

    # ---- flex: bias via score_mod, padded keys via mask_mod ----
    NT = NUM_TOKENS
    def score_mod(score, b, h, q_idx, kv_idx):
        qc = torch.clamp(q_idx - NT, min=0); kc = torch.clamp(kv_idx - NT, min=0)
        dq = coords[b, kc, 0] - coords[b, qc, 0]
        dr = coords[b, kc, 1] - coords[b, qc, 1]
        qi = torch.clamp(dq, -CM, CM) + CM; ri = torch.clamp(dr, -CM, CM) + CM
        cell_idx = LUT[qi * CW + ri]
        q_tok = q_idx < NT; k_tok = kv_idx < NT
        row = torch.where(q_tok & k_tok, torch.full_like(cell_idx, BIAS_TOKEN_TOKEN_ROW),
              torch.where(q_tok & ~k_tok, torch.full_like(cell_idx, BIAS_TOKEN_CELL_ROW),
              torch.where(~q_tok & k_tok, torch.full_like(cell_idx, BIAS_CELL_TOKEN_ROW), cell_idx)))
        return score + TABLE[row, h].to(score.dtype)

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx < NT) | mask[b, torch.clamp(kv_idx - NT, min=0)]

    bm = create_block_mask(mask_mod, b_sz, None, s, s, device=str(DEV))
    out = flex_c(q, k, v, score_mod=score_mod, block_mask=bm)

    # ---- PRODUCTION variant: pad masking folded into score_mod (block_mask=None) ----
    # This is the form that ships (§7 item 1). Captures n[b] (real cells per row)
    # and adds an additive fill to pad-key columns, matching build_attn_bias's
    # PAD_KEY_MASK_VALUE add. No create_block_mask => no per-shape BlockMask object.
    from hexfield.model import PAD_KEY_MASK_VALUE
    PAD_FILL = float(PAD_KEY_MASK_VALUE)   # -3e4 (the shipped finite fp16 mask)
    n_long = mask.sum(dim=1).to(torch.long)   # (b,) real cell count per row

    def score_mod_pad(score, b, h, q_idx, kv_idx):
        qc = torch.clamp(q_idx - NT, min=0); kc = torch.clamp(kv_idx - NT, min=0)
        dq = coords[b, kc, 0] - coords[b, qc, 0]
        dr = coords[b, kc, 1] - coords[b, qc, 1]
        qi = torch.clamp(dq, -CM, CM) + CM; ri = torch.clamp(dr, -CM, CM) + CM
        cell_idx = LUT[qi * CW + ri]
        q_tok = q_idx < NT; k_tok = kv_idx < NT
        row = torch.where(q_tok & k_tok, torch.full_like(cell_idx, BIAS_TOKEN_TOKEN_ROW),
              torch.where(q_tok & ~k_tok, torch.full_like(cell_idx, BIAS_TOKEN_CELL_ROW),
              torch.where(~q_tok & k_tok, torch.full_like(cell_idx, BIAS_CELL_TOKEN_ROW), cell_idx)))
        biased = score + TABLE[row, h].to(score.dtype)
        # pad-KEY columns (cell key beyond the row's real count) get the additive fill
        is_pad_key = (kv_idx >= NT) & (kc >= n_long[b])
        return torch.where(is_pad_key, biased + PAD_FILL, biased)

    out_pad = flex_c(q, k, v, score_mod=score_mod_pad, block_mask=None)

    d = (ref.float() - out.float()).abs().max().item()
    dpad = (ref.float() - out_pad.float()).abs().max().item()
    print(f"  B={b_sz} Npad={npad}: max|dattn_out|(blockmask)={d:.3e}  "
          f"max|dattn_out|(score_mod_pad)={dpad:.3e}", flush=True)
    if do_time:
        for fn, lbl in [(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=M.build_attn_bias(coords, mask).to(torch.float16)), "cur bias+sdpa"),
                        (lambda: flex_c(q, k, v, score_mod=score_mod, block_mask=bm), "flex")]:
            for _ in range(5): fn()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(20): fn()
            torch.cuda.synchronize()
            print(f"    {lbl:14s}: {(time.time()-t0)/20*1000:.2f} ms", flush=True)


print("torch", torch.__version__, "flex probe", flush=True)
with torch.no_grad():
    run(4, 512, seed=1)
    run(4, 512, seed=2)
    run(8, 1920, seed=3, do_time=True)
print("PROBE DONE", flush=True)
