"""Parity + timing for the build_attn_bias rewrite (LUT cell-index gather).

Compares the OLD cell-index machinery (abs/maximum/clamp/where/where over
(B,N,N)) against the NEW single precomputed-LUT gather, asserting BIT-IDENTICAL
cell indices and bit-identical full (B,heads,S,S) bias, then times each path.

Run (GPU free):
    PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
        scripts/_hexfield_bias_check.py
"""
from __future__ import annotations

import sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import torch
import hexfield.constants as C
from hexfield.geometry import rel_bias_index
from hexfield.model import HexfieldNet, PAD_KEY_MASK_VALUE

dev = torch.device("cuda")
NT = C.NUM_TOKENS


def old_cell_idx(coords, exact_lut):
    """The original lines 271-296 cell-index computation (reference)."""
    b, n, _ = coords.shape
    dq = coords[:, None, :, 0] - coords[:, :, None, 0]
    dr = coords[:, None, :, 1] - coords[:, :, None, 1]
    d = torch.maximum(torch.maximum(dq.abs(), dr.abs()), (dq + dr).abs())
    clamped_q = dq.clamp(-C.BIAS_DISK_RADIUS, C.BIAS_DISK_RADIUS) + C.BIAS_DISK_RADIUS
    clamped_r = dr.clamp(-C.BIAS_DISK_RADIUS, C.BIAS_DISK_RADIUS) + C.BIAS_DISK_RADIUS
    exact = exact_lut[(clamped_q * 17 + clamped_r).reshape(-1)].reshape(b, n, n)
    on_axis = (dq == 0) | (dr == 0) | (dq + dr == 0)
    ring_base = torch.where(on_axis, torch.full_like(d, C.BIAS_ON_AXIS_BASE),
                            torch.full_like(d, C.BIAS_OFF_AXIS_BASE))
    ring = ring_base + (d - C.BIAS_RING_MIN)
    cell_idx = torch.where(d <= C.BIAS_DISK_RADIUS, exact,
                           torch.where(d <= C.BIAS_RING_MAX, ring,
                                       torch.full_like(d, C.BIAS_FAR_ROW)))
    return cell_idx


def build_cell_lut():
    M = C.BIAS_RING_MAX + 1  # 17 — see proof in PR notes
    W = 2 * M + 1
    lut = torch.empty(W * W, dtype=torch.long)
    for dq in range(-M, M + 1):
        for dr in range(-M, M + 1):
            lut[(dq + M) * W + (dr + M)] = rel_bias_index(dq, dr)
    return lut, M, W


def new_cell_idx(coords, cell_lut, M, W):
    b, n, _ = coords.shape
    dq = coords[:, None, :, 0] - coords[:, :, None, 0]
    dr = coords[:, None, :, 1] - coords[:, :, None, 1]
    qi = dq.clamp(-M, M) + M
    ri = dr.clamp(-M, M) + M
    return cell_lut[(qi * W + ri).reshape(-1)].reshape(b, n, n)


# 1) exhaustive integer parity over a wide offset range (CPU)
cell_lut_cpu, M, W = build_cell_lut()
bad = 0
for dq in range(-40, 41):
    for dr in range(-40, 41):
        ref = rel_bias_index(dq, dr)
        qi = min(max(dq, -M), M) + M
        ri = min(max(dr, -M), M) + M
        got = int(cell_lut_cpu[qi * W + ri])
        if ref != got:
            bad += 1
            if bad < 10:
                print(f"  MISMATCH dq={dq} dr={dr} ref={ref} got={got}")
print(f"exhaustive integer parity over [-40,40]^2: {'OK' if bad == 0 else f'{bad} MISMATCHES'}")

# 2) tensor parity on random + structured coords, several shapes
m = HexfieldNet().eval().to(dev)
exact_lut = m._exact_lut
cell_lut = cell_lut_cpu.to(dev)
torch.manual_seed(0)
for (b, n) in [(3, 3328), (8, 1280), (64, 600), (1, 64)]:
    coords = torch.randint(-25, 26, (b, n, 2), dtype=torch.long, device=dev)
    a = old_cell_idx(coords, exact_lut)
    z = new_cell_idx(coords, cell_lut, M, W)
    same = torch.equal(a, z)
    print(f"  cell_idx parity B={b} N={n}: {'IDENTICAL' if same else 'DIFFER n=' + str((a!=z).sum().item())}")

# 3) timing the two cell-index paths at a late-game shape
def bench(fn, *a, reps=20):
    for _ in range(3): fn(*a)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): fn(*a)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000

coords = torch.randint(-25, 26, (3, 3328, 2), dtype=torch.long, device=dev)
told = bench(lambda: old_cell_idx(coords, exact_lut))
tnew = bench(lambda: new_cell_idx(coords, cell_lut, M, W))
print(f"cell_idx B=3 N=3328: OLD {told:.2f} ms | NEW {tnew:.2f} ms | {told/tnew:.2f}x")


def ref_full_bias_headlast(model, coords, mask):
    """Reference: old inference-path build_attn_bias (head-LAST fp16 gather +
    fill add over key dim 2 + permute+contiguous). The current model.py builds
    the SAME tensor head-FIRST; this asserts they are bit-identical."""
    b, n, _ = coords.shape
    cell_idx = old_cell_idx(coords, model._exact_lut)
    s = NT + n
    pair = coords.new_full((b, s, s), C.BIAS_TOKEN_TOKEN_ROW)
    pair[:, :NT, NT:] = C.BIAS_TOKEN_CELL_ROW
    pair[:, NT:, :NT] = C.BIAS_CELL_TOKEN_ROW
    pair[:, NT:, NT:] = cell_idx
    bias = model.bias_table.to(torch.float16)[pair]
    key_pad = torch.cat([mask.new_ones(b, NT), mask], dim=1)
    fill = torch.where(key_pad, 0.0, PAD_KEY_MASK_VALUE).to(bias.dtype)
    bias = bias + fill[:, None, :, None]
    return bias.permute(0, 3, 1, 2).contiguous()


print("\n=== full build_attn_bias parity (model head-first vs head-last ref) ===")
torch.nn.init.trunc_normal_(m.bias_table, std=1.0)  # non-zero table so diffs surface
with torch.no_grad():
    for (b, n) in [(3, 1024), (8, 512), (16, 300), (2, 2048)]:
        coords = torch.randint(-25, 26, (b, n, 2), dtype=torch.long, device=dev)
        mask = torch.rand(b, n, device=dev) > 0.15  # ~15% padding
        got = m.build_attn_bias(coords, mask)
        ref = ref_full_bias_headlast(m, coords, mask)
        ident = torch.equal(got, ref)
        print(f"  B={b} N={n}: shape={tuple(got.shape)} stride(-1)={got.stride(-1)} "
              f"{'BIT-IDENTICAL' if ident else 'DIFFER maxabs=' + str((got-ref).abs().max().item())}")

