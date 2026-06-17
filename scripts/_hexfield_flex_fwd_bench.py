"""Forward-pass benchmark, FlexAttention-aware. How much does attention take NOW?

Measures, at a large serve shape (B, Npad):
  1) FLEX ON  : full eager forward + an ablation that bypasses the flex attention
     (M._flex_call -> return v) to isolate attention's share vs the rest
     (conv trunk + q/k/v/out projections + MLP + heads).
  2) FLEX OFF : the old materialized-bias path (build_attn_bias + SDPA) ablated the
     same way, to show how much flex shrank the forward and the attention slice.
  3) COMPILED : torch.compile(dynamic=True) flex forward + a torch.profiler CUDA
     kernel breakdown (the flex kernel's % of the live compiled forward).

Run (GPU free): python scripts/_hexfield_flex_fwd_bench.py [Npad] [B]
"""
from __future__ import annotations
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
import torch
from torch.nn import functional as F
from hexfield.constants import NUM_FEATURES
from hexfield import model as M
from hexfield.model import HexfieldNet

DEV = torch.device("cuda")
CKPT = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/checkpoints/epoch_000034.pt"
NPAD = int(sys.argv[1]) if len(sys.argv) > 1 else 1920
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def make_batch(b, npad, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(b, npad, NUM_FEATURES, generator=g).to(DEV).half()  # f16 (serve path)
    coords = torch.randint(-22, 23, (b, npad, 2), generator=g).to(DEV).long()
    nbr = torch.randint(0, npad, (b, npad, 6), generator=g).to(DEV).long()
    mask = torch.ones(b, npad, dtype=torch.bool, device=DEV)
    return feats, nbr, mask, coords


m = HexfieldNet()
p = torch.load(CKPT, map_location="cpu", weights_only=False)
m.load_state_dict(p.get("model", p), strict=True)
m = m.to(DEV).eval()
assert M._flex_attention is not None, "flex_attention unavailable"
batch = make_batch(B, NPAD)
print(f"shape B={B} Npad={NPAD} (S={NPAD+8})", flush=True)


def timed(fn, reps=30, warmup=8):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000.0


def fwd():
    return m.forward_policy_value(*batch, request_moves_left=True)


# === FLEX ON: full vs attention-bypassed ===
m._serve_flex = True
full_flex = timed(fwd)
orig_flex = M._flex_call
M._flex_call = lambda q, k, v, score_mod: v        # skip the in-kernel bias+attention
noattn_flex = timed(fwd)
M._flex_call = orig_flex
att_flex = full_flex - noattn_flex
print("\n=== FLEX ON (eager) ===", flush=True)
print(f"  full forward            {full_flex:7.2f} ms", flush=True)
print(f"  attention (flex kernel) {att_flex:7.2f} ms = {100*att_flex/full_flex:4.1f}%   "
      f"(bias + scores + softmax + AV, all in-kernel)", flush=True)
print(f"  rest (conv+proj+mlp+heads) {noattn_flex:6.2f} ms = {100*noattn_flex/full_flex:4.1f}%", flush=True)

# === FLEX OFF: materialized bias + SDPA, ablated the same way ===
m._serve_flex = False
full_mat = timed(fwd)
orig_sdpa = F.scaled_dot_product_attention
M.F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, **kw: v
orig_bias = HexfieldNet.build_attn_bias
heads = m.attn_blocks[0].attn.heads
HexfieldNet.build_attn_bias = lambda self, coords, mask: torch.zeros(
    coords.shape[0], heads, 8 + coords.shape[1], 8 + coords.shape[1],
    device=coords.device, dtype=torch.float16)
noattn_mat = timed(fwd)
M.F.scaled_dot_product_attention = orig_sdpa
HexfieldNet.build_attn_bias = orig_bias
att_mat = full_mat - noattn_mat
print("\n=== FLEX OFF (materialized bias + SDPA, eager) ===", flush=True)
print(f"  full forward            {full_mat:7.2f} ms", flush=True)
print(f"  attention (bias build + SDPA) {att_mat:6.2f} ms = {100*att_mat/full_mat:4.1f}%", flush=True)
print(f"\n=> flex shrank the forward {full_mat:.2f} -> {full_flex:.2f} ms ({full_mat/full_flex:.2f}x); "
      f"attention {att_mat:.2f} -> {att_flex:.2f} ms ({att_mat/max(att_flex,1e-6):.2f}x)", flush=True)

# === COMPILED flex forward: time + profiler kernel breakdown (the live view) ===
m._serve_flex = True
torch._dynamo.config.suppress_errors = True
fpv = torch.compile(m.forward_policy_value, dynamic=True)
comp_ms = timed(lambda: fpv(*batch, request_moves_left=True), reps=20, warmup=8)
print(f"\n=== COMPILED flex forward: {comp_ms:.2f} ms ===", flush=True)
from torch.profiler import profile, ProfilerActivity
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fpv(*batch, request_moves_left=True)
        torch.cuda.synchronize()
rows = sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
tot = sum(e.self_device_time_total for e in rows) or 1
flex_share = sum(e.self_device_time_total for e in rows
                 if any(k in e.key.lower() for k in ("flex", "attention", "fmha")))
print(f"  flex-attention kernel share of compiled CUDA time: {100*flex_share/tot:.1f}%", flush=True)
print("  top kernels:", flush=True)
for e in rows[:8]:
    print(f"    {100*e.self_device_time_total/tot:5.1f}%  {e.key[:54]}", flush=True)
