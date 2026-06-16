"""Where does the large-support serve forward spend its GPU time — attention or not?

Profiles forward_policy_value at a realistic large-support shape (B, Npad) and
(1) gives a torch.profiler CUDA-kernel breakdown, (2) does a clean ABLATION:
times the full forward vs forwards with the 3 SDPA calls bypassed and with the
(B,heads,S,S) bias build bypassed, so attention's share is measured, not guessed.

Run (GPU free):  python scripts/_hexfield_forward_profile.py [Npad] [B]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import torch
from torch.nn import functional as F

from hexfield.constants import NUM_FEATURES
from hexfield import model as M
from hexfield.model import HexfieldNet

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"
CKPT = f"{RUN}/checkpoints/epoch_000034.pt"
DEV = torch.device("cuda")
NPAD = int(sys.argv[1]) if len(sys.argv) > 1 else 1920
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def make_batch(b, npad, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(b, npad, NUM_FEATURES, generator=g).to(DEV)
    coords = torch.randint(-22, 23, (b, npad, 2), generator=g).to(DEV).long()
    nbr = torch.randint(0, npad, (b, npad, 6), generator=g).to(DEV).long()
    mask = torch.ones(b, npad, dtype=torch.bool, device=DEV)
    return feats, nbr, mask, coords


def load():
    m = HexfieldNet()
    p = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(p.get("model", p), strict=True)
    return m.to(DEV).eval()


def timed(fn, batch, reps=20, warmup=5):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmup):
            fn(*batch, request_moves_left=True)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            fn(*batch, request_moves_left=True)
        torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000.0


m = load()
batch = make_batch(B, NPAD)
print(f"shape B={B} Npad={NPAD} (S={NPAD+8})", flush=True)

# --- ablation: bypass SDPA, and bypass the whole attention (bias build + blocks) ---
full_ms = timed(m.forward_policy_value, batch)

orig_sdpa = F.scaled_dot_product_attention
def no_sdpa(q, k, v, attn_mask=None, **kw):
    return v  # skip the O(S^2) attention matmul+softmax, keep proj/MLP/bias-build
F.scaled_dot_product_attention = no_sdpa
M.F.scaled_dot_product_attention = no_sdpa
nosdpa_ms = timed(m.forward_policy_value, batch)
F.scaled_dot_product_attention = orig_sdpa
M.F.scaled_dot_product_attention = orig_sdpa

orig_attn = HexfieldNet.build_attn_bias
def cheap_bias(self, coords, mask):
    b, n, _ = coords.shape
    s = 8 + n
    return torch.zeros(b, self.attn_blocks[0].attn.heads, s, s, device=coords.device, dtype=torch.float16)
HexfieldNet.build_attn_bias = cheap_bias
nobias_ms = timed(m.forward_policy_value, batch)
HexfieldNet.build_attn_bias = orig_attn

# bypass BOTH sdpa and bias build => "attention-free" forward
F.scaled_dot_product_attention = no_sdpa
M.F.scaled_dot_product_attention = no_sdpa
HexfieldNet.build_attn_bias = cheap_bias
noattn_ms = timed(m.forward_policy_value, batch)
F.scaled_dot_product_attention = orig_sdpa
M.F.scaled_dot_product_attention = orig_sdpa
HexfieldNet.build_attn_bias = orig_attn

print("\n=== ABLATION (eager fp16, ms/forward) ===", flush=True)
print(f"  full forward          : {full_ms:7.2f} ms", flush=True)
print(f"  - SDPA only bypassed  : {nosdpa_ms:7.2f} ms   -> SDPA share        {100*(full_ms-nosdpa_ms)/full_ms:5.1f}%", flush=True)
print(f"  - bias-build bypassed : {nobias_ms:7.2f} ms   -> bias-build share  {100*(full_ms-nobias_ms)/full_ms:5.1f}%", flush=True)
print(f"  - all attention gone  : {noattn_ms:7.2f} ms   -> ATTENTION total   {100*(full_ms-noattn_ms)/full_ms:5.1f}%", flush=True)
print(f"  (residual = conv trunk + projections/MLP + heads: {100*noattn_ms/full_ms:5.1f}%)", flush=True)

# --- profiler kernel breakdown (compiled, the real serve path) ---
print("\n=== torch.profiler CUDA kernels (top 14, compiled forward) ===", flush=True)
torch._dynamo.config.suppress_errors = True
fpv = torch.compile(m.forward_policy_value, dynamic=True)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for _ in range(6):
        fpv(*batch, request_moves_left=True)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fpv(*batch, request_moves_left=True)
        torch.cuda.synchronize()
ka = prof.key_averages()
rows = sorted(ka, key=lambda e: e.self_device_time_total, reverse=True)
tot = sum(e.self_device_time_total for e in rows) or 1
sdpa = 0.0
for e in rows[:14]:
    nm = e.key[:48]
    pct = 100 * e.self_device_time_total / tot
    print(f"  {pct:5.1f}%  {nm}", flush=True)
for e in rows:
    if any(k in e.key.lower() for k in ("attention", "fmha", "flash", "sdpa")):
        sdpa += e.self_device_time_total
print(f"  [SDPA-kernel share of CUDA time: {100*sdpa/tot:.1f}%]", flush=True)
