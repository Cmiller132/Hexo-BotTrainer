"""Tiny GPU sanity check for the restnet model: confirm CUDA + SDPA + materialized
attention forward/backward work, and report peak VRAM at small batches. Isolates a
'device not ready' / OOM cause before committing to a full prefit run."""
import sys, os
from pathlib import Path
ROOT = Path("/mnt/e/Hexo-BotTrainer-hexgt")
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models"):
    sys.path.insert(0, str(ROOT / "packages" / p / "python"))
sys.path.insert(0, str(ROOT / "packages" / "dense_cnn_restnet" / "python"))

import torch
from dense_cnn_restnet.architecture import RestnetNetwork
from dense_cnn_restnet.constants import INPUT_CHANNELS, BOARD_SIZE

print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("no CUDA")
print("device:", torch.cuda.get_device_name(0), flush=True)
dev = torch.device("cuda:0")

# minimal raw SDPA first (isolate the kernel from the model).
try:
    q = torch.randn(2, 4, 1681, 24, device=dev)
    mask = torch.randn(1, 4, 1681, 1681, device=dev)
    o = torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=mask)
    torch.cuda.synchronize()
    print("[ok] raw SDPA 1681-token forward works; out", tuple(o.shape), flush=True)
except Exception as e:
    print("[FAIL] raw SDPA:", repr(e), flush=True)

def trial(impl, batch):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    net = RestnetNetwork(channels=96, blocks_type="R_R_R_T_R_R_T_R",
                         attention_impl=impl, short_term_value_horizons=(1, 4, 8)).to(dev).train()
    x = torch.randn(batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=dev)
    out = net(x)
    loss = sum(v.float().pow(2).mean() for v in out.values())
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del net, x, out, loss
    torch.cuda.empty_cache()
    return peak

for impl in ("sdpa",):
    for b in (8, 32, 64, 96):
        try:
            peak = trial(impl, b)
            print(f"[ok] {impl} fwd+bwd batch={b}: peak {peak:.2f} GB", flush=True)
        except Exception as e:
            print(f"[FAIL] {impl} batch={b}: {repr(e)[:160]}", flush=True)
            break
print("sanity done", flush=True)
