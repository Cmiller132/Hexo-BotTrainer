"""torch.profiler over a few prefit steps — find where 2.7s/step goes (M3)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import time

import torch
from torch.profiler import ProfilerActivity, profile

from hexfield.batching import (
    collate_training,
    pair_budget_microbuckets,
    split_stvalue_columns,
    step_global_denominators,
)
from hexfield.model import HexfieldNet
from hexfield.prefit import make_optimizer, run_step
from hexfield.samples import STV_HORIZONS, expand_sample
from hexfield.shards import read_compact_shard

device = torch.device("cuda")

# Sanity: raw GEMM throughput under WSL2.
a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    a = (a @ a).clamp(-1, 1)
torch.cuda.synchronize()
gemm_tflops = 20 * 2 * 4096**3 / (time.time() - t0) / 1e12
print(f"raw fp16 GEMM: {gemm_tflops:.1f} TFLOPS")

model = HexfieldNet().to(device)
optimizer = make_optimizer(model)
scaler = torch.amp.GradScaler()

shard = sorted((REPO / "data" / "hexfield_bootstrap" / "train").glob("*.npz"))[0]
samples = read_compact_shard(shard)
chunk = samples[0:32]
expanded = [expand_sample(s) for s in chunk]
denoms = step_global_denominators(expanded, STV_HORIZONS)
buckets = []
for bucket in pair_budget_microbuckets(expanded):
    pad_to = -(-max(r.support.num_nodes for r in bucket) // 256) * 256
    buckets.append(split_stvalue_columns(collate_training(bucket, pad_to=pad_to), STV_HORIZONS))
print("bucket shapes:", [tuple(b["feats"].shape[:2]) for b in buckets])

grad_stats = []
for _ in range(3):  # warmup
    run_step(model, buckets, denoms, device, scaler, optimizer, grad_stats)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        run_step(model, buckets, denoms, device, scaler, optimizer, grad_stats)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=14))
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
