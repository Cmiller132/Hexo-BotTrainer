"""Micro-bench: timed train steps on synthetic-but-real buckets (M3 diagnosis).

Measures wall time per optimizer step (forward+loss+backward+step) on real
expanded rows at the production batch recipe, with the fp16 bias gather and
256-quantized shapes. Run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import torch

from hexfield.batching import (
    collate_training,
    pair_budget_microbuckets,
    split_stvalue_columns,
    step_global_denominators,
)
from hexfield.losses import hexfield_loss
from hexfield.model import HexfieldNet
from hexfield.prefit import make_optimizer, run_step
from hexfield.samples import STV_HORIZONS, expand_sample
from hexfield.shards import read_compact_shard

device = torch.device("cuda")
model = HexfieldNet().to(device)
optimizer = make_optimizer(model)
scaler = torch.amp.GradScaler()

shard = sorted((REPO / "data" / "hexfield_bootstrap" / "train").glob("*.npz"))[0]
samples = read_compact_shard(shard)
print(f"shard rows: {len(samples)}")

steps = []
grad_stats = []
for start in range(0, 32 * 12, 32):
    chunk = samples[start : start + 32]
    expanded = [expand_sample(s, symmetry=start % 12) for s in chunk]
    denoms = step_global_denominators(expanded, STV_HORIZONS)
    buckets = []
    for bucket in pair_budget_microbuckets(expanded):
        pad_to = -(-max(r.support.num_nodes for r in bucket) // 256) * 256
        buckets.append(split_stvalue_columns(collate_training(bucket, pad_to=pad_to), STV_HORIZONS))
    sizes = [tuple(b["feats"].shape[:2]) for b in buckets]
    torch.cuda.synchronize()
    t0 = time.time()
    run_step(model, buckets, denoms, device, scaler, optimizer, grad_stats)
    torch.cuda.synchronize()
    dt = time.time() - t0
    steps.append(dt)
    print(f"step {len(steps)}: {dt*1000:.0f} ms  buckets {sizes}  "
          f"mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)

warm = steps[2:]
print(f"steady-state: {sum(warm)/len(warm)*1000:.0f} ms/step; "
      f"epoch est: {sum(warm)/len(warm)*431495/32/60:.1f} min")
