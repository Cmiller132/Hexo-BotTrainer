"""Conclusive repro for #2 on CUDA: decode_binned_value can round a confident value a
few ULP outside [-1,1]. The real inference path runs the value head + softmax + sum on
the GPU (and via TensorRT FP16), whose parallel reductions round differently than CPU's
sequential sum (which caps at exactly +-1). read_value then hard-rejects the out-of-range
value and aborts the whole batched search."""
import torch

from hexo_models.dense_cnn.constants import VALUE_BINS
from hexo_models.dense_cnn.losses import decode_binned_value

print("cuda available:", torch.cuda.is_available())
torch.manual_seed(0)
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for device in devices:
    found = 0
    worst = 1.0
    for dtype in (torch.float32, torch.float16):
        for peak in (3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 16.0):
            n = 4_000_000
            logits = (torch.randn(n, VALUE_BINS, device=device) * 1.5).to(dtype)
            top = torch.randint(0, 2, (n,), device=device) * (VALUE_BINS - 1)
            logits[torch.arange(n, device=device), top] += torch.tensor(peak, device=device, dtype=dtype)
            v = decode_binned_value(logits)
            over = int(((v > 1.0) | (v < -1.0)).sum())
            found += over
            worst = max(worst, float(v.abs().max()))
            if over:
                ex = v[(v > 1.0) | (v < -1.0)][:3].tolist()
                print(f"  {device}/{dtype} peak={peak}: OVERFLOWS={over}/{n} examples={ex}")
    print(f"{device}: total overflows={found}  worst |value|={worst:.10f}  "
          f"clamped -> {min(1.0, worst):.6f}")
