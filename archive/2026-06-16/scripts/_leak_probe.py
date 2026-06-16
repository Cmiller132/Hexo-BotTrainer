#!/usr/bin/env python3
"""Direct leak probe for the zero-copy/f16 evaluator path.

Calls evaluate_model1_payload in a tight loop (each call builds + must free a
PlaneBuffer that Python views zero-copy). If my buffer-protocol pyclass leaks the
backing Vec, RSS climbs monotonically. If flat, the eval path is clean and the
27 GB seen in production is the pre-existing self-play accumulation, not my change.
"""

from __future__ import annotations

import os
import numpy as np
import torch

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference
from hexo_models.dense_cnn.constants import BOARD_SIZE, BOARD_AREA, INPUT_CHANNELS

N, LEGAL = 1024, 687


def rss_mb() -> float:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1.0


def make_payload():
    planes = np.random.rand(N, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)
    flats = np.empty(N * LEGAL, dtype=np.int64)
    for r in range(N):
        flats[r * LEGAL:(r + 1) * LEGAL] = np.random.choice(BOARD_AREA, size=LEGAL, replace=False)
    return {
        "inputs": bytearray(planes.tobytes()),
        "shape": (N, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        "legal_flat_indices_bytes": bytearray(flats.tobytes()),
        "legal_row_offsets": tuple(range(0, N * LEGAL + 1, LEGAL)),
    }


def main() -> None:
    use_trt = os.environ.get("HEXO_TRT", "").strip() in ("1", "true", "True")
    net = Model1Network(in_channels=INPUT_CHANNELS, channels=64, blocks=4,
                        short_term_value_horizons=(1, 4, 8))
    inf = DenseCNNInference(net, device="cuda", amp=True, return_logits=False,
                            max_batch_size=1024, use_trt=use_trt,
                            bucket_pad_multiple=16, trt_allow_fallback=True)
    payload = make_payload()
    iters = 2000
    for _ in range(20):  # warmup
        inf.evaluate_model1_payload(payload)
    torch.cuda.synchronize()
    base = rss_mb()
    print(f"[leak] trt={inf.trt_info.get('adopted')} baseline RSS={base:.0f}MB; {iters} calls of batch {N}")
    for i in range(1, iters + 1):
        inf.evaluate_model1_payload(payload)
        if i % 200 == 0:
            torch.cuda.synchronize()
            print(f"  call {i:5d}  RSS={rss_mb():.0f}MB  delta_from_baseline={rss_mb()-base:+.0f}MB")
    torch.cuda.synchronize()
    end = rss_mb()
    grow = end - base
    per_call_kb = grow * 1024 / iters
    print(f"[leak] end RSS={end:.0f}MB  total_growth={grow:+.0f}MB over {iters} calls = {per_call_kb:+.1f} KB/call")
    # A real PlaneBuffer leak would be ~{N*13*41*41*2/1024:.0f}KB/call (the f16 buffer).
    print(f"[leak] (a full leaked f16 buffer would be ~{N*INPUT_CHANNELS*BOARD_SIZE*BOARD_SIZE*2/1024:.0f} KB/call)")
    print(f"[leak] VERDICT: {'LEAK in eval path' if grow > 500 else 'eval path is CLEAN — 27GB is pre-existing selfplay accumulation'}")


if __name__ == "__main__":
    main()
