"""A7/PB measurement: cold cuDNN autotune cost with vs without batch bucketing.

cuDNN (benchmark=True) re-autotunes (~hundreds of ms) the first time it sees a
new batch shape. The evaluator was measured to see ~900 distinct shapes per
epoch -> ~830 s of autotune on every cold/relaunch epoch. Bucketing collapses
those to <=11 power-of-two shapes.

Run as a SUBPROCESS per mode so the per-process cuDNN algo cache starts cold:

    python analysis/a7_autotune_bench.py off
    python analysis/a7_autotune_bench.py on
"""

from __future__ import annotations

import sys
from time import perf_counter

import torch

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.constants import BOARD_SIZE, INPUT_CHANNELS
from hexo_models.dense_cnn.inference import DenseCNNInference, _bucket_batch_size

# Distinct leaf-batch sizes a search produces as games fill/drain. Real epochs
# see ~900; a few dozen distinct shapes is enough to expose the autotune tax.
SWEEP = list(range(3, 256, 8))  # 3, 11, 19, ... 251  -> 32 distinct shapes


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "off"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    net = Model1Network(in_channels=INPUT_CHANNELS, channels=64, blocks=4).eval()
    infer = DenseCNNInference(net, device=device, amp=True, optimize_for_inference=False)
    infer.pad_to_buckets = mode == "on"

    cap = infer.max_batch_size
    unique_inputs = sorted(set(SWEEP))
    unique_buckets = sorted({_bucket_batch_size(n, cap) for n in SWEEP})

    torch.cuda.synchronize(device)
    t0 = perf_counter()
    for n in SWEEP:
        x = torch.randn(n, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        infer._forward_inputs_device(x)
    torch.cuda.synchronize(device)
    total = perf_counter() - t0

    print(
        f"mode={mode:3s}  forwards={len(SWEEP)}  "
        f"unique_input_shapes={len(unique_inputs)}  "
        f"unique_bucket_shapes={len(unique_buckets)}  "
        f"first_pass_total={total:7.3f} s"
    )


if __name__ == "__main__":
    main()
