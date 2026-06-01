"""Forward decomposition + FP16 quantification for the evaluator.

Self-play move time is ~71% the evaluator forward (incl. Python marshal). This
bench separates the pure GPU forward from the full payload path and quantifies
FP16/autocast (already enabled via config.training.amp) at both the current
64x4 and the Phase-4 target 96x6, plus a numerics check (FP16 vs FP32 value &
policy-logit differences).

Usage: python analysis/forward_fp_bench.py
"""

from __future__ import annotations

import time

import torch

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.constants import BOARD_SIZE, INPUT_CHANNELS
from hexo_models.dense_cnn.inference import DenseCNNInference

SEED = 1234
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_inf(channels, blocks, amp):
    torch.manual_seed(SEED)
    model = Model1Network(channels=channels, blocks=blocks)
    return DenseCNNInference(model, device=DEVICE, amp=amp, return_logits=True,
                             max_batch_size=1024, optimize_for_inference=False)


def time_forward(inf, batch, iters=30):
    x = torch.randn(batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    for _ in range(5):  # warm (autotune + clocks)
        inf._forward_inputs_device(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = inf._forward_inputs_device(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3, out


def numerics(channels, blocks, batch=256):
    """Max abs diff of value & policy between FP32 and FP16 forwards."""
    inf32 = make_inf(channels, blocks, amp=False)
    inf16 = make_inf(channels, blocks, amp=True)  # same seed -> same weights
    x = torch.randn(batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.inference_mode():
        o32 = inf32._forward_inputs_device(x)
        o16 = inf16._forward_inputs_device(x)
    v = (o32["value"].float() - o16["value"].float()).abs().max().item()
    p = (o32["policy"].float() - o16["policy"].float()).abs().max().item()
    # policy is logits; also compare softmax to gauge decision impact
    ps32 = torch.softmax(o32["policy"].float(), dim=-1)
    ps16 = torch.softmax(o16["policy"].float(), dim=-1)
    psd = (ps32 - ps16).abs().max().item()
    return v, p, psd


def main() -> None:
    print(f"device={DEVICE}")
    print(f"{'model':>8} {'batch':>6} {'fp32 ms':>9} {'fp16 ms':>9} {'speedup':>8}")
    for (ch, bl, tag) in [(64, 4, "64x4"), (96, 6, "96x6")]:
        inf32 = make_inf(ch, bl, amp=False)
        inf16 = make_inf(ch, bl, amp=True)
        for batch in (64, 256, 1024):
            ms32, _ = time_forward(inf32, batch)
            ms16, _ = time_forward(inf16, batch)
            print(f"{tag:>8} {batch:>6} {ms32:>9.3f} {ms16:>9.3f} {ms32/max(ms16,1e-9):>7.2f}x")
    print("\nFP16-vs-FP32 numerics (max abs diff, batch 256):")
    for (ch, bl, tag) in [(64, 4, "64x4"), (96, 6, "96x6")]:
        v, p, psd = numerics(ch, bl)
        print(f"  {tag}: max|d value_logit|={v:.4e}  max|d policy_logit|={p:.4e}  max|d policy_softmax|={psd:.4e}")


if __name__ == "__main__":
    main()
