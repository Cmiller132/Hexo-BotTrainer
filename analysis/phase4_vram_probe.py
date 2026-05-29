"""Phase 4 gate: 96x6 + P7 model instantiates and a train step fits in 12 GB VRAM.

Builds the target Model 1 (channels=96, blocks=6, fully-conv P7 heads), runs a
real forward + AMP backward + optimizer step (mirroring DenseCNNTrainer) at
bs256 and bs128, and reports peak VRAM. Confirms the bs256->128 auto-fallback
headroom the calibration relies on. Does NOT train — one step per batch size.

Usage: python analysis/phase4_vram_probe.py
"""

from __future__ import annotations

import torch

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.constants import INPUT_CHANNELS, BOARD_SIZE, BOARD_AREA
from hexo_models.dense_cnn.losses import model1_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZONS = (1, 4, 8)


def synth_batch(n: int) -> dict:
    inp = torch.randn(n, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    pol = torch.rand(n, BOARD_AREA); pol /= pol.sum(-1, keepdim=True)
    opp = torch.rand(n, BOARD_AREA); opp /= opp.sum(-1, keepdim=True)
    batch = {
        "policy": pol, "opp_policy": opp,
        "value": torch.empty(n).uniform_(-1, 1),
    }
    for h in HORIZONS:
        batch[f"stvalue_{h}"] = torch.empty(n).uniform_(-1, 1)
        batch[f"stvalue_{h}_mask"] = torch.ones(n)
    return inp, batch


def run_step(model, optimizer, scaler, n) -> tuple[bool, float, str]:
    try:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        inp, batch = synth_batch(n)
        inp = inp.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True):
            out = model(inp)
            loss, _ = model1_loss(out, batch, policy_weight=1.0, value_weight=1.0,
                                  opp_policy_weight=0.25, short_term_value_weight=0.25)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(DEVICE)
        peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        return True, peak, f"loss={float(loss.detach().cpu()):.4f}"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False, 0.0, "OOM"
        raise


def main() -> None:
    if DEVICE.type != "cuda":
        raise SystemExit("CUDA required")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Model1Network(channels=96, blocks=6, short_term_value_horizons=HORIZONS)
    model.to(DEVICE, memory_format=torch.channels_last).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    params = sum(p.numel() for p in model.parameters())
    total_vram = torch.cuda.get_device_properties(DEVICE).total_memory / 1024**2
    print(f"device={torch.cuda.get_device_name(DEVICE)}  total VRAM={total_vram:.0f} MiB  model params={params:,}")
    for n in (256, 128):
        # two steps: first allocates, second is steady-state peak
        run_step(model, optimizer, scaler, n)
        ok, peak, info = run_step(model, optimizer, scaler, n)
        status = "OK" if ok else "OOM"
        print(f"  bs={n:>4}: {status:>3}  peak={peak:>8.0f} MiB ({100*peak/total_vram:>4.0f}% of VRAM)  {info}")


if __name__ == "__main__":
    main()
