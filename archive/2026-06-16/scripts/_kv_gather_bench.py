from __future__ import annotations

import argparse
import copy
import sys
import time
import tomllib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("dense_cnn_restnet", "hexo_train", "hexo_runner"):
    src = ROOT / "packages" / pkg / "python"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from dense_cnn_restnet.architecture import RestnetNetwork, optimized_restnet_for_inference
from dense_cnn_restnet.config import parse_model1_config
from dense_cnn_restnet.constants import BOARD_SIZE, INPUT_CHANNELS


def _build_model(config_path: Path) -> RestnetNetwork:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    cfg = parse_model1_config(raw["model"]["config"])
    arch = cfg.architecture
    return RestnetNetwork(
        in_channels=arch.input_channels,
        channels=arch.channels,
        blocks_type=arch.blocks_type,
        attention_heads=arch.attention_heads,
        mlp_ratio=arch.mlp_ratio,
        embed_kernel_size=arch.embed_kernel_size,
        residual_conv=arch.residual_conv,
        attention_impl=arch.attention_impl,
        attention_scope=arch.attention_scope,
        dropout=arch.dropout,
        short_term_value_horizons=arch.short_term_value_horizons,
        moves_left_head=arch.moves_left_head,
    ).eval()


@torch.inference_mode()
def _bench(model: torch.nn.Module, x: torch.Tensor, iters: int) -> float:
    for _ in range(5):
        _ = model.forward_policy_value(x)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        _ = model.forward_policy_value(x)
    torch.cuda.synchronize()
    return iters * int(x.shape[0]) / max(time.perf_counter() - started, 1.0e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense_cnn_restnet_main1.toml")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--min-speedup", type=float, default=1.10)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the KV-gather benchmark")

    base = _build_model(ROOT / args.config)
    dense = optimized_restnet_for_inference(copy.deepcopy(base), attention_kv_gather=False).cuda().half().eval()
    gathered = optimized_restnet_for_inference(copy.deepcopy(base), attention_kv_gather=True).cuda().half().eval()
    x = torch.zeros((args.batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), device="cuda", dtype=torch.float16)
    x[:, 0, BOARD_SIZE // 2, BOARD_SIZE // 2] = 1
    x[:, 3, :, :] = 1
    x = x.to(memory_format=torch.channels_last)

    ref = dense.forward_policy_value(x)
    got = gathered.forward_policy_value(x)
    policy_err = (ref["policy"].float() - got["policy"].float()).abs().max().item()
    value_err = (ref["value"].float() - got["value"].float()).abs().max().item()
    if policy_err > 1.0e-2 or value_err > 1.0e-2:
        raise SystemExit(f"KV-gather gate failed: policy_err={policy_err} value_err={value_err}")

    dense_pps = _bench(dense, x, args.iters)
    gathered_pps = _bench(gathered, x, args.iters)
    speedup = gathered_pps / max(dense_pps, 1.0e-9)
    print(
        {
            "dense_positions_per_second": dense_pps,
            "kv_gather_positions_per_second": gathered_pps,
            "speedup": speedup,
            "policy_max_abs_err": policy_err,
            "value_logit_max_abs_err": value_err,
            "adopt": speedup >= args.min_speedup,
        }
    )
    if speedup < args.min_speedup:
        raise SystemExit(f"KV-gather speedup {speedup:.3f} below threshold {args.min_speedup:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
