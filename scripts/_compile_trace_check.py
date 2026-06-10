"""CPU full-model torch.compile(fullgraph=True) trace check.

Reproduces the dynamo tracing the live GPU gate will do (device-independent for
graph-break / data-dependent-guard errors), so we can confirm the whole folded
forward_policy_value compiles before spending a live bounce. Builds the real
config architecture, folds + KV-gathers + freezes the bias, compiles fullgraph,
and checks the compiled output matches eager.
"""
from __future__ import annotations
import sys, tomllib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("dense_cnn_restnet", "hexo_train", "hexo_runner"):
    src = ROOT / "packages" / pkg / "python"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from dense_cnn_restnet.architecture import RestnetNetwork, RelPosMHSA, optimized_restnet_for_inference
from dense_cnn_restnet.config import parse_model1_config
from dense_cnn_restnet.constants import BOARD_SIZE, INPUT_CHANNELS


def main() -> int:
    raw = tomllib.loads((ROOT / "configs/dense_cnn_restnet_main1.toml").read_text())
    cfg = parse_model1_config(raw["model"]["config"]).architecture
    model = RestnetNetwork(
        in_channels=cfg.input_channels, channels=cfg.channels, blocks_type=cfg.blocks_type,
        attention_heads=cfg.attention_heads, mlp_ratio=cfg.mlp_ratio,
        embed_kernel_size=cfg.embed_kernel_size, residual_conv=cfg.residual_conv,
        attention_impl=cfg.attention_impl, attention_scope=cfg.attention_scope, dropout=cfg.dropout,
        short_term_value_horizons=cfg.short_term_value_horizons, moves_left_head=cfg.moves_left_head,
    ).eval()
    folded = optimized_restnet_for_inference(model, attention_kv_gather=True)
    for m in folded.modules():
        if isinstance(m, RelPosMHSA):
            m.freeze_inference_bias(dtype=torch.float32, device=torch.device("cpu"))

    x = torch.zeros((4, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE))
    x[:, 0, BOARD_SIZE // 2, BOARD_SIZE // 2] = 1
    with torch.inference_mode():
        eager = folded.forward_policy_value(x)
        compiled = torch.compile(folded.forward_policy_value, fullgraph=True, dynamic=False)
        got = compiled(x)
    pol = (eager["policy"] - got["policy"]).abs().max().item()
    val = (eager["value"] - got["value"]).abs().max().item()
    print(f"FULLGRAPH COMPILE OK  policy_max_err={pol:.2e}  value_max_err={val:.2e}")
    assert pol < 1e-4 and val < 1e-4, "compiled output diverged from eager"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
