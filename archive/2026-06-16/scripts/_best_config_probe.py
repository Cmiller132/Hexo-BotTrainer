"""Definitive lockstep-vs-continuous throughput probe with ALL GPU levers
(fp16 + KV-gather + torch.compile) on the real checkpoint. Settles which
scheduler to run in production. GPU must be free.
"""
from __future__ import annotations
import argparse, copy, sys, tomllib
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("dense_cnn_restnet", "hexo_train", "hexo_runner"):
    src = ROOT / "packages" / pkg / "python"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from dense_cnn_restnet.architecture import RestnetNetwork
from dense_cnn_restnet.config import parse_model1_config
from dense_cnn_restnet.inference import DenseCNNInference
from dense_cnn_restnet.performance import _benchmark_selfplay_setting


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions", type=int, default=2048)
    args = ap.parse_args()
    raw = tomllib.loads((ROOT / "configs/dense_cnn_restnet_main1.toml").read_text())
    cfg = parse_model1_config(raw["model"]["config"])
    a = cfg.architecture
    model = RestnetNetwork(
        in_channels=a.input_channels, channels=a.channels, blocks_type=a.blocks_type,
        attention_heads=a.attention_heads, mlp_ratio=a.mlp_ratio, embed_kernel_size=a.embed_kernel_size,
        residual_conv=a.residual_conv, attention_impl=a.attention_impl, attention_scope=a.attention_scope,
        dropout=a.dropout, short_term_value_horizons=a.short_term_value_horizons,
        moves_left_head=a.moves_left_head,
    ).eval()
    p = torch.load(args.checkpoint, map_location="cpu")
    state = p.get("model_state_dict") or p.get("state_dict") or p if isinstance(p, dict) else p
    model.load_state_dict(state, strict=False)

    def probe(scheduler: str):
        c = replace(cfg, selfplay=replace(cfg.selfplay, scheduler=scheduler))
        inf = DenseCNNInference(
            copy.deepcopy(model), device="cuda", amp=c.training.amp, return_logits=False,
            max_batch_size=max(c.performance.inference_batch_candidates),
            fp16_model=True, use_torch_compile=True, compile_allow_fallback=True,
            attention_kv_gather=True,
        )
        r = _benchmark_selfplay_setting(
            inference=inf, config=c, selfplay_batch_size=min(c.selfplay.active_games, c.selfplay.mcts_active_root_limit),
            virtual_batch_size=c.performance.mcts_virtual_batch_candidates[0],
            visits=c.selfplay.search_visits, probe_positions=args.positions,
        )
        return r, inf

    ls, ls_inf = probe("lockstep")
    co, co_inf = probe("continuous")
    print(f"positions={args.positions} (all levers: fp16 + KV-gather + compile)")
    print(f"  lockstep    pos_s={ls['positions_per_second']:.3f}  exact={ls['all_searches_exact']}  "
          f"compile_adopted={ls_inf.compile_info.get('adopted')}")
    print(f"  continuous  pos_s={co['positions_per_second']:.3f}  exact={co['all_searches_exact']}  "
          f"compile_adopted={co_inf.compile_info.get('adopted')}")
    winner = "lockstep" if ls["positions_per_second"] >= co["positions_per_second"] else "continuous"
    delta = abs(ls["positions_per_second"] - co["positions_per_second"]) / min(
        ls["positions_per_second"], co["positions_per_second"]) * 100
    print(f"  => {winner} faster by {delta:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
