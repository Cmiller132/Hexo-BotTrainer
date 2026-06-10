from __future__ import annotations

import argparse
import copy
import sys
import tomllib
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


def _load_config(path: Path):
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return parse_model1_config(raw["model"]["config"])


def _build_model(cfg) -> RestnetNetwork:
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


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict):
        # Production checkpoints (checkpoints.py) store the weights under
        # "model_state"; the other keys cover external/prefit formats.
        state = (
            payload.get("model_state")
            or payload.get("model_state_dict")
            or payload.get("state_dict")
            or payload
        )
    else:
        state = payload
    # strict: a checkpoint/architecture mismatch must abort the gate, not warn.
    model.load_state_dict(state, strict=True)


def _run_probe(model: torch.nn.Module, cfg, *, scheduler: str, positions: int):
    cfg = replace(cfg, selfplay=replace(cfg.selfplay, scheduler=scheduler, scheduler_flush_target=cfg.selfplay.scheduler_flush_target or cfg.performance.inference_batch_candidates[-1]))
    inference = DenseCNNInference(
        copy.deepcopy(model),
        device=cfg.device,
        amp=cfg.training.amp,
        return_logits=False,
        max_batch_size=max(cfg.performance.inference_batch_candidates),
        fp16_model=cfg.performance.inference_fp16_model,
        fp16_allow_fallback=cfg.performance.inference_fp16_allow_fallback,
        use_torch_compile=False,
        attention_kv_gather=cfg.performance.attention_kv_gather,
    )
    return _benchmark_selfplay_setting(
        inference=inference,
        config=cfg,
        selfplay_batch_size=min(cfg.selfplay.active_games, cfg.selfplay.mcts_active_root_limit),
        virtual_batch_size=cfg.performance.mcts_virtual_batch_candidates[0],
        visits=cfg.selfplay.search_visits,
        probe_positions=positions,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense_cnn_restnet_main1.toml")
    ap.add_argument("--checkpoint")
    ap.add_argument("--positions", type=int, default=64)
    args = ap.parse_args()

    cfg = _load_config(ROOT / args.config)
    model = _build_model(cfg)
    if args.checkpoint:
        _load_checkpoint(model, Path(args.checkpoint))

    lockstep = _run_probe(model, cfg, scheduler="lockstep", positions=args.positions)
    continuous = _run_probe(model, cfg, scheduler="continuous", positions=args.positions)
    print({"lockstep": lockstep, "continuous": continuous})
    for name, result in (("lockstep", lockstep), ("continuous", continuous)):
        if not result.get("all_searches_exact"):
            raise SystemExit(f"{name} did not return exact visit counts")
        expected = int(result["searched_positions"]) * int(cfg.selfplay.search_visits)
        if int(result["mcts_simulations"]) != expected:
            raise SystemExit(f"{name} sim mismatch: {result['mcts_simulations']} != {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
