"""GPU self-play smoke: load the BC step-6009 seed, play a few games, report
real pos/s + play-style aggregates. Confirms selfplay.py end-to-end on CUDA."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.selfplay import run_selfplay_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc/hexgt_bc_step006009.pt")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--active", type=int, default=24)
    ap.add_argument("--vbatch", type=int, default=64)
    ap.add_argument("--visits", type=int, default=128)
    ap.add_argument("--max-actions", type=int, default=200)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out-dir", default="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl/_smoke")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["arch"]
    model = HexgtNetwork(
        token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
        ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
        short_term_value_horizons=tuple(a["short_term_value_horizons"]),
    ).to(device).eval()
    model.load_state_dict(ck["model"])
    if args.compile and device.type == "cuda":
        model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)
    print(f"loaded BC seed step={ck.get('step')} params={sum(p.numel() for p in model.parameters()):,} device={device}", flush=True)

    cfg = parse_hexgt_config({
        "device": str(device),
        "selfplay": {
            "search_visits": args.visits,
            "max_actions": args.max_actions,
            "temperature": 1.0, "final_temperature": 0.2, "temperature_decay_moves": 30,
            "forced_playout_k": 2.0,
        },
    })

    t0 = time.perf_counter()
    res = run_selfplay_games(
        model, cfg, num_games=args.games, output_dir=Path(args.out_dir), epoch=0,
        device=str(device), fp16=(device.type == "cuda"), base_seed=123,
        active_games=args.active, virtual_batch_size=args.vbatch, collect_examples=2,
        progress=None,
    )
    print(f"\n=== self-play smoke ({time.perf_counter()-t0:.1f}s wall) ===", flush=True)
    d = res.as_dict()
    for k in ("completed_games", "truncated_games", "searched_positions", "raw_samples",
              "positions_per_second", "search_positions_per_second", "mean_candidate_count",
              "mean_visit_entropy", "mean_prior_entropy", "mean_top_visit_fraction"):
        print(f"  {k} = {d[k]:.3f}" if isinstance(d[k], float) else f"  {k} = {d[k]}", flush=True)
    g = res.example_games[0]
    print(f"\n  example game {g['game_id']} winner={g['winner']} turns={g['turns']}", flush=True)
    for m in g["moves"][:6]:
        print(f"    move {m['move']:3d} p={m['player']} v={m['root_value']:+.3f} "
              f"cand={m['candidates']:4d} vis_H={m['visit_entropy']:.2f} pri_H={m['prior_entropy']:.2f} "
              f"top={m['top_visit_fraction']:.2f} T={m['temperature']:.2f}", flush=True)


if __name__ == "__main__":
    main()
