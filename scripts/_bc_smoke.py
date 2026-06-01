"""Behavioral-clone smoke: train hexgt on real epoch-24 96x8 compact shards and
confirm the loss decreases (validates the full training path: compact shard ->
expand/recompute graph + targets -> GNN forward -> hexgt_loss -> optimizer step).

This is the step-3 BC validation. BC reads pre-recorded shards (no self-play), so
it is independent of the self-play featurization-throughput bottleneck.

Usage: python scripts/_bc_smoke.py --shards <96x8 selfplay dir> [--steps 120]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.trainer import HexgtTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shards", required=True)
    ap.add_argument("--epoch", default="epoch_000024")
    ap.add_argument("--max-shards", type=int, default=60)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="/mnt/e/Hexo-BotTrainer-hexgt/_hexgt_bc_smoke.pt")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    train_over = {}
    if args.lr is not None:
        train_over["learning_rate"] = args.lr
    if args.warmup is not None:
        train_over["warmup_steps"] = args.warmup
    cfg = parse_hexgt_config({"device": str(device), "training": train_over})
    arch = cfg.architecture
    model = HexgtNetwork(
        node_feature_dim=arch.node_feature_dim, token_dim=arch.token_dim,
        gnn_layers=arch.gnn_layers, ctx_layers=arch.ctx_layers,
        attention_heads=arch.attention_heads, ffn_dim=arch.ffn_dim,
        short_term_value_horizons=arch.short_term_value_horizons,
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    trainer = HexgtTrainer(model=model, config=cfg, optimizer=opt)
    print(f"device={device} params={nparams:,} n={arch.candidate_radius} batch={args.batch} "
          f"lr={cfg.training.learning_rate} warmup={cfg.training.warmup_steps}", flush=True)

    shard_dir = Path(args.shards)
    shards = sorted(shard_dir.glob(f"{args.epoch}_game_*.npz"))[: args.max_shards]
    if not shards:
        # fall back to any recent epoch
        allp = sorted(shard_dir.glob("*_game_*.npz"))
        epochs = sorted({p.name.split("_game_")[0] for p in allp})
        ep = epochs[-2] if len(epochs) > 1 else epochs[-1]
        shards = [p for p in allp if p.name.startswith(ep)][: args.max_shards]
        print(f"epoch {args.epoch} not found; using {ep}", flush=True)
    print(f"BC source: {len(shards)} shards from {shards[0].name.split('_game_')[0]}", flush=True)

    t0 = time.perf_counter()
    history = trainer.train_on_shards(shards, batch_size=args.batch, max_steps=args.steps)
    dt = time.perf_counter() - t0
    if not history:
        print("NO STEPS RAN (all rows pruned?)", flush=True)
        return

    def avg(hs, k):
        vs = [h[k] for h in hs if k in h]
        return sum(vs) / len(vs) if vs else float("nan")

    keys = [k for k in ("loss", "policy", "value", "opp_policy", "policy_loss", "value_loss") if k in history[0]]
    print(f"\nran {len(history)} steps in {dt:.1f}s ({dt/len(history)*1000:.0f} ms/step), "
          f"kept={trainer.kept_rows} pruned={trainer.pruned_rows} prune_rate={trainer.prune_rate:.1%}", flush=True)
    print(f"loss keys: {list(history[0].keys())}", flush=True)
    n = len(history)
    first = history[: max(1, n // 6)]
    last = history[-max(1, n // 6):]
    print(f"\n{'metric':>14} | {'first-sixth':>12} | {'last-sixth':>11} | {'delta':>9}")
    for k in keys:
        a, b = avg(first, k), avg(last, k)
        print(f"{k:>14} | {a:>12.4f} | {b:>11.4f} | {b-a:>+9.4f}")
    # step-by-step total loss trace (every ~10 steps)
    print("\ntotal-loss trace (step: loss):", flush=True)
    lk = "loss" if "loss" in history[0] else keys[0]
    for i in range(0, n, max(1, n // 12)):
        print(f"  {i:4d}: {history[i].get(lk, float('nan')):.4f}", flush=True)

    # save the BC checkpoint for eval (NOT into the live tree)
    out = Path(args.out)
    torch.save({"model": model.state_dict(), "train_state": trainer.train_state.to_dict(),
                "arch": {"token_dim": arch.token_dim, "gnn_layers": arch.gnn_layers,
                         "ctx_layers": arch.ctx_layers, "ffn_dim": arch.ffn_dim,
                         "attention_heads": arch.attention_heads,
                         "short_term_value_horizons": list(arch.short_term_value_horizons)}},
               out)
    print(f"\nsaved BC checkpoint -> {out}", flush=True)


if __name__ == "__main__":
    main()
