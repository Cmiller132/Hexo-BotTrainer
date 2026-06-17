"""Supervised PRETRAIN (behavioral clone) of the hexgnn architecture on buffer
samples — the hexgnn analog of scripts/_pretrain_model3.py.

Trains the transformer-free hexgnn model (GNN trunk + policy/value/opp heads, NO
STV) from init on N rows pulled from an existing self-play/replay buffer
(READ-ONLY), then writes a fresh checkpoint the RL driver (_rl_train_hexgnn.py)
seeds from. Mirrors the model-3 pretrain: optional soft-Z label de-saturation with
a NEUTRAL root_value proxy (value = (1-λ)·z) for buffers that predate root_value
storage; reports policy top-1 imitation accuracy + holdout value loss + the
same-board v(A)+v(B) optimism sanity sum.

It is ADDITIVE: imports the top-level `hexgnn` package (which reuses the built
native `hexo_models._rust.hexgt`), reads the buffer read-only, and writes only to
its own --out path.
"""
from __future__ import annotations

import argparse
import glob
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    sys.path.insert(0, str(ROOT / "packages" / p / "python"))
sys.path.insert(0, str(ROOT / "packages" / "hexgnn" / "python"))

import torch

import hexo_engine.api as eng
from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexgnn.architecture import HexgnnNetwork
from hexgnn.config import parse_hexgnn_config
from hexgnn.constants import FEATURE_SCHEMA_VERSION, VALUE_BINS
from hexgnn.expand import build_training_batch
from hexgnn.losses import binned_value_loss
from hexgnn.trainer import HexgnnTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/hexgnn_model.toml"))
    ap.add_argument("--buffer", default=str(ROOT / "runs/hexgt_rl_main3/selfplay"),
                    help="read-only directory of compact .npz shards to clone from")
    ap.add_argument("--samples", type=int, default=15000)
    ap.add_argument("--holdout", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--soft-z-lambda", type=float, default=0.0)
    ap.add_argument("--out", default=str(ROOT / "runs/hexgnn_rl/pretrain/hexgnn_pretrain.pt"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(7)
    cfg_raw = tomllib.loads(Path(args.config).read_text())["model"]["config"]
    cfg = parse_hexgnn_config(cfg_raw)
    a = cfg.architecture
    print(f"[arch] gnn_layers={a.gnn_layers} value_pma_seeds={a.value_pma_seeds} token_dim={a.token_dim} "
          f"heads={a.attention_heads} | soft_z_lambda(proxy)={args.soft_z_lambda}", flush=True)

    model = HexgnnNetwork(
        node_feature_dim=a.node_feature_dim, token_dim=a.token_dim, gnn_layers=a.gnn_layers,
        attention_heads=a.attention_heads, dropout=a.dropout,
        value_pma_seeds=a.value_pma_seeds, value_head_use_side=a.value_head_use_side,
    ).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparams:,} device={dev}", flush=True)

    shards = sorted(glob.glob(str(Path(args.buffer) / "*.npz")))
    if not shards:
        raise SystemExit(f"ABORT: no .npz shards under {args.buffer}")
    need = args.samples + args.holdout
    rows = []
    for sp in shards:
        rows.extend(read_compact_shard(Path(sp)))
        if len(rows) >= need:
            break
    rows = rows[:need]
    print(f"[data] gathered {len(rows)} rows from {Path(args.buffer).name} (read-only)", flush=True)

    lam = float(args.soft_z_lambda)
    if lam > 0.0:
        rows = [replace(r, value=(1.0 - lam) * float(r.value)) for r in rows]
    train_rows, eval_rows = rows[: args.samples], rows[args.samples:]
    print(f"[data] train={len(train_rows)} holdout={len(eval_rows)} | value target = (1-{lam})*z", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    trainer = HexgnnTrainer(model=model, config=cfg, optimizer=opt)
    hist = trainer.train_on_rows(train_rows, batch_size=cfg.training.batch_size, epochs=args.epochs)
    n = max(1, len(hist) // 5)
    first = {k: sum(h[k] for h in hist[:n]) / n for k in ("total", "policy", "value")}
    last = {k: sum(h[k] for h in hist[-n:]) / n for k in ("total", "policy", "value")}
    print(f"[train] steps={len(hist)} epochs={args.epochs}", flush=True)
    print(f"[train] loss first->last: total {first['total']:.4f}->{last['total']:.4f} | "
          f"policy {first['policy']:.4f}->{last['policy']:.4f} | value {first['value']:.4f}->{last['value']:.4f}", flush=True)

    model.eval()
    correct = total = 0
    vloss_sum = vbatches = 0.0
    with torch.no_grad():
        for s in range(0, len(eval_rows), 128):
            chunk = eval_rows[s: s + 128]
            batch, targets = build_training_batch(chunk, n=a.candidate_radius, prune_max_dropped_mass=1.0)
            if batch is None:
                continue
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            cg = batch["candidate_graph"].cpu()
            pol = out["policy"].float().cpu()
            tgt = targets["policy"].float().cpu()
            ng = int(batch["num_graphs"])
            for g in range(ng):
                m = (cg == g).nonzero(as_tuple=True)[0]
                if m.numel() == 0:
                    continue
                if int(pol[m].argmax()) == int(tgt[m].argmax()):
                    correct += 1
                total += 1
            vloss_sum += float(binned_value_loss(out["value"].float().cpu(), targets["value"].float().cpu()))
            vbatches += 1
    acc = correct / max(1, total)
    print(f"[eval] policy top-1 imitation acc = {acc:.3f} ({correct}/{total}) | "
          f"holdout value loss = {vloss_sum / max(1, vbatches):.4f}", flush=True)

    import hexgnn.graph_build as gb
    bins = torch.linspace(-1.0, 1.0, VALUE_BINS)

    def value_of(state):
        b = gb.batch_from_states([state], n=a.candidate_radius)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            logits = model(b)["value"].float().cpu()
        return float((torch.softmax(logits, dim=-1)[0] * bins).sum())

    sums = []
    for seed in (11, 23, 37, 51, 67):
        s = eng.new_game(seed=seed)
        for _ in range(12):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            eng.apply_action(s, acts[len(acts) // 2])
        if eng.terminal(s) is not None:
            continue
        va = value_of(s)
        acts = list(eng.legal_actions(s))
        eng.apply_action(s, acts[len(acts) // 2])
        if eng.terminal(s) is not None:
            continue
        vb = value_of(s)
        sums.append(va + vb)
        print(f"   seed {seed}: v(A)={va:+.3f} v(next)={vb:+.3f} sum={va + vb:+.3f}", flush=True)
    if sums:
        print(f"[sanity] mean v(A)+v(B) optimism sum = {sum(sums) / len(sums):+.3f} (0 = calibrated)", flush=True)

    arch_meta = {
        "token_dim": a.token_dim, "gnn_layers": a.gnn_layers, "attention_heads": a.attention_heads,
        "value_pma_seeds": a.value_pma_seeds, "value_head_use_side": a.value_head_use_side,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(), "arch": arch_meta,
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "rl_epoch": 0, "step": len(hist),
        "pretrain": {"samples": len(train_rows), "epochs": args.epochs, "soft_z_lambda_proxy": lam,
                     "policy_top1_acc": acc, "buffer": str(args.buffer)},
    }, out)
    print(f"[save] pretrained hexgnn model -> {out}", flush=True)


if __name__ == "__main__":
    main()
