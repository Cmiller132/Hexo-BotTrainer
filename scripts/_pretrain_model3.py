"""Phase C: supervised PRETRAIN of the new model-3 architecture (4 GNN layers +
PMA k=2 value head + STV heads + v3 threat features) on 15k samples from the
HALTED run's replay buffer (runs/hexgt_rl_main2/selfplay, READ-ONLY). Warm-starts
the mostly-new weights (trained from init, not grafted from the old 3-layer
mean+max checkpoint). Writes a fresh checkpoint Phase D seeds from.

soft-Z note: the epoch-42 buffer predates root_value storage (values are hard
+-1), so true per-position soft-Z (value = (1-λ)z + λ·root_value) cannot be
reconstructed. For the warm-start we apply soft-Z with a NEUTRAL root_value proxy
(v̂ = 0) -> value = (1-λ)·z (λ=0.5 -> 0.5·z), which de-saturates the value label
(the soft-Z purpose). Real per-position root_value soft-Z applies in the RL run.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

ROOT = Path("/mnt/e/Hexo-BotTrainer-hexgt")
for p in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    sys.path.insert(0, str(ROOT / "packages" / p / "python"))

import glob
import torch

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction
from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.constants import FEATURE_SCHEMA_VERSION, VALUE_BINS
from hexo_models.hexgt.expand import build_training_batch
from hexo_models.hexgt.losses import binned_value_loss
from hexo_models.hexgt.trainer import HexgtTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/hexgt_model3.toml"))
    ap.add_argument("--buffer", default=str(ROOT / "runs/hexgt_rl_main2/selfplay"))
    ap.add_argument("--samples", type=int, default=15000)
    ap.add_argument("--holdout", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--soft-z-lambda", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "runs/hexgt_rl_main3/pretrain/hexgt_model3_pretrain.pt"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(7)
    cfg_raw = tomllib.loads(Path(args.config).read_text())["model"]["config"]
    cfg = parse_hexgt_config(cfg_raw)
    a = cfg.architecture
    print(f"[arch] gnn_layers={a.gnn_layers} value_pma_seeds={a.value_pma_seeds} token_dim={a.token_dim} "
          f"stv={a.short_term_value_horizons} | soft_z_lambda(proxy)={args.soft_z_lambda}", flush=True)

    model = HexgtNetwork(
        node_feature_dim=a.node_feature_dim, token_dim=a.token_dim, gnn_layers=a.gnn_layers,
        ctx_layers=a.ctx_layers, attention_heads=a.attention_heads, ffn_dim=a.ffn_dim,
        dropout=a.dropout, short_term_value_horizons=a.short_term_value_horizons,
        value_pma_seeds=a.value_pma_seeds,
    ).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparams:,} device={dev}", flush=True)

    # --- gather rows (read-only) ---------------------------------------------
    shards = sorted(glob.glob(str(Path(args.buffer) / "*.npz")))
    need = args.samples + args.holdout
    rows = []
    for sp in shards:
        rows.extend(read_compact_shard(Path(sp)))
        if len(rows) >= need:
            break
    rows = rows[:need]
    print(f"[data] gathered {len(rows)} rows from {Path(args.buffer).name} (read-only)", flush=True)

    # soft-Z with neutral root_value proxy: value = (1-λ)·z.
    lam = float(args.soft_z_lambda)
    rows = [replace(r, value=(1.0 - lam) * float(r.value)) for r in rows]
    train_rows, eval_rows = rows[: args.samples], rows[args.samples:]
    print(f"[data] train={len(train_rows)} holdout={len(eval_rows)} | value target = (1-{lam})*z", flush=True)

    # --- train ----------------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    trainer = HexgtTrainer(model=model, config=cfg, optimizer=opt)
    hist = trainer.train_on_rows(train_rows, batch_size=cfg.training.batch_size, epochs=args.epochs)
    n = max(1, len(hist) // 5)
    first = {k: sum(h[k] for h in hist[:n]) / n for k in ("total", "policy", "value")}
    last = {k: sum(h[k] for h in hist[-n:]) / n for k in ("total", "policy", "value")}
    print(f"[train] steps={len(hist)} epochs={args.epochs}", flush=True)
    print(f"[train] loss first->last: total {first['total']:.4f}->{last['total']:.4f} | "
          f"policy {first['policy']:.4f}->{last['policy']:.4f} | value {first['value']:.4f}->{last['value']:.4f}", flush=True)

    # --- eval: policy top-1 imitation + value loss on holdout -----------------
    model.eval()
    correct = total = 0
    vloss_sum = vbatches = 0.0
    with torch.no_grad():
        for s in range(0, len(eval_rows), 128):
            chunk = eval_rows[s : s + 128]
            batch, targets = build_training_batch(chunk, n=a.candidate_radius,
                                                  horizons=a.short_term_value_horizons,
                                                  prune_max_dropped_mass=1.0)
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

    # --- sanity: same-board v(A)+v(B) optimism sum ----------------------------
    import hexo_models.hexgt.graph_build as gb
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
        mean_sum = sum(sums) / len(sums)
        print(f"[sanity] mean v(A)+v(B) optimism sum = {mean_sum:+.3f} (old mean+max head baseline was ~+0.82)", flush=True)

    # --- save pretrained checkpoint (NEW dir; does not touch the halted run) ---
    arch_meta = {
        "token_dim": a.token_dim, "gnn_layers": a.gnn_layers, "ctx_layers": a.ctx_layers,
        "ffn_dim": a.ffn_dim, "attention_heads": a.attention_heads,
        "short_term_value_horizons": list(a.short_term_value_horizons),
        "value_pma_seeds": a.value_pma_seeds,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(), "arch": arch_meta,
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "rl_epoch": 0, "step": len(hist),
        "pretrain": {"samples": len(train_rows), "epochs": args.epochs, "soft_z_lambda_proxy": lam,
                     "policy_top1_acc": acc, "buffer": str(args.buffer)},
    }, out)
    print(f"[save] pretrained model -> {out}", flush=True)


if __name__ == "__main__":
    main()
