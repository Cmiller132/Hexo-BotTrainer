"""Converged behavioral-clone training for hexgt (Model 2).

Streams recent 96x8 compact shards in RAM-bounded groups (load group -> train one
pass -> free), loops many passes, checkpoints periodically, and runs a held-out
imitation eval (policy CE / top-1 agreement vs the dense_cnn MCTS teacher / value
MAE) on an epoch the run never trains on. Crash-safe: checkpoints on exception.

This is the BC seed for the converged model. Run in the background; tail the log.

Usage:
  python scripts/_bc_train.py --shards <96x8 selfplay> --target-steps 15000
"""

from __future__ import annotations

import argparse
import random
import time
import traceback
from pathlib import Path

import torch

from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.expand import build_training_batch
from hexo_models.hexgt.losses import decode_binned_value, segment_softmax_cross_entropy
from hexo_models.hexgt.trainer import HexgtTrainer


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def seg_argmax(values, seg, num_seg, device):
    best_val = torch.full((num_seg,), float("-inf"), device=device).scatter_reduce(
        0, seg, values, reduce="amax", include_self=True)
    order = torch.arange(values.shape[0], device=device)
    is_max = values >= (best_val[seg] - 1e-6)
    cand = torch.where(is_max, order, torch.full_like(order, 1 << 30))
    return torch.full((num_seg,), 1 << 30, dtype=torch.long, device=device).scatter_reduce(
        0, seg, cand, reduce="amin", include_self=True)


@torch.no_grad()
def heldout_eval(model, rows, n, device, batch=128):
    model.eval()
    fp16 = device.type == "cuda"
    tot_ce = 0.0; tot = 0; agree = 0; val_abs = 0.0
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        b, t = build_training_batch(chunk, n=n, horizons=(), prune_max_dropped_mass=0.15)
        if b is None:
            continue
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        t = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
        ng = int(b["num_graphs"]); seg = b["candidate_graph"]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            out = model.forward_policy_value(b)
        pol = out["policy"].float()
        tot_ce += float(segment_softmax_cross_entropy(pol, t["policy"].float(), seg, ng)) * ng
        tot += ng
        pred = seg_argmax(pol, seg, ng, device)
        tgt = seg_argmax(t["policy"].float(), seg, ng, device)
        valid = (pred < (1 << 30)) & (tgt < (1 << 30))
        agree += int((valid & (pred == tgt)).sum())
        vt = t["value"]
        tgt_s = decode_binned_value(vt.float()) if vt.ndim == 2 else vt.float()
        val_abs += float((decode_binned_value(out["value"].float()) - tgt_s).abs().sum())
    model.train()
    return dict(ce=tot_ce / max(1, tot), top1=agree / max(1, tot), val_mae=val_abs / max(1, tot), graphs=tot)


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--train-epochs", default="20,21,23,24")
    ap.add_argument("--shards-per-epoch", type=int, default=120)
    ap.add_argument("--heldout-epoch", default="epoch_000022")
    ap.add_argument("--heldout-shards", type=int, default=60)
    ap.add_argument("--target-steps", type=int, default=15000)
    ap.add_argument("--group-size", type=int, default=40)
    ap.add_argument("--ckpt-every", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "bc_train.log", "a")
    rng = random.Random(args.seed)

    cfg = parse_hexgt_config({"device": str(device), "training": {"learning_rate": args.lr, "warmup_steps": args.warmup}})
    a = cfg.architecture
    model = HexgtNetwork(
        node_feature_dim=a.node_feature_dim, token_dim=a.token_dim, gnn_layers=a.gnn_layers,
        ctx_layers=a.ctx_layers, attention_heads=a.attention_heads, ffn_dim=a.ffn_dim,
        short_term_value_horizons=a.short_term_value_horizons,
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    trainer = HexgtTrainer(model=model, config=cfg, optimizer=opt)

    sp = Path(args.shards)
    train_shards = []
    for e in args.train_epochs.split(","):
        ep = f"epoch_{int(e):06d}"
        train_shards += sorted(sp.glob(f"{ep}_game_*.npz"))[: args.shards_per_epoch]
    rng.shuffle(train_shards)
    heldout_shards = sorted(sp.glob(f"{args.heldout_epoch}_game_*.npz"))[: args.heldout_shards]
    heldout_rows = []
    for p in heldout_shards:
        heldout_rows.extend(read_compact_shard(p))

    arch_meta = {"token_dim": a.token_dim, "gnn_layers": a.gnn_layers, "ctx_layers": a.ctx_layers,
                 "ffn_dim": a.ffn_dim, "attention_heads": a.attention_heads,
                 "short_term_value_horizons": list(a.short_term_value_horizons)}

    def save(tag):
        path = out_dir / f"hexgt_bc_{tag}.pt"
        torch.save({"model": model.state_dict(), "train_state": trainer.train_state.to_dict(),
                    "arch": arch_meta, "step": trainer.train_state.global_step}, path)
        latest = out_dir / "hexgt_bc_latest.pt"
        torch.save({"model": model.state_dict(), "train_state": trainer.train_state.to_dict(),
                    "arch": arch_meta, "step": trainer.train_state.global_step}, latest)
        log(f"  ckpt saved -> {path.name}", fh)

    log(f"=== converged BC start: params={nparams:,} n={a.candidate_radius} device={device} "
        f"lr={args.lr} warmup={args.warmup} target_steps={args.target_steps} ===", fh)
    log(f"train shards={len(train_shards)} (epochs {args.train_epochs}), "
        f"heldout rows={len(heldout_rows)} ({args.heldout_epoch})", fh)

    last_ckpt = 0; last_eval = 0
    t0 = time.perf_counter()
    try:
        pass_idx = 0
        while trainer.train_state.global_step < args.target_steps:
            pass_idx += 1
            rng.shuffle(train_shards)
            for group in chunks(train_shards, args.group_size):
                if trainer.train_state.global_step >= args.target_steps:
                    break
                gt = time.perf_counter()
                hist = trainer.train_on_shards(group, batch_size=args.batch)
                if not hist:
                    continue
                step = trainer.train_state.global_step
                def avg(k):
                    vs = [h[k] for h in hist if k in h]; return sum(vs) / len(vs) if vs else float("nan")
                msps = (time.perf_counter() - gt) / max(1, len(hist)) * 1000
                log(f"pass {pass_idx} step {step}: total={avg('total'):.4f} "
                    f"policy={avg('policy'):.4f} value={avg('value'):.4f} opp={avg('opp_policy'):.4f} "
                    f"prune={trainer.prune_rate:.1%} {msps:.0f}ms/step", fh)
                if step - last_eval >= args.eval_every:
                    last_eval = step
                    ev = heldout_eval(model, heldout_rows, args.n, device, batch=args.batch)
                    log(f"  >>> HELDOUT @ {step}: policy_CE={ev['ce']:.4f} top1={ev['top1']:.1%} "
                        f"val_mae={ev['val_mae']:.4f} (graphs={ev['graphs']})", fh)
                if step - last_ckpt >= args.ckpt_every:
                    last_ckpt = step
                    save(f"step{step:06d}")
        save("final")
        ev = heldout_eval(model, heldout_rows, args.n, device, batch=args.batch)
        log(f"=== DONE step={trainer.train_state.global_step} in {(time.perf_counter()-t0)/60:.1f} min. "
            f"FINAL heldout: policy_CE={ev['ce']:.4f} top1={ev['top1']:.1%} val_mae={ev['val_mae']:.4f} ===", fh)
    except Exception:
        log("EXCEPTION — saving crash checkpoint:\n" + traceback.format_exc(), fh)
        save("crash")
        raise
    finally:
        fh.close()


if __name__ == "__main__":
    main()
