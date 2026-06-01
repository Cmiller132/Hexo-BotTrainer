"""First eval of the BC-trained hexgt: held-out IMITATION quality vs the dense_cnn
MCTS teacher. No game-play / SealBot needed — measures, on positions the BC run
never saw (a different epoch), how well hexgt reproduces the teacher's policy +
value:
  - held-out policy cross-entropy (vs train CE -> generalization).
  - top-1 policy agreement: hexgt's argmax candidate == teacher's most-visited
    candidate (chance ~1/median_candidates ~0.4%).
  - value MAE: |decode(hexgt value) - teacher value|.

Usage: python scripts/_bc_eval.py --ckpt _hexgt_bc.pt --shards <dir> --epoch epoch_000022
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.expand import build_training_batch
from hexo_models.hexgt.losses import decode_binned_value, segment_softmax_cross_entropy


def _seg_argmax(values, seg, num_seg, device):
    """Per-segment argmax index (into the global candidate array) of `values`."""
    best_val = torch.full((num_seg,), float("-inf"), device=device)
    best_idx = torch.full((num_seg,), -1, dtype=torch.long, device=device)
    order = torch.arange(values.shape[0], device=device)
    # scatter-reduce amax then find matching index (stable enough for a metric)
    best_val = best_val.scatter_reduce(0, seg, values, reduce="amax", include_self=True)
    is_max = values >= (best_val[seg] - 1e-6)
    # first max per segment
    cand = torch.where(is_max, order, torch.full_like(order, 1 << 30))
    best_idx = torch.full((num_seg,), 1 << 30, dtype=torch.long, device=device).scatter_reduce(
        0, seg, cand, reduce="amin", include_self=True)
    return best_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/e/Hexo-BotTrainer-hexgt/_hexgt_bc.pt")
    ap.add_argument("--shards", required=True)
    ap.add_argument("--epoch", default="epoch_000022")
    ap.add_argument("--max-shards", type=int, default=120)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["arch"]
    model = HexgtNetwork(
        token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
        ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
        short_term_value_horizons=tuple(a["short_term_value_horizons"]),
    ).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {args.ckpt}: trained rows={ck['train_state'].get('total_num_data_rows')}", flush=True)

    shard_dir = Path(args.shards)
    shards = sorted(shard_dir.glob(f"{args.epoch}_game_*.npz"))[: args.max_shards]
    rows = []
    for p in shards:
        rows.extend(read_compact_shard(p))
    print(f"held-out: {len(rows)} rows from {args.epoch} ({len(shards)} shards)", flush=True)

    tot_ce = 0.0
    tot_graphs = 0
    agree = 0
    val_abs = 0.0
    val_n = 0
    fp16 = device.type == "cuda"
    with torch.no_grad():
        for start in range(0, len(rows), args.batch):
            chunk = rows[start : start + args.batch]
            batch, targets = build_training_batch(chunk, n=args.n, horizons=(), prune_max_dropped_mass=0.15)
            if batch is None:
                continue
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            targets = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
            ng = int(batch["num_graphs"])
            seg = batch["candidate_graph"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                out = model.forward_policy_value(batch)
            pol = out["policy"].float()
            ce = segment_softmax_cross_entropy(pol, targets["policy"].float(), seg, ng)
            tot_ce += float(ce) * ng
            tot_graphs += ng
            # top-1 agreement
            pred = _seg_argmax(pol, seg, ng, device)
            tgt = _seg_argmax(targets["policy"].float(), seg, ng, device)
            valid = (pred < (1 << 30)) & (tgt < (1 << 30))
            agree += int((valid & (pred == tgt)).sum())
            # value MAE
            vt = targets["value"]
            tgt_scalar = decode_binned_value(vt.float()) if vt.ndim == 2 else vt.float()
            pred_scalar = decode_binned_value(out["value"].float())
            val_abs += float((pred_scalar - tgt_scalar).abs().sum())
            val_n += ng

    print(f"\n=== held-out imitation eval ({args.epoch}, n={args.n}) ===", flush=True)
    print(f"  policy CE (held-out):   {tot_ce / max(1, tot_graphs):.4f}   (uniform~ln(cands)~5.6; BC train reached ~3.0)")
    print(f"  top-1 policy agreement: {agree / max(1, tot_graphs):.1%}   (chance ~0.4%)")
    print(f"  value MAE:              {val_abs / max(1, val_n):.4f}   (value in [-1,1])")
    print(f"  graphs evaluated:       {tot_graphs}")


if __name__ == "__main__":
    main()
