#!/usr/bin/env python
"""tap31 marginal-value probe: distill the Strix anchor with FULL v2 features.

Closes the validity gap of scripts/_tap31_probe.py (which withheld the window
planes and asked the nets to reconstruct them): here the students receive the
COMPLETE 46-plane v2 featurization — window planes included, exactly like the
real trunk — and must fit a strong evaluation function instead: the policy and
value of the Strix anchor checkpoint (~237k-step HeXONet, this repo's eval
opponent). If tap31's richer line read helps ON TOP of the hand-computed
planes, its edge should appear here; if the planes already saturate what the
read provides, ray7 should match it — cheap evidence against building the
tap31 kernels.

Teacher: hexo_strix load_strix_checkpoint + build_axis_graph per position
(CPU, at dataset-build time). Policy targets are the teacher's softmax over
its legal nodes, restricted to the hexfield support's legal set (radius-4
model support vs the teacher's radius-8 legality) and renormalized; coverage
is reported. Value is the teacher's tanh scalar (to-move perspective).

Students: the base probe's mechanisms unchanged (shared ray gather + own/opp
visibility halves; ray7 = alpha-collapsed read, tap31 = dense 31-tap read),
stem widened to 46 planes, plus a per-legal-node policy head and a masked
mean-pool value head. Loss = policy CE + 5 * value MSE. Reported per arm:
val policy KL to teacher, top-1 agreement, value MSE.

Caveats: positions come from the base script's line-biased random generator —
off-distribution for the teacher (its judgments there are noisier than in
real games), but identical targets for every arm, so the ARM ORDERING remains
meaningful; teacher output stats are printed so degenerate targets would be
visible. Micro-scale caveats of the base probe apply unchanged.

Run: <hexo-bot venv python> scripts/_tap31_distill_probe.py \
       [--ckpt ~/Downloads/checkpoint_00237000.pt] [--seed 0] [--arms ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HEXFIELD_EQ_FEATURE_VERSION", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "packages" / "hexo_strix" / "python"))

import torch
import torch.nn as nn

import _tap31_probe as base
from hexo_strix.graph import build_axis_graph
from hexo_strix.loader import load_strix_checkpoint

_MOVES_REMAINING = {"Opening": 1, "FirstStone": 2, "SecondStone": 1}


def label_with_teacher(ds: list[dict], ck, log) -> dict:
    """Attach 'pol' (L,) renormalized teacher policy over the hexfield legal
    rows, 'val' scalar, and 'cov' (teacher mass inside the hexfield legal set)
    to each position. Returns teacher-output stats."""

    mc, gc = ck.model_config, ck.game_config
    win_len = int(gc.get("win_length", 6))
    radius = int(gc.get("placement_radius", 8))
    rel = bool(mc.get("relative_stones", True))
    thr = bool(mc.get("threat_features", True))
    vals, ents, covs = [], [], []
    t0 = time.time()
    for i, pos in enumerate(ds):
        stones = [((q, r), 1 if owner == 0 else -1) for q, r, owner, _ in pos["records"]]
        to_move = 1 if pos["current_player"] == 0 else -1
        g = build_axis_graph(
            stones, to_move=to_move,
            moves_remaining=_MOVES_REMAINING[pos["phase"]],
            win_length=win_len, placement_radius=radius,
            relative_stones=rel, threat_features=thr,
        )
        with torch.no_grad():
            logits, value = ck.model(
                g.x, g.edge_index, g.legal_mask, g.stone_mask, edge_attr=g.edge_attr)
        probs = torch.softmax(logits, dim=0)
        pmap = {c: float(p) for c, p in zip(g.legal_coords, probs)}
        raw = np.array([pmap.get(c, 0.0) for c in pos["legal_coords"]], dtype=np.float32)
        cov = float(raw.sum())
        pos["pol"] = raw / max(cov, 1e-9)
        pos["val"] = float(value)
        pos["cov"] = cov
        vals.append(pos["val"])
        p = pos["pol"][pos["pol"] > 0]
        ents.append(float(-(p * np.log(p)).sum()))
        covs.append(cov)
        if (i + 1) % 500 == 0:
            log(f"  [teacher] {i+1}/{len(ds)} ({time.time()-t0:.0f}s)")
    return {
        "value_mean": float(np.mean(vals)), "value_std": float(np.std(vals)),
        "policy_entropy_mean": float(np.mean(ents)),
        "coverage_mean": float(np.mean(covs)),
        "coverage_p10": float(np.percentile(covs, 10)),
    }


def collate(batch: list[dict], device):
    npad = max(p["xf"].shape[0] for p in batch)
    b = len(batch)
    xf = np.zeros((b, npad, batch[0]["xf"].shape[1]), dtype=np.float32)
    pol = np.zeros((b, npad), dtype=np.float32)
    val = np.zeros((b,), dtype=np.float32)
    mask = np.zeros((b, npad), dtype=np.float32)
    legal = np.zeros((b, npad), dtype=bool)
    idx = np.full((b, npad, 6, 5), npad, dtype=np.int64)
    reach = np.zeros((b, npad, 2, 6), dtype=np.int64)
    for i, p in enumerate(batch):
        n = p["xf"].shape[0]
        nl = len(p["legal_coords"])
        xf[i, :n] = p["xf"]
        pol[i, :nl] = p["pol"]
        val[i] = p["val"]
        mask[i, :n] = 1.0
        legal[i, :nl] = True
        pi = p["idx"].astype(np.int64)
        idx[i, :n] = np.where(pi < 0, npad, pi)
        reach[i, :n] = p["reach"]
    tt = lambda a: torch.from_numpy(a).to(device)
    return tt(xf), tt(pol), tt(val), tt(mask), tt(legal), tt(idx), tt(reach)


class DistillNet(nn.Module):
    def __init__(self, nf: int, c: int, mech: str, depth: int) -> None:
        super().__init__()
        self.stem = nn.Linear(nf, c)
        self.blocks = nn.ModuleList(base.Block(c, mech) for _ in range(depth))
        self.pol_head = nn.Linear(c, 1)
        self.val_head = nn.Sequential(nn.Linear(c, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, xf, idx, reach, mask, legal):
        h = self.stem(xf) * mask.unsqueeze(-1)
        for blk in self.blocks:
            h = blk(h, idx, reach, mask)
        logits = self.pol_head(h).squeeze(-1).masked_fill(~legal, -1e9)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        return logits, self.val_head(pooled).squeeze(-1)


def losses(logits, vpred, pol, val):
    logp = torch.log_softmax(logits, dim=1)
    ce = -(pol * logp).sum(dim=1)          # per-position cross-entropy
    ent = -(pol * (pol + 1e-12).log()).sum(dim=1)
    kl = ce - ent                           # KL(teacher || student)
    mse = (vpred - val) ** 2
    return ce.mean(), kl.mean(), mse.mean()


def evaluate(net, val_batches):
    net.eval()
    kls, mses, hits, cnt = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for xf, pol, val, mask, legal, idx, reach in val_batches:
            logits, vpred = net(xf, idx, reach, mask, legal)
            _, kl, mse = losses(logits, vpred, pol, val)
            b = xf.shape[0]
            kls += float(kl) * b
            mses += float(mse) * b
            hits += int((logits.argmax(1) == pol.argmax(1)).sum())
            cnt += b
    net.train()
    return {"policy_kl": kls / cnt, "value_mse": mses / cnt, "top1": hits / cnt}


def run_arm(name, mech, c, depth, steps, nf, pool, val_batches, device, seed, log):
    torch.manual_seed(seed)
    net = DistillNet(nf, c, mech, depth).to(device)
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * (0.5 * (1 + math.cos(math.pi * s / steps))))
    rng = random.Random(seed)
    best, t0 = None, time.time()
    for step in range(steps):
        xf, pol, val, mask, legal, idx, reach = (
            t.to(device) for t in pool[rng.randrange(len(pool))])
        logits, vpred = net(xf, idx, reach, mask, legal)
        ce, _, mse = losses(logits, vpred, pol, val)
        loss = ce + 5.0 * mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 200 == 0 or step + 1 == steps:
            ev = evaluate(net, val_batches)
            if best is None or ev["policy_kl"] < best["policy_kl"]:
                best = ev
            log(f"  {name} step {step+1}/{steps} loss {loss.item():.4f} "
                f"KL {ev['policy_kl']:.4f} top1 {ev['top1']:.3f} "
                f"vMSE {ev['value_mse']:.4f} ({time.time()-t0:.0f}s)")
    return {"arm": name, "mech": mech, "C": c, "depth": depth, "params": params,
            "seed": seed, "steps": steps, "wall_s": round(time.time() - t0, 1), **best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path.home() / "Downloads/checkpoint_00237000.pt"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-positions", type=int, default=2200)
    ap.add_argument("--val-positions", type=int, default=200)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=26)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else (
        args.device if args.device != "auto" else "cpu")
    log = lambda m: print(m, flush=True)

    ck = load_strix_checkpoint(args.ckpt)
    log(f"[teacher] {Path(args.ckpt).name} train_steps={ck.train_steps} "
        f"model={ {k: ck.model_config[k] for k in ('hidden_dim','num_layers')} } "
        f"game={ck.game_config}")

    log(f"[gen] {args.train_positions}+{args.val_positions} positions ...")
    t0 = time.time()
    train_ds = base.gen_dataset(1234, args.train_positions, args.max_plies)
    val_ds = base.gen_dataset(9876, args.val_positions, args.max_plies)
    log(f"[gen] done in {time.time()-t0:.0f}s")
    stats = label_with_teacher(train_ds + val_ds, ck, log)
    log(f"[teacher] stats: { {k: round(v, 4) for k, v in stats.items()} }")

    nf = train_ds[0]["xf"].shape[1]
    bs = args.batch
    val_batches = [collate(val_ds[i:i + bs], device) for i in range(0, len(val_ds), bs)]
    order = sorted(range(len(train_ds)), key=lambda i: train_ds[i]["xf"].shape[0])
    pool = [collate([train_ds[j] for j in order[i:i + bs]], "cpu")
            for i in range(0, len(order) - bs + 1, bs)]
    log(f"[pool] {len(pool)} batches of {bs}; NF={nf}; device={device}")

    arms = [
        ("d2-ray7-C64",   "ray7", 64, 2, args.steps),
        ("d2-tap31-C64",  "tap31", 64, 2, args.steps),
        ("d2-ray7-C128",  "ray7", 128, 2, args.steps),
    ]
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",")}
        unknown = keep - {a[0] for a in arms}
        assert not unknown, f"unknown arms: {unknown}"
        arms = [a for a in arms if a[0] in keep]

    out = args.out or f"/tmp/tap31_distill_seed{args.seed}.json"
    results = []
    for name, mech, c, depth, steps in arms:
        log(f"[arm] {name}")
        results.append(run_arm(name, mech, c, depth, steps, nf, pool, val_batches,
                               device, args.seed, log))
        Path(out + ".partial").write_text(json.dumps(
            {"seed": args.seed, "device": device, "teacher": stats,
             "results": results}, indent=2))

    hdr = f"{'arm':<15}{'params':>9}{'polKL':>9}{'top1':>7}{'vMSE':>9}"
    log("\n" + hdr)
    log("-" * len(hdr))
    for r in results:
        log(f"{r['arm']:<15}{r['params']:>9}{r['policy_kl']:>9.4f}"
            f"{r['top1']:>7.3f}{r['value_mse']:>9.4f}")
    Path(out).write_text(json.dumps(
        {"seed": args.seed, "device": device, "teacher": stats, "results": results},
        indent=2))
    log(f"\n[out] {out}")


if __name__ == "__main__":
    main()
