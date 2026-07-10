#!/usr/bin/env python
"""tap31 representation micro-probe (MacBook-scale, reference math only).

Question (docs/PLAN_TAP31_CONV.md): does the 31-tap dense read (per-(direction,
distance) full weight blocks, Design A) learn line-pattern functions that the
shipped ray-tap read (per-channel alpha-collapsed distance profile) cannot, at
matched parameters?

Probe: tiny nets get ONLY occupancy planes (own/opp/empty/legal) and must
regress the v2 featurizer's 30 graded window planes (own/opp x line, live,
live3, live4, live5 x 3 axes) per node. Those targets are exact line-pattern
functions of the board (features.window_features_for_cell), so reconstruction
quality measures precisely the capability tap31 claims: reading exact
stone/gap arrangements along axis lines. Ground truth is free (the featurizer
computes it); inputs withhold it.

Both mechanisms share identical machinery (ray gather to 6 dirs x 5 dists,
hard own/opp visibility via the raylen wire on channel halves, center tap,
one GEMM); they differ ONLY in the read:

  ray7 : tap_d = sum_k alpha[k,:] * vis * x_{i+k d}   -> GEMM(7C -> C)
         (alpha here is per-FULL-channel (5, C), i.e. strictly MORE free than
         the real tied (5, C_ORBIT) — a conservative handicap for tap31)
  tap31: all 30 masked ray fibers enter the GEMM      -> GEMM(31C -> C)

Untied (no D6 weight tying): the tie divides params by 12 in both classes
equally and does not change their relative expressivity; the probe compares
mechanisms, not the tying.

Arms (d1 = single linear conv, no nonlinearity — pure function-class gap;
d2 = two pre-LN residual conv blocks = 4 convs — can depth + channels close
the gap?):

  d1-ray7-C64 vs d1-tap31-C64          raw linear expressivity gap
  d2-ray7-C64  vs d2-tap31-C32         param-matched (tap31 narrowed)
  d2-tap31-C64 vs d2-ray7-C128         param-matched (ray7 widened) — the
                                       Q7 width control at micro scale

Positions: random 1-then-2 playouts biased toward line extension (uniform
scatter makes live4/live5 degenerate-zero), 6-in-row completions rejected
(decision states only). Reported: per-family R^2 on held-out positions.

Caveats (also in the results doc): micro-scale, occupancy-only inputs, no
D6 tying, no MCTS in the loop — a positive result is necessary-condition
evidence for tap31, not sufficient; a null result (ray7 matches everywhere)
is strong evidence to stop before the GPU ladder.

Run:  <hexo-bot venv python> scripts/_tap31_probe.py [--device auto] [--seed 0]
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "hexfield_eq" / "python"))
os.environ.setdefault("HEXFIELD_EQ_FEATURE_VERSION", "2")

import torch
import torch.nn as nn

from hexfield_eq import features as F
from hexfield_eq.constants import (
    AXIS_PLANE_BASE,
    DIRECTIONS,
    F_EMPTY,
    F_LEGAL,
    F_OPP_STONE,
    F_OWN_STONE,
    RAY_REACH,
)
from hexfield_eq.features import AXIS_DELTAS

assert RAY_REACH == 5
IN_PLANES = [F_OWN_STONE, F_OPP_STONE, F_EMPTY, F_LEGAL]
TARGET_LO, TARGET_HI = AXIS_PLANE_BASE, AXIS_PLANE_BASE + 30  # line..live5, 30 planes
FAMILIES = ["line", "live", "live3", "live4", "live5"]  # 6 planes each (own+opp x 3 axes)

_AXES = [AXIS_DELTAS[n] for n in ("Q", "R", "QR")]
_DIRS6 = list(DIRECTIONS)


def _tap_axis_dir() -> list[tuple[int, int]]:
    """(axis, dir) per direction tap, from the same tables _raytap derives it."""
    out = []
    for dq, dr in _DIRS6:
        for ai, (aq, ar) in enumerate(_AXES):
            if (dq, dr) == (aq, ar):
                out.append((ai, 0))
                break
            if (dq, dr) == (-aq, -ar):
                out.append((ai, 1))
                break
    assert len(out) == len(_DIRS6)
    return out


# --- position generation --------------------------------------------------------


def _creates_win(occ: dict, cell: tuple[int, int], owner: int) -> bool:
    for aq, ar in _AXES:
        run = 1
        for sgn in (1, -1):
            q, r = cell
            while occ.get((q + sgn * aq, r + sgn * ar)) == owner:
                run += 1
                q += sgn * aq
                r += sgn * ar
        if run >= 6:
            return True
    return False


def _hexd(dq: int, dr: int) -> int:
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _propose(rng: random.Random, occ: dict, owner: int) -> tuple[int, int] | None:
    if not occ:
        return (0, 0)
    r_ = rng.random()
    if r_ < 0.55:  # grow one of the mover's runs (slide to its end; sometimes gap)
        own = [c for c, o in occ.items() if o == owner] or list(occ)
        q, r = rng.choice(own)
        dq, dr = rng.choice(_DIRS6)
        while occ.get((q + dq, r + dr)) == owner:
            q, r = q + dq, r + dr
        step = rng.choice((1, 1, 1, 2))
        return (q + step * dq, r + step * dr)
    base = rng.choice(list(occ))
    rad = 2 if r_ < 0.9 else 4
    dq, dr = rng.randint(-rad, rad), rng.randint(-rad, rad)
    if _hexd(dq, dr) > rad or (dq, dr) == (0, 0):
        return None
    return (base[0] + dq, base[1] + dr)


def gen_position(rng: random.Random, plies: int) -> dict | None:
    occ: dict[tuple[int, int], int] = {}
    records = []
    for i in range(plies):
        owner = F.record_player(i)
        for _ in range(40):
            cell = _propose(rng, occ, owner)
            if cell is None or cell in occ:
                continue
            if _creates_win(occ, cell, owner):
                continue
            occ[cell] = owner
            records.append((cell[0], cell[1], owner, i))
            break
        else:
            break
    ply = len(records)
    if ply < 6:
        return None
    phase = F.record_phase(ply)
    facts = F.PositionFacts(
        records=tuple(records),
        current_player=F.record_player(ply),
        phase=phase,
        first_stone=(records[-1][0], records[-1][1]) if phase == F.PHASE_SECOND else None,
    )
    sup, feats = F.build_position(facts)
    raylen = F.build_ray_lengths(facts, sup)  # (N, 12) u8
    n = sup.num_nodes
    coords = np.empty((n, 2), dtype=np.int32)
    for cell, row in sup.index.items():
        coords[row] = cell

    # idx_taps (N, 6, 5): row of the cell k steps along direction d; -1 = absent.
    idx = np.full((n, 6, 5), -1, dtype=np.int32)
    for row in range(n):
        q, r = coords[row]
        for d, (dq, dr) in enumerate(_DIRS6):
            for k in range(1, 6):
                j = sup.index.get((q + k * dq, r + k * dr))
                if j is None:
                    break  # off-support stops the walk (matches ray gather geometry)
                idx[row, d, k - 1] = j

    # reach (N, 2, 6): raylen wire slot side*6 + axis*2 + dir per (side, tap).
    reach = np.empty((n, 2, 6), dtype=np.uint8)
    for t, (axis, direc) in enumerate(_tap_axis_dir()):
        for side in range(2):
            reach[:, side, t] = raylen[:, side * 6 + axis * 2 + direc]

    return {
        "x": feats[:, IN_PLANES].astype(np.float32),
        "xf": feats.astype(np.float32),  # full feature planes (distill probe)
        "y": feats[:, TARGET_LO:TARGET_HI].astype(np.float32),
        "idx": idx,
        "reach": reach,
        "records": tuple(records),
        "current_player": facts.current_player,
        "phase": phase,
        "legal_coords": [tuple(coords[i]) for i in range(sup.legal_count)],
    }


def gen_dataset(seed: int, count: int, max_plies: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        pos = gen_position(rng, rng.randint(8, max_plies))
        if pos is not None:
            out.append(pos)
    return out


def collate(batch: list[dict], device) -> tuple:
    npad = max(p["x"].shape[0] for p in batch)
    b = len(batch)
    x = np.zeros((b, npad, len(IN_PLANES)), dtype=np.float32)
    y = np.zeros((b, npad, 30), dtype=np.float32)
    mask = np.zeros((b, npad), dtype=np.float32)
    idx = np.full((b, npad, 6, 5), npad, dtype=np.int64)  # pad/absent -> zero row
    reach = np.zeros((b, npad, 2, 6), dtype=np.int64)
    for i, p in enumerate(batch):
        n = p["x"].shape[0]
        x[i, :n] = p["x"]
        y[i, :n] = p["y"]
        mask[i, :n] = 1.0
        pi = p["idx"].astype(np.int64)
        idx[i, :n] = np.where(pi < 0, npad, pi)
        reach[i, :n] = p["reach"]
    tt = lambda a: torch.from_numpy(a).to(device)
    return tt(x), tt(y), tt(mask), tt(idx), tt(reach)


# --- mechanisms -----------------------------------------------------------------


def gather_rays(x: torch.Tensor, idx: torch.Tensor, reach: torch.Tensor) -> torch.Tensor:
    """(B, Np, 6, 5, C) visibility-masked ray fibers; own vis on channels [:C/2],
    opp vis on [C/2:] (the orbit-half side split of the real net)."""
    b, np_, c = x.shape
    x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
    flat = idx.reshape(b, np_ * 30, 1).expand(-1, -1, c)
    xg = x_ext.gather(1, flat).reshape(b, np_, 6, 5, c)
    kvec = torch.arange(1, 6, device=x.device)
    vis = (kvec.view(1, 1, 1, 1, 5) <= reach.unsqueeze(-1)).to(x.dtype)  # (B,Np,2,6,5)
    half = c // 2
    return torch.cat(
        [xg[..., :half] * vis[:, :, 0].unsqueeze(-1), xg[..., half:] * vis[:, :, 1].unsqueeze(-1)],
        dim=-1,
    )


class RayConv(nn.Module):
    """One conv, both mechanisms: read build -> single Linear GEMM."""

    def __init__(self, c: int, mech: str) -> None:
        super().__init__()
        self.mech = mech
        if mech == "ray7":
            alpha = torch.zeros(5, c)
            alpha[0] = 1.0  # real-net init: reproduces the plain 7-tap conv
            self.alpha = nn.Parameter(alpha)
            self.lin = nn.Linear(7 * c, c)
        elif mech == "tap31":
            self.lin = nn.Linear(31 * c, c)
        else:
            raise ValueError(mech)

    def forward(self, x, idx, reach):
        b, np_, c = x.shape
        xg = gather_rays(x, idx, reach)
        if self.mech == "ray7":
            taps = (xg * self.alpha.view(1, 1, 1, 5, c)).sum(dim=3)  # (B,Np,6,C)
            read = torch.cat([x.unsqueeze(2), taps], dim=2).reshape(b, np_, 7 * c)
        else:
            read = torch.cat([x.unsqueeze(2), xg.reshape(b, np_, 6 * 5, c)], dim=2)
            read = read.reshape(b, np_, 31 * c)
        return self.lin(read)


class Block(nn.Module):
    def __init__(self, c: int, mech: str) -> None:
        super().__init__()
        self.ln1, self.conv1 = nn.LayerNorm(c), RayConv(c, mech)
        self.ln2, self.conv2 = nn.LayerNorm(c), RayConv(c, mech)

    def forward(self, x, idx, reach, mask):
        h = self.conv1(self.ln1(x), idx, reach).relu() * mask.unsqueeze(-1)
        h = self.conv2(self.ln2(h), idx, reach)
        return (x + h) * mask.unsqueeze(-1)


class ProbeNet(nn.Module):
    def __init__(self, c: int, mech: str, depth: int) -> None:
        super().__init__()
        self.stem = nn.Linear(len(IN_PLANES), c)
        self.depth = depth
        if depth == 1:  # pure linear: stem -> one conv -> head, no nonlinearity
            self.conv = RayConv(c, mech)
        else:
            self.blocks = nn.ModuleList(Block(c, mech) for _ in range(depth))
        self.head = nn.Linear(c, 30)

    def forward(self, x, idx, reach, mask):
        h = self.stem(x) * mask.unsqueeze(-1)
        if self.depth == 1:
            h = self.conv(h, idx, reach) * mask.unsqueeze(-1)
        else:
            for blk in self.blocks:
                h = blk(h, idx, reach, mask)
        return self.head(h)


# --- train / eval ---------------------------------------------------------------


def evaluate(net, val_batches, plane_var) -> dict:
    net.eval()
    se = torch.zeros(30)
    cnt = 0.0
    with torch.no_grad():
        for x, y, mask, idx, reach in val_batches:
            pred = net(x, idx, reach, mask)
            se += (((pred - y) ** 2) * mask.unsqueeze(-1)).sum(dim=(0, 1)).cpu()
            cnt += mask.sum().item()
    net.train()
    mse = se.numpy() / cnt
    r2 = 1.0 - mse / plane_var
    fam = {f: float(np.mean(r2[i * 6:(i + 1) * 6])) for i, f in enumerate(FAMILIES)}
    return {"macro_r2": float(np.mean(r2)), "per_family_r2": fam}


def run_arm(name, mech, c, depth, steps, pool, val_batches, plane_var, device, seed, log):
    torch.manual_seed(seed)
    net = ProbeNet(c, mech, depth).to(device)
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * (0.5 * (1 + math.cos(math.pi * s / steps))))
    rng = random.Random(seed)
    best, t0 = None, time.time()
    for step in range(steps):
        x, y, mask, idx, reach = (t.to(device) for t in pool[rng.randrange(len(pool))])
        pred = net(x, idx, reach, mask)
        loss = (((pred - y) ** 2) * mask.unsqueeze(-1)).sum() / (mask.sum() * 30)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 200 == 0 or step + 1 == steps:
            ev = evaluate(net, val_batches, plane_var)
            if best is None or ev["macro_r2"] > best["macro_r2"]:
                best = ev
            log(f"  {name} step {step+1}/{steps} loss {loss.item():.5f} "
                f"macroR2 {ev['macro_r2']:.4f} ({time.time()-t0:.0f}s)")
    return {"arm": name, "mech": mech, "C": c, "depth": depth, "params": params,
            "seed": seed, "steps": steps, "wall_s": round(time.time() - t0, 1), **best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-positions", type=int, default=2200)
    ap.add_argument("--val-positions", type=int, default=200)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=26)
    ap.add_argument("--arms", default=None, help="comma-separated arm names (default: all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    log = lambda m: print(m, flush=True)

    log(f"[gen] {args.train_positions}+{args.val_positions} positions ...")
    t0 = time.time()
    train_ds = gen_dataset(1234, args.train_positions, args.max_plies)
    val_ds = gen_dataset(9876, args.val_positions, args.max_plies)
    log(f"[gen] done in {time.time()-t0:.0f}s; "
        f"mean nodes {np.mean([p['x'].shape[0] for p in train_ds]):.0f}")

    # Held-out plane variance (the R^2 denominator) + degenerate-target check.
    ally = np.concatenate([p["y"] for p in val_ds], axis=0)
    plane_var = ally.var(axis=0) + 1e-9
    fam_var = {f: float(np.mean(plane_var[i * 6:(i + 1) * 6])) for i, f in enumerate(FAMILIES)}
    log(f"[targets] val plane variance by family: "
        f"{ {k: round(v, 5) for k, v in fam_var.items()} }")

    bs = args.batch
    val_batches = [collate(val_ds[i:i + bs], device) for i in range(0, len(val_ds), bs)]

    # Fixed size-bucketed batch pool (sorted by node count -> minimal padding),
    # pre-collated once on CPU; steps sample batches from it.
    order = sorted(range(len(train_ds)), key=lambda i: train_ds[i]["x"].shape[0])
    pool = [collate([train_ds[j] for j in order[i:i + bs]], "cpu")
            for i in range(0, len(order) - bs + 1, bs)]
    log(f"[pool] {len(pool)} pre-collated batches of {bs}")

    if device == "mps":  # numeric parity guard vs CPU before trusting MPS
        torch.manual_seed(0)
        net = ProbeNet(32, "tap31", 2)
        b_cpu = collate(train_ds[:4], "cpu")
        with torch.no_grad():
            ref = net(b_cpu[0], b_cpu[3], b_cpu[4], b_cpu[2])
            net_m = net.to("mps")
            b_m = collate(train_ds[:4], "mps")
            got = net_m(b_m[0], b_m[3], b_m[4], b_m[2]).cpu()
        diff = (ref - got).abs().max().item()
        if diff > 1e-3:
            log(f"[device] MPS parity FAILED (max diff {diff:.2e}) -> CPU")
            device = "cpu"
            val_batches = [collate(val_ds[i:i + bs], device) for i in range(0, len(val_ds), bs)]
        else:
            log(f"[device] MPS parity ok (max diff {diff:.2e})")
    log(f"[device] {device}")

    arms = [
        ("d1-ray7-C64",   "ray7", 64, 1, args.steps * 3 // 5),
        ("d1-tap31-C64",  "tap31", 64, 1, args.steps * 3 // 5),
        ("d2-ray7-C64",   "ray7", 64, 2, args.steps),
        ("d2-tap31-C32",  "tap31", 32, 2, args.steps),
        ("d2-tap31-C64",  "tap31", 64, 2, args.steps),
        ("d2-ray7-C128",  "ray7", 128, 2, args.steps),
    ]
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",")}
        unknown = keep - {a[0] for a in arms}
        assert not unknown, f"unknown arms: {unknown}"
        arms = [a for a in arms if a[0] in keep]
    out = args.out or f"/tmp/tap31_probe_seed{args.seed}.json"
    results = []
    for name, mech, c, depth, steps in arms:
        log(f"[arm] {name}")
        results.append(run_arm(name, mech, c, depth, steps, pool, val_batches,
                               plane_var, device, args.seed, log))
        Path(out + ".partial").write_text(json.dumps(  # survive a killed run
            {"seed": args.seed, "device": device, "fam_var": fam_var,
             "results": results}, indent=2))

    hdr = f"{'arm':<16}{'params':>9}{'macroR2':>9}" + "".join(f"{f:>9}" for f in FAMILIES)
    log("\n" + hdr)
    log("-" * len(hdr))
    for r in results:
        log(f"{r['arm']:<16}{r['params']:>9}{r['macro_r2']:>9.4f}"
            + "".join(f"{r['per_family_r2'][f]:>9.4f}" for f in FAMILIES))

    Path(out).write_text(json.dumps(
        {"seed": args.seed, "device": device, "fam_var": fam_var, "results": results}, indent=2))
    log(f"\n[out] {out}")


if __name__ == "__main__":
    main()
