#!/usr/bin/env python3
"""CPU expressivity probe for ray-tap versus dense31 at depths one and two.

The target is the plan's broken-ray pattern: for at least one signed hex
direction, own stones at distances 1, 2, and 4 with an empty distance 3.
Synthetic inputs go through the repository's pure-Python featurizer.  Results
are emitted as a curve CSV and a compact final markdown table on stdout.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# A smaller but structurally identical regular fiber keeps all four CPU arms
# practical. These must be set before importing hexfield_eq constants/model.
os.environ.setdefault("HEXFIELD_EQ_CHANNELS", "48")
os.environ.setdefault("HEXFIELD_EQ_C_ORBIT", "4")

import torch
from torch import nn
from torch.nn import functional as F

from hexfield_eq import _raytap as RT
from hexfield_eq import equivariant as EQ
from hexfield_eq.batching import collate_rows
from hexfield_eq.constants import CHANNELS, DIRECTIONS, GROUP_ORDER, NUM_FEATURES
from hexfield_eq.features import PositionFacts, build_position, build_ray_lengths
from hexfield_eq.geometry import disk_offsets
from hexfield_eq.model import ConvBlock, EquivLinear, GroupAffineNorm, HexNodeConv


assert not torch.cuda.is_available(), "dense31 expressivity probe must run CPU-only"
torch.set_num_threads(8)


@dataclass
class Example:
    support: object
    feats: object
    raylen: object
    labels: torch.Tensor


def _make_example(seed: int) -> Example:
    rng = random.Random(seed)
    disk = disk_offsets(6)
    n_side = rng.randint(8, 24)
    cells = rng.sample(disk, 2 * n_side)
    records = tuple((q, r, i % 2, i) for i, (q, r) in enumerate(cells))
    facts = PositionFacts(
        records=records,
        current_player=seed % 2,
        phase="SecondStone",
        first_stone=cells[0],
    )
    support, feats = build_position(facts)
    raylen = build_ray_lengths(facts, support)
    owner = {(q, r): side for q, r, side, _ in records}
    own = facts.current_player
    labels = torch.zeros(support.num_nodes)
    for row, (q0, r0) in enumerate(support.coords[: support.legal_count]):
        for dq, dr in DIRECTIONS:  # all six signed axis directions
            values = [owner.get((int(q0) + k * dq, int(r0) + k * dr)) for k in range(1, 5)]
            if values[0] == own and values[1] == own and values[2] is None and values[3] == own:
                labels[row] = 1.0
                break
    return Example(support, feats, raylen, labels)


def _collate(examples: list[Example]):
    batch = collate_rows(
        [(ex.support, ex.feats) for ex in examples],
        raylen=[ex.raylen for ex in examples],
    )
    target = torch.zeros_like(batch["mask"], dtype=torch.float32)
    legal = torch.zeros_like(batch["mask"])
    for i, ex in enumerate(examples):
        count = ex.support.legal_count
        target[i, : ex.support.num_nodes] = ex.labels
        legal[i, :count] = True
    batch["target"] = target
    batch["legal"] = legal
    return batch


class ProbeNet(nn.Module):
    def __init__(self, mode: str, depth: int) -> None:
        super().__init__()
        self.mode = mode
        self.stem = HexNodeConv(NUM_FEATURES, CHANNELS)
        self.stem_ln = GroupAffineNorm(CHANNELS)
        self.blocks = nn.ModuleList([ConvBlock(CHANNELS, raytap=mode) for _ in range(depth)])
        # A tied 1x1 map to one regular orbit, followed by the invariant slot mean.
        self.read = EquivLinear(CHANNELS, GROUP_ORDER)

    def forward(self, batch) -> torch.Tensor:
        feats, nbr, mask, coords = (
            batch["feats"], batch["nbr"], batch["mask"], batch["coords"]
        )
        b, n = mask.shape
        self_idx = torch.arange(n).view(1, n, 1).expand(b, -1, -1)
        gather_idx = torch.cat([self_idx, nbr], dim=2)
        x = F.relu(self.stem_ln(self.stem(feats, gather_idx, mask)))
        ray_ctx = SimpleNamespace(
            idx_taps=RT.build_tap_gather_index(coords, mask),
            reach=RT.build_tap_reach(batch["raylen"]),
            ray_idx=None,
            raylen=batch["raylen"],
        )
        for block in self.blocks:
            x = block(x, gather_idx, mask, ray_ctx=ray_ctx)
        return EQ.group_pool(self.read(x)).squeeze(-1)


def _auc(logits: torch.Tensor, target: torch.Tensor) -> float:
    pos = target > 0.5
    p = int(pos.sum())
    q = int((~pos).sum())
    if p == 0 or q == 0:
        return float("nan")
    order = torch.argsort(logits)  # ascending ranks
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, logits.numel() + 1, dtype=torch.float64)
    return float((ranks[pos].sum() - p * (p + 1) / 2) / (p * q))


@torch.no_grad()
def _evaluate(model: nn.Module, examples: list[Example], batch_size: int) -> tuple[float, float]:
    model.eval()
    losses, logits_all, target_all = [], [], []
    for start in range(0, len(examples), batch_size):
        batch = _collate(examples[start : start + batch_size])
        logits = model(batch)
        select = batch["legal"]
        picked_logits = logits[select]
        picked_target = batch["target"][select]
        losses.append(F.binary_cross_entropy_with_logits(picked_logits, picked_target, reduction="sum"))
        logits_all.append(picked_logits)
        target_all.append(picked_target)
    logits = torch.cat(logits_all)
    target = torch.cat(target_all)
    return float(sum(losses) / target.numel()), _auc(logits, target)


def _train_arm(
    name: str,
    mode: str,
    depth: int,
    train: list[Example],
    val: list[Example],
    schedule: list[list[int]],
    args,
) -> tuple[list[dict], int, float]:
    torch.manual_seed(100 + depth)
    model = ProbeNet(mode, depth)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rows: list[dict] = []
    started = time.perf_counter()
    last_train = float("nan")
    for step in range(args.steps + 1):
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            bce, auc = _evaluate(model, val, args.batch_size)
            rows.append(
                dict(arm=name, step=step, val_bce=bce, val_auc=auc,
                     train_bce=last_train, params=params,
                     elapsed_s=time.perf_counter() - started)
            )
            print(f"{name} step={step:4d} val_bce={bce:.5f} auc={auc:.5f}", flush=True)
        if step == args.steps or time.perf_counter() - started >= args.max_seconds:
            break
        model.train()
        batch = _collate([train[i] for i in schedule[step]])
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(
            logits[batch["legal"]], batch["target"][batch["legal"]]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        frac = (step + 1) / args.steps
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        last_train = float(loss.detach())
    return rows, params, time.perf_counter() - started


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-positions", type=int, default=64)
    parser.add_argument("--val-positions", type=int, default=24)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/DENSE31_EXPRESSIVITY_CURVES.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    examples = [_make_example(10_000 + i) for i in range(args.train_positions + args.val_positions)]
    train = examples[: args.train_positions]
    val = examples[args.train_positions :]
    positives = sum(int(ex.labels[: ex.support.legal_count].sum()) for ex in examples)
    legal = sum(ex.support.legal_count for ex in examples)
    print(f"dataset legal={legal} positive={positives} rate={positives/legal:.6f}")
    rng = random.Random(20260710)
    schedule = [
        [rng.randrange(len(train)) for _ in range(args.batch_size)]
        for _ in range(args.steps)
    ]
    all_rows = []
    arms = (("P1-raytap-d1", "both", 1), ("P2-dense31-d1", "dense31", 1),
            ("P3-raytap-d2", "both", 2), ("P4-dense31-d2", "dense31", 2))
    summaries = []
    for name, mode, depth in arms:
        rows, params, elapsed = _train_arm(name, mode, depth, train, val, schedule, args)
        all_rows.extend(rows)
        summaries.append((name, params, rows[-1]["step"], rows[-1]["val_bce"], rows[-1]["val_auc"], elapsed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print("\n| arm | params | steps | val BCE | val AUC | seconds |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, params, steps, bce, auc, elapsed in summaries:
        print(f"| {name} | {params:,} | {steps} | {bce:.5f} | {auc:.5f} | {elapsed:.1f} |")
    print(f"\ncurves: {args.output}")


if __name__ == "__main__":
    main()
