#!/usr/bin/env python
"""P1' — ray-tap discrimination gate (SPEC_RAYTAP_CONV.md §6.2, work item W-P1).

PASS CRITERIA (fixed BEFORE the first run of this script; do not tune after):
    GATE_MARGIN     = 0.10   # ray-tap eval accuracy must beat the depth-1
                             # 1-ring baseline by >= 10 percentage points
    GATE_RAY_FLOOR  = 0.90   # and reach >= 90% eval accuracy outright
External simulation evidence for this task class: baseline ~75% (its Bayes
ceiling given distance-1 visibility is ~77% under the sampling distribution
below), ray variants ~100%. The gate is retained as a reproducible in-tree
artifact; a FAIL stops Phase R (spec §9.3 step 1).

The task (one-axis contiguity class, positions-not-counts): an 11-cell line
with the center cell empty; every other cell iid own/opp/empty. Label =
"placing at the center yields a contiguous own run >= 4 through the center".
The label depends on WHERE the stones sit (contiguity), not on how many are in
reach: a depth-1 1-ring conv sees only the distance-1 cells and is capped near
77%, while a single ray-tap layer (per-channel per-distance alpha over
visibility-masked ray aggregates, the §2.2 mechanism reduced to one axis)
exposes visible prefix counts o_j = sum_{k<=j} own_k, whose fixed-point
condition o_j = j decides run >= j exactly, so the task is representable at
depth 1.

Both models share the identical trunk shape (1x1 input lift -> one
3-tap conv [center, +1, -1] -> pointwise 2-layer MLP head) and training budget;
they differ ONLY in what the direction taps consume: the distance-1 neighbour
(baseline) vs the visibility-masked alpha-weighted ray aggregate initialized to
alpha = (1, 0, 0, 0, 0) (ray-tap; init-equivalent to the baseline, spec §2.2).
Visibility follows the ray_lengths_for_cell convention: the walk stops at AND
INCLUDES the first anti-side stone, passes through own-side stones and empties,
stops off-board; own-half channels use the own-side raylen, opp-half channels
the opp-side (the §2.2 orbit-half split reduced to channel halves).

Seeded, CPU-only, no repo imports (self-contained by design so the artifact
reproduces anywhere). Prints one JSON line and exits 0 on PASS / 1 on FAIL.

Run:  python scripts/raytap_p1_gate.py
"""

from __future__ import annotations

import json
import sys

import torch
from torch import nn

# --- gate constants (FIXED before first run; see module docstring) ---------------
GATE_MARGIN = 0.10
GATE_RAY_FLOOR = 0.90

SEED = 20260709
N_CELLS = 11            # line cells 0..10
CENTER = 5              # query cell (always empty pre-placement)
REACH = 5               # ray reach (WINDOW_LEN - 1)
RUN_TARGET = 4          # label: contiguous own run through center >= 4
P_OWN, P_OPP = 0.60, 0.15   # per-cell occupancy prior (rest empty)
N_TRAIN, N_EVAL = 4096, 4096
CHANNELS = 16           # trunk width (even: own/opp visibility halves)
HIDDEN = 32             # pointwise head hidden width
STEPS = 600
LR = 3e-2
BATCH = 512


def make_dataset(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """feats (n, N_CELLS, 3) one-hot [own, opp, empty]; labels (n,) float."""

    u = torch.rand(n, N_CELLS, generator=gen)
    occ = torch.full((n, N_CELLS), 2, dtype=torch.long)  # 2 = empty
    occ[u < P_OWN] = 0                                   # 0 = own
    occ[(u >= P_OWN) & (u < P_OWN + P_OPP)] = 1          # 1 = opp
    occ[:, CENTER] = 2                                   # center empty pre-placement
    feats = torch.nn.functional.one_hot(occ, 3).float()

    own = occ == 0
    # Contiguous own run through the center after placing own there.
    run = torch.ones(n)
    for j in range(1, REACH + 1):  # right arm
        alive = own[:, CENTER + 1 : CENTER + 1 + j].all(dim=1)
        run += alive.float()
    for j in range(1, REACH + 1):  # left arm
        alive = own[:, CENTER - j : CENTER].all(dim=1)
        run += alive.float()
    labels = (run >= RUN_TARGET).float()
    return feats, labels


def ray_lengths(occ_own: torch.Tensor, occ_opp: torch.Tensor) -> torch.Tensor:
    """(n, N_CELLS, 2 sides, 2 dirs) uint8 — the ray_lengths_for_cell walk on a
    line: from cell i, direction d, side s: step j = 1..REACH; off-board stops;
    an anti-side stone is included (terminal blocker) then stops; own-side
    stones and empties pass through. Side 0 (own) blocks on opp; side 1 (opp)
    blocks on own."""

    n, m = occ_own.shape
    out = torch.zeros(n, m, 2, 2, dtype=torch.uint8)
    anti = (occ_opp, occ_own)  # anti-side plane per side
    for i in range(m):
        for di, sgn in ((0, 1), (1, -1)):
            for s in range(2):
                length = torch.zeros(n, dtype=torch.uint8)
                blocked = torch.zeros(n, dtype=torch.bool)
                for j in range(1, REACH + 1):
                    y = i + sgn * j
                    if y < 0 or y >= m:
                        break
                    step = ~blocked
                    length = torch.where(step, torch.full_like(length, j), length)
                    blocked = blocked | (step & anti[s][:, y].bool())
                out[:, i, s, di] = length
    return out


class PointwiseHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CHANNELS, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (n, C) -> (n,)
        return self.net(x).squeeze(-1)


class Baseline1Ring(nn.Module):
    """lift -> 3-tap conv (center, +1, -1 neighbours) -> pointwise head at the
    center cell. Depth 1: reach is exactly one cell each side."""

    def __init__(self) -> None:
        super().__init__()
        self.lift = nn.Linear(3, CHANNELS)
        self.w0 = nn.Linear(CHANNELS, CHANNELS)
        self.wp = nn.Linear(CHANNELS, CHANNELS, bias=False)
        self.wm = nn.Linear(CHANNELS, CHANNELS, bias=False)
        self.head = PointwiseHead()

    def forward(self, feats: torch.Tensor, raylen: torch.Tensor) -> torch.Tensor:
        x = self.lift(feats)  # (n, N_CELLS, C)
        agg = (
            self.w0(x[:, CENTER])
            + self.wp(x[:, CENTER + 1])
            + self.wm(x[:, CENTER - 1])
        )
        return self.head(torch.relu(agg))


class RayTap1Layer(nn.Module):
    """Same trunk with the direction taps consuming the §2.2 ray aggregate:
    in_d[c] = sum_k alpha[k, c] * 1[k <= raylen_{s(c)}(center, d)] * x_{d,k}[c],
    alpha init (1, 0, 0, 0, 0) per channel (init-equivalent to the baseline)."""

    def __init__(self) -> None:
        super().__init__()
        self.lift = nn.Linear(3, CHANNELS)
        self.w0 = nn.Linear(CHANNELS, CHANNELS)
        self.wp = nn.Linear(CHANNELS, CHANNELS, bias=False)
        self.wm = nn.Linear(CHANNELS, CHANNELS, bias=False)
        alpha = torch.zeros(REACH, CHANNELS)
        alpha[0] = 1.0
        self.alpha = nn.Parameter(alpha)
        self.head = PointwiseHead()
        half = CHANNELS // 2
        # side of channel c: own visibility for [0, C/2), opp for [C/2, C)
        self.register_buffer(
            "side", (torch.arange(CHANNELS) >= half).long(), persistent=False
        )

    def _tap(self, x: torch.Tensor, raylen: torch.Tensor, di: int, sgn: int) -> torch.Tensor:
        n = x.shape[0]
        agg = torch.zeros(n, CHANNELS)
        rl = raylen[:, CENTER, :, di].float()          # (n, 2 sides)
        rl_c = rl[:, self.side]                        # (n, C) per-channel reach
        for k in range(1, REACH + 1):
            y = CENTER + sgn * k
            if y < 0 or y >= N_CELLS:
                continue
            vis = (rl_c >= k).float()                  # (n, C)
            agg = agg + self.alpha[k - 1] * vis * x[:, y]
        return agg

    def forward(self, feats: torch.Tensor, raylen: torch.Tensor) -> torch.Tensor:
        x = self.lift(feats)
        agg = (
            self.w0(x[:, CENTER])
            + self.wp(self._tap(x, raylen, 0, +1))
            + self.wm(self._tap(x, raylen, 1, -1))
        )
        return self.head(torch.relu(agg))


def train_and_eval(model: nn.Module, data: dict) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    n = data["train_x"].shape[0]
    gen = torch.Generator().manual_seed(SEED + 1)
    model.train()
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        logits = model(data["train_x"][idx], data["train_rl"][idx])
        loss = loss_fn(logits, data["train_y"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = (model(data["eval_x"], data["eval_rl"]) > 0).float()
    return float((pred == data["eval_y"]).float().mean())


def main() -> int:
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    train_x, train_y = make_dataset(N_TRAIN, gen)
    eval_x, eval_y = make_dataset(N_EVAL, gen)
    data = {
        "train_x": train_x, "train_y": train_y,
        "train_rl": ray_lengths(train_x[..., 0], train_x[..., 1]),
        "eval_x": eval_x, "eval_y": eval_y,
        "eval_rl": ray_lengths(eval_x[..., 0], eval_x[..., 1]),
    }

    torch.manual_seed(SEED + 10)
    base_acc = train_and_eval(Baseline1Ring(), data)
    torch.manual_seed(SEED + 10)
    ray_acc = train_and_eval(RayTap1Layer(), data)

    passed = (ray_acc - base_acc >= GATE_MARGIN) and (ray_acc >= GATE_RAY_FLOOR)
    print(json.dumps({
        "gate": "P1'",
        "baseline_acc": round(base_acc, 4),
        "raytap_acc": round(ray_acc, 4),
        "margin": round(ray_acc - base_acc, 4),
        "gate_margin": GATE_MARGIN,
        "gate_ray_floor": GATE_RAY_FLOOR,
        "positive_rate_eval": round(float(eval_y.mean()), 4),
        "pass": passed,
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
