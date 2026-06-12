"""Bradley-Terry rating fit over the main_4 anchored eval ladder.

Players: sealbot (pinned 0 Elo), m3ckpt5, m3ckpt10, and one player per main_4
eval epoch ("m4e15", ...). Edges: cached anchor results (fixed protocol
caveats noted) + every evaluation block found in main_4 diagnostics.

Usage: python _wf_r4_bt_rating.py
"""
import json
import math
from pathlib import Path

import numpy as np

RUN = Path("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4")
C5 = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3/checkpoints/epoch_000005.pt"
C10 = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3/checkpoints/epoch_000010.pt"

# Cached fixed edges: (player_a, player_b, wins_a, wins_b)
# - ckpt10 vs ckpt5: 96-game arena 2026-06-12 (no root temp — protocol differs
#   slightly from eval-phase games; small systematic, fine for trend use).
# - anchors vs sealbot: main_3 ep5/ep10 sealbot evals (128 games each).
EDGES = [
    ("m3c10", "m3c5", 72, 24),
    ("m3c5", "sealbot", 106, 22),
    ("m3c10", "sealbot", 106, 22),
]

for f in sorted(RUN.glob("diagnostics/epoch_*.json")):
    try:
        d = json.load(open(f))["metadata"]["result"]
    except Exception:
        continue
    ev = d.get("evaluation") or {}
    ep = ev.get("epoch")
    me = f"m4e{ep}"
    if ev.get("status") == "completed" and ev.get("games", 0) > 0:
        EDGES.append((me, "sealbot", ev["wins"], ev["losses"]))
    for ref in ev.get("references") or ([ev["reference"]] if ev.get("reference") else []):
        if not ref or ref.get("status") != "completed":
            continue
        opp = "m3c5" if "epoch_000005" in ref.get("checkpoint", "") else (
            "m3c10" if "epoch_000010" in ref.get("checkpoint", "") else ref.get("checkpoint"))
        EDGES.append((me, opp, ref["wins"], ref["losses"]))

players = sorted({p for e in EDGES for p in e[:2]})
idx = {p: i for i, p in enumerate(players)}
n = len(players)
r = np.zeros(n)  # log-strength; sealbot pinned at 0

def nll_grad(r):
    g = np.zeros(n)
    ll = 0.0
    for a, b, wa, wb in EDGES:
        ia, ib = idx[a], idx[b]
        d = r[ia] - r[ib]
        pa = 1.0 / (1.0 + math.exp(-d))
        ll -= wa * math.log(max(pa, 1e-12)) + wb * math.log(max(1 - pa, 1e-12))
        g[ia] -= wa - (wa + wb) * pa
        g[ib] += wa - (wa + wb) * pa
    return ll, g

for _ in range(500):
    ll, g = nll_grad(r)
    g[idx["sealbot"]] = 0.0
    r -= 0.002 * g
    r -= r[idx["sealbot"]]

# SEs from diagonal Fisher information (approximate, anchors treated as fixed)
fisher = np.zeros(n)
for a, b, wa, wb in EDGES:
    ia, ib = idx[a], idx[b]
    pa = 1.0 / (1.0 + math.exp(-(r[ia] - r[ib])))
    info = (wa + wb) * pa * (1 - pa)
    fisher[ia] += info
    fisher[ib] += info
se = np.where(fisher > 0, 1.0 / np.sqrt(np.maximum(fisher, 1e-9)), np.inf)

K = 400 / math.log(10)  # logit -> Elo
print(f"{'player':>10} {'elo':>8} {'+-95%':>7}   (sealbot pinned 0)")
for p in sorted(players, key=lambda p: -r[idx[p]]):
    print(f"{p:>10} {K*r[idx[p]]:8.0f} {1.96*K*se[idx[p]]:7.0f}")
print("\nedges used:")
for a, b, wa, wb in EDGES:
    print(f"  {a} {wa}-{wb} {b}")
