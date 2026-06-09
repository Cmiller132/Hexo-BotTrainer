"""Measure hexgt's candidate-count distribution at n=3 over recorded 96x8
positions (read-only), by game phase, to derive the Dirichlet total_alpha."""
from __future__ import annotations
import sys, glob, random
import numpy as np
from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.expand import reconstruct_state
from hexo_models.hexgt import rust_bridge

N = 3
shard_glob = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay/epoch_0000*_game_*.npz"
shards = sorted(glob.glob(shard_glob))
rng = random.Random(0)
rng.shuffle(shards)
counts = []
phase_counts = {"opening": [], "midgame": [], "endgame": []}
positions = 0
for sp in shards:
    if positions >= 1200:
        break
    try:
        rows = read_compact_shard(sp)
    except Exception:
        continue
    if not rows:
        continue
    # sample a few rows per shard across the game
    idxs = sorted(rng.sample(range(len(rows)), min(4, len(rows))))
    for i in idxs:
        row = rows[i]
        try:
            state = reconstruct_state(row.placement_history)
            cand = rust_bridge.candidate_ids(state, N)
            c = len(cand)
        except Exception:
            continue
        counts.append(c)
        ti = int(row.turn_index)
        bucket = "opening" if ti < 15 else ("midgame" if ti < 60 else "endgame")
        phase_counts[bucket].append(c)
        positions += 1

a = np.array(counts, dtype=np.float64)
print(f"n={N} candidate-count distribution over {len(a)} recorded 96x8 positions:")
for p in (5, 25, 50, 75, 95, 99):
    print(f"  p{p:02d} = {np.percentile(a, p):.0f}")
print(f"  mean = {a.mean():.1f}  min = {a.min():.0f}  max = {a.max():.0f}")
for ph, lst in phase_counts.items():
    if lst:
        arr = np.array(lst)
        print(f"  {ph:8s}: n={len(arr):4d} median={np.median(arr):.0f} p95={np.percentile(arr,95):.0f}")
med = np.median(a)
print(f"\nDERIVED Dirichlet total_alpha for per-candidate alpha targets:")
for target in (0.03, 0.04, 0.05):
    print(f"  per-cand alpha={target} -> total_alpha = {target*med:.1f} (at median {med:.0f})")
print(f"  (KataGo/dense_cnn rule: alpha_i = total_alpha/count; dense_cnn total_alpha=10.83)")
print(f"  dense_cnn-10.83 would give hexgt per-cand alpha = {10.83/med:.3f} at median {med:.0f}")
