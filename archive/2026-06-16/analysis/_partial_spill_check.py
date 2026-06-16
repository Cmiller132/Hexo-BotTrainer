"""Quantify the PARTIAL-spill policy corruption on real self-play shards.

The 41x41 square crop is not D6-closed. expand_sample raises (-> identity fallback)
only on TOTAL spill. PARTIAL spill (some policy actions rotate out of crop) is silent:
dropped actions + renormalized survivors. This measures how often that actually happens
by counting nonzero policy cells at identity vs each symmetry on REAL data.
"""
import glob
import os

import numpy as np

from hexo_models.dense_cnn import compact_io as cio
from hexo_models.dense_cnn import d6
from hexo_models.dense_cnn import samples as S

RUN = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8"
SP = os.path.join(RUN, "selfplay")
D6N = d6.D6_SIZE
shards = sorted(glob.glob(os.path.join(SP, "epoch_*_game_*.npz")))[-40:]

pairs = 0                 # (sample, non-id symmetry) attempts that expanded (no total spill)
total_spill = 0           # raised -> identity fallback (already handled)
partial_pairs = 0         # expanded but dropped >=1 policy action (SILENT corruption)
dropped_frac_sum = 0.0
samples_any_partial = 0
samples_scanned = 0
# bucket by game phase (turn fraction) to see if late-game is worse
late_partial = late_pairs = 0

for path in shards:
    try:
        rows = cio.read_compact_shard(path)
    except Exception:  # noqa: BLE001
        continue
    nmax = max((int(s.turn_index) for s in rows), default=1) + 1
    for s in rows[:25]:
        try:
            base = S.expand_sample(s, symmetry=0)
        except Exception:  # noqa: BLE001
            continue
        samples_scanned += 1
        n0 = int((base["policy"].numpy() > 0).sum())
        if n0 == 0:
            continue
        late = (int(s.turn_index) / nmax) > 0.66
        any_partial = False
        for k in range(1, D6N):
            try:
                e = S.expand_sample(s, symmetry=k)
            except ValueError:
                total_spill += 1
                continue
            pairs += 1
            if late:
                late_pairs += 1
            nk = int((e["policy"].numpy() > 0).sum())
            if nk < n0:
                partial_pairs += 1
                dropped_frac_sum += (n0 - nk) / n0
                any_partial = True
                if late:
                    late_partial += 1
        if any_partial:
            samples_any_partial += 1


def pct(a, b):
    return f"{100.0 * a / max(1, b):.2f}%"


print(f"shards={len(shards)}  samples_scanned={samples_scanned}")
print(f"symmetry expand attempts (non-id, no total-spill): {pairs}")
print(f"TOTAL-spill (already handled by identity fallback): {total_spill}")
print()
print(f"PARTIAL-spill pairs (SILENT policy corruption):     {partial_pairs} = {pct(partial_pairs, pairs)} of expand attempts")
print(f"  mean fraction of policy actions dropped when it happens: {pct(dropped_frac_sum, partial_pairs)}")
print(f"samples with >=1 partial-spill symmetry:            {samples_any_partial} = {pct(samples_any_partial, samples_scanned)} of samples")
print()
print(f"LATE-game (turn>66%) partial-spill rate: {pct(late_partial, late_pairs)}  (vs overall {pct(partial_pairs, pairs)})")
print("\nInterpretation: each affected row is expanded at a uniform-random symmetry every")
print("epoch, so ~(partial-rate) of an affected row's epochs train on a truncated policy target.")
