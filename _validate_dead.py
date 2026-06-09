"""Validate the strict dead-cell candidate rule on real dense_cnn 96x8 positions.

For each position: filtered candidate set (new build) vs the radius-3 (B) cells,
and the dense_cnn visit policy. Reports, by phase:
  - candidate count (filtered) vs the unfiltered radius-3 component -> the shrink.
  - DROPPED cells (radius-3 empties removed by the dead filter) and the dense_cnn
    VISIT-MASS that landed on them (must be ~0 -> the filter drops no real move).
  - coverage of dense_cnn visit mass by the filtered set vs the Phase-1 baseline.
Read-only.
"""
from __future__ import annotations
import glob, random
from collections import defaultdict
import numpy as np
from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.expand import reconstruct_state
from hexo_models.hexgt import rust_bridge
from hexo_engine.types import unpack_coord_id

SH = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay/epoch_0000*_game_*.npz"
N = 3


def hexdist(a, b):
    dq, dr = a[0] - b[0], a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def radius_cells(c, n):
    out = []
    for dq in range(-n, n + 1):
        for dr in range(max(-n, -dq - n), min(n, -dq + n) + 1):
            out.append((c[0] + dq, c[1] + dr))
    return out


def phase(t):
    return "opening" if t < 15 else ("mid" if t < 60 else "end")


def main():
    shards = sorted(glob.glob(SH))
    rng = random.Random(0); rng.shuffle(shards)
    by = defaultdict(lambda: dict(filt=[], unf=[], dropped=[], drop_mass=[], cov_mass=[], played_in=0, played_n=0))
    pos = 0
    for sp in shards:
        if pos >= 900:
            break
        try:
            rows = read_compact_shard(sp)
        except Exception:
            continue
        idxs = sorted(rng.sample(range(len(rows)), min(4, len(rows))))
        for i in idxs:
            row = rows[i]
            try:
                state = reconstruct_state(row.placement_history)
                filt = set(int(a) for a in rust_bridge.candidate_ids(state, N))  # action ids
            except Exception:
                continue
            filt_qr = set((int(unpack_coord_id(a).q), int(unpack_coord_id(a).r)) for a in filt)
            stones = set((int(e[0]), int(e[1])) for e in row.placement_history)
            B = set()
            for s in stones:
                for c in radius_cells(s, N):
                    if c not in stones:
                        B.add(c)
            dropped = B - filt_qr  # radius-3 empties removed (dead cells)
            # dense_cnn visit policy mass on dropped vs total + coverage by filtered
            total = sum(float(w) for _a, w in row.policy) or 1.0
            drop_mass = sum(float(w) for a, w in row.policy
                            if (int(unpack_coord_id(int(a)).q), int(unpack_coord_id(int(a)).r)) in dropped)
            cov_mass = sum(float(w) for a, w in row.policy if int(a) in filt)
            played = int(max(row.policy, key=lambda kv: kv[1])[0]) if row.policy else None
            d = by[phase(int(row.turn_index))]
            d["filt"].append(len(filt)); d["unf"].append(len(B)); d["dropped"].append(len(dropped))
            d["drop_mass"].append(drop_mass / total); d["cov_mass"].append(cov_mass / total)
            if played is not None:
                d["played_n"] += 1
                if played in filt:
                    d["played_in"] += 1
            pos += 1

    print(f"### Dead-cell rule validation — {pos} dense_cnn 96x8 positions, n={N}\n")
    print(f"{'phase':8} {'filt_cands':>10} {'radius3(B)':>10} {'dropped':>8} | "
          f"{'dropped dc-mass':>15} {'filt cov-mass':>13} {'played-in-set':>14}")
    allf, alldm, allcov, pin, pn = [], [], [], 0, 0
    for ph in ("opening", "mid", "end"):
        d = by[ph]
        if not d["filt"]:
            continue
        f = np.median(d["filt"]); u = np.median(d["unf"]); dr = np.mean(d["dropped"])
        dm = np.mean(d["drop_mass"]); cv = np.mean(d["cov_mass"])
        pi = d["played_in"] / max(1, d["played_n"])
        print(f"{ph:8} {f:10.0f} {u:10.0f} {dr:8.1f} | {dm:15.2%} {cv:13.2%} {pi:14.1%}")
        allf += d["filt"]; alldm += d["drop_mass"]; allcov += d["cov_mass"]; pin += d["played_in"]; pn += d["played_n"]
    print(f"\nOVERALL: filtered candidate median={np.median(allf):.0f} (Phase-1 unfiltered n=3 median ~220)")
    print(f"  dense_cnn visit-mass on DROPPED (dead) cells: mean={np.mean(alldm):.3%}  max={np.max(alldm):.2%}")
    print(f"  dense_cnn visit-mass COVERED by filtered set: mean={np.mean(allcov):.2%} (Phase-1 vmass ~90.2%)")
    print(f"  played-move (argmax) in filtered set: {pin}/{pn} = {pin/max(1,pn):.2%}")
    med = float(np.median(allf))
    print(f"\nDERIVED Dirichlet total_alpha (alpha_i=0.03 target): {0.03*med:.1f} (was 6.6 at median 220)")


if __name__ == "__main__":
    main()
