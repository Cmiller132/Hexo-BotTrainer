"""Adversarial verifier re-derivations (no NN). CPU, read-only.

1. Recompute dawdle-probe headline stats from RAW per-candidate evals in
   _wf_s3d_probe_out.json (ignoring the stored per_k/summary blocks).
2. Full-shard npz moves_left stats for main_3 ep5/ep9 (all rows, all shards)
   + run-constant (ml + turn_index) analysis vs .hxr decisive game lengths
   (resolves the 'plies vs decisions' / off-by-one unit question).
"""
import json
import glob
from pathlib import Path

import numpy as np

SCRIPTS = Path("/mnt/e/Hexo-BotTrainer-hexgt/scripts")
RUNS = Path("/mnt/e/Hexo-BotTrainer/runs")
CAP = 512.0

print("== 1. dawdle probe recompute from raw cands ==")
d = json.load(open(SCRIPTS / "_wf_s3d_probe_out.json"))
pos = d["positions"]
sq = [p for p in pos if p["tag"] == "squander"]
ct = [p for p in pos if p["tag"] != "squander"]
print(f"n_squander={len(sq)} n_other={len(ct)} tags={sorted(set(p['tag'] for p in pos))}")

KS = [0.05, 0.1, 0.2, 0.5, 1.0]


def recompute(p):
    v = np.array([c["v_leader"] for c in p["cands"]])
    ml = np.array([c["ml"] for c in p["cands"]])
    sgn = 1.0 if p["v_leader_base"] >= 0 else -1.0
    iv = int(np.argmax(v))
    out = {}
    for k in KS:
        u = v - k * sgn * ml / CAP
        out[k] = int(np.argmax(u)) != iv
    # exact critical k: min positive k where some j overtakes iv
    crit = np.inf
    for j in range(len(v)):
        if j == iv:
            continue
        dml = sgn * (ml[iv] - ml[j]) / CAP
        dv = v[iv] - v[j]
        if dml > 1e-12:
            kj = dv / dml
            if kj >= 0:
                crit = min(crit, kj)
    # near-v set spread
    near = v >= v[iv] - 0.05
    spread = float(ml[near].max() - ml[near].min()) if near.sum() >= 2 else 0.0
    # played above-median ml among near-v candidates
    played = next((i for i, c in enumerate(p["cands"]) if c["is_played"]), None)
    above = None
    if played is not None and near[played] and near.sum() >= 2:
        above = bool(ml[played] > np.median(ml[near]))
    return out, crit, spread, above, near.sum()


flip = {k: [0, 0] for k in KS}  # squander [count, n]; controls separately
flip_ct = {k: [0, 0] for k in KS}
crits, spreads, aboves = [], [], []
games_flipped_02, games_flipped_05 = set(), set()
for p in sq:
    out, crit, spread, above, _ = recompute(p)
    for k in KS:
        flip[k][1] += 1
        flip[k][0] += int(out[k])
    if out[0.2]:
        games_flipped_02.add(p["game_id"])
    if out[0.5]:
        games_flipped_05.add(p["game_id"])
    crits.append(crit)
    spreads.append(spread)
    if above is not None:
        aboves.append(above)
for p in ct:
    out, crit, spread, above, _ = recompute(p)
    for k in KS:
        flip_ct[k][1] += 1
        flip_ct[k][0] += int(out[k])

print("squander flip rates:", {k: f"{c}/{n} ({c/n:.3f})" for k, (c, n) in flip.items()})
print("control flip rates:", {k: f"{c}/{n} ({c/n:.3f})" for k, (c, n) in flip_ct.items()})
print(f"games hit at k=0.2: {len(games_flipped_02)}/8, k=0.5: {len(games_flipped_05)}/8")
fin = [c for c in crits if np.isfinite(c)]
print(f"critical-k: median(all, inf-aware)={np.median(crits):.3f} "
      f"n_finite={len(fin)}/{len(crits)} median_finite={np.median(fin):.3f}")
print(f"dawdle spread (near-v): median={np.median(spreads):.2f} "
      f"IQR={np.percentile(spreads,25):.2f}-{np.percentile(spreads,75):.2f} max={max(spreads):.2f}")
print(f"played above-median ml among near-v: {sum(aboves)}/{len(aboves)}")
lw = [p["leader_won"] for p in sq]
print(f"leader_won at squander points: {sum(lw)}/{len(lw)}")
# parity structure
flip_par, keep_par, bad = 0, 0, 0
for p in pos:
    vb = p["v_leader_base"]
    for c in p["cands"]:
        if c["terminal"] or c["stm_next"] is None:
            continue
        if c["stm_next"] == p["stm"]:  # mid-chain (same player to move)
            keep_par += 1
            if np.sign(c["v_leader"]) != np.sign(vb):
                bad += 1
print(f"mid-chain successors keeping sign(v_base): {keep_par - bad}/{keep_par}")
nterm = sum(p["n_terminal_cands"] for p in pos)
print(f"terminal candidates across all positions: {nterm}")
# v-gap and near-set medians for squander
vgaps, vranges, nnear = [], [], []
for p in sq:
    v = np.array([c["v_leader"] for c in p["cands"]])
    sv = np.sort(v)[::-1]
    vgaps.append(sv[0] - sv[1] if len(sv) > 1 else 0.0)
    vranges.append(sv[0] - sv[-1])
    nnear.append(int((v >= sv[0] - 0.05).sum()))
print(f"squander v-gap best-to-2nd median={np.median(vgaps):.4f} "
      f"v-range median={np.median(vranges):.4f} n_near median={np.median(nnear):.1f}")

print()
print("== 2. npz full-shard moves_left stats + unit analysis ==")
import sys
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python")
from hexo_utils.records import HexoRecordFile

# decisive game lengths per selfplay epoch (union for window tolerance)
len_by_ep = {}
for ep in range(1, 11):
    p = RUNS / "dense_cnn_restnet_main_3" / "selfplay" / f"epoch_{ep:06d}.hxr"
    if not p.exists():
        continue
    with HexoRecordFile.open(p) as rf:
        len_by_ep[ep] = np.array([len(r.action_ids) for r in rf.iter_records() if r.winner is not None])

for ep in (5, 9):
    paths = sorted(glob.glob(
        f"/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3/shuffleddata/*epoch_{ep:06d}/train/*.npz"))
    print(f"main_3 ep{ep}: {len(paths)} shards")
    mls, tis = [], []
    for p in paths:
        z = np.load(p)
        mls.append(np.asarray(z["moves_left"], dtype=np.float64))
        tis.append(np.asarray(z["turn_index"], dtype=np.float64))
    ml = np.concatenate(mls)
    ti = np.concatenate(tis)
    masked = ml < 0
    print(f"  rows={len(ml)} frac_masked={masked.mean():.4f} "
          f"mean_unmasked={ml[~masked].mean():.1f} max={ml.max():.0f} "
          f"frac_unmasked>512={(ml[~masked] > 512).mean():.4f} "
          f"turn_index max={ti.max():.0f}")
    s = (ml + ti)[~masked]
    lens_own = len_by_ep.get(ep, np.array([]))
    union = np.unique(np.concatenate(list(len_by_ep.values())))
    fr_L_own = np.isin(s, lens_own).mean()           # sum == L  (ml = L - index)
    fr_Lm1_own = np.isin(s + 1, lens_own).mean()     # sum == L-1 (ml = L - index - 1, current code)
    fr_L_un = np.isin(s, union).mean()
    fr_Lm1_un = np.isin(s + 1, union).mean()
    print(f"  ml+turn_index: max={s.max():.0f}; own-epoch lens max={lens_own.max() if len(lens_own) else 'NA'}")
    print(f"  frac sum==some L (own ep): {fr_L_own:.4f}; frac sum==some L-1 (own ep): {fr_Lm1_own:.4f}")
    print(f"  frac sum==some L (union eps): {fr_L_un:.4f}; frac sum==some L-1 (union): {fr_Lm1_un:.4f}")
    # distribution check: every game contributes its decisions; how many distinct sums vs lens
    print(f"  distinct sums={len(np.unique(s))} distinct own-ep lens={len(np.unique(lens_own))}")

print()
print("== 3. diagnostics loss spot-check ==")
for run, ep in (("dense_cnn_restnet_main_3", 9), ("dense_cnn_restnet_main_4", 12)):
    p = RUNS / run / "diagnostics" / f"epoch_{ep:06d}.json"
    d = json.load(open(p))
    v = d["metadata"]["result"]["training"]["loss_components"]["moves_left"]
    print(f"{run} ep{ep} moves_left loss: {v:.4f}")
print("DONE")
