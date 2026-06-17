#!/usr/bin/env python3
"""STAGE 1 (read-only): measure REALIZED exploration in dense_cnn_restnet_main1
self-play from the persisted per-game .npz shards + .json sidecars. No model, no
GPU, no tree search -- pure analysis of what the run actually produced.

Metrics (per ply-band + per epoch):
  - visit-policy entropy of the (pruned) training target, top-1 mass, effective
    move count exp(H), policy support size vs legal-move count.
  - played-move deviation: fraction of plies where the actually-played move
    (recovered from placement history) != argmax of the visit policy.
  - opening diversity: unique / entropy of 1st,2nd,3rd moves across games.
  - game length distribution + decisiveness (|value| at terminal-ish rows).
  - policy-surprise reweighting activity (from json: surprise mean, freq weight,
    raw_rows vs num_rows drop rate).
"""
from __future__ import annotations
import json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from dense_cnn_restnet.compact_io import read_compact_shard
from hexo_engine.types import unpack_coord_id

RUN = Path("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1")
SP = RUN / "selfplay"
EPOCHS = [int(x) for x in sys.argv[1:]] or [27, 28, 29, 30, 31]

def H(ws):
    ws = [w for w in ws if w > 0]
    return -sum(w * math.log(w) for w in ws) if ws else 0.0

def band(t):
    if t < 10: return "00-09"
    if t < 20: return "10-19"
    if t < 40: return "20-39"
    if t < 70: return "40-69"
    return "70+"

def hist_coord(h, k):
    return (int(h.coords[2 * k]), int(h.coords[2 * k + 1]))

agg = {}  # epoch -> dict
for ep in EPOCHS:
    npzs = sorted(SP.glob(f"epoch_{ep:06d}_game_*.npz"))
    if not npzs:
        continue
    lengths, decis_val, p0win = [], [], 0
    firsts, seconds, thirds, first3 = Counter(), Counter(), Counter(), Counter()
    band_rows = defaultdict(lambda: dict(H=[], top1=[], eff=[], supp=[], legal=[],
                                          dev=0, n=0))
    # json-side reweighting stats
    surprise, freqw, dropfrac = [], [], []
    ngames = 0
    for npz in npzs:
        try:
            rows = read_compact_shard(npz)
        except Exception:
            continue
        if not rows:
            continue
        ngames += 1
        js = npz.with_suffix(".json")
        if js.exists():
            try:
                j = json.load(open(js))
                surprise.append(j.get("policy_surprise_mean"))
                freqw.append(j.get("frequency_weight_mean"))
                rr, nr = j.get("raw_rows"), j.get("num_rows")
                if rr and nr is not None and rr > 0:
                    dropfrac.append(1.0 - nr / rr)
            except Exception:
                pass
        deep = max(rows, key=lambda r: len(r.placement_history))
        L = len(deep.placement_history)
        lengths.append(L)
        v = float(deep.value)
        decis_val.append(abs(v))
        # move sequence m_k = deep.placement_history[k]
        seq = [hist_coord(deep.placement_history, k) for k in range(L)]
        if L >= 1: firsts[seq[0]] += 1
        if L >= 2: seconds[seq[1]] += 1
        if L >= 3:
            thirds[seq[2]] += 1
            first3[(seq[0], seq[1], seq[2])] += 1
        for r in rows:
            if not r.policy:
                continue
            ws = [w for _, w in r.policy]
            tot = sum(ws)
            if tot <= 0:
                continue
            ws = [w / tot for w in ws]
            h = H(ws)
            b = band(r.turn_index)
            br = band_rows[b]
            br["H"].append(h)
            br["top1"].append(max(ws))
            br["eff"].append(math.exp(h))
            br["supp"].append(sum(1 for w in ws if w > 1e-6))
            br["legal"].append(len(r.legal_action_ids))
            br["n"] += 1
            t = r.turn_index
            if t < L:
                played = seq[t]
                # argmax of visit policy
                best_a = max(r.policy, key=lambda aw: aw[1])[0]
                ba = unpack_coord_id(best_a)
                if (int(ba.q), int(ba.r)) != played:
                    br["dev"] += 1
    if ngames == 0:
        continue
    def first_player_winrate():
        # value is from current player's perspective at the deepest (latest) row;
        # sign of game result for p0 is harder without player parity -> skip.
        return None
    agg[ep] = dict(
        ngames=ngames,
        len_mean=float(np.mean(lengths)), len_med=float(np.median(lengths)),
        len_p10=float(np.percentile(lengths, 10)), len_p90=float(np.percentile(lengths, 90)),
        len_min=int(min(lengths)), len_max=int(max(lengths)),
        decisive_gt09=float(np.mean([1 if d > 0.9 else 0 for d in decis_val])),
        decisive_gt05=float(np.mean([1 if d > 0.5 else 0 for d in decis_val])),
        absval_mean=float(np.mean(decis_val)),
        uniq_first=len(firsts), uniq_second=len(seconds), uniq_third=len(thirds),
        ent_first=H([c / sum(firsts.values()) for c in firsts.values()]) if firsts else 0,
        ent_second=H([c / sum(seconds.values()) for c in seconds.values()]) if seconds else 0,
        ent_third=H([c / sum(thirds.values()) for c in thirds.values()]) if thirds else 0,
        uniq_first3=len(first3),
        top_second=seconds.most_common(5),
        surprise_mean=float(np.mean([s for s in surprise if s is not None])) if surprise else None,
        freqw_mean=float(np.mean([f for f in freqw if f is not None])) if freqw else None,
        drop_frac=float(np.mean(dropfrac)) if dropfrac else None,
        bands={},
    )
    for b, br in sorted(band_rows.items()):
        if br["n"] == 0:
            continue
        agg[ep]["bands"][b] = dict(
            n=br["n"],
            H=round(float(np.mean(br["H"])), 4),
            top1=round(float(np.mean(br["top1"])), 4),
            eff=round(float(np.mean(br["eff"])), 2),
            supp=round(float(np.mean(br["supp"])), 2),
            legal=round(float(np.mean(br["legal"])), 1),
            dev_rate=round(br["dev"] / br["n"], 4),
        )

print(json.dumps(agg, indent=1, default=str))
