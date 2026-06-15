"""Supplementary analysis over _wf_s3d_probe_out.json + _wf_s3d_midgame_out.json.

Computes (pure JSON+numpy, no torch):
  A. per-position critical k (smallest k at which U-argmax departs v-argmax
     toward lower predicted ml), distributions per cohort;
  B. v-gap structure (best minus 2nd-best, best minus median candidate);
  C. played-move direction: was the played move an ml-dawdler among near-v
     candidates; rank shifts under U;
  D. terminal-win availability at squander points;
  E. near-v window sensitivity (spread within 0.01 as well as 0.05).
"""
import json
import numpy as np

CAP = 512.0
SCRIPTS = r"E:\Hexo-BotTrainer-hexgt\scripts"

probe = json.load(open(SCRIPTS + r"\_wf_s3d_probe_out.json"))
mid = json.load(open(SCRIPTS + r"\_wf_s3d_midgame_out.json"))
positions = probe["positions"] + mid["positions"]


def kcrit(pos):
    """Smallest k>0 flipping argmax_U away from argmax_v toward lower ml; inf if none."""
    v = np.array([c["v_leader"] for c in pos["cands"]])
    ml = np.array([c["ml"] for c in pos["cands"]]) / CAP
    sgn = 1.0 if pos["v_leader_base"] >= 0 else -1.0
    iv = int(np.argmax(v))
    best = np.inf
    for j in range(len(v)):
        if j == iv:
            continue
        d_ml = sgn * (ml[iv] - ml[j])          # >0 when j predicts fewer ml (sooner end for winner)
        if d_ml <= 1e-12:
            continue
        k_needed = (v[iv] - v[j]) / d_ml       # v[iv]-v[j] >= 0 by argmax
        if k_needed < best:
            best = k_needed
    return float(best)


def vgaps(pos):
    v = np.sort([c["v_leader"] for c in pos["cands"]])[::-1]
    g2 = float(v[0] - v[1]) if len(v) > 1 else None
    gm = float(v[0] - np.median(v))
    return g2, gm


def near_spread(pos, window):
    v = np.array([c["v_leader"] for c in pos["cands"]])
    ml = np.array([c["ml"] for c in pos["cands"]])
    near = v >= v.max() - window
    if near.sum() < 2:
        return 0.0, int(near.sum())
    return float(ml[near].max() - ml[near].min()), int(near.sum())


def played_dawdle_flags(pos):
    """Among near-v(0.05) candidates: is played move's ml above the median?"""
    v = np.array([c["v_leader"] for c in pos["cands"]])
    ml = np.array([c["ml"] for c in pos["cands"]])
    near = v >= v.max() - 0.05
    pi = next((i for i, c in enumerate(pos["cands"]) if c["is_played"]), None)
    if pi is None or not near[pi] or near.sum() < 3:
        return None
    med = float(np.median(ml[near]))
    return bool(ml[pi] > med)


def agg(rows, name):
    ks = np.array([kcrit(r) for r in rows])
    finite = ks[np.isfinite(ks)]
    g2 = np.array([vgaps(r)[0] for r in rows], dtype=float)
    gm = np.array([vgaps(r)[1] for r in rows], dtype=float)
    s001 = [near_spread(r, 0.01) for r in rows]
    s005 = [near_spread(r, 0.05) for r in rows]
    flags = [played_dawdle_flags(r) for r in rows]
    flags = [f for f in flags if f is not None]
    nterm = [r.get("n_terminal_cands", 0) for r in rows]
    out = {
        "n": len(rows),
        "n_games": len({r.get("game_id") for r in rows}),
        "kcrit_pcts_[10,25,50,75,90]": [round(float(np.percentile(ks, p)), 4) if np.isfinite(np.percentile(ks, p)) else None
                                         for p in (10, 25, 50, 75, 90)],
        "kcrit_finite_frac": round(float(np.mean(np.isfinite(ks))), 3),
        "flip_frac_at_k": {str(k): round(float(np.mean(ks <= k)), 3)
                           for k in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)},
        "vgap_best_minus_2nd_median": round(float(np.median(g2)), 5),
        "vgap_best_minus_median_median": round(float(np.median(gm)), 5),
        "spread_ml_w005_median": round(float(np.median([s for s, _ in s005])), 2),
        "spread_ml_w001_median": round(float(np.median([s for s, _ in s001])), 2),
        "n_near_w001_median": float(np.median([n for _, n in s001])),
        "played_above_median_ml_near": (f"{sum(flags)}/{len(flags)}" if flags else "n/a"),
        "terminal_cand_positions": int(sum(1 for t in nterm if t > 0)),
    }
    print(name, json.dumps(out, indent=1))
    return out


res = {}
for tag in ("squander", "control", "midgame"):
    rows = [r for r in positions if r["tag"] == tag]
    res[tag] = agg(rows, tag)

# squander by ply bucket: kcrit medians
sq = [r for r in positions if r["tag"] == "squander"]
for lo, hi in [(200, 260), (260, 300), (300, 340), (340, 400)]:
    rows = [r for r in sq if lo <= r["ply"] < hi]
    if rows:
        ks = np.array([kcrit(r) for r in rows])
        res.setdefault("squander_by_ply", {})[f"[{lo},{hi})"] = {
            "n": len(rows),
            "kcrit_median": round(float(np.median(ks)), 4) if np.isfinite(np.median(ks)) else None,
            "flip_frac_k02": round(float(np.mean(ks <= 0.2)), 3),
        }

# Played-rank shift at k=0.2 for squander (question 2 strict form)
pr_v = [r["played_rank_v"] for r in sq if r["played_rank_v"] is not None]
pr_u = [r["per_k"]["0.2"]["played_rank_u"] for r in sq if r["per_k"]["0.2"]["played_rank_u"] is not None]
res["squander_played_rank"] = {
    "rank_v_mean": round(float(np.mean(pr_v)), 2),
    "rank_u_k02_mean": round(float(np.mean(pr_u)), 2),
    "n": len(pr_v),
    "rank_worsens_under_u_k02": int(sum(1 for a, b in zip(pr_v, pr_u) if b > a)),
    "rank_improves_under_u_k02": int(sum(1 for a, b in zip(pr_v, pr_u) if b < a)),
}

# spot check 2 raw candidate tables (first squander flip at k<=0.1, one control)
flips01 = [r for r in sq if r["per_k"]["0.1"]["argmax_changed"]]
spot = []
for r in (flips01[:2] + [x for x in positions if x["tag"] == "control"][:1]):
    tbl = sorted(r["cands"], key=lambda c: -c["v_leader"])[:6]
    spot.append({
        "tag": r["tag"], "game": r.get("game_id"), "ply": r["ply"],
        "v_base": round(r["v_leader_base"], 3), "ml_base": round(r["ml_base"], 1),
        "top6": [{"v": round(c["v_leader"], 4), "ml": round(c["ml"], 1),
                  "prior": round(c["prior"], 3), "played": c["is_played"],
                  "term": c["terminal"]} for c in tbl],
    })
res["spot_checks"] = spot
print("SPOT", json.dumps(spot, indent=1))

json.dump(res, open(SCRIPTS + r"\_wf_s3d_kcrit_out.json", "w"), indent=1)
print("DONE")
