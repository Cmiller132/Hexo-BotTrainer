"""Phase 1: current exploration level from REAL recent shards (epoch 41, model e41).

For each sampled completed game: replay it, and at each recorded (searched) ply read
the actual MCTS visit policy from the shard and the actually-played move from the .hxr.
Compute, per ply-band:
  - realized visit entropy + top-1 mass (how spread / sharp the search is)
  - played-move deviation rate (played != top-visit move)
  - value cost of deviations via 1-ply value lookahead on the LIVE checkpoint
  - blunder proxy P(dV>0.1), P(dV>0.3)
  - analytical temperature sweep over the SAME visit distributions (T in 0/0.3/0.6/1.0
    and the live curve's temp at that ply) -> E[dV], P(dV>thr), sampling deviation prob.
Also dumps (v_model, z) pairs for value calibration (Phase 3).
"""
import sys, json, glob, math, statistics as st
from pathlib import Path
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/_expl")
import common as C
from hexo_utils.records import HexoRecordFile
from hexo_models.dense_cnn.compact_io import read_compact_shard

EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 41
N_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
TOPK = 8
SWEEP_T = [0.3, 0.6, 1.0]   # T=0 (greedy) is trivial: dev=0, dV=0

net, meta = C.load_model(EPOCH)
ev = C.Evaluator(net, n=4)

hxr = f"{C.SP_DIR}/epoch_{EPOCH:06d}.hxr"
recs = list(HexoRecordFile.open(hxr).iter_records())[:N_GAMES]

def reweight(pairs, T):
    inv = 1.0 / T
    z = [(a, max(w, 0.0) ** inv) for a, w in pairs]
    tot = sum(w for _, w in z)
    return [(a, (w / tot if tot > 0 else 0.0)) for a, w in z]

# accumulators per band
acc = {nm: {"vH": [], "top1": [], "ncand": [], "dev": [], "dev_dV": [],
            "all_p1": [], "all_p3": [], "curve_temp": [],
            "dev_flip": [],  # realized: played move's value sign opposite to top-visit
            "sw": {T: {"dev": [], "ev": [], "p1": [], "p3": [], "H": [], "flip": []} for T in SWEEP_T},
            "curve": {"dev": [], "ev": [], "p1": [], "p3": [], "flip": []}}
       for nm, _, _ in C.BANDS}
vz = []  # (v_model, z) calibration
npos = 0

for rec in recs:
    idx = C.shard_index_from_game_id(rec.game_id)
    shp = glob.glob(f"{C.SP_DIR}/epoch_{EPOCH:06d}_game_{idx:06d}.npz")
    if not shp:
        continue
    rows = read_compact_shard(Path(shp[0]))
    pol_by_turn = {}
    z_by_turn = {}
    for r in rows:
        t = int(r.turn_index)
        pol_by_turn[t] = [(int(a), float(w)) for a, w in r.policy if float(w) > 0]
        z_by_turn[t] = float(r.value)
    actions = [int(a) for a in rec.action_ids]
    s = C.new_game(seed=int(rec.seed))
    for t, played in enumerate(actions):
        if C.engine.terminal(s) is not None:
            break
        pol = pol_by_turn.get(t)
        if pol and len(pol) >= 2:
            tot = sum(w for _, w in pol)
            pol = [(a, w / tot) for a, w in pol]
            pol.sort(key=lambda x: -x[1])
            top_visit = pol[0][0]
            bnd = C.band_of(t)
            A = acc[bnd]
            A["vH"].append(C.entropy([w for _, w in pol]))
            A["top1"].append(pol[0][1])
            A["ncand"].append(len(pol))
            A["curve_temp"].append(C.cur_temp(t))
            # 1-ply value lookahead for topk candidates (self perspective = -value(child))
            topk = pol[:TOPK]
            vlook = {}
            for a, _w in topk:
                try:
                    s2 = C.clone(s)
                    C.apply(s2, a)
                    vlook[a] = -ev.value(s2)
                except Exception:
                    vlook[a] = None
            if vlook.get(top_visit) is None:
                C.apply(s, played); continue
            vtop = vlook[top_visit]
            dV = {a: (vtop - vlook[a]) for a, _w in topk if vlook.get(a) is not None}
            # realized played-move deviation
            dev = (played != top_visit)
            A["dev"].append(1.0 if dev else 0.0)
            if dev and played in dV:
                A["dev_dV"].append(dV[played])
                # outcome-flip proxy: played move's 1-ply value sign opposite to top
                if vtop * vlook[played] < 0:
                    A["dev_flip"].append(1.0)
                else:
                    A["dev_flip"].append(0.0)
            def _flip(a):
                return 1.0 if (vtop * vlook[a] < 0) else 0.0
            # blunder proxy over ALL topk weighted by visit prob (data the model trains on)
            wsum = sum(w for a, w in topk if a in dV)
            if wsum > 0:
                A["all_p1"].append(sum(w for a, w in topk if dV.get(a, 0) > 0.1) / wsum)
                A["all_p3"].append(sum(w for a, w in topk if dV.get(a, 0) > 0.3) / wsum)
            # analytical temperature sweep over the visit distribution (topk renormalized)
            base = [(a, w) for a, w in topk if a in dV]
            for T in SWEEP_T + ["curve"]:
                tt = C.cur_temp(t) if T == "curve" else T
                pT = reweight(base, tt)
                ev_dV = sum(p * dV[a] for a, p in pT)
                p1 = sum(p for a, p in pT if dV[a] > 0.1)
                p3 = sum(p for a, p in pT if dV[a] > 0.3)
                # P(sampled move != top-visit) under this temperature
                pdev = 1.0 - sum(p for a, p in pT if a == top_visit)
                H = C.entropy([p for _, p in pT])
                pflip = sum(p * _flip(a) for a, p in pT)  # E[outcome-flip] under this temp
                if T == "curve":
                    A["curve"]["dev"].append(pdev); A["curve"]["ev"].append(ev_dV)
                    A["curve"]["p1"].append(p1); A["curve"]["p3"].append(p3)
                    A["curve"]["flip"].append(pflip)
                else:
                    d = A["sw"][T]
                    d["dev"].append(pdev); d["ev"].append(ev_dV)
                    d["p1"].append(p1); d["p3"].append(p3); d["H"].append(H); d["flip"].append(pflip)
            # calibration: model value at this position vs z label
            vz.append((ev.value(s), z_by_turn.get(t, float("nan"))))
            npos += 1
        C.apply(s, played)

mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")

def corr(pairs):
    xs = [a for a, b in pairs if b == b]
    ys = [b for a, b in pairs if b == b]
    if len(xs) < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs)); dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")

out = {"epoch": EPOCH, "model_epoch": EPOCH, "games": len(recs), "positions": npos,
       "value_calibration_corr_vz": corr(vz), "n_vz": len(vz), "bands": {}}
print(f"\n=== PHASE 1: hexgnn_rl_main1 epoch {EPOCH} (model e{EPOCH}) | {npos} searched positions over {len(recs)} games ===")
print(f"value calibration corr(v,z) = {out['value_calibration_corr_vz']:+.3f}  (n={len(vz)})\n")
for nm, lo, hi in C.BANDS:
    A = acc[nm]
    n = len(A["dev"])
    if n == 0:
        continue
    rec_band = {
        "n": n,
        "mean_curve_temp": mean(A["curve_temp"]),
        "realized_visit_entropy": mean(A["vH"]),
        "realized_top1_mass": mean(A["top1"]),
        "mean_candidates": mean(A["ncand"]),
        "played_dev_rate": mean(A["dev"]),
        "mean_dV_of_deviations": mean(A["dev_dV"]),
        "n_deviations": len(A["dev_dV"]),
        "realized_dev_flip_rate": mean(A["dev_flip"]),   # of deviations, frac that flip value sign
        "blunder_all_p1": mean(A["all_p1"]),
        "blunder_all_p3": mean(A["all_p3"]),
        "curve_pred": {k: mean(A["curve"][k]) for k in ("dev", "ev", "p1", "p3", "flip")},
        "sweep": {T: {k: mean(A["sw"][T][k]) for k in ("dev", "ev", "p1", "p3", "H", "flip")} for T in SWEEP_T},
    }
    out["bands"][nm] = rec_band
    print(f"--- ply band {nm} (n={n}, live-curve temp~{rec_band['mean_curve_temp']:.2f}) ---")
    print(f"  realized: visitH={rec_band['realized_visit_entropy']:.2f} top1={rec_band['realized_top1_mass']:.2f} "
          f"ncand={rec_band['mean_candidates']:.1f}")
    print(f"  PLAYED-move deviation rate = {rec_band['played_dev_rate']:.0%}  "
          f"mean valΔ of deviations = {rec_band['mean_dV_of_deviations']:+.3f} (n_dev={rec_band['n_deviations']})  "
          f"value-sign-flip among deviations = {rec_band['realized_dev_flip_rate']:.0%}")
    print(f"  blunder proxy over searched policy mass: P(dV>0.1)={rec_band['blunder_all_p1']:.2f} P(dV>0.3)={rec_band['blunder_all_p3']:.2f}")
    print(f"  temp |  P(deviate) | E[valΔ] | P(Δ>0.1) | P(Δ>0.3) | P(flip) | samp-H")
    for T in SWEEP_T:
        d = rec_band["sweep"][T]
        print(f"  {T:.1f}  |    {d['dev']:.2f}     |  {d['ev']:+.3f} |   {d['p1']:.2f}   |   {d['p3']:.2f}   |  {d['flip']:.3f}  |  {d['H']:.2f}")
    c = rec_band["curve_pred"]
    print(f"  curve({rec_band['mean_curve_temp']:.2f}) | {c['dev']:.2f} | {c['ev']:+.3f} | {c['p1']:.2f} | {c['p3']:.2f} | flip={c['flip']:.3f}\n")

Path("/mnt/e/Hexo-BotTrainer-hexgt/_expl/out").mkdir(exist_ok=True)
with open("/mnt/e/Hexo-BotTrainer-hexgt/_expl/out/phase1.json", "w") as f:
    json.dump(out, f, indent=2)
with open("/mnt/e/Hexo-BotTrainer-hexgt/_expl/out/phase1_vz.json", "w") as f:
    json.dump(vz, f)
print("wrote _expl/out/phase1.json")
