"""hexgnn_rl_main1 exploration analysis (CPU, read-only).

Phases:
  1  Realized exploration level per ply band (0-20 / 20-50 / 50+): played-move
     deviation from visit-argmax, search/temp entropy, value cost of deviations
     (1-ply value lookahead via the live e41 checkpoint), blunder proxy ΔV>0.1/0.3,
     and an outcome-flip proxy (top1 winning -> played losing).
  2a TEMPERATURE sweep (post-hoc on the stored visit policies): expected deviation,
     E[ΔV], P(Δ>0.1/0.3), sampling entropy vs T in {0,0.3,0.6,1.0, ply-temp}.
  3  Cross-epoch stats: opening diversity (unique 1st/2nd/3rd moves + entropy),
     game-length distribution/trend, value calibration corr(v,z) via live ckpt,
     temperature-induced z-label-noise estimate.

Uses the same methodology as the prior hexgt _temp_cost.py, adapted to hexgnn
(GNN value via HexgnnInference.evaluate_states, CPU). Outputs JSON + printed tables.
"""
import sys, glob, re, math, json, random, statistics as st
from collections import Counter
from pathlib import Path
RNG = random.Random(20260606)

for _p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models"):
    sys.path.insert(0, f"packages/{_p}/python")
sys.path.insert(0, "packages/hexgnn/python")

import numpy as np
import torch
torch.set_grad_enabled(False)
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 6)

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexgnn.architecture import HexgnnNetwork
from hexgnn.inference import HexgnnInference
from hexo_models.dense_cnn.compact_io import read_compact_shard

RUN = "runs/hexgnn_rl_main1"
CKPT_EP = 41
N = 4                                   # current live candidate radius
TOPK = 8
GAMES_DEEP = int(sys.argv[1]) if len(sys.argv) > 1 else 50   # games/epoch for value-lookahead phases
DEEP_EPOCHS = [36, 38, 40, 41]          # epochs for the per-band value analysis
OPEN_EPOCHS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 38, 40, 41]  # opening/length trend
BANDS = [("0-20", 0, 20), ("20-50", 20, 50), ("50+", 50, 10**9)]
# live move-temperature curve: temp = floor + (init-floor)*2^(-ply/halflife)
TEMP_INIT, TEMP_FLOOR, TEMP_HALFLIFE = 1.0, 0.3, 33.0
def cur_temp(ply): return TEMP_FLOOR + (TEMP_INIT - TEMP_FLOOR) * 2.0 ** (-ply / TEMP_HALFLIFE)
SWEEP_T = [0.0, 0.3, 0.6, 1.0]

def band_of(t):
    for nm, lo, hi in BANDS:
        if lo <= t < hi:
            return nm
    return "50+"

# ---- model / value lookahead -------------------------------------------------
ck = torch.load(f"{RUN}/checkpoints/hexgnn_rl_epoch{CKPT_EP:06d}.pt", map_location="cpu", weights_only=False)
a = ck["arch"]
net = HexgnnNetwork(token_dim=a["token_dim"], gnn_layers=a["gnn_layers"],
                    attention_heads=a.get("attention_heads", 4),
                    value_pma_seeds=a.get("value_pma_seeds", 2),
                    value_head_use_side=a.get("value_head_use_side", True),
                    steerable_channels=a.get("steerable_channels", 0))
net.load_state_dict(ck["model"]); net.eval()
INF = HexgnnInference(net, device="cpu", fp16=False)

def values_of_states(states):
    """Batched value (side-to-move perspective) for a list of states."""
    if not states:
        return []
    out = []
    B = 48
    for i in range(0, len(states), B):
        res = INF.evaluate_states(states[i:i+B], n=N)
        out.extend(float(r["value"]) for r in res)
    return out

def records_for_epoch(ep):
    hxr = f"{RUN}/selfplay/epoch_{ep:06d}.hxr"
    if not glob.glob(hxr):
        return []
    return list(HexoRecordFile.open(hxr).iter_records())

def shard_rows(ep, idx):
    shp = f"{RUN}/selfplay/epoch_{ep:06d}_game_{idx:06d}.npz"
    if not glob.glob(shp):
        return None
    try:
        return read_compact_shard(Path(shp))
    except Exception:
        return None

def game_index(rec):
    m = re.search(r"selfplay-(\d+)$", rec.game_id)
    return int(m.group(1)) if m else None

def ent(weights):
    s = sum(weights)
    if s <= 0:
        return 0.0
    return -sum((w/s) * math.log(w/s) for w in weights if w > 0)

# ============================ PHASE 1 + 2a ====================================
# acc[band][T] for predicted sweep; realB[band] for realized played-move stats.
realB = {b[0]: {"n": 0, "dev": 0, "dV": [], "p1": 0, "p3": 0, "flip": 0,
                "searchH": [], "tempH": [], "top1": [], "ndev_val": 0,
                "vz_v": [], "vz_z": []} for b in BANDS}
sweep = {b[0]: {T: {"dev": [], "ev": [], "p1": [], "p3": [], "H": []} for T in SWEEP_T} for b in BANDS}
npos_deep = 0

for ep in DEEP_EPOCHS:
    recs = records_for_epoch(ep)
    if not recs:
        continue
    # .hxr records are in FINISH order (short games first) -> random-sample to avoid
    # a length bias that would starve the mid/late ply bands.
    recs = RNG.sample(recs, min(GAMES_DEEP, len(recs)))
    for rec in recs:
        idx = game_index(rec)
        if idx is None:
            continue
        rows = shard_rows(ep, idx)
        if not rows:
            continue
        pol_by_turn = {int(r.turn_index): list(r.policy) for r in rows}
        z_by_turn = {int(r.turn_index): float(r.value) for r in rows}
        coords = [unpack_coord_id(int(c)) for c in rec.action_ids]
        s = eng.new_game()
        for t, c in enumerate(coords):
            if eng.terminal(s) is not None:
                break
            if t in pol_by_turn:
                pol = [(int(act), float(w)) for act, w in pol_by_turn[t] if float(w) > 0]
                tot = sum(w for _, w in pol)
                if pol and tot > 0 and len(pol) >= 2:
                    pol.sort(key=lambda x: -x[1])
                    pv = [(act, w/tot) for act, w in pol]
                    topk = pv[:TOPK]
                    top1 = topk[0][0]
                    played = int(rec.action_ids[t])
                    bnd = band_of(t)
                    full_w = [w for _, w in pv]
                    # value of position (for v,z calibration) + 1-ply lookahead values
                    base_res = INF.evaluate_states([s], n=N)[0]
                    v_pos = float(base_res["value"])
                    move_states = {}
                    for act, _w in topk:
                        cc = unpack_coord_id(act); s2 = eng.clone_state(s)
                        try:
                            eng.apply_action(s2, PlacementAction(AxialCoord(q=int(cc.q), r=int(cc.r))))
                            move_states[act] = s2
                        except Exception:
                            move_states[act] = None
                    if played not in move_states:  # played may be outside top-k
                        cc = unpack_coord_id(played); s2 = eng.clone_state(s)
                        try:
                            eng.apply_action(s2, PlacementAction(AxialCoord(q=int(cc.q), r=int(cc.r))))
                            move_states[played] = s2
                        except Exception:
                            move_states[played] = None
                    keys = [k for k, v in move_states.items() if v is not None]
                    vals = values_of_states([move_states[k] for k in keys])
                    vmap = {k: -vv for k, vv in zip(keys, vals)}   # value to the mover
                    if top1 not in vmap:
                        eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r)))); continue
                    dV = {k: (vmap[top1] - vmap[k]) for k in vmap}
                    # ----- realized played-move stats -----
                    R = realB[bnd]
                    R["n"] += 1
                    R["searchH"].append(ent(full_w))
                    # temp-applied entropy at the ply's live temperature
                    Tcur = cur_temp(t)
                    if Tcur > 0:
                        z = [w ** (1.0/Tcur) for w in full_w]; zt = sum(z)
                        R["tempH"].append(-sum((zz/zt)*math.log(zz/zt) for zz in z if zz > 0))
                    R["top1"].append(pv[0][1])
                    R["vz_v"].append(v_pos); R["vz_z"].append(z_by_turn.get(t, 0.0))
                    if played != top1:
                        R["dev"] += 1
                        if played in dV:
                            d = dV[played]; R["dV"].append(d); R["ndev_val"] += 1
                            if d > 0.1: R["p1"] += 1
                            if d > 0.3: R["p3"] += 1
                            if vmap.get(top1, 0) > 0.25 and vmap.get(played, 0) < -0.25:
                                R["flip"] += 1
                    # ----- predicted temperature sweep -----
                    for T in SWEEP_T:
                        if T == 0.0:
                            pT = [(pv[0][0], 1.0)]
                        else:
                            inv = 1.0/T
                            kz = [(act, w ** inv) for act, w in topk if act in dV]
                            kzt = sum(zz for _, zz in kz)
                            pT = [(act, zz/kzt) for act, zz in kz]
                        ev = sum(p*dV[act] for act, p in pT if act in dV)
                        p1 = sum(p for act, p in pT if dV.get(act, 0) > 0.1)
                        p3 = sum(p for act, p in pT if dV.get(act, 0) > 0.3)
                        dev = 1.0 - sum(p for act, p in pT if act == top1)
                        # full-set sampling entropy
                        if T == 0.0:
                            H = 0.0
                        else:
                            fz = [w ** (1.0/T) for w in full_w]; fzt = sum(fz)
                            H = -sum((q/fzt)*math.log(q/fzt) for q in fz if q > 0)
                        A = sweep[bnd][T]
                        A["dev"].append(dev); A["ev"].append(ev); A["p1"].append(p1); A["p3"].append(p3); A["H"].append(H)
                    npos_deep += 1
            eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))

def mean(xs): return st.mean(xs) if xs else float("nan")
def corr(xs, ys):
    if len(xs) < 3: return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs)); dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx > 0 and dy > 0 else float("nan")

report = {"config": {"ckpt_epoch": CKPT_EP, "n": N, "temp": {"init": TEMP_INIT, "floor": TEMP_FLOOR, "halflife": TEMP_HALFLIFE},
                     "deep_epochs": DEEP_EPOCHS, "games_per_epoch": GAMES_DEEP, "positions": npos_deep}}

print(f"\n===== PHASE 1: REALIZED EXPLORATION (epochs {DEEP_EPOCHS}, {npos_deep} positions, e{CKPT_EP} value head) =====")
print(f"{'band':6} | {'n':5} | temp@band | dev% | meanΔV(dev) | P(ΔV>.1) | P(ΔV>.3) | flip% | searchH | tempH | top1visit | corr(v,z)")
p1res = {}
for nm, lo, hi in BANDS:
    R = realB[nm]
    if R["n"] == 0: continue
    rep_ply = {"0-20": 10, "20-50": 33, "50+": 70}[nm]
    nd = max(1, R["ndev_val"])
    row = {
        "n": R["n"], "temp_at_band": round(cur_temp(rep_ply), 3),
        "dev_rate": R["dev"]/R["n"], "mean_dV_dev": mean(R["dV"]),
        "P_dV_gt_0.1": R["p1"]/nd, "P_dV_gt_0.3": R["p3"]/nd, "flip_rate_of_dev": R["flip"]/nd,
        "search_entropy": mean(R["searchH"]), "temp_entropy": mean(R["tempH"]),
        "top1_visit": mean(R["top1"]), "corr_v_z": corr(R["vz_v"], R["vz_z"]),
    }
    p1res[nm] = row
    print(f"{nm:6} | {R['n']:5} | {row['temp_at_band']:8} | {row['dev_rate']*100:4.0f} | "
          f"{row['mean_dV_dev']:+.3f}     |  {row['P_dV_gt_0.1']:.2f}   |  {row['P_dV_gt_0.3']:.2f}   | "
          f"{row['flip_rate_of_dev']*100:4.0f} | {row['search_entropy']:.2f}   | {row['temp_entropy']:.2f}  | "
          f"{row['top1_visit']:.2f}     | {row['corr_v_z']:+.2f}")
report["phase1_realized"] = p1res

print(f"\n===== PHASE 2a: TEMPERATURE SWEEP (predicted, post-hoc on visit policies) =====")
swres = {}
for nm, lo, hi in BANDS:
    if realB[nm]["n"] == 0: continue
    print(f"--- band {nm} (live temp@band≈{p1res[nm]['temp_at_band']}) ---")
    print(f"  T     | E[dev] | E[ΔV]  | P(Δ>.1) | P(Δ>.3) | samplingH")
    swres[nm] = {}
    for T in SWEEP_T:
        A = sweep[nm][T]
        swres[nm][str(T)] = {"dev": mean(A["dev"]), "ev": mean(A["ev"]), "p1": mean(A["p1"]), "p3": mean(A["p3"]), "H": mean(A["H"])}
        print(f"  {T:.1f}   |  {mean(A['dev']):.2f}  | {mean(A['ev']):+.3f} |  {mean(A['p1']):.2f}   |  {mean(A['p3']):.2f}   |   {mean(A['H']):.2f}")
report["phase2a_temp_sweep"] = swres

# ============================ PHASE 3: cross-epoch stats ======================
print(f"\n===== PHASE 3: CROSS-EPOCH STATISTICS =====")
print(f"{'ep':3} | {'games':5} | len: med / mean / p10 / p90 | %<30 | uniq1 / uniq2 / uniq3 | openH(3mv) | top3share")
p3 = {}
for ep in OPEN_EPOCHS:
    recs = records_for_epoch(ep)
    if not recs:
        continue
    lens = [len(r.action_ids) for r in recs]
    m1 = [int(r.action_ids[0]) for r in recs if len(r.action_ids) >= 1]
    m2 = [int(r.action_ids[1]) for r in recs if len(r.action_ids) >= 2]
    m3 = [(int(r.action_ids[0]), int(r.action_ids[1]), int(r.action_ids[2])) for r in recs if len(r.action_ids) >= 3]
    c3 = Counter(m3)
    openH = ent([v for v in c3.values()])
    top3share = sum(v for _, v in c3.most_common(3)) / max(1, len(m3))
    lens_s = sorted(lens)
    def pct(p): return lens_s[min(len(lens_s)-1, int(p*len(lens_s)))]
    row = {"games": len(recs), "len_med": float(np.median(lens)), "len_mean": float(np.mean(lens)),
           "len_p10": pct(0.10), "len_p90": pct(0.90), "pct_lt30": 100.0*sum(1 for x in lens if x < 30)/len(lens),
           "uniq_m1": len(set(m1)), "uniq_m2": len(set(m2)), "uniq_3move": len(c3), "open_entropy_3mv": openH,
           "top3_open_share": top3share}
    p3[ep] = row
    print(f"{ep:3} | {len(recs):5} | {row['len_med']:4.0f} / {row['len_mean']:5.1f} / {row['len_p10']:3} / {row['len_p90']:4} | "
          f"{row['pct_lt30']:4.0f} | {row['uniq_m1']:5} / {row['uniq_m2']:5} / {row['uniq_3move']:5} | {openH:.2f}       | {top3share:.2f}")
report["phase3_stats"] = p3

# z-label noise estimate from temperature deviations (aggregate over deep bands)
tot_n = sum(realB[b]["n"] for b in realB)
tot_dev = sum(realB[b]["dev"] for b in realB)
tot_flip = sum(realB[b]["flip"] for b in realB)
print(f"\n--- z-label noise from temperature (λ=0 hard-z) ---")
print(f"  positions={tot_n}  played-deviation={tot_dev} ({100*tot_dev/max(1,tot_n):.0f}%)  "
      f"outcome-flip deviations (top1 V>0.25 & played V<-0.25)={tot_flip} "
      f"({100*tot_flip/max(1,tot_n):.1f}% of all recorded positions)")
report["zlabel_noise"] = {"positions": tot_n, "dev": tot_dev, "flip": tot_flip,
                          "flip_frac_all": tot_flip/max(1, tot_n)}

Path("docs/analysis").mkdir(parents=True, exist_ok=True)
with open("_hexgnn_explore_results.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nwrote _hexgnn_explore_results.json")
