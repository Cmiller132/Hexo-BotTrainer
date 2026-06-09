"""Quantify the current temperature curve's cost vs benefit, per ply band (CPU, read-only).

For real positions (recent self-play), using the search's own visit policy (from shards) +
1-ply value lookahead (apply each top move, eval value head):
  - blunder cost: E[value-drop] and P(drop>0.1/0.3) of a TEMP-SAMPLED move vs the top-visit move
  - diversity: entropy of the sampling distribution
  - deviation: P(sampled != top-visit)
swept over temperature, mapped to the current curve's temp at each band. Also the ACTUAL
realized deviation rate of the played move (from .hxr) as a cross-check.
"""
import glob, re, math, statistics as st
import importlib.util as u
import torch
spec = u.spec_from_file_location("exp", "_lambda0_experiment.py"); m = u.module_from_spec(spec); spec.loader.exec_module(m)
import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.features import build_graph_tensors
from hexo_models.hexgt.collate import collate_graphs
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt.losses import decode_binned_value
from hexo_models.dense_cnn.compact_io import read_compact_shard
from pathlib import Path
torch.set_grad_enabled(False)
RUN = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3"
EP = 21
TEMPS = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
TOPK = 10
BANDS = [("0-20", 0, 20), ("20-50", 20, 50), ("50-100", 50, 100), ("100+", 100, 10**9)]
def cur_temp(ply): return 0.3 + 0.7 * 2 ** (-ply / 44.26)

net = m.load_grafted(f"{RUN}/checkpoints/hexgt_rl_epoch{EP:06d}.pt", "cpu")[0]; net.eval()
def value_of(state):
    f = rust_bridge.graph_facts(state, 3); b = collate_graphs([build_graph_tensors(f)])
    return float(decode_binned_value(net.forward_policy_value(b)["value"].float())[0])

def band_of(t):
    for nm, lo, hi in BANDS:
        if lo <= t < hi: return nm
    return "100+"

# accumulators: acc[band][temp] = {ev: [], p1:[], p3:[], H:[], dev:[]}; real[band]={dev:0,n:0,dV:[]}
acc = {b[0]: {T: {"ev": [], "p1": [], "p3": [], "H": [], "dev": []} for T in TEMPS} for b in BANDS}
real = {b[0]: {"dev": 0, "n": 0, "dV": []} for b in BANDS}

hxr = f"{RUN}/selfplay/epoch_{EP:06d}.hxr"
records = list(HexoRecordFile.open(hxr).iter_records())[:16]
npos = 0
for rec in records:
    idx = int(re.search(r"selfplay-(\d+)$", rec.game_id).group(1))
    shp = f"{RUN}/selfplay/epoch_{EP:06d}_game_{idx:06d}.npz"
    if not glob.glob(shp): continue
    rows = read_compact_shard(Path(shp))
    pol_by_turn = {int(r.turn_index): list(r.policy) for r in rows}
    coords = [unpack_coord_id(int(c)) for c in rec.action_ids]
    s = eng.new_game()
    for t, c in enumerate(coords):
        if eng.terminal(s) is not None: break
        if t in pol_by_turn:
            pol = [(int(a), float(w)) for a, w in pol_by_turn[t] if float(w) > 0]
            tot = sum(w for _, w in pol)
            if pol and tot > 0 and len(pol) >= 2:
                pol.sort(key=lambda x: -x[1])
                pv = [(a, w / tot) for a, w in pol]              # full visit dist (for entropy)
                topk = pv[:TOPK]
                # 1-ply value of each top move (self perspective)
                vs = {}
                for a, _w in topk:
                    cc = unpack_coord_id(a); s2 = eng.clone_state(s)
                    try:
                        eng.apply_action(s2, PlacementAction(AxialCoord(q=int(cc.q), r=int(cc.r))))
                        vs[a] = -value_of(s2)
                    except Exception:
                        vs[a] = None
                top1 = topk[0][0]
                if vs.get(top1) is None:
                    eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r)))); continue
                dV = {a: (vs[top1] - vs[a]) for a, _w in topk if vs.get(a) is not None}
                bnd = band_of(t)
                full_w = [w for _, w in pv]
                for T in TEMPS:
                    inv = 1.0 / T
                    # sampling dist over full set (entropy) and over topk (dV expectation)
                    fz = [w ** inv for w in full_w]; fzt = sum(fz)
                    pT_full = [z / fzt for z in fz]
                    H = -sum(p * math.log(p) for p in pT_full if p > 0)
                    kz = [(a, (w ** inv)) for a, w in topk if a in dV]; kzt = sum(z for _, z in kz)
                    pT = [(a, z / kzt) for a, z in kz]
                    ev = sum(p * dV[a] for a, p in pT)
                    p1 = sum(p for a, p in pT if dV[a] > 0.1)
                    p3 = sum(p for a, p in pT if dV[a] > 0.3)
                    dev = 1.0 - (pT[0][1] if pT and pT[0][0] == top1 else 0.0)
                    A = acc[bnd][T]; A["ev"].append(ev); A["p1"].append(p1); A["p3"].append(p3); A["H"].append(H); A["dev"].append(dev)
                # actual played-move deviation + its dV
                played = int(rec.action_ids[t]) if t < len(rec.action_ids) else None
                real[bnd]["n"] += 1
                if played is not None and played != top1:
                    real[bnd]["dev"] += 1
                    if played in dV: real[bnd]["dV"].append(dV[played])
                npos += 1
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))

print(f"positions analyzed: {npos} (epoch {EP}, model e{EP})\n")
mean = lambda xs: st.mean(xs) if xs else float("nan")
for nm, lo, hi in BANDS:
    n = real[nm]["n"]
    if n == 0: continue
    rep = (lo + min(hi, lo + 40)) // 2 if hi < 10**9 else 150
    print(f"=== ply band {nm} (n={n}, current-curve temp≈{cur_temp(rep):.2f} at ply~{rep}) ===")
    print(f"  temp |  P(deviate) | E[valΔ] | P(Δ>0.1) | P(Δ>0.3) | sampling-entropy")
    for T in TEMPS:
        A = acc[nm][T]
        print(f"  {T:.1f}  |    {mean(A['dev']):.2f}     |  {mean(A['ev']):+.3f} |  {mean(A['p1']):.2f}    |  {mean(A['p3']):.2f}    |   {mean(A['H']):.2f}")
    rv = real[nm]["dV"]
    print(f"  ACTUAL played-move deviation rate = {real[nm]['dev']/n:.0%}; mean valΔ of deviations = {mean(rv):+.3f} (n_dev={len(rv)})\n")
