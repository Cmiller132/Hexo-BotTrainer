"""Phase 2b: Dirichlet-noise attribution via live CPU MCTS (read-only).

On a fixed set of real positions (sampled across ply bands from epoch-41 games),
run the SAME native hexgnn search varying ONLY the root Dirichlet noise fraction
eps in {0.0, 0.15, 0.30, 0.45} (total_alpha=6.6 live), averaged over several noise
seeds. Everything else is at the live config (c_puct=1.5, fpu=0.20, widening
mass=0.95/max=96, forced_playout_k=2.0, root_policy_temperature=1.0). Move
selection is greedy (move_temperature=0) so we isolate the noise's effect on the
SEARCHED visit distribution = the policy training target, not on the played move.

Metrics per eps: mean visit entropy, mean top1-visit fraction, mean effective-N,
and P(top-visit move != the eps=0 baseline move) — how often noise changes which
move the search would pick (and thus the argmax of the policy target).
"""
import sys, math, json, random, statistics as st
from collections import Counter
RNG = random.Random(20260606)

for _p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models"):
    sys.path.insert(0, f"packages/{_p}/python")
sys.path.insert(0, "packages/hexgnn/python")

import torch
torch.set_grad_enabled(False)
torch.set_num_threads(6)

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexgnn.architecture import HexgnnNetwork
from hexgnn.inference import HexgnnInference
from hexgnn.mcts import new_mcts_session

RUN = "runs/hexgnn_rl_main1"
CKPT_EP = 41
N = 4
VISITS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
EPS_SWEEP = [0.0, 0.15, 0.30, 0.45]
TOTAL_ALPHA = 6.6
SAMPLE_PLIES = [2, 6, 12, 22, 30, 40, 55, 70]   # spanning opening / mid / late
GAMES = 12

ck = torch.load(f"{RUN}/checkpoints/hexgnn_rl_epoch{CKPT_EP:06d}.pt", map_location="cpu", weights_only=False)
a = ck["arch"]
net = HexgnnNetwork(token_dim=a["token_dim"], gnn_layers=a["gnn_layers"],
                    attention_heads=a.get("attention_heads", 4),
                    value_pma_seeds=a.get("value_pma_seeds", 2),
                    value_head_use_side=a.get("value_head_use_side", True),
                    steerable_channels=a.get("steerable_channels", 0))
net.load_state_dict(ck["model"]); net.eval()
INF = HexgnnInference(net, device="cpu", fp16=False)
SESS = new_mcts_session(max_states=200000, n=N)

def band_of(t):
    return "0-20" if t < 20 else ("20-50" if t < 50 else "50+")

# ---- collect fixed positions -------------------------------------------------
_allrecs = list(HexoRecordFile.open(f"{RUN}/selfplay/epoch_{CKPT_EP:06d}.hxr").iter_records())
# prefer longer games so the late ply bands (55,70) are actually reachable.
_allrecs.sort(key=lambda r: -len(r.action_ids))
recs = _allrecs[:GAMES]
positions = []   # (band, ply, state)
for rec in recs:
    coords = [unpack_coord_id(int(c)) for c in rec.action_ids]
    s = eng.new_game()
    want = set(SAMPLE_PLIES)
    for t, c in enumerate(coords):
        if eng.terminal(s) is not None:
            break
        if t in want:
            positions.append((band_of(t), t, eng.clone_state(s)))
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
print(f"collected {len(positions)} positions across bands "
      f"{Counter(b for b, _, _ in positions)} | visits={VISITS} seeds={SEEDS}")

def search(state, key, eps, seed):
    r = SESS.run([key], [state], INF, visits=VISITS, c_puct=1.5,
                 temperature=1.0, seed=seed,
                 virtual_batch_size=16, active_root_limit=256,
                 root_dirichlet_total_alpha=(TOTAL_ALPHA if eps > 0 else None),
                 root_dirichlet_noise_fraction=(eps if eps > 0 else None),
                 root_policy_temperature=1.0, fpu_reduction=0.20, virtual_loss=1.0,
                 widening_policy_mass=0.95, widening_max_children=96, widening_min_children=2,
                 forced_playout_k=2.0, move_temperatures=[0.0])[0]
    SESS.discard(key)
    vp = [(int(act), float(w)) for act, w in r.visit_policy]
    tot = sum(w for _, w in vp) or 1.0
    vp = [(act, w/tot) for act, w in vp]
    vp.sort(key=lambda x: -x[1])
    H = -sum(p*math.log(p) for _, p in vp if p > 0)
    effN = math.exp(H)
    return {"top": vp[0][0], "top1": vp[0][1], "H": H, "effN": effN, "support": len(vp)}

# ---- sweep -------------------------------------------------------------------
# per band x eps accumulators
acc = {b: {e: {"H": [], "top1": [], "effN": [], "support": [], "changed": []} for e in EPS_SWEEP}
       for b in ["0-20", "20-50", "50+"]}
key = 0
for bi, (band, ply, state) in enumerate(positions):
    # baseline (eps=0) top move, averaged is deterministic but seed-varied search has
    # randomness from leaf tie-breaking; use seed 0 baseline per position.
    base = search(state, key, 0.0, 12345 + bi); key += 1
    base_top = base["top"]
    for e in EPS_SWEEP:
        for sd in range(SEEDS):
            r = search(state, key, e, 1000 + bi*97 + sd*7919); key += 1
            A = acc[band][e]
            A["H"].append(r["H"]); A["top1"].append(r["top1"]); A["effN"].append(r["effN"])
            A["support"].append(r["support"]); A["changed"].append(1.0 if r["top"] != base_top else 0.0)

def mean(xs): return st.mean(xs) if xs else float("nan")
report = {"visits": VISITS, "seeds": SEEDS, "total_alpha": TOTAL_ALPHA,
          "n_positions": len(positions), "bands": {}}
print(f"\n===== PHASE 2b: DIRICHLET eps SWEEP (live total_alpha={TOTAL_ALPHA}, visits={VISITS}) =====")
print("effect on the SEARCHED visit distribution (= policy target). top-move-change vs eps=0 baseline.")
for band in ["0-20", "20-50", "50+"]:
    if not acc[band][0.0]["H"]:
        continue
    print(f"\n--- band {band} ---")
    print(f"  eps   | visitH | effN | top1-visit | support | P(top-move != eps0)")
    report["bands"][band] = {}
    for e in EPS_SWEEP:
        A = acc[band][e]
        report["bands"][band][str(e)] = {"H": mean(A["H"]), "effN": mean(A["effN"]),
                                         "top1": mean(A["top1"]), "support": mean(A["support"]),
                                         "p_change": mean(A["changed"])}
        print(f"  {e:.2f}  |  {mean(A['H']):.2f}  | {mean(A['effN']):4.1f} |   {mean(A['top1']):.2f}     | "
              f"{mean(A['support']):5.1f}   |   {mean(A['changed']):.2f}")

with open("_hexgnn_dirichlet_results.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nwrote _hexgnn_dirichlet_results.json")
