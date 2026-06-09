"""Phase 2: NOISE vs TEMPERATURE attribution via controlled native-MCTS sims (CPU).

Reconstruct a fixed set of real positions across ply-bands. Holding positions + seed
fixed, sweep each knob INDEPENDENTLY through the actual native MCTS:
  (A) Dirichlet eps in {0.0, 0.15, 0.30, 0.45}  (temp fixed = live curve temp at the ply)
  (B) move temperature in {0.0, 0.3, 0.6, 1.0}  (eps fixed = live 0.30)
Measure per config: visit-policy entropy (= policy-target spread), top-1 mass
(= policy-target sharpness), and played-move deviation rate vs the search's own
top-visit move. Because the search is identical across move-temperatures at fixed
eps+seed, the temp sweep isolates the PLAYED-MOVE channel; the eps sweep isolates the
SEARCH / policy-target channel.

visits reduced from the live 1024 -> SIM_VISITS for CPU tractability (noted); the
ATTRIBUTION (relative effect of each knob) is what matters, not absolute throughput.
"""
import sys, json, statistics as st
from pathlib import Path
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/_expl")
import common as C
from hexo_utils.records import HexoRecordFile
from hexgnn.mcts import new_mcts_session

EPOCH = 41
SIM_VISITS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
SEED = 12345
ALPHA = 6.6           # live total_alpha
C_PUCT = 1.5
RPT = 1.0             # live root_policy_temperature
FPK = 2.0             # live forced_playout_k
WMASS, WMAXC, WMINC = 0.95, 96, 2
EPS_SWEEP = [0.0, 0.15, 0.30, 0.45]
TEMP_SWEEP = [0.0, 0.3, 0.6, 1.0]
# target plies per band to reconstruct fixed positions
TARGET_PLIES = [2, 4, 8, 12, 16, 25, 32, 40, 55, 70]

net, _ = C.load_model(EPOCH)
ev = C.Evaluator(net, n=4)

# --- reconstruct fixed positions from real games -------------------------------
recs = list(HexoRecordFile.open(f"{C.SP_DIR}/epoch_{EPOCH:06d}.hxr").iter_records())
positions = []  # (label, ply, state)
seen_ply = {}
for rec in recs:
    actions = [int(a) for a in rec.action_ids]
    for target in TARGET_PLIES:
        if target >= len(actions):
            continue
        if seen_ply.get(target, 0) >= 4:   # up to 4 positions per target ply
            continue
        s = C.new_game(seed=int(rec.seed))
        ok = True
        for t in range(target):
            if C.engine.terminal(s) is not None:
                ok = False; break
            C.apply(s, actions[t])
        if ok and C.engine.terminal(s) is None:
            positions.append((f"g{C.shard_index_from_game_id(rec.game_id)}@p{target}", target, s))
            seen_ply[target] = seen_ply.get(target, 0) + 1
    if all(seen_ply.get(t, 0) >= 4 for t in TARGET_PLIES if t < 80):
        break
print(f"reconstructed {len(positions)} fixed positions across plies {sorted(set(p for _,p,_ in positions))}")

def run_config(states, eps, temps):
    """One batched native search over all roots with the given eps + per-root temps."""
    sess = new_mcts_session(n=4)
    keys = list(range(len(states)))
    res = sess.run(
        keys, states, ev.inf,
        visits=SIM_VISITS, c_puct=C_PUCT, temperature=1.0, seed=SEED,
        virtual_batch_size=32,
        root_dirichlet_total_alpha=(ALPHA if eps > 0 else None),
        root_dirichlet_noise_fraction=(eps if eps > 0 else None),
        root_policy_temperature=RPT, forced_playout_k=FPK,
        widening_policy_mass=WMASS, widening_max_children=WMAXC, widening_min_children=WMINC,
        move_temperatures=list(temps),
    )
    return res

def summarize(results, plies):
    """Per-band aggregates of visit entropy, top1 mass, deviation rate."""
    bands = {nm: {"H": [], "top1": [], "dev": []} for nm, _, _ in C.BANDS}
    for r, ply in zip(results, plies):
        vp = list(r.visit_policy)
        if not vp:
            continue
        ws = [w for _, w in vp]
        tot = sum(ws)
        if tot <= 0:
            continue
        top_a = max(vp, key=lambda x: x[1])[0]
        b = bands[C.band_of(ply)]
        b["H"].append(C.entropy(ws))
        b["top1"].append(max(ws) / tot)
        b["dev"].append(0.0 if int(r.action_id) == int(top_a) else 1.0)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return {nm: {"H": mean(b["H"]), "top1": mean(b["top1"]), "dev": mean(b["dev"]), "n": len(b["H"])}
            for nm, b in bands.items()}

states = [s for _, _, s in positions]
plies = [p for _, p, _ in positions]
curve_temps = [C.cur_temp(p) for p in plies]

out = {"sim_visits": SIM_VISITS, "n_positions": len(positions), "seed": SEED,
       "eps_sweep": {}, "temp_sweep": {}}

# (A) eps sweep at the LIVE curve temperature (so played-move deviation reflects live temp)
print(f"\n=== (A) DIRICHLET eps sweep (move-temp fixed at live curve; visits={SIM_VISITS}) ===")
print("  band   |  eps=0.00        |  eps=0.15        |  eps=0.30(live)  |  eps=0.45")
print("         |  H   top1  dev   |  H   top1  dev   |  H   top1  dev   |  H   top1  dev")
eps_res = {eps: summarize(run_config(states, eps, curve_temps), plies) for eps in EPS_SWEEP}
out["eps_sweep"] = eps_res
for nm, _, _ in C.BANDS:
    row = f"  {nm:6s} |"
    for eps in EPS_SWEEP:
        d = eps_res[eps][nm]
        row += f"  {d['H']:.2f} {d['top1']:.2f} {d['dev']:.2f} |"
    print(row)

# (B) temperature sweep at the LIVE eps=0.30
print(f"\n=== (B) move-TEMPERATURE sweep (eps fixed at live 0.30; visits={SIM_VISITS}) ===")
print("  band   |  T=0.0           |  T=0.3           |  T=0.6           |  T=1.0")
print("         |  H   top1  dev   |  H   top1  dev   |  H   top1  dev   |  H   top1  dev")
temp_res = {T: summarize(run_config(states, 0.30, [T] * len(states)), plies) for T in TEMP_SWEEP}
out["temp_sweep"] = temp_res
for nm, _, _ in C.BANDS:
    row = f"  {nm:6s} |"
    for T in TEMP_SWEEP:
        d = temp_res[T][nm]
        row += f"  {d['H']:.2f} {d['top1']:.2f} {d['dev']:.2f} |"
    print(row)

print("\nNOTE: in (B), H and top1 (the policy TARGET) are ~constant across T — temperature")
print("does not touch the recorded search; it only changes which move is PLAYED (dev).")

Path("/mnt/e/Hexo-BotTrainer-hexgt/_expl/out").mkdir(exist_ok=True)
with open("/mnt/e/Hexo-BotTrainer-hexgt/_expl/out/phase2.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote _expl/out/phase2.json")
