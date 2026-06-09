"""How much does virtual-batch size change the SEARCH? (CPU, read-only, no GPU contention)
Same position + same seed + same params, vary vbatch. Compare vs vbatch=1 (fully
sequential MCTS = gold standard): played-move agreement, visit-distribution KL, root value."""
import glob, math, statistics as st
import torch
import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.architecture import HexgtNetwork, expand_stv_readout_columns
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session
torch.set_grad_enabled(False)
RUN = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3"

def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False); a = ck["arch"]
    net = HexgtNetwork(token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
        ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
        short_term_value_horizons=tuple(a.get("short_term_value_horizons", ())),
        value_pma_seeds=int(a.get("value_pma_seeds", 2)), value_head_use_side=bool(a.get("value_head_use_side", True))).eval()
    expand_stv_readout_columns(net, ck["model"]); net.load_state_dict(ck["model"], strict=False); return net

PARAMS = dict(visits=512, c_puct=1.5, temperature=1.0, root_dirichlet_total_alpha=6.6,
    root_dirichlet_noise_fraction=0.25, root_policy_temperature=1.0, fpu_reduction=0.2, virtual_loss=1.0,
    widening_policy_mass=0.95, widening_max_children=96, widening_min_children=2, forced_playout_k=2.0)
VBATCHES = [1, 8, 16, 32]
net = load(f"{RUN}/checkpoints/hexgt_rl_epoch000012.pt")
inf = HexgtInference(net, device="cpu", fp16=False)

# pick midgame positions across several games (plies 12/18/24/30) for n~16
hxr = sorted(glob.glob(f"{RUN}/selfplay/epoch_000006.hxr"))[0]
games = [[unpack_coord_id(int(c)) for c in r.action_ids] for r in HexoRecordFile.open(hxr).iter_records()][:8]
positions = []
for coords in games:
    s = eng.new_game()
    for i, c in enumerate(coords):
        if i in (12, 18, 24, 30) and eng.terminal(s) is None:
            positions.append((i, eng.clone_state(s)))
        if eng.terminal(s) is not None: break
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
    if len(positions) >= 16: break
positions = positions[:16]

def visit_map(res):
    m = {}
    for a, w in res.visit_policy: m[int(a)] = float(w)
    tot = sum(m.values()) or 1.0
    return {k: v/tot for k, v in m.items()}

def kl(p, q, eps=1e-9):
    return sum(pv*math.log((pv+eps)/(q.get(k,0.0)+eps)) for k, pv in p.items() if pv > 0)

agg = {vb: {"match": 0, "kl": [], "rvd": []} for vb in VBATCHES}
for pi, (ply, st0) in enumerate(positions):
    rows = []
    for vb in VBATCHES:
        sess = new_mcts_session(max_states=300000, n=3)
        res = sess.run([0], [eng.clone_state(st0)], inf, seed=4242, move_temperatures=[1.0],
                       virtual_batch_size=vb, active_root_limit=64, **PARAMS)[0]
        vm = visit_map(res); top = max(vm, key=vm.get) if vm else None
        rows.append((vb, vm, top, float(res.root_value)))
    ref_vm, ref_top, ref_rv = rows[0][1], rows[0][2], rows[0][3]
    for vb, vm, top, rv in rows:
        if vb == 1: continue
        agg[vb]["match"] += int(top == ref_top)
        agg[vb]["kl"].append(kl(ref_vm, vm))
        agg[vb]["rvd"].append(abs(rv - ref_rv))
n = len(positions)
print(f"\n=== vbatch vs vbatch=1 (sequential MCTS), {n} real midgame positions, same seed ===")
print(f"vbatch | rounds/move | top1 match w/ seq | mean visit-KL | mean |rootvalue Δ|")
print("-"*76)
for vb in VBATCHES:
    if vb == 1:
        print(f"  {vb:4d} |    {512//vb:4d}     |    (reference)    |    0.000      |    0.000")
        continue
    a = agg[vb]
    print(f"  {vb:4d} |    {512//vb:4d}     |    {a['match']}/{n}            |    {st.mean(a['kl']):.3f}      |    {st.mean(a['rvd']):.3f}")
