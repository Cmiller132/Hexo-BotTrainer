"""Does MORE search change e23's decisions? Search-convergence probe (CPU, read-only).
Same checkpoint + same positions + same seed, vary visits 512/1024/2048 (fixed vbatch).
If 512 already agrees with 2048 (top-move match high, visit-KL low, root-value stable),
the search is converged at 512 -> more sims won't change play -> won't fix the plateau."""
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
VISITS = [512, 1024, 2048]
def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False); a = ck["arch"]
    net = HexgtNetwork(token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
        ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
        short_term_value_horizons=tuple(a.get("short_term_value_horizons", ())),
        value_pma_seeds=int(a.get("value_pma_seeds", 2)), value_head_use_side=bool(a.get("value_head_use_side", True))).eval()
    expand_stv_readout_columns(net, ck["model"]); net.load_state_dict(ck["model"], strict=False); return net
def P(v): return dict(c_puct=1.5, temperature=1.0, root_dirichlet_total_alpha=6.6, root_dirichlet_noise_fraction=0.30,
    root_policy_temperature=1.0, fpu_reduction=0.2, virtual_loss=1.0, widening_policy_mass=0.95,
    widening_max_children=96, widening_min_children=2, forced_playout_k=2.0, visits=v)
net = load(f"{RUN}/checkpoints/hexgt_rl_epoch000023.pt")
inf = HexgtInference(net, device="cpu", fp16=False)
hxr = sorted(glob.glob(f"{RUN}/selfplay/epoch_000021.hxr"))[0]
games = [[unpack_coord_id(int(c)) for c in r.action_ids] for r in HexoRecordFile.open(hxr).iter_records()][:6]
positions = []
for coords in games:
    s = eng.new_game()
    for i, c in enumerate(coords):
        if i in (12, 20, 28) and eng.terminal(s) is None: positions.append(eng.clone_state(s))
        if eng.terminal(s) is not None: break
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
    if len(positions) >= 12: break
positions = positions[:12]
def vmap(res):
    d = {int(a): float(w) for a, w in res.visit_policy}; tot = sum(d.values()) or 1.0
    return {k: v/tot for k, v in d.items()}
def kl(p, q, eps=1e-9): return sum(pv*math.log((pv+eps)/(q.get(k,0.0)+eps)) for k, pv in p.items() if pv > 0)
agg = {v: {"match": 0, "kl": [], "rvd": []} for v in VISITS}
for st0 in positions:
    rows = []
    for v in VISITS:
        sess = new_mcts_session(max_states=400000, n=3)
        res = sess.run([0], [eng.clone_state(st0)], inf, seed=4242, move_temperatures=[1.0],
                       virtual_batch_size=64, active_root_limit=64, **P(v))[0]
        vm = vmap(res); rows.append((v, vm, max(vm, key=vm.get), float(res.root_value)))
    ref_vm, ref_top, ref_rv = rows[-1][1], rows[-1][2], rows[-1][3]  # 2048 = reference
    for v, vm, top, rv in rows:
        agg[v]["match"] += int(top == ref_top); agg[v]["kl"].append(kl(ref_vm, vm)); agg[v]["rvd"].append(abs(rv-ref_rv))
n = len(positions)
print(f"\n=== e23 search convergence, {n} real midgame positions, same seed (ref=2048 visits) ===")
print(f"visits | top-move match w/ 2048 | mean visit-KL vs 2048 | mean |root-value Δ| vs 2048")
for v in VISITS:
    a = agg[v]
    print(f"  {v:4d} |    {a['match']}/{n}              |    {st.mean(a['kl']):.3f}            |    {st.mean(a['rvd']):.3f}")
