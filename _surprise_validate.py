"""Validate policy-surprise weighting on REAL search data (CPU, read-only).

Runs the live MCTS (run params) over real midgame positions, builds samples exactly
as self-play does (visit policy + root prior), then runs materialize_policy_surprise_rows
PER GAME (as selfplay.py does) and reports the surprise + weight + copy distributions.
Question: do weights VARY (high-KL rows upweighted -> more copies) or collapse to 1.0?
"""
import glob, statistics as st
import torch
import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.architecture import HexgtNetwork, expand_stv_readout_columns
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session
from hexo_models.dense_cnn.samples import sample_from_state
from hexo_models.dense_cnn.replay import materialize_policy_surprise_rows, _policy_kl
torch.set_grad_enabled(False)

RUN = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3"
CKPT = f"{RUN}/checkpoints/hexgt_rl_epoch000010.pt"


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False); a = ck["arch"]
    net = HexgtNetwork(token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
                       ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
                       short_term_value_horizons=tuple(a.get("short_term_value_horizons", ())),
                       value_pma_seeds=int(a.get("value_pma_seeds", 2)),
                       value_head_use_side=bool(a.get("value_head_use_side", True))).eval()
    expand_stv_readout_columns(net, ck["model"]); net.load_state_dict(ck["model"], strict=False)
    return net


def search_params():
    return dict(visits=512, c_puct=1.5, temperature=1.0, root_dirichlet_total_alpha=6.6,
                root_dirichlet_noise_fraction=0.25, root_policy_temperature=1.0, fpu_reduction=0.2,
                virtual_loss=1.0, widening_policy_mass=0.95, widening_max_children=96,
                widening_min_children=2, forced_playout_k=2.0)


net = load(CKPT)
inf = HexgtInference(net, device="cpu", fp16=False)
hxr = sorted(glob.glob(f"{RUN}/selfplay/epoch_000006.hxr"))[0]
games = [[unpack_coord_id(int(c)) for c in r.action_ids] for r in HexoRecordFile.open(hxr).iter_records()][:3]

all_surprise, all_weight, all_copies = [], [], []
total_raw = total_eff = 0
for gi, coords in enumerate(games):
    sess = new_mcts_session(max_states=200000, n=3)
    s = eng.new_game(); samples = []
    for i, c in enumerate(coords):
        if eng.terminal(s) is None and 8 <= i <= 60:
            res = sess.run([gi], [s], inf, seed=12345 + i, move_temperatures=[1.0], **search_params())[0]
            visit = list(res.visit_policy); prior = list(res.root_prior_policy)
            smp = sample_from_state(s, game_id=f"g{gi}", turn_index=i, policy=visit, root_prior_policy=prior,
                                    metadata={})
            samples.append(smp)
        if eng.terminal(s) is not None:
            break
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
    if not samples:
        continue
    mat, stats = materialize_policy_surprise_rows(samples, seed=777 + gi, uniform_fraction=0.5, max_weight=8.0)
    # recover per-sample surprise/weight/copies
    surprises = [_policy_kl(x.policy, x.root_prior_policy) for x in samples]
    stot = sum(surprises); n = len(samples)
    weights = [min(8.0, 0.5 + 0.5 * n * (sp / stot)) if stot > 0 else 1.0 for sp in surprises]
    # copies = how many times each appears in mat (by id of game/turn)
    from collections import Counter
    cnt = Counter((x.metadata.get("game_id"), x.turn_index) for x in mat)
    copies = [cnt.get((f"g{gi}", x.turn_index), 0) for x in samples]
    all_surprise += surprises; all_weight += weights; all_copies += copies
    total_raw += n; total_eff += len(mat)
    print(f"game{gi}: n={n} effective={len(mat)} surprise[min/med/max]={min(surprises):.3f}/{st.median(surprises):.3f}/{max(surprises):.3f}")


def pct(x, p): s = sorted(x); k = int((len(s)-1)*p); return s[k]
print("\n=== AGGREGATE over real searched positions ===")
print(f"n positions: {len(all_surprise)}  total_raw={total_raw} total_effective={total_eff} ratio={total_eff/max(1,total_raw):.3f}")
print(f"surprise KL(visits||prior): min={min(all_surprise):.3f} p10={pct(all_surprise,.1):.3f} median={st.median(all_surprise):.3f} mean={st.mean(all_surprise):.3f} p90={pct(all_surprise,.9):.3f} max={max(all_surprise):.3f}")
print(f"frequency WEIGHT:           min={min(all_weight):.3f} p10={pct(all_weight,.1):.3f} median={st.median(all_weight):.3f} mean={st.mean(all_weight):.3f} p90={pct(all_weight,.9):.3f} max={max(all_weight):.3f} stdev={st.pstdev(all_weight):.3f}")
print(f"COPIES written per position: min={min(all_copies)} median={st.median(all_copies)} mean={st.mean(all_copies):.2f} max={max(all_copies)}")
# correlation surprise vs copies
ms, mc = st.mean(all_surprise), st.mean(all_copies)
num = sum((a-ms)*(b-mc) for a,b in zip(all_surprise, all_copies))
den = (sum((a-ms)**2 for a in all_surprise)*sum((b-mc)**2 for b in all_copies))**0.5
print(f"corr(surprise, copies) = {num/den if den else float('nan'):+.3f}  (positive => high-KL rows duplicated more)")
print(f"weight VARIES (stdev>0) => reweighting ACTIVE; ratio~1.0 is EXPECTED (volume-preserving, KataGo-style).")
