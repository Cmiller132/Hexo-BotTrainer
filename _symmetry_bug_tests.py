"""READ-ONLY CPU tests for a SUBTLE SYMMETRY / SOFT-Z bug. No GPU, no run mutation.

Test 1: probe validity. swap_owner must be a clean color relabel:
        (a) involution: swap(swap(f)) features == f features
        (b) node_feat diff between f and swap(f) touches ONLY owner-derived columns.
Test 2: MCTS max-bias. Search the empty board (no Dirichlet noise) with a
        near-calibrated net and compare root_value to the raw net value -> isolates
        whether root_value=+0.43 is search max-bias vs net optimism vs a sign bug.
"""
import glob, statistics as st
import numpy as np
import torch

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.features import build_graph_tensors
from hexo_models.hexgt.collate import collate_graphs
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt.losses import decode_binned_value
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session
from hexo_models.hexgt import constants as C

RADIUS = 3
RUN = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3"
torch.set_grad_enabled(False)

OWNER_COLS = {
    C.F_OWNER_OWN, C.F_OWNER_OPP, C.F_CAND_COMPLETE_OWN, C.F_CAND_COMPLETE_OPP,
    C.F_CAND_NWIN_OWN, C.F_CAND_NWIN_OPP, C.F_SIDE_STONES_OWN, C.F_SIDE_STONES_OPP,
    C.F_CAND_OWN_WIN3, C.F_CAND_OWN_WIN4, C.F_CAND_OWN_WIN5,
    C.F_CAND_OPP_WIN3, C.F_CAND_OPP_WIN4, C.F_CAND_OPP_WIN5,
    C.F_CAND_WIN_NOW_OWN,
}
for name in ("F_CAND_WIN_NOW_OPP", "F_CAND_OPP_THREAT"):
    if hasattr(C, name):
        OWNER_COLS.add(getattr(C, name))


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["arch"]
    net = HexgtNetwork(
        token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
        ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
        short_term_value_horizons=tuple(a.get("short_term_value_horizons", ())),
        value_pma_seeds=int(a.get("value_pma_seeds", 2)),
        value_head_use_side=bool(a.get("value_head_use_side", True)),
    ).eval()
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("short_term_value_heads")}
    net.load_state_dict(sd, strict=False)
    return net


def node_feat(facts):
    return np.asarray(build_graph_tensors(facts).node_feat, dtype=np.float64).copy()


def swap_owner(facts):
    own = list(facts["nodes"]["node_owner"])
    sw = [1 if o == 0 else (0 if o == 1 else o) for o in own]
    return {**facts, "nodes": {**facts["nodes"], "node_owner": sw}}


def raw_value(net, facts):
    b = collate_graphs([build_graph_tensors(facts)])
    return float(decode_binned_value(net.forward_policy_value(b)["value"].float())[0])


# ---- Test 1: probe validity ----
print("=" * 70)
print("TEST 1 — probe owner-swap validity (clean color relabel?)")
print("=" * 70)
hxr = sorted(glob.glob(f"{RUN}/selfplay/epoch_000006.hxr"))[0]
changed_cols = set()
n_pos = 0
involution_ok = True
for r in HexoRecordFile.open(hxr).iter_records():
    coords = [unpack_coord_id(int(c)) for c in r.action_ids]
    s = eng.new_game()
    for i, c in enumerate(coords):
        if eng.terminal(s) is None and 10 <= i <= 45:
            f = rust_bridge.graph_facts(s, RADIUS)
            if f.get("meta", {}).get("first_stone") is None:
                A = node_feat(f)
                B = node_feat(swap_owner(f))
                BB = node_feat(swap_owner(swap_owner(f)))
                if not np.allclose(A, BB):
                    involution_ok = False
                diff = np.where(np.abs(A - B).sum(axis=0) > 1e-6)[0]
                changed_cols.update(int(x) for x in diff)
                n_pos += 1
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
        if n_pos >= 200:
            break
    if n_pos >= 200:
        break
non_owner = changed_cols - OWNER_COLS
print(f"  positions tested: {n_pos}")
print(f"  involution swap(swap(f))==f : {'PASS' if involution_ok else 'FAIL'}")
print(f"  feature columns changed by swap: {sorted(changed_cols)}")
print(f"  owner-derived column set:        {sorted(OWNER_COLS)}")
print(f"  NON-owner columns changed (should be empty): {sorted(non_owner)}")
print(f"  VERDICT: {'probe is a CLEAN color relabel (valid)' if not non_owner and involution_ok else 'PROBE MAY BE BUGGY — stale non-owner feature!'}")

# ---- Test 2: MCTS max-bias on the empty board ----
print()
print("=" * 70)
print("TEST 2 — why is root_value(empty board) high? max-bias vs net vs sign-bug")
print("=" * 70)
for label, path in [("pretrain", f"{RUN}/pretrain/hexgt_model3_pretrain.pt"),
                    ("epoch6", f"{RUN}/checkpoints/hexgt_rl_epoch000006.pt")]:
    net = load_model(path)
    inf = HexgtInference(net, device="cpu", fp16=False)
    sess = new_mcts_session(max_states=200000, n=RADIUS)
    s0 = eng.new_game()
    raw0 = raw_value(net, rust_bridge.graph_facts(s0, RADIUS))
    # search the empty board, NO Dirichlet noise (isolate max-bias), greedy-ish
    res = sess.run([0], [s0], inf, visits=512, c_puct=1.5, temperature=1.0, seed=12345,
                   root_dirichlet_total_alpha=6.6, root_dirichlet_noise_fraction=0.25,
                   root_policy_temperature=1.0, fpu_reduction=0.2, virtual_loss=1.0,
                   widening_policy_mass=0.95, widening_max_children=96, widening_min_children=2,
                   forced_playout_k=2.0, move_temperatures=[1.0])[0]
    # also a couple plies in (more candidates -> real max over children)
    s1 = eng.new_game()
    eng.apply_action(s1, PlacementAction(unpack_coord_id(int(res.action_id))))
    raw1 = raw_value(net, rust_bridge.graph_facts(s1, RADIUS))
    res1 = sess.run([1], [s1], inf, visits=512, c_puct=1.5, temperature=1.0, seed=999,
                    root_dirichlet_total_alpha=6.6, root_dirichlet_noise_fraction=0.25,
                    root_policy_temperature=1.0, fpu_reduction=0.2, virtual_loss=1.0,
                    widening_policy_mass=0.95, widening_max_children=96, widening_min_children=2,
                    forced_playout_k=2.0, move_temperatures=[1.0])[0]
    print(f"  [{label}] empty board: raw net v = {raw0:+.3f}   MCTS root_value(no noise) = {res.root_value:+.3f}   (search adds {res.root_value-raw0:+.3f})  cands={len(list(res.root_prior_policy))}")
    print(f"  [{label}] ply-1 board: raw net v = {raw1:+.3f}   MCTS root_value(no noise) = {res1.root_value:+.3f}   (search adds {res1.root_value-raw1:+.3f})")
