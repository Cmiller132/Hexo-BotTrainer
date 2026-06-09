"""One-shot optimism/calibration probe for hexgt_rl_main3 (read-only, CPU).

Same methodology as scripts/_optimism.sh: at FirstStone midgame positions, evaluate
the IDENTICAL board from both perspectives via a facts-level owner-swap (0<->1),
which `build_graph_tensors` re-derives ALL features from — including the SIDE-node
own/opp stone counts the value head reads. mean(v_A + v_B): 0 = zero-sum-consistent,
>0 = optimism (both sides think they win). Old head was +0.82; pretrain seed -0.058.
"""
import glob
import statistics as st

import torch

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.features import build_graph_tensors
from hexo_models.hexgt.collate import collate_graphs
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt.losses import decode_binned_value

RADIUS = 3
RUN = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3"
CKPTS = [
    ("pretrain seed", f"{RUN}/pretrain/hexgt_model3_pretrain.pt"),
    ("RL epoch0", f"{RUN}/checkpoints/hexgt_rl_epoch000000.pt"),
    ("RL epoch1", f"{RUN}/checkpoints/hexgt_rl_epoch000001.pt"),
    ("RL epoch2", f"{RUN}/checkpoints/hexgt_rl_epoch000002.pt"),
    ("RL epoch3", f"{RUN}/checkpoints/hexgt_rl_epoch000003.pt"),
]
torch.set_grad_enabled(False)
m = None  # bound per-checkpoint in main loop


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
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    assert not [k for k in missing if not k.startswith("_")], f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    return net


def val_from_facts(facts):
    g = build_graph_tensors(facts)
    b = collate_graphs([g])
    return float(decode_binned_value(m.forward_policy_value(b)["value"].float())[0])


def swap_owner(facts):
    own = list(facts["nodes"]["node_owner"])
    sw = [1 if o == 0 else (0 if o == 1 else o) for o in own]
    return {**facts, "nodes": {**facts["nodes"], "node_owner": sw}}


def coords_of(f):
    rf = HexoRecordFile.open(f)
    return [[unpack_coord_id(int(c)) for c in r.action_ids] for r in rf.iter_records()]


def probe(filelist, label, maxpos=250):
    sums, both_pos, n = [], 0, 0
    for f in filelist:
        for coords in coords_of(f):
            if n >= maxpos:
                break
            s = eng.new_game()
            for i, c in enumerate(coords):
                if 10 <= i <= 50:
                    facts = rust_bridge.graph_facts(s, RADIUS)
                    if facts.get("meta", {}).get("first_stone") is None:
                        vA = val_from_facts(facts)
                        vB = val_from_facts(swap_owner(facts))
                        sums.append(vA + vB)
                        n += 1
                        if vA > 0.05 and vB > 0.05:
                            both_pos += 1
                        if n >= maxpos:
                            break
                eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
        if n >= maxpos:
            break
    if sums:
        print(f"\n[{label}] {n} FirstStone midgame positions (identical board, both perspectives)")
        print(f"   mean(vA+vB) = {st.mean(sums):+.4f}   median = {st.median(sums):+.4f}   stdev = {st.pstdev(sums):.4f}")
        print(f"   (0 = zero-sum-consistent / calibrated; >0 = optimism. old head +0.82, pretrain seed -0.058)")
        print(f"   BOTH sides predict winning (vA>0 AND vB>0): {both_pos}/{n} = {both_pos/n:.0%}")
        print(f"   |sum|>0.5: {sum(1 for x in sums if abs(x) > 0.5)}/{n}")
    else:
        print(f"[{label}] no eligible positions found")


sp = sorted(glob.glob(f"{RUN}/selfplay/epoch_000003.hxr")) + sorted(glob.glob(f"{RUN}/selfplay/epoch_000002.hxr"))
for label, path in CKPTS:
    m = load_model(path)
    probe(sp, label, 250)
