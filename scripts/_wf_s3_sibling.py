"""Sibling-discrimination probe for the moves_left head (read-only, CPU).

For a moves-left utility in MCTS, the gradient that matters is the SPREAD of
predicted moves-left across candidate children of the same parent. This probe
measures that spread at conversion-zone depths (true_ml = 10/30/60) and checks
the parent->child decrement against the ideal (-1 per decision).

Usage: python _wf_s3_sibling.py OUT_JSON
"""
from __future__ import annotations

import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np

PKG = "dense_cnn_restnet"
PKG_DIR = Path("/mnt/e/Hexo-BotTrainer-hexgt/packages/dense_cnn_restnet/python/dense_cnn_restnet")
if PKG not in sys.modules:
    stub = types.ModuleType(PKG)
    stub.__path__ = [str(PKG_DIR)]
    sys.modules[PKG] = stub

import torch  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(4)

import hexo_engine as engine  # noqa: E402
from hexo_engine.types import unpack_coord_id  # noqa: E402
from hexo_utils.records import HexoRecordFile  # noqa: E402

from dense_cnn_restnet.architecture import RestnetNetwork  # noqa: E402
from dense_cnn_restnet.constants import MOVES_LEFT_CAP  # noqa: E402
from dense_cnn_restnet.losses import decode_binned_value  # noqa: E402
from dense_cnn_restnet.samples import sample_from_state  # noqa: E402
from dense_cnn_restnet.input import build_input_planes  # noqa: E402
from dense_cnn_restnet.d6 import Axial  # noqa: E402

RUNS = Path("/mnt/e/Hexo-BotTrainer/runs")
ML_LEVELS = [10, 30, 60]
N_GAMES = 12
N_SIBLINGS = 12  # played + 11 random alternatives
rng = random.Random(7)

PAIRS = [
    ("m3c7", "dense_cnn_restnet_main_3", 7, "m3ep9", "dense_cnn_restnet_main_3", 9),
    ("m4c12", "dense_cnn_restnet_main_4", 12, "m4ep12", "dense_cnn_restnet_main_4", 12),
]


def build_net() -> RestnetNetwork:
    return RestnetNetwork(
        in_channels=13, channels=96, blocks_type="R_R_R_T_R_R_T_R",
        attention_heads=4, mlp_ratio=2, embed_kernel_size=3,
        residual_conv="hex", attention_impl="sdpa", attention_scope="disk",
        dropout=0.0, short_term_value_horizons=(2, 6, 16), moves_left_head=True,
    ).eval()


def load_ckpt(run: str, epoch: int) -> RestnetNetwork:
    net = build_net()
    payload = torch.load(RUNS / run / "checkpoints" / f"epoch_{epoch:06d}.pt",
                         map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("model"))
    missing, _ = net.load_state_dict(state, strict=False)
    bad = [k for k in missing if "disk" not in k and "relative_index" not in k]
    if bad:
        raise RuntimeError(f"{run} ep{epoch} missing keys: {bad[:8]}")
    return net


def planes_for(state):
    s = sample_from_state(state, game_id="t", turn_index=0,
                          policy=(), root_prior_policy=[(0, 1.0)])
    return build_input_planes(
        current_player=s.current_player, phase=s.phase, center=Axial(*s.center),
        stones=s.stones, legal_action_ids=s.legal_action_ids,
        placement_history=s.placement_history, first_stone=s.first_stone,
        own_hot=s.own_hot, opponent_hot=s.opponent_hot,
        opponent_last_turn=s.opponent_last_turn,
    ), list(s.legal_action_ids)


def pred_ml(net, planes_list):
    xs = torch.stack(planes_list)
    out = net(xs)
    norm = decode_binned_value(out["moves_left"].float())
    return ((norm + 1.0) * 0.5 * MOVES_LEFT_CAP).numpy()


def main():
    out_json = sys.argv[1]
    t0 = time.time()
    results = []
    for cname, crun, cep, sname, srun, sep in PAIRS:
        net = load_ckpt(crun, cep)
        with HexoRecordFile.open(RUNS / srun / "selfplay" / f"epoch_{sep:06d}.hxr") as rf:
            recs = [r for r in rf.iter_records() if r.winner is not None and len(r.action_ids) >= 80]
        recs.sort(key=lambda r: len(r.action_ids))
        idx = sorted(set(int(round(i)) for i in np.linspace(0, len(recs) - 1, N_GAMES)))
        picks = [recs[i] for i in idx]
        print(f"[{time.time()-t0:6.1f}s] {cname}|{sname}: {len(picks)} games", flush=True)
        for gi, rec in enumerate(picks):
            action_ids = [int(a) for a in rec.action_ids]
            L = len(action_ids)
            targets = {L - 1 - ml: ml for ml in ML_LEVELS if 1 <= L - 1 - ml < L}
            state = engine.new_game()
            for ply, aid in enumerate(action_ids):
                if ply in targets:
                    ml = targets[ply]
                    parent_planes, legal = planes_for(state)
                    if len(legal) < 2:
                        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(aid)))
                        continue
                    alts = [a for a in legal if a != aid]
                    rng.shuffle(alts)
                    cands = [aid] + alts[: N_SIBLINGS - 1]
                    child_planes, terminals, kept = [], 0, []
                    for a in cands:
                        cs = engine.clone_state(state)
                        engine.apply_action(cs, engine.PlacementAction(unpack_coord_id(a)))
                        if engine.terminal(cs) is not None:
                            terminals += 1
                            continue
                        cp, _ = planes_for(cs)
                        child_planes.append(cp)
                        kept.append(a)
                    if len(child_planes) >= 4:
                        preds = pred_ml(net, [parent_planes] + child_planes)
                        parent_pred = float(preds[0])
                        child_preds = preds[1:]
                        results.append({
                            "pair": f"{cname}|{sname}", "game_len": L, "true_ml": ml,
                            "n_legal": len(legal), "n_children": len(child_preds),
                            "n_terminal_children": terminals,
                            "parent_pred": round(parent_pred, 2),
                            "played_child_pred": round(float(child_preds[0]), 2) if kept and kept[0] == aid else None,
                            "child_pred_mean": round(float(child_preds.mean()), 2),
                            "child_pred_std": round(float(child_preds.std()), 3),
                            "child_pred_range": round(float(child_preds.max() - child_preds.min()), 2),
                            "mean_decrement": round(float(parent_pred - child_preds.mean()), 3),
                        })
                engine.apply_action(state, engine.PlacementAction(unpack_coord_id(aid)))
            print(f"[{time.time()-t0:6.1f}s] {cname}|{sname} game {gi+1}/{len(picks)} len={L}", flush=True)

    # aggregate by pair x true_ml
    agg = {}
    for pair in sorted({r["pair"] for r in results}):
        for ml in ML_LEVELS:
            rs = [r for r in results if r["pair"] == pair and r["true_ml"] == ml]
            if not rs:
                continue
            agg[f"{pair}|ml={ml}"] = {
                "n_positions": len(rs),
                "sib_std_mean": round(float(np.mean([r["child_pred_std"] for r in rs])), 3),
                "sib_range_mean": round(float(np.mean([r["child_pred_range"] for r in rs])), 2),
                "decrement_mean": round(float(np.mean([r["mean_decrement"] for r in rs])), 3),
                "parent_pred_mean": round(float(np.mean([r["parent_pred"] for r in rs])), 2),
                "parent_bias_mean": round(float(np.mean([r["parent_pred"] - r["true_ml"] for r in rs])), 2),
            }
    out = {"agg": agg, "positions": results}
    Path(out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
