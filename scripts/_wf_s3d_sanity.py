"""Step-1 sanity (CPU, read-only): ckpt5 moves_left forward smoke + npz target cap.

1. Load main_3 ckpt5, run full forward (with aux heads) on a few replayed
   positions from one squander game; decode moves_left under cap=512 and
   compare to true remaining decisions.
2. Inspect npz moves_left targets: keys, raw vs normalized, empirical cap.
"""
from __future__ import annotations

import json
import sys
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

import hexo_engine as engine  # noqa: E402
from hexo_engine.types import unpack_coord_id  # noqa: E402
from hexo_utils.records import HexoRecordFile  # noqa: E402

from dense_cnn_restnet.architecture import RestnetNetwork  # noqa: E402
from dense_cnn_restnet.losses import decode_binned_value  # noqa: E402
from dense_cnn_restnet.samples import sample_from_state  # noqa: E402
from dense_cnn_restnet.input import build_input_planes  # noqa: E402
from dense_cnn_restnet.d6 import Axial  # noqa: E402
from dense_cnn_restnet.constants import MOVES_LEFT_CAP  # noqa: E402

RUN = Path("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3")


def build_net() -> RestnetNetwork:
    return RestnetNetwork(
        in_channels=13, channels=96, blocks_type="R_R_R_T_R_R_T_R",
        attention_heads=4, mlp_ratio=2, embed_kernel_size=3,
        residual_conv="hex", attention_impl="sdpa", attention_scope="disk",
        dropout=0.0, short_term_value_horizons=(2, 6, 16), moves_left_head=True,
    ).eval()


def load_ckpt(epoch: int) -> RestnetNetwork:
    net = build_net()
    payload = torch.load(RUN / "checkpoints" / f"epoch_{epoch:06d}.pt",
                         map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("model"))
    missing, unexpected = net.load_state_dict(state, strict=False)
    bad = [k for k in missing if "disk" not in k and "relative_index" not in k]
    if bad:
        raise RuntimeError(f"missing keys: {bad[:8]}")
    print(f"ckpt{epoch} loaded; missing(non-buffer)={len(bad)} unexpected={len(unexpected)}")
    return net


def main():
    print(f"MOVES_LEFT_CAP (hexgt tree, in effect for main_3 training) = {MOVES_LEFT_CAP}")

    # ---- npz target inspection ----
    # marathon game from the squander corpus + one ep5 game
    for ep, gid in [(9, 60), (9, 56), (5, 0)]:
        p = RUN / "selfplay" / f"epoch_{ep:06d}_game_{gid:06d}.npz"
        if not p.exists():
            print(f"npz missing: {p}")
            continue
        z = np.load(p)
        keys = sorted(z.files)
        print(f"\nnpz ep{ep} game{gid}: keys={keys}")
        for k in keys:
            if "moves" in k or "turn" in k:
                a = z[k]
                print(f"  {k}: shape={a.shape} dtype={a.dtype} "
                      f"min={a.min():.4f} max={a.max():.4f} "
                      f"first5={np.asarray(a).ravel()[:5]} last5={np.asarray(a).ravel()[-5:]}")
        if "moves_left" in z.files and "turn_index" in z.files:
            ml, ti = z["moves_left"].astype(float), z["turn_index"].astype(float)
            n = len(ml)
            # if raw decisions: ml + turn_index should be ~constant = game decisions
            tot = ml + ti
            print(f"  rows={n}; ml+turn_index: min={tot.min():.1f} max={tot.max():.1f} "
                  f"median={np.median(tot):.1f}")
            print(f"  ml values at turn_index 0..4: {[float(ml[ti == t][0]) if (ti == t).any() else None for t in range(5)]}")

    # ---- forward smoke on real positions ----
    net = load_ckpt(5)
    with HexoRecordFile.open(RUN / "selfplay" / "epoch_000009.hxr") as rf:
        rec = next(r for r in rf.iter_records()
                   if getattr(r, "game_id") == "epoch-000009-selfplay-000060")
    action_ids = [int(a) for a in rec.action_ids]
    L = len(action_ids)
    print(f"\nsmoke game epoch-000009-selfplay-000060 len={L} winner={rec.winner}")
    test_plies = [10, 100, 250, 336, 500, L - 20]
    state = engine.new_game()
    planes = {}
    for ply, aid in enumerate(action_ids):
        if ply in test_plies:
            s = sample_from_state(state, game_id="t", turn_index=ply,
                                  policy=(), root_prior_policy=[(0, 1.0)])
            planes[ply] = build_input_planes(
                current_player=s.current_player, phase=s.phase, center=Axial(*s.center),
                stones=s.stones, legal_action_ids=s.legal_action_ids,
                placement_history=s.placement_history, first_stone=s.first_stone,
                own_hot=s.own_hot, opponent_hot=s.opponent_hot,
                opponent_last_turn=s.opponent_last_turn,
            )
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(aid)))
    xs = torch.stack([planes[p] for p in test_plies])
    out = net(xs)
    print("forward output keys:", sorted(out.keys()))
    v = decode_binned_value(out["value"].float()).numpy()
    mln = decode_binned_value(out["moves_left"].float()).numpy()
    ml_dec = (mln + 1.0) / 2.0 * MOVES_LEFT_CAP
    for i, p in enumerate(test_plies):
        true_rem = L - p
        print(f"  ply={p:4d} v_stm={v[i]:+.3f} ml_norm={mln[i]:+.4f} "
              f"ml_pred_decisions={ml_dec[i]:7.1f} true_remaining={true_rem}")
    print("DONE")


if __name__ == "__main__":
    main()
