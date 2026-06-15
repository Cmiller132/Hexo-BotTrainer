"""Verifier: fresh moves_left head audit on DIFFERENT games (CPU, read-only).

Independent re-derivation of three headline numbers using games NOT in the
original probe's length-stratified sample:
  - m3c7 on m3ep5: conv(<60) within-game Spearman (mean decode), [0,5) and
    [5,20) median-decode MAE, end(<20)-vs-mid([60,120)) pairwise acc.
  - m3c5 vs m4c12 on the same fresh m3ep5 games (flood before/after).

Sampling: decisive games sorted by length; EXCLUDE the 60 linspace indices the
probe used; draw 20 at random (seed 20260612) from the remainder, stratified
lightly by length quartile to keep mid/end buckets populated.
"""
from __future__ import annotations

import json
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
BATCH = 64
N_FRESH = 20
SEED = 20260612

CKPTS = [
    ("m3c5", "dense_cnn_restnet_main_3", 5),
    ("m3c7", "dense_cnn_restnet_main_3", 7),
    ("m4c12", "dense_cnn_restnet_main_4", 12),
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
    missing, unexpected = net.load_state_dict(state, strict=False)
    bad = [k for k in missing if "disk" not in k and "relative_index" not in k]
    if bad:
        raise RuntimeError(f"{run} ep{epoch} missing keys: {bad[:8]}")
    aux = [k for k in state if k.startswith("aux_value_reduction") or k.startswith("moves_left_head")]
    if len(aux) < 6:
        raise RuntimeError(f"{run} ep{epoch}: aux/moves_left keys absent: {aux}")
    return net


def fresh_games(run: str, epoch: int, n: int):
    with HexoRecordFile.open(RUNS / run / "selfplay" / f"epoch_{epoch:06d}.hxr") as rf:
        recs = [r for r in rf.iter_records() if r.winner is not None]
    recs.sort(key=lambda r: len(r.action_ids))
    used = set(int(round(i)) for i in np.linspace(0, len(recs) - 1, 60))
    avail = [i for i in range(len(recs)) if i not in used]
    rng = np.random.default_rng(SEED)
    # stratify: split avail into 4 quartile chunks, draw n/4 from each
    chunks = np.array_split(np.array(avail), 4)
    picks = []
    for ch in chunks:
        picks.extend(rng.choice(ch, size=n // 4, replace=False).tolist())
    return [recs[i] for i in sorted(picks)]


def game_planes(action_ids):
    state = engine.new_game()
    planes = []
    for ply, aid in enumerate(action_ids):
        s = sample_from_state(state, game_id="t", turn_index=ply,
                              policy=(), root_prior_policy=[(0, 1.0)])
        planes.append(build_input_planes(
            current_player=s.current_player, phase=s.phase, center=Axial(*s.center),
            stones=s.stones, legal_action_ids=s.legal_action_ids,
            placement_history=s.placement_history, first_stone=s.first_stone,
            own_hot=s.own_hot, opponent_hot=s.opponent_hot,
            opponent_last_turn=s.opponent_last_turn,
        ))
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(aid)))
    return planes


def predict_ml(net, planes):
    means, medians = [], []
    bins = torch.linspace(-1.0, 1.0, 65)
    for i in range(0, len(planes), BATCH):
        xs = torch.stack(planes[i:i + BATCH])
        out = net(xs)
        logits = out["moves_left"].float()
        norm = decode_binned_value(logits)
        means.append((norm + 1.0) * 0.5 * MOVES_LEFT_CAP)
        probs = torch.softmax(logits, dim=-1)
        cdf = probs.cumsum(dim=-1)
        med_idx = (cdf < 0.5).sum(dim=-1).clamp(max=64)
        medians.append((bins[med_idx] + 1.0) * 0.5 * MOVES_LEFT_CAP)
    return torch.cat(means).numpy(), torch.cat(medians).numpy()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main():
    t0 = time.time()
    recs = fresh_games("dense_cnn_restnet_main_3", 5, N_FRESH)
    lens = [len(r.action_ids) for r in recs]
    print(f"fresh m3ep5 games: n={len(recs)} len min/med/max={min(lens)}/{int(np.median(lens))}/{max(lens)}",
          flush=True)
    nets = {c: load_ckpt(run, ep) for c, run, ep in CKPTS}
    print(f"[{time.time()-t0:.0f}s] ckpts loaded", flush=True)

    per_game = []
    for gi, rec in enumerate(recs):
        aids = [int(a) for a in rec.action_ids]
        planes = game_planes(aids)
        row = {"L": len(aids)}
        for c in nets:
            row[c] = predict_ml(nets[c], planes)
        per_game.append(row)
        print(f"[{time.time()-t0:.0f}s] game {gi+1}/{len(recs)} len={len(aids)}", flush=True)

    print()
    for c, _, _ in CKPTS:
        conv_rho, pairwise, drop = [], [], []
        err_med_05, err_med_520, err_med_2060 = [], [], []
        for row in per_game:
            L = row["L"]
            true = L - 1 - np.arange(L, dtype=np.float64)
            mean_p, med_p = row[c]
            mean_p = np.asarray(mean_p, dtype=np.float64)
            med_p = np.asarray(med_p, dtype=np.float64)
            m = true < 60
            if m.sum() >= 10:
                conv_rho.append(spearman(mean_p[m], true[m]))
            end_m = true < 20
            mid_m = (true >= 60) & (true < 120)
            if end_m.sum() >= 5 and mid_m.sum() >= 5:
                e = mean_p[end_m][:, None]
                q = mean_p[mid_m][None, :]
                pairwise.append(float((e < q).mean()))
                drop.append(float(mean_p[-10:].mean() < mean_p[mid_m].mean()))
            err = med_p - true
            err_med_05.extend(err[(true >= 0) & (true < 5)].tolist())
            err_med_520.extend(err[(true >= 5) & (true < 20)].tolist())
            err_med_2060.extend(err[(true >= 20) & (true < 60)].tolist())
        e05 = np.abs(err_med_05).mean()
        e520 = np.abs(err_med_520).mean()
        e2060 = np.abs(err_med_2060).mean()
        print(f"{c}: conv<60 rho (mean dec) mean/med = {np.mean(conv_rho):.3f}/{np.median(conv_rho):.3f} "
              f"(n_games={len(conv_rho)})")
        print(f"     med-decode MAE [0,5)={e05:.1f} (n={len(err_med_05)})  "
              f"[5,20)={e520:.1f} (n={len(err_med_520)})  [20,60)={e2060:.1f} (n={len(err_med_2060)})")
        if pairwise:
            print(f"     end<20 vs mid[60,120) pairwise={np.mean(pairwise):.3f} "
                  f"(n_games={len(pairwise)}) end-drop={np.mean(drop):.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
