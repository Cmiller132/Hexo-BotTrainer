"""moves_left head quality audit (read-only, CPU).

Evaluates the moves_left auxiliary head of dense_cnn_restnet heads_v3 checkpoints
on replayed self-play games, by true-moves-left bucket.

Checkpoints: main_3 ep5, main_3 ep7, main_4 ep12 (all trained under MOVES_LEFT_CAP=512).
Game sources: main_3 ep5 (healthy), main_3 ep9 (marathon), main_4 ep12 (current).

Decode: pred_norm = decode_binned_value(out["moves_left"]) ; pred_ml = (pred_norm+1)/2*512
True:   true_ml at ply t of an L-decision game = L - t - 1   (decisions remaining AFTER
        this decision; matches samples.finalize / npz raw scalar convention).

Usage: python _wf_s3_mlaudit.py N_GAMES_PER_SOURCE OUT_JSON
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
BUCKETS = [(0, 5), (5, 20), (20, 60), (60, 120), (120, 10 ** 9)]
BUCKET_NAMES = ["[0,5)", "[5,20)", "[20,60)", "[60,120)", "[120+)"]

CKPTS = [
    ("m3c5", "dense_cnn_restnet_main_3", 5),
    ("m3c7", "dense_cnn_restnet_main_3", 7),
    ("m4c12", "dense_cnn_restnet_main_4", 12),
]
SOURCES = [
    ("m3ep5", "dense_cnn_restnet_main_3", 5),
    ("m3ep9", "dense_cnn_restnet_main_3", 9),
    ("m4ep12", "dense_cnn_restnet_main_4", 12),
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
    # fail-loud guard against the main_1 random-aux-head artifact
    aux = [k for k in state if k.startswith("aux_value_reduction") or k.startswith("moves_left_head")]
    if len(aux) < 6:
        raise RuntimeError(f"{run} ep{epoch}: aux/moves_left keys absent from checkpoint: {aux}")
    if unexpected:
        print(f"  NOTE {run} ep{epoch} unexpected keys: {unexpected[:6]}", flush=True)
    return net


def pick_games(run: str, epoch: int, n: int):
    with HexoRecordFile.open(RUNS / run / "selfplay" / f"epoch_{epoch:06d}.hxr") as rf:
        recs = [r for r in rf.iter_records() if r.winner is not None]
    recs.sort(key=lambda r: len(r.action_ids))
    if len(recs) <= n:
        return recs
    idx = sorted(set(int(round(i)) for i in np.linspace(0, len(recs) - 1, n)))
    return [recs[i] for i in idx]


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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def ols_slope(x: np.ndarray, y: np.ndarray):
    if len(x) < 3 or x.std() == 0:
        return None, None, None
    sl, ic = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(sl), float(ic), r


def main():
    n_games = int(sys.argv[1])
    out_json = sys.argv[2]
    t0 = time.time()

    nets = {}
    for cname, run, ep in CKPTS:
        nets[cname] = load_ckpt(run, ep)
        print(f"[{time.time()-t0:7.1f}s] loaded {cname} ({run} ep{ep})", flush=True)

    games = []  # raw per-game arrays for re-analysis
    for sname, run, ep in SOURCES:
        recs = pick_games(run, ep, n_games)
        lens = [len(r.action_ids) for r in recs]
        print(f"[{time.time()-t0:7.1f}s] {sname}: {len(recs)} games, len min/med/max = "
              f"{min(lens)}/{int(np.median(lens))}/{max(lens)}", flush=True)
        for gi, rec in enumerate(recs):
            action_ids = [int(a) for a in rec.action_ids]
            planes = game_planes(action_ids)
            row = {"source": sname, "game_len": len(action_ids), "winner": str(rec.winner)}
            for cname in nets:
                mean_ml, med_ml = predict_ml(nets[cname], planes)
                row[cname] = [round(float(p), 2) for p in mean_ml]
                row[cname + "_med"] = [round(float(p), 2) for p in med_ml]
            games.append(row)
            if (gi + 1) % 10 == 0 or gi == len(recs) - 1:
                print(f"[{time.time()-t0:7.1f}s] {sname} game {gi+1}/{len(recs)} len={len(action_ids)}", flush=True)

    # ---- aggregate ----
    agg = {"moves_left_cap": MOVES_LEFT_CAP, "n_games_per_source": n_games, "by": {}}
    for cname, _, _ in CKPTS:
        for sname, _, _ in SOURCES:
            gs = [g for g in games if g["source"] == sname]
            true_all, pred_all, med_all = [], [], []
            full_rho, conv_rho, mono1, mono5, conv_mono1 = [], [], [], [], []
            for g in gs:
                L = g["game_len"]
                t = np.arange(L, dtype=np.float64)
                true = L - 1 - t
                pred = np.asarray(g[cname], dtype=np.float64)
                true_all.append(true); pred_all.append(pred)
                med_all.append(np.asarray(g[cname + "_med"], dtype=np.float64))
                if L >= 10:
                    full_rho.append(spearman(pred, true))
                m = true < 60
                if m.sum() >= 10:
                    conv_rho.append(spearman(pred[m], true[m]))
                    pc = pred[m]
                    conv_mono1.append(float(np.mean(np.diff(pc) < 0)))
                d1 = np.diff(pred)
                mono1.append(float(np.mean(d1 < 0)))
                if L > 5:
                    mono5.append(float(np.mean(pred[5:] < pred[:-5])))
            true_all = np.concatenate(true_all); pred_all = np.concatenate(pred_all)
            med_all = np.concatenate(med_all)
            err = pred_all - true_all
            err_med = med_all - true_all
            buckets = {}
            for (lo, hi), bn in zip(BUCKETS, BUCKET_NAMES):
                m = (true_all >= lo) & (true_all < hi)
                if m.sum() == 0:
                    buckets[bn] = {"n": 0}
                    continue
                sl, ic, r = ols_slope(true_all[m], pred_all[m])
                buckets[bn] = {
                    "n": int(m.sum()),
                    "mae": round(float(np.abs(err[m]).mean()), 2),
                    "bias": round(float(err[m].mean()), 2),
                    "bias_median": round(float(np.median(err[m])), 2),
                    "pred_mean": round(float(pred_all[m].mean()), 2),
                    "pred_std": round(float(pred_all[m].std()), 2),
                    "slope": None if sl is None else round(sl, 3),
                    "pearson_r": None if r is None else round(r, 3),
                    "mae_med": round(float(np.abs(err_med[m]).mean()), 2),
                    "bias_med": round(float(err_med[m].mean()), 2),
                }
            sl, ic, r = ols_slope(true_all, pred_all)
            agg["by"][f"{cname}|{sname}"] = {
                "n_positions": int(len(true_all)),
                "n_games": len(gs),
                "overall": {
                    "mae": round(float(np.abs(err).mean()), 2),
                    "bias": round(float(err.mean()), 2),
                    "slope": round(sl, 3), "intercept": round(ic, 2), "pearson_r": round(r, 3),
                    "pred_min": round(float(pred_all.min()), 1),
                    "pred_max": round(float(pred_all.max()), 1),
                },
                "buckets": buckets,
                "within_game": {
                    "spearman_full_mean": round(float(np.mean(full_rho)), 3),
                    "spearman_full_median": round(float(np.median(full_rho)), 3),
                    "spearman_full_p10": round(float(np.percentile(full_rho, 10)), 3),
                    "spearman_conv_lt60_mean": round(float(np.mean(conv_rho)), 3) if conv_rho else None,
                    "spearman_conv_lt60_median": round(float(np.median(conv_rho)), 3) if conv_rho else None,
                    "spearman_conv_lt60_p10": round(float(np.percentile(conv_rho, 10)), 3) if conv_rho else None,
                    "n_games_conv": len(conv_rho),
                    "mono_adj_rate_mean": round(float(np.mean(mono1)), 3),
                    "mono_stride5_rate_mean": round(float(np.mean(mono5)), 3),
                    "mono_adj_rate_conv_lt60": round(float(np.mean(conv_mono1)), 3) if conv_mono1 else None,
                },
            }
            print(f"[{time.time()-t0:7.1f}s] aggregated {cname}|{sname}", flush=True)

    out = {"agg": agg, "games": games}
    Path(out_json).write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(agg, indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
