"""Read-only, CPU-only review of the dense_cnn_restnet value head.

Loads checkpoint(s), runs the network on stored self-play positions (compact
.npz shards expanded by the SAME Python encoder training uses), and reports:

  1. Calibration of the value head vs the true game outcome z (= sample.value,
     the hard +/-1 result from the side-to-move's perspective; soft_z_lambda=0).
  2. P0/P1 zero-sum / "optimism" probe: at FirstStone positions, evaluate the
     IDENTICAL board from both players' perspectives via a facts-level owner swap
     (current_player flip + own_hot<->opp_hot swap; opponent_last_turn ablated on
     BOTH sides so the comparison is symmetric). v(P0)+v(P1): 0 = zero-sum
     consistent, >0 = optimism (both sides think they win), <0 = pessimism.
  3. Per-color calibration / first-player-bias asymmetry.
  4. Auxiliary heads: moves_left and short-term value calibration.
  5. Multi-checkpoint trend.

Bypasses the package __init__.py (mid-merge in the working tree) via a sys.modules
stub. Forces CPU via CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import types
import glob
import math
import json
import time
import statistics as st
from dataclasses import replace
from pathlib import Path

import torch

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

PKG = "dense_cnn_restnet"
PKG_DIR = Path(r"E:/Hexo-BotTrainer-hexgt/packages/dense_cnn_restnet/python/dense_cnn_restnet")
if PKG not in sys.modules:
    stub = types.ModuleType(PKG)
    stub.__path__ = [str(PKG_DIR)]
    sys.modules[PKG] = stub

from dense_cnn_restnet.compact_io import read_compact_shard  # noqa: E402
from dense_cnn_restnet.input import build_input_planes  # noqa: E402
from dense_cnn_restnet.losses import decode_binned_value, value_bins, scalar_to_binned_target  # noqa: E402
from dense_cnn_restnet.architecture import RestnetNetwork  # noqa: E402
from dense_cnn_restnet.d6 import Axial  # noqa: E402
from dense_cnn_restnet.constants import MOVES_LEFT_CAP, VALUE_BINS  # noqa: E402
from torch.nn import functional as F  # noqa: E402

RUN = Path(r"E:/Hexo-BotTrainer/runs/dense_cnn_restnet_main1")
CKPT_DIR = RUN / "checkpoints"
SELFPLAY = RUN / "selfplay"
BATCH = 128
HORIZONS = (2, 6, 16)


def log(*a):
    print(*a, flush=True)


def build_net() -> RestnetNetwork:
    return RestnetNetwork(
        in_channels=13, channels=96, blocks_type="R_R_R_T_R_R_T_R",
        attention_heads=4, mlp_ratio=2, embed_kernel_size=3,
        residual_conv="hex", attention_impl="sdpa", attention_scope="disk",
        dropout=0.0, short_term_value_horizons=HORIZONS, moves_left_head=True,
    ).eval()


def load_ckpt(path: Path) -> RestnetNetwork:
    net = build_net()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("model"))
    missing, unexpected = net.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if "disk" not in k and "relative_index" not in k]
    if real_missing or unexpected:
        log(f"  [load WARN] missing={real_missing[:6]} unexpected={unexpected[:6]}")
    return net


def planes_of(sample, *, swap=False):
    cp = sample.current_player
    own_hot, opp_hot = sample.own_hot, sample.opponent_hot
    # opponent_last_turn is perspective-relative and its swapped replacement is
    # not stored, so ablate it on BOTH sides for a symmetric zero-sum comparison.
    last_turn = ()
    if swap:
        cp = "player1" if cp == "player0" else "player0"
        own_hot, opp_hot = opp_hot, own_hot
    return build_input_planes(
        current_player=cp, phase=sample.phase, center=Axial(*sample.center),
        stones=sample.stones, legal_action_ids=sample.legal_action_ids,
        placement_history=sample.placement_history, first_stone=sample.first_stone,
        own_hot=own_hot, opponent_hot=opp_hot, opponent_last_turn=last_turn,
    )


def planes_true(sample):
    """Faithful (un-ablated) encoding, exactly as training/inference sees it."""
    return build_input_planes(
        current_player=sample.current_player, phase=sample.phase,
        center=Axial(*sample.center), stones=sample.stones,
        legal_action_ids=sample.legal_action_ids,
        placement_history=sample.placement_history, first_stone=sample.first_stone,
        own_hot=sample.own_hot, opponent_hot=sample.opponent_hot,
        opponent_last_turn=sample.opponent_last_turn,
    )


def forward_values(net, planes_list, want_aux=False):
    """Batched forward -> dict of stacked head outputs (decoded where scalar)."""
    vparts, logit_parts = [], []
    ml_parts = []
    stv_parts = {h: [] for h in HORIZONS}
    for i in range(0, len(planes_list), BATCH):
        xs = torch.stack(planes_list[i:i + BATCH])
        out = net(xs) if want_aux else net.forward_policy_value(xs)
        vlog = out["value"].float()
        logit_parts.append(vlog)
        vparts.append(decode_binned_value(vlog))
        if want_aux:
            ml_parts.append(decode_binned_value(out["moves_left"].float()))
            for h in HORIZONS:
                stv_parts[h].append(decode_binned_value(out[f"stvalue_{h}"].float()))
    res = {"value": torch.cat(vparts), "value_logits": torch.cat(logit_parts)}
    if want_aux:
        res["moves_left"] = torch.cat(ml_parts)
        for h in HORIZONS:
            res[f"stvalue_{h}"] = torch.cat(stv_parts[h])
    return res


def load_rows(epochs, max_games_per_epoch=None):
    rows = []
    for ep in epochs:
        shards = sorted(SELFPLAY.glob(f"epoch_{ep:06d}_game_*.npz"))
        if max_games_per_epoch:
            shards = shards[:max_games_per_epoch]
        for sh in shards:
            try:
                rows.extend(read_compact_shard(sh))
            except Exception as e:
                log(f"  skip {sh.name}: {e}")
    return rows


def value_ce(logits, z):
    """Mean cross-entropy of 65-bin logits vs the soft 2-bin target for scalar z."""
    tgt = scalar_to_binned_target(torch.as_tensor(z, dtype=torch.float32))
    return float(-(tgt * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean())


def entropy_of_logits(logits):
    p = F.softmax(logits.float(), dim=-1)
    return float((-(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)).mean())


def pct(x, n):
    return f"{100*x/n:.1f}%" if n else "n/a"


# ----------------------------------------------------------------------------
# Pass A: calibration
# ----------------------------------------------------------------------------
def calibration(net, rows, label="epoch33", aux=True):
    planes = [planes_true(r) for r in rows]
    t0 = time.perf_counter()
    out = forward_values(net, planes, want_aux=aux)
    log(f"  [{label}] forward {len(rows)} positions in {time.perf_counter()-t0:.0f}s")
    v = out["value"]
    z = torch.tensor([r.value for r in rows])
    player = [r.current_player for r in rows]
    phase = [r.phase for r in rows]
    ml_tgt = torch.tensor([r.moves_left for r in rows])

    decisive = z.abs() > 0.5
    nd = int(decisive.sum())
    res = {"n": len(rows), "n_decisive": nd}
    err = (v - z)
    res["mae"] = float(err.abs().mean())
    res["rmse"] = float((err ** 2).mean() ** 0.5)
    res["mean_v"] = float(v.mean())
    res["mean_z"] = float(z.mean())
    res["mean_abs_v"] = float(v.abs().mean())
    res["sign_acc"] = float(((v.sign() == z.sign())[decisive]).float().mean())
    res["value_ce"] = value_ce(out["value_logits"], z)
    res["value_entropy_nats"] = entropy_of_logits(out["value_logits"])
    res["uniform_entropy_nats"] = math.log(VALUE_BINS)

    # reliability: bucket by predicted value, report realized mean z
    edges = [-1.01, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.01]
    rel = []
    for lo, hi in zip(edges, edges[1:]):
        m = (v >= lo) & (v < hi)
        c = int(m.sum())
        rel.append({"range": f"[{lo:+.1f},{hi:+.1f})", "n": c,
                    "mean_v": float(v[m].mean()) if c else None,
                    "mean_z": float(z[m].mean()) if c else None})
    res["reliability"] = rel

    # by closeness to terminal (moves_left target; -1 = truncated/masked)
    buckets = [(0, 5), (5, 15), (15, 40), (40, 80), (80, 1000)]
    bylen = []
    for lo, hi in buckets:
        m = (ml_tgt >= lo) & (ml_tgt < hi)
        c = int(m.sum())
        if not c:
            continue
        dd = decisive & m
        bylen.append({"moves_left": f"[{lo},{hi})", "n": c,
                      "mae": float(err[m].abs().mean()),
                      "mean_abs_v": float(v[m].abs().mean()),
                      "sign_acc": float(((v.sign() == z.sign())[dd]).float().mean()) if int(dd.sum()) else None,
                      "n_dec": int(dd.sum())})
    res["by_moves_left"] = bylen

    # by phase and by color
    def subset(mask):
        c = int(mask.sum())
        dd = decisive & mask
        return {"n": c, "mean_v": float(v[mask].mean()) if c else None,
                "mean_z": float(z[mask].mean()) if c else None,
                "mae": float(err[mask].abs().mean()) if c else None,
                "sign_acc": float(((v.sign() == z.sign())[dd]).float().mean()) if int(dd.sum()) else None,
                "n_dec": int(dd.sum())}
    ph = torch.tensor([p == "FirstStone" for p in phase])
    res["phase_first"] = subset(ph)
    res["phase_second"] = subset(~ph)
    p0 = torch.tensor([p == "player0" for p in player])
    res["player0"] = subset(p0)
    res["player1"] = subset(~p0)

    if not aux:
        return res

    # moves_left head calibration (decode [-1,1] -> [0, CAP]); mask -1 targets
    ml_pred01 = out["moves_left"]
    ml_pred = (ml_pred01 + 1.0) * 0.5 * MOVES_LEFT_CAP
    mlm = ml_tgt >= 0
    if int(mlm.sum()):
        mlt = ml_tgt[mlm].clamp(max=MOVES_LEFT_CAP)
        res["moves_left_mae"] = float((ml_pred[mlm] - mlt).abs().mean())
        res["moves_left_corr"] = float(torch.corrcoef(torch.stack([ml_pred[mlm], mlt]))[0, 1])
        res["moves_left_mean_pred"] = float(ml_pred[mlm].mean())
        res["moves_left_mean_tgt"] = float(mlt.mean())

    # short-term value head calibration vs stored stvalue targets (mask present)
    stv = {}
    for h in HORIZONS:
        tg, pr = [], []
        for i, r in enumerate(rows):
            d = dict(r.short_term_value)
            if h in d:
                tg.append(d[h]); pr.append(float(out[f"stvalue_{h}"][i]))
        if tg:
            tt = torch.tensor(tg); pp = torch.tensor(pr)
            stv[h] = {"n": len(tg), "mae": float((pp - tt).abs().mean()),
                      "mean_pred": float(pp.mean()), "mean_tgt": float(tt.mean()),
                      "sign_acc": float((pp.sign() == tt.sign()).float().mean())}
    res["stvalue"] = stv
    return res


# ----------------------------------------------------------------------------
# Pass B: zero-sum / optimism probe
# ----------------------------------------------------------------------------
def optimism(net, rows, label="epoch33", max_pos=4000):
    fs = [r for r in rows if r.phase == "FirstStone"][:max_pos]
    if not fs:
        return None
    pa = [planes_of(r, swap=False) for r in fs]
    pb = [planes_of(r, swap=True) for r in fs]
    t0 = time.perf_counter()
    va = forward_values(net, pa)["value"]
    vb = forward_values(net, pb)["value"]
    log(f"  [{label}] optimism on {len(fs)} FirstStone positions in {time.perf_counter()-t0:.0f}s")
    s = (va + vb)
    # v from each ABSOLUTE color's perspective (A is current_player; B is the other)
    a_is_p0 = torch.tensor([r.current_player == "player0" for r in fs])
    v_p0 = torch.where(a_is_p0, va, vb)
    v_p1 = torch.where(a_is_p0, vb, va)
    both_pos = int(((va > 0.05) & (vb > 0.05)).sum())
    return {
        "n": len(fs),
        "mean_sum": float(s.mean()), "median_sum": float(s.median()),
        "stdev_sum": float(s.std(unbiased=False)),
        "mean_va": float(va.mean()), "mean_vb": float(vb.mean()),
        "both_predict_win": both_pos, "both_predict_win_pct": both_pos / len(fs),
        "abs_sum_gt_0p5": int((s.abs() > 0.5).sum()),
        "mean_v_p0": float(v_p0.mean()), "mean_v_p1": float(v_p1.mean()),
        "first_player_bias": float(v_p0.mean() - v_p1.mean()),
    }


def fmt_cal(res):
    L = []
    L.append(f"N={res['n']} (decisive={res['n_decisive']})  "
             f"MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  "
             f"sign_acc={res['sign_acc']*100:.1f}%  value_CE={res['value_ce']:.3f}")
    L.append(f"mean v_pred={res['mean_v']:+.3f}  mean z={res['mean_z']:+.3f}  "
             f"mean|v|={res['mean_abs_v']:.3f}  "
             f"value entropy={res['value_entropy_nats']:.2f}/{res['uniform_entropy_nats']:.2f} nats")
    L.append("  reliability (pred bucket -> realized z):")
    for b in res["reliability"]:
        if b["n"]:
            L.append(f"    {b['range']}  n={b['n']:6d}  mean_v={b['mean_v']:+.3f}  mean_z={b['mean_z']:+.3f}")
    L.append("  by moves-left-to-end:")
    for b in res["by_moves_left"]:
        sa = f"{b['sign_acc']*100:.1f}%" if b["sign_acc"] is not None else "n/a"
        L.append(f"    ml={b['moves_left']:9s} n={b['n']:6d}  MAE={b['mae']:.3f}  "
                 f"mean|v|={b['mean_abs_v']:.3f}  sign_acc={sa} (n_dec={b['n_dec']})")
    for k in ("phase_first", "phase_second", "player0", "player1"):
        b = res[k]
        sa = f"{b['sign_acc']*100:.1f}%" if b["sign_acc"] is not None else "n/a"
        L.append(f"  {k:13s} n={b['n']:6d}  mean_v={b['mean_v']:+.3f}  mean_z={b['mean_z']:+.3f}  "
                 f"MAE={b['mae']:.3f}  sign_acc={sa}")
    if "moves_left_mae" in res:
        L.append(f"  moves_left head: MAE={res['moves_left_mae']:.1f} decisions  "
                 f"corr={res['moves_left_corr']:.3f}  "
                 f"mean_pred={res['moves_left_mean_pred']:.1f} mean_tgt={res['moves_left_mean_tgt']:.1f}")
    for h, b in res["stvalue"].items():
        L.append(f"  stvalue_{h}: n={b['n']} MAE={b['mae']:.3f} "
                 f"mean_pred={b['mean_pred']:+.3f} mean_tgt={b['mean_tgt']:+.3f} sign_acc={b['sign_acc']*100:.1f}%")
    return "\n".join(L)


def fmt_opt(o):
    return (f"N={o['n']} FirstStone positions (identical board, both perspectives)\n"
            f"  mean(vP0+vP1) = {o['mean_sum']:+.4f}   median = {o['median_sum']:+.4f}   "
            f"stdev = {o['stdev_sum']:.4f}\n"
            f"  (0 = zero-sum consistent; >0 = optimism, both think they win; <0 = pessimism)\n"
            f"  BOTH sides predict winning (vA>0.05 AND vB>0.05): {o['both_predict_win']}/{o['n']} "
            f"= {o['both_predict_win_pct']*100:.0f}%\n"
            f"  |vP0+vP1|>0.5: {o['abs_sum_gt_0p5']}/{o['n']}\n"
            f"  mean v(P0 perspective) = {o['mean_v_p0']:+.4f}   mean v(P1 perspective) = {o['mean_v_p1']:+.4f}\n"
            f"  first-player bias mean(vP0)-mean(vP1) = {o['first_player_bias']:+.4f}")


def main():
    out = {}
    log("=" * 78)
    log("VALUE HEAD REVIEW — dense_cnn_restnet_main1")
    log("=" * 78)

    # --- main pass: latest checkpoint on latest self-play epoch ---
    main_ckpt = CKPT_DIR / "epoch_000033.pt"
    log(f"\n## MAIN: {main_ckpt.name} on epoch-33 self-play\n")
    net = load_ckpt(main_ckpt)
    rows33 = load_rows([33])
    log(f"loaded {len(rows33)} positions from epoch-33 self-play")
    cal = calibration(net, rows33, "ep33")
    out["calibration_ep33"] = cal
    log("\n[CALIBRATION]")
    log(fmt_cal(cal))
    opt = optimism(net, rows33, "ep33", max_pos=6000)
    out["optimism_ep33"] = opt
    log("\n[ZERO-SUM / OPTIMISM]")
    log(fmt_opt(opt))

    # --- trend across checkpoints (fixed smaller set) ---
    log("\n## TREND across checkpoints (fixed sample)\n")
    trend_rows = load_rows([33], max_games_per_epoch=40)
    log(f"trend sample: {len(trend_rows)} positions "
        f"({sum(1 for r in trend_rows if r.phase=='FirstStone')} FirstStone)")
    trend = []
    for ep in (5, 12, 20, 24, 28, 32, 33):
        p = CKPT_DIR / f"epoch_{ep:06d}.pt"
        if not p.exists():
            continue
        n2 = load_ckpt(p)
        c = calibration(n2, trend_rows, f"ep{ep}", aux=False)
        o = optimism(n2, trend_rows, f"ep{ep}", max_pos=2000)
        row = {"epoch": ep, "mae": c["mae"], "sign_acc": c["sign_acc"],
               "value_ce": c["value_ce"], "mean_abs_v": c["mean_abs_v"],
               "opt_mean_sum": o["mean_sum"], "opt_both_win_pct": o["both_predict_win_pct"],
               "first_player_bias": o["first_player_bias"]}
        trend.append(row)
        log(f"  ep{ep:3d}: MAE={c['mae']:.3f} sign_acc={c['sign_acc']*100:.1f}% "
            f"value_CE={c['value_ce']:.3f} mean|v|={c['mean_abs_v']:.3f} | "
            f"optimism(vP0+vP1)={o['mean_sum']:+.4f} both_win={o['both_predict_win_pct']*100:.0f}% "
            f"fp_bias={o['first_player_bias']:+.3f}")
    out["trend"] = trend

    Path(r"E:/Hexo-BotTrainer-hexgt/scripts/_value_head_review_out.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    log("\nwrote scripts/_value_head_review_out.json")


if __name__ == "__main__":
    main()
