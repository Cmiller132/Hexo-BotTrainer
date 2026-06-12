"""Dawdle-discrimination probe (CPU, read-only).

At each of ~64 squander positions (main_3 ep9 marathons, from the archived
corpus) and ~16 healthy conversion controls (main_3 ep5 games, last 30 plies),
with the ckpt5 lens net:
  - enumerate top-K=12 prior candidates (+ the actually-played move),
  - apply each, evaluate the successor with the SAME net (full forward),
  - record v_leader(s') and predicted moves_left(s') (decisions, cap=512).

Outputs per-position candidate tables + aggregate metrics:
  (1) dawdle spread of ml among near-v candidates (within 0.05 of best v),
  (2) played-move rank under v vs under U_k = v - k*sign(v_base)*ml_norm,
  (3) flip rates (argmax_U != argmax_v) for squander vs control sets.
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

import hexo_engine as engine  # noqa: E402
from hexo_engine.types import unpack_coord_id  # noqa: E402
from hexo_utils.records import HexoRecordFile  # noqa: E402

from dense_cnn_restnet.architecture import RestnetNetwork  # noqa: E402
from dense_cnn_restnet.losses import decode_binned_value  # noqa: E402
from dense_cnn_restnet.samples import sample_from_state  # noqa: E402
from dense_cnn_restnet.input import build_input_planes, _flat_for_action_id  # noqa: E402
from dense_cnn_restnet.d6 import Axial  # noqa: E402
from dense_cnn_restnet.constants import MOVES_LEFT_CAP  # noqa: E402

RUN = Path("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3")
SCRIPTS = Path("/mnt/e/Hexo-BotTrainer-hexgt/scripts")
ARCHIVE = SCRIPTS / "archive"
K_TOP = 12
KS = [0.05, 0.1, 0.2, 0.5, 1.0]
NEAR_V = 0.05
BATCH = 64
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


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
    return net


def norm_player(p) -> str:
    return "player0" if "0" in str(p) else "player1"


def planes_for(state, ply):
    s = sample_from_state(state, game_id="t", turn_index=ply,
                          policy=(), root_prior_policy=[(0, 1.0)])
    x = build_input_planes(
        current_player=s.current_player, phase=s.phase, center=Axial(*s.center),
        stones=s.stones, legal_action_ids=s.legal_action_ids,
        placement_history=s.placement_history, first_stone=s.first_stone,
        own_hot=s.own_hot, opponent_hot=s.opponent_hot,
        opponent_last_turn=s.opponent_last_turn,
    )
    return s, x


def batched_forward(net, planes):
    """planes: list[tensor]; returns (v_stm, ml_decisions) arrays."""
    vs, mls = [], []
    for i in range(0, len(planes), BATCH):
        xs = torch.stack(planes[i:i + BATCH])
        out = net(xs)
        vs.append(decode_binned_value(out["value"].float()))
        mls.append(decode_binned_value(out["moves_left"].float()))
    v = torch.cat(vs).numpy()
    mln = torch.cat(mls).numpy()
    ml_dec = (mln + 1.0) / 2.0 * MOVES_LEFT_CAP
    return v, ml_dec


def probe_position(net, state, ply, leader, played_aid, tag):
    """state: engine state AT the decision ply (leader to move). Returns dict."""
    s, x_base = planes_for(state, ply)
    stm = norm_player(s.current_player)
    center = Axial(*s.center)
    legal = [int(a) for a in s.legal_action_ids]
    flats = {a: _flat_for_action_id(a, center=center) for a in legal}
    flats = {a: f for a, f in flats.items() if f is not None}

    out = net(x_base.unsqueeze(0))
    v_stm_base = float(decode_binned_value(out["value"].float())[0])
    ml_base = float((decode_binned_value(out["moves_left"].float())[0] + 1.0) / 2.0 * MOVES_LEFT_CAP)
    pol = out["policy"].float()[0].numpy()
    v_leader_base = v_stm_base if stm == leader else -v_stm_base

    logits = np.array([pol[flats[a]] for a in flats])
    e = np.exp(logits - logits.max())
    pri = e / e.sum()
    prior = dict(zip(flats.keys(), pri))
    cand_aids = [a for a, _ in sorted(prior.items(), key=lambda kv: -kv[1])[:K_TOP]]
    played_in_topk = played_aid in cand_aids
    if played_aid not in cand_aids and played_aid in legal:
        cand_aids.append(played_aid)

    cands, succ_planes, succ_idx = [], [], []
    for a in cand_aids:
        c = engine.clone_state(state)
        res = engine.apply_action(c, engine.PlacementAction(unpack_coord_id(int(a))))
        rec = {"aid": int(a), "prior": float(prior.get(a, 0.0)),
               "is_played": bool(a == played_aid), "terminal": bool(res.terminal)}
        if res.terminal:
            term = engine.terminal(c)
            w = norm_player(term.winner) if term.winner is not None else None
            rec["v_leader"] = 1.0 if w == leader else (-1.0 if w is not None else 0.0)
            rec["ml"] = 0.0
            rec["stm_next"] = None
        else:
            s2, x2 = planes_for(c, ply + 1)
            rec["stm_next"] = norm_player(s2.current_player)
            succ_idx.append(len(cands))
            succ_planes.append(x2)
        cands.append(rec)

    if succ_planes:
        v_stm, ml_dec = batched_forward(net, succ_planes)
        for j, ci in enumerate(succ_idx):
            r = cands[ci]
            r["v_leader"] = float(v_stm[j]) if r["stm_next"] == leader else -float(v_stm[j])
            r["ml"] = float(np.clip(ml_dec[j], 0.0, MOVES_LEFT_CAP))

    v_arr = np.array([c["v_leader"] for c in cands])
    ml_arr = np.array([c["ml"] for c in cands])
    sgn = 1.0 if v_leader_base >= 0 else -1.0  # leader winning at all probe points

    iv = int(np.argmax(v_arr))
    near = v_arr >= v_arr[iv] - NEAR_V
    spread = float(ml_arr[near].max() - ml_arr[near].min()) if near.sum() >= 2 else 0.0

    played_i = next((i for i, c in enumerate(cands) if c["is_played"]), None)
    rank_v = (1 + int(np.sum(v_arr > v_arr[played_i]))) if played_i is not None else None

    per_k = {}
    for k in KS:
        u = v_arr - k * sgn * (ml_arr / MOVES_LEFT_CAP)
        iu = int(np.argmax(u))
        per_k[str(k)] = {
            "argmax_changed": bool(iu != iv),
            "u_pick_ml_minus_v_pick_ml": float(ml_arr[iu] - ml_arr[iv]),
            "v_cost_of_u_pick": float(v_arr[iv] - v_arr[iu]),
            "u_pick_is_played": bool(cands[iu]["is_played"]),
            "played_rank_u": (1 + int(np.sum(u > u[played_i]))) if played_i is not None else None,
        }

    # within-position v-ml relation across candidates
    if len(cands) >= 3 and ml_arr.std() > 1e-9 and v_arr.std() > 1e-9:
        r_vml = float(np.corrcoef(v_arr, ml_arr)[0, 1])
    else:
        r_vml = None

    return {
        "tag": tag, "ply": int(ply), "leader": leader, "stm": stm,
        "n_legal": len(legal), "n_in_crop": len(flats),
        "n_cands": len(cands), "n_terminal_cands": int(sum(c["terminal"] for c in cands)),
        "played_in_topk": played_in_topk, "played_evaluated": played_i is not None,
        "v_leader_base": v_leader_base, "ml_base": ml_base,
        "best_v": float(v_arr[iv]), "n_near_v": int(near.sum()),
        "dawdle_spread_ml": spread,
        "ml_near_min": float(ml_arr[near].min()), "ml_near_max": float(ml_arr[near].max()),
        "played_rank_v": rank_v,
        "played_v_gap": (float(v_arr[iv] - v_arr[played_i]) if played_i is not None else None),
        "played_ml_minus_near_min": (float(ml_arr[played_i] - ml_arr[near].min())
                                     if played_i is not None else None),
        "corr_v_ml_cands": r_vml,
        "per_k": per_k,
        "cands": cands,
    }


def main():
    corpus = json.loads((ARCHIVE / "_wf_r4_hd_positions.json").read_text())
    events = corpus["events"]
    by_game = {}
    for ev in events:
        by_game.setdefault(ev["game_id"], []).append(ev)
    net = load_ckpt(5)
    log(f"ckpt5 loaded; {len(events)} squander events across {len(by_game)} games")

    results = []

    # ---- squander positions: replay ep9 marathons, clone at test plies ----
    with HexoRecordFile.open(RUN / "selfplay" / "epoch_000009.hxr") as rf:
        recs = {getattr(r, "game_id"): r for r in rf.iter_records()
                if getattr(r, "game_id") in by_game}
    for gid, evs in by_game.items():
        rec = recs[gid]
        action_ids = [int(a) for a in rec.action_ids]
        want = {ev["test_ply"]: ev for ev in evs}
        state = engine.new_game()
        for ply, aid in enumerate(action_ids):
            if ply in want:
                ev = want[ply]
                clone = engine.clone_state(state)
                res = probe_position(net, clone, ply, ev["leader"],
                                     int(ev["played_action_id_at_test_ply"]),
                                     tag="squander")
                res["game_id"] = gid
                res["game_len"] = len(action_ids)
                res["true_remaining"] = len(action_ids) - ply
                res["winner"] = str(rec.winner)
                res["leader_won"] = norm_player(rec.winner) == ev["leader"]
                results.append(res)
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(aid)))
        log(f"squander game {gid} len={len(action_ids)}: {len(evs)} positions done")

    # ---- healthy controls: ep5 median games, last 30 plies ----
    with HexoRecordFile.open(RUN / "selfplay" / "epoch_000005.hxr") as rf:
        recs5 = [r for r in rf.iter_records() if r.winner is not None]
    recs5.sort(key=lambda r: len(r.action_ids))
    mid = len(recs5) // 2
    ctrl_games = recs5[mid - 6: mid + 6]
    log(f"ep5: {len(recs5)} decisive games, median len="
        f"{len(recs5[mid].action_ids)}; using 8 median games")

    n_ctrl = 0
    for rec in ctrl_games:
        gid = getattr(rec, "game_id")
        action_ids = [int(a) for a in rec.action_ids]
        L = len(action_ids)
        winner = norm_player(rec.winner)
        lo = max(0, L - 30)
        # pass 1: eval v on last-30 plies where winner is to move
        state = engine.new_game()
        clones, metas = [], []
        for ply, aid in enumerate(action_ids):
            if lo <= ply <= L - 2:
                clones.append(engine.clone_state(state))
                metas.append(ply)
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(aid)))
        planes, stms = [], []
        for c, ply in zip(clones, metas):
            s, x = planes_for(c, ply)
            planes.append(x)
            stms.append(norm_player(s.current_player))
        v_stm, _ = batched_forward(net, planes)
        v_win = np.array([float(v) if st == winner else -float(v)
                          for v, st in zip(v_stm, stms)])
        qual, thr = [], None
        for t in (0.6, 0.5, 0.45):
            qual = [i for i in range(len(metas))
                    if stms[i] == winner and v_win[i] > t]
            if qual:
                thr = t
                break
        if not qual:
            log(f"control game {gid} len={L}: no qualifying ply (max v_win="
                f"{v_win.max():.3f}); skipped")
            continue
        picks = sorted({qual[0], qual[len(qual) // 2]})[:2]
        for i in picks:
            ply = metas[i]
            res = probe_position(net, clones[i], ply, winner,
                                 int(action_ids[ply]), tag="control")
            res["game_id"] = gid
            res["game_len"] = L
            res["true_remaining"] = L - ply
            res["winner"] = str(rec.winner)
            res["leader_won"] = True
            res["v_threshold_used"] = thr
            results.append(res)
            n_ctrl += 1
        log(f"control game {gid} len={L}: {len(picks)} positions "
            f"(plies {[metas[i] for i in picks]}, v_win "
            f"{[round(float(v_win[i]),3) for i in picks]}, thr={thr})")

    # ---- crash-safe raw dump before aggregation ----
    (SCRIPTS / "_wf_s3d_positions_raw.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    log(f"raw positions dumped ({len(results)} positions)")

    # ---- aggregate ----
    def agg(tagged):
        rows = [r for r in results if r["tag"] == tagged]
        n = len(rows)
        if not n:
            return {"n": 0}
        spread = np.array([r["dawdle_spread_ml"] for r in rows])
        out = {
            "n": n,
            "n_games": len({r["game_id"] for r in rows}),
            "dawdle_spread_ml_median": float(np.median(spread)),
            "dawdle_spread_ml_iqr": [float(np.percentile(spread, 25)),
                                     float(np.percentile(spread, 75))],
            "dawdle_spread_ml_max": float(spread.max()),
            "n_near_v_median": float(np.median([r["n_near_v"] for r in rows])),
            "played_rank_v_hist": {},
            "played_is_v_argmax_rate": float(np.mean([r["played_rank_v"] == 1 for r in rows
                                                      if r["played_rank_v"] is not None])),
            "corr_v_ml_median": float(np.median([r["corr_v_ml_cands"] for r in rows
                                                 if r["corr_v_ml_cands"] is not None])),
            "spearman_mlbase_trueremaining": None,
            "per_k": {},
        }

        def _spearman(a, b):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
                return None
            ra = np.argsort(np.argsort(a)).astype(float)
            rb = np.argsort(np.argsort(b)).astype(float)
            return float(np.corrcoef(ra, rb)[0, 1])

        out["spearman_mlbase_trueremaining"] = _spearman(
            [r["ml_base"] for r in rows], [r["true_remaining"] for r in rows])
        for r in rows:
            k = str(r["played_rank_v"])
            out["played_rank_v_hist"][k] = out["played_rank_v_hist"].get(k, 0) + 1
        for k in KS:
            ks = str(k)
            pk = [r["per_k"][ks] for r in rows]
            flips = [p["argmax_changed"] for p in pk]
            dml = np.array([p["u_pick_ml_minus_v_pick_ml"] for p in pk])
            vc = np.array([p["v_cost_of_u_pick"] for p in pk])
            pr_u = [p["played_rank_u"] for p in pk if p["played_rank_u"] is not None]
            pr_v = [r["played_rank_v"] for r in rows if r["played_rank_v"] is not None]
            # demote = played was v-argmax but not U-argmax
            demote = [1 for r in rows
                      if r["played_rank_v"] == 1 and r["per_k"][ks]["played_rank_u"] not in (None, 1)]
            out["per_k"][ks] = {
                "flip_rate": float(np.mean(flips)),
                "n_flips": int(np.sum(flips)),
                "flips_reduce_ml": int(np.sum(dml[np.array(flips)] < 0)) if any(flips) else 0,
                "median_ml_reduction_on_flip": (float(np.median(-dml[np.array(flips)]))
                                                if any(flips) else None),
                "median_v_cost_on_flip": (float(np.median(vc[np.array(flips)]))
                                          if any(flips) else None),
                "played_rank_u_mean": float(np.mean(pr_u)) if pr_u else None,
                "played_rank_v_mean": float(np.mean(pr_v)) if pr_v else None,
                "played_demoted_from_top": int(len(demote)),
                "played_demoted_base": int(sum(1 for r in rows if r["played_rank_v"] == 1)),
            }
        return out

    summary = {
        "ckpt_epoch": 5, "moves_left_cap": MOVES_LEFT_CAP,
        "k_values": KS, "near_v_window": NEAR_V, "k_top": K_TOP,
        "squander": agg("squander"),
        "control": agg("control"),
    }
    # ply buckets for squander flips
    for lo, hi in [(200, 260), (260, 300), (300, 340), (340, 400)]:
        rows = [r for r in results if r["tag"] == "squander" and lo <= r["ply"] < hi]
        if rows:
            summary.setdefault("squander_by_ply", {})[f"[{lo},{hi})"] = {
                "n": len(rows),
                "spread_median": float(np.median([r["dawdle_spread_ml"] for r in rows])),
                "flip_rate_k0.1": float(np.mean([r["per_k"]["0.1"]["argmax_changed"] for r in rows])),
                "flip_rate_k0.2": float(np.mean([r["per_k"]["0.2"]["argmax_changed"] for r in rows])),
            }

    out = {"summary": summary, "positions": results}
    (SCRIPTS / "_wf_s3d_probe_out.json").write_text(json.dumps(out, indent=1),
                                                    encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    log("DONE")


if __name__ == "__main__":
    main()
