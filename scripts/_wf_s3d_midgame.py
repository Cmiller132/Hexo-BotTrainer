"""Supplemental: mid-game healthy collateral probe (ep5 games, plies 40/60).

If a moves-left utility were applied UNGATED (no |v| threshold), would it flip
one-ply choices in ordinary healthy mid-game positions? Reuses probe_position
from _wf_s3d_probe. Leader := side to move (sign(v_base) decides utility sign
inside probe_position, matching ungated deployment).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/scripts")
import _wf_s3d_probe as P  # noqa: E402  (sets up PKG stub on import)

import numpy as np  # noqa: E402
import hexo_engine as engine  # noqa: E402
from hexo_engine.types import unpack_coord_id  # noqa: E402
from hexo_utils.records import HexoRecordFile  # noqa: E402

PLIES = (40, 60)


def main():
    net = P.load_ckpt(5)
    with HexoRecordFile.open(P.RUN / "selfplay" / "epoch_000005.hxr") as rf:
        recs5 = [r for r in rf.iter_records() if r.winner is not None]
    recs5.sort(key=lambda r: len(r.action_ids))
    mid = len(recs5) // 2
    games = recs5[mid - 26: mid - 14]  # 12 games, disjoint from main controls
    results = []
    for rec in games:
        gid = getattr(rec, "game_id")
        action_ids = [int(a) for a in rec.action_ids]
        L = len(action_ids)
        state = engine.new_game()
        for ply, aid in enumerate(action_ids):
            if ply in PLIES:
                clone = engine.clone_state(state)
                # leader := side to move; probe uses sign(v_leader_base) for U
                import dense_cnn_restnet.samples as _s  # noqa
                s, _ = P.planes_for(clone, ply)
                stm = P.norm_player(s.current_player)
                res = P.probe_position(net, clone, ply, stm, aid, tag="midgame")
                res["game_id"] = gid
                res["game_len"] = L
                res["true_remaining"] = L - ply
                results.append(res)
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(aid)))
        P.log(f"midgame {gid} len={L} done")

    out = {"n": len(results)}
    out["v_leader_base_absmedian"] = float(np.median([abs(r["v_leader_base"]) for r in results]))
    out["v_spread_median"] = float(np.median(
        [max(c["v_leader"] for c in r["cands"]) - min(c["v_leader"] for c in r["cands"])
         for r in results]))
    out["n_near_v_median"] = float(np.median([r["n_near_v"] for r in results]))
    out["dawdle_spread_ml_median"] = float(np.median([r["dawdle_spread_ml"] for r in results]))
    for k in ["0.05", "0.1", "0.2", "0.5", "1.0"]:
        flips = [r["per_k"][k]["argmax_changed"] for r in results]
        out[f"flip_rate_k{k}"] = float(np.mean(flips))
        out[f"n_flips_k{k}"] = int(np.sum(flips))
    Path("/mnt/e/Hexo-BotTrainer-hexgt/scripts/_wf_s3d_midgame_out.json").write_text(
        json.dumps({"summary": out, "positions": results}, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    P.log("DONE")


if __name__ == "__main__":
    main()
