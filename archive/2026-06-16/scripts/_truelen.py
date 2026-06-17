"""Reconcile TRUE game length (full .hxr records) vs the shard-based metric.
Read-only. Post-fix epochs (4+) have real full .hxr; compare per-game .hxr action
count (moves-to-terminal) against the shard's rows/game + deepest placement_history."""
from __future__ import annotations
import statistics as st
from pathlib import Path
import hexo_engine.api as eng
from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.expand import reconstruct_state
from hexo_runner.records import HexoRecordFile

SP = Path("runs/hexgt_rl_main2/selfplay")


def dist(x):
    x = sorted(x); n = len(x)
    return f"n={n} min={x[0]} med={x[n//2]} mean={sum(x)/n:.1f} max={x[-1]}"


def analyze_epoch(ep):
    hxr = SP / f"epoch_{ep:06d}.hxr"
    if not hxr.exists():
        print(f"epoch {ep}: no full .hxr (pre-fix)"); return
    recs = list(HexoRecordFile.open(hxr).iter_records())
    # TRUE length = .hxr action count, by game_id. Also is it terminal?
    hxr_len = {}
    term = 0
    for r in recs:
        idx = int(r.game_id.rsplit("-", 1)[-1])
        hxr_len[idx] = len(list(r.action_ids))
        s = eng.new_game()
        from hexo_engine import PlacementAction
        from hexo_engine.types import unpack_coord_id
        for aid in r.action_ids:
            eng.apply_action(s, PlacementAction(unpack_coord_id(int(aid))))
        if eng.terminal(s) is not None:
            term += 1
    # Shard rows + deepest placement_history, matched by game index.
    rows_len, deep_len, gaps = {}, {}, []
    for npz in SP.glob(f"epoch_{ep:06d}_game_*.npz"):
        idx = int(npz.stem.rsplit("_", 1)[-1])
        try:
            rs = read_compact_shard(npz)
        except Exception:
            continue
        if not rs:
            continue
        rows_len[idx] = len(rs)
        deep_len[idx] = max(len(r.placement_history) for r in rs)
    common = sorted(set(hxr_len) & set(deep_len))
    for i in common:
        gaps.append(hxr_len[i] - deep_len[i])
    print(f"\n===== RL epoch {ep} ({len(recs)} full games; {term}/{len(recs)} reach terminal) =====")
    print(f"  TRUE length (.hxr moves-to-terminal): {dist(list(hxr_len.values()))}")
    print(f"  shard rows/game                     : {dist(list(rows_len.values()))}")
    print(f"  shard deepest placement_history     : {dist(list(deep_len.values()))}")
    if gaps:
        print(f"  per-game gap (.hxr - shard_deepest) : {dist(gaps)}  <-- moves the shard omits")


def main():
    eps = sorted({int(p.stem.split("_")[1]) for p in SP.glob("epoch_*.hxr")})
    print("epochs with full .hxr:", eps)
    for ep in eps:
        analyze_epoch(ep)


if __name__ == "__main__":
    main()
