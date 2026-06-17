"""Read-only self-play game analysis across RL epochs: tests the "found an early
ending strategy not yet countered" hypothesis. Uses the per-game .npz shards
(sampled positions + winner-via-value); does NOT touch the run."""
from __future__ import annotations
import math, statistics as st
from collections import Counter
from pathlib import Path
import numpy as np
from hexo_models.dense_cnn.compact_io import read_compact_shard

RUN = Path("runs/hexgt_rl_main2/selfplay")


def entropy(probs):
    probs = [p for p in probs if p > 0]
    return -sum(p * math.log(p) for p in probs) if probs else 0.0


def game_records(epoch):
    for npz in sorted(RUN.glob(f"epoch_{epoch:06d}_game_*.npz")):
        try:
            rows = read_compact_shard(npz)
        except Exception:
            continue
        if rows:
            yield rows


def analyze(epoch):
    lengths, p0wins, openings, open3, dec = [], 0, Counter(), Counter(), 0
    priorH, visitH, top1v, effmoves = [], [], [], []
    n = 0
    for rows in game_records(epoch):
        n += 1
        deep = max(rows, key=lambda r: len(r.placement_history))
        L = len(deep.placement_history)
        lengths.append(L)
        val = float(deep.value)
        if abs(val) > 0.5:
            dec += 1
        winner = deep.current_player if val > 0 else ("player1" if deep.current_player == "player0" else "player0")
        if winner == "player0":
            p0wins += 1
        ordered = sorted(deep.placement_history, key=lambda h: h[4])
        mv = [(int(e[0]), int(e[1])) for e in ordered]
        if mv:
            openings[mv[0]] += 1
            open3[tuple(mv[:3])] += 1
        # per-position exploration (visit policy = row.policy, prior = root_prior_policy)
        for r in rows:
            pol = list(r.policy)
            if pol:
                ws = [float(w) for _a, w in pol]
                s = sum(ws) or 1.0
                ps = [w / s for w in ws]
                visitH.append(entropy(ps)); top1v.append(max(ps)); effmoves.append(math.exp(entropy(ps)))
            pri = list(r.root_prior_policy)
            if pri:
                ws = [float(w) for _a, w in pri]; s = sum(ws) or 1.0
                priorH.append(entropy([w / s for w in ws]))
    return dict(
        epoch=epoch, n=n,
        len_min=min(lengths), len_med=int(st.median(lengths)), len_mean=round(st.mean(lengths), 1),
        len_max=max(lengths), pct_short=round(100 * sum(1 for L in lengths if L < 40) / n, 1),
        pct_vshort=round(100 * sum(1 for L in lengths if L < 30) / n, 1),
        p0_winrate=round(100 * p0wins / n, 1), decisive=round(100 * dec / n, 1),
        open_distinct=len(openings), open_top1_share=round(100 * openings.most_common(1)[0][1] / n, 1),
        open_top5_share=round(100 * sum(c for _, c in openings.most_common(5)) / n, 1),
        open_entropy=round(entropy([c / n for c in openings.values()]), 2),
        open3_distinct=len(open3), open3_top1_share=round(100 * open3.most_common(1)[0][1] / n, 1),
        open3_top3_share=round(100 * sum(c for _, c in open3.most_common(3)) / n, 1),
        priorH=round(st.mean(priorH), 2) if priorH else None,
        visitH=round(st.mean(visitH), 2) if visitH else None,
        top1_visit=round(st.mean(top1v), 3) if top1v else None,
        eff_moves=round(st.mean(effmoves), 1) if effmoves else None,
        n_priorH=len(priorH), n_visitH=len(visitH),
        lengths=lengths, openings=openings, open3=open3,
    )


def main():
    res = [analyze(e) for e in (0, 1, 2)]
    keys = ["epoch", "n", "len_min", "len_med", "len_mean", "len_max", "pct_short", "pct_vshort",
            "decisive", "p0_winrate", "open_distinct", "open_top1_share", "open_top5_share",
            "open_entropy", "open3_distinct", "open3_top1_share", "open3_top3_share",
            "priorH", "visitH", "top1_visit", "eff_moves"]
    print(f"{'metric':<18} " + " ".join(f"{'ep'+str(r['epoch']):>10}" for r in res))
    for k in keys[2:]:
        print(f"{k:<18} " + " ".join(f"{str(r[k]):>10}" for r in res))
    # length histogram (bimodality) per epoch
    print("\nLENGTH HISTOGRAM (buckets of 20):")
    for r in res:
        h = Counter(min(L // 20, 7) for L in r["lengths"])
        bars = " ".join(f"{b*20}-{b*20+19}:{h.get(b,0)}" for b in range(8))
        print(f"  ep{r['epoch']}: {bars}")
    # short-game concentration test (the exploit signature)
    print("\nSHORT GAMES (<40 moves) — opening + winner concentration:")
    for r in res:
        short = [L for L in r["lengths"] if L < 40]
        print(f"  ep{r['epoch']}: {len(short)} short games | top opening3 share among ALL = {r['open3_top3_share']}% (distinct {r['open3_distinct']}/{r['n']})")
    # top recurring opening-3 sequences in latest epoch
    print("\nTOP RECURRING 3-MOVE OPENINGS (ep2):")
    for seq, c in res[-1]["open3"].most_common(6):
        print(f"  {c:>3}x  {seq}")


if __name__ == "__main__":
    main()
