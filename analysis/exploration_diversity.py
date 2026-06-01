"""Quantify opening / exploration diversity in dense_cnn self-play games.

Two data sources are used, each for what it is reliable at:

  * ``.hxr`` records (``epoch_0000NN.hxr``) hold the EXACT realized played
    action sequence per game, decoded to global axial coords via
    ``hexo_engine.types.unpack_coord_id``.  This is the canonical move log and
    is used for OPENING DIVERSITY.  Caveat: the ``.hxr`` for this run only
    stores a *subset* of the games (verified: 69 of 512 for epoch 1, 81 for the
    partial epoch 2) -- it is not the full 512.  Diversity is therefore reported
    over the recorded subset.

  * per-game ``.npz`` shards hold the post-MCTS visit policy
    (``policyTargetsNCHW``) and the root prior (``rootPolicyNCHW``) as 41x41
    planes, plus a legal mask and ``metadataInputNC`` (col 0 == turn index).
    These cover ALL 512 games and are used for SEARCH PEAKEDNESS.  The plane is
    in a per-position (and D6-augmented) crop frame, so its argmax does NOT
    equal the played move -- verified empirically -- hence the npz is NOT used
    to reconstruct global move sequences.  Entropy / top-1 mass are frame
    invariant, so peakedness is sound.

Run (PowerShell):
  $env:PYTHONPATH = (('hexo_engine','hexo_utils','hexo_runner','hexo_train','hexo_models','hexo_frontend') | ForEach-Object { "E:\\Hexo-BotTrainer\\packages\\$_\\python" }) -join ';'
  C:\\Python314\\python.exe E:\\Hexo-BotTrainer\\analysis\\exploration_diversity.py
"""

from __future__ import annotations

import glob
import math
import os
import re
from collections import Counter

import numpy as np

from hexo_engine.types import unpack_coord_id
from hexo_utils.records import HexoRecordFile

SELFPLAY = r"E:/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6/selfplay"
PREFIX_KS = (1, 2, 3, 5, 10)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def shannon_bits(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def entropy_nats_from_probs(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# OPENING DIVERSITY from .hxr (canonical realized moves)
# ---------------------------------------------------------------------------
def load_hxr_games(epoch: int):
    """Return list of games, each a tuple of (q, r) coords in play order."""
    path = os.path.join(SELFPLAY, f"epoch_{epoch:06d}.hxr")
    if not os.path.exists(path):
        return []
    try:
        f = HexoRecordFile.open(path)
        recs = list(f.iter_records())
    except Exception as exc:  # pragma: no cover - robustness
        print(f"  [skip] could not read {path}: {exc}")
        return []
    games = []
    for r in recs:
        try:
            coords = tuple((c.q, c.r) for c in (unpack_coord_id(a) for a in r.action_ids))
        except Exception:
            continue
        if coords:
            games.append(coords)
    return games


def opening_diversity(games, epoch: int):
    n = len(games)
    print(f"\n{'='*78}")
    print(f"OPENING DIVERSITY  epoch {epoch}   (source: .hxr realized moves, n={n} games)")
    print(f"{'='*78}")
    if n == 0:
        print("  no games")
        return

    # per-move distributions for moves 1,2,3
    for mv in (1, 2, 3):
        vals = [g[mv - 1] for g in games if len(g) >= mv]
        c = Counter(vals)
        print(f"\n  move {mv}: {len(c)} distinct value(s) over {len(vals)} games")
        for coord, cnt in c.most_common(6):
            print(f"      {coord}: {cnt}/{len(vals)}")
        if len(c) > 6:
            print(f"      ... (+{len(c)-6} more)")

    # distinct k-move prefixes
    print("\n  distinct k-move opening prefixes:")
    prefix_counters = {}
    for k in PREFIX_KS:
        prefixes = [g[:k] for g in games if len(g) >= k]
        c = Counter(prefixes)
        prefix_counters[k] = (c, len(prefixes))
        print(f"      k={k:>2}: {len(c):>4} distinct  over {len(prefixes)} games")

    # fraction sharing most-common 5-move prefix
    c5, n5 = prefix_counters[5]
    if n5:
        top_prefix, top_cnt = c5.most_common(1)[0]
        print(
            f"\n  most-common 5-move prefix shared by {top_cnt}/{n5} "
            f"games = {100*top_cnt/n5:.1f}%"
        )

    # entropy of move-1 distribution and 5-move-prefix distribution
    mv1 = Counter(g[0] for g in games if len(g) >= 1)
    h_mv1 = shannon_bits(list(mv1.values()))
    # max possible for move 1: number of distinct legal opening moves is 1
    # (origin only) per engine; fall back to log2(n) reference
    print("\n  entropy (bits):")
    print(
        f"      move-1 distribution : {h_mv1:.3f} bits "
        f"({len(mv1)} distinct;  max if uniform over distinct = {math.log2(len(mv1)) if len(mv1)>1 else 0.0:.3f})"
    )
    if n5:
        h_p5 = shannon_bits(list(c5.values()))
        print(
            f"      5-move-prefix distr : {h_p5:.3f} bits "
            f"({len(c5)} distinct;  max if all {n5} games unique = {math.log2(n5):.3f})"
        )

    # smoking-gun: share with identical move 1
    top1_coord, top1_cnt = mv1.most_common(1)[0]
    print(
        f"\n  SEED CHECK -- single most-common move 1 = {top1_coord} "
        f"played in {top1_cnt}/{sum(mv1.values())} games "
        f"= {100*top1_cnt/sum(mv1.values()):.1f}%"
    )

    # cheap pairwise longest-common-opening-prefix
    pairwise_lcp(games)


def pairwise_lcp(games, sample=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(games)
    if n < 2:
        return
    pairs = min(sample, n * (n - 1) // 2)
    total = 0
    for _ in range(pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        a, b = games[i], games[j]
        lcp = 0
        for x, y in zip(a, b):
            if x == y:
                lcp += 1
            else:
                break
        total += lcp
    print(f"  avg pairwise longest-common-opening-prefix over {pairs} random pairs: {total/pairs:.2f} moves")


# ---------------------------------------------------------------------------
# SEARCH PEAKEDNESS from .npz visit policies (all games)
# ---------------------------------------------------------------------------
def collect_peakedness(epoch: int, opening_max_turn=3, mid_min_turn=12, per_file_cap=None):
    """Gather visit-policy stats for opening vs mid-game rows across many games."""
    files = sorted(glob.glob(os.path.join(SELFPLAY, f"epoch_{epoch:06d}_game_*.npz")))
    opening = []  # list of (eff_moves, top1_mass, n_gt1pct)
    midgame = []
    files_read = 0
    for p in files:
        try:
            z = np.load(p)
            pt = z["policyTargetsNCHW"]
            md = z["metadataInputNC"]
        except Exception:
            continue
        files_read += 1
        turns = md[:, 0].astype(int)
        N = pt.shape[0]
        flat = pt.reshape(N, -1)
        for j in range(N):
            t = int(turns[j])
            row = flat[j].astype(np.float64)
            s = row.sum()
            if s <= 0:
                continue
            p_ = row / s
            ent = entropy_nats_from_probs(p_)
            eff = math.exp(ent)
            top1 = float(p_.max())
            n_gt = int((p_ > 0.01).sum())
            rec = (eff, top1, n_gt)
            if t <= opening_max_turn:
                opening.append(rec)
            elif t >= mid_min_turn:
                midgame.append(rec)
    return opening, midgame, files_read, len(files)


def report_peakedness(label, rows):
    if not rows:
        print(f"  {label}: no rows")
        return
    arr = np.array(rows, dtype=np.float64)
    eff, top1, ngt = arr[:, 0], arr[:, 1], arr[:, 2]
    print(
        f"  {label}  (n={len(rows)} positions):\n"
        f"      effective #moves exp(H): mean {eff.mean():.2f}  median {np.median(eff):.2f}  "
        f"p10 {np.percentile(eff,10):.2f}  p90 {np.percentile(eff,90):.2f}\n"
        f"      top-1 visit mass       : mean {top1.mean():.3f}  median {np.median(top1):.3f}  "
        f"p10 {np.percentile(top1,10):.3f}  p90 {np.percentile(top1,90):.3f}\n"
        f"      #moves > 1% mass       : mean {ngt.mean():.1f}  median {np.median(ngt):.1f}"
    )


def root_prior_peakedness(epoch: int, opening_max_turn=3, max_files=200):
    """Compare ROOT PRIOR (network policy) peakedness at the opening vs visit."""
    files = sorted(glob.glob(os.path.join(SELFPLAY, f"epoch_{epoch:06d}_game_*.npz")))[:max_files]
    prior_eff, visit_eff = [], []
    for p in files:
        try:
            z = np.load(p)
            rp = z["rootPolicyNCHW"]
            pt = z["policyTargetsNCHW"]
            md = z["metadataInputNC"]
        except Exception:
            continue
        turns = md[:, 0].astype(int)
        for j in range(rp.shape[0]):
            if int(turns[j]) > opening_max_turn:
                continue
            for src, acc in ((rp, prior_eff), (pt, visit_eff)):
                row = src[j].reshape(-1).astype(np.float64)
                s = row.sum()
                if s <= 0:
                    continue
                acc.append(math.exp(entropy_nats_from_probs(row / s)))
    if prior_eff and visit_eff:
        print(
            f"  opening ROOT PRIOR  eff#moves: mean {np.mean(prior_eff):.2f} (n={len(prior_eff)})\n"
            f"  opening VISIT POLICY eff#moves: mean {np.mean(visit_eff):.2f} (n={len(visit_eff)})"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def epoch_present(epoch: int) -> bool:
    return bool(glob.glob(os.path.join(SELFPLAY, f"epoch_{epoch:06d}_game_*.npz"))) or os.path.exists(
        os.path.join(SELFPLAY, f"epoch_{epoch:06d}.hxr")
    )


def main():
    print("Self-play exploration / opening diversity analysis")
    print(f"selfplay dir: {SELFPLAY}")
    print("Data sources: .hxr = realized moves (subset of games); .npz = visit policies (all games)")

    epochs = [e for e in (1, 2) if epoch_present(e)]
    print(f"epochs present: {epochs}")

    for ep in epochs:
        n_npz = len(glob.glob(os.path.join(SELFPLAY, f"epoch_{ep:06d}_game_*.npz")))
        games = load_hxr_games(ep)
        print(f"\n### EPOCH {ep}:  {n_npz} per-game NPZ shards, {len(games)} games in .hxr")

        opening_diversity(games, ep)

        print(f"\n{'-'*78}")
        print(f"SEARCH PEAKEDNESS  epoch {ep}   (source: .npz post-MCTS visit policy, ALL games)")
        print(f"{'-'*78}")
        opening, midgame, read, total = collect_peakedness(ep)
        print(f"  read {read}/{total} npz files (turn<=3 = opening, turn>=12 = mid-game)")
        report_peakedness("OPENING positions (turn<=3)", opening)
        report_peakedness("MID-GAME positions (turn>=12)", midgame)
        print("  prior-vs-visit at opening (first %d files):" % 200)
        root_prior_peakedness(ep)

    print(f"\n{'='*78}\nDONE\n{'='*78}")


if __name__ == "__main__":
    main()
