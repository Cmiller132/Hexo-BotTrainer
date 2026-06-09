"""Read-only exploration diagnostics for the hexgt RL self-play data (no GPU).

Measures, BY GAME PHASE (opening<15 / mid 15-60 / end>=60 plies):
  1. CONCENTRATION of the policy PRIOR, the post-search VISIT distribution, and
     the temperature-adjusted SELECTION distribution: top-1, top-5, effective moves.
  2. REALIZED exploration: fraction of actually-played moves that are NON-ARGMAX
     of the visit distribution (literal "did sampling pick a non-top move").
  3. DIVERSITY across games: unique first moves + first-move entropy, unique
     opening 6-ply lines, and the position-revisit (transposition) rate.
  5. TEMPERATURE tracking: selection entropy vs ply vs the schedule.

Source = a sample of self-play .npz shards (visit policy + played move) + the
example traces (prior_entropy / visit_entropy / temperature per move). C1 config:
temp 1.0->0.2 over 30 plies (floor 0.1), Dirichlet total_alpha=6.6, eps=0.25.
"""
from __future__ import annotations
import glob, json, math, random, sys
from collections import Counter, defaultdict
import numpy as np

from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_engine.types import unpack_coord_id

RUN = sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main"
N_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def phase(ply):
    return "opening" if ply < 15 else ("mid" if ply < 60 else "end")


def move_temperature(ply, initial=1.0, final=0.2, decay=30, floor=0.1):
    if decay <= 0:
        return initial
    frac = min(ply, decay) / decay
    return max(floor, initial + (final - initial) * frac)


def conc(weights):
    """top-1, top-5 cumulative, effective-move-count (perplexity) of a weight vec."""
    w = np.asarray([x for x in weights if x > 0], dtype=np.float64)
    s = w.sum()
    if s <= 0 or w.size == 0:
        return 0.0, 0.0, 0.0
    p = w / s
    p.sort()
    p = p[::-1]
    top1 = float(p[0])
    top5 = float(p[:5].sum())
    ent = float(-(p * np.log(p)).sum())
    return top1, top5, math.exp(ent)


def selection_probs(weights, T):
    w = np.asarray([x for x in weights if x > 0], dtype=np.float64)
    if w.size == 0:
        return w
    if T <= 1e-6:
        out = np.zeros_like(w); out[w.argmax()] = 1.0; return out
    x = np.power(w, 1.0 / T)
    return x / x.sum()


def main():
    shards = sorted(glob.glob(f"{RUN}/selfplay/epoch_*_game_*.npz"))
    rng = random.Random(0); rng.shuffle(shards)
    shards = shards[:N_GAMES]

    # accumulators per phase
    acc = lambda: defaultdict(list)
    visit = acc(); prior_sel = acc(); sel = acc(); nonargmax = acc(); legalc = acc()
    first_moves = []; opening_lines = []; pos_hash = Counter(); total_pos = 0

    for sp in shards:
        try:
            rows = read_compact_shard(sp)
        except Exception:
            continue
        rows = [r for r in rows if r.policy]
        rows.sort(key=lambda r: int(r.turn_index))
        if not rows:
            continue
        # played-move reconstruction: history grows by 1 per turn
        hist_sets = []
        for r in rows:
            hs = set((int(e[0]), int(e[1])) for e in r.placement_history)
            hist_sets.append(hs)
        # first move + opening line (action ids of played moves)
        played_qr = []
        for i in range(len(rows) - 1):
            d = hist_sets[i + 1] - hist_sets[i]
            played_qr.append(next(iter(d)) if len(d) == 1 else None)
        played_qr.append(None)  # last position's played move unknown

        for i, r in enumerate(rows):
            ply = int(r.turn_index); ph = phase(ply)
            weights = [float(w) for _a, w in r.policy]
            t1, t5, eff = conc(weights)
            visit[ph].append((t1, t5, eff))
            legalc[ph].append(len(r.legal_action_ids))
            T = move_temperature(ply)
            sprob = selection_probs(weights, T)
            if sprob.size:
                s1, s5, seff = conc(sprob)
                sel[ph].append((s1, s5, seff))
                nonargmax[ph].append(1.0 - float(sprob.max()))  # P(sample != top)
            # realized (actual) non-argmax: played vs visit-argmax
            pm = played_qr[i]
            if pm is not None:
                amax = max(r.policy, key=lambda kv: kv[1])[0]
                aqr = unpack_coord_id(int(amax))
                acted_non = 0.0 if (int(aqr.q), int(aqr.r)) == pm else 1.0
                nonargmax["ACTUAL_" + ph].append(acted_non)
            # transposition
            pos_hash[frozenset(hist_sets[i])] += 1; total_pos += 1
        # diversity
        if played_qr and played_qr[0] is not None:
            first_moves.append(played_qr[0])
        line = tuple(played_qr[:6])
        if all(x is not None for x in line) and len(line) == 6:
            opening_lines.append(line)

    # prior + temperature from example traces
    prior = acc(); visit_tr = acc(); temp_by_ply = defaultdict(list)
    for tf in sorted(glob.glob(f"{RUN}/eval/epoch_*_examples.json")):
        try:
            games = json.load(open(tf))
        except Exception:
            continue
        for g in games:
            for m in g.get("moves", []):
                ply = int(m["move"]); ph = phase(ply)
                prior[ph].append((m.get("prior_entropy"), m.get("candidates")))
                visit_tr[ph].append(m.get("visit_entropy"))
                temp_by_ply[ply].append((m.get("temperature"), m.get("visit_entropy"), m.get("top_visit_fraction")))

    def summ(rows3):
        if not rows3:
            return (float("nan"),) * 3
        a = np.array(rows3, dtype=float)
        return tuple(np.nanmean(a, axis=0))

    print(f"### Exploration diagnostics — {RUN.split('/')[-1]}  ({len(shards)} games sampled)\n")
    print("CONCENTRATION by phase (mean):")
    print(f"{'phase':8} {'legal':>6} | {'VISIT top1':>10} {'top5':>6} {'effN':>6} | "
          f"{'SELECT top1':>11} {'effN':>6} | {'PRIOR effN':>10} {'priorH':>7}")
    for ph in ("opening", "mid", "end"):
        v1, v5, veff = summ(visit[ph])
        s1, s5, seff = summ(sel[ph])
        lg = np.mean(legalc[ph]) if legalc[ph] else float("nan")
        pr = [p for p, _c in prior[ph] if p is not None]
        pH = np.mean(pr) if pr else float("nan")
        prEff = math.exp(pH) if pr else float("nan")
        print(f"{ph:8} {lg:6.0f} | {v1:10.3f} {v5:6.3f} {veff:6.1f} | "
              f"{s1:11.3f} {seff:6.1f} | {prEff:10.1f} {pH:7.2f}")

    print("\nREALIZED exploration — P(played != visit-argmax), by phase:")
    print(f"{'phase':8} {'expected(from sel dist)':>24} {'ACTUAL(played moves)':>22}")
    for ph in ("opening", "mid", "end"):
        exp = np.mean(nonargmax[ph]) if nonargmax[ph] else float("nan")
        act = np.mean(nonargmax["ACTUAL_" + ph]) if nonargmax["ACTUAL_" + ph] else float("nan")
        print(f"{ph:8} {exp:24.3f} {act:22.3f}")

    print("\nDIVERSITY across games:")
    fm = Counter(first_moves)
    fm_p = np.array(list(fm.values()), float); fm_p /= fm_p.sum()
    fm_ent = float(-(fm_p * np.log(fm_p)).sum()) if fm_p.size else 0.0
    print(f"  first move: {len(fm)} unique over {len(first_moves)} games "
          f"(entropy {fm_ent:.2f} nats = {math.exp(fm_ent):.1f} eff; top1 {fm.most_common(1)[0][1]/max(1,len(first_moves)):.0%})")
    ol = Counter(opening_lines)
    print(f"  opening 6-ply lines: {len(ol)} unique over {len(opening_lines)} games "
          f"({len(ol)/max(1,len(opening_lines)):.0%} unique; top line repeats {ol.most_common(1)[0][1] if ol else 0}x)")
    revisited = sum(1 for c in pos_hash.values() if c > 1)
    revisit_positions = sum(c for c in pos_hash.values() if c > 1)
    print(f"  transpositions: {revisited} positions seen >1x; "
          f"{revisit_positions/max(1,total_pos):.1%} of all positions are revisits "
          f"(of {total_pos} positions, {len(pos_hash)} unique)")

    print("\nTEMPERATURE tracking (selection sharpening vs ply, from traces):")
    print(f"{'ply':>4} {'T(sched)':>8} {'visitH':>7} {'top_visit_frac':>14}  (mean over example games)")
    for ply in (0, 2, 5, 10, 15, 20, 30, 45, 60, 90):
        rows = temp_by_ply.get(ply, [])
        if not rows:
            continue
        a = np.array([[r[0] or 0, r[1] or 0, r[2] or 0] for r in rows], float)
        print(f"{ply:4} {a[:,0].mean():8.2f} {a[:,1].mean():7.2f} {a[:,2].mean():14.3f}")


if __name__ == "__main__":
    main()
