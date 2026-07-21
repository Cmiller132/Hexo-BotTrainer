from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

AXES = ((1, 0), (0, 1), (1, -1))
ROOT = Path(__file__).resolve().parents[1]


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def scale(a, n):
    return a[0] * n, a[1] * n


def owner_at(index):
    if index == 0:
        return 0
    return 1 if ((index - 1) // 2) % 2 == 0 else 0


def mover_after(n):
    if n == 0:
        return 0, "Opening"
    post = n - 1
    return (1 if (post // 2) % 2 == 0 else 0), ("FirstStone" if post % 2 == 0 else "SecondStone")


def make_board(moves):
    return {tuple(c): owner_at(i) for i, c in enumerate(moves)}


def entries(board):
    keys = set()
    for cell in board:
        for axis in AXES:
            for offset in range(6):
                keys.add((add(cell, scale(axis, -offset)), axis))
    out = []
    for start, axis in keys:
        cells = tuple(add(start, scale(axis, i)) for i in range(6))
        c0 = sum(board.get(c) == 0 for c in cells)
        c1 = sum(board.get(c) == 1 for c in cells)
        if c0 and c1:
            continue
        if c0:
            out.append((0, c0, tuple(c for c in cells if c not in board), axis, cells))
        elif c1:
            out.append((1, c1, tuple(c for c in cells if c not in board), axis, cells))
    return out


def normal_candidates(es, claimant):
    out = set()
    for owner, count, empties, _, _ in es:
        if (owner == claimant and count >= 2) or (owner != claimant and count >= 4):
            out.update(empties)
    return out


def q1_support(es, claimant):
    out = {}
    for owner, count, empties, axis, _ in es:
        if owner == claimant and count == 1:
            for c in empties:
                per_axis = out.setdefault(c, Counter())
                per_axis[axis] += 1
    return out


def support_ok(counter, minimum):
    strengths = sorted(counter.values(), reverse=True)
    return len(strengths) >= 2 and strengths[1] >= minimum


def tau(sets):
    if not sets:
        return 0
    if any(not s for s in sets):
        return 3
    universe = set().union(*(set(s) for s in sets))
    if any(all(c in s for s in sets) for c in universe):
        return 1
    u = list(universe)
    for i, a in enumerate(u):
        for b in u[i + 1:]:
            if all(a in s or b in s for s in sets):
                return 2
    return 3


def forcing(es, claimant):
    threats = [empties for owner, count, empties, _, _ in es if owner == claimant and count >= 4]
    defender_now = any(owner != claimant and count >= 4 for owner, count, _, _, _ in es)
    return not defender_now and tau(threats) >= 2


def first_alone_forcing(root_entries, claimant, first):
    threats = []
    for owner, count, empties, _, _ in root_entries:
        if owner == claimant and count >= 3 and first in empties:
            threats.append(tuple(c for c in empties if c != first))
        elif owner != claimant and count >= 4 and first not in empties:
            return False
    return tau(threats) >= 2


def own_win_now(es, claimant, phase):
    budget = 2 if phase in ("Opening", "FirstStone") else 1
    return any(
        owner == claimant and (count == 5 or (count == 4 and budget == 2))
        for owner, count, _, _, _ in es
    )


def exact_second_universe(root_entries, claimant, first):
    out = set(normal_candidates(root_entries, claimant))
    for owner, count, empties, _, cells in root_entries:
        if owner == claimant and count >= 1 and first in empties:
            out.update(empties)
    out.discard(first)
    return out


def percentile(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int((len(xs) - 1) * q))]


def analyze(rows):
    result = {
        "positions": len(rows), "usable": 0, "first": 0, "second": 0,
        "eligible_positions": 0, "eligible_firsts": 0,
        "j2_positions": 0, "j2near_positions": 0,
        "current": [], "j2": [], "j2near": [], "legal_proxy": [],
    }
    examples = []
    for row in rows:
        moves = row["moves"]
        claimant, phase = mover_after(len(moves))
        if phase == "Opening":
            continue
        board = make_board(moves)
        es = entries(board)
        if own_win_now(es, claimant, phase):
            continue
        result["usable"] += 1
        result["first" if phase == "FirstStone" else "second"] += 1
        current_children = set()
        j2_pairs = set()
        near_pairs = set()
        eligible_firsts = 0
        if phase == "FirstStone":
            firsts = normal_candidates(es, claimant)
            for first in firsts:
                if not first_alone_forcing(es, claimant, first):
                    continue
                b1 = dict(board)
                b1[first] = claimant
                e1 = entries(b1)
                # Minimal gate: the first stone has already bought the full
                # defender turn; only the otherwise-free second is widened.
                assert forcing(e1, claimant)
                eligible_firsts += 1
                seconds = exact_second_universe(es, claimant, first)
                support = q1_support(e1, claimant)
                for second, per_axis in support.items():
                    if second in seconds or second == first or not support_ok(per_axis, 1):
                        continue
                    pair = tuple(sorted((first, second)))
                    j2_pairs.add(pair)
                    if support_ok(per_axis, 4):
                        near_pairs.add(pair)
            # Exact current-branch baseline is only material at an eligible
            # node; avoiding it elsewhere makes prevalence scans cheap.
            if eligible_firsts:
                for first in firsts:
                    b1 = dict(board)
                    b1[first] = claimant
                    for second in exact_second_universe(es, claimant, first):
                        b2 = dict(b1)
                        b2[second] = claimant
                        if forcing(entries(b2), claimant):
                            current_children.add(tuple(sorted((first, second))))
            result["legal_proxy"].append(0)
        else:
            # The first stone is already on the board. If it bought a tight
            # reply, widen only the free second placement.
            candidates = normal_candidates(es, claimant)
            if forcing(es, claimant):
                eligible_firsts = 1
                for second in candidates:
                    b2 = dict(board)
                    b2[second] = claimant
                    if forcing(entries(b2), claimant):
                        current_children.add(second)
                support = q1_support(es, claimant)
                for second, per_axis in support.items():
                    if second in candidates or not support_ok(per_axis, 1):
                        continue
                    j2_pairs.add(second)
                    if support_ok(per_axis, 4):
                        near_pairs.add(second)
            # Full legal is much larger; unique window-supported empties is a
            # conservative, reproducible lower proxy sufficient for ratios.
            result["legal_proxy"].append(len(set().union(*(set(e[2]) for e in es)) if es else set()))
        if eligible_firsts:
            result["eligible_positions"] += 1
            result["eligible_firsts"] += eligible_firsts
            result["current"].append(len(current_children))
            result["j2"].append(len(j2_pairs))
            result["j2near"].append(len(near_pairs))
            result["j2_positions"] += bool(j2_pairs)
            result["j2near_positions"] += bool(near_pairs)
            if len(examples) < 10:
                examples.append((row["pos_id"], phase, len(current_children), len(j2_pairs), len(near_pairs), eligible_firsts))
    return result, examples


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(name, result, examples):
    n = result["eligible_positions"]
    print(f"COHORT {name} positions={result['positions']} usable={result['usable']} first={result['first']} second={result['second']}")
    print(f"  eligible_positions={n} prevalence_all={n/result['positions']:.4%} prevalence_usable={n/max(1,result['usable']):.4%} eligible_firsts={result['eligible_firsts']}")
    print(f"  j2_positions={result['j2_positions']} ({result['j2_positions']/max(1,result['positions']):.4%}) j2near_positions={result['j2near_positions']} ({result['j2near_positions']/max(1,result['positions']):.4%})")
    for key in ("current", "j2", "j2near"):
        xs = result[key]
        print(f"  {key}: sum={sum(xs)} mean={sum(xs)/max(1,len(xs)):.2f} p50={percentile(xs,.5)} p90={percentile(xs,.9)} max={max(xs,default=0)}")
    ratios = [(c+j)/max(1,c) for c,j in zip(result["current"],result["j2"])]
    near_ratios = [(c+j)/max(1,c) for c,j in zip(result["current"],result["j2near"])]
    print(f"  current+J2 multiplier: mean={sum(ratios)/max(1,len(ratios)):.3f} p50={percentile(ratios,.5):.3f} p90={percentile(ratios,.9):.3f} max={max(ratios,default=0):.3f}")
    print(f"  current+J2near multiplier: mean={sum(near_ratios)/max(1,len(near_ratios)):.3f} p50={percentile(near_ratios,.5):.3f} p90={percentile(near_ratios,.9):.3f} max={max(near_ratios,default=0):.3f}")
    print(f"  examples={examples}")


sets_dir = ROOT / "scripts" / "tss_harness" / "sets"
move_rows = {}
for filename in ("selfplay_v1.jsonl", "human_v1.jsonl", "puzzle_v3.jsonl"):
    for row in read_jsonl(sets_dir / filename):
        move_rows[row["pos_id"]] = row

labels = read_jsonl(ROOT / "raws" / "lanec_labels.jsonl")
grind_ids = [row["pos_id"] for row in labels if row["source"] == "grind"]
grinds = [move_rows[pos_id] for pos_id in grind_ids if pos_id in move_rows]
puzzles = read_jsonl(sets_dir / "puzzle_v3.jsonl")
humans = read_jsonl(sets_dir / "human_v1.jsonl")
selfplay = read_jsonl(sets_dir / "selfplay_v1.jsonl")

for name, rows in (("grinds", grinds), ("puzzle_v3", puzzles), ("human_v1", humans), ("selfplay_v1", selfplay)):
    summarize(name, *analyze(rows))
