"""Real-cohort prevalence and root branching estimate for the J2 spare gate."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_atlas_width import (
    Board,
    IDS,
    classify_pair,
    live_axis_degree,
    mhs,
    owner_at,
    second_candidates,
    turn_start_candidates,
    window_cells,
)


ROOT = Path(__file__).resolve().parents[1]


def pct(values, q):
    if not values:
        return 0
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def first_alone_tight(board, claimant, first):
    """Whether first alone already creates a post-turn tau>=2 family and
    independently answers every turn-start defender threat.
    """
    if first not in board.legal_cells():
        return False, None, None
    post = board.clone()
    post.place(first, claimant)
    family = [
        empties
        for key, _count, empties in post.active(claimant, 4)
        if first in window_cells(key)
    ]
    tau = mhs(family)
    defender = [set(e) for _k, _c, e in board.active(1 - claimant, 4)]
    hits = all(first in cells for cells in defender)
    return bool(family) and hits and (tau is None or tau == 2), len(family), tau


def j2_seconds(board, claimant, first):
    """Novel J2 cells after first, relative to exact current S(P,first)."""
    current = second_candidates(board, claimant, first)
    post = board.clone()
    post.place(first, claimant)
    novel = [
        cell
        for cell in post.legal_cells() - current - {first}
        if live_axis_degree(post, claimant, cell, 1) >= 2
        # active(...,1) is pre-second; adding the candidate turns each such
        # supported axis into count>=2.  The cell itself is empty here.
    ]
    return current, novel


def measure_position(moves):
    n = len(moves)
    claimant = owner_at(n)
    phase = "first" if n == 1 or n % 2 == 1 else "second"
    if phase == "first":
        board = Board(moves)
        T = turn_start_candidates(board, claimant)
        pair_universe = set()
        tight_firsts = []
        j2 = set()
        for first in T:
            seconds = second_candidates(board, claimant, first)
            for second in seconds:
                pair = tuple(sorted((first, second)))
                pair_universe.add(pair)
            tight, family_n, tau = first_alone_tight(board, claimant, first)
            if tight:
                tight_firsts.append((first, family_n, tau))
                _current, novel = j2_seconds(board, claimant, first)
                j2.update(tuple(sorted((first, second))) for second in novel)
        j2 -= pair_universe
        return {
            "phase": phase,
            "T": len(T),
            "current_universe": len(pair_universe),
            "current_accepted": None,
            "tight_firsts": len(tight_firsts),
            "j2": len(j2),
        }

    # The root is after the claimant's first stone. Rebuild the actual turn
    # start so the tight-first predicate and S(P,a) use the same snapshot as
    # WideTurnGate would have used on a fresh-turn call.
    first = tuple(moves[-1])
    start = Board(moves[:-1])
    tight, family_n, tau = first_alone_tight(start, claimant, first)
    current, novel = j2_seconds(start, claimant, first) if tight else (set(), [])
    return {
        "phase": phase,
        "T": len(turn_start_candidates(Board(moves), claimant)),
        "current_universe": len(current),
        "current_accepted": None,
        "tight_firsts": int(tight),
        "j2": len(novel),
        "family_n": family_n,
        "tau": tau,
    }


def summarize(records):
    out = {"n": len(records)}
    out["phase"] = dict(Counter(r["phase"] for r in records))
    for key in ("T", "current_universe", "current_accepted", "tight_firsts", "j2"):
        values = [r[key] for r in records if r.get(key) is not None]
        out[key] = {
            "sum": sum(values),
            "mean": sum(values) / len(values) if values else 0,
            "p50": pct(values, 0.5),
            "p90": pct(values, 0.9),
            "max": max(values, default=0),
            "nonzero": sum(v > 0 for v in values),
        }
    affected = [r for r in records if r["j2"]]
    out["affected_n"] = len(affected)
    out["affected_j2"] = {
        "p50": pct([r["j2"] for r in affected], 0.5),
        "p90": pct([r["j2"] for r in affected], 0.9),
        "max": max((r["j2"] for r in affected), default=0),
    }
    fresh_affected = [r for r in affected if r["phase"] == "first"]
    out["fresh_affected_ratio"] = {
        "sum_current_universe": sum(r["current_universe"] for r in fresh_affected),
        "sum_j2": sum(r["j2"] for r in fresh_affected),
    }
    return out


def main():
    labels = []
    with (ROOT / "raws" / "lanec_labels.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("source") == "grind":
                labels.append(row)
    wanted = {r["pos_id"] for r in labels}
    positions = {}
    with (ROOT / "scripts" / "tss_harness" / "sets" / "selfplay_v1.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["pos_id"] in wanted:
                positions[row["pos_id"]] = row["moves"]

    cohorts = defaultdict(list)
    details = []
    for i, label in enumerate(labels, 1):
        if label["status"] == "win":
            cohort = "win"
        elif label.get("tt_saturation_suspect"):
            cohort = "cap_bound_raw"
        else:
            cohort = "width_exhaust_raw"
        record = measure_position(positions[label["pos_id"]])
        record["pos_id"] = label["pos_id"]
        record["cohort"] = cohort
        record["nodes"] = label["win_pass"]["deep_nodes"]
        cohorts[cohort].append(record)
        details.append(record)
        if i % 40 == 0:
            print(f"measured {i}/{len(labels)}", flush=True)

    result = {
        "method": "raw class uses tt_saturation_suspect; snapshot counts intentionally reported",
        "cohorts": {key: summarize(rows) for key, rows in sorted(cohorts.items())},
        "affected_examples": [r for r in details if r["j2"]][:20],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
