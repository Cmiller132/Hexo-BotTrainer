"""Reconstruct the five atlas-deep witnesses against the production VCF width.

Scratch-only measurement.  The geometry and gates mirror tss_solver.rs:
length-six axial windows on the three undirected axes, count>=2 turn-start
attacker candidates, G1 promotion after the first stone, and tau=2/overflow
post-pair dispatch.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT.parent / "opening-atlas" / "atlas-web" / "data" / "atlas.json"
PUZZLES = ROOT / "scripts" / "tss_harness" / "sets" / "puzzle_v3.jsonl"
IDS = {
    "oa-0153903c5a863630",
    "oa-23c6c04ad42d0904",
    "oa-611666d7d930eb1f",
    "oa-6fda812864c6d19a",
    "oa-773ca1a59e95f4e1",
}
AXES = ((1, 0), (0, 1), (1, -1))


def owner_at(index: int) -> int:
    if index == 0:
        return 0
    return 1 if ((index - 1) // 2) % 2 == 0 else 0


def hdist(a, b):
    dq, dr = a[0] - b[0], a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def window_cells(key):
    (q, r), axis = key
    dq, dr = AXES[axis]
    return tuple((q + i * dq, r + i * dr) for i in range(6))


class Board:
    def __init__(self, moves=()):
        self.stones: dict[tuple[int, int], int] = {}
        self.wcnt = defaultdict(lambda: [0, 0])
        for i, raw in enumerate(moves):
            self.place(tuple(raw), owner_at(i))

    def clone(self):
        other = Board()
        other.stones = dict(self.stones)
        other.wcnt = defaultdict(lambda: [0, 0], {k: list(v) for k, v in self.wcnt.items()})
        return other

    def place(self, cell, player):
        assert cell not in self.stones, cell
        self.stones[cell] = player
        for axis, (dq, dr) in enumerate(AXES):
            for off in range(6):
                key = ((cell[0] - off * dq, cell[1] - off * dr), axis)
                self.wcnt[key][player] += 1

    def empties(self, key):
        return tuple(c for c in window_cells(key) if c not in self.stones)

    def active(self, player, minimum=1):
        opponent = 1 - player
        return [
            (key, cnt[player], self.empties(key))
            for key, cnt in self.wcnt.items()
            if cnt[player] >= minimum and cnt[opponent] == 0
        ]

    def wins(self, player):
        return any(counts[player] == 6 for counts in self.wcnt.values())

    def legal_cells(self):
        out = set()
        for q, r in self.stones:
            for dq in range(-8, 9):
                for dr in range(-8, 9):
                    cell = (q + dq, r + dr)
                    if cell not in self.stones and hdist(cell, (q, r)) <= 8:
                        out.add(cell)
        return out


def mhs(sets, budget=2):
    sets = [set(s) for s in sets]
    if not sets:
        return 0
    if any(not s for s in sets):
        return None
    universe = sorted(set().union(*sets))
    if budget >= 1 and any(all(x in s for s in sets) for x in universe):
        return 1
    if budget >= 2:
        for i, x in enumerate(universe):
            for y in universe[i + 1 :]:
                if all(x in s or y in s for s in sets):
                    return 2
    return None


def turn_start_candidates(board, claimant):
    cells = set()
    for _key, _count, empties in board.active(claimant, 2):
        cells.update(empties)
    for _key, _count, empties in board.active(1 - claimant, 4):
        cells.update(empties)
    return cells


def second_candidates(board, claimant, first):
    # Board is the turn-start board, as in WideTurnGate.
    out = set(turn_start_candidates(board, claimant))
    for key, _count, empties in board.active(claimant, 1):
        if first in empties:
            out.update(c for c in empties if c != first)
    out.discard(first)
    return out


def post_pair_family(board, claimant, first, second):
    post = board.clone()
    post.place(first, claimant)
    post.place(second, claimant)
    family = []
    for key, count, empties in post.active(claimant, 4):
        cells = window_cells(key)
        if first in cells or second in cells:
            family.append((key, count, empties))
    return post, family


def live_axis_degree(board, claimant, cell, minimum=2):
    axes = set()
    for key, count, _empties in board.active(claimant, minimum):
        if cell in window_cells(key):
            axes.add(key[1])
    return len(axes)


def classify_pair(board, claimant, first, second):
    start = turn_start_candidates(board, claimant)
    order_ok = []
    for a, b in ((first, second), (second, first)):
        order_ok.append(a in start and b in second_candidates(board, claimant, a))
    post, family = post_pair_family(board, claimant, first, second)
    defender_threats = [set(e) for _k, _c, e in board.active(1 - claimant, 4)]
    hits_defender = all(first in s or second in s for s in defender_threats)
    tau = mhs([e for _k, _c, e in family])
    return {
        "first_in_T": first in start,
        "second_in_T": second in start,
        "order_ok": any(order_ok),
        "family_n": len(family),
        "family_counts": sorted(c for _k, c, _e in family),
        "tau": tau,
        "hits_defender": hits_defender,
        "accepted": any(order_ok) and bool(family) and hits_defender and (tau is None or tau == 2),
        "wins_now": post.wins(claimant),
    }


def classify_second(board_after_first, claimant, second, first):
    candidates = turn_start_candidates(board_after_first, claimant)
    post = board_after_first.clone()
    post.place(second, claimant)
    family = [(k, c, e) for k, c, e in post.active(claimant, 4) if first in window_cells(k) or second in window_cells(k)]
    defender_threats = [set(e) for _k, _c, e in board_after_first.active(1 - claimant, 4)]
    tau = mhs([e for _k, _c, e in family])
    return {
        "in_T_after_first": second in candidates,
        "family_n": len(family),
        "family_counts": sorted(c for _k, c, _e in family),
        "tau": tau,
        "hits_defender": all(second in s for s in defender_threats),
        "accepted": second in candidates and bool(family) and all(second in s for s in defender_threats) and (tau is None or tau == 2),
        "wins_now": post.wins(claimant),
    }


def pair_population(board, claimant):
    start = turn_start_candidates(board, claimant)
    seen = set()
    bins = defaultdict(int)
    examples = defaultdict(list)
    for first in sorted(start):
        for second in sorted(second_candidates(board, claimant, first)):
            pair = tuple(sorted((first, second)))
            if pair in seen:
                continue
            seen.add(pair)
            info = classify_pair(board, claimant, *pair)
            if not info["order_ok"] or not info["family_n"] or not info["hits_defender"]:
                kind = "gate_other"
            elif info["tau"] is None:
                kind = "tau_overflow"
            else:
                kind = f"tau_{info['tau']}"
            bins[kind] += 1
            if len(examples[kind]) < 4:
                examples[kind].append(pair)
    return len(start), len(seen), dict(sorted(bins.items())), dict(examples)


def cross_seed_extension_population(board, claimant):
    """Measure J2: outside-S seconds with post-pair live count>=2 support
    on at least two distinct axes.  Counts are unordered and exact on the
    finite engine legality frontier (radius eight from any post-first stone).
    """
    start = turn_start_candidates(board, claimant)
    current_pairs = set()
    extension_pairs = set()
    accepted_extension = set()
    by_first = {}
    for first in sorted(start):
        current_seconds = second_candidates(board, claimant, first)
        current_pairs.update(tuple(sorted((first, second))) for second in current_seconds)
        after_first = board.clone()
        after_first.place(first, claimant)
        novel = []
        accepted = []
        for second in sorted(after_first.legal_cells() - current_seconds - {first}):
            after_pair = after_first.clone()
            after_pair.place(second, claimant)
            if live_axis_degree(after_pair, claimant, second, 2) < 2:
                continue
            pair = tuple(sorted((first, second)))
            extension_pairs.add(pair)
            novel.append(second)
            info = classify_pair(board, claimant, first, second)
            if info["family_n"] and info["hits_defender"] and (info["tau"] is None or info["tau"] == 2):
                accepted_extension.add(pair)
                accepted.append(second)
        by_first[first] = (len(novel), len(accepted))
    return {
        "T": len(start),
        "current_unordered": len(current_pairs),
        "J2_unordered": len(extension_pairs - current_pairs),
        "J2_accepted": len(accepted_extension - current_pairs),
        "per_first_novel_accepted": by_first,
    }


def main():
    atlas = json.loads(ATLAS.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in atlas["rows"] if r["id"] in IDS}
    puzzle_labels = {}
    for line in PUZZLES.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["pos_id"].removeprefix("atlas_") in IDS:
            puzzle_labels[row["pos_id"].removeprefix("atlas_")] = row["labels"]

    for rid in sorted(rows):
        row = rows[rid]
        root_moves = row["moves"]
        claimant = 0 if row["side"] == "P0" else 1
        board = Board(root_moves)
        root_phase = "first" if len(root_moves) == 1 or len(root_moves) % 2 == 1 else "second"
        print(f"\n{rid} claimant=P{claimant} root_n={len(root_moves)} phase={root_phase} label={puzzle_labels[rid]}")
        if root_phase == "first":
            print("root_population", pair_population(board, claimant)[:3])
            print("root_J2", cross_seed_extension_population(board, claimant))
        else:
            actual_second = tuple(row["win_line"][0])
            legal = board.legal_cells()
            candidates = turn_start_candidates(board, claimant)
            print(
                "root_second_J2",
                {
                    "T": len(candidates),
                    "legal": len(legal),
                    "J2_outside_T": sum(
                        c not in candidates and live_axis_degree((lambda x: (x.place(c, claimant), x)[1])(board.clone()), claimant, c, 2) >= 2
                        for c in legal
                    ),
                    "actual_axis_degree": live_axis_degree(
                        (lambda x: (x.place(actual_second, claimant), x)[1])(board.clone()), claimant, actual_second, 2
                    ),
                },
            )
        line = [tuple(x) for x in row["win_line"]]
        cursor = 0
        global_i = len(root_moves)
        while cursor < len(line):
            player = owner_at(global_i)
            assert player == (claimant if (global_i == len(root_moves)) else owner_at(global_i))
            fresh = global_i == 0 or global_i % 2 == 1
            if fresh and cursor + 1 < len(line) and owner_at(global_i + 1) == player:
                a, b = line[cursor], line[cursor + 1]
                if player == claimant:
                    print(f"  ply+{cursor+1:02d} attacker pair {a} {b}: {classify_pair(board, claimant, a, b)}")
                board.place(a, player)
                board.place(b, player)
                cursor += 2
                global_i += 2
            else:
                cell = line[cursor]
                # This occurs for the n=8 witness: claimant completes a root pair.
                if player == claimant:
                    first = tuple(root_moves[-1])
                    print(f"  ply+{cursor+1:02d} attacker second {cell} (first={first}): {classify_second(board, claimant, cell, first)}")
                board.place(cell, player)
                cursor += 1
                global_i += 1
            if board.wins(player):
                print(f"  terminal P{player} after relative ply {cursor}")
                break


if __name__ == "__main__":
    main()
