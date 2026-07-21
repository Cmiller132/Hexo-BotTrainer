#!/usr/bin/env python3
"""Read-only geometry probes for the research-div reply-structure lane.

The script deliberately re-derives only rule-level geometry from frozen move
lists.  It does not call the solver and does not claim search verdicts.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


Coord = tuple[int, int]
AXES: tuple[Coord, ...] = ((1, 0), (0, 1), (1, -1))


def add(a: Coord, b: Coord) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def scale(k: int, a: Coord) -> Coord:
    return k * a[0], k * a[1]


def dist(a: Coord, b: Coord) -> int:
    q = a[0] - b[0]
    r = a[1] - b[1]
    return max(abs(q), abs(r), abs(q + r))


def ball(center: Coord, radius: int = 8) -> set[Coord]:
    cq, cr = center
    out: set[Coord] = set()
    for dq in range(-radius, radius + 1):
        lo = max(-radius, -dq - radius)
        hi = min(radius, -dq + radius)
        for dr in range(lo, hi + 1):
            out.add((cq + dq, cr + dr))
    return out


BALL8 = ball((0, 0), 8)


def shifted_ball(center: Coord) -> set[Coord]:
    return {(center[0] + q, center[1] + r) for q, r in BALL8}


def owner_at_index(i: int) -> int:
    if i == 0:
        return 0
    return 1 if ((i - 1) // 2) % 2 == 0 else 0


def phase_and_player(nstones: int) -> tuple[str, int, int]:
    if nstones == 0:
        return "opening", 0, 1
    placed_after_opening = nstones - 1
    turn = placed_after_opening // 2
    player = 1 if turn % 2 == 0 else 0
    if placed_after_opening % 2 == 0:
        return "first", player, 2
    return "second", player, 1


def board_from_moves(moves: list[Coord]) -> dict[Coord, int]:
    return {coord: owner_at_index(i) for i, coord in enumerate(moves)}


def support(board: dict[Coord, int]) -> set[Coord]:
    out: set[Coord] = set()
    for coord in board:
        out.update(shifted_ball(coord))
    return out


def legal_cells(board: dict[Coord, int]) -> set[Coord]:
    return support(board).difference(board)


def incident_windows(coord: Coord):
    for axis_id, axis in enumerate(AXES):
        for offset in range(6):
            start = add(coord, scale(-offset, axis))
            cells = tuple(add(start, scale(i, axis)) for i in range(6))
            yield axis_id, start, cells


def relevant_windows(board: dict[Coord, int]):
    seen: set[tuple[int, Coord]] = set()
    for coord in board:
        for axis_id, start, cells in incident_windows(coord):
            key = (axis_id, start)
            if key not in seen:
                seen.add(key)
                yield axis_id, start, cells


def dead_empty(coord: Coord, board: dict[Coord, int]) -> bool:
    if coord in board:
        return False
    for _axis, _start, cells in incident_windows(coord):
        owners = {board[c] for c in cells if c in board}
        if owners != {0, 1}:
            return False
    return True


def frontier_delta(coord: Coord, old_support: set[Coord]) -> int:
    return len(shifted_ball(coord).difference(old_support))


def threat_family(board: dict[Coord, int], attacker: int) -> list[frozenset[Coord]]:
    defender = 1 - attacker
    family: list[frozenset[Coord]] = []
    for _axis, _start, cells in relevant_windows(board):
        if any(board.get(c) == defender for c in cells):
            continue
        count = sum(board.get(c) == attacker for c in cells)
        if count >= 4:
            family.append(frozenset(c for c in cells if c not in board))
    return family


def own_win_now(board: dict[Coord, int], player: int, budget: int) -> bool:
    opponent = 1 - player
    for _axis, _start, cells in relevant_windows(board):
        if any(board.get(c) == opponent for c in cells):
            continue
        count = sum(board.get(c) == player for c in cells)
        if count == 5 or (count == 4 and budget == 2):
            return True
    return False


def tau_le2(family: list[frozenset[Coord]]) -> int | None:
    if not family:
        return 0
    universe = sorted(set().union(*family))
    if any(all(c in edge for edge in family) for c in universe):
        return 1
    for i, left in enumerate(universe):
        for right in universe[i + 1 :]:
            if all(left in edge or right in edge for edge in family):
                return 2
    return None


def hyper_components(family: list[frozenset[Coord]]) -> list[list[int]]:
    if not family:
        return []
    by_cell: dict[Coord, list[int]] = collections.defaultdict(list)
    for i, edge in enumerate(family):
        for c in edge:
            by_cell[c].append(i)
    adj: list[set[int]] = [set() for _ in family]
    for indices in by_cell.values():
        for i in indices:
            adj[i].update(indices)
    seen: set[int] = set()
    out: list[list[int]] = []
    for root in range(len(family)):
        if root in seen:
            continue
        stack = [root]
        seen.add(root)
        comp: list[int] = []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in adj[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        out.append(comp)
    return out


def common_hits(family: list[frozenset[Coord]]) -> set[Coord]:
    if not family:
        return set()
    return set.intersection(*(set(edge) for edge in family))


def minimum_pairs(family: list[frozenset[Coord]]) -> list[tuple[Coord, Coord]]:
    if not family:
        return []
    universe = sorted(set().union(*family))
    return [
        (left, right)
        for i, left in enumerate(universe)
        for right in universe[i + 1 :]
        if all(left in edge or right in edge for edge in family)
    ]


def covering_pairs_over(cells: set[Coord], family: list[frozenset[Coord]]) -> list[tuple[Coord, Coord]]:
    universe = sorted(cells)
    return [
        (left, right)
        for i, left in enumerate(universe)
        for right in universe[i + 1 :]
        if all(left in edge or right in edge for edge in family)
    ]


def d6(coord: Coord, symmetry: int) -> Coord:
    q, r = coord
    if symmetry >= 6:
        r = -q - r
    for _ in range(symmetry % 6):
        q, r = -r, q + r
    return q, r


def affine_stabilizers(board: dict[Coord, int], phase_first: Coord | None) -> list[tuple[int, Coord]]:
    if not board:
        return [(s, (0, 0)) for s in range(12)]
    anchor = min(board)
    owner = board[anchor]
    targets = [c for c, p in board.items() if p == owner]
    expected = set(board.items())
    out: list[tuple[int, Coord]] = []
    for s in range(12):
        ma = d6(anchor, s)
        for target in targets:
            t = target[0] - ma[0], target[1] - ma[1]
            transformed = {
                ((d6(c, s)[0] + t[0], d6(c, s)[1] + t[1]), p)
                for c, p in board.items()
            }
            if transformed != expected:
                continue
            if phase_first is not None:
                mf = d6(phase_first, s)
                if (mf[0] + t[0], mf[1] + t[1]) != phase_first:
                    continue
            out.append((s, t))
    return out


def orbit_count(cells: set[Coord], stabilizers: list[tuple[int, Coord]]) -> int:
    unseen = set(cells)
    count = 0
    while unseen:
        seed = next(iter(unseen))
        orbit = set()
        for s, t in stabilizers:
            mc = d6(seed, s)
            image = mc[0] + t[0], mc[1] + t[1]
            if image in cells:
                orbit.add(image)
        unseen.difference_update(orbit)
        count += 1
    return count


def parse_human(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        pos_id, coords = raw.split(";", 1)
        moves = [tuple(map(int, token.split(","))) for token in coords.split()]
        yield pos_id, moves


def parse_corpus_positions(path: Path) -> dict[str, dict]:
    positions: dict[str, dict] = {}
    current: str | None = None
    metadata: dict[str, str] = {}
    moves: list[Coord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("POS "):
            fields = dict(field.split("=", 1) for field in raw.split()[1:])
            current = fields["id"]
            metadata = fields
            moves = []
        elif raw == "END":
            if current is not None:
                positions[current] = {
                    "moves": moves,
                    "attacker": int(metadata["attacker"]),
                    "rem": int(metadata["rem"]),
                }
            current = None
        elif current is not None:
            q, r = map(int, raw.split())
            moves.append((q, r))
    return positions


def parse_corpus_lines(path: Path) -> dict[str, list[Coord]]:
    lines: dict[str, list[Coord]] = {}
    current: str | None = None
    moves: list[Coord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("LINE "):
            fields = dict(field.split("=", 1) for field in raw.split()[1:])
            current = fields["id"]
            moves = []
        elif raw == "END":
            if current is not None:
                lines[current] = moves
            current = None
        elif current is not None:
            q, r = map(int, raw.split())
            moves.append((q, r))
    return lines


def human_probe(path: Path) -> dict:
    rows = []
    threatened = []
    b2_tau1 = []
    b2_tau2 = []
    human_pair_boundary_checks = 0
    human_pair_boundary_violations: list[dict] = []
    dead_spare_children = []
    symmetry_rows = []
    for pos_id, moves in parse_human(path):
        board = board_from_moves(moves)
        phase, mover, budget = phase_and_player(len(moves))
        legal = legal_cells(board)
        old_support = support(board)
        dead = {c for c in legal if dead_empty(c, board)}
        inert = {c for c in legal if frontier_delta(c, old_support) == 0}
        family = threat_family(board, 1 - mover)
        tau = tau_le2(family)
        comps = hyper_components(family)
        phase_first = moves[-1] if phase == "second" else None
        stabilizers = affine_stabilizers(board, phase_first)
        orbits = orbit_count(legal, stabilizers)
        row = {
            "id": pos_id,
            "stones": len(board),
            "phase": phase,
            "budget": budget,
            "legal": len(legal),
            "dead": len(dead),
            "inert": len(inert),
            "threats": len(family),
            "tau": tau,
            "components": len(comps),
            "component_taus": [tau_le2([family[i] for i in comp]) for comp in comps],
            "affine_stabilizers": len(stabilizers),
            "legal_orbits": orbits,
        }
        rows.append(row)
        if family:
            threatened.append(row)
        if budget == 2 and family and tau == 1 and not own_win_now(board, mover, budget):
            hits = common_hits(family)
            b2_row = dict(row)
            b2_row["common_hits"] = len(hits)
            covers = covering_pairs_over(legal, family)
            b2_row["root_legal_cover_pairs"] = len(covers)
            b2_row["split_cover_pairs"] = sum(
                left not in hits and right not in hits for left, right in covers
            )
            b2_row["ordered_full_turns"] = len(legal) * max(0, len(legal) - 1)
            directed_new = sum(
                len(shifted_ball(hit).difference(old_support).difference(board)) for hit in hits
            )
            b2_row["directed_new_cover_pairs"] = directed_new
            b2_row["sequential_cover_leaves"] = 2 * len(covers) + directed_new
            b2_row["atomic_cover_children"] = len(covers) + directed_new
            b2_tau1.append(b2_row)
            for hit in sorted(hits):
                child = dict(board)
                child[hit] = mover
                child_legal = legal_cells(child)
                child_dead = {c for c in child_legal if dead_empty(c, child)}
                dead_spare_children.append(
                    {
                        "id": pos_id,
                        "hit": hit,
                        "legal": len(child_legal),
                        "dead": len(child_dead),
                    }
                )
        if budget == 2 and family and tau == 2 and not own_win_now(board, mover, budget):
            ps = minimum_pairs(family)
            tau2_row = dict(row)
            tau2_row.update(
                {
                    "universe": len(set().union(*family)),
                    "minimum_pairs": len(ps),
                    "kernel": len(set(sum((list(pair) for pair in ps), []))),
                }
            )
            b2_tau2.append(tau2_row)
            root_legal = legal_cells(board)
            attacker = 1 - mover
            for left, right in ps:
                human_pair_boundary_checks += 1
                reasons: list[str] = []
                if left not in root_legal or right not in root_legal:
                    reasons.append("pair cell not root-legal")
                for first, second in ((left, right), (right, left)):
                    child = dict(board)
                    child[first] = mover
                    residual = threat_family(child, attacker)
                    if tau_le2(residual) != 1:
                        reasons.append(f"residual tau after {first} != 1")
                    if second not in common_hits(residual):
                        reasons.append(f"mate {second} not residual common hit")
                    if second not in legal_cells(child):
                        reasons.append(f"mate {second} not child-legal")
                    if own_win_now(child, mover, 1):
                        reasons.append(f"defender win-now after first {first}")
                if reasons:
                    human_pair_boundary_violations.append(
                        {"id": pos_id, "pair": [left, right], "reasons": reasons}
                    )
        if len(stabilizers) > 1:
            symmetry_rows.append(row)

    def stats(values: list[int]) -> dict:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        return {
            "n": len(values),
            "sum": sum(values),
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "max": ordered[-1],
        }

    return {
        "input": str(path),
        "positions": len(rows),
        "root_legal": stats([r["legal"] for r in rows]),
        "root_dead": stats([r["dead"] for r in rows]),
        "root_inert": stats([r["inert"] for r in rows]),
        "positions_with_dead": sum(r["dead"] > 0 for r in rows),
        "dead_fraction_all_legal": sum(r["dead"] for r in rows) / max(1, sum(r["legal"] for r in rows)),
        "inert_fraction_all_legal": sum(r["inert"] for r in rows) / max(1, sum(r["legal"] for r in rows)),
        "threatened_positions": len(threatened),
        "threat_component_hist": dict(collections.Counter(r["components"] for r in threatened)),
        "b2_tau1_positions": b2_tau1,
        "b2_tau2_positions": b2_tau2,
        "pair_boundary_checks": human_pair_boundary_checks,
        "pair_boundary_violations": human_pair_boundary_violations,
        "dead_spare_child_count": len(dead_spare_children),
        "dead_spare_legal": stats([r["legal"] for r in dead_spare_children]),
        "dead_spare_dead": stats([r["dead"] for r in dead_spare_children]),
        "dead_spare_fraction": (
            sum(r["dead"] for r in dead_spare_children)
            / max(1, sum(r["legal"] for r in dead_spare_children))
        ),
        "affine_nontrivial_positions": symmetry_rows,
        "rows": rows,
    }


def line_probe(position_path: Path, line_path: Path) -> dict:
    positions = parse_corpus_positions(position_path)
    lines = parse_corpus_lines(line_path)
    turns = []
    pair_boundary_checks = 0
    pair_boundary_violations: list[dict] = []
    for pos_id, continuation in lines.items():
        record = positions[pos_id]
        history = list(record["moves"])
        attacker = record["attacker"]
        _phase, root_player, root_budget = phase_and_player(len(history))
        if root_player != attacker or root_budget != record["rem"]:
            raise ValueError(f"{pos_id} metadata disagrees with replay")
        current_turn: list[Coord] = []
        attacker_turn_index = 0
        for move in continuation:
            _phase, mover, _budget = phase_and_player(len(history))
            current_turn.append(move)
            history.append(move)
            _next_phase, next_mover, _next_budget = phase_and_player(len(history))
            if next_mover == mover:
                continue
            if mover == attacker:
                board = board_from_moves(history)
                family = threat_family(board, mover)
                comps = hyper_components(family)
                comp_families = [[family[i] for i in comp] for comp in comps]
                pairs = minimum_pairs(family)
                if tau_le2(family) == 2:
                    defender = 1 - mover
                    root_legal = legal_cells(board)
                    for left, right in pairs:
                        pair_boundary_checks += 1
                        reasons: list[str] = []
                        if left not in root_legal or right not in root_legal:
                            reasons.append("pair cell not root-legal")
                        if own_win_now(board, defender, 2):
                            reasons.append("defender win-now at root")
                        for first, second in ((left, right), (right, left)):
                            child = dict(board)
                            child[first] = defender
                            residual = threat_family(child, mover)
                            if tau_le2(residual) != 1:
                                reasons.append(f"residual tau after {first} != 1")
                            if second not in common_hits(residual):
                                reasons.append(f"mate {second} not residual common hit")
                            if second not in legal_cells(child):
                                reasons.append(f"mate {second} not child-legal")
                            if own_win_now(child, defender, 1):
                                reasons.append(f"defender win-now after first {first}")
                        if reasons:
                            pair_boundary_violations.append(
                                {"id": pos_id, "turn": attacker_turn_index, "pair": [left, right], "reasons": reasons}
                            )
                turns.append(
                    {
                        "id": pos_id,
                        "turn": attacker_turn_index,
                        "placements": list(current_turn),
                        "threats": len(family),
                        "tau": tau_le2(family),
                        "components": len(comps),
                        "component_sizes": [len(comp) for comp in comps],
                        "component_taus": [tau_le2(f) for f in comp_families],
                        "component_common_hits": [len(common_hits(f)) for f in comp_families],
                        "universe": len(set().union(*family)) if family else 0,
                        "minimum_pairs": len(pairs),
                        "kernel": len(set(sum((list(pair) for pair in pairs), []))),
                    }
                )
                attacker_turn_index += 1
            current_turn = []
    return {
        "positions": len(lines),
        "attacker_turns": len(turns),
        "component_hist": dict(collections.Counter(t["components"] for t in turns)),
        "tau_hist": dict(collections.Counter(str(t["tau"]) for t in turns)),
        "split_tau2_turns": [
            t
            for t in turns
            if t["tau"] == 2
            and len(t["component_taus"]) == 2
            and sorted(t["component_taus"]) == [1, 1]
        ],
        "pair_boundary_checks": pair_boundary_checks,
        "pair_boundary_violations": pair_boundary_violations,
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", type=Path)
    parser.add_argument("--positions", type=Path)
    parser.add_argument("--lines", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {}
    if args.human:
        result["human"] = human_probe(args.human)
    if args.positions and args.lines:
        result["forcing_lines"] = line_probe(args.positions, args.lines)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
