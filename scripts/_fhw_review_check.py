#!/usr/bin/env python3
"""Hostile machine checks for docs/_OPEN_FHW_REPORT.md.

Offline and standard-library only.  Every reported check occupies one output
line.  A failed check makes the process exit nonzero.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product
import sys


Cell = tuple[int, int]
Window = tuple[Cell, ...]
Position = dict[Cell, str]

AXES: tuple[Cell, ...] = ((1, 0), (0, 1), (1, -1))


def add(x: Cell, y: Cell) -> Cell:
    return x[0] + y[0], x[1] + y[1]


def scale(k: int, x: Cell) -> Cell:
    return k * x[0], k * x[1]


def dist(x: Cell, y: Cell) -> int:
    dq, dr = x[0] - y[0], x[1] - y[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def window(start: Cell, axis: Cell) -> Window:
    return tuple(add(start, scale(i, axis)) for i in range(6))


def incident_windows(cells: set[Cell] | Position) -> set[Window]:
    result: set[Window] = set()
    for c in cells:
        for axis in AXES:
            for offset in range(6):
                result.add(window(add(c, scale(-offset, axis)), axis))
    return result


def counts(pos: Position, w: Window) -> tuple[int, int]:
    return (
        sum(pos.get(c) == "A" for c in w),
        sum(pos.get(c) == "D" for c in w),
    )


def complete_windows(pos: Position, colour: str | None = None) -> list[Window]:
    wins: list[Window] = []
    for w in incident_windows(pos):
        ca, cd = counts(pos, w)
        if (colour in (None, "A") and ca == 6) or (
            colour in (None, "D") and cd == 6
        ):
            wins.append(w)
    return wins


def threat_empty_counter(pos: Position, colour: str = "A") -> Counter[frozenset[Cell]]:
    result: Counter[frozenset[Cell]] = Counter()
    for w in incident_windows(pos):
        ca, cd = counts(pos, w)
        own, other = (ca, cd) if colour == "A" else (cd, ca)
        if other == 0 and own >= 4:
            result[frozenset(c for c in w if c not in pos)] += 1
    return result


def d_alive_max(pos: Position) -> int:
    best = 0
    for w in incident_windows(pos):
        ca, cd = counts(pos, w)
        if ca == 0:
            best = max(best, cd)
    return best


def legal(pos: Position, c: Cell) -> bool:
    return c not in pos and any(dist(c, s) <= 8 for s in pos)


def legal_cells(pos: Position) -> set[Cell]:
    result: set[Cell] = set()
    for q, r in pos:
        for dq in range(-8, 9):
            for dr in range(-8, 9):
                c = q + dq, r + dr
                if c not in pos and dist(c, (q, r)) <= 8:
                    result.add(c)
    return result


def place(pos: Position, c: Cell, colour: str) -> Position:
    if not legal(pos, c):
        raise AssertionError(f"illegal {colour} placement at {c}")
    out = dict(pos)
    out[c] = colour
    return out


def tau(family: list[set[Cell]] | tuple[set[Cell], ...]) -> int:
    if not family:
        return 0
    if any(not edge for edge in family):
        return sys.maxsize
    universe = sorted(set().union(*family))
    for size in range(1, len(universe) + 1):
        for hit in combinations(universe, size):
            hit_set = set(hit)
            if all(hit_set & edge for edge in family):
                return size
    return sys.maxsize


def kernel(pos: Position, family: list[set[Cell]], budget: int) -> set[Cell]:
    result: set[Cell] = set()
    for d in legal_cells(pos):
        residual = [edge for edge in family if d not in edge]
        if tau(residual) <= budget - 1:
            result.add(d)
    return result


def connected_at_radius(pos: Position, radius: int = 8) -> bool:
    if not pos:
        return True
    unseen = set(pos)
    stack = [unseen.pop()]
    while stack:
        x = stack.pop()
        reached = {y for y in unseen if dist(x, y) <= radius}
        unseen.difference_update(reached)
        stack.extend(reached)
    return not unseen


RESULTS: list[tuple[str, bool, str]] = []


def check(identifier: str, condition: bool, detail: str) -> None:
    RESULTS.append((identifier, bool(condition), detail))


def section_1_1() -> None:
    history: list[tuple[str, Cell]] = [
        ("D", (0, 0)),  # fixed Opening; D4 locality starts afterward
        ("A", (0, -4)),
        ("A", (1, -4)),
        ("D", (0, 1)),
        ("D", (-1, -4)),
        ("A", (2, -4)),
        ("A", (3, -4)),
        ("D", (0, 2)),
        ("D", (0, -8)),
        ("A", (4, -4)),
        ("A", (-4, 8)),
        ("D", (0, 3)),
        ("D", (1, 8)),
        ("A", (-3, 8)),
        ("A", (-2, 8)),
        ("D", (-8, 0)),
        ("D", (8, -8)),
        ("A", (-1, 8)),
        ("A", (0, 8)),
    ]
    pos: Position = {history[0][1]: history[0][0]}
    legal_ok = True
    prefix_ok = not complete_windows(pos)
    for colour, c in history[1:]:
        legal_ok &= legal(pos, c)
        if legal(pos, c):
            pos[c] = colour
        prefix_ok &= not complete_windows(pos)
    check(
        "1.1-history-legality",
        legal_ok,
        "fixed Opening plus all 18 later placements satisfy radius-8 legality",
    )
    check(
        "1.1-prefix-nonterminal",
        prefix_ok,
        "neither colour has a complete window at any setup prefix",
    )

    w = tuple((0, r) for r in range(6))
    a, b = (5, -4), (-5, 8)
    expected = Counter(
        {
            frozenset({a}): 1,
            frozenset({a, (6, -4)}): 1,
            frozenset({b}): 1,
            frozenset({b, (-6, 8)}): 1,
        }
    )
    actual = threat_empty_counter(pos)
    check(
        "1.1-threat-family",
        actual == expected,
        f"complete A-threat empty multiset is the four displayed sets (found {dict(actual)})",
    )
    family = [set(edge) for edge in actual.elements()]
    k = kernel(pos, family, 2)
    check(
        "1.1-tau-kernel",
        tau(family) == 2 and k == {a, b},
        f"tau(F)=2 and K={{a,b}} (tau={tau(family)}, K={sorted(k)})",
    )
    check(
        "1.1-kernel-disjoint-W",
        not (k & set(w)),
        "the full extendable-hit kernel is disjoint from W",
    )

    u, v = (0, 4), (0, 5)
    both_start_legal = legal(pos, u) and legal(pos, v)
    after_u = place(pos, u, "D")
    after_v = place(after_u, v, "D")
    check(
        "1.1-defender-two-fill",
        both_start_legal
        and counts(after_u, w)[1] == 5
        and not complete_windows(after_u, "D")
        and counts(after_v, w)[1] == 6
        and bool(complete_windows(after_v, "D")),
        "u and v are turn-start legal; counts are 5 then 6 and D wins only on v",
    )


def section_1_2() -> None:
    w = tuple((q, 0) for q in range(6))
    a_gate = {
        (10, 0),
        (11, 0),
        (12, 0),
        (20, 0),
        (21, 0),
        (22, 0),
    }
    a_fork = {
        (-3, 20),
        (-2, 20),
        (-1, 20),
        (3, 20),
        (0, 17),
        (0, 18),
        (0, 19),
        (-3, 23),
        (-2, 22),
        (-1, 21),
    }
    d_base = {(0, 0), (1, 0), (2, 0), (9, 0), (19, 0)}
    ghost_n: Position = {c: "A" for c in a_gate | a_fork}
    ghost_n.update({c: "D" for c in d_base | {(0, 8)}})
    real_n: Position = {c: "A" for c in a_gate | a_fork}
    real_n.update({c: "D" for c in d_base | {(5, 0)}})
    check(
        "1.2-fragment-at-N",
        not threat_empty_counter(ghost_n)
        and d_alive_max(ghost_n) == 3
        and counts(ghost_n, w)[1] == 3
        and counts(real_n, w)[1] == 4
        and not complete_windows(ghost_n)
        and not complete_windows(real_n),
        "N has no A-threat, ghost D-alive maximum 3, and W counts ghost 3/real 4",
    )
    bare_n: Position = {c: "A" for c in a_gate | a_fork}
    bare_n.update({c: "D" for c in d_base})
    x, y = (5, 0), (0, 8)
    check(
        "1.2-divergent-replies",
        legal(bare_n, x)
        and legal(bare_n, y)
        and not complete_windows(place(bare_n, x, "D"), "D")
        and not complete_windows(place(bare_n, y, "D"), "D"),
        "x and y are legal nonwinning replies and yield X={x}, Y={y}",
    )

    ghost_q = place(place(ghost_n, (13, 0), "A"), (23, 0), "A")
    real_q = place(place(real_n, (13, 0), "A"), (23, 0), "A")
    e1, e2 = {(14, 0), (15, 0)}, {(24, 0), (25, 0)}
    expected = Counter({frozenset(e1): 1, frozenset(e2): 1})
    actual = threat_empty_counter(ghost_q)
    family = [set(edge) for edge in actual.elements()]
    k = kernel(ghost_q, family, 2)
    check(
        "1.2-gate-threats",
        actual == expected and tau(family) == 2 and not complete_windows(ghost_q),
        f"Q has exactly E1,E2 as A-threat empties and tau=2 (found {dict(actual)})",
    )
    check(
        "1.2-gate-kernel",
        k == e1 | e2 and not (k & set(w)),
        f"K=E1 union E2 and is disjoint from W (K={sorted(k)})",
    )

    r1, r2 = (3, 0), (4, 0)
    both_start_legal = legal(real_q, r1) and legal(real_q, r2)
    after_r1 = place(real_q, r1, "D")
    after_r2 = place(after_r1, r2, "D")
    check(
        "1.2-real-defender-win",
        counts(real_q, w)[1] == 4
        and both_start_legal
        and counts(after_r1, w)[1] == 5
        and not complete_windows(after_r1, "D")
        and counts(after_r2, w)[1] == 6
        and bool(complete_windows(after_r2, "D"))
        and all(c not in after_r2 for c in e1 | e2),
        "real D ignores the disjoint gate, reaches W counts 5/6, and wins with all threat cells empty",
    )

    u1 = window((-3, 20), (1, 0))
    u2 = window((0, 17), (0, 1))
    u3 = window((-3, 23), (1, -1))
    expected_pairs = (
        {(1, 20), (2, 20)},
        {(0, 21), (0, 22)},
        {(1, 19), (2, 18)},
    )
    loss_ok = True
    loss_details: list[str] = []
    for first_edge, second_edge in ((e1, e2), (e2, e1)):
        for h1, h2 in product(sorted(first_edge), sorted(second_edge)):
            after_h1 = place(ghost_q, h1, "D")
            after_h2 = place(after_h1, h2, "D")
            after_p = place(after_h2, (3, 0), "A")
            leaf = place(after_p, (0, 20), "A")
            empties = tuple(
                {c for c in u if c not in leaf} for u in (u1, u2, u3)
            )
            this_ok = (
                not complete_windows(after_h1)
                and not complete_windows(after_h2)
                and not complete_windows(after_p)
                and empties == expected_pairs
                and all(counts(leaf, u) == (4, 0) for u in (u1, u2, u3))
                and all(
                    empties[i].isdisjoint(empties[j])
                    for i, j in combinations(range(3), 2)
                )
                and tau([set(e) for e in empties]) == 3
                and d_alive_max(leaf) == 3
                and not complete_windows(leaf)
                and all(
                    h not in set(u1) | set(u2) | set(u3) for h in (h1, h2)
                )
            )
            loss_ok &= this_ok
            if not this_ok:
                loss_details.append(
                    f"hits {h1},{h2}: empties={empties}, Dmax={d_alive_max(leaf)}"
                )
    check(
        "1.2-loss-leaf",
        loss_ok,
        "all eight ordered exact-hit lines give the three displayed disjoint empty pairs, tau=3, and D-alive max 3"
        + ("; " + "; ".join(loss_details) if loss_details else ""),
    )


def section_1_3_and_1_4() -> None:
    a_runs = {(q, 0) for q in range(4)} | {(q, 0) for q in range(30, 34)}
    shared_d = {(-1, 0), (29, 0)}
    ghost: Position = {c: "A" for c in a_runs}
    ghost.update({c: "D" for c in shared_d | {(0, 8)}})
    real: Position = {c: "A" for c in a_runs}
    real.update({c: "D" for c in shared_d | {(4, 0)}})
    e1, e2 = {(4, 0), (5, 0)}, {(34, 0), (35, 0)}
    ghost_f = threat_empty_counter(ghost)
    real_f = threat_empty_counter(real)
    check(
        "1.3-mask-count",
        ghost_f == Counter({frozenset(e1): 1, frozenset(e2): 1})
        and tau([set(e) for e in ghost_f.elements()]) == 2
        and real_f == Counter({frozenset(e2): 1})
        and tau([set(e) for e in real_f.elements()]) == 1,
        "ghost masks give two disjoint threats/tau 2; real X at (4,0) kills the first and leaves tau 1",
    )

    # Section 1.4: after the two substituted hits, include the stated shared
    # A support t.  The report leaves its intervening shared A turn unnamed;
    # (33,1),(33,2) is a concrete inert witness for the install repair.
    shared_a = set(a_runs) | {(5, 25)}
    ghost_14: Position = {c: "A" for c in shared_a}
    ghost_14.update({c: "D" for c in shared_d | {(4, 0), (34, 0)}})
    real_14: Position = {c: "A" for c in shared_a}
    real_14.update({c: "D" for c in shared_d | {(5, 0), (35, 0)}})
    shared_turn = ((33, 1), (33, 2))
    turn_ok = all(
        legal(ghost_14, c) and legal(real_14, c) for c in shared_turn
    )
    for c in shared_turn:
        turn_ok &= legal(ghost_14, c) and legal(real_14, c)
        ghost_14 = place(ghost_14, c, "A")
        real_14 = place(real_14, c, "A")
        turn_ok &= not complete_windows(ghost_14) and not complete_windows(real_14)
    first, zeta = (5, 8), (5, 16)
    ghost_first_min = min(dist(first, s) for s in ghost_14)
    real_first_min = min(dist(first, s) for s in real_14)
    real_after_first = place(real_14, first, "D")
    ghost_zeta_min = min(dist(zeta, s) for s in ghost_14)
    real_zeta_min = min(dist(zeta, s) for s in real_after_first)
    check(
        "1.4-substitution-masks",
        set(real_14) - set(ghost_14) == {(5, 0), (35, 0)}
        and set(ghost_14) - set(real_14) == {(4, 0), (34, 0)},
        "substituted hits create exactly X={(5,0),(35,0)} and Y={(4,0),(34,0)}",
    )
    check(
        "1.4-shared-turn-witness",
        turn_ok,
        "the repair pair (33,1),(33,2) is a legal nonterminal shared A turn",
    )
    check(
        "1.4-first-link-distance",
        real_first_min == 8 and ghost_first_min == 9 and legal(real_14, first) and not legal(ghost_14, first),
        f"(5,8) is real-legal at distance 8 and ghost-illegal with nearest distance 9 (real={real_first_min}, ghost={ghost_first_min})",
    )
    check(
        "1.4-second-link-distance",
        real_zeta_min == 8 and ghost_zeta_min == 9 and legal(real_after_first, zeta) and not legal(ghost_14, zeta),
        f"zeta is real-legal at distance 8 and ghost-illegal with nearest distance 9 (real={real_zeta_min}, ghost={ghost_zeta_min})",
    )


def sharpness_position() -> tuple[Position, Window, Window, Cell]:
    a = {(q, 0) for q in range(5)} | {
        (-3, 20),
        (-2, 20),
        (-1, 20),
        (0, 17),
        (0, 18),
        (0, 19),
        (-3, 23),
        (-2, 22),
        (-1, 21),
        (0, -7),
    }
    d = {
        (-1, 0),
        (8, 0),
        (8, 8),
        (8, 16),
        (8, 24),
        (8, 32),
        (0, 40),
        (1, 40),
        (2, 40),
        (-8, 0),
        (0, -8),
        (-8, 8),
        (8, -8),
        (16, -8),
        (16, 0),
    }
    pos: Position = {c: "A" for c in a}
    pos.update({c: "D" for c in d})
    return pos, window((0, 40), (1, 0)), window((5, 0), (1, 0)), (5, 0)


def section_4() -> None:
    pos, w, w_prime, h = sharpness_position()
    check(
        "4.1-position",
        sum(v == "A" for v in pos.values()) == 15
        and sum(v == "D" for v in pos.values()) == 15
        and not complete_windows(pos)
        and connected_at_radius(pos),
        "the displayed 15+15 position is nonterminal and radius-8 connected",
    )
    check(
        "4.1-D-alive-maximum",
        d_alive_max(pos) == 3,
        f"maximum count in a D-alive window is exactly 3 (found {d_alive_max(pos)})",
    )
    t = window((0, 0), (1, 0))
    named_family = [{h}]
    named_k = kernel(pos, named_family, 1)
    check(
        "4.1-singleton-gate-threat",
        counts(pos, t) == (5, 0)
        and {c for c in t if c not in pos} == {h}
        and tau(named_family) == 1
        and named_k == {h},
        f"named T is a singleton-empty A-threat with tau=1 and K={{h}} (K={sorted(named_k)})",
    )

    old_e = 1 + 2
    new_q = max(1, int(h in w) + 2)
    check(
        "4.1-exposure-arithmetic",
        counts(pos, w)[1] == 3
        and old_e == 3
        and new_q == 2
        and counts(pos, w)[1] + old_e == 6
        and counts(pos, w)[1] + new_q == 5,
        "W starts at 3; old E=3 gives 6 while debited Q=2 gives 5",
    )

    after_h = place(pos, h, "D")
    after_p = place(after_h, (3, 20), "A")
    leaf = place(after_p, (0, 20), "A")
    u1 = window((-3, 20), (1, 0))
    u2 = window((0, 17), (0, 1))
    u3 = window((-3, 23), (1, -1))
    loss_empties = [{c for c in u if c not in leaf} for u in (u1, u2, u3)]
    loss_contract_ok = (
        not complete_windows(after_h)
        and not complete_windows(after_p)
        and tau(loss_empties) == 3
        and all(counts(leaf, u) == (4, 0) for u in (u1, u2, u3))
        and d_alive_max(leaf) == 3
        and not complete_windows(leaf)
    )
    after_3 = place(leaf, (3, 40), "D")
    after_4 = place(after_3, (4, 40), "D")
    survivor = u1
    survivor_empties = [c for c in survivor if c not in after_4]
    won = dict(after_4)
    attacker_line_ok = len(survivor_empties) == 2
    for i, c in enumerate(survivor_empties):
        attacker_line_ok &= legal(won, c)
        won = place(won, c, "A")
        if i == 0:
            attacker_line_ok &= not complete_windows(won, "A")
    attacker_line_ok &= bool(complete_windows(won, "A"))
    check(
        "4.1-loss-contract",
        loss_contract_ok,
        "after h and the two A moves, the named LOSS pairs are disjoint with tau=3 and D-alive max 3",
    )
    check(
        "4.1-attained-real-line",
        counts(after_3, w)[1] == 4
        and not complete_windows(after_3, "D")
        and counts(after_4, w)[1] == 5
        and not complete_windows(after_4, "D")
        and attacker_line_ok,
        "the legal two-fill LOSS remainder attains two W-hits/count 5, then A completes a surviving witness",
    )

    initial_wp = counts(pos, w_prime)
    q_with_indicator = max(1, int(h in w_prime) + 2)
    q_without_indicator = max(1, 0 + 2)
    line_after_6 = place(leaf, (6, 0), "D")
    line = place(line_after_6, (7, 0), "D")
    check(
        "4.2-dual-purpose-arithmetic",
        initial_wp == (0, 1)
        and h in w_prime
        and q_with_indicator == 3
        and q_without_indicator == 2,
        "W' starts with one D; Q=3 is attained algebraically and deleting 1[h in W'] gives the false value 2",
    )
    check(
        "4.2-dual-purpose-line",
        all(c in w_prime for c in (h, (6, 0), (7, 0)))
        and not complete_windows(line_after_6, "D")
        and counts(line, w_prime) == (0, 4)
        and not complete_windows(line, "D"),
        "h,(6,0),(7,0) are legal W' hits, realize three units of harm, and do not complete W'",
    )


def main() -> int:
    section_1_1()
    section_1_2()
    section_1_3_and_1_4()
    section_4()
    failures = 0
    for identifier, ok, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        failures += not ok
        print(f"{status} {identifier}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
