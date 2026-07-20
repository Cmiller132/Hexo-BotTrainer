"""Finite checks for coordinate certificates in _TIGHTNESS_FRONTIER_REPORT.md.

This is deliberately independent of the solver.  It enumerates every length-six
window incident to a finite position, computes threat families/transversals, and
checks the small distance/count certificates used in the report.
"""

from itertools import combinations


if not __debug__:
    raise RuntimeError("this verifier requires assertions; rerun without -O")


AXES = ((1, 0), (0, 1), (1, -1))


def add(x, y):
    return (x[0] + y[0], x[1] + y[1])


def scale(k, x):
    return (k * x[0], k * x[1])


def distance(x, y):
    dq = x[0] - y[0]
    dr = x[1] - y[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def window(start, axis):
    return frozenset(add(start, scale(i, axis)) for i in range(6))


def incident_windows(cells):
    result = set()
    for x in cells:
        for axis in AXES:
            for offset in range(6):
                result.add((add(x, scale(-offset, axis)), axis))
    return result


def threats(attacker, defender):
    result = []
    for start, axis in incident_windows(attacker):
        w = window(start, axis)
        if not (w & defender) and len(w & attacker) >= 4:
            result.append((start, axis, len(w & attacker), w - attacker - defender))
    return sorted(result, key=repr)


def complete_windows(own, other):
    result = []
    for start, axis in incident_windows(own):
        w = window(start, axis)
        if not (w & other) and len(w & own) == 6:
            result.append((start, axis))
    return result


def own_win_now(own, other, budget):
    threshold = 4 if budget == 2 else 5
    return any(count >= threshold for _, _, count, _ in threats(own, other))


def transversal_number(family):
    universe = sorted(set().union(*family)) if family else []
    for size in range(len(universe) + 1):
        for hit in combinations(universe, size):
            hit = frozenset(hit)
            if all(hit & edge for edge in family):
                return size
    raise AssertionError("finite family must have a transversal")


def smallest_obstruction(family, budget):
    for size in range(1, len(family) + 1):
        for selected in combinations(family, size):
            if transversal_number(selected) > budget:
                return size
    return None


def legal_cells(attacker, defender):
    stones = attacker | defender
    q0 = min(q for q, _ in stones) - 8
    q1 = max(q for q, _ in stones) + 8
    r0 = min(r for _, r in stones) - 8
    r1 = max(r for _, r in stones) + 8
    return {
        (q, r)
        for q in range(q0, q1 + 1)
        for r in range(r0, r1 + 1)
        if (q, r) not in stones
        and any(distance((q, r), stone) <= 8 for stone in stones)
    }


def kernel(attacker, defender, budget):
    family = [empty for _, _, _, empty in threats(attacker, defender)]
    result = set()
    for d in legal_cells(attacker, defender):
        residual = [edge for edge in family if d not in edge]
        if transversal_number(residual) <= budget - 1:
            result.add(d)
    return result


def exact_family(attacker, defender, expected):
    assert not (attacker & defender)
    assert (0, 0) in attacker | defender
    assert not complete_windows(attacker, defender)
    assert not complete_windows(defender, attacker)
    actual_threats = threats(attacker, defender)
    assert len(actual_threats) == len(expected)
    actual = {empty for _, _, _, empty in actual_threats}
    assert actual == set(expected), (actual, set(expected))
    return actual


def check_sparse_witnesses():
    # Triangle: the complete threat family needs all three members for tau > 1.
    vertices = {(4, 0), (5, 0), (4, 1)}
    named = [
        window((0, 0), (1, 0)),
        window((4, -4), (0, 1)),
        window((5, 0), (-1, 1)),
    ]
    attacker = set().union(*(w - vertices for w in named))
    defender = {(-1, 0), (4, -5), (-1, 6)}
    expected = [w - attacker - defender for w in named]
    family = exact_family(attacker, defender, expected)
    assert transversal_number(family) == 2
    assert smallest_obstruction(tuple(family), 1) == 3
    assert not own_win_now(defender, attacker, 1)
    print("triangle: threats=3 tau=2 minimum-witness=3")

    # Five-cycle: the complete threat family needs all five members for tau > 2.
    vertices = {(4, 0), (5, 0), (6, 0), (4, 2), (4, 1)}
    named = [
        window((0, 0), (1, 0)),
        window((5, 0), (1, 0)),
        window((6, 0), (-1, 1)),
        window((4, 1), (0, 1)),
        window((4, -4), (0, 1)),
    ]
    attacker = set().union(*(w - vertices for w in named))
    defender = {(-1, 0), (4, -5), (11, 0), (4, 7), (0, 6)}
    expected = [w - attacker - defender for w in named]
    assert len(attacker) == 20
    family = exact_family(attacker, defender, expected)
    assert transversal_number(family) == 3
    assert smallest_obstruction(tuple(family), 2) == 5
    assert not own_win_now(defender, attacker, 2)
    print("five-cycle: threats=5 tau=3 minimum-witness=5")


def check_t6_own_win_counterexample():
    attacker = {
        (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0),
        (1, -1), (2, -2), (3, -3),
        (31, 10), (32, 10), (33, 10), (34, 10), (35, 10),
        (59, 25), (66, 25),
    }
    defender = {
        (29, 10), (36, 10),
        (61, 25), (62, 25), (63, 25), (64, 25), (65, 25),
    }
    k = (30, 10)
    p = (0, 0)
    d = (60, 25)
    family = [empty for _, _, _, empty in threats(attacker, defender)]
    assert family == [frozenset({k})]
    assert kernel(attacker, defender, 1) == {k}
    assert own_win_now(defender, attacker, 1)
    assert d in legal_cells(attacker, defender)
    assert complete_windows(defender | {d}, attacker)

    defender_after_k = defender | {k}
    attacker_after_p = attacker | {p}
    attacker_at_leaf = attacker_after_p | {d}
    named = [
        frozenset({(-2, 0), (-1, 0)}),
        frozenset({(0, -2), (0, -1)}),
        frozenset({(-2, 2), (-1, 1)}),
    ]
    leaf_family = {
        empty for _, _, _, empty in threats(attacker_at_leaf, defender_after_k)
    }
    assert set(named) <= leaf_family
    assert transversal_number(named) == 3
    assert not own_win_now(defender_after_k, attacker_at_leaf, 2)
    assert distance(p, (0, 1)) == 1
    assert distance(d, (59, 25)) == 1
    print("T6 own-win omission: K1={k}; omitted d is immediate D completion")


def check_deadline_counterexamples():
    # Omitting the OR-COMPLETION role lets D occupy the designated cell.
    attacker = {(0, 0), (1, 0), (2, 0), (4, 0), (5, 0)}
    defender = set()
    c = (3, 0)
    assert own_win_now(attacker, defender, 2)
    assert complete_windows(attacker | {c}, defender)
    assert not threats(attacker, {c})

    # Dropping LOSS witness roles one D edge before leaf entry breaks tau transfer.
    attacker = {
        (0, 0), (1, 0), (4, 0), (5, 0),
        (0, 10), (1, 10), (4, 10), (5, 10),
    }
    defender = set()
    u = (2, 0)
    v = (2, 10)
    family = [empty for _, _, _, empty in threats(attacker, defender)]
    assert set(family) == {
        frozenset({(2, 0), (3, 0)}),
        frozenset({(2, 10), (3, 10)}),
    }
    assert transversal_number(family) == 2
    assert not threats(attacker, {u, v})
    print("deadline roles: OR-COMPLETION and leaf-entry protection checks pass")


def check_distance_and_count_bounds():
    # Exact-rank chain, r=2.
    o = (0, 0)
    z = (0, 16)
    x0 = (8, 0)
    y = (16, 0)
    a = (8, 8)
    f0 = (-8, 0)
    f1 = (-16, 0)
    assert distance(o, x0) == distance(x0, y) == 8
    assert min(distance(y, stone) for stone in (o, z)) == 16
    assert distance(f0, f1) == distance(z, a) == distance(a, y) == 8

    # Fixed-window virgin arithmetic for E=6..14.
    step = (8, -4)
    target = window((0, 0), (1, 0))
    for exposure in range(6, 15):
        relays = exposure - 6
        if relays:
            seed = scale(-relays, step)
            assert min(distance(seed, w) for w in target) == 8 * relays
            chain = [scale(-(relays - i), step) for i in range(relays)]
            assert all(distance(x, y_) == 8 for x, y_ in zip(chain, chain[1:]))
            assert distance(chain[-1], (0, 0)) == 8

    # Touched equality: two old D stones plus four fills attain six.
    touched = {(0, 0), (5, 0)}
    for fill in ((1, 0), (2, 0), (3, 0), (4, 0)):
        assert min(distance(fill, stone) for stone in touched) <= 8
        touched.add(fill)
    assert len(touched) == 6

    # T5 endpoints: B=4 needs radius 4; B=3 needs radius 3.
    assert distance((-5, 0), (-1, 0)) == 4
    assert distance((-5, 0), (-2, 0)) == 3
    print("distance/count chains: rank, virgin arithmetic, touched equality, T5")


def main():
    check_sparse_witnesses()
    check_t6_own_win_counterexample()
    check_deadline_counterexamples()
    check_distance_and_count_bounds()
    print("all tightness checks passed")


if __name__ == "__main__":
    main()
