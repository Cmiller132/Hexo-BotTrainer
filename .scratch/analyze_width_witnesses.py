from __future__ import annotations

from itertools import combinations

AXES = ((1, 0), (0, 1), (1, -1))

ROOTS = {
    "oa-0153903c5a863630": {
        "moves": [(0, 0), (-6, 1), (-7, 2), (0, 1), (0, 2), (-7, 1), (-8, 2)],
        "lift": [(0, -1), (-1, 2)],
    },
    "oa-773ca1a59e95f4e1": {
        "moves": [(0, 0), (-8, 0), (-8, 1), (1, -1), (2, -2), (-8, 2), (-9, 2)],
        "lift": [(3, -3), (3, -2)],
    },
    "oa-6fda812864c6d19a": {
        "moves": [(0, 0), (-4, 8), (4, -8), (-1, 0), (1, 0), (1, -1), (-1, 1), (-2, 0)],
        "lift": [(0, -2)],
    },
}


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def scale(a, n):
    return a[0] * n, a[1] * n


def owner_at(index: int) -> int:
    if index == 0:
        return 0
    return 1 if ((index - 1) // 2) % 2 == 0 else 0


def mover_after(n: int) -> tuple[int, str]:
    if n == 0:
        return 0, "Opening"
    post_opening = n - 1
    turn = post_opening // 2
    mover = 1 if turn % 2 == 0 else 0
    phase = "FirstStone" if post_opening % 2 == 0 else "SecondStone"
    return mover, phase


def board(moves):
    return {c: owner_at(i) for i, c in enumerate(moves)}


def windows_touching(cells):
    keys = set()
    for c in cells:
        for axis in AXES:
            for off in range(6):
                keys.add((add(c, scale(axis, -off)), axis))
    for start, axis in sorted(keys):
        yield tuple(add(start, scale(axis, i)) for i in range(6))


def active_windows(b, owner, extra_cells=()):
    out = []
    for w in windows_touching(tuple(b) + tuple(extra_cells)):
        mine = sum(b.get(c) == owner for c in w)
        theirs = sum(b.get(c) == 1 - owner for c in w)
        if mine and not theirs:
            out.append((mine, tuple(c for c in w if c not in b), w))
    return out


def candidate_reasons(b, claimant, c):
    reasons = []
    for cnt, empties, w in active_windows(b, claimant, (c,)):
        if c in empties and cnt >= 2:
            reasons.append((f"own-c{cnt}", w))
    for cnt, empties, w in active_windows(b, 1 - claimant, (c,)):
        if c in empties and cnt >= 4:
            reasons.append((f"block-c{cnt}", w))
    return reasons


def tau(sets):
    if not sets:
        return 0
    if any(not s for s in sets):
        return 99
    universe = sorted(set().union(*map(set, sets)))
    for k in (1, 2):
        if any(all(set(choice) & set(s) for s in sets) for choice in combinations(universe, k)):
            return k
    return 3


def normal_candidates(b, claimant):
    out = set()
    for cnt, empties, _ in active_windows(b, claimant):
        if cnt >= 2:
            out.update(empties)
    for cnt, empties, _ in active_windows(b, 1 - claimant):
        if cnt >= 4:
            out.update(empties)
    return out


def q1_candidates(b, claimant):
    out = set()
    for cnt, empties, _ in active_windows(b, claimant):
        if cnt == 1:
            out.update(empties)
    return out - normal_candidates(b, claimant)


def q1_degree(b, claimant, c):
    return sum(
        1
        for cnt, empties, _ in active_windows(b, claimant, (c,))
        if cnt == 1 and c in empties
    )


def q1_axis_support(b, claimant, c):
    support = {}
    for cnt, empties, w in active_windows(b, claimant, (c,)):
        if cnt == 1 and c in empties:
            axis = (w[1][0] - w[0][0], w[1][1] - w[0][1])
            support[axis] = support.get(axis, 0) + 1
    return tuple(sorted(support.values(), reverse=True))


def forcing_after(b, claimant):
    threats = [e for cnt, e, _ in active_windows(b, claimant) if cnt >= 4]
    defender_now = any(cnt >= 4 for cnt, _, _ in active_windows(b, 1 - claimant))
    return (not defender_now and tau(threats) >= 2), tau(threats), len(threats)


def current_pair_children(b, claimant):
    firsts = normal_candidates(b, claimant)
    seen = set()
    children = set()
    for first in firsts:
        b1 = dict(b)
        b1[first] = claimant
        seconds = normal_candidates(b1, claimant)
        # G1 promotion: count-one windows through the first stone.
        for cnt, empties, w in active_windows(b, claimant, (first,)):
            if cnt == 1 and first in empties:
                seconds.update(c for c in empties if c != first)
        for second in seconds:
            if second == first:
                continue
            pair = tuple(sorted((first, second)))
            if pair in seen:
                continue
            seen.add(pair)
            b2 = dict(b1)
            b2[second] = claimant
            if forcing_after(b2, claimant)[0]:
                children.add(pair)
    return firsts, children


def exact_second_universe(root_b, claimant, first):
    # WideTurnGate::second_candidates: T(P), count>=2 promotions through a,
    # and G1(P,a) from claimant count-one windows through a.
    out = set(normal_candidates(root_b, claimant))
    for cnt, empties, _ in active_windows(root_b, claimant, (first,)):
        if first not in empties:
            continue
        if cnt >= 1:
            out.update(empties)
    out.discard(first)
    return out


def j2_after_first(root_b, claimant, first, minimum_axis_support=1):
    b1 = dict(root_b)
    b1[first] = claimant
    exact = exact_second_universe(root_b, claimant, first)
    q1 = q1_candidates(b1, claimant)
    return {
        c for c in q1 - exact
        if len(q1_axis_support(b1, claimant, c)) >= 2
        and q1_axis_support(b1, claimant, c)[1] >= minimum_axis_support
    }


def tempo_seed_pair_children(b, claimant):
    firsts, current = current_pair_children(b, claimant)
    added = set()
    q1 = q1_candidates(b, claimant)
    first_alone = []
    for first in firsts:
        b1 = dict(b)
        b1[first] = claimant
        if not forcing_after(b1, claimant)[0]:
            continue
        first_alone.append(first)
        for second in q1:
            if second == first:
                continue
            pair = tuple(sorted((first, second)))
            if pair in current:
                continue
            b2 = dict(b1)
            b2[second] = claimant
            if forcing_after(b2, claimant)[0]:
                added.add(pair)
    return firsts, current, q1, first_alone, added


def show_window(w):
    return "[" + " ".join(f"{q},{r}" for q, r in w) + "]"


for rid, rec in ROOTS.items():
    moves = rec["moves"]
    claimant, phase = mover_after(len(moves))
    b = board(moves)
    print(f"\n{rid} claimant=P{claimant} phase={phase}")
    for i, c in enumerate(rec["lift"]):
        through = [
            (cnt, empties, w)
            for cnt, empties, w in active_windows(b, claimant, (c,))
            if c in empties
        ]
        reasons = candidate_reasons(b, claimant, c)
        print(f"  lift[{i}]={c} pre-candidate reasons={[(x, show_window(w)) for x,w in reasons]}")
        counts = {}
        for cnt, _, _ in through:
            counts[cnt] = counts.get(cnt, 0) + 1
        print(f"    claimant alive-window incidence before placement={counts}")
        b[c] = claimant
    threats = [(cnt, empties, w) for cnt, empties, w in active_windows(b, claimant) if cnt >= 4]
    counters = [(cnt, empties, w) for cnt, empties, w in active_windows(b, 1 - claimant) if cnt >= 4]
    print(f"  post-lift claimant threats={len(threats)} tau={tau([e for _,e,_ in threats])}")
    for cnt, empties, w in threats:
        print(f"    c{cnt} E={empties} W={show_window(w)}")
    print(f"  post-lift defender win-now windows={len(counters)}")
    for cnt, empties, w in counters:
        print(f"    c{cnt} E={empties} W={show_window(w)}")
    root_b = board(moves)
    if phase == "FirstStone":
        firsts, current, q1, forcing_firsts, added = tempo_seed_pair_children(root_b, claimant)
        lift_pair = tuple(sorted(rec["lift"]))
        print(
            f"  root branching firsts={len(firsts)} current_forcing_pairs={len(current)} "
            f"q1_only={len(q1)} forcing_firsts={len(forcing_firsts)} added_seed_pairs={len(added)} "
            f"lift_in_current={lift_pair in current} lift_in_added={lift_pair in added}"
        )
        hist = {}
        for c in q1:
            hist[q1_degree(root_b, claimant, c)] = hist.get(q1_degree(root_b, claimant, c), 0) + 1
        print(f"    q1-degree histogram={dict(sorted(hist.items()))}")
        for threshold in (2, 4, 6, 8, 10):
            strong = {c for c in q1 if q1_degree(root_b, claimant, c) >= threshold}
            strong_added = {pair for pair in added if any(c in strong for c in pair)}
            print(f"    degree>={threshold}: cells={len(strong)} added_pairs={len(strong_added)}")
        axis_hist = {}
        for c in q1:
            signature = q1_axis_support(root_b, claimant, c)
            axis_hist[signature] = axis_hist.get(signature, 0) + 1
        print(f"    q1-axis-support histogram={dict(sorted(axis_hist.items()))}")
        print(f"    lift seed axis support={q1_axis_support(root_b, claimant, rec['lift'][-1])}")
        for minimum in (1, 2, 3, 4, 5):
            junctions = {
                c for c in q1
                if len(q1_axis_support(root_b, claimant, c)) >= 2
                and q1_axis_support(root_b, claimant, c)[1] >= minimum
            }
            junction_added = {pair for pair in added if any(c in junctions for c in pair)}
            print(f"    two-axis min>={minimum}: cells={len(junctions)} added_pairs={len(junction_added)}")
        for minimum in (1, 2, 3, 4, 5):
            exact_added = set()
            per_first = []
            for first in forcing_firsts:
                j2 = j2_after_first(root_b, claimant, first, minimum)
                per_first.append((first, len(j2)))
                exact_added.update(tuple(sorted((first, c))) for c in j2)
            lift_first, lift_second = rec["lift"]
            print(
                f"    exact J2 min>={minimum}: per_first={per_first} unique_pairs={len(exact_added)} "
                f"lift_support_after_first={q1_axis_support({**root_b, lift_first: claimant}, claimant, lift_second)} "
                f"lift_in={tuple(sorted((lift_first,lift_second))) in exact_added}"
            )
    else:
        current = normal_candidates(root_b, claimant)
        q1 = q1_candidates(root_b, claimant)
        added = set()
        for second in q1:
            b2 = dict(root_b)
            b2[second] = claimant
            if forcing_after(b2, claimant)[0]:
                added.add(second)
        print(
            f"  root branching current_seconds={len(current)} q1_only={len(q1)} "
            f"added_seed_seconds={len(added)} lift_in_current={rec['lift'][0] in current} "
            f"lift_in_added={rec['lift'][0] in added}"
        )
        hist = {}
        for c in q1:
            hist[q1_degree(root_b, claimant, c)] = hist.get(q1_degree(root_b, claimant, c), 0) + 1
        print(f"    q1-degree histogram={dict(sorted(hist.items()))}")
        for threshold in (2, 4, 6, 8, 10):
            strong = {c for c in added if q1_degree(root_b, claimant, c) >= threshold}
            print(f"    degree>={threshold}: added_seconds={len(strong)}")
        axis_hist = {}
        for c in q1:
            signature = q1_axis_support(root_b, claimant, c)
            axis_hist[signature] = axis_hist.get(signature, 0) + 1
        print(f"    q1-axis-support histogram={dict(sorted(axis_hist.items()))}")
        print(f"    lift seed axis support={q1_axis_support(root_b, claimant, rec['lift'][-1])}")
        for minimum in (1, 2, 3, 4, 5):
            junctions = {
                c for c in added
                if len(q1_axis_support(root_b, claimant, c)) >= 2
                and q1_axis_support(root_b, claimant, c)[1] >= minimum
            }
            print(f"    two-axis min>={minimum}: added_seconds={len(junctions)}")
