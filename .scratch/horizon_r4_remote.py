#!/usr/bin/env python3
"""Finite evidence for Horizon R4 remote/excursion geometry.

This file deliberately separates three statements which are easy to conflate:

* ``static``: the hitting number of the residual one/two-cell threat family
  after an attacker set X has already been placed;
* ``activation``: whether an unoccupied pivot can turn a visible precursor
  into a static family of hitting number > 2;
* ``dynamic``: whether the attacker can force that precursor against all of
  the defender's earlier replies.  This program does *not* claim the third.

Only Python's standard library is used.  Coordinates are axial, with line
steps (1,0), (0,1), and (1,-1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

Cell = tuple[int, int]
Edge = frozenset[Cell]

AXES: tuple[Cell, ...] = ((1, 0), (0, 1), (1, -1))


def add(a: Cell, b: Cell) -> Cell:
    return a[0] + b[0], a[1] + b[1]


def mul(n: int, a: Cell) -> Cell:
    return n * a[0], n * a[1]


def translate(cells: Iterable[Cell], delta: Cell) -> frozenset[Cell]:
    return frozenset(add(c, delta) for c in cells)


def normalized_translation(cells: Iterable[Cell]) -> tuple[Cell, ...]:
    cells = tuple(cells)
    anchor = min(cells)
    return tuple(sorted((q - anchor[0], r - anchor[1]) for q, r in cells))


def line_key_and_coordinate(axis: int, c: Cell) -> tuple[int, int]:
    q, r = c
    if axis == 0:  # step (1,0)
        return r, q
    if axis == 1:  # step (0,1)
        return q, r
    return q + r, q  # step (1,-1)


def line_cell(axis: int, key: int, t: int) -> Cell:
    if axis == 0:
        return t, key
    if axis == 1:
        return key, t
    return t, key - t


def windows_with_at_least_four(
    attackers: frozenset[Cell], defenders: frozenset[Cell] = frozenset()
) -> tuple[tuple[Edge, ...], bool, tuple[tuple[int, int, int], ...]]:
    """Return distinct residual edges, terminal flag, and supporting windows.

    A defender-occupied window is dead.  A residual edge has size one or two.
    Supporting-window triples are ``(axis, line_key, start_coordinate)``.
    """
    by_line: dict[tuple[int, int], set[int]] = defaultdict(set)
    for c in attackers:
        for axis in range(3):
            key, t = line_key_and_coordinate(axis, c)
            by_line[axis, key].add(t)

    family: set[Edge] = set()
    supports: set[tuple[int, int, int]] = set()
    terminal = False
    for (axis, key), coords in by_line.items():
        starts: set[int] = set()
        for t in coords:
            starts.update(range(t - 5, t + 1))
        for start in starts:
            window = frozenset(line_cell(axis, key, t) for t in range(start, start + 6))
            if window & defenders:
                continue
            occupied = len(window & attackers)
            if occupied == 6:
                terminal = True
                supports.add((axis, key, start))
            elif occupied >= 4:
                family.add(window - attackers)
                supports.add((axis, key, start))
    return tuple(sorted(family, key=lambda e: (len(e), sorted(e)))), terminal, tuple(sorted(supports))


def minimum_cover(family: Iterable[Edge]) -> tuple[int, tuple[Cell, ...]]:
    """Exact minimum hitting set for a singleton/two-edge hypergraph."""
    edges = frozenset(frozenset(e) for e in family if e)
    forced = frozenset(next(iter(e)) for e in edges if len(e) == 1)
    edges = frozenset(e for e in edges if not e & forced)

    memo: dict[frozenset[Edge], tuple[int, tuple[Cell, ...]]] = {}

    def solve(rest: frozenset[Edge]) -> tuple[int, tuple[Cell, ...]]:
        if not rest:
            return 0, ()
        if rest in memo:
            return memo[rest]
        edge = min(rest, key=lambda e: (len(e), sorted(e)))
        best: tuple[int, tuple[Cell, ...]] | None = None
        for c in sorted(edge):
            sub = frozenset(e for e in rest if c not in e)
            n, witness = solve(sub)
            candidate = 1 + n, tuple(sorted((c,) + witness))
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        memo[rest] = best
        return best

    n, witness = solve(edges)
    full = tuple(sorted(set(witness) | set(forced)))
    return len(full), full


def analyze_set(attackers: frozenset[Cell]) -> dict:
    family, terminal, supports = windows_with_at_least_four(attackers)
    tau, cover = minimum_cover(family)
    lines = sorted(set((axis, key) for axis, key, _ in supports))
    result = {
        "stones": [list(c) for c in sorted(attackers)],
        "stone_count": len(attackers),
        "terminal": terminal,
        "supporting_line_count": len(lines),
        "supporting_window_count": len(supports),
        "residual_family": [[list(c) for c in sorted(e)] for e in family],
        "cover_number": tau,
        "cover_witness": [list(c) for c in cover],
    }
    if attackers:
        result["radius8_self_chain"] = radius_chain_certificate(attackers, 8)
    return result


def hex_distance(a: Cell, b: Cell) -> int:
    dq, dr = a[0] - b[0], a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def radius_chain_certificate(cells: frozenset[Cell], radius: int) -> dict:
    """Give a deterministic order connected by hops of at most ``radius``.

    The first cell is a seed whose legality must come from the interaction.
    Every later cell is then legal from an already placed attacker stone.
    """
    seed = (0, 0) if (0, 0) in cells else min(cells)
    placed = [seed]
    remaining = set(cells - {seed})
    hops: list[int] = []
    while remaining:
        choices = []
        for c in remaining:
            distance = min(hex_distance(c, old) for old in placed)
            choices.append((distance, c))
        distance, chosen = min(choices)
        if distance > radius:
            return {
                "conditional_on_seed_legal": True,
                "chainable": False,
                "stuck_distance": distance,
                "partial_order": [list(c) for c in placed],
            }
        placed.append(chosen)
        remaining.remove(chosen)
        hops.append(distance)
    return {
        "conditional_on_seed_legal": True,
        "chainable": True,
        "max_hop": max(hops, default=0),
        "order": [list(c) for c in placed],
    }


def cross_seven() -> frozenset[Cell]:
    # Two consecutive-four lines meet in their already occupied pivot.
    return frozenset(
        {(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)}
    )


def six_line_weave_twelve() -> frozenset[Cell]:
    # Active lines q=0,2; r=0,2; q+r=1,3.  Each has four consecutive
    # intersections and every pairwise intersection lies in X, so the six
    # consecutive-four residual families have disjoint support.
    return frozenset(
        {
            (0, 0), (0, 1), (0, 2), (0, 3),
            (2, -1), (2, 0), (2, 1), (2, 2),
            (1, 0), (3, 0), (-1, 2), (1, 2),
        }
    )


def triangle_nine() -> frozenset[Cell]:
    # Three consecutive-four lines r=0, q=0, and q+r=3.  Their three
    # pairwise intersections are occupied and distinct.
    return frozenset(
        {
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (0, 2), (0, 3),
            (1, 2), (2, 1),
        }
    )


def three_axis_star_ten() -> frozenset[Cell]:
    return frozenset(
        {(0, 0)}
        | {(t, 0) for t in range(1, 4)}
        | {(0, t) for t in range(1, 4)}
        | {(t, -t) for t in range(1, 4)}
    )


def disjoint_runs(count: int) -> frozenset[Cell]:
    return frozenset((10 * block + offset, 0) for block in range(count) for offset in range(4))


def one_axis_band_census() -> dict:
    """Exhaust the R3 [-5,10] carrier through twelve stones.

    This is an exact bounded-band census, not a classification of arbitrary
    far-separated one-axis components.  Requiring >=4 stones in [0,5] pins a
    base supporting window, as in R3.
    """
    carrier = tuple(range(-5, 11))
    base = frozenset(range(6))
    seen: set[tuple[int, ...]] = set()
    per_k: dict[int, Counter[int]] = defaultdict(Counter)
    max_examples: dict[int, dict] = {}
    terminal_skipped = Counter()
    for k in range(4, 13):
        for raw in combinations(carrier, k):
            S = frozenset(raw)
            if len(S & base) < 4:
                continue
            lo, hi = min(S), max(S)
            direct = tuple(x - lo for x in sorted(S))
            reflected = tuple(hi - x for x in sorted(S, reverse=True))
            key = min(direct, reflected)
            tagged = (k,) + key
            if tagged in seen:
                continue
            seen.add(tagged)
            X = frozenset((x, 0) for x in S)
            data = analyze_set(X)
            if data["terminal"]:
                terminal_skipped[k] += 1
                continue
            tau = data["cover_number"]
            per_k[k][tau] += 1
            old = max_examples.get(k)
            if old is None or tau > old["cover_number"]:
                max_examples[k] = data
    return {
        "scope": "exact for normalized subsets of [-5,10] meeting base [0,5] in >=4; not arbitrary separated components",
        "canonical_shapes_including_terminal": len(seen),
        "r3_prefix_k4_through_k6_including_terminal": sum(
            sum(per_k[k].values()) + terminal_skipped[k] for k in range(4, 7)
        ),
        "all_nonterminal_k4_through_k6_have_two_cover": all(
            tau <= 2 for k in range(4, 7) for tau in per_k[k]
        ),
        "canonical_nonterminal_by_k_and_cover": {
            str(k): dict(sorted(counter.items())) for k, counter in sorted(per_k.items())
        },
        "terminal_shapes_skipped": dict(sorted(terminal_skipped.items())),
        "max_examples": {str(k): value for k, value in sorted(max_examples.items())},
    }


def exact_seven_stone_two_axis_census() -> dict:
    """Exhaust every seven-stone set supported by two nonparallel windows.

    If seven stones support two distinct lines with >=4 stones each, the
    lines cannot be parallel and must share exactly one attacker stone.  Put
    their intersection at the origin, enumerate its positions 0..5 in both
    windows, and choose the other three attackers in each window.  Therefore
    this census is exhaustive up to axis symmetry and translation for the
    first multi-axis cardinality.
    """
    d0, d1 = AXES[0], AXES[1]
    seen: set[tuple[Cell, ...]] = set()
    hist = Counter()
    examples: dict[int, dict] = {}
    for pivot_i in range(6):
        w0 = tuple(mul(t - pivot_i, d0) for t in range(6))
        for pivot_j in range(6):
            w1 = tuple(mul(t - pivot_j, d1) for t in range(6))
            for rest0 in combinations([c for c in w0 if c != (0, 0)], 3):
                for rest1 in combinations([c for c in w1 if c != (0, 0)], 3):
                    X = frozenset(((0, 0),) + rest0 + rest1)
                    key = normalized_translation(X)
                    if key in seen:
                        continue
                    seen.add(key)
                    data = analyze_set(X)
                    assert not data["terminal"]
                    tau = data["cover_number"]
                    hist[tau] += 1
                    examples.setdefault(tau, data)
    return {
        "scope": "exhaustive up to translation, with one representative ordered axis pair fixed WLOG; reflections are retained, so the 1,600 count is not a full D6 quotient",
        "canonical_sets": len(seen),
        "cover_histogram": dict(sorted(hist.items())),
        "minimum_failure_k": 7,
        "failure_count": sum(n for tau, n in hist.items() if tau > 2),
        "examples": {str(k): value for k, value in sorted(examples.items())},
    }


def consecutive_cross_templates(pivot: Cell) -> tuple[frozenset[Cell], ...]:
    """Six-stone precursors: two 3-of-consecutive-4 arms around an empty pivot."""
    result: set[frozenset[Cell]] = set()
    for a0, a1 in combinations(range(3), 2):
        for pivot_index0 in range(4):
            offsets0 = [i - pivot_index0 for i in range(4) if i != pivot_index0]
            arm0 = {add(pivot, mul(t, AXES[a0])) for t in offsets0}
            for pivot_index1 in range(4):
                offsets1 = [i - pivot_index1 for i in range(4) if i != pivot_index1]
                arm1 = {add(pivot, mul(t, AXES[a1])) for t in offsets1}
                result.add(frozenset(arm0 | arm1))
    return tuple(sorted(result, key=lambda s: sorted(s)))


def dangerous_pivots_from_templates(
    X: frozenset[Cell], pivots: Iterable[Cell] | None = None
) -> tuple[Cell, ...]:
    if pivots is None:
        candidates: set[Cell] = set()
        for stone in X:
            for axis in AXES:
                for offset in range(-3, 4):
                    if offset:
                        candidates.add(add(stone, mul(offset, axis)))
        pivots = candidates - X
    dangerous = []
    for pivot in pivots:
        if pivot in X:
            continue
        ready_axes = 0
        for axis in AXES:
            ready = False
            for pivot_index in range(4):
                offsets = [i - pivot_index for i in range(4) if i != pivot_index]
                if all(add(pivot, mul(t, axis)) in X for t in offsets):
                    ready = True
                    break
            ready_axes += int(ready)
        if ready_axes < 2:
            continue
        family, terminal, _ = windows_with_at_least_four(X | {pivot})
        tau, _ = minimum_cover(family)
        if terminal or tau > 2:
            dangerous.append(pivot)
    return tuple(sorted(dangerous))


def defender_precover_number(X: frozenset[Cell], pivots: tuple[Cell, ...]) -> tuple[int, tuple[Cell, ...]]:
    """Minimum pre-block set neutralizing every listed one-cell activation.

    A candidate defender cell is either a pivot or belongs to a residual edge
    after a pivot activation.  A pivot remains dangerous after B when it is
    empty and the B-filtered residual family still has hitting number > 2 (or
    A immediately completes a six).  Search is exact over this dominance-
    complete candidate carrier.
    """
    candidate_cells: set[Cell] = set(pivots)
    for pivot in pivots:
        family, _, _ = windows_with_at_least_four(X | {pivot})
        candidate_cells.update(set().union(*family) if family else set())

    def safe(B: frozenset[Cell]) -> bool:
        for pivot in pivots:
            if pivot in B:
                continue
            family, terminal, _ = windows_with_at_least_four(X | {pivot}, B)
            tau, _ = minimum_cover(family)
            if terminal or tau > 2:
                return False
        return True

    ordered = tuple(sorted(candidate_cells))
    for size in range(0, len(ordered) + 1):
        for raw in combinations(ordered, size):
            B = frozenset(raw)
            if safe(B):
                return size, tuple(sorted(B))
    raise AssertionError("finite carrier must have a cover")


def triangle_six_pair_activation() -> dict:
    """Exact local h18-tail obstruction to a reserve-pair remote defense.

    X has three stones on each of three nonconcurrent axial lines and no
    current >=4-stone window.  D pre-covers with B, A activates with a pair,
    D gets one ordinary cover pair, and A has a final pair.  The computed
    pre-cover number is the least |B| making every A activation leave a
    two-coverable residual family.

    This is a local tail theorem only.  It does not say A can build X against
    D's earlier replies or that D lacks a win in the anchored interaction.
    """
    X = frozenset({(-2, 0), (-1, 2), (0, -1), (0, 0), (0, 1), (1, 0)})

    # A pair can raise a window to four stones only if it already contains at
    # least two members of X.  This gives an exact finite action carrier.
    action_cells: set[Cell] = set()
    for axis in range(3):
        lines: dict[int, set[int]] = defaultdict(set)
        for c in X:
            key, t = line_key_and_coordinate(axis, c)
            lines[key].add(t)
        for key, coords in lines.items():
            starts = {start for t in coords for start in range(t - 5, t + 1)}
            for start in starts:
                window = {line_cell(axis, key, t) for t in range(start, start + 6)}
                if len(window & X) >= 2:
                    action_cells.update(window - X)

    dangerous: list[tuple[tuple[Cell, Cell], int, bool, tuple[Edge, ...]]] = []
    cover_histogram = Counter()
    precover_cells: set[Cell] = set()
    for pair in combinations(sorted(action_cells), 2):
        family, terminal, _ = windows_with_at_least_four(X | set(pair))
        tau, _ = minimum_cover(family)
        if terminal or tau > 2:
            dangerous.append((pair, tau, terminal, family))
            cover_histogram[tau] += 1
            precover_cells.update(pair)
            if family:
                precover_cells.update(set().union(*family))

    def neutralizes(B: frozenset[Cell]) -> bool:
        for pair, _, _, _ in dangerous:
            if B & set(pair):
                continue
            family, terminal, _ = windows_with_at_least_four(X | set(pair), B)
            tau, _ = minimum_cover(family)
            if terminal or tau > 2:
                return False
        return True

    ordered_precover = tuple(sorted(precover_cells))
    precover_number = -1
    precover_witness: tuple[Cell, ...] = ()
    for size in range(len(ordered_precover) + 1):
        for raw in combinations(ordered_precover, size):
            if neutralizes(frozenset(raw)):
                precover_number = size
                precover_witness = raw
                break
        if precover_number >= 0:
            break
    assert precover_number == 3

    current = analyze_set(X)
    assert not current["terminal"] and current["cover_number"] == 0
    return {
        "scope": "exact local D-pair/A-pair/D-pair/A-pair tail; excludes earlier defender occupancy, defender wins, and forced reachability",
        "fresh_clock_relevance": "after six remote A stones, this is the placement-11..18 tail of fresh h18",
        "attacker_precursor": current,
        "attacker_action_carrier_size": len(action_cells),
        "dangerous_attacker_pair_count": len(dangerous),
        "dangerous_pair_cover_histogram_without_preblock": dict(sorted(cover_histogram.items())),
        "dangerous_attacker_pairs": [
            [list(pair[0]), list(pair[1])] for pair, _, _, _ in dangerous
        ],
        "dominance_complete_precover_carrier_size": len(precover_cells),
        "defender_precover_number": precover_number,
        "defender_precover_witness": [list(c) for c in precover_witness],
        "reserve_pair_fails": precover_number > 2,
        "dynamic_disclaimer": "the two earlier defender pairs may already pay this three-stone local tax; coupling that tax to the anchored interaction is the unresolved reachability theorem",
    }


def bounded_latent_pivot_census(relative_radius: int = 1) -> dict:
    """Union pairs of consecutive-cross precursors at nearby pivots.

    This is a deterministic bounded census.  One pivot is fixed at the
    origin; the other ranges over the axial hex ball of ``relative_radius``.
    It counts only activation pivots satisfying the consecutive-cross
    predicate used by ``dangerous_pivots_from_templates``: the pivot completes
    a consecutive four on at least two axes.  It does not enumerate every
    one-cell activation of the union state.  Pre-cover numbers are exact for
    the counted pivots, not for omitted non-cross activations.
    """
    origin = (0, 0)
    base_templates = consecutive_cross_templates(origin)
    deltas = []
    for q in range(-relative_radius, relative_radius + 1):
        for r in range(-relative_radius, relative_radius + 1):
            if max(abs(q), abs(r), abs(q + r)) <= relative_radius and (q, r) != origin:
                deltas.append((q, r))

    seen: set[tuple[Cell, ...]] = set()
    hist_by_k: dict[int, Counter[int]] = defaultdict(Counter)
    best: dict[str, object] | None = None
    first_unprecoverable: dict[str, object] | None = None
    max_pivots = 0

    def example_record(
        X: frozenset[Cell], pivots: tuple[Cell, ...], pre_n: int,
        pre_witness: tuple[Cell, ...]
    ) -> dict[str, object]:
        current_family, current_terminal, _ = windows_with_at_least_four(X)
        current_tau, _ = minimum_cover(current_family)
        activations = []
        for pivot in pivots:
            family, terminal, _ = windows_with_at_least_four(X | {pivot})
            tau, _ = minimum_cover(family)
            activations.append({
                "pivot": list(pivot),
                "terminal": terminal,
                "cover_number_without_preblock": tau,
            })
        return {
            "attackers": [list(c) for c in sorted(X)],
            "attacker_count": len(X),
            "current_terminal": current_terminal,
            "current_cover_number": current_tau,
            "counted_consecutive_cross_pivots": [list(c) for c in pivots],
            "counted_consecutive_cross_pivot_count": len(pivots),
            "counted_pivot_activations": activations,
            "defender_precover_number_for_counted_pivots": pre_n,
            "defender_precover_witness_for_counted_pivots": [list(c) for c in pre_witness],
        }

    for first in base_templates:
        for delta in deltas:
            for relative in consecutive_cross_templates(origin):
                second = translate(relative, delta)
                X = first | second
                if len(X) > 12:
                    continue
                key = normalized_translation(X)
                if key in seen:
                    continue
                seen.add(key)
                pivots = dangerous_pivots_from_templates(X)
                hist_by_k[len(X)][len(pivots)] += 1
                if len(pivots) > max_pivots:
                    max_pivots = len(pivots)
                    pre_n, pre_witness = defender_precover_number(X, pivots)
                    best = example_record(X, pivots, pre_n, pre_witness)
                if len(pivots) >= 3:
                    pre_n, pre_witness = defender_precover_number(X, pivots)
                    candidate = example_record(X, pivots, pre_n, pre_witness)
                    if pre_n > 2 and (
                        first_unprecoverable is None
                        or (len(X), -len(pivots), sorted(X))
                        < (
                            first_unprecoverable["attacker_count"],
                            -first_unprecoverable["counted_consecutive_cross_pivot_count"],
                            [tuple(c) for c in first_unprecoverable["attackers"]],
                        )
                    ):
                        first_unprecoverable = candidate
    return {
        "scope": f"unions of two consecutive-cross six-stone precursors; relative precursor centers in hex radius {relative_radius}",
        "quotient": "translation only; rotations and reflections are retained and may be counted separately",
        "pivot_predicate": "counted pivots complete a consecutive four on at least two axes and have terminal or cover number >2 after activation",
        "enumeration_limit": "not an exhaustive census of all dangerous one-cell activations of each union state; maxima and pre-cover numbers quantify only the counted consecutive-cross pivots",
        "dynamic_disclaimer": "state-level restricted activation census only; does not prove attacker can force the precursor through earlier defender turns",
        "canonical_unions": len(seen),
        "histogram_by_attacker_count_and_counted_consecutive_cross_pivots": {
            str(k): dict(sorted(counter.items())) for k, counter in sorted(hist_by_k.items())
        },
        "maximum_counted_consecutive_cross_pivots": max_pivots,
        "maximum_counted_consecutive_cross_pivots_at_attacker_count_le_10": max(
            pivot_count
            for k, counter in hist_by_k.items() if k <= 10
            for pivot_count in counter
        ),
        "example_maximizing_counted_consecutive_cross_pivots": best,
        "minimum_unprecoverable_example_for_counted_pivots": first_unprecoverable,
    }


def make_report(relative_radius: int) -> dict:
    cross = analyze_set(cross_seven())
    weave = analyze_set(six_line_weave_twelve())
    disjoint = {str(4 * n): analyze_set(disjoint_runs(n)) for n in range(1, 4)}
    return {
        "schema": 2,
        "claim_labels": {
            "static": "exact finite geometry for the displayed/censused attacker set",
            "activation": "exact at the displayed precursor state, before quantifying earlier defender replies",
            "dynamic": "not established by this artifact",
        },
        "static_named_obstructions": {
            "cross7": cross,
            "two_radius8_chained_runs8": disjoint["8"],
            "three_line_triangle9": analyze_set(triangle_nine()),
            "three_axis_star10": analyze_set(three_axis_star_ten()),
            "six_line_weave12": weave,
            "disjoint_consecutive_four_runs": disjoint,
        },
        "exact_seven_stone_two_axis_census": exact_seven_stone_two_axis_census(),
        "one_axis_band_census": one_axis_band_census(),
        "triangle_six_pair_activation": triangle_six_pair_activation(),
        "bounded_latent_pivot_census": bounded_latent_pivot_census(relative_radius),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(".scratch/horizon_r4_phase2.json"))
    parser.add_argument("--relative-radius", type=int, default=1)
    args = parser.parse_args()
    report = make_report(args.relative_radius)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.out.write_text(encoded, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({
        "out": str(args.out),
        "sha256": digest,
        "cross7_tau": report["static_named_obstructions"]["cross7"]["cover_number"],
        "weave12_tau": report["static_named_obstructions"]["six_line_weave12"]["cover_number"],
        "seven_canonical": report["exact_seven_stone_two_axis_census"]["canonical_sets"],
        "latent_unions": report["bounded_latent_pivot_census"]["canonical_unions"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
