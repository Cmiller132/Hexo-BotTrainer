#!/usr/bin/env python3
"""Exhaustive period-lattice search and verifier for 7-in-a-row pairings.

The searched quotient is Z^2 / L for

    L = <(2, 2), (0, 6)>.

Every axis step has order six in this 12-cell quotient.  At the density
threshold, a periodic pairing must select one unit edge (one phase) on each
of the two quotient cycles for each of the three axes.  The resulting six
phase choices form a small exact-cover instance.

This script uses only the Python standard library.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product


if not __debug__:
    raise RuntimeError("this verifier requires assertions; rerun without -O")


PERIOD = 6
AXES = (
    ("H", (1, 0)),
    ("V", (0, 1)),
    ("D", (1, -1)),
)

# Include every L-translate of these directed unit edges.
BASE_STARTS = {
    "H": ((0, 0), (0, 1)),
    "V": ((0, 3), (1, 3)),
    "D": ((1, 2), (1, 5)),
}


def quotient(cell: tuple[int, int]) -> tuple[int, int]:
    """Return the representative in F={0,1} x {0,...,5} modulo L."""

    q, r = cell
    multiple = q // 2
    return q - 2 * multiple, (r - 2 * multiple) % 6


FUNDAMENTAL_CELLS = tuple((q, r) for r in range(6) for q in range(2))
CELL_INDEX = {cell: index for index, cell in enumerate(FUNDAMENTAL_CELLS)}


def add(cell: tuple[int, int], step: tuple[int, int], n: int = 1) -> tuple[int, int]:
    return cell[0] + n * step[0], cell[1] + n * step[1]


def axis_cycles() -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    """List the two length-six quotient cycles for each axis."""

    result: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for name, step in AXES:
        unseen = set(FUNDAMENTAL_CELLS)
        while unseen:
            start = min(unseen, key=CELL_INDEX.__getitem__)
            cycle: list[tuple[int, int]] = []
            cell = start
            while cell not in cycle:
                cycle.append(cell)
                unseen.remove(cell)
                cell = quotient(add(cell, step))
            assert cell == start
            assert len(cycle) == PERIOD
            result.append((name, tuple(cycle)))
    assert len(result) == 6
    return tuple(result)


CYCLES = axis_cycles()


@dataclass(frozen=True)
class Option:
    group: int
    phase: int
    axis: str
    start: tuple[int, int]
    end: tuple[int, int]
    mask: int


def make_options() -> tuple[Option, ...]:
    """Make 36 exact-cover rows: six phases on each of six line cycles."""

    options: list[Option] = []
    vertex_columns = len(FUNDAMENTAL_CELLS)
    for group, (axis, cycle) in enumerate(CYCLES):
        for phase in range(PERIOD):
            start = cycle[phase]
            end = cycle[(phase + 1) % PERIOD]
            columns = (
                CELL_INDEX[start],
                CELL_INDEX[end],
                vertex_columns + group,
            )
            mask = sum(1 << column for column in columns)
            options.append(Option(group, phase, axis, start, end, mask))
    assert len(options) == 36
    return tuple(options)


OPTIONS = make_options()
CONSTRAINT_COUNT = len(FUNDAMENTAL_CELLS) + len(CYCLES)
ALL_CONSTRAINTS = (1 << CONSTRAINT_COUNT) - 1


def witness_option_ids() -> frozenset[int]:
    selected_starts = {
        (axis, quotient(start))
        for axis, starts in BASE_STARTS.items()
        for start in starts
    }
    ids = frozenset(
        index
        for index, option in enumerate(OPTIONS)
        if (option.axis, option.start) in selected_starts
    )
    assert len(ids) == 6
    assert {OPTIONS[index].group for index in ids} == set(range(6))
    return ids


WITNESS_OPTION_IDS = witness_option_ids()


def algorithm_x_count() -> tuple[int, int, bool]:
    """Enumerate every exact cover using MRV Algorithm X.

    Returns (solution count, recursive-state count, witness found).
    """

    constraint_options: list[list[int]] = [[] for _ in range(CONSTRAINT_COUNT)]
    for option_id, option in enumerate(OPTIONS):
        mask = option.mask
        while mask:
            bit = mask & -mask
            constraint_options[bit.bit_length() - 1].append(option_id)
            mask -= bit

    solutions = 0
    states = 0
    found_witness = False
    chosen: list[int] = []

    def visit(covered: int) -> None:
        nonlocal solutions, states, found_witness
        states += 1
        if covered == ALL_CONSTRAINTS:
            solutions += 1
            if frozenset(chosen) == WITNESS_OPTION_IDS:
                found_witness = True
            return

        uncovered = ALL_CONSTRAINTS ^ covered
        best: list[int] | None = None
        while uncovered:
            bit = uncovered & -uncovered
            column = bit.bit_length() - 1
            uncovered -= bit
            candidates = [
                option_id
                for option_id in constraint_options[column]
                if OPTIONS[option_id].mask & covered == 0
            ]
            if not candidates:
                return
            if best is None or len(candidates) < len(best):
                best = candidates

        assert best is not None
        for option_id in best:
            chosen.append(option_id)
            visit(covered | OPTIONS[option_id].mask)
            chosen.pop()

    visit(0)
    return solutions, states, found_witness


def direct_phase_count() -> tuple[int, bool]:
    """Independently enumerate the raw 6^6 phase assignments."""

    solutions = 0
    found_witness = False
    witness_phases = tuple(
        next(
            OPTIONS[option_id].phase
            for option_id in WITNESS_OPTION_IDS
            if OPTIONS[option_id].group == group
        )
        for group in range(len(CYCLES))
    )
    for phases in product(range(PERIOD), repeat=len(CYCLES)):
        incidence = [0] * len(FUNDAMENTAL_CELLS)
        for group, phase in enumerate(phases):
            cycle = CYCLES[group][1]
            incidence[CELL_INDEX[cycle[phase]]] += 1
            incidence[CELL_INDEX[cycle[(phase + 1) % PERIOD]]] += 1
        if all(value == 1 for value in incidence):
            solutions += 1
            if phases == witness_phases:
                found_witness = True
    return solutions, found_witness


def selected_start(axis: str, cell: tuple[int, int]) -> bool:
    residue = quotient(cell)
    return any(residue == quotient(start) for start in BASE_STARTS[axis])


def verify_quotient() -> tuple[Counter[int], Counter[int]]:
    """Check every vertex orbit and every start/axis window orbit."""

    incidence: Counter[tuple[int, int]] = Counter()
    for option_id in WITNESS_OPTION_IDS:
        option = OPTIONS[option_id]
        incidence[option.start] += 1
        incidence[option.end] += 1
    vertex_histogram = Counter(incidence[cell] for cell in FUNDAMENTAL_CELLS)

    window_histogram: Counter[int] = Counter()
    for start in FUNDAMENTAL_CELLS:
        for axis, step in AXES:
            contained = sum(
                selected_start(axis, add(start, step, offset))
                for offset in range(6)
            )
            window_histogram[contained] += 1
    return vertex_histogram, window_histogram


def verify_patch(radius: int = 30) -> tuple[Counter[int], Counter[int]]:
    """Directly check cells and windows whose starts lie in a square patch."""

    cell_histogram: Counter[int] = Counter()
    window_histogram: Counter[int] = Counter()
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            cell = (q, r)
            incidence = 0
            for axis, step in AXES:
                incidence += selected_start(axis, cell)
                incidence += selected_start(axis, add(cell, step, -1))
            cell_histogram[incidence] += 1

            for axis, step in AXES:
                contained = sum(
                    selected_start(axis, add(cell, step, offset))
                    for offset in range(6)
                )
                window_histogram[contained] += 1
    return cell_histogram, window_histogram


def main() -> None:
    solutions, states, found_witness = algorithm_x_count()
    direct_solutions, direct_found_witness = direct_phase_count()
    vertex_histogram, window_histogram = verify_quotient()
    patch_cells, patch_windows = verify_patch()

    assert solutions == 120
    assert states == 419
    assert direct_solutions == solutions
    assert found_witness
    assert direct_found_witness
    assert vertex_histogram == Counter({1: 12})
    assert window_histogram == Counter({1: 36})
    assert patch_cells == Counter({1: 61 * 61})
    assert patch_windows == Counter({1: 3 * 61 * 61})

    print("period_lattice=< (2,2), (0,6) >")
    print("quotient_cells=12 line_cycles=6 options=36 constraints=18")
    print(f"raw_phase_assignments={PERIOD ** len(CYCLES)}")
    print(f"algorithm_x_solutions={solutions} recursive_states={states}")
    print(
        f"direct_phase_solutions={direct_solutions} "
        f"witness_found={direct_found_witness}"
    )
    print(f"quotient_vertex_histogram={dict(vertex_histogram)}")
    print(f"quotient_window_histogram={dict(window_histogram)}")
    print(f"patch_vertex_histogram={dict(patch_cells)}")
    print(f"patch_window_histogram={dict(patch_windows)}")


if __name__ == "__main__":
    main()
