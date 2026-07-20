"""Exact verifier for the ES-global GREEDY-refutation witness.

The checker uses only integer arithmetic in Q(sqrt(3)).  Values printed in
the ``27 Phi`` and ``27 max-danger`` columns are numerators: ``a+b*sqrt(3)``
means the corresponding value is ``(a+b*sqrt(3))/27``.

It verifies both:

* a compact blanket position A={(0,0)}, D={(1,0)}; and
* a material-balanced, explicitly reachable enlargement obtained by adding
  an Attacker-filled radius-2 ball inside a Defender-filled radius-3 ring.

The dead enlargement is separated from every tactical cell.  It is translated
one step so the Defender's opening is the engine-mandated ``(0,0)``; after
undoing that translation, both forms have exactly the same alive windows,
greedy choices, and exhaustive branch table.  Every exact maximum-danger tie
is branched at every Defender stone.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache


if not __debug__:
    raise RuntimeError("this verifier requires assertions; rerun without -O")


Coord = tuple[int, int]
Window = tuple[int, int, int]
Q27 = tuple[int, int]  # (a,b) denotes (a+b*sqrt(3))/27

AXES: tuple[Coord, ...] = ((1, 0), (0, 1), (1, -1))
ATTACKER_MOVES: tuple[Coord, ...] = (
    (2, -4),
    (2, 2),
    (-5, 0),
    (-4, 0),
    (-3, 0),
    (-2, 0),
    (-1, 0),
)
WINNING_WINDOW = frozenset((q, 0) for q in range(-5, 1))
ENGINE_SHIFT: Coord = (-1, 0)

# 27*(sqrt(3)^(-e)), indexed by the number e of empty cells.
WEIGHT_27: dict[int, Q27] = {
    0: (27, 0),
    1: (0, 9),
    2: (9, 0),
    3: (0, 3),
    4: (3, 0),
    5: (0, 1),
}


def hdist(x: Coord, y: Coord) -> int:
    dq, dr = x[0] - y[0], x[1] - y[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def shifted(x: Coord, offset: Coord) -> Coord:
    return x[0] + offset[0], x[1] + offset[1]


def shifted_set(xs, offset: Coord) -> frozenset[Coord]:
    return frozenset(shifted(x, offset) for x in xs)


def ball(c: Coord, radius: int) -> frozenset[Coord]:
    cq, cr = c
    return frozenset(
        (cq + q, cr + r)
        for q in range(-radius, radius + 1)
        for r in range(-radius, radius + 1)
        if max(abs(q), abs(r), abs(q + r)) <= radius
    )


def windows_at(x: Coord):
    q, r = x
    for ai, (dq, dr) in enumerate(AXES):
        for off in range(6):
            yield q - off * dq, r - off * dr, ai


@lru_cache(maxsize=None)
def window_cells(w: Window) -> tuple[Coord, ...]:
    q, r, ai = w
    dq, dr = AXES[ai]
    return tuple((q + i * dq, r + i * dr) for i in range(6))


def qadd(x: Q27, y: Q27) -> Q27:
    return x[0] + y[0], x[1] + y[1]


def qsign(x: Q27) -> int:
    """Return the exact sign of a+b*sqrt(3), for integer a,b."""
    a, b = x
    if a == 0:
        return (b > 0) - (b < 0)
    if b == 0:
        return (a > 0) - (a < 0)
    if (a > 0) == (b > 0):
        return 1 if a > 0 else -1

    # The coefficients have opposite signs.  Compare |a| with |b|sqrt(3)
    # by squaring.  Equality cannot occur because sqrt(3) is irrational.
    delta = a * a - 3 * b * b
    assert delta != 0
    if a > 0:
        return 1 if delta > 0 else -1
    return -1 if delta > 0 else 1


def qcmp(x: Q27, y: Q27) -> int:
    return qsign((x[0] - y[0], x[1] - y[1]))


def qtext(x: Q27) -> str:
    a, b = x
    if b == 0:
        return str(a)
    bpart = "sqrt(3)" if abs(b) == 1 else f"{abs(b)}sqrt(3)"
    if b < 0:
        bpart = "-" + bpart
    if a == 0:
        return bpart
    return f"{a}{'+' if b > 0 else ''}{bpart}"


@dataclass(frozen=True)
class Position:
    A: frozenset[Coord]
    D: frozenset[Coord]

    @property
    def occupied(self) -> frozenset[Coord]:
        return self.A | self.D

    def legal(self, x: Coord) -> bool:
        return x not in self.occupied and any(hdist(x, y) <= 8 for y in self.occupied)

    def place(self, x: Coord, attacker: bool) -> "Position":
        assert x not in self.occupied
        if attacker:
            return Position(self.A | {x}, self.D)
        return Position(self.A, self.D | {x})


@lru_cache(maxsize=None)
def alive(pos: Position) -> tuple[tuple[Window, int, tuple[Coord, ...]], ...]:
    """All Attacker-touched, Defender-free windows, including s=6."""
    seen: set[Window] = set()
    out: list[tuple[Window, int, tuple[Coord, ...]]] = []
    for a in pos.A:
        for w in windows_at(a):
            if w in seen:
                continue
            seen.add(w)
            cells = window_cells(w)
            if any(x in pos.D for x in cells):
                continue
            s = sum(x in pos.A for x in cells)
            empties = tuple(x for x in cells if x not in pos.A)
            out.append((w, s, empties))
    return tuple(sorted(out))


@lru_cache(maxsize=None)
def profile(pos: Position) -> tuple[int, int, int, int, int, int]:
    bins = [0] * 7
    for _w, s, _empties in alive(pos):
        bins[s] += 1
    return tuple(bins[1:7])


def phi_27_from_profile(p: tuple[int, int, int, int, int, int]) -> Q27:
    n1, n2, n3, n4, n5, n6 = p
    return 3 * n2 + 9 * n4 + 27 * n6, n1 + 3 * n3 + 9 * n5


def phi_27(pos: Position) -> Q27:
    return phi_27_from_profile(profile(pos))


@lru_cache(maxsize=None)
def won(pos: Position, attacker: bool) -> bool:
    stones = pos.A if attacker else pos.D
    for x in stones:
        for w in windows_at(x):
            if all(c in stones for c in window_cells(w)):
                return True
    return False


@lru_cache(maxsize=None)
def greedy(pos: Position) -> tuple[tuple[Coord, ...], Q27 | None]:
    danger: dict[Coord, Q27] = defaultdict(lambda: (0, 0))
    for _w, s, empties in alive(pos):
        if s == 6:
            continue
        weight = WEIGHT_27[6 - s]
        for x in empties:
            danger[x] = qadd(danger[x], weight)
    if not danger:
        return (), None

    best: Q27 | None = None
    choices: list[Coord] = []
    for x, value in danger.items():
        comparison = 1 if best is None else qcmp(value, best)
        if comparison > 0:
            best = value
            choices = [x]
        elif comparison == 0:
            choices.append(x)
    assert best is not None
    return tuple(sorted(choices)), best


def danger_27(pos: Position, x: Coord) -> Q27:
    value = (0, 0)
    for _w, s, empties in alive(pos):
        if x in empties:
            value = qadd(value, WEIGHT_27[6 - s])
    return value


def pcounter(items) -> Counter:
    return Counter(items)


# Exact reachable-position profile distributions after every placement.
EXPECTED_PROFILES: dict[str, Counter] = {
    "P0": pcounter({(13, 0, 0, 0, 0, 0): 1}),
    "D0.1": pcounter({(8, 0, 0, 0, 0, 0): 4}),
    "D0.2": pcounter({(3, 0, 0, 0, 0, 0): 4}),
    "A1": pcounter({(21, 0, 0, 0, 0, 0): 4}),
    "A2": pcounter({(39, 0, 0, 0, 0, 0): 4}),
    "D1.1": pcounter({(32, 0, 0, 0, 0, 0): 2, (33, 0, 0, 0, 0, 0): 10}),
    "D1.2": pcounter({(27, 0, 0, 0, 0, 0): 20, (28, 0, 0, 0, 0, 0): 104}),
    "A3": pcounter({(43, 1, 0, 0, 0, 0): 20, (44, 1, 0, 0, 0, 0): 104}),
    "A4": pcounter(
        {
            (49, 4, 1, 0, 0, 0): 1,
            (50, 4, 1, 0, 0, 0): 5,
            (51, 4, 1, 0, 0, 0): 19,
            (52, 4, 1, 0, 0, 0): 99,
        }
    ),
    "D2.1": pcounter(
        {
            (39, 4, 1, 0, 0, 0): 1,
            (40, 4, 1, 0, 0, 0): 5,
            (41, 4, 1, 0, 0, 0): 38,
            (42, 4, 1, 0, 0, 0): 198,
        }
    ),
    "D2.2": pcounter(
        {
            (30, 4, 1, 0, 0, 0): 1,
            (31, 4, 1, 0, 0, 0): 24,
            (32, 4, 1, 0, 0, 0): 99,
        }
    ),
    "A5": pcounter(
        {
            (41, 1, 3, 1, 0, 0): 2,
            (42, 1, 3, 1, 0, 0): 11,
            (43, 1, 3, 1, 0, 0): 24,
            (44, 1, 3, 1, 0, 0): 87,
        }
    ),
    "A6": pcounter(
        {
            (49, 3, 1, 2, 1, 0): 2,
            (50, 3, 1, 2, 1, 0): 11,
            (51, 1, 1, 2, 1, 0): 1,
            (51, 3, 1, 2, 1, 0): 19,
            (52, 1, 1, 2, 1, 0): 7,
            (52, 2, 1, 2, 1, 0): 2,
            (52, 3, 1, 2, 1, 0): 62,
            (53, 1, 1, 2, 1, 0): 10,
            (53, 2, 1, 2, 1, 0): 10,
        }
    ),
    "D3.1": pcounter(
        {
            (48, 2, 0, 0, 1, 0): 2,
            (49, 2, 0, 0, 1, 0): 11,
            (50, 0, 0, 0, 1, 0): 1,
            (50, 2, 0, 0, 1, 0): 19,
            (51, 0, 0, 0, 1, 0): 7,
            (51, 1, 0, 0, 1, 0): 2,
            (51, 2, 0, 0, 1, 0): 62,
            (52, 0, 0, 0, 1, 0): 10,
            (52, 1, 0, 0, 1, 0): 10,
        }
    ),
    "D3.2": pcounter(
        {
            (39, 1, 0, 0, 1, 0): 2,
            (40, 0, 0, 0, 1, 0): 1,
            (40, 1, 0, 0, 1, 0): 11,
            (41, 0, 0, 0, 1, 0): 7,
            (41, 1, 0, 0, 1, 0): 19,
            (42, 0, 0, 0, 1, 0): 12,
            (42, 1, 0, 0, 1, 0): 62,
            (43, 0, 0, 0, 1, 0): 10,
        }
    ),
    "A7(WIN)": pcounter(
        {
            (41, 1, 0, 0, 0, 1): 1,
            (43, 0, 0, 0, 0, 1): 1,
            (43, 1, 0, 0, 0, 1): 7,
            (44, 0, 0, 0, 0, 1): 1,
            (46, 1, 0, 0, 0, 1): 2,
            (47, 0, 0, 0, 0, 1): 1,
            (47, 1, 0, 0, 0, 1): 6,
            (48, 0, 0, 0, 0, 1): 1,
            (48, 1, 0, 0, 0, 1): 5,
            (49, 0, 0, 0, 0, 1): 6,
            (49, 1, 0, 0, 0, 1): 37,
            (50, 0, 0, 0, 0, 1): 6,
            (50, 1, 0, 0, 0, 1): 1,
            (51, 0, 0, 0, 0, 1): 1,
            (51, 1, 0, 0, 0, 1): 6,
            (52, 0, 0, 0, 0, 1): 1,
            (52, 1, 0, 0, 0, 1): 8,
            (53, 0, 0, 0, 0, 1): 4,
            (53, 1, 0, 0, 0, 1): 5,
            (54, 0, 0, 0, 0, 1): 4,
            (54, 1, 0, 0, 0, 1): 16,
            (55, 0, 0, 0, 0, 1): 4,
        }
    ),
}


EXPECTED_GREEDY = {
    "D0.1": {
        "inputs": 1,
        "outputs": 4,
        "maxima": Counter({(0, 5): 1}),
        "choice_counts": Counter({4: 1}),
        "union": {(-1, 1), (0, -1), (0, 1), (1, -1)},
    },
    "D0.2": {
        "inputs": 4,
        "outputs": 4,
        "maxima": Counter({(0, 5): 4}),
        "choice_counts": Counter({2: 4}),
        "union": {(-1, 1), (0, -1), (0, 1), (1, -1)},
    },
    "D1.1": {
        "inputs": 4,
        "outputs": 12,
        "maxima": Counter({(0, 6): 2, (0, 7): 2}),
        "choice_counts": Counter({1: 2, 5: 2}),
        "union": {(2, r) for r in range(-3, 2)},
    },
    "D1.2": {
        "inputs": 12,
        "outputs": 124,
        "maxima": Counter({(0, 5): 12}),
        "choice_counts": Counter({10: 8, 11: 4}),
        "union": {
            (0, -4), (0, -2), (0, 2), (0, 4),
            (1, -4), (1, -3), (1, 2), (1, 3),
            (2, -5), (2, 3),
            (3, -5), (3, -4), (3, 1), (3, 2),
        },
    },
    "D2.1": {
        "inputs": 124,
        "outputs": 242,
        "maxima": Counter({(0, 10): 124}),
        "choice_counts": Counter({1: 6, 2: 118}),
        "union": {(-5, 1), (-4, -1)},
    },
    "D2.2": {
        "inputs": 242,
        "outputs": 124,
        "maxima": Counter({(0, 9): 6, (0, 10): 236}),
        "choice_counts": Counter({1: 242}),
        "union": {(-5, 1), (-4, -1)},
    },
    "D3.1": {
        "inputs": 124,
        "outputs": 124,
        "maxima": Counter({(21, 4): 124}),
        "choice_counts": Counter({1: 124}),
        "union": {(-6, 0)},
    },
    "D3.2": {
        "inputs": 124,
        "outputs": 124,
        "maxima": Counter({(3, 9): 106, (0, 10): 18}),
        "choice_counts": Counter({1: 124}),
        "union": {(-3, 1), (-2, -1)},
    },
}


@dataclass
class DefenderRow:
    stage: str
    inputs: int
    outputs: int
    maxima: Counter
    choice_counts: Counter
    choice_union: frozenset[Coord]


@dataclass
class Trace:
    stages: list[tuple[str, tuple[Position, ...]]]
    defender_rows: list[DefenderRow]


def dedup(positions) -> tuple[Position, ...]:
    return tuple({p: p for p in positions}.values())


def record_stage(stages, name: str, states: tuple[Position, ...]) -> None:
    got = Counter(profile(p) for p in states)
    assert got == EXPECTED_PROFILES[name], (name, got, EXPECTED_PROFILES[name])
    stages.append((name, states))


def defender_step(
    states: tuple[Position, ...],
    stage: str,
    rows: list[DefenderRow],
    offset: Coord,
) -> tuple[Position, ...]:
    maxima = Counter()
    choice_counts = Counter()
    choice_union: set[Coord] = set()
    children: list[Position] = []
    for pos in states:
        choices, best = greedy(pos)
        assert best is not None and qsign(best) > 0  # no filler branch
        maxima[best] += 1
        choice_counts[len(choices)] += 1
        choice_union.update(choices)
        for x in choices:
            assert pos.legal(x)  # also follows from positive alive-window danger
            child = pos.place(x, attacker=False)
            assert not won(child, attacker=False)
            children.append(child)
    out = dedup(children)
    expected = EXPECTED_GREEDY[stage]
    assert len(states) == expected["inputs"]
    assert len(out) == expected["outputs"]
    assert maxima == expected["maxima"]
    assert choice_counts == expected["choice_counts"]
    assert choice_union == shifted_set(expected["union"], offset)
    rows.append(
        DefenderRow(
            stage,
            len(states),
            len(out),
            maxima,
            choice_counts,
            frozenset(choice_union),
        )
    )
    return out


def attacker_step(
    states: tuple[Position, ...],
    stage: str,
    x: Coord,
    final: bool,
    winning_window: frozenset[Coord],
) -> tuple[Position, ...]:
    children = []
    for pos in states:
        assert pos.legal(x)
        child = pos.place(x, attacker=True)
        if final:
            assert winning_window <= child.A
            assert winning_window.isdisjoint(child.D)
            assert won(child, attacker=True)
        else:
            assert not won(child, attacker=True)
        children.append(child)
    return dedup(children)


def run_trace(start: Position, offset: Coord = (0, 0)) -> Trace:
    stages: list[tuple[str, tuple[Position, ...]]] = []
    rows: list[DefenderRow] = []
    states = (start,)
    record_stage(stages, "P0", states)

    states = defender_step(states, "D0.1", rows, offset)
    record_stage(stages, "D0.1", states)
    states = defender_step(states, "D0.2", rows, offset)
    record_stage(stages, "D0.2", states)

    move_index = 0
    for turn in range(3):
        for stone in range(2):
            move_index += 1
            states = attacker_step(
                states,
                f"A{move_index}",
                shifted(ATTACKER_MOVES[move_index - 1], offset),
                final=False,
                winning_window=shifted_set(WINNING_WINDOW, offset),
            )
            record_stage(stages, f"A{move_index}", states)
        states = defender_step(states, f"D{turn + 1}.1", rows, offset)
        record_stage(stages, f"D{turn + 1}.1", states)
        states = defender_step(states, f"D{turn + 1}.2", rows, offset)
        record_stage(stages, f"D{turn + 1}.2", states)

    states = attacker_step(
        states,
        "A7(WIN)",
        shifted(ATTACKER_MOVES[-1], offset),
        final=True,
        winning_window=shifted_set(WINNING_WINDOW, offset),
    )
    record_stage(stages, "A7(WIN)", states)
    assert len(states) == 124
    return Trace(stages, rows)


def expanded_witness() -> tuple[Position, list[tuple[str, Coord]]]:
    """Build and verify a legal history ending at expanded Defender-FirstStone P0."""
    center = shifted((-10, 10), ENGINE_SHIFT)
    inner = ball(center, 2)
    outer = ball(center, 3)
    ring = outer - inner
    gateway = shifted((-8, 8), ENGINE_SHIFT)
    assert len(inner) == 19 and len(ring) == 18 and gateway in inner

    compact = Position(frozenset({shifted((0, 0), ENGINE_SHIFT)}), frozenset({(0, 0)}))
    target = Position(compact.A | inner, compact.D | ring)

    # Every window through an inner Attacker stone meets the Defender ring.
    for x in inner:
        for w in windows_at(x):
            assert ring.intersection(window_cells(w))

    # D opening, then A pair, then nine complete D/A pair cycles.  This gives
    # |A|=20, |D|=19 at the next Defender FirstStone epoch.
    history: list[tuple[str, Coord]] = []
    pos = Position(frozenset(), frozenset())

    def put(side: str, x: Coord, opening: bool = False) -> None:
        nonlocal pos
        assert x not in pos.occupied
        if opening:
            assert not pos.occupied and x == (0, 0)
        assert opening or pos.legal(x)
        pos = pos.place(x, attacker=(side == "A"))
        history.append((side, x))
        assert not won(pos, attacker=True)
        assert not won(pos, attacker=False)

    put("D", (0, 0), opening=True)
    put("A", shifted((0, 0), ENGINE_SHIFT))
    put("A", gateway)
    remaining_a = sorted(inner - {gateway})
    ring_order = sorted(ring)
    for i in range(9):
        for x in ring_order[2 * i : 2 * i + 2]:
            put("D", x)
        for x in remaining_a[2 * i : 2 * i + 2]:
            put("A", x)

    expected_sides = ["D", "A", "A"] + [side for _ in range(9) for side in ("D", "D", "A", "A")]
    assert [side for side, _x in history] == expected_sides
    assert pos == target
    assert len(pos.A) == 20 and len(pos.D) == 19
    assert not won(pos, attacker=True) and not won(pos, attacker=False)
    assert profile(pos) == (13, 0, 0, 0, 0, 0)

    # The padding is at distance at least six from every tactical cell that
    # can occur in the exhaustive tree; therefore no length-six window (whose
    # diameter is five) can meet both components.
    tactical = {shifted((0, 0), ENGINE_SHIFT), (0, 0)}
    tactical.update(shifted_set(ATTACKER_MOVES, ENGINE_SHIFT))
    for spec in EXPECTED_GREEDY.values():
        tactical.update(shifted_set(spec["union"], ENGINE_SHIFT))
    assert min(hdist(x, y) for x in outer for y in tactical) == 6

    return target, history


def counter_text(counter: Counter, formatter=str) -> str:
    return "; ".join(
        f"{formatter(key)} x {count}"
        for key, count in sorted(counter.items(), key=lambda kv: str(kv[0]))
    )


def coords_text(coords) -> str:
    return "{" + ", ".join(map(str, sorted(coords))) + "}"


def print_tables(trace: Trace) -> None:
    print("\nEXACT PER-PLY PROFILE / POTENTIAL TABLE")
    print("(profile is (n1,n2,n3,n4,n5,n6); each potential entry is exact 27*Phi)")
    print("stage | states | profile multiplicities | 27 Phi multiplicities")
    print("---|---:|---|---")
    for stage, states in trace.stages:
        profiles = Counter(profile(p) for p in states)
        phis = Counter(phi_27(p) for p in states)
        print(
            f"{stage} | {len(states)} | {counter_text(profiles)} | "
            f"{counter_text(phis, qtext)}"
        )

    print("\nEXACT GREEDY-MAXIMUM TABLE")
    print("(each danger entry is exact 27*d; all are positive, so no filler occurs)")
    print("placement | inputs | choice-count multiplicities | 27 max-danger multiplicities | maximizer union | outputs")
    print("---|---:|---|---|---|---:")
    for row in trace.defender_rows:
        print(
            f"{row.stage} | {row.inputs} | {counter_text(row.choice_counts)} | "
            f"{counter_text(row.maxima, qtext)} | {coords_text(row.choice_union)} | "
            f"{row.outputs}"
        )


def incident_counts(pos: Position, x: Coord) -> tuple[int, ...]:
    return tuple(sorted(s for _w, s, empties in alive(pos) if x in empties))


def main() -> None:
    compact = Position(frozenset({(0, 0)}), frozenset({(1, 0)}))
    assert not won(compact, attacker=True) and not won(compact, attacker=False)
    assert profile(compact) == (13, 0, 0, 0, 0, 0)
    assert phi_27(compact) == (0, 13)
    assert qcmp(phi_27(compact), (27, 0)) < 0  # 13sqrt(3)/27 < 1

    expanded, history = expanded_witness()
    assert phi_27(expanded) == (0, 13)

    compact_trace = run_trace(compact)
    expanded_trace = run_trace(expanded, ENGINE_SHIFT)

    padding_a = expanded.A - shifted_set(compact.A, ENGINE_SHIFT)
    padding_d = expanded.D - shifted_set(compact.D, ENGINE_SHIFT)

    for (c_name, c_states), (e_name, e_states) in zip(
        compact_trace.stages, expanded_trace.stages, strict=True
    ):
        assert c_name == e_name
        translated_states = {
            Position(
                shifted_set(p.A, ENGINE_SHIFT) | padding_a,
                shifted_set(p.D, ENGINE_SHIFT) | padding_d,
            )
            for p in c_states
        }
        assert set(e_states) == translated_states

    # The dead, separated reachability gadget changes no tactical statistic.
    assert [
        (name, Counter(profile(p) for p in states), Counter(phi_27(p) for p in states))
        for name, states in compact_trace.stages
    ] == [
        (name, Counter(profile(p) for p in states), Counter(phi_27(p) for p in states))
        for name, states in expanded_trace.stages
    ]
    for compact_row, expanded_row in zip(
        compact_trace.defender_rows, expanded_trace.defender_rows, strict=True
    ):
        assert compact_row.stage == expanded_row.stage
        assert compact_row.inputs == expanded_row.inputs
        assert compact_row.outputs == expanded_row.outputs
        assert compact_row.maxima == expanded_row.maxima
        assert compact_row.choice_counts == expanded_row.choice_counts
        assert expanded_row.choice_union == shifted_set(
            compact_row.choice_union, ENGINE_SHIFT
        )

    # The immediate threat at (-1,0) is strictly below greedy's two maxima.
    pre_d3 = dict(compact_trace.stages)["A6"]
    assert Counter(incident_counts(p, (-6, 0)) for p in pre_d3) == Counter(
        {(1, 2, 3, 4, 4): 124}
    )
    assert Counter(incident_counts(p, (-1, 0)) for p in pre_d3) == Counter(
        {(4, 5): 124}
    )
    assert Counter(danger_27(p, (-1, 0)) for p in pre_d3) == Counter({(9, 9): 124})
    assert Counter(greedy(p)[1] for p in pre_d3) == Counter({(21, 4): 124})
    assert qcmp((21, 4), (9, 9)) > 0  # difference 12-5sqrt(3)>0

    post_d31 = dict(compact_trace.stages)["D3.1"]
    assert Counter(incident_counts(p, (-1, 0)) for p in post_d31) == Counter(
        {(5,): 124}
    )
    assert Counter(danger_27(p, (-1, 0)) for p in post_d31) == Counter({(0, 9): 124})
    assert Counter((greedy(p)[0], greedy(p)[1]) for p in post_d31) == Counter(
        {(((-3, 1),), (3, 9)): 106, (((-2, -1),), (0, 10)): 18}
    )
    assert qcmp((3, 9), (0, 9)) > 0
    assert qcmp((0, 10), (0, 9)) > 0

    print("GREEDY-REFUTATION VERIFIED")
    print("compact P0: A={(0,0)}, D={(1,0)}, Defender FirstStone")
    print("expanded engine-reachable P0: |A|=20, |D|=19, Defender FirstStone")
    print(f"explicit expanded-position history placements: {len(history)}")
    print("initial profile: (n1,n2,n3,n4,n5,n6)=(13,0,0,0,0,0)")
    print("initial Phi = 13sqrt(3)/27 = 13/(9sqrt(3)) < 1")
    print("fixed Attacker stones by turn:")
    print("  (2,-4),(2,2); (-5,0),(-4,0); (-3,0),(-2,0); (-1,0) WIN")
    print("  expanded engine history and continuation use the translation (-1,0)")
    print("all exact greedy ties branched; no filler; no Defender completion")
    print("critical final turn (compact coordinates):")
    print("  before D3.1: 27d(-6,0)=21+4sqrt(3) > 9+9sqrt(3)=27d(-1,0)")
    print("  the five Q-window counts at (-6,0) are 1,2,3,4,4")
    print("  after D3.1: 27d(-1,0)=9sqrt(3), while the next maximum is")
    print("  3+9sqrt(3) in 106 states or 10sqrt(3) in 18 states")
    print_tables(compact_trace)


if __name__ == "__main__":
    main()
