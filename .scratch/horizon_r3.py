#!/usr/bin/env python3
"""Horizon R3: true-rule legality, quotient universes, and next-rung probes.

This is research code.  It deliberately imports the frozen R2/H10 deciders so
that every optimized answer can be checked against its predecessor.  No search
limit is part of a verdict; command-line frames decide which roots to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".scratch"))

import deadline_ladder_r as phase_r  # noqa: E402
import horizon_h10 as h10  # noqa: E402
import horizon_production_h8 as production_h8  # noqa: E402
import horizon_r2 as r2  # noqa: E402


Cell = tuple[int, int]
LEGAL_RADIUS = 8


def bits(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        mask ^= bit
        yield bit


def hex_distance(a: Cell, b: Cell) -> int:
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def legal_cells(board: dict[Cell, int]) -> set[Cell]:
    """Exact non-opening legal carrier: empty cells within distance eight."""
    if not board:
        return {(0, 0)}
    out: set[Cell] = set()
    for q, r in board:
        for dq in range(-LEGAL_RADIUS, LEGAL_RADIUS + 1):
            lo = max(-LEGAL_RADIUS, -dq - LEGAL_RADIUS)
            hi = min(LEGAL_RADIUS, -dq + LEGAL_RADIUS)
            for dr in range(lo, hi + 1):
                cell = (q + dq, r + dr)
                if cell not in board:
                    out.add(cell)
    return out


def pair_is_legal(pair_cells: tuple[Cell, ...], occupied: set[Cell]) -> bool:
    """Whether an unordered one/two-placement turn has a legal ordering."""
    if not pair_cells:
        return True
    root_legal = lambda x: any(hex_distance(x, stone) <= LEGAL_RADIUS for stone in occupied)
    if len(pair_cells) == 1:
        return root_legal(pair_cells[0])
    x, y = pair_cells
    return (root_legal(x) and (root_legal(y) or hex_distance(x, y) <= LEGAL_RADIUS)) or (
        root_legal(y) and (root_legal(x) or hex_distance(x, y) <= LEGAL_RADIUS)
    )


def endpoint_schedule(n: int, horizon: int, target: int) -> tuple[int, ...]:
    """Schedule through the target's last placement, excluding trailing plies."""
    sched = r2.schedule(n, horizon)
    last = max(i for i, owner in enumerate(sched) if owner == target)
    return sched[: last + 1]


@dataclass(frozen=True)
class QModel:
    mover: int
    cells: tuple[Cell, ...]
    windows: dict[int, tuple[int, ...]]
    classes: tuple[int, ...]

    @property
    def physical_universe(self) -> int:
        return len(self.cells)

    @property
    def quotient_universe(self) -> int:
        return len(self.classes)


def incidence_classes(windows: tuple[tuple[int, ...], tuple[int, ...]], n: int) -> tuple[int, ...]:
    """Partition physical cells by their complete tagged-window incidence.

    Multiplicity is retained in each returned mask.  A search state may use
    zero, one, or several members of a class; only permutations within a class
    are quotiented.
    """
    signatures = [0] * n
    tag = 0
    for family in windows:
        for edge in family:
            for bit in bits(edge):
                signatures[bit.bit_length() - 1] |= 1 << tag
            tag += 1
    grouped: dict[int, int] = {}
    for i, signature in enumerate(signatures):
        grouped[signature] = grouped.get(signature, 0) | (1 << i)
    return tuple(grouped[key] for key in sorted(grouped))


def build_qmodel(row: dict, horizon: int, target: int) -> QModel:
    n = len(row["moves"])
    mover = phase_r.phase_player(n)[1]
    sched = endpoint_schedule(n, horizon, target)
    board = {tuple(c): phase_r.owner_at(i) for i, c in enumerate(row["moves"])}
    raw = r2.root_windows(board, sched)
    universe = tuple(sorted(set().union(*raw[0], *raw[1]) if raw[0] or raw[1] else set()))
    index = {cell: i for i, cell in enumerate(universe)}
    windows = {
        p: tuple(sum(1 << index[cell] for cell in edge) for edge in raw[p])
        for p in (0, 1)
    }
    classes = incidence_classes((windows[0], windows[1]), len(universe))
    return QModel(mover, universe, windows, classes)


def class_actions(active: int, classes: tuple[int, ...]) -> Iterator[int]:
    """One representative of every unordered pair modulo incidence twins."""
    available: list[tuple[int, int | None]] = []
    physical = active.bit_count()
    if not physical:
        yield 0
        return
    if physical == 1:
        yield active & -active
        return
    for cls in classes:
        live = cls & active
        if not live:
            continue
        first = live & -live
        rest = live ^ first
        second = rest & -rest if rest else None
        available.append((first, second))
    for i, (first, second) in enumerate(available):
        if second is not None:
            yield first | second
        for other, _ in available[i + 1 :]:
            yield first | other


def ordered_class_actions(
    active: int,
    classes: tuple[int, ...],
    own: Iterable[int],
    other: Iterable[int],
) -> Iterator[int]:
    """Exact class-action stream with an exactness-neutral tactical prefix."""
    own = tuple(own)
    other = tuple(other)
    physical = active.bit_count()
    if physical <= 1:
        yield 0 if not physical else active & -active
        return

    available: list[tuple[int, int | None]] = []
    for cls in classes:
        live = cls & active
        if live:
            first = live & -live
            rest = live ^ first
            available.append((first, (rest & -rest) if rest else None))

    seen: set[int] = set()

    def normalized_pair(required: int) -> int | None:
        """Canonical class action containing a one/two-cell requirement."""
        if required.bit_count() > 2 or required & ~active:
            return None
        chosen = required
        if chosen.bit_count() == 1:
            for first, second in available:
                for mate in (first, second):
                    if mate is not None and not mate & chosen:
                        return chosen | mate
        return chosen if chosen.bit_count() == 2 else None

    # Exact terminal prefixes first without scoring the quadratic action set.
    for edge in own:
        pair = normalized_pair(edge & active)
        if pair is not None and pair not in seen:
            seen.add(pair)
            yield pair

    def class_score(item: tuple[int, int | None]) -> int:
        first, _ = item
        return sum(bool(edge & first) for edge in own) + 2 * sum(bool(edge & first) for edge in other)

    ranked = sorted(available, key=class_score, reverse=True)

    def action(left: tuple[int, int | None], right: tuple[int, int | None] | None) -> int | None:
        if right is None:
            return left[0] | left[1] if left[1] is not None else None
        return left[0] | right[0]

    # A small tactical prefix normally finds existential responses quickly.
    for i, left in enumerate(ranked[:12]):
        pair = action(left, None)
        if pair is not None and pair not in seen:
            seen.add(pair)
            yield pair
        for right in ranked[i + 1 : 12]:
            pair = action(left, right)
            if pair not in seen:
                seen.add(pair)
                yield pair

    # Exhaustive fallback.  Pair order is semantically irrelevant.
    for i, left in enumerate(available):
        pair = action(left, None)
        if pair is not None and pair not in seen:
            seen.add(pair)
            yield pair
        for right in available[i + 1 :]:
            pair = action(left, right)
            if pair not in seen:
                seen.add(pair)
                yield pair


@dataclass(frozen=True)
class R3Decision:
    win: bool
    nodes: int
    physical_universe: int
    quotient_universe: int
    target_windows: int
    opponent_windows: int
    wall_ns: int
    witness: tuple[Cell, ...] | None = None


def decode_pair(pair: int, cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    return tuple(cells[i] for i in range(len(cells)) if pair & (1 << i))


def decide_fresh_current_h8(row: dict) -> R3Decision:
    """Exact fresh current-player WinWithin8 (= WinWithin6)."""
    started = time.perf_counter_ns()
    mover = phase_r.phase_player(len(row["moves"]))[1]
    model = build_qmodel(row, 6, mover)
    own, opp = model.windows[mover], model.windows[1 - mover]
    active = 0
    for edge in own + opp:
        active |= edge
    nodes = 0
    for pair in ordered_class_actions(active, model.classes, own, opp):
        nodes += 1
        if any(not edge & ~pair for edge in own):
            return R3Decision(True, nodes, model.physical_universe, model.quotient_universe,
                              len(own), len(opp), time.perf_counter_ns() - started,
                              decode_pair(pair, model.cells))
        if any(not edge & pair and edge.bit_count() <= 2 for edge in opp):
            continue
        threats = [edge & ~pair for edge in own if (edge & ~pair).bit_count() <= 2]
        if threats and h10._cover_two_witness(threats) is None:
            return R3Decision(True, nodes, model.physical_universe, model.quotient_universe,
                              len(own), len(opp), time.perf_counter_ns() - started,
                              decode_pair(pair, model.cells))
    return R3Decision(False, nodes, model.physical_universe, model.quotient_universe,
                      len(own), len(opp), time.perf_counter_ns() - started)


def decide_fresh_loss_h8(row: dict) -> R3Decision:
    """Exact fresh opponent WinWithin8, with incidence-class action quotient."""
    started = time.perf_counter_ns()
    mover = phase_r.phase_player(len(row["moves"]))[1]
    target = 1 - mover
    model = build_qmodel(row, 8, target)
    a_windows, d_windows = model.windows[mover], model.windows[target]
    active_a = 0
    for edge in a_windows + d_windows:
        active_a |= edge
    nodes = 0
    for a_pair in ordered_class_actions(active_a, model.classes, a_windows, d_windows):
        nodes += 1
        if any(not edge & ~a_pair for edge in a_windows):
            return R3Decision(False, nodes, model.physical_universe, model.quotient_universe,
                              len(d_windows), len(a_windows), time.perf_counter_ns() - started)
        live_d = tuple(edge for edge in d_windows if not edge & a_pair)
        live_a = tuple(edge & ~a_pair for edge in a_windows)
        active_d = 0
        for edge in live_d + live_a:
            active_d |= edge
        response = False
        for d_pair in ordered_class_actions(active_d, model.classes, live_d, live_a):
            nodes += 1
            if any(not edge & ~d_pair for edge in live_d):
                response = True
                break
            if any(not edge & d_pair and edge.bit_count() <= 2 for edge in live_a):
                continue
            threats = [edge & ~d_pair for edge in live_d if (edge & ~d_pair).bit_count() <= 2]
            if threats and h10._cover_two_witness(threats) is None:
                response = True
                break
        if not response:
            return R3Decision(False, nodes, model.physical_universe, model.quotient_universe,
                              len(d_windows), len(a_windows), time.perf_counter_ns() - started)
    return R3Decision(True, nodes, model.physical_universe, model.quotient_universe,
                      len(d_windows), len(a_windows), time.perf_counter_ns() - started)


def decide_second_current_h8(row: dict) -> R3Decision:
    """Exact A,D2,A2,D2,A endpoint from a SecondStone root."""
    started = time.perf_counter_ns()
    mover = phase_r.phase_player(len(row["moves"]))[1]
    model = build_qmodel(row, 8, mover)
    own, opp = model.windows[mover], model.windows[1 - mover]
    active = 0
    for edge in own + opp:
        active |= edge
    first_moves = [cls & -cls for cls in model.classes if cls & active] or [0]
    nodes = 0
    for a_bit in first_moves:
        nodes += 1
        if any(not edge & ~a_bit for edge in own):
            return R3Decision(True, nodes, model.physical_universe, model.quotient_universe,
                              len(own), len(opp), time.perf_counter_ns() - started,
                              decode_pair(a_bit, model.cells))
        a_future = tuple(edge & ~a_bit for edge in own if (edge & ~a_bit).bit_count() <= 3)
        d_live = tuple(edge for edge in opp if not edge & a_bit)
        active_d = 0
        for edge in a_future + d_live:
            active_d |= edge
        first_wins = True
        for d_pair in ordered_class_actions(active_d, model.classes, d_live, a_future):
            nodes += 1
            if any(not edge & ~d_pair for edge in d_live):
                first_wins = False
                break
            live_a = tuple(edge for edge in a_future if not edge & d_pair)
            live_d = tuple(edge & ~d_pair for edge in d_live)
            active_b = 0
            for edge in live_a:
                active_b |= edge
            for edge in live_d:
                if edge.bit_count() <= 2:
                    active_b |= edge
            reply = False
            for b_pair in ordered_class_actions(active_b, model.classes, live_a, live_d):
                nodes += 1
                if any(not edge & ~b_pair for edge in live_a):
                    reply = True
                    break
                if any(not edge & b_pair and (edge & ~b_pair).bit_count() <= 2 for edge in live_d):
                    continue
                final_cells = 0
                for edge in live_a:
                    residual = edge & ~b_pair
                    if residual.bit_count() == 1:
                        final_cells |= residual
                if final_cells.bit_count() > 2:
                    reply = True
                    break
            if not reply:
                first_wins = False
                break
        if first_wins:
            return R3Decision(True, nodes, model.physical_universe, model.quotient_universe,
                              len(own), len(opp), time.perf_counter_ns() - started,
                              decode_pair(a_bit, model.cells))
    return R3Decision(False, nodes, model.physical_universe, model.quotient_universe,
                      len(own), len(opp), time.perf_counter_ns() - started)


def decide_second_loss_h8(row: dict) -> R3Decision:
    """Exact opponent win by ply seven; the attacker quota is three, not four."""
    started = time.perf_counter_ns()
    mover = phase_r.phase_player(len(row["moves"]))[1]
    target = 1 - mover
    model = build_qmodel(row, 8, target)
    attacker, defender = model.windows[mover], model.windows[target]
    active = 0
    for edge in attacker + defender:
        active |= edge
    first_moves = [cls & -cls for cls in model.classes if cls & active] or [0]
    nodes = 0
    for a_bit in first_moves:
        nodes += 1
        if any(not edge & ~a_bit for edge in attacker):
            return R3Decision(False, nodes, model.physical_universe, model.quotient_universe,
                              len(defender), len(attacker), time.perf_counter_ns() - started)
        live_d0 = tuple(edge for edge in defender if not edge & a_bit)
        live_a0 = tuple(edge & ~a_bit for edge in attacker)
        active_d = 0
        for edge in live_d0:
            active_d |= edge
        for edge in live_a0:
            if edge.bit_count() <= 2:
                active_d |= edge
        response = False
        for d_pair in ordered_class_actions(active_d, model.classes, live_d0, live_a0):
            nodes += 1
            if any(not edge & ~d_pair for edge in live_d0):
                response = True
                break
            live_d = tuple(edge & ~d_pair for edge in live_d0)
            live_a = tuple(edge for edge in live_a0 if not edge & d_pair)
            active_b = 0
            for edge in live_a:
                if edge.bit_count() <= 2:
                    active_b |= edge
            for edge in live_d:
                if edge.bit_count() <= 2:
                    active_b |= edge
            all_replies_lose = True
            for b_pair in ordered_class_actions(active_b, model.classes, live_a, live_d):
                nodes += 1
                if any(not edge & ~b_pair for edge in live_a):
                    all_replies_lose = False
                    break
                if not any(not edge & b_pair and (edge & ~b_pair).bit_count() <= 2 for edge in live_d):
                    all_replies_lose = False
                    break
            if all_replies_lose:
                response = True
                break
        if not response:
            return R3Decision(False, nodes, model.physical_universe, model.quotient_universe,
                              len(defender), len(attacker), time.perf_counter_ns() - started)
    return R3Decision(True, nodes, model.physical_universe, model.quotient_universe,
                      len(defender), len(attacker), time.perf_counter_ns() - started)


def decide_both_h8(row: dict) -> tuple[R3Decision, R3Decision]:
    phase, mover, _ = phase_r.phase_player(len(row["moves"]))
    if phase == "first":
        return decide_fresh_current_h8(row), decide_fresh_loss_h8(row)
    if phase == "second":
        return decide_second_current_h8(row), decide_second_loss_h8(row)
    # The opening action is forced and neither side has six placements by h8.
    zero = R3Decision(False, 1, 0, 0, 0, 0, 0)
    return zero, zero


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))] if ordered else 0


def distribution(values: list[int]) -> dict[str, int]:
    return {
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "max": max(values) if values else 0,
    }


def cost_distribution(values: list[int]) -> dict[str, float]:
    return {
        "mean_us": sum(values) / len(values) / 1e3 if values else 0.0,
        "p50_us": percentile(values, 0.5) / 1e3,
        "p90_us": percentile(values, 0.9) / 1e3,
        "max_us": max(values) / 1e3 if values else 0.0,
        "total_ms": sum(values) / 1e6,
    }


def covering_pair_actions(family: list[int], active: int) -> list[int]:
    """Every normalized relevant pair hitting ``family`` (whose edges are tiny)."""
    if not family:
        return []
    physical = list(bits(active))
    if not physical:
        return []
    if len(physical) == 1:
        return physical if all(edge & physical[0] for edge in family) else []
    out: set[int] = set()
    for x in bits(family[0] & active):
        rest = [edge & active for edge in family if not edge & x]
        if not rest:
            for y in physical:
                if y != x:
                    out.add(x | y)
            continue
        common = rest[0]
        for edge in rest[1:]:
            common &= edge
        for y in bits(common):
            if y != x:
                out.add(x | y)
    return list(out)


def threat_making_pair_actions(family: Iterable[int], active: int) -> list[int]:
    """Every pair which can leave at least one residual of size at most two."""
    physical = list(bits(active))
    if not physical:
        return [0]
    if len(physical) == 1:
        return physical
    low = 0
    fours: list[int] = []
    for edge in family:
        size = edge.bit_count()
        if size <= 3:
            low |= edge
        elif size == 4:
            fours.append(edge)
    out: set[int] = set()
    for x in bits(low & active):
        for y in physical:
            if x != y:
                out.add(x | y)
    for edge in fours:
        edge_bits = list(bits(edge & active))
        for x, y in combinations(edge_bits, 2):
            out.add(x | y)
    return list(out)


def quotient_action_list(actions: Iterable[int], classes: tuple[int, ...], active: int) -> list[int]:
    """Canonicalize physical pair actions under a node-local incidence partition."""
    owner: dict[int, int] = {}
    live_classes: list[int] = []
    for ci, cls in enumerate(classes):
        live = cls & active
        if not live:
            continue
        live_classes.append(live)
        for bit in bits(live):
            owner[bit] = len(live_classes) - 1
    out: set[int] = set()
    for pair in actions:
        selected = list(bits(pair))
        if len(selected) <= 1:
            out.add(pair)
            continue
        ca, cb = owner[selected[0]], owner[selected[1]]
        if ca != cb:
            a = live_classes[ca] & -live_classes[ca]
            b = live_classes[cb] & -live_classes[cb]
            out.add(a | b)
        else:
            first = live_classes[ca] & -live_classes[ca]
            rest = live_classes[ca] ^ first
            if rest:
                out.add(first | (rest & -rest))
    return list(out)


def measure_h8_battery() -> dict:
    predecessor = json.loads((ROOT / ".scratch" / "horizon_production_h8.json").read_text(encoding="utf-8"))
    result: dict = {
        "metadata": {
            "exact": True,
            "horizon": 8,
            "scope": "all 6,443 frozen rows plus the overlapping 248-row grind audit",
            "before_artifact": ".scratch/horizon_production_h8.json",
        },
        "cohorts": {},
    }
    for name, rows in production_h8.all_cohorts().items():
        before_rows = {r["pos_id"]: r for r in predecessor["cohorts"][name]["rows"]}
        measured = []
        mismatches = []
        started = time.perf_counter_ns()
        for i, row in enumerate(rows, 1):
            current, loss = decide_both_h8(row)
            old = before_rows[row["pos_id"]]
            if (current.win, loss.win) != (old["current_win"], old["forced_loss"]):
                mismatches.append({
                    "pos_id": row["pos_id"],
                    "phase": old["phase"],
                    "before": [old["current_win"], old["forced_loss"]],
                    "after": [current.win, loss.win],
                })
            measured.append({
                "pos_id": row["pos_id"],
                "phase": old["phase"],
                "current_win": current.win,
                "forced_loss": loss.win,
                "current_before_universe": old["current_universe"],
                "current_endpoint_physical_universe": current.physical_universe,
                "current_after_universe": current.quotient_universe,
                "loss_before_universe": old["loss_universe"],
                "loss_endpoint_physical_universe": loss.physical_universe,
                "loss_after_universe": loss.quotient_universe,
                "current_before_wall_ns": old["current_wall_ns"],
                "current_after_wall_ns": current.wall_ns,
                "loss_before_wall_ns": old["loss_wall_ns"],
                "loss_after_wall_ns": loss.wall_ns,
                "current_before_nodes": old["current_nodes"],
                "current_after_nodes": current.nodes,
                "loss_before_nodes": old["loss_nodes"],
                "loss_after_nodes": loss.nodes,
            })
            if i % 100 == 0:
                print(f"{name}: {i}/{len(rows)}", flush=True)
        item = {
            "n": len(rows),
            "mismatches": mismatches,
            "current_wins": sum(r["current_win"] for r in measured),
            "forced_losses": sum(r["forced_loss"] for r in measured),
            "universe": {
                "current_before_physical": distribution([r["current_before_universe"] for r in measured]),
                "current_after_endpoint_physical": distribution([r["current_endpoint_physical_universe"] for r in measured]),
                "current_after_incidence_classes": distribution([r["current_after_universe"] for r in measured]),
                "loss_before_physical": distribution([r["loss_before_universe"] for r in measured]),
                "loss_after_endpoint_physical": distribution([r["loss_endpoint_physical_universe"] for r in measured]),
                "loss_after_incidence_classes": distribution([r["loss_after_universe"] for r in measured]),
            },
            "wall": {
                "current_before": cost_distribution([r["current_before_wall_ns"] for r in measured]),
                "current_after": cost_distribution([r["current_after_wall_ns"] for r in measured]),
                "loss_before": cost_distribution([r["loss_before_wall_ns"] for r in measured]),
                "loss_after": cost_distribution([r["loss_after_wall_ns"] for r in measured]),
                "frame_after_ms": (time.perf_counter_ns() - started) / 1e6,
            },
            "nodes": {
                "current_before": distribution([r["current_before_nodes"] for r in measured]),
                "current_after": distribution([r["current_after_nodes"] for r in measured]),
                "loss_before": distribution([r["loss_before_nodes"] for r in measured]),
                "loss_after": distribution([r["loss_after_nodes"] for r in measured]),
            },
            "rows": measured,
        }
        result["cohorts"][name] = item
    result["validation"] = {
        "rows_frozen": sum(result["cohorts"][n]["n"] for n in ("selfplay_v1", "human_v1", "puzzle_v3")),
        "grind_rows": result["cohorts"]["grinds"]["n"],
        "mismatches": sum(len(item["mismatches"]) for item in result["cohorts"].values()),
    }
    return result


@dataclass(frozen=True)
class H10R3Decision:
    win: bool
    h8_shortcut: bool
    nodes: int
    old_universe: int
    quotient_universe: int
    peak_branch_universe: int
    old_first_pairs: int
    legal_first_pairs: int
    wall_ns: int
    witness: tuple[Cell, ...] | None


def h10_quotient_classes(model: h10.H10Model) -> tuple[int, ...]:
    return incidence_classes(
        (model.target_windows + model.near_windows, model.opponent_windows),
        len(model.cells),
    )


def legal_h10_first_pairs(row: dict, model: h10.H10Model) -> tuple[int, ...]:
    """The H10 first-pair stream after applying the real radius-eight rule."""
    board = {tuple(cell): phase_r.owner_at(i) for i, cell in enumerate(row["moves"])}
    legal = legal_cells(board)
    legal_mask = sum(1 << i for i, cell in enumerate(model.cells) if cell in legal)
    kept = []
    for pair in model.first_pairs:
        first_legal = pair & legal_mask
        if not first_legal:
            continue
        if not pair & ~legal_mask or pair.bit_count() <= 1:
            kept.append(pair)
            continue
        first_bit = first_legal & -first_legal
        other_bit = (pair & ~first_bit) & -(pair & ~first_bit)
        if hex_distance(
            model.cells[first_bit.bit_length() - 1],
            model.cells[other_bit.bit_length() - 1],
        ) <= LEGAL_RADIUS:
            kept.append(pair)
    return tuple(kept)


def h10_peak_branch_universe(model: h10.H10Model, first_pairs: Iterable[int]) -> int:
    peak = 0
    for pair in first_pairs:
        anchored_future = tuple(
            edge & ~pair for edge in model.target_windows
            if (edge & ~pair).bit_count() <= 4
        )
        remote_future = tuple(
            edge & ~pair for edge in model.near_by_first_pair.get(pair, ())
        )
        d_live = tuple(edge for edge in model.opponent_windows if not edge & pair)
        active = pair
        for edge in anchored_future + remote_future + d_live:
            active |= edge
        peak = max(peak, active.bit_count())
    return peak


def decide_fresh_h10_r3(row: dict) -> H10R3Decision:
    """Exact true-rule H10 endpoint with legal-pair and incidence quotients."""
    started = time.perf_counter_ns()
    model = h10.build_model(row)
    classes = h10_quotient_classes(model)
    first_pairs = legal_h10_first_pairs(row, model)
    peak = 0

    old_h8 = r2.decide_fresh_current(row, 8)
    if old_h8.win:
        return H10R3Decision(
            True, True, old_h8.nodes, len(model.cells), len(classes), peak,
            len(model.first_pairs), len(first_pairs), time.perf_counter_ns() - started, None,
        )

    target = model.target_windows + model.near_windows
    opponent = model.opponent_windows
    nodes = 0
    for a_pair in first_pairs:
        nodes += 1
        if any(not edge & ~a_pair for edge in target):
            return H10R3Decision(
                True, False, nodes, len(model.cells), len(classes), peak,
                len(model.first_pairs), len(first_pairs), time.perf_counter_ns() - started,
                decode_pair(a_pair, model.cells),
            )
        anchored_future = tuple(
            edge & ~a_pair for edge in model.target_windows
            if (edge & ~a_pair).bit_count() <= 4
        )
        remote_future = tuple(
            edge & ~a_pair for edge in model.near_by_first_pair.get(a_pair, ())
        )
        a_future = anchored_future + remote_future
        d_live = tuple(edge for edge in opponent if not edge & a_pair)
        active_d = 0
        for edge in a_future + d_live:
            active_d |= edge
        peak = max(peak, (a_pair | active_d).bit_count())
        first_pair_wins = True
        for d_pair in h10._ordered_pair_iter(active_d, d_live, a_future):
            nodes += 1
            if any(not edge & ~d_pair for edge in d_live):
                first_pair_wins = False
                break
            live_a = tuple(edge for edge in a_future if not edge & d_pair)
            live_d = tuple(edge & ~d_pair for edge in d_live)
            active_b = 0
            for edge in live_a:
                active_b |= edge
            for edge in live_d:
                if edge.bit_count() <= 2:
                    active_b |= edge
            peak = max(peak, (a_pair | d_pair | active_b).bit_count())
            reply_wins = False
            for b_pair in h10._ordered_pair_iter(active_b, live_a, live_d):
                nodes += 1
                if any(not edge & ~b_pair for edge in live_a):
                    reply_wins = True
                    break
                if any(not edge & b_pair and (edge & ~b_pair).bit_count() <= 2 for edge in live_d):
                    continue
                threats = [edge & ~b_pair for edge in live_a if (edge & ~b_pair).bit_count() <= 2]
                if threats and h10._cover_two_witness(threats) is None:
                    reply_wins = True
                    break
            if not reply_wins:
                first_pair_wins = False
                break
        if first_pair_wins:
            return H10R3Decision(
                True, False, nodes, len(model.cells), len(classes), peak,
                len(model.first_pairs), len(first_pairs), time.perf_counter_ns() - started,
                decode_pair(a_pair, model.cells),
            )
    return H10R3Decision(
        False, False, nodes, len(model.cells), len(classes), peak,
        len(model.first_pairs), len(first_pairs), time.perf_counter_ns() - started, None,
    )


def validate_h10_cohort() -> dict:
    """Replay H10's completed 78-certificate boundary and its two new roots."""
    known, all_rows = h10._known_registry()
    eligible8 = sorted(
        pos_id for pos_id, info in known.items()
        if info["cert_depth"] is not None and info["cert_depth"] <= 8
    )
    new_ids = ["human_b132a09ccb4eb829_p101", "sp_20_p77"]
    rows = []
    misses = []
    universe_old = []
    universe_classes = []
    universe_peak = []
    pair_old = []
    pair_legal = []
    for pos_id in eligible8 + new_ids:
        row = all_rows[pos_id]
        phase = phase_r.phase_player(len(row["moves"]))[0]
        if phase == "first":
            decision = decide_fresh_h10_r3(row)
            win = decision.win
            detail = asdict(decision)
            universe_old.append(decision.old_universe)
            universe_classes.append(decision.quotient_universe)
            if not decision.h8_shortcut:
                universe_peak.append(decision.peak_branch_universe)
            pair_old.append(decision.old_first_pairs)
            pair_legal.append(decision.legal_first_pairs)
        else:
            # At non-fresh H10 roots the target quota is <6.  The R3 endpoint
            # clock and relevance quotient are exactly build_qmodel + R2 minimax;
            # H10's completed cohort already used this same finite path.
            mover = phase_r.phase_player(len(row["moves"]))[1]
            old = r2.decide_ladder(row, 10, mover)
            model = build_qmodel(row, 10, mover)
            win = old.win
            detail = {
                "win": win,
                "nodes": old.nodes,
                "old_universe": model.physical_universe,
                "quotient_universe": model.quotient_universe,
                "wall_ns": old.wall_ns,
            }
            universe_old.append(model.physical_universe)
            universe_classes.append(model.quotient_universe)
        if not win:
            misses.append(pos_id)
        witness = detail.get("witness")
        if witness:
            root_stones = [tuple(cell) for cell in row["moves"]]
            detail["witness_min_root_distances"] = [
                min(hex_distance(tuple(cell), stone) for stone in root_stones)
                for cell in witness
            ]
        rows.append({
            "pos_id": pos_id,
            "cert_depth": known[pos_id]["cert_depth"],
            "phase": phase,
            **detail,
        })
    by_cohort = {}
    for name, cohort_rows in r2.cohorts().items():
        cohort_ids = {row["pos_id"] for row in cohort_rows}
        selected = [row for row in rows if row["pos_id"] in cohort_ids]
        if not selected:
            continue
        by_cohort[name] = {
            "tested_references": len(selected),
            "before_V_or_U": distribution([row["old_universe"] for row in selected]),
            "after_incidence_classes": distribution([row["quotient_universe"] for row in selected]),
        }
    return {
        "tested": len(rows),
        "caught": len(rows) - len(misses),
        "misses": misses,
        "depth_le_8": len(eligible8),
        "fresh_depth_10_witnesses": new_ids,
        "universe": {
            "before_V_or_U": distribution(universe_old),
            "after_incidence_classes": distribution(universe_classes),
            "after_peak_pair_conditioned_nonshortcut": distribution(universe_peak),
        },
        "first_pairs_fresh": {
            "before_unbounded": distribution(pair_old),
            "after_true_legal": distribution(pair_legal),
        },
        "universe_by_cohort": by_cohort,
        "rows": rows,
    }


@dataclass(frozen=True)
class NextModel:
    mover: int
    cells: tuple[Cell, ...]
    anchored_mask: int
    target_anchored: tuple[int, ...]
    opponent_anchored: tuple[int, ...]
    target_windows: tuple[int, ...]
    opponent_windows: tuple[int, ...]
    near_windows: tuple[int, ...]
    classes: tuple[int, ...]
    first_pairs: tuple[int, ...]


def build_next_model(row: dict, horizon: int) -> NextModel:
    if horizon not in (13, 14):
        raise ValueError(horizon)
    n = len(row["moves"])
    phase, mover, _ = phase_r.phase_player(n)
    if not (phase == "first" or (phase == "second" and horizon == 13)):
        raise ValueError("next-rung model supports fresh h13/h14 and SecondStone h13")
    board = {tuple(cell): phase_r.owner_at(i) for i, cell in enumerate(row["moves"])}
    anchored = h10._root_anchored_windows(board, r2.schedule(n, horizon))
    anchored_cells = set().union(*anchored[0], *anchored[1]) if anchored[0] or anchored[1] else set()
    near = h10._near_empty_windows(board, anchored_cells)
    universe = tuple(sorted(anchored_cells | (set().union(*near) if near else set())))
    index = {cell: i for i, cell in enumerate(universe)}

    def mask(edge: frozenset[Cell]) -> int:
        return sum(1 << index[cell] for cell in edge)

    target_anchored = tuple(mask(edge) for edge in anchored[mover])
    opponent_anchored = tuple(mask(edge) for edge in anchored[1 - mover])
    near_masks = tuple(mask(edge) for edge in near)
    target = target_anchored + near_masks
    opponent = opponent_anchored + near_masks
    anchored_mask = sum(1 << index[cell] for cell in anchored_cells)
    classes = incidence_classes((target, opponent), len(universe))

    # Root-window ancestry at this rung: a useful first pair either contains an
    # anchored action (the other placement may seed a later root-empty line) or
    # puts both placements into one retained root-empty window.
    first: set[int] = set()
    full = (1 << len(universe)) - 1
    legal = legal_cells(board)
    legal_mask = sum(1 << i for i, cell in enumerate(universe) if cell in legal)
    if phase == "second":
        first.update(bits(full & legal_mask))
    elif anchored_mask:
        anchored_bits = list(bits(anchored_mask))
        all_bits = list(bits(full))
        for a in anchored_bits:
            ai = a.bit_length() - 1
            for b in all_bits:
                if a != b and (
                    b & legal_mask
                    or hex_distance(universe[ai], universe[b.bit_length() - 1]) <= LEGAL_RADIUS
                ):
                    first.add(a | b)
    for edge in near_masks:
        for pair in h10._pair_masks(edge):
            if pair & legal_mask:
                first.add(pair)
    if not first:
        first.add(0)

    legal_first = list(first)
    # Ordering only.  Window incidence is precomputed at cell granularity.
    score: dict[int, int] = {}
    for edge in target:
        for bit in bits(edge):
            score[bit] = score.get(bit, 0) + 2
    for edge in opponent:
        for bit in bits(edge):
            score[bit] = score.get(bit, 0) + 1
    legal_first.sort(key=lambda pair: sum(score.get(bit, 0) for bit in bits(pair)), reverse=True)
    return NextModel(
        mover=mover,
        cells=universe,
        anchored_mask=anchored_mask,
        target_anchored=target_anchored,
        opponent_anchored=opponent_anchored,
        target_windows=target,
        opponent_windows=opponent,
        near_windows=near_masks,
        classes=classes,
        first_pairs=tuple(legal_first),
    )


class HarnessTimeout(RuntimeError):
    """Raised by a measurement frame; never converted into a game verdict."""


@dataclass(frozen=True)
class NextDecision:
    win: bool
    h10_shortcut: bool
    horizon: int
    nodes: int
    first_pairs: int
    defender1_pairs: int
    attacker2_pairs: int
    defender2_pairs: int
    attacker3_pairs: int
    physical_universe: int
    quotient_universe: int
    wall_ns: int
    witness_first_pair: tuple[Cell, ...] | None


def decide_fresh_next(
    row: dict,
    horizon: int,
    *,
    harness_deadline_ns: int | None = None,
    use_h10_shortcut: bool = True,
    preferred_first_pair: tuple[Cell, Cell] | None = None,
) -> NextDecision:
    """H13/H14 interaction endpoint; timeout raises instead of answering.

    Exactness relies on the R3 interaction-normalization lemma documented in
    REPORT_HORIZON_R3.md.  All integer masks are arbitrary precision.
    """
    if horizon not in (13, 14):
        raise ValueError(horizon)
    started = time.perf_counter_ns()

    phase, mover, _ = phase_r.phase_player(len(row["moves"]))
    if use_h10_shortcut:
        old = decide_fresh_h10_r3(row) if phase == "first" else r2.decide_ladder(row, 10, mover)
    else:
        old = None
    if old is not None and old.win:
        old_nodes = old.nodes
        old_physical = old.old_universe if isinstance(old, H10R3Decision) else old.universe
        old_quotient = old.quotient_universe if isinstance(old, H10R3Decision) else old.universe
        old_witness = old.witness if isinstance(old, H10R3Decision) else None
        return NextDecision(
            True, True, horizon, old_nodes, 0, 0, 0, 0, 0,
            old_physical, old_quotient,
            time.perf_counter_ns() - started, old_witness,
        )

    model = build_next_model(row, horizon)
    final_capacity = 1 if phase == "first" and horizon == 13 else 2
    after_a1 = 5 if phase == "first" and horizon == 13 else 6
    after_a2 = 3 if phase == "first" and horizon == 13 else 4
    after_a3 = final_capacity
    nodes = n_a1 = n_d1 = n_a2 = n_d2 = n_a3 = 0

    def tick() -> None:
        nonlocal nodes
        nodes += 1
        if harness_deadline_ns is not None and (nodes & 1023) == 0:
            if time.perf_counter_ns() >= harness_deadline_ns:
                raise HarnessTimeout({
                    "nodes": nodes,
                    "first_pairs": n_a1,
                    "defender1_pairs": n_d1,
                    "attacker2_pairs": n_a2,
                    "defender2_pairs": n_d2,
                    "attacker3_pairs": n_a3,
                    "physical_universe": len(model.cells),
                    "quotient_universe": len(model.classes),
                    "wall_ns": time.perf_counter_ns() - started,
                })

    target0 = model.target_windows
    opponent0 = model.opponent_windows
    first_pairs = list(model.first_pairs)
    if preferred_first_pair is not None:
        index = {cell: i for i, cell in enumerate(model.cells)}
        if all(cell in index for cell in preferred_first_pair):
            preferred_mask = sum(1 << index[cell] for cell in preferred_first_pair)
            if preferred_mask in first_pairs:
                first_pairs.remove(preferred_mask)
                first_pairs.insert(0, preferred_mask)
    for a1 in first_pairs:
        tick(); n_a1 += 1
        if any(not edge & ~a1 for edge in target0):
            return NextDecision(
                True, False, horizon, nodes, n_a1, n_d1, n_a2, n_d2, n_a3,
                len(model.cells), len(model.classes), time.perf_counter_ns() - started,
                decode_pair(a1, model.cells),
            )
        # An untouched root-empty window would consume all six remaining A
        # placements.  The six-stone two-cover lemma makes that branch inert;
        # retain only near windows seeded by a1.  Root-anchored windows retain
        # their ordinary remaining-quota threshold.
        live_a1 = tuple(
            edge & ~a1 for edge in model.target_anchored
            if (edge & ~a1).bit_count() <= after_a1
        ) + tuple(
            edge & ~a1 for edge in model.near_windows
            if edge & a1 and (edge & ~a1).bit_count() <= after_a1
        )
        live_d1 = tuple(edge for edge in opponent0 if not edge & a1)
        active_d1 = 0
        for edge in live_a1 + live_d1:
            active_d1 |= edge
        a1_wins = True
        for d1 in h10._ordered_pair_iter(active_d1, live_d1, live_a1):
            tick(); n_d1 += 1
            if any(not edge & ~d1 for edge in live_d1):
                a1_wins = False
                break
            live_a_d1 = tuple(edge for edge in live_a1 if not edge & d1)
            live_d_d1 = tuple(edge & ~d1 for edge in live_d1)
            active_a2 = 0
            for edge in live_a_d1:
                if edge.bit_count() <= after_a1:
                    active_a2 |= edge
            for edge in live_d_d1:
                if edge.bit_count() <= 4:
                    active_a2 |= edge
            d1_answered = False
            for a2 in h10._ordered_pair_iter(active_a2, live_a_d1, live_d_d1):
                tick(); n_a2 += 1
                if any(not edge & ~a2 for edge in live_a_d1):
                    d1_answered = True
                    break
                live_a2 = tuple(
                    edge & ~a2 for edge in live_a_d1
                    if (edge & ~a2).bit_count() <= after_a2
                )
                live_d2 = tuple(
                    edge for edge in live_d_d1
                    if not edge & a2 and edge.bit_count() <= 4
                )
                active_d2 = 0
                for edge in live_a2 + live_d2:
                    active_d2 |= edge
                a2_wins = True
                for d2 in h10._ordered_pair_iter(active_d2, live_d2, live_a2):
                    tick(); n_d2 += 1
                    if any(not edge & ~d2 for edge in live_d2):
                        a2_wins = False
                        break
                    live_a_d2 = tuple(edge for edge in live_a2 if not edge & d2)
                    live_d_d2 = tuple(edge & ~d2 for edge in live_d2)
                    active_a3 = 0
                    for edge in live_a_d2:
                        if edge.bit_count() <= after_a2:
                            active_a3 |= edge
                    for edge in live_d_d2:
                        if edge.bit_count() <= 2:
                            active_a3 |= edge
                    d2_answered = False
                    d_threats = [edge for edge in live_d_d2 if edge.bit_count() <= 2]
                    if d_threats:
                        a3_actions = covering_pair_actions(d_threats, active_a3)
                    else:
                        a3_actions = threat_making_pair_actions(live_a_d2, active_a3)
                    dynamic_classes = incidence_classes(
                        (live_a_d2, live_d_d2), len(model.cells)
                    )
                    a3_actions = quotient_action_list(a3_actions, dynamic_classes, active_a3)
                    # Ordering only: completion/progress before pure covers.
                    a3_actions.sort(
                        key=lambda pair: (
                            int(any(not edge & ~pair for edge in live_a_d2)),
                            sum((edge & pair).bit_count() for edge in live_a_d2),
                        ),
                        reverse=True,
                    )
                    for a3 in a3_actions:
                        tick(); n_a3 += 1
                        if any(not edge & ~a3 for edge in live_a_d2):
                            d2_answered = True
                            break
                        if any(
                            not edge & a3 and (edge & ~a3).bit_count() <= 2
                            for edge in live_d_d2
                        ):
                            continue
                        threats = [
                            edge & ~a3 for edge in live_a_d2
                            if (edge & ~a3).bit_count() <= after_a3
                        ]
                        if not threats:
                            continue
                        if final_capacity == 1:
                            singleton_cells = 0
                            for edge in threats:
                                if edge.bit_count() == 1:
                                    singleton_cells |= edge
                            if singleton_cells.bit_count() > 2:
                                d2_answered = True
                                break
                        elif h10._cover_two_witness(threats) is None:
                            d2_answered = True
                            break
                    if not d2_answered:
                        a2_wins = False
                        break
                if a2_wins:
                    d1_answered = True
                    break
            if not d1_answered:
                a1_wins = False
                break
        if a1_wins:
            return NextDecision(
                True, False, horizon, nodes, n_a1, n_d1, n_a2, n_d2, n_a3,
                len(model.cells), len(model.classes), time.perf_counter_ns() - started,
                decode_pair(a1, model.cells),
            )
    return NextDecision(
        False, False, horizon, nodes, n_a1, n_d1, n_a2, n_d2, n_a3,
        len(model.cells), len(model.classes), time.perf_counter_ns() - started, None,
    )


def measure_next_rung(per_root_ms: int = 250) -> dict:
    known, all_rows = h10._known_registry()
    new_eligible = sorted(
        pos_id for pos_id, info in known.items()
        if info["cert_depth"] in (13, 14)
    )
    atlas_by_id = {
        f"atlas_full_{row['id']}": row
        for row in json.loads(phase_r.DEFAULT_ATLAS.read_text(encoding="utf-8"))["rows"]
    }
    attempts = []
    caught = []
    completed_negative = []
    for i, pos_id in enumerate(new_eligible, 1):
        row = all_rows[pos_id]
        horizon = int(known[pos_id]["cert_depth"])
        phase = phase_r.phase_player(len(row["moves"]))[0]
        model_started = time.perf_counter_ns()
        model = build_next_model(row, horizon)
        model_ns = time.perf_counter_ns() - model_started
        preferred = None
        atlas = atlas_by_id.get(pos_id)
        if atlas and len(atlas.get("win_line", ())) >= 2 and phase == "first":
            preferred = tuple(tuple(cell) for cell in atlas["win_line"][:2])
        deadline = time.perf_counter_ns() + per_root_ms * 1_000_000
        try:
            decision = decide_fresh_next(
                row,
                horizon,
                harness_deadline_ns=deadline,
                use_h10_shortcut=False,
                preferred_first_pair=preferred,
            )
            status = "caught" if decision.win else "completed_negative_mismatch"
            detail = asdict(decision)
            if decision.win:
                caught.append(pos_id)
            else:
                completed_negative.append(pos_id)
        except HarnessTimeout as exc:
            status = "timeout"
            detail = exc.args[0]
        attempts.append({
            "pos_id": pos_id,
            "cert_depth": horizon,
            "phase": phase,
            "status": status,
            "model_wall_ns": model_ns,
            "physical_universe": len(model.cells),
            "quotient_universe": len(model.classes),
            "normalized_first_actions": len(model.first_pairs),
            **detail,
        })
        if i % 25 == 0:
            print(f"next-rung certificates: {i}/{len(new_eligible)}", flush=True)

    cohorts = r2.cohorts()
    r2_measure = json.loads((ROOT / ".scratch" / "horizon_r2.json").read_text(encoding="utf-8"))
    production = json.loads((ROOT / ".scratch" / "horizon_production_h8.json").read_text(encoding="utf-8"))
    floors = {}
    for name, rows in cohorts.items():
        ids = {row["pos_id"] for row in rows}
        fresh_n = sum(phase_r.phase_player(len(row["moves"]))[0] == "first" for row in rows)
        if name in production["cohorts"]:
            h8_ids = set(production["cohorts"][name]["current_win_ids"])
        else:
            h8_ids = set(r2_measure["cohorts"][name]["horizons"]["8"]["current_win"]["win_ids"])
        certified10 = {
            pos_id for pos_id, info in known.items()
            if pos_id in ids and info["cert_depth"] is not None and info["cert_depth"] <= 10
        }
        certified14 = {
            pos_id for pos_id, info in known.items()
            if pos_id in ids and info["cert_depth"] is not None and info["cert_depth"] <= 14
        }
        floors[name] = {
            "rows": len(rows),
            "fresh_rows": fresh_n,
            "h8_exact_wins": len(h8_ids),
            "h10_certified_floor": len(h8_ids | certified10),
            "h14_certified_floor": len(h8_ids | certified14),
            "h14_delta_floor_over_h10": len((h8_ids | certified14) - (h8_ids | certified10)),
        }

    depth_counts = Counter(
        info["cert_depth"] for info in known.values() if info["cert_depth"] is not None
    )
    return {
        "metadata": {
            "exact_decider": True,
            "timeout_is_not_verdict": True,
            "per_new_certificate_harness_ms": per_root_ms,
        },
        "registry": {
            "unique": len(known),
            "depth_counts": {str(k): v for k, v in sorted(depth_counts.items())},
            "eligible_le_10": sum(v["cert_depth"] is not None and v["cert_depth"] <= 10 for v in known.values()),
            "new_depth_13_14": len(new_eligible),
            "new_caught": len(caught),
            "new_timeouts": sum(row["status"] == "timeout" for row in attempts),
            "completed_negative_mismatches": completed_negative,
        },
        "new_certificate_universe": {
            "before_physical_V": distribution([row["physical_universe"] for row in attempts]),
            "after_incidence_classes": distribution([row["quotient_universe"] for row in attempts]),
            "normalized_first_actions": distribution([row["normalized_first_actions"] for row in attempts]),
        },
        "cohort_bite_floors": floors,
        "attempts": attempts,
    }


def measure_legality_bridge() -> dict:
    h8_rows = [
        row
        for name, rows in production_h8.all_cohorts().items()
        if name != "grinds"
        for row in rows
    ]
    checked_cells = 0
    max_distance = 0
    violations = []
    for row in h8_rows:
        phase, mover, _ = phase_r.phase_player(len(row["moves"]))
        if phase == "opening":
            continue
        for target in (mover, 1 - mover):
            model = build_qmodel(row, 8, target)
            stones = [tuple(cell) for cell in row["moves"]]
            for cell in model.cells:
                distance = min(hex_distance(cell, stone) for stone in stones)
                checked_cells += 1
                max_distance = max(max_distance, distance)
                if distance > 5:
                    violations.append({"pos_id": row["pos_id"], "target": target, "cell": cell, "distance": distance})

    known, all_rows = h10._known_registry()
    tested_ids = sorted(
        pos_id for pos_id, info in known.items()
        if info["cert_depth"] is not None and info["cert_depth"] <= 8
    ) + ["human_b132a09ccb4eb829_p101", "sp_20_p77"]
    h10_anchored_cells = 0
    h10_max_distance = 0
    h10_violations = []
    for pos_id in tested_ids:
        row = all_rows[pos_id]
        if phase_r.phase_player(len(row["moves"]))[0] != "first":
            continue
        model = h10.build_model(row)
        stones = [tuple(cell) for cell in row["moves"]]
        for bit in bits(model.anchored_mask):
            cell = model.cells[bit.bit_length() - 1]
            distance = min(hex_distance(cell, stone) for stone in stones)
            h10_anchored_cells += 1
            h10_max_distance = max(h10_max_distance, distance)
            if distance > 5:
                h10_violations.append({"pos_id": pos_id, "cell": cell, "distance": distance})
    return {
        "engine_code_facts": {
            "hexfield_eq_constant": "packages/hexfield_eq/rust/src/constants.rs:5",
            "engine_write_legal_moves": "packages/hexo_engine/rust/src/state.rs:216-231",
            "engine_radius_update": "packages/hexo_engine/rust/src/legal.rs:135-138",
            "support_filter": "packages/hexfield_eq/rust/src/support.rs:106-108",
            "legal_radius": LEGAL_RADIUS,
        },
        "h8_full_battery_anchored_cells": {
            "unique_rows": len(h8_rows),
            "cell_references_checked": checked_cells,
            "max_min_distance_to_root_stone": max_distance,
            "violations": violations,
        },
        "h10_tested_fresh_anchored_cells": {
            "tested_ids_all_phases": len(tested_ids),
            "cell_references_checked": h10_anchored_cells,
            "max_min_distance_to_root_stone": h10_max_distance,
            "violations": h10_violations,
        },
    }


def measure_next_boundaries(seconds: int = 5) -> dict:
    known, rows = h10._known_registry()
    cases = [
        ("human_41e78c67c2ac8570_p20", 13, None),
        ("atlas_full_oa-c515cddcef6134b3", 14, ((2, 0), (2, 1))),
        ("sp_0_p51", 14, None),
    ]
    out = []
    for pos_id, horizon, preferred in cases:
        started = time.perf_counter_ns()
        try:
            decision = decide_fresh_next(
                rows[pos_id],
                horizon,
                harness_deadline_ns=started + seconds * 1_000_000_000,
                use_h10_shortcut=False,
                preferred_first_pair=preferred,
            )
            status = "win" if decision.win else "completed_negative_mismatch"
            detail = asdict(decision)
        except HarnessTimeout as exc:
            status = "timeout"
            detail = exc.args[0]
        out.append({
            "pos_id": pos_id,
            "cert_depth": known[pos_id]["cert_depth"],
            "phase": phase_r.phase_player(len(rows[pos_id]["moves"]))[0],
            "status": status,
            **detail,
        })
        print(f"next boundary {pos_id}: {status}", flush=True)
    return {
        "per_root_seconds": seconds,
        "timeout_is_not_verdict": True,
        "rows": out,
    }


def six_stone_two_cover() -> dict:
    """Exhaust the finite one-axis remainder of the <=6-stone lemma.

    A base threat is normalized to [0,5].  Any extra stone which participates
    in another threat is within five of one of its base stones, hence lies in
    [-5,10].  Translation and reflection are quotiented.
    """
    seen: set[tuple[int, ...]] = set()
    failures = []
    covers = Counter()
    examples: dict[int, dict] = {}
    carrier = tuple(range(-5, 11))
    base = frozenset(range(6))
    for size in range(4, 7):
        for raw in combinations(carrier, size):
            stones = frozenset(raw)
            if len(stones & base) < 4:
                continue
            lo, hi = min(stones), max(stones)
            normalized = tuple(x - lo for x in sorted(stones))
            reflected = tuple(hi - x for x in sorted(stones, reverse=True))
            key = min(normalized, reflected)
            if key in seen:
                continue
            seen.add(key)
            family = []
            for start in range(lo - 5, hi + 1):
                window = frozenset(range(start, start + 6))
                residual = window - stones
                if len(window & stones) >= 4 and residual:
                    family.append(residual)
            cells = sorted(set().union(*family)) if family else []
            index = {cell: i for i, cell in enumerate(cells)}
            masks = [sum(1 << index[x] for x in edge) for edge in family]
            witness = h10._cover_two_witness(masks)
            if witness is None:
                failures.append({"stones": sorted(stones), "residuals": [sorted(x) for x in family]})
                continue
            cover = [cells[i] for i in range(len(cells)) if witness & (1 << i)]
            covers[len(cover)] += 1
            examples.setdefault(len(cover), {"stones": sorted(stones), "cover": cover})
    return {
        "canonical_shapes": len(seen),
        "failures": failures,
        "all_have_two_cover": not failures,
        "cover_size_counts": dict(sorted(covers.items())),
        "examples": examples,
    }


def rung_table() -> list[dict]:
    representatives = {"opening": 0, "FirstStone": 1, "SecondStone": 2}
    rows = []
    for phase, n in representatives.items():
        attacker = phase_r.phase_player(n)[1]
        previous = None
        for horizon in range(1, 17):
            sched = r2.schedule(n, horizon)
            quotas = (sched.count(attacker), sched.count(1 - attacker))
            changed = previous is None or quotas[0] != previous[0]
            rows.append({
                "phase": phase,
                "horizon": horizon,
                "schedule": "".join("A" if p == attacker else "D" for p in sched),
                "attacker_placements": quotas[0],
                "defender_placements": quotas[1],
                "attacker_quota_changed": changed,
            })
            previous = quotas
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / ".scratch" / "horizon_r3.json")
    ap.add_argument("--h8-battery", action="store_true")
    ap.add_argument("--h10-cohort", action="store_true")
    ap.add_argument("--next-rung", action="store_true")
    ap.add_argument("--next-root-ms", type=int, default=250)
    ap.add_argument("--legality-audit", action="store_true")
    ap.add_argument("--next-boundaries", action="store_true")
    ap.add_argument("--boundary-seconds", type=int, default=5)
    ap.add_argument("--lemmas-only", action="store_true")
    args = ap.parse_args()
    result: dict = {
        "metadata": {
            "python": platform.python_version(),
            "head": "e118097075a2f46afcb30f8c38b0c2c98666eab0",
            "legal_radius": LEGAL_RADIUS,
        },
        "six_stone_two_cover": six_stone_two_cover(),
        "rung_table": rung_table(),
    }
    if args.h8_battery:
        result["h8_battery"] = measure_h8_battery()
    if args.h10_cohort:
        result["h10_validation"] = validate_h10_cohort()
    if args.next_rung:
        result["next_rung"] = measure_next_rung(args.next_root_ms)
    if args.legality_audit:
        result["legality_bridge"] = measure_legality_bridge()
    if args.next_boundaries:
        result["next_boundaries"] = measure_next_boundaries(args.boundary_seconds)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "six_stone_shapes": result["six_stone_two_cover"]["canonical_shapes"],
        "six_stone_failures": len(result["six_stone_two_cover"]["failures"]),
        "h8_mismatches": result.get("h8_battery", {}).get("validation", {}).get("mismatches"),
        "h10_caught": result.get("h10_validation", {}).get("caught"),
        "next_new_caught": result.get("next_rung", {}).get("registry", {}).get("new_caught"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
