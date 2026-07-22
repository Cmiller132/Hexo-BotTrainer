#!/usr/bin/env python3
"""Exact fresh-turn Connect6 attacker decider through h=10.

This builds alongside ``horizon_r2.py``.  The h=10 frontier is made finite by
retaining root-anchored windows plus root-empty windows which meet their finite
cell universe.  A wholly remote empty-root attack is represented by a constant
subgame; exhaustive one-dimensional shape enumeration proves that its final
threat family always has a two-cell cover.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".scratch"))
import deadline_ladder_r as phase_r  # noqa: E402
import horizon_r2 as r2  # noqa: E402


Cell = tuple[int, int]


@dataclass(frozen=True)
class H10Decision:
    win: bool
    h8_shortcut: bool
    nodes: int
    first_pairs: int
    defender_pairs: int
    second_pairs: int
    universe: int
    anchored_cells: int
    anchored_target_windows: int
    near_empty_windows: int
    opponent_windows: int
    wall_ns: int
    witness_first_pair: tuple[Cell, ...] | None


@dataclass(frozen=True)
class H10Model:
    mover: int
    cells: tuple[Cell, ...]
    anchored_mask: int
    target_windows: tuple[int, ...]
    near_windows: tuple[int, ...]
    opponent_windows: tuple[int, ...]
    first_pairs: tuple[int, ...]
    near_by_first_pair: dict[int, tuple[int, ...]]


def _bits(mask: int) -> Iterable[int]:
    while mask:
        bit = mask & -mask
        mask ^= bit
        yield bit


def _pair_masks(active: int) -> Iterable[int]:
    bits = list(_bits(active))
    if not bits:
        yield 0
    elif len(bits) == 1:
        yield bits[0]
    else:
        for a, b in combinations(bits, 2):
            yield a | b


def _ordered_pair_iter(active: int, own: tuple[int, ...], other: tuple[int, ...]) -> Iterable[int]:
    """Tactical prefix followed by every remaining normalized pair.

    Building and scoring the complete quadratic pair list dominated the first
    prototype even when the first defender reply refuted an attack.  This
    iterator scores individual cells, tries only a small exactness-neutral
    prefix, and then falls back to the exhaustive pair stream.
    """
    bits = list(_bits(active))
    if not bits:
        yield 0
        return
    if len(bits) == 1:
        yield bits[0]
        return

    seen: set[int] = set()

    def emit(pair: int) -> Iterable[int]:
        if pair not in seen:
            seen.add(pair)
            yield pair

    # A completion requiring one cell terminates before the second placement;
    # pair it with the strongest other active cell.  Two-cell completions are
    # already normalized pairs.
    cell_scores = {bit: 0 for bit in bits}
    for edge in own:
        for bit in _bits(edge & active):
            cell_scores[bit] += 1
    for edge in other:
        for bit in _bits(edge & active):
            cell_scores[bit] += 2
    ranked = sorted(bits, key=lambda bit: cell_scores[bit], reverse=True)
    for edge in own:
        rem = edge & active
        if rem.bit_count() == 2:
            yield from emit(rem)
        elif rem.bit_count() == 1:
            mate = next(bit for bit in ranked if bit != rem)
            yield from emit(rem | mate)

    # The top twelve cells give at most 66 high-impact pairs.  Negative roots
    # normally find their refuting D reply here.
    for a, b in combinations(ranked[:12], 2):
        yield from emit(a | b)
    for a, b in combinations(bits, 2):
        yield from emit(a | b)


def _cover_two_witness(family: list[int]) -> int | None:
    """Return a size-at-most-two hitting mask, or None."""
    if not family:
        return 0
    common = family[0]
    for edge in family[1:]:
        common &= edge
    if common:
        return common & -common
    for x in _bits(family[0]):
        rest = [edge for edge in family if not edge & x]
        if not rest:
            return x
        common = rest[0]
        for edge in rest[1:]:
            common &= edge
        if common:
            return x | (common & -common)
    return None


def remote_empty_constant() -> dict:
    """Exhaust every four-stone line shape relevant before the final A pair."""
    rows = []
    for tail in combinations(range(1, 6), 3):
        stones = frozenset((0, *tail))
        lo, hi = min(stones), max(stones)
        windows = []
        for start in range(hi - 5, lo + 1):
            window = frozenset(range(start, start + 6))
            if stones <= window:
                windows.append(window - stones)
        cells = sorted(set().union(*windows))
        index = {x: i for i, x in enumerate(cells)}
        masks = [sum(1 << index[x] for x in edge) for edge in windows]
        witness = _cover_two_witness(masks)
        rows.append({
            "stones": sorted(stones),
            "residuals": [sorted(edge) for edge in windows],
            "cover": [cells[i] for i in range(len(cells)) if witness is not None and witness & (1 << i)],
            "covered": witness is not None,
        })
    return {
        "constant": False,
        "shapes": len(rows),
        "all_have_two_cover": all(row["covered"] for row in rows),
        "rows": rows,
    }


def _root_anchored_windows(board: dict[Cell, int], sched: tuple[int, ...]) -> dict[int, list[frozenset[Cell]]]:
    quotas = Counter(sched)
    out: dict[int, list[frozenset[Cell]]] = {0: [], 1: []}
    seen: dict[int, set[frozenset[Cell]]] = {0: set(), 1: set()}
    for entry in phase_r.entries(board):
        owner = entry["owner"]
        empty = entry["empty"]
        # ``len(empty) < 6`` is the root-anchor: entries are pure windows, so
        # this says that the window contains at least one root owner stone.
        if len(empty) < 6 and len(empty) <= quotas[owner] and empty not in seen[owner]:
            seen[owner].add(empty)
            out[owner].append(empty)
    return out


def _near_empty_windows(board: dict[Cell, int], anchored_cells: set[Cell]) -> list[frozenset[Cell]]:
    """Root-empty windows meeting the finite root-anchored cell universe."""
    out: set[frozenset[Cell]] = set()
    for x, y in anchored_cells:
        for dx, dy in phase_r.AXES:
            for offset in range(6):
                window = frozenset(
                    (x + (i - offset) * dx, y + (i - offset) * dy)
                    for i in range(6)
                )
                if all(cell not in board for cell in window):
                    out.add(window)
    return sorted(out, key=lambda edge: tuple(sorted(edge)))


def build_model(row: dict) -> H10Model:
    n = len(row["moves"])
    phase, mover, _ = phase_r.phase_player(n)
    if phase != "first":
        raise ValueError("the translation quotient is needed only at a fresh FirstStone root")
    sched = r2.schedule(n, 10)
    board = {tuple(cell): phase_r.owner_at(i) for i, cell in enumerate(row["moves"])}
    anchored = _root_anchored_windows(board, sched)
    anchored_cells = set().union(*anchored[0], *anchored[1]) if anchored[0] or anchored[1] else set()
    near = _near_empty_windows(board, anchored_cells)
    universe = sorted(anchored_cells | (set().union(*near) if near else set()))
    index = {cell: i for i, cell in enumerate(universe)}

    def mask(edge: frozenset[Cell]) -> int:
        return sum(1 << index[cell] for cell in edge)

    target_anchored = tuple(mask(edge) for edge in anchored[mover])
    near_masks = tuple(mask(edge) for edge in near)
    opponent = tuple(mask(edge) for edge in anchored[1 - mover])
    anchored_mask = sum(1 << index[cell] for cell in anchored_cells)

    # A useful first pair is either wholly in the anchored universe or has
    # both cells in one near-empty window.  Otherwise every outside cell is in
    # a six-empty window still missing at least five stones, while it affects
    # no anchored window; replacing it by an anchored action is monotone.
    first: set[int] = set()
    near_by_pair_lists: dict[int, list[int]] = {}
    for pair in _pair_masks(anchored_mask):
        first.add(pair)
    for edge in near_masks:
        for pair in _pair_masks(edge):
            first.add(pair)
            near_by_pair_lists.setdefault(pair, []).append(edge)

    first_pairs = list(first)
    # Rank the complete root-pair set without rescanning every window per
    # pair.  This is ordering only: all pairs remain in the tuple.
    cell_score: dict[int, int] = {}
    for edge in target_anchored:
        for bit in _bits(edge):
            cell_score[bit] = cell_score.get(bit, 0) + 4
    for edge in opponent:
        for bit in _bits(edge):
            cell_score[bit] = cell_score.get(bit, 0) + 3
    for edge in near_masks:
        for bit in _bits(edge):
            cell_score[bit] = cell_score.get(bit, 0) + 1
    first_pairs.sort(
        key=lambda pair: (
            sum(cell_score.get(bit, 0) for bit in _bits(pair)),
            len(near_by_pair_lists.get(pair, ())),
            (pair & anchored_mask).bit_count(),
        ),
        reverse=True,
    )
    return H10Model(
        mover=mover,
        cells=tuple(universe),
        anchored_mask=anchored_mask,
        target_windows=target_anchored,
        near_windows=near_masks,
        opponent_windows=opponent,
        first_pairs=tuple(first_pairs),
        near_by_first_pair={pair: tuple(edges) for pair, edges in near_by_pair_lists.items()},
    )


def _decode_pair(pair: int, cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    return tuple(cells[i] for i in range(len(cells)) if pair & (1 << i))


def decide_fresh_h10(row: dict, *, h8_shortcut: bool = True) -> H10Decision:
    """Decide an exact current-mover win within ten physical placements."""
    started = time.perf_counter_ns()
    if h8_shortcut:
        old = r2.decide_fresh_current(row, 8)
        if old.win:
            return H10Decision(
                True, True, old.nodes, 0, 0, 0, old.universe, old.universe,
                old.target_windows, 0, old.opponent_windows,
                time.perf_counter_ns() - started, None,
            )

    model = build_model(row)
    target = model.target_windows + model.near_windows
    opponent = model.opponent_windows
    nodes = first_nodes = defender_nodes = second_nodes = 0

    for a_pair in model.first_pairs:
        nodes += 1
        first_nodes += 1
        if any(not (edge & ~a_pair) for edge in target):
            return H10Decision(
                True, False, nodes, first_nodes, defender_nodes, second_nodes,
                len(model.cells), model.anchored_mask.bit_count(),
                len(model.target_windows), len(model.near_windows), len(opponent),
                time.perf_counter_ns() - started, _decode_pair(a_pair, model.cells),
            )

        # D's first pair need only touch a D window or an A window that A can
        # still complete with its four remaining placements.
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
        first_pair_wins = True
        for d_pair in _ordered_pair_iter(active_d, d_live, a_future):
            nodes += 1
            defender_nodes += 1
            if any(not (edge & ~d_pair) for edge in d_live):
                first_pair_wins = False
                break

            live_a = tuple(edge for edge in a_future if not edge & d_pair)
            live_d = tuple(edge & ~d_pair for edge in d_live)
            # A's second pair must advance a still-completable A window and/or
            # suppress a D completion on the last defender pair.
            active_b = 0
            for edge in live_a:
                active_b |= edge
            for edge in live_d:
                if edge.bit_count() <= 2:
                    active_b |= edge
            reply_wins = False
            for b_pair in _ordered_pair_iter(active_b, live_a, live_d):
                nodes += 1
                second_nodes += 1
                if any(not (edge & ~b_pair) for edge in live_a):
                    reply_wins = True
                    break
                d_can_win = any(
                    not edge & b_pair and (edge & ~b_pair).bit_count() <= 2
                    for edge in live_d
                )
                if d_can_win:
                    continue
                threats = [edge & ~b_pair for edge in live_a if (edge & ~b_pair).bit_count() <= 2]
                if threats and _cover_two_witness(threats) is None:
                    reply_wins = True
                    break
            if not reply_wins:
                first_pair_wins = False
                break

        if first_pair_wins:
            return H10Decision(
                True, False, nodes, first_nodes, defender_nodes, second_nodes,
                len(model.cells), model.anchored_mask.bit_count(),
                len(model.target_windows), len(model.near_windows), len(opponent),
                time.perf_counter_ns() - started, _decode_pair(a_pair, model.cells),
            )

    return H10Decision(
        False, False, nodes, first_nodes, defender_nodes, second_nodes,
        len(model.cells), model.anchored_mask.bit_count(),
        len(model.target_windows), len(model.near_windows), len(opponent),
        time.perf_counter_ns() - started, None,
    )


def decide_h10(row: dict) -> H10Decision | r2.Decision:
    """Exact h=10 entry point; non-fresh roots remain inside R2 relevance."""
    phase = phase_r.phase_player(len(row["moves"]))[0]
    if phase == "first":
        return decide_fresh_h10(row)
    return r2.decide(row, 10)


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def _summary(decisions: list[H10Decision]) -> dict:
    walls = [d.wall_ns for d in decisions]
    nodes = [d.nodes for d in decisions]
    wins = sum(d.win for d in decisions)
    return {
        "n": len(decisions),
        "wins": wins,
        "win_pct": _pct(wins, len(decisions)),
        "h8_shortcuts": sum(d.h8_shortcut for d in decisions),
        "nodes": {
            "total": sum(nodes), "p50": _percentile(nodes, .5),
            "p90": _percentile(nodes, .9), "max": max(nodes),
        },
        "wall_us": {
            "total": sum(walls) / 1e3, "mean": sum(walls) / len(walls) / 1e3,
            "p50": _percentile(walls, .5) / 1e3,
            "p90": _percentile(walls, .9) / 1e3, "max": max(walls) / 1e3,
        },
        "universe": {
            "p50": _percentile([d.universe for d in decisions], .5),
            "p90": _percentile([d.universe for d in decisions], .9),
            "max": max(d.universe for d in decisions),
        },
    }


def _known_registry() -> tuple[dict[str, dict], dict[str, dict]]:
    cohorts = r2.cohorts()
    all_rows = {row["pos_id"]: row for rows in cohorts.values() for row in rows}
    walls = phase_r.wall_sources()
    known: dict[str, dict] = {}

    def add(pos_id: str, source: str, depth: int | None = None) -> None:
        depth = depth if depth and depth > 0 else None
        prior = known.get(pos_id)
        if prior is None or (prior["cert_depth"] is None and depth is not None):
            known[pos_id] = {"source": source, "cert_depth": depth}

    for row in cohorts["puzzle_v3"]:
        if row.get("labels", {}).get("verdict") == "win":
            add(row["pos_id"], "puzzle_v3 labeled WIN")
    for row in cohorts["forcing19"]:
        if row.get("labels", {}).get("verdict") == "win":
            add(row["pos_id"], "forcing19 expected WIN")
    for cohort_name, wall in walls.items():
        for pos_id, record in wall.items():
            if record["status"] == "win" and pos_id in all_rows:
                add(pos_id, f"{cohort_name} measured WIN", record.get("cert_depth"))

    atlas_rows = json.loads(phase_r.DEFAULT_ATLAS.read_text(encoding="utf-8"))["rows"]
    for atlas in atlas_rows:
        if atlas.get("status") != "WIN" or not atlas.get("certified"):
            continue
        pos_id = f"atlas_full_{atlas['id']}"
        all_rows[pos_id] = {"pos_id": pos_id, "moves": atlas["moves"]}
        derived = atlas.get("derived_horizon")
        depth = derived - atlas["placements"] if isinstance(derived, int) else None
        add(pos_id, "opening atlas certified WIN", depth)
    return known, all_rows


def measure(sets: list[str], limit: int | None = None) -> dict:
    cohorts = r2.cohorts()
    out: dict = {
        "metadata": {"exact": True, "horizons": [8, 10], "scope": "fresh FirstStone roots"},
        "remote_empty_subgame": remote_empty_constant(),
        "cohorts": {},
    }
    for name in sets:
        rows = [row for row in cohorts[name] if phase_r.phase_player(len(row["moves"]))[0] == "first"]
        if limit is not None:
            rows = rows[:limit]
        decisions: list[H10Decision] = []
        h8_ids = []
        h10_ids = []
        details = []
        for i, row in enumerate(rows, 1):
            old = r2.decide_fresh_current(row, 8)
            decision = decide_fresh_h10(row)
            decisions.append(decision)
            if old.win:
                h8_ids.append(row["pos_id"])
            if decision.win:
                h10_ids.append(row["pos_id"])
            details.append({"pos_id": row["pos_id"], "h8": old.win, **asdict(decision)})
            if i % 100 == 0:
                print(f"{name}: {i}/{len(rows)}", flush=True)
        violations = sorted(set(h8_ids) - set(h10_ids))
        out["cohorts"][name] = {
            "n": len(rows), "h8_wins": len(h8_ids), "h10_wins": len(h10_ids),
            "delta": len(h10_ids) - len(h8_ids),
            "h8_pct": _pct(len(h8_ids), len(rows)), "h10_pct": _pct(len(h10_ids), len(rows)),
            "h8_not_h10": violations, "h8_win_ids": h8_ids, "h10_win_ids": h10_ids,
            "h10_summary": _summary(decisions), "rows": details,
        }
    return out


def _even_sample(rows: list[dict], size: int | None) -> list[dict]:
    if size is None or size >= len(rows):
        return rows
    if size <= 1:
        return rows[:size]
    indexes = [round(i * (len(rows) - 1) / (size - 1)) for i in range(size)]
    return [rows[i] for i in indexes]


def measure_sampled(sets: list[str], sample_size: int) -> dict:
    """Deterministic cohort frame plus an all-root h8=>h10 nesting audit."""
    cohorts = r2.cohorts()
    out: dict = {
        "metadata": {
            "exact": True, "horizons": [8, 10], "scope": "fresh FirstStone roots",
            "measurement_frame": f"evenly spaced deterministic sample, at most {sample_size} roots per cohort",
        },
        "remote_empty_subgame": remote_empty_constant(),
        "cohorts": {},
    }
    for name in sets:
        population = [row for row in cohorts[name] if phase_r.phase_player(len(row["moves"]))[0] == "first"]
        selected = _even_sample(population, sample_size)
        h8_population_ids = []
        nesting_violations = []
        for row in population:
            old = r2.decide_fresh_current(row, 8)
            if old.win:
                h8_population_ids.append(row["pos_id"])
                # The h10 entry point deliberately returns this exact positive
                # result before entering the new quotient search.
                if not decide_fresh_h10(row).win:
                    nesting_violations.append(row["pos_id"])

        decisions = []
        h8_ids = []
        h10_ids = []
        details = []
        for row in selected:
            old = r2.decide_fresh_current(row, 8)
            decision = decide_fresh_h10(row)
            decisions.append(decision)
            if old.win:
                h8_ids.append(row["pos_id"])
            if decision.win:
                h10_ids.append(row["pos_id"])
            details.append({"pos_id": row["pos_id"], "h8": old.win, **asdict(decision)})
            print(f"{name}: sample {len(details)}/{len(selected)}", flush=True)
        out["cohorts"][name] = {
            "population_n": len(population), "sample_n": len(selected),
            "population_h8_wins": len(h8_population_ids),
            "population_h8_not_h10": nesting_violations,
            "sample_h8_wins": len(h8_ids), "sample_h10_wins": len(h10_ids),
            "sample_delta": len(h10_ids) - len(h8_ids),
            "sample_h8_pct": _pct(len(h8_ids), len(selected)),
            "sample_h10_pct": _pct(len(h10_ids), len(selected)),
            "sample_h8_win_ids": h8_ids, "sample_h10_win_ids": h10_ids,
            "h10_summary": _summary(decisions), "rows": details,
        }
    return out


def validate(result: dict) -> None:
    known, all_rows = _known_registry()
    eligible = [(pos_id, info) for pos_id, info in known.items()
                if info["cert_depth"] is not None and info["cert_depth"] <= 10]
    rows_out = []
    misses = []
    for pos_id, info in eligible:
        decision = decide_h10(all_rows[pos_id])
        if not decision.win:
            misses.append(pos_id)
        rows_out.append({
            "pos_id": pos_id, "cert_depth": info["cert_depth"],
            "phase": phase_r.phase_player(len(all_rows[pos_id]["moves"]))[0],
            "win": decision.win, "nodes": decision.nodes, "wall_us": decision.wall_ns / 1e3,
        })
    result["engine_certificates_h10"] = {
        "registry_unique": len(known), "eligible": len(eligible),
        "caught": len(eligible) - len(misses), "misses": misses, "rows": rows_out,
    }

    width_ids = [
        "oa-0153903c5a863630", "oa-23c6c04ad42d0904", "oa-611666d7d930eb1f",
        "oa-6fda812864c6d19a", "oa-773ca1a59e95f4e1",
    ]
    width = []
    for short_id in width_ids:
        row = all_rows[f"atlas_full_{short_id}"]
        decision = decide_h10(row)
        width.append({
            "pos_id": short_id, "phase": phase_r.phase_player(len(row["moves"]))[0],
            "win_within_10": decision.win, "nodes": decision.nodes,
            "universe": decision.universe, "wall_us": decision.wall_ns / 1e3,
        })
    result["width_exhaust_comparison"] = {
        "rows": width, "within_10_wins": sum(row["win_within_10"] for row in width),
        "known_j2near_ids": [
            "oa-0153903c5a863630", "oa-6fda812864c6d19a", "oa-773ca1a59e95f4e1",
        ],
    }


def bounded_audit() -> dict:
    """Reproducible completed evidence after the full negative battery timed out."""
    cohorts = r2.cohorts()
    known, all_rows = _known_registry()
    r2_measure = json.loads((ROOT / ".scratch" / "horizon_r2.json").read_text(encoding="utf-8"))
    r2_validation = json.loads((ROOT / ".scratch" / "horizon_r2_validation.json").read_text(encoding="utf-8"))
    out: dict = {
        "metadata": {
            "exact_decider": True, "horizons": [8, 10],
            "measurement_boundary": "full 123-certificate plus atlas run exceeded 1200 seconds and produced no completed output",
        },
        "remote_empty_subgame": remote_empty_constant(),
        "cohorts": {},
    }
    eligible = {pos_id: info for pos_id, info in known.items()
                if info["cert_depth"] is not None and info["cert_depth"] <= 10}
    old_eligible = {pos_id for pos_id, info in eligible.items() if info["cert_depth"] <= 8}
    new_eligible = {pos_id for pos_id, info in eligible.items() if 8 < info["cert_depth"] <= 10}

    for name, rows in cohorts.items():
        fresh_rows = [row for row in rows if phase_r.phase_player(len(row["moves"]))[0] == "first"]
        fresh_ids = {row["pos_id"] for row in fresh_rows}
        h8_ids = set(r2_measure["cohorts"][name]["horizons"]["8"]["current_win"]["win_ids"])
        # Evaluate every possible antecedent of h8=>h10.  Negative h8 roots
        # cannot violate the implication; positives take the exact shortcut.
        violations = []
        for row in fresh_rows:
            if row["pos_id"] in h8_ids and not decide_fresh_h10(row).win:
                violations.append(row["pos_id"])
        certified_delta = sorted(new_eligible & fresh_ids)
        floor = h8_ids | set(certified_delta)
        out["cohorts"][name] = {
            "population_n": len(fresh_rows), "h8_wins": len(h8_ids),
            "h8_not_h10": violations,
            "certified_h10_new_ids": certified_delta,
            "h10_certified_floor_wins": len(floor),
            "h8_pct": _pct(len(h8_ids), len(fresh_rows)),
            "h10_certified_floor_pct": _pct(len(floor), len(fresh_rows)),
            "interpretation": "h10 value is a certified lower bound, not an exhaustive cohort firing rate",
        }

    tested_new_ids = ["human_b132a09ccb4eb829_p101", "sp_20_p77"]
    tested_rows = []
    for pos_id in tested_new_ids:
        decision = decide_h10(all_rows[pos_id])
        tested_rows.append({
            "pos_id": pos_id, "cert_depth": eligible[pos_id]["cert_depth"],
            "win": decision.win, "nodes": decision.nodes,
            "wall_us": decision.wall_ns / 1e3,
            "witness_first_pair": getattr(decision, "witness_first_pair", None),
        })
    out["engine_certificates_h10"] = {
        "registry_unique": len(known), "eligible": len(eligible),
        "depth_le_8_caught_by_exact_shortcut": len(old_eligible),
        "new_depth_9_10": len(new_eligible), "new_tested": len(tested_rows),
        "new_caught": sum(row["win"] for row in tested_rows),
        "new_untested_after_timeout": len(new_eligible) - len(tested_rows),
        "tested_rows": tested_rows,
    }
    out["width_exhaust_comparison"] = {
        "h10_status": "not completed; exact negative evaluation entered the same universal tail",
        "h8_rows": r2_validation["width_exhaust_comparison"]["rows"],
        "known_j2near_ids": r2_validation["width_exhaust_comparison"]["known_j2near_ids"],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / ".scratch" / "horizon_h10.json")
    ap.add_argument("--sets", nargs="*", default=["selfplay_v1", "human_v1", "puzzle_v3", "grinds", "forcing19"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample-size", type=int)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument("--bounded-audit", action="store_true")
    args = ap.parse_args()
    if args.bounded_audit:
        result = bounded_audit()
    elif args.validation_only:
        result = {
            "metadata": {"exact": True, "horizons": [8, 10], "scope": "validation only"},
            "remote_empty_subgame": remote_empty_constant(), "cohorts": {},
        }
    elif args.sample_size is not None:
        result = measure_sampled(args.sets, args.sample_size)
    else:
        result = measure(args.sets, args.limit)
    if not args.bounded_audit and not args.skip_validation and (args.limit is None or args.validation_only):
        validate(result)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "cohorts": {
            name: {key: value for key, value in item.items()
                   if key in {"n", "h8_wins", "h10_wins", "delta", "h8_not_h10",
                              "population_n", "sample_n", "sample_h8_wins", "sample_h10_wins",
                              "sample_delta", "population_h8_not_h10"}}
            for name, item in result["cohorts"].items()
        },
        "eligible": result.get("engine_certificates_h10", {}).get("eligible"),
        "misses": result.get("engine_certificates_h10", {}).get("misses"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
