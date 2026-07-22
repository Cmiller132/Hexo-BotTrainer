#!/usr/bin/env python3
"""Exact finite-horizon Connect6 deciders and Phase-R2 measurements.

The game board is not bounded.  The search is finite because it retains every
root window that either player could complete by the requested deadline and
quotients all other placements as an inert move.  See REPORT_HORIZON_R2.md for
the proof and the precise endpoint statements.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".scratch"))
import deadline_ladder_r as phase_r  # noqa: E402


Cell = tuple[int, int]


def schedule(n: int, horizon: int) -> tuple[int, ...]:
    """Owners of the next ``horizon`` physical placements."""
    return tuple(phase_r.owner_at(n + i) for i in range(horizon))


def root_windows(board: dict[Cell, int], sched: tuple[int, ...]) -> dict[int, list[frozenset[Cell]]]:
    """All root-pure windows completable by their owner within ``sched``."""
    quotas = Counter(sched)
    out: dict[int, list[frozenset[Cell]]] = {0: [], 1: []}
    seen: dict[int, set[frozenset[Cell]]] = {0: set(), 1: set()}
    for entry in phase_r.entries(board):
        owner = entry["owner"]
        empty = entry["empty"]
        if len(empty) <= quotas[owner] and empty not in seen[owner]:
            seen[owner].add(empty)
            out[owner].append(empty)
    return out


@dataclass(frozen=True)
class Decision:
    win: bool
    nodes: int
    memo_hits: int
    universe: int
    target_windows: int
    opponent_windows: int
    wall_ns: int


class ExactHorizon:
    """Exact minimax on the finite root-relevance quotient."""

    def __init__(self, moves: list[list[int]] | list[tuple[int, int]], horizon: int, target: int | None = None):
        self.n = len(moves)
        self.sched = schedule(self.n, horizon)
        if not self.sched:
            raise ValueError("horizon must be positive")
        self.target = self.sched[0] if target is None else target
        # Moves strictly after the target's last placement cannot create a
        # target win, so they are outside the endpoint clock.
        last_target = max(i for i, p in enumerate(self.sched) if p == self.target)
        self.sched = self.sched[:last_target + 1]
        self.board = {tuple(c): phase_r.owner_at(i) for i, c in enumerate(moves)}
        windows = root_windows(self.board, self.sched)
        universe = sorted(set().union(*windows[0], *windows[1]) if windows[0] or windows[1] else set())
        self.cells = universe
        self.full = (1 << len(universe)) - 1
        index = {cell: i for i, cell in enumerate(universe)}
        self.windows = {
            p: tuple(sum(1 << index[c] for c in edge) for edge in windows[p])
            for p in (0, 1)
        }
        self.nodes = 0
        self.memo_hits = 0

    def decide(self) -> Decision:
        started = time.perf_counter_ns()
        target = self.target
        opponent = 1 - target
        sched = self.sched
        windows = self.windows
        full = self.full

        @lru_cache(maxsize=None)
        def search(step: int, p0: int, p1: int) -> bool:
            self.nodes += 1
            occupied = p0 | p1
            live: dict[int, list[int]] = {0: [], 1: []}
            residual: dict[int, list[int]] = {0: [], 1: []}
            for player, own, other in ((0, p0, p1), (1, p1, p0)):
                for edge in windows[player]:
                    if edge & other:
                        continue
                    rem = edge & ~own
                    if not rem:
                        return player == target
                    live[player].append(edge)
                    residual[player].append(rem)
            if step >= len(sched):
                return False

            # No target goal can be completed with its remaining placements.
            remaining_target = sum(p == target for p in sched[step:])
            if not residual[target] or min(x.bit_count() for x in residual[target]) > remaining_target:
                return False

            player = sched[step]
            other = 1 - player
            active = 0
            for rem in residual[player]:
                active |= rem
            for rem in residual[other]:
                active |= rem
            active &= full ^ occupied

            # An outside placement changes no deadline-relevant window.  If a
            # relevant cell exists, monotonicity makes it dominate that inert
            # placement for the mover; otherwise take the single inert class.
            if not active:
                return search(step + 1, p0, p1)

            choices = []
            bits = active
            while bits:
                bit = bits & -bits
                bits ^= bit
                own_gain = sum(bool(rem & bit) for rem in residual[player])
                blocks = sum(bool(rem & bit) for rem in residual[other])
                completes = any(rem == bit for rem in residual[player])
                choices.append((completes, own_gain + blocks, bit))
            choices.sort(reverse=True)

            if player == target:
                for _, _, bit in choices:
                    child = search(step + 1, p0 | bit, p1) if player == 0 else search(step + 1, p0, p1 | bit)
                    if child:
                        return True
                return False
            for _, _, bit in choices:
                child = search(step + 1, p0 | bit, p1) if player == 0 else search(step + 1, p0, p1 | bit)
                if not child:
                    return False
            return True

        result = search(0, 0, 0)
        info = search.cache_info()
        self.memo_hits = info.hits
        wall = time.perf_counter_ns() - started
        return Decision(result, self.nodes, self.memo_hits, len(self.cells),
                        len(self.windows[target]), len(self.windows[opponent]), wall)


def decide(row: dict, horizon: int, target: int | None = None) -> Decision:
    return ExactHorizon(row["moves"], horizon, target).decide()


def decide_ladder(row: dict, horizon: int, target: int | None = None) -> Decision:
    """Exact up-to-h decider with monotone smaller-clock short-circuiting."""
    total_nodes = total_hits = total_wall = 0
    last: Decision | None = None
    for rung in range(1, horizon + 1):
        last = decide(row, rung, target)
        total_nodes += last.nodes
        total_hits += last.memo_hits
        total_wall += last.wall_ns
        if last.win:
            return Decision(True, total_nodes, total_hits, last.universe,
                            last.target_windows, last.opponent_windows, total_wall)
    assert last is not None
    return Decision(False, total_nodes, total_hits, last.universe,
                    last.target_windows, last.opponent_windows, total_wall)


def _has_cover_two(family: list[int]) -> bool:
    """Whether nonempty bit-set edges have a hitting set of size at most 2."""
    if not family:
        return True
    common = family[0]
    for edge in family[1:]:
        common &= edge
    if common:
        return True
    first = family[0]
    while first:
        x = first & -first
        first ^= x
        rest = [edge for edge in family if not edge & x]
        if not rest:
            return True
        common = rest[0]
        for edge in rest[1:]:
            common &= edge
        if common:
            return True
    return False


def _pairs(active: int) -> Iterable[int]:
    bits = []
    while active:
        bit = active & -active
        active ^= bit
        bits.append(bit)
    if not bits:
        yield 0  # two inert placements
    elif len(bits) == 1:
        yield bits[0]  # relevant placement plus one inert placement
    else:
        for a, b in combinations(bits, 2):
            yield a | b


def _ordered_pairs(active: int, own: list[int] | tuple[int, ...], other: list[int] | tuple[int, ...]) -> list[int]:
    """Exact pair universe, ordered only to expose tactical witnesses early."""
    pairs = list(_pairs(active))
    def score(pair: int) -> tuple[int, int, int]:
        immediate = int(any(not (edge & ~pair) for edge in own))
        own_gain = sum((edge & pair).bit_count() for edge in own)
        blocks = sum(bool(edge & pair) for edge in other)
        return immediate, own_gain + 2 * blocks, own_gain
    pairs.sort(key=score, reverse=True)
    return pairs


def _fresh_model(row: dict, horizon: int) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    """Return mover, universe size, and root window masks for a fresh turn."""
    n = len(row["moves"])
    phase, mover, _ = phase_r.phase_player(n)
    if phase != "first":
        raise ValueError("fresh decider requires FirstStone")
    sched = schedule(n, horizon)
    board = {tuple(c): phase_r.owner_at(i) for i, c in enumerate(row["moves"])}
    windows = root_windows(board, sched)
    universe = sorted(set().union(*windows[0], *windows[1]) if windows[0] or windows[1] else set())
    index = {cell: i for i, cell in enumerate(universe)}
    masks = {p: tuple(sum(1 << index[c] for c in edge) for edge in windows[p]) for p in (0, 1)}
    return mover, len(universe), masks[mover], masks[1 - mover]


def decide_fresh_current(row: dict, horizon: int) -> Decision:
    """Exact current-mover win at a fresh turn for h=6 or h=8.

    The mover's last placement is ply 6 at both horizons.  A first pair wins
    exactly when it completes six or leaves a rank-at-most-two threat family
    with no two-cell cover, while allowing no opponent current-turn win.
    """
    if horizon not in (6, 8):
        raise ValueError(horizon)
    started = time.perf_counter_ns()
    mover, usize, own, opp = _fresh_model(row, 6)
    active = 0
    for edge in own + opp:
        active |= edge
    nodes = 0
    for pair in _ordered_pairs(active, own, opp):
        nodes += 1
        if any(not edge & ~pair for edge in own):
            return Decision(True, nodes, 0, usize, len(own), len(opp), time.perf_counter_ns() - started)
        opp_now = [edge for edge in opp if not edge & pair and edge.bit_count() <= 2]
        if opp_now:
            continue
        threats = [edge & ~pair for edge in own if (edge & ~pair).bit_count() <= 2]
        if threats and not _has_cover_two(threats):
            return Decision(True, nodes, 0, usize, len(own), len(opp), time.perf_counter_ns() - started)
    return Decision(False, nodes, 0, usize, len(own), len(opp), time.perf_counter_ns() - started)


def decide_fresh_forced_loss(row: dict, horizon: int) -> Decision:
    """Exact opponent-forced win from a fresh current-mover turn."""
    if horizon not in (6, 8):
        raise ValueError(horizon)
    started = time.perf_counter_ns()
    mover, usize, a_windows, d_windows = _fresh_model(row, horizon)
    active_a = 0
    for edge in a_windows + d_windows:
        active_a |= edge
    nodes = 0
    for a_pair in _ordered_pairs(active_a, list(a_windows), list(d_windows)):
        nodes += 1
        # One legal current-mover win refutes an opponent forcing strategy.
        if any(not edge & ~a_pair for edge in a_windows):
            return Decision(False, nodes, 0, usize, len(d_windows), len(a_windows), time.perf_counter_ns() - started)
        live_d = [edge for edge in d_windows if not edge & a_pair]
        live_a = [edge & ~a_pair for edge in a_windows]
        if horizon == 6:
            if not any(edge.bit_count() <= 2 for edge in live_d):
                return Decision(False, nodes, 0, usize, len(d_windows), len(a_windows), time.perf_counter_ns() - started)
            continue

        # At h=8 the opponent has a first pair now and a completion pair after
        # the current mover's second turn.  Find a pair that wins now or makes
        # an unanswerable threat while suppressing every intervening win-now.
        active_d = 0
        for edge in live_d + live_a:
            active_d |= edge
        response = False
        for d_pair in _pairs(active_d):
            nodes += 1
            if any(not edge & ~d_pair for edge in live_d):
                response = True
                break
            a_now = [edge for edge in live_a if not edge & d_pair and edge.bit_count() <= 2]
            if a_now:
                continue
            threats = [edge & ~d_pair for edge in live_d if (edge & ~d_pair).bit_count() <= 2]
            if threats and not _has_cover_two(threats):
                response = True
                break
        if not response:
            return Decision(False, nodes, 0, usize, len(d_windows), len(a_windows), time.perf_counter_ns() - started)
    return Decision(True, nodes, 0, usize, len(d_windows), len(a_windows), time.perf_counter_ns() - started)


def cohorts() -> dict[str, list[dict]]:
    out = {name: phase_r.read_jsonl(phase_r.SETS / f"{name}.jsonl")
           for name in ("selfplay_v1", "human_v1", "puzzle_v3")}
    by_id = {r["pos_id"]: r for r in out["selfplay_v1"]}
    grind_ids = [r["pos_id"] for r in phase_r.read_jsonl(phase_r.LABELS)
                 if r.get("source") == "grind"]
    out["grinds"] = [by_id[x] for x in grind_ids]
    out["forcing19"] = phase_r.parse_forcing(phase_r.FORCING)
    return out


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def percentile(xs: list[int], q: float) -> int:
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int((len(ys) - 1) * q))]


def summary(ds: list[Decision]) -> dict:
    walls = [d.wall_ns for d in ds]
    nodes = [d.nodes for d in ds]
    return {
        "n": len(ds), "wins": sum(d.win for d in ds), "win_pct": pct(sum(d.win for d in ds), len(ds)),
        "nodes": {"total": sum(nodes), "p50": percentile(nodes, .5), "p90": percentile(nodes, .9), "max": max(nodes)},
        "wall_us": {"total": sum(walls) / 1e3, "mean": sum(walls) / len(walls) / 1e3,
                    "p50": percentile(walls, .5) / 1e3, "p90": percentile(walls, .9) / 1e3,
                    "max": max(walls) / 1e3},
        "universe": {"p50": percentile([d.universe for d in ds], .5),
                     "p90": percentile([d.universe for d in ds], .9), "max": max(d.universe for d in ds)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / ".scratch" / "horizon_r2.json")
    ap.add_argument("--sets", nargs="*", default=["selfplay_v1", "human_v1", "puzzle_v3", "grinds", "forcing19"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--horizons", nargs="*", type=int, default=[6, 8])
    ap.add_argument("--fresh-only", action="store_true")
    args = ap.parse_args()
    all_cohorts = cohorts()
    result: dict = {"metadata": {"exact": True, "horizons": args.horizons}, "cohorts": {}}
    for name in args.sets:
        rows = all_cohorts[name]
        if args.fresh_only:
            rows = [r for r in rows if phase_r.phase_player(len(r["moves"]))[0] == "first"]
        if args.limit:
            rows = rows[:args.limit]
        item = {"n": len(rows), "horizons": {}}
        old_h2 = []
        old_h4_loss = []
        if args.fresh_only:
            for row in rows:
                f = phase_r.feature(row)
                if f["own_win_now"]:
                    old_h2.append(row["pos_id"])
                if f["forced_loss"]:
                    old_h4_loss.append(row["pos_id"])
            item["validated_base"] = {"h2_win_ids": old_h2, "h4_forced_loss_ids": old_h4_loss}
        for h in args.horizons:
            current = []
            dual = []
            for i, row in enumerate(rows, 1):
                mover = schedule(len(row["moves"]), 1)[0]
                if args.fresh_only:
                    current.append(decide_fresh_current(row, h))
                    dual.append(decide_fresh_forced_loss(row, h))
                else:
                    current.append(decide(row, h, mover))
                    dual.append(decide(row, h, 1 - mover))
            item["horizons"][str(h)] = {
                "current_win": {**summary(current), "win_ids": [r["pos_id"] for r, d in zip(rows, current) if d.win]},
                "forced_loss": {**summary(dual), "win_ids": [r["pos_id"] for r, d in zip(rows, dual) if d.win]},
            }
        if args.fresh_only:
            h6w = set(item["horizons"]["6"]["current_win"]["win_ids"])
            h8w = set(item["horizons"]["8"]["current_win"]["win_ids"])
            h6l = set(item["horizons"]["6"]["forced_loss"]["win_ids"])
            h8l = set(item["horizons"]["8"]["forced_loss"]["win_ids"])
            item["internal_validation"] = {
                "h2_not_h6": sorted(set(old_h2) - h6w),
                "h6_not_h8": sorted(h6w - h8w),
                "h4_loss_not_h6": sorted(set(old_h4_loss) - h6l),
                "h6_loss_not_h8": sorted(h6l - h8l),
            }
        result["cohorts"][name] = item
        print(name, json.dumps(item["horizons"], separators=(",", ":")))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
