#!/usr/bin/env python3
"""Production-shaped h<=8 firing rates on every frozen-set row."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".scratch"))
import deadline_ladder_r as phase_r  # noqa: E402
import horizon_r2 as r2  # noqa: E402
from horizon_h10 import _ordered_pair_iter  # noqa: E402


@dataclass(frozen=True)
class PartialDecision:
    win: bool
    nodes: int
    memo_hits: int
    universe: int
    target_windows: int
    opponent_windows: int
    wall_ns: int


def _partial_model(row: dict) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    n = len(row["moves"])
    phase, mover, _ = phase_r.phase_player(n)
    if phase != "second":
        raise ValueError("partial model requires SecondStone")
    sched = r2.schedule(n, 8)
    board = {tuple(cell): phase_r.owner_at(i) for i, cell in enumerate(row["moves"])}
    windows = r2.root_windows(board, sched)
    universe = sorted(set().union(*windows[0], *windows[1]) if windows[0] or windows[1] else set())
    index = {cell: i for i, cell in enumerate(universe)}
    masks = {
        player: tuple(sum(1 << index[cell] for cell in edge) for edge in windows[player])
        for player in (0, 1)
    }
    return mover, len(universe), masks[mover], masks[1 - mover]


def decide_second_current(row: dict) -> PartialDecision:
    """Exact A,D2,A2,D2,A endpoint for a SecondStone current mover."""
    started = time.perf_counter_ns()
    _, usize, target, opponent = _partial_model(row)
    active_a = 0
    for edge in target + opponent:
        active_a |= edge
    first_moves = list((bit for bit in _bits(active_a))) or [0]
    nodes = 0
    for a_bit in first_moves:
        nodes += 1
        if any(not edge & ~a_bit for edge in target):
            return PartialDecision(True, nodes, 0, usize, len(target), len(opponent), time.perf_counter_ns() - started)
        a_future = tuple(edge & ~a_bit for edge in target if (edge & ~a_bit).bit_count() <= 3)
        d_live = tuple(edge for edge in opponent if not edge & a_bit)
        active_d = 0
        for edge in a_future + d_live:
            active_d |= edge
        first_wins = True
        for d_pair in _ordered_pair_iter(active_d, d_live, a_future):
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
            for b_pair in _ordered_pair_iter(active_b, live_a, live_d):
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
            return PartialDecision(True, nodes, 0, usize, len(target), len(opponent), time.perf_counter_ns() - started)
    return PartialDecision(False, nodes, 0, usize, len(target), len(opponent), time.perf_counter_ns() - started)


def _bits(mask: int):
    while mask:
        bit = mask & -mask
        mask ^= bit
        yield bit


def decide_second_loss(row: dict) -> PartialDecision:
    """Exact opponent D win by the D2 endpoint at physical ply seven."""
    started = time.perf_counter_ns()
    _, usize, attacker, defender = _partial_model(row)
    active_a = 0
    for edge in attacker + defender:
        active_a |= edge
    first_moves = list(_bits(active_a)) or [0]
    nodes = 0
    for a_bit in first_moves:
        nodes += 1
        if any(not edge & ~a_bit for edge in attacker):
            return PartialDecision(False, nodes, 0, usize, len(defender), len(attacker), time.perf_counter_ns() - started)
        live_d0 = tuple(edge for edge in defender if not edge & a_bit)
        live_a0 = tuple(edge & ~a_bit for edge in attacker)
        active_d = 0
        for edge in live_d0:
            active_d |= edge
        for edge in live_a0:
            if edge.bit_count() <= 2:
                active_d |= edge
        response = False
        for d_pair in _ordered_pair_iter(active_d, live_d0, live_a0):
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
            for b_pair in _ordered_pair_iter(active_b, live_a, live_d):
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
            return PartialDecision(False, nodes, 0, usize, len(defender), len(attacker), time.perf_counter_ns() - started)
    return PartialDecision(True, nodes, 0, usize, len(defender), len(attacker), time.perf_counter_ns() - started)


def all_cohorts() -> dict[str, list[dict]]:
    frozen = {
        name: phase_r.read_jsonl(phase_r.SETS / f"{name}.jsonl")
        for name in ("selfplay_v1", "human_v1", "puzzle_v3")
    }
    by_id = {row["pos_id"]: row for row in frozen["selfplay_v1"]}
    grind_ids = [
        row["pos_id"] for row in phase_r.read_jsonl(phase_r.LABELS)
        if row.get("source") == "grind"
    ]
    frozen["grinds"] = [by_id[pos_id] for pos_id in grind_ids]
    return frozen


def decide_both(row: dict) -> tuple[r2.Decision, r2.Decision]:
    phase, mover, _ = phase_r.phase_player(len(row["moves"]))
    if phase == "first":
        return (
            r2.decide_fresh_current(row, 8),
            r2.decide_fresh_forced_loss(row, 8),
        )
    if phase == "second":
        return decide_second_current(row), decide_second_loss(row)
    # At the single opening root neither player receives six placements by
    # h=8, so R2's generic relevance quotient is exact.
    return r2.decide(row, 8, mover), r2.decide(row, 8, 1 - mover)


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    wins = sum(row["current_win"] for row in rows)
    losses = sum(row["forced_loss"] for row in rows)
    out = {
        "n": n,
        "current_wins": wins,
        "current_win_pct": pct(wins, n),
        "forced_losses": losses,
        "forced_loss_pct": pct(losses, n),
    }
    for stem in ("current", "loss"):
        nodes = [row[f"{stem}_nodes"] for row in rows]
        walls = [row[f"{stem}_wall_ns"] for row in rows]
        out[f"{stem}_cost"] = {
            "nodes_p50": percentile(nodes, .5), "nodes_p90": percentile(nodes, .9),
            "nodes_max": max(nodes), "wall_us_mean": sum(walls) / n / 1e3,
            "wall_us_p50": percentile(walls, .5) / 1e3,
            "wall_us_p90": percentile(walls, .9) / 1e3,
            "wall_us_max": max(walls) / 1e3,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / ".scratch" / "horizon_production_h8.json")
    ap.add_argument("--sets", nargs="*", default=["selfplay_v1", "human_v1", "puzzle_v3", "grinds"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--validate-existing", action="store_true")
    args = ap.parse_args()
    if args.validate_existing:
        result = json.loads(args.out.read_text(encoding="utf-8"))
        compared = []
        mismatches = []
        for row in all_cohorts()["selfplay_v1"]:
            if len(compared) >= 20:
                break
            phase, mover, _ = phase_r.phase_player(len(row["moves"]))
            if phase != "second":
                continue
            current = decide_second_current(row)
            loss = decide_second_loss(row)
            generic_current = r2.decide(row, 8, mover)
            generic_loss = r2.decide(row, 8, 1 - mover)
            compared.append(row["pos_id"])
            if (current.win, loss.win) != (generic_current.win, generic_loss.win):
                mismatches.append({
                    "pos_id": row["pos_id"],
                    "specialized": [current.win, loss.win],
                    "generic": [generic_current.win, generic_loss.win],
                })
        result["secondstone_specialized_vs_generic"] = {
            "positions": len(compared), "pos_ids": compared, "mismatches": mismatches,
        }
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result["secondstone_specialized_vs_generic"], sort_keys=True))
        return
    cohorts = all_cohorts()
    result: dict = {
        "metadata": {
            "exact": True, "horizon": 8,
            "scope": "all rows; current-attacker WinWithin8 and opponent ForcedLossWithin8",
        },
        "cohorts": {},
    }
    for name in args.sets:
        source = cohorts[name][args.offset:]
        if args.limit:
            source = source[:args.limit]
        measured = []
        started = time.perf_counter_ns()
        for i, row in enumerate(source, 1):
            if args.progress_every == 1:
                print(f"{name}: starting {i}/{len(source)} {row['pos_id']}", flush=True)
            current, loss = decide_both(row)
            phase = phase_r.phase_player(len(row["moves"]))[0]
            measured.append({
                "pos_id": row["pos_id"], "phase": phase,
                "current_win": current.win, "forced_loss": loss.win,
                "current_nodes": current.nodes, "loss_nodes": loss.nodes,
                "current_wall_ns": current.wall_ns, "loss_wall_ns": loss.wall_ns,
                "current_universe": current.universe, "loss_universe": loss.universe,
            })
            if i % args.progress_every == 0:
                print(f"{name}: {i}/{len(source)}", flush=True)
        phases = {}
        for phase in ("opening", "first", "second"):
            selected = [row for row in measured if row["phase"] == phase]
            if selected:
                phases[phase] = summarize(selected)
        result["cohorts"][name] = {
            "summary": summarize(measured), "by_phase": phases,
            "wall_ms": (time.perf_counter_ns() - started) / 1e6,
            "current_win_ids": [row["pos_id"] for row in measured if row["current_win"]],
            "forced_loss_ids": [row["pos_id"] for row in measured if row["forced_loss"]],
            "rows": measured,
        }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        name: {"summary": item["summary"], "by_phase": item["by_phase"]}
        for name, item in result["cohorts"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
