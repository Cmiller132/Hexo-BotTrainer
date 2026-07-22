#!/usr/bin/env python3
"""Horizon R4 phase-clock semantics and theorem-safe static scaling.

This artifact deliberately does not run a deep-game decider.  It records the
exact physical-placement clocks, the current-attacker collapse blocks, the
registry validation rungs, and two finite geometric facts relevant after h14:

* with at most eight attacker stones, every singleton completion threat is
  coverable by the defender's final pair; and
* nine stones already suffice for four singleton completion cells, so the
  analogous h21 reserve-pair statement is false without a dynamic domination
  theorem.

The radius-eight scaling table is a rule-only envelope, not an interaction
quotient and not a runtime prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / ".scratch" / "horizon_r4_registry.json"
DEFAULT_OUTPUT = ROOT / ".scratch" / "horizon_r4_phase3.json"

AXES = ((1, 0), (0, 1), (1, -1))
PHASE_ROOT_LENGTH = {
    "FirstStone": 1,
    "SecondStone": 2,
    "opening": 0,
}
HORIZON_FIRST = 13
HORIZON_LAST = 26
SCALING_HORIZONS = (13, 14, 17, 18, 21, 22, 24, 25, 26)
REGISTRY_HORIZONS = (14, 18, 22, 24, 26)
REGISTRY_DEPTHS = (13, 14, 17, 18, 21, 22, 25, 26)
PHASE_ALIGNED_DEPTH_PAIRS = ((13, 14), (17, 18), (21, 22), (25, 26))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def compact_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def owner_at(index: int) -> int:
    """Connect6 owner of the zero-based physical placement."""
    if index == 0:
        return 0
    return 1 if ((index - 1) // 2) % 2 == 0 else 0


def phase_schedule(root_length: int, horizon: int) -> tuple[str, ...]:
    attacker = 0 if root_length == 0 else owner_at(root_length)
    return tuple(
        "A" if owner_at(root_length + offset) == attacker else "D"
        for offset in range(horizon)
    )


def clock_entry(root_length: int, horizon: int) -> dict:
    schedule = phase_schedule(root_length, horizon)
    previous = phase_schedule(root_length, horizon - 1)
    attacker_count = schedule.count("A")
    previous_attacker_count = previous.count("A")
    return {
        "horizon": horizon,
        "schedule": "".join(schedule),
        "attacker_placements": attacker_count,
        "defender_placements": schedule.count("D"),
        "last_placement_owner": schedule[-1],
        "attacker_quota_changed": attacker_count != previous_attacker_count,
        "win_equal_to_previous_horizon": attacker_count == previous_attacker_count,
    }


def constant_quota_blocks(root_length: int) -> list[dict]:
    # Search beyond the reported interval so blocks touching h26 retain their
    # true right endpoint (fresh 26--28; partial/opening 25--27).
    entries = [clock_entry(root_length, h) for h in range(1, 33)]
    blocks: list[dict] = []
    start = 1
    quota = entries[0]["attacker_placements"]
    for horizon in range(2, 34):
        boundary = (
            horizon == 33
            or entries[horizon - 1]["attacker_placements"] != quota
        )
        if not boundary:
            continue
        end = horizon - 1
        if end >= HORIZON_FIRST and start <= HORIZON_LAST:
            blocks.append({
                "first_horizon": start,
                "last_horizon": end,
                "attacker_placements": quota,
                "intersects_reported_range": [
                    max(start, HORIZON_FIRST), min(end, HORIZON_LAST)
                ],
                "current_attacker_win_predicates_equal_within_block": True,
            })
        if horizon <= 32:
            start = horizon
            quota = entries[horizon - 1]["attacker_placements"]
    return blocks


def phase_clocks() -> dict:
    out = {}
    for phase, root_length in PHASE_ROOT_LENGTH.items():
        out[phase] = {
            "representative_root_placements": root_length,
            "opening_first_A_is_forced_origin": phase == "opening",
            "entries": [
                clock_entry(root_length, h)
                for h in range(HORIZON_FIRST, HORIZON_LAST + 1)
            ],
            "constant_attacker_quota_blocks_intersecting_h13_h26":
                constant_quota_blocks(root_length),
        }
    return out


def turn_segments(symbols: tuple[str, ...]) -> list[tuple[str, int]]:
    segments: list[tuple[str, int]] = []
    for symbol in symbols:
        if segments and segments[-1][0] == symbol:
            owner, length = segments[-1]
            segments[-1] = owner, length + 1
        else:
            segments.append((symbol, 1))
    return segments


def endpoint_shape(root_length: int, horizon: int, phase: str) -> dict:
    requested = phase_schedule(root_length, horizon)
    effective_horizon = max(
        index + 1 for index, symbol in enumerate(requested) if symbol == "A"
    )
    effective = requested[:effective_horizon]
    segments = turn_segments(effective)
    assert len(segments) >= 2
    assert segments[-2] == ("D", 2)
    assert segments[-1][0] == "A" and segments[-1][1] in (1, 2)
    prefix_blocks = len(segments) - 2
    fixed_prefix_blocks = 1 if phase == "opening" else 0
    return {
        "requested_horizon": horizon,
        "effective_last_attacker_horizon": effective_horizon,
        "requested_quota": {
            "A": requested.count("A"),
            "D": requested.count("D"),
        },
        "effective_quota": {
            "A": effective.count("A"),
            "D": effective.count("D"),
        },
        "final_attacker_capacity_after_last_defender_pair": segments[-1][1],
        "turn_blocks_before_exact_final_D_pair_A_suffix": prefix_blocks,
        "quantified_choice_blocks_before_suffix": prefix_blocks - fixed_prefix_blocks,
        "opening_fixed_blocks_before_suffix": fixed_prefix_blocks,
        "scope": (
            "exact decomposition of the full dynamic true-rule tree; it does "
            "not assert that a root-anchored finite quotient is sufficient"
        ),
    }


def one_dimensional_terminal(stones: frozenset[int]) -> bool:
    starts = range(min(stones) - 5, max(stones) + 1)
    return any(
        frozenset(range(start, start + 6)) <= stones
        for start in starts
    )


def singleton_threat_cells_1d(stones: frozenset[int]) -> frozenset[int]:
    starts = range(min(stones) - 5, max(stones) + 1)
    threats: set[int] = set()
    for start in starts:
        window = frozenset(range(start, start + 6))
        missing = window - stones
        if len(missing) == 1:
            threats.update(missing)
    return frozenset(threats)


def h17_singleton_cover_census() -> dict:
    carrier = tuple(range(-5, 11))
    base = frozenset(range(0, 6))
    rows = []
    overall_max = 0
    total_nonterminal = 0
    total_terminal_rejected = 0
    for k in range(5, 9):
        histogram: Counter[int] = Counter()
        examples: dict[int, dict] = {}
        terminal_rejected = 0
        candidate_count = 0
        for raw in combinations(carrier, k):
            stones = frozenset(raw)
            if len(stones & base) != 5:
                continue
            candidate_count += 1
            if one_dimensional_terminal(stones):
                terminal_rejected += 1
                continue
            threats = singleton_threat_cells_1d(stones)
            histogram[len(threats)] += 1
            examples.setdefault(len(threats), {
                "stones": list(raw),
                "singleton_completion_cells": sorted(threats),
            })
        nonterminal_count = sum(histogram.values())
        maximum = max(histogram, default=0)
        assert candidate_count == 6 * math.comb(10, k - 5)
        assert maximum <= 2
        total_nonterminal += nonterminal_count
        total_terminal_rejected += terminal_rejected
        overall_max = max(overall_max, maximum)
        rows.append({
            "attacker_stones_in_carrier": k,
            "raw_normalized_candidates": candidate_count,
            "terminal_candidates_rejected": terminal_rejected,
            "nonterminal_candidates": nonterminal_count,
            "singleton_threat_count_histogram": {
                str(key): histogram[key] for key in sorted(histogram)
            },
            "maximum_singleton_completion_cells": maximum,
            "first_examples_by_threat_count": {
                str(key): examples[key] for key in sorted(examples)
            },
        })
    assert overall_max == 2
    return {
        "claim_label": "PROOF-SKETCH plus MEASURED exhaustive finite census",
        "claim": (
            "Every nonterminal attacker set of at most eight stones has at "
            "most two distinct singleton completion cells."
        ),
        "geometric_reduction": [
            (
                "A singleton completion window contains five attacker stones. "
                "Two distinct geometric lines would require at least "
                "5+5-1=9 stones (or 10 for disjoint parallel lines), so at "
                "k<=8 all supporting windows lie on one line."
            ),
            (
                "Normalize one supporting interval to [0,5]. Any other "
                "five-stone support shares at least two of the <=8 stones and "
                "therefore lies in the finite carrier [-5,10]. Stones outside "
                "that carrier are irrelevant to this family."
            ),
            (
                "Opponent stones only delete pure attacker windows. Every "
                "completion cell is within distance five of an attacker stone, "
                "so both the defender cover and final attacker placement obey "
                "the radius-eight rule."
            ),
        ],
        "enumeration_scope": (
            "raw subsets of [-5,10] with exactly five stones in base [0,5]; "
            "translations/reflections and positions with multiple eligible "
            "base windows are deliberately not deduplicated"
        ),
        "rows": rows,
        "totals": {
            "nonterminal_raw_normalized_candidates": total_nonterminal,
            "terminal_candidates_rejected": total_terminal_rejected,
            "maximum_singleton_completion_cells": overall_max,
            "failures_of_two_cell_cover": 0,
        },
        "dynamic_limit": (
            "This closes only the static final-pair cover needed by fresh h17. "
            "It does not prove that remote excursions can be normalized away "
            "while the defender also answers the anchored interaction."
        ),
    }


def axial_windows(stones: frozenset[tuple[int, int]]) -> set[frozenset[tuple[int, int]]]:
    windows: set[frozenset[tuple[int, int]]] = set()
    for q, r in stones:
        for dq, dr in AXES:
            for offset in range(6):
                windows.add(frozenset(
                    (q + (index - offset) * dq, r + (index - offset) * dr)
                    for index in range(6)
                ))
    return windows


def analyze_hex_singletons(stones: frozenset[tuple[int, int]]) -> dict:
    terminal_windows = []
    supporting_windows = []
    threats: set[tuple[int, int]] = set()
    for window in axial_windows(stones):
        occupied = window & stones
        if len(occupied) == 6:
            terminal_windows.append(sorted(window))
        elif len(occupied) == 5:
            missing = window - stones
            threats.update(missing)
            supporting_windows.append({
                "window": sorted(window),
                "singleton": list(next(iter(missing))),
            })
    supporting_windows.sort(key=lambda item: (item["singleton"], item["window"]))
    return {
        "terminal": bool(terminal_windows),
        "terminal_window_count": len(terminal_windows),
        "singleton_completion_cells": [list(cell) for cell in sorted(threats)],
        "singleton_cover_number": len(threats),
        "supporting_windows": supporting_windows,
    }


def h21_singleton_obstruction() -> dict:
    cross9 = frozenset(
        {(index, 0) for index in range(5)}
        | {(0, index) for index in range(5)}
    )
    # Fresh h21 has ten attacker placements before its final defender-pair /
    # attacker-singleton suffix.  This nearby tenth stone leaves the four
    # displayed singleton threats intact and keeps the set nonterminal.
    fresh10 = cross9 | {(4, 1)}
    minimal = analyze_hex_singletons(cross9)
    fresh = analyze_hex_singletons(fresh10)
    assert not minimal["terminal"] and minimal["singleton_cover_number"] == 4
    assert not fresh["terminal"] and fresh["singleton_cover_number"] == 4
    return {
        "claim_label": "MEASURED exact finite geometry; dynamic reachability unproved",
        "claim": (
            "Nine stones already support four distinct singleton completion "
            "cells, so a reserved defender pair cannot by itself dominate the "
            "fresh-h21 remote endpoint."
        ),
        "minimal_cross9": {
            "stones": [list(cell) for cell in sorted(cross9)],
            **minimal,
        },
        "fresh_h21_budget_example10": {
            "stones": [list(cell) for cell in sorted(fresh10)],
            "radius8_direct_from_origin_max_distance": 5,
            **fresh,
        },
        "dynamic_limit": (
            "The construction is a local state obstruction, not a forcing "
            "strategy: earlier defender pairs may pre-block it, and coupling "
            "that tax to the anchored interaction remains the open theorem."
        ),
    }


def registry_ladder(registry_path: Path) -> dict:
    source = json.loads(registry_path.read_text(encoding="utf-8"))
    cumulative_out = {}
    for horizon in REGISTRY_HORIZONS:
        row = source["registry"]["cumulative"][str(horizon)]
        assert compact_json_sha256(row["ids"]) == row["ids_sha256_compact_json"]
        increment = row["increment_from_previous_requested_horizon"]
        assert compact_json_sha256(increment["ids"]) == increment["ids_sha256_compact_json"]
        cumulative_out[str(horizon)] = {
            "count": row["count"],
            "by_phase": row["by_phase"],
            "ids_sha256_compact_json": row["ids_sha256_compact_json"],
            "increment": {
                "after_horizon": increment["after_horizon"],
                "count": increment["count"],
                "by_phase": increment["by_phase"],
                "ids_sha256_compact_json": increment["ids_sha256_compact_json"],
            },
        }
    depths_out = {}
    for depth in REGISTRY_DEPTHS:
        row = source["registry"]["depths"][str(depth)]
        assert compact_json_sha256(row["ids"]) == row["ids_sha256_compact_json"]
        depths_out[str(depth)] = {
            "count": row["count"],
            "by_phase": row["by_phase"],
            "ids_sha256_compact_json": row["ids_sha256_compact_json"],
        }
    rungs_out = {}
    for second_depth, fresh_depth in PHASE_ALIGNED_DEPTH_PAIRS:
        second_row = source["registry"]["depths"][str(second_depth)]
        fresh_row = source["registry"]["depths"][str(fresh_depth)]
        ids = sorted(set(second_row["ids"]) | set(fresh_row["ids"]))
        by_phase = {
            phase: second_row["by_phase"].get(phase, 0)
            + fresh_row["by_phase"].get(phase, 0)
            for phase in ("FirstStone", "SecondStone", "opening")
        }
        assert len(ids) == second_row["count"] + fresh_row["count"]
        rungs_out[f"{second_depth}_{fresh_depth}"] = {
            "SecondStone_depth": second_depth,
            "FirstStone_depth": fresh_depth,
            "count": len(ids),
            "by_phase": by_phase,
            "ids_sha256_compact_json": compact_json_sha256(ids),
        }
    return {
        "source_path": registry_path.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(registry_path),
        "cert_depth_semantics": source["metadata"]["cert_depth_semantics"],
        "cumulative": cumulative_out,
        "exact_depths": depths_out,
        "phase_aligned_exact_depth_rungs": rungs_out,
    }


def static_scaling(clocks: dict) -> dict:
    rows = []
    by_phase_entries = {
        phase: {row["horizon"]: row for row in record["entries"]}
        for phase, record in clocks.items()
    }
    for horizon in SCALING_HORIZONS:
        radius = 8 * horizon
        ball_cells = 1 + 3 * radius * (radius + 1)
        phase_shapes = {}
        for phase, root_length in PHASE_ROOT_LENGTH.items():
            phase_shapes[phase] = endpoint_shape(root_length, horizon, phase)
            assert phase_shapes[phase]["requested_quota"]["A"] == (
                by_phase_entries[phase][horizon]["attacker_placements"]
            )
        ordered_upper = (216 ** horizon) * math.factorial(horizon)
        rows.append({
            "horizon": horizon,
            "radius8_rule_envelope_radius_from_one_existing_seed": radius,
            "single_seed_rule_envelope_cells_including_seed": ball_cells,
            "single_seed_rule_envelope_future_empty_cells": ball_cells - 1,
            "single_seed_ordered_placement_sequence_upper_bound": str(ordered_upper),
            "single_seed_ordered_placement_sequence_upper_bound_log10": math.log10(ordered_upper),
            "phase_endpoint_shapes": phase_shapes,
        })
    return {
        "claim_label": "CODE-FACT / PROOF-SKETCH analytical upper bounds, not runtime measurements",
        "scope": [
            (
                "Ignoring early terminal cutoffs, the union of cells reachable "
                "within h radius-eight-chained placements from one existing "
                "seed is exactly the axial hex ball of radius 8h, with "
                "1+3R(R+1) cells. A multi-stone root uses the union of such "
                "balls, not this single-seed count."
            ),
            (
                "At a state with N occupied cells, immediate legal empties are "
                "at most 216N: each occupied cell contributes a radius-eight "
                "ball of 217 cells and all N occupied cells are removed."
            ),
            (
                "Consequently a one-seed h-placement tree has at most "
                "216^h*h! ordered placement sequences. This is a deliberately "
                "loose tree-size upper bound, not a measured wall time or a "
                "claim about the candidate interaction quotient."
            ),
            (
                "The endpoint block counts are exact for the full dynamic tree. "
                "Each deeper fresh pair rung adds one forall-D / exists-A turn "
                "pair before the final cover suffix; they do not establish a "
                "tractable finite normalization."
            ),
        ],
        "rows": rows,
    }


def build(registry_path: Path) -> dict:
    clocks = phase_clocks()
    fresh_entries = {row["horizon"]: row for row in clocks["FirstStone"]["entries"]}
    second_entries = {row["horizon"]: row for row in clocks["SecondStone"]["entries"]}
    opening_entries = {row["horizon"]: row for row in clocks["opening"]["entries"]}
    assert fresh_entries[22]["attacker_placements"] == 12
    assert fresh_entries[23]["attacker_placements"] == 12
    assert fresh_entries[24]["attacker_placements"] == 12
    assert second_entries[21]["attacker_placements"] == 11
    assert second_entries[22]["attacker_placements"] == 11
    assert second_entries[23]["attacker_placements"] == 11
    assert second_entries[24]["attacker_placements"] == 12
    assert opening_entries[21]["attacker_placements"] == 11
    assert opening_entries[24]["attacker_placements"] == 12

    return {
        "metadata": {
            "purpose": "Horizon R4 phase-3 clock semantics and static theory boundary",
            "python": platform.python_version(),
            "script_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "claim_discipline": {
                "CODE-FACT": "direct schedule or formula evaluation",
                "MEASURED": "exhaustive finite enumeration in this program",
                "PROOF-SKETCH": "definition-level mathematical reduction, not Lean-checked here",
                "HYPOTHESIS": "dynamic remote/interaction coupling remains unproved",
            },
        },
        "phase_clocks_h13_through_h26": clocks,
        "clock_consequences": {
            "claim_label": "PROOF-SKETCH from PlayerCanForceWithin semantics and CODE-FACT schedules",
            "fresh_Win18_equals_Win19_equals_Win20": True,
            "fresh_Win22_equals_Win23_equals_Win24": True,
            "SecondStone_Win17_equals_Win18_equals_Win19": True,
            "SecondStone_Win21_equals_Win22_equals_Win23": True,
            "opening_Win17_equals_Win18_equals_Win19": True,
            "opening_Win21_equals_Win22_equals_Win23": True,
            "SecondStone_and_opening_h24_are_new_attacker_placements": True,
            "global_all_phase_h24_requires_distinct_partial_opening_endpoint": True,
            "definition_level_argument": [
                (
                    "The schedules being compared have the same prefix through "
                    "the last A placement; every added placement belongs to D."
                ),
                (
                    "At an ongoing leaf after that last A placement, unfolding "
                    "PlayerCanForceWithin for one or two D placements is false: "
                    "a D placement cannot return winner A, and an ongoing child "
                    "eventually reaches the zero-fuel false case. Legal cells "
                    "remain nonempty for a finite nonempty radius-eight board."
                ),
                (
                    "Substitute these equal false suffixes and induct backward "
                    "through the common prefix. A terminal A win in the common "
                    "prefix returns true immediately in every horizon."
                ),
            ],
            "formalization_status": (
                "definition-level proof sketch only; HorizonRound.lean was "
                "read-only and no new Lean theorem was added"
            ),
        },
        "exact_dynamic_endpoint_notation": {
            "E_c": (
                "After an explicit A turn, E_c is the exact full-board suffix "
                "forall legal D pair, exists legal A continuation of capacity "
                "c in {1,2}, with first-placement terminal prefixes retained. "
                "Equivalently, after excluding a D winning prefix, the family "
                "of all A residual sets of size <=c must have no hitting set of "
                "size <=2. Residual cells are radius-eight legal because they "
                "are within five of an existing A stone."
            ),
            "fresh_h17_h18": (
                "exists A1 forall D1 exists A2 forall D2 exists A3 forall D3 "
                "exists A4: E_1 (h17) / E_2 (h18)"
            ),
            "fresh_h21_h22": (
                "exists A1 forall D1 exists A2 forall D2 exists A3 forall D3 "
                "exists A4 forall D4 exists A5: E_1 (h21) / E_2 (h22)"
            ),
            "SecondStone_h17": (
                "exists a0 forall D1 exists A1 forall D2 exists A2 forall D3 "
                "exists A3: E_2; a0 is the root turn's one remaining stone"
            ),
            "SecondStone_h21": (
                "exists a0 forall D1 exists A1 forall D2 exists A2 forall D3 "
                "exists A3 forall D4 exists A4: E_2"
            ),
            "SecondStone_h24": (
                "exists a0 forall D1 exists A1 forall D2 exists A2 forall D3 "
                "exists A3 forall D4 exists A4 forall D5 exists A5: E_1"
            ),
            "opening_adjustment": (
                "Use the SecondStone clock but replace exists a0 by the fixed "
                "forced-origin placement (0,0)."
            ),
            "normalization_status": (
                "These formulas are exact on the full dynamic true-rule tree. "
                "Replacing that tree by the R3 root-anchored V quotient at "
                "h17 or beyond remains HYPOTHESIS pending an excursion/interaction "
                "coupling theorem."
            ),
        },
        "registry_validation_ladder": registry_ladder(registry_path),
        "fresh_h17_singleton_cover": h17_singleton_cover_census(),
        "h21_singleton_reserve_pair_obstruction": h21_singleton_obstruction(),
        "rule_only_static_scaling": static_scaling(clocks),
        "phase3_boundary": {
            "claim_label": "CODE-FACT clocks; MEASURED static geometry; HYPOTHESIS dynamic normalization",
            "h17_static_final_cover": "closed for <=8 attacker stones",
            "h17_full_interaction_normalization": "open",
            "h18": (
                "blocked by the separately audited six-stone pair-activation "
                "obstruction; earlier defender pre-cover versus anchored-defense "
                "tax remains the precise open coupling theorem"
            ),
            "h21_h22": (
                "the simple reserve-pair schema already fails statically: the "
                "displayed nine-stone cross has four singleton completion cells"
            ),
            "h24_scope_warning": (
                "fresh h24 collapses to h22, but SecondStone/opening h24 add a "
                "new attacker singleton and require their own endpoint"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.registry.resolve())
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out),
        "script_sha256": result["metadata"]["script_sha256"],
        "output_sha256": sha256_file(args.out),
        "h17_nonterminal_census": result["fresh_h17_singleton_cover"]["totals"]["nonterminal_raw_normalized_candidates"],
        "h17_failures": result["fresh_h17_singleton_cover"]["totals"]["failures_of_two_cell_cover"],
        "fresh_h22_h24_collapse": result["clock_consequences"]["fresh_Win22_equals_Win23_equals_Win24"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
