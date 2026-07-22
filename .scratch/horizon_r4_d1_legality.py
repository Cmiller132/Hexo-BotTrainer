#!/usr/bin/env python3
"""Horizon R4 audit of D1 radius-eight legality and later-stage saturation.

This script is deliberately cargo-free.  It reuses the frozen R3 model only
to expose a true-rule bug at the first defender pair and to check the repair's
load-bearing geometric invariant on every atlas principal variation available
for the depth-13/14 registry cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".scratch"))

import deadline_ladder_r as phase_r  # noqa: E402
import horizon_h10 as h10  # noqa: E402
import horizon_r3 as r3  # noqa: E402


Cell = tuple[int, int]


PLAYED_HINT_IDS = (
    "human_162b07d9a12084e2_p61",
    "human_194ce5fa25e2029d_p40",
    "human_1d7540b8789196ad_p54",
    "human_20bea7804fffee60_p15",
    "human_27057abe4ee1d907_p15",
    "human_37979e171ebe14b2_p79",
    "human_3c9423b1a047e24c_p22",
    "human_41e78c67c2ac8570_p20",
    "human_6023b2ef70e3ffc6_p76",
    "human_736f572521236d55_p17",
    "human_c77600999ef0c953_p45",
    "human_c77600999ef0c953_p49",
    "human_d4f2b0b1853d8565_p44",
    "sp_0_p51",
    "sp_13_p59",
    "sp_35_p27",
    "sp_4_p79",
    "sp_5_p61",
)


def distribution(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)

    def pick(q: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]

    return {
        "min": ordered[0],
        "p50": pick(0.5),
        "p90": pick(0.9),
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def mask(cells: list[Cell] | tuple[Cell, ...], index: dict[Cell, int]) -> int:
    return sum(1 << index[cell] for cell in cells)


def occupied_after(row: dict, extra: list[Cell] | tuple[Cell, ...]) -> set[Cell]:
    return {tuple(cell) for cell in row["moves"]} | set(extra)


def minimum_distance(cell: Cell, occupied: set[Cell]) -> int:
    return min(r3.hex_distance(cell, stone) for stone in occupied)


def active_cells(active: int, cells: tuple[Cell, ...]) -> list[Cell]:
    return [cells[i] for i in range(len(cells)) if active & (1 << i)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def played_line_hints(known: dict[str, dict], rows: dict[str, dict]) -> dict:
    """Recover one ordering-only A1 hint for each non-atlas runtime boundary.

    A hint is the next real-game action following the frozen registry prefix.
    The assertions deliberately verify both prefix identity and membership in
    the exact normalized first-action generator.  No played continuation is a
    certificate, a verdict, or permission to prune another action.
    """

    human_source = ROOT.parents[2] / "data" / "hexo-bootstrap-corpus" / "hexo_human_corpus.jsonl"
    selfplay_source = ROOT / "scripts" / "tss_harness" / "sets" / "selfplay_v1.jsonl"
    human_games = {record["game_hash"]: record for record in jsonl(human_source)}
    selfplay_games: dict[int, dict] = {}
    for record in jsonl(selfplay_source):
        game = int(record["meta"]["game"])
        if game not in selfplay_games or len(record["moves"]) > len(selfplay_games[game]["moves"]):
            selfplay_games[game] = record

    checked = []
    for pos_id in PLAYED_HINT_IDS:
        if match := re.fullmatch(r"human_([0-9a-f]{16})_p(\d+)", pos_id):
            source_kind = "human"
            source = human_source
            source_key: str | int = match.group(1)
            placements = int(match.group(2))
            record = human_games[source_key]
        else:
            match = re.fullmatch(r"sp_(\d+)_p(\d+)", pos_id)
            assert match is not None
            source_kind = "selfplay"
            source = selfplay_source
            source_key = int(match.group(1))
            placements = int(match.group(2))
            record = selfplay_games[source_key]

        row = rows[pos_id]
        assert placements == len(row["moves"])
        assert record["moves"][:placements] == row["moves"]
        phase, _, budget = phase_r.phase_player(placements)
        played = tuple(tuple(cell) for cell in record["moves"][placements : placements + budget])
        assert len(played) == budget
        model = r3.build_next_model(row, int(known[pos_id]["cert_depth"]))
        index = {cell: i for i, cell in enumerate(model.cells)}
        assert all(cell in index for cell in played)
        action = mask(played, index)
        generated = action in model.first_pairs
        assert generated
        checked.append({
            "pos_id": pos_id,
            "source_kind": source_kind,
            "source_key": source_key,
            "registry_prefix_placements": placements,
            "phase": phase,
            "action_placements": budget,
            "played_next_action": [list(cell) for cell in played],
            "member_of_exact_first_action_generator": generated,
            "source_record_placements": len(record["moves"]),
            "source_path": str(source),
        })

    assert len(checked) == 18
    assert Counter(row["source_kind"] for row in checked) == {"human": 13, "selfplay": 5}
    return {
        "claim": "MEASURED ordering-only heuristic; not a certificate, verdict, or pruning rule",
        "boundary_ids": len(checked),
        "source_counts": dict(sorted(Counter(row["source_kind"] for row in checked).items())),
        "all_registry_prefixes_match_source": True,
        "all_hints_member_of_exact_first_action_generator": True,
        "safe_use": "Move the matching action to the front of A1 ordering only; search all other actions normally if it fails.",
        "source_files": {
            "human": {"path": str(human_source), "sha256": sha256(human_source)},
            "selfplay": {"path": str(selfplay_source), "sha256": sha256(selfplay_source)},
        },
        "rows": checked,
    }


def live_after_a1(model: r3.NextModel, a1: int, after_a1: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    target = tuple(
        edge & ~a1
        for edge in model.target_anchored
        if (edge & ~a1).bit_count() <= after_a1
    ) + tuple(
        edge & ~a1
        for edge in model.near_windows
        if edge & a1 and (edge & ~a1).bit_count() <= after_a1
    )
    defender = tuple(edge for edge in model.opponent_windows if not edge & a1)
    return target, defender


def published_boundary(known_rows: dict[str, dict]) -> dict:
    pos_id = "atlas_oa-c515cddcef6134b3"
    row = known_rows[pos_id]
    model = r3.build_next_model(row, 14)
    index = {cell: i for i, cell in enumerate(model.cells)}
    preferred_cells = ((2, 0), (2, 1))
    a1 = mask(preferred_cells, index)
    target, defender = live_after_a1(model, a1, 6)
    active = 0
    for edge in target + defender:
        active |= edge
    occupied = occupied_after(row, preferred_cells)
    first_illegal = None
    legal_before = 0
    for rank, action in enumerate(h10._ordered_pair_iter(active, defender, target), 1):
        cells = r3.decode_pair(action, model.cells)
        if r3.pair_is_legal(cells, occupied):
            legal_before += 1
            continue
        first_illegal = {
            "rank_one_based": rank,
            "cells": [list(cell) for cell in cells],
            "minimum_post_a1_occupied_distances": [minimum_distance(cell, occupied) for cell in cells],
            "legal_actions_before_it": legal_before,
        }
        break
    assert first_illegal is not None
    assert first_illegal["cells"] == [[-19, 3], [-19, 4]]
    assert first_illegal["rank_one_based"] == 67
    return {
        "claim": "MEASURED",
        "pos_id": pos_id,
        "horizon": 14,
        "preferred_a1": [list(cell) for cell in preferred_cells],
        "active_d1_physical_cells": active.bit_count(),
        "post_a1_legal_active_cells": sum(
            minimum_distance(cell, occupied) <= r3.LEGAL_RADIUS
            for cell in active_cells(active, model.cells)
        ),
        "first_illegal_generated_d1": first_illegal,
    }


def atlas_pv_audit(known: dict[str, dict], rows: dict[str, dict]) -> dict:
    atlas = {
        f"atlas_full_{item['id']}": item
        for item in json.loads(phase_r.DEFAULT_ATLAS.read_text(encoding="utf-8"))["rows"]
    }
    checked = []
    endpoint_threats: list[int] = []
    stage_max_distances: dict[str, list[int]] = {"a2": [], "d2": [], "a3": []}
    for pos_id, info in sorted(known.items()):
        if info["cert_depth"] != 14 or pos_id not in atlas:
            continue
        row = rows[pos_id]
        line = [tuple(cell) for cell in atlas[pos_id]["win_line"]]
        model = r3.build_next_model(row, 14)
        index = {cell: i for i, cell in enumerate(model.cells)}
        assert len(line) >= 10
        assert all(cell in index for cell in line[:10])
        a1, d1, a2, d2, a3 = (
            mask(line[start : start + 2], index) for start in range(0, 10, 2)
        )
        assert a1 in model.first_pairs
        occupied = occupied_after(row, line[:2])
        assert r3.pair_is_legal(tuple(line[:2]), occupied_after(row, ()))
        live_a1, live_d1 = live_after_a1(model, a1, 6)
        assert not any(not edge & ~d1 for edge in live_d1)
        assert r3.pair_is_legal(tuple(line[2:4]), occupied)
        occupied.update(line[2:4])

        live_a_d1 = tuple(edge for edge in live_a1 if not edge & d1)
        live_d_d1 = tuple(
            edge & ~d1 for edge in live_d1 if (edge & ~d1).bit_count() <= 4
        )
        active_a2 = 0
        for edge in live_a_d1 + live_d_d1:
            active_a2 |= edge
        a2_cells = active_cells(active_a2, model.cells)
        a2_max = max((minimum_distance(cell, occupied) for cell in a2_cells), default=0)
        stage_max_distances["a2"].append(a2_max)
        assert a2_max <= 5
        assert not a2 & ~active_a2
        assert r3.pair_is_legal(tuple(line[4:6]), occupied)
        occupied.update(line[4:6])

        live_a2 = tuple(
            edge & ~a2
            for edge in live_a_d1
            if (edge & ~a2).bit_count() <= 4
        )
        live_d2 = tuple(edge for edge in live_d_d1 if not edge & a2)
        active_d2 = 0
        for edge in live_a2 + live_d2:
            active_d2 |= edge
        d2_cells = active_cells(active_d2, model.cells)
        d2_max = max((minimum_distance(cell, occupied) for cell in d2_cells), default=0)
        stage_max_distances["d2"].append(d2_max)
        assert d2_max <= 5
        assert not d2 & ~active_d2
        assert r3.pair_is_legal(tuple(line[6:8]), occupied)
        occupied.update(line[6:8])

        live_a_d2 = tuple(edge for edge in live_a2 if not edge & d2)
        live_d_d2 = tuple(
            edge & ~d2 for edge in live_d2 if (edge & ~d2).bit_count() <= 2
        )
        active_a3 = 0
        for edge in live_a_d2 + live_d_d2:
            active_a3 |= edge
        a3_cells = active_cells(active_a3, model.cells)
        a3_max = max((minimum_distance(cell, occupied) for cell in a3_cells), default=0)
        stage_max_distances["a3"].append(a3_max)
        assert a3_max <= 5
        assert not a3 & ~active_a3
        assert r3.pair_is_legal(tuple(line[8:10]), occupied)

        immediate = any(not edge & ~a3 for edge in live_a_d2)
        safe = not any(
            not edge & a3 and (edge & ~a3).bit_count() <= 2
            for edge in live_d_d2
        )
        threats = [
            edge & ~a3
            for edge in live_a_d2
            if (edge & ~a3).bit_count() <= 2
        ]
        endpoint = immediate or (safe and bool(threats) and h10._cover_two_witness(threats) is None)
        assert endpoint
        endpoint_threats.append(len(threats))
        checked.append({
            "pos_id": pos_id,
            "win_line_len": len(line),
            "a2_active_max_occupied_distance": a2_max,
            "d2_active_max_occupied_distance": d2_max,
            "a3_active_max_occupied_distance": a3_max,
            "pv_a3_endpoint_win": endpoint,
            "pv_a3_threats": len(threats),
        })

    assert len(checked) == 136
    return {
        "claim": "MEASURED",
        "atlas_roots_checked": len(checked),
        "all_first_five_pv_actions_generated_and_legal": True,
        "all_a2_d2_a3_active_cells_within_distance_five": True,
        "all_pv_a3_endpoints_win_in_r3_formula": True,
        "win_line_length_counts": dict(sorted(Counter(row["win_line_len"] for row in checked).items())),
        "a3_endpoint_threat_count_distribution": distribution(endpoint_threats),
        "stage_active_max_distance_distributions": {
            stage: distribution(values) for stage, values in stage_max_distances.items()
        },
        "rows": checked,
    }


def measure() -> dict:
    known, rows = h10._known_registry()
    eligible = sorted(
        (pos_id, info) for pos_id, info in known.items()
        if info["cert_depth"] in (13, 14)
    )
    fringe_rows = []
    for pos_id, info in eligible:
        row = rows[pos_id]
        model = r3.build_next_model(row, int(info["cert_depth"]))
        occupied = occupied_after(row, ())
        distances = [minimum_distance(cell, occupied) for cell in model.cells]
        fringe_rows.append({
            "pos_id": pos_id,
            "cert_depth": info["cert_depth"],
            "phase": phase_r.phase_player(len(row["moves"]))[0],
            "physical_universe": len(model.cells),
            "root_illegal_fringe_cells": sum(distance > r3.LEGAL_RADIUS for distance in distances),
            "maximum_root_distance": max(distances, default=0),
        })
    assert len(fringe_rows) == 155
    assert all(row["root_illegal_fringe_cells"] > 0 for row in fringe_rows)

    exact_move_groups: dict[tuple[Cell, ...], list[str]] = {}
    for pos_id, _ in eligible:
        exact_move_groups.setdefault(tuple(map(tuple, rows[pos_id]["moves"])), []).append(pos_id)
    duplicates = [ids for ids in exact_move_groups.values() if len(ids) > 1]

    return {
        "metadata": {
            "scope": "depth-13/14 registry IDs under the R3 finite quotient",
            "eligible_ids": len(eligible),
            "cargo_used": False,
            "timeout_is_not_verdict": True,
        },
        "root_fringe": {
            "claim": "MEASURED",
            "ids": len(fringe_rows),
            "ids_with_root_illegal_fringe": sum(row["root_illegal_fringe_cells"] > 0 for row in fringe_rows),
            "physical_universe_distribution": distribution([row["physical_universe"] for row in fringe_rows]),
            "root_illegal_fringe_distribution": distribution([row["root_illegal_fringe_cells"] for row in fringe_rows]),
            "maximum_root_distance_distribution": distribution([row["maximum_root_distance"] for row in fringe_rows]),
            "rows": fringe_rows,
        },
        "published_boundary_counterexample": published_boundary(rows),
        "atlas_pv_saturation": atlas_pv_audit(known, rows),
        "played_continuation_ordering_hints": played_line_hints(known, rows),
        "registry_identity_audit": {
            "claim": "CODE-FACT",
            "eligible_ids": len(eligible),
            "distinct_exact_move_lists": len(exact_move_groups),
            "duplicate_id_groups": duplicates,
        },
        "proof_status": {
            "d1_legal_pair_generator": {
                "claim": "PROOF-SKETCH",
                "predicate": "L1(x) and (L1(y) or dist(x,y)<=8), or symmetrically",
                "L1": "root_legal(cell) or within radius 8 of either A1 placement",
                "protocol_requirement": "root_legal flags, quotient-cell coordinates, and A1; root occupied coordinates are not required",
            },
            "empty_and_singleton_projection": {
                "claim": "PROOF-SKETCH relative to the R3 interaction normalization",
                "statement": "D1 turns with zero/one retained effects are represented by EMPTY/singleton actions; the omitted filler changes no retained window, and after quota pruning every future retained action is already legal without that filler",
                "initially_illegal_fringe": "Such a cell is in neither an A1-seeded/anchored attacker window nor an anchored defender window. Alone it can touch only pristine six-empty defender-near windows, leaving residual five with only four D placements left, so its retained effect is EMPTY.",
            },
            "later_stage_legality": {
                "claim": "PROOF-SKETCH plus 136-root measured PV audit",
                "statement": "After D1, every retained A window contains a root/A1 A stone. A retained D-near window has residual at most four, so both D1 stones lie in it; anchored D windows contain a root D stone. Every residual cell is therefore within hex distance five of an occupied stone. The invariant persists through A2, D2, A3, the D3 cover, and the final A completion.",
                "consequence": "Incidence-twin action quotienting is legality-safe from A2 onward, but not at D1 unless classes carry a legal physical-pair witness.",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / ".scratch" / "horizon_r4_d1_legality.json")
    args = parser.parse_args()
    result = measure()
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "eligible_ids": result["metadata"]["eligible_ids"],
        "fringe_cells": result["root_fringe"]["root_illegal_fringe_distribution"],
        "counterexample": result["published_boundary_counterexample"]["first_illegal_generated_d1"],
        "atlas_pv_checked": result["atlas_pv_saturation"]["atlas_roots_checked"],
        "played_hints_checked": result["played_continuation_ordering_hints"]["boundary_ids"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
