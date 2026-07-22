#!/usr/bin/env python3
"""Durable depth/phase inventory for the Horizon R4 validation ladder.

The registry definition is intentionally the inherited H10/R3 definition:
``horizon_h10._known_registry``.  This program does not solve positions or
touch engine state.  It materializes the exact sorted eligible ID sets and the
phase clocks needed to interpret them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".scratch"))

import deadline_ladder_r as phase_r  # noqa: E402
import horizon_h10 as h10  # noqa: E402
import horizon_r2 as r2  # noqa: E402


REQUESTED_CUMULATIVE_HORIZONS = (14, 18, 22, 24, 26)
PHASE_LABEL = {
    "opening": "opening",
    "first": "FirstStone",
    "second": "SecondStone",
}
PHASES = ("opening", "FirstStone", "SecondStone")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def compact_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def phase_of(row: dict) -> str:
    return PHASE_LABEL[phase_r.phase_player(len(row["moves"]))[0]]


def normalized_depth(value: object) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def input_paths() -> list[Path]:
    return [
        *(phase_r.SETS / f"{name}.jsonl"
          for name in ("selfplay_v1", "human_v1", "puzzle_v3")),
        phase_r.LABELS,
        phase_r.FORCING,
        *(phase_r.MAIN4 / f"records_main4_integration_gate2_{name}.jsonl"
          for name in ("selfplay_v1", "human_v1", "puzzle_v3")),
        phase_r.DEFAULT_ATLAS,
    ]


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def file_record(path: Path) -> dict:
    return {
        "path": display_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def count_breakdown(
    ids: Iterable[str],
    known: dict[str, dict],
    rows: dict[str, dict],
    cohort_ids: dict[str, set[str]],
) -> dict:
    selected = set(ids)
    phases = Counter(phase_of(rows[pos_id]) for pos_id in selected)
    sources = Counter(known[pos_id]["source"] for pos_id in selected)
    return {
        "count": len(selected),
        "by_phase": {phase: phases.get(phase, 0) for phase in PHASES},
        "by_source": dict(sorted(sources.items())),
        # Membership counts intentionally overlap: grinds are a self-play
        # subset, and some IDs occur in more than one frozen cohort.
        "by_cohort_membership": {
            name: len(selected & members)
            for name, members in sorted(cohort_ids.items())
        },
        "atlas_full_ids": sum(pos_id.startswith("atlas_full_") for pos_id in selected),
    }


def exact_set_record(
    ids: Iterable[str],
    known: dict[str, dict],
    rows: dict[str, dict],
    cohort_ids: dict[str, set[str]],
) -> dict:
    ordered = sorted(set(ids))
    return {
        **count_breakdown(ordered, known, rows, cohort_ids),
        "ids_sha256_compact_json": compact_json_sha256(ordered),
        "ids": ordered,
    }


def raw_registry_audit() -> dict:
    """Reconstruct all pre-dedup candidate references and audit ambiguity."""
    cohorts = r2.cohorts()
    all_rows = {row["pos_id"]: row for rows in cohorts.values() for row in rows}
    references: dict[str, list[dict]] = defaultdict(list)

    def add(pos_id: str, source: str, depth: object) -> None:
        references[pos_id].append({
            "source": source,
            "cert_depth": normalized_depth(depth),
        })

    for row in cohorts["puzzle_v3"]:
        if row.get("labels", {}).get("verdict") == "win":
            add(row["pos_id"], "puzzle_v3 labeled WIN", None)
    for row in cohorts["forcing19"]:
        if row.get("labels", {}).get("verdict") == "win":
            add(row["pos_id"], "forcing19 expected WIN", None)
    for cohort_name, wall in phase_r.wall_sources().items():
        for pos_id, record in wall.items():
            if record["status"] == "win" and pos_id in all_rows:
                add(
                    pos_id,
                    f"{cohort_name} measured WIN",
                    record.get("cert_depth"),
                )

    atlas_rows = json.loads(
        phase_r.DEFAULT_ATLAS.read_text(encoding="utf-8")
    )["rows"]
    atlas_ids: list[str] = []
    for atlas in atlas_rows:
        if atlas.get("status") != "WIN" or not atlas.get("certified"):
            continue
        pos_id = f"atlas_full_{atlas['id']}"
        atlas_ids.append(pos_id)
        derived = atlas.get("derived_horizon")
        depth = derived - atlas["placements"] if isinstance(derived, int) else None
        add(pos_id, "opening atlas certified WIN", depth)

    conflicts = []
    for pos_id, candidates in sorted(references.items()):
        nonnull = sorted({
            item["cert_depth"] for item in candidates
            if item["cert_depth"] is not None
        })
        if len(nonnull) > 1:
            conflicts.append({
                "pos_id": pos_id,
                "nonnull_depths": nonnull,
                "references": candidates,
            })

    seen_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, cohort_rows in cohorts.items():
        for row in cohort_rows:
            moves = json.dumps(row["moves"], separators=(",", ":"))
            seen_rows[row["pos_id"]].append((name, moves))
    move_conflicts = [
        {
            "pos_id": pos_id,
            "cohorts": sorted(name for name, _ in versions),
        }
        for pos_id, versions in sorted(seen_rows.items())
        if len({moves for _, moves in versions}) > 1
    ]

    return {
        "raw_candidate_references": sum(len(items) for items in references.values()),
        "unique_candidate_ids": len(references),
        "deduplicated_references": sum(len(items) for items in references.values()) - len(references),
        "ids_with_multiple_candidate_references": sum(
            len(items) > 1 for items in references.values()
        ),
        "conflicting_nonnull_depths": conflicts,
        "same_depth_multi_reference_ids": sum(
            len([item for item in items if item["cert_depth"] is not None]) > 1
            and len({
                item["cert_depth"] for item in items
                if item["cert_depth"] is not None
            }) == 1
            for items in references.values()
        ),
        "atlas_certified_win_references": len(atlas_ids),
        "atlas_certified_win_unique_ids": len(set(atlas_ids)),
        "atlas_id_collisions_with_frozen_cohorts": len(set(atlas_ids) & set(all_rows)),
        "cohort_ids_present_in_multiple_cohorts": sum(
            len(items) > 1 for items in seen_rows.values()
        ),
        "cohort_duplicate_move_conflicts": move_conflicts,
    }


def clock_table() -> dict:
    representatives = {
        "opening": 0,
        "FirstStone": 1,
        "SecondStone": 2,
    }
    out: dict[str, dict] = {}
    for phase, placements_made in representatives.items():
        attacker = phase_r.phase_player(placements_made)[1]
        entries = []
        previous_attacker_quota: int | None = None
        for horizon in range(1, 27):
            schedule = r2.schedule(placements_made, horizon)
            attacker_quota = schedule.count(attacker)
            defender_quota = schedule.count(1 - attacker)
            quota_changed = (
                previous_attacker_quota is None
                or attacker_quota != previous_attacker_quota
            )
            entries.append({
                "horizon": horizon,
                "schedule": "".join(
                    "A" if owner == attacker else "D" for owner in schedule
                ),
                "attacker_placements": attacker_quota,
                "defender_placements": defender_quota,
                "last_placement_owner": "A" if schedule[-1] == attacker else "D",
                "attacker_quota_changed": quota_changed,
                "win_same_as_previous_horizon": (
                    horizon > 1 and not quota_changed
                ),
                "opening_forced_origin": phase == "opening" and horizon == 1,
            })
            previous_attacker_quota = attacker_quota

        blocks = []
        start = 1
        for horizon in range(2, 28):
            boundary = (
                horizon == 27
                or entries[horizon - 1]["attacker_placements"]
                != entries[start - 1]["attacker_placements"]
            )
            if boundary:
                end = horizon - 1
                blocks.append({
                    "first_horizon": start,
                    "last_horizon": end,
                    "attacker_placements": entries[start - 1]["attacker_placements"],
                    "win_predicates_equal_within_block": True,
                })
                start = horizon
        out[phase] = {
            "root_placements_made_representative": placements_made,
            "attacker_player": attacker,
            "attacker_rung_horizons": [
                entry["horizon"] for entry in entries
                if entry["attacker_quota_changed"]
            ],
            "constant_attacker_quota_blocks": blocks,
            "entries": entries,
        }
    return out


def build() -> dict:
    known, rows = h10._known_registry()
    cohorts = r2.cohorts()
    cohort_ids = {
        name: {row["pos_id"] for row in cohort_rows}
        for name, cohort_rows in cohorts.items()
    }
    all_ids = sorted(known)
    stamped_ids = sorted(
        pos_id for pos_id in all_ids if known[pos_id]["cert_depth"] is not None
    )
    undated_ids = sorted(set(all_ids) - set(stamped_ids))
    depths = sorted({known[pos_id]["cert_depth"] for pos_id in stamped_ids})

    depth_records = {}
    for depth in depths:
        ids = [
            pos_id for pos_id in stamped_ids
            if known[pos_id]["cert_depth"] == depth
        ]
        depth_records[str(depth)] = exact_set_record(
            ids, known, rows, cohort_ids
        )

    cumulative = {}
    previous_ids: set[str] = set()
    previous_horizon = 0
    for horizon in REQUESTED_CUMULATIVE_HORIZONS:
        ids = {
            pos_id for pos_id in stamped_ids
            if known[pos_id]["cert_depth"] <= horizon
        }
        cumulative[str(horizon)] = {
            **exact_set_record(ids, known, rows, cohort_ids),
            "increment_from_previous_requested_horizon": {
                "after_horizon": previous_horizon,
                **exact_set_record(ids - previous_ids, known, rows, cohort_ids),
            },
        }
        previous_ids = ids
        previous_horizon = horizon

    audit = raw_registry_audit()
    reconstructed_ok = audit["unique_candidate_ids"] == len(known)

    greater_than_12 = [
        pos_id for pos_id in stamped_ids if known[pos_id]["cert_depth"] > 12
    ]
    depths_13_through_18 = [
        pos_id for pos_id in stamped_ids
        if 13 <= known[pos_id]["cert_depth"] <= 18
    ]
    eligible_le_18 = cumulative["18"]["count"]

    code_paths = [
        Path(__file__),
        ROOT / ".scratch" / "horizon_h10.py",
        ROOT / ".scratch" / "horizon_r2.py",
        ROOT / ".scratch" / "deadline_ladder_r.py",
    ]
    return {
        "metadata": {
            "purpose": "exact known-WIN validation ladder for Horizon R4",
            "registry_definition": ".scratch/horizon_h10.py::_known_registry",
            "eligibility_definition": "cert_depth is non-null and cert_depth <= horizon",
            "cert_depth_semantics": (
                "maximum exact-leaf resolution ply in the verified certificate, "
                "minus root placements; it is a sufficient deadline, not a claim "
                "of minimal winning depth"
            ),
            "python": platform.python_version(),
            "requested_cumulative_horizons": list(REQUESTED_CUMULATIVE_HORIZONS),
            "exact_sorted_ids_embedded": True,
            "id_hash_encoding": (
                "SHA-256 of UTF-8 compact JSON array, lexicographically sorted IDs, "
                "ensure_ascii=false"
            ),
            "cohort_count_note": (
                "cohort membership counts may overlap; grinds are a self-play subset "
                "and some IDs occur in multiple frozen cohorts"
            ),
        },
        "code_sources": [file_record(path) for path in code_paths],
        "data_sources": [file_record(path) for path in input_paths()],
        "registry": {
            "unique_known_wins": len(all_ids),
            "depth_stamped": len(stamped_ids),
            "undated": len(undated_ids),
            "all": exact_set_record(all_ids, known, rows, cohort_ids),
            "undated_set": exact_set_record(undated_ids, known, rows, cohort_ids),
            "depths": depth_records,
            "cumulative": cumulative,
        },
        "distribution_landmarks": {
            "literal_depth_gt_12": {
                "count": len(greater_than_12),
                "pct_of_depth_stamped": 100.0 * len(greater_than_12) / len(stamped_ids),
                "pct_of_all_registry": 100.0 * len(greater_than_12) / len(all_ids),
            },
            "depth_13_through_18": {
                "count": len(depths_13_through_18),
                "pct_of_depth_stamped": 100.0 * len(depths_13_through_18) / len(stamped_ids),
                "pct_of_all_registry": 100.0 * len(depths_13_through_18) / len(all_ids),
            },
            "eligible_le_18": {
                "count": eligible_le_18,
                "pct_of_depth_stamped": 100.0 * eligible_le_18 / len(stamped_ids),
                "pct_of_all_registry": 100.0 * eligible_le_18 / len(all_ids),
            },
        },
        "phase_clocks_through_26": clock_table(),
        "phase_aligned_rungs": [
            {
                "fresh_pair": [13, 14],
                "SecondStone_pair": [12, 13],
                "opening_pair_horizons": [12, 13],
                "observed_certificate_depths": {"FirstStone": 14, "SecondStone": 13},
            },
            {
                "fresh_pair": [17, 18],
                "SecondStone_pair": [16, 17],
                "opening_pair_horizons": [16, 17],
                "observed_certificate_depths": {"FirstStone": 18, "SecondStone": 17},
            },
            {
                "fresh_pair": [21, 22],
                "SecondStone_pair": [20, 21],
                "opening_pair_horizons": [20, 21],
                "observed_certificate_depths": {"FirstStone": 22, "SecondStone": 21},
            },
            {
                "fresh_pair": [25, 26],
                "SecondStone_pair": [24, 25],
                "opening_pair_horizons": [24, 25],
                "observed_certificate_depths": {"FirstStone": 26, "SecondStone": 25},
            },
        ],
        "clock_consequences": {
            "fresh_Win22_equals_Win23_equals_Win24": True,
            "SecondStone_Win21_equals_Win22_equals_Win23": True,
            "opening_Win21_equals_Win22_equals_Win23": True,
            "SecondStone_h24_is_new_attacker_placement": True,
            "opening_h24_is_new_attacker_placement": True,
            "global_all_phase_h24_requires_distinct_partial_opening_endpoint": True,
        },
        "deduplication_audit": audit,
        "consistency": {
            "inherited_expected_unique_2941": len(all_ids) == 2941,
            "inherited_expected_depth_stamped_2676": len(stamped_ids) == 2676,
            "inherited_expected_undated_265": len(undated_ids) == 265,
            "raw_reconstruction_unique_count_matches": reconstructed_ok,
            "no_conflicting_nonnull_depths": not audit["conflicting_nonnull_depths"],
            "no_cohort_duplicate_move_conflicts": not audit["cohort_duplicate_move_conflicts"],
            "all_registry_ids_have_rows": all(pos_id in rows for pos_id in all_ids),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / ".scratch" / "horizon_r4_registry.json",
    )
    args = parser.parse_args()
    result = build()
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cumulative = result["registry"]["cumulative"]
    print(json.dumps({
        "out": str(args.out),
        "unique": result["registry"]["unique_known_wins"],
        "depth_stamped": result["registry"]["depth_stamped"],
        "eligible_le_18": cumulative["18"]["count"],
        "eligible_le_22": cumulative["22"]["count"],
        "eligible_le_24": cumulative["24"]["count"],
        "eligible_le_26": cumulative["26"]["count"],
        "consistency": result["consistency"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
