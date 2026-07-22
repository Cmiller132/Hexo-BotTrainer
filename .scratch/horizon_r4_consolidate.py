"""Consolidate immutable Horizon R4 phase-1 evidence.

This script performs no search.  It checks shard coverage and recomputes all
rates and inherited-union floors directly from the row-level JSON evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch"
sys.path.insert(0, str(SCRATCH))

import horizon_h10 as h10  # noqa: E402
import horizon_r2 as r2  # noqa: E402

REGISTRY_INITIAL_ROWS = SCRATCH / "horizon_r4_phase1_registry_rows.jsonl"
REGISTRY_FINAL_ROWS = SCRATCH / "horizon_r4_phase1_registry_final_rows.jsonl"
SYNTHETIC_ROWS = SCRATCH / "horizon_r4_phase1_synthetic_rows.jsonl"

COHORT_ROW_FILES = (
    SCRATCH / "horizon_r4_phase1_cohorts_human_rows.jsonl",
    SCRATCH / "horizon_r4_phase1_cohorts_human_resume_rows.jsonl",
    SCRATCH / "horizon_r4_phase1_cohorts_small_rows.jsonl",
    SCRATCH / "horizon_r4_phase1_cohorts_selfplay_1_rows.jsonl",
    SCRATCH / "horizon_r4_phase1_cohorts_selfplay_2_rows.jsonl",
    SCRATCH / "horizon_r4_phase1_cohorts_selfplay_3_rows.jsonl",
    SCRATCH / "horizon_r4_phase1_cohorts_selfplay_4_rows.jsonl",
)

SUMMARY_FILES = (
    SCRATCH / "horizon_r4_phase1_registry.json",
    SCRATCH / "horizon_r4_phase1_registry_final.json",
    SCRATCH / "horizon_r4_phase1_retry.json",
    SCRATCH / "horizon_r4_phase1_retry2.json",
    SCRATCH / "horizon_r4_phase1_retry3.json",
    SCRATCH / "horizon_r4_phase1_retry4.json",
    SCRATCH / "horizon_r4_phase1_synthetic.json",
    SCRATCH / "horizon_r4_phase1_cohorts_small.json",
    SCRATCH / "horizon_r4_phase1_cohorts_human_resume.json",
    SCRATCH / "horizon_r4_phase1_cohorts_selfplay_1.json",
    SCRATCH / "horizon_r4_phase1_cohorts_selfplay_2.json",
    SCRATCH / "horizon_r4_phase1_cohorts_selfplay_3.json",
    SCRATCH / "horizon_r4_phase1_cohorts_selfplay_4.json",
)

COHORT_TOTALS = {
    "human_v1": 2720,
    "selfplay_v1": 3255,
    "puzzle_v3": 468,
    "grinds": 248,
    "forcing19": 19,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def compact_id_hash(ids: Iterable[str]) -> str:
    raw = json.dumps(sorted(set(ids)), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest().upper()


def distribution(rows: list[dict], key: str) -> dict[str, int | float]:
    values = sorted(int(row[key]) for row in rows if row.get(key) is not None)
    if not values:
        return {"n": 0, "p50": 0, "p90": 0, "max": 0, "total": 0, "mean": 0.0}
    at = lambda q: values[int(q * (len(values) - 1))]
    return {
        "n": len(values),
        "p50": at(0.5),
        "p90": at(0.9),
        "max": values[-1],
        "total": sum(values),
        "mean": sum(values) / len(values),
    }


def status_summary(rows: list[dict]) -> dict:
    statuses = Counter(row.get("status", "missing") for row in rows)
    completed = [row for row in rows if row.get("status") in ("win", "negative")]
    return {
        "attempted_memberships": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "wins": statuses["win"],
        "completed_negatives": statuses["negative"],
        "timeouts": statuses["timeout"],
        "errors": statuses["error"] + statuses["prep_error"],
        "completed_verdicts": len(completed),
        "native_win_rate_over_attempted": statuses["win"] / len(rows) if rows else 0.0,
        "timeout_rate_over_attempted": statuses["timeout"] / len(rows) if rows else 0.0,
        "distributions": {
            scope: {
                key: distribution(selected, key)
                for key in ("wall_ns", "model_wall_ns", "nodes", "universe", "first_action_total")
            }
            for scope, selected in (("all", rows), ("completed", completed))
        },
    }


def history_key(row: dict) -> tuple[tuple[int, int], ...]:
    return tuple((int(cell[0]), int(cell[1])) for cell in row["moves"])


def main() -> None:
    registry_initial = read_jsonl(REGISTRY_INITIAL_ROWS)
    registry_final = read_jsonl(REGISTRY_FINAL_ROWS)
    synthetic = read_jsonl(SYNTHETIC_ROWS)
    cohort_rows = [row for path in COHORT_ROW_FILES for row in read_jsonl(path)]

    assert len(registry_initial) == len({row["id"] for row in registry_initial}) == 155
    assert len(registry_final) == len({row["id"] for row in registry_final}) == 155
    assert {row["id"] for row in registry_initial} == {row["id"] for row in registry_final}
    assert Counter(row["status"] for row in registry_initial) == Counter({"win": 137, "timeout": 18})
    assert Counter(row["status"] for row in registry_final) == Counter({"win": 155})
    assert all(row.get("caught") is True for row in registry_final)
    assert len(synthetic) == 4
    assert Counter(row["status"] for row in synthetic) == Counter({"win": 3, "negative": 1})
    assert all(row.get("python_match") is True for row in synthetic)

    by_cohort: dict[str, list[dict]] = defaultdict(list)
    for row in cohort_rows:
        by_cohort[row["cohort"]].append(row)
    expected_supported = {
        "human_v1": 2720,
        "selfplay_v1": 3207,
        "puzzle_v3": 468,
        "grinds": 248,
        "forcing19": 19,
    }
    assert set(by_cohort) == set(expected_supported)
    for name, expected in expected_supported.items():
        ids = [row["id"] for row in by_cohort[name]]
        assert len(ids) == len(set(ids)) == expected, (name, len(ids), len(set(ids)), expected)

    h8 = read_json(SCRATCH / "horizon_r3_h8.json")["h8_battery"]["cohorts"]
    registry = read_json(SCRATCH / "horizon_r4_registry.json")
    registry_le14 = set(registry["registry"]["cumulative"]["14"]["ids"])
    known, registry_rows = h10._known_registry()
    source_cohorts = r2.cohorts()
    assert set(COHORT_TOTALS) <= set(source_cohorts)
    source_by_cohort = {
        name: {row["pos_id"]: row for row in source_cohorts[name]}
        for name in COHORT_TOTALS
    }
    assert set(known) == set(registry["registry"]["all"]["ids"])
    registry_le14_history_keys = {
        history_key(registry_rows[pos_id]) for pos_id in registry_le14
    }
    registry_le10 = {
        pos_id for pos_id, info in known.items()
        if info["cert_depth"] is not None and int(info["cert_depth"]) <= 10
    }
    registry_le10_history_keys = {
        history_key(registry_rows[pos_id]) for pos_id in registry_le10
    }
    registry_known_history_keys = {
        history_key(registry_rows[pos_id]) for pos_id in known
    }

    global_h8_win_ids = {
        row["pos_id"]
        for name in ("human_v1", "selfplay_v1", "puzzle_v3", "grinds")
        for row in h8[name]["rows"]
        if row["current_win"]
    }
    global_h8_history_keys = {
        history_key(source_by_cohort[name][row["pos_id"]])
        for name in ("human_v1", "selfplay_v1", "puzzle_v3", "grinds")
        for row in h8[name]["rows"]
        if row["current_win"]
    }
    semantic_baseline_keys = global_h8_history_keys | registry_le14_history_keys
    semantic_h10_baseline_keys = global_h8_history_keys | registry_le10_history_keys

    full_ids: dict[str, set[str]] = {}
    h8_wins: dict[str, set[str]] = {}
    for name in ("human_v1", "selfplay_v1", "puzzle_v3", "grinds"):
        rows = h8[name]["rows"]
        full_ids[name] = {row["pos_id"] for row in rows}
        h8_wins[name] = {row["pos_id"] for row in rows if row["current_win"]}
        assert len(full_ids[name]) == COHORT_TOTALS[name]
    full_ids["forcing19"] = {row["id"] for row in by_cohort["forcing19"]}
    h8_wins["forcing19"] = set()

    cohorts = {}
    for name in COHORT_TOTALS:
        measured = by_cohort[name]
        attempted_ids = {row["id"] for row in measured}
        native_wins = {row["id"] for row in measured if row["status"] == "win"}
        known_here = registry_le14 & full_ids[name]
        known10_here = registry_le10 & full_ids[name]
        literal_h10_union = h8_wins[name] | known10_here
        semantic_h10_inherited = {
            pos_id for pos_id in full_ids[name]
            if history_key(source_by_cohort[name][pos_id]) in semantic_h10_baseline_keys
        }
        inherited_union = h8_wins[name] | known_here
        after_union = inherited_union | native_wins
        native_new = native_wins - inherited_union
        source_rows = source_by_cohort[name]
        semantic_inherited = {
            pos_id for pos_id in full_ids[name]
            if history_key(source_rows[pos_id]) in semantic_baseline_keys
        }
        semantic_native_new = {
            pos_id for pos_id in native_wins
            if history_key(source_rows[pos_id]) not in semantic_baseline_keys
        }
        semantic_after = semantic_inherited | native_wins
        assert attempted_ids <= full_ids[name]
        cohorts[name] = {
            "frozen_memberships": COHORT_TOTALS[name],
            "supported_attempted": len(attempted_ids),
            "unsupported_opening": COHORT_TOTALS[name] - len(attempted_ids),
            "budget_ms": 25 if name in ("puzzle_v3", "grinds", "forcing19") else 10,
            **status_summary(measured),
            "native_win_ids": sorted(native_wins),
            "native_win_ids_sha256_compact_json": compact_id_hash(native_wins),
            "h8_exact_wins": len(h8_wins[name]),
            "registry_le14_members": len(known_here),
            "literal_h8_or_registry_le10_floor": len(literal_h10_union),
            "position_history_h8_or_registry_le10_floor": len(semantic_h10_inherited),
            "position_history_h10_plus_native_floor": len(semantic_h10_inherited | native_wins),
            "position_history_native_new_beyond_h10": len(native_wins - semantic_h10_inherited),
            "position_history_native_new_beyond_h10_ids": sorted(native_wins - semantic_h10_inherited),
            "inherited_h8_or_registry_floor": len(inherited_union),
            "native_new_beyond_h8_or_registry": len(native_new),
            "native_new_ids": sorted(native_new),
            "r4_union_floor": len(after_union),
            "r4_union_floor_rate_over_frozen": len(after_union) / COHORT_TOTALS[name],
            "position_history_inherited_floor": len(semantic_inherited),
            "position_history_inherited_ids": sorted(semantic_inherited),
            "position_history_native_new": len(semantic_native_new),
            "position_history_native_new_ids": sorted(semantic_native_new),
            "position_history_r4_union_floor": len(semantic_after),
            "position_history_r4_union_rate_over_frozen": len(semantic_after) / COHORT_TOTALS[name],
        }

    expected_union = {
        "human_v1": (176, 0, 176),
        "selfplay_v1": (107, 0, 107),
        "puzzle_v3": (23, 4, 27),
        "grinds": (0, 0, 0),
        "forcing19": (0, 0, 0),
    }
    for name, (before, new, after) in expected_union.items():
        actual = cohorts[name]
        assert (actual["inherited_h8_or_registry_floor"], actual["native_new_beyond_h8_or_registry"], actual["r4_union_floor"]) == (before, new, after)

    semantic_expected_new = {name: 0 for name in COHORT_TOTALS}
    assert {
        name: cohorts[name]["position_history_native_new"] for name in COHORT_TOTALS
    } == semantic_expected_new
    h10_expected = {
        "forcing19": (0, 0, 0),
        "grinds": (0, 0, 0),
        "human_v1": (163, 163, 164),
        "puzzle_v3": (21, 23, 27),
        "selfplay_v1": (102, 102, 102),
    }
    for name, expected in h10_expected.items():
        actual = cohorts[name]
        assert (
            actual["literal_h8_or_registry_le10_floor"],
            actual["position_history_h8_or_registry_le10_floor"],
            actual["position_history_h10_plus_native_floor"],
        ) == expected

    initial_summary = status_summary(registry_initial)
    final_summary = status_summary(registry_final)
    retry_ladder = []
    for path in SUMMARY_FILES[2:6]:
        data = read_json(path)
        retry_ladder.append({
            "file": str(path.relative_to(ROOT)),
            "budget_ms": data["metadata"]["per_root_ms"],
            "max_cache_entries": data["metadata"]["max_cache_entries"],
            **{key: data["summary"][key] for key in ("attempted", "wins", "timeouts", "completed_negatives", "errors")},
        })

    all_status = status_summary(cohort_rows)
    all_status["unique_position_ids"] = len({row["id"] for row in cohort_rows})
    all_status["duplicate_cross_cohort_memberships"] = len(cohort_rows) - all_status["unique_position_ids"]

    hash_paths = (
        Path(__file__),
        SCRATCH / "horizon_native" / "Cargo.toml",
        SCRATCH / "horizon_native" / "Cargo.lock",
        SCRATCH / "horizon_native" / "README.md",
        SCRATCH / "horizon_native" / "driver.py",
        SCRATCH / "horizon_native" / "src" / "lib.rs",
        SCRATCH / "horizon_native" / "src" / "main.rs",
        SCRATCH / "horizon_native" / ".target" / "release" / "horizon_native.exe",
        SCRATCH / "horizon_h10.py",
        SCRATCH / "horizon_r2.py",
        SCRATCH / "horizon_r3_h8.json",
        SCRATCH / "horizon_r4_registry.json",
        SCRATCH / "horizon_r4_python_boundary.py",
        SCRATCH / "horizon_r4_python_boundary.json",
        REGISTRY_INITIAL_ROWS,
        REGISTRY_FINAL_ROWS,
        SYNTHETIC_ROWS,
        *COHORT_ROW_FILES,
        *SUMMARY_FILES,
    )

    payload = {
        "metadata": {
            "schema": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "claim_class": "MEASURED unless a field explicitly says otherwise",
            "timeout_is_not_a_verdict": True,
            "native_threads": 1,
            "cohort_membership_overlap_note": "grinds/forcing/puzzle memberships can share position IDs with other cohorts; per-cohort rates use memberships",
        },
        "registry_validation": {
            "eligible": 155,
            "initial_compiled_pass": initial_summary,
            "retry_ladder": retry_ladder,
            "canonical_final_pass": final_summary,
            "caught": 155,
            "missed": 0,
            "negative_mismatches": 0,
            "canonical_id_sha256_compact_json": compact_id_hash(row["id"] for row in registry_final),
            "initial_to_final_p50_wall_speedup": initial_summary["distributions"]["all"]["wall_ns"]["p50"] / final_summary["distributions"]["all"]["wall_ns"]["p50"],
            "initial_to_final_p90_wall_speedup": initial_summary["distributions"]["all"]["wall_ns"]["p90"] / final_summary["distributions"]["all"]["wall_ns"]["p90"],
            "development_comparison_caveat": "initial and final runs used different cache limits and evolving source/binary snapshots; ratios describe the preserved development evidence, not a reproducible controlled benchmark",
        },
        "synthetic_python_crosscheck": {
            "cases": len(synthetic),
            "native_wins": sum(row["status"] == "win" for row in synthetic),
            "native_negatives": sum(row["status"] == "negative" for row in synthetic),
            "python_matches": sum(row.get("python_match") is True for row in synthetic),
            "mismatches": [row["id"] for row in synthetic if row.get("python_match") is not True],
            "case_results": [
                {
                    key: row.get(key)
                    for key in (
                        "id", "horizon", "phase", "status", "nodes",
                        "first_actions", "shortcut_d1_defender_completion",
                        "python_status", "python_nodes", "wall_ns", "python_wall_ns",
                    )
                }
                for row in synthetic
            ],
            "negative_legality_note": "the cross7 target has two-cell completion replies whose cells are within five of root stones; every Python D1 counterreply used by the proof therefore exists under true radius-eight legality",
            "summary": status_summary(synthetic),
        },
        "frozen_cohorts": {
            "all_supported_memberships": all_status,
            "per_cohort": cohorts,
            "budget_note": "human/self-play used 10 ms and 50,000 cache entries; puzzle/grinds/forcing used 25 ms and 100,000 entries; safe-point wall overruns remain timeouts",
            "interruption_note": "human shard 1 was killed by the 900 s shell cap after 2,134 durable rows; the disjoint 586-row resume completed",
            "identity_note": "literal pos_id floors preserve the R3 accounting convention; position-history floors transfer a proof across byte-identical ordered move lists and are the semantic figures",
        },
        "position_identity_audit": {
            "registry_ids": len(known),
            "registry_unique_ordered_move_histories": len(registry_known_history_keys),
            "registry_duplicate_id_excess_over_histories": len(known) - len(registry_known_history_keys),
            "registry_le14_ids": len(registry_le14),
            "registry_le14_unique_ordered_move_histories": len(registry_le14_history_keys),
            "registry_le10_ids": len(registry_le10),
            "registry_le10_unique_ordered_move_histories": len(registry_le10_history_keys),
            "global_h8_win_ids": len(global_h8_win_ids),
            "global_h8_unique_ordered_move_histories": len(global_h8_history_keys),
        },
        "integrity": {
            "files_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path)
                for path in hash_paths
            }
        },
    }

    out = SCRATCH / "horizon_r4_phase1.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "PHASE1_EVIDENCE_OK "
        f"registry={payload['registry_validation']['caught']}/155 "
        f"cohort_memberships={all_status['attempted_memberships']} "
        f"wins={all_status['wins']} negatives={all_status['completed_negatives']} "
        f"timeouts={all_status['timeouts']}"
    )


if __name__ == "__main__":
    main()
