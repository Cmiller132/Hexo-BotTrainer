import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (ROOT / "profile.jsonl").read_text().splitlines()]
by_cap = defaultdict(list)
for row in rows:
    by_cap[row["cap"]].append(row)

phases = [
    "attacker_generation",
    "second_candidate",
    "window",
    "defender_generation",
    "defender_plan",
    "pn_select_backprop",
    "tt",
    "state_make_unmake",
    "certificate",
    "setup",
    "other",
]


def median(values):
    return statistics.median(values)


def summarize_cap(cap_rows):
    reps = len(cap_rows[0]["baseline_wall_nanos"])
    battery_wall = [
        sum(row["baseline_wall_nanos"][rep] for row in cap_rows) for rep in range(reps)
    ]
    total_nodes = sum(row["nodes"] for row in cap_rows)
    phase_total = sum(row["phase_total_nanos"] for row in cap_rows)
    phase_nanos = {
        phase: sum(row[f"phase_{phase}_nanos"] for row in cap_rows) for phase in phases
    }
    outcomes = {}
    for status in ["win", "loss", "unknown_at_cap", "width_exhaust"]:
        selected = [row for row in cap_rows if row["final_status"] == status]
        wall = [
            sum(row["baseline_wall_nanos"][rep] for row in selected) for rep in range(reps)
        ]
        outcomes[status] = {
            "rows": len(selected),
            "nodes": sum(row["nodes"] for row in selected),
            "node_share_pct": 100 * sum(row["nodes"] for row in selected) / total_nodes,
            "wall_seconds": [value / 1e9 for value in wall],
            "wall_seconds_median": median(wall) / 1e9,
            "wall_share_pct_by_rep": [
                100 * value / battery_wall[rep] for rep, value in enumerate(wall)
            ],
            "wall_share_pct_median": median(
                [100 * value / battery_wall[rep] for rep, value in enumerate(wall)]
            ),
        }
    top20 = []
    for row in sorted(
        cap_rows, key=lambda item: median(item["baseline_wall_nanos"]), reverse=True
    )[:20]:
        item = {
            "set": row["set"],
            "pos_id": row["pos_id"],
            "final_status": row["final_status"],
            "nodes": row["nodes"],
            "root_pn": row["root_pn"],
            "root_dn": row["root_dn"],
            "wall_ms_samples": [value / 1e6 for value in row["baseline_wall_nanos"]],
            "wall_ms_median": median(row["baseline_wall_nanos"]) / 1e6,
        }
        top20.append(item)
    return {
        "rows": len(cap_rows),
        "nodes": total_nodes,
        "battery_wall_seconds": [value / 1e9 for value in battery_wall],
        "battery_wall_seconds_median": median(battery_wall) / 1e9,
        "ns_per_node_by_rep": [value / total_nodes for value in battery_wall],
        "ns_per_node_median": median(battery_wall) / total_nodes,
        "profiled_wall_seconds": sum(row["profiled_wall_nanos"] for row in cap_rows) / 1e9,
        "phase_total_seconds": phase_total / 1e9,
        "profile_clock_gap_seconds": sum(
            row["profile_clock_gap_nanos"] for row in cap_rows
        )
        / 1e9,
        "profile_overhead_vs_baseline_median_pct": 100
        * (sum(row["profiled_wall_nanos"] for row in cap_rows) - median(battery_wall))
        / median(battery_wall),
        "phases": {
            phase: {
                "seconds": phase_nanos[phase] / 1e9,
                "share_pct": 100 * phase_nanos[phase] / phase_total,
                "nanos_per_node": phase_nanos[phase] / total_nodes,
            }
            for phase in phases
        },
        "phase_share_sum_pct": sum(100 * value / phase_total for value in phase_nanos.values()),
        "outcomes": outcomes,
        "top20": top20,
    }


summary = {"caps": {str(cap): summarize_cap(cap_rows) for cap, cap_rows in by_cap.items()}}

cap500 = {(row["set"], row["pos_id"]): row for row in by_cap[500]}
cap750_unknown = [row for row in by_cap[750] if row["final_status"] == "unknown_at_cap"]
frozen = []
for row in cap750_unknown:
    prior = cap500[(row["set"], row["pos_id"])]
    if (
        prior["final_status"] == "unknown_at_cap"
        and prior["root_pn"] == row["root_pn"]
    ):
        frozen.append(row)
frozen_keys = {(row["set"], row["pos_id"]) for row in frozen}
frozen_500 = [row for row in by_cap[500] if (row["set"], row["pos_id"]) in frozen_keys]
frozen_750_wall = [
    sum(row["baseline_wall_nanos"][rep] for row in frozen)
    for rep in range(len(frozen[0]["baseline_wall_nanos"]))
]
frozen_500_wall = [
    sum(row["baseline_wall_nanos"][rep] for row in frozen_500)
    for rep in range(len(frozen[0]["baseline_wall_nanos"]))
]
frozen_increment_wall = [
    right - left for left, right in zip(frozen_500_wall, frozen_750_wall)
]
summary["stagnation_500_to_750"] = {
    "definition": "UNKNOWN at both caps with identical endpoint root pn",
    "expansion_increment": 250,
    "cap750_unknown_rows": len(cap750_unknown),
    "frozen_rows": len(frozen),
    "frozen_row_share_pct": 100 * len(frozen) / len(cap750_unknown),
    "frozen_cap500_wall_seconds": [value / 1e9 for value in frozen_500_wall],
    "frozen_cap750_wall_seconds": [value / 1e9 for value in frozen_750_wall],
    "frozen_cap750_wall_seconds_median": median(frozen_750_wall) / 1e9,
    "frozen_increment_wall_seconds": [value / 1e9 for value in frozen_increment_wall],
    "frozen_increment_wall_seconds_median": median(frozen_increment_wall) / 1e9,
    "frozen_increment_wall_share_pct": 100
    * median(frozen_increment_wall)
    / summary["caps"]["750"]["battery_wall_seconds_median"]
    / 1e9,
    "frozen_cap750_wall_share_pct": 100
    * median(frozen_750_wall)
    / median(
        [
            sum(row["baseline_wall_nanos"][rep] for row in by_cap[750])
            for rep in range(len(frozen[0]["baseline_wall_nanos"]))
        ]
    ),
}

phase_deltas = {}
for phase in phases:
    left = summary["caps"]["500"]["phases"][phase]
    right = summary["caps"]["750"]["phases"][phase]
    phase_deltas[phase] = {
        "share_pct_point_change": right["share_pct"] - left["share_pct"],
        "nanos_per_node_change": right["nanos_per_node"] - left["nanos_per_node"],
        "nanos_per_node_ratio": right["nanos_per_node"] / left["nanos_per_node"]
        if left["nanos_per_node"]
        else None,
    }
summary["phase_deltas_500_to_750"] = phase_deltas

(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
