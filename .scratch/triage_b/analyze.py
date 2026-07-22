#!/usr/bin/env python3
"""Transparent threshold analysis for TSS Unknown-classification Phase B.

The candidate family is deliberately small and inspectable: one threshold or
an AND-conjunction of two/three thresholds on distinct features. Thresholds
are a deterministic grid of up to 40 observed values per feature; there is no
fitted model or external ML package.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any


DECISION_POINTS = (100, 150, 200, 300, 500, 1000, 1500)
CLASSES = ("cap_bound", "exhaust", "provable")
PRECISION_BAR = 0.90
PROVABLE_CASUALTY_BAR = 2
PN_INFINITY = 1_000_000_000

FEATURE_DEFINITIONS = {
    "root_pn": "recursively refreshed root proof number",
    "root_dn": "recursively refreshed root disproof number",
    "root_child_count": "number of root children",
    "root_child_pn_sum": "sum of current root-child PN values",
    "root_child_dn_sum": "sum of current root-child DN values",
    "root_child_pn_min": "minimum current root-child PN",
    "root_child_pn_max": "maximum current root-child PN",
    "root_child_dn_min": "minimum current root-child DN",
    "root_child_dn_max": "maximum current root-child DN",
    "root_child_pn_zero": "root children already proven",
    "root_child_dn_zero": "root children already refuted",
    "child_pn_range": "max minus min root-child PN",
    "child_dn_range": "max minus min root-child DN",
    "child_pn_mean": "mean root-child PN",
    "child_dn_mean": "mean root-child DN",
    "top1_pn..top4_dn": "PN/DN of up to four root children ordered by current selection score",
    "top_pn_gap_12": "PN gap between selection-score ranks 1 and 2",
    "top_dn_gap_12": "DN gap between selection-score ranks 1 and 2",
    "open_nodes": "arena entries whose node is still Unexpanded",
    "cutoff_nodes": "arena entries at a staged depth cutoff",
    "max_depth": "maximum placement depth represented in the arena",
    "arena_size": "retained PN arena entries",
    "tt_entries": "positions admitted to the capped exact-key index",
    "tt_hits": "cumulative exact-key transposition hits",
    "tt_admission_rejections": "cumulative index admissions refused by the byte cap",
    "distinct_expanded_nodes": "distinct arena IDs expanded so far",
    "reselected_expansions": "expansions beyond first expansion of an arena ID",
    "max_node_expansions": "largest expansion count for one arena ID",
    "arena_per_expansion": "arena size divided by decision expansion N",
    "tt_index_fraction": "indexed entries divided by arena size",
    "open_fraction": "open nodes divided by arena size",
    "reselection_fraction": "reselected expansions divided by N",
    "distinct_fraction": "distinct expanded nodes divided by N",
    "*_delta_25": "feature change from the preceding 25-expansion snapshot",
    "*_delta_from_25": "feature change from the first snapshot at expansion 25",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def features(row: dict[str, Any], previous: dict[str, Any], first: dict[str, Any]) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        key: value
        for key, value in row.items()
        if key not in {"expansions", "solve_id"} and isinstance(value, (int, float))
    }
    count = max(1, row["root_child_count"])
    arena = max(1, row["arena_size"])
    n = row["expansions"]
    out.update(
        child_pn_range=row["root_child_pn_max"] - row["root_child_pn_min"],
        child_dn_range=row["root_child_dn_max"] - row["root_child_dn_min"],
        child_pn_mean=row["root_child_pn_sum"] / count,
        child_dn_mean=row["root_child_dn_sum"] / count,
        arena_per_expansion=row["arena_size"] / n,
        tt_index_fraction=row["tt_entries"] / arena,
        open_fraction=row["open_nodes"] / arena,
        reselection_fraction=row["reselected_expansions"] / n,
        distinct_fraction=row["distinct_expanded_nodes"] / n,
        arena_growth_25=row["arena_size"] - previous["arena_size"],
        depth_growth_25=row["max_depth"] - previous["max_depth"],
        root_pn_delta_25=row["root_pn"] - previous["root_pn"],
        root_dn_delta_25=row["root_dn"] - previous["root_dn"],
        child_pn_sum_delta_25=row["root_child_pn_sum"] - previous["root_child_pn_sum"],
        child_dn_sum_delta_25=row["root_child_dn_sum"] - previous["root_child_dn_sum"],
        root_pn_delta_from_25=row["root_pn"] - first["root_pn"],
        root_dn_delta_from_25=row["root_dn"] - first["root_dn"],
        depth_delta_from_25=row["max_depth"] - first["max_depth"],
        arena_delta_from_25=row["arena_size"] - first["arena_size"],
    )
    top = row["root_top"]
    for index in range(4):
        for number in ("pn", "dn"):
            out[f"top{index + 1}_{number}"] = top[index][number] if index < len(top) else None
    out["top_pn_gap_12"] = top[1]["pn"] - top[0]["pn"] if len(top) >= 2 else None
    out["top_dn_gap_12"] = top[1]["dn"] - top[0]["dn"] if len(top) >= 2 else None
    return out


def condition_text(condition: tuple[str, str, float | int]) -> str:
    feature, operation, threshold = condition
    if isinstance(threshold, float):
        rendered = f"{threshold:.6g}"
    else:
        rendered = str(threshold)
    return f"{feature} {operation} {rendered}"


def rule_metrics(mask: int, class_masks: dict[str, int], class_totals: dict[str, int]) -> dict[str, Any]:
    predicted = {name: (mask & class_masks[name]).bit_count() for name in CLASSES}
    selected = sum(predicted.values())
    true_positive = predicted["cap_bound"]
    precision = true_positive / selected if selected else 0.0
    recall = true_positive / class_totals["cap_bound"] if class_totals["cap_bound"] else 0.0
    confusion = {
        name: {
            "predicted_cap_bound": predicted[name],
            "predicted_other": class_totals[name] - predicted[name],
        }
        for name in CLASSES
    }
    return {
        "selected": selected,
        "precision": precision,
        "recall": recall,
        "provable_casualties": predicted["provable"],
        "confusion": confusion,
        "meets_bar": precision >= PRECISION_BAR
        and predicted["provable"] <= PROVABLE_CASUALTY_BAR,
    }


def rank_key(metric: dict[str, Any], purpose: str) -> tuple[float, ...]:
    tp = metric["confusion"]["cap_bound"]["predicted_cap_bound"]
    exhaust = metric["confusion"]["exhaust"]["predicted_cap_bound"]
    provable = metric["provable_casualties"]
    precision = metric["precision"]
    recall = metric["recall"]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if purpose == "bar":
        return (float(metric["meets_bar"]), recall, precision, -provable, -exhaust)
    if purpose == "f1":
        return (f1, precision, recall, -provable)
    if purpose.startswith("penalty"):
        weight = float(purpose.removeprefix("penalty"))
        return (tp - weight * (exhaust + 2 * provable), precision, recall)
    if purpose == "casualty":
        return (float(provable <= 2), tp - exhaust, precision, recall)
    raise ValueError(purpose)


def make_atoms(ids: list[str], matrix: dict[str, dict[str, float | int | None]]) -> list[tuple[int, tuple[tuple[str, str, float | int], ...]]]:
    atoms_by_mask: dict[tuple[str, int], tuple[int, tuple[tuple[str, str, float | int], ...]]] = {}
    feature_names = sorted(next(iter(matrix.values())))
    for feature in feature_names:
        observed = sorted({matrix[solve_id][feature] for solve_id in ids if matrix[solve_id][feature] is not None})
        if len(observed) > 40:
            observed = sorted({observed[round(index * (len(observed) - 1) / 39)] for index in range(40)})
        for operation in ("<=", ">="):
            for threshold in observed:
                mask = 0
                for index, solve_id in enumerate(ids):
                    value = matrix[solve_id][feature]
                    if value is not None and (value <= threshold if operation == "<=" else value >= threshold):
                        mask |= 1 << index
                if mask == 0:
                    continue
                condition = (feature, operation, threshold)
                atoms_by_mask.setdefault((feature, mask), (mask, (condition,)))
    return list(atoms_by_mask.values())


def beam_select(
    candidates: list[tuple[int, tuple[tuple[str, str, float | int], ...]]],
    metric,
    per_objective: int = 80,
) -> list[tuple[int, tuple[tuple[str, str, float | int], ...]]]:
    purposes = ("bar", "f1", "penalty0.5", "penalty1", "penalty2", "penalty4", "casualty")
    chosen: dict[tuple[int, tuple[str, ...]], tuple[int, tuple[tuple[str, str, float | int], ...]]] = {}
    for purpose in purposes:
        top = heapq.nlargest(
            per_objective,
            candidates,
            key=lambda candidate: rank_key(metric(candidate[0]), purpose),
        )
        for candidate in top:
            key = (candidate[0], tuple(condition[0] for condition in candidate[1]))
            chosen.setdefault(key, candidate)
    return list(chosen.values())


def analyze_rules(ids: list[str], matrix: dict[str, dict[str, float | int | None]], labels: dict[str, str]) -> dict[str, Any]:
    class_masks = {name: 0 for name in CLASSES}
    for index, solve_id in enumerate(ids):
        class_masks[labels[solve_id]] |= 1 << index
    class_totals = {name: class_masks[name].bit_count() for name in CLASSES}
    metric_cache: dict[int, dict[str, Any]] = {}

    def metric(mask: int) -> dict[str, Any]:
        if mask not in metric_cache:
            metric_cache[mask] = rule_metrics(mask, class_masks, class_totals)
        return metric_cache[mask]

    atoms = make_atoms(ids, matrix)
    atoms_by_feature: dict[str, list[tuple[int, tuple[tuple[str, str, float | int], ...]]]] = collections.defaultdict(list)
    for atom in atoms:
        atoms_by_feature[atom[1][0][0]].append(atom)

    best_by_depth: dict[str, Any] = {}
    broad_by_depth: dict[str, Any] = {}
    beam = beam_select(atoms, metric)
    current = atoms
    for depth in range(1, 4):
        def candidate_record(candidate: tuple[int, tuple[tuple[str, str, float | int], ...]]) -> dict[str, Any]:
            measured = metric(candidate[0])
            return {
                "conditions": [condition_text(condition) for condition in candidate[1]],
                "raw_conditions": [list(condition) for condition in candidate[1]],
                **measured,
            }

        qualifying = [candidate for candidate in current if metric(candidate[0])["meets_bar"]]
        if qualifying:
            best = max(qualifying, key=lambda candidate: rank_key(metric(candidate[0]), "bar"))
            best_by_depth[str(depth)] = candidate_record(best)
        broad = max(current, key=lambda candidate: rank_key(metric(candidate[0]), "f1"))
        broad_by_depth[str(depth)] = candidate_record(broad)
        if depth == 3:
            break

        # Conditions are stored in increasing feature-name order. Extending
        # only with a later feature avoids permutation duplicates and enforces
        # the preregistered distinct-feature limit.
        expanded: list[tuple[int, tuple[tuple[str, str, float | int], ...]]] = []
        for base_mask, conditions in beam:
            last_feature = conditions[-1][0]
            for feature in sorted(name for name in atoms_by_feature if name > last_feature):
                for atom_mask, atom_conditions in atoms_by_feature[feature]:
                    mask = base_mask & atom_mask
                    if mask:
                        expanded.append((mask, conditions + atom_conditions))
        current = expanded
        beam = beam_select(current, metric)

    all_qualifying = [record for record in best_by_depth.values()]
    overall = max(all_qualifying, key=lambda record: (record["recall"], record["precision"], -record["provable_casualties"])) if all_qualifying else None
    all_broad = list(broad_by_depth.values())
    broad_overall = max(
        all_broad,
        key=lambda record: (
            2 * record["precision"] * record["recall"] / (record["precision"] + record["recall"])
            if record["precision"] + record["recall"]
            else 0,
            record["precision"],
        ),
    )
    return {
        "candidate_family": "AND conjunctions of 1-3 numeric thresholds on distinct features; each feature uses a deterministic grid of up to 40 observed values",
        "atoms": len(atoms),
        "best_meeting_bar_by_condition_count": best_by_depth,
        "best_broad_f1_by_condition_count": broad_by_depth,
        "best_meeting_bar_up_to_3": overall,
        "best_broad_f1_up_to_3": broad_overall,
    }


def medians(ids: list[str], matrix: dict[str, dict[str, float | int | None]], labels: dict[str, str]) -> dict[str, dict[str, float | int | None]]:
    selected = (
        "root_pn",
        "root_dn",
        "child_pn_range",
        "child_dn_range",
        "open_nodes",
        "max_depth",
        "arena_size",
        "tt_hits",
        "tt_admission_rejections",
        "reselection_fraction",
        "arena_growth_25",
        "depth_growth_25",
        "root_pn_delta_from_25",
        "root_dn_delta_from_25",
    )
    out: dict[str, dict[str, float | int | None]] = {}
    for class_name in CLASSES:
        rows: dict[str, float | int | None] = {}
        class_ids = [solve_id for solve_id in ids if labels[solve_id] == class_name]
        for feature in selected:
            values = sorted(matrix[solve_id][feature] for solve_id in class_ids if matrix[solve_id][feature] is not None)
            if not values:
                rows[feature] = None
            else:
                middle = len(values) // 2
                rows[feature] = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
        out[class_name] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, default=Path(".scratch/triage_b/trajectory.jsonl"))
    parser.add_argument("--results", type=Path, default=Path(".scratch/triage_b/results.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("raws/lanec_labels.jsonl"))
    parser.add_argument("--positions", type=Path, default=Path("../v1-soak/raws/selfplay_positions.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path(".scratch/triage_b/summary.json"))
    args = parser.parse_args()

    result_rows = load_jsonl(args.results)
    trace_rows = load_jsonl(args.trajectory)
    results = {row["solve_id"]: row for row in result_rows}
    labels = {solve_id: row["class"] for solve_id, row in results.items()}
    snapshots: dict[str, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    for row in trace_rows:
        snapshots[row["solve_id"]][row["expansions"]] = row

    assert len(results) == 248
    assert len(snapshots) == 248
    assert all(expansion % 25 == 0 for rows in snapshots.values() for expansion in rows)
    assert collections.Counter(labels.values()) == {"cap_bound": 94, "exhaust": 97, "provable": 57}

    points: dict[str, Any] = {}
    for decision in DECISION_POINTS:
        live_ids = sorted(solve_id for solve_id, result in results.items() if result["expansions"] >= decision)
        matrix: dict[str, dict[str, float | int | None]] = {}
        sampled_at: dict[str, int] = {}
        for solve_id in live_ids:
            eligible = [expansion for expansion in snapshots[solve_id] if expansion <= decision]
            assert eligible, f"{solve_id}: live at {decision} without a snapshot"
            expansion = max(eligible)
            sampled_at[solve_id] = expansion
            row = snapshots[solve_id][expansion]
            previous = snapshots[solve_id].get(expansion - 25, row)
            first = snapshots[solve_id][min(snapshots[solve_id])]
            matrix[solve_id] = features(row, previous, first)
        live_counts = collections.Counter(labels[solve_id] for solve_id in live_ids)
        points[str(decision)] = {
            "snapshot_rule": "nearest snapshot at or below N",
            "sampled_expansion_min": min(sampled_at.values()),
            "sampled_expansion_max": max(sampled_at.values()),
            "live_counts": {name: live_counts[name] for name in CLASSES},
            "cap_bound_base_rate": live_counts["cap_bound"] / len(live_ids),
            "feature_medians": medians(live_ids, matrix, labels),
            "classifiers": analyze_rules(live_ids, matrix, labels),
        }

    artifact_paths = {
        "trajectory": args.trajectory,
        "results": args.results,
        "labels": args.labels,
        "positions": args.positions,
        "analyzer": Path(__file__),
    }
    summary = {
        "schema_version": 1,
        "measurement": {
            "positions": len(results),
            "trace_rows": len(trace_rows),
            "class_counts": dict(collections.Counter(labels.values())),
            "result_status_by_class": {
                name: dict(collections.Counter(row["status"] for row in result_rows if row["class"] == name))
                for name in CLASSES
            },
            "caps": {"node_cap": 5000, "tt_bytes_cap": 262144, "semantic_horizon": "unbounded", "goal": "win", "cold": True},
            "snapshot_interval": 25,
        },
        "interest_bar": {"precision_min": PRECISION_BAR, "provable_casualties_max": PROVABLE_CASUALTY_BAR},
        "feature_definitions": FEATURE_DEFINITIONS,
        "decision_points": points,
        "artifacts": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in artifact_paths.items()
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.summary), "bytes": args.summary.stat().st_size, "sha256": sha256(args.summary)}, sort_keys=True))


if __name__ == "__main__":
    main()
