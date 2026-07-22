#!/usr/bin/env python3
"""Batch/model bridge for the dependency-free Horizon native kernel.

The Rust process intentionally knows only the finite R3 interaction model.
This driver owns repository discovery, registry/cohort selection, atlas move
ordering, incremental JSONL evidence, and optional Python parity probes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, TextIO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / ".scratch"))

import deadline_ladder_r as phase_r  # noqa: E402
import horizon_h10 as h10  # noqa: E402
import horizon_r2 as r2  # noqa: E402
import horizon_r3 as r3  # noqa: E402


# Ordering only.  These are the next current-player placements in the original
# frozen continuation for the 18 roots that reached the first 10-second native
# boundary.  They are never used to prune an action or claim a proof.
CONTINUATION_HINTS: dict[str, list[list[int]]] = {
    "human_162b07d9a12084e2_p61": [[5, -8], [8, -8]],
    "human_194ce5fa25e2029d_p40": [[13, 1]],
    "human_1d7540b8789196ad_p54": [[-9, 6]],
    "human_20bea7804fffee60_p15": [[-4, 0], [1, -5]],
    "human_27057abe4ee1d907_p15": [[3, -2], [3, -1]],
    "human_37979e171ebe14b2_p79": [[11, -1], [10, 0]],
    "human_3c9423b1a047e24c_p22": [[3, -9]],
    "human_41e78c67c2ac8570_p20": [[2, -2]],
    "human_6023b2ef70e3ffc6_p76": [[-5, 13]],
    "human_736f572521236d55_p17": [[1, -4], [4, -4]],
    "human_c77600999ef0c953_p45": [[5, 0], [8, -3]],
    "human_c77600999ef0c953_p49": [[8, -1], [8, 0]],
    "human_d4f2b0b1853d8565_p44": [[20, -5]],
    "sp_0_p51": [[2, -7], [3, -8]],
    "sp_13_p59": [[9, -6], [9, -5]],
    "sp_35_p27": [[-4, 3], [-6, 6]],
    "sp_4_p79": [[7, -2], [7, -4]],
    "sp_5_p61": [[2, -9], [-1, -3]],
}

# Full A1 actions whose root and post-first-placement child were both accepted
# by the existing truth-pass verifier at the requested bounded depth.  They are
# still ordering hints only; no action is removed from the native search.
VERIFIED_ACTION_HINTS: dict[str, list[list[int]]] = {
    "human_20bea7804fffee60_p15": [[-1, 0], [-1, 2]],
    "human_6023b2ef70e3ffc6_p76": [[6, 13]],
    "sp_13_p59": [[9, -5], [6, -5]],
    "sp_35_p27": [[-1, 0], [-7, 8]],
    "sp_4_p79": [[-4, 4], [-2, 4]],
    "sp_5_p61": [[-1, -3], [3, -5]],
    "sp_0_p51": [[0, -6], [2, -8]],
}

# A verifier Choice proves only this first placement.  PREF_CELL moves the
# entire containing-pair block forward while retaining the exhaustive tail.
CERT_REQUIRED_HINTS: dict[str, list[int]] = {
    "sp_13_p59": [9, -5],
    "sp_35_p27": [-1, 0],
    "sp_4_p79": [-4, 4],
    "sp_5_p61": [-1, -3],
    "sp_0_p51": [0, -6],
}


def mask_indices(mask: int) -> list[int]:
    out = []
    while mask:
        bit = mask & -mask
        mask ^= bit
        out.append(bit.bit_length() - 1)
    return out


def load_atlas() -> dict[str, dict]:
    try:
        rows = json.loads(phase_r.DEFAULT_ATLAS.read_text(encoding="utf-8"))["rows"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return {}
    return {str(row["id"]): row for row in rows}


def atlas_row(pos_id: str, atlas: dict[str, dict]) -> dict | None:
    """Resolve both registry `atlas_full_` and frozen `atlas_` aliases."""
    candidates = [pos_id]
    for prefix in ("atlas_full_", "atlas_"):
        if pos_id.startswith(prefix):
            candidates.append(pos_id[len(prefix):])
    for candidate in candidates:
        if candidate in atlas:
            return atlas[candidate]
    return None


def preferred_cells(
    root: dict,
    model: r3.NextModel,
    atlas: dict[str, dict],
) -> tuple[tuple[int, ...] | None, str | None]:
    phase = phase_r.phase_player(len(root["moves"]))[0]
    manual = root.get("preferred_first_cells")
    verified = VERIFIED_ACTION_HINTS.get(root["pos_id"])
    row = atlas_row(root["pos_id"], atlas)
    atlas_line = row.get("win_line") if row else None
    continuation = CONTINUATION_HINTS.get(root["pos_id"])
    line = manual or verified or atlas_line or continuation
    source = (
        "synthetic_completion" if manual else
        "verified_root_child_resolve" if verified else
        "atlas_win_line" if atlas_line else
        "frozen_continuation" if continuation else None
    )
    if not line:
        return None, None
    wanted = 1 if phase == "second" else 2
    coords = [tuple(cell) for cell in line[:wanted]]
    index = {cell: i for i, cell in enumerate(model.cells)}
    if len(coords) != wanted or any(cell not in index for cell in coords):
        return None, None
    return tuple(index[cell] for cell in coords), source


def write_model(
    stream: TextIO,
    row: dict,
    horizon: int,
    timeout_ms: int,
    atlas: dict[str, dict],
) -> tuple[r3.NextModel, tuple[int, ...] | None, str | None, int | None]:
    model = r3.build_next_model(row, horizon)
    phase, _, _ = phase_r.phase_player(len(row["moves"]))
    if phase not in ("first", "second"):
        raise ValueError(f"unsupported root phase: {phase}")
    board = {tuple(cell): phase_r.owner_at(i) for i, cell in enumerate(row["moves"])}
    legal = r3.legal_cells(board)
    stream.write(f"MODEL {row['pos_id']} {horizon} {phase} {timeout_ms}\n")
    for i, (q, r) in enumerate(model.cells):
        anchored = int(bool(model.anchored_mask & (1 << i)))
        stream.write(f"CELL {q} {r} {anchored} {int((q, r) in legal)}\n")
    for edge in model.target_anchored:
        stream.write("TA " + " ".join(map(str, mask_indices(edge))) + "\n")
    for edge in model.opponent_anchored:
        stream.write("OA " + " ".join(map(str, mask_indices(edge))) + "\n")
    for edge in model.near_windows:
        stream.write("NE " + " ".join(map(str, mask_indices(edge))) + "\n")
    preferred, preferred_source = preferred_cells(row, model, atlas)
    index = {cell: i for i, cell in enumerate(model.cells)}
    required_coord = tuple(CERT_REQUIRED_HINTS[row["pos_id"]]) if row["pos_id"] in CERT_REQUIRED_HINTS else None
    preferred_required = index.get(required_coord) if required_coord is not None else None
    if preferred and (preferred_required is None or preferred_required in preferred):
        stream.write("PREF " + " ".join(map(str, preferred)) + "\n")
    if preferred_required is not None:
        stream.write(f"PREF_CELL {preferred_required}\n")
    stream.write("END\n")
    stream.flush()
    return model, preferred, preferred_source, preferred_required


def default_binary() -> Path:
    name = "horizon_native.exe" if os.name == "nt" else "horizon_native"
    return HERE / ".target" / "release" / name


def synthetic_rows() -> list[tuple[dict, int]]:
    # Three positions are nonterminal and have a one-placement completion for
    # the side to move.  The fourth is a fresh h13 negative: player 0's
    # seven-stone cross has four-cover number > 2 for its two-cell completion
    # family, so every A1 pair has a true-radius-eight legal D1 terminal reply.
    # It exercises the universal D1 layer in both implementations without
    # relying on Python R3's known illegal-fringe overgeneration.
    fresh = {
        "pos_id": "synthetic_fresh_five",
        "preferred_first_cells": [[5, 2], [5, 1]],
        "moves": [
            [0, 0], [0, 2], [1, 2], [-1, 0], [-1, 1], [2, 2], [3, 2],
            [0, -1], [1, -1], [4, 2], [2, 1], [2, -2], [3, -2],
        ],
    }
    second = {
        "pos_id": "synthetic_second_five",
        "preferred_first_cells": [[5, 0]],
        "moves": [
            [0, 0], [0, 1], [1, 1], [1, 0], [2, 0], [2, 1], [3, 1],
            [3, 0], [4, 0], [4, 1], [2, 2], [-1, -1],
        ],
    }
    cross7_negative = {
        "pos_id": "synthetic_fresh_cross7_negative",
        "moves": [
            [0, 0], [1, 1], [2, 1], [0, 1], [0, 2], [1, 2], [2, 2],
            [0, 3], [1, 0], [3, 1], [1, 3], [2, 0], [3, 0],
        ],
    }
    return [
        (fresh, 13),
        ({**fresh, "pos_id": "synthetic_fresh_five_h14"}, 14),
        (second, 13),
        (cross7_negative, 13),
    ]


def select_registry() -> tuple[list[tuple[dict, int, dict]], dict]:
    known, rows = h10._known_registry()
    selected = []
    for pos_id in sorted(known):
        depth = known[pos_id]["cert_depth"]
        if depth in (13, 14):
            selected.append((rows[pos_id], int(depth), {"cert_depth": int(depth), "expected": "win"}))
    return selected, {"registry_unique": len(known), "eligible_depth_13_14": len(selected)}


def select_cohorts(names: set[str] | None) -> tuple[list[tuple[dict, int, dict]], dict]:
    selected = []
    skipped = Counter()
    available = r2.cohorts()
    for cohort, rows in available.items():
        if names and cohort not in names:
            continue
        for row in rows:
            phase = phase_r.phase_player(len(row["moves"]))[0]
            if phase == "first":
                horizon = 14
            elif phase == "second":
                horizon = 13
            else:
                skipped["opening_unsupported"] += 1
                continue
            selected.append((row, horizon, {"cohort": cohort, "root_phase": phase}))
    return selected, {"selected": len(selected), "skipped": dict(skipped), "available_cohorts": sorted(available)}


def retry_ids(explicit: list[str] | None, prior_timeouts: Path | None) -> set[str]:
    selected = set(explicit or ())
    if prior_timeouts is None:
        return selected
    text = prior_timeouts.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    selected.update(
        str(row["id"])
        for row in rows
        if row.get("status") == "timeout" and row.get("id") is not None
    )
    return selected


def evidence_ids(path: Path | None) -> set[str]:
    """Return every position ID already present in a JSONL/JSON evidence file."""
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    rows = json.loads(text) if stripped.startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    return {str(row["id"]) for row in rows if row.get("id") is not None}


def run_native(
    cases: Iterable[tuple[dict, int, dict]],
    *,
    binary: Path,
    timeout_ms: int,
    max_cache: int,
    out_path: Path,
    atlas: dict[str, dict],
    limit: int | None,
    python_crosscheck: bool,
) -> tuple[list[dict], dict]:
    command = [str(binary), "--timeout-ms", str(timeout_ms), "--max-cache", str(max_cache)]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write("HORIZON_NATIVE_V1\n")
    process.stdin.flush()
    results = []
    counts = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", buffering=1) as evidence:
        for number, (row, horizon, metadata) in enumerate(cases, 1):
            if limit is not None and number > limit:
                break
            started = time.perf_counter_ns()
            try:
                model, preferred, preferred_source, preferred_required = write_model(process.stdin, row, horizon, timeout_ms, atlas)
            except Exception as exc:  # preserve the rest of a long cohort
                record = {
                    "id": row.get("pos_id", f"row_{number}"),
                    "horizon": horizon,
                    "status": "prep_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    **metadata,
                }
                evidence.write(json.dumps(record, sort_keys=True) + "\n")
                results.append(record)
                counts["prep_error"] += 1
                continue
            raw = process.stdout.readline()
            if not raw:
                raise RuntimeError(f"native process ended before result {number}; exit={process.poll()}")
            record = json.loads(raw)
            record.update(metadata)
            record["model_wall_ns"] = time.perf_counter_ns() - started - int(record.get("wall_ns", 0))
            record["preferred_first_action"] = list(preferred) if preferred else None
            record["preferred_first_action_source"] = preferred_source
            record["preferred_required_cell"] = preferred_required
            record["preferred_required_cell_source"] = "verifier_certificate_choice" if preferred_required is not None else None
            record["physical_universe_python"] = len(model.cells)
            record["quotient_universe_python"] = len(model.classes)
            if metadata.get("expected") == "win":
                record["completed_negative_mismatch"] = record["status"] == "negative"
                record["caught"] = record["status"] == "win"
            if python_crosscheck:
                py_started = time.perf_counter_ns()
                try:
                    witness_indices = record.get("witness_first_action") or ()
                    native_hint = tuple(model.cells[i] for i in witness_indices)
                    record["witness_first_action_cells"] = [list(cell) for cell in native_hint]
                    py = r3.decide_fresh_next(
                        row,
                        horizon,
                        use_h10_shortcut=False,
                        preferred_first_pair=native_hint or None,
                    )
                    record["python_status"] = "win" if py.win else "negative"
                    record["python_nodes"] = py.nodes
                    record["python_wall_ns"] = time.perf_counter_ns() - py_started
                    record["python_match"] = record["status"] == record["python_status"]
                except Exception as exc:
                    record["python_status"] = "error"
                    record["python_error"] = f"{type(exc).__name__}: {exc}"
                    record["python_match"] = False
            evidence.write(json.dumps(record, sort_keys=True) + "\n")
            evidence.flush()
            results.append(record)
            counts[record["status"]] += 1
            print(f"native {number}: {row['pos_id']} h{horizon} -> {record['status']}", flush=True)
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"native process exited with {return_code}")
    summary = {
        "attempted": len(results),
        "statuses": dict(sorted(counts.items())),
        "wins": sum(row.get("status") == "win" for row in results),
        "completed_negatives": sum(row.get("status") == "negative" for row in results),
        "timeouts": sum(row.get("status") == "timeout" for row in results),
        "errors": sum(row.get("status") in ("error", "prep_error") for row in results),
        "completed_negative_mismatches": [row["id"] for row in results if row.get("completed_negative_mismatch")],
        "caught_ids": [row["id"] for row in results if row.get("caught")],
        "python_crosscheck_mismatches": [row["id"] for row in results if python_crosscheck and not row.get("python_match")],
    }
    def distribution(rows: list[dict], key: str) -> dict[str, int | float]:
        values = sorted(int(row[key]) for row in rows if row.get(key) is not None)
        if not values:
            return {"n": 0, "p50": 0, "p90": 0, "max": 0, "total": 0, "mean": 0.0}
        at = lambda q: values[int(q * (len(values) - 1))]
        return {
            "n": len(values), "p50": at(0.5), "p90": at(0.9), "max": values[-1],
            "total": sum(values), "mean": sum(values) / len(values),
        }
    completed = [row for row in results if row.get("status") in ("win", "negative")]
    summary["distributions"] = {
        "all": {key: distribution(results, key) for key in ("wall_ns", "nodes", "universe", "first_action_total")},
        "completed": {key: distribution(completed, key) for key in ("wall_ns", "nodes", "universe", "first_action_total")},
    }
    per_cohort: dict[str, Counter] = {}
    for row in results:
        if "cohort" in row:
            per_cohort.setdefault(row["cohort"], Counter())[row.get("status", "missing")] += 1
    summary["per_cohort_status"] = {
        name: dict(sorted(statuses.items())) for name, statuses in sorted(per_cohort.items())
    }
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("registry", "cohorts", "synthetic"))
    parser.add_argument("--binary", type=Path, default=default_binary())
    parser.add_argument("--per-root-ms", type=int, default=0, help="0 disables the harness deadline")
    parser.add_argument("--max-cache", type=int, default=2_000_000)
    parser.add_argument("--out", type=Path, default=ROOT / ".scratch" / "horizon_r4_native_rows.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / ".scratch" / "horizon_r4_native_summary.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cohort", action="append", help="cohort name; repeat to select multiple")
    parser.add_argument("--id", action="append", help="run only this position ID; repeat as needed")
    parser.add_argument("--prior-timeouts", type=Path, help="JSONL/JSON-array evidence; rerun rows whose status is timeout")
    parser.add_argument(
        "--exclude-ids-from",
        type=Path,
        action="append",
        help="skip every position ID already recorded in this JSONL/JSON file; repeatable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.binary.exists():
        raise SystemExit(f"native binary not found: {args.binary}")
    atlas = load_atlas()
    if args.mode == "registry":
        cases, selection = select_registry()
        python_crosscheck = False
    elif args.mode == "cohorts":
        cases, selection = select_cohorts(set(args.cohort) if args.cohort else None)
        python_crosscheck = False
    else:
        cases = [(row, horizon, {"synthetic": True}) for row, horizon in synthetic_rows()]
        selection = {"synthetic_cases": len(cases)}
        python_crosscheck = True
    targets = retry_ids(args.id, args.prior_timeouts)
    if targets:
        before = len(cases)
        cases = [case for case in cases if case[0].get("pos_id") in targets]
        found = {case[0]["pos_id"] for case in cases}
        selection["target_filter_requested"] = sorted(targets)
        selection["target_filter_missing"] = sorted(targets - found)
        selection["target_filter_before"] = before
        selection["target_filter_after"] = len(cases)
    excluded_sources = args.exclude_ids_from or []
    excluded = set().union(*(evidence_ids(path) for path in excluded_sources))
    if excluded:
        before = len(cases)
        cases = [case for case in cases if case[0].get("pos_id") not in excluded]
        selection["excluded_ids_sources"] = [str(path) for path in excluded_sources]
        selection["excluded_ids_found"] = before - len(cases)
        selection["excluded_ids_after"] = len(cases)
    _, summary = run_native(
        cases,
        binary=args.binary,
        timeout_ms=args.per_root_ms,
        max_cache=args.max_cache,
        out_path=args.out,
        atlas=atlas,
        limit=args.limit,
        python_crosscheck=python_crosscheck,
    )
    payload = {
        "metadata": {
            "mode": args.mode,
            "binary": str(args.binary),
            "per_root_ms": args.per_root_ms,
            "max_cache_entries": args.max_cache,
            "timeout_is_not_verdict": True,
            "native_threads": 1,
            "row_evidence": str(args.out),
        },
        "selection": selection,
        "summary": summary,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
