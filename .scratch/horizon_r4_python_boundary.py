"""Audit inherited Python-vs-native action-space boundaries for Horizon R4."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch"
sys.path.insert(0, str(SCRATCH))

import deadline_ladder_r as phase_r  # noqa: E402
import horizon_h10 as h10  # noqa: E402
import horizon_r3 as r3  # noqa: E402


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def history_key(row: dict) -> tuple[tuple[int, int], ...]:
    return tuple((int(q), int(r)) for q, r in row["moves"])


def dist(values: list[int]) -> dict[str, int]:
    values = sorted(values)
    at = lambda q: values[int(q * (len(values) - 1))]
    return {"n": len(values), "min": values[0], "p50": at(0.5), "p90": at(0.9), "max": values[-1], "sum": sum(values)}


def main() -> None:
    known, rows = h10._known_registry()
    eligible = {
        pos_id: info for pos_id, info in known.items()
        if info["cert_depth"] in (13, 14)
    }
    native_rows = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (SCRATCH / "horizon_r4_phase1_registry_final_rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    assert set(native_rows) == set(eligible)

    details = []
    phase_counts = Counter()
    fresh_count_matches = []
    second_pair_excess = []
    for pos_id in sorted(eligible):
        row = rows[pos_id]
        horizon = int(eligible[pos_id]["cert_depth"])
        phase = phase_r.phase_player(len(row["moves"]))[0]
        model = r3.build_next_model(row, horizon)
        counts = Counter(mask.bit_count() for mask in model.first_pairs)
        native_count = int(native_rows[pos_id]["first_action_total"])
        item = {
            "id": pos_id,
            "phase": phase,
            "horizon": horizon,
            "python_first_actions": len(model.first_pairs),
            "python_empty_actions": counts[0],
            "python_singleton_actions": counts[1],
            "python_pair_actions": counts[2],
            "native_first_actions": native_count,
        }
        if phase == "first":
            item["counts_match"] = len(model.first_pairs) == native_count
            fresh_count_matches.append(item["counts_match"])
        else:
            board = {tuple(cell): phase_r.owner_at(i) for i, cell in enumerate(row["moves"])}
            legal = r3.legal_cells(board)
            correct_singletons = sum(cell in legal for cell in model.cells)
            item["correct_secondstone_singletons"] = correct_singletons
            item["native_matches_correct_singletons"] = native_count == correct_singletons
            item["python_phase_illegal_pair_excess"] = counts[2]
            second_pair_excess.append(counts[2])
        details.append(item)
        phase_counts[phase] += 1

    second = [row for row in details if row["phase"] == "second"]
    assert phase_counts == Counter({"first": 149, "second": 6})
    assert all(fresh_count_matches)
    assert all(row["native_matches_correct_singletons"] for row in second)
    assert all(row["python_phase_illegal_pair_excess"] > 0 for row in second)

    histories = {history_key(rows[pos_id]) for pos_id in eligible}
    payload = {
        "metadata": {
            "schema": 1,
            "claim_label": "CODE-FACT plus MEASURED exhaustive model reconstruction",
            "scope": "all registry IDs at exact certificate depths 13 or 14",
            "python_source": ".scratch/horizon_r3.py",
            "native_rows": ".scratch/horizon_r4_phase1_registry_final_rows.jsonl",
        },
        "eligible_ids": len(eligible),
        "eligible_unique_ordered_move_histories": len(histories),
        "phase_counts": dict(sorted(phase_counts.items())),
        "fresh": {
            "roots": len(fresh_count_matches),
            "python_native_first_action_count_matches": sum(fresh_count_matches),
            "mismatches": [row["id"] for row in details if row["phase"] == "first" and not row["counts_match"]],
        },
        "secondstone": {
            "roots": len(second),
            "all_native_counts_equal_correct_singleton_counts": all(row["native_matches_correct_singletons"] for row in second),
            "all_python_models_contain_phase_illegal_pairs": all(row["python_phase_illegal_pair_excess"] > 0 for row in second),
            "python_phase_illegal_pair_excess": dist(second_pair_excess),
            "details": second,
        },
        "interpretation": {
            "python_code_fact": "build_next_model adds legal singleton A1 actions for SecondStone, then unconditionally also adds two-cell pairs from retained near windows",
            "true_clock": "SecondStone has one placement remaining in its current turn, so those pair masks are phase-illegal",
            "native_correction": "the Rust parser/model accepts SecondStone h13 and enumerates only the legal singleton action quota",
            "parity_limit": "the immediate SecondStone synthetic win agrees in one node but does not validate the inherited Python exhaustive action space",
        },
        "source_sha256": {
            ".scratch/horizon_r3.py": digest(SCRATCH / "horizon_r3.py"),
            ".scratch/horizon_native/src/lib.rs": digest(SCRATCH / "horizon_native" / "src" / "lib.rs"),
            ".scratch/horizon_r4_phase1_registry_final_rows.jsonl": digest(SCRATCH / "horizon_r4_phase1_registry_final_rows.jsonl"),
            ".scratch/horizon_r4_python_boundary.py": digest(Path(__file__)),
        },
    }
    out = SCRATCH / "horizon_r4_python_boundary.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "PYTHON_BOUNDARY_OK "
        f"fresh={payload['fresh']['python_native_first_action_count_matches']}/149 "
        f"second_bug={payload['secondstone']['roots']}/6 "
        f"histories={len(histories)}/155"
    )


if __name__ == "__main__":
    main()
