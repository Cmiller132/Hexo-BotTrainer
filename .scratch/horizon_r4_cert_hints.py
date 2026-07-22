#!/usr/bin/env python3
"""Audit certificate-derived move-ordering hints for Horizon R4.

This is deliberately a read-only solver audit: it loads an already-staged
Windows Python extension, replays frozen JSONL positions, and runs the verified
deep-solve probe.  It does not build Cargo targets or touch package sources.

The archived V1 artifacts serialize only the certificate root Choice cell.
For FirstStone roots, this audit applies that certified cell and independently
re-solves the exact SecondStone state.  The resulting second cell is therefore
a newly MEASURED, verified ordering hint; it is not claimed to be a decoded
child ID from the original (unserialized) certificate arena.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import types
from pathlib import Path
from typing import Any


WORKTREE = Path(__file__).resolve().parents[1]
WORKTREES = WORKTREE.parent
V1_SOAK = WORKTREES / "v1-soak"
TRUTH_PASS = WORKTREES / "truth-pass"

STAGE = TRUTH_PASS / ".cargo-target" / "golden-stage-20260721"
EQ_PYD = STAGE / "hexfield_eq" / "_rust.pyd"
ENGINE_PYD = STAGE / "hexo_engine" / "_rust.pyd"

SELFPLAY_SET = WORKTREE / "scripts" / "tss_harness" / "sets" / "selfplay_v1.jsonl"
HUMAN_SET = WORKTREE / "scripts" / "tss_harness" / "sets" / "human_v1.jsonl"
MAIN4_DIR = (
    WORKTREE
    / "scripts"
    / "tss_harness"
    / "harness_runs"
    / "20260721_032725_main4_integration_gate2"
)
MAIN4_SELFPLAY = MAIN4_DIR / "records_main4_integration_gate2_selfplay_v1.jsonl"
MAIN4_HUMAN = MAIN4_DIR / "records_main4_integration_gate2_human_v1.jsonl"

INTERNALIZATION = (
    V1_SOAK / "scripts" / "tss_harness" / "sets" / "internalization_v1.jsonl"
)
INTERNALIZATION_MANIFEST = INTERNALIZATION.with_suffix(".manifest.json")
INTERNALIZATION_PIN = INTERNALIZATION.with_suffix(".sha256")
INTERNALIZATION_GENERATOR = (
    V1_SOAK / "scripts" / "tss_harness" / "mint_internalization_set.py"
)
SOAK_SELFPLAY = V1_SOAK / "raws" / "soak_selfplay.jsonl"
V1_SEARCH_RS = V1_SOAK / "packages" / "hexfield_eq" / "rust" / "src" / "search.rs"
V1_VERIFY_RS = V1_SOAK / "packages" / "hexfield_eq" / "rust" / "src" / "tss_verify.rs"
STAGED_SEARCH_RS = (
    TRUTH_PASS / "packages" / "hexfield_eq" / "rust" / "src" / "search.rs"
)
STAGED_VERIFY_RS = (
    TRUTH_PASS / "packages" / "hexfield_eq" / "rust" / "src" / "tss_verify.rs"
)

# These are the seven R4 timeout roots for which an archived certificate root
# Choice is available.  The parent lane may need only a subset after retries;
# keeping all seven makes the audit stable as the timeout frontier shrinks.
TARGETS = {
    "human_20bea7804fffee60_p15": "human",
    "human_6023b2ef70e3ffc6_p76": "human",
    "sp_0_p51": "selfplay",
    "sp_13_p59": "selfplay",
    "sp_35_p27": "selfplay",
    "sp_4_p79": "selfplay",
    "sp_5_p61": "selfplay",
}

ARCH_ENV = {
    "HEXFIELD_EQ_FEATURE_VERSION": "2",
    "HEXFIELD_EQ_CHANNELS": "192",
    "HEXFIELD_EQ_ATTENTION_HEADS": "3",
    "HEXFIELD_EQ_C_ORBIT": "16",
    "HEXFIELD_EQ_GROUP_ORDER": "12",
    "HEXFIELD_EQ_TRUNK": "CCACCACA",
    "HEXFIELD_EQ_RAYTAP": "both",
    "HEXFIELD_EQ_RAYTAP_LUT": "additive",
    "HEXFIELD_EQ_REG_LANE": "1",
    "HEXFIELD_EQ_RAY_BLOCKERS": "1",
}

PROBE_CONFIG = {
    "node_cap": 500,
    "goal": "win",
    "horizon": 16,
    "ladder": False,
    "zone": False,
    "wide": True,
    "with_stats": False,
}

PROBE_FIELDS = (
    "status",
    "placements",
    "has_cert",
    "cert_depth",
    "cert_root_move_q",
    "cert_root_move_r",
    "cert_choice_nodes",
    "cert_universal_nodes",
    "cert_zone_nodes",
    "cert_version",
    "deep_nodes",
    "deep_verify_failed",
    "horizon_cut",
    "horizon_cut_tall",
    "horizon_preflight_failed",
    "horizon_retry",
    "pair_omitted",
    "zone_verify_failed",
    "wall_nanos",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_jsonl(path: Path, id_key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if id_key in row:
                rows[str(row[id_key])] = row
    return rows


def load_extension(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension spec: {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_staged_modules() -> tuple[Any, Any, list[str]]:
    # The staged packages normally import NumPy through package __init__.  The
    # audit needs only the two PyO3 extensions, so minimal parent packages avoid
    # that unrelated runtime dependency.
    for package in ("hexo_engine", "hexfield_eq"):
        module = types.ModuleType(package)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[package] = module

    cleared_tss = sorted(key for key in os.environ if key.startswith("TSS_"))
    for key in cleared_tss:
        del os.environ[key]
    for key, value in ARCH_ENV.items():
        os.environ[key] = value

    engine = load_extension("hexo_engine._rust", ENGINE_PYD)
    eq = load_extension("hexfield_eq._rust", EQ_PYD)
    return engine, eq, cleared_tss


def build_state(engine: Any, moves: list[list[int]]) -> Any:
    state = engine.new_game()
    for q, r in moves:
        engine.apply_action(state, int(q), int(r))
    return state


def phase_record(engine: Any, state: Any) -> dict[str, Any]:
    raw = dict(engine.to_python_state(state))
    return {
        "current_player": raw["current_player"],
        "phase": raw["phase"],
        "placements_made": int(raw["placements_made"]),
    }


def run_probe(eq: Any, state: Any) -> dict[str, Any]:
    cfg = PROBE_CONFIG
    raw = dict(
        eq.hexfield_eq_deep_solve_probe(
            state,
            cfg["node_cap"],
            cfg["goal"],
            cfg["horizon"],
            cfg["ladder"],
            cfg["zone"],
            cfg["wide"],
            cfg["with_stats"],
        )
    )
    return {field: raw[field] for field in PROBE_FIELDS if field in raw}


def probe_choice(probe: dict[str, Any]) -> list[int] | None:
    if "cert_root_move_q" not in probe:
        return None
    return [int(probe["cert_root_move_q"]), int(probe["cert_root_move_r"])]


def assert_verified_win(probe: dict[str, Any], max_depth: int) -> None:
    assert probe["status"] == "win", probe
    assert probe["has_cert"] is True, probe
    assert int(probe["deep_verify_failed"]) == 0, probe
    assert int(probe["zone_verify_failed"]) == 0, probe
    assert int(probe["cert_depth"]) <= max_depth, probe
    assert probe_choice(probe) is not None, probe


def project_soak(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    fields = (
        "arm",
        "status",
        "cert_root_move_q",
        "cert_root_move_r",
        "cert_depth",
        "cert_choice_nodes",
        "cert_universal_nodes",
        "cert_zone_nodes",
        "deep_nodes",
        "deep_verify_failed",
        "pair_omitted",
    )
    return {key: row[key] for key in fields if key in row}


def project_main4(row: dict[str, Any]) -> dict[str, Any]:
    counters = row["counters"]
    return {
        "status": row["status"],
        "verified": bool(row["verified"]),
        "verify_failed": int(row["verify_failed"]),
        "cert_depth": int(counters["cert_depth"]),
        "deep_nodes": int(counters["deep_nodes"]),
        "has_cert": bool(counters["has_cert"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKTREE / ".scratch" / "horizon_r4_cert_hints.json",
    )
    args = parser.parse_args()

    positions: dict[str, dict[str, Any]] = {}
    positions.update(load_jsonl(SELFPLAY_SET, "pos_id"))
    positions.update(load_jsonl(HUMAN_SET, "pos_id"))
    internalization = load_jsonl(INTERNALIZATION, "pos_id")
    main4: dict[str, dict[str, Any]] = {}
    main4.update(load_jsonl(MAIN4_SELFPLAY, "pos_id"))
    main4.update(load_jsonl(MAIN4_HUMAN, "pos_id"))

    soak_h16: dict[str, dict[str, Any]] = {}
    with SOAK_SELFPLAY.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("arm") == "h16_flat_wide" and row.get("pos_id") in TARGETS:
                soak_h16[str(row["pos_id"])] = row

    manifest = json.loads(INTERNALIZATION_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["sha256"] == sha256(INTERNALIZATION)

    engine, eq, cleared_tss = load_staged_modules()
    results: dict[str, dict[str, Any]] = {}

    for pos_id, cohort in TARGETS.items():
        pos = positions[pos_id]
        archived = internalization[pos_id]
        assert pos["moves"] == archived["moves"], pos_id
        assert archived["verdict"] == "win", pos_id
        assert archived["has_cert_move"] is True, pos_id
        archived_choice = [int(x) for x in archived["cert_move"]]

        root_state = build_state(engine, pos["moves"])
        phase_before = phase_record(engine, root_state)
        root_probe = run_probe(eq, root_state)
        assert_verified_win(root_probe, 14)
        assert probe_choice(root_probe) == archived_choice, pos_id

        if cohort == "selfplay":
            soak = project_soak(soak_h16[pos_id])
            assert soak is not None
            assert soak["status"] == root_probe["status"]
            for key in (
                "cert_root_move_q",
                "cert_root_move_r",
                "cert_depth",
                "cert_choice_nodes",
                "cert_universal_nodes",
                "cert_zone_nodes",
                "deep_nodes",
                "deep_verify_failed",
                "pair_omitted",
            ):
                assert soak[key] == root_probe[key], (pos_id, key, soak[key], root_probe[key])
        else:
            soak = None

        after_first_state = build_state(engine, pos["moves"])
        engine.apply_action(after_first_state, *archived_choice)
        phase_after_first = phase_record(engine, after_first_state)

        child_probe: dict[str, Any] | None = None
        second_choice: list[int] | None = None
        phase_after_second: dict[str, Any] | None = None
        geometry_relation: dict[str, bool] | None = None

        if phase_before["phase"] == "FirstStone":
            assert phase_after_first["phase"] == "SecondStone", pos_id
            assert phase_after_first["current_player"] == phase_before["current_player"], pos_id
            child_probe = run_probe(eq, after_first_state)
            assert_verified_win(child_probe, 13)
            second_choice = probe_choice(child_probe)
            assert second_choice is not None

            after_second_state = build_state(
                engine, pos["moves"] + [archived_choice, second_choice]
            )
            phase_after_second = phase_record(engine, after_second_state)
            assert phase_after_second["phase"] == "FirstStone", pos_id
            assert phase_after_second["current_player"] != phase_before["current_player"], pos_id

            geometry_relation = {
                "depth_root_equals_child_plus_one": int(root_probe["cert_depth"])
                == int(child_probe["cert_depth"]) + 1,
                "choice_nodes_root_equals_child_plus_one": int(
                    root_probe["cert_choice_nodes"]
                )
                == int(child_probe["cert_choice_nodes"]) + 1,
                "universal_nodes_equal": root_probe["cert_universal_nodes"]
                == child_probe["cert_universal_nodes"],
                "zone_nodes_equal": root_probe["cert_zone_nodes"]
                == child_probe["cert_zone_nodes"],
                "pair_omitted_equal": root_probe["pair_omitted"]
                == child_probe["pair_omitted"],
            }
            assert all(geometry_relation.values()), (pos_id, geometry_relation)
            full_turn_ordering = [archived_choice, second_choice]
            clock_note = "fresh FirstStone root: two same-player placements"
        elif phase_before["phase"] == "SecondStone":
            assert phase_after_first["phase"] == "FirstStone", pos_id
            assert phase_after_first["current_player"] != phase_before["current_player"], pos_id
            full_turn_ordering = [archived_choice]
            clock_note = (
                "SecondStone root: the certified root Choice is the only remaining "
                "same-player placement; no second Choice exists before the phase changes"
            )
        else:
            raise AssertionError((pos_id, phase_before))

        results[pos_id] = {
            "claim": "MEASURED ordering-only",
            "cohort": cohort,
            "archived_cert_root_choice": archived_choice,
            "archived_cert_root_source": "internalization_v1.jsonl",
            "full_turn_ordering": full_turn_ordering,
            "clock_note": clock_note,
            "phase_proof": {
                "root": phase_before,
                "after_archived_root_choice": phase_after_first,
                "after_independent_child_choice": phase_after_second,
            },
            "root_h16_verified_probe": root_probe,
            "secondstone_h16_verified_probe": child_probe,
            "certificate_geometry_relation": geometry_relation,
            "same_original_arena_child_recovered": False,
            "second_cell_evidence": (
                "independent verified solve of the exact SecondStone child; "
                "not serialized in the original flat certificate artifact"
                if child_probe is not None
                else "not applicable: root clock is already SecondStone"
            ),
            "archive_cross_checks": {
                "internalization_row_matches_frozen_moves": True,
                "soak_h16_flat_wide": soak,
                "main4_registry": project_main4(main4[pos_id]),
            },
        }

    sources = {
        "audit_script": source_record(Path(__file__).resolve()),
        "staged_hexfield_eq_pyd": source_record(EQ_PYD),
        "staged_hexo_engine_pyd": source_record(ENGINE_PYD),
        "frozen_selfplay_set": source_record(SELFPLAY_SET),
        "frozen_human_set": source_record(HUMAN_SET),
        "internalization_set": source_record(INTERNALIZATION),
        "internalization_manifest": source_record(INTERNALIZATION_MANIFEST),
        "internalization_sha256_pin": source_record(INTERNALIZATION_PIN),
        "internalization_generator": source_record(INTERNALIZATION_GENERATOR),
        "soak_selfplay": source_record(SOAK_SELFPLAY),
        "main4_selfplay_registry": source_record(MAIN4_SELFPLAY),
        "main4_human_registry": source_record(MAIN4_HUMAN),
        "v1_certificate_projection_source": source_record(V1_SEARCH_RS),
        "v1_certificate_verifier_source": source_record(V1_VERIFY_RS),
        "staged_checkout_certificate_projection_source": source_record(STAGED_SEARCH_RS),
        "staged_checkout_certificate_verifier_source": source_record(STAGED_VERIFY_RS),
    }

    payload = {
        "schema": "horizon_r4_cert_hints_v1",
        "claim_class": "MEASURED ordering-only",
        "scope": {
            "package_edits": False,
            "cargo_builds": False,
            "solver_runs_single_process": True,
            "root_choice": (
                "artifact-certified Choice cell from the pinned internalization set, "
                "then independently reproduced by the staged verified solver"
            ),
            "second_choice": (
                "independent verified h16 solve at the exact SecondStone state; "
                "usable for ordering only"
            ),
            "negative_boundary": (
                "the soak/probe JSON projection contains no certificate node array or child ID, "
                "so it cannot establish that the second cell is literally the child Choice in "
                "the archived arena"
            ),
        },
        "reproduction": {
            "command": "python .scratch/horizon_r4_cert_hints.py --output .scratch/horizon_r4_cert_hints.json",
            "cwd": str(WORKTREE),
            "python": sys.version,
            "platform": platform.platform(),
            "architecture_environment": ARCH_ENV,
            "tss_environment_policy": "remove inherited TSS_* variables; use compiled defaults",
            "inherited_tss_variables_removed": cleared_tss,
            "probe_config": PROBE_CONFIG,
            "binary_loading": (
                "direct PyO3 extension loading from the pinned truth-pass golden stage; "
                "minimal parent modules avoid importing unrelated NumPy-dependent package code"
            ),
        },
        "internalization_manifest": manifest,
        "code_facts": [
            {
                "class": "CODE-FACT",
                "location": f"{STAGED_VERIFY_RS}:276",
                "fact": "CertNode::Choice stores both mv and child in the in-memory arena.",
            },
            {
                "class": "CODE-FACT",
                "location": f"{STAGED_SEARCH_RS}:5026",
                "fact": (
                    "deep_solve_probe derives aggregate certificate geometry, then emits only "
                    "the root Choice q/r; it does not emit the node arena or child ID."
                ),
            },
            {
                "class": "CODE-FACT",
                "location": f"{INTERNALIZATION_GENERATOR}:12",
                "fact": (
                    "internalization_v1 re-solves through the verified path and stores the "
                    "certificate-designated root Choice as cert_move."
                ),
            },
        ],
        "sources": sources,
        "results": results,
        "summary": {
            "roots": len(results),
            "fresh_root_full_pairs": sum(
                1 for result in results.values() if len(result["full_turn_ordering"]) == 2
            ),
            "secondstone_singletons": sum(
                1 for result in results.values() if len(result["full_turn_ordering"]) == 1
            ),
            "all_root_probes_verified": all(
                result["root_h16_verified_probe"]["deep_verify_failed"] == 0
                for result in results.values()
            ),
            "all_child_probes_verified": all(
                result["secondstone_h16_verified_probe"] is None
                or result["secondstone_h16_verified_probe"]["deep_verify_failed"] == 0
                for result in results.values()
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "CERT_HINTS_OK "
        f"roots={payload['summary']['roots']} "
        f"pairs={payload['summary']['fresh_root_full_pairs']} "
        f"secondstone={payload['summary']['secondstone_singletons']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
