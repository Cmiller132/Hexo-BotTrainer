#!/usr/bin/env python3
"""Phase-R2 internal, engine-certificate, and known-WIN validation battery."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".scratch"))
import deadline_ladder_r as r1  # noqa: E402
import horizon_r2 as r2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / ".scratch" / "horizon_r2_validation.json")
    args = ap.parse_args()

    cohorts = r2.cohorts()
    all_rows = {row["pos_id"]: row for rows in cohorts.values() for row in rows}
    walls = r1.wall_sources()
    known: dict[str, dict] = {}

    def add_known(pos_id: str, source: str, depth: int | None = None) -> None:
        depth = depth if depth and depth > 0 else None
        prior = known.get(pos_id)
        if prior is None or (prior["cert_depth"] is None and depth is not None):
            known[pos_id] = {"source": source, "cert_depth": depth}

    for row in cohorts["puzzle_v3"]:
        if row.get("labels", {}).get("verdict") == "win":
            add_known(row["pos_id"], "puzzle_v3 labeled WIN")
    for row in cohorts["forcing19"]:
        if row.get("labels", {}).get("verdict") == "win":
            add_known(row["pos_id"], "forcing19 expected WIN")
    for cohort_name, wall in walls.items():
        for pos_id, record in wall.items():
            if record["status"] == "win" and pos_id in all_rows:
                add_known(pos_id, f"{cohort_name} measured WIN", record.get("cert_depth"))

    atlas_rows = json.loads(r1.DEFAULT_ATLAS.read_text(encoding="utf-8"))["rows"]
    for row in atlas_rows:
        if row.get("status") != "WIN" or not row.get("certified"):
            continue
        pos_id = f"atlas_full_{row['id']}"
        all_rows[pos_id] = {"pos_id": pos_id, "moves": row["moves"]}
        derived = row.get("derived_horizon")
        depth = derived - row["placements"] if isinstance(derived, int) else None
        add_known(pos_id, "opening atlas certified WIN", depth)

    result: dict = {
        "known_win_registry": {
            "unique": len(known),
            "with_cert_depth": sum(v["cert_depth"] is not None for v in known.values()),
            "without_cert_depth": sum(v["cert_depth"] is None for v in known.values()),
        },
        "engine_certificates": {},
    }
    cached: dict[tuple[str, int], r2.Decision] = {}
    for horizon in (6, 8):
        eligible = [(pos_id, info) for pos_id, info in known.items()
                    if info["cert_depth"] is not None and info["cert_depth"] <= horizon]
        misses = []
        rows_out = []
        started = time.perf_counter_ns()
        for pos_id, info in eligible:
            row = all_rows[pos_id]
            phase = r1.phase_player(len(row["moves"]))[0]
            if phase == "first":
                decision = r2.decide_fresh_current(row, horizon)
            else:
                # The exact monotone ladder is itself the h-decider: it tries
                # smaller exact clocks first and searches the full requested
                # clock only if none decides WIN.
                decision = r2.decide_ladder(row, horizon)
            cached[(pos_id, horizon)] = decision
            if not decision.win:
                misses.append(pos_id)
            rows_out.append({"pos_id": pos_id, "cert_depth": info["cert_depth"], "phase": phase,
                             "win": decision.win, "nodes": decision.nodes,
                             "universe": decision.universe, "wall_us": decision.wall_ns / 1e3})
        result["engine_certificates"][str(horizon)] = {
            "eligible": len(eligible), "caught": len(eligible) - len(misses), "misses": misses,
            "wall_ms": (time.perf_counter_ns() - started) / 1e6, "rows": rows_out,
        }

    # The inherited 2,941-root audit is adapted by separating shallow ground
    # truth from later/undated wins.  A later win is compatible with exact
    # NoWinWithin8 and is never relabeled a game loss.
    result["adapted_false_dismissal"] = {
        "registry_unique": len(known),
        "h8_ground_truth_eligible": result["engine_certificates"]["8"]["eligible"],
        "h8_ground_truth_misses": result["engine_certificates"]["8"]["misses"],
        "later_than_8": sum(v["cert_depth"] is not None and v["cert_depth"] > 8 for v in known.values()),
        "undated": sum(v["cert_depth"] is None for v in known.values()),
        "interpretation": "NoWinWithin8 is a bounded partial refutation, never a full-game LOSS.",
    }

    width_ids = ["oa-0153903c5a863630", "oa-23c6c04ad42d0904",
                 "oa-611666d7d930eb1f", "oa-6fda812864c6d19a",
                 "oa-773ca1a59e95f4e1"]
    width_rows = []
    for short_id in width_ids:
        pos_id = f"atlas_full_{short_id}"
        row = all_rows[pos_id]
        phase = r1.phase_player(len(row["moves"]))[0]
        decision = r2.decide_fresh_current(row, 8) if phase == "first" else r2.decide(row, 8)
        width_rows.append({"pos_id": short_id, "phase": phase, "win_within_8": decision.win,
                           "nodes": decision.nodes, "universe": decision.universe,
                           "wall_us": decision.wall_ns / 1e3})
    result["width_exhaust_comparison"] = {
        "rows": width_rows,
        "within_8_wins": sum(x["win_within_8"] for x in width_rows),
        "known_j2near_ids": ["oa-0153903c5a863630", "oa-6fda812864c6d19a",
                             "oa-773ca1a59e95f4e1"],
        "interpretation": "The known J2near wins resolve in 21/22 plies, not within 8.",
    }

    measurement_path = ROOT / ".scratch" / "horizon_r2.json"
    if measurement_path.is_file():
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
        joins = {}
        internal = {}
        for name, item in measurement["cohorts"].items():
            exact_ids = set(item["horizons"]["8"]["current_win"]["win_ids"])
            engine = walls[name]
            joins[name] = {
                "exact_h8_wins": len(exact_ids),
                "with_engine_row": sum(x in engine for x in exact_ids),
                "engine_unknown": sorted(x for x in exact_ids if x in engine and engine[x]["status"] == "unknown"),
            }
            h4_loss = set(item["validated_base"]["h4_forced_loss_ids"])
            h6_loss = set(item["horizons"]["6"]["forced_loss"]["win_ids"])
            internal[name] = {
                "reported_nesting_mismatches": item["internal_validation"],
                "h4_vs_h6_loss_symmetric_difference": sorted(h4_loss ^ h6_loss),
            }
        result["engine_unknown_join"] = joins
        result["measurement_internal_audit"] = internal

    # Direct equality check between the specialized fresh-turn algebra and the
    # generic placement minimax on every shallow certified fresh WIN.
    comparison = []
    compared = []
    for pos_id, info in known.items():
        if info["cert_depth"] is None or info["cert_depth"] > 6:
            continue
        row = all_rows[pos_id]
        if r1.phase_player(len(row["moves"]))[0] != "first":
            continue
        fast = cached[(pos_id, 6)]
        # Keep this deliberately independent check production-shaped.  The
        # full certified battery above already covers all roots; twelve exact
        # generic searches test implementation agreement without turning the
        # validation into a second exhaustive cohort run.
        if len(compared) >= 12 or fast.universe > 60:
            continue
        generic = r2.decide(row, 6)
        compared.append(pos_id)
        if fast.win != generic.win:
            comparison.append(pos_id)
    result["specialized_vs_generic_h6"] = {"positions": len(compared),
                                             "pos_ids": compared, "mismatches": comparison}

    broad_compared = []
    broad_mismatches = []
    for cohort_name in ("selfplay_v1", "human_v1", "puzzle_v3", "grinds", "forcing19"):
        for row in cohorts[cohort_name]:
            if len(broad_compared) >= 40:
                break
            if r1.phase_player(len(row["moves"]))[0] != "first":
                continue
            fast_win = r2.decide_fresh_current(row, 6)
            if fast_win.universe > 30:
                continue
            fast_loss = r2.decide_fresh_forced_loss(row, 6)
            generic_win = r2.decide(row, 6)
            mover = r2.schedule(len(row["moves"]), 1)[0]
            generic_loss = r2.decide(row, 6, 1 - mover)
            broad_compared.append(row["pos_id"])
            if fast_win.win != generic_win.win or fast_loss.win != generic_loss.win:
                broad_mismatches.append({"pos_id": row["pos_id"],
                                         "fast": [fast_win.win, fast_loss.win],
                                         "generic": [generic_win.win, generic_loss.win]})
        if len(broad_compared) >= 40:
            break
    result["specialized_vs_generic_broad_h6"] = {
        "positions": len(broad_compared), "pos_ids": broad_compared,
        "mismatches": broad_mismatches,
    }

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "known": len(known),
        "h6": {k: result["engine_certificates"]["6"][k] for k in ("eligible", "caught", "misses")},
        "h8": {k: result["engine_certificates"]["8"][k] for k in ("eligible", "caught", "misses")},
        "generic_mismatches": comparison,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
