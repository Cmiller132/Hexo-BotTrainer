#!/usr/bin/env python3
"""Phase-R, read-only deadline-ladder measurements.

The analyzer intentionally re-derives rule-level window geometry from frozen
move lists.  It makes no engine changes and never turns a negative proxy into a
game value.  With ``--engine-check`` it also compares the two shallow tactical
predicates with the compiled engine's public threat-analysis diagnostic.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "scripts" / "tss_harness" / "sets"
FORCING = ROOT / "packages" / "hexfield_eq" / "rust" / "corpus" / "forcing_corpus_moves.txt"
LABELS = ROOT / "raws" / "lanec_labels.jsonl"
MAIN4 = ROOT / "scripts" / "tss_harness" / "harness_runs" / "20260721_032725_main4_integration_gate2"
DEFAULT_ATLAS = ROOT.parent / "opening-atlas" / "atlas-web" / "data" / "atlas.json"
AXES = ((1, 0), (0, 1), (1, -1))
PACKED_CELL_COUNT = 1 << 32


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_forcing(path: Path) -> list[dict]:
    lines = iter(path.read_text(encoding="utf-8").splitlines())
    out = []
    for raw in lines:
        raw = raw.strip()
        if not raw or raw.startswith("#") or raw == "END":
            continue
        assert raw.startswith("POS "), raw
        meta = dict(token.split("=", 1) for token in raw.split()[1:])
        moves = []
        while len(moves) < int(meta["nstones"]):
            q, r, *_ = next(lines).split()
            moves.append([int(q), int(r)])
        out.append({
            "pos_id": f"forcing_{meta['id']}",
            "source": "forcing19",
            "moves": moves,
            "labels": {"verdict": meta.get("expect", "") .lower()},
        })
    return out


def owner_at(index: int) -> int:
    if index == 0:
        return 0
    return 1 if ((index - 1) // 2) % 2 == 0 else 0


def phase_player(n: int) -> tuple[str, int, int]:
    if n == 0:
        return "opening", 0, 1
    post = n - 1
    player = 1 if (post // 2) % 2 == 0 else 0
    return ("first" if post % 2 == 0 else "second"), player, (2 if post % 2 == 0 else 1)


def cells(key: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    q, r, dq, dr = key
    return tuple((q + i * dq, r + i * dr) for i in range(6))


def entries(board: dict[tuple[int, int], int]) -> list[dict]:
    keys = set()
    for q, r in board:
        for dq, dr in AXES:
            for offset in range(6):
                keys.add((q - offset * dq, r - offset * dr, dq, dr))
    out = []
    for key in keys:
        cs = cells(key)
        c0 = sum(board.get(c) == 0 for c in cs)
        c1 = sum(board.get(c) == 1 for c in cs)
        if c0 and c1:
            continue
        if c0 or c1:
            owner = 0 if c0 else 1
            out.append({"owner": owner, "count": max(c0, c1),
                        "empty": frozenset(c for c in cs if c not in board), "key": key})
    return out


def min_hitting(family: list[frozenset], budget: int = 2) -> int | None:
    if not family:
        return 0
    if any(not edge for edge in family):
        return None
    universe = sorted(set().union(*family))
    for size in range(1, budget + 1):
        for choice in combinations(universe, size):
            if all(any(c in edge for c in choice) for edge in family):
                return size
    return None


def own_win_now(es: list[dict], player: int, budget: int) -> bool:
    return any(e["owner"] == player and (e["count"] == 5 or (e["count"] == 4 and budget == 2)) for e in es)


def forced_loss(es: list[dict], player: int, budget: int) -> tuple[bool, int | None, int]:
    family = [e["empty"] for e in es if e["owner"] != player and e["count"] >= 4]
    tau = min_hitting(family, budget)
    return (not own_win_now(es, player, budget) and tau is None), tau, len(family)


def phi_lt_one(es: list[dict], player: int) -> bool:
    bins = Counter(e["count"] for e in es if e["owner"] == player)
    a = bins[1] + 3 * bins[3] + 9 * bins[5]
    b = bins[2] + 3 * bins[4]
    return b <= 8 and a * a < 3 * (9 - b) * (9 - b)


def no_joint_carrier(es: list[dict], player: int, phase: str, win_now: bool) -> bool:
    if phase != "first" or win_now:
        return False
    c2 = [e["empty"] for e in es if e["owner"] == player and e["count"] == 2]
    c3 = [e["empty"] for e in es if e["owner"] == player and e["count"] == 3]
    if len(c3) >= 2:
        return False
    if c3 and any(c3[0] & edge for edge in c2):
        return False
    seen = set()
    for edge in c2:
        for pair in combinations(sorted(edge), 2):
            if pair in seen:
                return False
            seen.add(pair)
    return True


def normal_candidates(es: list[dict], player: int) -> set[tuple[int, int]]:
    out = set()
    for e in es:
        if (e["owner"] == player and e["count"] >= 2) or (e["owner"] != player and e["count"] >= 4):
            out.update(e["empty"])
    return out


def admissible_pair_exists(es: list[dict], player: int) -> bool:
    firsts = normal_candidates(es, player)
    defender_threats = [e["empty"] for e in es if e["owner"] != player and e["count"] >= 4]
    own = [e for e in es if e["owner"] == player]
    for first in firsts:
        seconds = set(firsts)
        for e in own:
            if e["count"] >= 1 and first in e["empty"]:
                seconds.update(e["empty"])
        seconds.discard(first)
        for second in seconds:
            if any(first not in edge and second not in edge for edge in defender_threats):
                continue
            family = []
            for e in own:
                added = int(first in e["empty"]) + int(second in e["empty"])
                if e["count"] + added >= 4:
                    family.append(frozenset(e["empty"] - {first, second}))
            if family and min_hitting(family, 2) != 1:
                return True
    return False


def census_lb(phase: str, census: int) -> int | None:
    if census > 5 or phase == "opening":
        return None
    if phase == "first":
        m = 6 - census if census >= 4 else min(7 - census, 6)
        return [1, 2, 5, 6, 9, 10][max(0, m - 1)]
    m = 6 - census if census >= 3 else min(7 - census, 6)
    return [1, 4, 5, 8, 9, 12][max(0, m - 1)]


def attacker_slots(phase: str, h: int) -> int:
    """Maximum current-mover placements among the next h physical placements."""
    if h <= 0:
        return 0
    if phase == "opening":
        # Opening is one placement, then the opponent gets a pair.
        return 1 + max(0, (h - 1) // 4 * 2) + (1 if (h - 1) % 4 >= 3 else 0)
    first_budget = 2 if phase == "first" else 1
    if h <= first_budget:
        return h
    remaining = h - first_budget
    full = remaining // 4
    tail = remaining % 4
    return first_budget + 2 * full + max(0, tail - 2)


def feature(row: dict) -> dict:
    board = {tuple(c): owner_at(i) for i, c in enumerate(row["moves"])}
    phase, player, budget = phase_player(len(row["moves"]))
    es = entries(board)
    win_now = own_win_now(es, player, budget)
    loss, tau, opp_count = forced_loss(es, player, budget)
    alive_counts = [e["count"] for e in es if e["owner"] == player]
    census = max(alive_counts, default=0)
    deficit = min([6 - c for c in alive_counts] + [6])
    joint = no_joint_carrier(es, player, phase, win_now)
    no_turn = phase == "first" and not win_now and not admissible_pair_exists(es, player)
    return {
        "pos_id": row["pos_id"], "n": len(row["moves"]), "phase": phase,
        "player": player, "budget": budget, "census": census,
        "census_lb": census_lb(phase, census), "stone_deficit": deficit,
        "own_win_now": win_now, "forced_loss": loss, "opp_tau": tau,
        "opp_threat_count": opp_count, "phi_lt_one": phi_lt_one(es, player),
        "no_joint_carrier": joint, "no_admissible_first_turn": no_turn,
        # This is deliberately a carrier surrogate, not a theorem deadline.
        "packed_fill_remaining": PACKED_CELL_COUNT - len(row["moves"]),
    }


def fires(f: dict, candidate: str, horizon: int | None) -> bool:
    if candidate == "census_lb":
        return horizon is not None and f["census_lb"] is not None and f["census_lb"] > horizon
    if candidate == "stone_deficit_lb":
        return horizon is not None and f["stone_deficit"] > attacker_slots(f["phase"], horizon)
    if candidate == "no_joint_carrier":
        return f["no_joint_carrier"]
    if candidate == "no_admissible_first_turn":
        return f["no_admissible_first_turn"]
    raise KeyError(candidate)


def percentile(xs: list[int], q: float) -> int | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int((len(ys) - 1) * q))]


def wall_sources() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    labels = read_jsonl(LABELS)
    for name, source in (("grinds", "grind"), ("forcing19", "forcing")):
        out[name] = {r["pos_id"]: {"status": r["status"], "wall_s": float(r["wall_s"])}
                     for r in labels if r.get("source") == source}
    for name in ("selfplay_v1", "human_v1", "puzzle_v3"):
        path = MAIN4 / f"records_main4_integration_gate2_{name}.jsonl"
        out[name] = {r["pos_id"]: {"status": r["status"], "wall_s": r["wall_nanos"] / 1e9,
                                    "cert_depth": r.get("counters", {}).get("cert_depth", 0)}
                     for r in read_jsonl(path)}
    return out


def engine_check(rows: Iterable[dict], feats: dict[str, dict], package_root: Path | None) -> dict:
    if package_root:
        sys.path.insert(0, str(package_root))
    api = importlib.import_module("hexo_engine.api")
    types = importlib.import_module("hexo_engine.types")
    rust = importlib.import_module("hexfield_eq._rust")
    mismatch = []
    total = 0
    for row in rows:
        state = api.new_game()
        for q, r in row["moves"]:
            result = api.apply_action(state, types.PlacementAction(types.AxialCoord(q=q, r=r)))
            if result is None:
                raise RuntimeError(f"illegal replay: {row['pos_id']} {(q, r)}")
        actual = dict(rust.hexfield_eq_threat_analysis(state))
        expected = feats[row["pos_id"]]
        total += 1
        if bool(actual["own_win_now"]) != expected["own_win_now"] or bool(actual["forced_loss"]) != expected["forced_loss"]:
            mismatch.append({"pos_id": row["pos_id"], "engine": actual,
                             "python": {"own_win_now": expected["own_win_now"], "forced_loss": expected["forced_loss"]}})
    return {"positions": total, "mismatches": len(mismatch), "examples": mismatch[:10]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / ".scratch" / "deadline_ladder_r.json")
    ap.add_argument("--engine-check", action="store_true")
    ap.add_argument("--package-root", type=Path)
    ap.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    args = ap.parse_args()

    cohorts = {name: read_jsonl(SETS / f"{name}.jsonl") for name in ("selfplay_v1", "human_v1", "puzzle_v3")}
    by_id = {r["pos_id"]: r for r in cohorts["selfplay_v1"]}
    grind_ids = [r["pos_id"] for r in read_jsonl(LABELS) if r.get("source") == "grind"]
    cohorts["grinds"] = [by_id[x] for x in grind_ids]
    cohorts["forcing19"] = parse_forcing(FORCING)

    all_rows = {r["pos_id"]: r for rows in cohorts.values() for r in rows}
    feats = {pos_id: feature(row) for pos_id, row in all_rows.items()}
    cohort_feature_ids = tuple(feats)
    walls = wall_sources()
    candidates = ("census_lb", "stone_deficit_lb", "no_joint_carrier", "no_admissible_first_turn")

    result: dict = {
        "metadata": {
            "packed_cell_count": PACKED_CELL_COUNT,
            "rules_board": "unbounded/Z^2; no sound finite fill deadline",
            "surrogate_warning": "2^32-n is an implementation-carrier surrogate, not a Lean/game deadline",
            "cohort_sizes": {k: len(v) for k, v in cohorts.items()},
        },
        "cohorts": {}, "false_dismissal_audit": {}, "small_h": {},
    }
    for name, rows in cohorts.items():
        fs = [feats[r["pos_id"]] for r in rows]
        ds = [f["packed_fill_remaining"] for f in fs]
        cohort = {
            "n": len(fs),
            "placements": {"min": min(f["n"] for f in fs), "p50": percentile([f["n"] for f in fs], .5),
                           "p90": percentile([f["n"] for f in fs], .9), "max": max(f["n"] for f in fs)},
            "sound_deadline": {"kind": "infinity", "reason": "unbounded board"},
            "packed_surrogate_D": {"min": min(ds), "p50": percentile(ds, .5), "p90": percentile(ds, .9), "max": max(ds)},
            "phase": dict(Counter(f["phase"] for f in fs)), "candidates": {},
            "shallow": {"h2_own_win": sum(f["own_win_now"] for f in fs),
                        "h4_forced_loss": sum(f["forced_loss"] for f in fs)},
        }
        wall = walls[name]
        unknown_ids = {x for x, v in wall.items() if v["status"] == "unknown" and x in feats}
        total_unknown_wall = sum(wall[x]["wall_s"] for x in unknown_ids)
        for candidate in candidates:
            at_d = {f["pos_id"] for f in fs if fires(f, candidate, f["packed_fill_remaining"])}
            at_inf = {f["pos_id"] for f in fs if fires(f, candidate, None)}
            rungs = {}
            for h in (2, 4, 8, 16, 32):
                at_h = {f["pos_id"] for f in fs if fires(f, candidate, h)}
                rungs[str(h)] = {
                    "hits": len(at_h), "pct": 100 * len(at_h) / len(fs),
                    "unknown_wall_dismissed_pct": 100 * sum(wall[x]["wall_s"] for x in unknown_ids & at_h) / total_unknown_wall if total_unknown_wall else 0,
                }
            cohort["candidates"][candidate] = {
                "rungs": rungs,
                "at_packed_D": len(at_d), "at_packed_D_pct": 100 * len(at_d) / len(fs),
                "at_infinity": len(at_inf), "at_infinity_pct": 100 * len(at_inf) / len(fs),
                "unknown_wall_source_n": len(unknown_ids), "unknown_wall_s": total_unknown_wall,
                "unknown_wall_dismissed_at_D_pct": 100 * sum(wall[x]["wall_s"] for x in unknown_ids & at_d) / total_unknown_wall if total_unknown_wall else 0,
                "unknown_wall_dismissed_at_infinity_pct": 100 * sum(wall[x]["wall_s"] for x in unknown_ids & at_inf) / total_unknown_wall if total_unknown_wall else 0,
            }
        result["cohorts"][name] = cohort

    # Stronger-than-requested audit for horizon-independent candidates: any
    # hit on a known WIN is reported, without relying on a possibly loose cap.
    known: dict[str, dict] = {}

    def add_known(pos_id: str, source: str, cert_depth: int | None = None) -> None:
        prior = known.get(pos_id)
        depth = cert_depth if cert_depth and cert_depth > 0 else None
        if prior is None or (prior["cert_depth"] is None and depth is not None):
            known[pos_id] = {"source": source, "cert_depth": depth}
    for row in cohorts["puzzle_v3"]:
        if row.get("labels", {}).get("verdict") == "win":
            add_known(row["pos_id"], "puzzle_v3 labeled WIN (includes atlas-certified rows)")
    for row in cohorts["forcing19"]:
        if row["labels"]["verdict"] == "win":
            add_known(row["pos_id"], "forcing19 expected WIN")
    for cohort_name, wall in walls.items():
        for pos_id, rec in wall.items():
            if rec["status"] == "win" and pos_id in feats:
                add_known(pos_id, f"{cohort_name} measured WIN", rec.get("cert_depth"))
    atlas_feature_ids = []
    if args.atlas.is_file():
        atlas_doc = json.loads(args.atlas.read_text(encoding="utf-8"))
        for row in atlas_doc["rows"]:
            if row.get("status") != "WIN" or not row.get("certified"):
                continue
            pos_id = f"atlas_full_{row['id']}"
            atlas_feature_ids.append(pos_id)
            feats[pos_id] = feature({"pos_id": pos_id, "moves": row["moves"]})
            derived = row.get("derived_horizon")
            depth = derived - row["placements"] if isinstance(derived, int) else None
            add_known(pos_id, "opening atlas certified WIN", depth)
    for candidate in candidates:
        violations = []
        depth_tested = 0
        for pos_id, info in known.items():
            if candidate in ("census_lb", "stone_deficit_lb"):
                horizon = info["cert_depth"]
                if horizon is None:
                    horizon = feats[pos_id]["packed_fill_remaining"]
                else:
                    depth_tested += 1
            else:
                horizon = None
            if fires(feats[pos_id], candidate, horizon):
                violations.append(pos_id)
        result["false_dismissal_audit"][candidate] = {
            "known_wins": len(known), "violations": len(violations),
            "atlas_certified_wins": len(atlas_feature_ids),
            "certified_depth_tests": depth_tested,
            "examples": [{"pos_id": x, **known[x]} for x in violations[:20]],
        }

    result["small_h"] = {
        "positions": len(all_rows),
        "h2_own_win": sum(feats[x]["own_win_now"] for x in cohort_feature_ids),
        "h4_forced_loss": sum(feats[x]["forced_loss"] for x in cohort_feature_ids),
        "definition_note": "h2 is phase-aware completion in the mover's remaining turn; h4 is the opponent-win dual from an unanswerable threat family (resolution <= b+2 <= 4)",
    }
    if args.engine_check:
        result["small_h"]["engine_check"] = engine_check(all_rows.values(), feats, args.package_root)

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "cohorts": result["metadata"]["cohort_sizes"],
                      "known_wins": len(known), "small_h": result["small_h"]}, sort_keys=True))


if __name__ == "__main__":
    main()
