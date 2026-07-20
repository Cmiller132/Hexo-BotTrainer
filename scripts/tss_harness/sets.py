"""Frozen position sets — build, pin, load. PLAN §2.

Sets are versioned files under scripts/tss_harness/sets/ with a sha256
sidecar manifest; NEVER edited in place (new version = new file + hash).
Every set row: {pos_id, source, moves, meta, labels?}. Each coverage set is
split deterministically into dev/holdout by pos_id hash (holdout consumed
only at adoption gates — PLAN §2 owner ruling).

Builders read the frozen V1 artifacts (raws/) so every fixture is anchored
to an OBSERVED behavior, not a synthetic hope. Canary fixture selection is
data-driven from the V1 raws + the fragment rerun (SOLVER_NOTES P3).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from .contract import Position, stable_hash

ROOT = Path(__file__).resolve().parent
SETS_DIR = ROOT / "sets"
WORKTREE = ROOT.parents[1]
RAWS = WORKTREE / "raws"

HOLDOUT_FRACTION = 0.2


def _holdout(pos_id: str) -> bool:
    h = int(hashlib.sha256(f"split:{pos_id}".encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < HOLDOUT_FRACTION


def load_set(name: str, split: str = "dev") -> list[Position]:
    """split: 'dev' | 'holdout' | 'all'."""
    path = SETS_DIR / f"{name}.jsonl"
    pinned = (SETS_DIR / f"{name}.sha256").read_text().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != pinned:
        raise RuntimeError(f"set {name} hash mismatch: {actual} != {pinned}")
    out = []
    for line in open(path):
        row = json.loads(line)
        is_hold = _holdout(row["pos_id"])
        if split == "dev" and is_hold:
            continue
        if split == "holdout" and not is_hold:
            continue
        out.append(Position(
            pos_id=row["pos_id"], source=row["source"],
            moves=tuple(row["moves"]), meta=row.get("meta", {}),
            labels=row.get("labels"),
        ))
    return out


def _write_pinned(name: str, rows: list[dict]) -> str:
    SETS_DIR.mkdir(exist_ok=True)
    path = SETS_DIR / f"{name}.jsonl"
    if path.exists():
        raise RuntimeError(
            f"{path} exists — sets are never edited in place; bump the version"
        )
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (SETS_DIR / f"{name}.sha256").write_text(f"{digest}  {name}.jsonl\n")
    return digest


def _v1_positions() -> dict[str, dict]:
    out = {}
    for line in open(RAWS / "selfplay_positions.jsonl"):
        row = json.loads(line)
        out[row["id"]] = row
    return out


def build_selfplay_v1() -> str:
    """SET-SELFPLAY-V1: the 3,255 frozen V1 selfplay positions."""
    rows = [
        {
            "pos_id": p["id"], "source": "selfplay", "moves": p["moves"],
            "meta": {
                "game": p.get("game"), "ply": p.get("ply"),
                "placements": p.get("placements"),
                "net_value": p.get("net_value"),
            },
        }
        for p in _v1_positions().values()
    ]
    rows.sort(key=lambda r: r["pos_id"])
    return _write_pinned("selfplay_v1", rows)


def build_canaries_v1() -> str:
    """Canary fixtures, selected from observed V1 behavior:

    - warmth_sequence: game 28 plies 40..47 in order (fragment rerun measured
      store engagement: sp_28_p47 71->2 nodes, sp_28_p44 203->62).
    - deep_win: unbounded-WIN positions with cert_depth > 16 whose h16 twin
      returned Unknown (the 39-win class; take the 3 cheapest by nodes).
    - wide_win: 3 stable quick wins (WIN in both arms, < 50 nodes) for the
      wide fixture assertion.
    """
    positions = _v1_positions()
    by_arm: dict[tuple[str, str], dict] = {}
    for line in open(RAWS / "soak_selfplay.jsonl"):
        r = json.loads(line)
        by_arm[(r["pos_id"], r["arm"])] = r

    rows = []
    for pid, p in sorted(positions.items()):
        if p.get("game") == 28 and 40 <= int(p.get("ply", -1)) <= 47:
            rows.append({
                "canary": "warmth_sequence", "pos_id": pid,
                "moves": p["moves"], "meta": {"ply": p["ply"]},
            })

    deep, quick = [], []
    for pid, p in positions.items():
        ub = by_arm.get((pid, "unbounded_wide"))
        h16 = by_arm.get((pid, "h16_flat_wide"))
        if not ub or not h16 or ub["status"] != "win":
            continue
        if h16["status"] == "unknown" and int(ub.get("cert_depth", 0)) > 16:
            deep.append((int(ub["deep_nodes"]), pid, p))
        if h16["status"] == "win" and int(ub["deep_nodes"]) < 50:
            quick.append((int(ub["deep_nodes"]), pid, p))
    for nodes, pid, p in sorted(deep)[:3]:
        rows.append({
            "canary": "deep_win", "pos_id": pid, "moves": p["moves"],
            "meta": {"unbounded_nodes": nodes},
        })
    for nodes, pid, p in sorted(quick)[:3]:
        rows.append({
            "canary": "wide_win", "pos_id": pid, "moves": p["moves"],
            "meta": {"unbounded_nodes": nodes},
        })

    kinds = {r["canary"] for r in rows}
    missing = {"warmth_sequence", "deep_win", "wide_win"} - kinds
    if missing:
        raise RuntimeError(f"canary selection came up empty for: {missing}")
    return _write_pinned("canaries_v1", rows)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "selfplay"):
        print("selfplay_v1:", build_selfplay_v1())
    if which in ("all", "canaries"):
        print("canaries_v1:", build_canaries_v1())


if __name__ == "__main__":
    main()
