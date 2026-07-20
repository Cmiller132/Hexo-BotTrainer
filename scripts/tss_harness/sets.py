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
    # newline="\n" always: the pin hashes bytes, so line endings must be
    # platform-independent (a CRLF-written file breaks the pin after git's
    # LF normalization — caught at first mint, 2026-07-20).
    with open(path, "w", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (SETS_DIR / f"{name}.sha256").write_text(
        f"{digest}  {name}.jsonl\n", newline="\n"
    )
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


def build_human_v1() -> str:
    """SET-HUMAN-V1: 2,720 human OOD positions from
    corpus_lib.load_human_positions(340, 8) (seed 1234, first-340-games
    slice — file-order bias documented in corpus_lib). V1 itself soaked a
    320-position subslice (40x8) across 10 arms; the builder hard-asserts
    those 320 observed pos_ids are all CONTAINED here, so V1 anchors remain
    comparable while the frozen set carries 8.5x the coverage. Any drift
    (corpus edit, sampler change) fails the mint."""
    sys.path.insert(0, str(WORKTREE / "scripts" / "_v1_soak"))
    import corpus_lib

    observed = set()
    for line in open(RAWS / "soak_human.jsonl"):
        observed.add(json.loads(line)["pos_id"])

    generated = corpus_lib.load_human_positions(340, 8)
    gen_ids = {p["id"] for p in generated}
    if not observed <= gen_ids:
        raise RuntimeError(
            f"human slice drift: {len(observed - gen_ids)} V1-observed "
            f"pos_ids missing from the regenerated slice "
            f"(generated {len(gen_ids)}, observed {len(observed)})"
        )

    rows = [
        {
            "pos_id": p["id"], "source": "human", "moves": p["moves"],
            "meta": {k: p[k] for k in ("placements", "elo") if k in p},
        }
        for p in generated
    ]
    rows.sort(key=lambda r: r["pos_id"])
    return _write_pinned("human_v1", rows)


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


def build_canaries_v2() -> str:
    """v2 = v1 rows + loss_pos fixtures for the loss_detection canary:
    positions the V1 goal=loss arms proved LOSS (verified, deep_verify_failed
    == 0) with BOTH loss arms agreeing — take the 3 cheapest by nodes. The
    canary anchors the discovery (SOLVER_NOTES §5) that SolveGoal::Both under
    the wide profile allocates zero budget to the loss attempt: an arm
    claiming loss detection must actually produce these verdicts."""
    v1 = SETS_DIR / "canaries_v1.jsonl"
    rows = [json.loads(line) for line in open(v1)]

    positions = _v1_positions()
    by_arm: dict[tuple[str, str], dict] = {}
    for line in open(RAWS / "soak_selfplay.jsonl"):
        r = json.loads(line)
        if r["arm"].endswith("_loss"):
            by_arm[(r["pos_id"], r["arm"])] = r

    losses = []
    for pid, p in sorted(positions.items()):
        a = by_arm.get((pid, "h16_flat_wide_loss"))
        b = by_arm.get((pid, "unbounded_wide_loss"))
        if not a or not b:
            continue
        if (a["status"] == b["status"] == "loss"
                and a["deep_verify_failed"] == 0 and b["deep_verify_failed"] == 0):
            losses.append((int(b["deep_nodes"]), pid, p))
    if len(losses) < 3:
        raise RuntimeError(f"only {len(losses)} agreed verified losses in V1 raws")
    for nodes, pid, p in sorted(losses)[:3]:
        rows.append({
            "canary": "loss_pos", "pos_id": pid, "moves": p["moves"],
            "meta": {"loss_nodes": nodes},
        })
    return _write_pinned("canaries_v2", rows)


def build_puzzle_v1() -> str:
    """SET-PUZZLE-V1 from Lane C labels (raws/lanec_labels.jsonl, two-pass
    win-then-loss, provenance in the .manifest.json sidecar).

    Included rows and label provenance:
    - every Lane-C-decided position (verified win/loss at the labeling cap);
    - atlas sample rows UNDECIDED at the labeling cap keep their certified
      atlas verdict (deep labels) — allowed only if the decided atlas
      cross-solves agreed with the mover-perspective mapping with ZERO
      disagreements, else the mint aborts;
    - forcing-corpus expect_win priors are contradiction-checked (a verified
      verdict against the prior aborts the mint — either the corpus
      annotation or the engine is wrong, and that must be resolved by a
      human, not papered over).

    must_solve marks labels EVERY sound arm must decide regardless of its
    goal protocol — which after two mint iterations (2026-07-20) means WIN
    labels proven <= 400 nodes ONLY. Loss labels can never be must_solve
    for arbitrary-goal arms: v1 required dedicated-loss verdicts (70 atlas
    fails), v2 tried "cheap losses <= 50 nodes" and STILL failed 15 human
    positions — mechanism: when the primal win search width-exhausts (at 2
    nodes on those), Both returns Unknown with the whole remaining budget
    unused; it never asks the loss question at any price. The loss-side
    requirement lives in the loss_detection canary, which every arm
    CLAIMING loss coverage must pass. Deeper labels gate only
    contradictions.
    """
    labels_path = RAWS / "lanec_labels.jsonl"
    labels = [json.loads(line) for line in open(labels_path)]

    # moves are re-derived from the same sources the labeler used
    sys.path.insert(0, str(WORKTREE / "scripts" / "_v1_soak"))
    import corpus_lib
    moves_by_id: dict[str, list] = {}
    for p in _v1_positions().values():
        moves_by_id[p["id"]] = p["moves"]
    for p in corpus_lib.load_forcing_corpus():
        moves_by_id[f"forcing_{p['id']}"] = p["moves"]
    for p in corpus_lib.load_spare_corpus():
        moves_by_id[f"spare_{p['id']}"] = p["moves"]
    for line in open(SETS_DIR / "human_v1.jsonl"):
        r = json.loads(line)
        moves_by_id[r["pos_id"]] = list(r["moves"])
    atlas_path = Path(
        "/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/opening-atlas/"
        "atlas-web/data/atlas.json")
    if atlas_path.exists():
        data = json.load(open(atlas_path))
        rows_a = data["rows"] if isinstance(data, dict) and "rows" in data else data
        for r in rows_a:
            moves_by_id[f"atlas_{r['id']}"] = r["moves"]

    atlas_decided = [l for l in labels
                     if l["source"] == "atlas" and l["status"] in ("win", "loss")]
    disagree = [l for l in atlas_decided
                if l["status"] != l["prior"]["atlas_status"].lower()]
    if disagree:
        raise RuntimeError(
            f"atlas perspective mapping broken on {len(disagree)} rows "
            f"({[l['pos_id'] for l in disagree[:5]]}) — resolve before minting")

    for l in labels:
        if l["source"] == "forcing" and l["status"] in ("win", "loss"):
            expect = l.get("prior", {}).get("expect_win")
            if expect is True and l["status"] == "loss":
                raise RuntimeError(f"forcing prior contradiction: {l['pos_id']}")

    rows = []
    for l in labels:
        moves = moves_by_id.get(l["pos_id"])
        if moves is None:
            raise RuntimeError(f"no moves source for {l['pos_id']}")
        if l["status"] in ("win", "loss"):
            nodes = int(l.get("nodes", 1 << 30))
            must = l["status"] == "win" and nodes <= 400
            rows.append({
                "pos_id": l["pos_id"], "source": l["source"], "moves": moves,
                "labels": {
                    "verdict": l["status"],
                    "must_solve": must,
                    "label_cap": l["cap"], "label_nodes": l.get("nodes"),
                    "protocol": "lanec_two_pass",
                },
            })
        elif l["source"] == "atlas" and l["status"] == "unknown":
            rows.append({
                "pos_id": l["pos_id"], "source": "atlas_deep", "moves": moves,
                "labels": {
                    "verdict": l["prior"]["atlas_status"].lower(),
                    "must_solve": False,
                    "label_cap": l["prior"].get("placements"),
                    "protocol": "atlas_certified",
                },
            })
    rows.sort(key=lambda r: r["pos_id"])
    return _write_pinned("puzzle_v3", rows)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "selfplay"):
        print("selfplay_v1:", build_selfplay_v1())
    if which in ("all", "human"):
        print("human_v1:", build_human_v1())
    if which in ("all", "canaries"):
        print("canaries_v1:", build_canaries_v1())
    if which in ("all", "canaries_v2"):
        print("canaries_v2:", build_canaries_v2())
    if which == "puzzle":
        print("puzzle_v3:", build_puzzle_v1())


if __name__ == "__main__":
    main()
