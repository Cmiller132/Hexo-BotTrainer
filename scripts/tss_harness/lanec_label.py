"""Lane C — ground-truth labeling for SET-PUZZLE-V1 (PLAN §2, owner ruling:
all four sources). Two-pass protocol per position (win pass then, if
Unknown, loss pass — each with the FULL node budget, sidestepping the
wide-profile Both starvation, SOLVER_NOTES §6 P4). Every verdict passes the
strict verifier (batch API = tss_solve_verified); labels carry provenance.

Budgets (first deep pass; TT is the harness-fixed 256KiB, so grinds that
stay Unknown at cap are flagged rather than believed — the big-TT/parallel
cargo lane re-visits them):
    grinds  50,000 nodes (100x the production cap that defined them)
    others  20,000 nodes

Atlas rows are already-certified WIN/LOSS: we do NOT relabel them; we
cross-solve a sample and require ZERO disagreements to validate the
status->mover-perspective mapping before any of them are minted.

Usage (harness-dev venv, worktree root):
    python scripts/tss_harness/lanec_label.py --pilot     # rate check
    python scripts/tss_harness/lanec_label.py --workers 6 \
        --out raws/lanec_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAWS = ROOT / "raws"
SETS = ROOT / "scripts" / "tss_harness" / "sets"
V1_SOAK = ROOT / "scripts" / "_v1_soak"
ATLAS_JSON = Path(
    "/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/opening-atlas/atlas-web/data/atlas.json"
)

GRIND_CAP = 50_000
BASE_CAP = 20_000
ATLAS_SAMPLE = 300
ATLAS_SEED = 20260720


def _engine():
    if str(V1_SOAK) not in sys.path:
        sys.path.insert(0, str(V1_SOAK))
    import arch_env  # noqa: F401
    import corpus_lib
    from hexfield_eq import _rust
    return corpus_lib, _rust


# ----------------------------------------------------------------- sources #

def _selfplay_moves() -> dict[str, list]:
    out = {}
    for line in open(RAWS / "selfplay_positions.jsonl"):
        r = json.loads(line)
        out[r["id"]] = r["moves"]
    return out


def load_tasks() -> list[dict]:
    corpus_lib, _ = _engine()
    tasks: list[dict] = []

    moves_by_id = _selfplay_moves()
    seen = set()
    for line in open(RAWS / "soak_selfplay.jsonl"):
        r = json.loads(line)
        if (r["arm"] == "unbounded_wide" and r["status"] == "unknown"
                and r["deep_nodes"] >= 500 and r["pos_id"] not in seen):
            seen.add(r["pos_id"])
            tasks.append({
                "pos_id": r["pos_id"], "source": "grind",
                "moves": moves_by_id[r["pos_id"]], "cap": GRIND_CAP,
            })

    for p in corpus_lib.load_forcing_corpus():
        tasks.append({
            "pos_id": f"forcing_{p['id']}", "source": "forcing",
            "moves": p["moves"], "cap": BASE_CAP,
            "prior": {k: p[k] for k in ("expect_win",) if k in p},
        })
    for p in corpus_lib.load_spare_corpus():
        tasks.append({
            "pos_id": f"spare_{p['id']}", "source": "spare",
            "moves": p["moves"], "cap": BASE_CAP,
            "prior": {k: p[k] for k in ("expect_win",) if k in p},
        })

    human_ids = set()
    for line in open(RAWS / "soak_human.jsonl"):
        human_ids.add(json.loads(line)["pos_id"])
    for line in open(SETS / "human_v1.jsonl"):
        r = json.loads(line)
        if r["pos_id"] in human_ids:
            tasks.append({
                "pos_id": r["pos_id"], "source": "human",
                "moves": r["moves"], "cap": BASE_CAP,
            })

    tasks.extend(load_atlas_tasks())
    return tasks


def load_atlas_tasks() -> list[dict]:
    """Stratified certified sample. The atlas verdict is carried as a PRIOR
    (atlas_status/claimant/side) — the mapping to mover-perspective is
    validated after solving, never assumed."""
    import random
    if not ATLAS_JSON.exists():
        print("atlas.json not found — skipping atlas source", file=sys.stderr)
        return []
    data = json.load(open(ATLAS_JSON))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    cert = [r for r in rows if r.get("certified")]
    rng = random.Random(ATLAS_SEED)
    wins = [r for r in cert if r["status"] == "WIN"]
    losses = [r for r in cert if r["status"] == "LOSS"]
    picked = (rng.sample(wins, min(ATLAS_SAMPLE // 2, len(wins)))
              + rng.sample(losses, min(ATLAS_SAMPLE // 2, len(losses))))
    return [{
        "pos_id": f"atlas_{r['id']}", "source": "atlas",
        "moves": r["moves"], "cap": BASE_CAP,
        "prior": {"atlas_status": r["status"], "claimant": r.get("claimant"),
                  "side": r.get("side"), "placements": r.get("placements")},
    } for r in picked]


# ------------------------------------------------------------------ worker #

def solve_shard(shard: list[dict]) -> list[dict]:
    corpus_lib, _rust = _engine()
    out = []
    for t in shard:
        try:
            state = corpus_lib.build_state(list(t["moves"]))
        except Exception as exc:  # data noise: report, never silently drop
            out.append({**t, "status": "build_error", "error": str(exc)})
            continue
        cap = int(t["cap"])
        rec = {"pos_id": t["pos_id"], "source": t["source"], "cap": cap,
               "prior": t.get("prior", {})}
        t0 = time.time()
        win = _rust.hexfield_eq_deep_solve_batch(
            [state], cap, "win", 0, False, False, True)[0]
        rec["win_pass"] = {k: win[k] for k in
                          ("status", "deep_nodes", "deep_verify_failed")}
        if win["status"] == "win" and win["deep_verify_failed"] == 0:
            rec.update(status="win", nodes=int(win["deep_nodes"]))
        else:
            loss = _rust.hexfield_eq_deep_solve_batch(
                [state], cap, "loss", 0, False, False, True)[0]
            rec["loss_pass"] = {k: loss[k] for k in
                               ("status", "deep_nodes", "deep_verify_failed")}
            if loss["status"] == "loss" and loss["deep_verify_failed"] == 0:
                rec.update(status="loss",
                           nodes=int(win["deep_nodes"]) + int(loss["deep_nodes"]))
            else:
                rec.update(status="unknown",
                           nodes=int(win["deep_nodes"]) + int(loss["deep_nodes"]))
                # 256KiB TT at these caps saturates; an Unknown here is a
                # budget statement, not a truth claim.
                rec["tt_saturation_suspect"] = int(win["deep_nodes"]) >= cap
        rec["wall_s"] = round(time.time() - t0, 3)
        out.append(rec)
    return out


# ------------------------------------------------------------------ driver #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(RAWS / "lanec_labels.jsonl"))
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    tasks = load_tasks()
    by_source: dict[str, int] = {}
    for t in tasks:
        by_source[t["source"]] = by_source.get(t["source"], 0) + 1
    print(f"tasks: {len(tasks)} {by_source}", flush=True)

    if args.pilot:
        pilot = ([t for t in tasks if t["source"] == "grind"][:3]
                 + [t for t in tasks if t["source"] == "forcing"][:5]
                 + [t for t in tasks if t["source"] == "atlas"][:6])
        t0 = time.time()
        recs = solve_shard(pilot)
        for r in recs:
            print(json.dumps({k: r.get(k) for k in
                              ("pos_id", "source", "status", "nodes",
                               "wall_s", "prior")}), flush=True)
        print(f"pilot wall: {time.time() - t0:.1f}s for {len(pilot)}", flush=True)
        return 0

    _, _rust = _engine()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    manifest = {
        "protocol": "two_pass_win_then_loss_full_budget",
        "git_rev": rev,
        "caps": {"grind": GRIND_CAP, "base": BASE_CAP},
        "engine_manifest": _rust.hexfield_eq_solver_manifest(
            BASE_CAP, 0, False, False, True),
        "atlas_source": str(ATLAS_JSON),
        "atlas_sample_seed": ATLAS_SEED,
    }
    out_path = Path(args.out)
    Path(str(out_path) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), newline="\n")

    n = max(1, args.workers)
    shards = [tasks[i::n] for i in range(n)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(n) as pool:
        results = pool.map(solve_shard, shards)
    flat = [r for shard in results for r in shard]
    order = {t["pos_id"]: i for i, t in enumerate(tasks)}
    flat.sort(key=lambda r: order.get(r["pos_id"], 1 << 30))
    with open(out_path, "w", newline="\n") as fh:
        for r in flat:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    from collections import Counter
    print(f"done in {time.time() - t0:.0f}s -> {out_path}", flush=True)
    for src in by_source:
        c = Counter(r["status"] for r in flat if r["source"] == src)
        print(f"  {src}: {dict(c)}", flush=True)

    # Atlas mapping validation: every decided cross-solve must agree with a
    # consistent orientation. Disagreements => mapping bug => atlas rows are
    # NOT usable as labels until resolved.
    agree = disagree = 0
    for r in flat:
        if r["source"] != "atlas" or r["status"] not in ("win", "loss"):
            continue
        # hypothesis: atlas status is mover-perspective (WIN = mover wins)
        want = r["prior"]["atlas_status"].lower()
        if r["status"] == want:
            agree += 1
        else:
            disagree += 1
            print(f"  ATLAS DISAGREE {r['pos_id']}: solved {r['status']} "
                  f"vs atlas {want} (claimant={r['prior'].get('claimant')} "
                  f"side={r['prior'].get('side')})", flush=True)
    print(f"atlas mapping: {agree} agree / {disagree} disagree "
          f"(decided cross-solves only)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
