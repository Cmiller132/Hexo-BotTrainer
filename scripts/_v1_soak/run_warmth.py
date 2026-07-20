"""V1 SOAK warmth-sensitivity arm: re-solve each self-play GAME's positions in
ply order on ONE persistent solver (warm shared fragment cache), to bound the
cold-vs-warm gap of the single-shot probe. Emits raw records tagged with a
'_warm' arm suffix, aligned by pos_id to the cold sweep for a paired delta.

Warmth is PER-GAME (one `hexfield_eq_deep_solve_batch` call per game = one
persistent solver over that game's plies), so sharding games across worker
processes preserves the semantics exactly.

Usage:
    python run_warmth.py <positions.jsonl> <out_raw.jsonl> [n_shards] [shard_idx]
"""

from __future__ import annotations

import arch_env  # noqa: F401

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from hexfield_eq import _rust

import corpus_lib

NODE_CAP = 500
# name, goal, horizon, ladder, zone, wide
WARM_ARMS = [
    ("h16_flat_wide_warm", "win", 16, False, False, True),
    ("unbounded_wide_warm", "win", 0, False, False, True),
]


def main():
    positions_path = sys.argv[1]
    out_path = Path(sys.argv[2])
    n_shards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    shard_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    positions = [json.loads(l) for l in open(positions_path)]
    # Group by game, keep ply order (self-play only; corpus has no 'game').
    games = defaultdict(list)
    for p in positions:
        if p.get("source") == "selfplay" and int(p["game"]) % n_shards == shard_idx:
            games[p["game"]].append(p)
    for g in games.values():
        g.sort(key=lambda p: p["ply"])

    fh = open(out_path, "w")
    n = 0
    vf_total = 0
    t0 = time.time()
    for (name, goal, horizon, ladder, zone, wide) in WARM_ARMS:
        for game, ps in sorted(games.items()):
            states = [corpus_lib.build_state(p["moves"]) for p in ps]
            results = _rust.hexfield_eq_deep_solve_batch(
                states, NODE_CAP, goal, horizon, ladder, zone, wide
            )
            for p, r in zip(ps, results):
                vf_total += int(r["deep_verify_failed"])
                rec = {
                    "pos_id": p["id"],
                    "source": "selfplay",
                    "arm": name,
                    "placements": int(p.get("placements", len(p["moves"]))),
                    "band": int(p.get("placements", len(p["moves"]))) // 10,
                    "net_value": p.get("net_value"),
                }
                rec.update(r)
                fh.write(json.dumps(rec) + "\n")
                n += 1
                if int(r["deep_verify_failed"]):
                    fh.flush()
                    raise SystemExit(f"FATAL deep_verify_failed at {p['id']} {name}")
    fh.close()
    print(f"warmth: {n} solves in {time.time()-t0:.1f}s, deep_verify_failed_total={vf_total}")


if __name__ == "__main__":
    main()
