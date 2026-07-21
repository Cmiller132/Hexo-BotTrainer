"""Solver wall-latency benchmark at the exact main_4 production profile.

Measures per-solve wall/nodes over the frozen harness sets via the engine's
serial batch API (same entry the harness adapter uses), at the production
config: cap 500, goal=both, dual_pass ON, unbounded horizon, wide, zone OFF.

Usage (WSL, hexgt-build venv, from the consolidate-main worktree):
    python scripts/tss_harness/_bench_solver_wall.py \
        --sets human_v1,selfplay_v1,puzzle_v3 --out /tmp/solver_wall.jsonl
    # contention shard (same sample solved by N concurrent processes):
    python scripts/tss_harness/_bench_solver_wall.py \
        --sample-file /tmp/sample.jsonl --out /tmp/shard_3.jsonl --quiet-report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE / "scripts" / "_v1_soak"))
import arch_env  # noqa: F401  MUST precede hexfield_eq import
import corpus_lib
from hexfield_eq import _rust

SETS_DIR = WORKTREE / "scripts" / "tss_harness" / "sets"

PROD = dict(node_cap=500, horizon=0, ladder=False, zone=False, wide=True,
            dual_pass=True, loss_reserve_nodes=0, group2=False)


def load_rows(names):
    rows = []
    for name in names:
        for line in open(SETS_DIR / f"{name}.jsonl"):
            row = json.loads(line)
            row["set"] = name
            rows.append(row)
    return rows


def solve_all(rows, cfg, chunk=256, progress=True):
    out = []
    for base in range(0, len(rows), chunk):
        part = rows[base:base + chunk]
        states = [corpus_lib.build_state([tuple(m) for m in r["moves"]])
                  for r in part]
        t0 = time.time()
        raw = _rust.hexfield_eq_deep_solve_batch(
            states, int(cfg["node_cap"]), "both", int(cfg["horizon"]),
            bool(cfg["ladder"]), bool(cfg["zone"]), bool(cfg["wide"]),
            bool(cfg["dual_pass"]),
            loss_reserve_nodes=int(cfg["loss_reserve_nodes"]),
            group2=bool(cfg["group2"]),
        )
        dt = time.time() - t0
        for r, res in zip(part, raw):
            rec = {
                "pos_id": r["pos_id"], "set": r["set"],
                "stones": len(r["moves"]),
                "status": res["status"],
                "wall_ns": int(res["wall_nanos"]),
                "nodes": int(res["deep_nodes"]),
                "verify_failed": int(res["deep_verify_failed"]),
            }
            for k, v in res.items():
                if isinstance(v, int) and k not in ("wall_nanos", "deep_nodes",
                                                    "deep_verify_failed"):
                    rec[k] = v
            out.append(rec)
        if progress:
            done = base + len(part)
            print(f"  {done}/{len(rows)} chunk {dt:.1f}s", file=sys.stderr)
    return out


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
    return s[i]


def report(records):
    walls = [r["wall_ns"] / 1e6 for r in records]
    print(f"n={len(records)}  "
          f"p50={pct(walls,50):.2f}ms p90={pct(walls,90):.1f}ms "
          f"p99={pct(walls,99):.1f}ms max={max(walls):.1f}ms "
          f"total={sum(walls)/1e3:.1f}s")
    by_status = {}
    for r in records:
        by_status.setdefault(r["status"], []).append(r)
    for st, rs in sorted(by_status.items()):
        w = [r["wall_ns"] / 1e6 for r in rs]
        n = [r["nodes"] for r in rs]
        print(f"  {st:8s} n={len(rs):5d} p50={pct(w,50):8.2f}ms "
              f"p90={pct(w,90):8.1f}ms p99={pct(w,99):8.1f}ms "
              f"nodes p50={pct(n,50):5.0f} p90={pct(n,90):5.0f}")
    # Tail anatomy: everything over 100 ms.
    tail = [r for r in records if r["wall_ns"] >= 100e6]
    if tail:
        tn = [r["nodes"] for r in tail]
        per_node = [r["wall_ns"] / 1e3 / max(1, r["nodes"]) for r in tail]
        stones = [r["stones"] for r in tail]
        print(f"  TAIL>=100ms n={len(tail)} "
              f"({100.0*len(tail)/len(records):.1f}%) "
              f"nodes p50={pct(tn,50):.0f} p90={pct(tn,90):.0f} "
              f"us/node p50={pct(per_node,50):.0f} p90={pct(per_node,90):.0f} "
              f"stones p50={pct(stones,50):.0f}")
        wall_share = sum(r["wall_ns"] for r in tail) / max(
            1, sum(r["wall_ns"] for r in records))
        print(f"  tail wall share: {100.0*wall_share:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="human_v1,selfplay_v1,puzzle_v3")
    ap.add_argument("--sample-file", default=None,
                    help="jsonl of rows to solve instead of --sets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--node-cap", type=int, default=PROD["node_cap"])
    ap.add_argument("--dual-pass", type=int, default=1)
    ap.add_argument("--quiet-report", action="store_true")
    args = ap.parse_args()

    if args.sample_file:
        rows = [json.loads(l) for l in open(args.sample_file)]
        for r in rows:
            r.setdefault("set", "sample")
    else:
        rows = load_rows(args.sets.split(","))
    if args.limit:
        rows = rows[: args.limit]

    cfg = dict(PROD)
    cfg["node_cap"] = args.node_cap
    cfg["dual_pass"] = bool(args.dual_pass)

    t0 = time.time()
    records = solve_all(rows, cfg, progress=not args.quiet_report)
    wall = time.time() - t0

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    vf = sum(r["verify_failed"] for r in records)
    print(f"WROTE {len(records)} records to {args.out}  "
          f"battery wall {wall:.1f}s  verify_failed_total={vf}")
    if vf:
        print("FATAL: verifier failures present", file=sys.stderr)
        sys.exit(2)
    if not args.quiet_report:
        report(records)


if __name__ == "__main__":
    main()
