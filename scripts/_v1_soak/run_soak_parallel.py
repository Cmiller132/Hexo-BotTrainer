"""V1 SOAK phase 2, parallel driver: shard the deep-solve arm sweep across
worker processes. Record schema, arm matrix, and the LOSS-subsample rule are
IDENTICAL to run_soak.py (arms imported from it; the LOSS flag is computed from
the GLOBAL post-load iteration index before sharding, so the subsample matches
what the serial driver would have picked). No thinning here — the full position
set is solved (max_positions is intentionally unsupported; parallelism makes
thinning unnecessary, removing the sampling caveat).

deep_verify_failed is asserted after every solve in every worker; any nonzero
aborts that worker (exit 2) and the parent propagates the failure with raws
preserved.

Usage:
    python run_soak_parallel.py <positions_spec> <out_raw.jsonl> <loss_every_n> <n_workers>
    python run_soak_parallel.py --worker <shard_in.jsonl> <shard_out.jsonl>
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def worker(in_path, out_path):
    import arch_env  # noqa: F401  keep import-time arch consistent across phases

    from hexfield_eq import _rust

    import corpus_lib
    import run_soak

    positions = [json.loads(l) for l in open(in_path)]
    fh = open(out_path, "w")
    n = 0
    t0 = time.time()
    try:
        for pos in positions:
            state = corpus_lib.build_state(pos["moves"])
            flags = run_soak.lambda1_flags(state)
            placements = int(pos.get("placements", len(pos["moves"])))
            band = placements // 10
            arms = list(run_soak.WIN_ARMS)
            if pos.get("_loss"):
                arms = arms + run_soak.LOSS_ARMS
            for (name, goal, horizon, ladder, zone, wide, with_stats) in arms:
                r = _rust.hexfield_eq_deep_solve_probe(
                    state, run_soak.NODE_CAP, goal, horizon, ladder, zone, wide, with_stats
                )
                n += 1
                vf = int(r["deep_verify_failed"])
                rec = {
                    "pos_id": pos.get("id"),
                    "source": pos.get("source", "unknown"),
                    "arm": name,
                    "placements": placements,
                    "band": band,
                    "hot": flags["hot"],
                    "opp_threats": flags["opp_threats"],
                    "min_hitting_set": flags["min_hitting_set"],
                    "net_value": pos.get("net_value"),
                    "expect_win": pos.get("expect_win"),
                }
                rec.update(r)
                fh.write(json.dumps(rec) + "\n")
                if vf:
                    fh.flush()
                    raise SystemExit(
                        f"FATAL deep_verify_failed={vf} at pos {pos.get('id')} arm {name}; "
                        f"raws preserved at {out_path}"
                    )
    finally:
        fh.close()
    print(f"worker {Path(in_path).name}: {len(positions)} positions, {n} solves "
          f"in {time.time()-t0:.1f}s", flush=True)


def parent(spec, out_path, loss_every_n, n_workers):
    import run_soak

    positions = run_soak.load_positions(spec)
    # Global-index LOSS flag BEFORE sharding (matches serial idx % loss_every_n).
    for idx, p in enumerate(positions):
        p["_loss"] = bool(loss_every_n > 0 and idx % loss_every_n == 0)
    shard_dir = out_path.parent / "_shards"
    shard_dir.mkdir(exist_ok=True)
    procs = []
    for i in range(n_workers):
        sh = positions[i::n_workers]
        spath = shard_dir / f"{out_path.stem}_in{i}.jsonl"
        opath = shard_dir / f"{out_path.stem}_out{i}.jsonl"
        with open(spath, "w") as f:
            for p in sh:
                f.write(json.dumps(p) + "\n")
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--worker", str(spath), str(opath)]
        ))
    n_loss = sum(1 for p in positions if p["_loss"])
    print(f"soaking {len(positions)} positions ({n_loss} with LOSS arms) from "
          f"{spec!r} across {n_workers} workers -> {out_path}", flush=True)
    rcs = [pr.wait() for pr in procs]
    n = 0
    vf = 0
    with open(out_path, "w") as out:
        for i in range(n_workers):
            opath = shard_dir / f"{out_path.stem}_out{i}.jsonl"
            if not opath.exists():
                continue
            for line in open(opath):
                out.write(line)
                n += 1
                vf += int(json.loads(line).get("deep_verify_failed", 0))
    print(f"PARALLEL DONE workers={n_workers} rcs={rcs} records={n} "
          f"deep_verify_failed_total={vf}", flush=True)
    if any(rcs) or vf:
        sys.exit(2)


def main():
    if sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        return
    spec = sys.argv[1]
    out_path = Path(sys.argv[2])
    loss_every_n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    parent(spec, out_path, loss_every_n, n_workers)


if __name__ == "__main__":
    main()
