"""Phase-4 validation: expand-time targets vs dense_cnn raw facts (real shards).

Reads recorded compact shards (read-only), expands each row through the shared
Rust path, and reports: dropped policy/opp visit mass by candidate_radius, the
fraction of rows with non-zero drop, and that value/short-term-value targets
round-trip from the row. Confirms the recompute-at-expand path is faithful and
that at n=8 dropped mass is ~0.

Usage:
  python scripts/_validate_hexgt_expand.py --runs <selfplay_dir> --max-games N
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from hexo_models.dense_cnn.compact_io import read_compact_shard, read_shard_horizons
from hexo_models.hexgt.expand import expand_row_to_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[
        "runs/dense_cnn_model1_target_96x8/selfplay",
    ])
    ap.add_argument("--max-games", type=int, default=20)
    ap.add_argument("--radii", nargs="*", type=int, default=[2, 8])
    args = ap.parse_args()

    npzs = []
    for run in args.runs:
        npzs.extend(sorted(Path(run).glob("*_game_*.npz")))
    npzs = npzs[: args.max_games]
    if not npzs:
        raise SystemExit(f"no shards under {args.runs}")

    per_n = {n: {"rows": 0, "drop_mass": [], "rows_with_drop": 0, "cands": []} for n in args.radii}
    value_ok = 0
    value_total = 0

    for npz in npzs:
        try:
            horizons = read_shard_horizons(npz)
            rows = read_compact_shard(npz)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {npz.name}: {exc}")
            continue
        for row in rows:
            total_visit = sum(w for _a, w in row.policy) or 1.0
            for n in args.radii:
                exp = expand_row_to_graph(row, n=n, horizons=horizons)
                per_n[n]["rows"] += 1
                frac = exp.dropped_policy_mass / total_visit
                per_n[n]["drop_mass"].append(frac)
                if exp.dropped_policy_mass > 1e-9:
                    per_n[n]["rows_with_drop"] += 1
                per_n[n]["cands"].append(int(exp.graph.candidate_ids.shape[0]))
                if n == args.radii[-1]:
                    value_total += 1
                    if abs(exp.value - float(row.value)) < 1e-6:
                        value_ok += 1

    print(f"\n=== hexgt Phase-4 expand validation ({len(npzs)} games) ===")
    print(f"value target round-trip: {value_ok}/{value_total}")
    for n in args.radii:
        d = per_n[n]
        if not d["rows"]:
            continue
        mean_drop = sum(d["drop_mass"]) / len(d["drop_mass"])
        p95_drop = sorted(d["drop_mass"])[min(len(d["drop_mass"]) - 1, int(0.95 * len(d["drop_mass"])))]
        med_cand = statistics.median(d["cands"])
        print(
            f"  n={n}: rows={d['rows']} mean_dropped_mass={mean_drop*100:.3f}% "
            f"p95={p95_drop*100:.3f}% rows_with_any_drop={100*d['rows_with_drop']/d['rows']:.1f}% "
            f"median_candidates={med_cand:.0f}"
        )


if __name__ == "__main__":
    main()
