"""Phase-1 validation gates for the hexgt candidate-set / graph construction.

HEXFORMER_REWRITE_PLAN.md §4.6 — measured against dense_cnn's RECORDED games
(read-only, from completed runs; never the live run):

  (1) Completeness/safety: fraction of dense_cnn's actually-PLAYED moves and
      MCTS-VISITED moves (by count and by visit mass) inside candidate_set,
      swept n = 2..8. Target ~= 100%.
  (2) Candidate / active-window size distribution by game phase.
  (3) Node- and edge-count distribution by type (the no-explosion gate):
      confirm total edges are LINEAR in (#nodes + #windows).

Usage:
  python scripts/_validate_hexgt_candidates.py [--runs DIR ...] [--max-games N]

Reconstructs each recorded position by replaying its placement history through
the engine, then calls the shared Rust builder (rust_bridge.graph_facts /
candidate_ids) — the same path sample-gen and MCTS will use.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import hexo_engine as engine
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id

from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt import rust_bridge

RADII = list(range(2, 9))


def _reconstruct(history) -> object:
    """Replay a placement history (idx-ordered) into a fresh engine state."""

    ordered = sorted(history, key=lambda h: h[4])  # entry[4] = placement_index
    state = engine.new_game()
    for q, r, *_rest in ((h[0], h[1]) for h in ordered):
        engine.apply_action(state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return state


def _coord(action_id: int) -> tuple[int, int]:
    c = unpack_coord_id(int(action_id))
    return (int(c.q), int(c.r))


def _phase_bucket(n_stones: int) -> str:
    if n_stones <= 8:
        return "opening"
    if n_stones <= 40:
        return "midgame"
    return "endgame"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="*",
        default=[
            "runs/dense_cnn_model1_target_96x8/selfplay",
        ],
    )
    ap.add_argument("--max-games", type=int, default=40)
    ap.add_argument("--default-n", type=int, default=3)
    ap.add_argument("--recent", action="store_true", help="sample the most recent (highest-epoch) games")
    args = ap.parse_args()

    npzs: list[Path] = []
    for run in args.runs:
        npzs.extend(sorted(Path(run).glob("*_game_*.npz")))
    npzs = sorted(npzs)
    if args.recent:
        # Highest-epoch games are the best data; skip the single latest epoch
        # (the run may have stopped mid-epoch, leaving partial files).
        def _epoch(p: Path) -> str:
            return p.name.split("_game_")[0]
        epochs = sorted({_epoch(p) for p in npzs})
        if len(epochs) > 1:
            npzs = [p for p in npzs if _epoch(p) != epochs[-1]]
        npzs = npzs[-args.max_games:]
    else:
        npzs = npzs[: args.max_games]
    if not npzs:
        raise SystemExit(f"no game npz files found under {args.runs}")
    print(f"sampling {len(npzs)} games: {npzs[0].name} .. {npzs[-1].name}")

    # coverage[phase][n] -> [played_total, played_cov, vcount_total, vcount_cov, vmass_total, vmass_cov]
    def _cov_block():
        return {n: [0, 0, 0, 0, 0.0, 0.0] for n in RADII}

    cov_by_phase: dict[str, dict] = defaultdict(_cov_block)
    cov = _cov_block()  # overall
    # hex-distance of MISSED played moves (at default-n) to nearest stone
    miss_dist: dict[int, int] = defaultdict(int)
    # size dists at default-n, by phase
    sizes: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    fallback_positions = 0
    total_positions = 0
    games_done = 0

    for npz in npzs:
        try:
            rows = read_compact_shard(npz)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {npz.name}: {exc}")
            continue
        if not rows:
            continue
        # Per-row candidate sets (coord space) for each n, plus stones set.
        row_cands: list[dict[int, set]] = []
        row_stones: list[set] = []
        for row in rows:
            stones = {(int(q), int(r)) for q, r, _p in row.stones}
            row_stones.append(stones)
            state = _reconstruct(row.placement_history)
            cands_by_n = {}
            for n in RADII:
                ids = rust_bridge.candidate_ids(state, n)
                cands_by_n[n] = {_coord(a) for a in ids}
            row_cands.append(cands_by_n)
            total_positions += 1

            # visited-move coverage (this position's MCTS visit distribution)
            visited = [(_coord(a), float(w)) for a, w in row.policy]
            mass = sum(w for _c, w in visited) or 1.0
            phase = _phase_bucket(len(stones))
            for n in RADII:
                cset = cands_by_n[n]
                vcc = sum(1 for c, _w in visited if c in cset)
                vmc = sum(w for c, w in visited if c in cset)
                for blk in (cov[n], cov_by_phase[phase][n]):
                    blk[2] += len(visited)
                    blk[3] += vcc
                    blk[4] += mass
                    blk[5] += vmc

            # size dists at default-n
            facts = rust_bridge.graph_facts(state, args.default_n)
            if facts["used_legal_fallback"]:
                fallback_positions += 1
            nc = facts["node_counts"]
            ec = facts["edge_counts"]
            sizes[phase]["candidate"].append(int(nc["candidate"]))
            sizes[phase]["window"].append(int(nc["window"]))
            sizes[phase]["total_nodes"].append(int(nc["total_nodes"]))
            sizes[phase]["total_edges"].append(int(ec["total_edges"]))
            sizes[phase]["edges_per_node_x100"].append(
                int(100 * ec["total_edges"] / max(1, nc["total_nodes"]))
            )

        # played-move coverage from consecutive rows (single new stone each)
        for i in range(len(rows) - 1):
            new = row_stones[i + 1] - row_stones[i]
            if len(new) != 1:
                continue  # game boundary / non-consecutive
            played = next(iter(new))
            phase = _phase_bucket(len(row_stones[i]))
            for n in RADII:
                for blk in (cov[n], cov_by_phase[phase][n]):
                    blk[0] += 1
                    if played in row_cands[i][n]:
                        blk[1] += 1
            # characterize misses at default-n: hex-distance to nearest stone
            if row_stones[i] and played not in row_cands[i][args.default_n]:
                d = min(
                    max(abs(played[0] - s[0]), abs(played[1] - s[1]),
                        abs((-played[0] - played[1]) - (-s[0] - s[1])))
                    for s in row_stones[i]
                )
                miss_dist[int(d)] += 1
        games_done += 1

    # ---- report ----
    print(f"\n=== hexgt Phase-1 validation ===")
    print(f"games={games_done} positions={total_positions} legal_fallback_positions={fallback_positions}")
    def _print_cov(block):
        print(f"  {'n':>3} | {'played%':>9} | {'visited_count%':>14} | {'visited_mass%':>13}")
        for n in RADII:
            p_tot, p_cov, vc_tot, vc_cov, vm_tot, vm_cov = block[n]
            played_pct = 100.0 * p_cov / p_tot if p_tot else float("nan")
            vcount_pct = 100.0 * vc_cov / vc_tot if vc_tot else float("nan")
            vmass_pct = 100.0 * vm_cov / vm_tot if vm_tot else float("nan")
            print(f"  {n:>3} | {played_pct:>8.3f}% | {vcount_pct:>13.3f}% | {vmass_pct:>12.3f}%")

    print("\n(1) COVERAGE sweep, OVERALL (target ~100%):")
    _print_cov(cov)
    for phase in ("opening", "midgame", "endgame"):
        if phase in cov_by_phase:
            print(f"\n(1) COVERAGE sweep, [{phase}]:")
            _print_cov(cov_by_phase[phase])
    if miss_dist:
        tot_miss = sum(miss_dist.values())
        print(f"\n  Missed PLAYED moves at n={args.default_n} by hex-dist to nearest stone (total {tot_miss}):")
        for d in sorted(miss_dist):
            print(f"      dist={d}: {miss_dist[d]:>5}  ({100.0*miss_dist[d]/tot_miss:.1f}%)")

    print(f"\n(2)/(3) SIZE + NO-EXPLOSION distribution at n={args.default_n} (median / p95):")
    for phase in ("opening", "midgame", "endgame"):
        if phase not in sizes:
            continue
        print(f"  [{phase}] positions={len(sizes[phase]['candidate'])}")
        for key in ("candidate", "window", "total_nodes", "total_edges", "edges_per_node_x100"):
            vals = sizes[phase][key]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            med = statistics.median(vals_sorted)
            p95 = vals_sorted[min(len(vals_sorted) - 1, int(0.95 * len(vals_sorted)))]
            label = key if key != "edges_per_node_x100" else "edges/node (x100)"
            print(f"      {label:>22}: median={med:>7.1f}  p95={p95:>7.1f}  max={vals_sorted[-1]:>7}")


if __name__ == "__main__":
    main()
