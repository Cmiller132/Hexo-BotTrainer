"""V1 SOAK phase 2 (CPU): the deep-solve arm sweep.

Reads a positions JSONL (from gen_positions.py or corpus_lib) and runs the
verified-path deep-solve probe across the pre-registered arm matrix, streaming
one raw record per (position, arm) to an output JSONL. Consumes NOTHING — every
solve is measurement only. ``deep_verify_failed`` is checked after every solve;
any nonzero ABORTS the run with the raws preserved (soundness backstop).

Arms (node_cap fixed at the production leaf cap 500):
  WIN-goal, wide leaf profile:   h16-flat / h16->h32 ladder / unbounded+cap,
                                 each also with tss_zone on.
  WIN-goal, narrow profile:      h16-flat / unbounded+cap (paired superiority).
  LOSS-goal (dual seat), wide:   h16-flat / unbounded, on a 1-in-N subsample.

Usage:
    python run_soak.py <positions.jsonl> <out_raw.jsonl> [max_positions] [loss_every_n]
"""

from __future__ import annotations

import arch_env  # noqa: F401  keep import-time arch consistent across phases

import json
import sys
import time
from pathlib import Path

from hexo_engine import api
from hexfield_eq import _rust

import corpus_lib

NODE_CAP = 500

# name, goal, horizon, ladder, zone, wide, with_stats
WIN_ARMS = [
    ("h16_flat_wide", "win", 16, False, False, True, True),
    ("h16_ladder_wide", "win", 16, True, False, True, False),
    ("unbounded_wide", "win", 0, False, False, True, True),
    ("h16_flat_zone", "win", 16, False, True, True, False),
    ("h16_ladder_zone", "win", 16, True, True, True, False),
    ("unbounded_zone", "win", 0, False, True, True, False),
    ("h16_flat_narrow", "win", 16, False, False, False, True),
    ("unbounded_narrow", "win", 0, False, False, False, True),
]
LOSS_ARMS = [
    ("h16_flat_wide_loss", "loss", 16, False, False, True, False),
    ("unbounded_wide_loss", "loss", 0, False, False, True, False),
]


def lambda1_flags(state):
    d = _rust.hexfield_eq_threat_analysis(state)
    hot = bool(d["own_win_now"]) or int(d["opp_threat_count"]) > 0
    return {
        "hot": hot,
        "own_win_now": bool(d["own_win_now"]),
        "opp_threats": int(d["opp_threat_count"]),
        "min_hitting_set": int(d["min_hitting_set"]),
        "l1_verdict": d["verdict"],
    }


def run(positions, out_path, loss_every_n):
    fh = open(out_path, "w")
    n_solves = 0
    verify_failed_total = 0
    t_start = time.time()
    try:
        for idx, pos in enumerate(positions):
            state = corpus_lib.build_state(pos["moves"])
            flags = lambda1_flags(state)
            placements = int(pos.get("placements", len(pos["moves"])))
            band = placements // 10  # stone-count band of width 10
            arms = list(WIN_ARMS)
            # Dual-seat LOSS subsample: every loss_every_n-th position in the
            # (post-thinning) iteration order. Deterministic but arbitrary w.r.t.
            # position content — a cost/rate estimate, not a matched subsample.
            if loss_every_n > 0 and idx % loss_every_n == 0:
                arms = arms + LOSS_ARMS
            for (name, goal, horizon, ladder, zone, wide, with_stats) in arms:
                r = _rust.hexfield_eq_deep_solve_probe(
                    state, NODE_CAP, goal, horizon, ladder, zone, wide, with_stats
                )
                n_solves += 1
                vf = int(r["deep_verify_failed"])
                verify_failed_total += vf
                rec = {
                    "pos_id": pos.get("id", f"pos{idx}"),
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
            if (idx + 1) % 200 == 0:
                fh.flush()
                rate = n_solves / (time.time() - t_start)
                print(f"  {idx+1}/{len(positions)} positions, {n_solves} solves, "
                      f"{rate:.0f} solves/s, verify_failed={verify_failed_total}", flush=True)
    finally:
        fh.close()
    dt = time.time() - t_start
    print(f"DONE: {len(positions)} positions, {n_solves} solves in {dt:.1f}s "
          f"({n_solves/dt:.0f}/s), deep_verify_failed_total={verify_failed_total}")
    return verify_failed_total


def load_positions(spec):
    """spec is either a .jsonl path, or 'forcing'/'spare'/'human:<ngames>x<plies>'."""
    if spec == "forcing":
        return [{**p, "source": "forcing"} for p in corpus_lib.load_forcing_corpus()]
    if spec == "spare":
        return [{**p, "source": "spare"} for p in corpus_lib.load_spare_corpus()]
    if spec.startswith("human:"):
        g, _, pl = spec[len("human:"):].partition("x")
        return corpus_lib.load_human_positions(int(g), int(pl))
    out = []
    with open(spec) as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def main():
    spec = sys.argv[1]
    out_path = Path(sys.argv[2])
    max_positions = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    loss_every_n = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    positions = load_positions(spec)
    if max_positions and len(positions) > max_positions:
        # Genuine stratified thinning by stone-count band (width-10) so every
        # game phase keeps its share: allocate the cap proportionally across
        # bands, then take a seeded random sample within each band. No silent
        # truncation — the per-band keep/drop is logged.
        import random
        from collections import defaultdict

        rng = random.Random(20260720)
        by_band = defaultdict(list)
        for p in positions:
            band = int(p.get("placements", len(p["moves"]))) // 10
            by_band[band].append(p)
        total = len(positions)
        kept = []
        for band, ps in sorted(by_band.items()):
            share = max(1, round(max_positions * len(ps) / total))
            rng.shuffle(ps)
            take = min(share, len(ps))
            kept.extend(ps[:take])
            print(f"  band {band} (stones {band*10}-{band*10+9}): {len(ps)} -> {take}")
        # Trim any rounding overflow deterministically.
        rng.shuffle(kept)
        dropped = len(positions) - len(kept[:max_positions])
        positions = kept[:max_positions]
        print(f"stratified thinning to {len(positions)} positions "
              f"(dropped {dropped}, seed 20260720)")
    print(f"soaking {len(positions)} positions from {spec!r} -> {out_path}")
    vf = run(positions, out_path, loss_every_n)
    if vf:
        sys.exit(2)


if __name__ == "__main__":
    main()
