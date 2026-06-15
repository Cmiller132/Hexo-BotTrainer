"""Late-game-inclusive throughput bench for the serve path.

Unlike the 60-75s early-game sweep, this runs a LONG window so the concurrent
games progress into the large-support late game that dominates a real epoch, and
prints the support-size histogram so we can SEE how late it got. Sweeps
active_games at fixed flush_target/cache to isolate the concurrency optimum and
expose the 64->192 regression. Reports pos/s, evals/s, mean flush size, peak
VRAM, and the full scheduler stats dict (flush counts, early stops, cache_len).

Run (GPU free):
    PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
        scripts/_hexfield_lategame_bench.py <ckpt.pt> <seconds> "64,96,128,192"
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import torch
from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield import _rust
from hexfield.geometry import unpack_action_id
from hexfield.inference import HexfieldEvaluator
from hexfield.model import HexfieldNet

CKPT = sys.argv[1]
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
AGS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["64", "128", "192"])]
FLUSH_TARGET = 1024
CACHE = 262144
VISITS = 512

device = torch.device("cuda")
model = HexfieldNet()
p = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(p.get("model", p), strict=True)
evaluator = HexfieldEvaluator(model, device=device)
print(f"loaded {CKPT}; window={SECONDS}s; active_games={AGS}", flush=True)


def run_config(active_games, refill):
    """refill=True  -> mirror the trainer with games_per_epoch >> active_games:
    every finished slot is REPLACED by a fresh game (steady-state early/mid/late
    MIX at constant concurrency=active_games). refill=False -> the cohort regime
    where active_games == games_per_epoch: finished slots are NOT replaced, so all
    games drain into deep late game together (the live ep32 regime)."""
    games = active_games
    states = {k: api.new_game() for k in range(games)}
    n = {"d": 0, "next_key": games, "live": games}
    support_hist: dict[int, int] = {}
    deadline = time.time() + SECONDS
    plies = {k: 0 for k in range(games)}

    def on_move(game_key, payload):
        n["d"] += 1
        st = states[game_key]
        s = len(api.legal_action_ids(st))
        bkt = ((s // 256) + 1) * 256
        support_hist[bkt] = support_hist.get(bkt, 0) + 1
        q, r = unpack_action_id(int(payload["action_id"]))
        res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
        plies[game_key] += 1
        ended = res.terminal or plies[game_key] >= 256
        if not ended:
            if time.time() > deadline:
                return None
            return ("advance", st)
        # game finished: refill with a fresh game (mirrors driver "replace") or stop
        if refill and time.time() <= deadline:
            nk = n["next_key"]; n["next_key"] += 1
            states[nk] = api.new_game(); plies[nk] = 0
            return ("replace", nk, states[nk])
        n["live"] -= 1
        return None

    session = _rust.HexfieldMctsSession(max_states=CACHE)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    stats = session.run_continuous(
        list(range(games)), tuple(states.values()), evaluator=evaluator, on_move=on_move,
        visits=VISITS, c_puct=1.5, base_seed=7, virtual_batch_size=4,
        flush_target=FLUSH_TARGET, active_root_limit=active_games,
        temperature_by_ply=[1.0] * 8 + [0.3] * 400,
        forced_playout_k=2.0, widening_policy_mass=0.95, widening_max_children=96,
        widening_min_children=2, root_dirichlet_total_alpha=10.83,
        root_dirichlet_noise_fraction=0.25, pcr_full_proportion=0.33, pcr_fast_visits=128,
        policy_init_fraction=0.25, policy_init_avg_plies=4.0, policy_init_max_plies=8,
        policy_init_temperature=1.4,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    return dt, n["d"], stats, peak, support_hist


REFILL = (sys.argv[5] != "cohort") if len(sys.argv) > 5 else True
results = []
for ag in AGS:
    print(f"\n=== active_games={ag} ft={FLUSH_TARGET} cache={CACHE} refill={REFILL} ===", flush=True)
    dt, decs, stats, peak, hist = run_config(ag, REFILL)
    flushed = stats.get("flushed_states", 0)
    rec = {
        "active_games": ag, "seconds": round(dt, 1), "decisions": decs,
        "pos_per_s": round(decs / dt, 2), "evals_per_s": round(flushed / dt, 0),
        "mean_flush": round(stats.get("mean_flush_states", 0), 1),
        "flush_count": stats.get("flush_count"),
        "no_progress_flushes": stats.get("no_progress_flushes"),
        "peak_vram_gib": round(peak, 2),
        "support_hist_256": dict(sorted(hist.items())),
    }
    results.append(rec)
    print(f"  pos/s={rec['pos_per_s']}  evals/s={rec['evals_per_s']:.0f}  "
          f"mean_flush={rec['mean_flush']}  peak_vram={rec['peak_vram_gib']}GiB", flush=True)
    print(f"  scheduler={json.dumps({k: stats.get(k) for k in stats}, default=str)[:400]}", flush=True)
    print(f"  support_hist(256-buckets)={rec['support_hist_256']}", flush=True)

out = sys.argv[4] if len(sys.argv) > 4 else "/tmp/hexfield_lategame_bench.json"
Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nwrote {out}")
best = max(results, key=lambda r: r["pos_per_s"])
print(f"BEST pos/s: active_games={best['active_games']} -> {best['pos_per_s']} pos/s")
