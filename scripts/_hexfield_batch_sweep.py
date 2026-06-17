"""Throughput sweep to maximize self-play serve speed by FEEDING the GPU better.

Run in the live torch venv with the run STOPPED (GPU free):

    PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
        scripts/_hexfield_batch_sweep.py <checkpoint.pt>

Sweeps active_games (with matched active_root_limit) x flush_target at PRODUCTION
512 visits. virtual_batch_size is FIXED at 4 (it is a search-quality knob — do not
change). Each config runs time-bounded (~SECONDS_PER_CONFIG) and reports
evals/s, decisions/s (pos/s), mean flush size, and peak VRAM; OOM is caught and
recorded so the VRAM ceiling is found, not hit blind. Also prints the per-decision
support-size (Npad proxy) histogram so we know whether small-S or large-S
positions dominate the wall-clock. Results -> JSON for the analysis workflow.

This is the campaign's measurement engine; it runs no training and writes no run
files (results go to the path in argv[2] or /tmp).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import numpy as np
import torch

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield import _rust
from hexfield.geometry import unpack_action_id
from hexfield.inference import HexfieldEvaluator
from hexfield.model import HexfieldNet

CKPT = sys.argv[1] if len(sys.argv) > 1 else None
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hexfield_batch_sweep.json"
SECONDS_PER_CONFIG = float(sys.argv[3]) if len(sys.argv) > 3 else 75.0
VISITS = int(sys.argv[4]) if len(sys.argv) > 4 else 512

# (active_games, active_root_limit, flush_target, cache_max_states). vbs FIXED
# at 4 (search-quality knob). NONE of these change search results — pure infra:
# active_games/active_root_limit = concurrency, flush_target = batch cadence,
# cache_max_states = transposition-cache size. flush_target must be > games*vbs.
CONFIGS = [
    (64,  64,  1024, 262144),    # ~current baseline (64 games)
    (128, 128, 1024, 262144),    # 2x games
    (192, 192, 1024, 262144),    # 3x games
    (256, 256, 2048, 262144),    # 4x games (ft must exceed 256*4)
    (128, 128, 2048, 262144),    # flush_target A/B @128
    (192, 192, 2048, 262144),    # flush_target A/B @192
    (128, 128, 1024, 1048576),   # 4x cache @128
    (192, 192, 1024, 1048576),   # 4x cache @192
]

device = torch.device("cuda")
model = HexfieldNet()
if CKPT:
    p = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(p.get("model", p), strict=True)
    print(f"loaded {CKPT}", flush=True)
evaluator = HexfieldEvaluator(model, device=device)  # compile gating + async-agnostic


def run_config(active_games, active_root_limit, flush_target, cache_max_states):
    games = active_games
    states = {k: api.new_game() for k in range(games)}
    plies = {k: 0 for k in range(games)}
    n = {"d": 0}
    support_hist: dict[int, int] = {}
    deadline = time.time() + SECONDS_PER_CONFIG

    def on_move(game_key, payload):
        n["d"] += 1
        st = states[game_key]
        # bucket the live support size (Npad proxy) into 64-quanta
        s = len(api.legal_action_ids(st))
        b = ((s // 64) + 1) * 64
        support_hist[b] = support_hist.get(b, 0) + 1
        q, r = unpack_action_id(int(payload["action_id"]))
        res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
        plies[game_key] += 1
        # wind the game down after the time budget so run_continuous returns
        if res.terminal or time.time() > deadline:
            return None
        return ("advance", st)

    session = _rust.HexfieldMctsSession(max_states=cache_max_states)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    stats = session.run_continuous(
        list(range(games)), tuple(states.values()), evaluator=evaluator, on_move=on_move,
        visits=VISITS, c_puct=1.5, base_seed=7, virtual_batch_size=4,
        flush_target=flush_target, active_root_limit=active_root_limit,
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
    return {
        "decisions": n["d"], "seconds": round(dt, 1),
        "pos_per_s": round(n["d"] / dt, 2),
        "evals_per_s": round(stats["flushed_states"] / dt, 0),
        "mean_flush": round(stats["mean_flush_states"], 1),
        "flushed_states": stats["flushed_states"],
        "peak_vram_gib": round(peak, 2),
        "support_hist": dict(sorted(support_hist.items())),
    }


results = []
for ag, arl, ft, cache in CONFIGS:
    label = f"active_games={ag} arl={arl} flush_target={ft} cache={cache}"
    print(f"\n=== {label} ===", flush=True)
    try:
        r = run_config(ag, arl, ft, cache)
        r.update(active_games=ag, active_root_limit=arl, flush_target=ft,
                 cache_max_states=cache, oom=False)
        print(f"  pos/s={r['pos_per_s']}  evals/s={r['evals_per_s']:.0f}  "
              f"mean_flush={r['mean_flush']}  peak_vram={r['peak_vram_gib']}GiB", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        r = {"active_games": ag, "active_root_limit": arl, "flush_target": ft, "cache_max_states": cache,
             "oom": True, "error": str(e)[:200]}
        print(f"  OOM at {label}", flush=True)
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001  -- record + continue the sweep
        r = {"active_games": ag, "active_root_limit": arl, "flush_target": ft, "cache_max_states": cache,
             "oom": False, "error": repr(e)[:300]}
        print(f"  ERROR at {label}: {r['error']}", flush=True)
    results.append(r)

Path(OUT).write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nwrote {OUT}", flush=True)
ok = [r for r in results if not r.get("oom") and "error" not in r]
if ok:
    best = max(ok, key=lambda r: r["evals_per_s"])
    print(f"BEST by evals/s: active_games={best['active_games']} "
          f"arl={best['active_root_limit']} flush_target={best['flush_target']} "
          f"-> {best['evals_per_s']:.0f} evals/s, {best['pos_per_s']} pos/s, "
          f"{best['peak_vram_gib']}GiB", flush=True)
