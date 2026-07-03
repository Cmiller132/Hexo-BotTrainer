"""M8 end-to-end self-play throughput (the honest pos/s vs the dense floor).

Runs the real continuous scheduler with the BC model over N games capped at a
small ply count, and reports decisions/s, evals/s (from scheduler flushes),
and mean support — accounting for the REAL self-play support distribution
(clustered, smaller than random scatter) + scheduler batching + select/eval."""

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
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 16
PLY_CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 20
FLUSH_TARGET = int(sys.argv[5]) if len(sys.argv) > 5 else 256
ARL = int(sys.argv[6]) if len(sys.argv) > 6 else GAMES
VBS = int(sys.argv[7]) if len(sys.argv) > 7 else 4

device = torch.device("cuda")
model = HexfieldNet()
if CKPT:
    p = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(p.get("model", p), strict=True)
evaluator = HexfieldEvaluator(model, device=device)

states = {k: api.new_game() for k in range(GAMES)}
plies = {k: 0 for k in range(GAMES)}
support_samples = []
decisions = {"n": 0}


def on_move(game_key, payload):
    decisions["n"] += 1
    st = states[game_key]
    support_samples.append(len(api.legal_action_ids(st)))  # rough support proxy
    q, r = unpack_action_id(int(payload["action_id"]))
    res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
    plies[game_key] += 1
    if res.terminal or plies[game_key] >= PLY_CAP:
        return None
    return ("advance", st)


session = _rust.HexfieldMctsSession(max_states=262144)
torch.cuda.synchronize()
t0 = time.time()
stats = session.run_continuous(
    list(range(GAMES)), tuple(states.values()), evaluator=evaluator, on_move=on_move,
    visits=VISITS, c_puct=1.5, base_seed=7, virtual_batch_size=VBS, flush_target=FLUSH_TARGET,
    active_root_limit=ARL, temperature_by_ply=[1.0] * 8 + [0.3] * 200,
    forced_playout_k=2.0, widening_policy_mass=0.95, widening_max_children=96,
    widening_min_children=2, root_dirichlet_total_alpha=10.83,
    root_dirichlet_noise_fraction=0.25, pcr_full_proportion=0.33, pcr_fast_visits=128,
    policy_init_fraction=0.25, policy_init_avg_plies=4.0, policy_init_max_plies=8,
    policy_init_temperature=1.4,
)
torch.cuda.synchronize()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2**30

print(f"games={GAMES} visits={VISITS} ply_cap={PLY_CAP} device={device}")
print(f"decisions={decisions['n']}  seconds={dt:.1f}")
print(f"DECISIONS/S (pos/s): {decisions['n']/dt:.2f}")
print(f"flushed_states={stats['flushed_states']}  flush_count={stats['flush_count']}  "
      f"mean_flush={stats['mean_flush_states']:.1f}")
print(f"EVALS/S (unique forwards): {stats['flushed_states']/dt:.0f}")
print(f"full/fast/init moves: {stats['full_moves']}/{stats['fast_moves']}/{stats['init_moves']}")
print(f"early_stops fast/full: {stats['early_stops_fast']}/{stats['early_stops_full']}  "
      f"saved={stats['early_stop_visits_saved']}")
print(f"peak VRAM: {peak:.2f} GiB")
phase_keys = [
    "select_seconds", "submit_seconds", "finish_seconds", "backup_seconds",
    "complete_seconds", "loop_iterations", "completes_skipped",
    "no_progress_flushes",
]
if "select_seconds" in stats:
    print("phases: " + "  ".join(f"{k.replace('_seconds','')}={stats[k]:.1f}" if isinstance(stats.get(k), float) else f"{k}={stats.get(k)}" for k in phase_keys))
ev = stats.get("evaluation")
if isinstance(ev, dict):
    print("eval_stats: " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(ev.items())))
rep = evaluator.perf_trace_report()
if rep:
    print(f"perf_trace: {rep}")
print(f"DENSE FLOOR: restnet ~9.7 pos/s @ 256v; 0.8x = 7.8 pos/s (note: hexfield 512v vs 256v)")
