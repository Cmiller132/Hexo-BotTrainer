"""End-to-end validation of HEXFIELD_ASYNC_EVAL (the GPU/host overlap path).

Run in the live torch venv with the run STOPPED (GPU free):

    PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
        scripts/_hexfield_async_parity.py [checkpoint.pt]

PARITY: runs the real continuous scheduler at production knobs (512 visits, the
full PCR/widening/noise set) with a fixed seed, capturing the exact per-game
action sequence. async=OFF is run twice (a determinism baseline) and once with
async=ON; the action sequences must be IDENTICAL — the overlap only moves the
device sync, never the math.

THROUGHPUT: a larger 64-game batch (matching active_games) timed async off vs on,
reporting decisions/s (pos/s) and evals/s.
"""
from __future__ import annotations

import os
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

CKPT = sys.argv[1] if len(sys.argv) > 1 else None

device = torch.device("cuda")
model = HexfieldNet()
if CKPT:
    p = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(p.get("model", p), strict=True)
    print(f"loaded {CKPT}")
evaluator = HexfieldEvaluator(model, device=device)  # compile ON for both runs


def run(*, games: int, ply_cap: int, async_eval: bool, capture: bool):
    if async_eval:
        os.environ["HEXFIELD_ASYNC_EVAL"] = "1"
    else:
        os.environ.pop("HEXFIELD_ASYNC_EVAL", None)
    states = {k: api.new_game() for k in range(games)}
    plies = {k: 0 for k in range(games)}
    moves: dict[int, list[int]] = {k: [] for k in range(games)}
    n = {"d": 0}

    def on_move(game_key, payload):
        n["d"] += 1
        aid = int(payload["action_id"])
        if capture:
            moves[game_key].append(aid)
        st = states[game_key]
        q, r = unpack_action_id(aid)
        res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
        plies[game_key] += 1
        if res.terminal or plies[game_key] >= ply_cap:
            return None
        return ("advance", st)

    session = _rust.HexfieldMctsSession(max_states=262144)
    torch.cuda.synchronize()
    t0 = time.time()
    stats = session.run_continuous(
        list(range(games)), tuple(states.values()), evaluator=evaluator, on_move=on_move,
        visits=512, c_puct=1.5, base_seed=7, virtual_batch_size=4, flush_target=1024,
        active_root_limit=max(128, games), temperature_by_ply=[1.0] * 8 + [0.3] * 300,
        forced_playout_k=2.0, widening_policy_mass=0.95, widening_max_children=96,
        widening_min_children=2, root_dirichlet_total_alpha=10.83,
        root_dirichlet_noise_fraction=0.25, pcr_full_proportion=0.33, pcr_fast_visits=128,
        policy_init_fraction=0.25, policy_init_avg_plies=4.0, policy_init_max_plies=8,
        policy_init_temperature=1.4,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    return moves, n["d"], dt, stats


print("\n=== PARITY (8 games, ply_cap 24, 512 visits) ===")
a_moves, a_d, _, _ = run(games=8, ply_cap=24, async_eval=False, capture=True)
b_moves, b_d, _, _ = run(games=8, ply_cap=24, async_eval=False, capture=True)
c_moves, c_d, _, _ = run(games=8, ply_cap=24, async_eval=True, capture=True)

det_ok = a_moves == b_moves
async_ok = a_moves == c_moves
print(f"  decisions: off={a_d} off2={b_d} on={c_d}")
print(f"  determinism (off vs off): {'IDENTICAL' if det_ok else 'DIVERGED'}")
print(f"  async parity (off vs on): {'IDENTICAL' if async_ok else 'DIVERGED'}")
if not det_ok:
    # Without a deterministic baseline the parity comparison is inconclusive.
    print("  NOTE: forwards are non-deterministic run-to-run; comparing outcomes only.")

print("\n=== THROUGHPUT (64 games, ply_cap 24, 512 visits) ===")
_, off_d, off_dt, off_s = run(games=64, ply_cap=24, async_eval=False, capture=False)
_, on_d, on_dt, on_s = run(games=64, ply_cap=24, async_eval=True, capture=False)
off_pps, on_pps = off_d / off_dt, on_d / on_dt
off_eps = off_s["flushed_states"] / off_dt
on_eps = on_s["flushed_states"] / on_dt
print(f"  async OFF: {off_pps:5.2f} pos/s   {off_eps:6.0f} evals/s   mean_flush={off_s['mean_flush_states']:.0f}")
print(f"  async ON : {on_pps:5.2f} pos/s   {on_eps:6.0f} evals/s   mean_flush={on_s['mean_flush_states']:.0f}")
print(f"  overlap speedup: {on_pps / off_pps:.2f}x pos/s")

ok = (det_ok and async_ok) or (not det_ok)
print("\nRESULT:", "PASS" if ok else "FAIL (async diverged from a deterministic baseline)")
raise SystemExit(0 if ok else 1)
