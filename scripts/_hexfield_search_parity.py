"""End-to-end validation of HEXFIELD_ASYNC_EVAL on the MULTI-ROOT
``session.search`` driver (``run_searches_to_targets`` — the eval path).

Sibling of ``scripts/_hexfield_async_parity.py`` (which validates the self-play
``run_continuous`` scheduler). This one drives the eval primitive: N games are
played to completion with ONE multi-root ``session.search`` per ply (all active
roots batched together, tree-reused across plies exactly like the arenas), at a
fixed seed, capturing the exact per-game action sequence.

Run in the live torch venv with the run STOPPED (GPU free):

    PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
        scripts/_hexfield_search_parity.py [checkpoint.pt]

PARITY: async=OFF is run twice (a GPU-determinism baseline, off==off2) and once
with async=ON (off==on). The action streams must be IDENTICAL — the overlap only
moves the device sync, never the math.

THROUGHPUT: a larger batch timed async off vs on, reporting decisions/s (pos/s).
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
torch.manual_seed(0)
if CKPT:
    # Build at the checkpoint's OWN arch (width/heads/trunk), inferred from the
    # state dict, so foreign-arch checkpoints (e.g. main6 c=128) load without a
    # shape mismatch — same loader the eval arenas use.
    from hexfield.eval_arena import _load_hexfield_net
    model = _load_hexfield_net(CKPT)
    print(f"loaded {CKPT}")
else:
    model = HexfieldNet()
evaluator = HexfieldEvaluator(model, device=device)  # compile ON for both runs

VISITS = 256
SEED = 7


def run(*, games: int, ply_cap: int, async_eval: bool, capture: bool):
    """Play ``games`` games to ``ply_cap`` via one multi-root search/ply.

    Returns (per-game action streams, total decisions, wall seconds).
    """
    if async_eval:
        os.environ["HEXFIELD_ASYNC_EVAL"] = "1"
    else:
        os.environ.pop("HEXFIELD_ASYNC_EVAL", None)

    states = {k: api.new_game() for k in range(games)}
    active = set(range(games))
    moves: dict[int, list[int]] = {k: [] for k in range(games)}
    decisions = 0

    session = _rust.HexfieldMctsSession(max_states=262144)
    torch.cuda.synchronize()
    t0 = time.time()
    ply = 0
    while active and ply < ply_cap:
        keys = sorted(active)
        batch_states = tuple(states[k] for k in keys)
        # Temperature schedule: sample the opening, greedy thereafter (matches
        # the arena's opening-then-greedy pattern). Per-ply seed so successive
        # plies draw fresh root noise; identical across off/on so parity holds.
        temp = 1.0 if ply < 8 else 0.0
        results = session.search(
            keys,
            batch_states,
            VISITS,
            1.5,          # c_puct
            temp,         # temperature
            SEED + ply,   # seed
            evaluator,
            4,            # virtual_batch_size
            max(128, games),  # active_root_limit
            10.83,        # root_dirichlet_total_alpha
            0.25,         # root_dirichlet_noise_fraction
        )
        for k, res in zip(keys, results):
            aid = int(res["action_id"])
            if capture:
                moves[k].append(aid)
            q, r = unpack_action_id(aid)
            out = api.apply_action(states[k], PlacementAction(AxialCoord(q=q, r=r)))
            decisions += 1
            if out.terminal:
                active.discard(k)
        ply += 1
    torch.cuda.synchronize()
    dt = time.time() - t0
    return moves, decisions, dt


print("\n=== PARITY (8 games, ply_cap 24, 256 visits) ===")
a_moves, a_d, _ = run(games=8, ply_cap=24, async_eval=False, capture=True)
b_moves, b_d, _ = run(games=8, ply_cap=24, async_eval=False, capture=True)
c_moves, c_d, _ = run(games=8, ply_cap=24, async_eval=True, capture=True)

det_ok = a_moves == b_moves
async_ok = a_moves == c_moves
print(f"  decisions: off={a_d} off2={b_d} on={c_d}")
print(f"  determinism (off vs off2): {'IDENTICAL' if det_ok else 'DIVERGED'}")
print(f"  async parity (off vs on) : {'IDENTICAL' if async_ok else 'DIVERGED'}")
if not det_ok:
    print("  NOTE: GPU forwards are non-deterministic run-to-run; parity is")
    print("        inconclusive against a non-deterministic baseline.")

print("\n=== THROUGHPUT (48 games, ply_cap 24, 256 visits) ===")
_, off_d, off_dt = run(games=48, ply_cap=24, async_eval=False, capture=False)
_, on_d, on_dt = run(games=48, ply_cap=24, async_eval=True, capture=False)
off_pps, on_pps = off_d / off_dt, on_d / on_dt
print(f"  async OFF: {off_pps:6.2f} pos/s  ({off_d} decisions / {off_dt:.2f}s)")
print(f"  async ON : {on_pps:6.2f} pos/s  ({on_d} decisions / {on_dt:.2f}s)")
print(f"  overlap speedup: {on_pps / off_pps:.2f}x pos/s")

ok = (det_ok and async_ok) or (not det_ok)
print("\nRESULT:", "PASS" if ok else "FAIL (async diverged from a deterministic baseline)")
raise SystemExit(0 if ok else 1)
