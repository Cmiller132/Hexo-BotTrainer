"""A1 throughput / GPU-duty bench: self-play-like search on CUDA.

Drives 256 roots at 128 visits, vbatch=4 (the production self-play shape) and
reports wall/move, searched positions/sec, and the GPU forward fraction. Run the
same script against the serial and the A1 build to compare: A1 overlaps the
Python/Torch forward with rayon selection, so wall should drop and the forward
fraction (eval_s / wall) should rise toward 1.0.

Usage: python analysis/a1_throughput_bench.py [label]
"""

from __future__ import annotations

import sys
import time

import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference
from hexo_models.dense_cnn.mcts import new_mcts_session

import os
SEED = 1234
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GAMES = int(os.environ.get("HEXO_ROOTS", "256"))
WARM_MOVES = 4
TIMED_MOVES = int(os.environ.get("HEXO_MOVES", "6"))
VISITS = int(os.environ.get("HEXO_VISITS", "128"))
VBATCH = int(os.environ.get("HEXO_VBATCH", "4"))


def build_inference() -> DenseCNNInference:
    torch.manual_seed(SEED)
    model = Model1Network(channels=64, blocks=4)
    inf = DenseCNNInference(model, device=DEVICE, amp=False, return_logits=False,
                            max_batch_size=1024, optimize_for_inference=True)
    if DEVICE == "cuda":
        # Bucketing keeps cudnn autotune cheap; leave benchmark on (A7).
        torch.backends.cudnn.benchmark = True
    return inf


def search(session, games, inf, visits):
    playable = [g for g in games if engine.terminal(g["state"]) is None]
    if not playable:
        return None, []
    t0 = time.perf_counter()
    results = session.run(
        [g["key"] for g in playable], [g["state"] for g in playable], inf,
        visits=visits, c_puct=1.5, temperature=1.0, seed=SEED, virtual_batch_size=VBATCH,
        active_root_limit=256, root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
        root_policy_temperature=1.1, fpu_reduction=0.20, virtual_loss=1.0,
        widening_policy_mass=0.95, widening_max_children=32, widening_min_children=2,
    )
    wall = time.perf_counter() - t0
    return wall, list(zip(playable, results))


def advance(gr):
    for g, res in gr:
        engine.apply_action(g["state"], engine.PlacementAction(unpack_coord_id(res.action_id)))


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else DEVICE
    inf = build_inference()
    session = new_mcts_session(max_states=262144)
    games = [{"key": i, "state": engine.new_game(seed=300 + i)} for i in range(N_GAMES)]
    for _ in range(WARM_MOVES):
        _, gr = search(session, games, inf, 32)
        if not gr:
            break
        advance(gr)

    walls, evals, encs, parses, npos = [], [], [], [], []
    for _ in range(TIMED_MOVES):
        wall, gr = search(session, games, inf, VISITS)
        if not gr:
            break
        bd = dict(gr[0][1].diagnostics.get("batch", {}))
        ev = bd.get("evaluation", {})
        walls.append(wall)
        evals.append(ev.get("evaluator_seconds", 0.0))
        encs.append(ev.get("encoding_seconds", 0.0))
        parses.append(ev.get("parse_seconds", 0.0))
        npos.append(len(gr))
        advance(gr)

    n = max(len(walls), 1)
    wall = sum(walls) / n
    eval_s = sum(evals) / n
    enc_s = sum(encs) / n
    parse_s = sum(parses) / n
    pos = sum(npos) / n
    print(f"[{label}] device={DEVICE} roots={int(pos)} visits={VISITS} vbatch={VBATCH} moves={n}")
    print(f"  wall/move        : {wall*1000:8.1f} ms")
    print(f"  positions/sec    : {pos / wall:8.1f}")
    print(f"  forward/move     : {eval_s*1000:8.1f} ms   (fraction of wall: {eval_s/wall:5.2f})")
    print(f"  encode/move      : {enc_s*1000:8.1f} ms   (fraction of wall: {enc_s/wall:5.2f})")
    print(f"  parse/move       : {parse_s*1000:8.1f} ms   (fraction of wall: {parse_s/wall:5.2f})")
    print(f"  tree+orch/move   : {(wall-eval_s-enc_s-parse_s)*1000:8.1f} ms   (fraction: {(wall-eval_s-enc_s-parse_s)/wall:5.2f})")


if __name__ == "__main__":
    main()
