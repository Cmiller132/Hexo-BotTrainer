"""Aligned MCTS search-quality probe: serial vs A1 on identical positions.

The plain trajectory harness diverges after the first near-tied move (different
argmax -> different game -> incomparable). This one advances each game by the
argmax of the *root prior* (NN-derived, deterministic on CPU, identical across
builds) instead of the search result, so both builds traverse exactly the same
positions. For each position it records the full visit-policy distribution.

Diff two runs with `mcts_aligned_diff.py` to get, per position:
  * total-variation distance between the visit-policy distributions
  * whether the visit argmax (the move self-play would pick at temp 0) agrees

A1 is pipelined virtual-loss MCTS, so it is not bit-identical; the test of
quality-neutrality is that these distributions stay close and the argmax mostly
agrees on the same positions.

Usage: python analysis/mcts_aligned_harness.py out.json
"""

from __future__ import annotations

import json
import sys

import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference
from hexo_models.dense_cnn.mcts import new_mcts_session

SEED = 1234
N_GAMES = 12
N_MOVES = 10
VISITS = 128
import os
VBATCH = int(os.environ.get("HEXO_VBATCH", "4"))


def build_inference() -> DenseCNNInference:
    torch.manual_seed(SEED)
    model = Model1Network(channels=64, blocks=4)
    return DenseCNNInference(
        model, device="cpu", amp=False, return_logits=False,
        max_batch_size=1024, optimize_for_inference=False,
    )


def run() -> dict:
    inf = build_inference()
    session = new_mcts_session(max_states=262144)
    games = [{"key": i, "state": engine.new_game(seed=200 + i)} for i in range(N_GAMES)]

    positions = []
    for move_idx in range(N_MOVES):
        playable = [g for g in games if engine.terminal(g["state"]) is None]
        if not playable:
            break
        results = session.run(
            [g["key"] for g in playable],
            [g["state"] for g in playable],
            inf,
            visits=VISITS, c_puct=1.5, temperature=1.0, seed=SEED,
            virtual_batch_size=VBATCH, active_root_limit=256,
            root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
            root_policy_temperature=1.1, fpu_reduction=0.20, virtual_loss=1.0,
            widening_policy_mass=0.95, widening_max_children=32, widening_min_children=2,
        )
        for g, res in zip(playable, results):
            vp = {int(a): round(float(w), 6) for a, w in res.visit_policy}
            positions.append({"move": move_idx, "game": g["key"], "visit_policy": vp})
            # Advance by the ROOT-PRIOR argmax (search-independent -> positions
            # stay aligned across builds), NOT by the visit argmax.
            prior = res.root_prior_policy
            advance_action = max(prior, key=lambda kv: kv[1])[0]
            engine.apply_action(g["state"], engine.PlacementAction(unpack_coord_id(int(advance_action))))

    return {"config": {"seed": SEED, "games": N_GAMES, "moves": N_MOVES, "visits": VISITS,
                       "vbatch": VBATCH}, "n_positions": len(positions), "positions": positions}


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "aligned.json"
    result = run()
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"wrote {out}: {result['n_positions']} positions")


if __name__ == "__main__":
    main()
