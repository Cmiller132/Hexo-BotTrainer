"""Root-parallel vs single-tree-large-vbatch: quality at equal latency.

Both levers cut single-game latency by fattening the NN forward (fewer passes):
  * single-tree large vbatch -> one tree, big virtual-loss window (perturbs search)
  * root parallel            -> N diverse trees, small window each, visits summed

This bench plays a game forward and, at each position, compares the move chosen
by each fast lever against a strong reference search (single tree, many sims,
small vbatch). The lever that AGREES MORE with the reference at a given latency
is the better quality/speed trade. Root parallelism uses the existing per-root
rayon selection (no new sync) and keeps each tree's window small.

Usage: python analysis/root_parallel_bench.py
"""

from __future__ import annotations

import time
from collections import defaultdict

import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference
from hexo_models.dense_cnn.mcts import new_mcts_session

SEED = 1234
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_inference() -> DenseCNNInference:
    torch.manual_seed(SEED)
    model = Model1Network(channels=64, blocks=4)
    inf = DenseCNNInference(model, device=DEVICE, amp=False, return_logits=False,
                            max_batch_size=1024, optimize_for_inference=True)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
    return inf


def run_one(sess, state, inf, *, visits, vbatch, n_roots, seed):
    """Run n_roots diverse trees on the same state; return (summed_argmax, wall)."""
    t0 = time.perf_counter()
    results = sess.run(
        list(range(n_roots)), [state] * n_roots, inf, visits=visits, c_puct=1.5,
        temperature=0.0, seed=seed, virtual_batch_size=vbatch, active_root_limit=256,
        root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
        root_policy_temperature=1.1, fpu_reduction=0.20, virtual_loss=1.0,
        widening_policy_mass=0.95, widening_max_children=32, widening_min_children=2,
    )
    wall = time.perf_counter() - t0
    summed = defaultdict(float)
    for res in results:
        for action, weight in res.visit_policy:
            summed[int(action)] += float(weight) * res.visits
    best = max(summed.items(), key=lambda kv: (kv[1], -kv[0]))[0] if summed else None
    return best, wall


def main() -> None:
    inf = build_inference()
    # Reference: strong single-tree search (1024 sims, small window).
    REF = dict(visits=1024, vbatch=4, n_roots=1)
    LEVERS = {
        "single vb4   (512s)": dict(visits=512, vbatch=4, n_roots=1),
        "single vb8   (512s)": dict(visits=512, vbatch=8, n_roots=1),
        "single vb16  (512s)": dict(visits=512, vbatch=16, n_roots=1),
        "single vb32  (512s)": dict(visits=512, vbatch=32, n_roots=1),
    }
    agree = {k: 0 for k in LEVERS}
    walls = {k: [] for k in LEVERS}
    ref_walls = []
    n_positions = 0

    state = engine.new_game(seed=4242)
    # advance the shared game by the reference's choice so positions are realistic
    fresh = lambda: new_mcts_session(max_states=262144)
    for mv in range(12):
        if engine.terminal(state) is not None:
            break
        ref_move, rw = run_one(fresh(), state, inf, seed=mv, **REF)
        if mv > 0:
            ref_walls.append(rw)
            n_positions += 1
            for name, cfg in LEVERS.items():
                mv_choice, w = run_one(fresh(), state, inf, seed=mv, **cfg)
                walls[name].append(w)
                if mv_choice == ref_move:
                    agree[name] += 1
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(ref_move)))

    print(f"device={DEVICE}  positions={n_positions}  reference={REF}  ref ms/move={sum(ref_walls)/max(len(ref_walls),1)*1e3:.1f}")
    print(f"{'lever':>22} {'ms/move':>9} {'agree-with-ref':>16}")
    for name in LEVERS:
        ms = sum(walls[name]) / max(len(walls[name]), 1) * 1e3
        print(f"{name:>22} {ms:>9.1f} {agree[name]:>7}/{n_positions} ({100*agree[name]/max(n_positions,1):>4.0f}%)")


if __name__ == "__main__":
    main()
