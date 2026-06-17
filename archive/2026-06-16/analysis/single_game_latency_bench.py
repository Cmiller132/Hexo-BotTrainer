"""Single-game (eval/play) MCTS latency vs virtual-batch size and sims.

The eval/play path (`player.decide`) searches ONE root. Latency = passes x
(forward + select + backup). With a small virtual batch each forward carries few
leaves, so latency is dominated by many tiny, fixed-overhead forwards. This bench
measures ms/move for one game across virtual_batch_size at 128 and 512 sims, to
see how much "fewer, fatter forwards" alone cuts single-game latency (the cheap
lever) before committing to shared-tree parallelism.

Usage: python analysis/single_game_latency_bench.py
"""

from __future__ import annotations

import time

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


def one_move(session, state, inf, *, visits, vbatch, seed):
    t0 = time.perf_counter()
    res = session.run(
        [0], [state], inf, visits=visits, c_puct=1.5, temperature=0.0, seed=seed,
        virtual_batch_size=vbatch, active_root_limit=256,
        root_policy_temperature=1.1, fpu_reduction=0.20, virtual_loss=1.0,
        widening_policy_mass=0.95, widening_max_children=32, widening_min_children=2,
    )[0]
    wall = time.perf_counter() - t0
    bd = dict(res.diagnostics.get("batch", {}))
    ev = bd.get("evaluation", {})
    chunks = ev.get("evaluator_chunks", 0) or 0
    encoded = ev.get("encoded_states", 0) or 0
    return wall, res, (encoded / chunks if chunks else 0.0), chunks


def play_game(inf, *, visits, vbatch, n_moves, base_seed):
    """Play a fresh game forward, timing each NEW position (realistic eval).

    The session is reused across the game's moves (subtree reuse via advance_root,
    like real eval), but every move searches a genuinely new position so leaves
    are not just cache hits. Move 0 is discarded as warmup.
    """
    sess = new_mcts_session(max_states=262144)
    state = engine.new_game(seed=base_seed)
    walls, passes, batches = [], [], []
    for mv in range(n_moves):
        if engine.terminal(state) is not None:
            break
        w, res, avg_batch, chunks = one_move(sess, state, inf, visits=visits, vbatch=vbatch, seed=mv)
        if mv > 0:  # discard the first (cold autotune/clocks) move
            walls.append(w); passes.append(chunks); batches.append(avg_batch)
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(res.action_id)))
    n = max(len(walls), 1)
    return sum(walls) / n * 1e3, sum(passes) / n, sum(batches) / n


def main() -> None:
    inf = build_inference()
    print(f"device={DEVICE}  (single-root eval/play path, fresh position each move)")
    print(f"{'visits':>7} {'vbatch':>7} {'ms/move':>9} {'fwd_passes':>11} {'avg_fwd_batch':>14}")
    for visits in (128, 512):
        for vbatch in (4, 8, 16, 32, 64):
            ms, passes, batch = play_game(inf, visits=visits, vbatch=vbatch, n_moves=7, base_seed=777)
            print(f"{visits:>7} {vbatch:>7} {ms:>9.1f} {passes:>11.1f} {batch:>14.1f}")


if __name__ == "__main__":
    main()
