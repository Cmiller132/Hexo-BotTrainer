"""Phase-5 make-or-break gate: hexgt dynamic-GNN forward throughput on GPU FP16.

The feasibility question (HEXFORMER_REWRITE_PLAN.md Phase 5): can a no-pad dynamic
GNN run fast enough in the 512-sim self-play loop without TensorRT? The bottleneck
is the search forward (`forward_policy_value`) — especially the per-graph Python
attention loop. This probe measures NN forward throughput (positions/sec) at
realistic batch sizes and n=3 graphs, then derives an implied self-play pos/s to
compare against dense_cnn (~58 for 64x4, ~23-38 for 96x8 @512 sims).

Usage: python scripts/_profile_hexgt_forward.py [--device cuda] [--sims 512]
"""

from __future__ import annotations

import argparse
import random
import time

import torch

import hexo_engine.api as eng
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.constants import (
    DEFAULT_ATTENTION_HEADS, DEFAULT_CTX_LAYERS, DEFAULT_FFN_DIM, DEFAULT_GNN_LAYERS, DEFAULT_TOKEN_DIM,
)
from hexo_models.hexgt.graph_build import batch_from_states


def make_states_random(n_states: int, seed: int = 0):
    """Random-play pool — NOTE: scatters stones, so n-radius candidate counts are
    badly INFLATED vs real clustered play. Use --shards for a realistic measure."""
    rng = random.Random(seed)
    states = []
    tries = 0
    while len(states) < n_states and tries < n_states * 40:
        tries += 1
        s = eng.new_game(seed=rng.randint(0, 10**7))
        steps = rng.randint(4, 110)
        for _ in range(steps):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            if not acts:
                break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is None:
            states.append(s)
    return states


def make_states_from_shards(n_states: int, shard_dir: str, seed: int = 0):
    """Realistic pool: reconstruct positions from recorded 96x8 compact shards."""
    from pathlib import Path
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.expand import reconstruct_state

    rng = random.Random(seed)
    npzs = sorted(Path(shard_dir).glob("*_game_*.npz"))
    # recent completed epochs, skip the latest (possibly partial)
    epochs = sorted({p.name.split("_game_")[0] for p in npzs})
    if len(epochs) > 1:
        npzs = [p for p in npzs if p.name.split("_game_")[0] != epochs[-1]]
    rng.shuffle(npzs)
    states = []
    for npz in npzs:
        try:
            rows = read_compact_shard(npz)
        except Exception:
            continue
        for row in rows:
            states.append(reconstruct_state(row.placement_history))
            if len(states) >= n_states:
                return states
    return states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--sims", type=int, default=512)
    ap.add_argument("--batches", nargs="*", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--shards", default=None, help="reconstruct realistic positions from this 96x8 selfplay dir")
    ap.add_argument("--compile", action="store_true", help="wrap forward in torch.compile")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    fp16 = device.type == "cuda"
    print(f"device={device} fp16={fp16} n={args.n} sims={args.sims}")

    model = HexgtNetwork(
        token_dim=DEFAULT_TOKEN_DIM, gnn_layers=DEFAULT_GNN_LAYERS, ctx_layers=DEFAULT_CTX_LAYERS,
        attention_heads=DEFAULT_ATTENTION_HEADS, ffn_dim=DEFAULT_FFN_DIM,
        short_term_value_horizons=(1, 4, 8),
    ).to(device).eval()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params={nparams:,} compile={args.compile}")
    fwd = model.forward_policy_value
    if args.compile:
        fwd = torch.compile(model.forward_policy_value, dynamic=True)

    max_b = max(args.batches)
    if args.shards:
        pool = make_states_from_shards(max_b, args.shards, seed=1)
        print(f"state pool: {len(pool)} (REAL 96x8 positions)")
    else:
        pool = make_states_random(max_b, seed=1)
        print(f"state pool: {len(pool)} (random-play — candidate counts INFLATED)")

    print(f"\n{'batch':>6} | {'nodes':>8} | {'cands':>8} | {'edges':>9} | {'ms/fwd':>8} | {'pos/s':>9} | {'impl. selfplay pos/s':>20}")
    for b in args.batches:
        states = pool[:b]
        batch = batch_from_states(states, args.n)
        dev_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        nodes = int(batch["node_feat"].shape[0])
        cands = int(batch["candidate_index"].shape[0])
        edges = int(batch["edge_index"].shape[1])

        # warmup
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            for _ in range(3):
                fwd(dev_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            for _ in range(args.iters):
                fwd(dev_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters

        ms = dt * 1000.0
        pos_s = b / dt  # NN positions (leaves) per second
        # implied self-play searched-pos/s: each searched position costs `sims`
        # leaf evals (no cache); leaf-throughput / sims.
        impl = pos_s / args.sims
        print(f"{b:>6} | {nodes:>8} | {cands:>8} | {edges:>9} | {ms:>8.1f} | {pos_s:>9.0f} | {impl:>20.1f}")


if __name__ == "__main__":
    main()
