"""End-to-end hexgt MCTS smoke + REAL cache-assisted self-play throughput.

Drives the native HexgtMctsSession (transposition cache + tree reuse + nucleus
widening) against an untrained GNN evaluator over real 96x8 positions, advancing
several moves so the cache + subtree reuse are exercised. Reports:
  - correctness: selected actions are legal; search completes.
  - cache-assist: NN-leaf forwards per searched position (cache_hits, unique).
  - throughput: searched positions/sec (the true comparison to dense_cnn ~23).

Usage: python scripts/_mcts_selfplay_probe.py --shards <dir> [--games 256]
       [--visits 128] [--moves 6] [--vbatch 8] [--compile]
"""

from __future__ import annotations

import argparse
import random
import time

import torch

import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.constants import (
    DEFAULT_ATTENTION_HEADS, DEFAULT_CTX_LAYERS, DEFAULT_FFN_DIM, DEFAULT_GNN_LAYERS, DEFAULT_TOKEN_DIM,
)
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session


def _states_from_shards(n_states, shard_dir, seed=1):
    from pathlib import Path
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.expand import reconstruct_state
    rng = random.Random(seed)
    npzs = sorted(Path(shard_dir).glob("*_game_*.npz"))
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
            s = reconstruct_state(row.placement_history)
            if eng.terminal(s) is None:
                states.append(s)
            if len(states) >= n_states:
                return states
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--visits", type=int, default=128)
    ap.add_argument("--moves", type=int, default=6)
    ap.add_argument("--vbatch", type=int, default=8)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    model = HexgtNetwork(
        token_dim=DEFAULT_TOKEN_DIM, gnn_layers=DEFAULT_GNN_LAYERS, ctx_layers=DEFAULT_CTX_LAYERS,
        attention_heads=DEFAULT_ATTENTION_HEADS, ffn_dim=DEFAULT_FFN_DIM,
        short_term_value_horizons=(),
    ).to(device).eval()
    if args.compile:
        model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)
    inference = HexgtInference(model, device=device, fp16=(device.type == "cuda"))

    # --- correctness smoke (small) ------------------------------------------
    smoke_states = _states_from_shards(4, args.shards, seed=7)
    sess = new_mcts_session(max_states=1 << 20, n=args.n)
    keys = list(range(len(smoke_states)))
    results = sess.run(
        keys, smoke_states, inference, visits=16, c_puct=1.5, seed=1,
        virtual_batch_size=4, widening_policy_mass=0.95, widening_max_children=32,
        widening_min_children=2, fpu_reduction=0.20, virtual_loss=1.0,
        root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
        root_policy_temperature=1.1,
    )
    for s, res in zip(smoke_states, results):
        act = eng.PlacementAction(unpack_coord_id(res.action_id))
        assert act in list(eng.legal_actions(s)), f"selected illegal action {res.action_id}"
        assert res.visits > 0
    diag0 = dict(results[0].diagnostics).get("batch", {})
    print(f"SMOKE OK: {len(results)} games searched, all actions legal. "
          f"sample visits={results[0].visits}, root_value={results[0].root_value:.3f}")
    print(f"  tree diag: {dict(diag0.get('tree', {}))}")
    print(f"  eval diag: {dict(diag0.get('evaluation', {}))}")

    # --- throughput: advance several moves over `games` games ---------------
    pool = _states_from_shards(args.games, args.shards, seed=1)
    games = {i: s for i, s in enumerate(pool)}
    sess = new_mcts_session(max_states=1 << 21, n=args.n)
    print(f"\n=== self-play throughput: {len(games)} games x {args.moves} moves, "
          f"visits={args.visits} vbatch={args.vbatch} compile={args.compile} ===")

    total_searched = 0
    total_unique = 0
    total_hits = 0
    total_requested = 0
    enc_s = ev_s = parse_s = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    # warm one move (compile + cache fill) before timing
    warm_keys = list(games.keys())[:min(64, len(games))]
    sess.run([k for k in warm_keys], [games[k] for k in warm_keys], inference,
             visits=args.visits, c_puct=1.5, seed=99, virtual_batch_size=args.vbatch,
             widening_policy_mass=0.95, widening_max_children=32, widening_min_children=2,
             fpu_reduction=0.20, virtual_loss=1.0, root_policy_temperature=1.1)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for move in range(args.moves):
        active = [(k, s) for k, s in games.items() if eng.terminal(s) is None]
        if not active:
            break
        akeys = [k for k, _ in active]
        astates = [s for _, s in active]
        results = sess.run(
            akeys, astates, inference, visits=args.visits, c_puct=1.5, seed=1000 + move,
            virtual_batch_size=args.vbatch, widening_policy_mass=0.95,
            widening_max_children=32, widening_min_children=2, fpu_reduction=0.20,
            virtual_loss=1.0, root_dirichlet_total_alpha=10.83,
            root_dirichlet_noise_fraction=0.25, root_policy_temperature=1.1,
            temperature=1.0,
        )
        total_searched += len(active)
        d = dict(dict(results[0].diagnostics).get("batch", {}).get("evaluation", {}))
        total_unique += int(d.get("unique_states", 0))
        total_hits += int(d.get("cache_hits", 0))
        total_requested += int(d.get("requested_states", 0))
        enc_s += float(d.get("encoding_seconds", 0.0))
        ev_s += float(d.get("evaluator_seconds", 0.0))
        parse_s += float(d.get("parse_seconds", 0.0))
        for (k, s), res in zip(active, results):
            eng.apply_action(s, eng.PlacementAction(unpack_coord_id(res.action_id)))
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    pos_s = total_searched / dt
    fwd_per_pos = total_unique / max(1, total_searched)
    hit_rate = total_hits / max(1, total_requested)
    print(f"searched positions: {total_searched} in {dt:.2f}s")
    print(f"  -> {pos_s:.1f} searched positions/sec  (dense_cnn 96x8 ~23, 64x4 ~58)")
    print(f"  NN-leaf forwards/searched-pos: {fwd_per_pos:.1f}  (of {args.visits} sims; lower = more cache assist)")
    print(f"  cache hit-rate: {hit_rate:.1%}  (requested={total_requested}, unique={total_unique}, hits={total_hits})")
    print(f"  time split (sum over moves): rust_graph_encode={enc_s:.1f}s  py_evaluator={ev_s:.1f}s  parse={parse_s:.1f}s  (wall={dt:.1f}s)")
    if device.type == "cuda":
        print(f"  peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
