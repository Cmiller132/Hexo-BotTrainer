#!/usr/bin/env python
"""Spec D acceptance perf report: inference positions/sec for the three attention
scopes (full / disk / content) on mid-game positions.

Builds the run-config ResTNet, encodes a batch of mid-game positions via the
native dense_cnn encoder, and times ``forward_policy_value`` for each scope,
reporting positions/sec and peak CUDA memory. ``content`` (gathered O(K²)) is the
one that should win at mid-game, where the relevant token count K (occupied ∪
legal ∩ disk) is far below N=1681; ``full`` is expected to OOM at batch 256 on a
12 GB card (which is itself the motivation for the scope).

Run on a FREE GPU (or coordinate so it does not contend with a live run):

    PYTHONPATH=packages/dense_cnn_restnet/python:packages/hexo_engine/python:\
packages/hexo_utils/python:packages/hexo_runner/python:packages/hexo_train/python:\
packages/hexo_models/python \
      python scripts/_restnet_attention_scope_perf.py --batch 256 --iters 20

Paste the printed table into the PR description.
"""

from __future__ import annotations

import argparse
import time

import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from dense_cnn_restnet import rust_bridge
from dense_cnn_restnet.architecture import RestnetNetwork

# The live run's architecture (configs/dense_cnn_restnet_main1.toml).
ARCH = dict(
    in_channels=13,
    channels=96,
    blocks_type="R_R_R_T_R_R_T_R",
    attention_heads=4,
    mlp_ratio=2,
    embed_kernel_size=3,
    residual_conv="hex",
    attention_impl="sdpa",
    short_term_value_horizons=(1, 4, 8),
)


def _midgame_inputs(batch: int, *, plies: int, seed0: int) -> torch.Tensor:
    """Encode `batch` mid-game positions (≈`plies` deep) into (B, 13, 41, 41)."""

    states = []
    g = seed0
    while len(states) < batch:
        state = engine.new_game(seed=g)
        g += 1
        for step in range(plies):
            if engine.terminal(state) is not None:
                break
            payload = rust_bridge.model1_batch_inputs([state])
            legal = [int(a) for a in payload["legal_action_ids"][0]]
            if not legal:
                break
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(legal[(g + step) % len(legal)])))
        if engine.terminal(state) is None:
            states.append(state)
    payload = rust_bridge.model1_batch_inputs(states)
    shape = tuple(int(s) for s in payload["shape"])
    return torch.frombuffer(bytearray(payload["inputs"]), dtype=torch.float32).reshape(shape)


def _bench_scope(model: RestnetNetwork, inputs: torch.Tensor, scope: str, *, iters: int, device: str) -> dict:
    model.set_attention_scope(scope)
    model.eval()
    result = {"scope": scope, "positions_per_second": None, "peak_mem_mb": None, "status": "ok"}
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(3):  # warmup
                model.forward_policy_value(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                model.forward_policy_value(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        result["positions_per_second"] = (iters * inputs.shape[0]) / dt
        if device == "cuda":
            result["peak_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except RuntimeError as exc:  # typically CUDA OOM for full @ large batch
        result["status"] = f"FAILED: {type(exc).__name__}: {str(exc)[:120]}"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--plies", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    print(f"device={device} batch={args.batch} iters={args.iters} plies≈{args.plies}")
    inputs = _midgame_inputs(args.batch, plies=args.plies, seed0=args.seed)
    # Relevant-token stats over the batch (content's working set).
    own = inputs[:, 0]; opp = inputs[:, 1]; legal = inputs[:, 3]
    k = ((own + opp + legal) > 0).reshape(args.batch, -1).sum(1)
    print(f"relevant tokens K per position: min={int(k.min())} mean={float(k.float().mean()):.0f} "
          f"max={int(k.max())} (N=1681)")

    inputs = inputs.to(device)
    if device == "cuda":
        inputs = inputs.to(memory_format=torch.channels_last)
    model = RestnetNetwork(**ARCH).to(device)

    rows = [_bench_scope(model, inputs, scope, iters=args.iters, device=device) for scope in ("full", "disk", "content")]
    print("\n=== attention-scope inference perf (mid-game) ===")
    print(f"{'scope':<10}{'pos/sec':>12}{'peak MB':>12}  status")
    for r in rows:
        pps = f"{r['positions_per_second']:.1f}" if r["positions_per_second"] else "-"
        mem = f"{r['peak_mem_mb']:.0f}" if r["peak_mem_mb"] else "-"
        print(f"{r['scope']:<10}{pps:>12}{mem:>12}  {r['status']}")


if __name__ == "__main__":
    main()
