"""Shared harness for the dense_cnn 96x6+P7 inference-backend benchmarks.

Everything the variant scripts need to be RIGOROUS and comparable:

- Build the EXACT production network from the target_96x6 TOML and load
  ``model_state`` strict from the bootstrap prefit checkpoint.
- Build the production *inference* model the same way the evaluator does
  (``optimized_model1_for_inference`` + channels_last + cuDNN autotune).
- Generate REPRESENTATIVE inputs by driving real engine games and encoding
  them through the same Rust path the evaluator uses (cached to .npy).
- A timing harness that fully warms up (GPU clock ramp + cuDNN autotune) before
  measuring, then reports mean / std / p50 / p95 over many CUDA-event-timed
  iterations -- never a single number.
- A correctness gate: max-abs error of policy logits AND value vs the FP32
  reference, on representative inputs.

Definitions used throughout (stated so the report is unambiguous):
  * "forward"      = one network evaluation of one position (one MCTS leaf).
  * "forwards/s"   = positions evaluated per second = batch_rows / batch_latency.
                     This is the user's "~1287 forwards/s" metric.
  * batch latency  = wall time for one forward_policy_value call on a batch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# hexo_models is a PEP-420 namespace whose INSTALLED copy is stale (see NOTES.md
# "Environment gotcha"). Point at the worktree package python/ dirs so we test
# worktree code, never a stale wheel.
_PKG_PY = [
    REPO / "packages" / p / "python"
    for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend")
]
for _p in _PKG_PY:
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import numpy as np
import torch

CONFIG_PATH = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT_PATH = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"
INPUT_CACHE = Path(__file__).resolve().parent / "_bench_inputs.npy"

INPUT_CHANNELS = 13
BOARD = 41


# --------------------------------------------------------------------------- #
# Model + weights
# --------------------------------------------------------------------------- #
def _load_model_config() -> dict:
    import tomllib

    with open(CONFIG_PATH, "rb") as fh:
        raw = tomllib.load(fh)
    return raw["model"]["config"]


def build_training_model() -> torch.nn.Module:
    """Build Model1Network exactly as DenseCNNPlugin.build_model does, weights loaded."""
    from hexo_models.dense_cnn.architecture import Model1Network
    from hexo_models.dense_cnn.config import parse_model1_config

    parsed = parse_model1_config(_load_model_config())
    arch = parsed.architecture
    model = Model1Network(
        in_channels=arch.input_channels,
        channels=arch.channels,
        blocks=arch.residual_blocks,
        dropout=arch.dropout,
        short_term_value_horizons=arch.short_term_value_horizons,
    )
    payload = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    model.load_state_dict(state)  # strict by default
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] built channels={arch.channels} blocks={arch.residual_blocks} "
          f"horizons={arch.short_term_value_horizons} params={n_params/1e6:.3f}M "
          f"epoch={payload.get('epoch')}", flush=True)
    return model


def build_inference_model(device: str = "cuda", channels_last: bool = True) -> torch.nn.Module:
    """The production inference clone: conv/BN folded, eval, on device, channels_last.

    Mirrors DenseCNNInference.__init__ (optimize_for_inference=True path).
    """
    from hexo_models.dense_cnn.architecture import optimized_model1_for_inference

    model = build_training_model()
    if device == "cuda":
        opt = optimized_model1_for_inference(model)  # returns CPU eval clone
        opt = opt.to(device=device)
        if channels_last:
            opt = opt.to(memory_format=torch.channels_last)
        opt.eval()
        return opt
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Representative inputs (real engine states encoded via the production Rust path)
# --------------------------------------------------------------------------- #
def generate_inputs(n: int = 2048, seed_base: int = 70_000) -> np.ndarray:
    """Drive random-legal games to varied depths, encode via rust_bridge.

    Returns float32 array (n, 13, 41, 41) -- the exact tensor the evaluator
    feeds the model (same Rust encoder). Cached to disk.
    """
    if INPUT_CACHE.exists():
        arr = np.load(INPUT_CACHE)
        if arr.shape[0] >= n and arr.shape[1:] == (INPUT_CHANNELS, BOARD, BOARD):
            print(f"[inputs] loaded cache {arr.shape} from {INPUT_CACHE.name}", flush=True)
            return arr[:n]

    import random

    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id

    from hexo_models.dense_cnn import rust_bridge

    rng = random.Random(1234)
    collected: list[np.ndarray] = []
    collected_rows = 0
    snapshot_states: list[object] = []
    game_idx = 0
    # Spread positions across many games and many depths so the pool has a
    # realistic mix of board occupancies (early/mid/late), not one distribution.
    while collected_rows + len(snapshot_states) < n:
        state = engine.new_game(seed=seed_base + game_idx)
        game_idx += 1
        depth_cap = rng.randint(2, 120)
        for _ in range(depth_cap):
            if engine.terminal(state) is not None:
                break
            aids = engine.legal_action_ids(state)
            if not aids:
                break
            aid = rng.choice(aids)
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(aid)))
        if engine.terminal(state) is not None:
            continue
        snapshot_states.append(engine.clone_state(state))
        if len(snapshot_states) >= 256:
            payload = rust_bridge.model1_batch_inputs(snapshot_states)
            shape = tuple(int(x) for x in payload["shape"])
            buf = np.frombuffer(bytes(payload["inputs"]), dtype=np.float32).reshape(shape).copy()
            collected.append(buf)
            collected_rows += buf.shape[0]
            snapshot_states = []
    if snapshot_states:
        payload = rust_bridge.model1_batch_inputs(snapshot_states)
        shape = tuple(int(x) for x in payload["shape"])
        buf = np.frombuffer(bytes(payload["inputs"]), dtype=np.float32).reshape(shape).copy()
        collected.append(buf)
    arr = np.concatenate(collected, axis=0)[:n]
    np.save(INPUT_CACHE, arr)
    nz = float((arr != 0).mean())
    print(f"[inputs] generated {arr.shape} nonzero_frac={nz:.3f} -> cached {INPUT_CACHE.name}", flush=True)
    return arr


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def warmup(fn, *, seconds: float = 6.0, min_iters: int = 30):
    """Run fn until BOTH (a) a generous wall-time budget elapses (GPU clock ramps
    from idle ~210MHz to boost) AND (b) min_iters completed (cuDNN autotune
    converges on the shapes fn uses). This is the fix for the fake "11 pos/s".
    """
    import time

    cuda_sync()
    t0 = time.perf_counter()
    it = 0
    while True:
        fn()
        it += 1
        if it >= min_iters and (time.perf_counter() - t0) >= seconds:
            break
    cuda_sync()


def time_iters(fn, *, iters: int = 200) -> dict:
    """Time fn for `iters` iterations using CUDA events; return latency stats (ms)."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    cuda_sync()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    cuda_sync()
    lat = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)], dtype=np.float64)  # ms
    lat.sort()
    return {
        "mean_ms": float(lat.mean()),
        "std_ms": float(lat.std()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "min_ms": float(lat.min()),
        "iters": iters,
    }


def gpu_clock_mhz() -> int:
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.current.graphics", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def fwd_per_s(latency_ms: float, batch: int) -> float:
    return batch / (latency_ms / 1000.0)


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def correctness_report(policy_ref, value_ref, policy_test, value_test) -> dict:
    pol_err = max_abs_err(policy_ref, policy_test)
    val_err = max_abs_err(value_ref, value_test)
    # Also report on the quantities that actually matter downstream:
    #  - policy as softmax-over-legal would be; here report softmax(full) MAE.
    #  - decoded scalar value (after decode_binned_value) abs error.
    from hexo_models.dense_cnn.losses import decode_binned_value

    v_ref = decode_binned_value(value_ref.float())
    v_test = decode_binned_value(value_test.float())
    decoded_val_err = float((v_ref - v_test).abs().max().item())
    p_ref = torch.softmax(policy_ref.float(), dim=-1)
    p_test = torch.softmax(policy_test.float(), dim=-1)
    softmax_mae = float((p_ref - p_test).abs().mean().item())
    softmax_max = float((p_ref - p_test).abs().max().item())
    return {
        "policy_logit_max_abs_err": pol_err,
        "value_logit_max_abs_err": val_err,
        "decoded_value_max_abs_err": decoded_val_err,
        "policy_softmax_mae": softmax_mae,
        "policy_softmax_max_abs_err": softmax_max,
    }
