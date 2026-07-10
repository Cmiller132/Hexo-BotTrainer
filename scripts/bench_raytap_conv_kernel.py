#!/usr/bin/env python
"""K1 bench gate (SPEC_RAYTAP_CONV.md §2.4, work item W-K1).

Times the fused ray-tap conv+LN kernel (hexfield_eq::hex_conv_ln_raytap)
against the baseline fused conv+LN (hexfield_eq::hex_conv_ln) at serve shapes
(B*Npad ~ 24k, C = 192 by default), on REAL positions (engine games at
mid-game plies, real supports + raylen so the visibility load-skip sees the
true truncation distribution). Also checks parity of the fused kernel vs the
reference ray-tap path at a randomized (trained-stand-in) alpha — the T5
fused-path leg.

Gate (§2.4): equipped conv+LN wall-clock within ~10% of the baseline fused
conv+LN. Pre-agreed fallback if missed after reasonable tuning: (a) record the
figure and relax to <= 20%, or (b) promote conv2 (half the equipped convs) as
the candidate configuration. Either way Phase L proceeds.

RUN ONLY ON AN IDLE GPU — never against the live soak.

  python scripts/bench_raytap_conv_kernel.py [--channels 192] [--rows 24576]
      [--npad 256] [--iters 200] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np
import torch


def build_inputs(channels: int, rows: int, npad: int, seed: int = 0):
    """One padded serve-shape group from real engine positions: feats-shaped
    random fp16 activations (the conv input is trunk activations, not
    features — random is representative), real nbr/coords/mask/raylen."""

    from hexo_engine import api
    from hexo_engine.types import AxialCoord, PlacementAction
    from hexfield_eq import _rust
    from hexfield_eq.geometry import unpack_action_id
    from hexfield_eq._raytap import build_tap_reach, build_ray_gather_index

    b = max(2, rows // npad)
    rng = random.Random(seed)
    states = []
    while len(states) < b:
        st = api.new_game()
        for _ in range(rng.randint(20, 50)):
            ids = api.legal_action_ids(st)
            if not ids:
                break
            q, r = unpack_action_id(rng.choice(ids))
            res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
            if res.terminal:
                break
        if api.terminal(st) is None:
            states.append(st)
    rws = _rust.featurize_states(states)
    NBR_SENTINEL = 0xFFFF

    nbr = torch.full((b, npad, 6), npad, dtype=torch.int64)
    coords = torch.zeros(b, npad, 2, dtype=torch.int64)
    mask = torch.zeros(b, npad, dtype=torch.bool)
    raylen = torch.zeros(b, npad, 12, dtype=torch.uint8)
    for k, rw in enumerate(rws):
        n = min(int(rw["num_nodes"]), npad)
        nb = np.frombuffer(rw["nbr"], dtype=np.int32).reshape(-1, 6)[:n]
        nb64 = nb.astype(np.int64)
        nb64[(nb64 < 0) | (nb64 >= npad)] = npad
        nbr[k, :n] = torch.from_numpy(nb64)
        coords[k, :n] = torch.from_numpy(
            np.frombuffer(rw["coords"], dtype=np.int16).reshape(-1, 2)[:n].astype(np.int64)
        )
        mask[k, :n] = True
        raylen[k, :n] = torch.from_numpy(
            np.frombuffer(rw["raylen"], dtype=np.uint8).reshape(-1, 12)[:n].copy()
        )

    dev = "cuda"
    nbr, coords, mask, raylen = (
        t.to(dev) for t in (nbr, coords, mask, raylen)
    )
    self_idx = torch.arange(npad, device=dev).reshape(1, npad, 1).expand(b, -1, -1)
    gather_idx = torch.cat([self_idx, nbr], dim=2)
    torch.manual_seed(seed)
    x = torch.randn(b, npad, channels, device=dev, dtype=torch.float16)
    x = x * mask.unsqueeze(-1)
    ray_idx = build_ray_gather_index(coords, mask)
    reach = build_tap_reach(raylen)
    return x, gather_idx, mask, ray_idx, reach


def time_op(fn, iters: int, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--rows", type=int, default=24576)
    ap.add_argument("--npad", type=int, default=256)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "K1 bench needs a (IDLE) CUDA GPU"
    from hexfield_eq import constants as C
    from hexfield_eq._triton_conv import (
        hex_conv_ln,
        hex_conv_ln_raytap,
        _conv_ln_raytap_ref,
    )

    ch = args.channels
    corb = C.C_ORBIT if C.GROUP_ORDER == 12 else ch
    x, gather_idx, mask, ray_idx, reach = build_inputs(ch, args.rows, args.npad)
    b, npad, _ = x.shape

    torch.manual_seed(1)
    weight = (torch.randn(7, ch, ch, device="cuda") * 0.05).half().float()
    bias = torch.randn(ch, device="cuda") * 0.02
    lnw = torch.ones(ch, device="cuda")
    lnb = torch.zeros(ch, device="cuda")
    # Trained-alpha stand-in: decaying nonzero profile, per-channel jitter.
    alpha = torch.zeros(5, ch, device="cuda")
    for k in range(5):
        alpha[k] = 0.9 ** k * (1.0 + 0.1 * torch.randn(ch, device="cuda"))
    eps = 1e-5

    # Parity leg (T5 on the fused path): fused kernel vs the reference ray-tap
    # conv+LN at the trained-stand-in alpha.
    out_fused = hex_conv_ln_raytap(
        x, gather_idx, mask, weight, bias, lnw, lnb, ray_idx, reach, alpha,
        eps, True, corb,
    )
    out_ref = _conv_ln_raytap_ref(
        x, gather_idx, mask, weight, bias, lnw, lnb, ray_idx, reach, alpha,
        eps, True, corb,
    )
    par = float((out_fused.float() - out_ref.float()).abs().max())

    ms_base = time_op(
        lambda: hex_conv_ln(x, gather_idx, mask, weight, bias, lnw, lnb, eps, True),
        args.iters,
    )
    ms_rt = time_op(
        lambda: hex_conv_ln_raytap(
            x, gather_idx, mask, weight, bias, lnw, lnb, ray_idx, reach,
            alpha, eps, True, corb,
        ),
        args.iters,
    )
    overhead = ms_rt / ms_base - 1.0
    result = {
        "gate": "K1",
        "shape": {"B": b, "Npad": npad, "C": ch, "rows": b * npad},
        "baseline_conv_ln_ms": round(ms_base, 4),
        "raytap_conv_ln_ms": round(ms_rt, 4),
        "overhead": round(overhead, 4),
        "gate_primary": 0.10,
        "gate_fallback": 0.20,
        "parity_max_abs_vs_reference": par,
        "pass_primary": overhead <= 0.10,
        "pass_fallback": overhead <= 0.20,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0 if result["pass_primary"] else (2 if result["pass_fallback"] else 1)


if __name__ == "__main__":
    sys.exit(main())
