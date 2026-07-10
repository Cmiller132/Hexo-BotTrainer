#!/usr/bin/env python
"""Race the three equipped-conv serve paths on captured REAL inputs:

  ref   — _conv_ln_raytap_ref (masked-gather taps + GEMM + eager fp32 LN)
  k1    — hex_conv_ln_raytap (fused taps+GEMM+LN, whole-Cout-row programs)
  split — hex_ray_taps7 -> cuBLAS fp16 GEMM -> hex_ln_mask

Parity is asserted against ref on the SAME inputs (fp16 LN outputs, so the
tolerance is the serve class, not 1e-6). Inputs are captured from a real
forward at three game phases via a ConvBlock pre-hook, so the reach
distribution (and the invisible-row load skip) is production-realistic.

  set -a; source scripts/prefit_env/hexfield_eq_raytap_a5.env; set +a
  python scripts/_bench_raytap_paths.py <ckpt>

RUN ONLY ON AN IDLE GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for pkg in ("hexfield_eq", "hexo_engine", "hexo_utils"):
    p = REPO / "packages" / pkg / "python"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from hexo_engine import api  # noqa: E402
from hexo_engine.types import AxialCoord, PlacementAction  # noqa: E402

from hexfield_eq import _triton_conv as TC  # noqa: E402
from hexfield_eq.batching import collate_rows  # noqa: E402
from hexfield_eq.engine_facts import facts_from_state  # noqa: E402
from hexfield_eq.features import build_features, build_ray_lengths  # noqa: E402
from hexfield_eq.model import (  # noqa: E402
    HexfieldNet,
    infer_net_kwargs_from_state_dict,
)
from hexfield_eq.support import build_support  # noqa: E402

CKPT = sys.argv[1]
BATCH = 96
PHASES = (3, 8, 20, 40)
WARMUP, ITERS = 10, 50
BLOCKS = (0, 4)  # early + late trunk conv blocks


def make_states(plies, n, seed):
    import random

    out = []
    for i in range(n):
        rng = random.Random(seed * 100003 + i)
        st = api.new_game()
        for _ in range(plies):
            facts = facts_from_state(st)
            sup = build_support(facts.stones())
            legal = sup.legal_coords().tolist()
            if not legal:
                break
            q, r = legal[rng.randrange(len(legal))]
            res = api.apply_action(st, PlacementAction(AxialCoord(q=int(q), r=int(r))))
            if res.terminal:
                st = api.new_game()
        out.append(st)
    return out


def featurize(states):
    rows, raylens = [], []
    for st in states:
        facts = facts_from_state(st)
        sup = build_support(facts.stones())
        rows.append((sup, build_features(facts, sup)))
        raylens.append(build_ray_lengths(facts, sup))
    return collate_rows(rows, raylen=raylens)


def time_path(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main() -> int:
    device = torch.device("cuda")
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = payload.get("model", payload)
    meta = payload.get("meta") or {}
    model = HexfieldNet(**infer_net_kwargs_from_state_dict(sd, meta))
    model.load_state_dict(sd, strict=True)
    model = model.eval().half().to(device)

    captured: dict[int, tuple] = {}

    def grab(i):
        def pre(mod, args, kwargs):
            ctx = args[3] if len(args) > 3 else kwargs.get("ray_ctx")
            captured[i] = (args[0], args[1], args[2], ctx)
        return pre

    handles = [
        model.conv_blocks[i].register_forward_pre_hook(grab(i), with_kwargs=True)
        for i in BLOCKS
    ]

    results = {}
    with torch.no_grad():
        for plies in PHASES:
            b = featurize(make_states(plies, BATCH, seed=plies))
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            if b["feats"].dtype == torch.float32:
                b["feats"] = b["feats"].half()
            captured.clear()
            model.forward_policy_value(
                b["feats"], b["nbr"], b["mask"], b["coords"], b.get("raylen")
            )
            phase_res = {}
            for i in BLOCKS:
                x, gidx, mask, ctx = captured[i]
                blk = model.conv_blocks[i]
                conv, ln = blk.conv1, blk.ln1
                w, bias = conv._materialize()
                alpha = conv._alpha_full()
                corb = conv.alpha.shape[1]
                cin, cout = conv.in_channels, conv.out_channels
                bsz, npad = x.shape[0], x.shape[1]
                args_common = (
                    x, gidx, mask, w, bias, ln.weight, ln.bias,
                    ctx.ray_idx, ctx.reach, alpha,
                )

                def run_ref():
                    return TC._conv_ln_raytap_ref(*args_common, ln.eps, True, corb)

                def run_k1():
                    return TC.hex_conv_ln_raytap(*args_common, ln.eps, True, corb)

                def run_split():
                    t7 = TC.hex_ray_taps7(x, ctx.ray_idx, ctx.reach, alpha, corb)
                    g = t7.reshape(bsz * npad, 7 * cin) @ w.reshape(7 * cin, cout).to(
                        t7.dtype
                    )
                    return TC.hex_ln_mask(
                        g.view(bsz, npad, cout), bias, ln.weight, ln.bias,
                        mask, ln.eps, True,
                    )

                ref = run_ref()
                d_k1 = (run_k1().float() - ref.float()).abs().max().item()
                d_split = (run_split().float() - ref.float()).abs().max().item()
                phase_res[f"block{i}"] = {
                    "npad": int(npad),
                    "parity_max_abs": {"k1": round(d_k1, 5), "split": round(d_split, 5)},
                    "ms": {
                        "ref": round(time_path(run_ref), 3),
                        "k1": round(time_path(run_k1), 3),
                        "split": round(time_path(run_split), 3),
                    },
                }
            results[plies] = phase_res

    for h in handles:
        h.remove()
    print(json.dumps({"ckpt": CKPT, "batch": BATCH, "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
