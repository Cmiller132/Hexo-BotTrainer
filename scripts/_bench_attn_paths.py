#!/usr/bin/env python
"""Race the A-block attention serve paths on captured REAL inputs:

  ref   — _attn_ref (materialized fp32 scores + table2[pair] bias; parity anchor)
  flex  — _flex_call with the flex-pair score_mod (what the sprint profile
          accidentally measured: the harness env lacked HEXFIELD_FLEX_PAIR=1,
          so RelPosAttention fell through to plain flex)
  k     — hexfield_eq::attn_pair (the bespoke Triton kernel LIVE serve runs)

Also times _build_pair_u8 (the (B, S, S) uint8 pair-index build) separately —
it runs once per LIVE forward (flex-pair path only) and never appeared in the
sprint profile, and it is what PAIR_CEILING exists to bound.

Inputs are captured from a real forward at three game phases via a
RelPosAttention pre-hook run under the FULL production attention env, so the
carrier is a _FlexPairBias WITH seq_lens. Tile constants (HEXFIELD_ATTN_BM/BN/
WARPS/STAGES) are read at import in _triton_attn; sweep them by relaunching
this script under different env, e.g.:

  set -a; source scripts/prefit_env/hexfield_eq_raytap_a5.env; set +a
  export HEXFIELD_SERVE_FLEX=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_TRITON_ATTN=1
  export HEXFIELD_TRITON_CONV=1 HEXFIELD_TRITON_CONV_LN=1 HEXFIELD_TRITON_RAYTAP7=1
  for bm in 64 128; do HEXFIELD_ATTN_BM=$bm python scripts/_bench_attn_paths.py <ckpt>; done

RUN ONLY ON AN IDLE GPU.
"""

from __future__ import annotations

import json
import math
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

from hexfield_eq import _triton_attn as TA  # noqa: E402
from hexfield_eq import model as M  # noqa: E402
from hexfield_eq.batching import collate_rows  # noqa: E402
from hexfield_eq.engine_facts import facts_from_state  # noqa: E402
from hexfield_eq.features import build_features, build_ray_lengths  # noqa: E402
from hexfield_eq.model import (  # noqa: E402
    HexfieldNet,
    infer_net_kwargs_from_state_dict,
)
from hexfield_eq.support import build_support  # noqa: E402

CKPT = sys.argv[1]
# --sweep: tile-tuning mode — skip the flex lane (its per-shape torch.compile
# dominates wall time and its timing is config-independent).
SWEEP = "--sweep" in sys.argv[2:]
# LIVE-REALISTIC batch per phase: plan_groups caps a group at PAIR_CEILING /
# S_pad^2 rows (3.8e7 default -> ~255 / ~51 / ~19 at Npad 380/856/1396), so a
# fixed batch-96 bench overstates late-game GEMM efficiency. Bench BOTH the
# live group size and the flat 96 for comparability with the sprint profile.
PHASES = (3, 8, 20, 40, 80)
BATCHES = ("live", 96)
WARMUP, ITERS = 10, 50
BLOCK = 1  # middle A block (all three share shapes; tables differ trivially)


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
    if not (M._SERVE_FLEX and M._FLEX_PAIR and M._attn_pair_fused is not None):
        print(
            "env error: need HEXFIELD_SERVE_FLEX=1 HEXFIELD_FLEX_PAIR=1 "
            "HEXFIELD_TRITON_ATTN=1 (read at import)",
            file=sys.stderr,
        )
        return 2

    device = torch.device("cuda")
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = payload.get("model", payload)
    meta = payload.get("meta") or {}
    model = HexfieldNet(**infer_net_kwargs_from_state_dict(sd, meta))
    model.load_state_dict(sd, strict=True)
    model = model.eval().half().to(device)

    captured: dict[str, tuple] = {}

    def pre(mod, args, kwargs):
        # RelPosAttention.forward(seq, attn_bias)
        captured["x"] = (args[0], args[1] if len(args) > 1 else kwargs["attn_bias"])

    attn_mod = model.attn_blocks[BLOCK].attn
    handle = attn_mod.register_forward_pre_hook(pre, with_kwargs=True)

    results = {}
    with torch.no_grad():
        for plies in PHASES:
            for bs in BATCHES:
                probe = featurize(make_states(plies, 1, seed=plies))
                npad_probe = probe["feats"].shape[1]
                s_probe = npad_probe + M.NUM_TOKENS
                if bs == "live":
                    from hexfield_eq.inference import PAIR_CEILING

                    n_rows = max(4, min(96, int(PAIR_CEILING // (s_probe**2))))
                else:
                    n_rows = bs
                b = featurize(make_states(plies, n_rows, seed=plies))
                b = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in b.items()
                }
                if b["feats"].dtype == torch.float32:
                    b["feats"] = b["feats"].half()
                captured.clear()
                model.forward_policy_value(
                    b["feats"], b["nbr"], b["mask"], b["coords"], b.get("raylen")
                )
                seq, bias = captured["x"]
                assert isinstance(bias, M._FlexPairBias) and bias.seq_lens is not None, (
                    "captured bias is not a seq_lens _FlexPairBias — env mismatch "
                    "(run with HEXFIELD_TRITON_ATTN2 unset: the bench builds the "
                    "v2 inputs itself from the batch)"
                )
                bsz, s, c = seq.shape
                h, d = attn_mod.heads, attn_mod.head_dim
                q = attn_mod.q_proj(seq).reshape(bsz, s, h, d).transpose(1, 2)
                k = attn_mod.k_proj(seq).reshape(bsz, s, h, d).transpose(1, 2)
                v = attn_mod.v_proj(seq).reshape(bsz, s, h, d).transpose(1, 2)
                pair, table2, seq_lens = bias.pair, bias.table2, bias.seq_lens
                score_mod = bias.make_score_mod()
                # v2 (coords-direct) inputs, built the way trunk() builds them.
                co_i32 = b["coords"].to(torch.int32).contiguous()
                mask_u8 = b["mask"].to(torch.uint8)
                lut = model._cell_bias_lut_u8
                # Row-validity mask for parity: the bespoke kernels store zeros
                # for whole q tiles beyond seq_lens where the reference computes
                # (downstream-masked) garbage — compare live rows only.
                live_q = (
                    torch.arange(s, device=seq.device)[None, :]
                    < seq_lens[:, None]
                )[:, None, :, None]

                def run_ref():
                    return TA._attn_ref(q, k, v, pair, table2, seq_lens)

                def run_flex():
                    return M._flex_call(q, k, v, score_mod)

                def run_kernel():
                    return M._attn_pair_fused(q, k, v, pair, table2, seq_lens)

                def run_v2():
                    return TA.attn_coords(
                        q, k, v, co_i32, mask_u8, table2, lut, seq_lens
                    )

                def run_pair_build():
                    return model._build_pair_u8(b["coords"], b["mask"])

                def run_v2_build():
                    return (
                        b["coords"].to(torch.int32).contiguous(),
                        b["mask"].to(torch.uint8),
                    )

                # The materialized ref peaks ~3x (B, H, S, S) fp32; at the
                # biggest shapes that can OOM a 12 GB card — degrade to a
                # flex-vs-kernel-only row rather than dying.
                d_flex = float("nan")
                try:
                    ref = run_ref().float() * live_q
                    if not SWEEP:
                        d_flex = (
                            (run_flex().float() * live_q) - ref
                        ).abs().max().item()
                    d_k = ((run_kernel().float() * live_q) - ref).abs().max().item()
                    d_v2 = ((run_v2().float() * live_q) - ref).abs().max().item()
                    del ref
                    ref_ms = round(time_path(run_ref), 3) if not SWEEP else None
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    d_flex = d_k = d_v2 = float("nan")
                    ref_ms = None
                sdpa_gflop = 4 * bsz * h * s * s * d / 1e9
                ms = {
                    "ref": ref_ms,
                    "flex": None if SWEEP else round(time_path(run_flex), 3),
                    "kernel": round(time_path(run_kernel), 3),
                    "v2": round(time_path(run_v2), 3),
                    "pair_build": round(time_path(run_pair_build), 3),
                    "v2_build": round(time_path(run_v2_build), 3),
                }
                results[f"ply{plies}_b{n_rows}"] = {
                    "batch": bsz,
                    "s": s,
                    "parity_max_abs": {
                        "flex": round(d_flex, 5),
                        "kernel": round(d_k, 5),
                        "v2": round(d_v2, 5),
                    },
                    "ms": ms,
                    "kernel_tflops": round(sdpa_gflop / ms["kernel"], 1),
                    "v2_tflops": round(sdpa_gflop / ms["v2"], 1),
                    "flex_tflops": (
                        None if SWEEP else round(sdpa_gflop / ms["flex"], 1)
                    ),
                }

    handle.remove()
    cfg = {
        "BM": TA._BM, "BN": TA._BN, "warps": TA._WARPS, "stages": TA._STAGES,
    }
    print(json.dumps({"ckpt": CKPT, "tile": cfg, "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
