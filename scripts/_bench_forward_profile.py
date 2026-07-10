#!/usr/bin/env python
"""Per-component timing of the hexfield_eq serve forward (forward_policy_value).

Mirrors the production fp16 half-module serve path (HEXFIELD_SERVE_HALF: fp16
module, fp32 value tops, fp16 feats) at production microbatch size, EXCEPT
CUDA graphs stay OFF: per-module attribution needs eager block boundaries.
Absolute ms therefore reads slightly worse than the graphed production serve;
the component *shares* are the point. Source the arch env first.

  set -a; source scripts/prefit_env/hexfield_eq_raytap_a5.env; set +a
  python scripts/_bench_forward_profile.py <ckpt> [--device cpu] [--smoke]

RUN ONLY ON AN IDLE GPU — never against the live soak.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for pkg in ("hexfield_eq", "hexo_engine", "hexo_utils"):
    p = REPO / "packages" / pkg / "python"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from hexo_engine import api  # noqa: E402
from hexo_engine.types import AxialCoord, PlacementAction  # noqa: E402

from hexfield_eq import constants as C  # noqa: E402
from hexfield_eq.batching import collate_rows  # noqa: E402
from hexfield_eq.engine_facts import facts_from_state  # noqa: E402
from hexfield_eq.features import build_features, build_ray_lengths  # noqa: E402
from hexfield_eq.model import (  # noqa: E402
    HexfieldNet,
    infer_net_kwargs_from_state_dict,
)
from hexfield_eq.support import build_support  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
DEVICE = "cpu" if "--device" in sys.argv and "cpu" in sys.argv else "cuda"
SMOKE = "--smoke" in sys.argv

BATCH = 96          # production virtual_batch_size
PHASES = (10, 40, 80)   # plies replayed before featurizing
WARMUP = 2 if SMOKE else 10
ITERS = 2 if SMOKE else 40


def make_states(plies: int, n: int, seed: int):
    """n states at the given ply depth via seeded random legal replay."""
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


def main() -> int:
    device = torch.device(DEVICE)
    half = device.type == "cuda"

    if CKPT:
        payload = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = payload.get("model", payload)
        meta = payload.get("meta") or {}
        model = HexfieldNet(**infer_net_kwargs_from_state_dict(sd, meta))
        model.load_state_dict(sd, strict=True)
    else:
        model = HexfieldNet()
    model.eval()
    if half:
        # Mirror HexfieldEvaluator._serve_half: fp16 module, fp32 scalar tops.
        model = model.half()
        model.value_reduction.float()
        model.value_head.float()
    model = model.to(device)

    # ---- hooks: CUDA-event pairs per module of interest -------------------
    records: dict[str, list] = defaultdict(list)
    enabled = {"on": False}

    def hook(name: str, module):
        def pre(mod, args):
            if not enabled["on"]:
                return
            if device.type == "cuda":
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
            else:
                import time as _t
                ev = _t.perf_counter()
            records[name].append([ev, None])

        def post(mod, args, out):
            if not enabled["on"]:
                return
            if device.type == "cuda":
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
            else:
                import time as _t
                ev = _t.perf_counter()
            records[name][-1][1] = ev

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    buckets: dict[str, str] = {}   # module name -> bucket

    for i, blk in enumerate(model.conv_blocks):
        hook(f"conv_blocks.{i}", blk)
        buckets[f"conv_blocks.{i}"] = "conv_blocks"
    for i, blk in enumerate(getattr(model, "attn_blocks", []) or []):
        hook(f"attn_blocks.{i}", blk)
        buckets[f"attn_blocks.{i}"] = "attention"
    for i, reg in enumerate(getattr(model, "registers", []) or []):
        hook(f"registers.{i}", reg)
        buckets[f"registers.{i}"] = "register_refresh"
    for hname in ("policy_conv", "policy_expand", "policy_head",
                  "value_reduction", "value_head"):
        m = getattr(model, hname, None)
        if isinstance(m, torch.nn.Module):
            hook(f"heads.{hname}", m)
            buckets[f"heads.{hname}"] = "heads"

    # ---- inputs ------------------------------------------------------------
    batches = {}
    for plies in PHASES:
        b = featurize(make_states(plies, BATCH, seed=plies))
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        if half and b["feats"].dtype == torch.float32:
            b["feats"] = b["feats"].half()
        batches[plies] = b

    results = {}
    for plies, b in batches.items():
        args = (b["feats"], b["nbr"], b["mask"], b["coords"])
        raylen = b.get("raylen")
        with torch.no_grad():
            for _ in range(WARMUP):
                model.forward_policy_value(*args, raylen)
            if device.type == "cuda":
                torch.cuda.synchronize()
            records.clear()
            enabled["on"] = True
            total_evs = []
            for _ in range(ITERS):
                if device.type == "cuda":
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    model.forward_policy_value(*args, raylen)
                    e1.record()
                    total_evs.append((e0, e1))
                else:
                    import time as _t
                    t0 = _t.perf_counter()
                    model.forward_policy_value(*args, raylen)
                    total_evs.append((t0, _t.perf_counter()))
            if device.type == "cuda":
                torch.cuda.synchronize()
            enabled["on"] = False

        def ms(pair):
            if device.type == "cuda":
                return pair[0].elapsed_time(pair[1])
            return (pair[1] - pair[0]) * 1e3

        total = sum(ms(p) for p in total_evs) / ITERS
        per_mod = {k: sum(ms(p) for p in v) / ITERS for k, v in records.items()}
        per_bucket: dict[str, float] = defaultdict(float)
        for k, v in per_mod.items():
            per_bucket[buckets[k]] += v
        per_bucket["other (stem/tokens/pool/glue)"] = max(
            0.0, total - sum(per_bucket.values())
        )
        results[plies] = {
            "npad": int(b["feats"].shape[1]),
            "batch": BATCH,
            "total_ms_per_fwd": round(total, 3),
            "fwd_per_s": round(1000.0 / total, 1),
            "buckets_ms": {k: round(v, 3) for k, v in sorted(
                per_bucket.items(), key=lambda kv: -kv[1])},
            "buckets_pct": {k: round(100.0 * v / total, 1) for k, v in sorted(
                per_bucket.items(), key=lambda kv: -kv[1])},
            "slowest_modules": dict(sorted(
                ((k, round(v, 3)) for k, v in per_mod.items()),
                key=lambda kv: -kv[1])[:6]),
        }

    print(json.dumps({
        "ckpt": CKPT,
        "device": str(device),
        "half_serve": half,
        "arch": {"trunk": model._trunk_layout, "raytap": model._raytap,
                 "feature_version": C.FEATURE_VERSION, "channels": C.CHANNELS},
        "cuda_graphs": "OFF (required for per-module attribution)",
        "iters": ITERS,
        "phases": results,
    }, indent=1))

    # ---- kernel-level view (one phase, mid-game) ---------------------------
    if device.type == "cuda" and not SMOKE:
        from torch.profiler import ProfilerActivity, profile

        b = batches[PHASES[1]]
        args = (b["feats"], b["nbr"], b["mask"], b["coords"])
        with torch.no_grad(), profile(
            activities=[ProfilerActivity.CUDA], record_shapes=False
        ) as prof:
            for _ in range(5):
                model.forward_policy_value(*args, b.get("raylen"))
        print("\n== top CUDA kernels (5 fwd @ ply 40) ==")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=14))
    return 0


if __name__ == "__main__":
    sys.exit(main())
