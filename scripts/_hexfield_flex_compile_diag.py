"""Diagnose the serve-flex forward compile across Npad shapes + size-1 groups.

Gate 3: prove the HEXFIELD_SERVE_FLEX=1 forward compiles (NOT eager-fallback),
reuses ONE graph across an Npad ladder + the size-1 pad_batch case, never trips
CantSplit, and never recompile-storms. Surfaces the real error if it raises.

Run (GPU free, HEXFIELD_SERVE_FLEX=1):
  /root/.venvs/hexgt-build/bin/python scripts/_hexfield_flex_compile_diag.py [mode]
    mode = dynamic (default) | innercompile
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import torch

from hexfield.constants import NUM_FEATURES
from hexfield.model import HexfieldNet

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"
CKPT = f"{RUN}/checkpoints/epoch_000034.pt"
DEV = torch.device("cuda")
MODE = sys.argv[1] if len(sys.argv) > 1 else "dynamic"


def load_model():
    m = HexfieldNet()
    p = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(p.get("model", p), strict=True)
    return m.to(DEV).eval()


def make_batch(b, npad, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(b, npad, NUM_FEATURES, generator=g).to(DEV)
    coords = torch.randint(-20, 21, (b, npad, 2), generator=g).to(DEV).long()
    nbr = torch.randint(0, npad, (b, npad, 6), generator=g).to(DEV).long()
    # leave a pad tail so the score_mod pad-key branch is exercised
    n = max(1, int(npad * 0.7))
    mask = torch.zeros(b, npad, dtype=torch.bool, device=DEV)
    mask[:, :n] = True
    return feats, nbr, mask, coords


def main():
    print(f"torch {torch.__version__}  mode={MODE}  SERVE_FLEX={os.environ.get('HEXFIELD_SERVE_FLEX')}", flush=True)
    m = load_model()
    assert m._serve_flex, "HEXFIELD_SERVE_FLEX=1 must be set so the flex path is live"
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 64
    fpv = torch.compile(m.forward_policy_value, dynamic=True)

    # (B, Npad) ladder incl. size-1 (pad_batch -> duplicate to batch 2) + big,
    # PLUS many 64-quantized buckets close together (the real serve quantum) to
    # surface any per-shape recompile storm.
    cases = [(8, 256), (8, 512), (2, 1024), (8, 1536), (8, 1920), (2, 2560),
             (1, 1600), (1, 2304), (1, 2560),
             (8, 320), (8, 384), (8, 448), (8, 576), (8, 704), (8, 832),
             (8, 1088), (8, 1216), (8, 1408), (8, 1728), (2, 2048), (2, 2432)]
    n_compiles = {"c": 0}

    for i, (bsz, npad) in enumerate(cases):
        b_eff = max(2, bsz)  # mimic the serve pad_batch dup for size-1
        batch = make_batch(b_eff, npad, seed=i)
        for t in batch:
            torch._dynamo.mark_dynamic(t, 0)
            torch._dynamo.mark_dynamic(t, 1)
        try:
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                fpv(*batch, request_moves_left=True)
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
            tag = "compile/slow" if dt > 1000 else "reuse"
            if dt > 1000:
                n_compiles["c"] += 1
            print(f"  B={b_eff:>2} Npad={npad:>5}  {dt:9.1f} ms  ({tag})", flush=True)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  B={b_eff:>2} Npad={npad:>5}  RAISED {type(e).__name__}", flush=True)
            for line in tb.splitlines():
                if any(k in line for k in ("CantSplit", "divisible", "unbacked",
                                            "InductorError", "Error", "score_mod")):
                    print("     " + line.strip()[:180], flush=True)
            break
    print(f"slow(compile-like) flushes: {n_compiles['c']}", flush=True)


if __name__ == "__main__":
    main()
