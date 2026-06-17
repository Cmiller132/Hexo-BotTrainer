"""Diagnose the hexfield serve-forward compile across shapes.

Goal: find the REAL root cause of the compile-all failure (do not trust prior
notes) and test whether a single DYNAMIC-Npad compile can serve every shape.

Three experiments, selected by argv[1]:
  static   - compile forward_policy_value once per Npad (current shipped path);
             time each compile; flags any that stall hard (watchdog via wall time).
  dynamic  - mark Npad dynamic + automatic_dynamic ON, suppress_errors OFF, so
             the CantSplit (or whatever) RAISES instead of silently falling to
             eager. Compiles ONCE, then runs a small then a deep shape.
  parity   - eager vs compiled outputs on the same inputs (max abs diff).

Synthetic inputs are fine for compile-success + eager/compiled parity: parity is
eager==compiled on identical tensors, independent of input realism.

Run (GPU free):
  /root/.venvs/hexgt-build/bin/python scripts/_hexfield_compile_diag.py <mode>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import numpy as np
import torch

from hexfield.constants import NUM_FEATURES, NUM_TOKENS
from hexfield.model import HexfieldNet

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"
CKPT = f"{RUN}/checkpoints/epoch_000034.pt"
DEV = torch.device("cuda")


def load_model():
    m = HexfieldNet()
    p = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(p.get("model", p), strict=True)
    return m.to(DEV).eval()


def make_batch(b, npad, seed=0):
    """Realistic-ish padded batch for (b, npad): a contiguous hex-ish coord
    block, valid neighbour indices, all rows fully live (mask True). Enough to
    exercise the bias gather + sdpa; exact values irrelevant for compile/parity."""
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(b, npad, NUM_FEATURES, generator=g).to(DEV)
    # coords: spread over a plausible axial range so the bias LUT hits many rows
    coords = torch.randint(-20, 21, (b, npad, 2), generator=g).to(DEV).long()
    # neighbours: random valid node indices (0..npad-1); a few point to pad row
    nbr = torch.randint(0, npad, (b, npad, 6), generator=g).to(DEV).long()
    nbr[:, :, 0] = torch.randint(0, npad + 1, (b, npad), generator=g).to(DEV)
    mask = torch.ones(b, npad, dtype=torch.bool, device=DEV)
    return feats, nbr, mask, coords


def run_fwd(fpv, batch, ml=True):
    feats, nbr, mask, coords = batch
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        return fpv(feats, nbr, mask, coords, request_moves_left=ml)


def mode_static():
    m = load_model()
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.cache_size_limit = 256
    shapes = [256, 384, 512, 768, 1024, 1536, 2048, 2560, 3008]
    print(f"{'Npad':>6} {'compile_s':>10} {'run_ms':>8}  status", flush=True)
    for npad in shapes:
        fpv = torch.compile(m.forward_policy_value)
        batch = make_batch(4, npad)
        for t in batch[:1]:
            pass
        # mark batch dim dynamic, Npad static (current production scheme)
        for t in (batch[0], batch[1], batch[2], batch[3]):
            torch._dynamo.mark_static(t, 1)
        try:
            t0 = time.time()
            with torch.no_grad():
                run_fwd(fpv, batch)
            torch.cuda.synchronize()
            comp = time.time() - t0
            t0 = time.time()
            with torch.no_grad():
                for _ in range(5):
                    run_fwd(fpv, batch)
            torch.cuda.synchronize()
            run_ms = (time.time() - t0) / 5 * 1000
            print(f"{npad:>6} {comp:>10.1f} {run_ms:>8.2f}  ok", flush=True)
        except Exception as e:
            print(f"{npad:>6}  EXC {type(e).__name__}: {str(e)[:120]}", flush=True)
        del fpv
        torch._dynamo.reset()


def mode_dynamic():
    m = load_model()
    # The point: ONE compile, Npad a free symbol, errors RAISE (no silent eager).
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 64
    fpv = torch.compile(m.forward_policy_value, dynamic=True)
    print("compiling dynamic (production mode: incl. size-1 groups, batch CONCRETE)...", flush=True)
    # Mimic the serve path EXACTLY: mark dim0 dynamic ONLY when batch>1; size-1
    # groups (one big late-game row) leave batch concrete=1 with Npad dynamic —
    # the suspected CantSplit trigger.
    cases = [(8, 256), (1, 800), (5, 512), (1, 1600), (2, 2560)]
    for i, (bsz, npad) in enumerate(cases):
        batch = make_batch(bsz, npad, seed=i)
        for t in batch:
            if bsz > 1:
                torch._dynamo.mark_dynamic(t, 0)  # batch dynamic
            torch._dynamo.mark_dynamic(t, 1)  # Npad dynamic
        try:
            t0 = time.time()
            with torch.no_grad():
                run_fwd(fpv, batch)
            torch.cuda.synchronize()
            dt = time.time() - t0
            tag = "FIRST-COMPILE" if i == 0 else "reuse"
            print(f"  Npad={npad:>5}  {dt*1000:8.1f} ms  ({tag})", flush=True)
        except Exception as e:
            import traceback
            print(f"  Npad={npad:>5}  RAISED {type(e).__name__}", flush=True)
            tb = traceback.format_exc()
            # surface the CantSplit / divisibility line
            for line in tb.splitlines():
                if any(k in line for k in ("CantSplit", "divisible", "GuardOn", "Split", "Error")):
                    print("     " + line.strip()[:160], flush=True)
            print("     " + str(e)[:300], flush=True)
            break


def mode_parity():
    m = load_model()
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.automatic_dynamic_shapes = True
    fpv_c = torch.compile(m.forward_policy_value, dynamic=True)
    for npad in [512, 1024, 2048]:
        batch = make_batch(4, npad, seed=7)
        for t in batch:
            torch._dynamo.mark_dynamic(t, 1)
        with torch.no_grad():
            oe = run_fwd(m.forward_policy_value, batch)
            oc = run_fwd(fpv_c, batch)
        dv = (oe["value"].float() - oc["value"].float()).abs().max().item()
        dp = (oe["policy"].float() - oc["policy"].float()).abs().max().item()
        print(f"  Npad={npad:>5}  max|dvalue|={dv:.2e}  max|dpolicy|={dp:.2e}", flush=True)


def mode_compare():
    """Apples-to-apples: dynamic-compiled vs static-compiled (the SHIPPED path)
    on identical inputs. If dyn==static, dynamic is numerically the production
    path. Also report each vs eager to show the compile-fp16 noise floor."""
    m = load_model()
    torch._dynamo.config.suppress_errors = False
    for npad in [512, 1536]:
        batch = make_batch(4, npad, seed=11)
        with torch.no_grad():
            oe = run_fwd(m.forward_policy_value, batch)
        # static-compiled (Npad static, batch dynamic) — current production scheme
        torch._dynamo.reset()
        torch._dynamo.config.automatic_dynamic_shapes = False
        fpv_s = torch.compile(m.forward_policy_value)
        for t in batch:
            torch._dynamo.mark_static(t, 1)
        with torch.no_grad():
            os_ = run_fwd(fpv_s, batch)
        # dynamic-compiled (Npad free)
        torch._dynamo.reset()
        torch._dynamo.config.automatic_dynamic_shapes = True
        fpv_d = torch.compile(m.forward_policy_value, dynamic=True)
        for t in batch:
            torch._dynamo.mark_dynamic(t, 1)
        with torch.no_grad():
            od = run_fwd(fpv_d, batch)
        def md(a, b, k):
            return (a[k].float() - b[k].float()).abs().max().item()
        print(f"Npad={npad}", flush=True)
        print(f"  dyn-vs-static  value={md(od,os_,'value'):.2e} policy={md(od,os_,'policy'):.2e} "
              f"ml={md(od,os_,'moves_left'):.2e}", flush=True)
        print(f"  static-vs-eager value={md(os_,oe,'value'):.2e} policy={md(os_,oe,'policy'):.2e}", flush=True)
        print(f"  dyn-vs-eager    value={md(od,oe,'value'):.2e} policy={md(od,oe,'policy'):.2e}", flush=True)
        del fpv_s, fpv_d
        torch._dynamo.reset()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dynamic"
    print(f"torch {torch.__version__}  mode={mode}  ckpt={CKPT}", flush=True)
    {"static": mode_static, "dynamic": mode_dynamic, "parity": mode_parity,
     "compare": mode_compare}[mode]()
