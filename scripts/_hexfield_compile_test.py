"""Test whether torch.compile speeds the inference forward (fuses the bias-index
elementwise ops + reduces kernel-launch overhead) and how it handles the varying
(B, N) shapes of self-play. Compares eager vs compiled at two shapes."""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
import torch
import hexfield.constants as C
from hexfield.model import HexfieldNet

dev = torch.device("cuda")
m = HexfieldNet().eval().to(dev)

def mk(B, N):
    return (torch.randn(B, N, C.NUM_FEATURES, device=dev),
            torch.randint(0, N, (B, N, 6), dtype=torch.long, device=dev),
            torch.ones(B, N, dtype=torch.bool, device=dev),
            torch.randint(-20, 21, (B, N, 2), dtype=torch.long, device=dev))

def bench(fn, args, reps=10):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        fn(*args)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(reps): fn(*args)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000

a1 = mk(64, 600); a2 = mk(48, 360)
print(f"EAGER  B64 N600: {bench(m, a1):.1f} ms")
print(f"EAGER  B48 N360: {bench(m, a2):.1f} ms")

cm = torch.compile(m, dynamic=True)
print("compiling (warmup)...")
t0 = time.time()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    cm(*a1)
torch.cuda.synchronize()
print(f"  first-call (compile) B64 N600: {(time.time()-t0)*1000:.0f} ms")
t0 = time.time()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    cm(*a2)
torch.cuda.synchronize()
print(f"  first-call B48 N360 (recompile?): {(time.time()-t0)*1000:.0f} ms")
print(f"COMPILED B64 N600: {bench(cm, a1):.1f} ms")
print(f"COMPILED B48 N360: {bench(cm, a2):.1f} ms")

# parity: compiled vs eager output
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    oe = m(*a1); oc = cm(*a1)
for k in oe:
    d = (oe[k].float() - oc[k].float()).abs().max().item()
    print(f"  parity {k}: max abs diff {d:.2e}")
