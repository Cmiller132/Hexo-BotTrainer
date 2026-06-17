"""Profile the hexfield model forward at a representative (B, N) to find the
hot ops and which SDPA backend fires (flash vs mem-efficient vs math). The
relpos additive bias may be blocking FlashAttention -> slow backend. Parity-
safe speedups target whatever dominates here."""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
import torch
import hexfield.constants as C
from hexfield.model import HexfieldNet

B = int(sys.argv[1]) if len(sys.argv) > 1 else 64
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
dev = torch.device("cuda")
m = HexfieldNet().eval().to(dev)
feats = torch.randn(B, N, C.NUM_FEATURES, device=dev)
nbr = torch.randint(0, N, (B, N, 6), dtype=torch.long, device=dev)
mask = torch.ones(B, N, dtype=torch.bool, device=dev)
coords = torch.randint(-20, 21, (B, N, 2), dtype=torch.long, device=dev)
args = (feats, nbr, mask, coords)

def run():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return m(*args)

run(); torch.cuda.synchronize()
# time
t0 = time.time()
for _ in range(10): run()
torch.cuda.synchronize()
print(f"B={B} N={N}: {(time.time()-t0)/10*1000:.1f} ms/fwd  -> {B/((time.time()-t0)/10):.0f} states/s (10-rep avg already passed)")

# which SDPA backends are even available / used: profile
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        run()
    torch.cuda.synchronize()
print("\n=== TOP CUDA OPS (self cuda time) ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18))
# detect SDPA backend by kernel name
names = " ".join(e.key.lower() for e in prof.key_averages())
print("\n=== SDPA backend hints (kernel names present) ===")
for tag in ["flash", "efficient_attention", "fmha", "cutlassf", "mem_eff", "scaled_dot_product", "bmm", "gemm"]:
    print(f"  {tag}: {'YES' if tag in names else '-'}")
