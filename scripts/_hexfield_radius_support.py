"""What-if: legal radius 4 instead of 8 — how much does the SUPPORT shrink?

The serve forward is O(support^2) (attention). The support = legal(dist<=R) | stones
| halo(dist==R+1). This measures the support-size distribution at R=8 (current) vs
R=4 on real DEEP late-game states, to predict the forward speedup (~ (N8/N4)^2 on
the attention part). NOTE: this is a game-rule change (the engine would also need
R=4 legality); the model was trained at R=8, so this is a throughput what-if only.
"""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
import numpy as np
from hexo_runner.records import HexoRecordFile
from hexfield.geometry import unpack_action_id
import hexfield.support as sup

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"


def collect_deep(n=80, min_ply=110):
    """Replay real games; at a deep ply capture the placed stones (q,r) list."""
    samples = []
    for hxr in sorted(glob.glob(f"{RUN}/selfplay/*.hxr"), key=os.path.getmtime, reverse=True):
        try:
            rf = HexoRecordFile.open(hxr)
        except Exception:
            continue
        with rf:
            for rec in rf.iter_records():
                aids = [int(a) for a in rec.action_ids]
                if len(aids) < min_ply:
                    continue
                target = len(aids) - 4  # near the end = deepest support
                stones = [unpack_action_id(a) for a in aids[:target]]
                samples.append(stones)
                if len(samples) >= n:
                    return samples
    return samples


samples = collect_deep()
print(f"deep states: {len(samples)}  (stones mean {np.mean([len(s) for s in samples]):.0f})", flush=True)

res = {}
for radius in (8, 4):
    sup.LEGAL_RADIUS = radius
    sup.HALO_DIST = radius + 1
    N, L = [], []
    for stones in samples:
        s = sup._build_support([(int(q), int(r)) for q, r in stones])
        N.append(s.num_nodes); L.append(s.legal_count)
    N = np.array(N); L = np.array(L)
    res[radius] = N
    print(f"R={radius}: support N mean {N.mean():.0f} p50 {np.median(N):.0f} p90 {np.percentile(N,90):.0f} "
          f"max {N.max()} | legal mean {L.mean():.0f}", flush=True)

r = res[8] / res[4]
print(f"\nsupport shrink R8->R4: {res[8].mean():.0f} -> {res[4].mean():.0f}  "
      f"(mean ratio {r.mean():.2f}x fewer nodes)", flush=True)
print(f"predicted attention (O(N^2)) speedup ~ {(r**2).mean():.1f}x on the attention slice", flush=True)
