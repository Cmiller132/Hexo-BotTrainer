"""M8 cost curve: evals/s vs support size N, batched at the pair ceiling, plus
a one-forward profile of where the time goes (attention vs conv vs bias gather).
Tests the §5.5 projections at the REAL self-play mid-game median (N~900), not
the random-scatter N~2000 mix."""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import numpy as np
import torch

from hexfield import _rust
from hexfield.model import HexfieldNet
from hexfield.inference import NBR_SENTINEL
from hexfield.constants import NUM_FEATURES, NUM_TOKENS
from _hexfield_perf import midgame_states


def states_near(target_n, tol, want, pool):
    out = []
    for s in pool:
        n = _rust.featurize_states([s])[0]["num_nodes"]
        if abs(n - target_n) <= tol:
            out.append((s, n))
        if len(out) >= want:
            break
    return out


def make_batch(rows_states, device):
    rows = _rust.featurize_states([s for s, _ in rows_states])
    pad = max(r["num_nodes"] for r in rows)
    pad = -(-pad // 256) * 256
    g = len(rows)
    feats = torch.zeros(g, pad, NUM_FEATURES)
    nbr = torch.full((g, pad, 6), pad, dtype=torch.long)
    mask = torch.zeros(g, pad, dtype=torch.bool)
    coords = torch.zeros(g, pad, 2, dtype=torch.long)
    for k, r in enumerate(rows):
        n = r["num_nodes"]
        feats[k, :n] = torch.from_numpy(np.frombuffer(r["feats"], dtype=np.float32).reshape(n, -1).copy())
        nb = np.frombuffer(r["nbr"], dtype=np.int32).reshape(n, 6).astype(np.int64)
        nb = np.where(nb < 0, pad, nb)
        nbr[k, :n] = torch.from_numpy(nb)
        mask[k, :n] = True
        coords[k, :n] = torch.from_numpy(np.frombuffer(r["coords"], dtype=np.int16).reshape(n, 2).astype(np.int64).copy())
    return feats.to(device), nbr.to(device), mask.to(device), coords.to(device), pad


def main():
    device = torch.device("cuda")
    model = HexfieldNet()
    if len(sys.argv) > 1:
        p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
        model.load_state_dict(p.get("model", p), strict=True)
    model.to(device).eval()

    pool = midgame_states(900, seed=21)
    PAIR = 3.8e7
    print("N_target  batch  Npad  ms/forward  evals/s  VRAM_GiB")
    for target in (300, 600, 900, 1200, 1800):
        picked = states_near(target, 80, 64, pool)
        if len(picked) < 4:
            print(f"{target}: only {len(picked)} states, skip")
            continue
        pad_guess = -(-(target + 80) // 256) * 256
        bmax = max(1, int(PAIR // (pad_guess + NUM_TOKENS) ** 2))
        picked = picked[:bmax]
        feats, nbr, mask, coords, pad = make_batch(picked, device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                model.forward_policy_value(feats, nbr, mask, coords)  # warm
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            reps = 5
            for _ in range(reps):
                with torch.autocast("cuda", dtype=torch.float16):
                    model.forward_policy_value(feats, nbr, mask, coords)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / reps
        g = len(picked)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{target:>8} {g:>6} {pad:>5} {dt*1000:>10.1f} {g/dt:>8.0f} {peak:>8.2f}")

    # profile one N~900 forward
    print("\n--- profile @ N~900 ---")
    picked = states_near(900, 80, 16, pool)
    feats, nbr, mask, coords, pad = make_batch(picked, device)
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            model.forward_policy_value(feats, nbr, mask, coords)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.float16):
                    model.forward_policy_value(feats, nbr, mask, coords)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()
