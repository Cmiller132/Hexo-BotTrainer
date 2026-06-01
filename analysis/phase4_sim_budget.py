"""Phase 4: MCTS simulation-budget vs action-space growth (logged data only).

The recorded visit target (policyTargetsNCHW) is the DELTA visit distribution for
that turn, normalized over the ~128 visits added this search call (mcts.rs
visit_policy with a baseline; zero-delta moves are dropped). So:
  nonzero cells               = # moves that received >=1 visit this turn
  weight * search_visits      ~ approx visit count for that move
We relate legal-move growth and visit coverage to move number, then sims/move.
"""
import numpy as np, glob, os, sys
RUN = "E:/Hexo-BotTrainer/runs/dense_cnn_model1_scratch_64/selfplay"
SIMS = 128

def collect(epoch, max_games=256):
    rows = []
    wsum = []
    for f in sorted(glob.glob(os.path.join(RUN, f"epoch_{epoch:06d}_game_*.npz")))[:max_games]:
        d = np.load(f); md = d["metadataInputNC"]
        if md.shape[0] == 0:
            continue
        N = md.shape[0]
        vt = d["policyTargetsNCHW"].reshape(N, -1)
        legal = d["legalMaskNCHW"].reshape(N, -1)
        turn = md[:, 0]
        legal_n = legal.sum(1)
        n_visited = (vt > 0).sum(1)                      # moves with >=1 new visit
        approx_visits = vt * SIMS
        n_ge2 = (approx_visits >= 1.5).sum(1)            # moves with >=2 visits
        top1 = vt.max(1)
        # effective # moves of the visit target
        p = vt / vt.sum(1, keepdims=True).clip(1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(np.where(p > 0, p * np.log(p), 0.0)).sum(1)
        eff = np.exp(H)
        for i in range(N):
            rows.append((turn[i], legal_n[i], n_visited[i], n_ge2[i], top1[i], eff[i]))
        wsum.append(vt.sum(1))
    wsum = np.concatenate(wsum)
    return np.array(rows, dtype=np.float64), wsum

C = ["turn", "legal_n", "n_visited", "n_ge2", "top1", "eff"]
BINS = [0, 10, 20, 30, 40, 60, 80, 100, 130, 200]

if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    a, wsum = collect(ep)
    print(f"EPOCH {ep}  positions={a.shape[0]}")
    print(f"sanity: visit-weight row sums  min={wsum.min():.4f} med={np.median(wsum):.4f} max={wsum.max():.4f} (expect ~1.0)\n")
    hdr = (f"{'move#':>10} {'n':>6} {'legalN':>8} {'visited':>8} {'vis>=2':>7} "
           f"{'%legalVis':>9} {'sims/legal':>10} {'sims/vis':>9} {'top1':>6} {'effVIS':>6}")
    print(hdr); print("-"*len(hdr))
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum() < 8: continue
        s = a[m]
        def mn(c): return s[:, C.index(c)].mean()
        legalN, vis, vge2 = mn("legal_n"), mn("n_visited"), mn("n_ge2")
        print(f"{f'[{lo},{hi})':>10} {int(m.sum()):>6} {legalN:>8.0f} {vis:>8.1f} {vge2:>7.1f} "
              f"{100*vis/legalN:>8.2f}% {SIMS/legalN:>10.3f} {SIMS/max(vis,1):>9.1f} {mn('top1'):>6.3f} {mn('eff'):>6.2f}")
    # correlations
    print(f"\ncorr(n_visited, move#)   = {np.corrcoef(a[:,0], a[:,C.index('n_visited')])[0,1]:+.3f}")
    print(f"corr(legal_n,  move#)    = {np.corrcoef(a[:,0], a[:,C.index('legal_n')])[0,1]:+.3f}")
    print(f"corr(n_visited, legal_n) = {np.corrcoef(a[:,C.index('legal_n')], a[:,C.index('n_visited')])[0,1]:+.3f}")
