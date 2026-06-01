"""Phase 1: entropy / diffuseness analysis from LOGGED selfplay NPZ data only.

No model inference. For every recorded decision position we read:
  - turn_index (metadata col 0)  -> move number
  - legalMaskNCHW.sum()          -> legal move count
  - policyTargetsNCHW            -> POST-MCTS visit distribution (training target)
  - rootPolicyNCHW               -> root prior used by search (CAVEAT: includes
                                    root_policy_temperature=1.1 + Dirichlet noise)
  - valueTargetsN                -> game outcome z for that position's player

We quantify, binned by move number and by legal-move count:
  entropy, top-1 prob, effective #moves (exp H), and entropy normalized by
  log(legal_count) (1.0 == uniform over legal moves).
"""
import numpy as np, glob, os, sys, json

RUN = "E:/Hexo-BotTrainer/runs/dense_cnn_model1_scratch_64/selfplay"

def dist_stats(planeNCHW, legalNCHW):
    """Per-row entropy, top1, effective-moves, normalized entropy over legal set."""
    N = planeNCHW.shape[0]
    p = planeNCHW.reshape(N, -1).astype(np.float64)
    legal = legalNCHW.reshape(N, -1).astype(bool)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    p = p / s
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    H = -(p * logp).sum(axis=1)            # nats
    top1 = p.max(axis=1)
    eff = np.exp(H)                        # effective number of moves
    legal_n = legal.sum(axis=1).astype(np.float64)
    legal_n_safe = np.where(legal_n > 1, legal_n, 2.0)
    Hnorm = H / np.log(legal_n_safe)       # 1.0 == uniform over legal
    return H, top1, eff, legal_n, Hnorm

def collect(epoch, max_games=256):
    files = sorted(glob.glob(os.path.join(RUN, f"epoch_{epoch:06d}_game_*.npz")))[:max_games]
    rows = []
    for f in files:
        d = np.load(f)
        md = d["metadataInputNC"]
        if md.shape[0] == 0:
            continue
        turn = md[:, 0]
        vis = d["policyTargetsNCHW"]
        root = d["rootPolicyNCHW"]
        legal = d["legalMaskNCHW"]
        val = d["valueTargetsN"]
        Hv, t1v, effv, ln, Hnv = dist_stats(vis, legal)
        Hr, t1r, effr, _, Hnr = dist_stats(root, legal)
        for i in range(md.shape[0]):
            rows.append((turn[i], ln[i], Hv[i], t1v[i], effv[i], Hnv[i],
                         Hr[i], t1r[i], effr[i], Hnr[i], val[i]))
    return np.array(rows, dtype=np.float64)

COLS = ["turn", "legal_n", "H_visit", "top1_visit", "eff_visit", "Hnorm_visit",
        "H_root", "top1_root", "eff_root", "Hnorm_root", "value"]

def binned(arr, xcol, bins):
    xi = COLS.index(xcol)
    x = arr[:, xi]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 5:
            continue
        sub = arr[m]
        row = {"bin": f"[{lo:.0f},{hi:.0f})", "n": int(m.sum())}
        for c in ["legal_n", "eff_visit", "top1_visit", "Hnorm_visit",
                  "eff_root", "top1_root", "Hnorm_root"]:
            row[c] = float(np.mean(sub[:, COLS.index(c)]))
        out.append(row)
    return out

def fmt_table(rows, xlabel):
    hdr = f"{xlabel:>12} {'n':>6} {'legalN':>8} | {'effVIS':>7} {'top1V':>6} {'HnormV':>7} | {'effROOT':>8} {'top1R':>6} {'HnormR':>7}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['bin']:>12} {r['n']:>6} {r['legal_n']:>8.1f} | "
                     f"{r['eff_visit']:>7.2f} {r['top1_visit']:>6.3f} {r['Hnorm_visit']:>7.3f} | "
                     f"{r['eff_root']:>8.2f} {r['top1_root']:>6.3f} {r['Hnorm_root']:>7.3f}")
    return "\n".join(lines)

if __name__ == "__main__":
    epochs = [int(x) for x in sys.argv[1:]] or [21]
    move_bins = [0, 10, 20, 30, 40, 60, 80, 100, 130, 200]
    legal_bins = [0, 10, 20, 40, 60, 80, 100, 130, 160, 200, 300, 500]
    summary = {}
    for ep in epochs:
        arr = collect(ep)
        print(f"\n############ EPOCH {ep}  (positions={arr.shape[0]}) ############")
        print("\n=== by MOVE NUMBER (turn_index) ===")
        bm = binned(arr, "turn", move_bins)
        print(fmt_table(bm, "move#"))
        print("\n=== by LEGAL-MOVE COUNT ===")
        bl = binned(arr, "legal_n", legal_bins)
        print(fmt_table(bl, "legalN"))
        # correlations
        c_eff_turn = np.corrcoef(arr[:, COLS.index("turn")], arr[:, COLS.index("eff_visit")])[0, 1]
        c_hn_turn = np.corrcoef(arr[:, COLS.index("turn")], arr[:, COLS.index("Hnorm_visit")])[0, 1]
        c_eff_legal = np.corrcoef(arr[:, COLS.index("legal_n")], arr[:, COLS.index("eff_visit")])[0, 1]
        print(f"\ncorr(eff_visit, move#)={c_eff_turn:.3f}  corr(Hnorm_visit, move#)={c_hn_turn:.3f}  corr(eff_visit, legalN)={c_eff_legal:.3f}")
        summary[ep] = {"by_move": bm, "by_legal": bl,
                       "corr_eff_visit_move": c_eff_turn,
                       "corr_Hnormvisit_move": c_hn_turn}
    with open("E:/Hexo-BotTrainer/analysis/phase1_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nwrote analysis/phase1_summary.json")
