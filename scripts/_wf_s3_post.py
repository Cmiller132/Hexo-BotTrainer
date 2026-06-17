"""Post-analysis of _wf_s3_full_out.json: head vs null baselines, compact tables.

Baselines per source (oracle-fit on the same sample, so they are STRONG nulls):
  - const: best constant predictor (mean true_ml of the source sample)
  - ply-lin: OLS true_ml ~ a*ply + b (knows only the move number, not the board)

Usage: python _wf_s3_post.py IN_JSON OUT_TXT
"""
import json
import sys

import numpy as np

def _spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


BUCKETS = [(0, 5), (5, 20), (20, 60), (60, 120), (120, 10 ** 9)]
BUCKET_NAMES = ["[0,5)", "[5,20)", "[20,60)", "[60,120)", "[120+)"]
CKPTS = ["m3c5", "m3c7", "m4c12"]
SOURCES = ["m3ep5", "m3ep9", "m4ep12"]


def main():
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    games = d["games"]
    out = []
    P = out.append

    for sname in SOURCES:
        gs = [g for g in games if g["source"] == sname]
        true_all, ply_all = [], []
        for g in gs:
            L = g["game_len"]
            t = np.arange(L, dtype=np.float64)
            true_all.append(L - 1 - t)
            ply_all.append(t)
        true_all = np.concatenate(true_all)
        ply_all = np.concatenate(ply_all)
        lens = sorted(g["game_len"] for g in gs)
        P(f"== source {sname}: {len(gs)} games, {len(true_all)} positions, "
          f"len p10/p50/p90 = {lens[len(lens)//10]}/{lens[len(lens)//2]}/{lens[(len(lens)*9)//10]}")

        c = true_all.mean()
        a, b = np.polyfit(ply_all, true_all, 1)
        base_const = np.full_like(true_all, c)
        base_ply = a * ply_all + b
        P(f"   nulls: const={c:.1f}; ply-lin slope={a:.3f} icpt={b:.1f}")

        preds = {"null_const": base_const, "null_plylin": base_ply}
        for cname in CKPTS:
            preds[cname] = np.concatenate([np.asarray(g[cname], dtype=np.float64) for g in gs])
            if cname + "_med" in gs[0]:
                preds[cname + "_med"] = np.concatenate([np.asarray(g[cname + "_med"], dtype=np.float64) for g in gs])

        hdr = f"   {'bucket':9s} {'n':>6s} " + " ".join(f"{k:>12s}" for k in preds)
        P(hdr + "   (MAE per bucket)")
        for (lo, hi), bn in zip(BUCKETS, BUCKET_NAMES):
            m = (true_all >= lo) & (true_all < hi)
            if m.sum() == 0:
                continue
            row = f"   {bn:9s} {int(m.sum()):6d} "
            row += " ".join(f"{np.abs(p[m] - true_all[m]).mean():12.1f}" for p in preds.values())
            P(row)
        P(f"   {'bias':9s} {'':6s} " + " ".join(
            f"{(p - true_all).mean():12.1f}" for p in preds.values()) + "   (overall bias)")

        # phase-aware within-game metrics (Hexo period-4 ply structure:
        # 2 placements/turn x 2 players; the head oscillates with phase)
        for cname in CKPTS:
            s4, rho4, rho4c = [], [], []
            for g in gs:
                L = g["game_len"]
                pred = np.asarray(g[cname], dtype=np.float64)
                true = L - 1 - np.arange(L, dtype=np.float64)
                if L > 4:
                    s4.append(float(np.mean(pred[4:] < pred[:-4])))
                for off in range(4):
                    ps, ts = pred[off::4], true[off::4]
                    if len(ps) >= 8:
                        rho4.append(_spear(ps, ts))
                    m = ts < 60
                    if m.sum() >= 6:
                        rho4c.append(_spear(ps[m], ts[m]))
            P(f"   phase-aware {cname}: mono_stride4={np.mean(s4):.3f} "
              f"rho_phase_full={np.mean(rho4):.3f} rho_phase_conv<60={np.mean(rho4c):.3f} "
              f"(n_seq={len(rho4)}/{len(rho4c)})")
        P("")

    text = "\n".join(out)
    open(sys.argv[2], "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
