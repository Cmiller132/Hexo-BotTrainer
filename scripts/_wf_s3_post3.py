"""Extra within-game discrimination metrics from _wf_s3_full_out.json.

Per ckpt x source:
  - conv(<60) Spearman using the MEDIAN-of-bins decode (the agg block used mean decode).
  - end-vs-mid pairwise discrimination: within each game, fraction of
    (end position true_ml<20, mid position 60<=true_ml<120) pairs where
    pred_end < pred_mid. 0.5 = chance. This is the signal a moves-left
    utility would actually lean on ("are we close to done?").
  - net end-drop: fraction of games where mean pred over the last 10 plies
    < mean pred over plies with true_ml in [60,120).

Usage: python _wf_s3_post3.py IN_JSON OUT_TXT
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


CKPTS = ["m3c5", "m3c7", "m4c12"]
SOURCES = ["m3ep5", "m3ep9", "m4ep12"]


def main():
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    games = d["games"]
    out = []
    P = out.append
    for sname in SOURCES:
        gs = [g for g in games if g["source"] == sname]
        P(f"== source {sname}: {len(gs)} games")
        for cname in CKPTS:
            conv_rho_med, pairwise, drop = [], [], []
            for g in gs:
                L = g["game_len"]
                true = L - 1 - np.arange(L, dtype=np.float64)
                pred = np.asarray(g[cname], dtype=np.float64)
                pmed = np.asarray(g[cname + "_med"], dtype=np.float64)
                m = true < 60
                if m.sum() >= 10:
                    conv_rho_med.append(_spear(pmed[m], true[m]))
                end_m = true < 20
                mid_m = (true >= 60) & (true < 120)
                if end_m.sum() >= 5 and mid_m.sum() >= 5:
                    e = pred[end_m][:, None]
                    q = pred[mid_m][None, :]
                    pairwise.append(float((e < q).mean()))
                    drop.append(float(pred[-10:].mean() < pred[mid_m].mean()))
            P(f"   {cname}: conv<60 rho (median decode) mean/med/p10 = "
              f"{np.mean(conv_rho_med):.3f}/{np.median(conv_rho_med):.3f}/"
              f"{np.percentile(conv_rho_med, 10):.3f} (n={len(conv_rho_med)})")
            if pairwise:
                P(f"        end(<20)-vs-mid([60,120)) pairwise acc mean/med/p10 = "
                  f"{np.mean(pairwise):.3f}/{np.median(pairwise):.3f}/"
                  f"{np.percentile(pairwise, 10):.3f} (n_games={len(pairwise)}); "
                  f"end-drop frac = {np.mean(drop):.3f}")
        P("")
    text = "\n".join(out)
    open(sys.argv[2], "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
