"""Print compact per-bucket tables from _wf_s3_full_out.json agg block.

Usage: python _wf_s3_tables.py IN_JSON
"""
import json
import sys

CKPTS = ["m3c5", "m3c7", "m4c12"]
SOURCES = ["m3ep5", "m3ep9", "m4ep12"]
BUCKET_NAMES = ["[0,5)", "[5,20)", "[20,60)", "[60,120)", "[120+)"]

d = json.load(open(sys.argv[1], encoding="utf-8"))
agg = d["agg"]["by"]

for sname in SOURCES:
    print(f"\n===== source {sname} =====")
    for cname in CKPTS:
        v = agg[f"{cname}|{sname}"]
        o, w = v["overall"], v["within_game"]
        print(f"\n  {cname}: n_pos={v['n_positions']} n_games={v['n_games']} | overall MAE={o['mae']} "
              f"bias={o['bias']} slope={o['slope']} r={o['pearson_r']} pred_range=[{o['pred_min']},{o['pred_max']}]")
        print(f"    within-game: rho_full mean/med/p10 = {w['spearman_full_mean']}/{w['spearman_full_median']}/{w['spearman_full_p10']}"
              f" | rho_conv(<60) mean/med/p10 = {w['spearman_conv_lt60_mean']}/{w['spearman_conv_lt60_median']}/{w['spearman_conv_lt60_p10']}"
              f" (n={w['n_games_conv']})")
        print(f"    monotonic-decrease rate: adj={w['mono_adj_rate_mean']} stride5={w['mono_stride5_rate_mean']}"
              f" conv_adj={w['mono_adj_rate_conv_lt60']}")
        print(f"    {'bucket':9s} {'n':>6s} {'MAE':>8s} {'bias':>8s} {'biasMed':>8s} {'predMean':>9s} {'predStd':>8s} {'slope':>7s} {'r':>6s} {'MAEmed':>7s}")
        for bn in BUCKET_NAMES:
            b = v["buckets"][bn]
            if b.get("n", 0) == 0:
                print(f"    {bn:9s} {0:6d}")
                continue
            print(f"    {bn:9s} {b['n']:6d} {b['mae']:8.1f} {b['bias']:8.1f} {b['bias_median']:8.1f} "
                  f"{b['pred_mean']:9.1f} {b['pred_std']:8.1f} "
                  f"{(b['slope'] if b['slope'] is not None else float('nan')):7.2f} "
                  f"{(b['pearson_r'] if b['pearson_r'] is not None else float('nan')):6.2f} {b['mae_med']:7.1f}")
