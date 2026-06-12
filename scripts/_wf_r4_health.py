"""Per-epoch health gates for a dense_cnn_restnet run (CPU, no model forward).

Full-population census: diagnostics JSON + .hxr winners/lengths + npz targets.
Designed for babysitting dense_cnn_restnet_main_4 (works on main_3 for
calibration). Prints one line per epoch plus GATE verdict lines.

Usage: python _wf_r4_health.py RUN_NAME [MAX_EP]
e.g.   python _wf_r4_health.py dense_cnn_restnet_main_4
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python")
from hexo_utils.records import HexoRecordFile  # noqa: E402

RUNS = Path("/mnt/e/Hexo-BotTrainer/runs")

# Gates calibrated on main_3: ep1-7 healthy (dec/game 85-109, ml80 share <=0.45,
# top1 0.57-0.65), ep8-9 sick (170/266 dec/game, ml80 0.52/0.65, top1 falling).
GATES = {
    "dec_per_game_max": 130.0,        # FAIL above (marathon spiral)
    "dec_growth_watch": 1.15,         # WATCH if dec/game grows >15% vs prev epoch
    "dec_growth_fail": 1.30,          # FAIL if >30% in one epoch or >15% twice
    "value_ce_rise_watch": 0.02,      # WATCH single-epoch rise
    "ml80_share_watch": 0.45,         # share of rows with moves_left >= 80
    "ml80_share_fail": 0.55,
    "top1_fullpop_min": 0.51,         # audit watch item 2 threshold
    "p0_share_lo": 0.44,
    "p0_share_hi": 0.56,
}


def epoch_stats(run: Path, ep: int):
    diag = run / "diagnostics" / f"epoch_{ep:06d}.json"
    if not diag.exists():
        return None
    d = json.load(open(diag))
    if d.get("status") != "completed" and "result" not in d.get("metadata", {}):
        return None
    res = d["metadata"]["result"]
    sp = res["selfplay"]
    tr = res.get("training") or {}
    comp = tr.get("loss_components") or {}
    tc = sp.get("temperature_control", {})
    out = {
        "ep": ep,
        "dec_per_game": sp["total_decisions"] / max(sp.get("games_finished", 1), 1),
        "rows": sp.get("raw_samples"),
        "pos_s": sp.get("search_positions_per_second", 0.0),
        "trunc": sp.get("truncated_games", 0),
        "ema": tc.get("expected_game_length", float("nan")),
        "policy_ce": comp.get("policy", float("nan")),
        "value_ce": comp.get("value", float("nan")),
        "eval": None,
    }
    ev = res.get("evaluation")
    if ev:
        out["eval"] = {k: v for k, v in ev.items() if isinstance(v, (int, float, str))}

    # .hxr census: winners + lengths
    hxr = run / "selfplay" / f"epoch_{ep:06d}.hxr"
    if hxr.exists():
        with HexoRecordFile.open(hxr) as rf:
            recs = list(rf.iter_records())
        lens = sorted(len(r.action_ids) for r in recs)
        winners = Counter(str(r.winner) for r in recs)
        n = max(len(recs), 1)
        p0 = sum(v for k, v in winners.items() if "0" in k)
        out["n_games"] = len(recs)
        out["len_p90"] = lens[int(0.9 * len(lens))] if lens else 0
        out["len_max"] = lens[-1] if lens else 0
        out["p0_share"] = p0 / n

    # npz census: ml80 share, full-pop visit-target top1, z mean
    ml_all, top1s, zs = [], [], []
    for f in sorted(glob.glob(str(run / "selfplay" / f"epoch_{ep:06d}_game_*.npz"))):
        try:
            z = np.load(f, allow_pickle=True)
            ml = z["moves_left"]
            ml_all.append(ml)
            zs.append(z["value"])
            w, off = z["pol_w"], z["pol_off"]
            for i in range(len(off) - 1):
                seg = w[off[i]:off[i + 1]]
                s = seg.sum()
                if s > 0:
                    top1s.append(float(seg.max() / s))
        except Exception:
            continue
    if ml_all:
        ml = np.concatenate(ml_all)
        out["ml80_share"] = float((ml >= 80).mean())
        out["npz_rows"] = int(ml.size)
        out["z_mean"] = float(np.concatenate(zs).mean())
        out["top1_fullpop"] = float(np.mean(top1s)) if top1s else float("nan")
    return out


def main():
    run = RUNS / sys.argv[1]
    max_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    prev = None
    dec_watch_streak = 0
    vce_rise_streak = 0
    p0_out_streak = 0
    seen_any = False
    for ep in range(1, max_ep + 1):
        s = epoch_stats(run, ep)
        if s is None:
            if seen_any:
                break
            continue
        seen_any = True
        flags = []
        if s["dec_per_game"] > GATES["dec_per_game_max"]:
            flags.append("FAIL:dec/game>%d" % GATES["dec_per_game_max"])
        if prev:
            g = s["dec_per_game"] / max(prev["dec_per_game"], 1e-9)
            if g > GATES["dec_growth_fail"]:
                flags.append(f"FAIL:dec growth x{g:.2f}")
            elif g > GATES["dec_growth_watch"]:
                dec_watch_streak += 1
                flags.append(f"{'FAIL' if dec_watch_streak >= 2 else 'WATCH'}:dec growth x{g:.2f} ({dec_watch_streak} consec)")
            else:
                dec_watch_streak = 0
            dv = s["value_ce"] - prev["value_ce"]
            if dv > GATES["value_ce_rise_watch"]:
                vce_rise_streak += 1
                flags.append(f"{'FAIL' if vce_rise_streak >= 2 else 'WATCH'}:valueCE +{dv:.3f} ({vce_rise_streak} consec)")
            else:
                vce_rise_streak = 0
        if s.get("ml80_share") is not None:
            if s["ml80_share"] > GATES["ml80_share_fail"]:
                flags.append(f"FAIL:ml80 {s['ml80_share']:.2f}")
            elif s["ml80_share"] > GATES["ml80_share_watch"]:
                flags.append(f"WATCH:ml80 {s['ml80_share']:.2f}")
        if s.get("top1_fullpop") and s["top1_fullpop"] < GATES["top1_fullpop_min"]:
            flags.append(f"FAIL:top1 {s['top1_fullpop']:.3f}")
        if s.get("p0_share") is not None and not (GATES["p0_share_lo"] <= s["p0_share"] <= GATES["p0_share_hi"]):
            p0_out_streak += 1
            flags.append(f"{'FAIL' if p0_out_streak >= 2 else 'WATCH'}:P0 {s['p0_share']:.2f} ({p0_out_streak} consec)")
        else:
            p0_out_streak = 0
        if s["trunc"]:
            flags.append(f"WATCH:trunc={s['trunc']}")

        print(f"ep{s['ep']:>2}: dec/g={s['dec_per_game']:6.1f} rows={s.get('npz_rows', s['rows'])} "
              f"polCE={s['policy_ce']:.3f} valCE={s['value_ce']:.3f} "
              f"ml80={s.get('ml80_share', float('nan')):.2f} top1={s.get('top1_fullpop', float('nan')):.3f} "
              f"P0={s.get('p0_share', float('nan')):.2f} p90={s.get('len_p90', 0)} EMA={s['ema']:.0f} "
              f"pos/s={s['pos_s']:.1f}"
              + (f"  GATES: {'; '.join(flags)}" if flags else "  GATES: all pass"))
        if s["eval"]:
            print(f"      eval: {s['eval']}")
        prev = s
    sys.stdout.flush()


if __name__ == "__main__":
    main()
