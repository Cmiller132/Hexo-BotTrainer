"""V1-raw follow-up 2: structural characterization of the cap-bound Unknown
grinds (deep_nodes >= cap) in the unbounded_wide arm — the A5 census
fail-fast target category (73.5% of arm wall). A free selfplay-cohort
preview of whether a FirstStone census/deadline dismissal could plausibly
fire on them, plus deletion evidence for dead machinery (warmth, interior
gate) inside the grind class specifically.

Usage: python grind_characterization.py <raw.jsonl> [arm]
"""

from __future__ import annotations

import json
import sys
from collections import Counter

CAP = 500


def dist(vals, name, buckets=None):
    vals = sorted(vals)
    n = len(vals)
    if not n:
        print(f"  {name}: (empty)")
        return
    q = lambda p: vals[min(n - 1, int(p * n))]
    print(f"  {name}: min={vals[0]} p25={q(0.25)} p50={q(0.5)} p75={q(0.75)} p90={q(0.9)} max={vals[-1]}")
    if buckets:
        c = Counter()
        for v in vals:
            for lo, hi, lab in buckets:
                if lo <= v < hi:
                    c[lab] += 1
                    break
        print(f"    buckets: {dict(c)}")


def main():
    path = sys.argv[1]
    arm = sys.argv[2] if len(sys.argv) > 2 else "unbounded_wide"
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("arm") == arm:
                rows.append(r)

    unk = [r for r in rows if r["status"] == "unknown"]
    grind = [r for r in unk if r["deep_nodes"] >= CAP]
    fx = [r for r in unk if r["deep_nodes"] < CAP]
    wins = [r for r in rows if r["status"] == "win"]

    print(f"=== arm={arm}: solves={len(rows)} unknown={len(unk)} "
          f"cap-bound={len(grind)} frontier-exhausted={len(fx)} wins={len(wins)}")

    for label, group in (("CAP-BOUND GRINDS", grind), ("PROVEN WINS (contrast)", wins),
                         ("FRONTIER-EXHAUSTED UNK (contrast)", fx)):
        if not group:
            continue
        print(f"\n--- {label} (n={len(group)})")
        dist([r["opp_threats"] for r in group], "opp_threats")
        dist([r["min_hitting_set"] for r in group], "min_hitting_set")
        dist([r["band"] for r in group], "band")
        dist([r["placements"] for r in group], "placements")
        hot = sum(1 for r in group if r["hot"])
        print(f"  hot: {hot}/{len(group)} ({100*hot/len(group):.1f}%)")
        dist([round(r["net_value"], 2) for r in group], "net_value")
        dist([r["wall_nanos"] // 1_000_000 for r in group], "wall_ms")
        # dead-machinery audit inside this class
        ig_ev = sum(r.get("stats_interior_gate_evaluations", 0) for r in group)
        ig_dis = sum(r.get("stats_interior_gate_dismissals", 0) for r in group)
        frag = sum(r.get("stats_fragment_lookups", 0) for r in group)
        tt_h = sum(r.get("stats_tt_hits", 0) for r in group)
        tt_e = sum(r.get("stats_tt_entries", 0) for r in group)
        print(f"  interior_gate: evals={ig_ev} dismissals={ig_dis}; fragment_lookups={frag}; "
              f"tt_hits={tt_h} tt_entries={tt_e} (hit/entry={tt_h/max(1,tt_e):.2f})")

    # A5 plausibility cut: FirstStone census dismissal keys on the attacker
    # lacking a fast enough win — proxy: low opp_threats / low band grinds are
    # the ones a census bound could kill early.
    if grind:
        lo = [r for r in grind if r["opp_threats"] <= 1]
        lo_w = sum(r["wall_nanos"] for r in lo) / 1e9
        tot_w = sum(r["wall_nanos"] for r in grind) / 1e9
        print(f"\nA5 proxy: grinds with opp_threats<=1: n={len(lo)}/{len(grind)} "
              f"wall={lo_w:.1f}s of {tot_w:.1f}s grind wall ({100*lo_w/max(1e-9,tot_w):.1f}%)")
        # net calibration on grinds: does the net already 'know' these?
        conf = [r for r in grind if abs(r["net_value"]) > 0.5]
        print(f"net-confident grinds (|v|>0.5): {len(conf)}/{len(grind)}")


if __name__ == "__main__":
    main()
