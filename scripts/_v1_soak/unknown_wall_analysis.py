"""V1-raw follow-up: where does Unknown-solve wall time sit? Splits each WIN
arm's Unknown solves into cap-bound (deep_nodes >= cap: the budget-exhausted
grinds a census/deadline dismissal could cut short) vs frontier-exhausted
(search space closed below the cap: already-cheap structural Unknowns).
A free selfplay-cohort preview of the residue map's A5-sizing question.

Usage: python unknown_wall_analysis.py <raw.jsonl> [arm ...]
"""

from __future__ import annotations

import json
import sys

CAP = 500


def main():
    path = sys.argv[1]
    arms = sys.argv[2:] or ["unbounded_wide", "h16_flat_wide"]
    by_arm = {a: [] for a in arms}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("arm") in by_arm:
                by_arm[r["arm"]].append(r)
    for arm, rows in by_arm.items():
        total = sum(r["wall_nanos"] for r in rows) / 1e9
        unk = [r for r in rows if r["status"] == "unknown"]
        unk_w = sum(r["wall_nanos"] for r in unk) / 1e9
        capb = [r for r in unk if r["deep_nodes"] >= CAP]
        capb_w = sum(r["wall_nanos"] for r in capb) / 1e9
        fx = [r for r in unk if r["deep_nodes"] < CAP]
        fx_w = sum(r["wall_nanos"] for r in fx) / 1e9
        fx.sort(key=lambda r: -r["wall_nanos"])
        top20 = sum(r["wall_nanos"] for r in fx[:20]) / 1e9
        hi = [r for r in capb if r["wall_nanos"] > 50e6]
        hi_w = sum(r["wall_nanos"] for r in hi) / 1e9
        print(f"=== {arm}: n={len(rows)} total_wall={total:.1f}s")
        print(f"  unknown: n={len(unk)} wall={unk_w:.1f}s ({100*unk_w/total:.1f}% of arm wall)")
        print(f"    cap-bound (nodes>={CAP}): n={len(capb)} wall={capb_w:.1f}s ({100*capb_w/total:.1f}%)")
        print(f"    cap-bound >50ms:          n={len(hi)} wall={hi_w:.1f}s")
        print(f"    frontier-exhausted:       n={len(fx)} wall={fx_w:.1f}s ({100*fx_w/total:.1f}%); top-20 hold {top20:.1f}s")
        for r in fx[:3]:
            print(f"      big fx: pos={r['pos_id']} wall={r['wall_nanos']/1e6:.0f}ms "
                  f"nodes={r['deep_nodes']} placements={r['placements']} hot={r.get('hot')}")


if __name__ == "__main__":
    main()
