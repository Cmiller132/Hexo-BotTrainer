"""V1 SOAK tactical-anchor reporter: per-position verdict + time-to-verdict +
zone delta for the forcing/spare corpus raws. Ties the trainer leaf solver to
the campaign's puzzle gate (does the production leaf cap find the certified win?)
and surfaces where the horizon ladder + zone AND-generation activate.

Usage:
    python anchors_report.py <out.json> <corpus_raw.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

# The arms whose verdict we report per position (WIN-goal).
REPORT_ARMS = [
    "h16_flat_wide",
    "h16_ladder_wide",
    "unbounded_wide",
    "h16_flat_zone",
    "h16_ladder_zone",
    "unbounded_zone",
    "h16_flat_narrow",
    "unbounded_narrow",
]


def main():
    out_path = sys.argv[1]
    raw_path = sys.argv[2]
    by_pos = defaultdict(dict)
    expect = {}
    for line in open(raw_path):
        r = json.loads(line)
        by_pos[r["pos_id"]][r["arm"]] = r
        if r.get("expect_win") is not None:
            expect[r["pos_id"]] = r["expect_win"]

    positions = []
    arm_found = defaultdict(int)
    arm_zone_pos = defaultdict(int)
    n_expect_win = 0
    soundness_violations = []
    for pos_id, arms in sorted(by_pos.items()):
        exp = expect.get(pos_id)
        if exp:
            n_expect_win += 1
        row = {"pos_id": pos_id, "expect_win": exp, "arms": {}}
        for a in REPORT_ARMS:
            r = arms.get(a)
            if not r:
                continue
            row["arms"][a] = {
                "status": r["status"],
                "wall_us": round(r["wall_nanos"] / 1000.0, 1),
                "cert_depth": r["cert_depth"],
                "deep_nodes": r["deep_nodes"],
                "zone_nodes": r["zone_nodes"],
            }
            if r["status"] == "win":
                arm_found[a] += 1
            if r["zone_nodes"] > 0:
                arm_zone_pos[a] += 1
            # Soundness: a NO position must never come back WIN.
            if exp is False and r["status"] == "win":
                soundness_violations.append((pos_id, a))
        positions.append(row)

    summary = {
        "n_positions": len(positions),
        "n_expect_win": n_expect_win,
        "arm_win_found": dict(arm_found),
        "arm_zone_positions": dict(arm_zone_pos),
        "soundness_violations": soundness_violations,
        "positions": positions,
    }
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"positions={len(positions)} expect_win={n_expect_win}")
    print("WIN found (of all positions) per arm:")
    for a in REPORT_ARMS:
        print(f"  {a:20s} win_found={arm_found[a]:3d}  zone_positions={arm_zone_pos[a]:3d}")
    if soundness_violations:
        print(f"!!! SOUNDNESS VIOLATIONS (NO->WIN): {soundness_violations}")
    else:
        print("soundness: no NO->WIN (clean)")


if __name__ == "__main__":
    main()
