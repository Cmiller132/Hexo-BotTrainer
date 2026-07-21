"""Second hint oracle: threat-proximity statics. Weight each empty cell by
proximity to EXISTING stones (any color): w = sum over stones of
1/(1+hexdist) — hot cells near the action first, cold periphery last.
Rationale: V1 measured proven wins as 45.5% "hot" vs 6% for grinds; the
policy-prior oracle failed because it encodes move QUALITY, not proof
locality. This oracle encodes locality only.

Also runs the batch-order-dependence probe (reverse-order solve of the
same positions, no hints) to quantify the TT-carryover caveat.

Usage: /root/.venvs/order-dev/bin/python scripts/tss_harness/_ab_order_threat.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "_v1_soak"))

import arch_env  # noqa: F401
import corpus_lib
from hexfield_eq import _rust
from tss_harness.sets import load_set

CAP = 500


def hexdist(a, b) -> int:
    aq, ar = a
    bq, br = b
    return max(abs(aq - bq), abs(ar - br), abs((-aq - ar) - (-bq - br)))


def threat_hints(moves: list) -> list:
    stones = [(int(q), int(r)) for q, r in moves]
    if not stones:
        return []
    # candidate universe: ring around the stones (radius 3 halo)
    cells = set()
    for (q, r) in stones:
        for dq in range(-3, 4):
            for dr in range(-3, 4):
                if hexdist((0, 0), (dq, dr)) <= 3:
                    cells.add((q + dq, r + dr))
    cells -= set(stones)
    out = []
    for c in cells:
        w = sum(1.0 / (1 + hexdist(c, s)) for s in stones)
        out.append((c[0], c[1], float(w)))
    out.sort(key=lambda t: -t[2])
    return out[:64]


def coverage(rs):
    return Counter(r["status"] for r in rs if r["deep_verify_failed"] == 0)


def paired(kept, a, b, label):
    up = down = flips = 0
    for pid, ra, rb in zip(kept, a, b):
        ad = ra["status"] in ("win", "loss") and ra["deep_verify_failed"] == 0
        bd = rb["status"] in ("win", "loss") and rb["deep_verify_failed"] == 0
        if bd and not ad:
            up += 1
        if ad and not bd:
            down += 1
        if ad and bd and ra["status"] != rb["status"]:
            flips += 1
    print(f"{label}: +{up} / -{down} / contradictions {flips}", flush=True)
    return flips


def main() -> int:
    positions = load_set("human_v1", "dev")
    states, hints, kept = [], [], []
    for p in positions:
        h = threat_hints(list(p.moves))
        states.append(corpus_lib.build_state(list(p.moves)))
        hints.append(h)
        kept.append(p.pos_id)
    print(f"n={len(kept)}", flush=True)

    base = _rust.hexfield_eq_deep_solve_batch(
        states, CAP, "both", 0, False, False, True, True)
    print("baseline:", dict(coverage(base)), flush=True)

    hinted = _rust.hexfield_eq_deep_solve_batch(
        states, CAP, "both", 0, False, False, True, True,
        ordering_hints=hints)
    print("threat-hinted:", dict(coverage(hinted)), flush=True)
    alarms = paired(kept, base, hinted, "threat oracle vs baseline")

    # order-dependence probe: same arm, reversed batch order
    rev = _rust.hexfield_eq_deep_solve_batch(
        list(reversed(states)), CAP, "both", 0, False, False, True, True)
    rev = list(reversed(rev))
    print("reversed-order:", dict(coverage(rev)), flush=True)
    alarms += paired(kept, base, rev, "batch order (fwd vs rev)")
    return 1 if alarms else 0


if __name__ == "__main__":
    raise SystemExit(main())
