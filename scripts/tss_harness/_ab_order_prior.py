"""Paired A/B: net-prior ordering hints vs baseline on the selfplay dev
split (priors were recorded per-position at generation by gen_positions.py
from the ep90 net's root prior policy — no fresh GPU eval needed).

Both arms: cap 500, unbounded, wide, goal=both, dual_pass=True (the
current production-candidate profile). Arms run as separate batch calls
(fresh solver per call; hinted solves additionally run cold by design).

Usage: /root/.venvs/order-dev/bin/python scripts/tss_harness/_ab_order_prior.py
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


def action_coord_map() -> dict[int, tuple[int, int]]:
    """Invert hexo_engine's coord->action_id over the board universe."""
    from hexo_engine import api
    from hexo_engine.types import AxialCoord, PlacementAction

    inv: dict[int, tuple[int, int]] = {}
    for q in range(-15, 16):
        for r in range(-15, 16):
            try:
                aid = api.action_id(PlacementAction(AxialCoord(q=q, r=r)))
            except Exception:
                continue
            inv[int(aid)] = (q, r)
    return inv


def main() -> int:
    positions = load_set("selfplay_v1", "dev")
    raw_by_id = {}
    # selfplay_positions.jsonl is untracked V1 data; the v1-soak worktree
    # holds the canonical copy.
    v1_raws = Path("/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/v1-soak/raws")
    for line in open(v1_raws / "selfplay_positions.jsonl"):
        r = json.loads(line)
        raw_by_id[r["id"]] = r

    inv = action_coord_map()
    print(f"action map: {len(inv)} cells", flush=True)

    states, hints, kept = [], [], []
    n_no_prior = 0
    for p in positions:
        raw = raw_by_id.get(p.pos_id)
        prior = (raw or {}).get("prior") or []
        h = [(inv[aid][0], inv[aid][1], float(w))
             for aid, w in prior if aid in inv]
        if not h:
            n_no_prior += 1
            continue
        states.append(corpus_lib.build_state(list(p.moves)))
        hints.append(h)
        kept.append(p.pos_id)
    print(f"positions with priors: {len(kept)} (skipped {n_no_prior})", flush=True)

    base = _rust.hexfield_eq_deep_solve_batch(
        states, CAP, "both", 0, False, False, True, True)
    hinted = _rust.hexfield_eq_deep_solve_batch(
        states, CAP, "both", 0, False, False, True, True,
        ordering_hints=hints)

    cov = lambda rs: Counter(  # noqa: E731
        r["status"] for r in rs if r["deep_verify_failed"] == 0)
    print("baseline:", dict(cov(base)), flush=True)
    print("hinted:  ", dict(cov(hinted)), flush=True)

    up, down, flips = [], [], []
    node_saved = node_spent = 0
    for pid, b, h in zip(kept, base, hinted):
        bd = b["status"] in ("win", "loss") and b["deep_verify_failed"] == 0
        hd = h["status"] in ("win", "loss") and h["deep_verify_failed"] == 0
        if hd and not bd:
            up.append(pid)
        if bd and not hd:
            down.append(pid)
        if bd and hd and b["status"] != h["status"]:
            flips.append(pid)   # both verified, different verdicts = ALARM
        if bd and hd:
            d = int(b["deep_nodes"]) - int(h["deep_nodes"])
            if d > 0:
                node_saved += d
            else:
                node_spent += -d
    vf = sum(int(r["deep_verify_failed"]) for r in base + hinted)
    print(f"upgrades {len(up)} downgrades {len(down)} "
          f"CONTRADICTIONS {len(flips)} vf_total {vf}", flush=True)
    print(f"on both-decided: nodes saved {node_saved}, added {node_spent}", flush=True)
    if up[:10]:
        print("first upgrades:", up[:10], flush=True)
    if flips:
        print("!!! soundness alarm:", flips, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
