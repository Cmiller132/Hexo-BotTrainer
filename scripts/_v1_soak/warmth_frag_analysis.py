"""Paired warm-vs-cold analysis for the TSS_SHARED_FRAGMENTS=1 warmth rerun.

Compares raws/soak_warmth_frag_s{0,1}.jsonl (fragment store ON) against
raws/soak_warmth.jsonl (the V1 run: same driver, store env-gated OFF =
cold control). Deterministic solver => any deep_nodes/status delta is the
fragment store's causal effect.

Usage: python warmth_frag_analysis.py <raws_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    raws = Path(sys.argv[1] if len(sys.argv) > 1 else "raws")
    old = {}
    for line in open(raws / "soak_warmth.jsonl"):
        r = json.loads(line)
        old[(r["pos_id"], r["arm"])] = r
    new = {}
    for part in ("s0", "s1"):
        for line in open(raws / f"soak_warmth_frag_{part}.jsonl"):
            r = json.loads(line)
            new[(r["pos_id"], r["arm"])] = r

    per_arm = {}
    flips = []
    savers = []
    for k, r in new.items():
        o = old.get(k)
        if o is None:
            continue
        arm = k[1]
        st = per_arm.setdefault(arm, dict(
            n=0, changed=0, nodes_saved=0, nodes_added=0,
            wall_old=0, wall_new=0, flips=0))
        st["n"] += 1
        d = r["deep_nodes"] - o["deep_nodes"]
        if d != 0:
            st["changed"] += 1
            if d < 0:
                st["nodes_saved"] -= d
                savers.append((k, o["deep_nodes"], r["deep_nodes"],
                               o["status"], r["status"]))
            else:
                st["nodes_added"] += d
        st["wall_old"] += o["wall_nanos"]
        st["wall_new"] += r["wall_nanos"]
        if o["status"] != r["status"]:
            st["flips"] += 1
            flips.append((k, o["status"], r["status"],
                          o["deep_nodes"], r["deep_nodes"]))

    for arm, st in sorted(per_arm.items()):
        print(f"{arm}: n={st['n']} changed={st['changed']} "
              f"nodes_saved={st['nodes_saved']} nodes_added={st['nodes_added']} "
              f"wall {st['wall_old']/1e9:.1f}s -> {st['wall_new']/1e9:.1f}s "
              f"verdict_flips={st['flips']}")
    print(f"\nVERDICT FLIPS ({len(flips)}):")
    for f in flips:
        print("  ", f)
    savers.sort(key=lambda s: s[1] - s[2], reverse=True)
    print(f"\nTOP NODE SAVERS ({len(savers)} total):")
    for s in savers[:12]:
        print("  ", s)


if __name__ == "__main__":
    main()
