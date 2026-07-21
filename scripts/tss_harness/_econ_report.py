"""Scratch: node economics per arm on the human quick sample."""
import json
import os
import sys

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness_runs")
ARMS = [
    ("20260720_211755_baseline_production_v2", "baseline_production_v2"),
    ("20260720_214526_cap1000_both", "cap1000_both"),
    ("20260720_214535_cap2000_both", "cap2000_both"),
    ("20260720_214548_cap4000_both", "cap4000_both"),
    ("20260720_214610_twopass_cap500", "twopass_cap500"),
    ("20260720_214616_twopass_cap2000", "twopass_cap2000"),
]

set_name = sys.argv[1] if len(sys.argv) > 1 else "human_v1"
print(f"{'arm':24s} {'n':>5} {'dec':>5} {'W':>4} {'L':>4} "
      f"{'Mnodes':>7} {'nd/pos':>7} {'nd/dec':>8}")
for d, a in ARMS:
    f = os.path.join(RUNS, d, f"records_{a}_{set_name}.jsonl")
    if not os.path.exists(f):
        print(f"{a:24s} missing")
        continue
    recs = [json.loads(line) for line in open(f)]
    n = len(recs)
    nodes = sum(r["cost"] for r in recs)
    wins = sum(1 for r in recs if r["status"] == "win" and r["verified"])
    losses = sum(1 for r in recs if r["status"] == "loss" and r["verified"])
    dec = wins + losses
    print(f"{a:24s} {n:>5} {dec:>5} {wins:>4} {losses:>4} "
          f"{nodes/1e6:>7.2f} {nodes//n:>7} {nodes//max(dec,1):>8}")
