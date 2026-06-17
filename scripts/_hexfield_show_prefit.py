"""Print compact per-epoch prefit metrics from diagnostics.jsonl."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
path = REPO / "runs" / "hexfield_bc_1" / "diagnostics.jsonl"
keys = ("epoch", "top1", "value_ece", "value_optimism", "train_total",
        "train_policy", "train_value", "probe_d6_kl", "probe_policy_kl_prev")
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    print({k: (round(d[k], 4) if isinstance(d.get(k), float) else d.get(k)) for k in keys})
