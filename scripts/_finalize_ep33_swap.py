#!/usr/bin/env python
"""Swap the 6e-4-retrained ep33 in and make all records consistent.

- checkpoints/epoch_000033.pt  <- epoch_000033_6e4.pt (the retrained 6e-4 net)
- diagnostics/epoch_000033.json: overwrite ONLY the lr-affected training metrics
  (loss_*, grad_norm_*, clip_*, amp_scale, seconds) with the 6e-4 retrain values;
  keep the lr-independent governor/window/self-play/eval fields as-is.
- diagnostics/config.normalized.json: learning_rate -> 0.0006.
Originals already backed up to pre_6e4_epoch33_lr1e3.{pt,json}.
"""
import json
import os
from pathlib import Path

RUN = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_3")
new = json.load(open("/tmp/retrain_ep33_result.json"))

# 1) checkpoint swap
side = RUN / "checkpoints/epoch_000033_6e4.pt"
dest = RUN / "checkpoints/epoch_000033.pt"
os.replace(side, dest)
print(f"swapped checkpoint <- {side.name}")

# 2) diagnostics: merge lr-affected fields into the original ep33 training subtree
diag_path = RUN / "diagnostics/epoch_000033.json"
diag = json.load(open(RUN / "diagnostics/pre_6e4_epoch33_lr1e3.json"))  # the original full record
tr = diag["metadata"]["result"]["training"]
LR_AFFECTED = [
    "loss_total", "loss_policy", "loss_value", "loss_opp_policy", "loss_cell_q",
    "loss_moves_left", "loss_stvalue_2", "loss_stvalue_6", "loss_stvalue_16",
    "grad_norm_mean", "grad_norm_ema", "grad_norm_p95", "grad_norm_trunk_conv",
    "grad_norm_trunk_attn", "grad_norm_heads", "clip_value_mean", "clip_fraction",
    "amp_scale", "seconds",
]
changed = {}
for k in LR_AFFECTED:
    if k in new:
        changed[k] = (tr.get(k), new[k])
        tr[k] = new[k]
tr["retrained_at_lr"] = 6e-4
tr["retrain_note"] = "ep33 training re-run at lr=6e-4 (was 1e-3); LR micro-sweep showed 1e-3 worst in band"
json.dump(diag, open(diag_path, "w"), indent=2, default=str)
print(f"updated {diag_path.name}: " + ", ".join(f"{k} {a}->{b}" for k, (a, b) in list(changed.items())[:6]) + " ...")

# 3) config.normalized.json learning_rate -> 6e-4
cfg_path = RUN / "diagnostics/config.normalized.json"
if cfg_path.exists():
    cfg = json.load(open(cfg_path))
    n = [0]
    def fix(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "learning_rate" and isinstance(v, (int, float)):
                    o[k] = 6e-4; n[0] += 1
                else:
                    fix(v)
        elif isinstance(o, list):
            for v in o:
                fix(v)
    fix(cfg)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"config.normalized.json: set {n[0]} learning_rate field(s) -> 6e-4")
print("DONE")
