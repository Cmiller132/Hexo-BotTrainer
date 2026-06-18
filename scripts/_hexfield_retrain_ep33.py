#!/usr/bin/env python
"""Faithfully re-train epoch 33 of hexfield_main_3 at lr=6e-4 (was 1e-3).

ep33's training step ran at lr=1e-3, which the held-out LR micro-sweep showed is
the worst choice in [5e-4,1e-3]. This recreates ep33 as if it had been trained at
6e-4, changing ONLY the learning rate:

  - start from epoch_000032.pt (weights + WARM AdamW moments) exactly as the live
    run continued into ep33;
  - rebuild ep33's EXACT training window by replicating select_training_samples
    with the production seeds (run.seed=1, epoch=33) against the unchanged buffer
    (ep33's self-play games are still present), so the data + order + D6 are
    identical to the original ep33 -- only the optimizer lr is 6e-4;
  - run the REAL HexfieldTrainer.train_passes (same AMP + adaptive clip + loss),
    so the resulting checkpoint + diagnostics are production-faithful;
  - save to a SIDE file (epoch_000033_6e4.pt) and dump metrics -- the caller
    verifies, then swaps it in and updates the record files.

Materialized bias path (HEXFIELD_TRAIN_FLEX=0): parity-equivalent to the live
flex train path per the deploy verify, far cheaper, identical lr behaviour.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HEXFIELD_TRAIN_FLEX", "0")
os.environ.setdefault("HEXFIELD_SUPPORT_RADIUS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hexfield.buffer_manifest import scan_or_update_manifest  # noqa: E402
from hexfield.checkpoints import load_into, save_checkpoint  # noqa: E402
from hexfield.config import parse_hexfield_config  # noqa: E402
from hexfield.model import HexfieldNet  # noqa: E402
from hexfield.train_state import HexfieldTrainState  # noqa: E402
from hexfield.trainer import HexfieldTrainer  # noqa: E402
from hexfield.window import (  # noqa: E402
    _select_files_for_rows, _split_by_md5, build_window_split,
    compute_katago_window_rows, keep_prob as _keep_prob, select_recent_window,
)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

ROOT = "/mnt/e/Hexo-BotTrainer-hexgt"
RUNDIR = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_3"
CONFIG = f"{ROOT}/configs/hexfield_main_3.toml"
SAMPLES_DIR = Path(f"{RUNDIR}/samples")
CKPT_IN = f"{RUNDIR}/checkpoints/epoch_000032.pt"
CKPT_OUT = f"{RUNDIR}/checkpoints/epoch_000033_6e4.pt"
EPOCH = 33
NEW_LR = 6e-4
DEV = torch.device("cuda")

with open(CONFIG, "rb") as fh:
    toml = tomllib.load(fh)
cfg = parse_hexfield_config(toml["model"]["config"])
tcfg = cfg.training
SEED = int(toml["run"]["seed"])
RUN_NAME = toml["run"]["name"]

# ----- model + optimizer (production param groups) loaded from ep32 (WARM) -----
model = HexfieldNet().to(DEV)
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (decay if (p.ndim >= 2 and "bias_table" not in name and name != "tokens") else no_decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg.weight_decay},
                         {"params": no_decay, "weight_decay": 0.0}], lr=NEW_LR)
trainer = HexfieldTrainer(model=model, config=cfg, optimizer=opt)

payload = torch.load(open(CKPT_IN, "rb"), map_location="cpu", weights_only=False)
load_into(model, payload, optimizer=opt)            # weights + WARM Adam moments
for st in opt.state.values():
    for k, v in st.items():
        if isinstance(v, torch.Tensor) and v.device != DEV:
            st[k] = v.to(DEV)
for g in opt.param_groups:                          # lr 1e-3 (ep32 saved) -> 6e-4
    g["lr"] = NEW_LR
trainer.train_state = HexfieldTrainState.from_dict(payload.get("meta", {}).get("train_state"))
print(f"loaded ep32 (warm Adam); lr set to {NEW_LR}", flush=True)

# ----- rebuild ep33's EXACT window: replicate select_training_samples(epoch=33) -
manifest = scan_or_update_manifest(SAMPLES_DIR)
total_rows = int(manifest.total_rows)
desired = max(int(compute_katago_window_rows(
    total_rows, min_rows=tcfg.shuffle_min_rows,
    expand_window_per_row=tcfg.shuffle_expand_window_per_row,
    taper_window_exponent=tcfg.shuffle_taper_window_exponent,
    taper_window_scale=tcfg.shuffle_taper_window_scale)), int(tcfg.shuffle_min_rows))
selected_window, used = select_recent_window(manifest.entries, desired)
kp = _keep_prob(used, int(tcfg.shuffle_keep_target_rows))
train_entries, _ = _split_by_md5(selected_window, validation_fraction=float(tcfg.validation_fraction))
sel_rng = np.random.default_rng(SEED + EPOCH * 65_537)
selected_files, selected_rows = _select_files_for_rows(
    train_entries, int(tcfg.train_samples_per_epoch), sel_rng)
effective_rows = min(int(tcfg.train_samples_per_epoch), int(selected_rows))
keep_rng = np.random.default_rng(SEED + EPOCH)
window = build_window_split(selected_files, keep_prob=kp, rng=keep_rng, samples_dir=SAMPLES_DIR)
print(f"window: total_rows={total_rows} desired={desired} used={used} keep_prob={kp:.3f} "
      f"selected_rows={selected_rows} effective_rows={effective_rows} window_rows={window.n}", flush=True)

# faithful governor debit (mimic select_training_samples) so saved train_state matches
trainer.train_state.train_bucket_level = max(0.0, trainer.train_state.train_bucket_level - effective_rows)
trainer.train_state.train_steps_since_last_reload += 1
trainer._last_select[EPOCH] = {
    "effective_rows": int(effective_rows), "window_start": 0, "reuse_ratio": 0.0,
    "train_bucket_level": float(trainer.train_state.train_bucket_level),
}

# ----- run the REAL train step (seed=1, epoch=33 => identical aug/perm to live) -
ctx = SimpleNamespace(
    config=SimpleNamespace(run=SimpleNamespace(seed=SEED)),
    diagnostics_dir=Path("/tmp"),
)
result = trainer.train_passes(
    passes=1, sample_window=window, sample_symmetries=None, ctx=ctx, components=None, epoch=EPOCH)
print("\n===== ep33 RETRAIN @ 6e-4 result =====", flush=True)
print(json.dumps(result, indent=2), flush=True)

# ----- save side checkpoint + dump metrics ------------------------------------
extra = {"run": RUN_NAME, "train_state": trainer.train_state.to_dict()}
save_checkpoint(Path(CKPT_OUT), model=model, optimizer=opt, epoch=EPOCH, extra=extra)
with open("/tmp/retrain_ep33_result.json", "w") as fh:
    json.dump(result, fh, indent=2, default=str)
print(f"\nsaved {CKPT_OUT}", flush=True)
print("wrote /tmp/retrain_ep33_result.json", flush=True)
