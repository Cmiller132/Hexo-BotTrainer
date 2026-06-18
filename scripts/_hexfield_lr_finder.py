#!/usr/bin/env python
"""LR range test (Leslie-Smith) for hexfield_main_3.

A single-trajectory sweep: load the current checkpoint weights into a FRESH
AdamW (production param groups), then ramp the learning rate exponentially from
LR_MIN to LR_MAX over N optimizer steps, recording the per-step training loss.
The loss-vs-lr curve reveals the usable lr band: loss is flat/noisy at tiny lr,
descends through the productive band, bottoms out, then diverges. The classic
pick is ~the steepest-descent lr (and well below the divergence knee).

Faithfulness: reuses the EXACT production train-step internals — the same
window selection, D6 expand, pair-budget micro-bucketing, step-global
denominators + policy-surprise weights, and hexfield_loss with the config
weights. Two deliberate deviations, standard for an LR range test and both
documented inline: (1) fp32 (no AMP/GradScaler) so every lr is actually applied
and the curve is clean — the optimal-lr LOCATION is AMP-invariant (the train
path is parity-equivalent to flex per the deploy verify); (2) NO grad clipping,
so the true divergence knee is visible instead of being masked by the clip.

Run with PYTHONPATH=<root>/packages/hexfield/python and the build venv python.
Reads only; never writes a checkpoint.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

# materialized bias path (no flex compile churn across the many shapes this
# sweep touches); numerically parity-equivalent to the flex train path, so the
# loss-vs-lr curve is identical. Set BEFORE importing model.
os.environ.setdefault("HEXFIELD_TRAIN_FLEX", "0")
os.environ.setdefault("HEXFIELD_SUPPORT_RADIUS", "4")

from hexfield.batching import (  # noqa: E402
    PAD_QUANTUM,
    collate_training,
    pair_budget_microbuckets,
    policy_surprise_weights,
    split_stvalue_columns,
    step_global_denominators,
)
from hexfield.buffer_manifest import scan_or_update_manifest  # noqa: E402
from hexfield.checkpoints import load_into  # noqa: E402
from hexfield.config import parse_hexfield_config  # noqa: E402
from hexfield.expand_backends import expand_rows  # noqa: E402
from hexfield.losses import hexfield_loss  # noqa: E402
from hexfield.model import HexfieldNet  # noqa: E402
from hexfield.samples import STV_HORIZONS  # noqa: E402
from hexfield.window import (  # noqa: E402
    _split_by_md5,
    build_window_split,
    compute_katago_window_rows,
    keep_prob as _keep_prob,
    select_recent_window,
)

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11
    import tomli as tomllib  # type: ignore

# ----- config -----------------------------------------------------------------
ROOT = "/mnt/e/Hexo-BotTrainer-hexgt"
RUNDIR = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_3"
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{RUNDIR}/checkpoints/epoch_000031.pt"
CONFIG = f"{ROOT}/configs/hexfield_main_3.toml"
SAMPLES_DIR = __import__("pathlib").Path(f"{RUNDIR}/samples")

N_STEPS = 150
LR_MIN = 1e-6
LR_MAX = 1.5e-1
SEED = 1
SMOOTH = 0.92  # EMA over per-step loss
DIVERGE_FACTOR = 4.0

dev = torch.device("cuda")
torch.manual_seed(SEED)

# ----- config + model + fresh optimizer (production param groups) -------------
with open(CONFIG, "rb") as fh:
    toml = tomllib.load(fh)
cfg = parse_hexfield_config(toml["model"]["config"])
tcfg = cfg.training

model = HexfieldNet().to(dev)
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
load_into(model, payload, optimizer=None)  # weights only -> fresh Adam moments
model.train()

decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim >= 2 and "bias_table" not in name and name != "tokens":
        decay.append(p)
    else:
        no_decay.append(p)
opt = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": tcfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=LR_MIN,
)
print(f"loaded {CKPT}  epoch={payload.get('meta',{}).get('epoch')}", flush=True)

# ----- build a representative window (production selection) --------------------
manifest = scan_or_update_manifest(SAMPLES_DIR)
total_rows = int(manifest.total_rows)
desired = max(
    int(
        compute_katago_window_rows(
            total_rows,
            min_rows=tcfg.shuffle_min_rows,
            expand_window_per_row=tcfg.shuffle_expand_window_per_row,
            taper_window_exponent=tcfg.shuffle_taper_window_exponent,
            taper_window_scale=tcfg.shuffle_taper_window_scale,
        )
    ),
    int(tcfg.shuffle_min_rows),
)
selected_window, used = select_recent_window(manifest.entries, desired)
kp = _keep_prob(used, int(tcfg.shuffle_keep_target_rows))
train_entries, _ = _split_by_md5(selected_window, validation_fraction=0.0)
window = build_window_split(
    train_entries, keep_prob=kp, rng=np.random.default_rng(SEED), samples_dir=SAMPLES_DIR
)
print(f"window: total_rows={total_rows} desired={desired} used={used} "
      f"keep_prob={kp:.3f} window_rows={window.n}", flush=True)

# expand only the rows we need (N_STEPS nominal batches of batch_rows), sampled
# from the window; request a margin for off-legal (radius-4) drops.
batch_rows = int(tcfg.batch_rows)
need = N_STEPS * batch_rows
margin = int(need * 1.4) + batch_rows
idx_rng = np.random.default_rng(SEED + 777)
pick = idx_rng.permutation(window.n)[: min(window.n, margin)]
d6 = np.random.default_rng(SEED + 13).integers(0, 12, size=len(pick), dtype=np.int64)
expanded, valid = expand_rows(
    window, pick, d6, STV_HORIZONS,
    tolerate_off_legal=True, backend="serial", workers=0, pool=None,
)
survivors = [r for r, ok in zip(expanded, valid) if ok]
print(f"expanded {len(pick)} -> {len(survivors)} survivors "
      f"(off-legal dropped {len(pick)-len(survivors)})", flush=True)
if len(survivors) < need:
    print(f"WARN: only {len(survivors)} survivors for {need} needed; "
          f"reducing steps", flush=True)
    N_STEPS = len(survivors) // batch_rows

# ----- the sweep --------------------------------------------------------------
lrs = [LR_MIN * (LR_MAX / LR_MIN) ** (i / (N_STEPS - 1)) for i in range(N_STEPS)]
hist = []
smoothed = None
best = math.inf
print(f"\n{'step':>4} {'lr':>11} {'loss':>9} {'smooth':>9} {'pol':>7} {'val':>7}", flush=True)
for step in range(N_STEPS):
    lr = lrs[step]
    for g in opt.param_groups:
        g["lr"] = lr
    batch_rows_list = survivors[step * batch_rows : (step + 1) * batch_rows]
    if len(batch_rows_list) < batch_rows:
        break
    denoms = step_global_denominators(
        batch_rows_list, STV_HORIZONS,
        policy_surprise_uniform_fraction=tcfg.policy_surprise_uniform_fraction,
        policy_surprise_max_weight=tcfg.policy_surprise_max_weight,
    )
    sw, _ = policy_surprise_weights(
        [r.policy_surprise for r in batch_rows_list],
        tcfg.policy_surprise_uniform_fraction, tcfg.policy_surprise_max_weight,
    )
    w_by_row = {id(r): w for r, w in zip(batch_rows_list, sw)}
    opt.zero_grad(set_to_none=True)
    step_loss = 0.0
    comp_acc: dict[str, float] = {}
    for bucket in pair_budget_microbuckets(batch_rows_list, quantize=PAD_QUANTUM):
        pad_to = -(-max(r.support.num_nodes for r in bucket) // PAD_QUANTUM) * PAD_QUANTUM
        batch = split_stvalue_columns(
            collate_training(bucket, pad_to=pad_to,
                             row_weights=[w_by_row[id(r)] for r in bucket]),
            STV_HORIZONS,
        )
        batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        loss, comps = hexfield_loss(
            out, batch,
            policy_weight=tcfg.policy_weight, value_weight=tcfg.value_weight,
            opp_policy_weight=tcfg.opp_policy_weight,
            short_term_value_weight=tcfg.short_term_value_weight,
            moves_left_weight=tcfg.moves_left_weight,
            q_head_weight=tcfg.q_head_weight, denominators=denoms,
        )
        loss.backward()  # fp32, no scaler
        step_loss += float(loss.detach())
        for k, v in comps.items():
            comp_acc[k] = comp_acc.get(k, 0.0) + float(v.detach())
    # NO grad clip (range test wants the true divergence knee)
    opt.step()

    finite = math.isfinite(step_loss)
    smoothed = step_loss if smoothed is None else SMOOTH * smoothed + (1 - SMOOTH) * step_loss
    best = min(best, smoothed) if finite else best
    hist.append({"step": step, "lr": lr, "loss": step_loss, "smooth": smoothed,
                 "pol": comp_acc.get("loss_policy"), "val": comp_acc.get("loss_value")})
    print(f"{step:>4} {lr:>11.3e} {step_loss:>9.4f} {smoothed:>9.4f} "
          f"{comp_acc.get('loss_policy', float('nan')):>7.3f} "
          f"{comp_acc.get('loss_value', float('nan')):>7.3f}", flush=True)
    if not finite or (smoothed > DIVERGE_FACTOR * best and step > 10):
        print(f"--- diverged at step {step} lr={lr:.3e} ---", flush=True)
        break

# ----- analysis ---------------------------------------------------------------
fin = [h for h in hist if math.isfinite(h["smooth"])]
min_h = min(fin, key=lambda h: h["smooth"])
# steepest descent: most-negative d(smooth)/d(log10 lr)
steep = None
steep_slope = 0.0
for a, b in zip(fin, fin[1:]):
    dll = math.log10(b["lr"]) - math.log10(a["lr"])
    if dll <= 0:
        continue
    slope = (b["smooth"] - a["smooth"]) / dll
    if slope < steep_slope:
        steep_slope = slope
        steep = b
out = {
    "checkpoint": CKPT,
    "n_steps_run": len(hist),
    "current_lr": tcfg.learning_rate,
    "min_loss": {"lr": min_h["lr"], "smooth": min_h["smooth"]},
    "steepest_descent_lr": (steep["lr"] if steep else None),
    "diverged": (not math.isfinite(hist[-1]["smooth"])) or (hist[-1]["smooth"] > DIVERGE_FACTOR * best),
    "diverge_lr": hist[-1]["lr"],
    "history": hist,
}
with open("/tmp/lr_finder_result.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\n===== SUMMARY =====", flush=True)
print(f"current production lr : {tcfg.learning_rate:.3e}", flush=True)
print(f"min-loss lr           : {min_h['lr']:.3e}  (smooth loss {min_h['smooth']:.4f})", flush=True)
if steep:
    print(f"steepest-descent lr   : {steep['lr']:.3e}  (the productive-band center)", flush=True)
print(f"divergence near       : {hist[-1]['lr']:.3e}", flush=True)
print("wrote /tmp/lr_finder_result.json", flush=True)
