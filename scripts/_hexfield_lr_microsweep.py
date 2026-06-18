#!/usr/bin/env python
"""Replicated, held-out LR micro-sweep for hexfield_main_3 in [5e-4, 1e-3].

The single-trajectory LR range test is too noisy to resolve the ideal lr within
a 2x band (5e-4..1e-3 differ by ~1% train loss, inside the per-step batch
noise). This experiment removes the three confounds:

  1. BATCH-OVERFIT  -> measure improvement on a FIXED held-out slice that no
     trial ever trains on (not the training batches), so a hot lr that just
     memorizes its batch does not look better.
  2. SEED / DATA NOISE -> replicate each lr across N seeds (different training
     subset + order) and report mean +/- stderr.
  3. FRESH-vs-WARM optimizer -> load the PRODUCTION optimizer state (warm Adam
     moments) so the measured optimum is for THIS run's continuation, not a
     generic from-scratch finder.

Each trial: rebuild model+AdamW, load the ep32 checkpoint (weights + warm Adam),
set the candidate lr, train STEPS optimizer steps on a seed-specific draw from a
shared expanded pool using the EXACT production step (AMP fp16 + GradScaler +
adaptive grad-clip + step-global denominators + policy-surprise weights +
hexfield_loss), and record the held-out loss before/after. The metric is the
held-out loss IMPROVEMENT per lr (mean +/- stderr over seeds).

Materialized bias path (HEXFIELD_TRAIN_FLEX=0): numerically parity-equivalent to
the live flex train path, far cheaper across the many shapes a sweep touches, so
the lr ranking is identical and the run is fast. Reads only; writes no checkpoint.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

os.environ.setdefault("HEXFIELD_TRAIN_FLEX", "0")
os.environ.setdefault("HEXFIELD_SUPPORT_RADIUS", "4")

from hexfield.batching import (  # noqa: E402
    PAD_QUANTUM, collate_training, pair_budget_microbuckets,
    policy_surprise_weights, split_stvalue_columns, step_global_denominators,
)
from hexfield.buffer_manifest import scan_or_update_manifest  # noqa: E402
from hexfield.checkpoints import load_into  # noqa: E402
from hexfield.config import parse_hexfield_config  # noqa: E402
from hexfield.expand_backends import expand_rows  # noqa: E402
from hexfield.losses import hexfield_loss  # noqa: E402
from hexfield.model import HexfieldNet  # noqa: E402
from hexfield.samples import STV_HORIZONS  # noqa: E402
from hexfield.window import (  # noqa: E402
    _split_by_md5, build_window_split, compute_katago_window_rows,
    keep_prob as _keep_prob, select_recent_window,
)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore
from pathlib import Path

# ----- experiment config ------------------------------------------------------
ROOT = "/mnt/e/Hexo-BotTrainer-hexgt"
RUNDIR = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_3"
CKPT = f"{RUNDIR}/checkpoints/epoch_000033.pt"
CONFIG = f"{ROOT}/configs/hexfield_main_3.toml"
SAMPLES_DIR = Path(f"{RUNDIR}/samples")

LRS = [5.0e-4, 6.0e-4, 7.1e-4, 8.4e-4, 1.0e-3]   # log-spaced across the band
SEEDS = [0, 1, 2]
STEPS = 300                      # optimizer steps per trial (~20% of a real epoch)
HOLDOUT_ROWS = 2500              # fixed eval slice, never trained on
POOL_MARGIN = 1.5                # expand this * (max train rows needed) for seed draws
DEV = torch.device("cuda")

with open(CONFIG, "rb") as fh:
    toml = tomllib.load(fh)
cfg = parse_hexfield_config(toml["model"]["config"])
tcfg = cfg.training
BATCH = int(tcfg.batch_rows)

# ----- build the window + expand a shared pool (once) -------------------------
manifest = scan_or_update_manifest(SAMPLES_DIR)
total_rows = int(manifest.total_rows)
desired = max(int(compute_katago_window_rows(
    total_rows, min_rows=tcfg.shuffle_min_rows,
    expand_window_per_row=tcfg.shuffle_expand_window_per_row,
    taper_window_exponent=tcfg.shuffle_taper_window_exponent,
    taper_window_scale=tcfg.shuffle_taper_window_scale)), int(tcfg.shuffle_min_rows))
selected_window, used = select_recent_window(manifest.entries, desired)
kp = _keep_prob(used, int(tcfg.shuffle_keep_target_rows))
train_entries, _ = _split_by_md5(selected_window, validation_fraction=0.0)
window = build_window_split(train_entries, keep_prob=kp,
                            rng=np.random.default_rng(12345), samples_dir=SAMPLES_DIR)
print(f"window_rows={window.n} total_rows={total_rows} keep_prob={kp:.3f}", flush=True)

need_train = int(STEPS * BATCH * POOL_MARGIN)
n_expand = min(window.n, need_train + HOLDOUT_ROWS + BATCH)
pick = np.random.default_rng(999).permutation(window.n)[:n_expand]
d6 = np.random.default_rng(31).integers(0, 12, size=len(pick), dtype=np.int64)
expanded, valid = expand_rows(window, pick, d6, STV_HORIZONS,
                              tolerate_off_legal=True, backend="serial", workers=0, pool=None)
surv = [r for r, ok in zip(expanded, valid) if ok]
print(f"expanded {len(pick)} -> {len(surv)} survivors", flush=True)
holdout = surv[:HOLDOUT_ROWS]
train_pool = surv[HOLDOUT_ROWS:]
print(f"holdout={len(holdout)} train_pool={len(train_pool)} "
      f"(need {STEPS*BATCH}/trial)", flush=True)


def make_batches(rows):
    """Yield nominal batches of BATCH rows -> production microbucket/collate/loss inputs."""
    for s in range(0, len(rows), BATCH):
        chunk = rows[s:s + BATCH]
        if len(chunk) < BATCH:
            break
        denoms = step_global_denominators(
            chunk, STV_HORIZONS,
            policy_surprise_uniform_fraction=tcfg.policy_surprise_uniform_fraction,
            policy_surprise_max_weight=tcfg.policy_surprise_max_weight)
        sw, _ = policy_surprise_weights([r.policy_surprise for r in chunk],
                                        tcfg.policy_surprise_uniform_fraction,
                                        tcfg.policy_surprise_max_weight)
        wbr = {id(r): w for r, w in zip(chunk, sw)}
        buckets = []
        for bucket in pair_budget_microbuckets(chunk, quantize=PAD_QUANTUM):
            pad_to = -(-max(r.support.num_nodes for r in bucket) // PAD_QUANTUM) * PAD_QUANTUM
            batch = split_stvalue_columns(
                collate_training(bucket, pad_to=pad_to,
                                 row_weights=[wbr[id(r)] for r in bucket]), STV_HORIZONS)
            batch = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            buckets.append(batch)
        yield denoms, buckets


def loss_of(model, batch, denoms):
    out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    loss, comps = hexfield_loss(
        out, batch, policy_weight=tcfg.policy_weight, value_weight=tcfg.value_weight,
        opp_policy_weight=tcfg.opp_policy_weight,
        short_term_value_weight=tcfg.short_term_value_weight,
        moves_left_weight=tcfg.moves_left_weight,
        q_head_weight=tcfg.q_head_weight, denominators=denoms)
    return loss, comps


# Precompute held-out batches ONCE (same slice for every eval).
holdout_batches = list(make_batches(holdout))


@torch.no_grad()
def eval_holdout(model):
    model.eval()
    tot, n = 0.0, 0
    for denoms, buckets in holdout_batches:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for b in buckets:
                l, _ = loss_of(model, b, denoms)
                tot += float(l); n += 1
    return tot / max(1, n)


def build_warm():
    """Fresh model+AdamW with production param groups, loaded with ep32 weights+warm Adam."""
    model = HexfieldNet().to(DEV)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if (p.ndim >= 2 and "bias_table" not in name and name != "tokens") else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tcfg.learning_rate)
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    load_into(model, payload, optimizer=opt)         # weights + WARM Adam moments
    for st in opt.state.values():                    # move Adam state to GPU
        for k, v in st.items():
            if isinstance(v, torch.Tensor) and v.device != DEV:
                st[k] = v.to(DEV)
    return model, opt


def run_trial(lr, seed):
    model, opt = build_warm()
    for g in opt.param_groups:
        g["lr"] = lr
    scaler = torch.amp.GradScaler(enabled=True)
    grad_ema = None
    # seed-specific training draw from the shared pool
    idx = np.random.default_rng(1000 + seed).permutation(len(train_pool))[:STEPS * BATCH]
    rows = [train_pool[int(i)] for i in idx]
    before = eval_holdout(model)
    model.train()
    last_losses = []
    for step, (denoms, buckets) in enumerate(make_batches(rows)):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for b in buckets:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, _ = loss_of(model, b, denoms)
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())
        scaler.unscale_(opt)
        # production adaptive clip
        if not tcfg.adaptive_clip or step < tcfg.clip_warmup_steps or grad_ema is None:
            clip_v = float(tcfg.grad_clip)
        else:
            clip_v = float(tcfg.clip_c) * float(grad_ema)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_v)
        if torch.isfinite(norm):
            d = float(tcfg.clip_ema_decay)
            grad_ema = float(norm) if grad_ema is None else d * grad_ema + (1 - d) * float(norm)
        scaler.step(opt); scaler.update()
        if step >= STEPS - 50:
            last_losses.append(step_loss)
    after = eval_holdout(model)
    return {"lr": lr, "seed": seed, "holdout_before": before, "holdout_after": after,
            "holdout_delta": before - after, "train_loss_tail": float(np.mean(last_losses))}


# ----- run all trials sequentially (single GPU) -------------------------------
results = []
t0 = time.time()
before0 = None
for lr in LRS:
    for seed in SEEDS:
        r = run_trial(lr, seed)
        before0 = r["holdout_before"] if before0 is None else before0
        results.append(r)
        print(f"lr={lr:.2e} seed={seed} holdout {r['holdout_before']:.4f}->{r['holdout_after']:.4f} "
              f"delta={r['holdout_delta']:+.4f} train_tail={r['train_loss_tail']:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

# ----- aggregate --------------------------------------------------------------
agg = {}
for lr in LRS:
    ds = [r["holdout_delta"] for r in results if r["lr"] == lr]
    ts = [r["train_loss_tail"] for r in results if r["lr"] == lr]
    mean = float(np.mean(ds)); sd = float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
    agg[f"{lr:.2e}"] = {"lr": lr, "n": len(ds), "holdout_delta_mean": mean,
                        "holdout_delta_stderr": sd / math.sqrt(max(1, len(ds))),
                        "train_tail_mean": float(np.mean(ts)),
                        "overfit_gap": float(np.mean(ts)) - (before0 - mean) if before0 else None}
out = {"checkpoint": CKPT, "steps": STEPS, "seeds": SEEDS, "lrs": LRS,
       "holdout_rows": len(holdout), "holdout_before_ref": before0,
       "trials": results, "aggregate": agg}
with open("/tmp/lr_microsweep_result.json", "w") as fh:
    json.dump(out, fh, indent=2)

print("\n===== AGGREGATE (held-out loss improvement; higher = better) =====", flush=True)
print(f"{'lr':>10} {'n':>3} {'delta_mean':>12} {'stderr':>9} {'train_tail':>11}", flush=True)
for lr in LRS:
    a = agg[f"{lr:.2e}"]
    print(f"{lr:>10.2e} {a['n']:>3} {a['holdout_delta_mean']:>12.4f} "
          f"{a['holdout_delta_stderr']:>9.4f} {a['train_tail_mean']:>11.4f}", flush=True)
best = max(LRS, key=lambda l: agg[f"{l:.2e}"]["holdout_delta_mean"])
print(f"\nbest held-out improvement at lr = {best:.2e}", flush=True)
print("wrote /tmp/lr_microsweep_result.json", flush=True)
