"""Phase-A BC prefit trainer + probe harness (spec §7 Phase A, §6.4 probes).

Optimizer per §6.3: AdamW lr 1e-3, wd 1e-4 on matrix weights only (no decay:
biases, LN params, token inits, bias table), AMP + GradScaler, grad-clip 1.0,
500-step linear LR warmup on fresh initializations, no EMA. One optimizer
step per nominal 32-row batch via pair-budget micro-buckets with step-global
denominators (exactly equal to monolithic batches under LN — tested theorem).

Per-epoch artifacts in the run dir: checkpoint, diagnostics.jsonl line, and a
probe npz over ~1024 frozen validation rows (policy churn KL vs the previous
epoch, entropy, value ECE vs frozen realized outcomes, D6-consistency KL).
"""

from __future__ import annotations

import os

# PERF NOTE — HEXFIELD_TRAIN_FLEX is deliberately NOT defaulted on for the prefit.
# Production self-play/training (systemd HEXFIELD_TRAIN_FLEX=1) gets a big win from
# the in-kernel FlexAttention score_mod because its forward runs on a SMALL, STABLE
# set of shapes. The prefit is different: its pair-budget micro-buckets pack a
# VARIABLE number of rows per step (the batch/seq dim swings, e.g. 11 vs 3), which
# the inner flex compile (torch.compile(dynamic=False) in model.py) cannot cache —
# it recompiles per shape and then falls back to the UNFUSED flex path that
# materializes the full scores matrix. MEASURED: enabling train-flex here made the
# prefit ~1.85 s/step vs ~0.33 s/step eager (5.6x SLOWER). So the prefit stays on
# the eager materialized-bias path. (A user can still force flex for experiments via
# HEXFIELD_TRAIN_FLEX=1 in the env; matching production flex speed would instead
# require stabilizing the prefit's batch shapes — a separate, larger change.)

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .batching import (
    PAD_QUANTUM,
    collate_training,
    pair_budget_microbuckets,
    split_stvalue_columns,
    step_global_denominators,
)
from .geometry import apply_d6
from .losses import decode_binned_value, hexfield_loss
from .model import HexfieldNet
from .samples import STV_HORIZONS, expand_sample
from .shards import read_compact_shard

NOMINAL_BATCH_ROWS = 32
WARMUP_STEPS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PROBE_ROWS = 1024


def make_optimizer(model: HexfieldNet) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim >= 2 and "bias_table" not in name and name != "tokens":
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LR,
    )


class BootstrapShards(IterableDataset):
    """Yields ready nominal-step payloads: (list of micro-bucket batches, denoms)."""

    def __init__(self, shard_paths: list[Path], seed: int, epoch: int, train: bool):
        self.shard_paths = list(shard_paths)
        self.seed = seed
        self.epoch = epoch
        self.train = train

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        workers = info.num_workers if info else 1
        rng = random.Random(f"{self.seed}/{self.epoch}/{wid}")
        order = list(self.shard_paths)
        rng.shuffle(order)
        my_shards = order[wid::workers]
        buffer = []
        for path in my_shards:
            buffer.extend(read_compact_shard(path))
            if len(buffer) < 4096 and path is not my_shards[-1]:
                continue
            rng.shuffle(buffer)
            yield from self._emit(buffer, rng)
            buffer = []
        if buffer:
            rng.shuffle(buffer)
            yield from self._emit(buffer, rng)

    def _emit(self, rows, rng):
        for start in range(0, len(rows), NOMINAL_BATCH_ROWS):
            chunk = rows[start : start + NOMINAL_BATCH_ROWS]
            expanded = [
                expand_sample(s, symmetry=rng.randrange(12) if self.train else 0)
                for s in chunk
            ]
            denoms = step_global_denominators(expanded, STV_HORIZONS)
            buckets = []
            for bucket in pair_budget_microbuckets(expanded, quantize=PAD_QUANTUM):
                # Quantize Npad to PAD_QUANTUM-multiples: a small, repeating set
                # of tensor shapes keeps the CUDA caching allocator from
                # fragmenting toward the VRAM ceiling (the §5.3 static-shape
                # discipline, applied to training). The split above assumed this
                # same quantum, so the live (B,4,S,S) transient honours the
                # PAIR_BUDGET (§6.3).
                pad_to = -(-max(r.support.num_nodes for r in bucket) // PAD_QUANTUM) * PAD_QUANTUM
                batch = collate_training(bucket, pad_to=pad_to)
                buckets.append(split_stvalue_columns(batch, STV_HORIZONS))
            yield buckets, denoms


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def run_step(model, buckets, denoms, device, scaler, optimizer, grad_stats) -> dict:
    optimizer.zero_grad(set_to_none=True)
    components_sum: dict[str, float] = {}
    any_backward = False
    for batch in buckets:
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        loss, components = hexfield_loss(out, batch, denominators=denoms)
        if not torch.isfinite(loss):
            # PCR fast-value-row buckets occasionally yield a non-finite cell_q in the
            # OFFLINE expand_sample path (production's rust expand is clean). Skip the
            # bucket instead of crashing the warm-start — cell_q is auxiliary and a rare
            # skipped bucket's policy/value gradient is negligible for a BC prefit. The
            # printed count lets us confirm it stays rare.
            print(f"  [skip] non-finite bucket (cell_q={float(components.get('cell_q', float('nan'))):.3g})", flush=True)
            continue
        scaler.scale(loss).backward()
        any_backward = True
        for key, val in components.items():
            components_sum[key] = components_sum.get(key, 0.0) + float(val.detach())
    if not any_backward:
        # The whole nominal step was non-finite (every bucket skipped) — no grads were
        # scaled, so skip the optimizer update rather than tripping GradScaler's
        # "No inf checks were recorded" assert. Rare (a lone all-fast cell_q bucket).
        return components_sum
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    grad_stats.append(float(grad_norm))
    scaler.step(optimizer)
    scaler.update()
    return components_sum


@torch.no_grad()
def evaluate(model, shard_paths, device, max_rows: int = 50_000) -> dict:
    model.eval()
    top1_hits = 0
    rows_seen = 0
    policy_ce_sum = 0.0
    value_ce_sum = 0.0
    evs: list[float] = []
    zs: list[float] = []
    for path in shard_paths:
        samples = read_compact_shard(path)
        for start in range(0, len(samples), NOMINAL_BATCH_ROWS):
            chunk = samples[start : start + NOMINAL_BATCH_ROWS]
            expanded = [expand_sample(s, symmetry=0) for s in chunk]
            denoms = step_global_denominators(expanded, STV_HORIZONS)
            chunk_policy = 0.0
            chunk_value = 0.0
            chunk_rows = len(expanded)
            # eval collate pads to raw max N (pad_to=None), so the split must
            # use the raw quantum (no 256-rounding) for its budget to match the
            # actual transient here — keeps prior eval bucketing unchanged.
            for bucket in pair_budget_microbuckets(expanded, quantize=1):
                batch = _to_device(
                    split_stvalue_columns(collate_training(bucket), STV_HORIZONS), device
                )
                out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
                _, comps = hexfield_loss(out, batch, denominators=denoms)
                # components use the chunk-global denominator, so summing the
                # buckets reconstructs the chunk-mean CE exactly
                chunk_policy += float(comps["policy"])
                chunk_value += float(comps["value"])
                b = batch["value"].shape[0]
                ev = decode_binned_value(out["value"])
                evs.extend(ev.tolist())
                zs.extend(batch["value"].tolist())
                npad = out["policy"].shape[1]
                prefix = torch.arange(npad, device=device).unsqueeze(0) < batch[
                    "legal_counts"
                ].unsqueeze(1)
                masked_logits = out["policy"].masked_fill(~prefix, float("-inf"))
                pred = masked_logits.argmax(dim=1)
                tgt = batch["policy"].argmax(dim=1)
                top1_hits += int((pred == tgt).sum())
                rows_seen += b
            policy_ce_sum += chunk_policy * chunk_rows
            value_ce_sum += chunk_value * chunk_rows
        if rows_seen >= max_rows:
            break
    evs_arr = np.asarray(evs)
    zs_arr = np.asarray(zs)
    deciles = np.quantile(evs_arr, np.linspace(0, 1, 11))
    ece = 0.0
    for k in range(10):
        lo, hi = deciles[k], deciles[k + 1]
        sel = (evs_arr >= lo) & (evs_arr <= hi if k == 9 else evs_arr < hi)
        if sel.sum() == 0:
            continue
        ece += sel.mean() * abs(evs_arr[sel].mean() - zs_arr[sel].mean())
    model.train()
    return {
        "val_rows": rows_seen,
        "top1": top1_hits / max(rows_seen, 1),
        "policy_ce": policy_ce_sum / max(rows_seen, 1),
        "value_ce": value_ce_sum / max(rows_seen, 1),
        "value_ece": float(ece),
        "value_optimism": float(evs_arr.mean() - zs_arr.mean()),
    }


@torch.no_grad()
def run_probe(model, probe_samples, device, prev_probs: list[np.ndarray] | None):
    """Frozen-probe forward: policy churn KL, entropy, E[v], D6-consistency KL."""

    model.eval()
    probs_out: list[np.ndarray] = []
    entropy: list[float] = []
    evs: list[float] = []
    d6_kl: list[float] = []
    kl_prev: list[float] = []
    for i, sample in enumerate(probe_samples):
        ident = expand_sample(sample, symmetry=0)
        sym = (i % 11) + 1
        rot = expand_sample(sample, symmetry=sym)
        batch = _to_device(collate_training([ident]), device)
        out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        l = ident.support.legal_count
        p = torch.softmax(out["policy"][0, :l].float(), dim=0)
        probs_out.append(p.cpu().numpy())
        entropy.append(float(-(p * (p + 1e-12).log()).sum()))
        evs.append(float(decode_binned_value(out["value"])[0]))
        batch_r = _to_device(collate_training([rot]), device)
        out_r = model(batch_r["feats"], batch_r["nbr"], batch_r["mask"], batch_r["coords"])
        q_rot = torch.softmax(out_r["policy"][0, :l].float(), dim=0).cpu().numpy()
        # map identity slot i -> rotated slot of sigma(cell_i)
        perm = np.empty(l, dtype=np.int64)
        for slot in range(l):
            q_, r_ = ident.support.coords[slot].tolist()
            perm[slot] = rot.support.index[apply_d6(sym, q_, r_)]
        q = q_rot[perm]
        p_np = probs_out[-1]
        d6_kl.append(float(np.sum(p_np * (np.log(p_np + 1e-12) - np.log(q + 1e-12)))))
        if prev_probs is not None:
            prev = prev_probs[i]
            kl_prev.append(float(np.sum(p_np * (np.log(p_np + 1e-12) - np.log(prev + 1e-12)))))
    model.train()
    metrics = {
        "probe_entropy": float(np.mean(entropy)),
        "probe_ev_mean": float(np.mean(evs)),
        "probe_d6_kl": float(np.mean(d6_kl)),
        "probe_policy_kl_prev": float(np.mean(kl_prev)) if kl_prev else None,
    }
    return probs_out, metrics


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, global_step: int):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, scaler=None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dir with train/ and val/ shards")
    parser.add_argument("--out", required=True, help="run dir")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit-steps", type=int, default=0, help="smoke: stop each epoch early")
    parser.add_argument("--resume", default="")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_shards = sorted(Path(args.data, "train").glob("shard_*.npz"))
    val_shards = sorted(Path(args.data, "val").glob("shard_*.npz"))
    if not train_shards or not val_shards:
        raise SystemExit(f"no shards under {args.data}")

    torch.manual_seed(args.seed)
    model = HexfieldNet().to(device)
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    global_step = 0
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(Path(args.resume), model, optimizer, scaler)
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"]) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    # torch.compile the FORWARD (dynamic=True). Measured ~1.45x on this prefit
    # (0.278 -> 0.192 s/step steady) after a ~12s one-time warmup. dynamic=True is
    # essential: the pair-budget micro-buckets vary the batch/seq dim, so a static
    # compile would recompile per shape (that is exactly why HEXFIELD_TRAIN_FLEX,
    # whose inner flex op is dynamic=False, was ~5.6x SLOWER here). Keep `model`
    # (the eager module) for make_optimizer and save_checkpoint — the compiled
    # wrapper prefixes state_dict keys with "_orig_mod." which would break the
    # checkpoint loader. Set HEXFIELD_PREFIT_COMPILE=0 to force plain eager.
    fwd = model
    if os.environ.get("HEXFIELD_PREFIT_COMPILE", "1") == "1":
        try:
            fwd = torch.compile(model, dynamic=True)
        except Exception as exc:  # pragma: no cover - older torch / compile unavailable
            print(f"torch.compile unavailable ({exc}); using eager forward", flush=True)

    # Frozen probe rows: first PROBE_ROWS rows of the (fixed-order) val shards.
    probe_samples = []
    for path in val_shards:
        probe_samples.extend(read_compact_shard(path))
        if len(probe_samples) >= PROBE_ROWS:
            break
    probe_samples = probe_samples[:PROBE_ROWS]
    prev_probs = None

    diag_path = out_dir / "diagnostics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        dataset = BootstrapShards(train_shards, args.seed, epoch, train=True)
        loader = DataLoader(
            dataset, batch_size=None, num_workers=args.workers,
            persistent_workers=False, prefetch_factor=4 if args.workers else None,
        )
        t0 = time.time()
        comp_totals: dict[str, float] = {}
        grad_stats: list[float] = []
        steps = 0
        for buckets, denoms in loader:
            # linear warmup on fresh inits only (resumes skip via global_step)
            if global_step < WARMUP_STEPS:
                lr = LR * (global_step + 1) / WARMUP_STEPS
                for group in optimizer.param_groups:
                    group["lr"] = lr
            comps = run_step(fwd, buckets, denoms, device, scaler, optimizer, grad_stats)
            for key, val in comps.items():
                comp_totals[key] = comp_totals.get(key, 0.0) + val
            steps += 1
            global_step += 1
            if steps % 25 == 0:
                print(f"epoch {epoch} step {steps} total {comp_totals['total']/steps:.4f} "
                      f"({(time.time()-t0)/steps:.3f}s/step)", flush=True)
            if args.limit_steps and steps >= args.limit_steps:
                break

        grads = np.asarray(grad_stats)
        val_metrics = evaluate(fwd, val_shards, device)
        prev_probs, probe_metrics = run_probe(fwd, probe_samples, device, prev_probs)
        record = {
            "epoch": epoch,
            "steps": steps,
            "global_step": global_step,
            "seconds": round(time.time() - t0, 1),
            **{f"train_{k}": v / max(steps, 1) for k, v in comp_totals.items()},
            "grad_norm_mean": float(grads.mean()) if len(grads) else None,
            "grad_norm_p95": float(np.percentile(grads, 95)) if len(grads) else None,
            "clip_fraction": float((grads > GRAD_CLIP).mean()) if len(grads) else None,
            "amp_scale": float(scaler.get_scale()),
            **val_metrics,
            **probe_metrics,
        }
        with open(diag_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        np.savez_compressed(
            out_dir / f"probe_epoch{epoch}.npz",
            probs=np.asarray(prev_probs, dtype=object),
            entropy=probe_metrics["probe_entropy"],
            d6_kl=probe_metrics["probe_d6_kl"],
        )
        save_checkpoint(out_dir / f"checkpoint_epoch{epoch}.pt", model, optimizer, scaler, epoch, global_step)


if __name__ == "__main__":
    main()
