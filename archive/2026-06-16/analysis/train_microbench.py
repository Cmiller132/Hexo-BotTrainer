"""In-situ training-loop micro-benchmark: faithfully replays the production inner
loop (`train_passes` -> `_batch_from_npz` -> `_optimizer_step`) over REAL shuffled
shards on the free GPU, timing decompress vs GPU-step per batch. Sizes the
~250-380 s/epoch training residual the earlier report could not explain.

Read-only. Reads real shards; touches no run state.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import torch

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.losses import model1_loss
from hexo_models.dense_cnn.trainer import _batch_from_npz
from hexo_models.dense_cnn.replay import INPUT_KEY

HORIZONS = (1, 4, 8)
BATCH = 256
DEV = torch.device("cuda")
TARGET_ROWS = 30000  # a few real shards (production target is 100000)


def shards():
    base = r"E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64\shuffleddata"
    gens = sorted(d for d in glob.glob(os.path.join(base, "*-epoch_*")) if not d.endswith(".tmp"))
    for g in reversed(gens):
        s = sorted(glob.glob(os.path.join(g, "train", "*.npz")))
        if s:
            return s
    raise SystemExit("no shards")


def build():
    torch.backends.cudnn.benchmark = True
    m = Model1Network(channels=64, blocks=4, short_term_value_horizons=HORIZONS).to(
        DEV, memory_format=torch.channels_last)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return m, opt, scaler


def opt_step(m, opt, scaler, batch):
    inputs = batch.pop("input").to(DEV, non_blocking=True, memory_format=torch.channels_last)
    batch = {k: v.to(DEV, non_blocking=True) for k, v in batch.items()}
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=True):
        out = m(inputs)
        loss, _ = model1_loss(out, batch, policy_weight=1.0, value_weight=1.0,
                              opp_policy_weight=0.25, short_term_value_weight=0.25)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    return float(loss.detach().cpu().item())


def run(mode: str, files, m, opt, scaler):
    """mode='current' re-indexes per batch (production bug); 'loadonce' decompresses
    each shard's arrays once."""
    # warm GPU
    warm = torch.randn(BATCH, 13, 41, 41, device=DEV).to(memory_format=torch.channels_last)
    for _ in range(15):
        with torch.autocast(device_type="cuda", enabled=True):
            m(warm)
    torch.cuda.synchronize()

    t_decompress = 0.0
    t_step = 0.0
    t_open = 0.0
    rows_done = 0
    steps = 0
    wall0 = time.perf_counter()
    for fp in files:
        if rows_done >= TARGET_ROWS:
            break
        t = time.perf_counter()
        npz = np.load(fp)
        rows = int(npz[INPUT_KEY].shape[0])
        if mode == "loadonce":
            arrays = {k: npz[k] for k in npz.files}

            class Mem:
                def __getitem__(self, k):
                    return arrays[k]
            src = Mem()
        else:
            src = npz
        t_open += time.perf_counter() - t
        off = 0
        while off < rows and rows_done < TARGET_ROWS:
            take = min(BATCH, rows - off, TARGET_ROWS - rows_done)
            t = time.perf_counter()
            batch = _batch_from_npz(src, off, off + take, HORIZONS)
            t_decompress += time.perf_counter() - t
            off += take
            rows_done += take
            t = time.perf_counter()
            opt_step(m, opt, scaler, batch)
            torch.cuda.synchronize()
            t_step += time.perf_counter() - t
            steps += 1
        npz.close()
    wall = time.perf_counter() - wall0
    return {
        "mode": mode, "rows": rows_done, "steps": steps, "wall_s": wall,
        "open_s": t_open, "decompress_s": t_decompress, "gpu_step_s": t_step,
        "wall_per_step_ms": wall / steps * 1000.0,
        "decompress_per_step_ms": t_decompress / steps * 1000.0,
        "gpu_step_per_step_ms": t_step / steps * 1000.0,
        "proj_epoch_391steps_s": wall / steps * 391,
    }


def main():
    files = shards()
    print(f"{len(files)} shards available; using up to {TARGET_ROWS} rows")
    out = {}
    for mode in ("current", "loadonce"):
        m, opt, scaler = build()
        r = run(mode, files, m, opt, scaler)
        out[mode] = r
        print(mode, json.dumps(r, indent=1))
        del m, opt, scaler
        torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "train_microbench_summary.json"),
                        "w", encoding="utf-8"), indent=2)
    print("wrote train_microbench_summary.json")


if __name__ == "__main__":
    main()
