"""Reconcile the training GPU-step cost: why does the in-situ loop show ~480 ms/
step while a static-resident-batch bench showed ~260 ms? Isolates, in ONE process:
  (i)   static resident batch, step repeated (caches hottest)
  (ii)  fresh H2D each step from PRE-decompressed real batches (no decompress in loop)
  (iii) fresh H2D + per-step decompress (full production 'current' loop)
All with cuda.synchronize per step. Read-only."""
from __future__ import annotations

import glob, json, os, time
import numpy as np
import torch

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.losses import model1_loss
from hexo_models.dense_cnn.trainer import _batch_from_npz
from hexo_models.dense_cnn.replay import INPUT_KEY

H = (1, 4, 8)
BS = 256
DEV = torch.device("cuda")


def shard():
    base = r"E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64\shuffleddata"
    g = sorted(d for d in glob.glob(os.path.join(base, "*-epoch_*")) if not d.endswith(".tmp"))[-1]
    return sorted(glob.glob(os.path.join(g, "train", "*.npz")))


def build():
    torch.backends.cudnn.benchmark = True
    m = Model1Network(channels=64, blocks=4, short_term_value_horizons=H).to(DEV, memory_format=torch.channels_last)
    m.train()
    return m, torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4), torch.amp.GradScaler("cuda", enabled=True)


def step(m, opt, scaler, dev_batch):
    b = dict(dev_batch)
    inp = b.pop("input")
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=True):
        out = m(inp)
        loss, _ = model1_loss(out, b, policy_weight=1.0, value_weight=1.0,
                              opp_policy_weight=0.25, short_term_value_weight=0.25)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    return float(loss.detach().cpu().item())


def to_dev(cpu):
    out = {}
    for k, v in cpu.items():
        out[k] = v.to(DEV, non_blocking=True, memory_format=torch.channels_last) if k == "input" else v.to(DEV, non_blocking=True)
    return out


def main():
    files = shard()
    m, opt, scaler = build()
    # pre-decompress ~40 real batches into CPU tensors
    cpu_batches = []
    with np.load(files[0]) as d:
        arrays = {k: d[k] for k in d.files}

        class Mem:
            def __getitem__(self, k):
                return arrays[k]
        mem = Mem()
        rows = arrays[INPUT_KEY].shape[0]
        for off in range(0, min(rows, BS * 40), BS):
            cpu_batches.append(_batch_from_npz(mem, off, min(off + BS, rows), H))
    nb = len(cpu_batches)
    # warm
    for _ in range(20):
        step(m, opt, scaler, to_dev(cpu_batches[0]))
    torch.cuda.synchronize()

    # (i) static resident batch
    dev_static = to_dev(cpu_batches[0])
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(100):
        step(m, opt, scaler, dev_static); torch.cuda.synchronize()
    static_ms = (time.perf_counter() - t0) / 100 * 1000

    # (ii) fresh H2D each step from pre-decompressed CPU batches (cycle through them)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(100):
        dev = to_dev(cpu_batches[i % nb])
        step(m, opt, scaler, dev); torch.cuda.synchronize()
    fresh_h2d_ms = (time.perf_counter() - t0) / 100 * 1000

    # (iii) per-step decompress (re-index, production bug) + H2D + step
    with np.load(files[0]) as d:
        rows = int(d[INPUT_KEY].shape[0])
        torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
        for off in range(0, min(rows, BS * 100), BS):
            cb = _batch_from_npz(d, off, min(off + BS, rows), H)
            dev = to_dev(cb)
            step(m, opt, scaler, dev); torch.cuda.synchronize(); n += 1
        current_ms = (time.perf_counter() - t0) / n * 1000

    out = {
        "static_resident_ms": static_ms,
        "fresh_h2d_predecompressed_ms": fresh_h2d_ms,
        "current_reindex_each_step_ms": current_ms,
        "delta_fresh_minus_static_ms": fresh_h2d_ms - static_ms,
        "delta_current_minus_fresh_ms": current_ms - fresh_h2d_ms,
    }
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "train_step_reconcile_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
