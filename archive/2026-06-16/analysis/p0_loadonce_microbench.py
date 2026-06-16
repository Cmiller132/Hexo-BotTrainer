"""P0 verification: NPZ load-once is byte-identical and much faster.

Compares the OLD data path (re-index ``NpzFile[KEY][start:stop]`` per batch,
which re-decompresses the whole array every access) against the NEW path
(materialize each array once per shard, then slice in RAM). Asserts the produced
batches are bit-identical and reports the wall-clock for one full pass over a
real shuffled train shard at the configured training batch size.

Usage: python analysis/p0_loadonce_microbench.py [shard.npz] [batch_size]
"""

from __future__ import annotations

import glob
import sys
from time import perf_counter

import numpy as np
import torch

from hexo_models.dense_cnn.trainer import _batch_from_npz, _materialize_npz, _NPZ_BATCH_KEYS
from hexo_models.dense_cnn.replay import INPUT_KEY

HORIZONS = (1, 4, 8)


def _find_shard() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    hits = sorted(
        glob.glob("runs/dense_cnn_model1_scratch_64/shuffleddata/*/train/*.npz")
    )
    if not hits:
        raise SystemExit("no shuffled train shard found; pass a path explicitly")
    return hits[0]


def main() -> None:
    shard = _find_shard()
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    with np.load(shard) as data:
        rows = int(data[INPUT_KEY].shape[0])
        sizes = {k: data[k].shape for k in _NPZ_BATCH_KEYS}
    print(f"shard={shard}")
    print(f"rows={rows} batch_size={batch_size} key_shapes={sizes}")

    # --- equivalence: OLD per-batch indexing vs NEW materialize-once ---
    mism = 0
    with np.load(shard) as data:
        arrays = _materialize_npz(data)
        for off in range(0, rows, batch_size):
            stop = min(off + batch_size, rows)
            old = _batch_from_npz(data, off, stop, HORIZONS)      # NpzFile path
            new = _batch_from_npz(arrays, off, stop, HORIZONS)    # in-RAM dict
            assert old.keys() == new.keys()
            for k in old:
                if not torch.equal(old[k], new[k]):
                    mism += 1
    print(f"equivalence: {'IDENTICAL' if mism == 0 else f'{mism} MISMATCHED batches'}")
    assert mism == 0, "P0 changed numerics!"

    # --- timing: OLD (re-decompress per batch) ---
    t0 = perf_counter()
    with np.load(shard) as data:
        for off in range(0, rows, batch_size):
            _batch_from_npz(data, off, min(off + batch_size, rows), HORIZONS)
    old_s = perf_counter() - t0

    # --- timing: NEW (materialize once, slice in RAM) ---
    t0 = perf_counter()
    with np.load(shard) as data:
        arrays = _materialize_npz(data)
    for off in range(0, rows, batch_size):
        _batch_from_npz(arrays, off, min(off + batch_size, rows), HORIZONS)
    new_s = perf_counter() - t0

    n_batches = (rows + batch_size - 1) // batch_size
    print(f"OLD per-batch decompress : {old_s*1000:8.1f} ms  ({old_s/n_batches*1000:.2f} ms/batch)")
    print(f"NEW load-once + slice    : {new_s*1000:8.1f} ms  ({new_s/n_batches*1000:.2f} ms/batch)")
    print(f"speedup                  : {old_s/max(new_s,1e-9):6.1f}x  ({n_batches} batches)")


if __name__ == "__main__":
    main()
