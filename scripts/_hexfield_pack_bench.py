"""Micro-bench: the per-row host-pack loop in _forward_group vs a vectorized
scatter. Checks BIT-IDENTICAL packed buffers and times both, for an early-game
group (many small rows) and a late-game group (few large rows)."""
from __future__ import annotations
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
import numpy as np
import hexfield.constants as C

NBR_SENTINEL = 0xFFFF
F = C.NUM_FEATURES
rng = np.random.default_rng(0)


def make_group(sizes):
    sizes = np.asarray(sorted(sizes, reverse=True), dtype=np.int64)
    total = int(sizes.sum())
    feats = rng.standard_normal((total, F)).astype(np.float32)
    qr = rng.integers(-20, 21, size=(total, 2)).astype(np.int64)
    nbr = rng.integers(0, 1000, size=(total, 6)).astype(np.uint16)
    nbr[rng.random((total, 6)) < 0.2] = NBR_SENTINEL
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    return feats, qr, nbr, offsets, sizes


def pack_loop(feats, qr, nbr, offsets, sizes, pad_to):
    g = len(sizes)
    np_feats = np.zeros((g, pad_to, F), dtype=np.float32)
    np_nbr = np.full((g, pad_to, 6), pad_to, dtype=np.int64)
    np_mask = np.zeros((g, pad_to), dtype=np.bool_)
    np_coords = np.zeros((g, pad_to, 2), dtype=np.int64)
    for k in range(g):
        n = int(sizes[k]); o = int(offsets[k])
        np_feats[k, :n] = feats[o:o + n]
        row_nbr = nbr[o:o + n].astype(np.int64)
        np_nbr[k, :n] = np.where(row_nbr == NBR_SENTINEL, pad_to, row_nbr)
        np_mask[k, :n] = True
        np_coords[k, :n] = qr[o:o + n]
    return np_feats, np_nbr, np_mask, np_coords


def pack_vec(feats, qr, nbr, offsets, sizes, pad_to):
    g = len(sizes)
    total = int(offsets[g])
    np_feats = np.zeros((g, pad_to, F), dtype=np.float32)
    np_nbr = np.full((g, pad_to, 6), pad_to, dtype=np.int64)
    np_mask = np.zeros((g, pad_to), dtype=np.bool_)
    np_coords = np.zeros((g, pad_to, 2), dtype=np.int64)
    k_per = np.repeat(np.arange(g), sizes)
    # position within each row: arange(total) minus each row's start offset
    p_per = np.arange(total) - np.repeat(offsets[:g], sizes)
    dest = k_per * pad_to + p_per
    np_feats.reshape(g * pad_to, F)[dest] = feats[:total]
    np_coords.reshape(g * pad_to, 2)[dest] = qr[:total]
    np_mask.reshape(g * pad_to)[dest] = True
    nbr64 = nbr[:total].astype(np.int64)
    np_nbr.reshape(g * pad_to, 6)[dest] = np.where(nbr64 == NBR_SENTINEL, pad_to, nbr64)
    return np_feats, np_nbr, np_mask, np_coords


def bench(fn, args, reps=200):
    for _ in range(5): fn(*args)
    t0 = time.time()
    for _ in range(reps): r = fn(*args)
    return (time.time() - t0) / reps * 1000, r


for label, sizes, pad_to in [
    ("early g=64 pad=256", list(rng.integers(40, 256, size=64)), 256),
    ("mid g=20 pad=768", list(rng.integers(300, 768, size=20)), 768),
    ("late g=5 pad=3328", list(rng.integers(2000, 3328, size=5)), 3328),
]:
    data = make_group(sizes)
    a = pack_loop(*data, pad_to)
    b = pack_vec(*data, pad_to)
    ident = all(np.array_equal(x, y) for x, y in zip(a, b))
    tl, _ = bench(pack_loop, (*data, pad_to))
    tv, _ = bench(pack_vec, (*data, pad_to))
    print(f"{label:22s} parity={'IDENTICAL' if ident else 'DIFFER'}  loop {tl:.3f} ms | vec {tv:.3f} ms | {tl/tv:.2f}x")
