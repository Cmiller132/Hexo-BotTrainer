"""Isolate the Rust featurize+collate path and measure its real parallelism.

Calls `rust_bridge.hexgt_featurize_states` (the GIL-released featurize+collate)
on a fixed batch in a tight loop while sampling per-core CPU, to answer: does the
`.par_iter()` actually spread across the 32 threads, or is it effectively serial
(serial `collate()` dominating, or the rayon pool not engaging)?

Usage: python scripts/_feat_probe.py --shards <dir> --states 1024 --iters 20
"""
from __future__ import annotations

import argparse
import os
import random
import threading
import time
from pathlib import Path


class CoreSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.cores_busy = []
        self.util = []
        self.ncpu = 0

    @staticmethod
    def _read():
        out = {}
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu"):
                    break
                p = line.split()
                vals = list(map(int, p[1:]))
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                out[p[0]] = (sum(vals), idle)
        return out

    def run(self):
        prev = self._read()
        while not self.stop_flag.wait(self.interval):
            cur = self._read()
            tt, ti = cur["cpu"]; pt, pi = prev["cpu"]
            if tt > pt:
                self.util.append(100 * (1 - (ti - pi) / (tt - pt)))
            busy = 0; n = 0
            for k in cur:
                if k == "cpu":
                    continue
                n += 1
                ct, ci = cur[k]; ptk, pik = prev.get(k, (ct, ci))
                d = ct - ptk
                if d > 0 and (1 - (ci - pik) / d) > 0.5:
                    busy += 1
            self.ncpu = n
            self.cores_busy.append(busy)
            prev = cur

    def stop(self):
        self.stop_flag.set(); self.join(timeout=2)


def load_states(n, shard_dir, seed=1):
    import hexo_engine.api as eng
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.expand import reconstruct_state
    rng = random.Random(seed)
    npzs = sorted(Path(shard_dir).glob("*_game_*.npz"))
    rng.shuffle(npzs)
    states = []
    for npz in npzs:
        try:
            rows = read_compact_shard(npz)
        except Exception:
            continue
        for row in rows:
            s = reconstruct_state(row.placement_history)
            if eng.terminal(s) is None:
                states.append(s)
            if len(states) >= n:
                return states
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--states", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    from hexo_models.hexgt import rust_bridge

    featurize = rust_bridge._hexgt_module().hexgt_featurize_states
    states = tuple(load_states(args.states, args.shards))
    print(f"loaded {len(states)} states; RAYON_NUM_THREADS="
          f"{os.environ.get('RAYON_NUM_THREADS', '(unset)')}")

    # warm
    featurize(states, args.n)

    sampler = CoreSampler(); sampler.start()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        featurize(states, args.n)
    dt = time.perf_counter() - t0
    sampler.stop()

    total_states = len(states) * args.iters
    per_state_us = dt / total_states * 1e6
    print(f"featurize+collate+dict: {dt:.3f}s for {total_states} states "
          f"({args.iters} iters x {len(states)})")
    print(f"  -> {per_state_us:.1f} us/state   ({total_states/dt:.0f} states/s)")
    if sampler.cores_busy:
        cb = sampler.cores_busy
        print(f"  cores busy (>50%): mean={sum(cb)/len(cb):.1f}/{sampler.ncpu} "
              f"peak={max(cb)}/{sampler.ncpu}")
        print(f"  system CPU util: mean={sum(sampler.util)/len(sampler.util):.0f}% "
              f"peak={max(sampler.util):.0f}%  ({len(cb)} samples)")


if __name__ == "__main__":
    main()
