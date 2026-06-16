"""Throughput sweep for the post-optimization 512-sim regime.

Now that featurization is parallel and overlapped with the GPU forward, re-sweep
the concurrency (active_games) and the forward pad-budget to use the spare VRAM
headroom. One compiled model is reused across configs (amortizes warmup); each
config rebuilds only the lightweight HexgtInference (for its pad-budget) and runs
a warmup + timed moves. Reports pos/s + peak VRAM (torch alloc/reserved +
nvidia-smi) + process RAM + GPU util per config.

Usage: python scripts/_sweep_active.py --shards <dir> [--moves 4]
"""
from __future__ import annotations

import argparse
import random
import subprocess
import threading
import time
from pathlib import Path

import torch

import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session

SEARCH = dict(
    c_puct=1.5, root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
    root_policy_temperature=1.1, fpu_reduction=0.20, virtual_loss=1.0,
    widening_policy_mass=0.95, widening_max_children=96, widening_min_children=2,
    forced_playout_k=2.0, temperature=1.0,
)


class GpuSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval; self.stop_flag = threading.Event()
        self.util = []; self.mem = []

    def run(self):
        while not self.stop_flag.wait(self.interval):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=2).decode()
                u, m = out.strip().split("\n")[0].split(",")
                self.util.append(float(u)); self.mem.append(float(m))
            except Exception:
                pass

    def stop(self):
        self.stop_flag.set(); self.join(timeout=3)


def proc_rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def histories(n, shard_dir, seed=1):
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.expand import reconstruct_state
    rng = random.Random(seed)
    npzs = sorted(Path(shard_dir).glob("*_game_*.npz")); rng.shuffle(npzs)
    hs = []
    for p in npzs:
        try:
            rows = read_compact_shard(p)
        except Exception:
            continue
        for r in rows:
            s = reconstruct_state(r.placement_history)
            if eng.terminal(s) is None:
                hs.append(list(r.placement_history))
            if len(hs) >= n:
                return hs
    return hs


def states_from(hists):
    from hexo_models.hexgt.expand import reconstruct_state
    return [reconstruct_state(h) for h in hists]


def measure(model, hists, *, visits, active, vbatch, budget, moves):
    inf = HexgtInference(model, device="cuda", fp16=True, forward_pad_budget=budget)
    games = {i: s for i, s in enumerate(states_from(hists[:active]))}
    sess = new_mcts_session(max_states=1 << 21, n=3)
    kw = dict(visits=visits, virtual_batch_size=vbatch, active_root_limit=active, **SEARCH)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    # warmup (compile for these shapes + cache fill)
    wk = list(games.keys())
    sess.run(wk, [games[k] for k in wk], inf, seed=99, **kw)
    torch.cuda.synchronize()
    gpu = GpuSampler(); gpu.start(); rss_peak = proc_rss_mb()
    t0 = time.perf_counter(); searched = 0
    for mv in range(moves):
        active_g = [(k, s) for k, s in games.items() if eng.terminal(s) is None]
        if not active_g:
            break
        res = sess.run([k for k, _ in active_g], [s for _, s in active_g], inf, seed=1000 + mv, **kw)
        searched += len(active_g)
        rss_peak = max(rss_peak, proc_rss_mb())
        for (k, s), r in zip(active_g, res):
            eng.apply_action(s, eng.PlacementAction(unpack_coord_id(r.action_id)))
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    gpu.stop()
    return dict(
        pos_s=searched / dt, searched=searched, dt=dt,
        vram_alloc=torch.cuda.max_memory_allocated() / 1e9,
        vram_reserved=torch.cuda.max_memory_reserved() / 1e9,
        smi_mem_peak=max(gpu.mem) if gpu.mem else 0,
        gpu_util=sum(gpu.util) / len(gpu.util) if gpu.util else 0,
        rss_peak=rss_peak,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--moves", type=int, default=4)
    ap.add_argument("--visits", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    model = HexgtNetwork(token_dim=168, gnn_layers=3, ctx_layers=3, attention_heads=4,
                         ffn_dim=336, short_term_value_horizons=()).to("cuda").eval()
    model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)
    hists = histories(200, args.shards, seed=1)
    print(f"loaded {len(hists)} positions; visits={args.visits} moves={args.moves} repeat={args.repeat}\n")

    # (active, vbatch, budget_k) configs. Baseline 200k vs raised budgets; the
    # active sweep at a raised budget the headroom allows.
    configs = [
        (64, 64, 200), (64, 64, 600),
        (96, 64, 600),
        (128, 64, 600), (128, 64, 1000), (128, 96, 1000),
        (192, 64, 600), (192, 64, 1000),
    ]
    rows = []
    hdr = f"{'active':>6} {'vbatch':>6} {'budget':>7} | {'pos/s(best)':>11} {'pos/s(med)':>10} | {'VRAMalloc':>9} {'VRAMresv':>8} {'smiMB':>6} | {'RSSmb':>6} {'gpu%':>5}"
    print(hdr); print("-" * len(hdr))
    for active, vbatch, bk in configs:
        runs = []
        for _ in range(args.repeat):
            runs.append(measure(model, hists, visits=args.visits, active=active,
                                vbatch=vbatch, budget=bk * 1000, moves=args.moves))
        ps = sorted(r["pos_s"] for r in runs)
        best = ps[-1]; med = ps[len(ps) // 2]
        r0 = max(runs, key=lambda r: r["pos_s"])
        print(f"{active:>6} {vbatch:>6} {bk:>6}k | {best:>11.2f} {med:>10.2f} | "
              f"{r0['vram_alloc']:>8.2f}G {r0['vram_reserved']:>7.2f}G {r0['smi_mem_peak']:>6.0f} | "
              f"{r0['rss_peak']:>6.0f} {r0['gpu_util']:>4.0f}%")
        rows.append((active, vbatch, bk, best, med, r0))
    best_cfg = max(rows, key=lambda r: r[3])
    print(f"\nBEST: active={best_cfg[0]} vbatch={best_cfg[1]} budget={best_cfg[2]}k "
          f"-> {best_cfg[3]:.2f} pos/s (VRAM {best_cfg[5]['vram_reserved']:.2f}G reserved, "
          f"smi {best_cfg[5]['smi_mem_peak']:.0f}MB, RSS {best_cfg[5]['rss_peak']:.0f}MB)")


if __name__ == "__main__":
    main()
