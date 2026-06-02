"""Detailed performance profile of hexgt self-play at the 512-sim config.

Drives the native HexgtMctsSession over REAL 96x8 positions at the production
search params (visits=512, active=64, vbatch=64, compile) and produces a full
breakdown:

  TIME    -- per-round attribution from the Rust stage timers (encoding_seconds,
             evaluator_seconds, parse_seconds, payload_seconds, dedup_seconds,
             cache_insert_seconds) + tree-ops remainder (wall - sum). With
             --gpu-split, cuda-event timing splits the evaluator into pure GPU
             forward vs transport/post.
  TPUT    -- searched positions/sec, forwards/searched-pos, cache hit-rate.
  UTIL    -- background samplers: GPU util/mem (nvidia-smi), system CPU util
             (/proc/stat), process + system RAM (/proc/self/status, /proc/meminfo).
  MEM     -- VRAM allocated/reserved peak; tree + cache byte drivers.

Usage:
  python scripts/_perf_512.py --shards <dir> --visits 512 --active 64 --moves 4 \
      --vbatch 64 --compile [--gpu-split] [--profiler]
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session

# Production search params (config C1 / configs/hexgt_model2.toml [selfplay]).
SEARCH = dict(
    c_puct=1.5,
    root_dirichlet_total_alpha=10.83,
    root_dirichlet_noise_fraction=0.25,
    root_policy_temperature=1.1,
    fpu_reduction=0.20,
    virtual_loss=1.0,
    widening_policy_mass=0.95,
    widening_max_children=96,
    widening_min_children=2,
    forced_playout_k=2.0,
    temperature=1.0,
)
SECONDS_KEYS = [
    "encoding_seconds", "payload_seconds", "evaluator_seconds",
    "parse_seconds", "dedup_seconds", "cache_insert_seconds",
]


# --------------------------------------------------------------------------- #
# Background samplers
# --------------------------------------------------------------------------- #
class CpuSampler(threading.Thread):
    """System-wide CPU util + per-core busy count from /proc/stat."""

    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.util = []          # whole-machine util %
        self.cores_busy = []    # # logical cores > 50% busy
        self.ncpu = 0

    @staticmethod
    def _read():
        out = {}
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu"):
                    break
                p = line.split()
                name = p[0]
                vals = list(map(int, p[1:]))
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                total = sum(vals)
                out[name] = (total, idle)
        return out

    def run(self):
        prev = self._read()
        while not self.stop_flag.wait(self.interval):
            cur = self._read()
            tot_t, tot_i = cur["cpu"]; pt, pi = prev["cpu"]
            dt = tot_t - pt
            if dt > 0:
                self.util.append(100.0 * (1 - (tot_i - pi) / dt))
            busy = 0; ncpu = 0
            for k in cur:
                if k == "cpu":
                    continue
                ncpu += 1
                ct, ci = cur[k]; ptk, pik = prev.get(k, (ct, ci))
                d = ct - ptk
                if d > 0 and (1 - (ci - pik) / d) > 0.5:
                    busy += 1
            self.ncpu = ncpu
            self.cores_busy.append(busy)
            prev = cur

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2)

    def summary(self):
        if not self.util:
            return {}
        return dict(
            cpu_util_mean=sum(self.util) / len(self.util),
            cpu_util_peak=max(self.util),
            cores_busy_mean=sum(self.cores_busy) / len(self.cores_busy),
            cores_busy_peak=max(self.cores_busy),
            ncpu=self.ncpu,
            samples=len(self.util),
        )


class GpuSampler(threading.Thread):
    """GPU util + mem via nvidia-smi polling."""

    def __init__(self, interval=0.15):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.util = []
        self.mem = []

    def run(self):
        while not self.stop_flag.wait(self.interval):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=2).decode()
                u, m = out.strip().split("\n")[0].split(",")
                self.util.append(float(u)); self.mem.append(float(m))
            except Exception:
                pass

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=3)

    def summary(self):
        if not self.util:
            return {}
        return dict(
            gpu_util_mean=sum(self.util) / len(self.util),
            gpu_util_peak=max(self.util),
            gpu_mem_used_peak_mb=max(self.mem),
            gpu_mem_used_mean_mb=sum(self.mem) / len(self.mem),
            samples=len(self.util),
        )


def read_ram_mb():
    """(process RSS, process peak RSS, system used, system total) in MB."""
    rss = peak = 0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1024
    mt = ma = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mt = int(line.split()[1]) / 1024
            elif line.startswith("MemAvailable:"):
                ma = int(line.split()[1]) / 1024
    return rss, peak, (mt - ma), mt


# --------------------------------------------------------------------------- #
# State loading
# --------------------------------------------------------------------------- #
def histories_from_shards(n_states, shard_dir, seed=1):
    """Return placement-histories of real nonterminal positions. We keep the
    histories (not live states) because `eng.apply_action` mutates states in
    place; each run rebuilds fresh states so runs are independent + comparable."""
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.expand import reconstruct_state
    rng = random.Random(seed)
    npzs = sorted(Path(shard_dir).glob("*_game_*.npz"))
    epochs = sorted({p.name.split("_game_")[0] for p in npzs})
    if len(epochs) > 1:
        npzs = [p for p in npzs if p.name.split("_game_")[0] != epochs[-1]]
    rng.shuffle(npzs)
    hists = []
    for npz in npzs:
        try:
            rows = read_compact_shard(npz)
        except Exception:
            continue
        for row in rows:
            s = reconstruct_state(row.placement_history)
            if eng.terminal(s) is None:
                hists.append(list(row.placement_history))
            if len(hists) >= n_states:
                return hists
    return hists


def states_from_histories(hists):
    from hexo_models.hexgt.expand import reconstruct_state
    return [reconstruct_state(h) for h in hists]


# --------------------------------------------------------------------------- #
# Core run
# --------------------------------------------------------------------------- #
def run_breakdown(label, inference, hists, *, visits, active, vbatch, moves,
                  n, gpu_split, sample_util):
    pool = states_from_histories(hists[:active])  # fresh states each run
    games = {i: s for i, s in enumerate(pool[:active])}
    sess = new_mcts_session(max_states=1 << 21, n=n)

    # Optional cuda-event timing of the pure GPU forward (perturbs wall a bit).
    gpu_fwd_ms = [0.0]
    if gpu_split:
        orig = inference.model.forward_policy_value
        def timed(batch):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            out = orig(batch)
            ev1.record()
            ev1.synchronize()
            gpu_fwd_ms[0] += ev0.elapsed_time(ev1)
            return out
        inference.model.forward_policy_value = timed

    if inference.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # warmup one move (compile + cache fill) -- not timed
    wk = list(games.keys())
    sess.run([k for k in wk], [games[k] for k in wk], inference,
             visits=visits, seed=99, virtual_batch_size=vbatch,
             active_root_limit=active, **SEARCH)
    if inference.device.type == "cuda":
        torch.cuda.synchronize()
    gpu_fwd_ms[0] = 0.0  # discard warmup

    acc = {k: 0.0 for k in SECONDS_KEYS}
    total_searched = total_unique = total_hits = total_requested = 0
    chunks = 0
    max_chunk = 0
    enc_states = 0
    tree_nodes = 0
    tree_active_edge_bytes = tree_hidden_prior_bytes = 0
    cache_peak = 0

    cpu = GpuS = None
    if sample_util:
        cpu = CpuSampler(); cpu.start()
        GpuS = GpuSampler(); GpuS.start()
    ram_peak = 0.0
    sys_used_peak = 0.0

    if inference.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for move in range(moves):
        active_games = [(k, s) for k, s in games.items() if eng.terminal(s) is None]
        if not active_games:
            break
        akeys = [k for k, _ in active_games]
        astates = [s for _, s in active_games]
        results = sess.run(akeys, astates, inference, visits=visits,
                           seed=1000 + move, virtual_batch_size=vbatch,
                           active_root_limit=active, **SEARCH)
        total_searched += len(active_games)
        batch = dict(dict(results[0].diagnostics).get("batch", {}))
        d = dict(batch.get("evaluation", {}))
        tree = dict(batch.get("tree", {}))
        for k in SECONDS_KEYS:
            acc[k] += float(d.get(k, 0.0))
        total_unique += int(d.get("unique_states", 0))
        total_hits += int(d.get("cache_hits", 0))
        total_requested += int(d.get("requested_states", 0))
        chunks += int(d.get("evaluator_chunks", 0))
        max_chunk = max(max_chunk, int(d.get("max_chunk_states", 0)))
        enc_states += int(d.get("encoded_states", 0))
        cache_peak = max(cache_peak, int(d.get("cache_size_peak", 0)))
        tree_nodes = max(tree_nodes, int(tree.get("node_count", 0)))
        tree_active_edge_bytes = max(tree_active_edge_bytes, int(tree.get("active_edge_bytes", 0)))
        tree_hidden_prior_bytes = max(tree_hidden_prior_bytes, int(tree.get("hidden_prior_bytes", 0)))
        rss, _, sys_used, sys_total = read_ram_mb()
        ram_peak = max(ram_peak, rss)
        sys_used_peak = max(sys_used_peak, sys_used)
        for (k, s), res in zip(active_games, results):
            eng.apply_action(s, eng.PlacementAction(unpack_coord_id(res.action_id)))
    if inference.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    if cpu:
        cpu.stop(); GpuS.stop()
    rss, peak_rss, sys_used, sys_total = read_ram_mb()
    ram_peak = max(ram_peak, peak_rss)

    summed = sum(acc.values())
    tree_ops = wall - summed
    out = {
        "label": label, "visits": visits, "active": active, "vbatch": vbatch,
        "moves_done": move + 1, "wall": wall,
        "searched": total_searched, "pos_s": total_searched / wall,
        "fwd_per_pos": total_unique / max(1, total_searched),
        "hit_rate": total_hits / max(1, total_requested),
        "requested": total_requested, "unique": total_unique, "hits": total_hits,
        "chunks_total": chunks, "max_chunk_states": max_chunk,
        "encoded_states": enc_states,
        "seconds": acc, "tree_ops": tree_ops, "summed": summed,
        "gpu_fwd_s": gpu_fwd_ms[0] / 1000.0 if gpu_split else None,
        "vram_alloc_peak_gb": (torch.cuda.max_memory_allocated() / 1e9
                               if inference.device.type == "cuda" else 0),
        "vram_reserved_peak_gb": (torch.cuda.max_memory_reserved() / 1e9
                                  if inference.device.type == "cuda" else 0),
        "ram_proc_peak_mb": ram_peak, "ram_sys_used_peak_mb": sys_used_peak,
        "ram_sys_total_mb": sys_total,
        "tree_nodes_peak": tree_nodes,
        "tree_active_edge_bytes": tree_active_edge_bytes,
        "tree_hidden_prior_bytes": tree_hidden_prior_bytes,
        "cache_size_peak": cache_peak,
        "cpu": cpu.summary() if cpu else {},
        "gpu": GpuS.summary() if GpuS else {},
    }
    if gpu_split:
        inference.model.forward_policy_value = orig
    return out


def print_breakdown(o):
    print(f"\n===== {o['label']}  (visits={o['visits']} active={o['active']} "
          f"vbatch={o['vbatch']}) =====")
    print(f"  moves={o['moves_done']}  wall={o['wall']:.2f}s  searched={o['searched']}")
    print(f"  THROUGHPUT: {o['pos_s']:.2f} searched pos/s   "
          f"fwd/pos={o['fwd_per_pos']:.1f}/{o['visits']}   "
          f"cache_hit={o['hit_rate']:.1%}")
    print(f"  forwards: requested={o['requested']} unique={o['unique']} hits={o['hits']} "
          f"chunks={o['chunks_total']} max_chunk={o['max_chunk_states']}")
    w = o["wall"]
    print("  TIME BREAKDOWN (% of wall):")
    label_map = {
        "encoding_seconds": "Rust featurize+graph-encode",
        "payload_seconds": "Rust payload dict build",
        "evaluator_seconds": "Py evaluator (transport+GPU fwd+softmax+d2h)",
        "parse_seconds": "Rust parse (legality/sort/normalize)",
        "dedup_seconds": "Rust dedup (transposition)",
        "cache_insert_seconds": "Rust cache insert",
    }
    for k in SECONDS_KEYS:
        s = o["seconds"][k]
        print(f"    {label_map[k]:<46} {s:7.2f}s  {100*s/w:5.1f}%")
    print(f"    {'MCTS tree ops (select/expand/backup/glue)':<46} "
          f"{o['tree_ops']:7.2f}s  {100*o['tree_ops']/w:5.1f}%")
    if o["gpu_fwd_s"] is not None:
        ev = o["seconds"]["evaluator_seconds"]
        gf = o["gpu_fwd_s"]
        print(f"    -- within evaluator: pure GPU forward = {gf:.2f}s "
              f"({100*gf/w:.1f}% wall, {100*gf/max(ev,1e-9):.0f}% of evaluator); "
              f"transport+post = {ev-gf:.2f}s ({100*(ev-gf)/w:.1f}% wall)")
    print(f"  VRAM: alloc_peak={o['vram_alloc_peak_gb']:.2f}GB  "
          f"reserved_peak={o['vram_reserved_peak_gb']:.2f}GB  (ceiling 12GB)")
    print(f"  RAM: proc_peak={o['ram_proc_peak_mb']:.0f}MB  "
          f"sys_used_peak={o['ram_sys_used_peak_mb']:.0f}MB / {o['ram_sys_total_mb']:.0f}MB")
    print(f"  tree: nodes_peak={o['tree_nodes_peak']} "
          f"active_edge_bytes={o['tree_active_edge_bytes']/1e6:.1f}MB "
          f"hidden_prior_bytes={o['tree_hidden_prior_bytes']/1e6:.1f}MB  "
          f"cache_peak={o['cache_size_peak']}")
    if o["gpu"]:
        g = o["gpu"]
        print(f"  GPU sampler: util_mean={g['gpu_util_mean']:.0f}% "
              f"util_peak={g['gpu_util_peak']:.0f}% "
              f"mem_peak={g['gpu_mem_used_peak_mb']:.0f}MB ({g['samples']} samples)")
    if o["cpu"]:
        c = o["cpu"]
        print(f"  CPU sampler: util_mean={c['cpu_util_mean']:.0f}% "
              f"util_peak={c['cpu_util_peak']:.0f}% "
              f"cores_busy_mean={c['cores_busy_mean']:.1f}/{c['ncpu']} "
              f"cores_busy_peak={c['cores_busy_peak']}/{c['ncpu']}")


def run_torch_profiler(inference, pool, *, visits, active, vbatch, n):
    """Capture a single 512-sim search round under torch.profiler to attribute
    GPU kernels (transformer attn vs GNN einsum vs softmax)."""
    from torch.profiler import profile, ProfilerActivity
    games = {i: s for i, s in enumerate(states_from_histories(pool[:active]))}
    sess = new_mcts_session(max_states=1 << 21, n=n)
    wk = list(games.keys())
    # warm
    sess.run(wk, [games[k] for k in wk], inference, visits=visits, seed=7,
             virtual_batch_size=vbatch, active_root_limit=active, **SEARCH)
    torch.cuda.synchronize()
    print("\n===== TORCH PROFILER: one 512-sim search round (GPU kernels) =====")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        sess.run(wk, [games[k] for k in wk], inference, visits=visits, seed=8,
                 virtual_batch_size=vbatch, active_root_limit=active, **SEARCH)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--active", type=int, default=64)
    ap.add_argument("--visits", type=int, default=512)
    ap.add_argument("--moves", type=int, default=4)
    ap.add_argument("--vbatch", type=int, default=64)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--gpu-split", action="store_true")
    ap.add_argument("--profiler", action="store_true")
    ap.add_argument("--also-128", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # Config arch (configs/hexgt_model2.toml): 168/g3/c3/h4/ffn336.
    model = HexgtNetwork(
        token_dim=168, gnn_layers=3, ctx_layers=3, attention_heads=4,
        ffn_dim=336, short_term_value_horizons=(),
    ).to(device).eval()
    if args.compile:
        model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)
    inference = HexgtInference(model, device=device, fp16=(device.type == "cuda"))

    # generous pool so all `active` games have a starting state
    hists = histories_from_shards(max(args.active, 64) + 8, args.shards, seed=1)
    print(f"loaded {len(hists)} real nonterminal positions; device={device} "
          f"compile={args.compile}")

    # Main clean run (no gpu-split sync) for throughput + util.
    o512 = run_breakdown("512-sim clean (throughput+util)", inference, hists,
                         visits=args.visits, active=args.active, vbatch=args.vbatch,
                         moves=args.moves, n=args.n, gpu_split=False, sample_util=True)
    print_breakdown(o512)

    # GPU-split run (cuda events; perturbs wall, isolates pure GPU compute).
    if args.gpu_split and device.type == "cuda":
        try:
            og = run_breakdown("512-sim gpu-split (perturbed wall)", inference, hists,
                               visits=args.visits, active=args.active, vbatch=args.vbatch,
                               moves=max(2, args.moves - 1), n=args.n,
                               gpu_split=True, sample_util=False)
            print_breakdown(og)
        except Exception as e:
            print(f"[gpu-split run failed: {e!r}]")

    # 128-sim comparison for scaling.
    if args.also_128:
        try:
            o128 = run_breakdown("128-sim clean (scaling cmp)", inference, hists,
                                 visits=128, active=args.active, vbatch=args.vbatch,
                                 moves=args.moves, n=args.n, gpu_split=False, sample_util=True)
            print_breakdown(o128)
            print(f"\nSCALING 128->512: pos/s {o128['pos_s']:.2f} -> {o512['pos_s']:.2f} "
                  f"({o512['pos_s'] and o128['pos_s']/o512['pos_s']:.2f}x slower)  "
                  f"fwd/pos {o128['fwd_per_pos']:.1f} -> {o512['fwd_per_pos']:.1f}")
        except Exception as e:
            print(f"[128-sim run failed: {e!r}]")

    if args.profiler and device.type == "cuda":
        run_torch_profiler(inference, hists, visits=args.visits, active=args.active,
                           vbatch=args.vbatch, n=args.n)


if __name__ == "__main__":
    main()
