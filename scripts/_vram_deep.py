"""Deep CUDA-memory profile of the hexgt self-play forward.

Answers, with REAL torch.cuda allocator numbers (not estimates):
  * how much VRAM is FIXED CUDA/cuBLAS/cuDNN context vs model weights vs the
    transient forward working set,
  * the LIVE peak (`max_memory_allocated`) vs allocator-RESERVED peak
    (`max_memory_reserved`) gap (= fragmentation / pool overhead nvidia-smi sees),
  * per-submodule attribution of the live working set (GNN msg-pass vs trunk
    context-transformer attention vs PMA value pool vs heads) via memory hooks on
    an EAGER single forward over the largest realistic chunk,
  * whether reserved VRAM ACCUMULATES across many self-play moves (the
    torch.compile dynamic-shape recompile / inductor-workspace hypothesis),
  * the win from PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (run the script
    twice, with and without the env var).

Drives the REAL native HexgtMctsSession over real positions at the production
search params, exactly like scripts/_perf_512.py, so the shapes match self-play.

Usage (inside the hexgt-build venv, GPU free):
  python scripts/_vram_deep.py --shards <selfplay_dir> \
      --visits 512 --active 64 --vbatch 64 --moves 6 [--compile] [--snapshot OUT.pickle]
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.graph_build import batch_from_states
from hexo_models.hexgt.inference import HexgtInference, _plan_chunks, _subbatch
from hexo_models.hexgt.architecture import precompute_attention_layout
from hexo_models.hexgt.mcts import new_mcts_session

MB = 1024 * 1024

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


def smi_used_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=4).decode()
        return float(out.strip().split("\n")[0])
    except Exception:
        return -1.0


def mem(tag: str) -> dict:
    torch.cuda.synchronize()
    d = dict(
        tag=tag,
        alloc=torch.cuda.memory_allocated() / MB,
        reserved=torch.cuda.memory_reserved() / MB,
        max_alloc=torch.cuda.max_memory_allocated() / MB,
        max_reserved=torch.cuda.max_memory_reserved() / MB,
        smi=smi_used_mb(),
    )
    return d


def pr(d: dict) -> None:
    print(f"  [{d['tag']:<34}] alloc={d['alloc']:8.1f}  reserved={d['reserved']:8.1f}  "
          f"max_alloc={d['max_alloc']:8.1f}  max_reserved={d['max_reserved']:8.1f}  "
          f"smi={d['smi']:8.0f}  MiB", flush=True)


# --------------------------------------------------------------------------- #
def histories_from_shards(n_states, shard_dir, seed=1):
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.expand import reconstruct_state
    rng = random.Random(seed)
    npzs = sorted(Path(shard_dir).glob("*_game_*.npz"))
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


def build_model(device):
    # configs/hexgt_model3.toml architecture: 168 / gnn4 / ctx3 / h4 / ffn336 / pma2.
    model = HexgtNetwork(
        token_dim=168, gnn_layers=4, ctx_layers=3, attention_heads=4,
        ffn_dim=336, short_term_value_horizons=(4, 12, 24), value_pma_seeds=2,
    ).to(device).eval()
    return model


# --------------------------------------------------------------------------- #
def module_breakdown(model, batch, device):
    """EAGER forward over one real batch, attributing the transient working set to
    each top-level stage via pre/post memory hooks (peak alloc delta per stage)."""
    print("\n===== MODULE-LEVEL LIVE WORKING-SET BREAKDOWN (eager, fp16) =====")
    # Build attention layout eagerly (as inference does) so we time the model only.
    num_graphs = int(batch["num_graphs"])
    b = {**batch, **precompute_attention_layout(batch["node_graph"], batch["node_type"], num_graphs)}

    records = []  # (name, alloc_before, peak_during, alloc_after)
    handles = []

    def mk_pre(name):
        def pre(mod, inp):
            torch.cuda.synchronize()
            mod.__a0 = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        return pre

    def mk_post(name):
        def post(mod, inp, out):
            torch.cuda.synchronize()
            a0 = getattr(mod, "__a0", torch.cuda.memory_allocated())
            records.append((name,
                            a0 / MB,
                            torch.cuda.max_memory_allocated() / MB,
                            torch.cuda.memory_allocated() / MB))
        return post

    named = []
    for i, m in enumerate(model.gnn):
        named.append((f"gnn[{i}] RelMsgPass", m))
    for i, m in enumerate(model.transformer):
        named.append((f"transformer[{i}] GraphTfmr", m))
    named.append(("value_pool PMA", model.value_pool))
    named.append(("policy_head", model.policy_head))
    named.append(("value_head", model.value_head))
    named.append(("node_in", model.node_in))
    for name, m in named:
        handles.append(m.register_forward_pre_hook(mk_pre(name)))
        handles.append(m.register_forward_hook(mk_post(name)))

    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / MB
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        _ = model.forward_policy_value(b)
    torch.cuda.synchronize()
    whole_peak = torch.cuda.max_memory_allocated() / MB
    for h in handles:
        h.remove()

    print(f"  batch: num_graphs={num_graphs}  nodes={int(batch['node_feat'].shape[0])}  "
          f"cands={int(batch['candidate_index'].shape[0])}  edges={int(batch['edge_index'].shape[1])}")
    print(f"  base alloc before forward = {base:.1f} MiB ; whole-forward peak = {whole_peak:.1f} MiB "
          f"(transient working set = {whole_peak - base:.1f} MiB)")
    print(f"  {'stage':<26} {'peak_during':>12} {'transient(Δpeak-in)':>20} {'retained(out-in)':>18}")
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for name, a0, peak, a1 in records:
        key = name.split("[")[0].strip()
        agg[key][0] = max(agg[key][0], peak)
        agg[key][1] += (peak - a0)
        agg[key][2] += (a1 - a0)
        agg[key][3] += 1
    for name, a0, peak, a1 in records:
        print(f"  {name:<26} {peak:>12.1f} {peak - a0:>20.1f} {a1 - a0:>18.1f}")
    print("  --- aggregated by stage (transient = sum of per-call Δpeak; retained = sum out-in) ---")
    for key, (mx, tr, ret, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {key:<26} calls={n:<3} max_peak_during={mx:8.1f}  sum_transient={tr:8.1f}  sum_retained={ret:8.1f} MiB")


# --------------------------------------------------------------------------- #
def drive_search(model, inference, hists, *, visits, active, vbatch, moves, n,
                 snapshot=None, accum_csv=None):
    pool = states_from_histories(hists[:active])
    games = {i: s for i, s in enumerate(pool[:active])}
    sess = new_mcts_session(max_states=1 << 21, n=n)
    keys = list(games.keys())

    # warmup (compile + cache fill), not measured
    sess.run(keys, [games[k] for k in keys], inference, visits=visits, seed=99,
             virtual_batch_size=vbatch, active_root_limit=active, **SEARCH)
    torch.cuda.synchronize()
    pr(mem("after warmup move (compiled)"))

    torch.cuda.reset_peak_memory_stats()
    per_move = []
    for move in range(moves):
        active_games = [(k, s) for k, s in games.items() if eng.terminal(s) is None]
        if not active_games:
            break
        akeys = [k for k, _ in active_games]
        astates = [s for _, s in active_games]
        if snapshot and move == 1:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
        results = sess.run(akeys, astates, inference, visits=visits, seed=1000 + move,
                           virtual_batch_size=vbatch, active_root_limit=active, **SEARCH)
        torch.cuda.synchronize()
        if snapshot and move == 1:
            torch.cuda.memory._dump_snapshot(snapshot)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"  [snapshot dumped after move {move} -> {snapshot}]")
        d = mem(f"after search move {move}")
        per_move.append(d)
        pr(d)
        for (k, s), res in zip(active_games, results):
            eng.apply_action(s, eng.PlacementAction(unpack_coord_id(res.action_id)))

    print("\n===== STEADY-STATE PEAK (over measured moves) =====")
    mx_a = max(d["max_alloc"] for d in per_move)
    mx_r = max(d["max_reserved"] for d in per_move)
    mx_smi = max(d["smi"] for d in per_move)
    print(f"  LIVE peak (max_memory_allocated)     = {mx_a:8.1f} MiB")
    print(f"  RESERVED peak (max_memory_reserved)  = {mx_r:8.1f} MiB")
    print(f"  nvidia-smi peak (whole GPU)          = {mx_smi:8.0f} MiB")
    print(f"  reserved-minus-live gap (allocator overhead/frag) = {mx_r - mx_a:8.1f} MiB")
    return per_move


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--visits", type=int, default=512)
    ap.add_argument("--active", type=int, default=64)
    ap.add_argument("--vbatch", type=int, default=64)
    ap.add_argument("--moves", type=int, default=6)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--pad-budget", type=int, default=None)
    ap.add_argument("--no-module-breakdown", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','<unset>')}")
    print(f"torch {torch.__version__}  compile={args.compile}  "
          f"pad_budget={args.pad_budget or os.environ.get('HEXGT_FORWARD_PAD_BUDGET','200000(default)')}")

    print("\n===== FIXED COSTS =====")
    print(f"  baseline nvidia-smi (before any CUDA alloc) = {smi_used_mb():.0f} MiB")
    # create CUDA context
    _ctx = torch.zeros(1, device=device)
    torch.cuda.synchronize()
    pr(mem("CUDA context + 1 tensor"))
    ctx_smi = smi_used_mb()

    model = build_model(device)
    model = model.half()  # weights to fp16 to match autocast inference footprint note
    torch.cuda.synchronize()
    nparams = sum(p.numel() for p in model.parameters())
    pr(mem(f"model on GPU ({nparams/1e6:.2f}M params)"))
    model = model.float()  # autocast handles fp16; keep params fp32 like real run

    if args.compile:
        model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)

    inference = HexgtInference(model, device=device, fp16=True,
                               forward_pad_budget=args.pad_budget)

    hists = histories_from_shards(max(args.active, 64) + 16, args.shards, seed=1)
    print(f"\nloaded {len(hists)} real nonterminal positions")

    # Module breakdown on a realistic single chunk (eager, uncompiled inner model).
    if not args.no_module_breakdown:
        st = states_from_histories(hists[:args.active])
        b = batch_from_states(st, args.n)
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        cg = b["candidate_graph"]
        counts = torch.bincount(cg, minlength=int(b["num_graphs"]))
        budget = inference.forward_pad_budget
        chunks = _plan_chunks(counts, budget)
        # take the LARGEST chunk (worst-case padded forward)
        big = max(chunks, key=lambda g: g.numel() * int(counts[g].max()))
        sub, _ = _subbatch(b, big.to(device))
        print(f"  (largest-of-{len(chunks)} chunk for module breakdown: "
              f"{int(sub['num_graphs'])} graphs, pad budget {budget})")
        module_breakdown(model, sub, device)
        del b, sub; gc.collect(); torch.cuda.empty_cache()

    print("\n===== REAL SEARCH (native MCTS session) =====")
    drive_search(model, inference, hists, visits=args.visits, active=args.active,
                 vbatch=args.vbatch, moves=args.moves, n=args.n, snapshot=args.snapshot)

    print("\n===== torch.cuda.memory_summary() =====")
    print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
