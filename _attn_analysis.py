"""Read-only analysis: token counts + attention FLOPs, current (ctx-self +
cand->ctx cross) vs hypothetical full all-to-all self-attention. CPU only; does
not touch the GPU / live run."""
import glob
import time

import numpy as np
import torch

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.features import build_graph_tensors
from hexo_models.hexgt.collate import collate_graphs
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt.constants import (
    NODE_TYPE_SIDE, NODE_TYPE_STONE, NODE_TYPE_CANDIDATE, NODE_TYPE_WINDOW,
)

torch.set_grad_enabled(False)
RADIUS = 3
D = 168          # token_dim (run)
H = 4            # attention heads
CTX_LAYERS = 3   # run

# Collect single-position graphs from real self-play games across the midgame.
hxr_files = sorted(glob.glob("/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3/selfplay/epoch_00000*.hxr"))
graphs = []
counts = []  # (side, stone, cand, window, total) per graph
for f in hxr_files:
    rf = HexoRecordFile.open(f)
    for rec in rf.iter_records():
        coords = [unpack_coord_id(int(c)) for c in rec.action_ids]
        s = eng.new_game()
        for i, c in enumerate(coords):
            if i in (10, 18, 26, 34, 42):
                g = build_graph_tensors(rust_bridge.graph_facts(s, RADIUS))
                graphs.append(g)
                nt = np.asarray(g.node_type)
                counts.append((
                    int((nt == NODE_TYPE_SIDE).sum()), int((nt == NODE_TYPE_STONE).sum()),
                    int((nt == NODE_TYPE_CANDIDATE).sum()), int((nt == NODE_TYPE_WINDOW).sum()),
                    int(nt.shape[0]),
                ))
            eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
        if len(graphs) >= 400:
            break
    if len(graphs) >= 400:
        break

c = np.asarray(counts)
side, stone, cand, window, total = (c[:, i] for i in range(5))
ctx = side + stone + window  # context tokens (non-candidate)
print(f"positions sampled: {len(c)}")
print(f"per-position token counts (mean / p50 / p95 / max):")
for name, arr in [("SIDE", side), ("STONE", stone), ("WINDOW", window),
                  ("CANDIDATE", cand), ("context(ctx)", ctx), ("TOTAL", total)]:
    print(f"  {name:14s} mean={arr.mean():7.1f}  p50={np.percentile(arr,50):6.0f}  p95={np.percentile(arr,95):6.0f}  max={arr.max():5d}")

# Attention FLOP comparison (the QK^T + AV scaled-dot-product term, the N^2 part).
# current per graph: ctx self-attn over ctx tokens (ctx^2) + cand->ctx cross (cand*ctx)
# full per graph: self-attn over all tokens (total^2)
cur = ctx.astype(np.float64) ** 2 + cand.astype(np.float64) * ctx.astype(np.float64)
full = total.astype(np.float64) ** 2
print(f"\nAttention score-matrix size per graph (proportional to attn FLOPs, per layer):")
print(f"  current (ctx^2 + cand*ctx):  mean={cur.mean():,.0f}")
print(f"  full   (total^2):            mean={full.mean():,.0f}")
print(f"  full / current ratio:        mean={ (full/cur).mean():.2f}x   (median {np.median(full/cur):.2f}x)")

# Microbench: actual nn.MultiheadAttention cost, current split vs full, at a
# representative vbatch=64 padded batch (CPU; relative cost is the signal).
def bench(B, reps=20):
    max_ctx = int(np.percentile(ctx, 95))
    max_cand = int(np.percentile(cand, 95))
    max_tot = int(np.percentile(total, 95))
    mha = torch.nn.MultiheadAttention(D, H, batch_first=True).eval()
    x_ctx = torch.randn(B, max_ctx, D)
    x_cand = torch.randn(B, max_cand, D)
    x_tot = torch.randn(B, max_tot, D)
    def run_current():
        for _ in range(CTX_LAYERS):
            mha(x_ctx, x_ctx, x_ctx, need_weights=False)   # ctx self
            mha(x_cand, x_ctx, x_ctx, need_weights=False)  # cand->ctx
    def run_full():
        for _ in range(CTX_LAYERS):
            mha(x_tot, x_tot, x_tot, need_weights=False)   # all-to-all self
    for fn in (run_current, run_full):  # warm
        fn()
    t0 = time.perf_counter()
    for _ in range(reps): run_current()
    t_cur = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps): run_full()
    t_full = (time.perf_counter() - t0) / reps
    print(f"\n[CPU microbench B={B}, padded max_ctx={max_ctx} max_cand={max_cand} max_tot={max_tot}, {CTX_LAYERS} layers]")
    print(f"  current attn stack: {t_cur*1000:7.2f} ms   full attn stack: {t_full*1000:7.2f} ms   ratio {t_full/t_cur:.2f}x")

bench(64)
