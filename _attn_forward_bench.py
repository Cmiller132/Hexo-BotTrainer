"""Forward-level cost: real HexgtNetwork forward (current ctx-self + cand->ctx
cross) vs a full all-to-all self-attention variant, on a representative vbatch=64
batch built from real positions. CPU only (does not touch the live GPU run).

The ratio of the FULL FORWARD (not just the attention op) is what maps to pos/s,
since attention is only part of the forward (GNN + QKV/out projections + FFN + PMA
+ heads are all linear in token count and unchanged)."""
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.features import build_graph_tensors
from hexo_models.hexgt.collate import collate_graphs
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt import architecture as A

torch.set_grad_enabled(False)
torch.manual_seed(0)
B = 64

# Build a vbatch=64 batch of real midgame positions.
graphs = []
for f in sorted(glob.glob("/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3/selfplay/epoch_000006.hxr")):
    rf = HexoRecordFile.open(f)
    for rec in rf.iter_records():
        coords = [unpack_coord_id(int(c)) for c in rec.action_ids]
        s = eng.new_game()
        for i, c in enumerate(coords):
            if i in (14, 22, 30):
                graphs.append(build_graph_tensors(rust_bridge.graph_facts(s, 3)))
            eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
            if len(graphs) >= B:
                break
        if len(graphs) >= B:
            break
    if len(graphs) >= B:
        break
batch = collate_graphs(graphs[:B])
print(f"batch: {B} graphs, {int(batch['node_feat'].shape[0])} total nodes "
      f"({batch['node_feat'].shape[0]/B:.0f}/graph)")

model = A.HexgtNetwork(token_dim=168, gnn_layers=4, ctx_layers=3, ffn_dim=336,
                       attention_heads=4, short_term_value_horizons=(4, 12, 24),
                       value_pma_seeds=2).eval()

# --- full all-to-all self-attention variant: monkeypatch the transformer layer ---
# Each graph's ALL tokens (ctx + candidates) attend to each other in one MHA, then
# a shared FFN. Uses the same padded-by-graph machinery but over every node.
_orig_forward = A.GraphTransformerLayer.forward

def _full_forward(self, h, layout):
    # Rebuild an all-token padded layout from the union of ctx and cand indices.
    # layout.ctx_index/cand_index are (B, max) padded with 0; combine per graph.
    out = h.clone()
    B_ = layout.ctx_index.shape[0]
    # gather all real node rows per graph by concatenating ctx + cand valid rows.
    ctx_idx, cand_idx = layout.ctx_index, layout.cand_index
    ctx_val, cand_val = layout.ctx_valid, layout.cand_valid
    rows_per = []
    maxn = 0
    for b in range(B_):
        r = torch.cat([ctx_idx[b][ctx_val[b]], cand_idx[b][cand_val[b]]]) if cand_idx.shape[1] else ctx_idx[b][ctx_val[b]]
        rows_per.append(r); maxn = max(maxn, int(r.shape[0]))
    idx = h.new_zeros((B_, maxn), dtype=torch.long)
    pad = torch.ones((B_, maxn), dtype=torch.bool)
    for b, r in enumerate(rows_per):
        idx[b, :r.shape[0]] = r; pad[b, :r.shape[0]] = False
    x = h[idx]                                              # (B, maxn, D)
    a, _ = self.ctx_attn(x, x, x, key_padding_mask=pad, need_weights=False)
    x = self.norm_ctx1(x + a)
    x = self.norm_ctx2(x + self.ffn_ctx(x))
    for b, r in enumerate(rows_per):
        out[r] = x[b, :r.shape[0]].to(out.dtype)
    return out

def bench(fn, reps=15):
    for _ in range(3):
        model.forward(batch)
    t0 = time.perf_counter()
    for _ in range(reps):
        model.forward(batch)
    return (time.perf_counter() - t0) / reps

t_cur = bench(None)
A.GraphTransformerLayer.forward = _full_forward
t_full = bench(None)
A.GraphTransformerLayer.forward = _orig_forward

print(f"\nfull model forward @ B={B} (CPU, mean of 15):")
print(f"  current (ctx-self + cand->ctx cross): {t_cur*1000:7.1f} ms")
print(f"  full all-to-all self-attention:       {t_full*1000:7.1f} ms")
fwd_ratio = t_full / t_cur
print(f"  forward slowdown factor:              {fwd_ratio:.2f}x  (+{(fwd_ratio-1)*100:.0f}%)")

# Map to self-play pos/s. Forward is ~78% of self-play wall (notes); the rest
# (MCTS tree ops, featurize, host overhead) is unchanged.
fwd_frac = 0.78
for base in (6.5, 8.0):
    new_wall = fwd_frac * fwd_ratio + (1 - fwd_frac)
    print(f"  if forward={fwd_frac:.0%} of wall & current self-play ~{base:.1f} pos/s "
          f"-> ~{base/new_wall:.1f} pos/s ({(1/new_wall-1)*100:+.0f}%)")
