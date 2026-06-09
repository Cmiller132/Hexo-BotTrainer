"""Deep VRAM profile of the hexgt forward: where does the memory go, and how does
it scale with leaf-batch size + candidate-count padding?"""
from __future__ import annotations
import glob
import numpy as np
import torch

from hexo_models.hexgt.architecture import HexgtNetwork, build_attention_layout
from hexo_models.hexgt.constants import NODE_TYPE_CANDIDATE
from hexo_models.hexgt.graph_build import batch_from_states
from hexo_models.hexgt.expand import reconstruct_state
from hexo_models.dense_cnn.compact_io import read_compact_shard

device = torch.device("cuda")
MB = 1024 * 1024


def gather_states(n_states):
    shards = sorted(glob.glob("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay/epoch_0000*_game_*.npz"))
    states = []
    for sp in shards:
        rows = read_compact_shard(sp)
        # sample across the game so candidate counts vary (opening->endgame)
        for i in range(0, len(rows), max(1, len(rows) // 6)):
            states.append(reconstruct_state(rows[i].placement_history))
            if len(states) >= n_states:
                return states
    return states


def main():
    ck = torch.load("/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc/hexgt_bc_step006009.pt", map_location=device, weights_only=False)
    a = ck["arch"]
    model = HexgtNetwork(token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
                         ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
                         short_term_value_horizons=tuple(a["short_term_value_horizons"])).to(device).eval()
    model.load_state_dict(ck["model"])

    print(f"{'graphs':>7} {'tot_cand':>9} {'max_cand':>9} {'pad_eff':>8} {'peak_MB':>9} {'MB/graph':>9}")
    for ng in (256, 512, 1024, 2048):
        states = gather_states(ng)
        batch = batch_from_states(states, 3)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        cg = batch["candidate_graph"]
        num_graphs = int(batch["num_graphs"])
        counts = torch.bincount(cg, minlength=num_graphs)
        tot_cand = int(counts.sum()); max_cand = int(counts.max())
        pad_eff = tot_cand / (num_graphs * max_cand)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model.forward_policy_value(batch)
        peak = torch.cuda.max_memory_allocated() / MB
        print(f"{num_graphs:>7} {tot_cand:>9} {max_cand:>9} {pad_eff:>7.1%} {peak:>9.0f} {peak/num_graphs:>9.2f}")
        del batch; torch.cuda.empty_cache()

    # Stage breakdown at a representative size.
    print("\n--- stage breakdown @ 1024 graphs (peak MB per stage) ---")
    states = gather_states(1024)
    batch = batch_from_states(states, 3)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    num_graphs = int(batch["num_graphs"])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        h = model.node_in(batch["node_feat"])
        ei = batch["edge_index"]; et = batch.get("edge_type"); ea = batch.get("edge_attr")
        for layer in model.gnn:
            h = layer(h, ei, et, ea)
        print(f"  after GNN ({model.gnn_layers} layers): peak {torch.cuda.max_memory_allocated()/MB:.0f} MB")
        is_cand = batch["node_type"] == NODE_TYPE_CANDIDATE
        layout = build_attention_layout(batch["node_graph"], is_cand, num_graphs)
        print(f"  attention layout: ctx pad {tuple(layout.ctx_index.shape)}, cand pad {tuple(layout.cand_index.shape)}")
        torch.cuda.reset_peak_memory_stats()
        for layer in model.transformer:
            h = layer(h, layout)
        print(f"  transformer ({model.ctx_layers} layers): peak {torch.cuda.max_memory_allocated()/MB:.0f} MB (THE driver)")
        # size of the dominant padded tensors
        B, mc = layout.cand_index.shape
        D = model.token_dim; ffn = model.ffn_dim
        print(f"  padded cand tensor (B={B}, max_cand={mc}, D={D}) fp16 = {B*mc*D*2/MB:.0f} MB")
        print(f"  cand FFN intermediate (B, max_cand, ffn={ffn}) fp16 = {B*mc*ffn*2/MB:.0f} MB")


if __name__ == "__main__":
    main()
