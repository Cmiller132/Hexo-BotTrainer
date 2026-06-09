"""Re-profile: chunked (compressed) forward vs the old single-forward peak VRAM,
at the leaf-batch sizes self-play actually produces (which OOM'd before)."""
from __future__ import annotations
import glob
import torch

from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.graph_build import batch_from_states
from hexo_models.hexgt.expand import reconstruct_state
from hexo_models.dense_cnn.compact_io import read_compact_shard

device = torch.device("cuda")
MB = 1024 * 1024


def gather_states(n):
    shards = sorted(glob.glob("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay/epoch_0000*_game_*.npz"))
    states = []
    for sp in shards:
        rows = read_compact_shard(sp)
        for i in range(0, len(rows), max(1, len(rows) // 6)):
            states.append(reconstruct_state(rows[i].placement_history))
            if len(states) >= n:
                return states
    return states


def peak_of(infer, batch):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    out, _ = infer.forward_batch({k: (v.clone() if hasattr(v, "clone") else v) for k, v in batch.items()})
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / MB


def main():
    ck = torch.load("/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc/hexgt_bc_step006009.pt", map_location=device, weights_only=False)
    a = ck["arch"]
    model = HexgtNetwork(token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
                         ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
                         short_term_value_horizons=tuple(a["short_term_value_horizons"])).to(device).eval()
    model.load_state_dict(ck["model"])

    single = HexgtInference(model, device=device, fp16=True, forward_pad_budget=10**12)  # no chunking
    chunked = HexgtInference(model, device=device, fp16=True)  # default 200k budget

    print(f"{'graphs':>7} {'single_MB':>10} {'chunked_MB':>11} {'reduction':>10}")
    for ng in (512, 1024, 2048, 4096):
        states = gather_states(ng)
        batch = batch_from_states(states, 3)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        try:
            s = peak_of(single, batch)
            ss = f"{s:.0f}"
        except torch.cuda.OutOfMemoryError:
            s = None; ss = "OOM"
        torch.cuda.empty_cache()
        try:
            c = peak_of(chunked, batch)
            cs = f"{c:.0f}"
        except torch.cuda.OutOfMemoryError:
            c = None; cs = "OOM"
        red = f"{s/c:.1f}x" if (s and c) else ("-" if c is None else "inf")
        print(f"{int(batch['num_graphs']):>7} {ss:>10} {cs:>11} {red:>10}", flush=True)
        del batch; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
