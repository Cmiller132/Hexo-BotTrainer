"""Quantify the candidate-radius sparsification: pos/s + mean per-leaf
edges/nodes/candidates for n=3 vs n=2, td96/gnn2 @512 visits active=512.
Wraps the evaluator to accumulate the payload edge/node/candidate totals."""
import time
import numpy as np
import torch
import hexgnn.selfplay as SP
from hexgnn.inference import HexgnnInference
from hexgnn.config import parse_hexgnn_config
from hexgnn.architecture import HexgnnNetwork

TD, GNN, ACTIVE, NUMG, MAXA, VIS, VB = 96, 2, 512, 512, 20, 512, 16


class CountInf(HexgnnInference):
    N = 0; E = 0; ND = 0; C = 0
    def evaluate_featurized_batch(self, payload):
        CountInf.N += int(payload["num_graphs"]); CountInf.E += int(payload["total_edges"])
        CountInf.ND += int(payload["total_nodes"]); CountInf.C += int(payload["total_candidates"])
        return super().evaluate_featurized_batch(payload)


def run(n):
    CountInf.N = CountInf.E = CountInf.ND = CountInf.C = 0
    SP.HexgnnInference = CountInf
    m = HexgnnNetwork(token_dim=TD, gnn_layers=GNN).to("cuda").eval()
    m.forward_policy_value = torch.compile(m.forward_policy_value, dynamic=True)
    cfg = parse_hexgnn_config({"device": "cuda",
        "architecture": {"candidate_radius": n, "token_dim": TD, "gnn_layers": GNN, "attention_heads": 4, "value_pma_seeds": 2},
        "samples": {"soft_z_lambda": 0.0},
        "selfplay": dict(search_visits=VIS, active_games=ACTIVE, mcts_active_root_limit=ACTIVE, c_puct=1.5,
                         root_dirichlet_noise_enabled=True, root_dirichlet_total_alpha=6.6, root_dirichlet_noise_fraction=0.30,
                         root_policy_temperature=1.0, widening_max_children=96, widening_min_children=2, widening_policy_mass=0.95,
                         max_actions=MAXA, temperature=1.0, temperature_floor=0.3, temperature_halflife=33,
                         forced_playout_k=2.0, pcr_enabled=False)})
    t0 = time.perf_counter()
    r = SP.run_selfplay_games(m, cfg, num_games=NUMG, output_dir=f"/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_smoke/nsweep_n{n}",
                              epoch=0, device="cuda", fp16=True, base_seed=1, active_games=ACTIVE,
                              virtual_batch_size=VB, collect_examples=0)
    nl = max(1, CountInf.N)
    print(f">>> n={n}: pos/s={r.positions_per_second:.1f} searched={r.searched_positions} wall={time.perf_counter()-t0:.0f}s "
          f"| per-leaf: nodes={CountInf.ND/nl:.0f} edges={CountInf.E/nl:.0f} cand={CountInf.C/nl:.0f} "
          f"(leaves total={CountInf.N})", flush=True)


if __name__ == "__main__":
    run(3)
    run(2)
