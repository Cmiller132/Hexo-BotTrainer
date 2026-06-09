"""Quick pos/s probes on the smallest sensible hexgnn (td96/gnn2) to bound the
achievable ceiling @512 visits and demonstrate the visits-scaling path to 200.
Each setting: active=512, num_games=512, short games (consistent opening-region
sample for a relative read). argv ignored; settings are inline."""
import time
import torch
from hexgnn.architecture import HexgnnNetwork
from hexgnn.config import parse_hexgnn_config
from hexgnn.selfplay import run_selfplay_games

TD, GNN, ACTIVE, NUMG, MAXA, VB = 96, 2, 512, 512, 16, 16
SETTINGS = [
    dict(visits=512, pcr=True, fast=85),    # best case @512 visits (cheap PCR)
    dict(visits=128, pcr=False, fast=0),    # mid
    dict(visits=64, pcr=False, fast=0),     # path toward 200
]

m = HexgnnNetwork(token_dim=TD, gnn_layers=GNN).to("cuda").eval()
m.forward_policy_value = torch.compile(m.forward_policy_value, dynamic=True)
p = sum(q.numel() for q in m.parameters())
print(f"MODEL td{TD}/gnn{GNN} params={p:,} active={ACTIVE} num_games={NUMG} max_actions={MAXA} vbatch={VB}", flush=True)

for s in SETTINGS:
    sp = dict(search_visits=s["visits"], active_games=ACTIVE, mcts_active_root_limit=ACTIVE,
              c_puct=1.5, root_dirichlet_noise_enabled=True, root_dirichlet_total_alpha=6.6,
              root_dirichlet_noise_fraction=0.30, root_policy_temperature=1.0,
              widening_max_children=96, widening_min_children=2, widening_policy_mass=0.95,
              max_actions=MAXA, temperature=1.0, temperature_floor=0.3, temperature_halflife=33,
              forced_playout_k=2.0, pcr_enabled=s["pcr"], pcr_full_proportion=0.5, pcr_fast_visits=s["fast"])
    cfg = parse_hexgnn_config({"device": "cuda",
                               "architecture": {"candidate_radius": 3, "token_dim": TD, "gnn_layers": GNN,
                                                "attention_heads": 4, "value_pma_seeds": 2},
                               "samples": {"soft_z_lambda": 0.0}, "selfplay": sp})
    t0 = time.perf_counter()
    r = run_selfplay_games(m, cfg, num_games=NUMG, output_dir="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_smoke/posps_m",
                           epoch=0, device="cuda", fp16=True, base_seed=1, active_games=ACTIVE,
                           virtual_batch_size=VB, collect_examples=0)
    print(f">>> visits={s['visits']} pcr={s['pcr']}(fast={s['fast']}): pos/s={r.positions_per_second:.1f} "
          f"searched={r.searched_positions} recorded={r.recorded_positions} "
          f"mean_visits/move={r.mean_search_visits:.0f} wall={time.perf_counter()-t0:.0f}s", flush=True)
