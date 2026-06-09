"""Measure hexgnn self-play pos/s at a given config. Prints a per-round progress
line (cumulative pos/s) so steady-state throughput can be read quickly, then a
final summary. argv: td gnn pcr(0/1) visits active num_games max_actions vbatch
"""
import sys
import time
import torch

td = int(sys.argv[1]); gnn = int(sys.argv[2]); pcr = bool(int(sys.argv[3]))
visits = int(sys.argv[4]); active = int(sys.argv[5]); num_games = int(sys.argv[6])
max_actions = int(sys.argv[7]); vbatch = int(sys.argv[8]) if len(sys.argv) > 8 else 16
N = int(sys.argv[9]) if len(sys.argv) > 9 else 2
STEER = int(sys.argv[10]) if len(sys.argv) > 10 else 0

from hexgnn.architecture import HexgnnNetwork
from hexgnn.config import parse_hexgnn_config
from hexgnn.selfplay import run_selfplay_games

sp = dict(
    search_visits=visits, active_games=active, mcts_active_root_limit=active,
    c_puct=1.5, root_dirichlet_noise_enabled=True, root_dirichlet_total_alpha=6.6,
    root_dirichlet_noise_fraction=0.30, root_policy_temperature=1.0,
    widening_max_children=96, widening_min_children=2, widening_policy_mass=0.95,
    max_actions=max_actions, temperature=1.0, temperature_floor=0.3, temperature_halflife=33,
    forced_playout_k=2.0, pcr_enabled=pcr, pcr_full_proportion=0.5, pcr_fast_visits=85,
)
cfg = parse_hexgnn_config({"device": "cuda",
                           "architecture": {"candidate_radius": N, "token_dim": td, "gnn_layers": gnn,
                                            "attention_heads": 4, "value_pma_seeds": 2},
                           "samples": {"soft_z_lambda": 0.0}, "selfplay": sp})
m = HexgnnNetwork(token_dim=td, gnn_layers=gnn, steerable_channels=STEER).to("cuda").eval()
m.forward_policy_value = torch.compile(m.forward_policy_value, dynamic=True)
p = sum(q.numel() for q in m.parameters())
print(f"CONFIG td{td}/gnn{gnn} params={p:,} pcr={pcr} visits={visits} active={active} "
      f"num_games={num_games} max_actions={max_actions} vbatch={vbatch}", flush=True)

t0 = time.perf_counter()
def cb(msg):
    print(f"[{time.perf_counter()-t0:6.1f}s] {msg}", flush=True)
r = run_selfplay_games(m, cfg, num_games=num_games, output_dir="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_smoke/posps",
                       epoch=0, device="cuda", fp16=True, base_seed=1, active_games=active,
                       virtual_batch_size=vbatch, collect_examples=0, progress=cb)
print(f">>> FINAL td{td}/gnn{gnn} pcr={pcr}: searched={r.searched_positions} recorded={r.recorded_positions} "
      f"pos/s={r.positions_per_second:.1f} search_pos/s={r.search_positions_per_second:.1f} "
      f"mean_visits/move={r.mean_search_visits:.0f} wall={time.perf_counter()-t0:.1f}s", flush=True)
