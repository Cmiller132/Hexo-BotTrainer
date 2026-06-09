"""GPU self-play throughput: hexgnn vs hexgt model-3 at the PRODUCTION config
(visits=1024, PCR p=0.5 / fast=170, vbatch=16, active=128), compiled + fp16.

Runs each model's own run_selfplay_games (random weights — forward cost is
weight-independent), compiling forward_policy_value like the driver does. Reports
measured positions/sec (searched + search-only) and the speedup. A progress
callback prints running pos/s so a long run can be read/aborted mid-flight.

argv: [num_games] [active] [max_actions] [visits]   (defaults below)
"""
import sys
import time

import torch

NUM_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 128
ACTIVE = int(sys.argv[2]) if len(sys.argv) > 2 else 128
MAX_ACTIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
VISITS = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
VBATCH = 16
OUT = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_smoke/gpu_bench"

SP_KNOBS = dict(
    search_visits=VISITS, active_games=ACTIVE, mcts_active_root_limit=ACTIVE,
    c_puct=1.5, root_dirichlet_noise_enabled=True, root_dirichlet_total_alpha=6.6,
    root_dirichlet_noise_fraction=0.30, root_policy_temperature=1.0,
    widening_max_children=96, widening_min_children=2, widening_policy_mass=0.95,
    max_actions=MAX_ACTIONS, temperature=1.0, temperature_floor=0.3,
    temperature_halflife=33, forced_playout_k=2.0,
    pcr_enabled=True, pcr_full_proportion=0.5, pcr_fast_visits=170,
)


def progress(tag):
    def cb(msg):
        print(f"[{tag}] {msg}", flush=True)
    return cb


def run_model3():
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.config import parse_hexgt_config
    from hexo_models.hexgt.selfplay import run_selfplay_games
    cfg = parse_hexgt_config({
        "device": "cuda",
        "architecture": {"candidate_radius": 3, "token_dim": 168, "gnn_layers": 4,
                         "ctx_layers": 3, "ffn_dim": 336, "attention_heads": 4,
                         "short_term_value_horizons": [4, 12, 24], "value_pma_seeds": 2},
        "samples": {"soft_z_lambda": 0.0},
        "selfplay": SP_KNOBS,
    })
    m = HexgtNetwork(token_dim=168, gnn_layers=4, ctx_layers=3, attention_heads=4, ffn_dim=336,
                     short_term_value_horizons=(4, 12, 24), value_pma_seeds=2).to("cuda").eval()
    m.forward_policy_value = torch.compile(m.forward_policy_value, dynamic=True)
    p = sum(q.numel() for q in m.parameters())
    t0 = time.perf_counter()
    r = run_selfplay_games(m, cfg, num_games=NUM_GAMES, output_dir=OUT + "_m3", epoch=0,
                           device="cuda", fp16=True, base_seed=1, active_games=ACTIVE,
                           virtual_batch_size=VBATCH, collect_examples=0, progress=progress("m3"))
    return p, r, time.perf_counter() - t0


def run_hexgnn():
    from hexgnn.architecture import HexgnnNetwork
    from hexgnn.config import parse_hexgnn_config
    from hexgnn.selfplay import run_selfplay_games
    cfg = parse_hexgnn_config({
        "device": "cuda",
        "architecture": {"candidate_radius": 3, "token_dim": 168, "gnn_layers": 4,
                         "attention_heads": 4, "value_pma_seeds": 2},
        "samples": {"soft_z_lambda": 0.0},
        "selfplay": SP_KNOBS,
    })
    m = HexgnnNetwork().to("cuda").eval()
    m.forward_policy_value = torch.compile(m.forward_policy_value, dynamic=True)
    p = sum(q.numel() for q in m.parameters())
    t0 = time.perf_counter()
    r = run_selfplay_games(m, cfg, num_games=NUM_GAMES, output_dir=OUT + "_gnn", epoch=0,
                           device="cuda", fp16=True, base_seed=1, active_games=ACTIVE,
                           virtual_batch_size=VBATCH, collect_examples=0, progress=progress("gnn"))
    return p, r, time.perf_counter() - t0


def main():
    print(f"config: num_games={NUM_GAMES} active={ACTIVE} max_actions={MAX_ACTIONS} "
          f"visits={VISITS} vbatch={VBATCH} PCR(p=0.5,fast=170) compiled+fp16", flush=True)
    p_m3, r_m3, w_m3 = run_model3()
    print(f">>> model-3: params={p_m3:,} searched={r_m3.searched_positions} "
          f"pos/s={r_m3.positions_per_second:.2f} search_pos/s={r_m3.search_positions_per_second:.2f} "
          f"wall={w_m3:.1f}s mean_visits/move={r_m3.mean_search_visits:.0f}", flush=True)
    p_g, r_g, w_g = run_hexgnn()
    print(f">>> hexgnn : params={p_g:,} searched={r_g.searched_positions} "
          f"pos/s={r_g.positions_per_second:.2f} search_pos/s={r_g.search_positions_per_second:.2f} "
          f"wall={w_g:.1f}s mean_visits/move={r_g.mean_search_visits:.0f}", flush=True)
    print(f">>> SELF-PLAY SPEEDUP (pos/s) hexgnn / model-3 = "
          f"{r_g.positions_per_second / max(1e-9, r_m3.positions_per_second):.2f}x", flush=True)


if __name__ == "__main__":
    main()
