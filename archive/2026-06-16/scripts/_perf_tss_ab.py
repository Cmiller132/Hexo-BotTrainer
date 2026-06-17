"""A/B self-play throughput harness for the TSS no-scan short-circuit.

Drives the REAL production self-play path (`run_selfplay_games`) with the live
main3 config + pretrained seed, so pos/s is directly comparable to the run. The
ONLY thing that differs between two invocations is the env var
`HEXGT_TSS_NOSCAN_FASTPATH` (1 = O(1) has_threats() short-circuit, 0 = legacy full
`threats()` scan in every position). Same seed -> byte-identical games -> a clean
controlled measurement of the threat-scan cost.

Usage (run twice, env toggled):
  HEXGT_TSS_NOSCAN_FASTPATH=0 python scripts/_perf_tss_ab.py --games 64 --tag BEFORE
  HEXGT_TSS_NOSCAN_FASTPATH=1 python scripts/_perf_tss_ab.py --games 64 --tag AFTER
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.constants import FEATURE_SCHEMA_VERSION, feature_slots_after
from hexo_models.hexgt.selfplay import run_selfplay_games

# Reuse the driver's exact model-build / graft helpers so the network matches the run.
import _rl_train as drv


def _gpu_util_sampler(stop_evt, out):
    import subprocess
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            out.append(int(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        stop_evt.wait(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-ckpt", default=str(ROOT / "runs/hexgt_rl_main3/pretrain/hexgt_model3_pretrain.pt"))
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--active", type=int, default=64)
    ap.add_argument("--vbatch", type=int, default=64)
    ap.add_argument("--visits", type=int, default=512)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--tag", default="RUN")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    fastpath = os.environ.get("HEXGT_TSS_NOSCAN_FASTPATH", "1") != "0"

    # cfg mirrors _rl_launch_main3.sh / configs/hexgt_model3.toml selfplay knobs.
    cfg = parse_hexgt_config({
        "device": str(device),
        "architecture": {"candidate_radius": args.n,
                         "short_term_value_horizons": list(drv.STV_HORIZONS)},
        "selfplay": {
            "search_visits": args.visits, "max_actions": 512, "c_puct": 1.5,
            "root_policy_temperature": 1.0, "root_dirichlet_noise_enabled": True,
            "root_dirichlet_total_alpha": 6.6, "root_dirichlet_noise_fraction": 0.25,
            "temperature": 1.0, "final_temperature": 0.2, "temperature_decay_moves": 30,
            "temperature_floor": 0.1, "forced_playout_k": 2.0,
            "widening_max_children": 96, "widening_min_children": 2,
            "widening_policy_mass": 0.95,
        },
    })

    ck = torch.load(args.seed_ckpt, map_location=device, weights_only=False)
    arch_meta = dict(ck["arch"]); arch_meta["short_term_value_horizons"] = list(drv.STV_HORIZONS)
    model = drv.build_model(arch_meta, device)
    drv.expand_value_readout_columns(model, ck["model"])
    model.load_state_dict(ck["model"], strict=False)
    loaded_fsv = int(ck.get("feature_schema_version", 1))
    if loaded_fsv < FEATURE_SCHEMA_VERSION:
        drv.zero_init_expanded_feature_columns(model, feature_slots_after(loaded_fsv))
    model.eval()
    if not args.no_compile and device.type == "cuda":
        model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)

    out_dir = ROOT / "runs" / "_perf_ab_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm one self-play batch (compile + cache fill) BEFORE the timed run so the
    # measured pos/s is steady-state, not warmup-dominated.
    _ = run_selfplay_games(model, cfg, num_games=args.active, output_dir=out_dir,
                           epoch=9000, device=str(device), fp16=(device.type == "cuda"),
                           base_seed=1, active_games=args.active, virtual_batch_size=args.vbatch,
                           collect_examples=0)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    stop_evt = threading.Event(); utils = []
    t = threading.Thread(target=_gpu_util_sampler, args=(stop_evt, utils), daemon=True); t.start()

    t0 = time.perf_counter()
    sp = run_selfplay_games(model, cfg, num_games=args.games, output_dir=out_dir,
                            epoch=9001, device=str(device), fp16=(device.type == "cuda"),
                            base_seed=12345, active_games=args.active, virtual_batch_size=args.vbatch,
                            collect_examples=0)
    wall = time.perf_counter() - t0
    stop_evt.set(); t.join(timeout=1)

    mean_util = sum(utils) / len(utils) if utils else float("nan")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    print(f"[{args.tag}] fastpath={'ON' if fastpath else 'OFF'} "
          f"games={sp.completed_games}C/{sp.truncated_games}T searched={sp.searched_positions} "
          f"wall={wall:.1f}s | pos/s={sp.positions_per_second:.2f} "
          f"| GPU_util_mean={mean_util:.0f}% peak_alloc={peak_gb*1000:.0f}MB "
          f"| len_med={sp.game_length_median:.0f} forced={sp.forced_move_fraction:.1%}")
    print("PERF_AB_DONE")


if __name__ == "__main__":
    main()
