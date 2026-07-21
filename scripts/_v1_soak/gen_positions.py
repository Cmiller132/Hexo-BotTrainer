"""V1 SOAK phase 1 (GPU): fresh in-regime position generation at ep90.

Runs the production continuous MCTS self-play scheduler with the ep90 net at the
run's 256-visit search config, and records every ROOT position it plays through
(the on-policy trajectory the trainer's root/leaf channels would solve), tagged
with the net's root value and top-K prior for the §8 internalization baseline
and the §9 net-vs-proof calibration.

Deep-solver CONSUMPTION is disabled for generation (mode=0, no root guard, no
async/park) so trajectories are deterministic and generation is fast; the
play-shaping interior guard (Lever 0, live in main_3) is retained. The positions
are the measurement slice; the solver economics are measured offline in phase 2.

Usage:
    python gen_positions.py <out.jsonl> <n_games> [max_plies]
"""

from __future__ import annotations

import arch_env  # noqa: F401  MUST precede any hexfield_eq import (sets NUM_FEATURES etc.)

import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield_eq import _rust
from hexfield_eq.config import build_divergence_overrides, parse_hexfield_config
from hexfield_eq.geometry import unpack_action_id
from hexfield_eq.inference import build_serve_evaluator
from hexfield_eq.serve_env import prime_serve_env

RUN_DIR = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_3")
CKPT = RUN_DIR / "checkpoints" / "epoch_000090.pt"
TOPK_PRIOR = 24


def load_net_and_cfg():
    prime_serve_env()
    from hexfield_eq import eval_arena  # after prime_serve_env, before model import

    raw = tomllib.load(open(RUN_DIR / "_resume_config.toml", "rb"))
    cfg = parse_hexfield_config(raw["model"]["config"])
    net = eval_arena._load_hexfield_net(CKPT)
    return net, cfg


def main():
    out_path = Path(sys.argv[1])
    n_games = int(sys.argv[2])
    max_plies = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    # Search-visit budget used only to SOURCE phase-diverse positions. Lower than
    # the 256 training regime purely for throughput (reaching late-game positions
    # in minutes not an hour); the deep-solve measurement — the actual object —
    # runs identically on the resulting states regardless of how they were
    # reached. Documented as a generation-only deviation in the report.
    gen_visits = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    # Optional 5th arg "tss_on": enable the play-shaping interior guard, for the
    # paired guard-on vs guard-off throughput A/B (same seed, same config).
    tss_on = len(sys.argv) > 5 and sys.argv[5] == "tss_on"

    net, cfg = load_net_and_cfg()
    sp = cfg.selfplay
    evaluator = build_serve_evaluator(net, cfg, role="selfplay", auto_match_serve_env=True)

    # Production divergence map, minus deep-solver consumption (see module doc).
    div = build_divergence_overrides(sp)
    div = dict(div)
    div["tss_solver_mode"] = 0
    div["tss_solver_root_guard"] = False
    div["tss_solver_async"] = False
    div["tss_solver_park"] = False

    states = {k: api.new_game() for k in range(n_games)}
    plies = {k: 0 for k in range(n_games)}
    moves = {k: [] for k in range(n_games)}
    out_fh = open(out_path, "w")  # incremental write: crash/kill keeps partials
    n_written = [0]

    def on_move(game_key, payload):
        st = states[game_key]
        # The search ran on `st` (the root == current move list). Record it.
        rv = float(payload.get("root_value", 0.0))
        prior = []
        if "root_prior_policy_action_ids_bytes" in payload:
            ids = np.frombuffer(bytes(payload["root_prior_policy_action_ids_bytes"]), dtype=np.uint32)
            ws = np.frombuffer(bytes(payload["root_prior_policy_weights_bytes"]), dtype=np.float32)
            order = np.argsort(-ws)[:TOPK_PRIOR]
            prior = [[int(ids[i]), float(ws[i])] for i in order]
        played = int(payload["action_id"])
        out_fh.write(json.dumps(
            {
                "id": f"sp_{game_key}_p{plies[game_key]}",
                "source": "selfplay",
                "game": int(game_key),
                "ply": plies[game_key],
                "placements": len(moves[game_key]),
                "moves": list(moves[game_key]),
                "net_value": rv,
                "played": played,
                "prior": prior,
            }
        ) + "\n")
        n_written[0] += 1
        if n_written[0] % 500 == 0:
            out_fh.flush()
        q, r = unpack_action_id(played)
        res = api.apply_action(st, PlacementAction(AxialCoord(q=int(q), r=int(r))))
        moves[game_key].append([int(q), int(r)])
        plies[game_key] += 1
        if res is None or api.terminal(st) is not None or plies[game_key] >= max_plies:
            return None
        return ("advance", st)

    session = _rust.HexfieldMctsSession(max_states=sp.cache_max_states)
    torch.cuda.synchronize()
    t0 = time.time()
    noise = {}
    if sp.root_dirichlet_noise_fraction > 0:
        noise = dict(
            root_dirichlet_total_alpha=sp.root_dirichlet_total_alpha,
            root_dirichlet_noise_fraction=sp.root_dirichlet_noise_fraction,
        )
    session.run_continuous(
        list(states.keys()),
        tuple(states.values()),
        evaluator=evaluator,
        on_move=on_move,
        visits=gen_visits,
        c_puct=sp.c_puct,
        base_seed=20260720,
        virtual_batch_size=sp.virtual_batch_size,
        flush_target=sp.flush_target,
        active_root_limit=min(sp.active_root_limit, n_games),
        temperature_by_ply=cfg.temperature_by_ply(),
        root_policy_temperature=sp.root_policy_temperature,
        root_policy_temperature_early=sp.root_policy_temperature_early or None,
        root_policy_temperature_halflife=sp.root_policy_temperature_halflife or None,
        fpu_reduction=sp.fpu_reduction,
        virtual_loss=sp.virtual_loss,
        widening_policy_mass=sp.widening_policy_mass,
        widening_max_children=sp.widening_max_children,
        widening_min_children=sp.widening_min_children,
        forced_playout_k=sp.forced_playout_k,
        pcr_full_proportion=sp.pcr_full_proportion,
        pcr_fast_visits=sp.pcr_fast_visits,
        pcr_fast_temperature=sp.pcr_fast_temperature,
        policy_init_fraction=sp.policy_init_fraction,
        policy_init_avg_plies=sp.policy_init_avg_plies,
        policy_init_max_plies=sp.policy_init_max_plies,
        policy_init_temperature=sp.policy_init_temperature,
        # Generation runs with the TSS path OFF. NOTE (corrected): measured
        # guard-on vs guard-off early-window rates were ~near parity (~118 vs
        # ~131 records/min), NOT a large speedup — the throughput lever that
        # actually mattered was lowering the search-visit budget (gen_visits,
        # 256->48). Guard-off is kept only for generation determinism/simplicity;
        # it shifts play only at proven-tactical roots (rare), a minor position-
        # distribution deviation documented in the report. The measured object
        # is the solver on positions, not the play policy.
        tss_enabled=tss_on,
        root_fpu_reduction=sp.root_fpu_reduction,
        root_fpu_zero_under_noise=sp.root_fpu_zero_under_noise,
        search_parity_mode=sp.search_parity_mode,
        divergence_overrides=div,
        **noise,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0

    out_fh.flush()
    out_fh.close()
    finished = sum(1 for k in plies if api.terminal(states[k]) is not None)
    print(
        f"generated {n_written[0]} positions from {n_games} games "
        f"({finished} terminal) in {dt:.1f}s "
        f"({n_written[0] / dt * 60:.1f} moves/min, visits={gen_visits}, "
        f"tss={'on' if tss_on else 'off'}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
