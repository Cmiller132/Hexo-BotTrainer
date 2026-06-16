#!/usr/bin/env python3
"""Locate the self-play memory consumer: run 256 concurrent games through the real
native MCTS session for many sequential plies (as self-play does) and watch RSS vs
total tree-node count vs eval-cache size. Confirms whether the working set grows
with game length (tree reuse over long cold-start games) and which structure drives
it — so the mitigation (active_games / cache size / max_actions) is grounded."""

from __future__ import annotations

import os
import numpy as np
import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference
from hexo_models.dense_cnn.mcts import new_mcts_session
from hexo_models.dense_cnn.samples import sample_from_state
from hexo_models.dense_cnn.constants import INPUT_CHANNELS

N_GAMES = int(os.environ.get("PROBE_GAMES", "256"))
PLIES = int(os.environ.get("PROBE_PLIES", "60"))
CACHE = int(os.environ.get("PROBE_CACHE", "131072"))
PENDING = os.environ.get("PROBE_PENDING", "").strip() in ("1", "true", "True")
VISITS = 256


def rss_mb() -> float:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1.0


def main() -> None:
    use_trt = os.environ.get("HEXO_TRT", "").strip() in ("1", "true", "True")
    net = Model1Network(in_channels=INPUT_CHANNELS, channels=64, blocks=4,
                        short_term_value_horizons=(1, 4, 8))
    inf = DenseCNNInference(net, device="cuda", amp=True, return_logits=False,
                            max_batch_size=1024, use_trt=use_trt,
                            bucket_pad_multiple=16, trt_allow_fallback=True)
    games = [{"search_key": i, "game_id": f"g{i}", "state": engine.new_game(seed=9000 + i),
              "actions": [], "pending": []}
             for i in range(N_GAMES)]
    sess = new_mcts_session(max_states=CACHE)
    print(f"[grow] games={N_GAMES} plies={PLIES} cache={CACHE} visits={VISITS} "
          f"pending_accum={PENDING} trt={inf.trt_info.get('adopted')}")
    print(f"[grow] baseline RSS={rss_mb():.0f}MB")
    for ply in range(1, PLIES + 1):
        playable = [g for g in games if engine.terminal(g["state"]) is None]
        if not playable:
            print(f"[grow] all terminal at ply {ply}")
            break
        searches = sess.run(
            [g["search_key"] for g in playable], [g["state"] for g in playable], inf,
            visits=VISITS, c_puct=1.5, temperature=1.0, seed=7, virtual_batch_size=4,
            active_root_limit=256, root_dirichlet_total_alpha=10.83,
            root_dirichlet_noise_fraction=0.25, root_policy_temperature=1.1,
            fpu_reduction=0.20, virtual_loss=1.0, widening_policy_mass=0.95,
            widening_max_children=32, widening_min_children=2, forced_playout_k=2.0,
            move_temperatures=[1.0] * len(playable),
        )
        for g, s in zip(playable, searches):
            if PENDING:
                # Mirror selfplay.py: capture a compact sample per searched position
                # and retain it in the game's pending list until the game terminates.
                sample = sample_from_state(
                    g["state"], game_id=g["game_id"], turn_index=len(g["actions"]),
                    policy=s.visit_policy, root_prior_policy=s.root_prior_policy,
                    metadata={"epoch": 1, "search_visits": s.visits},
                )
                g["pending"].append((sample.current_player, sample, s.root_value))
            engine.apply_action(g["state"], engine.PlacementAction(unpack_coord_id(s.action_id)))
            g["actions"].append(s.action_id)
        if ply % 5 == 0 or ply == 1:
            tree = dict(dict(searches[0].diagnostics or {}).get("batch", {})).get("tree", {})
            ev = dict(dict(searches[0].diagnostics or {}).get("batch", {})).get("evaluation", {})
            print(f"  ply {ply:3d}  active={len(playable):3d}  RSS={rss_mb():6.0f}MB  "
                  f"sess_trees={len(sess):3d}  nodes={tree.get('node_count'):>9}  "
                  f"edge_bytes={int(tree.get('active_edge_bytes',0))//(1024*1024):4d}MB  "
                  f"cache={ev.get('cache_size'):>6}")
    print(f"[grow] end RSS={rss_mb():.0f}MB after {PLIES} plies")


if __name__ == "__main__":
    main()
