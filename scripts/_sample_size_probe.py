#!/usr/bin/env python3
"""Measure exactly what makes a pending self-play sample ~320 KB.

Plays one game to N plies and, at several depths, builds the real Model1SampleData
(via sample_from_state) and reports the recursive (deep) byte size of each field.
Reveals which fields dominate and how they scale with turn_index."""

from __future__ import annotations

import os
import sys
from dataclasses import fields

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

from hexo_models.dense_cnn.samples import sample_from_state, Model1SampleData
from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference
from hexo_models.dense_cnn.mcts import new_mcts_session
from hexo_models.dense_cnn.constants import INPUT_CHANNELS


def deep_size(obj, seen=None):
    """Recursive sizeof over the containers we actually use (tuple/list/dict/str/num)."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, (tuple, list)):
        size += sum(deep_size(x, seen) for x in obj)
    elif isinstance(obj, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    return size


def sample_deep_size(s: Model1SampleData):
    per = {}
    for f in fields(s):
        per[f.name] = deep_size(getattr(s, f.name))
    return per


def main() -> None:
    use_trt = os.environ.get("HEXO_TRT", "").strip() in ("1", "true", "True")
    net = Model1Network(in_channels=INPUT_CHANNELS, channels=64, blocks=4,
                        short_term_value_horizons=(1, 4, 8))
    inf = DenseCNNInference(net, device="cuda", amp=True, return_logits=False,
                            max_batch_size=1024, use_trt=use_trt,
                            bucket_pad_multiple=16, trt_allow_fallback=True)
    sess = new_mcts_session(max_states=131072)

    st = engine.new_game(seed=12345)
    depths = {10, 30, 60, 100, 150, 200}
    maxd = max(depths)
    ply = 0
    while ply < maxd and engine.terminal(st) is None:
        ply += 1
        # REAL MCTS search to get the production visit_policy / root_prior_policy.
        res = sess.run([0], [st], inf, visits=256, c_puct=1.5, temperature=1.0, seed=7,
                       virtual_batch_size=4, active_root_limit=256,
                       root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
                       root_policy_temperature=1.1, fpu_reduction=0.20, virtual_loss=1.0,
                       widening_policy_mass=0.95, widening_max_children=32,
                       widening_min_children=2, forced_playout_k=2.0,
                       move_temperatures=[1.0])[0]
        if ply in depths:
            s = sample_from_state(st, game_id="probe", turn_index=ply,
                                  policy=res.visit_policy, root_prior_policy=res.root_prior_policy,
                                  metadata={"epoch": 1})
            per = sample_deep_size(s)
            total = sum(per.values())
            print(f"\n=== ply {ply}: total deep size = {total/1024:.1f} KB  "
                  f"(stones={len(s.stones)}, hist={len(s.placement_history)}, "
                  f"legal={len(s.legal_action_ids)}, visit_pol={len(s.policy)}, "
                  f"root_prior={len(s.root_prior_policy)}) ===")
            for name, sz in sorted(per.items(), key=lambda kv: -kv[1]):
                if sz > 200:
                    print(f"   {name:22s} {sz/1024:8.1f} KB  ({sz/max(total,1)*100:4.1f}%)")
        engine.apply_action(st, engine.PlacementAction(unpack_coord_id(res.action_id)))
    print("\n[done]")


if __name__ == "__main__":
    main()
