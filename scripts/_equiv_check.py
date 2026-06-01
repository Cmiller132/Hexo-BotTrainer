#!/usr/bin/env python3
"""Prove the sample compaction is functionally identical.

For real searched positions, build the NEW compact sample (byte-backed policies +
array legal_action_ids) and a reference sample holding the SAME values in the OLD
representation (tuples of Python (int,float) / ints), then expand BOTH through
expand_sample under every D6 symmetry and assert all training tensors are
bit-identical (torch.equal). If they match, the NPZ rows fed to training are
unchanged."""

from __future__ import annotations

import os
from dataclasses import replace

import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference
from hexo_models.dense_cnn.mcts import new_mcts_session
from hexo_models.dense_cnn.samples import sample_from_state, finalize_game_samples, expand_sample
from hexo_models.dense_cnn.d6 import D6Symmetry, D6_SIZE
from hexo_models.dense_cnn.constants import INPUT_CHANNELS


def to_old_repr(sample):
    """Reconstruct the pre-compaction representation (tuples) from a compact sample,
    preserving exact values, so expansion can be compared storage-type vs storage-type."""
    return replace(
        sample,
        policy=tuple((int(a), float(w)) for a, w in sample.policy),
        root_prior_policy=tuple((int(a), float(w)) for a, w in sample.root_prior_policy),
        opp_policy=tuple((int(a), float(w)) for a, w in sample.opp_policy),
        legal_action_ids=tuple(int(x) for x in sample.legal_action_ids),
    )


def main() -> None:
    use_trt = os.environ.get("HEXO_TRT", "").strip() in ("1", "true", "True")
    net = Model1Network(in_channels=INPUT_CHANNELS, channels=64, blocks=4,
                        short_term_value_horizons=(1, 4, 8))
    inf = DenseCNNInference(net, device="cuda", amp=True, return_logits=False,
                            max_batch_size=1024, use_trt=use_trt,
                            bucket_pad_multiple=16, trt_allow_fallback=True)
    sess = new_mcts_session(max_states=131072)

    st = engine.new_game(seed=777)
    pending = []
    checked = 0
    mism = 0
    for ply in range(40):
        if engine.terminal(st) is not None:
            break
        res = sess.run([0], [st], inf, visits=256, c_puct=1.5, temperature=1.0, seed=7,
                       virtual_batch_size=4, active_root_limit=256,
                       root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
                       root_policy_temperature=1.1, fpu_reduction=0.20, virtual_loss=1.0,
                       widening_policy_mass=0.95, widening_max_children=32,
                       widening_min_children=2, forced_playout_k=2.0,
                       move_temperatures=[1.0])[0]
        s = sample_from_state(st, game_id="eq", turn_index=ply,
                              policy=res.visit_policy, root_prior_policy=res.root_prior_policy,
                              metadata={"epoch": 1})
        pending.append((s.current_player, s, res.root_value))
        engine.apply_action(st, engine.PlacementAction(unpack_coord_id(res.action_id)))

    # Finalize (fills value/opp_policy/short_term_value) — exercise the full row.
    finalized = finalize_game_samples(pending, winner="player0", horizons=(1, 4, 8))
    for s in finalized:
        old = to_old_repr(s)
        for sym in range(D6_SIZE):
            new_t = expand_sample(s, symmetry=D6Symmetry(sym))
            old_t = expand_sample(old, symmetry=D6Symmetry(sym))
            assert set(new_t) == set(old_t), f"key mismatch {set(new_t) ^ set(old_t)}"
            for k in new_t:
                checked += 1
                if not torch.equal(new_t[k], old_t[k]):
                    mism += 1
                    d = (new_t[k].float() - old_t[k].float()).abs().max().item()
                    print(f"  MISMATCH ply={s.turn_index} sym={sym} key={k} maxabs={d}")
    print(f"[equiv] compared {checked} tensors across {len(finalized)} samples x {D6_SIZE} symmetries")
    print(f"[equiv] VERDICT: {'IDENTICAL — compaction is functionally lossless' if mism == 0 else f'{mism} MISMATCHES'}")


if __name__ == "__main__":
    main()
