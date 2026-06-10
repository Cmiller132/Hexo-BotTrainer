"""Per-root MCTS overrides — the single-call heterogeneous-cap path that coalesces
KataGo PCR's full/fast move mix into ONE full-width batched search (recovering the
throughput the two-call split lost).

Key correctness properties:
  * non-interference: an index-0 root's result is identical whether or not other
    (per-root-cap) batchmates share the call -- batching does not corrupt per-root
    search (the "bit-identical per root given its cap" property);
  * uniform per-root vectors == the scalar fallback (byte-identical generalization);
  * each root respects its own cap / forced-k / noise (caps honored, a fast root's
    exported policy is unaffected by another root's forced-playout pruning);
  * determinism; and input validation.
Tiny CPU model + few visits so it stays a fast gate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hexo_engine.api as eng
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session


def _midgame_state(seed: int, plies: int = 10):
    s = eng.new_game(seed=seed)
    for _ in range(plies):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        eng.apply_action(s, acts[len(acts) // 2])
    return s


def _inference():
    model = HexgtNetwork(short_term_value_horizons=()).to("cpu").eval()
    return HexgtInference(model, device="cpu", fp16=False)


def _kw(**over):
    base = dict(
        c_puct=1.5, seed=1, virtual_batch_size=8,
        widening_policy_mass=0.95, widening_max_children=8, widening_min_children=2,
        fpu_reduction=0.2, virtual_loss=1.0, root_policy_temperature=1.1,
    )
    base.update(over)
    return base


def _key(res):
    """Comparable, deterministic fingerprint of a SearchResult."""
    return (
        int(res.action_id),
        float(res.root_value),
        int(res.visits),
        tuple((int(a), float(w)) for a, w in res.visit_policy),
    )


def test_index0_root_result_is_invariant_to_batchmates():
    """Non-interference: the index-0 root (same seed -> same per-root randomness)
    produces an IDENTICAL result whether searched alone or alongside another root
    with a different cap. Batching coalesces forwards without changing per-root search."""
    inf = _inference()
    X, Y = _midgame_state(1), _midgame_state(2)

    alone = new_mcts_session(max_states=1 << 16, n=3).run(
        [0], [X], inf, **_kw(visits=8))[0]
    batched = new_mcts_session(max_states=1 << 16, n=3).run(
        [0, 1], [X, Y], inf, **_kw(visits=8, per_root_visits=[8, 16]))[0]

    assert _key(alone) == _key(batched), "index-0 root result changed when a batchmate was added"


def test_uniform_per_root_equals_scalar_fallback():
    """Passing per-root vectors filled with the scalar values reproduces the scalar
    path byte-for-byte (the per-root code is a faithful generalization)."""
    inf = _inference()
    states = [_midgame_state(3), _midgame_state(4)]

    scalar = new_mcts_session(max_states=1 << 16, n=3).run(
        [0, 1], states, inf, **_kw(visits=16, forced_playout_k=2.0,
                                   root_dirichlet_total_alpha=6.6, root_dirichlet_noise_fraction=0.3))
    per_root = new_mcts_session(max_states=1 << 16, n=3).run(
        [0, 1], states, inf, **_kw(visits=16, forced_playout_k=2.0,
                                   root_dirichlet_total_alpha=6.6, root_dirichlet_noise_fraction=0.3,
                                   per_root_visits=[16, 16], per_root_forced_playout_k=[2.0, 2.0],
                                   per_root_noise=[True, True]))
    assert [_key(r) for r in scalar] == [_key(r) for r in per_root]


def test_mixed_caps_are_each_respected():
    """Each root searches to its OWN cap in a single mixed call."""
    inf = _inference()
    res = new_mcts_session(max_states=1 << 16, n=3).run(
        [0, 1], [_midgame_state(5), _midgame_state(6)], inf,
        **_kw(visits=40, per_root_visits=[40, 6],
              per_root_forced_playout_k=[2.0, 0.0], per_root_noise=[True, False]))
    assert res[0].visits == 40, f"full root cap not respected: {res[0].visits}"
    assert res[1].visits == 6, f"fast root cap not respected: {res[1].visits}"


def test_fast_root_export_unaffected_by_other_root_forced_k():
    """Per-root forced-k isolation: a k=0 (fast) root's exported policy is identical
    whether the OTHER root has k=2.0 (pruning) or k=0 -- pruning is applied per-root,
    not leaked across the batch."""
    inf = _inference()
    states = [_midgame_state(7), _midgame_state(8)]
    common = dict(visits=24, per_root_visits=[24, 24], per_root_noise=[True, True])

    mixed = new_mcts_session(max_states=1 << 16, n=3).run(
        [0, 1], states, inf, **_kw(per_root_forced_playout_k=[2.0, 0.0], **common))
    allzero = new_mcts_session(max_states=1 << 16, n=3).run(
        [0, 1], states, inf, **_kw(per_root_forced_playout_k=[0.0, 0.0], **common))
    # Root 1 has k=0 in both -> identical export (root 0's k=2.0 must not leak in).
    assert _key(mixed[1]) == _key(allzero[1])


def test_per_root_call_is_deterministic():
    inf = _inference()
    states = [_midgame_state(9), _midgame_state(10)]
    kw = _kw(visits=20, per_root_visits=[20, 8],
             per_root_forced_playout_k=[2.0, 0.0], per_root_noise=[True, False])
    a = new_mcts_session(max_states=1 << 16, n=3).run([0, 1], states, inf, **kw)
    b = new_mcts_session(max_states=1 << 16, n=3).run([0, 1], states, inf, **kw)
    assert [_key(r) for r in a] == [_key(r) for r in b]


def test_per_root_length_and_value_validation():
    inf = _inference()
    states = [_midgame_state(11), _midgame_state(12)]
    sess = new_mcts_session(max_states=1 << 16, n=3)
    with pytest.raises(Exception):
        sess.run([0, 1], states, inf, **_kw(visits=8, per_root_visits=[8]))  # wrong length
    with pytest.raises(Exception):
        new_mcts_session(max_states=1 << 16, n=3).run(
            [0, 1], states, inf, **_kw(visits=8, per_root_visits=[8, 0]))    # zero cap
