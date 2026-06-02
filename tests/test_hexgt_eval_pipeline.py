"""Correctness gate for the featurize<->forward eval pipeline (mcts_eval.rs).

The eval splits a batch of unique leaves into chunks and overlaps featurizing the
next chunk (a GIL-free rayon worker) with the current chunk's forward + parse.
Chunking + pipelining must NOT change results: each graph is featurized and
forwarded independently (per-graph attention), so a search is bit-identical
regardless of the chunk size or pipeline depth. We force the multi-chunk path
with a small `HEXGT_EVAL_CHUNK_STATES` and compare against the single-chunk path.

Runs on CPU/fp32 (deterministic) so the comparison is exact, not approximate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hexo_engine as _eng_pkg  # noqa: F401  (ensure native module importable)
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
    # Deterministic CPU/fp32 evaluator so chunking differences are exact, and a
    # fixed seed so the random init is identical across runs in this process.
    torch.manual_seed(20260602)
    model = HexgtNetwork(short_term_value_horizons=()).to("cpu").eval()
    return HexgtInference(model, device="cpu", fp16=False)


def _run(inf, *, chunk_states, pipeline_depth, monkeypatch):
    # Several roots x a wide virtual batch => many unique leaves per round, so a
    # small chunk size genuinely splits into multiple pipelined chunks.
    monkeypatch.setenv("HEXGT_EVAL_CHUNK_STATES", str(chunk_states))
    monkeypatch.setenv("HEXGT_EVAL_PIPELINE_DEPTH", str(pipeline_depth))
    states = [_midgame_state(s) for s in (1, 2, 3, 4)]
    sess = new_mcts_session(max_states=1 << 16, n=3)
    return sess.run(
        list(range(len(states))), states, inf,
        visits=48, c_puct=1.5, seed=7, virtual_batch_size=8,
        widening_policy_mass=0.95, widening_max_children=8, widening_min_children=2,
        fpu_reduction=0.2, virtual_loss=1.0, root_policy_temperature=1.1,
    )


def _signature(results):
    return [
        (r.action_id, r.visits, round(r.root_value, 9),
         tuple((a, round(w, 9)) for a, w in r.visit_policy))
        for r in results
    ]


def test_multichunk_pipeline_matches_single_chunk(monkeypatch):
    inf = _inference()
    # Single chunk: all leaves in one featurize+forward (no pipeline).
    single = _signature(_run(inf, chunk_states=100000, pipeline_depth=2, monkeypatch=monkeypatch))
    # Many small chunks => the producer/consumer pipeline is exercised heavily.
    multi = _signature(_run(inf, chunk_states=4, pipeline_depth=2, monkeypatch=monkeypatch))
    assert multi == single, "pipelined multi-chunk eval must be bit-identical to single-chunk"


def test_pipeline_depth_is_result_invariant(monkeypatch):
    inf = _inference()
    depth1 = _signature(_run(inf, chunk_states=4, pipeline_depth=1, monkeypatch=monkeypatch))
    depth4 = _signature(_run(inf, chunk_states=4, pipeline_depth=4, monkeypatch=monkeypatch))
    assert depth1 == depth4, "pipeline depth must not change search results"


def test_multichunk_pipeline_selects_legal_actions(monkeypatch):
    inf = _inference()
    monkeypatch.setenv("HEXGT_EVAL_CHUNK_STATES", "4")
    states = [_midgame_state(s) for s in (5, 6, 7)]
    sess = new_mcts_session(max_states=1 << 16, n=3)
    from hexo_engine.types import unpack_coord_id
    results = sess.run(
        list(range(len(states))), states, inf,
        visits=48, c_puct=1.5, seed=11, virtual_batch_size=8,
        widening_policy_mass=0.95, widening_max_children=8, widening_min_children=2,
        fpu_reduction=0.2, virtual_loss=1.0, root_policy_temperature=1.1,
    )
    for s, res in zip(states, results):
        assert eng.PlacementAction(unpack_coord_id(res.action_id)) in list(eng.legal_actions(s))
        assert res.visits > 0
