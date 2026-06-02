"""Parallel (batched) head-to-head reproduces the sequential driver exactly.

The parallel eval steps many games at once, searching each side's batch of
positions in one forward. Because eval is deterministic (greedy, no Dirichlet,
fixed per-game seed), the per-game result must be invariant to the async batch
composition. This test pins that: sequential `run_head_to_head` and
`run_head_to_head_parallel` must return identical W/L/D + mean turns on the same
games. Tiny CPU model + small search so it is a fast gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _torch():
    return pytest.importorskip("torch")


def _cfg_and_model():
    from hexo_models.hexgt.config import HexgtArchitectureConfig, HexgtConfig, HexgtSelfPlayConfig
    from hexo_models.hexgt.architecture import HexgtNetwork

    cfg = HexgtConfig(
        architecture=HexgtArchitectureConfig(token_dim=48, gnn_layers=2, ctx_layers=2, ffn_dim=64, candidate_radius=2),
        selfplay=HexgtSelfPlayConfig(search_visits=8, widening_max_children=8, widening_min_children=2),
    )
    a = cfg.architecture
    model = HexgtNetwork(
        token_dim=a.token_dim, gnn_layers=a.gnn_layers, ctx_layers=a.ctx_layers,
        attention_heads=a.attention_heads, ffn_dim=a.ffn_dim,
    ).cpu().eval()
    return cfg, model


def test_parallel_matches_sequential(tmp_path) -> None:
    _torch()
    from hexo_models.hexgt.evaluation import (
        HexgtBatchedSearcher,
        make_hexgt_factory,
        run_head_to_head,
        run_head_to_head_parallel,
    )

    cfg, model = _cfg_and_model()
    visits = cfg.selfplay.search_visits
    games, seed, max_actions = 6, 31, 30

    # Sequential: deterministic HexgtPlayer factories (vbatch=1 -> sequential search).
    make = make_hexgt_factory(model, cfg, device="cpu", fp16=False, virtual_batch_size=1)
    seq = run_head_to_head(make, make, games=games, output_dir=tmp_path, base_seed=seed, max_actions=max_actions)

    # Parallel: one batched searcher per side, all games stepped together.
    sa = HexgtBatchedSearcher(model, cfg, device="cpu", fp16=False, visits=visits)
    sb = HexgtBatchedSearcher(model, cfg, device="cpu", fp16=False, visits=visits)
    par = run_head_to_head_parallel(sa, sb, games=games, base_seed=seed, max_actions=max_actions)

    assert (par.wins, par.losses, par.draws) == (seq.wins, seq.losses, seq.draws), (
        f"parallel {(par.wins, par.losses, par.draws)} != sequential {(seq.wins, seq.losses, seq.draws)}"
    )
    assert par.mean_turns == seq.mean_turns
    assert par.games == seq.games == games


def test_parallel_concurrency_cap_invariant(tmp_path) -> None:
    """Result is invariant to how many games run concurrently (active_limit)."""

    _torch()
    from hexo_models.hexgt.evaluation import HexgtBatchedSearcher, run_head_to_head_parallel

    cfg, model = _cfg_and_model()
    visits = cfg.selfplay.search_visits
    kw = dict(games=6, base_seed=17, max_actions=30)

    def play(limit):
        sa = HexgtBatchedSearcher(model, cfg, device="cpu", fp16=False, visits=visits)
        sb = HexgtBatchedSearcher(model, cfg, device="cpu", fp16=False, visits=visits)
        return run_head_to_head_parallel(sa, sb, active_limit=limit, **kw)

    r_full = play(6)
    r_small = play(2)
    assert (r_full.wins, r_full.losses, r_full.draws) == (r_small.wins, r_small.losses, r_small.draws)
    assert r_full.mean_turns == r_small.mean_turns
