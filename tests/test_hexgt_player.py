"""hexgt runner-player + head-to-head harness wiring (Phase 5+ stubs -> real).

Validates that a `HexgtPlayer` plugs into the `hexo_runner` match driver and
that `run_head_to_head` plays complete games and tallies a score, plus that the
deterministic eval path (greedy, no noise, fixed seed, vbatch=1) is repeatable.
Tiny CPU model + small search so it stays a fast suite gate.
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


def _tiny_config():
    from hexo_models.hexgt.config import (
        HexgtArchitectureConfig,
        HexgtConfig,
        HexgtSelfPlayConfig,
    )

    return HexgtConfig(
        architecture=HexgtArchitectureConfig(token_dim=48, gnn_layers=2, ctx_layers=2, ffn_dim=64, candidate_radius=2),
        selfplay=HexgtSelfPlayConfig(search_visits=4, widening_max_children=8, widening_min_children=2),
    )


def _tiny_model(cfg):
    from hexo_models.hexgt.architecture import HexgtNetwork

    a = cfg.architecture
    return HexgtNetwork(
        token_dim=a.token_dim, gnn_layers=a.gnn_layers, ctx_layers=a.ctx_layers,
        attention_heads=a.attention_heads, ffn_dim=a.ffn_dim,
    ).cpu().eval()


def test_head_to_head_completes_and_scores(tmp_path) -> None:
    _torch()
    from hexo_models.hexgt.evaluation import make_hexgt_factory, run_head_to_head

    cfg = _tiny_config()
    model = _tiny_model(cfg)
    make = make_hexgt_factory(model, cfg, device="cpu", fp16=False, virtual_batch_size=1)

    result = run_head_to_head(make, make, games=2, output_dir=tmp_path, base_seed=7, max_actions=24)

    assert result.games == 2
    # every game is either decided (win/loss) or a draw/timeout
    assert result.wins + result.losses + result.draws == 2
    assert 0.0 <= result.win_rate <= 1.0
    assert result.mean_turns > 0


def test_eval_path_is_deterministic(tmp_path) -> None:
    _torch()
    from hexo_models.hexgt.evaluation import make_hexgt_factory, run_head_to_head

    cfg = _tiny_config()
    model = _tiny_model(cfg)
    # vbatch=1 -> sequential search -> repeatable greedy/no-noise eval.
    make = make_hexgt_factory(model, cfg, device="cpu", fp16=False, virtual_batch_size=1)

    r1 = run_head_to_head(make, make, games=2, output_dir=tmp_path / "a", base_seed=11, max_actions=24)
    r2 = run_head_to_head(make, make, games=2, output_dir=tmp_path / "b", base_seed=11, max_actions=24)

    assert (r1.wins, r1.losses, r1.draws) == (r2.wins, r2.losses, r2.draws)
    assert r1.mean_turns == r2.mean_turns
