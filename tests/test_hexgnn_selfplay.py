"""CPU end-to-end mini self-play: drive the native MCTS + hexgnn evaluator + TSS
machinery for a couple of games at tiny visits and confirm it writes sane compact
shards the trainer can read back (and a CPU train pass over them lowers loss).

This exercises the whole transformer-free stack: rust_bridge (shared native
hexgt), HexgnnInference.evaluate_featurized_batch, HexgnnMctsSession, the
dense_cnn-shared finalize/compact-shard write path with horizons=(), and
HexgnnTrainer.train_on_shards -> expand -> hexgnn_loss.
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
_HEXGNN = ROOT / "packages" / "hexgnn" / "python"
if str(_HEXGNN) not in sys.path:
    sys.path.insert(0, str(_HEXGNN))


def _torch():
    return pytest.importorskip("torch")


def _cfg():
    from hexgnn.config import parse_hexgnn_config

    return parse_hexgnn_config(
        {
            "device": "cpu",
            "architecture": {"candidate_radius": 3, "token_dim": 32, "gnn_layers": 2},
            "selfplay": {
                "search_visits": 8,
                "max_actions": 40,
                "active_games": 2,
                "widening_max_children": 16,
                "root_dirichlet_noise_enabled": True,
                "root_dirichlet_total_alpha": 6.6,
                "root_dirichlet_noise_fraction": 0.25,
            },
        }
    )


def test_cpu_mini_selfplay_writes_sane_shards(tmp_path) -> None:
    torch = _torch()
    from hexgnn.architecture import HexgnnNetwork
    from hexgnn.selfplay import run_selfplay_games
    from hexo_models.dense_cnn.compact_io import read_compact_shard

    cfg = _cfg()
    model = HexgnnNetwork(token_dim=cfg.architecture.token_dim, gnn_layers=cfg.architecture.gnn_layers)
    model.eval()

    result = run_selfplay_games(
        model,
        cfg,
        num_games=2,
        output_dir=tmp_path,
        epoch=0,
        device="cpu",
        fp16=False,
        base_seed=123,
        active_games=2,
        virtual_batch_size=2,
        collect_examples=1,
    )
    assert result.completed_games + result.truncated_games == 2
    assert result.searched_positions > 0
    assert result.shard_paths, "no compact shards were written"

    rows = []
    for sp in result.shard_paths:
        rows.extend(read_compact_shard(Path(sp)))
    assert rows, "shards contained no rows"
    for r in rows[:50]:
        assert -1.0 <= float(r.value) <= 1.0
        assert len(list(r.policy)) > 0
        # hexgnn writes NO short-term-value targets.
        assert len(list(getattr(r, "short_term_value", ()))) == 0

    # The shards must train: a few CPU steps should lower the loss.
    from hexgnn.trainer import HexgnnTrainer

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = HexgnnTrainer(model=model, config=cfg, optimizer=opt)
    hist = trainer.train_on_shards([Path(p) for p in result.shard_paths], batch_size=16, max_steps=12)
    assert hist, "no training steps ran over the self-play shards"
    assert all(k in hist[0] for k in ("total", "policy", "value", "opp_policy"))
