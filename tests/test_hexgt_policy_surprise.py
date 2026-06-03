"""KataGo policy-surprise weighting for hexgt self-play (Task 3).

hexgt reuses dense_cnn's `materialize_policy_surprise_rows` to repeat finalized
self-play positions by a frequency weight driven by KL(visit-policy || root-prior)
-- positions where the search most disagreed with the network prior carry more
training signal. These tests pin (a) the core weighting property and (b) the
self-play integration + on/off config gate.
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


def _sample(game_id, policy, prior):
    from hexo_models.dense_cnn.samples import Model1SampleData

    return Model1SampleData(
        game_id=game_id,
        turn_index=0,
        current_player="player0",
        phase="placement",
        center=(0, 0),
        stones=(),
        legal_action_ids=(),
        policy=tuple(policy),
        root_prior_policy=tuple(prior),
    )


def test_policy_surprise_weight_scales_with_kl() -> None:
    """A position whose visit policy disagrees with the prior (high KL) is
    up-weighted (frequency weight > 1) and duplicated at least as often as a
    position whose visits match the prior (KL ~ 0)."""

    from hexo_models.dense_cnn.replay import materialize_policy_surprise_rows

    # "low": visits == prior -> KL ~ 0.  "high": visits sharply disagree -> high KL.
    low = _sample("low", policy=[(1, 0.5), (2, 0.5)], prior=[(1, 0.5), (2, 0.5)])
    high = _sample("high", policy=[(1, 0.99), (2, 0.01)], prior=[(1, 0.01), (2, 0.99)])

    materialized, stats = materialize_policy_surprise_rows(
        [low, high], seed=1, uniform_fraction=0.5, max_weight=8.0
    )
    by_id: dict[str, list] = {}
    for row in materialized:
        by_id.setdefault(row.game_id, []).append(row)

    # The high-KL position has frequency weight 1.5 (>1) -> always >=1 copy, and it
    # carries a positive policy_surprise; the low-KL position weight is 0.5.
    assert "high" in by_id
    high_row = by_id["high"][0]
    assert high_row.frequency_weight > 1.0
    assert high_row.policy_surprise > 0.0
    assert stats["policy_surprise_mean"] > 0.0
    assert len(by_id.get("high", [])) >= len(by_id.get("low", []))


def _tiny_config(*, policy_surprise_enabled: bool):
    from hexo_models.hexgt.config import (
        HexgtArchitectureConfig,
        HexgtConfig,
        HexgtSampleConfig,
        HexgtSelfPlayConfig,
        HexgtTrainingConfig,
    )

    return HexgtConfig(
        architecture=HexgtArchitectureConfig(
            token_dim=48, gnn_layers=2, ctx_layers=2, ffn_dim=64, candidate_radius=2
        ),
        selfplay=HexgtSelfPlayConfig(
            search_visits=6, widening_max_children=8, widening_min_children=2, max_actions=16
        ),
        training=HexgtTrainingConfig(batch_size=8, warmup_steps=2, learning_rate=1e-3),
        samples=HexgtSampleConfig(policy_surprise_enabled=policy_surprise_enabled),
    )


def _tiny_model(cfg):
    from hexo_models.hexgt.architecture import HexgtNetwork

    a = cfg.architecture
    return HexgtNetwork(
        token_dim=a.token_dim, gnn_layers=a.gnn_layers, ctx_layers=a.ctx_layers,
        attention_heads=a.attention_heads, ffn_dim=a.ffn_dim,
    ).cpu().eval()


def test_selfplay_policy_surprise_on_off(tmp_path) -> None:
    """With the gate ON, self-play applies row duplication (effective == written
    rows, a populated frequency-weight mean) and still produces trainable shards;
    with it OFF, rows are written 1:1 and no surprise weighting is recorded."""

    _torch()
    from hexo_models.dense_cnn.compact_io import compact_row_count, read_compact_shard
    from hexo_models.hexgt.expand import build_training_batch
    from hexo_models.hexgt.selfplay import run_selfplay_games

    # ON
    cfg_on = _tiny_config(policy_surprise_enabled=True)
    model = _tiny_model(cfg_on)
    res_on = run_selfplay_games(
        model, cfg_on, num_games=2, output_dir=tmp_path / "on", epoch=0,
        device="cpu", fp16=False, base_seed=5, active_games=2, virtual_batch_size=2,
    )
    written_on = sum(compact_row_count(Path(p)) for p in res_on.shard_paths)
    assert res_on.effective_samples == written_on
    assert res_on.frequency_weight_mean > 0.0  # weighting ran (>=1.0 even at zero surprise)
    assert res_on.raw_samples > 0
    # Duplicated shards still decode + expand into a trainer batch.
    nonempty = [p for p in res_on.shard_paths if compact_row_count(Path(p)) > 0]
    assert nonempty, "policy-surprise self-play wrote only empty shards"
    rows = read_compact_shard(Path(nonempty[0]))
    batch, targets = build_training_batch(rows, n=cfg_on.architecture.candidate_radius, horizons=())
    assert batch is not None and int(batch["num_graphs"]) >= 1

    # OFF (default): rows written 1:1, no surprise recorded.
    cfg_off = _tiny_config(policy_surprise_enabled=False)
    model2 = _tiny_model(cfg_off)
    res_off = run_selfplay_games(
        model2, cfg_off, num_games=2, output_dir=tmp_path / "off", epoch=0,
        device="cpu", fp16=False, base_seed=5, active_games=2, virtual_batch_size=2,
    )
    written_off = sum(compact_row_count(Path(p)) for p in res_off.shard_paths)
    assert res_off.effective_samples == written_off == res_off.raw_samples
    assert res_off.policy_surprise_mean == 0.0
    assert res_off.frequency_weight_mean == 0.0
