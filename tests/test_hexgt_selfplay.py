"""hexgt game-driven self-play (`selfplay.py`) — the RL loop's data producer.

Validates that self-play writes the dense_cnn COMPACT shard format `expand.py`
reads, that those shards round-trip through the trainer's read path
(`build_training_batch`) AND through `HexgtTrainer.train_on_shards` (so a
self-play epoch genuinely closes the RL loop), and that play-style aggregates +
example traces are captured. Tiny CPU model + tiny search so it stays a fast gate.
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
        HexgtTrainingConfig,
    )

    return HexgtConfig(
        architecture=HexgtArchitectureConfig(
            token_dim=48, gnn_layers=2, ctx_layers=2, ffn_dim=64, candidate_radius=2
        ),
        selfplay=HexgtSelfPlayConfig(
            search_visits=6,
            widening_max_children=8,
            widening_min_children=2,
            max_actions=20,
            # Exercise the temperature decay + forced-playout paths in the loop.
            temperature=1.0,
            final_temperature=0.2,
            temperature_decay_moves=8,
            forced_playout_k=2.0,
        ),
        training=HexgtTrainingConfig(batch_size=8, warmup_steps=2, learning_rate=1e-3),
    )


def _tiny_model(cfg):
    from hexo_models.hexgt.architecture import HexgtNetwork

    a = cfg.architecture
    return HexgtNetwork(
        token_dim=a.token_dim, gnn_layers=a.gnn_layers, ctx_layers=a.ctx_layers,
        attention_heads=a.attention_heads, ffn_dim=a.ffn_dim,
    ).cpu().eval()


def test_selfplay_writes_trainable_compact_shards(tmp_path) -> None:
    _torch()
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.expand import build_training_batch
    from hexo_models.hexgt.selfplay import run_selfplay_games

    cfg = _tiny_config()
    model = _tiny_model(cfg)

    result = run_selfplay_games(
        model, cfg, num_games=3, output_dir=tmp_path, epoch=0,
        device="cpu", fp16=False, base_seed=5, active_games=3,
        virtual_batch_size=2, collect_examples=2,
    )

    # One compact shard per game, all on disk.
    assert len(result.shard_paths) == 3
    for p in result.shard_paths:
        assert Path(p).exists()

    # Self-play also writes ONE per-epoch full-game .hxr record (one record/game).
    from hexo_runner.records import HexoRecordFile
    hxr = tmp_path / "epoch_000000.hxr"
    assert hxr.exists(), "self-play did not write the per-epoch .hxr game records"
    assert len(list(HexoRecordFile.open(hxr).iter_records())) == 3
    assert result.completed_games + result.truncated_games == 3
    assert result.searched_positions > 0
    assert result.raw_samples > 0
    assert result.positions_per_second > 0.0
    assert result.mean_candidate_count > 0.0

    # Shards decode as dense_cnn compact rows AND expand into a trainer batch.
    rows = read_compact_shard(Path(result.shard_paths[0]))
    assert rows, "self-play shard decoded to zero rows"
    batch, targets = build_training_batch(
        rows, n=cfg.architecture.candidate_radius, horizons=(), prune_max_dropped_mass=1.0
    )
    assert batch is not None
    assert int(batch["num_graphs"]) >= 1
    assert targets["policy"].numel() == batch["candidate_graph"].numel()


def test_selfplay_shards_train_one_step(tmp_path) -> None:
    """A self-play epoch's shards drive a real `HexgtTrainer` optimizer step."""

    _torch()
    import torch

    from hexo_models.hexgt.selfplay import run_selfplay_games
    from hexo_models.hexgt.trainer import HexgtTrainer

    cfg = _tiny_config()
    model = _tiny_model(cfg)
    result = run_selfplay_games(
        model, cfg, num_games=2, output_dir=tmp_path, epoch=1,
        device="cpu", fp16=False, base_seed=9, active_games=2, virtual_batch_size=2,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    trainer = HexgtTrainer(model=model, config=cfg, optimizer=opt)
    history = trainer.train_on_shards(
        [Path(p) for p in result.shard_paths], batch_size=8, max_steps=2
    )
    assert history, "trainer produced no steps from self-play shards"
    assert "total" in history[0]
    assert history[0]["total"] == history[0]["total"]  # finite (not NaN)


def test_hxr_record_holds_full_game_through_winning_move(tmp_path) -> None:
    """The .hxr record must contain the COMPLETE game incl. the terminal/winning
    move + the correct winner (unlike the .npz shards, which sample only
    nonterminal positions and so omit the final move). Deterministic: play a real
    game to terminal with fixed-RNG legal moves (no model), persist it via the
    self-play record writer, then re-read and verify it replays to that terminal +
    winner with the full move count."""
    _torch()
    import hexo_engine as engine
    import hexo_engine.api as eng
    from hexo_engine.types import pack_coord_id, unpack_coord_id
    from hexo_runner.records import HexoRecordFile
    from hexo_models.hexgt.selfplay import _RECORD_PLAYERS, _write_game_record

    # Deterministic terminal game: always play the most central available cell, so
    # both colours pack densely and a 6-in-a-line forms quickly (random play on the
    # sparse board spreads out and rarely lines up). No model/search involved.
    state = eng.new_game(seed=1)
    action_ids: list[int] = []
    terminal = None
    for _ in range(600):
        terminal = eng.terminal(state)
        if terminal is not None:
            break
        legal = list(eng.legal_actions(state))
        act = min(legal, key=lambda a: (abs(a.coord.q) + abs(a.coord.r), a.coord.q, a.coord.r))
        action_ids.append(int(pack_coord_id(act.coord)))
        eng.apply_action(state, act)
    assert terminal is not None and terminal.winner is not None, "central-fill game did not terminate"
    winner = str(getattr(terminal.winner, "value", terminal.winner))

    path = tmp_path / "epoch_000000.hxr"
    rf = HexoRecordFile.create(path, engine.engine_metadata(), _RECORD_PLAYERS)
    _write_game_record(rf, {"game_id": "t-0", "seed": 0, "actions": action_ids},
                       winner=winner, truncated=False, max_actions=9999)
    rf.close()

    rec = list(HexoRecordFile.open(path).iter_records())[0]
    # Full move sequence preserved, INCLUDING the terminal/winning move.
    assert [int(a) for a in rec.action_ids] == action_ids
    assert str(rec.winner) == winner
    # Replaying the recorded moves reaches the same terminal + winner.
    s2 = eng.new_game()
    for aid in rec.action_ids:
        eng.apply_action(s2, engine.PlacementAction(unpack_coord_id(int(aid))))
    t2 = eng.terminal(s2)
    assert t2 is not None, ".hxr game does not replay to a terminal (winning move missing)"
    assert str(getattr(t2.winner, "value", t2.winner)) == winner


def test_selfplay_captures_example_traces(tmp_path) -> None:
    _torch()
    from hexo_models.hexgt.selfplay import run_selfplay_games

    cfg = _tiny_config()
    model = _tiny_model(cfg)
    result = run_selfplay_games(
        model, cfg, num_games=2, output_dir=tmp_path, epoch=2,
        device="cpu", fp16=False, base_seed=3, active_games=2, collect_examples=2,
    )

    assert len(result.example_games) == 2
    game = result.example_games[0]
    assert game["turns"] > 0
    assert game["moves"], "example game has no per-move trace"
    move = game["moves"][0]
    for key in ("action_id", "root_value", "visit_entropy", "prior_entropy", "top_visit_fraction"):
        assert key in move
    # Temperature should decay across the game (initial 1.0 -> final 0.2).
    temps = [m["temperature"] for m in game["moves"]]
    assert temps[0] >= temps[-1]
