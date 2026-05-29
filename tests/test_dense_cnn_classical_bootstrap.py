from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for package in ("hexo_engine", "hexo_models", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_recording_classical_player_preserves_global_game_order() -> None:
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id
    from hexo_models.dense_cnn.samples import finalize_game_samples
    from hexo_runner.player import DecisionResult, PlayerIdentity
    from scripts.bootstrap_dense_cnn_classical import RecordingClassicalPlayer

    class FirstLegalPlayer:
        def __init__(self, player_id: str) -> None:
            self.identity = PlayerIdentity(player_id, player_id)

        def setup_worker(self, _context: Any) -> None:
            return

        def start_game(self, _context: Any) -> None:
            return

        def decide(self, state: object) -> DecisionResult:
            action_id = int(engine.legal_action_ids(state)[0])
            return DecisionResult(action=engine.PlacementAction(unpack_coord_id(action_id)))

        def observe_transition(self, _transition: Any) -> None:
            return

        def finish_game(self, _final_summary: Any) -> None:
            return

        def close(self) -> None:
            return

    pending: list[tuple[str, object, float]] = []
    counter = [0]
    player0 = RecordingClassicalPlayer(
        FirstLegalPlayer("classical-p0"),  # type: ignore[arg-type]
        sample_sink=pending,
        action_counter=counter,
        source="unit_classical",
        compression_level=1,
    )
    player1 = RecordingClassicalPlayer(
        FirstLegalPlayer("classical-p1"),  # type: ignore[arg-type]
        sample_sink=pending,
        action_counter=counter,
        source="unit_classical",
        compression_level=1,
    )

    state = engine.new_game(seed=7)
    for player in (player0, player1, player1):
        decision = player.decide(state)
        engine.apply_action(state, decision.action)

    decoded = [sample.decode() for _player, sample, _value in pending]
    assert [sample.turn_index for sample in decoded] == [0, 1, 2]
    assert [sample.current_player for sample in decoded] == ["player0", "player1", "player1"]

    finalized = finalize_game_samples(pending, winner="player1", horizons=(), truncated=False)
    assert finalized[0].opp_policy == decoded[1].policy


def test_bootstrap_reuses_classical_sample_checkpoint(tmp_path: Path) -> None:
    import torch
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id
    from hexo_models.dense_cnn.samples import SampleBuffer, sample_from_state
    from scripts.bootstrap_dense_cnn_classical import JsonDiagnostics, load_samples_from_checkpoint

    state = engine.new_game(seed=11)
    action_id = int(engine.legal_action_ids(state)[0])
    sample = sample_from_state(
        state,
        game_id="classical-sealbot-unit",
        turn_index=0,
        policy={action_id: 1.0},
        value=1.0,
        metadata={"sample_source": "classical_sealbot_best_bootstrap"},
    )
    source = SampleBuffer(capacity=200_000, compression_level=1)
    source.append(sample)
    checkpoint = tmp_path / "samples.pt"
    torch.save({"epoch": 6, "sample_buffer": source.state_dict(), "metadata": {"run": "unit"}}, checkpoint)

    target = SampleBuffer(capacity=200_000, compression_level=1)
    result = load_samples_from_checkpoint(
        buffer=target,
        samples_checkpoint=checkpoint,
        sample_count=1,
        require_classical=True,
        diagnostics=JsonDiagnostics(tmp_path / "diagnostics"),
    )

    assert target.sample_count == 1
    assert result["samples"] == 1
    assert result["source_counts"] == {"classical_sealbot_best_bootstrap": 1}


def test_bootstrap_rejects_non_classical_sample_checkpoint(tmp_path: Path) -> None:
    import pytest
    import torch
    import hexo_engine as engine
    from hexo_models.dense_cnn.samples import SampleBuffer, sample_from_state
    from scripts.bootstrap_dense_cnn_classical import JsonDiagnostics, load_samples_from_checkpoint

    state = engine.new_game(seed=12)
    action_id = int(engine.legal_action_ids(state)[0])
    sample = sample_from_state(
        state,
        game_id="epoch-000001-selfplay-000001",
        turn_index=0,
        policy={action_id: 1.0},
        value=0.0,
        metadata={"sample_source": "mcts"},
    )
    source = SampleBuffer(capacity=200_000, compression_level=1)
    source.append(sample)
    checkpoint = tmp_path / "samples.pt"
    torch.save({"epoch": 1, "sample_buffer": source.state_dict()}, checkpoint)

    with pytest.raises(ValueError, match="non-classical"):
        load_samples_from_checkpoint(
            buffer=SampleBuffer(capacity=200_000, compression_level=1),
            samples_checkpoint=checkpoint,
            sample_count=1,
            require_classical=True,
            diagnostics=JsonDiagnostics(tmp_path / "diagnostics"),
        )
