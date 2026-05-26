from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (
    TESTS,
    ROOT / "packages" / "hexo_engine" / "python",
    ROOT / "packages" / "hexo_runner" / "python",
    ROOT / "packages" / "hexo_frontend" / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


WINNING_P0 = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0))
FILLER_P1 = ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2))


class ScriptedPlayer:
    def __init__(self, player_id: str, moves: tuple[tuple[int, int], ...], *, mutate_decision_clone: bool = False) -> None:
        from hexo_runner import PlayerIdentity

        self.identity = PlayerIdentity(player_id=player_id)
        self.base_moves = tuple(moves)
        self.moves: list[tuple[int, int]] = []
        self.mutate_decision_clone = mutate_decision_clone
        self.setup_worker_count = 0
        self.start_game_count = 0
        self.finish_game_count = 0
        self.close_count = 0
        self.observed: list[object] = []

    def setup_worker(self, context: object) -> None:
        self.setup_worker_count += 1

    def start_game(self, context: object) -> None:
        self.start_game_count += 1
        self.moves = list(self.base_moves)

    def decide(self, state: object) -> object:
        from hexo_engine import AxialCoord, PlacementAction, apply_action, legal_actions
        from hexo_runner import DecisionResult

        actions = list(legal_actions(state))
        if self.mutate_decision_clone and actions:
            apply_action(state, actions[0])
        if not self.moves:
            raise RuntimeError("script exhausted")
        q, r = self.moves.pop(0)
        return DecisionResult(action=PlacementAction(AxialCoord(q, r)), diagnostics={"scripted": True})

    def observe_transition(self, transition: object) -> None:
        self.observed.append(transition)

    def finish_game(self, final_summary: object) -> None:
        self.finish_game_count += 1

    def close(self) -> None:
        self.close_count += 1


class IllegalPlayer(ScriptedPlayer):
    def __init__(self, player_id: str) -> None:
        super().__init__(player_id, ())

    def decide(self, state: object) -> object:
        from hexo_engine import AxialCoord, PlacementAction
        from hexo_runner import DecisionResult

        return DecisionResult(action=PlacementAction(AxialCoord(99, 99)))


class ExplodingPlayer(ScriptedPlayer):
    def decide(self, state: object) -> object:
        raise RuntimeError("boom")


class MutatingObserverPlayer(ScriptedPlayer):
    def observe_transition(self, transition: object) -> None:
        from hexo_engine import apply_action, legal_actions

        self.observed.append(transition)
        actions = list(legal_actions(transition.state))
        if actions:
            apply_action(transition.state, actions[0])


class RecordingIllegalObserver(IllegalPlayer):
    def __init__(self, player_id: str) -> None:
        super().__init__(player_id)
        self.python_states: list[object] = []

    def observe_transition(self, transition: object) -> None:
        from hexo_engine import to_python_state

        self.observed.append(transition)
        self.python_states.append(to_python_state(transition.state))


@dataclass
class ScriptedFactory:
    player_id: str
    moves: tuple[tuple[int, int], ...]
    created: int = 0
    instances: list[ScriptedPlayer] = field(default_factory=list)

    def create_player(self) -> ScriptedPlayer:
        self.created += 1
        player = ScriptedPlayer(self.player_id, self.moves)
        self.instances.append(player)
        return player


@dataclass
class ConditionalFactory:
    player_id: str
    moves: tuple[tuple[int, int], ...]
    abort_game_id: str | None = None
    created: int = 0
    instances: list[ScriptedPlayer] = field(default_factory=list)

    def create_player(self) -> ScriptedPlayer:
        self.created += 1
        if self.abort_game_id is None:
            player = ScriptedPlayer(self.player_id, self.moves)
        else:
            player = ConditionalPlayer(self.player_id, self.moves, self.abort_game_id)
        self.instances.append(player)
        return player


class ConditionalPlayer(ScriptedPlayer):
    def __init__(self, player_id: str, moves: tuple[tuple[int, int], ...], abort_game_id: str) -> None:
        super().__init__(player_id, moves)
        self.abort_game_id = abort_game_id
        self.current_game_id = ""

    def start_game(self, context: object) -> None:
        super().start_game(context)
        self.current_game_id = context.game_id

    def decide(self, state: object) -> object:
        if self.current_game_id == self.abort_game_id:
            return IllegalPlayer(self.identity.player_id).decide(state)
        return super().decide(state)


def action_from_record(action_id: int) -> object:
    from hexo_engine import PlacementAction
    from hexo_engine.types import unpack_coord_id

    return PlacementAction(unpack_coord_id(action_id))


def records_from_result(result: object) -> tuple[object, ...]:
    from hexo_runner.records import HexoRecordFile

    with HexoRecordFile.open(result.record_ref["path"]) as record_file:
        return record_file.iter_records()


class RunnerRewriteTests(unittest.TestCase):
    def test_completed_game_writes_compact_replayable_record(self) -> None:
        from hexo_engine import Player, apply_action, new_game, terminal
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import GameStatus, HEXO_RECORD_SCHEMA_VERSION
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(
                GameSpec(game_id="scripted", seed=7),
                (ScriptedPlayer("p0", WINNING_P0), ScriptedPlayer("p1", FILLER_P1)),
                tmp,
            )
            records = records_from_result(result)

        self.assertEqual(result.status, GameStatus.COMPLETED)
        self.assertEqual(result.winner, "player0")
        self.assertEqual(result.turns, 12)
        self.assertEqual(HEXO_RECORD_SCHEMA_VERSION, 1)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.status, "completed")
        self.assertEqual(len(record.action_ids), 12)
        self.assertIsNone(record.abort)

        replay = new_game(seed=record.seed)
        for action_id in record.action_ids:
            apply_action(replay, action_from_record(action_id))
        self.assertEqual(terminal(replay).winner, Player.PLAYER_0)
        self.assertEqual(record.replay().winner, Player.PLAYER_0)

    def test_player_can_mutate_decision_clone_without_corrupting_primary_state(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_engine.types import unpack_coord_id
        from hexo_runner.records import GameStatus
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(
                GameSpec(game_id="clone-isolation"),
                (ScriptedPlayer("p0", ((0, 0),), mutate_decision_clone=True), IllegalPlayer("p1")),
                tmp,
            )
            records = records_from_result(result)

        self.assertEqual(result.status, GameStatus.ABORTED)
        self.assertEqual(result.turns, 1)
        self.assertEqual(len(records[0].action_ids), 1)
        coord = unpack_coord_id(records[0].action_ids[0])
        self.assertEqual((coord.q, coord.r), (0, 0))

    def test_illegal_action_aborts_loudly_and_writes_aborted_record(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import GameStatus
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(GameSpec(game_id="illegal"), (IllegalPlayer("p0"), ScriptedPlayer("p1", ())), tmp)
            records = records_from_result(result)

        self.assertEqual(result.status, GameStatus.ABORTED)
        self.assertEqual(result.abort.stage, "engine.apply_action")
        self.assertIn("opening placement", result.abort.message)
        record = records[0]
        self.assertEqual(record.status, "aborted")
        self.assertEqual(record.action_ids, ())
        self.assertIsNone(record.winner)
        self.assertEqual(record.abort.stage, "engine.apply_action")

    def test_player_exception_aborts_with_stage_type_and_message(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import GameStatus
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(GameSpec(game_id="explode"), (ExplodingPlayer("p0", ()), ScriptedPlayer("p1", ())), tmp)
            records = records_from_result(result)

        self.assertEqual(result.status, GameStatus.ABORTED)
        self.assertEqual(result.abort.stage, "player.decide:p0")
        self.assertEqual(result.abort.exception_type, "RuntimeError")
        self.assertEqual(result.abort.message, "boom")
        self.assertEqual(records[0].abort.stage, "player.decide:p0")

    def test_observers_receive_independent_cloned_states(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.session import GameSpec

        p0 = MutatingObserverPlayer("p0", ((0, 0),))
        p1 = RecordingIllegalObserver("p1")
        with tempfile.TemporaryDirectory() as tmp:
            run_match(GameSpec(game_id="observer-clones"), (p0, p1), tmp)

        self.assertEqual(p1.python_states[0].placements_made, 1)
        self.assertGreater(len(p0.observed), 0)
        self.assertGreater(len(p1.observed), 0)

    def test_hexo_record_file_writes_one_record_per_game(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import HexoRecordFile
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(
                GameSpec(game_id="hxr"),
                (ScriptedPlayer("p0", WINNING_P0), ScriptedPlayer("p1", FILLER_P1)),
                tmp,
            )
            path = Path(result.record_ref["path"])
            with HexoRecordFile.open(path) as record_file:
                records = record_file.iter_records()

            self.assertEqual(path.suffix, ".hxr")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].game_id, "hxr")
            self.assertEqual(records[0].status, "completed")
            self.assertEqual(len(records[0].action_ids), 12)

    def test_batch_reuses_player_instances_within_worker_and_surfaces_aborts(self) -> None:
        from hexo_runner.modes.batch import run_batch
        from hexo_runner.records import GameStatus
        from hexo_runner.session import BatchSpec, GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            p0_factory = ConditionalFactory("p0", WINNING_P0, abort_game_id="bad")
            p1_factory = ScriptedFactory("p1", FILLER_P1)
            result = run_batch(
                BatchSpec(
                    batch_id="reuse",
                    games=(GameSpec("ok-1"), GameSpec("bad"), GameSpec("ok-2")),
                    player_factories=(p0_factory, p1_factory),
                    output_dir=tmp,
                    worker_count=1,
                    chunk_size=1,
                )
            )

        self.assertEqual(result.total_games, 3)
        self.assertEqual(result.completed, 2)
        self.assertEqual(result.aborted, 1)
        self.assertEqual(p0_factory.created, 1)
        self.assertEqual(p1_factory.created, 1)
        self.assertEqual(p1_factory.instances[0].start_game_count, 3)
        self.assertEqual([item.status for item in result.results].count(GameStatus.ABORTED), 1)
        self.assertEqual(result.aborts[0].stage, "engine.apply_action")

    def test_batch_runs_across_process_workers_and_writes_worker_hxr(self) -> None:
        from hexo_runner.modes.batch import run_batch
        from hexo_runner.records import HexoRecordFile
        from hexo_runner.session import BatchSpec, GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_batch(
                BatchSpec(
                    batch_id="process",
                    games=tuple(GameSpec(f"game-{index}") for index in range(4)),
                    player_factories=(ScriptedFactory("p0", WINNING_P0), ScriptedFactory("p1", FILLER_P1)),
                    output_dir=tmp,
                    worker_count=2,
                    chunk_size=1,
                )
            )
            record_count = 0
            for ref in result.record_refs:
                with HexoRecordFile.open(ref["path"]) as record_file:
                    record_count += len(record_file.iter_records())

        self.assertEqual(result.total_games, 4)
        self.assertEqual(result.completed, 4)
        self.assertEqual(result.aborted, 0)
        self.assertEqual(result.worker_count, 2)
        self.assertEqual(record_count, 4)

    def test_frontend_controller_still_uses_generic_runner(self) -> None:
        from hexo_frontend.web import ManualMatchController

        controller = ManualMatchController()
        try:
            state = controller.state()
            self.assertEqual(state["legal"], [{"q": 0, "r": 0}])

            state = controller.submit_move(0, 0)
            self.assertEqual(len(state["placements"]), 1)
            self.assertEqual(state["current_player"], "player1")

            state = controller.reset()
            self.assertEqual(state["placements"], [])
            self.assertEqual(state["legal"], [{"q": 0, "r": 0}])

            with self.assertRaises(ValueError):
                controller.submit_move(42, 42)
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
