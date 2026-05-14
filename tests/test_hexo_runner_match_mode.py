from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_engine", "hexo_runner", "hexo_frontend"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class MemorySink:
    def __init__(self) -> None:
        self.entries: list[object] = []
        self.closed: tuple[str, object | None] | None = None

    def write_entry(self, entry: object) -> None:
        self.entries.append(entry)

    def close_game(self, game_id: str, terminal: object | None = None) -> object:
        self.closed = (game_id, terminal)
        return {"game_id": game_id, "entries": len(self.entries), "terminal": terminal is not None}


class ScriptedPlayer:
    def __init__(self, player_id: str, moves: list[tuple[int, int]], *, mutate_clone: bool = False) -> None:
        from hexo_runner.player import PlayerIdentity

        self.identity = PlayerIdentity(player_id=player_id)
        self.moves = list(moves)
        self.mutate_clone = mutate_clone
        self.initialized = False
        self.closed = False
        self.observed: list[object] = []
        self.requests: list[object] = []

    def initialize(self, session_context: object) -> None:
        self.initialized = True

    def decide(self, request: object) -> object:
        from hexo_engine import AxialCoord, PlacementAction, apply_action
        from hexo_runner.player import DecisionResult

        self.requests.append(request)
        legal_actions = list(getattr(request, "legal_actions"))
        if self.mutate_clone and legal_actions:
            apply_action(getattr(request, "state").state_ref, legal_actions[0])
        if not self.moves:
            raise RuntimeError("script exhausted")
        q, r = self.moves.pop(0)
        return DecisionResult(action=PlacementAction(AxialCoord(q, r)), diagnostics={"scripted": True})

    def observe_transition(self, transition: object) -> None:
        self.observed.append(transition)

    def close(self, final_summary: object) -> None:
        self.closed = True


class IllegalPlayer(ScriptedPlayer):
    def __init__(self, player_id: str) -> None:
        super().__init__(player_id, [])

    def decide(self, request: object) -> object:
        from hexo_engine import AxialCoord, PlacementAction
        from hexo_runner.player import DecisionResult

        self.requests.append(request)
        return DecisionResult(action=PlacementAction(AxialCoord(99, 99)))


class RunnerMatchModeTests(unittest.TestCase):
    def test_scripted_players_complete_game_and_write_records(self) -> None:
        from hexo_engine import Player
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import GameStatus
        from hexo_runner.session import GameSpec

        p0 = ScriptedPlayer("p0", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)], mutate_clone=True)
        p1 = ScriptedPlayer("p1", [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)])
        sink = MemorySink()

        result = run_match(GameSpec(game_id="scripted", seed=7), (p0, p1), sink)

        self.assertEqual(result.status, GameStatus.COMPLETED)
        self.assertEqual(result.winner, Player.PLAYER_0)
        self.assertEqual(result.terminal.reason, "six_in_line")
        self.assertEqual(len(sink.entries), 12)
        self.assertEqual(result.record_ref, {"game_id": "scripted", "entries": 12, "terminal": True})
        self.assertTrue(p0.initialized and p1.initialized)
        self.assertTrue(p0.closed and p1.closed)
        self.assertEqual(len(p0.observed), 12)
        self.assertEqual(len(p1.observed), 12)

        for index, entry in enumerate(sink.entries):
            before = entry.before_snapshot.payload["placements"]
            after = entry.after_snapshot.payload["placements"]
            self.assertEqual(len(before), index)
            self.assertEqual(len(after), index + 1)
            self.assertIn("before_state_id", entry.metadata)
            self.assertIn("after_state_id", entry.metadata)
            self.assertIn("action_id", entry.metadata)
            self.assertIn("decision_diagnostics", entry.metadata)
            self.assertIn("transition_metadata", entry.metadata)

    def test_decision_request_exposes_typed_cloneable_engine_view(self) -> None:
        from hexo_engine import legal_actions, snapshot
        from hexo_runner.modes.match import run_match
        from hexo_runner.player import EngineDecisionView
        from hexo_runner.session import GameSpec

        p0 = ScriptedPlayer("p0", [(0, 0)], mutate_clone=True)
        p1 = IllegalPlayer("p1")
        sink = MemorySink()

        run_match(GameSpec(game_id="view-contract"), (p0, p1), sink)

        request = p0.requests[0]
        self.assertIsInstance(request.state, EngineDecisionView)
        self.assertEqual(request.seed, None)
        self.assertEqual(request.state.snapshot.payload["placements"], [])
        self.assertEqual(len(snapshot(request.state.state_ref).payload["placements"]), 1)
        self.assertEqual(len(sink.entries[0].before_snapshot.payload["placements"]), 0)
        self.assertEqual(len(sink.entries[0].after_snapshot.payload["placements"]), 1)
        self.assertIn("board", request.state.game_state)
        self.assertIn("window_store", request.state.tactics)
        self.assertIn("entries", request.state.tactics["window_store"])
        self.assertGreater(len(legal_actions(request.state.state_ref)), 1)

    def test_decision_result_requires_an_action(self) -> None:
        from hexo_runner.player import DecisionResult

        with self.assertRaises(ValueError):
            DecisionResult(action=None)

    def test_illegal_action_aborts_without_corrupting_primary_state(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import GameStatus
        from hexo_runner.session import GameSpec

        p0 = IllegalPlayer("p0")
        p1 = ScriptedPlayer("p1", [])
        sink = MemorySink()

        result = run_match(GameSpec(game_id="illegal"), (p0, p1), sink)

        self.assertEqual(result.status, GameStatus.ABORTED)
        self.assertIsNone(result.winner)
        self.assertEqual(len(sink.entries), 0)
        self.assertEqual(result.metadata["reason"], "runner_error")
        self.assertIn("not legal", result.metadata["error"])

    def test_engine_rejects_pair_actions_atomically(self) -> None:
        from hexo_engine import AxialCoord, IllegalActionError, PairAction, apply_action, new_game, snapshot

        state = new_game()

        with self.assertRaises(IllegalActionError):
            apply_action(state, PairAction((AxialCoord(0, 0), AxialCoord(0, 0))))

        self.assertEqual(snapshot(state).payload["placements"], [])

    def test_frontend_controller_uses_generic_runner(self) -> None:
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

    def test_frontend_no_longer_imports_create_match(self) -> None:
        web_source = (ROOT / "packages" / "hexo_frontend" / "python" / "hexo_frontend" / "web.py").read_text()
        app_source = (ROOT / "packages" / "hexo_frontend" / "python" / "hexo_frontend" / "static" / "app.js").read_text()

        self.assertNotIn("create_match", web_source)
        self.assertNotIn("/api/undo", app_source)


if __name__ == "__main__":
    unittest.main()
