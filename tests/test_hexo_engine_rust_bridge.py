from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_engine",):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class RustEngineBridgeTests(unittest.TestCase):
    def test_bridge_metadata_reports_rust_backend(self) -> None:
        import hexo_engine as engine

        metadata = engine.engine_metadata()

        self.assertEqual(metadata["backend"], "rust-pyo3")
        self.assertEqual(metadata["rules_version"], 1)

    def test_legal_actions_are_sorted(self) -> None:
        import hexo_engine as engine

        state = engine.new_game()
        engine.apply_action(state, engine.PlacementAction(engine.AxialCoord(0, 0)))

        coords = [(action.coord.q, action.coord.r) for action in engine.legal_actions(state)]

        self.assertEqual(coords, sorted(coords))

    def test_clone_mutation_does_not_affect_original(self) -> None:
        import hexo_engine as engine

        state = engine.new_game()
        clone = engine.clone_state(state)
        engine.apply_action(clone, engine.PlacementAction(engine.AxialCoord(0, 0)))

        self.assertEqual(engine.to_python_state(state).placements_made, 0)
        self.assertEqual(engine.to_python_state(clone).placements_made, 1)

    def test_illegal_action_does_not_mutate_state(self) -> None:
        import hexo_engine as engine

        state = engine.new_game()

        with self.assertRaises(engine.IllegalActionError):
            engine.apply_action(state, engine.PlacementAction(engine.AxialCoord(99, 99)))

        self.assertEqual(engine.to_python_state(state).placements_made, 0)
        self.assertIsNone(engine.terminal(state))

    def test_python_state_mirror_tracks_terminal_win(self) -> None:
        import hexo_engine as engine

        state = engine.new_game()
        for q, r in [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (2, 0),
            (1, 1),
            (1, 2),
            (3, 0),
            (4, 0),
            (2, 1),
            (2, 2),
            (5, 0),
        ]:
            engine.apply_action(state, engine.PlacementAction(engine.AxialCoord(q, r)))

        terminal = engine.terminal(state)
        mirror = engine.to_python_state(state)

        self.assertEqual(terminal.winner, engine.Player.PLAYER_0)
        self.assertEqual(terminal.reason, "six_in_line")
        self.assertEqual(mirror.terminal.winner, engine.Player.PLAYER_0)
        self.assertEqual(mirror.placements_made, 12)
        self.assertGreater(mirror.board.windows.len, 0)


if __name__ == "__main__":
    unittest.main()
