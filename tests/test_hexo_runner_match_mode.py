from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_engine", "hexo_runner", "hexo_frontend"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class RunnerMatchModeTests(unittest.TestCase):
    def test_opening_and_two_stone_turns(self) -> None:
        from hexo_runner.modes.match import create_match

        match = create_match()

        self.assertEqual(match.view()["legal_actions"], [{"q": 0, "r": 0}])
        state = match.play(0, 0)
        self.assertEqual(state["engine_state"]["current_player"], "Player1")
        self.assertEqual(state["engine_state"]["phase"], "FirstStone")

        state = match.play(1, 0)
        self.assertEqual(state["engine_state"]["current_player"], "Player1")
        self.assertIn("SecondStone", state["engine_state"]["phase"])

        state = match.play(2, 0)
        self.assertEqual(state["engine_state"]["current_player"], "Player0")
        self.assertEqual(state["engine_state"]["phase"], "FirstStone")

    def test_six_in_line_wins_after_single_placement(self) -> None:
        from hexo_runner.modes.match import create_match

        match = create_match()
        state: dict[str, object] = {}
        for q, r in [
            (0, 0),
            (0, 1), (0, 2),
            (1, 0), (2, 0),
            (1, 1), (1, 2),
            (3, 0), (4, 0),
            (2, 1), (2, 2),
            (5, 0),
        ]:
            state = match.play(q, r)

        self.assertEqual(state["terminal"]["winner"], "Player0")
        self.assertEqual(state["terminal"]["reason"], "six_in_line")
        self.assertEqual(state["legal_actions"], [])

    def test_undo_rebuilds_state(self) -> None:
        from hexo_runner.modes.match import create_match

        match = create_match()
        match.play(0, 0)
        match.play(1, 0)

        state = match.undo()
        self.assertEqual(len(state["engine_state"]["placement_history"]), 1)
        self.assertEqual(state["engine_state"]["current_player"], "Player1")
        self.assertEqual(state["engine_state"]["phase"], "FirstStone")

    def test_runner_exposes_raw_engine_windows(self) -> None:
        from hexo_runner.modes.match import create_match

        match = create_match()
        state: dict[str, object] = {}
        for q, r in [
            (0, 0),
            (0, 1), (0, 2),
            (1, 0), (2, 0),
            (1, 1), (1, 2),
            (3, 0), (4, 0),
        ]:
            state = match.play(q, r)

        tactics = state["tactics"]
        entries = tactics["window_store"]["entries"]
        self.assertGreater(tactics["window_store"]["len"], 0)
        self.assertNotIn("windows", tactics)
        self.assertNotIn("immediate_wins", tactics)
        self.assertNotIn("must_blocks", tactics)
        self.assertNotIn("summary", tactics)
        self.assertTrue(
            any(
                entry["key"]["axis"] == "Q"
                and entry["key"]["start"] == {"q": -1, "r": 0}
                and entry["masks"] == [62, 0]
                for entry in entries
            )
        )

    def test_frontend_derives_tactics_dashboard_from_raw_engine_data(self) -> None:
        from hexo_frontend.dashboard import dashboard_state
        from hexo_runner.modes.match import create_match

        match = create_match()
        raw: dict[str, object] = {}
        for q, r in [
            (0, 0),
            (0, 1), (0, 2),
            (1, 0), (2, 0),
            (1, 1), (1, 2),
            (3, 0), (4, 0),
        ]:
            raw = match.play(q, r)

        state = dashboard_state(raw)
        tactics = state["tactics"]
        self.assertGreater(tactics["window_count"], 0)
        self.assertGreaterEqual(tactics["threat_count"], 1)
        self.assertTrue(
            any(
                threat["player"] == "player0"
                and threat["axis"] == "Q"
                and threat["own_count"] >= 5
                for threat in tactics["threats"]
            )
        )
        self.assertTrue(
            any(block["player"] == "player1" and block["q"] == 5 and block["r"] == 0 for block in tactics["must_blocks"])
        )
        self.assertTrue(
            any(win["player"] == "player0" and win["q"] == 5 and win["r"] == 0 for win in tactics["immediate_wins"])
        )


if __name__ == "__main__":
    unittest.main()
