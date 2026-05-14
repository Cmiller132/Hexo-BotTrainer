"""Direct match runner mode.

Match mode starts a single match and returns the result.
This is the public entry point for running one game. It should build a session,
call the shared loop, and return a compact `GameResult`.
"""

from __future__ import annotations

from dataclasses import dataclass

from hexo_engine import (
    AxialCoord,
    EngineSnapshot,
    EngineStateRef,
    IllegalActionError,
    PlacementAction,
    apply_action,
    game_state,
    legal_actions,
    load_snapshot,
    new_game,
    snapshot as engine_snapshot,
    tactics,
    terminal,
)


@dataclass(slots=True)
class InteractiveMatch:
    """Step-by-step match session for human or remote-controlled play."""

    engine_state: EngineStateRef

    @classmethod
    def new(cls) -> "InteractiveMatch":
        return cls(engine_state=new_game())

    def reset(self) -> dict[str, object]:
        self.engine_state = new_game()
        return self.view()

    def play(self, q: int, r: int) -> dict[str, object]:
        apply_action(self.engine_state, PlacementAction(AxialCoord(q=q, r=r)))
        return self.view()

    def undo(self) -> dict[str, object]:
        snap = engine_snapshot(self.engine_state)
        placements = list(snap.payload.get("placements", []))
        if placements:
            placements.pop()
            self.engine_state = load_snapshot(
                EngineSnapshot(version=snap.version, payload={"placements": placements})
            )
        return self.view()

    def view(self) -> dict[str, object]:
        snap = engine_snapshot(self.engine_state)
        state = game_state(self.engine_state)
        legal = [_action_payload(action) for action in legal_actions(self.engine_state)]

        return {
            "engine_state": state,
            "legal_actions": legal,
            "legal_count": len(legal),
            "terminal": _terminal_payload(terminal(self.engine_state)),
            "tactics": tactics(self.engine_state),
            "snapshot": {
                "version": snap.version,
                "placements": snap.payload.get("placements", []),
            },
        }


def create_match(config: object | None = None) -> InteractiveMatch:
    """Create an interactive match session backed by the engine API."""

    return InteractiveMatch.new()


def run_match(config: object | None = None) -> object:
    """Run one game through session setup and the shared loop."""

    return create_match(config)


def _action_payload(action: object) -> dict[str, int]:
    if isinstance(action, PlacementAction):
        return {"q": action.coord.q, "r": action.coord.r}
    raise IllegalActionError(f"Unsupported match action type: {type(action).__name__}")


def _terminal_payload(outcome: object) -> dict[str, object] | None:
    if outcome is None:
        return None
    winner = getattr(outcome, "winner", None)
    return {
        "winner": _raw_player(winner),
        "reason": getattr(outcome, "reason", None),
        "metadata": dict(getattr(outcome, "metadata", {}) or {}),
    }


def _raw_player(player: object) -> str | None:
    value = getattr(player, "value", player)
    if value == "player0":
        return "Player0"
    if value == "player1":
        return "Player1"
    return None
