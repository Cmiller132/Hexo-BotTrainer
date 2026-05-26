"""Centralized adapter for the public hexo_engine API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import hexo_engine as engine


class HexoEngineAdapter:
    """Small wrapper so the runner never scatters direct engine calls."""

    def metadata(self) -> Mapping[str, Any]:
        return _jsonable(engine.engine_metadata())

    def new_game(self, *, seed: int | None = None, scenario: object | None = None) -> engine.HexoState:
        return engine.new_game(seed=seed, scenario=scenario)

    def clone_state(self, state: engine.HexoState) -> engine.HexoState:
        return engine.clone_state(state)

    def current_player(self, state: engine.HexoState) -> engine.Player:
        return engine.current_player(state)

    def player_index(self, player: engine.Player) -> int:
        if player == engine.Player.PLAYER_0:
            return 0
        if player == engine.Player.PLAYER_1:
            return 1
        raise ValueError(f"Unknown engine player: {player!r}")

    def player_role(self, player: engine.Player) -> str:
        return str(player)

    def apply_action(self, state: engine.HexoState, action: engine.Action) -> engine.TransitionResult:
        return engine.apply_action(state, action)

    def terminal(self, state: engine.HexoState) -> engine.TerminalResult | None:
        return engine.terminal(state)

    def action_id(self, action: engine.Action) -> str:
        return engine.action_id(action)

    def action_payload(self, action: object) -> Mapping[str, Any]:
        if isinstance(action, engine.PlacementAction):
            return {"type": "placement", "q": action.coord.q, "r": action.coord.r}
        return {"type": type(action).__name__, "repr": repr(action)}

    def terminal_payload(self, terminal: object | None) -> Mapping[str, Any] | None:
        if terminal is None:
            return None
        if isinstance(terminal, engine.TerminalResult):
            return {
                "winner": str(terminal.winner) if terminal.winner is not None else None,
                "reason": terminal.reason,
                "metadata": _jsonable(terminal.metadata),
            }
        return _jsonable(terminal)

    def transition_payload(self, transition: object) -> Mapping[str, Any]:
        if isinstance(transition, engine.TransitionResult):
            return {
                "next_player": str(transition.next_player) if transition.next_player is not None else None,
                "terminal": transition.terminal,
                "metadata": _jsonable(transition.metadata),
            }
        return _jsonable(transition)


def _jsonable(value: object) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)
