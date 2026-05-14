"""Python package boundary for the Rust Hexo engine.

The engine owns canonical rules, legal transitions, tactical state, and stable
action identity. Python callers should come through this package instead of
duplicating game logic.
"""

from .api import (
    HexoState,
    action_id,
    apply_action,
    clone_state,
    current_player,
    engine_metadata,
    legal_actions,
    new_game,
    terminal,
    to_python_state,
    turn_placement,
    validate_action,
)
from .errors import (
    EngineUnavailableError,
    HexoEngineError,
    IllegalActionError,
)
from .types import (
    Action,
    ActionId,
    AxialCoord,
    PairAction,
    PlacementAction,
    Player,
    PythonBoard,
    PythonHexoState,
    PythonMoveRecord,
    PythonPlacementRecord,
    PythonTerminal,
    PythonWindowEntry,
    PythonWindowKey,
    PythonWindowStore,
    TerminalResult,
    TransitionResult,
    TurnPhase,
    TurnPlacement,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "ActionId",
    "AxialCoord",
    "EngineUnavailableError",
    "HexoState",
    "HexoEngineError",
    "IllegalActionError",
    "PairAction",
    "PlacementAction",
    "Player",
    "PythonBoard",
    "PythonHexoState",
    "PythonMoveRecord",
    "PythonPlacementRecord",
    "PythonTerminal",
    "PythonWindowEntry",
    "PythonWindowKey",
    "PythonWindowStore",
    "TerminalResult",
    "TransitionResult",
    "TurnPhase",
    "TurnPlacement",
    "action_id",
    "apply_action",
    "clone_state",
    "current_player",
    "engine_metadata",
    "legal_actions",
    "new_game",
    "terminal",
    "to_python_state",
    "turn_placement",
    "validate_action",
]
