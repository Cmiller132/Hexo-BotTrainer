"""Python package boundary for the Rust Hexo engine.

The engine owns canonical rules, legal transitions, snapshots, raw tactics data,
and stable state/action identity. Python callers should come through this
package instead of duplicating game logic.
"""

from .api import (
    EngineStateRef,
    action_id,
    apply_action,
    current_player,
    engine_metadata,
    game_state,
    legal_actions,
    load_snapshot,
    new_game,
    snapshot,
    state_id,
    tactics,
    terminal,
    turn_placement,
    validate_action,
)
from .errors import (
    EngineUnavailableError,
    HexoEngineError,
    IllegalActionError,
    IncompatibleSnapshotError,
    SnapshotError,
)
from .types import (
    Action,
    ActionId,
    AxialCoord,
    EngineSnapshot,
    PairAction,
    PlacementAction,
    Player,
    StateId,
    TacticalSummary,
    TerminalResult,
    TransitionResult,
    TurnPlacement,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "ActionId",
    "AxialCoord",
    "EngineSnapshot",
    "EngineStateRef",
    "EngineUnavailableError",
    "HexoEngineError",
    "IllegalActionError",
    "IncompatibleSnapshotError",
    "PairAction",
    "PlacementAction",
    "Player",
    "SnapshotError",
    "StateId",
    "TacticalSummary",
    "TerminalResult",
    "TransitionResult",
    "TurnPlacement",
    "action_id",
    "apply_action",
    "current_player",
    "engine_metadata",
    "game_state",
    "legal_actions",
    "load_snapshot",
    "new_game",
    "snapshot",
    "state_id",
    "tactics",
    "terminal",
    "turn_placement",
    "validate_action",
]
