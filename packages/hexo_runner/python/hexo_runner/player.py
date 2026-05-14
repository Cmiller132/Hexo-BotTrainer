"""Runner player contract.

Participants can be model-backed, scripted, human-controlled, remote, random,
or search-based. The runner sees only this contract and submits accepted actions
to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from hexo_engine import (
    Action,
    EngineSnapshot,
    EngineStateRef,
    Player,
    TerminalResult,
    TurnPlacement,
)


@dataclass(frozen=True, slots=True)
class PlayerIdentity:
    """Stable identity for a participant in runner output."""

    player_id: str
    label: str | None = None


@dataclass(frozen=True, slots=True)
class EngineDecisionView:
    """Stable engine view passed to a player for one decision.

    The runner builds this from public `hexo_engine` calls immediately before
    asking a player to move. `state_ref` is a clone of the primary engine state,
    made by replaying `snapshot`; players may mutate it for search without
    changing the authoritative game held by the runner.
    """

    # Cloned, player-owned engine state. Safe for MCTS/search mutation.
    state_ref: EngineStateRef
    # Replayable snapshot captured from the primary state before the decision.
    snapshot: EngineSnapshot
    # Stable ID for the primary state before the decision.
    state_id: str
    # Engine player whose turn it is.
    current_player: Player
    # Current single-placement slot in the turn sequence.
    turn_placement: TurnPlacement
    # Raw engine state payload: board, history, current player, phase, terminal.
    game_state: Mapping[str, Any]
    # Engine-generated legal actions available at this decision point.
    legal_actions: Sequence[Action]
    # Raw engine tactical/window data. Dashboard/model interpretation is external.
    tactics: Mapping[str, Any]
    # Terminal result if this view is ever built for a terminal state.
    terminal: TerminalResult | None
    # Runner/engine provenance for diagnostics.
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Context passed to the active participant.

    The active player receives this object in `RunnerPlayer.decide`. The
    authoritative state is not included. The player gets ergonomic top-level
    fields plus `state`, the typed `EngineDecisionView` described above.
    """

    # Runner/game identity.
    game_id: str
    turn_index: int
    # Engine player to act, duplicated from `state.current_player`.
    current_player: Player
    # Full typed engine view, including the cloned state ref and raw tactics.
    state: EngineDecisionView
    # Engine legal actions, duplicated from `state.legal_actions` for convenience.
    legal_actions: Sequence[Action]
    # Session seed/provenance, if any.
    seed: int | None = None
    is_evaluation: bool = False
    # Runner metadata such as mode, state_id, and engine capabilities.
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Participant response consumed by the runner.

    Players return only an action plus optional diagnostics. There is no
    refusal/forfeit path; errors should be raised and will abort the game.
    """

    # The action the runner will submit to `hexo_engine.apply_action`.
    action: Action
    # Player-owned debug data transported into the position record.
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action is None:
            raise ValueError("DecisionResult.action is required.")


@dataclass(frozen=True, slots=True)
class TransitionEvent:
    """Notification sent to participants after an engine transition.

    This is sent after the action has been accepted and applied to the primary
    engine state.
    """

    game_id: str
    turn_index: int
    action: object
    transition: object


@dataclass(frozen=True, slots=True)
class FinalSummary:
    """Final runner summary passed to players during cleanup."""

    game_id: str
    result: object
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RunnerPlayer(Protocol):
    """Protocol implemented by all runner participants."""

    identity: PlayerIdentity

    def initialize(self, session_context: object) -> None:
        """Prepare the player for a session."""

    def decide(self, request: DecisionRequest) -> DecisionResult:
        """Choose an action for the runner to submit to the engine."""

    def observe_transition(self, transition: TransitionEvent) -> None:
        """Observe an accepted engine transition."""

    def close(self, final_summary: FinalSummary) -> None:
        """Release player resources after a game or series."""
