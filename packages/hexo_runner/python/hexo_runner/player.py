"""Runner player contract.

Participants can be model-backed, scripted, human-controlled, remote, random,
or search-based. The runner sees only this contract and submits accepted actions
to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from hexo_engine import Action, EngineStateRef


@dataclass(frozen=True, slots=True)
class PlayerIdentity:
    """Stable identity for a participant in runner output."""

    player_id: str
    label: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Participant response consumed by the runner.

    Players receive only a cloned `EngineStateRef` in `decide`, query whatever
    they need from `hexo_engine`, and return one action plus optional
    diagnostics. There is no refusal/forfeit path; errors abort the game.
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

    def decide(self, state: EngineStateRef) -> DecisionResult:
        """Choose an action from a cloned, player-owned engine state."""

    def observe_transition(self, transition: TransitionEvent) -> None:
        """Observe an accepted engine transition."""

    def close(self, final_summary: FinalSummary) -> None:
        """Release player resources after a game or series."""
