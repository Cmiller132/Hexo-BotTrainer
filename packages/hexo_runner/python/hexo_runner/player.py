"""Runner player contract.

Participants can be model-backed, scripted, human-controlled, remote, random,
or search-based. The runner sees only this contract and submits accepted actions
to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class PlayerIdentity:
    """Stable identity for a participant in runner output."""

    player_id: str
    label: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Context passed to the active participant."""

    game_id: str
    turn_index: int
    current_player: object
    state: object
    legal_actions: Sequence[object]
    budget: object | None = None
    tactics: object | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Participant response consumed by the runner."""

    action: object | None
    elapsed_ms: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    refusal: str | None = None


@dataclass(frozen=True, slots=True)
class TransitionEvent:
    """Notification sent to participants after an engine transition."""

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
        """Choose an action or return a controlled refusal."""

    def observe_transition(self, transition: TransitionEvent) -> None:
        """Observe an accepted engine transition."""

    def close(self, final_summary: FinalSummary) -> None:
        """Release player resources after a game or series."""
