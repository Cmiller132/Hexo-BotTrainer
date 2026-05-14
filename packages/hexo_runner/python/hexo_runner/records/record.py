"""Durable core game record boundary.

The runner records the authoritative position trail and game metadata while a
game runs. It does not record model tensors, policy semantics, or training
targets. Model-owned training samples may keep references to these core records
for debugging, but they are written separately during self-play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..player import PlayerIdentity


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """Core record for one accepted engine transition.

    Metadata should include `before_state_id`, `after_state_id`, `action_id`,
    `decision_diagnostics`, and `transition_metadata` when produced by the
    generic runner loop.
    """

    game_id: str
    turn_index: int
    player_id: str
    before_snapshot: object
    action: object
    after_snapshot: object
    terminal: object | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GameRecord:
    """Complete core record for one game."""

    game_id: str
    players: Sequence[PlayerIdentity]
    entries: Sequence[PositionRecord]
    seed: int | None = None
    terminal: object | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RecordSink(Protocol):
    """Destination for core game record entries."""

    def write_entry(self, entry: PositionRecord) -> None:
        """Persist or forward one core position record as the game runs."""

    def close_game(self, game_id: str, terminal: object | None = None) -> object:
        """Finalize a game record and return a storage reference or manifest."""


class RecordAnalyzer(Protocol):
    """Post-game analyzer for durable records."""

    def analyze(self, record: GameRecord) -> Mapping[str, Any]:
        """Return derived statistics without mutating the source record."""
