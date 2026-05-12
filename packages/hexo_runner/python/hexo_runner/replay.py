"""Runner replay recording boundary.

The runner records execution metadata around engine transitions. It does not
reinterpret game legality or turn snapshots, and it leaves model diagnostics
opaque for the model package that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class RunnerReplayEntry:
    """Runner-owned metadata for one recorded transition."""

    game_id: str
    turn_index: int
    player_id: str
    engine_record: object
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class ReplaySink(Protocol):
    """Destination for runner replay entries."""

    def write(self, entry: RunnerReplayEntry) -> None:
        """Persist or forward one runner replay entry."""
