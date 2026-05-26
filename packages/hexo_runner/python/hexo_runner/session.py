"""Runner game and batch specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .player import GameContext, PlayerFactory


@dataclass(frozen=True, slots=True)
class GameSpec:
    """Inputs needed to create one engine-backed game.

    The runner passes `seed` and `scenario` through to `hexo_engine.new_game`;
    it does not interpret game rules or scenario contents itself.
    """

    game_id: str
    seed: int | None = None
    scenario: object | None = None
    mode: str = "match"
    is_evaluation: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


SessionSpec = GameSpec
SessionContext = GameContext


@dataclass(frozen=True, slots=True)
class BatchSpec:
    """Local multiprocessing batch request for this machine."""

    batch_id: str
    games: Sequence[GameSpec]
    player_factories: tuple[PlayerFactory, PlayerFactory]
    output_dir: str | Path = Path("data/replay")
    worker_count: int | None = None
    chunk_size: int = 32
    metadata: Mapping[str, Any] = field(default_factory=dict)
