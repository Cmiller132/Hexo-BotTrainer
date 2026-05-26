"""Runner game and batch specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .player import GameContext, PlayerFactory


@dataclass(frozen=True, slots=True)
class GameSpec:
    """Inputs needed to create one engine-backed game.

    The runner passes `seed` through to `hexo_engine.new_game`. Durable runner
    records do not persist scenarios yet, so recorded runs require
    `scenario=None`.
    """

    game_id: str
    seed: int | None = None
    scenario: object | None = None
    mode: str = "match"
    is_evaluation: bool = False
    max_actions: int = 1024
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_actions <= 0:
            raise ValueError("GameSpec.max_actions must be positive.")


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
