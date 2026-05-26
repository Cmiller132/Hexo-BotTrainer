"""Python runner package for headless Hexo execution.

The runner owns sessions, player lifecycle, single-game loops, run modes, and
record/result emission. It applies actions through `hexo_engine` and treats
model/search internals as player-owned details.
"""

__version__ = "0.1.0"

from .player import (
    DecisionResult,
    FinalSummary,
    GameContext,
    PlayerFactory,
    PlayerIdentity,
    RunnerPlayer,
    TransitionEvent,
    WorkerContext,
)
from .records import BatchResult, GameRecordV1, GameResult, GameStatus, JsonlRecordSink, MemoryRecordSink
from .session import BatchSpec, GameSpec

__all__ = [
    "BatchResult",
    "BatchSpec",
    "DecisionResult",
    "FinalSummary",
    "GameContext",
    "GameRecordV1",
    "GameResult",
    "GameSpec",
    "GameStatus",
    "JsonlRecordSink",
    "MemoryRecordSink",
    "PlayerFactory",
    "PlayerIdentity",
    "RunnerPlayer",
    "TransitionEvent",
    "WorkerContext",
]
