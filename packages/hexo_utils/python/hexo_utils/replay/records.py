"""Shared replay record shapes.

The engine contributes accepted actions and snapshots. The runner contributes
players, seeds, budgets, timings, and run outcome. Models may attach opaque
diagnostics that remain owned by the producing model package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EngineReplayRecord:
    """Rules-authoritative transition data for one accepted action."""

    game_id: str
    turn_index: int
    before_snapshot: object
    action: object
    after_snapshot: object
    terminal: object | None = None


@dataclass(frozen=True, slots=True)
class RunnerReplayRecord:
    """Execution metadata recorded around an engine transition."""

    game_id: str
    turn_index: int
    player_id: str
    elapsed_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelDiagnosticsRecord:
    """Opaque model diagnostics transported alongside replay."""

    game_id: str
    turn_index: int
    model_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
