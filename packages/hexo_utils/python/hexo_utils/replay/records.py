"""Shared replay record shapes.

The engine contributes accepted actions and snapshots. The runner contributes
players, seeds, and run outcome. Models may attach opaque extension records
that remain owned by the producing model package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


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
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyLogitRecord:
    """Common policy output over the legal actions offered by the engine.

    `legal_action_ids` defines the order of `logits`. Model packages can turn
    that compact vector into their own target tensors. More complex policy
    heads, pair policies, search traces, or architecture-specific data should
    be stored as model extension records instead of being modeled here. Large
    arrays may be represented by references rather than kept resident in RAM.
    """

    game_id: str
    turn_index: int
    model_id: str
    legal_action_ids: tuple[str, ...]
    selected_action_id: str | None = None
    logits: object | None = None
    logits_ref: object | None = None
    value: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelExtensionRecord:
    """Opaque model-owned extension attached to a replay decision.

    The shared replay layer only knows the extension namespace and version.
    The model package owns the payload schema and parsing code. Large extension
    payloads may be stored out-of-line behind `payload_ref`.
    """

    game_id: str
    turn_index: int
    model_id: str
    namespace: str
    schema_version: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    payload_ref: object | None = None


@dataclass(frozen=True, slots=True)
class ReplayDecisionRecord:
    """Joined decision record used by samplers and training readers."""

    engine: EngineReplayRecord
    runner: RunnerReplayRecord
    policy: PolicyLogitRecord | None = None
    model_extensions: Sequence[ModelExtensionRecord] = field(default_factory=tuple)
