"""Training replay record shapes.

Core game records are written by `hexo_runner.records`. This module describes
training-oriented replay layers that can reference those core records, attach a
common policy output over legal actions, and carry model-owned extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


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
    """Training-facing replay record for one recorded decision.

    `source_record_ref` points at the runner's durable core position record.
    This keeps the replay layer extensible for model training without making
    utils responsible for owning the authoritative game record.
    """

    game_id: str
    turn_index: int
    source_record_ref: object
    legal_action_ids: tuple[str, ...]
    policy: PolicyLogitRecord | None = None
    model_extensions: Sequence[ModelExtensionRecord] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
