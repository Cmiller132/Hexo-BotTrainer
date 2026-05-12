"""Self-play runner mode.

Self-play generates games from model-backed players and writes durable game
records for training. It should not build model tensors itself; model packages
read records and decide how to encode them for their own training loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """State and legal actions passed to a model/search adapter."""

    state: object
    legal_actions: Sequence[object]
    is_evaluation: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


class InferenceAdapter(Protocol):
    """Policy/value bridge used by self-play players and search.

    A model package implements this adapter by loading its checkpoint and
    returning common policy logits over the supplied legal actions plus a value
    estimate. Any model-specific extras stay in model-owned extension records.
    """

    def evaluate(self, request: InferenceRequest) -> Mapping[str, Any]:
        """Return policy/value data for the supplied engine context."""


def run_selfplay_cycle(config: object) -> object:
    """Generate self-play records once player, batch, and storage wiring exist.

    Intended flow:

    1. Load self-play config and model/checkpoint references.
    2. Ask the model package to create an `InferenceAdapter`.
    3. Wrap the adapter in runner players.
    4. Build match configs and call batch mode.
    5. Write game records with engine history, runner metadata, common policy
       logits, and model-owned extension records.
    6. Return a manifest or record reference for the training package to read.
    """

    raise NotImplementedError("self-play will be built on batch mode and records.")
