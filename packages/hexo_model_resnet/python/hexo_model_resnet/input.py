"""State-to-input conversion for the ResNet model family.

This module owns how engine state, legal actions, and optional tactical facts
become ResNet tensors. Shared encoders from `hexo_utils.encoding` may be used
when their semantics match this model's representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn


@dataclass(frozen=True, slots=True)
class ResNetInput:
    """Model input tensors and action mapping for one decision or batch."""

    state_tensor: object
    legal_mask: object
    action_ids: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by ResNet input encoding.")


def build_input(state: object, legal_actions: object, *, config: object | None = None) -> ResNetInput:
    """Convert engine context into ResNet input tensors."""

    _not_implemented("build_input")
