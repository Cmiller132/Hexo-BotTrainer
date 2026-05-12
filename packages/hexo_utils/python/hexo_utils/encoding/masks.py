"""Shared action mask helpers.

Masks are derived from engine-provided legal actions. They should never decide
legality themselves; their job is to translate legal action identities into the
shape expected by shared encoders or model packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, Sequence


@dataclass(frozen=True, slots=True)
class ActionMask:
    """Model-facing legal-action mask plus the action ids it represents."""

    mask: object
    action_ids: tuple[str, ...]


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by engine legal actions.")


def build_legal_mask(legal_actions: Sequence[object], *, shape: tuple[int, ...]) -> ActionMask:
    """Build a model-shaped mask from engine-provided legal actions."""

    _not_implemented("build_legal_mask")


def mask_from_action_ids(action_ids: Sequence[str], *, shape: tuple[int, ...]) -> ActionMask:
    """Build a mask when replay or search already carries action identities."""

    _not_implemented("mask_from_action_ids")
