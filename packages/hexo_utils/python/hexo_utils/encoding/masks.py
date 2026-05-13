"""Shared action mask helpers.

Masks are derived from engine-provided legal actions. They should never decide
legality themselves; their job is to translate legal action identities into the
shape expected by shared encoders or model packages.

Threat filtering is allowed here only as a common post-process over engine
facts: start with engine legal actions, read engine tactical summaries, and
return a smaller legal-action list for models that want forced tactical play.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, Sequence

@dataclass(frozen=True, slots=True)
class ActionMask:
    """Model-facing legal-action mask plus the action ids it represents."""

    mask: object
    action_ids: tuple[str, ...]
    legal_actions: tuple[object, ...] = ()


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by engine legal actions.")


def build_legal_mask(legal_actions: Sequence[object], *, shape: tuple[int, ...]) -> ActionMask:
    """Build a model-shaped mask from engine-provided legal actions."""

    _not_implemented("build_legal_mask")


def mask_from_action_ids(action_ids: Sequence[str], *, shape: tuple[int, ...]) -> ActionMask:
    """Build a mask when samples or search already carry action identities."""

    _not_implemented("mask_from_action_ids")


def filter_threat_legal_actions(
    legal_actions: Sequence[object],
    tactical_summary: object,
    *,
    include_winning: bool = True,
    include_blocking: bool = True,
) -> tuple[object, ...]:
    """Return only forced tactical legal actions when engine tactics expose them.

    Pseudo-code for the eventual implementation:

    ```text
    winning_ids = tactical_summary.immediate_win_action_ids if include_winning
    blocking_ids = tactical_summary.must_block_action_ids if include_blocking
    forced_ids = winning_ids or blocking_ids
    if forced_ids is empty:
        return tuple(legal_actions)
    return tuple(action for action in legal_actions if action.id in forced_ids)
    ```

    The function must never add actions, invent tactics, or validate legality.
    It can only strip the engine-provided legal list based on engine-provided
    tactical facts.
    """

    _not_implemented("filter_threat_legal_actions")


def build_threat_legal_mask(
    legal_actions: Sequence[object],
    tactical_summary: object,
    *,
    shape: tuple[int, ...],
) -> ActionMask:
    """Build a mask after optional threat filtering of engine legal actions."""

    _not_implemented("build_threat_legal_mask")
