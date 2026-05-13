"""Default replay target builders for common policy/value models.

This module handles only the shared case: a policy logit for each legal action
plus an optional scalar value. Model-specific heads, pair targets, search
traces, and auxiliary labels remain model-owned extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hexo_utils.encoding.symmetry import (
    IDENTITY_D6,
    ActionSymmetryMapper,
    D6Symmetry,
    transform_action_ids,
)

from .records import ReplayDecisionRecord


@dataclass(frozen=True, slots=True)
class LegalPolicyValueTarget:
    """Default target for models trained on legal-action policy logits/value."""

    legal_action_ids: tuple[str, ...]
    policy_logits: object | None = None
    policy_logits_ref: object | None = None
    selected_action_id: str | None = None
    value: float | None = None
    symmetry: D6Symmetry = IDENTITY_D6
    metadata: Mapping[str, Any] = field(default_factory=dict)


def build_legal_policy_value_target(
    record: ReplayDecisionRecord,
    *,
    symmetry: D6Symmetry = IDENTITY_D6,
    action_mapper: ActionSymmetryMapper | None = None,
) -> LegalPolicyValueTarget:
    """Build the shared policy/value target for one replay decision.

    The policy vector stays paired with legal-action order. Under symmetry, the
    action ids are transformed but their associated logits remain in the same
    sequence positions. Dense tensors, masks, and model-specific targets are
    still built by the model package after it receives this target.
    """

    if record.policy is None:
        raise ValueError("ReplayDecisionRecord has no common policy record")

    if symmetry != IDENTITY_D6 and action_mapper is None:
        raise ValueError("non-identity symmetry requires an action_mapper")

    legal_action_ids = tuple(record.legal_action_ids)
    selected_action_id = record.policy.selected_action_id

    if action_mapper is not None:
        legal_action_ids = transform_action_ids(legal_action_ids, symmetry, action_mapper)
        if selected_action_id is not None:
            selected_action_id = action_mapper.transform_action_id(selected_action_id, symmetry)

    _validate_logits_shape(legal_action_ids, record.policy.logits)

    return LegalPolicyValueTarget(
        legal_action_ids=legal_action_ids,
        policy_logits=record.policy.logits,
        policy_logits_ref=record.policy.logits_ref,
        selected_action_id=selected_action_id,
        value=record.policy.value,
        symmetry=symmetry,
        metadata=record.policy.metadata,
    )


def _validate_logits_shape(action_ids: Sequence[str], logits: object | None) -> None:
    """Catch obvious mismatches without requiring a tensor dependency."""

    if logits is None or not hasattr(logits, "__len__"):
        return
    if len(logits) != len(action_ids):
        raise ValueError("policy logits length must match legal_action_ids length")
