"""Shared D6 symmetry contracts for hex-board training data.

The shared layer owns how a symmetry is identified and transported.
`hexo_train` owns when a training sample receives a symmetry. Engine/model code
owns how concrete coordinates, action ids, tensors, and custom targets are
transformed under that symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


D6_SIZE = 12


@dataclass(frozen=True, slots=True)
class D6Symmetry:
    """One element of the hex board dihedral symmetry group."""

    index: int

    def __post_init__(self) -> None:
        if not 0 <= self.index < D6_SIZE:
            raise ValueError(f"D6 symmetry index must be in [0, {D6_SIZE}); got {self.index}")


IDENTITY_D6 = D6Symmetry(0)


class ActionSymmetryMapper(Protocol):
    """Model or engine adapter that transforms stable action ids."""

    def transform_action_id(self, action_id: str, symmetry: D6Symmetry) -> str:
        """Return the action id after applying `symmetry`."""


def transform_action_ids(
    action_ids: Sequence[str],
    symmetry: D6Symmetry,
    mapper: ActionSymmetryMapper,
) -> tuple[str, ...]:
    """Transform stable action ids while preserving their policy-logit order."""

    return tuple(mapper.transform_action_id(action_id, symmetry) for action_id in action_ids)
