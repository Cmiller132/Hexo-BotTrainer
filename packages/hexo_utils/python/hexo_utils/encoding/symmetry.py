"""Shared D6 symmetry contracts for hex-board training data.

The shared layer owns how a symmetry is identified and sampled. Engine/model
code owns how concrete coordinates, action ids, tensors, and custom targets are
transformed under that symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
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


def choose_d6_symmetry(
    *,
    seed: int,
    epoch: int,
    sample_index: int,
    game_id: str,
    turn_index: int,
) -> D6Symmetry:
    """Choose a deterministic pseudo-random D6 symmetry for one sample."""

    material = f"{seed}:{epoch}:{sample_index}:{game_id}:{turn_index}".encode("utf-8")
    digest = blake2b(material, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return D6Symmetry(value % D6_SIZE)


def transform_action_ids(
    action_ids: Sequence[str],
    symmetry: D6Symmetry,
    mapper: ActionSymmetryMapper,
) -> tuple[str, ...]:
    """Transform stable action ids while preserving their policy-logit order."""

    return tuple(mapper.transform_action_id(action_id, symmetry) for action_id in action_ids)
