"""Training sample buffer and sampling boundary.

Model packages write trainable samples during self-play. Shared sampling
utilities should select those samples and preserve schema metadata without
constructing model tensors or choosing training-time augmentation policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Sequence

@dataclass(frozen=True, slots=True)
class SampleRequest:
    """Request for a reusable sample buffer."""

    count: int
    seed: int | None = None
    required_extensions: Sequence[str] = field(default_factory=tuple)
    filters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleBatch:
    """A sampled set of training samples plus provenance metadata."""

    records: Sequence[object]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by sample buffer helpers.")


def sample_training_samples(source: object, request: SampleRequest) -> SampleBatch:
    """Sample trainable records without constructing model tensors.

    The eventual sampler should:

    1. read model-written sample chunks by schema version;
    2. select samples deterministically from `seed` and filters;
    3. keep legal-action ordering and action ids intact;
    4. include only requested model payload namespaces when asked;
    5. return samples for `hexo_train` to attach training-time selections such
       as D6 symmetries before model decoding.
    """

    _not_implemented("sample_training_samples")
