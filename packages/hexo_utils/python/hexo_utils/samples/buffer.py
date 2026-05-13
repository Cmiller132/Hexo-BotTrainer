"""Shared sample-buffer mechanics.

The samples package has two layers:

1. model-facing sample records and targets, which describe what a training row
   means;
2. storage mechanics, which describe where rows live, how they are indexed, and
   which subset is visible to an epoch.

This file owns the second layer. It intentionally does not know how a model
turns records into tensors, what a policy head means, or how final values are
computed. Keeping the mechanical buffer pieces together makes the current
placeholder-backed flow easier to follow and leaves one obvious file to split
later if real chunk manifests, indexes, or samplers become large enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence


@dataclass(frozen=True, slots=True)
class SampleStore:
    """Directory and metadata for trainable sample chunks.

    The store is the root handle passed between self-play, finalization,
    indexing, and training-window selection. Today it only creates the backing
    directory; future chunk writers can hang manifests and format metadata from
    this same object without changing the training pipeline contract.
    """

    path: Path
    mode: str = "append"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleWriteResult:
    """Summary returned after appending finalized records to a store."""

    count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleIndex:
    """Searchable summary over finalized sample chunks.

    Indexing is separated from the store handle because training should select
    rows from an indexed view rather than scan every chunk directly. The current
    implementation is a placeholder until real chunk IO lands.
    """

    store: SampleStore
    sample_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleWindow:
    """A deterministic slice of indexed samples visible to one training pass."""

    index: SampleIndex
    window_size: int | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleRequest:
    """Request for sampled records from a reusable sample buffer."""

    count: int
    seed: int | None = None
    required_extensions: Sequence[str] = field(default_factory=tuple)
    filters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleBatch:
    """A sampled set of training rows plus provenance metadata."""

    records: Sequence[object]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def open_sample_store(
    path: str | Path,
    *,
    mode: str = "append",
    metadata: Mapping[str, Any] | None = None,
) -> SampleStore:
    """Open or create the directory that will contain sample chunks.

    The training pipeline calls this once when constructing shared defaults.
    Model packages and epoch helpers then pass the returned handle around
    instead of rediscovering paths or duplicating directory setup.
    """

    store_path = Path(path)
    store_path.mkdir(parents=True, exist_ok=True)
    return SampleStore(
        path=store_path,
        mode=mode,
        metadata=dict(metadata or {}),
    )


def append_samples(
    store: SampleStore,
    records: Sequence[object],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> SampleWriteResult:
    """Append finalized sample records to the store.

    This is still placeholder-backed: it validates the shared call shape and
    reports how many records would have been written. Real storage should add
    chunked writes and manifest updates here while keeping model-owned payloads
    opaque to utils.
    """

    _ = store
    return SampleWriteResult(count=len(records), metadata=dict(metadata or {}))


def refresh_sample_index(store: SampleStore) -> SampleIndex:
    """Refresh and return the searchable index for a sample store.

    Once chunk storage exists, this function should rebuild or load the compact
    metadata needed for deterministic sampling. For now it preserves the
    training pipeline boundary without pretending a real index exists.
    """

    return SampleIndex(
        store=store,
        sample_count=0,
        metadata={"note": "Indexing is a placeholder until chunk IO exists."},
    )


def build_sample_window(
    index: SampleIndex,
    *,
    window_size: int | None = None,
    seed: int | None = None,
) -> SampleWindow:
    """Build the sample subset used by one epoch's training passes.

    `window_size` comes from `samples.train_sample_count`. A value of `None`
    means "use the full indexed buffer" once real indexing is implemented.
    """

    return SampleWindow(
        index=index,
        window_size=window_size,
        seed=seed,
        metadata={"note": "Window selection is a placeholder."},
    )


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

    _ = source, request
    _not_implemented("sample_training_samples")


def _not_implemented(operation: str) -> NoReturn:
    """Raise a consistent message for buffer mechanics that are not real yet."""

    raise NotImplementedError(f"{operation} will be backed by sample buffer helpers.")
