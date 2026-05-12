"""Shared crop encoding boundary.

Model packages may use this module when they agree on a common board crop
representation. The engine still supplies canonical state and legal actions;
this module only describes how reusable encoders should package that state for
model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn


@dataclass(frozen=True, slots=True)
class CropWindow:
    """A square model view centered on an engine coordinate."""

    center: object
    size: int


@dataclass(frozen=True, slots=True)
class EncodedCrop:
    """Transport shape for a shared crop tensor and its metadata."""

    tensor: object
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by the shared encoder.")


def build_crop_window(state: object, *, size: int) -> CropWindow:
    """Choose the crop window for a model input from an engine state."""

    _not_implemented("build_crop_window")


def encode_crop(state: object, window: CropWindow) -> EncodedCrop:
    """Encode an engine state into the shared crop representation."""

    _not_implemented("encode_crop")
