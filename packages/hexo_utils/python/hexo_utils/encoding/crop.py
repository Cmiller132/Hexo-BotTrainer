"""Optional shared crop encoding boundary.

Model packages may use this module when they agree on a common board crop
representation. The engine still supplies canonical state and legal actions;
this module only describes reusable crop requests and transport shapes. Models
may bypass this package entirely when they need whole-board, tokenized,
transformer, or otherwise custom input representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, NoReturn, Sequence


class CropShape(StrEnum):
    """Shared crop shapes supported by reusable encoders."""

    SQUARE = "square"
    CIRCULAR = "circular"
    MODEL_DEFINED = "model_defined"


@dataclass(frozen=True, slots=True)
class CropRequest:
    """Model request for zero, one, or many reusable crop windows."""

    size: int
    shape: CropShape = CropShape.SQUARE
    centers: Sequence[object] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CropWindow:
    """One model view centered on an engine coordinate or model-defined point."""

    center: object
    size: int
    shape: CropShape = CropShape.SQUARE
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EncodedCrop:
    """Transport shape for a shared crop tensor and its metadata."""

    tensor: object
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EncodedCrops:
    """Transport shape for models that request multiple shared crops."""

    crops: Sequence[EncodedCrop]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by the shared encoder.")


def build_crop_windows(state: object, request: CropRequest) -> tuple[CropWindow, ...]:
    """Choose reusable crop windows for a model input from an engine state."""

    _not_implemented("build_crop_windows")


def build_crop_window(state: object, *, size: int) -> CropWindow:
    """Compatibility helper for models that need one square crop."""

    _not_implemented("build_crop_window")


def encode_crop(state: object, window: CropWindow) -> EncodedCrop:
    """Encode an engine state into the shared crop representation."""

    _not_implemented("encode_crop")


def encode_crops(state: object, windows: Sequence[CropWindow]) -> EncodedCrops:
    """Encode many crop windows without defining model-specific tensors."""

    _not_implemented("encode_crops")
