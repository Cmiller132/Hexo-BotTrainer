"""Shared encoding helpers for Hexo model packages.

The encoding package contains reusable crop and mask contracts. Model packages
can opt into these helpers when their tensor semantics match the shared
representation.
"""

from .crop import (
    CropRequest,
    CropShape,
    CropWindow,
    EncodedCrop,
    EncodedCrops,
    build_crop_window,
    build_crop_windows,
    encode_crop,
    encode_crops,
)
from .masks import (
    ActionMask,
    build_legal_mask,
    build_threat_legal_mask,
    filter_threat_legal_actions,
    mask_from_action_ids,
)
from .symmetry import (
    D6_SIZE,
    IDENTITY_D6,
    ActionSymmetryMapper,
    D6Symmetry,
    transform_action_ids,
)

__all__ = [
    "ActionMask",
    "ActionSymmetryMapper",
    "CropRequest",
    "CropShape",
    "CropWindow",
    "D6_SIZE",
    "D6Symmetry",
    "EncodedCrop",
    "EncodedCrops",
    "IDENTITY_D6",
    "build_crop_window",
    "build_crop_windows",
    "build_legal_mask",
    "build_threat_legal_mask",
    "encode_crop",
    "encode_crops",
    "filter_threat_legal_actions",
    "mask_from_action_ids",
    "transform_action_ids",
]
