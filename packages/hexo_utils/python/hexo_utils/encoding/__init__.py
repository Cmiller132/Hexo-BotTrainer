"""Shared encoding helpers for Hexo model packages.

The encoding package contains reusable crop and mask contracts. Model packages
can opt into these helpers when their tensor semantics match the shared
representation.
"""

from .crop import CropWindow, EncodedCrop, build_crop_window, encode_crop
from .masks import ActionMask, build_legal_mask, mask_from_action_ids

__all__ = [
    "ActionMask",
    "CropWindow",
    "EncodedCrop",
    "build_crop_window",
    "build_legal_mask",
    "encode_crop",
    "mask_from_action_ids",
]
