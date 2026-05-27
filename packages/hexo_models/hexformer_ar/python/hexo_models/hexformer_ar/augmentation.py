"""D6 augmentation for Hexformer sparse samples."""

from __future__ import annotations

from dataclasses import replace

import torch

from .d6 import transform_action_id, transform_coord
from .coordinates import Axial
from .input import SparseDecisionInput


def transform_sparse_input(sample: SparseDecisionInput, symmetry: object) -> SparseDecisionInput:
    """Apply one D6 transform to sparse coordinates, action IDs, and local crops."""

    index = int(getattr(symmetry, "index", symmetry))
    if index == 0:
        return sample
    anchor = _anchor(sample)
    candidate_action_ids = tuple(
        transform_action_id(action_id, index, center=anchor)
        for action_id in sample.candidate_action_ids
    )
    candidate_coords = _transform_coord_tensor(sample.candidate_coords, index)
    stone_coords = _transform_coord_tensor(sample.stone_coords, index)
    window_coords = _transform_coord_tensor(sample.window_coords, index)
    local_window_coords = _transform_coord_tensor(sample.local_window_coords, index)
    candidate_features = _replace_feature_coords(sample.candidate_features, candidate_coords, 5, 10)
    stone_features = _replace_feature_coords(sample.stone_features, stone_coords, 2, 7)
    window_features = _replace_feature_coords(sample.window_features, window_coords, 10, 15)
    window_features = _transform_window_axes(window_features, index)
    local_input = _transform_local_crop(sample.local_input, index)
    local_inputs = torch.stack([_transform_local_crop(item, index) for item in sample.local_inputs], dim=0) if sample.local_inputs.numel() else sample.local_inputs
    rel_edge_features = _transform_edge_features(sample.rel_edge_features, index)
    return replace(
        sample,
        candidate_action_ids=candidate_action_ids,
        candidate_features=candidate_features,
        candidate_coords=candidate_coords,
        stone_features=stone_features,
        stone_coords=stone_coords,
        window_features=window_features,
        window_coords=window_coords,
        local_input=local_input,
        local_inputs=local_inputs,
        local_window_coords=local_window_coords,
        rel_edge_features=rel_edge_features,
        metadata={**dict(sample.metadata), "d6_symmetry": index},
    )


def _anchor(sample: SparseDecisionInput) -> Axial:
    raw = sample.metadata.get("anchor", (0, 0))
    return Axial(int(raw[0]), int(raw[1]))


def _transform_coord_tensor(coords: torch.Tensor, symmetry: int) -> torch.Tensor:
    if coords.numel() == 0:
        return coords.clone()
    out = coords.clone()
    for row in range(coords.shape[0]):
        coord = transform_coord(Axial(int(coords[row, 0].item()), int(coords[row, 1].item())), symmetry)
        ds = -coord.q - coord.r
        distance = max(abs(coord.q), abs(coord.r), abs(ds))
        out[row, 0:5] = torch.tensor([coord.q, coord.r, ds, distance, distance], dtype=out.dtype, device=out.device)
    return out


def _replace_feature_coords(features: torch.Tensor, coords: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    if features.numel() == 0:
        return features.clone()
    out = features.clone()
    width = min(stop - start, coords.shape[-1], max(0, out.shape[-1] - start))
    if width > 0:
        out[:, start : start + width] = coords[:, :width].to(dtype=out.dtype, device=out.device)
    return out


def _transform_edge_features(features: torch.Tensor, symmetry: int) -> torch.Tensor:
    if features.numel() == 0 or features.shape[-1] < 3:
        return features.clone()
    out = features.clone()
    for row in range(features.shape[0]):
        coord = transform_coord(Axial(int(features[row, 0].item()), int(features[row, 1].item())), symmetry)
        ds = -coord.q - coord.r
        distance = max(abs(coord.q), abs(coord.r), abs(ds))
        out[row, 0:4] = torch.tensor([coord.q, coord.r, ds, distance], dtype=out.dtype, device=out.device)
        if out.shape[-1] >= 7:
            out[row, 4] = 1.0 if coord.q == 0 else 0.0
            out[row, 5] = 1.0 if coord.r == 0 else 0.0
            out[row, 6] = 1.0 if ds == 0 else 0.0
    return out


def _transform_window_axes(features: torch.Tensor, symmetry: int) -> torch.Tensor:
    if features.numel() == 0 or features.shape[-1] < 3:
        return features
    out = features.clone()
    for row in range(features.shape[0]):
        axis = int(torch.argmax(features[row, 0:3]).item())
        transformed_axis = _transform_axis(axis, symmetry)
        out[row, 0:3] = 0.0
        out[row, transformed_axis] = 1.0
    return out


def _transform_axis(axis_index: int, symmetry: int) -> int:
    vector = (Axial(1, 0), Axial(0, 1), Axial(1, -1))[axis_index]
    transformed = transform_coord(vector, symmetry)
    pair = (transformed.q, transformed.r)
    if pair in {(1, 0), (-1, 0)}:
        return 0
    if pair in {(0, 1), (0, -1)}:
        return 1
    if pair in {(1, -1), (-1, 1)}:
        return 2
    raise RuntimeError(f"D6 transform produced non-axis vector {pair!r}")


def _transform_local_crop(crop: torch.Tensor, symmetry: int) -> torch.Tensor:
    if crop.numel() == 0:
        return crop.clone()
    out = torch.zeros_like(crop)
    height = crop.shape[-2]
    width = crop.shape[-1]
    row_half = height // 2
    col_half = width // 2
    for row in range(height):
        for col in range(width):
            transformed = transform_coord(Axial(col - col_half, row - row_half), symmetry)
            dst_col = transformed.q + col_half
            dst_row = transformed.r + row_half
            if 0 <= dst_col < width and 0 <= dst_row < height:
                out[..., dst_row, dst_col] = crop[..., row, col]
    return out
