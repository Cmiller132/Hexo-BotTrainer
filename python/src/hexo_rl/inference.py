"""Synchronous batched inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .model_api import ModelPlugin, TensorBatch


@dataclass(frozen=True)
class NetworkOutput:
    policy_logits: list[float]
    value: float
    height: int
    width: int


def _as_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def encoded_state_tensor(state: Mapping[str, Any]) -> torch.Tensor:
    """Return an encoded state as [C, H, W], including Rust flat-plane output."""

    raw_planes = state.get("state_tensor", state.get("planes"))
    if raw_planes is None:
        raise KeyError("Encoded state must include state_tensor or planes")

    tensor = _as_tensor(raw_planes, dtype=torch.float32)
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim != 1:
        raise ValueError(f"Encoded state planes must be 1D or 3D, got shape {tuple(tensor.shape)}")

    crop_size = int(state.get("crop_size", 0))
    plane_count = int(state.get("plane_count", 0))
    if crop_size <= 0 or plane_count <= 0:
        raise KeyError("Flat encoded planes require crop_size and plane_count")
    expected = crop_size * crop_size * plane_count
    if tensor.numel() != expected:
        raise ValueError(f"Flat encoded planes have {tensor.numel()} cells, expected {expected}")
    return tensor.reshape(plane_count, crop_size, crop_size)


def legal_mask_from_state(state: Mapping[str, Any], state_tensor: torch.Tensor) -> torch.Tensor:
    raw_mask = state.get("legal_mask")
    if raw_mask is not None:
        mask = _as_tensor(raw_mask, dtype=torch.float32)
        return mask.reshape(state_tensor.shape[-2], state_tensor.shape[-1])
    if state_tensor.shape[0] > 2:
        return state_tensor[2].to(dtype=torch.float32)
    return torch.ones_like(state_tensor[0], dtype=torch.float32)


def collate_encoded_states(encoded_states: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Convert Rust/Python encoded state dictionaries into a tensor batch."""

    if not encoded_states:
        raise ValueError("Cannot collate an empty inference batch")

    planes = []
    legal_masks = []
    for state in encoded_states:
        state_tensor = encoded_state_tensor(state)
        planes.append(state_tensor)
        legal_masks.append(legal_mask_from_state(state, state_tensor))

    return {
        "state_tensor": torch.stack(planes, dim=0),
        "legal_mask": torch.stack(legal_masks, dim=0),
    }


def move_to_device(batch: TensorBatch, device: torch.device | str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def convert_outputs_for_rust(outputs: Mapping[str, torch.Tensor]) -> list[NetworkOutput]:
    policy = outputs["policy_logits"].detach().float().cpu()
    value = outputs["value"].detach().float().cpu()
    if policy.ndim != 3:
        raise ValueError("policy_logits must have shape [B, H, W]")
    if value.ndim != 1:
        value = value.reshape(-1)

    results: list[NetworkOutput] = []
    for idx in range(policy.shape[0]):
        logits = policy[idx]
        results.append(
            NetworkOutput(
                policy_logits=logits.reshape(-1).tolist(),
                value=float(value[idx].item()),
                height=int(logits.shape[0]),
                width=int(logits.shape[1]),
            )
        )
    return results


class InferenceServer:
    """Thin synchronous inference facade used by the Rust bridge."""

    def __init__(
        self,
        model: torch.nn.Module,
        plugin: ModelPlugin,
        device: torch.device | str,
        batch_size: int,
        *,
        amp: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.plugin = plugin
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.amp = amp
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, encoded_states: Iterable[Mapping[str, Any]]) -> list[NetworkOutput]:
        states = list(encoded_states)
        outputs: list[NetworkOutput] = []
        for start in range(0, len(states), self.batch_size):
            batch = collate_encoded_states(states[start : start + self.batch_size])
            batch = move_to_device(batch, self.device)
            use_amp = self.amp and self.device.type == "cuda"
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                model_outputs = self.plugin.forward_inference(self.model, batch)
            outputs.extend(convert_outputs_for_rust(model_outputs))
        return outputs
