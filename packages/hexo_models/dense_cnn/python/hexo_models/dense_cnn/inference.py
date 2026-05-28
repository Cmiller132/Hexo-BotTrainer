"""Inference adapter for the dense CNN model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from . import rust_bridge
from .constants import BOARD_SIZE, PLANE_LEGAL
from .d6 import unpack_coord_pair
from .losses import decode_binned_value
from .samples import CompressedSample, Model1SampleData, expand_sample, stack_expanded


@dataclass(frozen=True, slots=True)
class InferenceResult:
    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    value: float
    legal_action_ids: tuple[int, ...]
    legal_priors: dict[int, float]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class DenseCNNInference:
    """Owns tensor construction, device movement, and policy/value decoding."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | str = "cpu",
        amp: bool = False,
        return_logits: bool = True,
        max_batch_size: int | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.amp = bool(amp and self.device.type == "cuda")
        self.return_logits = bool(return_logits)
        self.max_batch_size = max(1, int(max_batch_size or 1024))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.model.to(device=self.device, memory_format=torch.channels_last)
        else:
            self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer_state(self, state: object) -> InferenceResult:
        return self.infer_states([state])[0]

    @torch.no_grad()
    def infer_states(self, states: Sequence[object]) -> list[InferenceResult]:
        if not states:
            return []
        return self._infer_states_rust(states)

    @torch.no_grad()
    def _infer_states_rust(self, states: Sequence[object]) -> list[InferenceResult]:
        payload = rust_bridge.model1_batch_inputs(states)
        shape = tuple(int(item) for item in payload["shape"])
        inputs = torch.frombuffer(payload["inputs"], dtype=torch.float32).reshape(shape)
        outputs = self._forward_inputs_device(inputs)
        policy_batch = outputs["policy"]
        value_batch = outputs["value"]
        values = decode_binned_value(value_batch)
        legal_action_rows = tuple(tuple(int(item) for item in row) for row in payload["legal_action_ids"])
        legal_flat_rows = tuple(tuple(int(item) for item in row) for row in payload["legal_flat_indices"])
        results: list[InferenceResult] = []
        for index, (legal_action_ids, legal_flat_indices) in enumerate(zip(legal_action_rows, legal_flat_rows)):
            policy_logits = policy_batch[index]
            value_logits = value_batch[index]
            results.append(
                InferenceResult(
                    policy_logits=policy_logits.cpu() if self.return_logits else torch.empty(0),
                    value_logits=value_logits.cpu() if self.return_logits else torch.empty(0),
                    value=float(values[index].item()),
                    legal_action_ids=legal_action_ids,
                    legal_priors=_legal_priors_from_flats(policy_logits, legal_action_ids, legal_flat_indices),
                    diagnostics={"device": str(self.device), "amp": self.amp, "batched": len(states) > 1, "encoder": "rust"},
                )
            )
        return results

    @torch.no_grad()
    def infer_samples(
        self,
        samples: Sequence[CompressedSample | Model1SampleData],
        *,
        batch_size: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Run batch inference over compact samples without changing them."""

        if not samples:
            return []
        resolved_batch = max(1, int(batch_size or len(samples)))
        outputs: list[dict[str, torch.Tensor]] = []
        for start in range(0, len(samples), resolved_batch):
            chunk = samples[start : start + resolved_batch]
            batch = stack_expanded([expand_sample(sample) for sample in chunk])
            model_outputs = self.infer_inputs(batch["input"])
            for row in range(model_outputs["policy"].shape[0]):
                outputs.append({key: value[row].cpu() for key, value in model_outputs.items()})
        return outputs

    @torch.no_grad()
    def infer_batch(
        self,
        samples: Sequence[CompressedSample | Model1SampleData],
    ) -> list[InferenceResult]:
        """Return decoded inference results for a compact-sample batch."""

        if not samples:
            return []
        expanded = [expand_sample(sample) for sample in samples]
        batch = stack_expanded(expanded)
        model_outputs = self.infer_inputs(batch["input"], batch_size=len(samples))
        results: list[InferenceResult] = []
        for index, sample in enumerate(samples):
            data = sample.decode() if isinstance(sample, CompressedSample) else sample
            policy_logits = model_outputs["policy"][index]
            value_logits = model_outputs["value"][index]
            value = float(decode_binned_value(value_logits.unsqueeze(0))[0].item())
            priors = _legal_priors(policy_logits, data.legal_action_ids, data.center)
            results.append(
                InferenceResult(
                    policy_logits=policy_logits,
                    value_logits=value_logits,
                    value=value,
                    legal_action_ids=data.legal_action_ids,
                    legal_priors=priors,
                    diagnostics={"device": str(self.device), "amp": self.amp, "batched": True},
                )
            )
        return results

    @torch.no_grad()
    def infer_inputs(self, inputs: torch.Tensor, *, batch_size: int | None = None) -> dict[str, torch.Tensor]:
        """Run batch inference over already-dense model input tensors."""

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        resolved_batch = max(1, int(batch_size or inputs.shape[0]))
        chunks: list[dict[str, torch.Tensor]] = []
        for start in range(0, inputs.shape[0], resolved_batch):
            chunk = inputs[start : start + resolved_batch].to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                output = self.model(chunk)
            chunks.append({key: value.detach().float().cpu() for key, value in output.items()})
        return {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
        }

    @torch.no_grad()
    def _forward_inputs_device(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = _to_inference_device(inputs, self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            if hasattr(self.model, "forward_policy_value"):
                output = self.model.forward_policy_value(inputs)
            else:
                output = self.model(inputs)
        return {key: value.detach().float() for key, value in output.items()}

    @torch.no_grad()
    def evaluate_model1_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Torch callback used by the Rust batched MCTS bridge."""

        shape = tuple(int(item) for item in payload["shape"])
        inputs = torch.frombuffer(payload["inputs"], dtype=torch.float32).reshape(shape)
        max_batch = self.max_batch_size
        if inputs.shape[0] > max_batch:
            output_chunks = []
            for start in range(0, inputs.shape[0], max_batch):
                output_chunks.append({
                    key: value.cpu()
                    for key, value in self._forward_inputs_device(inputs[start : start + max_batch]).items()
                })
            policy_batch = torch.cat([chunk["policy"] for chunk in output_chunks], dim=0)
            value_batch = torch.cat([chunk["value"] for chunk in output_chunks], dim=0)
        else:
            outputs = self._forward_inputs_device(inputs)
            policy_batch = outputs["policy"]
            value_batch = outputs["value"]
        values_tensor = decode_binned_value(value_batch).cpu().contiguous()
        priors: list[torch.Tensor] = []
        selected_ordinals: torch.Tensor | None = None
        selected_flats: torch.Tensor | None = None
        selected_offsets: Sequence[int] | None = None
        if payload.get("legal_mask_from_inputs") and int(payload.get("max_prior_candidates") or 0) > 0:
            priors_tensor, selected_flats, selected_offsets = _topk_legal_priors_from_input_mask(
                policy_batch=policy_batch,
                inputs=inputs,
                max_candidates=int(payload["max_prior_candidates"]),
            )
        elif "legal_flat_indices_bytes" in payload:
            if len(payload["legal_flat_indices_bytes"]) == 0:
                priors_tensor = torch.empty(0, dtype=torch.float32)
            else:
                flat_index_buffer = torch.frombuffer(payload["legal_flat_indices_bytes"], dtype=torch.int64)
                offsets = tuple(int(item) for item in payload["legal_row_offsets"])
                counts = torch.as_tensor(
                    [max(0, offsets[index + 1] - offsets[index]) for index in range(len(offsets) - 1)],
                    dtype=torch.long,
                )
                if int(counts.sum().item()) > 0:
                    max_candidates = int(payload.get("max_prior_candidates") or 0)
                    if max_candidates > 0:
                        priors_tensor, selected_ordinals, selected_offsets = _topk_legal_priors(
                            policy_batch=policy_batch,
                            flat_index_buffer=flat_index_buffer,
                            offsets=offsets,
                            max_candidates=max_candidates,
                        )
                    else:
                        row_ids = torch.repeat_interleave(torch.arange(len(counts), dtype=torch.long), counts)
                        flat_indices = flat_index_buffer.to(policy_batch.device, non_blocking=True)
                        row_ids_device = row_ids.to(policy_batch.device, non_blocking=True)
                        selected = policy_batch[row_ids_device, flat_indices]
                        max_per_row = torch.full(
                            (len(counts),),
                            float("-inf"),
                            dtype=selected.dtype,
                            device=selected.device,
                        )
                        max_per_row.scatter_reduce_(0, row_ids_device, selected, reduce="amax", include_self=True)
                        exp = torch.exp(selected - max_per_row[row_ids_device])
                        sum_per_row = torch.zeros((len(counts),), dtype=selected.dtype, device=selected.device)
                        sum_per_row.scatter_add_(0, row_ids_device, exp)
                        priors_tensor = (exp / sum_per_row[row_ids_device].clamp_min(1.0e-8)).cpu().contiguous()
                        selected_ordinals = None
                        selected_offsets = None
                else:
                    priors_tensor = torch.empty(0, dtype=torch.float32)
                    selected_ordinals = torch.empty(0, dtype=torch.int64)
                    selected_offsets = [0 for _ in range(len(offsets))]
        else:
            for logits, flat_indices in zip(policy_batch, payload["legal_flat_indices"]):
                flats = tuple(int(item) for item in flat_indices)
                if not flats:
                    continue
                index = torch.as_tensor(flats, dtype=torch.long, device=logits.device)
                priors.append(torch.softmax(logits.index_select(0, index), dim=0).cpu())
            priors_tensor = (
                torch.cat(priors).contiguous()
                if priors
                else torch.empty(0, dtype=torch.float32)
            )
        result: dict[str, Any] = {
            "values_bytes": values_tensor.numpy().tobytes(),
            "priors_bytes": priors_tensor.numpy().tobytes(),
        }
        if "legal_flat_indices_bytes" in payload and selected_ordinals is not None and selected_offsets is not None:
            result["selected_legal_ordinals_bytes"] = selected_ordinals.contiguous().numpy().tobytes()
            result["selected_row_offsets"] = tuple(int(item) for item in selected_offsets)
        if payload.get("legal_mask_from_inputs") and selected_flats is not None and selected_offsets is not None:
            result["selected_flat_indices_bytes"] = selected_flats.contiguous().numpy().tobytes()
            result["selected_row_offsets"] = tuple(int(item) for item in selected_offsets)
        return result


def _legal_priors(
    logits: torch.Tensor,
    legal_action_ids: Sequence[int],
    center: tuple[int, int],
) -> dict[int, float]:
    if not legal_action_ids:
        return {}
    flats: list[int] = []
    valid_ids: list[int] = []
    center_q, center_r = int(center[0]), int(center[1])
    half = BOARD_SIZE // 2
    for action_id in legal_action_ids:
        q, r = unpack_coord_pair(action_id)
        row = r - center_r + half
        col = q - center_q + half
        if not 0 <= row < BOARD_SIZE or not 0 <= col < BOARD_SIZE:
            continue
        valid_ids.append(int(action_id))
        flats.append(row * BOARD_SIZE + col)
    if not flats:
        prior = 1.0 / len(legal_action_ids)
        return {int(action_id): prior for action_id in legal_action_ids}
    index = torch.as_tensor(flats, dtype=torch.long, device=logits.device)
    probs = torch.softmax(logits.index_select(0, index), dim=0).tolist()
    return {action_id: float(prob) for action_id, prob in zip(valid_ids, probs)}


def _legal_priors_from_flats(
    logits: torch.Tensor,
    legal_action_ids: Sequence[int],
    legal_flat_indices: Sequence[int],
) -> dict[int, float]:
    if not legal_action_ids:
        return {}
    if not legal_flat_indices:
        prior = 1.0 / len(legal_action_ids)
        return {int(action_id): prior for action_id in legal_action_ids}
    index = torch.as_tensor(legal_flat_indices, dtype=torch.long, device=logits.device)
    probs = torch.softmax(logits.index_select(0, index), dim=0).tolist()
    return {
        int(action_id): float(prob)
        for action_id, prob in zip(legal_action_ids, probs)
    }


def _to_inference_device(inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda" and inputs.ndim == 4:
        return inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
    return inputs.to(device, non_blocking=True)


def _topk_legal_priors(
    *,
    policy_batch: torch.Tensor,
    flat_index_buffer: torch.Tensor,
    offsets: tuple[int, ...],
    max_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    row_count = len(offsets) - 1
    if row_count <= 0:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.int64), [0]

    counts = torch.as_tensor(
        [max(0, offsets[index + 1] - offsets[index]) for index in range(row_count)],
        dtype=torch.long,
    )
    if int(counts.sum().item()) <= 0:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.int64), [0 for _ in range(row_count + 1)]

    device = policy_batch.device
    k = min(max(1, int(max_candidates)), int(policy_batch.shape[1]), int(counts.max().item()))
    row_ids = torch.repeat_interleave(torch.arange(row_count, dtype=torch.long), counts).to(device, non_blocking=True)
    flat_indices = flat_index_buffer.to(device, non_blocking=True)
    masked_logits = torch.full_like(policy_batch, float("-inf"))
    masked_logits[row_ids, flat_indices] = policy_batch[row_ids, flat_indices]
    values, selected_flats = torch.topk(masked_logits, k=k, dim=1, largest=True, sorted=True)

    ordinal_map = torch.full(
        (row_count, int(policy_batch.shape[1])),
        -1,
        dtype=torch.long,
        device=device,
    )
    ordinal_source = torch.cat(
        [torch.arange(int(count), dtype=torch.long, device=device) for count in counts if int(count) > 0]
    )
    ordinal_map[row_ids, flat_indices] = ordinal_source
    selected_ordinal_matrix = ordinal_map.gather(1, selected_flats)
    valid = selected_ordinal_matrix >= 0
    masked_values = values.masked_fill(~valid, float("-inf"))
    prob_matrix = torch.softmax(masked_values, dim=1).masked_fill(~valid, 0.0).float().cpu()
    ordinal_matrix = selected_ordinal_matrix.cpu()
    valid_cpu = valid.cpu()

    priors: list[torch.Tensor] = []
    ordinals: list[torch.Tensor] = []
    selected_offsets = [0]
    for row_index in range(row_count):
        row_valid = valid_cpu[row_index]
        keep = int(row_valid.sum().item())
        if keep == 0:
            selected_offsets.append(selected_offsets[-1])
            continue
        priors.append(prob_matrix[row_index, row_valid])
        ordinals.append(ordinal_matrix[row_index, row_valid].to(dtype=torch.int64))
        selected_offsets.append(selected_offsets[-1] + keep)
    if priors:
        return torch.cat(priors).contiguous(), torch.cat(ordinals).contiguous(), selected_offsets
    return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.int64), selected_offsets


def _topk_legal_priors_from_input_mask(
    *,
    policy_batch: torch.Tensor,
    inputs: torch.Tensor,
    max_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    row_count = int(policy_batch.shape[0])
    if row_count <= 0:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.int64), [0]

    legal_mask = inputs[:, PLANE_LEGAL].reshape(row_count, -1).to(
        device=policy_batch.device,
        dtype=torch.bool,
        non_blocking=True,
    )
    if not bool(legal_mask.any().item()):
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.int64), [0 for _ in range(row_count + 1)]

    k = min(max(1, int(max_candidates)), int(policy_batch.shape[1]))
    masked_logits = policy_batch.masked_fill(~legal_mask, float("-inf"))
    values, selected_flats = torch.topk(masked_logits, k=k, dim=1, largest=True, sorted=True)
    valid = torch.isfinite(values)
    masked_values = values.masked_fill(~valid, float("-inf"))
    prob_matrix = torch.softmax(masked_values, dim=1).masked_fill(~valid, 0.0).float().cpu()
    flat_matrix = selected_flats.cpu()
    valid_cpu = valid.cpu()

    priors: list[torch.Tensor] = []
    flats: list[torch.Tensor] = []
    selected_offsets = [0]
    for row_index in range(row_count):
        row_valid = valid_cpu[row_index]
        keep = int(row_valid.sum().item())
        if keep == 0:
            selected_offsets.append(selected_offsets[-1])
            continue
        priors.append(prob_matrix[row_index, row_valid])
        flats.append(flat_matrix[row_index, row_valid].to(dtype=torch.int64))
        selected_offsets.append(selected_offsets[-1] + keep)
    if priors:
        return torch.cat(priors).contiguous(), torch.cat(flats).contiguous(), selected_offsets
    return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.int64), selected_offsets
