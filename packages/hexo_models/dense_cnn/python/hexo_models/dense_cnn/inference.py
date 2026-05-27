"""Inference adapter for the dense CNN model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from . import rust_bridge
from .constants import BOARD_SIZE
from .d6 import Axial, D6Symmetry, unpack_coord_pair
from .input import build_input_planes
from .losses import decode_binned_value
from .samples import CompressedSample, Model1SampleData, expand_sample, sample_from_state, stack_expanded


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
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.amp = bool(amp and self.device.type == "cuda")
        self.return_logits = bool(return_logits)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer_state(self, state: object) -> InferenceResult:
        return self.infer_states([state])[0]

    @torch.no_grad()
    def infer_states(self, states: Sequence[object]) -> list[InferenceResult]:
        if not states:
            return []
        if rust_bridge.is_available():
            return self._infer_states_fast(states)
        return self._infer_states_python(states)

    @torch.no_grad()
    def _infer_states_fast(self, states: Sequence[object]) -> list[InferenceResult]:
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
    def _infer_states_python(self, states: Sequence[object]) -> list[InferenceResult]:
        samples = tuple(
            sample_from_state(state, game_id=f"inference-{index}", turn_index=0)
            for index, state in enumerate(states)
        )
        inputs = torch.stack([_input_from_sample(sample) for sample in samples], dim=0)
        outputs = self._forward_inputs_device(inputs)
        policy_batch = outputs["policy"]
        value_batch = outputs["value"]
        results: list[InferenceResult] = []
        values = decode_binned_value(value_batch)
        for index, sample in enumerate(samples):
            policy_logits = policy_batch[index]
            value_logits = value_batch[index]
            legal_action_ids = sample.legal_action_ids
            results.append(
                InferenceResult(
                    policy_logits=policy_logits.cpu() if self.return_logits else torch.empty(0),
                    value_logits=value_logits.cpu() if self.return_logits else torch.empty(0),
                    value=float(values[index].item()),
                    legal_action_ids=legal_action_ids,
                    legal_priors=_legal_priors(policy_logits, legal_action_ids, sample.center),
                    diagnostics={"device": str(self.device), "amp": self.amp, "batched": len(states) > 1, "encoder": "python"},
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
        inputs = inputs.to(self.device, non_blocking=True)
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
        max_batch = 2048
        if inputs.shape[0] > max_batch:
            output_chunks = []
            for start in range(0, inputs.shape[0], max_batch):
                output_chunks.append(self._forward_inputs_device(inputs[start : start + max_batch]))
            policy_batch = torch.cat([chunk["policy"] for chunk in output_chunks], dim=0)
            value_batch = torch.cat([chunk["value"] for chunk in output_chunks], dim=0)
        else:
            outputs = self._forward_inputs_device(inputs)
            policy_batch = outputs["policy"]
            value_batch = outputs["value"]
        values_tensor = decode_binned_value(value_batch).cpu().contiguous()
        priors: list[torch.Tensor] = []
        if "legal_flat_indices_bytes" in payload:
            flat_index_buffer = torch.frombuffer(payload["legal_flat_indices_bytes"], dtype=torch.int64)
            offsets = tuple(int(item) for item in payload["legal_row_offsets"])
            counts = torch.as_tensor(
                [max(0, offsets[index + 1] - offsets[index]) for index in range(len(offsets) - 1)],
                dtype=torch.long,
            )
            if int(counts.sum().item()) > 0:
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
            else:
                priors_tensor = torch.empty(0, dtype=torch.float32)
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
        return {
            "values_bytes": values_tensor.numpy().tobytes(),
            "priors_bytes": priors_tensor.numpy().tobytes(),
        }


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


def _input_from_sample(sample: Model1SampleData) -> torch.Tensor:
    center = Axial(*sample.center)
    return build_input_planes(
        current_player=sample.current_player,
        phase=sample.phase,
        center=center,
        stones=sample.stones,
        legal_action_ids=sample.legal_action_ids,
        placement_history=sample.placement_history,
        first_stone=sample.first_stone,
        own_hot=sample.own_hot,
        opponent_hot=sample.opponent_hot,
        opponent_last_turn=sample.opponent_last_turn,
        symmetry=D6Symmetry(0),
    )
