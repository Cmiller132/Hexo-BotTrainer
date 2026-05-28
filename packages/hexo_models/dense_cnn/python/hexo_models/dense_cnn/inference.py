"""Inference and Rust-MCTS evaluator adapter for dense CNN Model 1.

`DenseCNNInference` is the only Python/Torch evaluator used by production MCTS.
Direct state inference asks Rust to encode live `hexo_engine.HexoState` objects
into Model 1 tensors, runs the PyTorch network, and projects legal priors back
onto packed action ids.

Native MCTS uses `evaluate_model1_payload` as a strict byte callback. Rust sends
contiguous tensor bytes plus either explicit legal flat-index rows or a request
to derive top-k legal priors from the legal plane. Python returns contiguous
`values_bytes` and `priors_bytes` that Rust parses exactly. Shape, dtype, and
byte-count checks live here because this is the Python/Torch boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from . import rust_bridge
from .architecture import Model1Network, optimized_model1_for_inference
from .constants import BOARD_SIZE, INPUT_CHANNELS, PLANE_LEGAL
from .d6 import unpack_coord_pair
from .losses import decode_binned_value
from .samples import CompressedSample, Model1SampleData, expand_sample, stack_expanded


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Decoded model result for one root/sample.

    The dense policy head is a flat crop tensor, but callers usually care about
    engine action ids. `legal_priors` is the policy head projected onto legal
    packed coordinates and normalized over that legal set.
    """

    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    value: float
    legal_action_ids: tuple[int, ...]
    legal_priors: dict[int, float]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class DenseCNNInference:
    """Owns tensor construction, device movement, and policy/value decoding.

    The class accepts any `torch.nn.Module` with Model 1-compatible outputs. It
    handles CPU/GPU placement, optional AMP, optional CUDA inference clone
    optimization, and the exact Rust evaluator callback used by MCTS.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | str = "cpu",
        amp: bool = False,
        return_logits: bool = True,
        max_batch_size: int | None = None,
        optimize_for_inference: bool = True,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.model = (
            optimized_model1_for_inference(model)
            if optimize_for_inference and self.device.type == "cuda" and _contains_model1_network(model)
            else model
        )
        self.amp = bool(amp and self.device.type == "cuda")
        self.return_logits = bool(return_logits)
        self.max_batch_size = 1024 if max_batch_size is None else int(max_batch_size)
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.model.to(device=self.device, memory_format=torch.channels_last)
        else:
            self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            self._warm_up_cuda()

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
        """Encode live states in Rust, run Torch once, and decode legal priors."""

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
        resolved_batch = len(samples) if batch_size is None else int(batch_size)
        if resolved_batch <= 0:
            raise ValueError("batch_size must be > 0")
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
        resolved_batch = int(inputs.shape[0]) if batch_size is None else int(batch_size)
        if resolved_batch <= 0:
            raise ValueError("batch_size must be > 0")
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

    @torch.inference_mode()
    def _forward_inputs_device(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = _to_inference_device(inputs, self.device)
        return self._forward_device_inputs(inputs)

    @torch.inference_mode()
    def _forward_device_inputs(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            if hasattr(self.model, "forward_policy_value"):
                output = self.model.forward_policy_value(inputs)
            else:
                output = self.model(inputs)
        return {key: value.detach().float() for key, value in output.items()}

    @torch.inference_mode()
    def _warm_up_cuda(self) -> None:
        """Prime cuDNN algorithm selection and GPU clocks before timed search."""

        warmup_batch = min(1024, self.max_batch_size)
        dtype = torch.float16 if self.amp else torch.float32
        inputs = torch.zeros(
            (warmup_batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            device=self.device,
            dtype=dtype,
        ).to(memory_format=torch.channels_last)
        for _ in range(8):
            self._forward_device_inputs(inputs)
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def evaluate_model1_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Torch callback used by the Rust batched MCTS bridge.

        Rust owns search and sends a strict byte payload. Python owns only the
        neural network pass and legal-prior extraction. Returned bytes are parsed
        by `rust/src/mcts_eval.rs`, so the order and lengths here are part of the
        native evaluator ABI.
        """

        shape = _payload_shape(payload)
        input_dtype_name = str(payload.get("input_dtype", "float32"))
        if input_dtype_name == "float32":
            input_dtype = torch.float32
            item_bytes = 4
        elif input_dtype_name == "float16":
            input_dtype = torch.float16
            item_bytes = 2
        else:
            raise ValueError(f"unsupported dense_cnn MCTS input_dtype {input_dtype_name!r}")
        _require_byte_length("inputs", payload["inputs"], _shape_product(shape), item_bytes)
        inputs = torch.frombuffer(payload["inputs"], dtype=input_dtype).reshape(shape)
        if inputs.dtype != torch.float32 and (self.device.type != "cuda" or not self.amp):
            inputs = inputs.float()
        max_batch = self.max_batch_size
        if inputs.shape[0] > max_batch:
            # MCTS can ask for a large leaf batch. Chunking happens here because
            # only the Python inference wrapper knows the safe Torch batch size
            # for the active model/device.
            output_chunks = []
            for start in range(0, inputs.shape[0], max_batch):
                output_chunks.append({
                    key: value.cpu()
                    for key, value in self._forward_inputs_device(inputs[start : start + max_batch]).items()
                })
            policy_batch = torch.cat([chunk["policy"] for chunk in output_chunks], dim=0)
            value_batch = torch.cat([chunk["value"] for chunk in output_chunks], dim=0)
            mask_inputs = inputs
        else:
            device_inputs = _to_inference_device(inputs, self.device)
            outputs = self._forward_device_inputs(device_inputs)
            policy_batch = outputs["policy"]
            value_batch = outputs["value"]
            mask_inputs = device_inputs
        values_tensor = decode_binned_value(value_batch).cpu().contiguous()
        selected_flats: torch.Tensor | None = None
        selected_offsets: Sequence[int] | None = None
        if payload.get("legal_mask_from_inputs"):
            # Candidate-limited mode sends the legal plane inside the input
            # tensor. Python chooses the top-k legal crop cells and returns both
            # priors and selected flat indices so Rust can prove they are legal.
            max_candidates = _positive_payload_int(payload, "max_prior_candidates")
            priors_tensor, selected_flats, selected_offsets = _topk_legal_priors_from_input_mask(
                policy_batch=policy_batch,
                inputs=mask_inputs,
                max_candidates=max_candidates,
            )
        else:
            # Full-prior mode is used when Rust has already sent the exact legal
            # crop flats. Priors are softmax-normalized per row over that list.
            if "legal_flat_indices_bytes" not in payload or "legal_row_offsets" not in payload:
                raise ValueError(
                    "dense_cnn MCTS evaluator payload requires legal_flat_indices_bytes and legal_row_offsets"
                )
            offsets = _legal_row_offsets(payload["legal_row_offsets"], rows=shape[0])
            selected_count = offsets[-1]
            _require_byte_length("legal_flat_indices_bytes", payload["legal_flat_indices_bytes"], selected_count, 8)
            if selected_count == 0:
                priors_tensor = torch.empty(0, dtype=torch.float32)
            else:
                flat_index_buffer = torch.frombuffer(payload["legal_flat_indices_bytes"], dtype=torch.int64)
                counts = torch.as_tensor(
                    [offsets[index + 1] - offsets[index] for index in range(len(offsets) - 1)],
                    dtype=torch.long,
                )
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
                positive_rows = counts.to(sum_per_row.device) > 0
                if bool((sum_per_row[positive_rows] <= 0).any().item()):
                    raise ValueError("dense_cnn MCTS evaluator payload has a legal row with zero prior mass")
                priors_tensor = (exp / sum_per_row[row_ids_device]).cpu().contiguous()
        result: dict[str, Any] = {
            "values_bytes": values_tensor.numpy().tobytes(),
            "priors_bytes": priors_tensor.numpy().tobytes(),
        }
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
            raise ValueError(f"legal action {int(action_id)} is outside the dense_cnn inference crop")
        valid_ids.append(int(action_id))
        flats.append(row * BOARD_SIZE + col)
    if not flats:
        raise ValueError("dense_cnn inference received legal actions but no in-crop policy indices")
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
        raise ValueError("dense_cnn inference received legal actions but no legal flat indices")
    index = torch.as_tensor(legal_flat_indices, dtype=torch.long, device=logits.device)
    probs = torch.softmax(logits.index_select(0, index), dim=0).tolist()
    return {
        int(action_id): float(prob)
        for action_id, prob in zip(legal_action_ids, probs)
    }


def _to_inference_device(inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    non_blocking = bool(inputs.is_cuda or inputs.is_pinned())
    if device.type == "cuda" and inputs.ndim == 4:
        return inputs.to(device, non_blocking=non_blocking, memory_format=torch.channels_last)
    return inputs.to(device, non_blocking=non_blocking)


def _contains_model1_network(model: torch.nn.Module) -> bool:
    return any(isinstance(module, Model1Network) for module in model.modules())


def _payload_shape(payload: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Read the evaluator input tensor shape before constructing a tensor view."""

    shape = tuple(int(item) for item in payload["shape"])
    if len(shape) != 4:
        raise ValueError(f"dense_cnn MCTS input shape must have 4 dimensions, got {shape!r}")
    if any(item <= 0 for item in shape):
        raise ValueError(f"dense_cnn MCTS input shape dimensions must be positive, got {shape!r}")
    return shape


def _shape_product(shape: Sequence[int]) -> int:
    result = 1
    for item in shape:
        result *= int(item)
    return result


def _require_byte_length(name: str, value: object, expected_items: int, bytes_per_item: int) -> None:
    """Reject byte buffers that cannot represent the declared tensor/vector."""

    expected = int(expected_items) * int(bytes_per_item)
    actual = len(value)  # type: ignore[arg-type]
    if actual != expected:
        raise ValueError(f"{name} has {actual} bytes, expected {expected}")


def _positive_payload_int(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"dense_cnn MCTS evaluator payload missing {key}")
    value = int(payload[key])
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _legal_row_offsets(value: object, *, rows: int) -> tuple[int, ...]:
    offsets = tuple(int(item) for item in value)  # type: ignore[arg-type]
    if len(offsets) != rows + 1:
        raise ValueError(f"legal_row_offsets has {len(offsets)} entries, expected {rows + 1}")
    if offsets[0] != 0:
        raise ValueError("legal_row_offsets must start at 0")
    for left, right in zip(offsets, offsets[1:]):
        if right < left:
            raise ValueError("legal_row_offsets must be monotonically nondecreasing")
    return offsets


def _topk_legal_priors_from_input_mask(
    *,
    policy_batch: torch.Tensor,
    inputs: torch.Tensor,
    max_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return top-k legal priors and flat crop indices for candidate-limited MCTS."""

    row_count = int(policy_batch.shape[0])
    if row_count <= 0:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.int64), [0]
    policy_width = int(policy_batch.shape[1])
    if max_candidates <= 0:
        raise ValueError("max_prior_candidates must be > 0")
    if max_candidates > policy_width:
        raise ValueError(
            f"max_prior_candidates {max_candidates} exceeds dense_cnn policy width {policy_width}"
        )

    legal_mask = inputs[:, PLANE_LEGAL].reshape(row_count, -1).to(
        device=policy_batch.device,
        dtype=torch.bool,
        non_blocking=True,
    )
    if not bool(legal_mask.any().item()):
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.int64), [0 for _ in range(row_count + 1)]

    # `topk` runs over masked logits. Invalid cells become `-inf`, then the
    # finite mask controls both softmax normalization and selected-row offsets.
    k = int(max_candidates)
    masked_logits = policy_batch.masked_fill(~legal_mask, float("-inf"))
    values, selected_flats = torch.topk(masked_logits, k=k, dim=1, largest=True, sorted=True)
    valid = torch.isfinite(values)
    masked_values = values.masked_fill(~valid, float("-inf"))
    prob_matrix = torch.softmax(masked_values, dim=1).masked_fill(~valid, 0.0).float().cpu()
    flat_matrix = selected_flats.cpu()
    valid_cpu = valid.cpu()
    if bool(valid_cpu.all().item()):
        selected_offsets = list(range(0, (row_count + 1) * k, k))
        return (
            prob_matrix.reshape(-1).contiguous(),
            flat_matrix.reshape(-1).to(dtype=torch.int64).contiguous(),
            selected_offsets,
        )

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
