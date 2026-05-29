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
from .constants import BOARD_SIZE, INPUT_CHANNELS
from .losses import decode_binned_value


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
        # A7 (production form of PB): keep cuDNN autotune ON, but pad forward
        # batches to a small set of power-of-two buckets so the evaluator sees a
        # handful of shapes instead of ~900. cuDNN re-autotunes (~925 ms) on every
        # never-seen batch shape; without bucketing a cold/relaunch epoch pays
        # ~830 s of autotune thrash (and once hung >10 min). Bucketing bounds the
        # distinct shapes to ~log2(max_batch)+1, so autotune converges in seconds
        # and stays converged, while padded rows are discarded (per-sample conv/
        # eval-BN/FC ops never let padding leak into the real rows).
        self.pad_to_buckets = self.device.type == "cuda"
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
        rows = int(inputs.shape[0])
        padded = self._pad_to_bucket(inputs, rows)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            if hasattr(self.model, "forward_policy_value"):
                output = self.model.forward_policy_value(padded)
            else:
                output = self.model(padded)
        # Slice padding off every head; rows [:rows] are the real samples.
        return {key: value.detach().float()[:rows] for key, value in output.items()}

    def _pad_to_bucket(self, inputs: torch.Tensor, rows: int) -> torch.Tensor:
        """Pad the batch dim up to a power-of-two bucket (A7 shape stabilization)."""

        if not self.pad_to_buckets:
            return inputs
        target = _bucket_batch_size(rows, self.max_batch_size)
        if target == rows:
            return inputs
        padded = torch.zeros((target, *inputs.shape[1:]), dtype=inputs.dtype, device=inputs.device)
        if inputs.ndim == 4:
            padded = padded.to(memory_format=torch.channels_last)
        padded[:rows].copy_(inputs)
        return padded

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

        Rust owns search and sends a strict byte payload: the f32 input planes
        plus the per-row legal crop flats Rust derived from the engine. Python
        runs the network and returns row-major `values_bytes` and softmax
        `priors_bytes` over each row's legal flats. The order and lengths here are
        part of the native evaluator ABI parsed by `rust/src/mcts_eval.rs`.
        """

        shape = _payload_shape(payload)
        _require_byte_length("inputs", payload["inputs"], _shape_product(shape), 4)
        inputs = torch.frombuffer(payload["inputs"], dtype=torch.float32).reshape(shape)
        max_batch = self.max_batch_size
        if inputs.shape[0] > max_batch:
            # MCTS can ask for a large leaf batch. Chunking happens here because
            # only the Python inference wrapper knows the safe Torch batch size.
            output_chunks = [
                {key: value.cpu() for key, value in self._forward_inputs_device(inputs[start : start + max_batch]).items()}
                for start in range(0, inputs.shape[0], max_batch)
            ]
            policy_batch = torch.cat([chunk["policy"] for chunk in output_chunks], dim=0)
            value_batch = torch.cat([chunk["value"] for chunk in output_chunks], dim=0)
        else:
            outputs = self._forward_inputs_device(_to_inference_device(inputs, self.device))
            policy_batch = outputs["policy"]
            value_batch = outputs["value"]
        values_tensor = decode_binned_value(value_batch).cpu().contiguous()

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
            counts = torch.as_tensor(
                [offsets[index + 1] - offsets[index] for index in range(len(offsets) - 1)],
                dtype=torch.long,
            )
            row_ids = torch.repeat_interleave(torch.arange(len(counts), dtype=torch.long), counts)
            flat_indices = torch.frombuffer(payload["legal_flat_indices_bytes"], dtype=torch.int64).to(
                policy_batch.device, non_blocking=True
            )
            row_ids_device = row_ids.to(policy_batch.device, non_blocking=True)
            selected = policy_batch[row_ids_device, flat_indices]
            max_per_row = torch.full((len(counts),), float("-inf"), dtype=selected.dtype, device=selected.device)
            max_per_row.scatter_reduce_(0, row_ids_device, selected, reduce="amax", include_self=True)
            exp = torch.exp(selected - max_per_row[row_ids_device])
            sum_per_row = torch.zeros((len(counts),), dtype=selected.dtype, device=selected.device)
            sum_per_row.scatter_add_(0, row_ids_device, exp)
            if bool((sum_per_row[counts.to(sum_per_row.device) > 0] <= 0).any().item()):
                raise ValueError("dense_cnn MCTS evaluator payload has a legal row with zero prior mass")
            priors_tensor = (exp / sum_per_row[row_ids_device]).cpu().contiguous()
        return {
            "values_bytes": values_tensor.numpy().tobytes(),
            "priors_bytes": priors_tensor.numpy().tobytes(),
        }


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


def _bucket_batch_size(rows: int, cap: int) -> int:
    """Smallest power-of-two >= ``rows``, clamped to ``cap`` (A7 batch buckets).

    Batches at or above ``cap`` are returned unchanged: the evaluator already
    chunks oversized leaf batches to ``max_batch_size`` before the forward, so
    those land on the ``cap`` bucket. Padding is therefore at most ~2x the real
    rows, while the number of distinct shapes cuDNN must autotune is bounded to
    ``log2(cap) + 1``.
    """

    if rows <= 0 or rows >= cap:
        return rows
    size = 1
    while size < rows:
        size <<= 1
    return min(size, cap)


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


