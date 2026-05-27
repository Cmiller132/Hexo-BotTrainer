"""Inference helpers for sparse candidate policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .config import HexformerConfig
from .input import (
    SparseDecisionInput,
    build_sparse_input,
    collate_sparse_inputs,
    sparse_input_from_payload,
)
from .losses import wdl_value_from_logits


@dataclass(frozen=True, slots=True)
class HexformerInferenceResult:
    legal_action_ids: tuple[int, ...]
    legal_priors: Mapping[int, float]
    value: float
    wdl: tuple[float, float, float]
    distance: float


class HexformerInference:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        config: HexformerConfig,
        device: torch.device | str | None = None,
        amp: bool | None = None,
    ) -> None:
        self.model = model
        self.config = config
        resolved = torch.device(device or config.device)
        self.device = resolved if resolved.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
        self.amp = config.training.amp if amp is None else bool(amp)
        self.model.to(self.device)
        self.model.eval()

    def infer_state(self, state: object) -> HexformerInferenceResult:
        return self.infer_states((state,))[0]

    def infer_states(self, states: Sequence[object]) -> tuple[HexformerInferenceResult, ...]:
        sparse = [
            build_sparse_input(
                state,
                architecture=self.config.architecture,
                candidates=self.config.candidates,
            )
            for state in states
        ]
        return self.infer_sparse(sparse)

    def infer_sparse(self, sparse_inputs: Sequence[SparseDecisionInput]) -> tuple[HexformerInferenceResult, ...]:
        if not sparse_inputs:
            return ()
        batch = collate_sparse_inputs(sparse_inputs)
        batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=self.amp and self.device.type == "cuda"):
                outputs = self.model(batch)
        probs = torch.softmax(outputs["policy_logits"].float(), dim=-1).detach().cpu()
        wdl_probs = torch.softmax(outputs["wdl_logits"].float(), dim=-1).detach().cpu()
        values = wdl_value_from_logits(outputs["wdl_logits"].float()).detach().cpu()
        distances = outputs["distance"].float().detach().cpu()
        results: list[HexformerInferenceResult] = []
        for index, sparse in enumerate(sparse_inputs):
            count = len(sparse.candidate_action_ids)
            priors = {
                int(action_id): float(prob)
                for action_id, prob in zip(sparse.candidate_action_ids, probs[index, :count])
            }
            total = sum(priors.values())
            if total <= 0.0 and priors:
                uniform = 1.0 / len(priors)
                priors = {action_id: uniform for action_id in priors}
            elif total > 0.0:
                priors = {action_id: value / total for action_id, value in priors.items()}
            results.append(
                HexformerInferenceResult(
                    legal_action_ids=tuple(sparse.candidate_action_ids),
                    legal_priors=priors,
                    value=float(values[index].item()),
                    wdl=tuple(float(item) for item in wdl_probs[index].tolist()),
                    distance=float(distances[index].item()),
                )
            )
        return tuple(results)

    @torch.no_grad()
    def evaluate_mcts_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Torch callback used by the Rust Hexformer AR MCTS bridge."""

        sparse = tuple(
            sparse_input_from_payload(item)
            for item in payload["sparse_inputs"]
        )
        results = self.infer_sparse(sparse)
        values = torch.tensor(
            [result.value for result in results],
            dtype=torch.float32,
        ).contiguous()
        candidate_rows = tuple(
            tuple(int(action_id) for action_id in result.legal_action_ids)
            for result in results
        )
        priors = torch.tensor(
            [
                float(result.legal_priors.get(int(action_id), 0.0))
                for result in results
                for action_id in result.legal_action_ids
            ],
            dtype=torch.float32,
        ).contiguous()
        return {
            "values_bytes": values.numpy().tobytes(),
            "candidate_action_ids": candidate_rows,
            "priors_bytes": priors.numpy().tobytes(),
        }
