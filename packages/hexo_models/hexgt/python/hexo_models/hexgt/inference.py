"""Torch (FP16) inference for hexgt — the search evaluator path (Phase 5).

A truly dynamic GNN does not export to TensorRT (§6.1), so inference is plain
torch with optional FP16 autocast. This module evaluates a batch of live engine
states into per-candidate priors + a scalar value, which is exactly what the
MCTS evaluator callback needs (the dense_cnn `{values_bytes, priors_bytes}`
contract, but priors are per-candidate in CSR order — `candidate_ids`).

`evaluate_states` is the Python-driven path (build graph from states → forward →
per-graph softmax). The zero-copy Rust-payload transport is layered on top once
the throughput gate (this module + `_profile_hexgt_forward.py`) says go.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .constants import DEFAULT_CANDIDATE_RADIUS
from .graph_build import batch_from_states
from .losses import decode_binned_value, segment_log_softmax


class HexgtInference:
    """Batched torch evaluator for live engine states."""

    def __init__(self, model: torch.nn.Module, *, device: str | torch.device = "cuda", fp16: bool = True) -> None:
        self.device = torch.device(device)
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.model = model.to(self.device).eval()

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return out

    @torch.no_grad()
    def forward_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        batch = self._to_device(batch)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
            out = self.model.forward_policy_value(batch)
        return out, batch

    @torch.no_grad()
    def evaluate_states(self, states: Sequence[Any], n: int = DEFAULT_CANDIDATE_RADIUS) -> list[dict[str, Any]]:
        """Return per-state {candidate_ids, priors, value}. priors are softmax of
        the policy logits over that state's candidate set (CSR order)."""

        batch = batch_from_states(states, n)
        out, dev_batch = self.forward_batch(batch)
        num_graphs = int(batch["num_graphs"])
        cand_graph = dev_batch["candidate_graph"]
        policy = out["policy"].float()
        log_probs = segment_log_softmax(policy, cand_graph, num_graphs)
        priors = log_probs.exp().cpu().numpy()
        values = decode_binned_value(out["value"].float()).cpu().numpy()
        cand_ids = batch["candidate_ids"].cpu().numpy()
        cand_graph_cpu = batch["candidate_graph"].cpu().numpy()

        results = []
        for gid in range(num_graphs):
            mask = cand_graph_cpu == gid
            results.append(
                {
                    "candidate_ids": cand_ids[mask],
                    "priors": priors[mask].astype(np.float32),
                    "value": float(np.clip(values[gid], -1.0, 1.0)),
                }
            )
        return results
