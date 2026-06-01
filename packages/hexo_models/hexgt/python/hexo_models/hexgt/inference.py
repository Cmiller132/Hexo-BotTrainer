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

from .collate import collate_graphs
from .constants import DEFAULT_CANDIDATE_RADIUS
from .features import build_graph_tensors
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
    def evaluate_graph_facts(self, payload: dict) -> dict[str, Any]:
        """MCTS evaluator callback (the Rust `mcts_eval` byte contract).

        The Rust session builds one graph-facts dict per unique leaf state and
        hands them here as ``payload["graph_facts"]``. We featurize + collate
        (the EXACT training graph path), forward, and return byte buffers:

        - ``values_bytes``: f32[num_graphs], current-player value.
        - ``candidate_action_ids_bytes``: u32, packed coords, CSR-concatenated.
        - ``candidate_row_offsets``: int[num_graphs+1], CSR segment offsets.
        - ``priors_bytes``: f32, per-candidate softmax priors, CSR-aligned.

        Candidate ids come straight from the collated batch (CSR order), so the
        Rust side zips ids<->priors positionally and never re-derives order.
        """

        facts_list = list(payload["graph_facts"])
        if not facts_list:
            return {
                "values_bytes": b"",
                "candidate_action_ids_bytes": b"",
                "candidate_row_offsets": [0],
                "priors_bytes": b"",
            }
        graphs = [build_graph_tensors(f) for f in facts_list]
        batch = collate_graphs(graphs)
        out, dev_batch = self.forward_batch(batch)
        num_graphs = int(batch["num_graphs"])
        cand_graph = dev_batch["candidate_graph"]
        policy = out["policy"].float()
        log_probs = segment_log_softmax(policy, cand_graph, num_graphs)
        priors = log_probs.exp().cpu().numpy().astype(np.float32, copy=False)
        values = decode_binned_value(out["value"].float()).cpu().numpy()
        values = np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)
        cand_ids = batch["candidate_ids"].cpu().numpy().astype(np.uint32, copy=False)
        cand_graph_cpu = batch["candidate_graph"].cpu().numpy()

        counts = np.bincount(cand_graph_cpu, minlength=num_graphs)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return {
            "values_bytes": np.ascontiguousarray(values).tobytes(),
            "candidate_action_ids_bytes": np.ascontiguousarray(cand_ids).tobytes(),
            "candidate_row_offsets": [int(o) for o in offsets],
            "priors_bytes": np.ascontiguousarray(priors).tobytes(),
        }

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
