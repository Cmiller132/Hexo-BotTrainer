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

import os
import sys
from typing import Any, Sequence

import numpy as np
import torch

from .architecture import precompute_attention_layout
from .constants import DEFAULT_CANDIDATE_RADIUS
from .graph_build import batch_from_states
from .losses import decode_binned_value, segment_log_softmax


# Default per-chunk padded-candidate budget (graphs_in_chunk * chunk_max_cand).
# The transformer's peak VRAM is ~linear in this product (~12 B/slot for the 168/
# 336 trunk), so ~200k slots ≈ ~2.5 GB/chunk. A LARGER budget means fewer, fatter
# GPU forwards (higher util/throughput) at the cost of more VRAM + padding waste,
# so size it to the available headroom. Tunable via the constructor or the
# HEXGT_FORWARD_PAD_BUDGET env var (the run launcher sets it to exploit headroom).
DEFAULT_FORWARD_PAD_BUDGET = 200_000


def _plan_chunks(counts: torch.Tensor, budget: int) -> list[torch.Tensor]:
    """Partition graph ids into chunks (each a LongTensor of ids), SORTED by
    candidate count so a chunk pads only to its LOCAL max, and bounded so
    ``len(chunk) * chunk_max_cand <= budget``. This is the compression: a global
    batch is ~22% packing-efficient (a few endgame graphs force huge padding);
    sorted chunks are ~90%+, and the budget caps peak VRAM regardless of how many
    leaves MCTS submits."""

    order = torch.argsort(counts)
    cs = counts[order].tolist()  # ascending -> cs[i] is the running chunk max
    bounds = []
    start = 0
    for i in range(len(cs)):
        ln = i - start + 1
        if ln > 1 and ln * cs[i] > budget:
            bounds.append((start, i))
            start = i
    bounds.append((start, len(cs)))
    # Candidate-count sorting decides chunk MEMBERSHIP (group similar sizes), but
    # each chunk's ids are returned ASCENDING: the model's per-graph layout
    # (`_padded_index`) requires nodes in contiguous ascending-graph-id blocks,
    # which the original collation guarantees and a 0-based remap of an ascending
    # id subset preserves.
    return [order[a:b].sort().values for a, b in bounds]


def _subbatch(batch: dict, gids: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Extract a self-contained sub-batch for the graphs in `gids` (re-indexed to
    0-based). Returns (sub_batch, candidate_mask) where candidate_mask selects the
    sub-batch's candidates from the full candidate array (in original order), so
    per-candidate outputs scatter straight back. Graphs are independent, so the
    sub-batch forward is bit-identical to the same graphs in the full batch."""

    device = batch["node_feat"].device
    ng = int(batch["num_graphs"])
    gset = torch.zeros(ng, dtype=torch.bool, device=device)
    gset[gids] = True
    remap = torch.full((ng,), -1, dtype=torch.long, device=device)
    remap[gids] = torch.arange(gids.numel(), device=device)

    ngph = batch["node_graph"]
    nsel = gset[ngph]
    node_map = torch.full((ngph.numel(),), -1, dtype=torch.long, device=device)
    node_map[nsel] = torch.arange(int(nsel.sum()), device=device)

    sub: dict = {
        "node_feat": batch["node_feat"][nsel],
        "node_type": batch["node_type"][nsel],
        "node_graph": remap[ngph[nsel]],
        "num_graphs": int(gids.numel()),
    }
    ei = batch["edge_index"]
    if ei.numel():
        esel = nsel[ei[0]]  # an edge sits within one graph -> src selected <=> dst selected
        sub["edge_index"] = node_map[ei[:, esel]]
        sub["edge_type"] = batch["edge_type"][esel]
        sub["edge_attr"] = batch["edge_attr"][esel]
    else:
        sub["edge_index"] = ei
        sub["edge_type"] = batch.get("edge_type")
        sub["edge_attr"] = batch.get("edge_attr")

    cg = batch["candidate_graph"]
    csel = gset[cg]
    sub["candidate_index"] = node_map[batch["candidate_index"][csel]]
    sub["candidate_graph"] = remap[cg[csel]]
    return sub, csel


class HexgtInference:
    """Batched torch evaluator for live engine states."""

    #: Host tensors with at least this many elements are staged through pinned
    #: (page-locked) memory for a faster async HtoD copy; smaller ones are not
    #: worth the pinning overhead. Tunable via the constructor.
    DEFAULT_PIN_MIN_NUMEL = 4096

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "cuda",
        fp16: bool = True,
        forward_pad_budget: int | None = None,
        pin_host_transfer: bool = True,
        pin_min_numel: int = DEFAULT_PIN_MIN_NUMEL,
    ) -> None:
        self.device = torch.device(device)
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.model = model.to(self.device).eval()
        # Resolve the chunk pad-budget: explicit arg > HEXGT_FORWARD_PAD_BUDGET env
        # > module default. Lets the run launcher dial it to the VRAM headroom
        # without threading a new arg through every construction site.
        if forward_pad_budget is None:
            env_budget = os.environ.get("HEXGT_FORWARD_PAD_BUDGET")
            forward_pad_budget = int(env_budget) if env_budget else DEFAULT_FORWARD_PAD_BUDGET
        self.forward_pad_budget = int(forward_pad_budget)
        # Env override lets the knob be A/B-profiled without code changes.
        env_pin = os.environ.get("HEXGT_PIN_HOST")
        if env_pin is not None:
            pin_host_transfer = env_pin not in ("0", "false", "False")
        self.pin_host_transfer = bool(pin_host_transfer) and self.device.type == "cuda"
        self.pin_min_numel = int(pin_min_numel)
        # --- Non-finite-logit sanitization audit counters -------------------------
        # The forward (fp16 autocast) can emit inf/NaN logits on rare pathological
        # positions; `_count_and_sanitize` replaces them with neutral finite values so
        # the run survives, but a sanitized output is a near-uniform prior + neutral
        # value that must NOT silently become a training target. These counters make
        # the guard auditable: callers (self-play) snapshot them per search round to
        # FLAG + exclude affected positions, and surface the totals in diagnostics.
        # A non-zero/rising count is a red flag (model instability, e.g. fp16 training
        # underflow before the GradScaler fix, or a genuinely broken state), not a
        # benign event. Counts are cumulative over this evaluator's lifetime; self-play
        # builds a fresh evaluator per epoch, so the per-epoch total is read at end.
        self.sanitized_logit_total = 0      # count of individual non-finite logit values
        self.sanitized_forward_calls = 0    # forward batches that contained >=1 non-finite
        self.sanitized_graph_total = 0      # graphs (positions) with >=1 non-finite logit

    def _count_and_sanitize(
        self,
        raw_policy: torch.Tensor,
        raw_value: torch.Tensor,
        candidate_graph: torch.Tensor | None,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Count + replace non-finite logits, returning (policy, value, n_bad).

        fp16 autocast can overflow to inf/NaN on rare extreme positions (e.g. very
        deep endgames with huge candidate sets). A non-finite logit poisons the
        per-graph softmax/value and the Rust evaluator rejects it (crashing the run).
        We replace non-finite LOGITS with neutral finite values so the softmax stays
        well-defined (the affected position gets a near-uniform prior + neutral value
        instead of taking down a multi-hour run), but -- unlike a silent guard -- we
        COUNT the event, log it with the running totals, and let self-play exclude the
        affected positions from training. Returns the (sanitized) tensors and the
        number of non-finite logit values found (0 == untouched)."""

        bad_policy = ~torch.isfinite(raw_policy)
        bad_value = ~torch.isfinite(raw_value)
        n_bad = int(bad_policy.sum().item()) + int(bad_value.sum().item())
        if not n_bad:
            return raw_policy, raw_value, 0
        affected = torch.zeros(num_graphs, dtype=torch.bool, device=raw_value.device)
        if bad_value.ndim >= 2:
            affected |= bad_value.any(dim=-1)
        else:
            affected |= bad_value.reshape(-1)[:num_graphs]
        if candidate_graph is not None and bool(bad_policy.any().item()):
            affected[candidate_graph[bad_policy].to(torch.long)] = True
        n_graphs_bad = int(affected.sum().item())
        self.sanitized_logit_total += n_bad
        self.sanitized_forward_calls += 1
        self.sanitized_graph_total += n_graphs_bad
        print(
            f"[hexgt.inference] sanitized {n_bad} non-finite logit(s) (fp16 overflow) "
            f"over {n_graphs_bad}/{num_graphs} graph(s) "
            f"[run totals: {self.sanitized_logit_total} logits in "
            f"{self.sanitized_forward_calls} batch(es), {self.sanitized_graph_total} positions]",
            file=sys.stderr,
            flush=True,
        )
        raw_policy = torch.nan_to_num(raw_policy, nan=0.0, posinf=30.0, neginf=-30.0)
        raw_value = torch.nan_to_num(raw_value, nan=0.0, posinf=30.0, neginf=-30.0)
        return raw_policy, raw_value, n_bad

    def _host_to_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move one host tensor to the device. Large buffers go through pinned
        (page-locked) memory + a non-blocking DMA, which is ~2x the pageable-copy
        bandwidth and lets kernel launches overlap the transfer (the default
        pageable HtoD was ~10% of GPU time at 512 sims). `frombuffer` views are
        non-writable, so `pin_memory` makes a writable pinned copy; torch's
        caching host allocator keeps that copy alive until the async DMA
        completes, so this is safe with `non_blocking=True`."""
        if not self.pin_host_transfer or t.device.type != "cpu":
            return t.to(self.device)
        if t.numel() < self.pin_min_numel:
            return t.to(self.device)
        return t.pin_memory().to(self.device, non_blocking=True)

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = self._host_to_device(v) if isinstance(v, torch.Tensor) else v
        return out

    @torch.no_grad()
    def forward_batch(self, batch: dict) -> tuple[dict[str, torch.Tensor], dict]:
        """Forward returning ``(out, dev_batch)``: ``out`` holds RAW per-candidate
        policy logits + per-graph value in the batch's original candidate/graph
        order, and ``dev_batch`` is the on-device batch (with attention layout) the
        callers read ``candidate_graph`` from. When the padded size would exceed the
        budget, the forward is split into sorted, bounded chunks (compression: ~4x
        less wasted padding + capped peak VRAM); each graph is independent so the
        reassembled output is identical to a single forward."""

        batch = self._to_device(batch)
        num_graphs = int(batch["num_graphs"])
        # Build the attention layout EAGERLY here (outside the compiled
        # forward_policy_value) and pass it in, so the compiled forward has no
        # data-dependent nonzero/bincount/.item -> no graph-break / recompile.
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
            if num_graphs <= 1:
                batch = {**batch, **precompute_attention_layout(batch["node_graph"], batch["node_type"], num_graphs)}
                return self.model.forward_policy_value(batch), batch
            cg = batch["candidate_graph"]
            counts = torch.bincount(cg, minlength=num_graphs)
            padded = int(counts.max()) * num_graphs
            if padded <= self.forward_pad_budget:
                batch = {**batch, **precompute_attention_layout(batch["node_graph"], batch["node_type"], num_graphs)}
                return self.model.forward_policy_value(batch), batch
            # Chunked path: sorted, budget-bounded sub-batches.
            chunks = _plan_chunks(counts, self.forward_pad_budget)
            policy = torch.empty(cg.numel(), device=self.device, dtype=torch.float32)
            value: torch.Tensor | None = None
            for gids in chunks:
                sub, csel = _subbatch(batch, gids)
                sub = {**sub, **precompute_attention_layout(sub["node_graph"], sub["node_type"], sub["num_graphs"])}
                out = self.model.forward_policy_value(sub)
                policy[csel] = out["policy"].float()
                if value is None:
                    value = torch.empty((num_graphs, out["value"].shape[-1]), device=self.device, dtype=torch.float32)
                value[gids] = out["value"].float()
        return {"policy": policy, "value": value}, batch

    @torch.no_grad()
    def evaluate_featurized_batch(self, payload: dict) -> dict[str, Any]:
        """MCTS evaluator callback consuming a RUST-featurized + collated batch
        (the parallel `features.rs` path — no per-graph Python featurization).

        `payload` carries the collated batch as ZERO-COPY buffer-protocol objects
        (see `features.rs::collated_to_py_dict`): `node_feat`/`edge_attr` (f32),
        `node_type`/`node_graph`/`edge_index` (i64, edge_index packed 2*E)/
        `edge_type`/`candidate_index`/`candidate_graph` (i64), plus scalar sizes.
        Candidate ids stay Rust-side, so we return ONLY `{values_bytes,
        priors_bytes}` with priors in the SAME packed candidate order Rust built
        (zipped positionally there via `candidate_counts`). Featurization (the
        former Python bottleneck) now runs in Rust rayon; this path is just
        frombuffer + GNN forward + per-graph softmax.
        """

        num_graphs = int(payload["num_graphs"])
        tc = int(payload["total_candidates"])
        if num_graphs == 0 or tc == 0:
            return {"values_bytes": b"", "priors_bytes": b""}
        tn = int(payload["total_nodes"])
        te = int(payload["total_edges"])
        fd = int(payload["feat_dim"])
        ad = int(payload["attr_dim"])
        node_feat = torch.frombuffer(payload["node_feat"], dtype=torch.float32).reshape(tn, fd)
        node_type = torch.frombuffer(payload["node_type"], dtype=torch.int64)
        node_graph = torch.frombuffer(payload["node_graph"], dtype=torch.int64)
        if te:
            edge_index = torch.frombuffer(payload["edge_index"], dtype=torch.int64).reshape(2, te)
            edge_type = torch.frombuffer(payload["edge_type"], dtype=torch.int64)
            edge_attr = torch.frombuffer(payload["edge_attr"], dtype=torch.float32).reshape(te, ad)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.int64)
            edge_type = torch.zeros((0,), dtype=torch.int64)
            edge_attr = torch.zeros((0, ad), dtype=torch.float32)
        candidate_index = torch.frombuffer(payload["candidate_index"], dtype=torch.int64)
        candidate_graph = torch.frombuffer(payload["candidate_graph"], dtype=torch.int64)
        batch = {
            "node_feat": node_feat, "node_type": node_type, "node_graph": node_graph,
            "edge_index": edge_index, "edge_type": edge_type, "edge_attr": edge_attr,
            "candidate_index": candidate_index, "candidate_graph": candidate_graph,
            "num_graphs": num_graphs,
        }
        out, dev_batch = self.forward_batch(batch)
        cg = dev_batch["candidate_graph"]
        # Count + sanitize non-finite trunk outputs (see `_count_and_sanitize`): the
        # affected positions get a near-uniform prior + neutral value to keep the run
        # alive, but the event is recorded so self-play can FLAG/exclude them from
        # training targets instead of silently learning the neutral output.
        raw_policy, raw_value, _n_bad = self._count_and_sanitize(
            out["policy"].float(), out["value"].float(), cg, num_graphs
        )
        log_probs = segment_log_softmax(raw_policy, cg, num_graphs)
        priors = log_probs.exp().cpu().numpy().astype(np.float32, copy=False)
        values = decode_binned_value(raw_value).cpu().numpy()
        values = np.nan_to_num(np.clip(values, -1.0, 1.0), nan=0.0).astype(np.float32, copy=False)
        return {
            "values_bytes": np.ascontiguousarray(values).tobytes(),
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
        # Same fp16-overflow guard as evaluate_featurized_batch, now counted + logged
        # (this Python-driven path was previously silent).
        policy, raw_value, _n_bad = self._count_and_sanitize(
            out["policy"].float(), out["value"].float(), cand_graph, num_graphs
        )
        log_probs = segment_log_softmax(policy, cand_graph, num_graphs)
        priors = log_probs.exp().cpu().numpy()
        values = decode_binned_value(raw_value).cpu().numpy()
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
