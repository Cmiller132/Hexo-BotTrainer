"""HexfieldEvaluator: serve-side consumer of the wire ABI.

Consumes the Rust payload (CSR over support nodes, rows pre-sorted by support
size descending), packs rows into quantized static shapes under the inference
pair ceiling, runs `forward_policy_value`, and returns the reply: `values_bytes`
(f32 x B, clamped [-1, 1]), `priors_bytes` (f32 x sum L_g, positional over each
row's legal prefix, fp32 softmax), and `moves_left_bytes` (f32 x B) when
requested.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch._dynamo  # noqa: F401  (mark_dynamic / config used in the serve path)

from .constants import NUM_FEATURES, NUM_TOKENS
from .losses import decode_binned_value, decode_moves_left
from .model import HexfieldNet

NBR_SENTINEL = 0xFFFF
# Upper bound on B * (S_pad + NUM_TOKENS)^2 per group, bounding the fp16
# (B, 4, S, S) bias transient. Inference ceiling, separate from any training
# pair budget.
PAIR_CEILING = 3.8e7
# Padding quantum in nodes: row sizes are rounded up to a multiple of this.
QUANT_NODES = 64
# A group stops extending when the next row is more than this fraction (or
# QUANT_NODES nodes) smaller than the group's padded anchor, bounding per-row
# padding waste.
WASTE_FRACTION = 0.18


def _ceil_quant(n: int) -> int:
    return max(QUANT_NODES, -(-int(n) // QUANT_NODES) * QUANT_NODES)


def plan_groups(sizes) -> list[tuple[int, int, int]]:
    """Group rows sorted DESCENDING by size. Returns (start, end, pad_to)
    tuples. pad_to is QUANT_NODES-quantized from the first (largest) row of the
    group, so pad_to >= every row in the group. A group stops extending when
    (a) adding the next row would exceed PAIR_CEILING, or (b) the next row is
    smaller than pad_to by more than the WASTE_FRACTION bound."""
    n = len(sizes)
    groups: list[tuple[int, int, int]] = []
    start = 0
    while start < n:
        pad_to = _ceil_quant(int(sizes[start]))
        floor = pad_to - max(QUANT_NODES, int(WASTE_FRACTION * pad_to))
        end = start + 1
        while end < n:
            if (end - start + 1) * (pad_to + NUM_TOKENS) ** 2 > PAIR_CEILING:
                break
            if int(sizes[end]) < floor:  # exceeds padding-waste bound -> split
                break
            end += 1
        groups.append((start, end, pad_to))
        start = end
    return groups


class HexfieldEvaluator:
    def __init__(self, model: HexfieldNet, device: torch.device | str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        # Compile the serve forward with dynamic=True: both varying dims, batch
        # (dim 0) and cell-count Npad (dim 1), are marked dynamic (in
        # _run_forward), so a single graph parameterized by symbolic (B, Npad) is
        # built on the first flush and reused for all later shapes.
        # Enabled on cuda; opt out with HEXFIELD_NO_COMPILE=1. Falls back to eager
        # on any compile error.
        self._raw_fpv = self.model.forward_policy_value
        self._compiled_fpv = self._raw_fpv
        self._use_compile = (
            self.device.type == "cuda"
            and os.environ.get("HEXFIELD_NO_COMPILE") != "1"
        )
        # When set, the per-group decode/softmax/gather (the two device syncs)
        # runs in result() instead of submit_payload(), so submit only enqueues
        # forwards. Enabled by HEXFIELD_DEFER_DECODE=1; default off.
        self._defer_decode = os.environ.get("HEXFIELD_DEFER_DECODE") == "1"
        # Keep feats f16 through pack + H2D on cuda. HEXFIELD_F32_FEATS=1 forces
        # the f32 path.
        self._f16_feats = (
            self.device.type == "cuda"
            and os.environ.get("HEXFIELD_F32_FEATS") != "1"
        )
        # When set, grouping, per-group padding, and f16/int buffer assembly run
        # in Rust, exposed as zero-copy buffers consumed via torch.frombuffer +
        # .to(device). Enabled by HEXFIELD_RUST_PACK=1. Gated on _f16_feats: the
        # Rust pack emits f16 feats only, so the f32 path falls back to the
        # CSR/Python pack.
        self._rust_pack = (
            self.device.type == "cuda"
            and self._f16_feats
            and os.environ.get("HEXFIELD_RUST_PACK") == "1"
        )
        if self._use_compile:
            # suppress_errors drops to eager for any shape that fails to compile.
            # cache_size_limit covers the specializations in use (request_moves_left
            # True/False and the batch-size-1 guard).
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.automatic_dynamic_shapes = True
            torch._dynamo.config.cache_size_limit = max(
                64, torch._dynamo.config.cache_size_limit
            )
            try:
                self._compiled_fpv = torch.compile(self._raw_fpv, dynamic=True)
            except Exception:
                self._compiled_fpv = self._raw_fpv

    def __call__(self, payload: dict) -> dict:
        return self.evaluate_payload(payload)

    @torch.no_grad()
    def evaluate_payload(self, payload: dict) -> dict:
        """Synchronous serve: submit the payload and immediately drain it."""
        return self.result(self.submit_payload(payload))

    @torch.no_grad()
    def submit_payload(self, payload: dict) -> dict:
        """Parse the request and enqueue every forward group on the GPU without
        synchronizing: decoded outputs stay on-device and no .cpu() runs here.
        Returns an opaque handle; call result(handle) to drain it. Paired with
        result() so the caller can overlap other work between the two."""
        if int(payload["abi"]) != 1:
            raise ValueError(f"unsupported hexfield ABI {payload['abi']}")
        b, total_nodes = (int(x) for x in payload["shape"])
        offsets = np.asarray(payload["node_row_offsets"], dtype=np.int64)
        if offsets.shape[0] != b + 1 or int(offsets[-1]) != total_nodes:
            raise ValueError("node_row_offsets inconsistent with shape")
        legal_counts = np.frombuffer(payload["legal_counts"], dtype=np.int32)
        if legal_counts.shape[0] != b:
            raise ValueError("legal_counts byte count mismatch")
        request_ml = bool(payload.get("request_moves_left", False))

        if self._rust_pack:
            return self._submit_rust_pack(payload, b, offsets, legal_counts, request_ml)

        feats16 = np.frombuffer(payload["node_feats"], dtype=np.float16)
        if feats16.shape[0] != total_nodes * NUM_FEATURES:
            raise ValueError("node_feats byte count mismatch")
        # Wire feats are f16. Keep them f16 when _f16_feats is set (cuda); the
        # CPU path and the F32_FEATS toggle upcast to f32.
        feats = (
            feats16.reshape(total_nodes, NUM_FEATURES)
            if self._f16_feats
            else feats16.astype(np.float32).reshape(total_nodes, NUM_FEATURES)
        )
        qr = np.frombuffer(payload["node_qr"], dtype=np.int16).reshape(total_nodes, 2)
        nbr = np.frombuffer(payload["nbr"], dtype=np.uint16).reshape(total_nodes, 6)

        sizes = (offsets[1:] - offsets[:-1]).astype(np.int64)
        # Each group appends GPU tensors to these buffers; the single .cpu() sync
        # happens in result(). gpu_priors holds one flat tensor per group (the
        # group's rows' legal-prefix priors, row-major). plan_groups emits groups
        # in ascending row order, so concatenating them yields the full row-order
        # flat-priors layout.
        gpu_priors: list[torch.Tensor] = []
        gpu_values: list[torch.Tensor] = []
        gpu_ml: list[torch.Tensor] = []
        # Defer mode: collect raw per-group outputs and decode in result().
        deferred: list | None = [] if self._defer_decode else None

        for start, end, pad_to in plan_groups(sizes):
            self._forward_group(
                feats, qr, nbr, offsets, sizes, legal_counts, start, end, pad_to,
                request_ml, gpu_values, gpu_ml, gpu_priors, deferred,
            )

        if self._defer_decode:
            return {
                "b": b,
                "request_ml": request_ml,
                "legal_counts": legal_counts,
                "deferred": deferred,
            }
        # Concatenate on-GPU; the D2H syncs happen in result().
        return {
            "b": b,
            "request_ml": request_ml,
            "legal_counts": legal_counts,
            "values_gpu": torch.cat(gpu_values),
            "ml_gpu": torch.cat(gpu_ml) if request_ml else None,
            "priors_gpu": torch.cat(gpu_priors),
        }

    @torch.no_grad()
    def _submit_rust_pack(self, payload, b, offsets, legal_counts, request_ml) -> dict:
        """Consume the Rust serve-pack (HEXFIELD_RUST_PACK).

        Hands the CSR-flat wire bytes (f16 feats, i16 coords, u16 nbr) + the i64
        row offsets to `_rust.build_serve_groups`, which runs the same grouping as
        plan_groups and assembles each group's padded buffers: feats (f16, pad=0),
        nbr (i32, fill=pad_to, sentinel->pad_to), mask (u8, 1 at real nodes),
        coords (i32, pad=0). Each group's four buffers come back as read-only
        zero-copy buffers; torch.frombuffer views them and .to(device) copies them
        to the GPU. The int32 nbr/coords are cast to int64 on-device (the model's
        gather needs int64). The forward tail is the shared _run_forward."""
        from hexfield import _rust  # only the rust-pack path needs it

        dev = self.device
        groups = _rust.build_serve_groups(
            payload["node_feats"],
            payload["node_qr"],
            payload["nbr"],
            offsets.tolist(),
        )

        gpu_priors: list[torch.Tensor] = []
        gpu_values: list[torch.Tensor] = []
        gpu_ml: list[torch.Tensor] = []
        deferred: list | None = [] if self._defer_decode else None

        for grp in groups:
            start = grp["start"]
            end = grp["end"]
            gn = grp["g"]
            p = grp["pad_to"]
            # frombuffer views the zero-copy Rust buffer; .to(dev) copies it to
            # the GPU.
            d_feats = (
                torch.frombuffer(grp["feats"], dtype=torch.float16)
                .reshape(gn, p, NUM_FEATURES)
                .to(dev, non_blocking=True)
            )
            d_nbr = (
                torch.frombuffer(grp["nbr"], dtype=torch.int32)
                .reshape(gn, p, 6)
                .to(dev, non_blocking=True)
                .to(torch.int64)
            )
            d_mask = (
                torch.frombuffer(grp["mask"], dtype=torch.uint8)
                .reshape(gn, p)
                .to(dev, non_blocking=True)
                .to(torch.bool)
            )
            d_coords = (
                torch.frombuffer(grp["coords"], dtype=torch.int32)
                .reshape(gn, p, 2)
                .to(dev, non_blocking=True)
                .to(torch.int64)
            )
            self._run_forward(
                d_feats, d_nbr, d_mask, d_coords, gn, request_ml, legal_counts,
                start, end, gpu_values, gpu_ml, gpu_priors, deferred,
            )

        if self._defer_decode:
            return {
                "b": b,
                "request_ml": request_ml,
                "legal_counts": legal_counts,
                "deferred": deferred,
            }
        return {
            "b": b,
            "request_ml": request_ml,
            "legal_counts": legal_counts,
            "values_gpu": torch.cat(gpu_values),
            "ml_gpu": torch.cat(gpu_ml) if request_ml else None,
            "priors_gpu": torch.cat(gpu_priors),
        }

    @torch.no_grad()
    def result(self, handle: dict) -> dict:
        """Drain a submit_payload() handle. The .cpu() calls here are the single
        device->host sync for the whole flush."""
        b = handle["b"]
        request_ml = handle["request_ml"]
        legal_counts = handle["legal_counts"]

        # Defer mode: the per-group decode/softmax/gather was held out of submit.
        # Run it now (before the D2H below), then fall through to the concat+.cpu()
        # path.
        if "deferred" in handle:
            gpu_values, gpu_ml, gpu_priors = [], [], []
            for out, start, end in handle["deferred"]:
                value, ml, priors_flat = self._decode_group(
                    out, legal_counts, start, end, request_ml
                )
                gpu_values.append(value)
                if request_ml:
                    gpu_ml.append(ml)
                gpu_priors.append(priors_flat)
            handle["values_gpu"] = torch.cat(gpu_values)
            handle["priors_gpu"] = torch.cat(gpu_priors)
            handle["ml_gpu"] = torch.cat(gpu_ml) if request_ml else None

        values_out = handle["values_gpu"].cpu().numpy().astype(np.float32, copy=False)
        # priors_gpu is torch.cat over the per-row legal-prefix priors in row
        # order, so flat_priors is the sum(legal_counts) positional layout the Rust
        # parser walks. Emitted as one contiguous f32 buffer; the row split happens
        # Rust-side from legal_counts.
        flat_priors = np.ascontiguousarray(
            handle["priors_gpu"].cpu().numpy(), dtype=np.float32
        )

        reply = {
            "values_bytes": values_out.tobytes(),
            "priors_bytes": flat_priors.tobytes(),
        }
        if request_ml:
            reply["moves_left_bytes"] = (
                handle["ml_gpu"].cpu().numpy().astype(np.float32, copy=False).tobytes()
            )
        return reply

    def _forward_group(
        self, feats, qr, nbr, offsets, sizes, legal_counts, start, end, pad_to,
        request_ml, gpu_values, gpu_ml, gpu_priors, deferred=None,
    ) -> None:
        g = end - start
        # Build the padded (g, pad_to, *) numpy buffers per field, then one
        # from_numpy + .to(device) per field. Fields: feats (f16 on cuda, f32
        # otherwise; pad 0), nbr (int64, sentinel->pad_to, pad pad_to), mask (bool,
        # True at real nodes), coords (int64, pad 0).
        feat_dtype = np.float16 if self._f16_feats else np.float32
        np_feats = np.zeros((g, pad_to, NUM_FEATURES), dtype=feat_dtype)
        np_nbr = np.full((g, pad_to, 6), pad_to, dtype=np.int64)
        np_mask = np.zeros((g, pad_to), dtype=np.bool_)
        np_coords = np.zeros((g, pad_to, 2), dtype=np.int64)
        for k in range(g):
            row = start + k
            n = int(sizes[row])
            o = int(offsets[row])
            np_feats[k, :n] = feats[o : o + n]
            row_nbr = nbr[o : o + n].astype(np.int64)
            np_nbr[k, :n] = np.where(row_nbr == NBR_SENTINEL, pad_to, row_nbr)
            np_mask[k, :n] = True
            np_coords[k, :n] = qr[o : o + n].astype(np.int64)
        batch_feats = torch.from_numpy(np_feats)
        batch_nbr = torch.from_numpy(np_nbr)
        batch_mask = torch.from_numpy(np_mask)
        batch_coords = torch.from_numpy(np_coords)

        device = self.device
        use_fp16 = device.type == "cuda"
        # On cuda, pin each host buffer and copy with non_blocking=True so the H2D
        # can overlap queued GPU work; the consuming forward runs on the same
        # stream. CPU device uses a plain blocking copy.

        def _h2d(t):
            return t.pin_memory().to(device, non_blocking=True) if use_fp16 else t.to(device)

        d_feats = _h2d(batch_feats)
        d_nbr = _h2d(batch_nbr)
        d_mask = _h2d(batch_mask)
        d_coords = _h2d(batch_coords)
        self._run_forward(
            d_feats, d_nbr, d_mask, d_coords, g, request_ml, legal_counts,
            start, end, gpu_values, gpu_ml, gpu_priors, deferred,
        )

    def _run_forward(
        self, d_feats, d_nbr, d_mask, d_coords, g, request_ml, legal_counts,
        start, end, gpu_values, gpu_ml, gpu_priors, deferred,
    ) -> None:
        """Shared forward tail for the CSR (_forward_group) and Rust-pack
        (_submit_rust_pack) packers. Takes the four device tensors in their final
        dtypes (feats f16/f32, nbr int64, mask bool, coords int64) and runs the
        compiled/eager forward: the batch-1 duplicate-to-2 guard, mark_dynamic,
        autocast, and the defer-or-decode path."""
        device = self.device
        use_fp16 = device.type == "cuda"
        # Mark both varying dims dynamic: batch (dim 0) and cell-count Npad (dim 1).
        # A concrete batch of 1 is specialized away by dynamo, leaving Npad the
        # sole free symbol, which trips Inductor's CantSplit on the attention
        # head-merge reshape. A size-1 group is duplicated to batch 2 (the model is
        # per-row batch-inert, so row 0's outputs are unchanged) and the twin is
        # sliced off after.
        use_compiled = self._use_compile and self._compiled_fpv is not self._raw_fpv
        fpv = self._compiled_fpv if use_compiled else self._raw_fpv
        pad_batch = use_compiled and g == 1
        if pad_batch:
            d_feats = d_feats.repeat(2, 1, 1)
            d_nbr = d_nbr.repeat(2, 1, 1)
            d_mask = d_mask.repeat(2, 1)
            d_coords = d_coords.repeat(2, 1, 1)
        if use_compiled:
            for t in (d_feats, d_nbr, d_mask, d_coords):
                torch._dynamo.mark_dynamic(t, 0)  # batch
                torch._dynamo.mark_dynamic(t, 1)  # Npad
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            out = fpv(
                d_feats,
                d_nbr,
                d_mask,
                d_coords,
                request_moves_left=request_ml,
            )
        if pad_batch:  # drop the duplicated twin row -> back to g == 1
            out = {k: v[:g] for k, v in out.items()}
        # Defer mode: stash the raw forward outputs and run the per-group
        # decode/softmax/gather in result(). That decode has two device syncs (the
        # group_counts H2D and the priors[legal] gather), so running it here would
        # make submit_payload block on each group's forward.
        if deferred is not None:
            deferred.append((out, start, end))
            return
        value, ml, priors_flat = self._decode_group(out, legal_counts, start, end, request_ml)
        gpu_values.append(value)
        if request_ml:
            gpu_ml.append(ml)
        gpu_priors.append(priors_flat)

    def _decode_group(self, out, legal_counts, start, end, request_ml):
        """Per-group serve decode: binned value, moves-left, and the flattened
        legal-prefix prior gather. Contains the two device syncs (group_counts H2D
        and the priors[legal] gather); invoked from submit (immediate) or result
        (deferred). Decoded value/ml are (g,) GPU tensors; priors[legal] flattens
        each row's first legal_counts[row] entries in row order (`legal` is
        row-major and l==0 rows select nothing)."""
        value = decode_binned_value(out["value"].float())
        ml = decode_moves_left(out["moves_left"].float()) if request_ml else None
        logits = out["policy"].float()
        # Set columns at index >= the row's legal count to -inf before one batched
        # softmax. Model logits are mask-zeroed (not -inf), so a bare slice softmax
        # would let the zeros enter the denominator. The -inf columns contribute
        # exp(-inf)=0, so each [:l] slice equals torch.softmax(logits[k, :l]).
        group_counts = torch.from_numpy(
            np.ascontiguousarray(legal_counts[start:end])
        ).to(logits.device, dtype=torch.long)
        col_idx = torch.arange(logits.shape[1], device=logits.device)
        legal = col_idx.unsqueeze(0) < group_counts.unsqueeze(1)  # (g, Npad)
        masked = logits.masked_fill(~legal, float("-inf"))
        priors = torch.softmax(masked, dim=1)  # fp32, GPU; rows with l==0 -> NaN
        return value, ml, priors[legal]
