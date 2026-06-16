"""HexfieldEvaluator — the serve-side half of the §5.2 ABI.

Consumes the Rust payload (CSR over support nodes, rows pre-sorted by support
size descending), packs rows into 256-quantized static shapes under the
inference pair ceiling (§5.3), runs `forward_policy_value`, and returns the
reply: `values_bytes` (f32 x B, clamped [-1, 1]), `priors_bytes` (f32 x sum
L_g, positional over each row's legal prefix, fp32 softmax), and
`moves_left_bytes` (f32 x B, median-of-bins decisions) when requested.
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
# B * S_pad^2 <= this keeps the fp16 (B, 4, S, S) bias transient <= ~305 MB
# (§5.3 — the INFERENCE ceiling, distinct from the training pair budget).
PAIR_CEILING = 3.8e7
# Pad quantum: finer (64 vs the old 256) cuts padded-cell waste, the dominant
# serve inefficiency on skewed support mixes (M8 workflow finding).
QUANT_NODES = 64
# Bound padding waste: don't pad a row up to a group anchor more than ~18%
# larger than the row's own size (or 64 nodes), so the squared attention waste
# (sum B*S_pad^2) stays small where S is large. Coalescing for batch size is
# still allowed where padding is cheap (small S).
WASTE_FRACTION = 0.18


def _ceil_quant(n: int) -> int:
    return max(QUANT_NODES, -(-int(n) // QUANT_NODES) * QUANT_NODES)


def plan_groups(sizes) -> list[tuple[int, int, int]]:
    """Padding-aware grouping over rows sorted DESCENDING by size. Returns
    (start, end, pad_to) groups. pad_to is the 64-quantized anchor (largest
    row in the group) so pad_to >= every row (pad-inertness preserved). A
    group stops extending when (a) the pair ceiling would be exceeded or
    (b) the next row is smaller than the anchor pad by more than the waste
    bound — keeping squared padding waste low at large S."""
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
            if int(sizes[end]) < floor:  # too much padding waste -> split
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
        # Serve forward compile (spec §5.3). It is the throughput bottleneck
        # (~70% is the rel-pos bias machinery: many small elementwise/gather
        # kernels); torch.compile fuses them for ~2.4x at fp16-tolerance parity.
        # Eval/self-play only — training never goes through HexfieldEvaluator.
        #
        # ONE dynamic compile serves EVERY shape. Both varying dims — batch (dim 0)
        # and cell-count Npad (dim 1) — are marked dynamic (in _forward_group) and
        # compile() is invoked with dynamic=True, so Inductor builds a single graph
        # parameterized by symbolic (B, Npad) on the first (small) flush and reuses
        # it for all later shapes, deep late-game included. This REPLACES the old
        # static-per-Npad scheme and its `HEXFIELD_COMPILE_MAX_NPAD<=1024` cutoff.
        #
        # Why the old scheme existed, and why it was wrong: a prior note claimed
        # dynamic Npad raises `CantSplit: 96*s+768 not divisible by s+8` (the
        # attention out-reshape CHANNELS*(Npad+8) over seq-len Npad+8) and forces a
        # silent eager fallback, so it pinned Npad static and compiled ~48 buckets —
        # which then HUNG the single self-play thread the first time a deep shape
        # compiled. MEASURED FALSE on torch 2.12 (scripts/_hexfield_compile_diag.py,
        # 2026-06-16, epoch-34 ckpt): `compile(dynamic=True)` + mark_dynamic on the
        # Npad dim compiles ONCE (~27 s on Npad=256) and serves Npad 512..2560 with
        # NO recompile and NO CantSplit (4–13 ms reuse). It is also numerically the
        # SAME path as the shipped static compile: dyn-vs-static == static-vs-eager
        # == one fp16 ulp (the compile noise floor, not added error). Because there
        # is now exactly ONE compile — on the first small shape — the deep-shape
        # compile that hung can never occur; the cutoff is gone.
        # Opt out with HEXFIELD_NO_COMPILE=1; falls back to eager on any error.
        self._raw_fpv = self.model.forward_policy_value
        self._compiled_fpv = self._raw_fpv
        self._use_compile = (
            self.device.type == "cuda"
            and os.environ.get("HEXFIELD_NO_COMPILE") != "1"
        )
        # Defer the per-group decode/softmax/gather (the two device syncs) from
        # submit_payload to result(), so submit only enqueues forwards and the
        # pre-backup select pass overlaps them. Opt-in A/B knob; default OFF.
        self._defer_decode = os.environ.get("HEXFIELD_DEFER_DECODE") == "1"
        # Keep feats f16 through pack+H2D (cuda) — half the feats H2D + no astype
        # copy. HEXFIELD_F32_FEATS=1 forces the old f32 path (A/B only).
        self._f16_feats = (
            self.device.type == "cuda"
            and os.environ.get("HEXFIELD_F32_FEATS") != "1"
        )
        if self._use_compile:
            # Keep the eager fallback (suppress_errors) as an unattended-run
            # safety net: if some never-before-seen shape ever fails to compile,
            # the run drops to eager for it instead of dying. automatic_dynamic
            # stays ON (it cannot hurt once dynamic=True already generalizes Npad).
            # cache_size_limit covers the handful of real specializations
            # (request_moves_left True/False, and a batch-size-1 guard).
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
        """Synchronous serve (eval arena + the non-overlapped self-play path):
        enqueue the forward and immediately read it back."""
        return self.result(self.submit_payload(payload))

    @torch.no_grad()
    def submit_payload(self, payload: dict) -> dict:
        """Phase 1 of the async serve split (§5.3 overlap): parse the request and
        ENQUEUE every forward group on the GPU, but do NOT synchronize — the
        decoded outputs stay on-device and no .cpu() runs here. Returns an opaque
        handle. The caller (Rust) can then run the pre-backup select pass with the
        GIL released while these kernels execute, and only afterwards call
        result(handle) to drain them. The math is identical to evaluate_payload;
        only the host/GPU sync point moves."""
        if int(payload["abi"]) != 1:
            raise ValueError(f"unsupported hexfield ABI {payload['abi']}")
        b, total_nodes = (int(x) for x in payload["shape"])
        offsets = np.asarray(payload["node_row_offsets"], dtype=np.int64)
        if offsets.shape[0] != b + 1 or int(offsets[-1]) != total_nodes:
            raise ValueError("node_row_offsets inconsistent with shape")
        feats16 = np.frombuffer(payload["node_feats"], dtype=np.float16)
        if feats16.shape[0] != total_nodes * NUM_FEATURES:
            raise ValueError("node_feats byte count mismatch")
        # Keep feats f16 (dense_cnn-style): the wire is already f16 and the serve
        # forward runs f16 under autocast, so the old astype(float32) was a wasteful
        # host copy that doubled the feats H2D — autocast downcasts the upcast back
        # to the identical f16 value. Pack/H2D below build f16 on cuda (half size,
        # no astype), f32 only on the rare CPU path / the F32_FEATS A/B toggle.
        feats = (
            feats16.reshape(total_nodes, NUM_FEATURES)
            if self._f16_feats
            else feats16.astype(np.float32).reshape(total_nodes, NUM_FEATURES)
        )
        qr = np.frombuffer(payload["node_qr"], dtype=np.int16).reshape(total_nodes, 2)
        nbr = np.frombuffer(payload["nbr"], dtype=np.uint16).reshape(total_nodes, 6)
        legal_counts = np.frombuffer(payload["legal_counts"], dtype=np.int32)
        if legal_counts.shape[0] != b:
            raise ValueError("legal_counts byte count mismatch")
        request_ml = bool(payload.get("request_moves_left", False))

        sizes = (offsets[1:] - offsets[:-1]).astype(np.int64)
        # Single-D2H discipline (§5.3): every group appends GPU tensors to these
        # buffers; the ONE .cpu() sync happens later, in result(). gpu_priors holds
        # ONE flat tensor PER GROUP (the group's rows' legal-prefix priors already
        # concatenated row-major); plan_groups emits groups in ascending row order,
        # so concatenating them is the full row-order flat-priors layout.
        gpu_priors: list[torch.Tensor] = []
        gpu_values: list[torch.Tensor] = []
        gpu_ml: list[torch.Tensor] = []
        # DEFER mode: collect raw per-group outputs; decode in result() (see
        # _forward_group) so submit only enqueues forwards and select can overlap.
        deferred: list | None = [] if self._defer_decode else None

        # Padding-aware grouping (rows arrive size-descending): 64-quantized
        # Npad, batch under the pair ceiling, split to bound padding waste.
        for start, end, pad_to in plan_groups(sizes):
            self._forward_group(
                feats, qr, nbr, offsets, sizes, legal_counts, start, end, pad_to,
                request_ml, gpu_values, gpu_ml, gpu_priors, deferred,
            )

        if self._defer_decode:
            # Raw forwards enqueued; the syncing decode happens in result().
            return {
                "b": b,
                "request_ml": request_ml,
                "legal_counts": legal_counts,
                "deferred": deferred,
            }
        # Concatenate on-GPU (still no D2H); the syncs happen in result().
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
        """Phase 2: drain a submit_payload() handle. The .cpu() calls here are the
        single device->host sync for the whole flush; bytes are identical to the
        synchronous path."""
        b = handle["b"]
        request_ml = handle["request_ml"]
        legal_counts = handle["legal_counts"]

        # DEFER mode: the per-group decode/softmax/gather was held out of submit so
        # the forwards could overlap the select pass. Do it now (still BEFORE the one
        # D2H below), then fall through to the identical concat+.cpu() path.
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
        # priors_gpu is already torch.cat over the per-row legal-prefix priors in
        # row order, so flat_priors IS the sum(legal_counts) positional layout the
        # Rust parser walks — the old per-row slice + np.concatenate rebuilt the
        # identical array. Emit it directly (one contiguous f32 buffer). legal_counts
        # is unused here now but kept in the handle for the parser's row split.
        flat_priors = np.ascontiguousarray(
            handle["priors_gpu"].cpu().numpy(), dtype=np.float32
        )
        _ = legal_counts  # row split happens Rust-side from legal_counts on the wire

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
        # Vectorized host pack: build the padded (g, pad_to, *) numpy buffers in
        # one pass per field, then a single from_numpy + .to(device) per field.
        # Byte-for-byte identical to the prior per-row from_numpy/torch.where
        # loop (same fp32 feats, same sentinel->pad_to neighbor remap, same
        # int64 coords, same bool mask), only without g separate host copies.
        # f16 feats on cuda -> half the H2D bytes + no astype copy (CPU path stays
        # f32; numpy upcasts the f16 source on assignment there).
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
        # Pinned + non_blocking H2D (CUDA): page-lock each freshly-allocated host
        # buffer so the driver can DMA it asynchronously and overlap the copies
        # with queued GPU work, instead of a synchronous pageable bounce copy.
        # Bit-identical — pinning changes only the host allocator and non_blocking
        # only the copy timing; the consuming forward runs on the same stream so
        # ordering holds. CPU device keeps the plain blocking copy.

        def _h2d(t):
            return t.pin_memory().to(device, non_blocking=True) if use_fp16 else t.to(device)

        d_feats = _h2d(batch_feats)
        d_nbr = _h2d(batch_nbr)
        d_mask = _h2d(batch_mask)
        d_coords = _h2d(batch_coords)
        # One dynamic graph for every shape (see __init__): mark BOTH varying dims
        # dynamic — batch (dim 0) and cell-count Npad (dim 1) — so the single
        # compiled graph absorbs all (B, Npad) without recompiling.
        #
        # A CONCRETE batch of 1 is the one shape dynamo refuses to keep symbolic
        # (it always specializes a size-1 dim away). That leaves Npad the sole free
        # symbol, and Inductor then trips on the attention head-merge transpose-copy
        # — domain CHANNELS*(Npad+NUM_TOKENS), which it tiles as (S, CHANNELS) and
        # cannot split by the compound seq-len S = Npad+NUM_TOKENS:
        # `CantSplit: 96*s+768 not divisible by s+8` (REPRODUCED on torch 2.12,
        # scripts/_hexfield_compile_diag.py). With batch >= 2 the batch dim stays a
        # free symbol and the very same graph compiles cleanly for every Npad. So
        # duplicate a size-1 compiled group to batch 2 (the model is pad-/batch-
        # inert per row, §6.3, so row 0's outputs are unchanged by a twin) and slice
        # the twin off after. Cost is one extra row on the rare singleton group.
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
                torch._dynamo.mark_dynamic(t, 0)  # batch (>= 2 here) dynamic
                torch._dynamo.mark_dynamic(t, 1)  # Npad dynamic
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            out = fpv(
                d_feats,
                d_nbr,
                d_mask,
                d_coords,
                request_moves_left=request_ml,
            )
        if pad_batch:  # drop the duplicated twin row -> back to the true g == 1
            out = {k: v[:g] for k, v in out.items()}
        # DEFER mode (HEXFIELD_DEFER_DECODE): stash the RAW forward outputs and do
        # the per-group decode/softmax/gather later, in result(). The decode has two
        # device syncs — the group_counts H2D and the priors[legal] boolean gather's
        # nonzero — so doing it HERE makes submit_payload block on each group's
        # forward, defeating the submit->select overlap (submit can't return until
        # the GPU is done). Deferring to result() lets submit only ENQUEUE the
        # forwards (truly async); the pre-backup select pass then overlaps them.
        # Math is identical (same ops/order) -> bit-identical outputs.
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
        legal-prefix prior gather. Holds the two device syncs (group_counts H2D +
        the priors[legal] nonzero); invoked from submit (immediate) or result
        (deferred). Decoded values/ml are (g,) GPU tensors; priors[legal] flattens
        each row's first legal_counts[row] entries in row order (== the old per-row
        slice loop, since `legal` is row-major and l==0 rows select nothing)."""
        value = decode_binned_value(out["value"].float())
        ml = decode_moves_left(out["moves_left"].float()) if request_ml else None
        logits = out["policy"].float()
        # Set columns at index >= the row's legal count to -inf before one batched
        # softmax (logits are mask-ZEROED, not -inf, in the model, so a bare slice
        # softmax would let the zeros pollute the denominator). The -inf columns add
        # exp(-inf)=0 to num+denom, so each [:l] slice equals torch.softmax(logits[k, :l]).
        group_counts = torch.from_numpy(
            np.ascontiguousarray(legal_counts[start:end])
        ).to(logits.device, dtype=torch.long)
        col_idx = torch.arange(logits.shape[1], device=logits.device)
        legal = col_idx.unsqueeze(0) < group_counts.unsqueeze(1)  # (g, Npad)
        masked = logits.masked_fill(~legal, float("-inf"))
        priors = torch.softmax(masked, dim=1)  # fp32, GPU; rows with l==0 -> NaN
        return value, ml, priors[legal]
