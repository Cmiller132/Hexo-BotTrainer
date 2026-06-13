"""HexfieldEvaluator — the serve-side half of the §5.2 ABI.

Consumes the Rust payload (CSR over support nodes, rows pre-sorted by support
size descending), packs rows into 256-quantized static shapes under the
inference pair ceiling (§5.3), runs `forward_policy_value`, and returns the
reply: `values_bytes` (f32 x B, clamped [-1, 1]), `priors_bytes` (f32 x sum
L_g, positional over each row's legal prefix, fp32 softmax), and
`moves_left_bytes` (f32 x B, median-of-bins decisions) when requested.
"""

from __future__ import annotations

import numpy as np
import torch

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

    def __call__(self, payload: dict) -> dict:
        return self.evaluate_payload(payload)

    @torch.no_grad()
    def evaluate_payload(self, payload: dict) -> dict:
        if int(payload["abi"]) != 1:
            raise ValueError(f"unsupported hexfield ABI {payload['abi']}")
        b, total_nodes = (int(x) for x in payload["shape"])
        offsets = np.asarray(payload["node_row_offsets"], dtype=np.int64)
        if offsets.shape[0] != b + 1 or int(offsets[-1]) != total_nodes:
            raise ValueError("node_row_offsets inconsistent with shape")
        feats16 = np.frombuffer(payload["node_feats"], dtype=np.float16)
        if feats16.shape[0] != total_nodes * NUM_FEATURES:
            raise ValueError("node_feats byte count mismatch")
        feats = feats16.astype(np.float32).reshape(total_nodes, NUM_FEATURES)
        qr = np.frombuffer(payload["node_qr"], dtype=np.int16).reshape(total_nodes, 2)
        nbr = np.frombuffer(payload["nbr"], dtype=np.uint16).reshape(total_nodes, 6)
        legal_counts = np.frombuffer(payload["legal_counts"], dtype=np.int32)
        if legal_counts.shape[0] != b:
            raise ValueError("legal_counts byte count mismatch")
        request_ml = bool(payload.get("request_moves_left", False))

        sizes = (offsets[1:] - offsets[:-1]).astype(np.int64)
        values_out = np.empty(b, dtype=np.float32)
        ml_out = np.empty(b, dtype=np.float32) if request_ml else None
        prior_chunks: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * b
        # Single-D2H discipline (§5.3): every group appends GPU tensors to these
        # buffers; ONE .cpu() sync happens at the very end, not per row/group.
        gpu_priors: list[torch.Tensor] = [None] * b  # type: ignore[list-item]
        gpu_values: list[torch.Tensor] = []
        gpu_ml: list[torch.Tensor] = []

        # Padding-aware grouping (rows arrive size-descending): 64-quantized
        # Npad, batch under the pair ceiling, split to bound padding waste.
        for start, end, pad_to in plan_groups(sizes):
            self._forward_group(
                feats, qr, nbr, offsets, sizes, legal_counts, start, end, pad_to,
                request_ml, gpu_values, gpu_ml, gpu_priors,
            )

        # ONE device->host sync for the whole flush.
        values_out[:] = torch.cat(gpu_values).cpu().numpy()
        if request_ml:
            ml_out[:] = torch.cat(gpu_ml).cpu().numpy()
        # Concatenate all per-row priors on-GPU (variable length), one D2H.
        flat_priors = torch.cat([gpu_priors[i] for i in range(b)]).cpu().numpy()
        pos = 0
        for i in range(b):
            l = int(legal_counts[i])
            prior_chunks[i] = flat_priors[pos : pos + l].astype(np.float32, copy=False)
            pos += l

        reply = {
            "values_bytes": values_out.tobytes(),
            "priors_bytes": np.concatenate(prior_chunks).astype(np.float32).tobytes(),
        }
        if request_ml:
            reply["moves_left_bytes"] = ml_out.tobytes()
        return reply

    def _forward_group(
        self, feats, qr, nbr, offsets, sizes, legal_counts, start, end, pad_to,
        request_ml, gpu_values, gpu_ml, gpu_priors,
    ) -> None:
        g = end - start
        # Vectorized host pack: build the padded (g, pad_to, *) numpy buffers in
        # one pass per field, then a single from_numpy + .to(device) per field.
        # Byte-for-byte identical to the prior per-row from_numpy/torch.where
        # loop (same fp32 feats, same sentinel->pad_to neighbor remap, same
        # int64 coords, same bool mask), only without g separate host copies.
        np_feats = np.zeros((g, pad_to, NUM_FEATURES), dtype=np.float32)
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
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            out = self.model.forward_policy_value(
                batch_feats.to(device),
                batch_nbr.to(device),
                batch_mask.to(device),
                batch_coords.to(device),
                request_moves_left=request_ml,
            )
        # Stay on-GPU: no .cpu() here. Decoded values/ml are (g,) GPU tensors;
        # per-row legal-prefix softmaxes are concatenated on-GPU. The single
        # D2H sync happens once in evaluate_payload (§5.3).
        gpu_values.append(decode_binned_value(out["value"].float()))
        if request_ml:
            gpu_ml.append(decode_moves_left(out["moves_left"].float()))
        logits = out["policy"].float()
        # Batched legal-prefix softmax. Each row's prior is softmax over its
        # first legal_counts[row] columns. Policy logits are mask-zeroed in the
        # model (illegal columns are 0.0, NOT -inf), so a bare softmax over a
        # fixed slice would let those zeros pollute the denominator; instead we
        # set every column at index >= the row's legal count to -inf before one
        # batched softmax. softmax subtracts the per-row max over the *legal*
        # prefix in both forms (the -inf columns contribute exp(-inf)=0 to
        # numerator and denominator), so each [:l] slice is numerically
        # identical to the prior per-row torch.softmax(logits[k, :l]).
        group_counts = torch.from_numpy(
            np.ascontiguousarray(legal_counts[start:end])
        ).to(logits.device, dtype=torch.long)
        col_idx = torch.arange(logits.shape[1], device=logits.device)
        legal = col_idx.unsqueeze(0) < group_counts.unsqueeze(1)  # (g, Npad)
        masked = logits.masked_fill(~legal, float("-inf"))
        priors = torch.softmax(masked, dim=1)  # fp32, GPU; rows with l==0 -> NaN
        for k in range(g):
            row = start + k
            l = int(legal_counts[row])
            # Slice the legal prefix; rows with l == 0 yield an empty tensor, so
            # any all-(-inf)-row NaNs are never read (matches the old behavior).
            gpu_priors[row] = priors[k, :l]
