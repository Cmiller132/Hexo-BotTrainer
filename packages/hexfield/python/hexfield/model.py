"""hexfield network — spec §2 (trunk), §2.3 (attention/bias), §2.4/§3 (heads).

Trunk interleave C C C A C C A C A over variable-N node sets: 6 plain
post-activation conv residual blocks (restnet's `ResidualBlock` form — NOT
dense_cnn's gated block, §12.5) with LayerNorm everywhere, and 3 pre-norm
transformer blocks over the joint sequence [8 summary tokens ; cells], each with
its OWN learned 237-row relative-position bias table. No cuDNN convs exist anywhere —
HexNodeConv is a gather + one GEMM.

Batch conventions (built by `batching.py`):
- feats  (B, Npad, F) f32; pad rows all-zero
- nbr    (B, Npad, 6) long, row-local; missing/pad -> Npad (the appended
  zero row — conv zero-padding semantics)
- mask   (B, Npad) bool, True at real nodes
- coords (B, Npad, 2) long axial (q, r); pad coords arbitrary (never read
  through the bias because pad KEY columns are additively masked and pad
  QUERY rows are re-zeroed after every block)

Pad-inertness is an invariant, not an accident: convs and attention re-apply
the node mask after every parameter-carrying op, so a row's outputs are
bit-identical regardless of how much padding shares its batch (exactness
trio, §6.3).
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .constants import (
    ATTENTION_HEADS,
    BIAS_CELL_TOKEN_ROW,
    BIAS_DISK_RADIUS,
    BIAS_FAR_ROW,
    BIAS_OFF_AXIS_BASE,
    BIAS_ON_AXIS_BASE,
    BIAS_RING_MAX,
    BIAS_RING_MIN,
    BIAS_ROWS,
    BIAS_TOKEN_CELL_ROW,
    BIAS_TOKEN_TOKEN_ROW,
    CHANNELS,
    HEAD_DIM,
    MLP_RATIO,
    NUM_FEATURES,
    NUM_TOKENS,
    VALUE_BINS,
)
from .geometry import disk_offsets, rel_bias_index

# Finite in fp16 — closes restnet's documented -1e9 -> -inf saturation hazard.
PAD_KEY_MASK_VALUE = -3.0e4

STV_HORIZONS = (2, 6, 16)

# FlexAttention serve path (opt-in, default OFF). When HEXFIELD_SERVE_FLEX=1 the
# NO-GRAD serve forward computes the rel-pos bias INSIDE the attention kernel via
# a score_mod (coords + _cell_bias_lut + bias_table gather) and folds the pad-key
# mask into the score (PAD_KEY_MASK_VALUE additive fill) instead of materializing
# the (B, heads, S, S) bias — the profiled ~65-69% serve cost. The TRAINING (grad)
# path is untouched (stays on build_attn_bias + _BiasGather). flag-gated + read
# once at import; the import is guarded so a torch without flex still loads.
_SERVE_FLEX = os.environ.get("HEXFIELD_SERVE_FLEX") == "1"
# FlexAttention TRAINING path (opt-in, default OFF). When HEXFIELD_TRAIN_FLEX=1
# the GRAD-enabled training forward also computes the rel-pos bias INSIDE the
# attention kernel via the SAME per-block score_mod (coords + _cell_bias_lut +
# bias_table gather + pad-key fill) instead of materializing the (B, heads, S, S)
# bias — so the dense bias is not built during training either. Crucially the
# train carrier passes the FP32 master bias_tables[i] (NOT the fp16 cast the serve
# carrier uses) so the table's gradient accumulates in fp32 — same rationale as
# the _BiasGather fp32-master backward (the fp16 path overflows the hot far/ring
# rows under AMP+GradScaler). Empirically proven on torch 2.12.0+cu130 / triton
# 3.7.0: flex fwd+backward compiles, grads match materialized SDPA to ~1e-6 (fp32)
# incl the bias-table grad, finite under AMP+GradScaler. Flag-gated + read once at
# import; default OFF == today's materialized behavior bit-for-bit.
_TRAIN_FLEX = os.environ.get("HEXFIELD_TRAIN_FLEX") == "1"
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    # Inner-compiled flex_attention (dynamic=True so ONE flex kernel serves every
    # Npad). The serve forward_policy_value is itself torch.compile(dynamic=True);
    # tracing the flex_attention HOP *into* that outer graph trips the flex
    # subgraph tracer when the score_mod captures graph-internal tensors that flow
    # through several frames + a method ('lift_tracked_freevar_to_input should not
    # be called on root SubgraphTracer'). So we DECOUPLE: _flex_call is
    # torch.compiler.disable'd — the outer graph BREAKS at the attention, and the
    # flex op compiles through its OWN inner graph (the probe's proven-working
    # standalone form). The conv trunk / projections / heads stay in the outer
    # compiled graph; only the flex sub-op runs in its own. (§7 item 8 — the
    # nested lowering does recompile/assert, so the inner compile is required.)
    _flex_compiled = torch.compile(_flex_attention, dynamic=False)

    @torch.compiler.disable(recursive=False)
    def _flex_call(q, k, v, score_mod):
        return _flex_compiled(q, k, v, score_mod=score_mod)

except Exception:  # pragma: no cover - older torch without flex
    _flex_attention = None
    _flex_call = None


class _FlexBias:
    """Carrier for the serve-flex attention path. Built ONCE PER BLOCK in
    trunk() (each block gets its own bias_tables[i]) and passed in place of the
    materialized attn_bias tensor; RelPosAttention.forward detects it and routes
    to flex_attention.

    It holds the RAW inputs the score_mod needs (coords, mask, the fp16 bias
    table, the cell LUT) — NOT a pre-built closure. The score_mod closure is
    constructed locally in RelPosAttention.forward (same frame as the flex call)
    and invoked through the torch.compiler.disable'd _flex_call, which graph-breaks
    the outer dynamic compile so the flex HOP lowers in its OWN inner graph
    (see _flex_call) rather than being inlined into the parent graph — where the
    score_mod's captured graph tensors tripped the flex subgraph tracer
    ('lift_tracked_freevar_to_input should not be called on root SubgraphTracer').
    No (B, heads, S, S) tensor is ever materialized."""

    __slots__ = ("coords", "mask", "table", "lut", "m", "w")

    def __init__(self, coords, mask, table, lut, m) -> None:
        self.coords = coords
        self.mask = mask
        self.table = table
        self.lut = lut
        self.m = m
        self.w = 2 * m + 1

    def make_score_mod(self):
        """Build the flex score_mod closure (called inside RelPosAttention.forward,
        the same frame as the flex_attention call). Computes the SAME additive bias
        the materialized build_attn_bias adds to the scores — coords + _cell_bias_lut
        + the fp16 bias_table gather — plus the pad-KEY additive fill
        (PAD_KEY_MASK_VALUE) folded in via the bool mask (no reduction)."""

        nt = NUM_TOKENS
        coords = self.coords
        mask = self.mask
        table = self.table
        lut = self.lut
        m = self.m
        w = self.w
        pad_fill = PAD_KEY_MASK_VALUE

        def score_mod(score, b, h, q_idx, kv_idx):
            qc = torch.clamp(q_idx - nt, min=0)
            kc = torch.clamp(kv_idx - nt, min=0)
            dq = coords[b, kc, 0] - coords[b, qc, 0]
            dr = coords[b, kc, 1] - coords[b, qc, 1]
            qi = torch.clamp(dq, -m, m) + m
            ri = torch.clamp(dr, -m, m) + m
            cell_idx = lut[qi * w + ri]
            q_tok = q_idx < nt
            k_tok = kv_idx < nt
            row = torch.where(
                q_tok & k_tok,
                torch.full_like(cell_idx, BIAS_TOKEN_TOKEN_ROW),
                torch.where(
                    q_tok & ~k_tok,
                    torch.full_like(cell_idx, BIAS_TOKEN_CELL_ROW),
                    torch.where(
                        ~q_tok & k_tok,
                        torch.full_like(cell_idx, BIAS_CELL_TOKEN_ROW),
                        cell_idx,
                    ),
                ),
            )
            biased = score + table[row, h].to(score.dtype)
            # pad-KEY columns: a cell key (kv_idx >= nt) whose row's mask is False.
            is_pad_key = (kv_idx >= nt) & ~mask[b, kc]
            return torch.where(is_pad_key, biased + pad_fill, biased)

        return score_mod


class _BiasGather(torch.autograd.Function):
    """table[pair] with a histogram backward.

    The generic indexing backward scatter-adds ~19M gradient elements into the
    237-row table with raw atomics — profiled at 86% of total step time. With
    only BIAS_ROWS destination classes, the gradient is a per-head bincount:
    milliseconds instead of ~640 ms."""

    @staticmethod
    def forward(ctx, table: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(pair)
        ctx.rows = table.shape[0]
        return table[pair]

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (pair,) = ctx.saved_tensors
        flat = pair.reshape(-1)
        g = grad.reshape(-1, grad.shape[-1])
        acc = torch.float64 if grad.dtype == torch.float64 else torch.float32
        cols = [
            torch.bincount(flat, weights=g[:, h].to(acc), minlength=ctx.rows)
            for h in range(g.shape[1])
        ]
        return torch.stack(cols, dim=1).to(grad.dtype), None


class HexNodeConv(nn.Module):
    """Direction-typed 7-tap hex convolution: gather (B,N,7,Cin) -> one GEMM.

    Tap 0 = center; taps 1-6 = the fixed direction order D (the rotate60
    orbit of (1,0)). Mathematically dense_cnn's masked-3x3 hex conv family —
    anchored executably by the M1 oracle test against `HexConv2d`.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(7, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        # PyTorch conv default init with fan_in = 7 * C_in (spec §2.2).
        fan_in = 7 * in_channels
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, x: torch.Tensor, gather_idx: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """x (B, Npad, Cin); gather_idx (B, Npad, 7) with tap 0 = self and
        missing -> Npad; mask (B, Npad) bool. Returns (B, Npad, Cout) with
        pad rows exactly zero (the conv bias would otherwise leak into them).
        """

        b, n, c = x.shape
        x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)  # zero row at index Npad
        flat = gather_idx.reshape(b, n * 7, 1).expand(-1, -1, c)
        gathered = x_ext.gather(1, flat).reshape(b, n, 7 * c)
        out = gathered @ self.weight.reshape(7 * c, self.out_channels) + self.bias
        return out * mask.unsqueeze(-1)


class LayerScale(nn.Module):
    """Per-channel learned residual-branch scale (gamma), init 1e-4."""

    def __init__(self, channels: int, init: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class ConvBlock(nn.Module):
    """Post-activation residual block (restnet `ResidualBlock` form, LN)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = HexNodeConv(channels, channels)
        self.ln1 = nn.LayerNorm(channels)
        self.conv2 = HexNodeConv(channels, channels)
        self.ln2 = nn.LayerNorm(channels)
        self.ls = LayerScale(channels)

    def forward(
        self, x: torch.Tensor, gather_idx: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        m = mask.unsqueeze(-1)
        y = F.relu(self.ln1(self.conv1(x, gather_idx, mask))) * m
        y = self.ln2(self.conv2(y, gather_idx, mask)) * m
        return F.relu(x + self.ls(y))


class RelPosAttention(nn.Module):
    """4-head MHSA over the joint [tokens ; cells] sequence with this block's
    bias table gathered as an additive mask. Two numerically identical
    implementations: 'sdpa' (production) and 'materialized' (test oracle)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = ATTENTION_HEADS
        self.head_dim = HEAD_DIM
        self.scale = 1.0 / math.sqrt(HEAD_DIM)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.impl = "sdpa"

    def forward(self, seq: torch.Tensor, attn_bias) -> torch.Tensor:
        b, s, c = seq.shape
        h, d = self.heads, self.head_dim
        q = self.q_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        k = self.k_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        v = self.v_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        # Serve-flex: the rel-pos bias + pad mask live INSIDE the kernel via a
        # score_mod (no materialized (B,heads,S,S) tensor). block_mask is
        # deliberately None (score_mod-only, §7 item 1) so no per-shape BlockMask
        # object enters the dynamic-Npad compile. The head-merge reshape below is
        # the SAME one the batch-2 dup guards against CantSplit on — unchanged.
        # The score_mod is built HERE (same frame as the flex call) — see _FlexBias.
        if isinstance(attn_bias, _FlexBias):
            score_mod = attn_bias.make_score_mod()
            out = _flex_call(q, k, v, score_mod)
            out = out.transpose(1, 2).reshape(b, s, c)
            return self.out_proj(out)
        # Match the bias dtype to q under autocast: a dtype mismatch silently
        # drops sdpa to the slow math fallback. -3.0e4 stays finite in fp16.
        attn_bias = attn_bias.to(q.dtype)
        if self.impl == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        elif self.impl == "materialized":
            scores = (q @ k.transpose(-2, -1)) * self.scale + attn_bias
            out = torch.softmax(scores, dim=-1) @ v
        else:  # pragma: no cover - config validation
            raise ValueError(f"unknown attention impl: {self.impl}")
        out = out.transpose(1, 2).reshape(b, s, c)
        return self.out_proj(out)


class AttnBlock(nn.Module):
    """Pre-norm transformer block (restnet block semantics, GELU, ratio 2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.attn = RelPosAttention(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, MLP_RATIO * channels)
        self.fc2 = nn.Linear(MLP_RATIO * channels, channels)
        self.ls_attn = LayerScale(channels)
        self.ls_mlp = LayerScale(channels)

    def forward(
        self, seq: torch.Tensor, attn_bias: torch.Tensor, seq_mask: torch.Tensor
    ) -> torch.Tensor:
        m = seq_mask.unsqueeze(-1)
        seq = seq + self.ls_attn(self.attn(self.ln1(seq), attn_bias) * m)
        seq = seq + self.ls_mlp(self.fc2(F.gelu(self.fc1(self.ln2(seq)))) * m)
        return seq


class HexfieldNet(nn.Module):
    """The full network: stem, C C C A C C C A C C A, LN_final, heads."""

    def __init__(self) -> None:
        super().__init__()
        c = CHANNELS
        self.stem = HexNodeConv(NUM_FEATURES, c)
        self.stem_ln = nn.LayerNorm(c)
        self.conv_blocks = nn.ModuleList([ConvBlock(c) for _ in range(8)])
        self.attn_blocks = nn.ModuleList([AttnBlock(c) for _ in range(3)])
        self.tokens = nn.Parameter(torch.empty(NUM_TOKENS, c))
        # Per-block relative-position bias tables: each of the 3 attention blocks
        # gets its OWN (BIAS_ROWS, heads) table instead of one shared table. Each
        # is zero-init (a fresh run is bit-identical to the old single-zero-table
        # net); a migrated checkpoint copies the old shared table into all 3.
        self.bias_tables = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(BIAS_ROWS, ATTENTION_HEADS))
                for _ in range(len(self.attn_blocks))
            ]
        )
        self.ln_final = nn.LayerNorm(c)

        # Heads. Policy heads read cells; value/aux read tokens + the masked
        # mean-pool of cells (§2.4 — adopted §12.6).
        self.policy_conv = HexNodeConv(c, c)
        self.policy_head = nn.Linear(c, 1)
        self.opp_policy_conv = HexNodeConv(c, c)
        self.opp_policy_head = nn.Linear(c, 1)
        # Train-only per-cell Q head (#4): emitted in forward() only, never serve.
        self.cell_q_conv = HexNodeConv(c, c)
        self.cell_q_head = nn.Linear(c, VALUE_BINS)
        self.value_reduction = nn.Linear(3 * c, c)
        self.value_head = nn.Linear(c, VALUE_BINS)
        self.aux_reduction = nn.Linear(3 * c, c)
        self.stv_heads = nn.ModuleDict(
            {str(h): nn.Linear(c, VALUE_BINS) for h in STV_HORIZONS}
        )
        self.ml_reduction = nn.Linear(3 * c, c)  # moves_left's own reduction (tokens 4,5)
        self.moves_left_head = nn.Linear(c, VALUE_BINS)

        # Exact-offset LUT for the on-GPU pair-index build: (17*17,) long,
        # row = lut[(dq+8)*17 + (dr+8)] valid wherever hex-dist <= 8.
        lut = torch.zeros((2 * BIAS_DISK_RADIUS + 1) ** 2, dtype=torch.long)
        for dq, dr in disk_offsets(BIAS_DISK_RADIUS):
            lut[(dq + BIAS_DISK_RADIUS) * 17 + (dr + BIAS_DISK_RADIUS)] = rel_bias_index(dq, dr)
        self.register_buffer("_exact_lut", lut, persistent=False)

        # FULL cell-bias LUT over the entire (dq, dr) offset domain — the single
        # gather that replaces the per-forward abs/maximum/clamp/where/where bias
        # machinery (profiled ~70% of the serve forward, all memory-bound O(B*S^2)
        # elementwise over the cell-cell pair grid). row = _cell_bias_lut[
        # (clamp(dq,-M,M)+M)*W + (clamp(dr,-M,M)+M)], M = BIAS_RING_MAX+1 = 17,
        # W = 2M+1 = 35. BIT-IDENTICAL to the old machinery, not just fp16-close:
        # for |dq|,|dr| <= M the table IS rel_bias_index(dq,dr); when either coord
        # is clamped the true offset has hex-dist >= M+ (so the old path's far row)
        # AND the clamped offset lands on a |coord|==M cell whose hex-dist >= M >
        # BIAS_RING_MAX, so the table also yields the far row — the clamp only ever
        # fires on offsets that were already far. (Verified exhaustively over
        # [-40,40]^2 and on random tensors by scripts/_hexfield_bias_check.py.)
        self._cell_bias_M = BIAS_RING_MAX + 1
        cw = 2 * self._cell_bias_M + 1
        cell_lut = torch.empty(cw * cw, dtype=torch.long)
        for dq in range(-self._cell_bias_M, self._cell_bias_M + 1):
            for dr in range(-self._cell_bias_M, self._cell_bias_M + 1):
                cell_lut[(dq + self._cell_bias_M) * cw + (dr + self._cell_bias_M)] = (
                    rel_bias_index(dq, dr)
                )
        self.register_buffer("_cell_bias_lut", cell_lut, persistent=False)

        # Serve-flex flag, read ONCE (compile-time constant for the dynamo graph).
        # Only ever used on the no-grad serve path; training uses the materialized
        # build_attn_bias + _BiasGather unless _train_flex (below) is on.
        self._serve_flex = _SERVE_FLEX and _flex_attention is not None
        # Train-flex flag, read ONCE. When on AND grad is enabled, the training
        # forward takes the per-block flex path (fp32-table carrier) instead of the
        # materialized build_attn_bias. Default OFF == today's materialized behavior
        # bit-for-bit; serve (no-grad) path is unaffected by this flag.
        self._train_flex = _TRAIN_FLEX and _flex_attention is not None

        self._init_weights()

    def _init_weights(self) -> None:
        """trunc_normal(0.02) Linears; LN (1, 0); convs PyTorch default
        (done in HexNodeConv); per-block bias tables zero (already, from the
        ParameterList constructor — must NOT be touched here); tokens
        trunc_normal(0.02). Residual-branch identity at step 0 now comes from
        LayerScale(init=1e-4) on every branch (ConvBlock.ls; AttnBlock
        ls_attn/ls_mlp), so the explicit residual-closing zero-inits are gone.
        LayerScale.gamma is neither Linear nor LayerNorm, so the generic loops
        leave its 1e-4 fill untouched."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.tokens, std=0.02)

    def set_attention_impl(self, impl: str) -> None:
        for block in self.attn_blocks:
            block.attn.impl = impl

    # --- pair index + bias (pair built once per batch; bias built per A block) ---

    def _build_pair(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Block-INDEPENDENT pieces of the bias build, computed ONCE per forward
        and reused by all 3 attention blocks (was inlined in build_attn_bias).

        Returns (pair, key_pad):
        - pair (B, S, S) long: the per-pair bias-table ROW index. S = NUM_TOKENS +
          Npad. Tokens sit at slots 0-7 with no board position; token keys are
          never masked, so a fully-masked softmax row is structurally impossible.
        - key_pad (B, S) bool: True at live keys (the pad-KEY additive fill mask).

        coords (B, Npad, 2) long; mask (B, Npad) bool."""

        b, n, _ = coords.shape
        dq = coords[:, None, :, 0] - coords[:, :, None, 0]  # (B, N, N) key - query
        dr = coords[:, None, :, 1] - coords[:, :, None, 1]
        # Single precomputed-LUT gather over the whole offset domain — bit-exact
        # replacement for the old abs/maximum/clamp/where/where pipeline (see the
        # _cell_bias_lut construction note in __init__). One clamp + one mul-add +
        # one gather instead of ~10 full (B, N, N) elementwise/select kernels.
        m = self._cell_bias_M
        w = 2 * m + 1
        qi = dq.clamp(-m, m) + m
        ri = dr.clamp(-m, m) + m
        cell_idx = self._cell_bias_lut[(qi * w + ri).reshape(-1)].reshape(b, n, n)

        s = NUM_TOKENS + n
        pair = coords.new_full((b, s, s), BIAS_TOKEN_TOKEN_ROW)
        pair[:, :NUM_TOKENS, NUM_TOKENS:] = BIAS_TOKEN_CELL_ROW
        pair[:, NUM_TOKENS:, :NUM_TOKENS] = BIAS_CELL_TOKEN_ROW
        pair[:, NUM_TOKENS:, NUM_TOKENS:] = cell_idx

        # Pad-cell KEY columns: additive, finite in fp16. Token keys untouched.
        # The mask MUST be added before the returned attn_mask is materialized so
        # it has stride(-1) == 1 on the key axis: a non-stride-1 attn_mask silently
        # forces F.scaled_dot_product_attention onto the FP32 MATH backend (the
        # profiled ampere_sgemm + fp32 softmax) instead of the fused fp16
        # mem-efficient kernel.
        key_pad = torch.cat(
            [mask.new_ones(b, NUM_TOKENS), mask], dim=1
        )  # (B, S) True = live key
        return pair, key_pad

    def build_attn_bias(
        self, pair: torch.Tensor, key_pad: torch.Tensor, block: int
    ) -> torch.Tensor:
        """(B, heads, S, S) additive bias for attention block `block`, using that
        block's OWN table self.bias_tables[block] + the pad-key mask.

        `pair` (B, S, S) row indices and `key_pad` (B, S) live-key mask come from
        _build_pair (built once per forward, shared across the 3 blocks)."""

        table = self.bias_tables[block]
        if torch.is_grad_enabled():
            # TRAINING: gather in FP32 via _BiasGather. Casting the table to fp16
            # before _BiasGather made its backward accumulate fp16 gradients; the
            # hot far/ring rows receive ~19M scatter-adds and overflow to inf under
            # AMP+GradScaler (HIGH audit fix). The fp32 master-table bincount
            # backward keeps the gradient finite. Add the mask in head-LAST layout
            # (key = dim 2), then ONE permute+contiguous to (B, heads, Sq, Sk).
            bias = _BiasGather.apply(table, pair)  # (B, Sq, Sk, heads) fp32
            fill = torch.where(key_pad, 0.0, PAD_KEY_MASK_VALUE).to(bias.dtype)
            bias = bias + fill[:, None, :, None]  # broadcast over key axis (dim 2)
            return bias.permute(0, 3, 1, 2).contiguous()  # (B, heads, Sq, Sk)

        # INFERENCE (no grad — self-play/eval, the throughput path): no backward
        # exists, so build the ENTIRE (B,heads,S,S) bias in fp16 directly in the
        # head-FIRST layout. Indexing the TRANSPOSED table (heads, ROWS) yields a
        # contiguous (heads, B, Sq, Sk); permute(1,0,2,3) to (B, heads, Sq, Sk) is
        # then a stride(-1)==1 VIEW, and the pad-mask add is the SINGLE full-tensor
        # materialization — eliminating the separate permute+contiguous() copy the
        # head-last path needed (one fewer (B,heads,S,S) write per forward). The
        # arithmetic (table[pair] + fill broadcast over keys) is bit-identical to
        # the head-last fp16 inference path (guarded by
        # test_sdpa_equals_materialized_fp16_cuda + scripts/_hexfield_bias_check.py).
        bias_t = table.to(torch.float16).t().contiguous()  # (heads, ROWS)
        bias = bias_t[:, pair]                       # (heads, B, Sq, Sk) contiguous
        bias = bias.permute(1, 0, 2, 3)              # (B, heads, Sq, Sk) view, stride(-1)=1
        fill = torch.where(key_pad, 0.0, PAD_KEY_MASK_VALUE).to(bias.dtype)
        return bias + fill[:, None, None, :]         # broadcast over key axis (dim 3)

    def _build_flex_bias(
        self, coords: torch.Tensor, mask: torch.Tensor, block: int
    ) -> "_FlexBias":
        """Serve-flex (no-grad) equivalent of build_attn_bias for attention block
        `block`. Packages the RAW tensors the score_mod needs (coords, mask, fp16
        bias table for THIS block, cell LUT) into a _FlexBias carrier; the actual
        score_mod closure is built downstream in RelPosAttention.forward (same
        frame as the flex call — see _FlexBias). No (B, heads, S, S) tensor is
        materialized; block_mask is None (the pad mask is folded into the score,
        §7 item 1). Validated to <=3e-4 parity vs build_attn_bias by
        scripts/_hexfield_flex_probe.py (score_mod_pad variant).

        Each block now hands the (unchanged) score_mod a DIFFERENT table object
        (self.bias_tables[block]) — per-block on serve is ~free (the score_mod
        just reads a different table).

        Pad-key boundary is read DIRECTLY from the bool mask (an input tensor,
        fully static shape), NOT mask.sum(): a reduction inside the score_mod
        produces a data-dependent (unbacked) symint the dynamic-Npad Inductor
        lowering cannot bind. (ROWS, heads) fp16 table, NO transpose — the score_mod
        indexes table[row, h] (per-element row gather, scalar h)."""

        table = self.bias_tables[block].to(torch.float16)
        return _FlexBias(coords, mask, table, self._cell_bias_lut, self._cell_bias_M)

    def _build_train_flex_bias(
        self, coords: torch.Tensor, mask: torch.Tensor, block: int
    ) -> "_FlexBias":
        """Train-flex (GRAD-enabled) equivalent of build_attn_bias for attention
        block `block`. Identical to _build_flex_bias EXCEPT it passes the FP32
        MASTER table self.bias_tables[block] directly (NO .to(fp16) cast): the
        score_mod's `table[row, h].to(score.dtype)` read is differentiable, so the
        table gradient flows back through the flex backward and accumulates in fp32
        — same fp32-master rationale as build_attn_bias's _BiasGather training
        branch (a fp16 table would let the hot far/ring rows' ~19M-element grad
        overflow to inf under AMP+GradScaler). coords/mask/lut are non-learned
        inputs, so no autograd subtlety there.

        The score_mod itself is reused UNCHANGED (make_score_mod): it indexes
        table[row, h] and casts to score.dtype, which is autocast-safe and keeps a
        fp32 master while the matmul runs in the autocast compute dtype. No
        (B, heads, S, S) tensor is materialized; block_mask stays None (pad mask
        folded into the score). Empirically grad-matches materialized SDPA to ~1e-6
        in fp32 incl the bias-table grad, finite under AMP+GradScaler."""

        table = self.bias_tables[block]  # fp32 master — NOT cast to fp16 (see above)
        return _FlexBias(coords, mask, table, self._cell_bias_lut, self._cell_bias_M)

    # --- forward ---------------------------------------------------------------

    def trunk(
        self,
        feats: torch.Tensor,
        nbr: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (cells (B,Npad,C), tokens (B,8,C), gather_idx) after
        LN_final."""

        b, n, _ = feats.shape
        self_idx = torch.arange(n, device=feats.device).reshape(1, n, 1).expand(b, -1, -1)
        gather_idx = torch.cat([self_idx, nbr], dim=2)  # (B, Npad, 7), tap 0 = self

        x = F.relu(self.stem_ln(self.stem(feats, gather_idx, mask))) * mask.unsqueeze(-1)
        # FlexAttention bias path is chosen per execution mode:
        #   serve_flex — no-grad serve forward + HEXFIELD_SERVE_FLEX (fp16 carrier);
        #   train_flex — grad-enabled forward + HEXFIELD_TRAIN_FLEX (fp32 master
        #                carrier so the bias-table grad accumulates in fp32).
        # Either takes the in-kernel score_mod (no (B,heads,S,S) materialization);
        # when NEITHER is active, the materialized build_attn_bias (+ _BiasGather)
        # branch runs UNCHANGED — and with both flags OFF that is the only path, so
        # default behavior is bit-for-bit today's. PER-BLOCK BIAS: each attention
        # block uses its OWN bias_tables[i]. The pair/key_pad index machinery is
        # block-INDEPENDENT and built ONCE here; only the per-block (B,heads,S,S)
        # materialization (or flex score_mod repacking) differs across blocks.
        # block_bias(i) returns the right object.
        grad_on = torch.is_grad_enabled()
        serve_flex = self._serve_flex and not grad_on
        train_flex = self._train_flex and grad_on
        flex = serve_flex or train_flex
        if not flex:
            pair, key_pad = self._build_pair(coords, mask)

        def block_bias(i: int):
            if train_flex:
                return self._build_train_flex_bias(coords, mask, i)
            if serve_flex:
                return self._build_flex_bias(coords, mask, i)
            return self.build_attn_bias(pair, key_pad, i)

        seq_mask = torch.cat([mask.new_ones(b, NUM_TOKENS), mask], dim=1)

        tokens = self.tokens.unsqueeze(0).expand(b, -1, -1)
        x = self.conv_blocks[0](x, gather_idx, mask)
        x = self.conv_blocks[1](x, gather_idx, mask)
        x = self.conv_blocks[2](x, gather_idx, mask)
        seq = self.attn_blocks[0](torch.cat([tokens, x], dim=1), block_bias(0), seq_mask)
        tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
        x = self.conv_blocks[3](x, gather_idx, mask)
        x = self.conv_blocks[4](x, gather_idx, mask)
        x = self.conv_blocks[5](x, gather_idx, mask)
        seq = self.attn_blocks[1](torch.cat([tokens, x], dim=1), block_bias(1), seq_mask)
        tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
        x = self.conv_blocks[6](x, gather_idx, mask)
        x = self.conv_blocks[7](x, gather_idx, mask)
        seq = self.attn_blocks[2](torch.cat([tokens, x], dim=1), block_bias(2), seq_mask)
        seq = self.ln_final(seq)
        tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
        return x * mask.unsqueeze(-1), tokens, gather_idx

    def _pooled(self, cells: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean of LN_final cell vectors (pad rows excluded)."""

        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(cells.dtype)
        return (cells * mask.unsqueeze(-1)).sum(dim=1) / counts

    def forward(
        self,
        feats: torch.Tensor,
        nbr: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cells, tokens, gather_idx = self.trunk(feats, nbr, mask, coords)
        pooled = self._pooled(cells, mask)
        out = {
            "policy": self._policy_logits(
                self.policy_conv, self.policy_head, cells, gather_idx, mask
            ),
            "opp_policy": self._policy_logits(
                self.opp_policy_conv, self.opp_policy_head, cells, gather_idx, mask
            ),
            "value": self.value_head(
                F.relu(self.value_reduction(self._value_input(tokens, 0, 1, pooled)))
            ),
        }
        out["cell_q"] = self._cell_q_logits(
            self.cell_q_conv, self.cell_q_head, cells, gather_idx, mask
        )
        aux = F.relu(self.aux_reduction(self._value_input(tokens, 2, 3, pooled)))
        for horizon, head in self.stv_heads.items():
            out[f"stvalue_{horizon}"] = head(aux)
        ml = F.relu(self.ml_reduction(self._value_input(tokens, 4, 5, pooled)))
        out["moves_left"] = self.moves_left_head(ml)
        return out

    def forward_policy_value(
        self,
        feats: torch.Tensor,
        nbr: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        *,
        request_moves_left: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Serve forward: policy + value always; the aux reduction +
        moves-left top only when requested; opp-policy never (train-only)."""

        cells, tokens, gather_idx = self.trunk(feats, nbr, mask, coords)
        pooled = self._pooled(cells, mask)
        out = {
            "policy": self._policy_logits(
                self.policy_conv, self.policy_head, cells, gather_idx, mask
            ),
            "value": self.value_head(
                F.relu(self.value_reduction(self._value_input(tokens, 0, 1, pooled)))
            ),
        }
        if request_moves_left:
            ml = F.relu(self.ml_reduction(self._value_input(tokens, 4, 5, pooled)))
            out["moves_left"] = self.moves_left_head(ml)
        return out

    @staticmethod
    def _value_input(
        tokens: torch.Tensor, a: int, b: int, pooled: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([tokens[:, a], tokens[:, b], pooled], dim=1)

    @staticmethod
    def _policy_logits(
        conv: HexNodeConv,
        head: nn.Linear,
        cells: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        y = F.relu(conv(cells, gather_idx, mask))
        return head(y).squeeze(-1) * mask  # (B, Npad); legality is structural

    @staticmethod
    def _cell_q_logits(
        conv: HexNodeConv,
        head: nn.Linear,
        cells: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        y = F.relu(conv(cells, gather_idx, mask))
        return head(y) * mask.unsqueeze(-1)  # (B, Npad, VALUE_BINS); pad rows zeroed
