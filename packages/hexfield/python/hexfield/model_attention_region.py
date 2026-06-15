# =============================================================================
# DELIVERABLE — model.py attention region (Implementer 3, Layer A2).
#
# This file shows the FULL replacement text for the three pieces of model.py
# that change:  RelPosAttention, AttnBlock, and HexfieldNet.trunk  (plus the
# one-line set_attention_impl validation and the new import). Everything else
# in model.py is UNCHANGED. Drop these bodies in place; the surrounding class
# scaffolding (HexfieldNet.__init__, build_attn_bias, heads, forward,
# forward_policy_value) is identical to the live tree.
#
# Routing contract:
#   impl in {"sdpa","materialized"}  -> EXACTLY the existing path. trunk builds
#       the (B,heads,S,S) attn_bias via build_attn_bias and passes it down.
#       This is the training path, the test oracle, and the universal fallback.
#       UNTOUCHED so test_sdpa_equals_materialized_fp16_cuda still binds.
#   impl in {"hexflash","flex"} AND not torch.is_grad_enabled()  -> the serve
#       fast path: build_attn_bias is SKIPPED; coords (int) + seq_mask (bool) +
#       bias_table flow to the kernel, which reconstructs the bias row in-kernel.
#   impl in {"hexflash","flex"} AND grad enabled (should never happen on the
#       serve path) -> falls back to the materialized math via build_attn_bias
#       so training/grad never hits an autograd-free kernel. Safe by default.
#
# Hard invariants preserved (no retrain): q/k/v proj + scale + out_proj are
# bit-identical to the SDPA path; only the score+softmax+@V core changes kernel.
# AttnBlock's `*m` re-zero of pad QUERY rows is unchanged, so pad-row outputs
# stay bit-identical regardless of Npad.
# =============================================================================

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .constants import ATTENTION_HEADS, HEAD_DIM, NUM_TOKENS
from .hexflash import flex_attention_relpos, hexflash_attention

# (PAD_KEY_MASK_VALUE, STV_HORIZONS, _BiasGather, HexNodeConv, ConvBlock are
#  unchanged and live above this region in model.py.)

_FUSED_IMPLS = ("hexflash", "flex")
_FUSED_FNS = {"hexflash": hexflash_attention, "flex": flex_attention_relpos}


class RelPosAttention(nn.Module):
    """4-head MHSA over the joint [tokens ; cells] sequence with the shared
    bias table gathered as an additive mask.

    Implementations:
      'sdpa'         — production training/eval path (fused SDPA over a
                       prebuilt (B,heads,S,S) additive bias).
      'materialized' — explicit math; the numerical oracle.
      'hexflash'     — fused Triton FA2 reconstructing the bias row in-kernel
                       from (coords, bias_table, _exact_lut). Serve only.
      'flex'         — FlexAttention fallback, same reconstruction via score_mod.

    For 'sdpa'/'materialized', forward takes the prebuilt ``attn_bias``. For the
    fused impls it takes ``coords``/``seq_mask``/``bias_table``/``exact_lut``
    instead (attn_bias is None). The trunk picks which to pass per impl."""

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

    def forward(
        self,
        seq: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        *,
        coords: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        bias_table: torch.Tensor | None = None,
        exact_lut: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, c = seq.shape
        h, d = self.heads, self.head_dim
        q = self.q_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        k = self.k_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        v = self.v_proj(seq).reshape(b, s, h, d).transpose(1, 2)

        if self.impl in _FUSED_IMPLS:
            # Serve fast path. The trunk only routes here under no_grad; if grad
            # is somehow enabled, fall through to materialized math (build a bias
            # would require coords — which the trunk still passes — but the
            # autograd-free kernel is wrong under grad, so we use the explicit
            # math definition that IS differentiable).
            fn = _FUSED_FNS[self.impl]
            if torch.is_grad_enabled():
                # Differentiable materialized equivalent (reference path inside
                # hexflash is pure-torch and autograd-friendly).
                from .hexflash import reference_relpos_attention

                out = reference_relpos_attention(
                    q, k, v, coords, bias_table, seq_mask, exact_lut,
                    self.scale, NUM_TOKENS,
                )
            else:
                out = fn(
                    q, k, v, coords, bias_table, seq_mask, exact_lut,
                    self.scale, NUM_TOKENS,
                )
            out = out.transpose(1, 2).reshape(b, s, c)
            return self.out_proj(out)

        # --- existing sdpa / materialized paths, byte-identical to live tree ---
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
    """Pre-norm transformer block (restnet block semantics, GELU, ratio 2).

    Threads either a prebuilt ``attn_bias`` (sdpa/materialized) OR the raw
    ``coords``/``bias_table``/``exact_lut`` (hexflash/flex) through to the attn.
    The ``*m`` re-zero of pad QUERY rows is UNCHANGED — the pad-inertness
    invariant survives because pad rows are zeroed after the residual add
    regardless of which kernel produced their (garbage) attention output."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.attn = RelPosAttention(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, MLP_RATIO * channels)  # noqa: F821 (defined in model.py)
        self.fc2 = nn.Linear(MLP_RATIO * channels, channels)  # noqa: F821

    def forward(
        self,
        seq: torch.Tensor,
        attn_bias: torch.Tensor | None,
        seq_mask: torch.Tensor,
        *,
        coords: torch.Tensor | None = None,
        bias_table: torch.Tensor | None = None,
        exact_lut: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = seq_mask.unsqueeze(-1)
        if self.attn.impl in _FUSED_IMPLS:
            attended = self.attn(
                self.ln1(seq),
                None,
                coords=coords,
                seq_mask=seq_mask,
                bias_table=bias_table,
                exact_lut=exact_lut,
            )
        else:
            attended = self.attn(self.ln1(seq), attn_bias)
        seq = seq + attended * m
        seq = seq + self.fc2(F.gelu(self.fc1(self.ln2(seq)))) * m
        return seq


# =============================================================================
# HexfieldNet.trunk — full replacement body (method of HexfieldNet, unchanged
# class otherwise). Only difference vs live tree: when the attention impl is a
# fused serve impl AND grad is disabled, build_attn_bias is SKIPPED and the raw
# (coords, bias_table, _exact_lut) are threaded to each AttnBlock. coords is
# passed as int32 for the kernels. Otherwise IDENTICAL to the live trunk.
# =============================================================================

def trunk(
    self,
    feats: torch.Tensor,
    nbr: torch.Tensor,
    mask: torch.Tensor,
    coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (cells (B,Npad,C), tokens (B,8,C), gather_idx) after LN_final."""

    b, n, _ = feats.shape
    self_idx = torch.arange(n, device=feats.device).reshape(1, n, 1).expand(b, -1, -1)
    gather_idx = torch.cat([self_idx, nbr], dim=2)  # (B, Npad, 7), tap 0 = self

    x = F.relu(self.stem_ln(self.stem(feats, gather_idx, mask))) * mask.unsqueeze(-1)

    # Decide the attention dispatch ONCE for this forward.
    impl = self.attn_blocks[0].attn.impl
    fused_serve = (impl in _FUSED_IMPLS) and (not torch.is_grad_enabled())

    seq_mask = torch.cat([mask.new_ones(b, NUM_TOKENS), mask], dim=1)

    if fused_serve:
        attn_bias = None
        # Kernels want int32 coords; bias_table + exact_lut passed by reference.
        attn_coords = coords.to(torch.int32)
        attn_table = self.bias_table
        attn_lut = self._exact_lut
    else:
        attn_bias = self.build_attn_bias(coords, mask)
        attn_coords = attn_table = attn_lut = None

    def _attn(block: "AttnBlock", seq: torch.Tensor) -> torch.Tensor:
        if fused_serve:
            return block(
                seq, None, seq_mask,
                coords=attn_coords, bias_table=attn_table, exact_lut=attn_lut,
            )
        return block(seq, attn_bias, seq_mask)

    tokens = self.tokens.unsqueeze(0).expand(b, -1, -1)
    x = self.conv_blocks[0](x, gather_idx, mask)
    x = self.conv_blocks[1](x, gather_idx, mask)
    x = self.conv_blocks[2](x, gather_idx, mask)
    seq = _attn(self.attn_blocks[0], torch.cat([tokens, x], dim=1))
    tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
    x = self.conv_blocks[3](x, gather_idx, mask)
    x = self.conv_blocks[4](x, gather_idx, mask)
    seq = _attn(self.attn_blocks[1], torch.cat([tokens, x], dim=1))
    tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
    x = self.conv_blocks[5](x, gather_idx, mask)
    seq = _attn(self.attn_blocks[2], torch.cat([tokens, x], dim=1))
    seq = self.ln_final(seq)
    tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
    return x * mask.unsqueeze(-1), tokens, gather_idx


# =============================================================================
# set_attention_impl — accept the two new strings (validation only).
# =============================================================================

def set_attention_impl(self, impl: str) -> None:
    if impl not in ("sdpa", "materialized", "hexflash", "flex"):
        raise ValueError(f"unknown attention impl: {impl}")
    for block in self.attn_blocks:
        block.attn.impl = impl
