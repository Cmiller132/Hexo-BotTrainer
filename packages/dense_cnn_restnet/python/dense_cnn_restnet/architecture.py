"""PyTorch architecture for dense_cnn_restnet — a faithful ResTNet trunk.

This is a fork of the dense_cnn `architecture.py`. It keeps the dense_cnn input
contract (the 13-plane 41x41 crop from `input.py` / `rust/src/encoding.rs`) and
the dense_cnn search surface (`forward_policy_value`: `policy` + `value`), with
this training head surface:

- `policy`: logits over the 41x41 dense crop.
- `value`: a 65-bin scalar value distribution in `[-1, 1]`.
- `opp_policy`: an auxiliary policy target from the next opponent MCTS policy.
- `stvalue_<horizon>`: optional short-term value heads.
- `moves_left`: optional auxiliary head over remaining decisions, binned onto
  the same 65-bin support via `[0, MOVES_LEFT_CAP] -> [-1, 1]`.

All value-style heads share one `ValueReduction` (conv1x1 -> Linear 1681->64);
each head is a 64->65 Linear top.

What differs from dense_cnn is the TRUNK. Instead of a homogeneous stack of
gated residual blocks, the trunk interleaves Residual (R) and Transformer (T)
blocks per the ResTNet paper (Ju, Wu, Shih, Wu, "Bridging Local and Global
Knowledge via Transformer in Board Games", IJCAI 2025, arXiv:2410.05347). The
implementation mirrors the authors' released code
(github.com/rlglab/restnet, `restnet/learner/network/block_unit.py`):

- Trunk = a `blocks_type` string split on `_`, each token `R` -> `ResidualBlock`
  or `T` -> `TransformerBlock` (e.g. RRTRRT = "R_R_T_R_R_T").
- `ResidualBlock`: plain post-activation AlphaZero block
  (Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> add input -> ReLU). NOT gated. The
  3x3 conv is hex-masked (`HexConv2d`) by default for hex-board correctness; set
  `residual_conv="plain"` for a literal `nn.Conv2d` paper match.
- `TransformerBlock`: pre-norm standard Transformer
  (x = x + MSA(LN(x)); x = x + MLP(LN(x))), GELU MLP at ratio 2, 4 heads, token
  embedding dim == channel count (no projection at the R<->T boundary).
- Attention (`RelPosMHSA`): full O(N^2) multi-head self-attention over all
  41x41 = 1681 board-cell tokens, with a learned relative-position bias table
  shaped ((2H-1)(2W-1), heads), scale 1/sqrt(C/heads). Performance is an explicit
  non-goal: the attention is faithful full attention, not windowed/downsampled.
  Two numerically-identical implementations are provided:
    * impl="materialized" — the literal MSA_rel (the correctness oracle, tested).
    * impl="sdpa" (default) — `F.scaled_dot_product_attention` with the relative
      bias passed as an additive mask. Same math, but never materializes the
      (B, heads, N, N) score matrix, so it fits training in GPU memory. NOT a
      windowing/downsampling compromise.
- Stem (`EmbedNet` semantics): Conv(embed_kernel_size) -> BN -> ReLU.
- Init: `trunc_normal_(std=0.02)` on every Linear; LayerNorm/BatchNorm bias 0,
  weight 1 (ViT-style, as in the released code).

The R<->T boundary is a pure reshape with no learned parameters: a conv feature
map `(B, C, H, W)` flattens to `(B, H*W, C)` tokens (row-major) and back. Every
block accepts either form and reshapes itself; `trunk()` guarantees a
`(B, C, 41, 41)` feature map for the dense_cnn heads.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .constants import (
    BOARD_AREA,
    BOARD_SIZE,
    INPUT_CHANNELS,
    PLANE_LEGAL,
    PLANE_OPPONENT_STONES,
    PLANE_OWN_STONES,
    VALUE_BINS,
)

DEFAULT_CHANNELS = 96
DEFAULT_BLOCKS_TYPE = "R_R_R_T_R_R_T_R"  # 8-block R3-family default (see parse_blocks_type)
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_MLP_RATIO = 2
DEFAULT_EMBED_KERNEL_SIZE = 3

# Attention scope (Spec C / D): "full" attends over all 1681 tokens (the original
# behavior, bit-for-bit), "disk" statically masks the 420 corner KEY columns so
# attention never spends mass on out-of-crop cells, and "content" (Spec D) gathers
# only the relevant tokens (occupied ∪ legal ∩ disk) into a true O(K²) attention.
ATTENTION_SCOPES = ("full", "disk", "content")
DEFAULT_ATTENTION_SCOPE = "disk"
# Additive mask added to non-disk KEY columns under "disk"/"content". Finite in
# fp32/bf16, but NOTE: under fp16 autocast the cast saturates -1e9 to -inf. That
# is still NaN-safe HERE because only KEY columns are masked and queries are
# unrestricted, so no attention row is ever fully masked (softmax over a
# partially -inf row is well defined; verified against the fp32 oracle). The one
# latent hazard is a FULLY-masked row — only reachable in "content" scope for a
# sample with zero relevant tokens inside a mixed batch, which real positions
# cannot produce (in-disk legal moves always exist).
_DISK_MASK_VALUE = -1.0e9
# Gathered ("content") attention pads the per-sample relevant-token count up to a
# multiple of this so the forward sees a small set of token-length buckets (cuDNN/
# kernel-shape stability), capped at the full board.
_CONTENT_BUCKET_MULTIPLE = 64


def _select_content_tokens(relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample relevant-token selection for gathered ("content") attention.

    ``relevance`` is a ``(B, N)`` bool mask (occupied ∪ legal ∩ disk). Returns
    ``(idx, valid)``:

    - ``idx`` ``(B, K_pad)`` long — each row's relevant token indices (ascending
      board order), with padding slots set to 0 (a valid index, masked out below).
    - ``valid`` ``(B, K_pad)`` bool — True for the ``K_i`` real tokens of each
      sample, False for padding.

    ``K_pad`` is the per-batch maximum relevant count, rounded up to a multiple of
    ``_CONTENT_BUCKET_MULTIPLE`` and capped at ``N``. A batch with no relevant
    tokens returns width-0 tensors (the caller treats the block as identity).
    """

    b, n = relevance.shape
    counts = relevance.sum(dim=1)  # (B,)
    k_max = int(counts.max().item()) if b > 0 else 0
    if k_max <= 0:
        return (
            relevance.new_zeros((b, 0), dtype=torch.long),
            relevance.new_zeros((b, 0), dtype=torch.bool),
        )
    k_pad = min(n, ((k_max + _CONTENT_BUCKET_MULTIPLE - 1) // _CONTENT_BUCKET_MULTIPLE) * _CONTENT_BUCKET_MULTIPLE)
    # A stable descending sort of the 0/1 mask lists every relevant index first, in
    # ascending board order (stability preserves the original order within ties).
    order = torch.argsort(relevance.to(torch.int8), dim=1, descending=True, stable=True)  # (B, N)
    idx = order[:, :k_pad].contiguous()  # (B, K_pad)
    valid = torch.arange(k_pad, device=relevance.device).unsqueeze(0) < counts.unsqueeze(1)  # (B, K_pad)
    idx = torch.where(valid, idx, torch.zeros_like(idx))  # padding slots -> index 0 (masked out)
    return idx, valid


def parse_blocks_type(blocks_type: str) -> tuple[str, ...]:
    """Parse a ResTNet `blocks_type` string into a tuple of block kinds.

    The string is a `_`-separated sequence of single-letter block kinds, exactly
    as in the released code (`alphazero_network.py`: `blocks_type.split("_")`):

    - ``"R"`` -> a residual block.
    - ``"T"`` -> a transformer block.

    Examples: ``"R_R_T_R_R_T"`` is RRTRRT (6 blocks, 2 T); the paper's canonical
    10-block ``R3(RRT)`` is ``"R_R_R_T_R_R_T_R_R_T"``. Rejects empty strings and
    unknown kinds (fail-fast, matching the project's config convention).
    """

    if not isinstance(blocks_type, str) or not blocks_type.strip():
        raise ValueError("blocks_type must be a non-empty string like 'R_R_T_R_R_T'")
    kinds = tuple(token for token in blocks_type.split("_"))
    if any(kind not in ("R", "T") for kind in kinds):
        bad = sorted({kind for kind in kinds if kind not in ("R", "T")})
        raise ValueError(f"blocks_type contains unsupported block kind(s): {bad!r}; only 'R' and 'T' are allowed")
    if not kinds:
        raise ValueError("blocks_type must contain at least one block")
    return kinds


class HexConv2d(nn.Conv2d):
    """3x3 convolution with invalid square-grid hex corners masked out.

    The dense crop is stored as a square tensor for GPU efficiency, but axial
    hex adjacency has six neighbors instead of eight. Masking `(0, 0)` and
    `(2, 2)` in every 3x3 kernel makes the local receptive field match the
    axial directions used by the engine. (Copied verbatim from dense_cnn.)
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if self.kernel_size != (3, 3):
            raise ValueError("HexConv2d requires kernel_size=3")
        mask = torch.ones_like(self.weight)
        mask[:, :, 0, 0] = 0.0
        mask[:, :, 2, 2] = 0.0
        self.register_buffer("hex_mask", mask, persistent=False)
        self._cached_inference_weight: torch.Tensor | None = None
        self._cached_inference_weight_version: int | None = None
        self._cached_inference_weight_device: torch.device | None = None
        self._cached_inference_weight_dtype: torch.dtype | None = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.masked_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def masked_weight(self) -> torch.Tensor:
        """Return the masked kernel, caching it only for eval/no-grad use."""

        if self.training or torch.is_grad_enabled():
            return self.weight * self.hex_mask
        version = int(getattr(self.weight, "_version", 0))
        if (
            self._cached_inference_weight is None
            or self._cached_inference_weight_version != version
            or self._cached_inference_weight_device != self.weight.device
            or self._cached_inference_weight_dtype != self.weight.dtype
        ):
            self._cached_inference_weight = (self.weight * self.hex_mask).detach()
            self._cached_inference_weight_version = version
            self._cached_inference_weight_device = self.weight.device
            self._cached_inference_weight_dtype = self.weight.dtype
        return self._cached_inference_weight


def _make_conv3x3(channels: int, residual_conv: str) -> nn.Conv2d:
    """A 3x3, padding-1, bias-free conv: hex-masked (default) or plain."""

    if residual_conv == "hex":
        return HexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
    if residual_conv == "plain":
        return nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
    raise ValueError(f"residual_conv must be 'hex' or 'plain', got {residual_conv!r}")


def conv_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C) tokens in row-major board order."""

    return x.flatten(2).transpose(1, 2)


def tokens_to_conv(x: torch.Tensor, channels: int, height: int, width: int) -> torch.Tensor:
    """(B, H*W, C) tokens -> (B, C, H, W) feature map (inverse of conv_to_tokens)."""

    return x.transpose(1, 2).reshape(x.shape[0], channels, height, width)


class ResidualBlock(nn.Module):
    """Plain post-activation AlphaZero residual block (ResTNet `R`).

    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (add input) -> ReLU. Mirrors the
    released code's `ResidualBlock`. Accepts a conv map or a token sequence (it
    reshapes tokens back to a conv map first), and outputs a conv map.
    """

    def __init__(self, channels: int, *, residual_conv: str = "hex") -> None:
        super().__init__()
        self.channels = int(channels)
        self.conv1 = _make_conv3x3(channels, residual_conv)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = _make_conv3x3(channels, residual_conv)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = tokens_to_conv(x, self.channels, BOARD_SIZE, BOARD_SIZE)
        identity = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(identity + x, inplace=True)


class RelPosMHSA(nn.Module):
    """Full O(N^2) multi-head self-attention with a learned relative-position bias.

    Faithful to the released `MSA_rel`: 4 heads (configurable), embedding dim ==
    channel count, scale `1/sqrt(C/heads)`, and a learned bias table of shape
    `((2H-1)(2W-1), heads)` indexed by the 2D (drow, dcol) offset between tokens
    (Shaw 2018-style, Swin-indexed). For the square 41x41 board this is exactly
    the released construction.

    Two numerically-identical forward paths share the same parameters:

    - ``impl="materialized"``: builds the `(B, heads, N, N)` score matrix
      explicitly (the correctness oracle; memory-heavy at N=1681).
    - ``impl="sdpa"`` (default): `F.scaled_dot_product_attention` with the
      relative bias as an additive `attn_mask`. SDPA's default scale is
      `1/sqrt(head_dim)`, which equals the paper's `1/sqrt(C/heads)`, and the
      additive mask reproduces ``dots = q@k^T * scale + bias`` exactly — same
      math, no materialized N^2 score matrix. This is NOT a
      windowed/local/downsampled approximation.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        board_h: int,
        board_w: int,
        *,
        dropout: float = 0.0,
        impl: str = "sdpa",
        attention_scope: str = DEFAULT_ATTENTION_SCOPE,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        if board_h != board_w:
            # The relative-index scheme below assumes a square grid (as does the
            # released code, which reuses `H-1` for both axes). The dense crop is
            # 41x41, so this always holds here.
            raise ValueError(f"RelPosMHSA requires a square token grid, got {board_h}x{board_w}")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = channels // num_heads
        self.board_h = int(board_h)
        self.board_w = int(board_w)
        self.token_len = board_h * board_w
        self.scaling = self.head_dim ** -0.5
        self.set_impl(impl)
        self.set_attention_scope(attention_scope)

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.out_drop = nn.Dropout(dropout)

        # Learned relative-position bias table, ((2H-1)(2W-1), heads).
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * board_h - 1) * (2 * board_w - 1), num_heads)
        )
        # Precompute the flat offset index for every (query, key) token pair, in
        # the same row-major order as conv_to_tokens (token t -> row t//W, col t%W).
        rows = torch.arange(board_h).repeat_interleave(board_w)  # (N,)
        cols = torch.arange(board_w).repeat(board_h)  # (N,)
        rel_rows = rows[:, None] - rows[None, :] + (board_h - 1)  # (N, N) in [0, 2H-2]
        rel_cols = cols[:, None] - cols[None, :] + (board_w - 1)  # (N, N) in [0, 2W-2]
        relative_index = rel_rows * (2 * board_w - 1) + rel_cols  # (N, N) in [0, (2H-1)(2W-1)-1]
        self.register_buffer("relative_index", relative_index.reshape(-1), persistent=False)
        # Static radius-(H//2) hex-disk membership for every KEY token, from the SAME
        # (row, col) grids as relative_index (Spec C). The 420 corner cells are
        # out-of-crop, so under "disk"/"content" scope a large negative additive bias
        # masks their KEY columns. persistent=False -> NOT in state_dict, so this is
        # checkpoint-compatible and adds no params. Queries are never masked.
        half = board_h // 2
        drow = rows - half
        dcol = cols - half
        disk_flat = torch.maximum(torch.maximum(drow.abs(), dcol.abs()), (drow + dcol).abs()) <= half  # (N,) bool
        self.register_buffer("disk_flat", disk_flat, persistent=False)
        disk_key_bias = torch.zeros(self.token_len, dtype=torch.float32)
        disk_key_bias[~disk_flat] = _DISK_MASK_VALUE  # additive -1e9 on corner KEY columns
        self.register_buffer("disk_key_bias", disk_key_bias, persistent=False)
        # Performance-only inference cache for the additive bias mask (B-speedup):
        # the (1, heads, N, N) mask is a pure function of relative_bias_table, so
        # under eval/no-grad (self-play, frozen weights) it is gathered ONCE and
        # reused across forwards instead of re-gathering ~N*N*heads elements every
        # call. Version-keyed (like HexConv2d.masked_weight) so it auto-invalidates
        # the instant the table changes. Plain attributes -> NOT in state_dict, so
        # this is bit-identical and checkpoint-compatible. Training (grad enabled)
        # always recomputes, so gradients to the table are unaffected.
        self._cached_bias: torch.Tensor | None = None
        self._cached_bias_key: tuple | None = None

    def set_impl(self, impl: str) -> None:
        if impl not in ("sdpa", "materialized"):
            raise ValueError(f"attention impl must be 'sdpa' or 'materialized', got {impl!r}")
        self.impl = impl

    def set_attention_scope(self, scope: str) -> None:
        if scope not in ATTENTION_SCOPES:
            raise ValueError(f"attention_scope must be one of {ATTENTION_SCOPES}, got {scope!r}")
        self.attention_scope = scope
        # The cached eval bias depends on the scope's key mask, so invalidate it.
        self._cached_bias = None
        self._cached_bias_key = None

    def _compute_relative_bias(self) -> torch.Tensor:
        """Gather the additive bias `(1, heads, N, N)` fresh from the learned table.

        Under "disk"/"content" scope the corner KEY columns receive a large negative
        additive bias (``disk_key_bias``) so attention never spends mass on
        out-of-crop cells; "full" returns the unmasked bias bit-for-bit.
        """

        bias = self.relative_bias_table[self.relative_index]  # (N*N, heads)
        bias = bias.view(self.token_len, self.token_len, self.num_heads)  # (N, N, heads): [query, key, head]
        out = bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N): [batch, head, query, key]
        if self.attention_scope in ("disk", "content"):
            out = out + self.disk_key_bias.view(1, 1, 1, self.token_len)  # mask corner KEY columns
        return out

    def _relative_bias(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Additive bias `(1, heads, N, N)` in the consumer's dtype/device.

        Eval + no-grad reuses a cached, contiguous copy (gathered once); training
        recomputes so the table still receives gradient. Numerically identical.
        """

        if self.training or torch.is_grad_enabled():
            return self._compute_relative_bias().to(device=device, dtype=dtype)
        version = int(getattr(self.relative_bias_table, "_version", 0))
        key = (version, dtype, device)
        if self._cached_bias is None or self._cached_bias_key != key:
            self._cached_bias = (
                self._compute_relative_bias().to(device=device, dtype=dtype).detach().contiguous()
            )
            self._cached_bias_key = key
        return self._cached_bias

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        shape = (b, n, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).permute(0, 2, 1, 3)  # (B, heads, N, head_dim)
        k = self.k_proj(x).view(shape).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(shape).permute(0, 2, 1, 3)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl == "sdpa":
            return self._forward_sdpa(x)
        return self._forward_materialized(x)

    def _forward_materialized(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self._project(x)
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # (B, heads, N, N)
        dots = dots + self._relative_bias(dtype=dots.dtype, device=dots.device)
        attn = torch.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.channels)
        return self.out_drop(self.out_proj(out))

    def _forward_sdpa(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self._project(x)
        attn_mask = self._relative_bias(dtype=q.dtype, device=q.device)  # additive bias; SDPA scale = 1/sqrt(head_dim)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.channels)
        return self.out_drop(self.out_proj(out))

    def _attend_with_bias(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Project + attend `x` (B, L, C) with an explicit additive `bias`.

        ``bias`` is broadcastable to ``(B, heads, L, L)``. Respects ``self.impl``
        ('sdpa' or 'materialized') with the same math as the dense paths, but over
        an arbitrary token length L and per-sample bias (used by gathered attention).
        """

        b, length, _ = x.shape
        q, k, v = self._project(x)  # (B, heads, L, head_dim)
        if self.impl == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(dtype=q.dtype))
        else:
            dots = torch.matmul(q, k.transpose(-2, -1)) * self.scaling + bias.to(dtype=q.dtype)
            out = torch.matmul(torch.softmax(dots, dim=-1), v)
        out = out.permute(0, 2, 1, 3).reshape(b, length, self.channels)
        return self.out_drop(self.out_proj(out))

    def forward_gathered(self, tok: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Gathered O(K²) attention over `tok` (B, K_pad, C) for "content" scope.

        ``idx``/``valid`` come from :func:`_select_content_tokens`. The per-sample
        ``(K_pad, K_pad, heads)`` relative-bias submatrix is gathered from the SAME
        learned table/index as the dense path, so for a relevant query attending to
        the relevant keys the bias is identical to dense attention — only the
        non-relevant keys are absent (gathered out) instead of statically masked.
        Padded KEY columns get the large-negative mask; queries are unrestricted.
        """

        b, k_pad, _ = tok.shape
        rel2d = self.relative_index.view(self.token_len, self.token_len)  # (N, N)
        rows = rel2d[idx]  # (B, K_pad, N): rows for each gathered query token
        sub = torch.gather(rows, 2, idx.unsqueeze(1).expand(b, k_pad, k_pad))  # (B, K_pad, K_pad)
        bias = self.relative_bias_table[sub]  # (B, K_pad, K_pad, heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, heads, K_pad, K_pad): [batch, head, query, key]
        # Mask padded KEY columns. (Disk masking is implicit: relevance ⊆ disk, so
        # every gathered token is already in-disk.)
        key_pad = (~valid).view(b, 1, 1, k_pad).to(bias.dtype) * _DISK_MASK_VALUE
        bias = bias + key_pad
        return self._attend_with_bias(tok, bias)


class TransformerBlock(nn.Module):
    """Pre-norm standard Transformer block (ResTNet `T`).

    `x = x + MSA(LN(x)); x = x + MLP(LN(x))`, GELU MLP at `mlp_ratio` (default 2),
    LayerNorm, no dropout by default. Accepts a conv map or tokens; OUTPUTS tokens
    `(B, H*W, C)` (the next residual block reshapes back, matching the released
    code). Token embedding dim == channel count, so the R<->T boundary needs no
    projection.
    """

    def __init__(
        self,
        channels: int,
        *,
        heads: int = DEFAULT_ATTENTION_HEADS,
        mlp_ratio: int = DEFAULT_MLP_RATIO,
        board_h: int = BOARD_SIZE,
        board_w: int = BOARD_SIZE,
        dropout: float = 0.0,
        attention_impl: str = "sdpa",
        attention_scope: str = DEFAULT_ATTENTION_SCOPE,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.attention_scope = str(attention_scope)
        self.ln1 = nn.LayerNorm(channels)
        self.attn = RelPosMHSA(
            channels, heads, board_h, board_w, dropout=dropout, impl=attention_impl, attention_scope=attention_scope
        )
        self.ln2 = nn.LayerNorm(channels)
        hidden = int(mlp_ratio) * int(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor, content_sel: tuple | None = None) -> torch.Tensor:
        if x.dim() == 4:
            x = conv_to_tokens(x)
        if self.attention_scope == "content" and content_sel is not None:
            return self._forward_content(x, content_sel)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def _forward_content(self, x: torch.Tensor, content_sel: tuple) -> torch.Tensor:
        """Gathered-token transformer block (Spec D): attention + MLP over only the
        relevant tokens, scattered back so non-relevant cells pass through unchanged.
        """

        relevance, idx, valid = content_sel  # (B, N) bool, (B, K_pad) long, (B, K_pad) bool
        b, n, c = x.shape
        if idx.shape[1] == 0:
            return x  # no relevant tokens in the whole batch -> block is identity
        idx_exp = idx.unsqueeze(-1).expand(b, idx.shape[1], c)  # (B, K_pad, C)
        tok = x.gather(1, idx_exp)  # (B, K_pad, C); padding slots duplicate token 0
        tok = tok + self.attn.forward_gathered(self.ln1(tok), idx, valid)
        tok = tok + self.mlp(self.ln2(tok))
        # Scatter the updated tokens back to their board positions. Padded slots are
        # routed to a throwaway row (index n) and dropped; non-relevant cells keep x.
        safe_idx = torch.where(valid, idx, torch.full_like(idx, n))  # invalid -> dump row
        scattered = x.new_zeros((b, n + 1, c)).scatter(
            1, safe_idx.unsqueeze(-1).expand(b, idx.shape[1], c), tok
        )[:, :n, :]
        return torch.where(relevance.unsqueeze(-1), scattered, x)


class PolicyHead(nn.Module):
    """Fully-convolutional dense crop policy head — one logit per crop cell (P7).

    Copied verbatim from dense_cnn. The training target `policyTargetsNCHW` is
    spatial `(N, 1, 41, 41)`, so a fully-conv head produces it directly; flattening
    to `(N, BOARD_AREA)` keeps the exact downstream contract (cross-entropy target
    shape, flat-crop-index gather in inference / `mcts_eval.rs`).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = HexConv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, 41, 41) -> (N, 1, 41, 41) -> (N, BOARD_AREA)
        return self.head(self.activation(self.conv(x))).flatten(start_dim=1)


class ValueReduction(nn.Module):
    """Shared spatial reduction for every value-style head.

    Conv1x1(C->1) -> ReLU -> flatten -> Linear(BOARD_AREA->64) -> ReLU, producing
    one 64-d value embedding consumed by all the scalar-distribution tops (value,
    stvalue_<h>, moves_left). This replaces the per-head copies of the same
    reduction (4 x ~112k params -> 1 x ~108k + ~4k per top): the dominant
    1681->64 Linear is supervised by every value-style target at once instead of
    four private ones. Composition (conv+ReLU -> Linear -> ReLU -> top Linear) is
    identical to the old per-head ValueBinnedHead, so old checkpoints migrate
    function-preservingly for the main value head (see
    scripts/_restnet_migrate_heads_v2.py). The spatial Linear (vs global pooling)
    is deliberate: the crop is center-relative, so absolute crop position is
    meaningful.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(BOARD_AREA, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.conv(x).flatten(start_dim=1))


class RestnetNetwork(nn.Module):
    """ResTNet trunk + dense_cnn heads, matching the dense_cnn training surface.

    `forward` returns all training heads (`policy`, `value`, `opp_policy`, and
    `stvalue_<h>`). `forward_policy_value` returns just the two outputs Rust needs
    for MCTS search. The trunk interleaves residual and transformer blocks per
    `blocks_type` and always returns a `(B, C, 41, 41)` feature map for the heads
    (a token->conv reshape is applied if the last trunk block is a transformer
    block, which emits tokens).
    """

    def __init__(
        self,
        *,
        in_channels: int = INPUT_CHANNELS,
        channels: int = DEFAULT_CHANNELS,
        blocks_type: str = DEFAULT_BLOCKS_TYPE,
        attention_heads: int = DEFAULT_ATTENTION_HEADS,
        mlp_ratio: int = DEFAULT_MLP_RATIO,
        embed_kernel_size: int = DEFAULT_EMBED_KERNEL_SIZE,
        residual_conv: str = "hex",
        attention_impl: str = "sdpa",
        attention_scope: str = DEFAULT_ATTENTION_SCOPE,
        dropout: float = 0.0,
        short_term_value_horizons: tuple[int, ...] = (),
        moves_left_head: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.channels = int(channels)
        self.board_size = BOARD_SIZE
        self.blocks_type = str(blocks_type)
        self.attention_impl = str(attention_impl)
        self.attention_scope = str(attention_scope)
        self.residual_conv = str(residual_conv)
        self.short_term_value_horizons = tuple(int(item) for item in short_term_value_horizons)
        block_kinds = parse_blocks_type(self.blocks_type)

        # Stem (EmbedNet semantics): Conv(embed_kernel_size) -> BN -> ReLU. A hex
        # mask only applies to the 3x3 case; a kernel_size=1 stem is a plain conv.
        if int(embed_kernel_size) == 3 and residual_conv == "hex":
            self.stem_conv: nn.Conv2d = HexConv2d(self.in_channels, self.channels, kernel_size=3, padding=1, bias=False)
        else:
            k = int(embed_kernel_size)
            self.stem_conv = nn.Conv2d(self.in_channels, self.channels, kernel_size=k, padding=k // 2, bias=False)
        self.stem_bn = nn.BatchNorm2d(self.channels)
        self.activation = nn.ReLU(inplace=True)

        blocks: list[nn.Module] = []
        for kind in block_kinds:
            if kind == "R":
                blocks.append(ResidualBlock(self.channels, residual_conv=residual_conv))
            else:
                blocks.append(
                    TransformerBlock(
                        self.channels,
                        heads=attention_heads,
                        mlp_ratio=mlp_ratio,
                        board_h=BOARD_SIZE,
                        board_w=BOARD_SIZE,
                        dropout=dropout,
                        attention_impl=attention_impl,
                        attention_scope=attention_scope,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        self.policy_head = PolicyHead(self.channels)
        self.opp_policy_head = PolicyHead(self.channels)
        # All value-style heads share one spatial reduction; each head is a
        # 64 -> VALUE_BINS top. moves_left predicts remaining decisions binned
        # over [0, MOVES_LEFT_CAP] via the same [-1, 1] support (training-only
        # auxiliary; absent from forward_policy_value).
        self.value_reduction = ValueReduction(self.channels)
        self.value_head = nn.Linear(64, VALUE_BINS)
        self.short_term_value_heads = nn.ModuleDict(
            {str(horizon): nn.Linear(64, VALUE_BINS) for horizon in self.short_term_value_horizons}
        )
        self.moves_left_head = nn.Linear(64, VALUE_BINS) if moves_left_head else None

        # Static radius-(H//2) hex-disk membership for every board cell, used by the
        # "content" scope to intersect relevance with the disk. persistent=False ->
        # NOT in state_dict (adds no params; checkpoints cross-load across scopes).
        half = BOARD_SIZE // 2
        rows = torch.arange(BOARD_SIZE).repeat_interleave(BOARD_SIZE)
        cols = torch.arange(BOARD_SIZE).repeat(BOARD_SIZE)
        drow = rows - half
        dcol = cols - half
        disk_flat = torch.maximum(torch.maximum(drow.abs(), dcol.abs()), (drow + dcol).abs()) <= half
        self.register_buffer("_disk_flat", disk_flat, persistent=False)

        # ViT-style init from the released code: trunc_normal on Linear, LN/BN
        # to (bias 0, weight 1). Conv layers keep PyTorch's default init.
        self.apply(_init_weights_trunc_normal)

    def set_attention_impl(self, impl: str) -> None:
        """Switch all transformer blocks between 'sdpa' and 'materialized'."""

        for module in self.modules():
            if isinstance(module, RelPosMHSA):
                module.set_impl(impl)
        self.attention_impl = impl

    def set_attention_scope(self, scope: str) -> None:
        """Switch all transformer blocks between 'full' / 'disk' / 'content' scope."""

        for module in self.modules():
            if isinstance(module, (RelPosMHSA, TransformerBlock)):
                module.attention_scope = scope
            if isinstance(module, RelPosMHSA):
                module.set_attention_scope(scope)
        self.attention_scope = scope

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        # "content" scope derives the relevant-token selection ONCE from the input
        # planes (before the stem) and threads it through every transformer block.
        content_sel = self._content_selection(x) if self.attention_scope == "content" else None
        x = self.activation(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                x = block(x, content_sel)
            else:
                x = block(x)
        if x.dim() == 3:
            x = tokens_to_conv(x, self.channels, BOARD_SIZE, BOARD_SIZE)
        return x

    def _content_selection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Relevant-token selection for "content" scope, from the input planes.

        Relevance = (occupied ∪ legal) ∩ disk, read from ``PLANE_OWN_STONES`` /
        ``PLANE_OPPONENT_STONES`` / ``PLANE_LEGAL`` of the raw input ``x``
        (B, INPUT_CHANNELS, 41, 41), flattened to ``(B, N)`` in row-major board
        order (the token order). Returns ``(relevance, idx, valid)``.
        """

        planes = x[:, PLANE_OWN_STONES] + x[:, PLANE_OPPONENT_STONES] + x[:, PLANE_LEGAL]  # (B, H, W)
        relevance = (planes > 0).reshape(x.shape[0], -1) & self._disk_flat  # (B, N) bool
        idx, valid = _select_content_tokens(relevance)
        return relevance, idx, valid

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(x)
        value_embedding = self.value_reduction(features)
        outputs = {
            "policy": self.policy_head(features),
            "value": self.value_head(value_embedding),
            "opp_policy": self.opp_policy_head(features),
        }
        for horizon, head in self.short_term_value_heads.items():
            outputs[f"stvalue_{horizon}"] = head(value_embedding)
        if self.moves_left_head is not None:
            outputs["moves_left"] = self.moves_left_head(value_embedding)
        return outputs

    def forward_policy_value(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference-only forward path for search."""

        features = self.trunk(x)
        return {
            "policy": self.policy_head(features),
            "value": self.value_head(self.value_reduction(features)),
        }

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"RestnetNetwork input must be rank 4, got shape {tuple(x.shape)}")
        if x.shape[1:] != (self.in_channels, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                "RestnetNetwork input shape after batch must be "
                f"({self.in_channels}, {BOARD_SIZE}, {BOARD_SIZE}), got {tuple(x.shape[1:])}"
            )


def _init_weights_trunc_normal(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)


def optimized_restnet_for_inference(model: nn.Module) -> nn.Module:
    """Return a cloned eval-only model with the residual convs/BNs folded away.

    Folds each plain `ResidualBlock` (Conv+BN -> Conv) and replaces remaining
    `HexConv2d` (stem, policy heads) with plain `nn.Conv2d` carrying the masked
    weight, exactly as the dense_cnn optimizer does for its blocks.
    `TransformerBlock` (LayerNorm + attention) has no conv/BN to fold and is left
    untouched — its eval-mode output is already exact. Outputs are numerically
    identical to the unfolded eval model.

    (Performance is an explicit non-goal for this lineage; this exists only so the
    CUDA inference path mirrors dense_cnn's and stays correct. The forward is
    invoked via `forward_policy_value` during search.)
    """

    optimized = copy.deepcopy(model).to("cpu").eval()
    for module in optimized.modules():
        if isinstance(module, ResidualBlock):
            _fuse_residual_block(module)
    _replace_remaining_hex_convs(optimized)
    optimized.eval()
    return optimized


def _fuse_residual_block(block: ResidualBlock) -> None:
    fused1 = fuse_conv_bn_eval(_conv_as_plain_conv2d(block.conv1).eval(), block.bn1.eval())
    fused2 = fuse_conv_bn_eval(_conv_as_plain_conv2d(block.conv2).eval(), block.bn2.eval())
    block.conv1 = fused1
    block.bn1 = nn.Identity()
    block.conv2 = fused2
    block.bn2 = nn.Identity()


def _replace_remaining_hex_convs(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, HexConv2d):
            _set_child_module(module, name, _conv_as_plain_conv2d(child))
        else:
            _replace_remaining_hex_convs(child)


def _set_child_module(parent: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(parent, nn.Sequential):
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def _conv_as_plain_conv2d(conv: nn.Conv2d) -> nn.Conv2d:
    """Copy a conv into a plain `nn.Conv2d`; folds the hex mask for `HexConv2d`."""

    weight = conv.masked_weight().detach() if isinstance(conv, HexConv2d) else conv.weight.detach()
    converted = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    converted.weight.data.copy_(weight)
    if conv.bias is not None:
        converted.bias.data.copy_(conv.bias.detach())
    return converted
