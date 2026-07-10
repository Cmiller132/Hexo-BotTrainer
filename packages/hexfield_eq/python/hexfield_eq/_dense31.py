"""CPU reference and recompute-backward path for the dense 31-tap conv.

Tap order is center followed by five distance shells, each in
``constants.DIRECTIONS`` order.  Visibility is delegated to the same
``_raytap._masked_gather`` helper used by the alpha ray-tap operator, keeping
the own/opp orbit-half semantics and sentinel handling identical.
"""

from __future__ import annotations

import torch

from .constants import RAY_REACH
from ._raytap import _masked_gather, _tap_flat_index, _tap_vis


def dense31_gather(
    x: torch.Tensor,
    idx_taps: torch.Tensor,
    reach: torch.Tensor,
    corb: int,
) -> torch.Tensor:
    """Return ``(B, N, 31*C)`` center + shell-major masked ray fibers."""

    b, n, c = x.shape
    x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
    reach_l = reach.to(torch.long)
    # Each item is (B,N,K,C); stacking directions after K produces
    # (B,N,K,D,C), whose contiguous flatten is shell-major then direction.
    rays = torch.stack(
        [_masked_gather(x_ext, idx_taps, reach_l, corb, d) for d in range(6)],
        dim=3,
    )
    return torch.cat([x.unsqueeze(2), rays.reshape(b, n, 30, c)], dim=2).reshape(
        b, n, 31 * c
    )


def dense31_conv_naive(
    x: torch.Tensor,
    idx_taps: torch.Tensor,
    reach: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    corb: int,
) -> torch.Tensor:
    """Plain-autograd oracle; it intentionally saves the gathered tensor."""

    gathered = dense31_gather(x, idx_taps, reach, corb)
    out = gathered @ weight.reshape(31 * x.shape[-1], weight.shape[-1]) + bias
    return out * mask.unsqueeze(-1)


class _Dense31ConvFn(torch.autograd.Function):
    """Dense31 gather + GEMM without saving the ``(B,N,31*C)`` gather.

    Backward regathers only for ``grad_weight`` and scatter-adds the ray-input
    gradient through the same hard visibility masks.  The generated tied
    weight is an input, so its upstream gather remains ordinary autograd.
    """

    @staticmethod
    def forward(ctx, x, idx_taps, reach, weight, bias, mask, corb):
        out = dense31_conv_naive(x, idx_taps, reach, weight, bias, mask, corb)
        ctx.save_for_backward(x, idx_taps, reach, weight, mask)
        ctx.corb = int(corb)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        x, idx_taps, reach, weight, mask = ctx.saved_tensors
        corb = ctx.corb
        b, n, c = x.shape
        cout = weight.shape[-1]
        need_x, _, _, need_weight, need_bias, _, _ = ctx.needs_input_grad
        go = grad_out * mask.unsqueeze(-1).to(grad_out.dtype)

        grad_weight = None
        if need_weight:
            gathered = dense31_gather(x, idx_taps, reach, corb)
            grad_weight = (
                gathered.reshape(b * n, 31 * c).transpose(0, 1)
                @ go.reshape(b * n, cout)
            ).reshape_as(weight)

        grad_bias = go.sum(dim=(0, 1)) if need_bias else None
        grad_x = None
        if need_x:
            grad_taps = (
                go.reshape(b * n, cout)
                @ weight.reshape(31 * c, cout).transpose(0, 1)
            ).reshape(b, n, 31, c)
            grad_x_ext = x.new_zeros(b, n + 1, c)
            grad_x_ext[:, :n] += grad_taps[:, :, 0]
            ray_grad = grad_taps[:, :, 1:].reshape(b, n, RAY_REACH, 6, c)
            reach_l = reach.to(torch.long)
            half = corb // 2
            groups = c // corb
            for d in range(6):
                vis = _tap_vis(reach_l, d)
                gd = (
                    ray_grad[:, :, :, d]
                    .view(b, n, RAY_REACH, groups, 2, half)
                    * vis.view(b, n, RAY_REACH, 1, 2, 1).to(x.dtype)
                ).reshape(b, n * RAY_REACH, c)
                grad_x_ext.scatter_add_(1, _tap_flat_index(idx_taps, d, c), gd)
            grad_x = grad_x_ext[:, :n]

        return grad_x, None, None, grad_weight, grad_bias, None, None


def dense31_conv(
    x: torch.Tensor,
    idx_taps: torch.Tensor,
    reach: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    corb: int,
) -> torch.Tensor:
    """Production entry point using recomputation in backward."""

    return _Dense31ConvFn.apply(x, idx_taps, reach, weight, bias, mask, corb)
