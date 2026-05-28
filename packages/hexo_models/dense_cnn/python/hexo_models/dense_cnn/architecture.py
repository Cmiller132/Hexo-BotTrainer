"""Model 1 hex convolutional network."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .constants import (
    BOARD_AREA,
    BOARD_SIZE,
    DEFAULT_BLOCKS,
    DEFAULT_CHANNELS,
    INPUT_CHANNELS,
    VALUE_BINS,
)


class HexConv2d(nn.Conv2d):
    """3x3 convolution with the invalid square-grid hex corners masked out."""

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


class GatedResBlock(nn.Module):
    """Residual block with a sigmoid gate applied to the main branch."""

    def __init__(self, channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.main = nn.Sequential(
            HexConv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            HexConv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.gate = nn.Sequential(
            HexConv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        return residual + self.main(x) * self.gate(residual)


class PolicyHead(nn.Module):
    """Dense crop policy logits over crop cells."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(2 * BOARD_AREA, BOARD_AREA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.conv(x).flatten(start_dim=1))


class ValueBinnedHead(nn.Module):
    """65-bin KataGo-style value head."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(BOARD_AREA, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, VALUE_BINS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.conv(x).flatten(start_dim=1))


class Model1Network(nn.Module):
    """Model 1 trunk and heads exactly matching the training target surface."""

    def __init__(
        self,
        *,
        in_channels: int = INPUT_CHANNELS,
        channels: int = DEFAULT_CHANNELS,
        blocks: int = DEFAULT_BLOCKS,
        dropout: float = 0.0,
        lookahead_horizons: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.channels = int(channels)
        self.blocks_count = int(blocks)
        self.board_size = BOARD_SIZE
        self.lookahead_horizons = tuple(int(item) for item in lookahead_horizons)

        self.conv_in = HexConv2d(self.in_channels, self.channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(
            *[GatedResBlock(self.channels, dropout=dropout) for _ in range(self.blocks_count)]
        )
        self.policy_head = PolicyHead(self.channels)
        self.value_head = ValueBinnedHead(self.channels)
        self.opp_policy_head = PolicyHead(self.channels)
        self.lookahead_heads = nn.ModuleDict(
            {str(horizon): ValueBinnedHead(self.channels) for horizon in self.lookahead_horizons}
        )

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return self.blocks(self.activation(self.conv_in(x)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(x)
        outputs = {
            "policy": self.policy_head(features),
            "value": self.value_head(features),
            "opp_policy": self.opp_policy_head(features),
        }
        for horizon, head in self.lookahead_heads.items():
            outputs[f"lookahead_{horizon}"] = head(features)
        return outputs

    def forward_policy_value(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference-only forward path for search."""

        features = self.trunk(x)
        return {
            "policy": self.policy_head(features),
            "value": self.value_head(features),
        }

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Model1Network input must be rank 4, got shape {tuple(x.shape)}")
        if x.shape[1:] != (self.in_channels, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                "Model1Network input shape after batch must be "
                f"({self.in_channels}, {BOARD_SIZE}, {BOARD_SIZE}), got {tuple(x.shape[1:])}"
            )


def optimized_model1_for_inference(model: nn.Module) -> nn.Module:
    """Return a cloned eval-only model with HexConv/BatchNorm overhead folded away."""

    optimized = copy.deepcopy(model).to("cpu").eval()
    for module in optimized.modules():
        if isinstance(module, GatedResBlock):
            _fuse_gated_res_block(module)
    _replace_remaining_hex_convs(optimized)
    optimized.eval()
    return optimized


def _fuse_gated_res_block(block: GatedResBlock) -> None:
    main = block.main
    gate = block.gate
    if (
        isinstance(main, nn.Sequential)
        and len(main) >= 5
        and isinstance(main[0], HexConv2d)
        and isinstance(main[1], nn.BatchNorm2d)
        and isinstance(main[3], HexConv2d)
        and isinstance(main[4], nn.BatchNorm2d)
    ):
        main[0] = fuse_conv_bn_eval(_hex_conv_as_conv2d(main[0]).eval(), main[1].eval())
        main[1] = nn.Identity()
        main[3] = fuse_conv_bn_eval(_hex_conv_as_conv2d(main[3]).eval(), main[4].eval())
        main[4] = nn.Identity()
    if (
        isinstance(gate, nn.Sequential)
        and len(gate) >= 2
        and isinstance(gate[0], HexConv2d)
        and isinstance(gate[1], nn.BatchNorm2d)
    ):
        gate[0] = fuse_conv_bn_eval(_hex_conv_as_conv2d(gate[0]).eval(), gate[1].eval())
        gate[1] = nn.Identity()


def _replace_remaining_hex_convs(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, HexConv2d):
            _set_child_module(module, name, _hex_conv_as_conv2d(child))
        else:
            _replace_remaining_hex_convs(child)


def _set_child_module(parent: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(parent, nn.Sequential):
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def _hex_conv_as_conv2d(conv: HexConv2d) -> nn.Conv2d:
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
    converted.weight.data.copy_(conv.masked_weight().detach())
    if conv.bias is not None:
        converted.bias.data.copy_(conv.bias.detach())
    return converted
