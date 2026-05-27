"""Model 1 hex convolutional network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight * self.hex_mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


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
