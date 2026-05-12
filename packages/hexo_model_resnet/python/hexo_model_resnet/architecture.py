"""Residual policy/value network for placement-level Hexo decisions."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class PolicyHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class ValueHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv(x)
        if valid_mask is None:
            pooled = h.mean(dim=(2, 3))
        else:
            mask = valid_mask.to(dtype=h.dtype).unsqueeze(1)
            denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
            pooled = (h * mask).sum(dim=(2, 3)) / denom
        return self.mlp(pooled).squeeze(-1)


class HexoNet(nn.Module):
    """Small residual CNN with policy logits over crop cells and scalar value."""

    def __init__(self, in_channels: int = 12, channels: int = 64, blocks: int = 6) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.blocks_count = blocks
        self.stem = ConvBlock(in_channels, channels)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.blocks(self.stem(x))
        policy_logits = self.policy_head(h)
        if valid_mask is not None:
            policy_logits = policy_logits.masked_fill(valid_mask <= 0, -1.0e9)
        value = self.value_head(h, valid_mask)
        return {
            "policy_logits": policy_logits,
            "value": value,
        }

