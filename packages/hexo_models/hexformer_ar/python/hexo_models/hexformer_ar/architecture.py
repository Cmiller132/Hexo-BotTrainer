"""Hybrid HexaConv + sparse relative-attention model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .config import HexformerArchitectureConfig


@dataclass(frozen=True, slots=True)
class HexformerOutputs:
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    distance: torch.Tensor
    opp_policy_logits: torch.Tensor
    threat_logits: torch.Tensor
    rz_logits: torch.Tensor
    lookahead_logits: Mapping[str, torch.Tensor]


class HexConv2d(nn.Conv2d):
    """3x3 masked convolution over axial-neighborhood cells embedded in a square crop."""

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


class GatedHexBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.main = nn.Sequential(
            HexConv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
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
        return x + self.main(x) * self.gate(x)


class LocalHexEncoder(nn.Module):
    def __init__(self, config: HexformerArchitectureConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            HexConv2d(config.local_input_channels, config.local_channels, kernel_size=3, padding=1),
            nn.GELU(),
            *[GatedHexBlock(config.local_channels, config.dropout) for _ in range(config.local_blocks)],
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(config.local_channels, config.token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        return self.proj(self.pool(features).flatten(start_dim=1))


class GraphGPSBlock(nn.Module):
    """Simple GraphGPS-style layer: local coordinate message plus global attention."""

    def __init__(self, token_dim: int, heads: int, dropout: float, edge_feature_dim: int) -> None:
        super().__init__()
        self.local_norm = nn.LayerNorm(token_dim)
        self.local_message = nn.Sequential(
            nn.Linear(token_dim + 5, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )
        self.edge_message = nn.Sequential(
            nn.Linear(token_dim + edge_feature_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )
        self.attn_norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 4, token_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_in = torch.cat([self.local_norm(tokens), coords], dim=-1)
        tokens = tokens + self.local_message(local_in)
        if edge_index is not None and edge_features is not None and edge_mask is not None and edge_index.shape[1] > 0:
            tokens = tokens + self._edge_aggregate(tokens, edge_index, edge_features, edge_mask)
        attn_in = self.attn_norm(tokens)
        attn_out, _weights = self.attn(attn_in, attn_in, attn_in, key_padding_mask=~mask, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)

    def _edge_aggregate(
        self,
        tokens: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        norm_tokens = self.local_norm(tokens)
        aggregate = torch.zeros_like(tokens)
        counts = torch.zeros((*tokens.shape[:2], 1), dtype=tokens.dtype, device=tokens.device)
        token_count = tokens.shape[1]
        for batch_index in range(tokens.shape[0]):
            valid = edge_mask[batch_index].bool()
            if not bool(valid.any()):
                continue
            edges = edge_index[batch_index, valid].to(device=tokens.device, dtype=torch.long)
            features = edge_features[batch_index, valid].to(device=tokens.device, dtype=tokens.dtype)
            in_bounds = (
                (edges[:, 0] >= 0)
                & (edges[:, 0] < token_count)
                & (edges[:, 1] >= 0)
                & (edges[:, 1] < token_count)
            )
            if not bool(in_bounds.any()):
                continue
            edges = edges[in_bounds]
            features = features[in_bounds]
            messages = self.edge_message(torch.cat([norm_tokens[batch_index, edges[:, 0]], features], dim=-1))
            aggregate[batch_index].index_add_(0, edges[:, 1], messages)
            counts[batch_index].index_add_(0, edges[:, 1], torch.ones((edges.shape[0], 1), dtype=tokens.dtype, device=tokens.device))
        return aggregate / counts.clamp_min(1.0)


class HexformerAR(nn.Module):
    """Sparse candidate-pointer policy/value model."""

    def __init__(self, config: HexformerArchitectureConfig | None = None) -> None:
        super().__init__()
        self.config = config or HexformerArchitectureConfig()
        cfg = self.config
        self.local_encoder = LocalHexEncoder(cfg)
        self.global_proj = nn.Linear(cfg.global_feature_dim, cfg.token_dim)
        self.candidate_proj = nn.Linear(cfg.candidate_feature_dim + 5, cfg.token_dim)
        self.stone_proj = nn.Linear(cfg.stone_feature_dim + 5, cfg.token_dim)
        self.window_proj = nn.Linear(cfg.window_feature_dim + 5, cfg.token_dim)
        self.type_embedding = nn.Embedding(5, cfg.token_dim)
        self.layers = nn.ModuleList(
            [
                GraphGPSBlock(cfg.token_dim, cfg.attention_heads, cfg.dropout, cfg.rel_edge_feature_dim)
                for _ in range(cfg.gps_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.token_dim)
        self.policy_head = nn.Sequential(nn.Linear(cfg.token_dim * 2, cfg.token_dim), nn.GELU(), nn.Linear(cfg.token_dim, 1))
        self.opp_policy_head = nn.Sequential(nn.Linear(cfg.token_dim * 2, cfg.token_dim), nn.GELU(), nn.Linear(cfg.token_dim, 1))
        self.wdl_head = nn.Sequential(nn.Linear(cfg.token_dim * 2, cfg.token_dim), nn.GELU(), nn.Linear(cfg.token_dim, 3))
        self.distance_head = nn.Sequential(nn.Linear(cfg.token_dim * 2, cfg.token_dim), nn.GELU(), nn.Linear(cfg.token_dim, 1))
        self.threat_head = nn.Linear(cfg.token_dim, cfg.threat_classes)
        self.relevance_head = nn.Linear(cfg.token_dim, 1)
        self.lookahead_heads = nn.ModuleDict(
            {str(horizon): nn.Linear(cfg.token_dim * 2, 3) for horizon in cfg.lookahead_horizons}
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        candidate_features = batch["candidate_features"]
        candidate_mask = batch["candidate_mask"].bool()
        candidate_coords = batch["candidate_coords"].to(dtype=candidate_features.dtype)
        stone_features = batch["stone_features"]
        stone_mask = batch["stone_mask"].bool()
        stone_coords = batch["stone_coords"].to(dtype=stone_features.dtype)
        window_features = batch["window_features"]
        window_mask = batch["window_mask"].bool()
        window_coords = batch["window_coords"].to(dtype=window_features.dtype)
        global_features = batch["global_features"]
        local_tokens, local_window_coords, local_window_mask = self._encode_local_windows(batch)
        local_summary = _masked_mean(local_tokens, local_window_mask)
        global_token = self.global_proj(global_features) + local_summary

        candidate_tokens = self.candidate_proj(torch.cat([candidate_features, candidate_coords], dim=-1))
        stone_tokens = self.stone_proj(torch.cat([stone_features, stone_coords], dim=-1))
        window_tokens = self.window_proj(torch.cat([window_features, window_coords], dim=-1))
        local_tokens = local_tokens + self.type_embedding.weight[4]
        candidate_tokens = candidate_tokens + self.type_embedding.weight[1]
        stone_tokens = stone_tokens + self.type_embedding.weight[2]
        window_tokens = window_tokens + self.type_embedding.weight[3]
        global_token = global_token + self.type_embedding.weight[0]

        tokens = torch.cat([global_token.unsqueeze(1), local_tokens, candidate_tokens, stone_tokens, window_tokens], dim=1)
        zero_global = torch.zeros((tokens.shape[0], 1, 5), dtype=tokens.dtype, device=tokens.device)
        coords = torch.cat([zero_global, local_window_coords, candidate_coords, stone_coords, window_coords], dim=1)
        mask = torch.cat(
            [
                torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device),
                local_window_mask,
                candidate_mask,
                stone_mask,
                window_mask,
            ],
            dim=1,
        )
        edge_index = batch.get("rel_edge_index")
        edge_features = batch.get("rel_edge_features")
        edge_mask = batch.get("rel_edge_mask")
        for layer in self.layers:
            tokens = layer(tokens, coords, mask, edge_index=edge_index, edge_features=edge_features, edge_mask=edge_mask)
        tokens = self.final_norm(tokens)
        state_token = tokens[:, 0, :]
        candidate_start = 1 + local_tokens.shape[1]
        candidate_encoded = tokens[:, candidate_start : candidate_start + candidate_tokens.shape[1], :]
        fused_candidate = torch.cat([candidate_encoded, state_token.unsqueeze(1).expand_as(candidate_encoded)], dim=-1)
        state_summary = torch.cat([state_token, local_summary], dim=-1)

        policy_logits = self.policy_head(fused_candidate).squeeze(-1)
        opp_policy_logits = self.opp_policy_head(fused_candidate).squeeze(-1)
        rz_logits = self.relevance_head(candidate_encoded).squeeze(-1)
        invalid = ~candidate_mask
        policy_logits = policy_logits.masked_fill(invalid, torch.finfo(policy_logits.dtype).min)
        opp_policy_logits = opp_policy_logits.masked_fill(invalid, torch.finfo(opp_policy_logits.dtype).min)
        rz_logits = rz_logits.masked_fill(invalid, torch.finfo(rz_logits.dtype).min)
        outputs: dict[str, torch.Tensor] = {
            "policy_logits": policy_logits,
            "policy": policy_logits,
            "wdl_logits": self.wdl_head(state_summary),
            "distance": self.distance_head(state_summary).squeeze(-1),
            "opp_policy_logits": opp_policy_logits,
            "opp_policy": opp_policy_logits,
            "threat_logits": self.threat_head(candidate_encoded),
            "rz_logits": rz_logits,
        }
        for horizon, head in self.lookahead_heads.items():
            outputs[f"lookahead_{horizon}"] = head(state_summary)
        return outputs

    def _encode_local_windows(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_inputs = batch.get("local_inputs")
        if local_inputs is None:
            local_inputs = batch["local_input"].unsqueeze(1)
        batch_size, local_count = local_inputs.shape[:2]
        flat = local_inputs.reshape(batch_size * local_count, *local_inputs.shape[2:])
        tokens = self.local_encoder(flat).reshape(batch_size, local_count, -1)
        coords = batch.get("local_window_coords")
        if coords is None:
            coords = torch.zeros((batch_size, local_count, 5), dtype=tokens.dtype, device=tokens.device)
        else:
            coords = coords.to(device=tokens.device, dtype=tokens.dtype)
        mask = batch.get("local_window_mask")
        if mask is None:
            mask = torch.ones((batch_size, local_count), dtype=torch.bool, device=tokens.device)
        else:
            mask = mask.to(device=tokens.device, dtype=torch.bool)
        return tokens, coords, mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(dtype=values.dtype, device=values.device)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
