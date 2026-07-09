"""Phase-A G6: typed toy net and the Phase-B regular-attention boundary."""

from __future__ import annotations

import math
import os
import random
import sys

os.environ["HEXFIELD_EQ_GROUP_ORDER"] = "12"
os.environ["HEXFIELD_EQ_FEATURE_VERSION"] = "1"
os.environ["HEXFIELD_EQ_SUPPORT_RADIUS"] = "1"
for _name in (
    "HEXFIELD_TRITON_CONV",
    "HEXFIELD_TRITON_CONV_LN",
    "HEXFIELD_TRITON_ATTN",
    "HEXFIELD_EQ_TRITON_RAY",
    "HEXFIELD_SERVE_FLEX",
    "HEXFIELD_TRAIN_FLEX",
    "HEXFIELD_FLEX_PAIR",
    "HEXFIELD_TRAIN_FLEX_PAIR",
):
    os.environ[_name] = "0"

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

# model.py imports flex_attention even when all serve/train gates are off; on
# Windows that imports Triton.  Phase A is CPU/no-Triton, so force the existing
# guarded import down its documented no-flex fallback before importing model.py.
sys.modules["torch.nn.attention.flex_attention"] = None

from hexfield_eq import constants as C  # noqa: E402
from hexfield_eq import equivariant as production_eq  # noqa: E402
from hexfield_eq.constants import (  # noqa: E402
    BIAS_CELL_TOKEN_ROW,
    BIAS_TOKEN_CELL_ROW,
    BIAS_TOKEN_TOKEN_ROW,
    DIRECTIONS,
    NUM_TOKENS,
)
from hexfield_eq.features import (  # noqa: E402
    PositionFacts,
    build_position,
    record_phase,
    record_player,
    transform_facts,
)
from hexfield_eq.geometry import apply_d6, rel_bias_index  # noqa: E402
from hexfield_eq.model import (  # noqa: E402
    AttnBlock as ProductionAttnBlock,
    ConvBlock as ProductionConvBlock,
    GroupAffineNorm as ProductionNorm,
    HexNodeConv as ProductionConv,
)
from hexfield_eq.reps import (  # noqa: E402
    TypedConv,
    TypedGroupAffineNorm,
    TypedLayerScale,
    TypedLinear,
    TypedStem,
    expand_per_instance,
    head_perm,
    head_perm_inv,
    production_conv_coefficients,
    production_linear_coefficients,
    scale_signature,
    signature_instances,
    signature_width,
    typed_group_pool,
)

assert not any(name == "triton" or name.startswith("triton.") for name in sys.modules)

SIG_51 = (("reg", 2), ("mirror", 2), ("point", 1), ("axis", 2), ("triv", 3))
SIG_96 = (("reg", 4), ("mirror", 4), ("axis", 4), ("triv", 12))


class ToyConvBlock(nn.Module):
    """Production-form post-activation typed residual conv block."""

    def __init__(self, signature) -> None:
        super().__init__()
        self.conv1 = TypedConv(signature, signature, dtype=torch.float64)
        self.ln1 = TypedGroupAffineNorm(signature, dtype=torch.float64)
        self.conv2 = TypedConv(signature, signature, dtype=torch.float64)
        self.ln2 = TypedGroupAffineNorm(signature, dtype=torch.float64)
        self.ls = TypedLayerScale(signature, dtype=torch.float64)

    def forward(self, x: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.ln1(self.conv1(x, nbr)))
        y = self.ln2(self.conv2(y, nbr))
        return F.relu(x + self.ls(y))


class ToyBoundaryAttention(nn.Module):
    """Typed residual stream -> pure regular q/k/v -> typed residual stream."""

    def __init__(self, signature, k_attn: int) -> None:
        super().__init__()
        self.signature = signature
        self.k_attn = k_attn
        self.regular = (("reg", k_attn),)
        self.width = 12 * k_attn
        self.heads = 3
        self.head_dim = 4 * k_attn
        self.q_proj = TypedLinear(signature, self.regular, dtype=torch.float64)
        self.k_proj = TypedLinear(signature, self.regular, dtype=torch.float64)
        self.v_proj = TypedLinear(signature, self.regular, dtype=torch.float64)
        self.out_proj = TypedLinear(self.regular, signature, dtype=torch.float64)
        self.register_buffer("hp", head_perm(k_attn), persistent=False)
        self.register_buffer("hp_inv", head_perm_inv(k_attn), persistent=False)
        joint, n_joint = production_eq.joint_bias_lut()
        self.register_buffer("joint", joint, persistent=False)
        self.theta = nn.Parameter(torch.zeros(n_joint, dtype=torch.float64))

    def attention_bias(self, coords: torch.Tensor) -> torch.Tensor:
        """Materialize the exactly jointly (pair-row, head)-tied bias."""

        batch, nodes, _ = coords.shape
        size = NUM_TOKENS + nodes
        pair = torch.full(
            (batch, size, size),
            BIAS_TOKEN_TOKEN_ROW,
            dtype=torch.long,
            device=coords.device,
        )
        pair[:, :NUM_TOKENS, NUM_TOKENS:] = BIAS_TOKEN_CELL_ROW
        pair[:, NUM_TOKENS:, :NUM_TOKENS] = BIAS_CELL_TOKEN_ROW
        for b in range(batch):
            for query in range(nodes):
                qq, qr = (int(v) for v in coords[b, query].tolist())
                for key in range(nodes):
                    kq, kr = (int(v) for v in coords[b, key].tolist())
                    pair[b, NUM_TOKENS + query, NUM_TOKENS + key] = rel_bias_index(
                        kq - qq, kr - qr
                    )
        table = self.theta[self.joint]  # (rows, heads)
        return table[pair].permute(0, 3, 1, 2).contiguous()

    def forward(self, seq: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        batch, size, _ = seq.shape
        q = self.q_proj(seq).index_select(-1, self.hp)
        k = self.k_proj(seq).index_select(-1, self.hp)
        v = self.v_proj(seq).index_select(-1, self.hp)
        q = q.reshape(batch, size, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, size, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, size, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attended = torch.softmax(scores + self.attention_bias(coords), dim=-1) @ v
        attended = attended.transpose(1, 2).reshape(batch, size, self.width)
        attended = attended.index_select(-1, self.hp_inv)
        return self.out_proj(attended)


class ToyAttentionBlock(nn.Module):
    """Production-form pre-norm attention block with typed MLP boundaries."""

    def __init__(self, signature, k_attn: int) -> None:
        super().__init__()
        hidden = scale_signature(signature, 2)
        self.ln1 = TypedGroupAffineNorm(signature, dtype=torch.float64)
        self.attn = ToyBoundaryAttention(signature, k_attn)
        self.ln2 = TypedGroupAffineNorm(signature, dtype=torch.float64)
        self.fc1 = TypedLinear(signature, hidden, dtype=torch.float64)
        self.fc2 = TypedLinear(hidden, signature, dtype=torch.float64)
        self.ls_attn = TypedLayerScale(signature, dtype=torch.float64)
        self.ls_mlp = TypedLayerScale(signature, dtype=torch.float64)

    def forward(self, seq: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        seq = seq + self.ls_attn(self.attn(self.ln1(seq), coords))
        seq = seq + self.ls_mlp(self.fc2(F.gelu(self.fc1(self.ln2(seq)))))
        return seq


class ToyRegisterRefresh(nn.Module):
    """Sigmoid-gated, unnormalized SUM lane with regular internal heads."""

    def __init__(self, signature, k_attn: int) -> None:
        super().__init__()
        self.signature = signature
        self.regular = (("reg", k_attn),)
        self.head_dim = 4 * k_attn
        self.width = 12 * k_attn
        self.ln_q = TypedGroupAffineNorm(signature, dtype=torch.float64)
        self.ln_kv = TypedGroupAffineNorm(signature, dtype=torch.float64)
        self.q_proj = TypedLinear(signature, self.regular, dtype=torch.float64)
        self.k_proj = TypedLinear(signature, self.regular, dtype=torch.float64)
        self.v_proj = TypedLinear(signature, self.regular, dtype=torch.float64)
        self.out_proj = TypedLinear(self.regular, signature, dtype=torch.float64)
        self.gate_bias = nn.Parameter(torch.full((NUM_TOKENS,), -2.5, dtype=torch.float64))
        self.sum_scale = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
        self.register_buffer("hp", head_perm(k_attn), persistent=False)
        self.register_buffer("hp_inv", head_perm_inv(k_attn), persistent=False)

    def forward(self, tokens: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
        batch, token_count, _ = tokens.shape
        nodes = cells.shape[1]
        q = self.q_proj(self.ln_q(tokens)).index_select(-1, self.hp)
        kv = self.ln_kv(cells)
        k = self.k_proj(kv).index_select(-1, self.hp)
        v = self.v_proj(kv).index_select(-1, self.hp)
        q = q.reshape(batch, token_count, 3, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, nodes, 3, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, nodes, 3, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        gates = torch.sigmoid(scores + self.gate_bias.view(1, 1, token_count, 1))
        update = (gates @ v) * self.sum_scale
        update = update.transpose(1, 2).reshape(batch, token_count, self.width)
        update = update.index_select(-1, self.hp_inv)
        return tokens + self.out_proj(update)


class TypedToyNet(nn.Module):
    """Complete Phase-B-boundary rehearsal used by the G6 equivariance gate."""

    def __init__(self, signature, *, k_attn: int = 4) -> None:
        super().__init__()
        self.signature = signature
        self.stem = TypedStem(signature, dtype=torch.float64)
        self.stem_norm = TypedGroupAffineNorm(signature, dtype=torch.float64)
        self.conv_blocks = nn.ModuleList([ToyConvBlock(signature) for _ in range(2)])
        self.refresh = ToyRegisterRefresh(signature, k_attn)
        self.attn = ToyAttentionBlock(signature, k_attn)
        self.final_norm = TypedGroupAffineNorm(signature, dtype=torch.float64)
        self.token_base = nn.Parameter(
            torch.empty(NUM_TOKENS, signature_instances(signature), dtype=torch.float64)
        )
        nn.init.normal_(self.token_base, std=0.05)

        self.policy_conv = TypedConv(signature, signature, dtype=torch.float64)
        self.policy_expand = TypedLinear(
            signature, scale_signature(signature, 2), dtype=torch.float64
        )
        self.policy_head = nn.Linear(
            2 * signature_instances(signature), 1, dtype=torch.float64
        )
        self.value_read = TypedLinear(
            signature, scale_signature(signature, 2), dtype=torch.float64
        )
        self.value_head = nn.Linear(
            4 * signature_instances(signature), 1, dtype=torch.float64
        )

    def forward(
        self, features: torch.Tensor, nbr: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cells = F.relu(self.stem_norm(self.stem(features, nbr)))
        cells = self.conv_blocks[0](cells, nbr)
        cells = self.conv_blocks[1](cells, nbr)
        token_dense = expand_per_instance(self.token_base, self.signature)
        tokens = token_dense.unsqueeze(0).expand(features.shape[0], -1, -1)
        tokens = self.refresh(tokens, cells)
        joint = self.attn(torch.cat((tokens, cells), dim=1), coords)
        tokens, cells = joint[:, :NUM_TOKENS], joint[:, NUM_TOKENS:]
        cells = self.final_norm(cells)

        policy_features = F.relu(self.policy_conv(cells, nbr))
        policy_features = self.policy_expand(policy_features)
        policy = self.policy_head(typed_group_pool(policy_features, scale_signature(self.signature, 2)))
        token_read = typed_group_pool(
            self.value_read(tokens.mean(dim=1)), scale_signature(self.signature, 2)
        )
        cell_read = typed_group_pool(
            self.value_read(cells.mean(dim=1)), scale_signature(self.signature, 2)
        )
        value = self.value_head(torch.cat((token_read, cell_read), dim=-1))
        return policy.squeeze(-1), value.squeeze(-1)


def _random_legal_facts(seed: int, moves: int) -> PositionFacts:
    """Generate a legal connected placement prefix without engine dependencies."""

    rng = random.Random(seed)
    occupied = {(0, 0)}
    records = [(0, 0, record_player(0), 0)]
    for ordinal in range(1, moves):
        candidates = {
            (q + dq, r + dr)
            for q, r in occupied
            for dq, dr in DIRECTIONS
            if (q + dq, r + dr) not in occupied
        }
        q, r = rng.choice(sorted(candidates))
        occupied.add((q, r))
        records.append((q, r, record_player(ordinal), ordinal))
    phase = record_phase(moves)
    first_stone = (records[-1][0], records[-1][1]) if phase == "SecondStone" else None
    return PositionFacts(
        records=tuple(records),
        current_player=record_player(moves),
        phase=phase,
        first_stone=first_stone,
    )


def _inputs(facts: PositionFacts):
    support, features = build_position(facts)
    return (
        support,
        torch.from_numpy(features).to(torch.float64).unsqueeze(0),
        torch.from_numpy(support.nbr.astype("int64")).unsqueeze(0),
        torch.from_numpy(support.coords.astype("int64")).unsqueeze(0),
    )


def _randomize(module: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.08)


def test_two_typed_toynets_are_equivariant_on_real_oracle_features() -> None:
    """Two required signatures x five positions x all D6 policy/value checks."""

    facts_list = [_random_legal_facts(100 + i, 5 + i) for i in range(5)]
    for net_index, signature in enumerate((SIG_51, SIG_96)):
        torch.manual_seed(0)
        model = TypedToyNet(signature, k_attn=4).eval()
        _randomize(model, 10 + net_index)
        with torch.no_grad():
            for facts in facts_list:
                support, features, nbr, coords = _inputs(facts)
                base_policy, base_value = model(features, nbr, coords)
                for g in range(12):
                    transformed_facts = transform_facts(facts, g)
                    support_g, features_g, nbr_g, coords_g = _inputs(transformed_facts)
                    policy_g, value_g = model(features_g, nbr_g, coords_g)
                    node_permutation = torch.tensor(
                        [
                            support_g.index[apply_d6(g, int(q), int(r))]
                            for q, r in support.coords.tolist()
                        ],
                        dtype=torch.long,
                    )
                    torch.testing.assert_close(
                        policy_g[:, node_permutation], base_policy, atol=1e-9, rtol=0
                    )
                    torch.testing.assert_close(
                        value_g, base_value, atol=1e-9, rtol=0
                    )


def _copy_norm(typed: TypedGroupAffineNorm, production) -> None:
    with torch.no_grad():
        typed.gamma.copy_(production.gamma.to(torch.float64))
        typed.beta.copy_(production.beta.to(torch.float64))


def _copy_scale(typed: TypedLayerScale, production) -> None:
    with torch.no_grad():
        typed.gamma.copy_(production.gamma.to(torch.float64))


def _copy_conv(typed: TypedConv, production) -> None:
    with torch.no_grad():
        typed.coefficients["reg__from__reg"].copy_(
            production_conv_coefficients(production.w_base).to(torch.float64)
        )
        typed.bias_base.copy_(production.bias_base.to(torch.float64))


def _copy_linear(typed: TypedLinear, production) -> None:
    with torch.no_grad():
        typed.coefficients["reg__from__reg"].copy_(
            production_linear_coefficients(production.wb).to(torch.float64)
        )
        typed.bias_base.copy_(production.bias_base.to(torch.float64))


def test_pure_regular_block_structure_matches_production_primitives() -> None:
    """Stem + two C blocks + one A block agree under matched parameters."""

    assert C.CHANNELS == 96 and C.C_ORBIT == 8 and C.NUM_FEATURES == 25
    signature = (("reg", C.C_ORBIT),)
    torch.manual_seed(0)
    typed_stem = TypedStem(signature, dtype=torch.float64)
    production_stem = ProductionConv(C.NUM_FEATURES, C.CHANNELS).to(torch.float64)
    typed_stem_norm = TypedGroupAffineNorm(signature, dtype=torch.float64)
    production_stem_norm = ProductionNorm(C.CHANNELS).to(torch.float64)
    production_blocks = nn.ModuleList(
        [ProductionConvBlock(C.CHANNELS).to(torch.float64) for _ in range(2)]
    )
    typed_blocks = nn.ModuleList([ToyConvBlock(signature) for _ in range(2)])
    production_attn = ProductionAttnBlock(C.CHANNELS, 3).to(torch.float64)
    production_attn.attn.impl = "materialized"
    typed_attn = ToyAttentionBlock(signature, C.C_ORBIT)

    with torch.no_grad():
        typed_stem.w0.copy_(production_stem.w0)
        typed_stem.bias_base.copy_(production_stem.bias_base)
    _copy_norm(typed_stem_norm, production_stem_norm)
    for typed_block, production_block in zip(typed_blocks, production_blocks, strict=True):
        _copy_conv(typed_block.conv1, production_block.conv1)
        _copy_conv(typed_block.conv2, production_block.conv2)
        _copy_norm(typed_block.ln1, production_block.ln1)
        _copy_norm(typed_block.ln2, production_block.ln2)
        _copy_scale(typed_block.ls, production_block.ls)
    _copy_norm(typed_attn.ln1, production_attn.ln1)
    _copy_norm(typed_attn.ln2, production_attn.ln2)
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        _copy_linear(getattr(typed_attn.attn, name), getattr(production_attn.attn, name))
    _copy_linear(typed_attn.fc1, production_attn.fc1)
    _copy_linear(typed_attn.fc2, production_attn.fc2)
    _copy_scale(typed_attn.ls_attn, production_attn.ls_attn)
    _copy_scale(typed_attn.ls_mlp, production_attn.ls_mlp)
    with torch.no_grad():
        typed_attn.attn.theta.copy_(torch.randn_like(typed_attn.attn.theta) * 0.05)

    facts = _random_legal_facts(999, 7)
    support, features, nbr, coords = _inputs(facts)
    nodes = support.num_nodes
    mask = torch.ones((1, nodes), dtype=torch.bool)
    self_index = torch.arange(nodes).view(1, nodes, 1)
    safe_nbr = torch.where(nbr >= 0, nbr, torch.full_like(nbr, nodes))
    gather = torch.cat((self_index, safe_nbr), dim=-1)

    typed_x = F.relu(typed_stem_norm(typed_stem(features, nbr)))
    production_x = F.relu(production_stem_norm(production_stem(features, gather, mask)))
    torch.testing.assert_close(typed_x, production_x, atol=1e-12, rtol=0)
    for typed_block, production_block in zip(typed_blocks, production_blocks, strict=True):
        typed_x = typed_block(typed_x, nbr)
        production_x = production_block(production_x, gather, mask)
        torch.testing.assert_close(typed_x, production_x, atol=1e-11, rtol=0)

    token_base = torch.randn(NUM_TOKENS, C.C_ORBIT, dtype=torch.float64)
    tokens = expand_per_instance(token_base, signature).unsqueeze(0)
    typed_seq = torch.cat((tokens, typed_x), dim=1)
    production_seq = torch.cat((tokens, production_x), dim=1)
    typed_output = typed_attn(typed_seq, coords)
    bias = typed_attn.attn.attention_bias(coords)
    seq_mask = torch.ones(typed_seq.shape[:2], dtype=torch.bool)
    production_output = production_attn(production_seq, bias, seq_mask)
    torch.testing.assert_close(typed_output, production_output, atol=1e-10, rtol=0)
    assert not any(name == "triton" or name.startswith("triton.") for name in sys.modules)
