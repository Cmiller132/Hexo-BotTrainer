r"""Spec C/D tests for dense_cnn_restnet attention scopes.

Spec C — static disk KEY mask:
  * "full" reproduces the pre-change (unmasked) attention bit-for-bit;
  * "disk" drives corner-key attention to ~0 (queries unrestricted, no NaN);
  * sdpa and materialized agree under every scope;
  * the disk buffers are NOT in the state_dict (checkpoints cross-load between scopes).

Spec D — gathered-token attention ("content") lives lower in this file.

CPU-only. Run:
    python -m pytest tests\test_dense_cnn_restnet_attention.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("dense_cnn_restnet", "hexo_train", "hexo_runner"):
    _src = _ROOT / "packages" / _pkg / "python"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from dense_cnn_restnet.architecture import (  # noqa: E402
    _DISK_MASK_VALUE,
    RelPosMHSA,
    RestnetNetwork,
    TransformerBlock,
)
from dense_cnn_restnet.architecture import (  # noqa: E402
    _DISK_MASK_VALUE as _NEG,
    _select_content_tokens,
)
from dense_cnn_restnet.constants import BOARD_AREA, BOARD_SIZE, INPUT_CHANNELS, VALUE_BINS  # noqa: E402
from dense_cnn_restnet.losses import model1_loss  # noqa: E402

torch.manual_seed(0)


def _manual_attention(mhsa: RelPosMHSA, x: torch.Tensor, bias: torch.Tensor):
    """Reference attention given an explicit additive bias; returns (out, attn)."""

    b, n, _ = x.shape
    shape = (b, n, mhsa.num_heads, mhsa.head_dim)
    q = mhsa.q_proj(x).view(shape).permute(0, 2, 1, 3)
    k = mhsa.k_proj(x).view(shape).permute(0, 2, 1, 3)
    v = mhsa.v_proj(x).view(shape).permute(0, 2, 1, 3)
    dots = torch.matmul(q, k.transpose(-2, -1)) * mhsa.scaling + bias
    attn = torch.softmax(dots, dim=-1)
    out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, n, mhsa.channels)
    return mhsa.out_drop(mhsa.out_proj(out)), attn


def _raw_bias(mhsa: RelPosMHSA) -> torch.Tensor:
    """The original (pre-change) unmasked relative bias, (1, heads, N, N)."""

    bias = mhsa.relative_bias_table[mhsa.relative_index]
    bias = bias.view(mhsa.token_len, mhsa.token_len, mhsa.num_heads)
    return bias.permute(2, 0, 1).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Spec C: "full" reproduces the pre-change behavior bit-for-bit.
# --------------------------------------------------------------------------- #
def test_full_scope_bias_is_unmasked():
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=6, board_w=6, attention_scope="full").eval()
    with torch.no_grad():
        mhsa.relative_bias_table.normal_(std=0.3)
    bias = mhsa._compute_relative_bias()
    assert torch.equal(bias, _raw_bias(mhsa))  # identical, no -1e9 anywhere
    assert torch.isfinite(bias).all()


def test_full_scope_forward_matches_manual_unmasked():
    torch.manual_seed(1)
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=6, board_w=6, attention_scope="full").eval()
    with torch.no_grad():
        mhsa.relative_bias_table.normal_(std=0.3)
    x = torch.randn(2, 36, 16)
    with torch.no_grad():
        mhsa.set_impl("materialized")
        out = mhsa(x)
        ref, _ = _manual_attention(mhsa, x, _raw_bias(mhsa))
    assert torch.allclose(out, ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# Spec C: "disk" masks corner-key columns -> ~0 attention there, no NaN.
# --------------------------------------------------------------------------- #
def test_disk_scope_zeros_corner_key_attention():
    torch.manual_seed(2)
    # Use the real board so the disk/corner split is exercised (1261 / 420).
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=BOARD_SIZE, board_w=BOARD_SIZE,
                      attention_scope="disk").eval()
    with torch.no_grad():
        mhsa.relative_bias_table.normal_(std=0.3)
    n = mhsa.token_len
    assert int(mhsa.disk_flat.sum()) == 1261
    bias = mhsa._compute_relative_bias()
    # Corner KEY columns carry the large negative mask; disk columns are finite.
    corner = ~mhsa.disk_flat
    assert torch.all(bias[..., corner] <= _DISK_MASK_VALUE / 2)
    assert torch.isfinite(bias[..., mhsa.disk_flat]).all()
    x = torch.randn(2, n, 16)
    with torch.no_grad():
        mhsa.set_impl("materialized")
        _, attn = _manual_attention(mhsa, x, bias)
    # No NaN (queries are unrestricted) and corner keys receive ~0 weight.
    assert torch.isfinite(attn).all()
    corner_mass = attn[..., corner].sum(dim=-1)  # (B, heads, N_query)
    assert float(corner_mass.max()) < 1e-6
    # Every query row still sums to 1 over the disk keys.
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)


# --------------------------------------------------------------------------- #
# Spec C: sdpa == materialized under every scope.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scope", ["full", "disk"])
def test_sdpa_equals_materialized_per_scope(scope):
    torch.manual_seed(3)
    h = w = 6
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=h, board_w=w, attention_scope=scope).eval()
    with torch.no_grad():
        mhsa.relative_bias_table.normal_(std=0.3)
    x = torch.randn(2, h * w, 16)
    with torch.no_grad():
        mhsa.set_impl("materialized")
        out_mat = mhsa(x)
        mhsa.set_impl("sdpa")
        out_sdpa = mhsa(x)
    assert torch.allclose(out_mat, out_sdpa, atol=1e-5, rtol=1e-4)


def test_network_sdpa_equals_materialized_disk_scope():
    torch.manual_seed(4)
    net = RestnetNetwork(channels=16, blocks_type="R_T_R_T", attention_scope="disk",
                         short_term_value_horizons=(1,)).eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, RelPosMHSA):
                m.relative_bias_table.normal_(std=0.1)
    x = torch.randn(2, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        net.set_attention_impl("materialized")
        out_mat = net(x)
        net.set_attention_impl("sdpa")
        out_sdpa = net(x)
    for key in out_mat:
        assert torch.allclose(out_mat[key], out_sdpa[key], atol=1e-4, rtol=1e-4), key


# --------------------------------------------------------------------------- #
# Spec C: training == eval (cache) under disk, and cache invalidates on table edit.
# --------------------------------------------------------------------------- #
def test_disk_cache_matches_training_and_invalidates():
    torch.manual_seed(5)
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=6, board_w=6, attention_scope="disk")
    with torch.no_grad():
        mhsa.relative_bias_table.normal_(std=0.2)
    x = torch.randn(2, 36, 16)
    mhsa.train()
    with torch.enable_grad():
        out_train = mhsa(x)
    mhsa.eval()
    with torch.no_grad():
        out_eval1 = mhsa(x)
        out_eval2 = mhsa(x)
    assert torch.allclose(out_train, out_eval1, atol=1e-6)
    assert torch.equal(out_eval1, out_eval2)
    with torch.no_grad():
        # A NON-uniform edit (a uniform shift is softmax-invariant): the version
        # bump must invalidate the cache so the new table is reflected.
        mhsa.relative_bias_table.add_(torch.randn_like(mhsa.relative_bias_table))
        out_eval3 = mhsa(x)
    assert not torch.allclose(out_eval1, out_eval3)


def test_set_attention_scope_changes_output_and_invalidates_cache():
    torch.manual_seed(6)
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=BOARD_SIZE, board_w=BOARD_SIZE,
                      attention_scope="full").eval()
    with torch.no_grad():
        mhsa.relative_bias_table.normal_(std=0.3)
    x = torch.randn(1, mhsa.token_len, 16)
    with torch.no_grad():
        out_full = mhsa(x)
        mhsa.set_attention_scope("disk")
        out_disk = mhsa(x)
    assert not torch.allclose(out_full, out_disk)


# --------------------------------------------------------------------------- #
# Spec C/global: disk buffers absent from state_dict; checkpoints cross-load.
# --------------------------------------------------------------------------- #
def test_disk_buffers_not_in_state_dict():
    full = RestnetNetwork(channels=16, blocks_type="R_T", attention_scope="full")
    disk = RestnetNetwork(channels=16, blocks_type="R_T", attention_scope="disk")
    assert set(full.state_dict().keys()) == set(disk.state_dict().keys())
    for key in full.state_dict():
        assert "disk_flat" not in key and "disk_key_bias" not in key


def test_checkpoint_cross_loads_between_scopes():
    torch.manual_seed(7)
    full = RestnetNetwork(channels=16, blocks_type="R_T_R", attention_scope="full")
    disk = RestnetNetwork(channels=16, blocks_type="R_T_R", attention_scope="disk")
    # A checkpoint trained under one scope must strict-load into the other.
    missing, unexpected = disk.load_state_dict(full.state_dict(), strict=True)
    assert not missing and not unexpected
    # After loading, disk still applies its key mask (different output from full).
    full.eval()
    disk.eval()
    x = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        assert not torch.allclose(full(x)["policy"], disk(x)["policy"])


# =========================================================================== #
# Spec D — gathered-token ("content") attention.
# =========================================================================== #
def _dense_block_reference(block: TransformerBlock, x_tokens: torch.Tensor, relevance: torch.Tensor):
    """Oracle: a full dense transformer block whose attention masks NON-relevant
    KEY columns (relevance, not disk). For a relevant query this is identical to the
    gathered "content" block (same relative bias, same relevant key set), so it is
    the reference the content path must match at relevant cells. Materialized math.
    """

    mhsa = block.attn
    raw = _raw_bias(mhsa)  # (1, heads, N, N) unmasked
    key_mask = torch.where(relevance, torch.zeros((), dtype=raw.dtype), torch.full((), _NEG, dtype=raw.dtype))
    bias = raw + key_mask.view(relevance.shape[0], 1, 1, relevance.shape[1])  # (B, heads, N, N)
    normed = block.ln1(x_tokens)
    attn_out = mhsa._attend_with_bias(normed, bias)  # materialized (mhsa.impl)
    x1 = x_tokens + attn_out
    return x1 + block.mlp(block.ln2(x1))


def _random_relevance(b: int, n: int, *, seed: int, density: float = 0.3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rel = torch.rand(b, n, generator=g) < density
    rel[:, 0] = True  # guarantee at least one relevant token per sample
    return rel


def test_select_content_tokens_shapes_and_bucketing():
    rel = _random_relevance(3, BOARD_SIZE * BOARD_SIZE, seed=1)
    idx, valid = _select_content_tokens(rel)
    k_pad = idx.shape[1]
    assert k_pad % 64 == 0 or k_pad == BOARD_SIZE * BOARD_SIZE
    assert k_pad >= int(rel.sum(1).max())
    # valid count per row == relevant count; valid indices are exactly the relevant cells.
    for b in range(3):
        assert int(valid[b].sum()) == int(rel[b].sum())
        sel = set(int(i) for i in idx[b][valid[b]].tolist())
        assert sel == set(torch.nonzero(rel[b]).flatten().tolist())


def test_content_equivalence_oracle():
    # content (materialized) == full-with-relevance-key-mask at relevant cells;
    # non-relevant cells pass through unchanged.
    torch.manual_seed(10)
    h = w = 6
    n = h * w
    block = TransformerBlock(channels=16, heads=4, board_h=h, board_w=w, attention_scope="content").eval()
    with torch.no_grad():
        block.attn.relative_bias_table.normal_(std=0.3)
    block.attn.set_impl("materialized")
    x = torch.randn(3, n, 16)
    relevance = _random_relevance(3, n, seed=11, density=0.4)
    idx, valid = _select_content_tokens(relevance)
    with torch.no_grad():
        out = block(x, (relevance, idx, valid))
        ref = _dense_block_reference(block, x, relevance)
    rel3 = relevance.unsqueeze(-1)
    # Relevant cells match the dense oracle to fp32 tolerance.
    assert torch.allclose(out[rel3.expand_as(out)], ref[rel3.expand_as(ref)], atol=1e-5, rtol=1e-4)
    # Non-relevant cells are exactly the block input (identity residual).
    nonrel = ~rel3.expand_as(out)
    assert torch.equal(out[nonrel], x[nonrel])


def test_content_sdpa_equals_materialized():
    torch.manual_seed(12)
    h = w = 6
    n = h * w
    block = TransformerBlock(channels=16, heads=4, board_h=h, board_w=w, attention_scope="content").eval()
    with torch.no_grad():
        block.attn.relative_bias_table.normal_(std=0.3)
    x = torch.randn(2, n, 16)
    relevance = _random_relevance(2, n, seed=13, density=0.5)
    sel = _select_content_tokens(relevance)
    with torch.no_grad():
        block.attn.set_impl("materialized")
        out_mat = block(x, (relevance, *sel))
        block.attn.set_impl("sdpa")
        out_sdpa = block(x, (relevance, *sel))
    assert torch.allclose(out_mat, out_sdpa, atol=1e-5, rtol=1e-4)


def test_content_batch_invariance():
    # A sample's output must not depend on other samples sharing the batch (padding
    # to a larger K_pad must never leak). Run a low-relevance sample alone and again
    # batched with a near-full-relevance sample (which enlarges K_pad).
    torch.manual_seed(14)
    n = BOARD_SIZE * BOARD_SIZE
    block = TransformerBlock(channels=16, heads=4, board_h=BOARD_SIZE, board_w=BOARD_SIZE,
                             attention_scope="content").eval()
    with torch.no_grad():
        block.attn.relative_bias_table.normal_(std=0.2)
    block.attn.set_impl("materialized")
    x_small = torch.randn(1, n, 16)
    rel_small = torch.zeros(1, n, dtype=torch.bool)
    rel_small[0, torch.arange(0, n, 137)] = True  # sparse relevance -> small K
    x_big = torch.randn(1, n, 16)
    rel_big = torch.rand(1, n) < 0.8  # dense relevance -> large K (bigger K_pad)

    def run(x, rel):
        idx, valid = _select_content_tokens(rel)
        with torch.no_grad():
            return block(x, (rel, idx, valid))

    out_alone = run(x_small, rel_small)
    # Batched together: the big sample forces a larger K_pad bucket for BOTH.
    x_batch = torch.cat([x_small, x_big], 0)
    rel_batch = torch.cat([rel_small, rel_big], 0)
    idx_b, valid_b = _select_content_tokens(rel_batch)
    with torch.no_grad():
        out_batch = block(x_batch, (rel_batch, idx_b, valid_b))
    assert torch.allclose(out_alone[0], out_batch[0], atol=1e-5, rtol=1e-4)


def test_content_zero_relevant_is_identity():
    torch.manual_seed(15)
    n = 6 * 6
    block = TransformerBlock(channels=16, heads=4, board_h=6, board_w=6, attention_scope="content").eval()
    with torch.no_grad():
        block.attn.relative_bias_table.normal_(std=0.3)
    x = torch.randn(2, n, 16)
    relevance = torch.zeros(2, n, dtype=torch.bool)  # nothing relevant
    idx, valid = _select_content_tokens(relevance)
    with torch.no_grad():
        out = block(x, (relevance, idx, valid))
    assert torch.equal(out, x)  # whole block is identity


def test_content_network_forward_and_backward():
    torch.manual_seed(16)
    net = RestnetNetwork(channels=16, blocks_type="R_T_R_T", attention_scope="content",
                         short_term_value_horizons=(1,))
    net.train()
    x = torch.randn(2, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    # Give the input planes real occupied/legal structure so relevance is non-trivial.
    x[:, 0, 20, 20] = 1.0
    x[:, 3, 18:23, 18:23] = 1.0
    out = net(x)
    assert out["policy"].shape == (2, BOARD_AREA)
    assert out["value"].shape == (2, VALUE_BINS)
    batch = {
        "policy": torch.softmax(torch.randn(2, BOARD_AREA), 1),
        "opp_policy": torch.softmax(torch.randn(2, BOARD_AREA), 1),
        "value": torch.empty(2).uniform_(-1, 1),
        "stvalue_1": torch.empty(2).uniform_(-1, 1),
    }
    total, _ = model1_loss(out, batch, opp_policy_weight=0.25, short_term_value_weight=0.25)
    assert torch.isfinite(total)
    total.backward()
    rel = next(m.relative_bias_table for m in net.modules() if isinstance(m, RelPosMHSA))
    assert rel.grad is not None and torch.isfinite(rel.grad).all()


def test_content_checkpoint_cross_loads_with_disk_and_full():
    torch.manual_seed(17)
    content = RestnetNetwork(channels=16, blocks_type="R_T_R", attention_scope="content")
    disk = RestnetNetwork(channels=16, blocks_type="R_T_R", attention_scope="disk")
    full = RestnetNetwork(channels=16, blocks_type="R_T_R", attention_scope="full")
    # Identical state_dict keys across all three scopes (content buffers persistent=False).
    assert set(content.state_dict()) == set(disk.state_dict()) == set(full.state_dict())
    for key in content.state_dict():
        assert "disk_flat" not in key
    missing, unexpected = content.load_state_dict(disk.state_dict(), strict=True)
    assert not missing and not unexpected
    missing, unexpected = disk.load_state_dict(content.state_dict(), strict=True)
    assert not missing and not unexpected


# =========================================================================== #
# Stage D - inference-only KV-gathered disk attention.
# =========================================================================== #
def test_kv_gathered_disk_matches_dense_disk_attention():
    torch.manual_seed(18)
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=BOARD_SIZE, board_w=BOARD_SIZE,
                      attention_scope="disk").eval()
    with torch.no_grad():
        mhsa.relative_bias_table.normal_(std=0.2)
    x = torch.randn(2, BOARD_AREA, 16)
    with torch.no_grad():
        mhsa.set_impl("sdpa")
        mhsa.set_kv_gather(False)
        dense = mhsa(x)
        mhsa.set_kv_gather(True)
        gathered = mhsa(x)
    assert torch.allclose(gathered, dense, atol=1e-5, rtol=1e-4)


def test_kv_gather_is_eval_only_training_uses_dense_path():
    torch.manual_seed(19)
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=6, board_w=6,
                      attention_scope="disk")
    x = torch.randn(2, 36, 16)
    mhsa.train()
    with torch.enable_grad():
        mhsa.set_kv_gather(False)
        dense = mhsa(x)
        mhsa.set_kv_gather(True)
        gathered_flag = mhsa(x)
    assert torch.allclose(gathered_flag, dense, atol=1e-6)


def test_kv_gather_buffers_not_in_state_dict():
    dense = RelPosMHSA(channels=16, num_heads=4, board_h=BOARD_SIZE, board_w=BOARD_SIZE,
                       attention_scope="disk")
    gathered = RelPosMHSA(channels=16, num_heads=4, board_h=BOARD_SIZE, board_w=BOARD_SIZE,
                          attention_scope="disk")
    gathered.set_kv_gather(True)
    assert set(dense.state_dict()) == set(gathered.state_dict())
    for key in gathered.state_dict():
        assert "disk_indices" not in key and "kv_bias" not in key


# =========================================================================== #
# Frozen inference bias (torch.compile-friendly static bias buffers).
# =========================================================================== #
def test_frozen_bias_matches_unfrozen_dense_and_kv():
    torch.manual_seed(20)
    for kv_gather in (False, True):
        mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=BOARD_SIZE, board_w=BOARD_SIZE,
                          attention_scope="disk").eval()
        with torch.no_grad():
            mhsa.relative_bias_table.normal_(std=0.2)
        mhsa.set_kv_gather(kv_gather)
        x = torch.randn(2, BOARD_AREA, 16)
        with torch.no_grad():
            unfrozen = mhsa(x)
            mhsa.freeze_inference_bias(dtype=torch.float32, device=torch.device("cpu"))
            assert mhsa._bias_frozen
            frozen = mhsa(x)
        assert torch.allclose(frozen, unfrozen, atol=1e-6), f"kv_gather={kv_gather}"


def test_set_kv_gather_clears_frozen_bias():
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=BOARD_SIZE, board_w=BOARD_SIZE,
                      attention_scope="disk").eval()
    mhsa.freeze_inference_bias(dtype=torch.float32, device=torch.device("cpu"))
    assert mhsa._bias_frozen
    mhsa.set_kv_gather(True)
    assert not mhsa._bias_frozen and mhsa._frozen_bias is None


def test_frozen_kv_gather_forward_is_torch_compile_traceable():
    """fullgraph compile of the frozen KV-gather path must NOT raise the
    data-dependent-guard error that the runtime bias cache triggered. CPU trace
    reproduces the dynamo failure mode device-independently."""

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    torch.manual_seed(21)
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=BOARD_SIZE, board_w=BOARD_SIZE,
                      attention_scope="disk").eval()
    with torch.no_grad():
        mhsa.relative_bias_table.normal_(std=0.2)
    mhsa.set_kv_gather(True)
    mhsa.freeze_inference_bias(dtype=torch.float32, device=torch.device("cpu"))
    x = torch.randn(2, BOARD_AREA, 16)
    with torch.no_grad():
        eager = mhsa(x)
        compiled = torch.compile(mhsa, fullgraph=True, dynamic=False)
        got = compiled(x)
    assert torch.allclose(got, eager, atol=1e-5, rtol=1e-4)
