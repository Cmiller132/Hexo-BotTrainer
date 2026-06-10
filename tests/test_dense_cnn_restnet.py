r"""CPU tests for the dense_cnn_restnet (faithful ResTNet) fork.

These cover the parts that differ from dense_cnn — the interleaved R/T trunk, the
relative-position attention, the SDPA <-> materialized equivalence, the R<->T
token/conv boundary, and head/loss compatibility with the dense_cnn binned-value
training contract — plus a from-scratch init sanity check and coexistence with
the original dense_cnn lineage. All tests are CPU-only.

Run (from the repo root), pointing PYTHONPATH at the worktree packages:

    $env:PYTHONPATH = "packages\dense_cnn_restnet\python;packages\hexo_train\python;packages\hexo_runner\python"
    python -m pytest tests\test_dense_cnn_restnet.py -q

(The file also self-bootstraps those source dirs onto sys.path, so a bare
`python -m pytest tests\test_dense_cnn_restnet.py` works too as long as the native
hexo_models/hexo_engine wheels are importable.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Self-bootstrap the worktree package source dirs (additive; never installs).
_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("dense_cnn_restnet", "hexo_train", "hexo_runner"):
    _src = _ROOT / "packages" / _pkg / "python"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from dense_cnn_restnet.architecture import (  # noqa: E402
    DEFAULT_BLOCKS_TYPE,
    RelPosMHSA,
    ResidualBlock,
    RestnetNetwork,
    TransformerBlock,
    conv_to_tokens,
    optimized_restnet_for_inference,
    parse_blocks_type,
    tokens_to_conv,
)
from dense_cnn_restnet.config import parse_model1_config  # noqa: E402
from dense_cnn_restnet.constants import BOARD_AREA, BOARD_SIZE, INPUT_CHANNELS, VALUE_BINS  # noqa: E402
from dense_cnn_restnet.losses import model1_loss  # noqa: E402

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Block-pattern parsing
# --------------------------------------------------------------------------- #
def test_parse_blocks_type_valid():
    assert parse_blocks_type("R_R_T_R_R_T") == ("R", "R", "T", "R", "R", "T")
    assert parse_blocks_type("R_R_R_T_R_R_T_R_R_T") == tuple("RRRTRRTRRT")
    assert parse_blocks_type(DEFAULT_BLOCKS_TYPE) == ("R", "R", "R", "T", "R", "R", "T", "R")
    assert parse_blocks_type("T") == ("T",)


@pytest.mark.parametrize("bad", ["", "  ", "R-R-T", "R_X_T", "R_RT", "Q"])
def test_parse_blocks_type_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_blocks_type(bad)


def test_default_trunk_is_8block_r3_family():
    net = RestnetNetwork(channels=16)
    kinds = [type(b).__name__ for b in net.blocks]
    assert kinds == [
        "ResidualBlock", "ResidualBlock", "ResidualBlock", "TransformerBlock",
        "ResidualBlock", "ResidualBlock", "TransformerBlock", "ResidualBlock",
    ]
    assert sum(k == "TransformerBlock" for k in kinds) == 2


# --------------------------------------------------------------------------- #
# Relative-position index + bias correctness (the easy-to-get-wrong part)
# --------------------------------------------------------------------------- #
def test_relative_index_matches_bruteforce_3x3():
    h = w = 3
    mhsa = RelPosMHSA(channels=4, num_heads=1, board_h=h, board_w=w)
    idx = mhsa.relative_index.view(h * w, h * w)
    for i in range(h * w):
        ri, ci = divmod(i, w)  # row-major token order, matching conv_to_tokens
        for j in range(h * w):
            rj, cj = divmod(j, w)
            expected = (ri - rj + (h - 1)) * (2 * w - 1) + (ci - cj + (w - 1))
            assert int(idx[i, j]) == expected
    # All indices land inside the table.
    assert int(idx.min()) == 0
    assert int(idx.max()) < (2 * h - 1) * (2 * w - 1)
    # The diagonal (offset 0,0) is the same table row for every token.
    assert len({int(idx[t, t]) for t in range(h * w)}) == 1


def test_relative_bias_gather_is_correct():
    h = w = 3
    heads = 2
    # "full" scope exercises the raw gather (no disk key mask); the mask is covered
    # separately in test_dense_cnn_restnet_attention.py.
    mhsa = RelPosMHSA(channels=4, num_heads=heads, board_h=h, board_w=w, attention_scope="full")
    # Fill the table with distinct per-(offset,head) values, then check the
    # (1, heads, N, N) materialized bias picks the right row for each pair.
    with torch.no_grad():
        rng = torch.arange(mhsa.relative_bias_table.numel(), dtype=torch.float32)
        mhsa.relative_bias_table.copy_(rng.view_as(mhsa.relative_bias_table))
    bias = mhsa._compute_relative_bias()  # (1, heads, N, N)
    idx = mhsa.relative_index.view(h * w, h * w)
    for head in range(heads):
        for i in (0, 4, 8):
            for j in (0, 1, 7):
                assert bias[0, head, i, j].item() == pytest.approx(
                    mhsa.relative_bias_table[int(idx[i, j]), head].item()
                )


# --------------------------------------------------------------------------- #
# SDPA <-> materialized equivalence (same params, both attention paths)
# --------------------------------------------------------------------------- #
def test_msa_sdpa_equals_materialized():
    torch.manual_seed(1)
    h = w = 6
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=h, board_w=w).eval()
    # Random (non-zero) relative bias so the additive-mask path is exercised.
    with torch.no_grad():
        mhsa.relative_bias_table.normal_(std=0.3)
    x = torch.randn(2, h * w, 16)
    with torch.no_grad():
        mhsa.set_impl("materialized")
        out_mat = mhsa(x)
        mhsa.set_impl("sdpa")
        out_sdpa = mhsa(x)
    assert torch.allclose(out_mat, out_sdpa, atol=1e-5, rtol=1e-4)


def test_relative_bias_inference_cache_matches_fresh():
    # The B-speedup eval cache for the additive bias mask must be numerically
    # identical to the fresh (training-path) gather, and must invalidate when the
    # learned table changes.
    torch.manual_seed(7)
    h = w = 6
    mhsa = RelPosMHSA(channels=16, num_heads=4, board_h=h, board_w=w)
    with torch.no_grad():
        mhsa.relative_bias_table.normal_(std=0.2)
    x = torch.randn(2, h * w, 16)
    mhsa.train()
    with torch.enable_grad():
        out_fresh = mhsa(x)
    mhsa.eval()
    with torch.no_grad():
        out_cached1 = mhsa(x)  # populates the cache
        out_cached2 = mhsa(x)  # hits the cache
    assert torch.allclose(out_fresh, out_cached1, atol=1e-6)
    assert torch.equal(out_cached1, out_cached2)  # cached path is bit-identical run-to-run
    # mutate the table -> cache must invalidate (version bump)
    with torch.no_grad():
        mhsa.relative_bias_table.add_(0.5)
        out_cached3 = mhsa(x)
    assert not torch.allclose(out_cached1, out_cached3)


def test_transformer_block_sdpa_equals_materialized():
    torch.manual_seed(2)
    h = w = 5
    blk = TransformerBlock(channels=16, heads=4, mlp_ratio=2, board_h=h, board_w=w).eval()
    with torch.no_grad():
        blk.attn.relative_bias_table.normal_(std=0.2)
    x = torch.randn(3, 16, h, w)  # conv-form input
    with torch.no_grad():
        blk.attn.set_impl("materialized")
        out_mat = blk(x)
        blk.attn.set_impl("sdpa")
        out_sdpa = blk(x)
    assert out_mat.shape == (3, h * w, 16)  # block emits tokens
    assert torch.allclose(out_mat, out_sdpa, atol=1e-5, rtol=1e-4)


def test_network_sdpa_equals_materialized():
    torch.manual_seed(3)
    net = RestnetNetwork(channels=16, blocks_type="R_T_R_T", short_term_value_horizons=(1,)).eval()
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
# R<->T token/conv boundary
# --------------------------------------------------------------------------- #
def test_token_conv_roundtrip():
    x = torch.randn(2, 7, 5, 5)
    tokens = conv_to_tokens(x)
    assert tokens.shape == (2, 25, 7)
    back = tokens_to_conv(tokens, 7, 5, 5)
    assert torch.equal(back, x)


def test_residual_block_accepts_tokens_and_conv():
    blk = ResidualBlock(8).eval()
    conv = torch.randn(2, 8, BOARD_SIZE, BOARD_SIZE)
    tok = conv_to_tokens(conv)
    with torch.no_grad():
        out_from_conv = blk(conv)
        out_from_tok = blk(tok)
    assert out_from_conv.shape == (2, 8, BOARD_SIZE, BOARD_SIZE)
    assert torch.allclose(out_from_conv, out_from_tok, atol=1e-6)


# --------------------------------------------------------------------------- #
# Full network forward shapes / head contract
# --------------------------------------------------------------------------- #
def test_forward_shapes_full_board():
    net = RestnetNetwork(channels=16, short_term_value_horizons=(1, 4, 8)).eval()
    x = torch.randn(2, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        out = net(x)
    assert out["policy"].shape == (2, BOARD_AREA)
    assert out["value"].shape == (2, VALUE_BINS)
    assert out["opp_policy"].shape == (2, BOARD_AREA)
    for h in (1, 4, 8):
        assert out[f"stvalue_{h}"].shape == (2, VALUE_BINS)


def test_forward_policy_value_subset():
    net = RestnetNetwork(channels=16, short_term_value_horizons=(1,)).eval()
    x = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        out = net.forward_policy_value(x)
    assert set(out) == {"policy", "value"}
    assert out["policy"].shape == (1, BOARD_AREA)
    assert out["value"].shape == (1, VALUE_BINS)


def test_trunk_returns_conv_map_even_when_last_block_is_T():
    # Last block T emits tokens; trunk() must reshape back to a conv map.
    net = RestnetNetwork(channels=8, blocks_type="R_T").eval()
    x = torch.randn(2, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        feats = net.trunk(x)
    assert feats.shape == (2, 8, BOARD_SIZE, BOARD_SIZE)


def test_input_validation_rejects_wrong_shape():
    net = RestnetNetwork(channels=8)
    with pytest.raises(ValueError):
        net(torch.randn(2, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE + 1))
    with pytest.raises(ValueError):
        net(torch.randn(INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE))  # rank 3


# --------------------------------------------------------------------------- #
# From-scratch init sanity (no NaNs) + loss/backward compatibility
# --------------------------------------------------------------------------- #
def test_fresh_init_has_finite_params_and_forward():
    net = RestnetNetwork(channels=16, short_term_value_horizons=(1, 4))
    for name, p in net.named_parameters():
        assert torch.isfinite(p).all(), name
    # Relative-bias tables start at zero (released-code init).
    for m in net.modules():
        if isinstance(m, RelPosMHSA):
            assert torch.count_nonzero(m.relative_bias_table) == 0
    net.eval()
    x = torch.randn(2, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        out = net(x)
    for key, value in out.items():
        assert torch.isfinite(value).all(), key


def _synthetic_batch(n: int, horizons=(1, 4)):
    policy = torch.softmax(torch.randn(n, BOARD_AREA), dim=1)
    opp = torch.softmax(torch.randn(n, BOARD_AREA), dim=1)
    value = torch.empty(n).uniform_(-1.0, 1.0)
    batch = {"policy": policy, "opp_policy": opp, "value": value}
    for h in horizons:
        batch[f"stvalue_{h}"] = torch.empty(n).uniform_(-1.0, 1.0)
    return batch


def test_loss_and_backward_match_binned_value_contract():
    horizons = (1, 4)
    net = RestnetNetwork(channels=16, blocks_type="R_R_T_R", short_term_value_horizons=horizons)
    net.train()
    x = torch.randn(3, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    outputs = net(x)
    batch = _synthetic_batch(3, horizons)
    total, components = model1_loss(outputs, batch, opp_policy_weight=0.25, short_term_value_weight=0.25)
    assert torch.isfinite(total)
    for key in ("policy", "value", "opp_policy", "stvalue_1", "stvalue_4", "total"):
        assert key in components and torch.isfinite(components[key]), key
    total.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads)
    # The relative-bias table must receive gradient (attention is actually used).
    rel = next(m.relative_bias_table for m in net.modules() if isinstance(m, RelPosMHSA))
    assert rel.grad is not None and torch.isfinite(rel.grad).all()


# --------------------------------------------------------------------------- #
# Inference-fold equivalence + config knobs
# --------------------------------------------------------------------------- #
def test_optimized_inference_matches_unfolded():
    net = RestnetNetwork(channels=16, blocks_type="R_T_R", short_term_value_horizons=(1,)).eval()
    folded = optimized_restnet_for_inference(net).eval()
    x = torch.randn(2, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        a = net.forward_policy_value(x)
        b = folded.forward_policy_value(x)
    assert torch.allclose(a["policy"], b["policy"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(a["value"], b["value"], atol=1e-4, rtol=1e-4)


def test_plain_residual_conv_builds_and_runs():
    net = RestnetNetwork(channels=16, blocks_type="R_T_R", residual_conv="plain").eval()
    x = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        out = net(x)
    assert out["policy"].shape == (1, BOARD_AREA)


def test_config_parses_restnet_arch_keys():
    cfg = parse_model1_config(
        {
            "architecture": {
                "channels": 96,
                "blocks_type": "R_R_R_T_R_R_T_R",
                "attention_heads": 4,
                "mlp_ratio": 2,
                "embed_kernel_size": 3,
                "residual_conv": "hex",
                "attention_impl": "sdpa",
                # Horizons must be even (turn-aligned) since the heads_v2
                # even-offset EMA; see test_dense_cnn_restnet_heads_v2.py.
                "short_term_value_horizons": [2, 6, 16],
            }
        }
    )
    arch = cfg.architecture
    assert arch.channels == 96
    assert arch.blocks_type == "R_R_R_T_R_R_T_R"
    assert arch.attention_heads == 4 and arch.mlp_ratio == 2
    assert arch.attention_impl == "sdpa"


def test_config_rejects_unknown_arch_key():
    with pytest.raises(ValueError):
        parse_model1_config({"architecture": {"nonsense_key": 1}})


# --------------------------------------------------------------------------- #
# Coexistence with the original dense_cnn lineage
# --------------------------------------------------------------------------- #
def test_coexists_with_dense_cnn():
    from hexo_models.dense_cnn.architecture import Model1Network
    from hexo_models.dense_cnn.plugin import get_plugin as dense_plugin
    from dense_cnn_restnet.plugin import get_plugin as restnet_plugin

    assert RestnetNetwork is not Model1Network
    assert dense_plugin().name == "hexo_models.dense_cnn"
    assert restnet_plugin().name == "dense_cnn_restnet"
    # The fork reuses the SAME native accelerator submodule (no Rust rebuild).
    from dense_cnn_restnet import rust_bridge as restnet_bridge
    from hexo_models.dense_cnn import rust_bridge as dense_bridge
    assert restnet_bridge._dense_cnn_module() is dense_bridge._dense_cnn_module()
