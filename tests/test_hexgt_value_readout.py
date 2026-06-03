"""Global-pooled value readout (Phase 2): the value head reads
[SIDE hub | mean-pool | max-pool] over all node embeddings, not just the SIDE
token. Covers shape/forward, permutation- (hence D6-) invariance of the pooling,
and the checkpoint-expansion 'no cold start' property that lets the pre-pool
epoch-42 checkpoint load with an IDENTICAL first-step value output.

(End-to-end D6 value-invariance through the shared Rust path is additionally
gated by test_hexgt_d6::test_model_is_d6_equivariant.)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _torch():
    return pytest.importorskip("torch")


def _eng():
    return importlib.import_module("hexo_engine.api")


def _replay(coords):
    eng = _eng()
    from hexo_engine.types import AxialCoord, PlacementAction

    state = eng.new_game()
    for q, r in coords:
        eng.apply_action(state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return state


def _random_coords(seed: int, length: int):
    import random

    eng = _eng()
    from hexo_engine.types import unpack_coord_id

    rng = random.Random(seed)
    state = eng.new_game(seed=seed)
    coords = []
    for _ in range(length):
        if eng.terminal(state) is not None:
            break
        acts = list(eng.legal_actions(state))
        if not acts:
            break
        a = rng.choice(acts)
        c = unpack_coord_id(int(eng.action_id(a)))
        coords.append((int(c.q), int(c.r)))
        eng.apply_action(state, a)
    return coords


def _batch(coords, n=4):
    from hexo_models.hexgt.graph_build import batch_from_states

    return batch_from_states([_replay(coords)], n=n)


def _model(token_dim=48, **kw):
    from hexo_models.hexgt.architecture import HexgtNetwork

    return HexgtNetwork(
        token_dim=token_dim, gnn_layers=2, ctx_layers=2, attention_heads=4, ffn_dim=64, **kw
    )


# --- shape / wiring -----------------------------------------------------------

def test_value_head_input_is_three_blocks() -> None:
    torch = _torch()
    from hexo_models.hexgt.architecture import VALUE_READOUT_MULT

    td = 48
    model = _model(token_dim=td)
    assert VALUE_READOUT_MULT == 3
    # first Linear of the value head reads [side|mean|max] = 3*token_dim.
    assert model.value_head[0].weight.shape[1] == VALUE_READOUT_MULT * td
    # STV/value bin width unchanged.
    from hexo_models.hexgt.constants import VALUE_BINS

    assert model.value_head[-1].weight.shape[0] == VALUE_BINS


def test_forward_value_shape_and_readout_width() -> None:
    torch = _torch()
    model = _model(short_term_value_horizons=(4,))
    model.eval()
    batch = _batch(_random_coords(11, 14))
    with torch.no_grad():
        out = model(batch)
        node_emb = model._encode_nodes(batch)
        vr = model._value_readout(batch, node_emb)
    g = int(batch["num_graphs"])
    from hexo_models.hexgt.constants import VALUE_BINS

    assert out["value"].shape == (g, VALUE_BINS)
    assert vr.shape == (g, 3 * model.token_dim)
    # STV head still reads the SIDE-hub width (token_dim), unchanged.
    assert out["stvalue_4"].shape == (g, VALUE_BINS)


# --- permutation (=> D6) invariance of the pooling ----------------------------

def test_global_pool_is_permutation_invariant() -> None:
    """Shuffling node order within a graph must not change the pooled value
    readout (mean/max are symmetric; D6 permutes nodes bijectively)."""

    torch = _torch()
    model = _model()
    model.eval()
    batch = _batch(_random_coords(23, 16))
    with torch.no_grad():
        node_emb = model._encode_nodes(batch)
        base = model._value_readout(batch, node_emb)
        # permute node embeddings + node_graph together; pooled result must match.
        perm = torch.randperm(node_emb.shape[0])
        shuffled = dict(batch)
        shuffled_emb = node_emb.index_select(0, perm)
        shuffled["node_graph"] = batch["node_graph"].index_select(0, perm)
        # SIDE-hub block needs node_type too (also permuted)
        shuffled["node_type"] = batch["node_type"].index_select(0, perm)
        shuf = model._value_readout(shuffled, shuffled_emb)
    assert torch.allclose(base, shuf, atol=1e-5)


# --- checkpoint expansion: identical first-step value (no cold start) ---------

def test_expand_value_readout_gives_identical_output() -> None:
    torch = _torch()
    import torch.nn.functional as F
    from hexo_models.hexgt.architecture import expand_value_readout_columns

    td = 48
    model = _model(token_dim=td)
    model.eval()
    batch = _batch(_random_coords(37, 14))

    # Synthesize a PRE-POOL checkpoint: value_head.0.weight is SIDE-only (td, td).
    torch.manual_seed(7)
    w_old = torch.randn(td, td)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["value_head.0.weight"] = w_old.clone()  # narrow, old shape

    # Reference: what the OLD SIDE-only head produced = head over the SIDE readout.
    with torch.no_grad():
        node_emb = model._encode_nodes(batch)
        side = model._graph_readout(batch, node_emb)
        b0 = sd["value_head.0.bias"]
        h = F.relu(side @ w_old.t() + b0)
        ref = h @ sd["value_head.2.weight"].t() + sd["value_head.2.bias"]

    expanded = expand_value_readout_columns(model, sd)
    assert expanded is True
    assert tuple(sd["value_head.0.weight"].shape) == (td, 3 * td)
    # old weight in the SIDE block, ZERO in mean/max blocks
    assert torch.equal(sd["value_head.0.weight"][:, :td], w_old)
    assert torch.count_nonzero(sd["value_head.0.weight"][:, td:]) == 0

    info = model.load_state_dict(sd, strict=False)
    assert not info.unexpected_keys
    assert all(k.startswith("short_term_value_heads") for k in info.missing_keys)

    with torch.no_grad():
        out = model(batch)["value"]
    assert torch.allclose(out, ref, atol=1e-5), "value output changed after expansion (cold start!)"


def test_expand_value_readout_is_noop_when_already_wide() -> None:
    from hexo_models.hexgt.architecture import expand_value_readout_columns

    model = _model(token_dim=32)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    before = sd["value_head.0.weight"].clone()
    assert expand_value_readout_columns(model, sd) is False
    import torch

    assert torch.equal(sd["value_head.0.weight"], before)


def test_expand_value_readout_rejects_unexpected_shape() -> None:
    import torch
    from hexo_models.hexgt.architecture import expand_value_readout_columns

    model = _model(token_dim=32)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["value_head.0.weight"] = torch.randn(32, 17)  # neither old (32) nor new (96) width
    with pytest.raises(ValueError):
        expand_value_readout_columns(model, sd)
