"""hexgnn global-pooled value readout: the value head reads [SIDE hub | PMA_k]
over all post-GNN node embeddings (the transformer-free lineage's only global
readout). Covers shape/wiring, permutation- (hence D6-) invariance of the PMA
pool, the PMA-only toggle, and the checkpoint-expansion 'no cold start' property.
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
_HEXGNN = ROOT / "packages" / "hexgnn" / "python"
if str(_HEXGNN) not in sys.path:
    sys.path.insert(0, str(_HEXGNN))


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
    from hexgnn.graph_build import batch_from_states

    return batch_from_states([_replay(coords)], n=n)


def _model(token_dim=48, **kw):
    from hexgnn.architecture import HexgnnNetwork

    return HexgnnNetwork(token_dim=token_dim, gnn_layers=2, attention_heads=4, **kw)


def test_value_head_reads_side_plus_pma_seeds() -> None:
    _torch()

    td = 48
    model = _model(token_dim=td)  # default k=2
    k = model.value_pma_seeds
    assert k == 2
    assert tuple(model.value_pool.seeds.shape) == (k, td)
    assert model.value_head[0].weight.shape[1] == (1 + k) * td
    from hexgnn.constants import VALUE_BINS

    assert model.value_head[-1].weight.shape[0] == VALUE_BINS
    m1 = _model(token_dim=td, value_pma_seeds=1)
    assert m1.value_pool.seeds.shape[0] == 1
    assert m1.value_head[0].weight.shape[1] == 2 * td


def test_pma_pool_output_shape() -> None:
    torch = _torch()
    model = _model()
    model.eval()
    batch = _batch(_random_coords(14, 14))
    g = int(batch["num_graphs"])
    with torch.no_grad():
        node_emb = model._encode_nodes(batch)
        pooled = model.value_pool(node_emb, batch["node_graph"], g)
    assert pooled.shape == (g, model.value_pma_seeds * model.token_dim)


def test_value_readout_pma_only_toggle() -> None:
    torch = _torch()
    model = _model(value_head_use_side=False)
    model.eval()
    batch = _batch(_random_coords(11, 14))
    g = int(batch["num_graphs"])
    with torch.no_grad():
        node_emb = model._encode_nodes(batch)
        vr = model._value_readout(batch, node_emb)
        out = model(batch)
    # PMA-only readout width = k * token_dim (no SIDE block).
    assert vr.shape == (g, model.value_pma_seeds * model.token_dim)
    assert model.value_head[0].weight.shape[1] == model.value_pma_seeds * model.token_dim
    from hexgnn.constants import VALUE_BINS

    assert out["value"].shape == (g, VALUE_BINS)


def test_forward_value_shape_and_readout_width() -> None:
    torch = _torch()
    model = _model()
    model.eval()
    batch = _batch(_random_coords(11, 14))
    with torch.no_grad():
        out = model(batch)
        node_emb = model._encode_nodes(batch)
        vr = model._value_readout(batch, node_emb)
    g = int(batch["num_graphs"])
    from hexgnn.constants import VALUE_BINS

    assert out["value"].shape == (g, VALUE_BINS)
    assert vr.shape == (g, (1 + model.value_pma_seeds) * model.token_dim)


def test_pma_value_readout_is_permutation_invariant() -> None:
    """Shuffling node order within a graph must not change the PMA value readout
    (Set-Transformer PMA is permutation-invariant; D6 permutes nodes bijectively,
    so this is the readout's D6-invariance property)."""

    torch = _torch()
    model = _model()
    model.eval()
    batch = _batch(_random_coords(23, 16))
    with torch.no_grad():
        node_emb = model._encode_nodes(batch)
        base = model._value_readout(batch, node_emb)
        perm = torch.randperm(node_emb.shape[0])
        shuffled = dict(batch)
        shuffled_emb = node_emb.index_select(0, perm)
        shuffled["node_graph"] = batch["node_graph"].index_select(0, perm)
        shuffled["node_type"] = batch["node_type"].index_select(0, perm)
        shuf = model._value_readout(shuffled, shuffled_emb)
    assert torch.allclose(base, shuf, atol=1e-5)


def test_expand_value_readout_gives_identical_output() -> None:
    torch = _torch()
    import torch.nn.functional as F
    from hexgnn.architecture import expand_value_readout_columns

    td = 48
    model = _model(token_dim=td)
    model.eval()
    batch = _batch(_random_coords(37, 14))

    torch.manual_seed(7)
    w_old = torch.randn(td, td)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["value_head.0.weight"] = w_old.clone()

    with torch.no_grad():
        node_emb = model._encode_nodes(batch)
        side = model._graph_readout(batch, node_emb)
        b0 = sd["value_head.0.bias"]
        h = F.relu(side @ w_old.t() + b0)
        ref = h @ sd["value_head.2.weight"].t() + sd["value_head.2.bias"]

    expanded = expand_value_readout_columns(model, sd)
    assert expanded is True
    assert tuple(sd["value_head.0.weight"].shape) == (td, 3 * td)
    assert torch.equal(sd["value_head.0.weight"][:, :td], w_old)
    assert torch.count_nonzero(sd["value_head.0.weight"][:, td:]) == 0

    info = model.load_state_dict(sd, strict=False)
    assert not info.unexpected_keys
    assert not info.missing_keys  # hexgnn has no STV heads to leave un-loaded

    with torch.no_grad():
        out = model(batch)["value"]
    assert torch.allclose(out, ref, atol=1e-5), "value output changed after expansion (cold start!)"


def test_expand_value_readout_is_noop_when_already_wide() -> None:
    import torch
    from hexgnn.architecture import expand_value_readout_columns

    model = _model(token_dim=32)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    before = sd["value_head.0.weight"].clone()
    assert expand_value_readout_columns(model, sd) is False
    assert torch.equal(sd["value_head.0.weight"], before)


def test_expand_value_readout_rejects_unexpected_shape() -> None:
    import torch
    from hexgnn.architecture import expand_value_readout_columns

    model = _model(token_dim=32)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["value_head.0.weight"] = torch.randn(32, 17)
    with pytest.raises(ValueError):
        expand_value_readout_columns(model, sd)
