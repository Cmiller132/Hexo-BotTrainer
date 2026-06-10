"""Masked policy cross-entropy and weights-only initialize_from for restnet.

Covers the 2026-06-09 training-loss change: the self policy CE is masked to the
position's in-disk legal cells (matching the serve-time contract, where the
native evaluator gathers and renormalizes priors over legal cells only), the
mask is threaded through shard expansion as ``legal_mask``, and
``initialize_from`` warm starts load weights only (no optimizer/train state).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
_RESTNET = ROOT / "packages" / "dense_cnn_restnet" / "python"
if str(_RESTNET) not in sys.path:
    sys.path.insert(0, str(_RESTNET))

torch = pytest.importorskip("torch")
F = torch.nn.functional
d6 = importlib.import_module("dense_cnn_restnet.d6")
losses = importlib.import_module("dense_cnn_restnet.losses")
samples = importlib.import_module("dense_cnn_restnet.samples")
compact_io = importlib.import_module("dense_cnn_restnet.compact_io")
checkpoints = importlib.import_module("dense_cnn_restnet.checkpoints")


def _aid(q: int, r: int) -> int:
    return int(d6.pack_coord_id(d6.Axial(q, r)))


# --- soft_cross_entropy mask semantics ---------------------------------------


def _manual_masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    rows = []
    for row_logits, row_target, row_mask in zip(logits, target, mask):
        sub_logits = row_logits[row_mask]
        sub_target = row_target[row_mask]
        sub_target = sub_target / sub_target.sum()
        rows.append(-(sub_target * F.log_softmax(sub_logits, dim=-1)).sum())
    return float(torch.stack(rows).mean())


def test_masked_ce_equals_restricted_softmax():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(3, 9, generator=generator)
    mask = torch.zeros(3, 9, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, 2:6] = True
    mask[2, ::2] = True
    target = torch.rand(3, 9, generator=generator) * mask
    out = losses.soft_cross_entropy(logits, target, mask=mask)
    assert out.dtype == torch.float32
    assert abs(float(out) - _manual_masked_ce(logits, target, mask)) < 1.0e-5


def test_masked_ce_upcasts_half_logits():
    # Under fp16 autocast the model emits half logits; the mask path must upcast
    # BEFORE the -1e9 fill (in fp16 it would saturate to -inf and 0 * -inf = NaN).
    logits = torch.randn(2, 6)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, :4] = True
    target = torch.rand(2, 6) * mask
    full = losses.soft_cross_entropy(logits, target, mask=mask)
    half = losses.soft_cross_entropy(logits.half(), target, mask=mask)
    assert torch.isfinite(half)
    assert abs(float(half) - float(full)) < 5.0e-3  # half-precision logits only


def test_masked_ce_rejects_target_mass_outside_mask():
    logits = torch.zeros(1, 4)
    mask = torch.tensor([[True, True, False, False]])
    target = torch.tensor([[0.5, 0.0, 0.5, 0.0]])
    with pytest.raises(ValueError, match="outside the mask"):
        losses.soft_cross_entropy(logits, target, mask=mask)


def test_masked_ce_zero_gradient_on_masked_logits():
    logits = torch.randn(2, 7, requires_grad=True)
    mask = torch.zeros(2, 7, dtype=torch.bool)
    mask[:, :3] = True
    target = torch.rand(2, 7) * mask
    losses.soft_cross_entropy(logits, target, mask=mask).backward()
    assert torch.allclose(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
    assert logits.grad[mask].abs().sum() > 0


def test_model1_loss_uses_legal_mask_and_stays_optional():
    generator = torch.Generator().manual_seed(11)
    outputs = {
        "policy": torch.randn(2, 8, generator=generator),
        "value": torch.randn(2, 65, generator=generator),
    }
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :4] = True
    batch = {
        "policy": torch.rand(2, 8, generator=generator) * mask,
        "value": torch.zeros(2),
    }
    unmasked_total, _ = losses.model1_loss(outputs, batch)
    masked_total, masked_components = losses.model1_loss(outputs, {**batch, "legal_mask": mask})
    expected = _manual_masked_ce(outputs["policy"], batch["policy"], mask)
    assert abs(float(masked_components["policy"]) - expected) < 1.0e-5
    assert float(masked_total) != float(unmasked_total)


# --- legal_mask plumbing through shard expansion ------------------------------


def test_expand_shard_emits_legal_mask_covering_policy(tmp_path):
    legal = tuple((q, r) for q in range(-2, 3) for r in range(-2, 3) if abs(q + r) <= 2)
    sample = samples.Model1SampleData(
        game_id="g",
        turn_index=3,
        current_player="player0",
        phase="Placement",
        center=(0, 0),
        stones=((4, -2, "player1"),),
        legal_action_ids=tuple(_aid(q, r) for q, r in legal),
        policy=tuple((_aid(q, r), 1.0) for q, r in legal[:3]),
        root_prior_policy=tuple((_aid(q, r), 1.0) for q, r in legal),
        value=1.0,
        short_term_value=((1, 0.5),),
    )
    shard = tmp_path / "rows.npz"
    compact_io.write_compact_shard(shard, [sample] * 4, short_term_value_horizons=(1,))
    arrays = compact_io.expand_shard_to_arrays(shard, np.arange(4) % d6.D6_SIZE, (1,))
    assert arrays["legal_mask"].dtype == np.bool_
    assert arrays["legal_mask"].shape == arrays["policy"].shape
    assert arrays["legal_mask"].any(axis=1).all()
    # The training invariant the masked CE relies on: no policy mass off-mask.
    assert (arrays["policy"][~arrays["legal_mask"]] == 0.0).all()


# --- initialize_from is weights-only ------------------------------------------


def _loader_fixture(tmp_path):
    source = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-3)
    source(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    path = tmp_path / "warm.pt"
    torch.save(
        {"model_state": source.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": 0},
        path,
    )
    target = torch.nn.Linear(3, 2)
    components = SimpleNamespace(
        model=SimpleNamespace(model=target, optimizer=torch.optim.AdamW(target.parameters()), trainer=None)
    )
    return path, source, components


def test_initialize_from_loads_weights_only(tmp_path):
    path, source, components = _loader_fixture(tmp_path)
    ctx = SimpleNamespace(
        config=SimpleNamespace(checkpoint=SimpleNamespace(resume_from=None, initialize_from=path))
    )
    result = checkpoints.DenseCNNCheckpointLoader().load(path, ctx=ctx, components=components)
    assert result["status"] == "loaded"
    assert result["weights_only"] is True
    assert torch.equal(components.model.model.weight, source.weight)
    assert not components.model.optimizer.state_dict()["state"]  # fresh moments


def test_resume_from_restores_optimizer(tmp_path):
    path, _source, components = _loader_fixture(tmp_path)
    ctx = SimpleNamespace(
        config=SimpleNamespace(checkpoint=SimpleNamespace(resume_from=path, initialize_from=None))
    )
    result = checkpoints.DenseCNNCheckpointLoader().load(path, ctx=ctx, components=components)
    assert result["status"] == "loaded"
    assert result["weights_only"] is False
    assert components.model.optimizer.state_dict()["state"]  # moments restored
