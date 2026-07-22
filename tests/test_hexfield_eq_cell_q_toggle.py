"""Per-cell Q head toggle (HEXFIELD_EQ_CELL_Q / HexfieldNet(cell_q=...)).

main_5 drops the train-only per-cell Q head (owner ruling 2026-07-22). The
toggle is an arch knob: it changes the state-dict key set, so it must ride the
checkpoint meta (arch_meta) and be inferred deterministically by
``infer_net_kwargs_from_state_dict`` — meta first, key-set evidence second —
exactly like the register-lane toggles. These tests pin:

  * toggle-off drops every ``cell_q_*`` parameter and records ``cell_q: False``
    in arch_meta; the default build keeps head + meta True (no regression);
  * a toggle-off checkpoint rebuilds PURELY from meta and strict-loads
    bit-for-bit;
  * meta-less inference recovers the toggle from the key set alone (present ->
    on, absent -> off), so a foreign-env loader cannot mis-shape the rebuild.

Runs in the Windows CPU suite (CUDA_VISIBLE_DEVICES=-1) and the hexgt-build
venv via PYTHONPATH=packages/hexfield_eq/python. CPU-only, no rust.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexfield_eq.checkpoints import load_into, save_checkpoint
from hexfield_eq.model import HexfieldNet, infer_net_kwargs_from_state_dict


def _cell_q_keys(sd: dict) -> set[str]:
    return {k for k in sd if k.startswith("cell_q_")}


def test_default_build_keeps_cell_q_head() -> None:
    """Default (env absent -> HEXFIELD_EQ_CELL_Q=1) is unchanged: the head's
    three modules are present and arch_meta records True."""

    model = HexfieldNet().eval()
    keys = _cell_q_keys(model.state_dict())
    assert any(k.startswith("cell_q_conv.") for k in keys)
    assert any(k.startswith("cell_q_expand.") for k in keys)
    assert any(k.startswith("cell_q_head.") for k in keys)
    assert model.arch_meta()["cell_q"] is True


def test_cell_q_off_drops_keys_and_records_meta() -> None:
    model = HexfieldNet(cell_q=False).eval()
    assert _cell_q_keys(model.state_dict()) == set()
    assert model.arch_meta()["cell_q"] is False


def test_cell_q_off_rebuild_from_meta_strict_load_bitwise(tmp_path: Path) -> None:
    """Build off -> save -> rebuild PURELY from persisted meta -> strict load
    succeeds and every tensor is bit-identical (mirrors the checkpoint-meta
    round-trip test)."""

    torch.manual_seed(0)
    model = HexfieldNet(cell_q=False).eval()
    orig_sd = model.state_dict()

    ckpt = tmp_path / "epoch_000005.pt"
    save_checkpoint(ckpt, model=model, optimizer=None, epoch=5, extra={"run": "cq_test"})

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    kwargs = infer_net_kwargs_from_state_dict(payload["model"], payload["meta"])
    assert kwargs["cell_q"] is False

    rebuilt = HexfieldNet(**kwargs)
    # Identical key set BEFORE loading — this is what makes strict load succeed.
    assert set(rebuilt.state_dict().keys()) == set(orig_sd.keys())
    load_into(rebuilt, payload, optimizer=None)
    rebuilt_sd = rebuilt.state_dict()
    for key, val in orig_sd.items():
        assert torch.equal(rebuilt_sd[key], val), f"tensor mismatch after reload: {key}"


@pytest.mark.parametrize("toggle", [True, False])
def test_metaless_inference_recovers_toggle_from_key_set(toggle: bool) -> None:
    """No meta at all: the key set is affirmative evidence either way, so a
    foreign-env process (opposite HEXFIELD_EQ_CELL_Q) still rebuilds the right
    shape."""

    sd = HexfieldNet(cell_q=toggle).state_dict()
    kwargs = infer_net_kwargs_from_state_dict(sd, meta=None)
    assert kwargs["cell_q"] is toggle


def test_cell_q_off_train_forward_omits_component() -> None:
    """The train forward() must not emit (or touch) the head when off: losses.py
    keys the component on ``"cell_q" in outputs``, so emission IS the contract.
    Exercised through the model's own pair build on a toy geometry."""

    from hexfield_eq.constants import NUM_FEATURES

    torch.manual_seed(0)
    b, n = 2, 19
    feats = torch.randn(b, n, NUM_FEATURES)
    g = torch.Generator().manual_seed(1)
    coords = torch.randint(-4, 5, (b, n, 2), generator=g, dtype=torch.int64)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[:, n - 3 :] = False
    # batching.py convention: (B, N, 6) neighbor indices, missing -> N (the
    # appended zero row); all-missing = isolated cells, enough for a shape smoke.
    nbr = torch.full((b, n, 6), n, dtype=torch.int64)

    for toggle, expect in ((True, True), (False, False)):
        model = HexfieldNet(cell_q=toggle).eval()
        with torch.no_grad():
            out = model(feats, nbr, mask, coords)
        assert ("cell_q" in out) is expect
        assert "policy" in out and "value" in out and "soft_policy" in out
