from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEXO_MODELS_PATH = ROOT / "packages" / "hexo_models" / "python"
if str(HEXO_MODELS_PATH) not in sys.path:
    sys.path.insert(0, str(HEXO_MODELS_PATH))


def _torch() -> Any:
    return pytest.importorskip("torch")


def _api() -> SimpleNamespace:
    try:
        package = importlib.import_module("hexo_models.dense_cnn")
    except ModuleNotFoundError as exc:
        if exc.name == "hexo_models.dense_cnn":
            pytest.xfail("hexo_models.dense_cnn package is expected to land with the Model 1 implementation")
        raise

    return SimpleNamespace(
        Model1Network=package.Model1Network,
        Model1Config=package.Model1Config,
        HexConv2d=package.HexConv2d,
        GatedResBlock=package.GatedResBlock,
        decode_binned_value=package.decode_binned_value,
        binned_value_loss=package.binned_value_loss,
    )


def _make_model(api: SimpleNamespace, **overrides: Any) -> Any:
    attempts: list[Callable[[], Any]] = []

    try:
        config = api.Model1Config(**overrides)
    except TypeError:
        config = None
    if config is not None:
        attempts.extend(
            [
                lambda: api.Model1Network(config),
                lambda: api.Model1Network(config=config),
            ]
        )

    attempts.extend(
        [
            lambda: api.Model1Network(**overrides),
            lambda: api.Model1Network(),
        ]
    )

    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(str(exc))

    raise AssertionError(f"Could not construct Model1Network with overrides {overrides!r}: {errors}")


def _conv_weight(module: Any) -> Any:
    if hasattr(module, "weight"):
        return module.weight
    nested = getattr(module, "conv", None)
    if nested is not None and hasattr(nested, "weight"):
        return nested.weight
    raise AssertionError("HexConv2d must expose its convolution weight directly or through .conv.weight")


def _effective_weight(module: Any) -> Any:
    weight = _conv_weight(module)
    for method_name in ("effective_weight", "masked_weight"):
        method = getattr(module, method_name, None)
        if callable(method):
            return method().detach()
    for mask_name in ("mask", "hex_mask", "weight_mask", "kernel_mask"):
        mask = getattr(module, mask_name, None)
        if mask is not None:
            return weight.detach() * mask.detach().to(device=weight.device, dtype=weight.dtype)
    return weight.detach()


def _manual_binned_targets(values: Any, *, torch: Any, bins: int = 65) -> Any:
    clipped = values.clamp(-1.0, 1.0)
    positions = (clipped + 1.0) * ((bins - 1) / 2.0)
    lower = positions.floor().long().clamp(0, bins - 1)
    upper = positions.ceil().long().clamp(0, bins - 1)
    upper_weight = positions - lower.to(dtype=positions.dtype)
    lower_weight = 1.0 - upper_weight

    target = torch.zeros((values.shape[0], bins), dtype=values.dtype, device=values.device)
    target.scatter_add_(1, lower.unsqueeze(1), lower_weight.unsqueeze(1))
    target.scatter_add_(1, upper.unsqueeze(1), upper_weight.unsqueeze(1))
    return target


def test_default_model_architecture_uses_goal_1_shapes_and_depth() -> None:
    torch = _torch()
    api = _api()

    model = _make_model(api)
    model.eval()

    hex_convs = [module for module in model.modules() if isinstance(module, api.HexConv2d)]
    assert hex_convs, "Model1Network must start with and use HexConv2d layers"
    first_weight = _conv_weight(hex_convs[0])
    assert tuple(first_weight.shape[:2]) == (96, 13)

    gated_blocks = [module for module in model.modules() if isinstance(module, api.GatedResBlock)]
    assert len(gated_blocks) == 6

    with torch.no_grad():
        outputs = model(torch.randn(2, 13, 41, 41))

    assert outputs["policy"].shape == (2, 41 * 41)
    assert outputs["value"].shape == (2, 65)


def test_hexo_models_root_is_container_without_legacy_dense_cnn_wrappers() -> None:
    package = importlib.import_module("hexo_models")

    assert not hasattr(package, "Model1Network")
    assert not hasattr(package, "HexConv2d")
    assert not hasattr(package, "SampleBuffer")
    for legacy_module in (
        "architecture",
        "config",
        "d6",
        "losses",
        "plugin",
        "samples",
    ):
        with pytest.raises(ModuleNotFoundError) as exc_info:
            importlib.import_module(f"hexo_models.{legacy_module}")
        assert exc_info.value.name == f"hexo_models.{legacy_module}"


def test_hex_conv2d_masks_invalid_square_grid_corners() -> None:
    torch = _torch()
    api = _api()
    conv = api.HexConv2d(1, 1, kernel_size=3, padding=1, bias=False)

    with torch.no_grad():
        _conv_weight(conv).fill_(1.0)

    x = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    y = conv(x)

    expected_kernel = torch.ones(3, 3)
    expected_kernel[0, 0] = 0.0
    expected_kernel[2, 2] = 0.0
    assert torch.equal(_effective_weight(conv)[0, 0].cpu(), expected_kernel)
    assert y[0, 0, 1, 1].item() == pytest.approx(35.0)


def test_hex_conv2d_caches_masked_weight_only_for_inference() -> None:
    torch = _torch()
    api = _api()
    conv = api.HexConv2d(1, 1, kernel_size=3, padding=1, bias=False)
    conv.eval()

    with torch.no_grad():
        _conv_weight(conv).fill_(1.0)
        first = conv.masked_weight()
        second = conv.masked_weight()
        _conv_weight(conv)[0, 0, 1, 1] = 2.0
        third = conv.masked_weight()

    assert first.data_ptr() == second.data_ptr()
    assert third.data_ptr() != second.data_ptr()
    assert third[0, 0, 1, 1].item() == pytest.approx(2.0)

    conv.train()
    x = torch.ones((1, 1, 3, 3))
    y = conv(x).sum()
    y.backward()
    grad = _conv_weight(conv).grad[0, 0]
    assert grad[0, 0].item() == pytest.approx(0.0)
    assert grad[2, 2].item() == pytest.approx(0.0)
    assert grad[1, 1].item() != pytest.approx(0.0)


def test_inference_optimizer_folds_hex_convs_without_changing_outputs() -> None:
    torch = _torch()
    api = _api()
    architecture = importlib.import_module("hexo_models.dense_cnn.architecture")
    model = api.Model1Network(channels=8, blocks=1, lookahead_horizons=(1,)).eval()
    optimized = architecture.optimized_model1_for_inference(model).eval()
    inputs = torch.randn(2, model.in_channels, model.board_size, model.board_size)

    with torch.no_grad():
        expected = model.forward_policy_value(inputs)
        actual = optimized.forward_policy_value(inputs)

    assert not any(isinstance(module, api.HexConv2d) for module in optimized.modules())
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1.0e-5, atol=1.0e-6)


def test_gated_res_block_preserves_shape_and_exposes_sigmoid_gate() -> None:
    torch = _torch()
    api = _api()
    block = api.GatedResBlock(8)
    block.eval()
    x = torch.randn(2, 8, 41, 41)

    with torch.no_grad():
        y = block(x)

    assert y.shape == x.shape

    gate = getattr(block, "gate", None)
    assert gate is not None, "GatedResBlock must expose its gate branch as .gate"
    with torch.no_grad():
        gate_values = gate(x)

    assert gate_values.shape == x.shape
    assert torch.all(gate_values >= 0.0)
    assert torch.all(gate_values <= 1.0)


def test_forward_returns_policy_value_lookahead_and_opponent_policy_logits() -> None:
    torch = _torch()
    api = _api()
    model = _make_model(api, lookahead_horizons=(1, 4))
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(3, 13, 41, 41))

    assert outputs["policy"].shape == (3, 41 * 41)
    assert outputs["value"].shape == (3, 65)
    assert outputs["lookahead_1"].shape == (3, 65)
    assert outputs["lookahead_4"].shape == (3, 65)
    assert outputs["opp_policy"].shape == (3, 41 * 41)


def test_decode_binned_value_uses_65_softmax_bins_from_minus_one_to_one() -> None:
    torch = _torch()
    api = _api()
    logits = torch.stack(
        [
            torch.linspace(-2.0, 2.0, 65),
            torch.linspace(1.0, -1.0, 65),
        ]
    )

    decoded = api.decode_binned_value(logits)

    bins = torch.linspace(-1.0, 1.0, 65, dtype=logits.dtype, device=logits.device)
    expected = (torch.softmax(logits, dim=-1) * bins).sum(dim=-1)
    assert decoded.shape == (2,)
    assert torch.allclose(decoded, expected, atol=1.0e-6)


def test_binned_value_loss_interpolates_scalar_targets_between_bins() -> None:
    torch = _torch()
    api = _api()
    logits = torch.stack(
        [
            torch.linspace(-1.0, 1.0, 65),
            torch.linspace(1.0, -1.0, 65),
            torch.sin(torch.linspace(0.0, 3.14, 65)),
        ]
    )
    targets = torch.tensor([-1.0, 0.2, 1.0])

    loss = api.binned_value_loss(logits, targets)

    target_distribution = _manual_binned_targets(targets, torch=torch)
    expected = -(target_distribution * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    assert loss.shape == ()
    assert torch.allclose(loss, expected, atol=1.0e-6)


def test_binned_value_loss_respects_sample_mask() -> None:
    torch = _torch()
    api = _api()
    logits = torch.stack(
        [
            torch.linspace(-1.0, 1.0, 65),
            torch.linspace(1.0, -1.0, 65),
            torch.sin(torch.linspace(0.0, 3.14, 65)),
        ]
    )
    targets = torch.tensor([-1.0, 0.2, 1.0])
    mask = torch.tensor([1.0, 0.0, 1.0])

    loss = api.binned_value_loss(logits, targets, mask=mask)

    target_distribution = _manual_binned_targets(targets, torch=torch)
    per_item = -(target_distribution * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    expected = (per_item * mask).sum() / mask.sum()
    assert loss.shape == ()
    assert torch.allclose(loss, expected, atol=1.0e-6)
