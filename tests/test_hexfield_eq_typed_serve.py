"""Serve-path gates for the Phase-B mixed quotient fiber.

The package architecture is import-time configured, so pytest stays a small
controller and every substantive check runs in a fresh child process.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHONPATH = os.pathsep.join(
    str(ROOT / path)
    for path in (
        "packages/hexfield_eq/python",
        "packages/hexo_engine/python",
        "packages/hexo_utils/python",
    )
)


def _child_env(*, cuda: bool) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HEXFIELD")
    }
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": PYTHONPATH,
            "HEXFIELD_EQ_FEATURE_VERSION": "1",
            "HEXFIELD_EQ_TYPE_SIG": "reg:4,mirror:6,point:2,axis:8,triv:8",
            "HEXFIELD_EQ_ATTN_ORBIT": "8",
            "HEXFIELD_EQ_CHANNELS": "128",
            "HEXFIELD_EQ_TRUNK": "CLA",
            "HEXFIELD_EQ_REG_LANE": "1",
            "HEXFIELD_EQ_REG_TOK_READ": "0",
            "HEXFIELD_EQ_SUPPORT_RADIUS": "4",
            "HEXFIELD_EQ_RAY_BLOCKERS": "1",
            "HEXFIELD_SERVE_FLEX": "1" if cuda else "0",
            "HEXFIELD_FLEX_PAIR": "1" if cuda else "0",
            "HEXFIELD_TRITON_CONV": "1" if cuda else "0",
            "HEXFIELD_TRITON_CONV_LN": "1" if cuda else "0",
            "HEXFIELD_TRITON_ATTN": "1" if cuda else "0",
            "HEXFIELD_EQ_TRITON_RAY": "1" if cuda else "0",
            "HEXFIELD_SERVE_HALF": "1" if cuda else "0",
            "HEXFIELD_CUDA_GRAPHS": "1" if cuda else "0",
            "HEXFIELD_NO_COMPILE": "1",
        }
    )
    if not cuda:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def _run_child(mode: str, *, cuda: bool = False) -> str:
    proc = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--child", mode],
        cwd=ROOT,
        env=_child_env(cuda=cuda),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"typed serve child {mode!r} failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc.stdout


def test_mixed_eager_fold_cache_and_cpu_evaluator() -> None:
    assert "typed-serve-cpu: ok" in _run_child("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="typed CUDA serve gate")
def test_mixed_half_kernels_and_cuda_graphs() -> None:
    assert "typed-serve-cuda: ok" in _run_child("cuda", cuda=True)


def _board_tensors():
    import numpy as np

    from hexfield_eq.constants import DIRECTIONS, NUM_FEATURES, RAYLEN_SLOTS
    from hexfield_eq.geometry import disk_offsets

    cells = disk_offsets(1)
    nodes = len(cells)
    by_coord = {coord: index for index, coord in enumerate(cells)}
    nbr = torch.full((1, nodes, 6), nodes, dtype=torch.long)
    for row, (q, r) in enumerate(cells):
        for direction, (dq, dr) in enumerate(DIRECTIONS):
            target = by_coord.get((q + dq, r + dr))
            if target is not None:
                nbr[0, row, direction] = target
    feats = torch.linspace(
        -0.75, 0.75, nodes * NUM_FEATURES, dtype=torch.float32
    ).reshape(1, nodes, NUM_FEATURES)
    coords = torch.tensor([[list(coord) for coord in cells]], dtype=torch.long)
    mask = torch.ones((1, nodes), dtype=torch.bool)
    raylen = torch.ones((1, nodes, RAYLEN_SLOTS), dtype=torch.uint8)

    wire_nbr = nbr[0].numpy().copy()
    wire_nbr[wire_nbr == nodes] = 0xFFFF
    payload = {
        "abi": 1,
        "shape": (1, nodes),
        "node_feats": feats[0].numpy().astype(np.float16).tobytes(),
        "node_qr": coords[0].numpy().astype(np.int16).tobytes(),
        "nbr": wire_nbr.astype(np.uint16).tobytes(),
        "raylen": raylen[0].numpy().tobytes(),
        "node_row_offsets": [0, nodes],
        "legal_counts": np.asarray([nodes], dtype=np.int32).tobytes(),
        "request_moves_left": True,
    }
    # Mirror the f16 wire rounding in the direct reference.
    feats = feats.half().float()
    return payload, (feats, nbr, mask, coords, raylen)


def _randomize(model, seed: int, scale: float = 0.08) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * scale)


def _cpu_child() -> None:
    import numpy as np

    from hexfield_eq.inference import HexfieldEvaluator
    from hexfield_eq.losses import decode_binned_value, decode_moves_left
    from hexfield_eq.model import HexfieldNet

    payload, tensors = _board_tensors()
    model = HexfieldNet().eval()
    _randomize(model, 31)

    # Runtime permutations (grad enabled) and no-grad folded caches must agree
    # under the magnitude-scaled tolerance documented by perm-fold.py.
    runtime = model.forward_policy_value(*tensors, request_moves_left=True)
    with torch.no_grad():
        folded = model.forward_policy_value(*tensors, request_moves_left=True)
    for key in runtime:
        tolerance = max(1.0e-4, 2.0e-6 * float(runtime[key].detach().abs().max()))
        torch.testing.assert_close(folded[key], runtime[key], atol=tolerance, rtol=0)

    # Every mixed coefficient participates in the version-keyed dense cache.
    modules = [model.attn_blocks[0].attn.q_proj, model.conv_blocks[0].conv1]
    for module in modules:
        with torch.no_grad():
            previous, _ = module._materialize()
            for parameter in module.coefficients.values():
                before = previous
                parameter.view(-1)[0].add_(0.125)
                current, _ = module._materialize()
                assert current is not before
                assert not torch.equal(current, before)
                previous = current

    reply = HexfieldEvaluator(model, device="cpu").evaluate_payload(dict(payload))
    with torch.no_grad():
        direct = model.forward_policy_value(*tensors, request_moves_left=True)
    value = decode_binned_value(direct["value"].float()).numpy()
    prior = torch.softmax(direct["policy"][0].float(), dim=0).numpy()
    moves = decode_moves_left(direct["moves_left"].float()).numpy()
    np.testing.assert_allclose(
        np.frombuffer(reply["values_bytes"], dtype=np.float32), value, atol=3e-3, rtol=0
    )
    np.testing.assert_allclose(
        np.frombuffer(reply["priors_bytes"], dtype=np.float32), prior, atol=3e-3, rtol=0
    )
    np.testing.assert_allclose(
        np.frombuffer(reply["moves_left_bytes"], dtype=np.float32),
        moves,
        atol=1.0,
        rtol=1e-3,
    )
    print("typed-serve-cpu: ok")


def _cuda_child() -> None:
    import copy
    import numpy as np

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA child launched without CUDA")
    from hexfield_eq import model as model_module
    from hexfield_eq.inference import HexfieldEvaluator
    from hexfield_eq.model import HexfieldNet

    payload, _ = _board_tensors()
    master = HexfieldNet().eval()
    _randomize(master, 37, scale=0.03)
    cpu_reply = HexfieldEvaluator(copy.deepcopy(master), device="cpu").evaluate_payload(
        dict(payload)
    )

    calls = {"conv": 0, "conv_ln": 0, "attn": 0, "ray": 0}

    def wrap(name, function):
        if function is None:
            raise AssertionError(f"{name} fused function unavailable")

        def counted(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)

        return counted

    model_module._hex_conv_fused = wrap("conv", model_module._hex_conv_fused)
    model_module._hex_conv_ln_fused = wrap("conv_ln", model_module._hex_conv_ln_fused)
    model_module._attn_pair_fused = wrap("attn", model_module._attn_pair_fused)
    model_module._ray_attn_fused = wrap("ray", model_module._ray_attn_fused)

    evaluator = HexfieldEvaluator(copy.deepcopy(master), device="cuda")
    assert evaluator._serve_half and evaluator._use_graphs
    gpu_reply = evaluator.evaluate_payload(dict(payload))
    torch.cuda.synchronize()
    assert evaluator._graph_cache is not None
    assert evaluator._graph_cache._graphs, "CUDA graph capture did not produce an entry"
    assert all(count > 0 for count in calls.values()), calls

    for key, tolerance in (
        ("values_bytes", 3e-3),
        ("priors_bytes", 3e-3),
        ("moves_left_bytes", 1.0),
    ):
        np.testing.assert_allclose(
            np.frombuffer(gpu_reply[key], dtype=np.float32),
            np.frombuffer(cpu_reply[key], dtype=np.float32),
            atol=tolerance,
            rtol=0 if key != "moves_left_bytes" else 1e-3,
        )
    print("typed-serve-cuda: ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--child", "cpu"]:
        _cpu_child()
    elif sys.argv[1:] == ["--child", "cuda"]:
        _cuda_child()
    else:
        raise SystemExit(f"unknown typed serve child arguments: {sys.argv[1:]}")
