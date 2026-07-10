"""Checkpoint metadata gates for production quotient representations.

Every child process starts from a scrubbed ``HEXFIELD*`` environment before
importing :mod:`hexfield_eq`.  This is load-bearing: feature version, residual
signature, and attention orbit are import-time architecture inputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


NOMINATED = (
    ("reg:8,mirror:8,axis:4,triv:4", 8, 160),
    ("reg:4,mirror:6,point:2,axis:8,triv:8", 16, 128),
    ("reg:4,mirror:6,point:2,axis:8,triv:8", 8, 128),
)
ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATH = os.pathsep.join(
    str(ROOT / path)
    for path in (
        "packages/hexfield_eq/python",
        "packages/hexo_engine/python",
        "packages/hexo_utils/python",
    )
)

_BACKEND_ENV = {
    "HEXFIELD_SERVE_FLEX": "0",
    "HEXFIELD_TRAIN_FLEX": "0",
    "HEXFIELD_FLEX_PAIR": "0",
    "HEXFIELD_TRAIN_FLEX_PAIR": "0",
    "HEXFIELD_TRITON_CONV": "0",
    "HEXFIELD_TRITON_ATTN": "0",
    "HEXFIELD_TRITON_CONV_LN": "0",
    "HEXFIELD_EQ_TRITON_RAY": "0",
    "HEXFIELD_CUDA_GRAPHS": "0",
}


def _child_env(signature: str, attn_orbit: int, feature_version: int) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("HEXFIELD")}
    env.update(_BACKEND_ENV)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": PACKAGE_PATH,
            "HEXFIELD_EQ_GROUP_ORDER": "12",
            "HEXFIELD_EQ_TYPE_SIG": signature,
            "HEXFIELD_EQ_ATTN_ORBIT": str(attn_orbit),
            "HEXFIELD_EQ_FEATURE_VERSION": str(feature_version),
            "HEXFIELD_EQ_TRUNK": "CCLACCLACLA",
            "HEXFIELD_EQ_REG_LANE": "1",
            "HEXFIELD_EQ_REG_TOK_READ": "0",
            "HEXFIELD_EQ_SUPPORT_RADIUS": "4",
            "HEXFIELD_EQ_RAY_BLOCKERS": "1",
            # Phase-R may already be present when this suite is run.  Keep its
            # default path explicitly off without depending on its import.
            "HEXFIELD_EQ_RAYTAP": "0",
        }
    )
    return env


def _run_child(
    mode: str,
    *,
    signature: str,
    attn_orbit: int,
    feature_version: int = 1,
) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--child",
            mode,
            signature,
            str(attn_orbit),
            str(feature_version),
        ],
        env=_child_env(signature, attn_orbit, feature_version),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"typed checkpoint child failed ({mode}, {signature}, K={attn_orbit}, "
        f"feature-v{feature_version})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("signature,attn_orbit,channels", NOMINATED)
@pytest.mark.parametrize("feature_version,feature_width", ((1, 25), (2, 46)))
def test_nominated_arch_meta_round_trip_both_feature_versions(
    signature: str,
    attn_orbit: int,
    channels: int,
    feature_version: int,
    feature_width: int,
) -> None:
    got = _run_child(
        "roundtrip",
        signature=signature,
        attn_orbit=attn_orbit,
        feature_version=feature_version,
    )
    assert got == {
        "attn_orbit": attn_orbit,
        "channels": channels,
        "feature_width": feature_width,
        "type_sig": signature,
    }


def test_state_dict_only_inference_is_pure_regular_only() -> None:
    got = _run_child(
        "no-meta",
        signature=NOMINATED[1][0],
        attn_orbit=NOMINATED[1][1],
    )
    assert got == {
        "foreign_stem_shape": [7, 25, 192],
        "mixed_rejected": True,
        "partial_rejected": 2,
        "pure_attn_orbit": 16,
        "pure_type_sig": "reg:16",
    }


def test_strict_loaders_reject_exact_metadata_mismatches_and_accept_legacy() -> None:
    got = _run_child(
        "loaders",
        signature=NOMINATED[2][0],
        attn_orbit=NOMINATED[2][1],
    )
    assert got == {
        "checkpoints_legacy": True,
        "checkpoints_match": True,
        "prefit_legacy": True,
        "prefit_match": True,
        "rejected": 8,
    }


def test_warm_start_warns_on_signature_mismatch_only() -> None:
    got = _run_child(
        "warm-start",
        signature=NOMINATED[2][0],
        attn_orbit=NOMINATED[2][1],
    )
    assert got == {"exact_silent": True, "mismatch_warned": True}


def _guarded_imports():
    """Import production model code without letting optional Flex load Triton."""

    import torch

    optional_name = "torch.nn.attention.flex_attention"
    missing = object()
    previous = sys.modules.get(optional_name, missing)
    triton_before = {
        name for name in sys.modules if name == "triton" or name.startswith("triton.")
    }
    sys.modules[optional_name] = None
    try:
        from hexfield_eq import constants
        from hexfield_eq.checkpoints import load_into, warm_start_into
        from hexfield_eq.model import HexfieldNet, infer_net_kwargs_from_state_dict
    finally:
        if previous is missing:
            sys.modules.pop(optional_name, None)
        else:
            sys.modules[optional_name] = previous
    triton_after = {
        name for name in sys.modules if name == "triton" or name.startswith("triton.")
    }
    assert triton_after == triton_before
    # Importing the complete prefit harness pulls in trainer-side Torch
    # utilities which may themselves register Triton helpers on some Torch
    # builds.  The optional Flex import above is the model-import guard; load
    # prefit only after that assertion and reuse the already-imported model.
    from hexfield_eq.prefit import load_checkpoint

    assert not torch.cuda.is_available()
    return (
        constants,
        load_into,
        warm_start_into,
        HexfieldNet,
        infer_net_kwargs_from_state_dict,
        load_checkpoint,
    )


def _roundtrip(signature: str, attn_orbit: int, feature_version: int) -> dict:
    (
        constants,
        _load_into,
        _warm_start,
        net_cls,
        infer_kwargs,
        _load_checkpoint,
    ) = _guarded_imports()
    model = net_cls().eval()
    meta = model.arch_meta()
    assert meta["type_sig"] == signature
    assert meta["attn_orbit"] == attn_orbit
    assert meta["channels"] == constants.CHANNELS
    assert meta["in_channels"] == meta["feature_width"] == constants.NUM_FEATURES
    assert constants.FEATURE_VERSION == feature_version

    state = model.state_dict()
    kwargs = infer_kwargs(state, meta)
    assert kwargs["type_sig"] == signature
    assert kwargs["attn_orbit"] == attn_orbit
    rebuilt = net_cls(**kwargs).eval()
    rebuilt_state = rebuilt.state_dict()
    assert set(rebuilt_state) == set(state)
    assert {key: tuple(value.shape) for key, value in rebuilt_state.items()} == {
        key: tuple(value.shape) for key, value in state.items()
    }
    rebuilt.load_state_dict(state, strict=True)
    return {
        "type_sig": meta["type_sig"],
        "attn_orbit": meta["attn_orbit"],
        "channels": meta["channels"],
        "feature_width": meta["feature_width"],
    }


def _no_meta() -> dict:
    import torch

    (
        _constants,
        _load_into,
        _warm_start,
        net_cls,
        infer_kwargs,
        _load_checkpoint,
    ) = _guarded_imports()

    pure = net_cls(channels=192, type_sig="reg:16", attn_orbit=16).eval()
    pure_state = pure.state_dict()
    pure_kwargs = infer_kwargs(pure_state)
    assert pure_kwargs["type_sig"] == "reg:16"
    assert pure_kwargs["attn_orbit"] == 16
    pure_rebuilt = net_cls(**pure_kwargs).eval()
    pure_rebuilt.load_state_dict(pure_state, strict=True)
    # The process-global nominated build is C=128/K=16, while this legacy
    # checkpoint is pure reg:16 (C=192). The historical stem generator closes
    # over import-time CHANNELS, so this materialization is the regression gate
    # for signature-parameterized foreign pure-regular reconstruction.
    with torch.no_grad():
        pure_weight, pure_bias = pure.stem._materialize()
        rebuilt_weight, rebuilt_bias = pure_rebuilt.stem._materialize()
    assert tuple(pure_weight.shape) == (7, _constants.NUM_FEATURES, 192)
    torch.testing.assert_close(rebuilt_weight, pure_weight, atol=0, rtol=0)
    torch.testing.assert_close(rebuilt_bias, pure_bias, atol=0, rtol=0)

    mixed = net_cls().eval()
    try:
        infer_kwargs(mixed.state_dict())
    except ValueError as exc:
        message = str(exc)
        assert "mixed quotient checkpoint" in message
        assert "meta.type_sig" in message or "meta.attn_orbit" in message
    else:  # pragma: no cover - this is the safety property under test
        raise AssertionError("mixed no-meta inference did not fail loudly")

    partial_rejected = 0
    mixed_meta = mixed.arch_meta()
    for present, missing in (
        ("type_sig", "attn_orbit"),
        ("attn_orbit", "type_sig"),
    ):
        try:
            infer_kwargs(mixed.state_dict(), {present: mixed_meta[present]})
        except ValueError as exc:
            assert "quotient metadata is incomplete" in str(exc)
            assert missing in str(exc)
            partial_rejected += 1
        else:  # pragma: no cover - safety property under test
            raise AssertionError(f"foreign inference accepted meta missing {missing}")

    return {
        "foreign_stem_shape": list(pure_weight.shape),
        "pure_type_sig": pure_kwargs["type_sig"],
        "pure_attn_orbit": pure_kwargs["attn_orbit"],
        "mixed_rejected": True,
        "partial_rejected": partial_rejected,
    }


def _assert_rejected(call, field: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert field in str(exc)
    else:  # pragma: no cover - this is the safety property under test
        raise AssertionError(f"loader accepted mismatched {field}")


def _loaders(signature: str, attn_orbit: int) -> dict:
    import tempfile

    import torch

    (
        _constants,
        load_into,
        _warm_start,
        net_cls,
        _infer_kwargs,
        load_checkpoint,
    ) = _guarded_imports()
    model = net_cls().eval()
    state = model.state_dict()
    meta = model.arch_meta()
    legacy_meta = {
        key: value for key, value in meta.items() if key not in ("type_sig", "attn_orbit")
    }

    # Matching metadata and old metadata with both new fields absent load.
    load_into(model, {"model": state, "meta": dict(meta)}, optimizer=None)
    load_into(model, {"model": state, "meta": legacy_meta}, optimizer=None)

    rejected = 0
    for field, wrong in (("type_sig", "reg:16"), ("attn_orbit", attn_orbit + 8)):
        bad_meta = dict(meta)
        bad_meta[field] = wrong
        _assert_rejected(
            lambda bad_meta=bad_meta: load_into(
                model, {"model": state, "meta": bad_meta}, optimizer=None
            ),
            field,
        )
        rejected += 1
    for present, missing in (
        ("type_sig", "attn_orbit"),
        ("attn_orbit", "type_sig"),
    ):
        partial_meta = {**legacy_meta, present: meta[present]}
        _assert_rejected(
            lambda partial_meta=partial_meta: load_into(
                model, {"model": state, "meta": partial_meta}, optimizer=None
            ),
            missing,
        )
        rejected += 1

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"

        torch.save({"model": state, "meta": dict(meta)}, path)
        load_checkpoint(path, model)
        torch.save({"model": state, "meta": legacy_meta}, path)
        load_checkpoint(path, model)

        for field, wrong in (("type_sig", "reg:16"), ("attn_orbit", attn_orbit + 8)):
            bad_meta = dict(meta)
            bad_meta[field] = wrong
            torch.save({"model": state, "meta": bad_meta}, path)
            _assert_rejected(lambda: load_checkpoint(path, model), field)
            rejected += 1
        for present, missing in (
            ("type_sig", "attn_orbit"),
            ("attn_orbit", "type_sig"),
        ):
            partial_meta = {**legacy_meta, present: meta[present]}
            torch.save({"model": state, "meta": partial_meta}, path)
            _assert_rejected(lambda: load_checkpoint(path, model), missing)
            rejected += 1

    assert meta["type_sig"] == signature
    return {
        "checkpoints_match": True,
        "checkpoints_legacy": True,
        "prefit_match": True,
        "prefit_legacy": True,
        "rejected": rejected,
    }


def _warm_start() -> dict:
    import warnings

    (
        _constants,
        _load_into,
        warm_start,
        net_cls,
        _infer,
        _load_checkpoint,
    ) = _guarded_imports()
    target = net_cls(
        channels=128,
        type_sig=NOMINATED[2][0],
        attn_orbit=8,
        trunk_layout="CA",
    ).eval()
    source = net_cls(
        channels=160,
        type_sig=NOMINATED[0][0],
        attn_orbit=8,
        trunk_layout="CA",
    ).eval()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warm_start(target, target.state_dict())
    assert caught == []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = warm_start(target, source.state_dict())
    assert len(caught) == 1
    message = str(caught[0].message)
    assert caught[0].category is RuntimeWarning
    assert "missing=" in message and "unexpected=" in message
    assert "shape_mismatch=" in message
    assert "type signature likely differs" in message
    assert summary["missing"] or summary["unexpected"] or summary["shape_mismatch"]
    return {"exact_silent": True, "mismatch_warned": True}


def _child_main(argv: list[str]) -> None:
    if len(argv) != 5 or argv[0] != "--child":
        raise SystemExit("expected --child MODE TYPE_SIG ATTN_ORBIT FEATURE_VERSION")
    _, mode, signature, attn_orbit_raw, feature_version_raw = argv
    attn_orbit = int(attn_orbit_raw)
    feature_version = int(feature_version_raw)
    if mode == "roundtrip":
        result = _roundtrip(signature, attn_orbit, feature_version)
    elif mode == "no-meta":
        result = _no_meta()
    elif mode == "loaders":
        result = _loaders(signature, attn_orbit)
    elif mode == "warm-start":
        result = _warm_start()
    else:
        raise SystemExit(f"unknown child mode: {mode}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    _child_main(sys.argv[1:])
