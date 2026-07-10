"""Phase-B D8 compatibility gates for production quotient representations.

The load-bearing rule is that the default equivariant build remains the exact
``reg:16`` architecture used by the live ``hexfield_eq_main_1`` checkpoint.
This file deliberately imports no ``hexfield_eq`` module in the pytest process:
all architecture knobs are import-time environment variables, so each probe
runs in a fresh, CUDA-hidden child interpreter with a controlled environment.

Two complete key/shape/dtype manifests are checked in below as compressed JSON
literals.  Compression keeps this test reviewable while still retaining every
key (rather than reducing the evidence to a hash):

* arm-4 pure regular: 321 tensors, captured from Phase A / epoch 20;
* GROUP_ORDER=1 passthrough: 179 tensors, captured from the Phase-A reference.

The optional live gate is enabled only by ``HEXFIELD_EQ_D8_CHECKPOINT``.  It
strict-loads that checkpoint in both this worktree and a separate Phase-A
reference tree, then compares raw policy/value/moves-left outputs on one shared
serialized input at the locked ``atol=1e-5, rtol=0`` tolerance.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import zlib

import numpy as np
import pytest


_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_REFERENCE_ROOT = (
    Path(r"E:\Hexo-BotTrainer-hexgt")
    if os.name == "nt"
    else Path("/mnt/e/Hexo-BotTrainer-hexgt")
)

_DISABLED_BACKENDS = {
    "HEXFIELD_TRITON_CONV": "0",
    "HEXFIELD_TRITON_CONV_LN": "0",
    "HEXFIELD_TRITON_ATTN": "0",
    "HEXFIELD_EQ_TRITON_RAY": "0",
    "HEXFIELD_SERVE_FLEX": "0",
    "HEXFIELD_TRAIN_FLEX": "0",
    "HEXFIELD_FLEX_PAIR": "0",
    "HEXFIELD_TRAIN_FLEX_PAIR": "0",
    "HEXFIELD_NO_COMPILE": "1",
}

_ARM4_ENV = {
    "HEXFIELD_EQ_CHANNELS": "192",
    "HEXFIELD_EQ_GROUP_ORDER": "12",
    "HEXFIELD_EQ_C_ORBIT": "16",
    "HEXFIELD_EQ_ATTENTION_HEADS": "3",
    "HEXFIELD_EQ_SUPPORT_RADIUS": "4",
    "HEXFIELD_EQ_TRUNK": "CCLACCLACLA",
    "HEXFIELD_EQ_REG_LANE": "1",
    "HEXFIELD_EQ_REG_TOK_READ": "0",
    # D-Delta2: these are the only new-project settings active for D8.
    "HEXFIELD_EQ_FEATURE_VERSION": "1",
    "HEXFIELD_EQ_RAYTAP": "0",
    **_DISABLED_BACKENDS,
}

_PASSTHROUGH_ENV = {
    "HEXFIELD_EQ_GROUP_ORDER": "1",
    "HEXFIELD_EQ_FEATURE_VERSION": "1",
    "HEXFIELD_EQ_RAYTAP": "0",
    **_DISABLED_BACKENDS,
}

_ARM4_MANIFEST_COUNT = 321
_ARM4_MANIFEST_SHA256 = (
    "1ce1cd95e41eaba486974ec4d8c69758cd7117307aeec6dd1ee4dcb17be8be28"
)
_ARM4_MANIFEST_ZLIB_B64 = """
eNqlmtFS4kAQRf/FZ4qy74zs7rdYVipiUNZACAlh9+83ZAUDiDP3zqPl7XQyfY4OYR4f7/K2XWfPZTV/b6b308NP0/dss61+
T5+XeZM9501xN3m02dPkrq2287fpoqzy1uHuafJd8f75UIWJzSbxtdWuTWl9Kpea1ymt64TGXUrjTmu8mNtZQ4fIomMjh9hG
4J/sUDR6oqibK9f9ExVtHtvjkH/NVyuiAGQDkA2abBgpW7QqN0yNpUhuCZJbmuSWJLmlSG4JkluK5JYguSmSmyC5KZKbILmR
khsruZGSGyu5KZKbIDlSJEeC5EiTHEmSI0VyJEiOFMmRIDkUySFIDkVyCJKDlBys5CAlBys5FMkRK/nuT7YtXnbzdlmth3n0
2ZmPyO6L5etbO6QnePj66sN827d+bab3ffKnhVIWlUJMKis/ek4QTlp0Et8l50VZZnU2r9ZdDNvj+P6Y/TH51taPmuLPJl+/
xJh6XhC29CP/VuQvJx4egslPGh5uWXl4zM+N5uEni1qkL8qiF+uqFlpLiC3Df3uu8wFjLwtANgDZoGHyps3VEuZq2lwtYa5G
ztXYuRo5V2PnauRcoc0VCXOFNlckzBXkXMHOFeRcwc4V5FydNleXMFenzdUlzNWRc3XsXB05V8fO1ZFz9dpcfcJcvTZXnzBX
T87Vs3P15Fw9O1cfMddlH98ed2PHVbqxRT9lj/u8fnN+47K/q+W6zapFtq32wx6uL4D7MXGndP/7mR+y5TpbLNd5GViKUyzw
RKsy+jPHWTTmI8eq6oomK4tFG7OFvUyPt7E37qjabLJNVS7nf6P3+ZclsayP6uL3+9dF4T3/qGa8aBYVPq2Z3dr5k8ulLBW9
TOwSRS8PtzTb/K/8bdnN2vA7n+tS5g3bN9VK6zqhca237RLadlLbyHdrVzVhOi9KQD9V3Iu1s5Lw/96reOD/wkUe3OXBXT7u
ndplTcQrtVGJJShtutKWpLSlKG0JSpuutCUobbrSJihtvNImKG280sYpbaTSxiltpNImKG280khQGrrSSFIaKUojQWnoSiNB
aehKQ1AavNIQlAavNDilQSoNTmmQSkNQOvaLr23xumzaYnv4v/6at0V2/KAYTBPqX1dFsDgq6j9Qv3eh9b3KE0/eF9Tc9Wvm
8tSfq6/quOWqpWa10qrZrbJmnpeHHsFwJ91Xx9+XUSibhLIpKBuJsrEoG4eykSibiLJpKJuEsikoG4OySSibgjIolCGhDAVl
kCiDRRkcyiBRhogyNJQhoQwFZTAoQ0IZCsqOQtlJKDsFZUei7FiUHYeyI1F2IspOQ9lJKDsFZceg7CSUnYKyp1D2EspeQdmT
KHsWZc+h7EmUvYiy11D2EspeQdkzKHsJZS+gfDgeyMA85AWcx3Xs7VFIjypiqTuW1GyPmmshoX1eyS5dLTastXbxiA/xTrw7
CXMjMTcRc9MwNxpz4zE3FnOjMTcZc1MxNxFz0zA3DnMTMTcNc5CYQ8QcGuagMQePOVjMQWMOGXOomEPEHBrm4DCHiDnxUbKp
+vNJ5CGaq5rYkzTjwvjjNF9Uhb8PGReFD9ZcpcOna/q1XkUt1iG3v/+/OL/QHy67GcvKdcCvYyogVdN2w2P078dmgVNqZ9GI
I2qfeURfGdyF42856o7b6r1YD5e7xUqXl7si5kjfKBjR+H869jTiZXp8IPHBvmDw6R9ud3up
"""

_PASSTHROUGH_MANIFEST_COUNT = 179
_PASSTHROUGH_MANIFEST_SHA256 = (
    "570627a31644c61b769eb1b0de37c46f19e7ac2b3265394188af489c308004d9"
)
_PASSTHROUGH_MANIFEST_ZLIB_B64 = """
eNqlmN1So0AQRt8l11TK7mFgfRbLmiJIlHXCRCBR334hq0h+kP6ay1R9Pc304UCYh4dV1raV2/iQvzbru3X/a/3q9nX4u96U
WbOKHu6Tx2jVhjp/WW99yFrDq8fot7r3onx+aU+Vkbw4HFpl26FS1/hN2fZtSdOjsulR2XSb03czumdhwdClK5G2YWhPfX68
F9m1+YqgLn1+1EVUwWAHRjs07gT0OdvtMqBo5/dIDSmNpiVGk9poWmY0KY2mJUaT0mhaYjShRpPKaAKNJpXRBBpNsNEEGk2w
0aQxmhRGs9JoXmI0q43mZUaz0mheYjQrjeYlRjNqNKuMZtBoVhnNoNEMG82g0QwbzRqjWWr04cPVxdMhb8tQfW/D/IkF4WEP
XTwia2436Jd027ooXJttfNH9e+gqYhsZWZqgNP+Wzgvv3ZvLQ3Wc4TVODrtMo2l/vgqKj31WPc25cx6WePNV8VJkw+KJnU0O
Kyd20pV+iz9/6/pfc7bcqpDN6KqQ4VasaTX/BLjOz/l5WcFgB0Y7NHMij/MEkyQtSYJJkpYkgSQJJkkgSYJJEkiSYZKsJckw
SdaSZJAkwyQZJMkwSQZJGpik0ZI0MEmjJWlAkgYmaUCSBiZpQJIxTDLWkoxhkrGWZAySjGGSMUgyhknGIEkLk7RakhYmabUk
LUjSwiQtSNLCJC1IMoFJJlqSCUwy0ZJMQJIJTDIBSSYwyQQkmcIkUy3JFCaZakmmIMkUJpmCJFOYZCogWXbxevSZOnWSMOTO
DhEmFvWV25ZV5md2N8TmNrXz8iOPs6zsxGMXjkXjfLFtJV/sl+nxV/vURYX93u2DL/NPyWnGZVp0846KZKca1wWSk41R1XhW
JAr/NJg83wj1pmxd2Lo6vHdBNukQK6s2iU8h+SjhMUIjxMcnHh04tiZ0t6N8LFdx0WzGVbIB3aiQTGlcNj+qq7RkXm2xm5tR
HxnNhW30S9L5SrBen5p72DXt8bSN7uwhmXkUnUUlz6GfAhYvzeDK8ouWXXMbXovqtN7U/I+ZPxSSR/coKOn8Py5+81zGz14+
5i69cSc+/gO/N60d
"""


def _canonical_manifest(rows: list[list[object]]) -> bytes:
    return json.dumps(rows, separators=(",", ":")).encode("utf-8")


def _decode_manifest(payload: str) -> list[list[object]]:
    raw = zlib.decompress(base64.b64decode(payload))
    rows = json.loads(raw)
    assert isinstance(rows, list)
    return rows


_ARM4_MANIFEST = _decode_manifest(_ARM4_MANIFEST_ZLIB_B64)
_PASSTHROUGH_MANIFEST = _decode_manifest(_PASSTHROUGH_MANIFEST_ZLIB_B64)


def _pythonpath_for_root(root: Path) -> str:
    package_dirs = []
    for relative in (
        "packages/hexfield_eq/python",
        "packages/hexo_engine/python",
        "packages/hexo_utils/python",
        "packages/hexo_train/python",
        "packages/hexo_runner/python",
        "packages/hexo_models/python",
        "packages/hexo_strix/python",
        "packages/dense_cnn_restnet/python",
    ):
        candidate = root / relative
        if candidate.is_dir():
            package_dirs.append(str(candidate))
    if not package_dirs or not (root / "packages/hexfield_eq/python").is_dir():
        raise AssertionError(f"not a hexfield_eq repository root: {root}")
    return os.pathsep.join(package_dirs)


def _child_env(root: Path, overrides: dict[str, str]) -> dict[str, str]:
    # Every architecture/backend knob is import-time.  Strip both the eq and
    # legacy namespaces first so the invoking shell cannot contaminate D8.
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEXFIELD")}
    env.update(overrides)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": _pythonpath_for_root(root),
        }
    )
    # D8 requires these to be genuinely unset, not explicit aliases.
    env.pop("HEXFIELD_EQ_TYPE_SIG", None)
    env.pop("HEXFIELD_EQ_ATTN_ORBIT", None)
    return env


def _run_child(
    script: str,
    *,
    root: Path,
    overrides: dict[str, str],
    extra_env: dict[str, str] | None = None,
    timeout: int = 300,
) -> dict:
    env = _child_env(root, overrides)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"child failed under {root}:\nstdout={proc.stdout[-6000:]}"
        f"\nstderr={proc.stderr[-6000:]}"
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"child produced no verdict under {root}"
    return json.loads(lines[-1])


_MANIFEST_CHILD = r"""
import hashlib
import json
import sys

import torch

# Windows torch imports Triton through flex_attention even with every backend
# gate disabled.  Force model.py's guarded import down its no-flex fallback.
sys.modules["torch.nn.attention.flex_attention"] = None

from hexfield_eq import constants as C
from hexfield_eq.model import HexfieldNet

torch.set_num_threads(1)
model = HexfieldNet().eval()
rows = [
    [key, list(tensor.shape), str(tensor.dtype)]
    for key, tensor in sorted(model.state_dict().items())
]
raw = json.dumps(rows, separators=(",", ":")).encode("utf-8")
print(json.dumps({
    "rows": rows,
    "count": len(rows),
    "sha256": hashlib.sha256(raw).hexdigest(),
    "group_order": C.GROUP_ORDER,
    "feature_version": int(getattr(C, "FEATURE_VERSION", 1)),
    "num_features": C.NUM_FEATURES,
    "type_sig": getattr(C, "TYPE_SIG", None),
    "attn_orbit": getattr(C, "ATTN_ORBIT", None),
    "cuda_available": torch.cuda.is_available(),
}, separators=(",", ":")))
"""


_LIVE_CHILD = r"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.modules["torch.nn.attention.flex_attention"] = None

from hexfield_eq import constants as C
from hexfield_eq.checkpoints import load_into
from hexfield_eq.model import HexfieldNet

torch.set_num_threads(1)
checkpoint = os.environ["D8_CHECKPOINT"]
input_path = os.environ["D8_INPUT"]
output_path = os.environ["D8_OUTPUT"]

payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
model = HexfieldNet().eval()
rows = [
    [key, list(tensor.shape), str(tensor.dtype)]
    for key, tensor in sorted(model.state_dict().items())
]
raw = json.dumps(rows, separators=(",", ":")).encode("utf-8")
load_into(model, payload, optimizer=None)

arrays = np.load(input_path, allow_pickle=False)
feats = torch.from_numpy(arrays["feats"])
nbr = torch.from_numpy(arrays["nbr"])
mask = torch.from_numpy(arrays["mask"])
coords = torch.from_numpy(arrays["coords"])
raylen = torch.from_numpy(arrays["raylen"])
with torch.no_grad():
    out = model.forward_policy_value(
        feats,
        nbr,
        mask,
        coords,
        raylen,
        request_moves_left=True,
    )
np.savez(
    output_path,
    policy=out["policy"].detach().cpu().numpy(),
    value=out["value"].detach().cpu().numpy(),
    moves_left=out["moves_left"].detach().cpu().numpy(),
)
print(json.dumps({
    "count": len(rows),
    "sha256": hashlib.sha256(raw).hexdigest(),
    "feature_version": int(getattr(C, "FEATURE_VERSION", 1)),
    "num_features": C.NUM_FEATURES,
    "type_sig": getattr(C, "TYPE_SIG", None),
    "attn_orbit": getattr(C, "ATTN_ORBIT", None),
    "type_sig_env_present": "HEXFIELD_EQ_TYPE_SIG" in os.environ,
    "attn_orbit_env_present": "HEXFIELD_EQ_ATTN_ORBIT" in os.environ,
    "raytap_env": os.environ.get("HEXFIELD_EQ_RAYTAP"),
    "cuda_available": torch.cuda.is_available(),
}, separators=(",", ":")))
"""


def _assert_manifest(
    actual: list[list[object]], expected: list[list[object]], *, label: str
) -> None:
    if actual == expected:
        return
    actual_by_key = {row[0]: row[1:] for row in actual}
    expected_by_key = {row[0]: row[1:] for row in expected}
    missing = sorted(expected_by_key.keys() - actual_by_key.keys())
    unexpected = sorted(actual_by_key.keys() - expected_by_key.keys())
    changed = [
        (key, expected_by_key[key], actual_by_key[key])
        for key in sorted(expected_by_key.keys() & actual_by_key.keys())
        if expected_by_key[key] != actual_by_key[key]
    ]
    pytest.fail(
        f"{label} state manifest drift: missing={missing[:8]} "
        f"unexpected={unexpected[:8]} changed={changed[:8]}"
    )


def _write_deterministic_input(path: Path) -> None:
    coords2 = np.asarray(
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)],
        dtype=np.int64,
    )
    directions = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
    index = {tuple(coord): k for k, coord in enumerate(coords2.tolist())}
    n = len(coords2)
    nbr2 = np.full((n, 6), n, dtype=np.int64)
    for i, (q, r) in enumerate(coords2):
        for d, (dq, dr) in enumerate(directions):
            nbr2[i, d] = index.get((int(q + dq), int(r + dr)), n)
    raw = np.arange(n * 25, dtype=np.int32).reshape(1, n, 25)
    feats = (((raw * 37 + 11) % 101) - 50).astype(np.float32) / np.float32(50.0)
    np.savez(
        path,
        feats=feats,
        nbr=nbr2[None, ...],
        mask=np.ones((1, n), dtype=np.bool_),
        coords=coords2[None, ...],
        # Radius-1 support has no live key beyond distance one; ones exercise
        # the blockers-on L path without inventing a longer visible ray.
        raylen=np.ones((1, n, 12), dtype=np.uint8),
    )


def test_embedded_phase_a_manifests_are_self_consistent() -> None:
    assert len(_ARM4_MANIFEST) == _ARM4_MANIFEST_COUNT
    assert (
        hashlib.sha256(_canonical_manifest(_ARM4_MANIFEST)).hexdigest()
        == _ARM4_MANIFEST_SHA256
    )
    assert len(_PASSTHROUGH_MANIFEST) == _PASSTHROUGH_MANIFEST_COUNT
    assert (
        hashlib.sha256(_canonical_manifest(_PASSTHROUGH_MANIFEST)).hexdigest()
        == _PASSTHROUGH_MANIFEST_SHA256
    )

    arm = {row[0]: (row[1], row[2]) for row in _ARM4_MANIFEST}
    assert arm["stem.w0"] == ([7, 192, 25], "torch.float32")
    assert arm["conv_blocks.0.conv1.w_base"] == (
        [7, 12, 16, 16],
        "torch.float32",
    )
    assert arm["attn_blocks.0.attn.q_proj.wb"] == (
        [12, 16, 16],
        "torch.float32",
    )
    assert arm["tokens"] == ([6, 16], "torch.float32")

    passthrough = {
        row[0]: (row[1], row[2]) for row in _PASSTHROUGH_MANIFEST
    }
    assert passthrough["stem.weight"] == ([7, 25, 96], "torch.float32")
    assert passthrough["conv_blocks.0.conv1.weight"] == (
        [7, 96, 96],
        "torch.float32",
    )
    assert passthrough["attn_blocks.0.attn.q_proj.weight"] == (
        [96, 96],
        "torch.float32",
    )
    assert passthrough["tokens"] == ([6, 96], "torch.float32")


def test_pure_regular_arm4_state_manifest_matches_phase_a() -> None:
    got = _run_child(_MANIFEST_CHILD, root=_REPO, overrides=_ARM4_ENV)
    assert got["count"] == _ARM4_MANIFEST_COUNT
    assert got["sha256"] == _ARM4_MANIFEST_SHA256
    assert got["group_order"] == 12
    assert got["feature_version"] == 1 and got["num_features"] == 25
    assert got["type_sig"] == "reg:16"
    assert got["attn_orbit"] == 16
    assert got["cuda_available"] is False
    _assert_manifest(got["rows"], _ARM4_MANIFEST, label="pure-reg arm4")


def test_passthrough_state_manifest_matches_phase_a() -> None:
    got = _run_child(_MANIFEST_CHILD, root=_REPO, overrides=_PASSTHROUGH_ENV)
    assert got["count"] == _PASSTHROUGH_MANIFEST_COUNT
    assert got["sha256"] == _PASSTHROUGH_MANIFEST_SHA256
    assert got["group_order"] == 1
    assert got["feature_version"] == 1 and got["num_features"] == 25
    assert got["cuda_available"] is False
    _assert_manifest(got["rows"], _PASSTHROUGH_MANIFEST, label="passthrough")


def test_live_checkpoint_reproduces_phase_a_logits(tmp_path: Path) -> None:
    checkpoint_env = os.environ.get("HEXFIELD_EQ_D8_CHECKPOINT")
    if not checkpoint_env:
        pytest.skip("HEXFIELD_EQ_D8_CHECKPOINT is unset")
    checkpoint = Path(checkpoint_env).expanduser()
    if not checkpoint.is_file():
        pytest.skip(f"D8 checkpoint is absent: {checkpoint}")

    reference_root = Path(
        os.environ.get("HEXFIELD_EQ_D8_REFERENCE_ROOT", str(_DEFAULT_REFERENCE_ROOT))
    ).expanduser()
    if not reference_root.is_dir():
        pytest.fail(f"D8 Phase-A reference root is absent: {reference_root}")
    if reference_root.resolve() == _REPO.resolve():
        pytest.fail("D8 reference root resolves to the Phase-B worktree itself")

    input_path = tmp_path / "d8_input.npz"
    reference_output = tmp_path / "phase_a_output.npz"
    candidate_output = tmp_path / "phase_b_output.npz"
    _write_deterministic_input(input_path)

    common = {
        "D8_CHECKPOINT": str(checkpoint.resolve()),
        "D8_INPUT": str(input_path.resolve()),
    }
    ref = _run_child(
        _LIVE_CHILD,
        root=reference_root,
        overrides=_ARM4_ENV,
        extra_env={**common, "D8_OUTPUT": str(reference_output.resolve())},
    )
    candidate = _run_child(
        _LIVE_CHILD,
        root=_REPO,
        overrides=_ARM4_ENV,
        extra_env={**common, "D8_OUTPUT": str(candidate_output.resolve())},
    )

    for label, verdict in (("Phase A", ref), ("Phase B", candidate)):
        assert verdict["count"] == _ARM4_MANIFEST_COUNT, label
        assert verdict["sha256"] == _ARM4_MANIFEST_SHA256, label
        assert verdict["feature_version"] == 1, label
        assert verdict["num_features"] == 25, label
        assert verdict["type_sig_env_present"] is False, label
        assert verdict["attn_orbit_env_present"] is False, label
        assert verdict["raytap_env"] == "0", label
        assert verdict["cuda_available"] is False, label
    assert candidate["type_sig"] == "reg:16"
    assert candidate["attn_orbit"] == 16

    with np.load(reference_output, allow_pickle=False) as expected, np.load(
        candidate_output, allow_pickle=False
    ) as actual:
        assert set(expected.files) == set(actual.files) == {
            "policy",
            "value",
            "moves_left",
        }
        for key in expected.files:
            assert actual[key].shape == expected[key].shape, key
            np.testing.assert_allclose(
                actual[key],
                expected[key],
                atol=1e-5,
                rtol=0,
                err_msg=f"D8 live checkpoint {key} mismatch",
            )
