"""Production typed-fiber construction and full-network equivariance gates.

Every architecture/feature-version case runs in a fresh child interpreter:
``constants.py`` reads all shape knobs at import, and importing production
``model.py`` must not pull Triton into this CPU suite.  The controller strips
every inherited ``HEXFIELD*`` variable, sets the complete child architecture,
forces ``CUDA_VISIBLE_DEVICES=-1``, and installs the same optional
``flex_attention`` sentinel used by the Phase-A toy-net proof.

The full-network checks use Python-oracle features from real ``PositionFacts``
and rebuild support, neighbours, ray lengths, and features after each of all 12
D6 transforms.  They exercise a compact ``CLA`` trunk (C, blocker-aware L, A)
plus the production policy/value heads, keeping the CPU gate practical while
covering every typed boundary family.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


_REPO = Path(__file__).resolve().parents[1]
_PACKAGE_PATHS = (
    _REPO / "packages" / "hexfield_eq" / "python",
    _REPO / "packages" / "hexo_engine" / "python",
    _REPO / "packages" / "hexo_utils" / "python",
)

SIG_CONSERVATIVE = "reg:8,mirror:8,axis:4,triv:4"
SIG_DIVERSIFIED = "reg:4,mirror:6,point:2,axis:8,triv:8"

_BACKEND_OFF = (
    "HEXFIELD_TRITON_CONV",
    "HEXFIELD_TRITON_CONV_LN",
    "HEXFIELD_TRITON_ATTN",
    "HEXFIELD_EQ_TRITON_RAY",
    "HEXFIELD_SERVE_FLEX",
    "HEXFIELD_TRAIN_FLEX",
    "HEXFIELD_FLEX_PAIR",
    "HEXFIELD_TRAIN_FLEX_PAIR",
    "HEXFIELD_CUDA_GRAPHS",
    "HEXFIELD_SERVE_HALF",
)


def _child_env(
    *,
    signature: str | None,
    attn_orbit: int | None,
    feature_version: int = 1,
    layout: str = "CLA",
) -> dict[str, str]:
    """A complete import-time architecture with no inherited shape knobs."""

    env = {key: value for key, value in os.environ.items() if not key.startswith("HEXFIELD")}
    pythonpath = os.pathsep.join(str(path) for path in _PACKAGE_PATHS)
    inherited_path = env.get("PYTHONPATH")
    if inherited_path:
        pythonpath = os.pathsep.join((pythonpath, inherited_path))
    env.update(
        {
            "PYTHONPATH": pythonpath,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HEXFIELD_EQ_GROUP_ORDER": "12",
            "HEXFIELD_EQ_FEATURE_VERSION": str(feature_version),
            "HEXFIELD_EQ_ATTENTION_HEADS": "3",
            "HEXFIELD_EQ_TRUNK": layout,
            "HEXFIELD_EQ_SUPPORT_RADIUS": "1",
            "HEXFIELD_EQ_RAY_BLOCKERS": "1",
            "HEXFIELD_EQ_REG_LANE": "0",
            "HEXFIELD_EQ_REG_TOK_READ": "0",
        }
    )
    env.update({name: "0" for name in _BACKEND_OFF})
    if signature is not None:
        env["HEXFIELD_EQ_TYPE_SIG"] = signature
    if attn_orbit is not None:
        env["HEXFIELD_EQ_ATTN_ORBIT"] = str(attn_orbit)
    return env


def _run_child(
    mode: str,
    *,
    signature: str,
    attn_orbit: int,
    feature_version: int,
    timeout: int = 120,
) -> dict:
    proc = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "child", "--mode", mode],
        cwd=_REPO,
        env=_child_env(
            signature=signature,
            attn_orbit=attn_orbit,
            feature_version=feature_version,
        ),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"typed-model child failed: mode={mode}, sig={signature}, "
        f"K={attn_orbit}, feature_version={feature_version}\n"
        f"stdout:\n{proc.stdout[-6000:]}\nstderr:\n{proc.stderr[-4000:]}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _run_import_failure(
    *,
    signature: str,
    attn_orbit: int | None,
    layout: str,
    channels: int | None = None,
    c_orbit: int | None = None,
) -> str:
    env = _child_env(
        signature=signature,
        attn_orbit=attn_orbit,
        feature_version=1,
        layout=layout,
    )
    if channels is not None:
        env["HEXFIELD_EQ_CHANNELS"] = str(channels)
    if c_orbit is not None:
        env["HEXFIELD_EQ_C_ORBIT"] = str(c_orbit)
    proc = subprocess.run(
        [sys.executable, "-B", "-c", "import hexfield_eq.constants"],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0, "invalid import-time architecture unexpectedly succeeded"
    return proc.stdout + proc.stderr


def _assert_report(
    report: dict,
    *,
    signature: str,
    attn_orbit: int,
    feature_version: int,
    channels: int,
) -> None:
    assert report["type_sig"] == signature
    assert report["attn_orbit"] == attn_orbit
    assert report["feature_version"] == feature_version
    assert report["num_features"] == (25 if feature_version == 1 else 46)
    assert report["channels"] == channels
    assert report["finite"] is True
    assert report["new_triton_modules"] == []


def test_three_nominated_configs_and_v1_full_net_equivariance() -> None:
    """All nominated (sig,K); both unique sigs get all-12 full-net checks."""

    conservative = _run_child(
        "equivariance",
        signature=SIG_CONSERVATIVE,
        attn_orbit=8,
        feature_version=1,
    )
    _assert_report(
        conservative,
        signature=SIG_CONSERVATIVE,
        attn_orbit=8,
        feature_version=1,
        channels=160,
    )
    assert conservative["max_policy_error"] < 1e-4
    assert conservative["max_value_error"] < 1e-4

    diversified_full = _run_child(
        "equivariance",
        signature=SIG_DIVERSIFIED,
        attn_orbit=16,
        feature_version=1,
    )
    _assert_report(
        diversified_full,
        signature=SIG_DIVERSIFIED,
        attn_orbit=16,
        feature_version=1,
        channels=128,
    )
    assert diversified_full["max_policy_error"] < 1e-4
    assert diversified_full["max_value_error"] < 1e-4

    diversified_fast = _run_child(
        "smoke",
        signature=SIG_DIVERSIFIED,
        attn_orbit=8,
        feature_version=1,
    )
    _assert_report(
        diversified_fast,
        signature=SIG_DIVERSIFIED,
        attn_orbit=8,
        feature_version=1,
        channels=128,
    )


def test_feature_versions_typed_stem_full_net_and_pure_reg_head_parity() -> None:
    """Both feature maps cover the stem; v2 also gets an all-12 mixed full net."""

    for version in (1, 2):
        parity = _run_child(
            "pure-reg-parity",
            signature="reg:16",
            attn_orbit=16,
            feature_version=version,
        )
        _assert_report(
            parity,
            signature="reg:16",
            attn_orbit=16,
            feature_version=version,
            channels=192,
        )
        assert parity["stem_max_error"] <= 1e-12
        assert parity["input_rep_exact"] is True
        assert parity["head_perm_exact"] is True
        assert parity["head_perm6_exact"] is True

    v2_full = _run_child(
        "equivariance",
        signature=SIG_DIVERSIFIED,
        attn_orbit=8,
        feature_version=2,
    )
    _assert_report(
        v2_full,
        signature=SIG_DIVERSIFIED,
        attn_orbit=8,
        feature_version=2,
        channels=128,
    )
    assert v2_full["max_policy_error"] < 1e-4
    assert v2_full["max_value_error"] < 1e-4


def test_type_signature_env_requires_canonical_order() -> None:
    error = _run_import_failure(
        signature="mirror:8,reg:8",
        attn_orbit=8,
        layout="A",
    )
    assert "canonical order" in error


def test_mixed_signature_env_requires_explicit_attention_orbit() -> None:
    error = _run_import_failure(
        signature="reg:4,mirror:4",
        attn_orbit=None,
        layout="A",
    )
    assert "ATTN_ORBIT is required" in error


def test_explicit_pure_signature_rejects_conflicting_legacy_c_orbit() -> None:
    error = _run_import_failure(
        signature="reg:16",
        attn_orbit=16,
        layout="A",
        channels=192,
        c_orbit=8,
    )
    assert "C_ORBIT=8 disagrees" in error


def test_l_layout_rejects_odd_attention_orbit() -> None:
    error = _run_import_failure(
        signature="reg:5",
        attn_orbit=5,
        layout="LA",
    )
    # The current fast-path legal set contains only even K, so its earlier
    # head-dimension check can reject K=5 before the dedicated L-layout check.
    # Either diagnostic is the code-ground-truth rejection of an odd L width.
    assert (
        "must be even for an 'L' layout" in error
        or "4*HEXFIELD_EQ_ATTN_ORBIT=20 must be one of" in error
    )


def _triton_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name == "triton" or name.startswith("triton.")
    }


def _child_modules():
    """Import production modules behind the CPU FlexAttention sentinel."""

    torch = importlib.import_module("torch")
    triton_before = _triton_modules()
    flex_key = "torch.nn.attention.flex_attention"
    had_flex = flex_key in sys.modules
    previous_flex = sys.modules.get(flex_key)
    sys.modules[flex_key] = None
    try:
        constants = importlib.import_module("hexfield_eq.constants")
        equivariant = importlib.import_module("hexfield_eq.equivariant")
        features = importlib.import_module("hexfield_eq.features")
        geometry = importlib.import_module("hexfield_eq.geometry")
        model = importlib.import_module("hexfield_eq.model")
        reps = importlib.import_module("hexfield_eq.reps")
    finally:
        if had_flex:
            sys.modules[flex_key] = previous_flex
        else:
            sys.modules.pop(flex_key, None)
    assert _triton_modules() == triton_before
    assert not torch.cuda.is_available()
    return torch, constants, equivariant, features, geometry, model, reps, triton_before


def _fixed_facts(features):
    cells = ((0, 0), (1, 0), (0, 1), (-1, 1), (1, -1))
    records = tuple(
        (q, r, features.record_player(index), index)
        for index, (q, r) in enumerate(cells)
    )
    moves = len(records)
    phase = features.record_phase(moves)
    first_stone = (
        (records[-1][0], records[-1][1])
        if phase == "SecondStone"
        else None
    )
    return features.PositionFacts(
        records=records,
        current_player=features.record_player(moves),
        phase=phase,
        first_stone=first_stone,
    )


def _oracle_inputs(torch, features, facts):
    support, feature_array = features.build_position(facts)
    nodes = support.num_nodes
    nbr = torch.from_numpy(support.nbr.astype("int64")).unsqueeze(0)
    nbr = torch.where(nbr >= 0, nbr, torch.full_like(nbr, nodes))
    return (
        support,
        torch.from_numpy(feature_array).unsqueeze(0),
        nbr,
        torch.ones(1, nodes, dtype=torch.bool),
        torch.from_numpy(support.coords.astype("int64")).unsqueeze(0),
        torch.from_numpy(features.build_ray_lengths(facts, support)).unsqueeze(0),
    )


def _full_net_child(mode: str, modules: tuple) -> dict:
    torch, constants, _eq, features, geometry, model_module, _reps, _before = modules
    torch.manual_seed(20260709)
    net = model_module.HexfieldNet().eval()
    net.set_attention_impl("materialized")
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.08)
        stem_weight, _stem_bias = net.stem._materialize()

    assert net.stem.kind == "stem"
    assert tuple(stem_weight.shape) == (
        7,
        constants.NUM_FEATURES,
        constants.CHANNELS,
    )
    meta = net.arch_meta()
    assert meta["type_sig"] == constants.TYPE_SIG
    assert meta["attn_orbit"] == constants.ATTN_ORBIT

    facts = _fixed_facts(features)
    support, feats, nbr, mask, coords, raylen = _oracle_inputs(torch, features, facts)
    assert feats.dtype == torch.float32
    assert feats.shape[-1] == constants.NUM_FEATURES
    with torch.no_grad():
        base = net.forward_policy_value(feats, nbr, mask, coords, raylen)
    finite = all(bool(torch.isfinite(value).all()) for value in base.values())
    assert finite

    max_policy_error = 0.0
    max_value_error = 0.0
    if mode == "equivariance":
        with torch.no_grad():
            for group_element in range(12):
                transformed = features.transform_facts(facts, group_element)
                support_g, feats_g, nbr_g, mask_g, coords_g, raylen_g = _oracle_inputs(
                    torch, features, transformed
                )
                output_g = net.forward_policy_value(
                    feats_g, nbr_g, mask_g, coords_g, raylen_g
                )
                node_permutation = torch.tensor(
                    [
                        support_g.index[
                            geometry.apply_d6(group_element, int(q), int(r))
                        ]
                        for q, r in support.coords.tolist()
                    ],
                    dtype=torch.long,
                )
                policy_g = output_g["policy"].index_select(1, node_permutation)
                policy_error = float((policy_g - base["policy"]).abs().max())
                value_error = float((output_g["value"] - base["value"]).abs().max())
                max_policy_error = max(max_policy_error, policy_error)
                max_value_error = max(max_value_error, value_error)
                torch.testing.assert_close(
                    policy_g,
                    base["policy"],
                    atol=1e-4,
                    rtol=0,
                    msg=f"policy covariance g={group_element}",
                )
                torch.testing.assert_close(
                    output_g["value"],
                    base["value"],
                    atol=1e-4,
                    rtol=0,
                    msg=f"value invariance g={group_element}",
                )

    return {
        "mode": mode,
        "type_sig": constants.TYPE_SIG,
        "attn_orbit": constants.ATTN_ORBIT,
        "feature_version": constants.FEATURE_VERSION,
        "num_features": constants.NUM_FEATURES,
        "channels": constants.CHANNELS,
        "nodes": support.num_nodes,
        "finite": finite,
        "max_policy_error": max_policy_error,
        "max_value_error": max_value_error,
    }


def _pure_reg_parity_child(modules: tuple) -> dict:
    torch, constants, equivariant, _features, _geometry, _model, reps, _before = modules
    assert constants.TYPE_SIG == "reg:16"
    assert constants.ATTN_ORBIT == 16

    input_rep_exact = True
    for group_element in range(12):
        input_rep_exact = input_rep_exact and torch.equal(
            reps.input_rep_matrix(group_element),
            equivariant._in_rep_matrix()[group_element].to(torch.float64),
        )

    torch.manual_seed(20260709)
    w0 = torch.randn(
        7,
        constants.CHANNELS,
        constants.NUM_FEATURES,
        dtype=torch.float64,
    )
    typed_stem = reps.typed_stem_weight(w0, constants.TYPE_SIGNATURE)
    production_stem = equivariant.gen_stem_weight(w0)
    stem_max_error = float((typed_stem - production_stem).abs().max())
    torch.testing.assert_close(typed_stem, production_stem, atol=1e-12, rtol=0)

    k_attn = constants.ATTN_ORBIT
    head_perm_exact = torch.equal(reps.head_perm(k_attn), equivariant.head_perm())
    head_perm_exact = head_perm_exact and torch.equal(
        reps.head_perm_inv(k_attn), equivariant.head_perm_inv()
    )
    head_perm6_exact = torch.equal(reps.head_perm6(k_attn), equivariant.head_perm6())
    head_perm6_exact = head_perm6_exact and torch.equal(
        reps.head_perm6_inv(k_attn), equivariant.head_perm6_inv()
    )
    assert input_rep_exact and head_perm_exact and head_perm6_exact

    return {
        "mode": "pure-reg-parity",
        "type_sig": constants.TYPE_SIG,
        "attn_orbit": constants.ATTN_ORBIT,
        "feature_version": constants.FEATURE_VERSION,
        "num_features": constants.NUM_FEATURES,
        "channels": constants.CHANNELS,
        "finite": bool(torch.isfinite(typed_stem).all()),
        "stem_max_error": stem_max_error,
        "input_rep_exact": input_rep_exact,
        "head_perm_exact": head_perm_exact,
        "head_perm6_exact": head_perm6_exact,
    }


def _child_main(mode: str) -> None:
    modules = _child_modules()
    if mode in ("smoke", "equivariance"):
        report = _full_net_child(mode, modules)
    elif mode == "pure-reg-parity":
        report = _pure_reg_parity_child(modules)
    else:  # pragma: no cover - argparse owns validation
        raise ValueError(mode)
    triton_before = modules[-1]
    new_triton = sorted(_triton_modules() - triton_before)
    assert not new_triton
    report["new_triton_modules"] = new_triton
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("child",))
    parser.add_argument(
        "--mode",
        required=True,
        choices=("smoke", "equivariance", "pure-reg-parity"),
    )
    args = parser.parse_args()
    _child_main(args.mode)
