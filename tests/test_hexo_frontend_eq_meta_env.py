"""CPU-only import-seeding gates for the dashboard's hexfield_eq worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHONPATH = os.pathsep.join(
    str(ROOT / relative)
    for relative in (
        "packages/hexo_frontend/python",
        "packages/hexfield_eq/python",
        "packages/hexo_engine/python",
        "packages/hexo_utils/python",
        "packages/hexo_train/python",
        "packages/hexo_runner/python",
    )
)
SIGNATURE = "reg:8,mirror:8,axis:4,triv:4"


def _env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("HEXFIELD")}
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": PYTHONPATH,
            "HEXFIELD_TRITON_CONV": "0",
            "HEXFIELD_TRITON_CONV_LN": "0",
            "HEXFIELD_TRITON_ATTN": "0",
            "HEXFIELD_EQ_TRITON_RAY": "0",
            "HEXFIELD_SERVE_FLEX": "0",
            "HEXFIELD_TRAIN_FLEX": "0",
            "HEXFIELD_FLEX_PAIR": "0",
            "HEXFIELD_TRAIN_FLEX_PAIR": "0",
            "HEXFIELD_CUDA_GRAPHS": "0",
            "HEXFIELD_SERVE_HALF": "0",
        }
    )
    return env


def _run(mode: str, feature_width: int = 25) -> dict:
    proc = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--child", mode, str(feature_width)],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fresh_worker_seeds_mixed_signature_and_attention_orbit() -> None:
    assert _run("fresh", 25) == {
        "attn_orbit": 8,
        "channels": 160,
        "feature_version": 1,
        "num_features": 25,
        "type_signature": [["reg", 8], ["mirror", 8], ["axis", 4], ["triv", 4]],
    }


def test_fresh_worker_seeds_feature_v2_from_metadata_width() -> None:
    report = _run("fresh", 46)
    assert report["feature_version"] == 2
    assert report["num_features"] == 46
    assert report["channels"] == 160


def test_already_imported_mismatch_requires_worker_restart() -> None:
    report = _run("mismatch")
    assert report["rejected"] is True
    assert "restart the debug worker" in report["message"]
    assert "type_sig" in report["message"]


def test_legacy_meta_seeds_exactly_the_pre_phase_b_keys() -> None:
    report = _run("legacy")
    assert report == {
        "HEXFIELD_EQ_ATTENTION_HEADS": "3",
        "HEXFIELD_EQ_CHANNELS": "192",
        "HEXFIELD_EQ_C_ORBIT": "16",
        "HEXFIELD_EQ_GROUP_ORDER": "12",
        "HEXFIELD_EQ_RAY_BLOCKERS": "1",
        "HEXFIELD_EQ_REG_LANE": "1",
        "HEXFIELD_EQ_REG_TOK_READ": "0",
        "HEXFIELD_EQ_SUPPORT_RADIUS": "4",
        "HEXFIELD_EQ_TRUNK": "CCLACCLACLA",
    }


def _meta(feature_width: int) -> dict:
    return {
        "support_radius": 1,
        "channels": 160,
        "group_order": 12,
        "c_orbit": 8,
        "attention_heads": 3,
        "trunk_layout": "CLA",
        "reg_lane": False,
        "reg_tok_read": False,
        "ray_blockers": True,
        "type_sig": SIGNATURE,
        "attn_orbit": 8,
        "feature_width": feature_width,
    }


def _child(mode: str, feature_width: int) -> dict:
    # Keep optional Flex inert on a Windows CPU host, as the typed suites do.
    # Unlike those suites this namespace import includes hexfield_eq.inference,
    # which pulls Triton whenever it is installed — expected for the worker.
    sys.modules["torch.nn.attention.flex_attention"] = None
    from hexo_frontend import debug_infer

    if mode == "legacy":
        seeded: dict[str, str] = {}
        debug_infer._seed_eq_meta_env(
            {
                "support_radius": 4,
                "channels": 192,
                "group_order": 12,
                "c_orbit": 16,
                "attention_heads": 3,
                "trunk_layout": "CCLACCLACLA",
                "reg_lane": True,
                "reg_tok_read": False,
                "ray_blockers": True,
            },
            seeded,
        )
        return seeded

    meta = _meta(feature_width)
    eq = debug_infer._hexfield_eq(meta)
    if mode == "mismatch":
        bad = dict(meta, type_sig="reg:4,mirror:6,point:2,axis:8,triv:8", channels=128)
        try:
            debug_infer._check_eq_meta_matches_import(eq, bad)
        except ValueError as exc:
            return {"rejected": True, "message": str(exc)}
        raise AssertionError("already-imported signature mismatch was accepted")

    return {
        "type_signature": eq.constants.TYPE_SIGNATURE,
        "attn_orbit": eq.constants.ATTN_ORBIT,
        "channels": eq.constants.CHANNELS,
        "feature_version": eq.constants.FEATURE_VERSION,
        "num_features": eq.constants.NUM_FEATURES,
    }


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--child":
        raise SystemExit("expected --child MODE FEATURE_WIDTH")
    print(json.dumps(_child(sys.argv[2], int(sys.argv[3])), sort_keys=True))
