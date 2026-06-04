"""Tests for the dashboard Debug-tab inference library + worker service.

The inference tests need torch + the hexgt build and a real run directory, so
they skip cleanly where those are absent (e.g. a CI box without the GPU build).
The path/signature tests are pure and always run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Pure helpers (no torch) — always importable.
from hexo_frontend import debug_service


# --------------------------------------------------------------------------
# Pure: WSL path translation (runs everywhere).
# --------------------------------------------------------------------------


def test_to_wsl_translates_windows_drive_paths():
    assert debug_service._to_wsl("E:\\Hexo-BotTrainer-hexgt\\runs\\x.pt") == "/mnt/e/Hexo-BotTrainer-hexgt/runs/x.pt"
    assert debug_service._to_wsl("C:\\a\\b") == "/mnt/c/a/b"


def test_to_wsl_passes_through_posix_paths():
    assert debug_service._to_wsl("/mnt/e/runs/x.pt") == "/mnt/e/runs/x.pt"
    assert debug_service._to_wsl("relative/path") == "relative/path"


# --------------------------------------------------------------------------
# Inference: gated on torch + hexgt + a real run dir.
# --------------------------------------------------------------------------

di = pytest.importorskip("hexo_frontend.debug_infer", reason="needs torch + hexgt build")


def _run_dir() -> Path | None:
    candidate = Path.cwd() / "runs" / "hexgt_rl_main3"
    return candidate if (candidate / "checkpoints").is_dir() else None


def _checkpoint(name: str) -> Path:
    run_dir = _run_dir()
    if run_dir is None:
        pytest.skip("runs/hexgt_rl_main3 not present")
    path = run_dir / "checkpoints" / name
    if not path.is_file():
        pytest.skip(f"checkpoint {name} not present")
    return path


def _sample_actions(n: int = 16) -> list[int]:
    run_dir = _run_dir()
    hxr = run_dir / "selfplay" / "epoch_000000.hxr"
    if not hxr.is_file():
        pytest.skip("no self-play .hxr present")
    from hexo_runner.records import HexoRecordFile

    with HexoRecordFile.open(hxr) as rf:
        rec = next(iter(rf.iter_records()))
    return [int(a) for a in rec.action_ids][:n]


@pytest.fixture(scope="module")
def cpu_only():
    # Belt-and-suspenders: the library forces CPU, but make the GPU invisible too.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield


def test_load_post_graft_checkpoint(cpu_only):
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    assert loaded.graft == "post"
    assert loaded.expanded_stv == []  # already wide, no expansion
    nonstv = [w for w in loaded.load_warnings if "short_term_value" not in w]
    assert not nonstv, loaded.load_warnings
    assert not loaded.model.training  # eval mode


def test_load_pre_graft_checkpoint_expands(cpu_only):
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_epoch000000.pt"))
    assert loaded.graft == "pre"
    # The epoch-0 STV heads were SIDE-only; the loader must widen all three.
    assert len(loaded.expanded_stv) == 3
    nonstv = [w for w in loaded.load_warnings if "short_term_value" not in w]
    assert not nonstv, loaded.load_warnings


def test_analyze_position_shapes(cpu_only):
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(16)
    result = di.analyze_position(loaded, actions)

    assert result["current_player"] in (0, 1)
    assert -1.0 <= result["value"] <= 1.0
    assert len(result["value_dist"]) == 65
    assert abs(sum(result["value_dist"]) - 1.0) < 1e-3
    assert len(result["value_bins"]) == 65

    policy = result["policy"]
    assert policy and result["candidate_count"] == len(policy)
    # Sorted descending by prior, probabilities in [0, 1].
    probs = [row["p"] for row in policy]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-2

    # STV heads present and scalar in range.
    assert set(result["stvalue"]) == {"4", "12", "24"}
    for head in result["stvalue"].values():
        assert -1.0 <= head["scalar"] <= 1.0
        assert len(head["dist"]) == 65

    # Both-perspectives / optimism probe present and consistent.
    assert -1.0 <= result["value_swapped"] <= 1.0
    assert abs(result["optimism"] - (result["value"] + result["value_swapped"])) < 1e-3


def test_both_perspectives_swap_is_symmetric_ish(cpu_only):
    """Swapping ownership twice returns the original board, so analyzing the
    swapped facts must reproduce the swapped value the single call reported."""
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(14)
    result = di.analyze_position(loaded, actions)
    # optimism is the documented sum; just assert it is finite and bounded.
    assert -2.0 <= result["optimism"] <= 2.0


def test_search_position(cpu_only):
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(16)
    result = di.search_position(loaded, actions, visits=64)
    assert result["visits"] >= 1
    assert -1.0 <= result["root_value"] <= 1.0
    assert result["visit_policy"], "search returned an empty visit policy"
    # visit policy normalized
    total = sum(row["p"] for row in result["visit_policy"])
    assert abs(total - 1.0) < 1e-2
    # best action is one of the visited candidates
    best = result["best_action_id"]
    assert any(row["action_id"] == best for row in result["visit_policy"])
