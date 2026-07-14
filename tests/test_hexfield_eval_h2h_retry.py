"""Retry semantics of the multistage eval's multi-checkpoint pass.

Under ``mode=run_concurrent`` the checkpoint (h2h) pass shares the GPU with the
live driver's selfplay; a freshly restarted driver's CUDA-graph capture burst
can surface transient errors ("RuntimeError: CUDA driver error: device not
ready" cost hexfield_eq_main_2 ep20 AND ep30 their entire h2h leg — the
champion edge dropped, verdict permanently INCONCLUSIVE). The pass is
fail-soft, so before 2026-07-11 ONE transient exception silently discarded
the epoch's champion verdict.

Pinned here (parametrized over BOTH twins, hexfield and hexfield_eq):

  1. A transient failure is retried and the h2h edges are recovered; the
     attempt count is recorded, no error is reported.
  2. A persistent failure exhausts HEXFIELD_EVAL_H2H_RETRIES + 1 attempts,
     keeps the original fail-soft envelope (anchors still rated, stage D
     completes, error string recorded), and the verdict degrades exactly as
     before (no champion edge -> no primary).
  3. First-try success calls the arena exactly once and reports no attempt
     count (diagnostics JSON unchanged for the healthy path).
  4. HEXFIELD_EVAL_H2H_RETRIES=0 restores single-attempt behavior.

CPU-only: the arena is stubbed via the injection seams; the retry delay is
forced to 0 through HEXFIELD_EVAL_H2H_RETRY_DELAY.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in ("hexo_engine/python", "hexfield/python", "hexfield_eq/python"):
    _src = str(_REPO / "packages" / _p)
    if _src not in sys.path:
        sys.path.insert(0, _src)

from hexfield_eval_kit import _make_run, _paired_match, _sealbot_match  # noqa: E402


@pytest.fixture(params=["hexfield", "hexfield_eq"])
def pkg(request):
    """(multistage_eval module, parse_hexfield_config) for each twin."""
    mse = importlib.import_module(f"{request.param}.multistage_eval")
    config = importlib.import_module(f"{request.param}.config")
    return mse, config.parse_hexfield_config


class _FlakyMulti:
    """play_multi_checkpoint_match stub: raises `fail_times` times, then
    returns a full per-opponent match dict."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, candidate_ckpt, opponents, per, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("CUDA driver error: device not ready")
        cand = kw.get("candidate_label", "cand")
        return {
            label: _paired_match(cand, label, max(1, per // 2), 2)
            for label, _ckpt in opponents
        }


def _play_sealbot(ckpt, n, **kw):
    return _sealbot_match(kw.get("label", "hexfield"), n, 0.6)


def _run_concurrent(mse, parse_config, tmp_path, flaky, monkeypatch, *, retries=None):
    monkeypatch.setenv("HEXFIELD_EVAL_H2H_RETRY_DELAY", "0")
    if retries is not None:
        monkeypatch.setenv("HEXFIELD_EVAL_H2H_RETRIES", str(retries))
    else:
        monkeypatch.delenv("HEXFIELD_EVAL_H2H_RETRIES", raising=False)
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40), bc=False)
    cfg = parse_config({"multi_stage_eval": {"sprt": {"enabled": False}}})
    return mse.run_multistage_eval_concurrent(
        run,
        run / "checkpoints" / "epoch_000040.pt",
        cfg,
        candidate_epoch=40,
        write_diagnostics=False,
        play_multi_checkpoint_match=flaky,
        play_sealbot_match=_play_sealbot,
    )


def _stage_c(report: dict) -> dict:
    return next(s for s in report["stages"] if s["stage"] == "C_deep")


def test_transient_failure_is_retried_and_h2h_recovers(pkg, tmp_path, monkeypatch):
    mse, parse_config = pkg
    flaky = _FlakyMulti(fail_times=1)
    rep = _run_concurrent(mse, parse_config, tmp_path, flaky, monkeypatch)
    sc = _stage_c(rep)
    assert flaky.calls == 2
    assert sc.get("multi_checkpoint_attempts") == 2
    assert "multi_checkpoint_error" not in sc
    # The h2h edges made it into the stage (anchors + checkpoint opponents).
    ckpt_played = [p for p in sc["opponents_played"] if p not in ("sealbot",)]
    assert ckpt_played, "checkpoint opponents recovered by the retry"
    # The champion edge exists again -> the primary hypothesis is testable.
    assert rep["verdict"]["primary"] is not None


def test_persistent_failure_keeps_fail_soft_envelope(pkg, tmp_path, monkeypatch):
    mse, parse_config = pkg
    flaky = _FlakyMulti(fail_times=99)
    rep = _run_concurrent(
        mse, parse_config, tmp_path, flaky, monkeypatch, retries=2
    )
    sc = _stage_c(rep)
    assert flaky.calls == 3  # 1 + 2 retries
    assert sc.get("multi_checkpoint_attempts") == 3
    assert "device not ready" in sc["multi_checkpoint_error"]
    # Fail-soft: the anchor edge still rates, stage D still completes.
    assert "sealbot" in sc["opponents_played"]
    assert rep["verdict"]["label"] == "INCONCLUSIVE"
    assert rep["verdict"]["primary"] is None


def test_first_try_success_is_single_call_and_unreported(pkg, tmp_path, monkeypatch):
    mse, parse_config = pkg
    flaky = _FlakyMulti(fail_times=0)
    rep = _run_concurrent(mse, parse_config, tmp_path, flaky, monkeypatch)
    sc = _stage_c(rep)
    assert flaky.calls == 1
    assert "multi_checkpoint_attempts" not in sc
    assert "multi_checkpoint_error" not in sc


def test_zero_retries_restores_single_attempt(pkg, tmp_path, monkeypatch):
    mse, parse_config = pkg
    flaky = _FlakyMulti(fail_times=99)
    rep = _run_concurrent(
        mse, parse_config, tmp_path, flaky, monkeypatch, retries=0
    )
    sc = _stage_c(rep)
    assert flaky.calls == 1
    assert "device not ready" in sc["multi_checkpoint_error"]
