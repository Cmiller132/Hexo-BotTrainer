"""Incremental pool persistence of the concurrent multistage eval wave.

The 2026-07-14 hexfield_eq_main_3 ep70 GPU segfault landed mid-wave (during
the multi-checkpoint pass) and discarded BOTH fully-played anchor edges
(Strix 64 games + SealBot 32): the concurrent path built every edge in memory
and pooled only at Stage D, so a process death anywhere in the wave lost all
of it. The fix persists each edge to ``eval_pool.json`` the moment its match
completes (``_persist_edge_incremental``), with Stage D silently skipping the
already-persisted edges.

Pinned here (parametrized over BOTH twins, hexfield and hexfield_eq):

  1. A wave that dies mid-pass (the multi-checkpoint stub raises a
     BaseException the fail-soft envelope does not catch — the in-test
     stand-in for a process-killing crash) leaves the already-played SealBot
     edge durably in the on-disk pool.
  2. A completed wave pools each (epoch, {a, b}) edge exactly once — the
     incremental persists and Stage D's append never double-count, and the
     report's edge list matches the pooled rows for the epoch.
  3. ``write_diagnostics=False`` still writes nothing (in-memory mode
     unchanged by the durability path).

CPU-only: the arena is stubbed via the injection seams.
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


class _ProcessDeath(BaseException):
    """Not an Exception: sails through the wave's fail-soft try/except exactly
    like a segfault ends the process before Stage D."""


def _crash_multi(candidate_ckpt, opponents, per, **kw):
    raise _ProcessDeath


def _healthy_multi(candidate_ckpt, opponents, per, **kw):
    cand = kw.get("candidate_label", "cand")
    return {
        label: _paired_match(cand, label, max(1, per // 2), 2)
        for label, _ckpt in opponents
    }


def _play_sealbot(ckpt, n, **kw):
    return _sealbot_match(kw.get("label", "hexfield"), n, 0.6)


def _run_concurrent(mse, parse_config, tmp_path, multi, *, write_diagnostics):
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40), bc=False)
    cfg = parse_config({"multi_stage_eval": {"sprt": {"enabled": False}}})
    report = mse.run_multistage_eval_concurrent(
        run,
        run / "checkpoints" / "epoch_000040.pt",
        cfg,
        candidate_epoch=40,
        write_diagnostics=write_diagnostics,
        play_multi_checkpoint_match=multi,
        play_sealbot_match=_play_sealbot,
    )
    return run, cfg, report


def _pool_rows(mse, run, cfg, epoch):
    doc = mse._load_pool(mse._pool_path(run, cfg.multi_stage_eval))
    return [r for r in doc.get("edges", []) if r.get("epoch") == epoch]


def test_crash_mid_wave_keeps_completed_edges(pkg, tmp_path):
    mse, parse_config = pkg
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40), bc=False)
    cfg = parse_config({"multi_stage_eval": {"sprt": {"enabled": False}}})
    with pytest.raises(_ProcessDeath):
        mse.run_multistage_eval_concurrent(
            run,
            run / "checkpoints" / "epoch_000040.pt",
            cfg,
            candidate_epoch=40,
            write_diagnostics=True,
            play_multi_checkpoint_match=_crash_multi,
            play_sealbot_match=_play_sealbot,
        )
    rows = _pool_rows(mse, run, cfg, 40)
    assert any(
        "sealbot" in (r.get("a"), r.get("b")) for r in rows
    ), f"SealBot edge lost by the mid-wave crash; pooled rows: {rows}"


def test_completed_wave_pools_each_edge_exactly_once(pkg, tmp_path):
    mse, parse_config = pkg
    run, cfg, report = _run_concurrent(
        mse, parse_config, tmp_path, _healthy_multi, write_diagnostics=True
    )
    rows = _pool_rows(mse, run, cfg, 40)
    keys = [(r["epoch"], frozenset((r["a"], r["b"]))) for r in rows]
    assert len(keys) == len(set(keys)), f"double-pooled edge: {rows}"
    # Every edge the wave reports made it to disk (sealbot + each checkpoint).
    assert len(rows) == len(report["edges"])
    # Stage D saw nothing to flag as a duplicate-epoch rerun.
    stage_d = next(s for s in report["stages"] if s["stage"] == "D_pool")
    assert not stage_d.get("duplicate_edges_skipped")


def test_in_memory_mode_writes_nothing(pkg, tmp_path):
    mse, parse_config = pkg
    run, cfg, _report = _run_concurrent(
        mse, parse_config, tmp_path, _healthy_multi, write_diagnostics=False
    )
    assert not mse._pool_path(run, cfg.multi_stage_eval).exists()
