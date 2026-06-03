"""Eval-schedule + restart-backfill logic (scripts/_rl_train.py).

A restart must never drop a scheduled eval: the schedule keys off the ABSOLUTE
epoch (so resuming can't shift it), and on resume a due-but-missing eval for the
last completed epoch is backfilled. These tests cover the pure decision logic
(no model/GPU needed)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    p = ROOT / "packages" / sub / "python"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def rl():
    pytest.importorskip("torch")
    return importlib.import_module("_rl_train")


def test_eval_due_is_absolute_epoch_keyed(rl):
    # eval_every=3 -> eval at 0,3,6,9 ... and the final epoch, regardless of where
    # a resume starts (the schedule is a function of the absolute epoch only).
    due = [e for e in range(10) if rl.eval_due(e, 3, total_epochs=60)]
    assert due == [0, 3, 6, 9]
    assert rl.eval_due(59, 3, total_epochs=60) is True   # final epoch always evals
    assert rl.eval_due(4, 3, total_epochs=60) is False


def test_eval_missing_keys_off_result_json(rl, tmp_path):
    assert rl.eval_missing(tmp_path, 3) is True            # no result yet
    rl.eval_result_path(tmp_path, 3).write_text("{}", encoding="utf-8")
    assert rl.eval_missing(tmp_path, 3) is False           # result present -> done


def test_resume_backfills_due_but_missing_eval(rl, tmp_path):
    # Simulate the bug: resume started at epoch 4 (prev=3 was an eval epoch), but
    # epoch-3's eval JSON is missing (killed mid-eval). The backfill must fire.
    def should_backfill(start_epoch, eval_every, total, eval_dir):
        prev = start_epoch - 1
        return prev >= 0 and rl.eval_due(prev, eval_every, total) and rl.eval_missing(eval_dir, prev)

    assert should_backfill(4, 3, 60, tmp_path) is True            # prev=3 due + missing
    rl.eval_result_path(tmp_path, 3).write_text("{}", encoding="utf-8")
    assert should_backfill(4, 3, 60, tmp_path) is False           # now present -> skip
    assert should_backfill(5, 3, 60, tmp_path) is False           # prev=4 not an eval epoch
    assert should_backfill(0, 3, 60, tmp_path) is False           # nothing before epoch 0
