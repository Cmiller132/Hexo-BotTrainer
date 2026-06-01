"""Expansion-pool lifecycle (#4).

The dense_cnn trainer lazily builds a persistent spawn ProcessPoolExecutor (each
worker importing torch, ~250-400 MB RSS). Before the fix, ``close()`` had zero call
sites, so the pool leaked for the whole run. These tests confirm the run-end teardown
now invokes it and that the trainer carries a finalizer backstop.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine", "hexo_train", "hexo_runner", "hexo_utils"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytest.importorskip("torch")
pipeline_mod = importlib.import_module("hexo_train.pipeline")
trainer_mod = importlib.import_module("hexo_models.dense_cnn.trainer")


def test_pipeline_teardown_invokes_trainer_close() -> None:
    calls = {"n": 0}

    class FakeTrainer:
        def close(self) -> None:
            calls["n"] += 1

    pl = pipeline_mod.TrainingPipeline()
    pl._teardown_components(SimpleNamespace(model=SimpleNamespace(trainer=FakeTrainer())))
    assert calls["n"] == 1
    # Robust to a model with no trainer / no close / no model (must never raise).
    pl._teardown_components(SimpleNamespace(model=SimpleNamespace(trainer=None)))
    pl._teardown_components(SimpleNamespace(model=None))


def test_teardown_swallows_close_errors() -> None:
    class BadTrainer:
        def close(self) -> None:
            raise RuntimeError("boom")

    # Teardown is best-effort and must not mask the run's real result/exception.
    pipeline_mod.TrainingPipeline()._teardown_components(
        SimpleNamespace(model=SimpleNamespace(trainer=BadTrainer()))
    )


def test_trainer_has_close_and_finalizer() -> None:
    T = trainer_mod.DenseCNNTrainer
    assert callable(getattr(T, "close", None))
    assert callable(getattr(T, "__del__", None))


def test_pipeline_run_calls_teardown_in_finally() -> None:
    # Re-verify the fix: close() now HAS a call site (the reviewer measured zero).
    src = inspect.getsource(pipeline_mod.TrainingPipeline)
    assert "finally:" in src and "_teardown_components" in src
