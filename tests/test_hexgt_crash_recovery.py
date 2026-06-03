"""Crash-recovery: transient-CUDA retry guard + re-train-on-crash resume.

The CUDA `device not ready` (a WSL2/consumer-GPU TDR-style transient) crashed
training twice — once in the forward SDPA, once in loss.backward(). Two fixes:
  - trainer.optimizer_step RETRIES a transient CUDA error (synchronize + backoff)
    before giving up; real errors (OOM/illegal/shape) still raise immediately.
  - _rl_train resume RE-TRAINS an epoch whose training crashed mid-way (using the
    existing self-play shards, no re-self-play, no epoch skip) instead of advancing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    p = ROOT / "packages" / package / "python"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _torch():
    return pytest.importorskip("torch")


# --- transient-error classifier ----------------------------------------------

def test_is_transient_cuda_error_classification() -> None:
    from hexo_models.hexgt.trainer import _is_transient_cuda_error as t
    assert t(RuntimeError("CUDA driver error: device not ready"))
    assert t(RuntimeError("cudaErrorNotReady"))
    # non-transient must NOT match (so they still raise)
    assert not t(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not t(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not t(RuntimeError("device-side assert triggered"))
    assert not t(RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED"))
    assert not t(ValueError("some shape error"))
    # mixed: a non-transient marker present -> treated non-transient (raise)
    assert not t(RuntimeError("device not ready ... out of memory"))


# --- re-train-on-crash resume decision ---------------------------------------

def test_resume_plan_complete_advances_incomplete_retrains() -> None:
    import _rl_train as rl
    assert rl.resume_plan(13, True) == (14, False)    # completed -> advance
    assert rl.resume_plan(14, False) == (14, True)    # crashed mid-train -> re-train 14
    assert rl.resume_plan(0, True) == (1, False)
    # old checkpoints lack the flag -> ck.get(..., True) -> complete -> advance
    assert rl.resume_plan(7, True)[1] is False


def test_should_skip_selfplay_only_first_epoch_when_incomplete() -> None:
    import _rl_train as rl
    # re-training epoch 14: skip self-play ONLY for epoch 14, only on resume-incomplete
    assert rl.should_skip_selfplay(14, 14, True) is True
    assert rl.should_skip_selfplay(15, 14, True) is False   # next epoch does self-play
    assert rl.should_skip_selfplay(14, 14, False) is False  # normal resume: no skip


# --- retry guard: trainer.optimizer_step --------------------------------------

def _trainer_and_batch():
    torch = _torch()
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.config import parse_hexgt_config
    from hexo_models.hexgt.graph_build import batch_from_states
    from hexo_models.hexgt.trainer import HexgtTrainer
    import hexo_engine.api as eng
    import random

    cfg = parse_hexgt_config({"device": "cpu", "architecture": {"candidate_radius": 3},
                              "training": {"batch_size": 4}})
    torch.manual_seed(0)
    model = HexgtNetwork(token_dim=48, gnn_layers=2, ctx_layers=2, attention_heads=4, ffn_dim=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = HexgtTrainer(model=model, config=cfg, optimizer=opt)
    trainer.cuda_retry_backoff = 0.0  # no sleep in tests

    rng = random.Random(1)
    s = eng.new_game(seed=1)
    for _ in range(20):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        if not acts:
            break
        eng.apply_action(s, rng.choice(acts))
    batch = batch_from_states([s], n=3)
    g = int(batch["num_graphs"]); ncand = int(batch["candidate_graph"].shape[0])
    targets = {
        "candidate_graph": batch["candidate_graph"], "num_graphs": g,
        "policy": torch.ones(ncand), "opp_policy": torch.ones(ncand),
        "value": torch.zeros(g),
    }
    return trainer, model, batch, targets


def _flaky_forward(model, fail_times, exc):
    real = model.forward
    state = {"n": 0}

    def fwd(batch):
        state["n"] += 1
        if state["n"] <= fail_times:
            raise exc
        return real(batch)

    model.forward = fwd
    return state


def test_optimizer_step_retries_transient_then_succeeds() -> None:
    torch = _torch()
    trainer, model, batch, targets = _trainer_and_batch()
    state = _flaky_forward(model, 1, RuntimeError("CUDA driver error: device not ready"))
    report = trainer.optimizer_step(batch, targets)  # should retry once, then succeed
    assert state["n"] == 2, "expected one retry then success"
    assert "total" in report and trainer.train_state.global_step == 1


def test_optimizer_step_raises_on_oom_no_retry() -> None:
    trainer, model, batch, targets = _trainer_and_batch()
    state = _flaky_forward(model, 99, RuntimeError("CUDA out of memory. Tried to allocate"))
    with pytest.raises(RuntimeError, match="out of memory"):
        trainer.optimizer_step(batch, targets)
    assert state["n"] == 1, "OOM must NOT be retried"
    assert trainer.train_state.global_step == 0


def test_optimizer_step_raises_after_exhausting_retries() -> None:
    trainer, model, batch, targets = _trainer_and_batch()
    trainer.cuda_retry_attempts = 3
    state = _flaky_forward(model, 99, RuntimeError("CUDA driver error: device not ready"))
    with pytest.raises(RuntimeError, match="device not ready"):
        trainer.optimizer_step(batch, targets)
    assert state["n"] == 3, "should try exactly cuda_retry_attempts times"
    assert trainer.train_state.global_step == 0
