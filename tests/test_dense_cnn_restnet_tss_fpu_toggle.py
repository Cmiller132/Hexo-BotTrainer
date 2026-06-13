"""TSS / root-FPU-zeroing config toggles for dense_cnn_restnet (2026-06-12).

Covers: the `selfplay.tss_enabled` and `selfplay.root_fpu_zero_under_noise`
config keys (default true = current behavior), the rust_bridge old-.so
compatibility contract (the new positional args are appended ONLY when a
config opts out, so the live run's in-memory native module never sees them),
and — when the installed native module is new enough — that explicit-true
flags are byte-equivalent to omitting them, and that the disabled search
accepts the flags end-to-end.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
_RESTNET = ROOT / "packages" / "dense_cnn_restnet" / "python"
if str(_RESTNET) not in sys.path:
    sys.path.insert(0, str(_RESTNET))

config_module = importlib.import_module("dense_cnn_restnet.config")
rust_bridge = importlib.import_module("dense_cnn_restnet.rust_bridge")


# --- config keys ----------------------------------------------------------------


def test_config_defaults_keep_current_behavior():
    parsed = config_module.parse_model1_config({})
    assert parsed.selfplay.tss_enabled is True
    assert parsed.selfplay.root_fpu_zero_under_noise is True


def test_config_parses_opt_outs():
    parsed = config_module.parse_model1_config(
        {"selfplay": {"tss_enabled": False, "root_fpu_zero_under_noise": False}}
    )
    assert parsed.selfplay.tss_enabled is False
    assert parsed.selfplay.root_fpu_zero_under_noise is False


# --- rust_bridge old-.so compatibility contract ----------------------------------

# Positional arg counts of the native signatures BEFORE the flags were added.
_SEARCH_LEGACY_ARGC = 20
_CONTINUOUS_LEGACY_ARGC = 28


class _CapturingSession:
    def __init__(self):
        self.search_args = None
        self.continuous_args = None

    def search(self, *args):
        self.search_args = args
        return ()

    def run_continuous(self, *args):
        self.continuous_args = args
        return {}


def _search_kwargs(session, **extra):
    return dict(
        session=session,
        game_keys=[1],
        states=["state"],
        visits=4,
        c_puct=1.5,
        temperature=1.0,
        seed=0,
        evaluator=object(),
        **extra,
    )


def test_search_omits_flags_when_unset():
    session = _CapturingSession()
    rust_bridge.model1_mcts_session_search(**_search_kwargs(session))
    assert len(session.search_args) == _SEARCH_LEGACY_ARGC


def test_search_appends_flags_only_on_opt_out():
    session = _CapturingSession()
    rust_bridge.model1_mcts_session_search(
        **_search_kwargs(session, tss_enabled=False, root_fpu_zero_under_noise=False)
    )
    assert len(session.search_args) == _SEARCH_LEGACY_ARGC + 2
    assert session.search_args[-2] is False
    assert session.search_args[-1] is False


def test_search_appends_both_when_either_set():
    # Positional forwarding cannot skip a middle arg: setting either flag
    # appends both (the unset one as None = native default).
    session = _CapturingSession()
    rust_bridge.model1_mcts_session_search(**_search_kwargs(session, tss_enabled=False))
    assert len(session.search_args) == _SEARCH_LEGACY_ARGC + 2
    assert session.search_args[-2] is False
    assert session.search_args[-1] is None


def _continuous_kwargs(session, **extra):
    return dict(
        session=session,
        game_keys=[1],
        states=["state"],
        evaluator=object(),
        on_move=object(),
        visits=4,
        c_puct=1.5,
        base_seed=0,
        virtual_batch_size=2,
        flush_target=2,
        active_root_limit=4,
        temperature_by_ply=[1.0],
        **extra,
    )


def test_continuous_omits_flags_when_unset():
    session = _CapturingSession()
    rust_bridge.model1_mcts_session_run_continuous(**_continuous_kwargs(session))
    assert len(session.continuous_args) == _CONTINUOUS_LEGACY_ARGC


def test_continuous_appends_flags_only_on_opt_out():
    session = _CapturingSession()
    rust_bridge.model1_mcts_session_run_continuous(
        **_continuous_kwargs(session, tss_enabled=False, root_fpu_zero_under_noise=False)
    )
    assert len(session.continuous_args) == _CONTINUOUS_LEGACY_ARGC + 2
    assert session.continuous_args[-2] is False
    assert session.continuous_args[-1] is False


# --- native end-to-end (requires a module built >= 2026-06-12) -------------------


def _native_search(*, seed, **flags):
    """One 8-visit search on a fresh game/session with a trivial uniform net
    (the continuous-scheduler test pattern: sub-ms forward, real Rust search)."""

    torch = pytest.importorskip("torch")
    engine = importlib.import_module("hexo_engine")
    constants = importlib.import_module("dense_cnn_restnet.constants")
    inference_module = importlib.import_module("dense_cnn_restnet.inference")
    mcts_module = importlib.import_module("dense_cnn_restnet.mcts")

    class _UniformPV(torch.nn.Module):
        def forward_policy_value(self, inputs):
            batch = int(inputs.shape[0])
            return {
                "policy": torch.zeros((batch, constants.BOARD_AREA), dtype=torch.float32),
                "value": torch.zeros((batch, constants.VALUE_BINS), dtype=torch.float32),
            }

        def forward(self, inputs):
            return self.forward_policy_value(inputs)

    session = mcts_module.new_mcts_session(max_states=4096)
    inference = inference_module.DenseCNNInference(
        _UniformPV(), device="cpu", amp=False, max_batch_size=32
    )
    state = engine.new_game(seed=7)
    results = session.run(
        [1],
        [state],
        inference,
        visits=8,
        c_puct=1.5,
        temperature=1.0,
        seed=seed,
        root_dirichlet_total_alpha=10.83,
        root_dirichlet_noise_fraction=0.25,
        fpu_reduction=0.20,
        **flags,
    )
    return results[0]


def test_native_explicit_true_matches_omitted():
    baseline = _native_search(seed=11)
    try:
        explicit = _native_search(seed=11, tss_enabled=True, root_fpu_zero_under_noise=True)
    except TypeError as exc:
        pytest.skip(f"native module predates tss/fpu flags (rebuild pending): {exc}")
    assert explicit.action_id == baseline.action_id
    assert list(explicit.visit_policy) == list(baseline.visit_policy)
    assert explicit.root_value == baseline.root_value


def test_native_accepts_opt_outs():
    try:
        result = _native_search(seed=13, tss_enabled=False, root_fpu_zero_under_noise=False)
    except TypeError as exc:
        pytest.skip(f"native module predates tss/fpu flags (rebuild pending): {exc}")
    assert result.visits == 8
    assert len(list(result.visit_policy)) >= 1
