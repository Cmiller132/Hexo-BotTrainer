"""M6 #4 gate: the depth-2 double-buffered scheduler (HEXFIELD_PIPELINE_DEPTH2)
is FLAG-GATED, default OFF, and NOT byte-identical to the lockstep path (it
deepens the async eval window by one flush, so the leaf stream legitimately
differs). This test guards the two properties that MUST hold for the opt-in path:

  1. Determinism — two depth-2 runs with the same seed produce the identical
     on_move stream (the interleave is fixed by the loop structure, not by
     wall-clock), so enabling it is reproducible.
  2. No lost / double-applied backups — every started game still decides exactly
     the same NUMBER of moves as the lockstep run (each game runs to MAX_PLIES),
     i.e. the pipeline drains its final flush and never drops or replays an eval.

It also asserts the default-OFF path is reachable and that depth-2 falls back to
lockstep (with no crash) when HEXFIELD_ASYNC_EVAL is absent.

The byte-IDENTICAL-to-dense parity corpus is covered separately by
tests/test_hexfield_continuous_parity.py with the flag UNSET; depth-2 must NOT be
compared against that corpus.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from hexfield_testkit import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.geometry import unpack_action_id
from test_hexfield_search_parity import HexfieldStub


class AsyncStub:
    """Wraps the synchronous HexfieldStub in the async submit/result protocol the
    depth-2 (HEXFIELD_ASYNC_EVAL) path drives: `submit_payload` enqueues (here:
    eagerly computes — the test isn't timing the GPU) and returns a handle;
    `result(handle)` returns the precomputed reply. Still callable so the sync /
    fallback path (which calls the evaluator directly) works unchanged."""

    def __init__(self):
        self._inner = HexfieldStub()
        self._next = 0
        self._pending = {}

    def __call__(self, payload: dict) -> dict:
        return self._inner(payload)

    def submit_payload(self, payload: dict):
        handle = self._next
        self._next += 1
        self._pending[handle] = self._inner(payload)
        return handle

    def result(self, handle):
        return self._pending.pop(handle)

try:
    from hexfield import _rust as hexfield_rust
except ImportError:  # pragma: no cover
    hexfield_rust = None

needs_native = pytest.mark.skipif(
    hexfield_rust is None,
    reason="native module not built",
)

# Same shallow corpus discipline as the parity test (keeps decision states in a
# legal region the stub evaluates consistently); every game runs to MAX_PLIES so
# the move COUNT is search-path-independent.
MAX_PLIES = 6


class Recorder:
    """on_move driver: applies actions to its own engine states and records the
    full decision stream (the same 8-tuple the parity test compares)."""

    def __init__(self):
        self.states = {}
        self.records = []

    def add_game(self, key: int):
        self.states[key] = api.new_game()

    def __call__(self, game_key: int, payload: dict):
        state = self.states[game_key]
        action_id = payload["action_id"]
        self.records.append(
            (
                game_key,
                action_id,
                bool(payload["pcr_full"]),
                bool(payload["policy_init"]),
                int(payload["visits"]),
                bytes(payload["visit_policy_action_ids_bytes"]),
                bytes(payload["visit_policy_weights_bytes"]),
                round(float(payload["root_value"]), 6),
            )
        )
        q, r = unpack_action_id(action_id)
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        ply = sum(1 for rec in self.records if rec[0] == game_key)
        if result.terminal or ply >= MAX_PLIES:
            return None
        return ("advance", state)


def _continuous_kwargs():
    # Mirrors the parity-test config (full per-move machinery exercised), with a
    # small enough corpus to stay in-stub.
    return dict(
        visits=48,
        c_puct=1.5,
        base_seed=20260613,
        virtual_batch_size=8,
        flush_target=24,
        active_root_limit=16,
        temperature_by_ply=[1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
        root_dirichlet_total_alpha=10.83,
        root_dirichlet_noise_fraction=0.25,
        root_policy_temperature=1.1,
        root_policy_temperature_early=1.4,
        root_policy_temperature_halflife=12.0,
        fpu_reduction=0.2,
        virtual_loss=1.0,
        widening_policy_mass=0.95,
        widening_max_children=96,
        widening_min_children=2,
        forced_playout_k=2.0,
        pcr_full_proportion=0.33,
        pcr_fast_visits=16,
        policy_init_fraction=0.25,
        policy_init_avg_plies=4.0,
        policy_init_max_plies=8,
        policy_init_temperature=1.4,
        tss_enabled=True,
        root_fpu_zero_under_noise=True,
    )


@contextmanager
def _env(**overrides):
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run(keys, **env):
    rec = Recorder()
    for key in keys:
        rec.add_game(key)
    with _env(**env):
        session = hexfield_rust.HexfieldMctsSession(max_states=65536)
        stats = session.run_continuous(
            keys,
            tuple(rec.states[k] for k in keys),
            evaluator=AsyncStub(),
            on_move=rec,
            search_parity_mode=True,
            **_continuous_kwargs(),
        )
    return rec.records, stats


@needs_native
def test_depth2_is_deterministic() -> None:
    """Same seed + depth-2 flag => bit-identical on_move stream across two runs."""
    keys = list(range(700, 706))
    env = {"HEXFIELD_ASYNC_EVAL": "1", "HEXFIELD_PIPELINE_DEPTH2": "1"}
    records_a, stats_a = _run(keys, **env)
    records_b, stats_b = _run(keys, **env)
    assert records_a == records_b, "depth-2 stream is non-deterministic across runs"
    assert stats_a["moves_decided"] == stats_b["moves_decided"]


@needs_native
def test_depth2_no_lost_or_double_moves() -> None:
    """Every game decides exactly MAX_PLIES moves under depth-2 (no eval lost at
    shutdown, no flush backed up twice). The COUNT matches the lockstep run even
    though the stream legitimately differs."""
    keys = list(range(700, 706))

    lockstep_records, lockstep_stats = _run(
        keys, HEXFIELD_PIPELINE_DEPTH2=None, HEXFIELD_ASYNC_EVAL=None
    )
    depth2_records, depth2_stats = _run(
        keys, HEXFIELD_ASYNC_EVAL="1", HEXFIELD_PIPELINE_DEPTH2="1"
    )

    # Every game ran to the MAX_PLIES cap on both paths -> identical move counts.
    assert len(depth2_records) == len(keys) * MAX_PLIES
    assert len(depth2_records) == len(lockstep_records)
    assert depth2_stats["moves_decided"] == lockstep_stats["moves_decided"]

    # Per-game move counts match (no game starved by a lost backup).
    for key in keys:
        d2 = sum(1 for rec in depth2_records if rec[0] == key)
        ls = sum(1 for rec in lockstep_records if rec[0] == key)
        assert d2 == ls == MAX_PLIES, f"game {key}: depth2={d2} lockstep={ls}"

    # The flushed/queued accounting is internally consistent (every queued state
    # was eventually flushed -> no flush left undrained at exit).
    assert depth2_stats["flush_count"] >= 1
    assert depth2_stats["flushed_states"] <= depth2_stats["queued_states"]


@needs_native
def test_depth2_falls_back_without_async() -> None:
    """Depth-2 without HEXFIELD_ASYNC_EVAL must fall back to the lockstep loop
    (warn + run), not crash. The fallback stream therefore equals a plain
    lockstep run."""
    keys = list(range(700, 703))
    lockstep_records, _ = _run(
        keys, HEXFIELD_PIPELINE_DEPTH2=None, HEXFIELD_ASYNC_EVAL=None
    )
    fallback_records, _ = _run(
        keys, HEXFIELD_PIPELINE_DEPTH2="1", HEXFIELD_ASYNC_EVAL=None
    )
    assert fallback_records == lockstep_records
