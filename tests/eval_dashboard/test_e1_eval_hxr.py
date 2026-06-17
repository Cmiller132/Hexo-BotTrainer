"""E1 — eval .hxr must persist real game actions (not header-only stubs).

Proof that eval games WILL populate the dashboard History: build synthetic eval
``_Game`` objects WITH a real action list, run them through the actual writer
``_write_eval_hxr`` (the exact function the live eval calls), and assert the
written .hxr decodes with ``num_records > 0`` and the actions round-trip through
the real Rust-backed ``HexoRecordFile`` codec.

Also proves the E1 hardening: a 0-record write (all games had empty .actions) is
now LOUD (warning logged) + machine-visible (``stats['games_written'] == 0``)
instead of being silently swallowed.

CPU-only, no torch, no GPU.

Run:
  wsl bash -lc 'cd /mnt/e/hexgt-evaldash && HEXFIELD_SUPPORT_RADIUS=4 \
    PYTHONPATH=packages/hexfield/python:packages/dense_cnn_restnet/python \
    /root/.venvs/hexgt-build/bin/python tests/eval_dashboard/test_e1_eval_hxr.py'
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from hexo_runner.records import HexoRecordFile

from hexfield.eval_arena import _write_eval_hxr
from hexfield.geometry import pack_action_id


class _StubGame:
    """Minimal stand-in for eval_arena._Game with exactly the attrs the writer
    reads: index, a_is_p0, seed, winner, plies, actions (packed action ids)."""

    def __init__(self, index, a_is_p0, seed, winner, actions):
        self.index = index
        self.a_is_p0 = a_is_p0
        self.seed = seed
        self.winner = winner  # "A" | "B" | None
        self.actions = list(actions)
        self.plies = len(actions)


def _make_actions(coords):
    return [pack_action_id(q, r) for (q, r) in coords]


def test_actions_roundtrip_to_hxr():
    diag = Path(tempfile.mkdtemp()) / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    a_coords = [(0, 0), (1, 0), (-1, 1)]
    b_coords = [(0, 0), (0, 1)]
    games = [
        _StubGame(0, a_is_p0=True, seed=11, winner="A", actions=_make_actions(a_coords)),
        _StubGame(1, a_is_p0=False, seed=11, winner="B", actions=_make_actions(b_coords)),
    ]

    stats: dict[str, int] = {}
    path = _write_eval_hxr(games, diag, "ep35", "ep30", stats=stats)
    assert path is not None, "writer returned None despite 2 games with real actions"
    assert Path(path).is_file(), f"hxr file not written: {path}"
    assert stats.get("games_written") == 2, stats
    assert stats.get("games_skipped") == 0, stats

    rf = HexoRecordFile.open(path)
    recs = list(rf.iter_records())
    assert len(recs) == 2, f"expected 2 records, got {len(recs)}"

    # Round-trip: action_ids non-empty, lengths match, game_id pattern, winner set.
    by_id = {r.game_id: r for r in recs}
    g0 = next(r for gid, r in by_id.items() if gid.endswith("g0-candP0"))
    g1 = next(r for gid, r in by_id.items() if gid.endswith("g1-candP1"))
    assert list(g0.action_ids), "game 0 has no action_ids"
    assert len(list(g0.action_ids)) == len(a_coords), "game 0 action count mismatch"
    assert len(list(g1.action_ids)) == len(b_coords), "game 1 action count mismatch"
    assert g0.game_id.startswith("ep35-ep35-vs-ep30-"), g0.game_id
    assert all(r.winner is not None for r in recs), "winner not set on decided games"
    print(f"[E1] PASS roundtrip: {len(recs)} records, g0={g0.game_id} "
          f"actions={len(list(g0.action_ids))} winner={g0.winner}")


def test_zero_record_is_loud():
    diag = Path(tempfile.mkdtemp()) / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    # All games have empty .actions (the exact regression that emptied live .hxr).
    games = [
        _StubGame(0, a_is_p0=True, seed=1, winner="A", actions=[]),
        _StubGame(1, a_is_p0=False, seed=1, winner="B", actions=[]),
    ]

    handler = _CaptureHandler()
    logger = logging.getLogger("hexfield.eval")
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        stats: dict[str, int] = {}
        path = _write_eval_hxr(games, diag, "ep35", "ep30", stats=stats)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    assert path is None, "expected None when 0 of N games written"
    assert stats.get("games_written") == 0, stats
    assert stats.get("games_skipped") == 2, stats
    warned = [r for r in handler.records if r.levelno >= logging.WARNING]
    assert any("wrote 0 of 2 games" in r.getMessage() for r in warned), (
        "0-record write was NOT loud; warnings=" + repr([r.getMessage() for r in warned])
    )
    print(f"[E1] PASS loudness: 0-record write logged WARNING "
          f"({[r.getMessage() for r in warned]})")


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


if __name__ == "__main__":
    test_actions_roundtrip_to_hxr()
    test_zero_record_is_loud()
    print("E1 ALL GREEN")
