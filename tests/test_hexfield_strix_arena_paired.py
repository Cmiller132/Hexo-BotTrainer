"""CPU smoke test for the PAIRED ``eval_arena.play_strix_match`` loop.

The strix-anchor tests mock ``play_strix_match`` wholesale, so the arena's own
leader/follower CRN replay + pentanomial assembly is never exercised there. This
drives the REAL round loop through the four injection seams
(``build_hexfield_evaluator`` / ``make_session`` / ``build_strix_player`` /
``make_batch_server``) with fakes that play real ``hexo_engine`` games, so no
GPU / torch / ``hexo_rs`` wheel is needed.

The load-bearing check: the two games of a CRN pair share an OPENING line even
though their seats are swapped. The fakes make hexfield pick the FIRST legal move
and strix the LAST — so a follower that searched independently would produce a
different opening (its seats are swapped), and opening equality can only hold if
the follower REPLAYS the leader's recorded line.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "packages" / "hexo_engine" / "python"))
sys.path.insert(0, str(_REPO / "packages" / "hexfield" / "python"))

engine = pytest.importorskip("hexo_engine")
from hexo_engine import api  # noqa: E402
from hexo_engine.types import pack_coord_id as _pack  # noqa: E402

from hexfield import eval_arena  # noqa: E402
from hexfield.config import parse_hexfield_config  # noqa: E402


class _FakeSession:
    """Multi-root hexfield session: each root plays its FIRST legal move."""

    def __init__(self) -> None:
        self.discarded: list[int] = []

    def search(self, indices, states, **kw):
        out = []
        for st in states:
            coord = api.legal_actions(st)[0].coord
            out.append({"action_id": _pack(coord), "root_value": 0.0, "visits": 1})
        return out

    def discard(self, index) -> None:
        self.discarded.append(index)


class _FakeStrix:
    """Strix player: plays the LAST legal move (distinct from hexfield's first)."""

    def __init__(self, seed: int, is_leader: bool) -> None:
        self.seed = seed
        self.is_leader = is_leader

    def decide(self, state):
        return SimpleNamespace(action=api.legal_actions(state)[-1])


class _FakeServer:
    def __init__(self) -> None:
        self.closed = 0
        self.n_forwards = 0
        self.n_graphs = 0
        self.max_seen_batch = 0

    def close(self) -> None:
        self.closed += 1


def _cfg(max_plies: int):
    cfg = parse_hexfield_config({})
    cfg = dataclasses.replace(
        cfg, selfplay=dataclasses.replace(cfg.selfplay, max_game_plies=max_plies)
    )
    return cfg


def test_paired_strix_arena_shares_opening_and_builds_pentanomial(tmp_path: Path) -> None:
    session = _FakeSession()
    server = _FakeServer()
    n_games = 4
    opening_plies = 4

    result = eval_arena.play_strix_match(
        "hexfield.pt", "strix.pt", n_games,
        config=_cfg(max_plies=16),
        opening_plies=opening_plies,
        paired_openings=True,
        diagnostics_dir=str(tmp_path),
        build_hexfield_evaluator=lambda: object(),
        make_session=lambda: session,
        build_strix_player=lambda seed, is_leader: _FakeStrix(seed, is_leader),
        make_batch_server=lambda: server,
    )

    # Pentanomial block present, one row per CRN pair, net-A-centric shape.
    pent = result["pentanomial"]
    assert pent is not None
    assert pent["n_pairs"] == n_games // 2
    assert result["meta"]["paired_openings"] is True
    assert result["meta"]["kind"] == "hexfield_vs_strix"

    # The load-bearing invariant: within a pair the seat-swapped games traverse
    # the SAME opening line (follower replayed the leader) — impossible under the
    # first/last asymmetry unless the follower replays.
    games = {g["index"]: g for g in result["games"]}
    for pair_index in range(n_games // 2):
        leader, follower = games[2 * pair_index], games[2 * pair_index + 1]
        assert leader["a_seat"] == "P0" and follower["a_seat"] == "P1"  # seats swapped
        shared = min(len(leader["opening"]), len(follower["opening"]), opening_plies)
        assert shared > 0
        assert leader["opening"][:shared] == follower["opening"][:shared]

    # The loop terminated cleanly: every game discarded, server drained.
    assert sorted(session.discarded) == list(range(n_games))
    assert server.closed >= 1


def test_unpaired_strix_arena_has_no_pentanomial(tmp_path: Path) -> None:
    """``paired_openings=False`` keeps the legacy independent-Bernoulli layout:
    seats alternate by index and there is no pentanomial block."""

    session = _FakeSession()
    server = _FakeServer()
    result = eval_arena.play_strix_match(
        "hexfield.pt", "strix.pt", 4,
        config=_cfg(max_plies=16),
        opening_plies=4,
        paired_openings=False,
        diagnostics_dir=str(tmp_path),
        build_hexfield_evaluator=lambda: object(),
        make_session=lambda: session,
        build_strix_player=lambda seed, is_leader: _FakeStrix(seed, is_leader),
        make_batch_server=lambda: server,
    )
    assert result["pentanomial"] is None
    assert result["meta"]["paired_openings"] is False
    # Seats still alternate by game index (even -> hexfield player0).
    games = {g["index"]: g for g in result["games"]}
    assert games[0]["a_seat"] == "P0" and games[1]["a_seat"] == "P1"
    assert sorted(session.discarded) == [0, 1, 2, 3]
