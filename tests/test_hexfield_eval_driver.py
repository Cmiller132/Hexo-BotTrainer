"""Unit tests for the shared eval-driver primitives (``hexfield.eval_driver``).

Pure CPU: a deterministic fake engine (monkeypatched onto ``eval_driver.api``)
plus the fake multi-root session / evaluator from ``hexfield_eval_kit``. No torch,
no native ``.so``. Exercises ``EvalGame`` / ``build_paired_games`` (CRN paired
construction, leader/follower, singleton, unpaired seed streams, net-B factory),
``finalize`` / ``settle`` / ``record_move`` / ``follower_opening_action``, and the
``run_hexfield_ply`` multi-root chunking + positional zip-scatter (never re-sort).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in ("hexo_engine/python", "hexfield/python"):
    _src = str(_REPO / "packages" / _p)
    if _src not in sys.path:
        sys.path.insert(0, _src)

from hexfield import eval_driver  # noqa: E402  (torch-free, .so-free import)
from hexfield.config import build_eval_search_kwargs, parse_hexfield_config  # noqa: E402
from hexfield.eval_driver import (  # noqa: E402
    EvalGame,
    OpponentAdapter,
    build_paired_games,
    finalize,
    follower_opening_action,
    record_move,
    run_hexfield_ply,
    settle,
)
from hexfield.geometry import unpack_action_id  # noqa: E402

from hexfield_eval_kit import (  # noqa: E402
    _FakeApi,
    _FakeEvaluator,
    _FakeSession,
    _FakeState,
    _Terminal,
)


@pytest.fixture
def fake_api(monkeypatch):
    """A deterministic fake engine monkeypatched onto ``eval_driver.api``.

    Winners are decided by a seat the test sets on the state (``winner_seat``);
    ``decide_winner`` returns seat 0 by default (only consulted when a move drives
    a state to ``game_len``). Long ``game_len`` so tests control termination.
    """

    api = _FakeApi(game_len=1_000, decide_winner=lambda s: 0)
    monkeypatch.setattr(eval_driver, "api", api)
    return api


def _new_state(api: _FakeApi):
    return api.new_game()


# =========================================================================== #
# build_paired_games.
# =========================================================================== #
def test_build_paired_games_paired_structure(fake_api):
    games, pair_members = build_paired_games(
        8, game_seed_base=100, paired_openings=True, new_state=lambda seed: _new_state(fake_api)
    )

    assert len(games) == 8
    assert sorted(pair_members) == [0, 1, 2, 3]
    for pair_index, members in pair_members.items():
        assert len(members) == 2
        g0, g1 = members
        # game 0 is the leader (net A as player0); game 1 the seat-swapped follower.
        assert g0.index == 2 * pair_index
        assert g1.index == 2 * pair_index + 1
        assert g0.a_is_p0 is True and g1.a_is_p0 is False
        assert g0.is_leader is True and g0.leader is g0
        assert g1.is_leader is False and g1.leader is g0
        # Shared CRN seed within the pair: game_seed_base + pair_index.
        assert g0.seed == g1.seed == 100 + pair_index
        assert g0.pair_index == g1.pair_index == pair_index
        # No net-B factory -> b_player unset.
        assert g0.b_player is None and g1.b_player is None
        assert g0.state is not None and g1.state is not None


def test_build_paired_games_odd_singleton(fake_api):
    games, pair_members = build_paired_games(
        3, game_seed_base=0, paired_openings=True, new_state=lambda seed: _new_state(fake_api)
    )
    assert len(games) == 3
    assert len(pair_members[0]) == 2
    # Last pair is a singleton leader only (no seat-swapped follower).
    assert len(pair_members[1]) == 1
    lone = pair_members[1][0]
    assert lone.index == 2
    assert lone.is_leader is True and lone.leader is lone
    assert lone.a_is_p0 is True


def test_build_paired_games_unpaired_default_seeds(fake_api):
    games, pair_members = build_paired_games(
        4, game_seed_base=50, paired_openings=False, new_state=lambda seed: _new_state(fake_api)
    )
    assert pair_members == {}
    assert [g.index for g in games] == [0, 1, 2, 3]
    # Seats alternate by index; seed = base + index (no side offset).
    assert [g.a_is_p0 for g in games] == [True, False, True, False]
    assert [g.seed for g in games] == [50, 51, 52, 53]
    assert all(g.pair_index == -1 for g in games)


def test_build_paired_games_unpaired_side_seed_offset(fake_api):
    offset = {"a": 0, "b": 500_009_999}
    games, _ = build_paired_games(
        4,
        game_seed_base=10,
        paired_openings=False,
        new_state=lambda seed: _new_state(fake_api),
        side_seed_offset=offset,
    )
    # Even index -> "a" offset (0); odd index -> "b" offset (decorrelated stream).
    assert [g.seed for g in games] == [
        10 + 0 + 0,
        10 + 1 + 500_009_999,
        10 + 2 + 0,
        10 + 3 + 500_009_999,
    ]


def test_build_paired_games_b_player_factory(fake_api):
    seen_seeds: list[int] = []

    def factory(seed: int) -> str:
        seen_seeds.append(seed)
        return f"bp-{seed}"

    games, _ = build_paired_games(
        4,
        game_seed_base=7,
        paired_openings=True,
        new_state=lambda seed: _new_state(fake_api),
        b_player_factory=factory,
    )
    # One net-B player per game, built with that game's (CRN) seed.
    assert [g.b_player for g in games] == ["bp-7", "bp-7", "bp-8", "bp-8"]
    assert seen_seeds == [7, 7, 8, 8]


# =========================================================================== #
# finalize / settle.
# =========================================================================== #
def _terminal_game(fake_api: _FakeApi, *, a_is_p0: bool, winner_seat: int) -> EvalGame:
    st = _FakeState(game_len=1_000)
    st.winner_seat = winner_seat
    return EvalGame(index=0, pair_index=-1, a_is_p0=a_is_p0, seed=0, state=st)


def test_finalize_winner_is_net_a_centric(fake_api):
    discarded: list[int] = []
    on_discard = lambda g: discarded.append(g.index)

    # Net A holds seat 0 and seat 0 won -> net A wins.
    g = _terminal_game(fake_api, a_is_p0=True, winner_seat=0)
    finalize(g, budget_hit=False, on_discard=on_discard)
    assert g.status == "completed" and g.winner == "A" and g.done is True

    # Net A holds seat 1 but seat 0 won -> net B wins.
    g = _terminal_game(fake_api, a_is_p0=False, winner_seat=0)
    finalize(g, budget_hit=False, on_discard=on_discard)
    assert g.winner == "B"

    assert discarded == [0, 0]  # on_discard fired once per finalize


def test_finalize_non_terminal_status(fake_api):
    on_discard = lambda g: None
    # Non-terminal + budget hit -> aborted_budget.
    g = EvalGame(index=0, pair_index=-1, a_is_p0=True, seed=0, state=_new_state(fake_api))
    finalize(g, budget_hit=True, on_discard=on_discard)
    assert g.status == "aborted_budget" and g.winner is None and g.done is True

    # Non-terminal, no budget -> truncated.
    g = EvalGame(index=1, pair_index=-1, a_is_p0=True, seed=0, state=_new_state(fake_api))
    finalize(g, budget_hit=False, on_discard=on_discard)
    assert g.status == "truncated" and g.winner is None


def test_settle_returns_and_finalizes(fake_api):
    on_discard = lambda g: None

    # Not terminal, under max_plies -> no finalize.
    g = EvalGame(index=0, pair_index=-1, a_is_p0=True, seed=0, state=_new_state(fake_api))
    g.plies = 3
    assert settle(g, 10, budget_hit=False, on_discard=on_discard) is False
    assert g.done is False

    # At max_plies -> finalize (truncated).
    g.plies = 10
    assert settle(g, 10, budget_hit=False, on_discard=on_discard) is True
    assert g.done is True and g.status == "truncated"

    # Terminal position settles regardless of ply count.
    st = _FakeState(game_len=1_000)
    st.winner_seat = 1
    g2 = EvalGame(index=1, pair_index=-1, a_is_p0=False, seed=0, state=st)
    assert settle(g2, 10_000, budget_hit=False, on_discard=on_discard) is True
    assert g2.status == "completed" and g2.winner == "A"  # A holds seat 1, seat 1 won


# =========================================================================== #
# record_move / follower_opening_action.
# =========================================================================== #
def test_record_move_bookkeeping(fake_api):
    settled: list[int] = []
    settle_fn = lambda g: settled.append(g.plies)

    from hexfield.geometry import pack_action_id

    g = EvalGame(index=0, pair_index=-1, a_is_p0=True, seed=0, state=_new_state(fake_api))
    a0 = pack_action_id(1, -1)
    a1 = pack_action_id(0, 2)
    a2 = pack_action_id(-2, 1)

    # opening_plies=2: only the first two actions extend the opening line.
    record_move(g, a0, is_a=True, opening_plies=2, settle_fn=settle_fn)
    record_move(g, a1, is_a=False, opening_plies=2, settle_fn=settle_fn)
    record_move(g, a2, is_a=True, opening_plies=2, settle_fn=settle_fn)

    assert g.plies == 3
    assert g.a_decisions == 2  # only the is_a=True moves counted
    assert g.actions == [a0, a1, a2]  # full ordered stream, both players
    assert g.opening == [a0, a1]  # capped at opening_plies
    assert settled == [1, 2, 3]  # settle_fn fired after each move, post-increment

    # The move actually advanced the fake engine.
    assert g.state.ply == 3
    assert g.state.actions[0] == a0


def test_follower_opening_action_replay_and_fallback(fake_api):
    leader = EvalGame(index=0, pair_index=0, a_is_p0=True, seed=0, state=_new_state(fake_api))
    leader.opening = [111, 222, 333]
    follower = EvalGame(index=1, pair_index=0, a_is_p0=False, seed=0, state=_new_state(fake_api))
    follower.is_leader = False
    follower.leader = leader

    # Follower reads the leader's recorded action for its current ply.
    follower.plies = 0
    assert follower_opening_action(follower) == 111
    follower.plies = 2
    assert follower_opening_action(follower) == 333
    # Past the leader's recorded line -> None (caller falls back to a search).
    follower.plies = 3
    assert follower_opening_action(follower) is None


# =========================================================================== #
# run_hexfield_ply: multi-root chunking + positional zip-scatter.
# =========================================================================== #
def _record_fn(opening_plies=0):
    def record(g, action_id, *, is_a):
        record_move(g, action_id, is_a=is_a, opening_plies=opening_plies, settle_fn=lambda gg: None)

    return record


def test_run_hexfield_ply_scatter_order_and_chunking(fake_api):
    _FakeSession.calls = []
    session = _FakeSession()
    evaluator = _FakeEvaluator("A", strength=1)
    sp = parse_hexfield_config({}).selfplay
    search_kwargs = build_eval_search_kwargs(sp, visits=64, virtual_batch_size=8, active_root_limit=8)

    # Five games at DISTINCT plies so each expects a distinct greedy move; if the
    # driver mis-scattered (or re-sorted) results, the applied actions would not
    # line up with each game's own position.
    games = []
    for i in range(5):
        st = _new_state(fake_api)
        st.ply = i * 2  # 0,2,4,6,8 -> distinct mod 7 -> distinct moves
        games.append(EvalGame(index=100 + i, pair_index=-1, a_is_p0=True, seed=0, state=st))

    seed = 777
    expected = [_FakeSession._move_for(seed, 0.0, g.state.ply) for g in games]

    applied = run_hexfield_ply(
        session,
        evaluator,
        games,
        search_kwargs=search_kwargs,
        root_limit=2,
        overrides=None,
        seed=seed,
        move_temperature_fn=lambda g: 0.0,
        record_fn=_record_fn(),
        is_a=True,
    )

    assert applied == 5
    # Each game got the move for ITS pre-move ply (positional scatter, not sorted).
    for g, exp in zip(games, expected):
        assert g.actions[-1] == exp

    # root_limit=2 over 5 games -> chunk sizes 2,2,1, keys in positional order.
    assert [c["n_roots"] for c in _FakeSession.calls] == [2, 2, 1]
    assert [k for c in _FakeSession.calls for k in c["game_keys"]] == [100 + i for i in range(5)]
    # search_kwargs threaded through to the session.
    assert all(c["visits"] == 64 for c in _FakeSession.calls)
    assert all(c["virtual_batch_size"] == 8 for c in _FakeSession.calls)


def test_run_hexfield_ply_key_fn_override(fake_api):
    _FakeSession.calls = []
    session = _FakeSession()
    evaluator = _FakeEvaluator("A", strength=1)
    sp = parse_hexfield_config({}).selfplay
    search_kwargs = build_eval_search_kwargs(sp, visits=32, virtual_batch_size=4, active_root_limit=8)

    games = []
    for i in range(3):
        g = EvalGame(index=i, pair_index=-1, a_is_p0=True, seed=0, state=_new_state(fake_api))
        g.cand_key = 1_000_000 + i  # candidate-session namespaced key
        games.append(g)

    run_hexfield_ply(
        session,
        evaluator,
        games,
        search_kwargs=search_kwargs,
        root_limit=8,
        overrides=None,
        seed=1,
        move_temperature_fn=lambda g: 0.0,
        record_fn=_record_fn(),
        is_a=True,
        key_fn=lambda g: g.cand_key,
    )
    assert _FakeSession.calls[0]["game_keys"] == [1_000_000, 1_000_001, 1_000_002]


def test_run_hexfield_ply_length_mismatch_raises(fake_api):
    class _BadSession(_FakeSession):
        def search(self, game_keys, states, **kw):
            return []  # wrong length on purpose

    sp = parse_hexfield_config({}).selfplay
    search_kwargs = build_eval_search_kwargs(sp, visits=16, virtual_batch_size=4, active_root_limit=8)
    games = [EvalGame(index=0, pair_index=-1, a_is_p0=True, seed=0, state=_new_state(fake_api))]

    with pytest.raises(RuntimeError, match="returned 0 results for 1 games"):
        run_hexfield_ply(
            _BadSession(),
            _FakeEvaluator("A", 1),
            games,
            search_kwargs=search_kwargs,
            root_limit=8,
            overrides=None,
            seed=0,
            move_temperature_fn=lambda g: 0.0,
            record_fn=_record_fn(),
            is_a=True,
        )


# =========================================================================== #
# OpponentAdapter protocol surface.
# =========================================================================== #
def test_opponent_adapter_is_runtime_checkable():
    class _Adapter:
        label = "stub"

        def start(self, games):
            return None

        def advance(self, batch, *, round_index, seed_base, record_move, opening_plies, opening_temperature):
            return 0

        def close(self):
            return None

    assert isinstance(_Adapter(), OpponentAdapter)
    # Missing the protocol methods -> not an adapter.
    assert not isinstance(object(), OpponentAdapter)


def test_unpack_roundtrip_smoke():
    # Guards the geometry dependency the driver relies on for record_move.
    from hexfield.geometry import pack_action_id

    q, r = unpack_action_id(pack_action_id(2, -3))
    assert (q, r) == (2, -3)
