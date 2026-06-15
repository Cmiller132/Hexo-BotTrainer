"""Pure-CPU EQUIVALENCE tests for the CONCURRENT MULTI-OPPONENT checkpoint runner
(``eval_arena.play_multi_checkpoint_match``), with NO torch/CUDA and NO native .so.

The multi-opponent runner plays the candidate (always net A) vs MANY checkpoint
opponents in ONE batched concurrent pass — sharing ONE candidate forward across
every opponent's candidate-to-move games per round (the speed win) while each
opponent searches in its OWN session. The WHOLE safety net for this rewrite is a
single property:

    play_multi_checkpoint_match over K opponents produces, FOR EACH opponent, the
    SAME per-opponent result (score a_wins/b_wins/decided, pentanomial histogram +
    per-pair rows, net-A-centric per-game winners, seats, game lengths) as calling
    the existing serial ``play_checkpoint_match`` once per opponent with the same
    seeds/config.

These tests pin exactly that, reusing the fake engine + _FakeSession +
build_evaluators/make_session seams from test_hexfield_eval_arena_concurrent.py.
The fake session's chosen move is a pure function of (search RNG seed, position) —
NOT of the evaluator — so the candidate's moves are identical regardless of which
opponent group it is grouped with; only the per-game WINNER depends on net
strength (decided by the fake engine per opponent). At temperature 0 the move is
seed-independent (greedy argmax), so the cross-opponent greedy merge is exact; the
opening leaders are searched per-opponent with that opponent's own open_seed so the
``seed+root_index`` per-root sampling stream matches the serial run bit-for-bit.

Everything is faked; the native MCTS extension and the real engine .so are never
imported, so this collects + runs on a CPU-only interpreter and never touches the
live training run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "hexo_engine" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "hexfield" / "python"))

from hexfield import eval_arena  # noqa: E402
from hexfield.config import parse_hexfield_config  # noqa: E402
from hexfield.geometry import pack_action_id  # noqa: E402
from hexo_engine.types import Player  # noqa: E402

# Reuse the single-opponent concurrent suite's fakes verbatim (the agreed seams).
import test_hexfield_eval_arena_concurrent as single  # noqa: E402

_FakeState = single._FakeState
_FakeEvaluator = single._FakeEvaluator
_FakeSession = single._FakeSession
_Terminal = single._Terminal


# --------------------------------------------------------------------------- #
# A tracking fake engine that decides each game's winner net-relatively, keyed
# by the OPPONENT each freshly-created game belongs to. Both the serial driver
# (one play_checkpoint_match per opponent) and the concurrent driver (one
# play_multi_checkpoint_match) create paired games in (a_is_p0=True, a_is_p0=False)
# order; we replay that ordering so each state's (opponent, net-A seat) is known
# and the winner is the stronger net's engine seat — exactly the engine-relative
# outcome the real engine produces.
# --------------------------------------------------------------------------- #
def _seat_iter(n_games: int) -> list[bool]:
    """The arena's per-game net-A seat in creation order (paired: True, False,...)."""
    out: list[bool] = []
    n_pairs = (n_games + 1) // 2
    for p in range(n_pairs):
        out.append(True)
        if 2 * p + 1 < n_games:
            out.append(False)
    return out


def _winner_seat_for(a_strength: int, b_strength: int, a_is_p0: bool) -> int:
    a_seat = 0 if a_is_p0 else 1
    b_seat = 1 - a_seat
    if a_strength == b_strength:
        return 0  # deterministic, keeps it decided
    return a_seat if a_strength > b_strength else b_seat


class _TrackingApi:
    """Stand-in for hexo_engine.api. Each new game is tagged with the (opponent,
    a_is_p0) the caller is about to assign, via a creation-order plan."""

    Player = Player

    def __init__(self, *, game_len: int, plan: list[tuple[str, bool]],
                 strength_by_opp: dict[str, tuple[int, int]]) -> None:
        self._game_len = game_len
        self._plan = plan
        self._strength = strength_by_opp
        self._created = 0
        self._tag: dict[int, tuple[str, bool]] = {}

    def new_game(self, *, seed=None, scenario=None):
        st = _FakeState(game_len=self._game_len)
        self._tag[id(st)] = self._plan[self._created]
        self._created += 1
        return st

    def current_player(self, state):
        return Player.PLAYER_0 if state.mover_seat() == 0 else Player.PLAYER_1

    def apply_action(self, state, action):
        coord = action.coord
        state.actions.append(pack_action_id(coord.q, coord.r))
        state.ply += 1
        if state.ply >= state.game_len:
            opp, a_is_p0 = self._tag[id(state)]
            a_str, b_str = self._strength[opp]
            state.winner_seat = _winner_seat_for(a_str, b_str, a_is_p0)

    def terminal(self, state):
        if state.winner_seat is None:
            return None
        return _Terminal("player0" if state.winner_seat == 0 else "player1")


# --------------------------------------------------------------------------- #
# Drivers.
# --------------------------------------------------------------------------- #
def _strength_by_opp(opponents, candidate_strength):
    return {label: (candidate_strength, opp_strength) for label, opp_strength in opponents}


def _run_serial(monkeypatch, *, opponents, n_games, candidate_strength,
                game_len, cfg, **kw):
    """Reference: call play_checkpoint_match ONCE per opponent, each on a fresh
    fake engine whose creation plan is just that opponent's games."""
    out: dict[str, dict] = {}
    strength = _strength_by_opp(opponents, candidate_strength)
    for label, opp_strength in opponents:
        _FakeSession.calls = []
        plan = [(label, a_is_p0) for a_is_p0 in _seat_iter(n_games)]
        api = _TrackingApi(game_len=game_len, plan=plan, strength_by_opp=strength)
        monkeypatch.setattr(eval_arena, "api", api)

        def _build_evaluators(_label=label, _opp=opp_strength):
            return _FakeEvaluator("A", candidate_strength), _FakeEvaluator(_label, _opp)

        sessions: list[_FakeSession] = []

        def factory():
            s = _FakeSession()
            sessions.append(s)
            return s

        out[label] = eval_arena.play_checkpoint_match(
            "cand.pt", f"{label}.pt", n_games,
            config=cfg, label_a="cand", label_b=label,
            paired_openings=True, make_session=factory,
            build_evaluators=_build_evaluators,
            **kw,
        )
    return out


def _run_concurrent(monkeypatch, *, opponents, n_games, candidate_strength,
                    game_len, cfg, **kw):
    """The runner under test: ONE play_multi_checkpoint_match over all opponents.

    The concurrent runner creates ALL opponent groups up front (opponent 0's
    pairs, then opponent 1's pairs, ...), so the creation plan is the concatenation
    of each opponent's seat sequence in roster order."""
    _FakeSession.calls = []
    strength = _strength_by_opp(opponents, candidate_strength)
    plan: list[tuple[str, bool]] = []
    for label, _ in opponents:
        plan.extend((label, a_is_p0) for a_is_p0 in _seat_iter(n_games))
    api = _TrackingApi(game_len=game_len, plan=plan, strength_by_opp=strength)
    monkeypatch.setattr(eval_arena, "api", api)

    sessions: list[_FakeSession] = []

    def factory():
        s = _FakeSession()
        sessions.append(s)
        return s

    def build_candidate():
        return _FakeEvaluator("A", candidate_strength)

    opp_strength = dict(opponents)

    def build_opponent(label, ckpt):
        return _FakeEvaluator(label, opp_strength[label])

    result = eval_arena.play_multi_checkpoint_match(
        "cand.pt",
        [(label, f"{label}.pt") for label, _ in opponents],
        n_games,
        config=cfg, candidate_label="cand",
        make_session=factory,
        build_candidate_evaluator=build_candidate,
        build_opponent_evaluator=build_opponent,
        **kw,
    )
    return result, sessions


# --------------------------------------------------------------------------- #
# Comparators.
# --------------------------------------------------------------------------- #
def _score_tuple(match: dict) -> tuple:
    s = match["score"]
    return (s["completed"], s["truncated"], s["aborted_budget"],
            s["a_wins"], s["b_wins"], s["decided"])


def _penta_tuple(match: dict):
    p = match.get("pentanomial")
    if p is None:
        return None
    return (
        p["n_pairs"], p["n_full_pairs"], p["n_informative_pairs"],
        tuple(sorted(p["histogram_a_wins"].items())),
        tuple(
            (q["pair_index"], q["n_games"], q["n_decided"], q["a_wins"],
             q["b_wins"], q["pentanomial_a_score"], tuple(q["game_indices"]))
            for q in p["pairs"]
        ),
    )


def _winners(match: dict) -> list:
    return [(g["index"], g["a_seat"], g["status"], g["winner"], g["plies"])
            for g in match["games"]]


def _assert_match_equivalent(label, serial_match, conc_match):
    assert _score_tuple(serial_match) == _score_tuple(conc_match), (
        f"[{label}] score differs:\n serial={serial_match['score']}\n conc  ={conc_match['score']}"
    )
    assert _penta_tuple(serial_match) == _penta_tuple(conc_match), (
        f"[{label}] pentanomial differs:\n serial={serial_match['pentanomial']}\n conc  ={conc_match['pentanomial']}"
    )
    assert _winners(serial_match) == _winners(conc_match), (
        f"[{label}] per-game winners/seats/status differ:\n"
        f" serial={_winners(serial_match)}\n conc  ={_winners(conc_match)}"
    )
    assert serial_match["game_lengths"] == conc_match["game_lengths"], f"[{label}] lengths differ"


# =========================================================================== #
# 1. The core equivalence: concurrent == serial per opponent.
# =========================================================================== #
def test_multi_equals_serial_per_opponent(monkeypatch):
    """K=3 opponents at distinct strengths (candidate beats one, loses to one,
    ties one): each opponent's concurrent result == its serial play_checkpoint_match
    result on the same seeds/config."""

    cfg = parse_hexfield_config({})
    opponents = [("opp_weak", 1), ("opp_strong", 5), ("opp_even", 3)]
    kwargs = dict(opponents=opponents, n_games=8, candidate_strength=3,
                  game_len=6, cfg=cfg, opening_plies=2, game_seed_base=100)

    serial = _run_serial(monkeypatch, **kwargs)
    conc, sessions = _run_concurrent(monkeypatch, **kwargs)

    assert set(conc) == set(serial) == {l for l, _ in opponents}
    for label, _ in opponents:
        _assert_match_equivalent(label, serial[label], conc[label])


def test_multi_equals_serial_longer_opening(monkeypatch):
    """A longer opening (so several rounds are temperature-sampled) still matches:
    the per-opponent open_seed + per-root index reproduces the serial line."""

    cfg = parse_hexfield_config({})
    opponents = [("bc_prefit", 1), ("ep5", 2), ("ep10", 4), ("champ", 3)]
    kwargs = dict(opponents=opponents, n_games=6, candidate_strength=3,
                  game_len=10, cfg=cfg, opening_plies=4, game_seed_base=42)

    serial = _run_serial(monkeypatch, **kwargs)
    conc, _ = _run_concurrent(monkeypatch, **kwargs)
    for label, _ in opponents:
        _assert_match_equivalent(label, serial[label], conc[label])


def test_multi_equals_serial_single_opponent(monkeypatch):
    """Degenerate K=1: the multi-runner reduces to play_checkpoint_match exactly."""

    cfg = parse_hexfield_config({})
    opponents = [("solo", 5)]
    kwargs = dict(opponents=opponents, n_games=8, candidate_strength=2,
                  game_len=6, cfg=cfg, opening_plies=3, game_seed_base=7)
    serial = _run_serial(monkeypatch, **kwargs)
    conc, _ = _run_concurrent(monkeypatch, **kwargs)
    _assert_match_equivalent("solo", serial["solo"], conc["solo"])


def test_multi_equals_serial_odd_game_count(monkeypatch):
    """Odd n_games -> a singleton final pair per opponent; still equivalent."""

    cfg = parse_hexfield_config({})
    opponents = [("a", 1), ("b", 4)]
    kwargs = dict(opponents=opponents, n_games=5, candidate_strength=2,
                  game_len=6, cfg=cfg, opening_plies=2, game_seed_base=3)
    serial = _run_serial(monkeypatch, **kwargs)
    conc, _ = _run_concurrent(monkeypatch, **kwargs)
    for label, _ in opponents:
        _assert_match_equivalent(label, serial[label], conc[label])


def test_multi_equals_serial_with_truncation(monkeypatch):
    """Games that never decide before max_game_plies truncate identically (no
    draws in hexo): the truncated/undecided bookkeeping matches the serial run."""

    cfg = parse_hexfield_config({"selfplay": {"max_game_plies": 3}})
    opponents = [("x", 2), ("y", 5)]
    kwargs = dict(opponents=opponents, n_games=6, candidate_strength=3,
                  game_len=10, cfg=cfg, opening_plies=2, game_seed_base=11)
    serial = _run_serial(monkeypatch, **kwargs)
    conc, _ = _run_concurrent(monkeypatch, **kwargs)
    for label, _ in opponents:
        _assert_match_equivalent(label, serial[label], conc[label])
        assert conc[label]["score"]["truncated"] == 6


# =========================================================================== #
# 2. Concurrency mechanics: the candidate forward is SHARED across opponents.
# =========================================================================== #
def test_candidate_greedy_forward_is_shared_across_opponents(monkeypatch):
    """The candidate's GREEDY plies across ALL opponents are merged into ONE
    multi-root call whose roots span more than one opponent's games (the
    cross-opponent batch is the speed win), and the candidate session is a SINGLE
    persistent session, NOT one-per-opponent."""

    cfg = parse_hexfield_config({})
    opponents = [("o1", 2), ("o2", 4), ("o3", 1)]
    conc, sessions = _run_concurrent(
        monkeypatch, opponents=opponents, n_games=6, candidate_strength=3,
        game_len=8, cfg=cfg, opening_plies=2, game_seed_base=0,
    )

    # KEY_STRIDE namespaces candidate keys per opponent (opp_index*1_000_000+local).
    # A greedy candidate batch is one whose roots include keys from >1 opponent
    # namespace, at temperature 0.
    KEY_STRIDE = 1_000_000
    cross_opponent_greedy = []
    for c in _FakeSession.calls:
        if all(t == 0.0 for t in c["move_temperatures"]) and c["n_roots"] > 1:
            namespaces = {k // KEY_STRIDE for k in c["game_keys"]}
            if len(namespaces) > 1:
                cross_opponent_greedy.append(c)
    assert cross_opponent_greedy, (
        "expected at least one greedy candidate batch spanning >1 opponent "
        "(the shared candidate forward)"
    )

    # Sessions: 1 candidate + 3 opponents = 4 (NOT one-per-game).
    assert len(sessions) == 1 + len(opponents)


def test_candidate_trees_discarded_no_leak(monkeypatch):
    """Every game's candidate tree is discarded on the candidate session (global
    keys) and every opponent tree on that opponent's session (local keys), so no
    tree leaks across opponent groups."""

    cfg = parse_hexfield_config({})
    opponents = [("o1", 2), ("o2", 4)]
    n_games = 6
    conc, sessions = _run_concurrent(
        monkeypatch, opponents=opponents, n_games=n_games, candidate_strength=3,
        game_len=6, cfg=cfg, opening_plies=2, game_seed_base=0,
    )
    KEY_STRIDE = 1_000_000
    # The candidate session is the one that saw keys from multiple namespaces.
    cand_session = None
    for s in sessions:
        ns = {k // KEY_STRIDE for k in s.discarded}
        if len(ns) > 1:
            cand_session = s
            break
    assert cand_session is not None, "could not identify the shared candidate session"
    # Candidate discarded exactly the global keys for every game of every opponent.
    expected_cand = sorted(
        opp_index * KEY_STRIDE + local
        for opp_index in range(len(opponents))
        for local in range(n_games)
    )
    assert sorted(cand_session.discarded) == expected_cand

    # Each opponent session discarded exactly its own local game indices 0..n-1.
    opp_sessions = [s for s in sessions if s is not cand_session]
    for s in opp_sessions:
        assert sorted(s.discarded) == list(range(n_games))


def test_result_dicts_are_play_checkpoint_match_shape(monkeypatch):
    """Each per-opponent result is the exact drop-in shape the orchestrator's
    downstream (_checkpoint_edge_counts) consumes, AND it consumes it."""

    from hexfield import multistage_eval as mse

    cfg = parse_hexfield_config({})
    opponents = [("o1", 2), ("o2", 4)]
    conc, _ = _run_concurrent(
        monkeypatch, opponents=opponents, n_games=8, candidate_strength=3,
        game_len=6, cfg=cfg, opening_plies=2, game_seed_base=0,
    )
    for label, _ in opponents:
        match = conc[label]
        assert set(match) >= {"meta", "score", "game_lengths", "opening_dedup",
                              "games", "pentanomial"}
        assert match["meta"]["label_a"] == "cand"
        assert match["meta"]["label_b"] == label
        assert match["meta"]["multi_opponent"] is True
        assert match["meta"]["games_requested"] == 8
        # Downstream effective-count extraction works unchanged.
        wa, wb, n_eff, prov = mse._checkpoint_edge_counts(match)
        assert n_eff > 0.0
        paired = mse._pentanomial_to_paired_result(match["pentanomial"])
        assert paired is not None and paired.n_pairs == 4
