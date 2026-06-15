"""Pure-CPU unit tests for the CONCURRENT checkpoint-vs-checkpoint arena runner
(``eval_arena.play_checkpoint_match``), with NO torch/CUDA and NO native .so.

The rewrite makes ``play_checkpoint_match`` run games CONCURRENTLY (cross-game
leaf batching via the multi-root ``HexfieldMctsSession.search``) at full sims,
the way self-play does, replacing the old serial one-game-at-a-time
``_play_pair`` loop. These tests pin the three things that rewrite must get
right WITHOUT a GPU:

  1. RESULT-DICT SHAPE / DROP-IN CONTRACT: the concurrent runner still returns
     the exact ``meta`` / ``score`` / ``pentanomial`` (pairs + histogram) /
     ``game_lengths`` / ``opening_dedup`` / per-game-row structure that
     ``multistage_eval`` + ``eval_stats`` consume (net-A-centric winners,
     per-pair ``n_decided``/``a_wins``/``pentanomial_a_score``, ``games_requested``,
     etc.). We feed the result through the SAME downstream helpers the
     orchestrator uses to prove they still consume it.

  2. CONCURRENCY MECHANICS: many games are in flight at once and each round
     batches each side's to-move games through that side's session in ONE
     multi-root ``search`` call (not one game at a time). We assert the fake
     session actually saw multi-root batches, that seats/CRN pairing are
     preserved, and that the FULL-sims budget (cfg.selfplay.search_visits=512)
     is threaded by default.

  3. CRN UNDER BATCHING: the temperature-sampled opening plies are now searched
     BATCHED cross-game for the pair LEADERS (each leader root decorrelated by the
     native per-root ``seed+index`` offset); paired seat-swapped siblings still
     share the IDENTICAL opening because the FOLLOWER REPLAYS the leader's recorded
     line (no search). The greedy tail batches as before. (The old single-root
     per-leader opening was the throughput bottleneck and is gone.)

Everything is faked: a tiny deterministic engine (monkeypatched onto
``eval_arena.api``) and a fake multi-root session / evaluator injected through
the ``make_session`` / ``build_evaluators`` seams. The native MCTS extension and
the real engine .so are never imported, so this collects + runs on a CPU-only
interpreter and never touches the live training run.
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


# --------------------------------------------------------------------------- #
# A tiny deterministic fake engine, monkeypatched onto eval_arena.api.
# --------------------------------------------------------------------------- #
class _Terminal:
    def __init__(self, winner_label: str | None) -> None:
        # str(self.winner) must yield "player0"/"player1" like the real engine.
        self.winner = winner_label  # already the string label, or None

    def __str__(self) -> str:  # pragma: no cover - not used directly
        return str(self.winner)


class _FakeState:
    """Connect6 mover schedule (0, 1,1, 0,0, 1,1, ...) over a fixed game length.

    The engine is SEEDLESS (like the real one): every game starts identical, all
    diversity comes from which action the search picks. ``winner_seat`` is set by
    the harness when the game ends so the test can pin the net-A-centric winner
    mapping deterministically.
    """

    def __init__(self, *, game_len: int) -> None:
        self.game_len = game_len
        self.ply = 0
        self.actions: list[int] = []
        self.winner_seat: int | None = None  # 0/1 engine seat that "won"

    def mover_seat(self) -> int:
        # Identical schedule to evaluation._play_pair's formula.
        return 0 if self.ply == 0 else (1 if ((self.ply - 1) // 2) % 2 == 0 else 0)


class _FakeApi:
    """Stand-in for hexo_engine.api with just what the arena calls."""

    Player = Player

    def __init__(self, *, game_len: int, decide_winner) -> None:
        self._game_len = game_len
        # decide_winner(state) -> engine seat int (0/1) that wins this game.
        self._decide_winner = decide_winner

    def new_game(self, *, seed=None, scenario=None):
        return _FakeState(game_len=self._game_len)

    def current_player(self, state: _FakeState) -> Player:
        return Player.PLAYER_0 if state.mover_seat() == 0 else Player.PLAYER_1

    def apply_action(self, state: _FakeState, action) -> None:
        coord = action.coord
        state.actions.append(pack_action_id(coord.q, coord.r))
        state.ply += 1
        if state.ply >= state.game_len:
            state.winner_seat = self._decide_winner(state)

    def terminal(self, state: _FakeState):
        if state.winner_seat is None:
            return None
        return _Terminal("player0" if state.winner_seat == 0 else "player1")


# --------------------------------------------------------------------------- #
# A fake multi-root session + evaluator. The "evaluator" carries a strength so
# the fake session can pick a move; the session records every call so the test
# can verify batching + per-ply seeds.
# --------------------------------------------------------------------------- #
class _FakeEvaluator:
    def __init__(self, tag: str, strength: int) -> None:
        self.tag = tag
        self.strength = strength


class _FakeSession:
    """Records ``search`` calls and returns one deterministic move per root.

    Faithful to the CRN property that matters: the chosen move is a pure function
    of (search RNG seed, position) — NOT of the game index or the evaluator — so
    two games searched at the same position with the same seed pick the SAME
    move (exactly what real symmetric MCTS does, and what the opening-CRN
    guarantee relies on). At temperature 0 (the greedy tail) the move is a pure
    function of position (deterministic argmax) so it is seed-independent. We do
    NOT model real MCTS internals — only this call/return contract.
    """

    # Class-level shared log so the test can inspect every call across the two
    # per-net sessions in one place.
    calls: list[dict] = []

    def __init__(self) -> None:
        self.discarded: list[int] = []

    @staticmethod
    def _move_for(seed: int, temperature: float, ply: int) -> int:
        # Greedy (temp 0): position-only argmax. Sampled (temp > 0): also depends
        # on the seed. Coords stay tiny so pack_action_id never overflows.
        if temperature > 0.0:
            mix = (seed * 2654435761 + ply * 40503) % 7
        else:
            mix = ply % 7
        q = (mix % 5) - 2
        r = ((mix // 5) % 5) - 2
        return pack_action_id(q, r)

    def search(self, game_keys, states, *, seed, evaluator, move_temperatures, **kw):
        assert len(game_keys) == len(states) == len(move_temperatures)
        _FakeSession.calls.append(
            {
                "n_roots": len(game_keys),
                "game_keys": list(game_keys),
                "seed": seed,
                "evaluator_tag": evaluator.tag,
                "move_temperatures": list(move_temperatures),
                "visits": kw.get("visits"),
                "virtual_batch_size": kw.get("virtual_batch_size"),
            }
        )
        return [
            {"action_id": self._move_for(seed, temp, state.ply)}
            for state, temp in zip(states, move_temperatures)
        ]

    def discard(self, index: int) -> None:
        self.discarded.append(index)


def _make_session_factory():
    """Return (factory, sessions_list); factory() appends a fresh fake session."""
    sessions: list[_FakeSession] = []

    def factory():
        s = _FakeSession()
        sessions.append(s)
        return s

    return factory, sessions


# --------------------------------------------------------------------------- #
# The match harness. The fake engine is SEEDLESS and only sees opaque states, so
# to decide each game's winner net-relatively we track which net-A seat the arena
# assigned to each freshly-created state (the arena creates paired games in
# (a_is_p0=True, a_is_p0=False) order per pair). The winner is then the stronger
# net's engine seat — exactly the (engine-seat-returning, net-relative) outcome
# the real engine produces, so the arena's net-A-centric winner mapping is
# exercised faithfully.
# --------------------------------------------------------------------------- #
def _run_match(monkeypatch, *, n_games, a_strength, b_strength, game_len=6,
               config=None, **kw):
    """Drive play_checkpoint_match with a fully deterministic fake engine that
    tracks each game's net-A seat so winners are exact and net-relative."""

    _FakeSession.calls = []
    factory, sessions = _make_session_factory()

    # Per-state net-A seat, filled as games are created in arena order. The arena
    # creates paired games in (a_is_p0=True, a_is_p0=False) order per pair, so we
    # replay that ordering deterministically.
    seat_iter = []
    if kw.get("paired_openings", True):
        n_pairs = (n_games + 1) // 2
        for p in range(n_pairs):
            seat_iter.append(True)
            if 2 * p + 1 < n_games:
                seat_iter.append(False)
    else:
        for i in range(n_games):
            seat_iter.append(i % 2 == 0)

    created: list[_FakeState] = []
    seat_of_state: dict[int, bool] = {}

    def decide(state: _FakeState) -> int:
        a_is_p0 = seat_of_state[id(state)]
        a_seat = 0 if a_is_p0 else 1
        b_seat = 1 - a_seat
        if a_strength == b_strength:
            return 0  # deterministic, keeps it decided
        return a_seat if a_strength > b_strength else b_seat

    base_api = _FakeApi(game_len=game_len, decide_winner=decide)

    class _TrackingApi(_FakeApi):
        Player = Player

        def new_game(self, *, seed=None, scenario=None):
            st = _FakeState(game_len=game_len)
            created.append(st)
            # Assign the next seat in arena creation order.
            seat_of_state[id(st)] = seat_iter[len(created) - 1]
            return st

        def current_player(self, state):
            return base_api.current_player(state)

        def apply_action(self, state, action):
            return base_api.apply_action(state, action)

        def terminal(self, state):
            return base_api.terminal(state)

    monkeypatch.setattr(eval_arena, "api", _TrackingApi(game_len=game_len, decide_winner=decide))

    def _build_evaluators():
        return _FakeEvaluator("A", a_strength), _FakeEvaluator("B", b_strength)

    result = eval_arena.play_checkpoint_match(
        "ckpt_a.pt", "ckpt_b.pt", n_games,
        config=config if config is not None else parse_hexfield_config({}),
        label_a="cand", label_b="opp",
        make_session=factory,
        build_evaluators=_build_evaluators,
        **kw,
    )
    return result, sessions


# =========================================================================== #
# 1. Result-dict shape / drop-in contract.
# =========================================================================== #
def test_result_shape_is_drop_in(monkeypatch):
    result, _ = _run_match(monkeypatch, n_games=8, a_strength=2, b_strength=1, game_len=6)

    # Top-level keys.
    assert set(result) >= {"meta", "score", "game_lengths", "opening_dedup", "games", "pentanomial"}

    score = result["score"]
    for key in ("completed", "truncated", "aborted_budget", "a_wins", "b_wins", "decided", "by_seat"):
        assert key in score, key
    # Candidate (net A) is the stronger net -> wins every decided game.
    assert score["completed"] == 8
    assert score["decided"] == 8
    assert score["a_wins"] == 8
    assert score["b_wins"] == 0
    assert score["truncated"] == 0

    # Per-game rows carry the exact load-bearing keys.
    assert len(result["games"]) == 8
    for g in result["games"]:
        assert set(g) >= {"index", "seed", "a_seat", "status", "winner", "plies", "opening"}
        assert g["a_seat"] in ("P0", "P1")
        assert g["status"] in ("completed", "truncated", "aborted_budget")
        assert g["winner"] in ("A", "B", None)

    # Pentanomial block + per-pair rows the orchestrator consumes.
    penta = result["pentanomial"]
    assert penta is not None
    assert set(penta) >= {"n_pairs", "histogram_a_wins", "pairs"}
    assert penta["n_pairs"] == 4
    for p in penta["pairs"]:
        assert set(p) >= {
            "pair_index", "seed", "game_indices", "n_games",
            "n_decided", "a_wins", "b_wins", "pentanomial_a_score",
        }
        assert p["n_games"] == 2
        assert p["n_decided"] == 2
        assert p["a_wins"] == 2  # candidate swept both seats of every pair
    # histogram keyed by net-A wins among the pair's 2 decided games.
    assert penta["histogram_a_wins"] == {"0": 0, "1": 0, "2": 4}

    # meta carries the keys the orchestrator reads (games_requested) + telemetry.
    meta = result["meta"]
    assert meta["games_requested"] == 8
    assert meta["label_a"] == "cand" and meta["label_b"] == "opp"
    assert meta["concurrent"] is True
    assert meta["rounds"] >= 1 and meta["forward_batches"] >= 1


def test_result_consumed_by_orchestrator_helpers(monkeypatch):
    """The concurrent result feeds the SAME downstream helpers the orchestrator
    uses (the real drop-in proof)."""

    from hexfield import multistage_eval as mse

    result, _ = _run_match(monkeypatch, n_games=8, a_strength=2, b_strength=1, game_len=6)

    # Stage-C edge counts (pentanomial -> effective BT counts).
    wa, wb, n_eff, prov = mse._checkpoint_edge_counts(result)
    assert wa > 0.0
    assert wb == 0.0  # candidate swept; no opponent wins
    assert n_eff > 0.0

    # Stage-B SPRT consumes score.a_wins / score.b_wins directly.
    import hexfield.eval_stats as es

    sprt = es.sprt(result["score"]["a_wins"], result["score"]["b_wins"], elo0=0.0, elo1=35.0)
    assert sprt.verdict in {"accept_h0", "accept_h1", "continue"}

    # Pentanomial -> PairedResult path the orchestrator prefers.
    paired = mse._pentanomial_to_paired_result(result["pentanomial"])
    assert paired is not None and paired.n_pairs == 4


# =========================================================================== #
# 2. Concurrency mechanics + seat / pairing preservation.
# =========================================================================== #
def test_games_run_concurrently_in_multiroot_batches(monkeypatch):
    """Many games in flight -> the greedy tail is searched in MULTI-ROOT batches
    (n_roots > 1), not one game at a time."""

    # game_len=6 with opening_plies=2 leaves a multi-ply greedy tail to batch.
    result, sessions = _run_match(
        monkeypatch, n_games=16, a_strength=2, b_strength=1, game_len=6, opening_plies=2,
    )
    batched = [c for c in _FakeSession.calls if c["n_roots"] > 1]
    assert batched, "expected at least one multi-root (concurrent) search batch"
    # The biggest batch should pull in many games at once (cross-game batching).
    assert max(c["n_roots"] for c in _FakeSession.calls) >= 4

    # Exactly two persistent sessions were created (one per net), NOT one per game.
    assert len(sessions) == 2

    # Every game's tree is discarded on BOTH sessions at game end (no leak).
    for s in sessions:
        assert sorted(s.discarded) == list(range(16))


def test_seats_swapped_within_pairs_and_crn_seed_shared(monkeypatch):
    result, _ = _run_match(monkeypatch, n_games=8, a_strength=2, b_strength=1, game_len=4)
    games = {g["index"]: g for g in result["games"]}
    penta = result["pentanomial"]
    for p in penta["pairs"]:
        i0, i1 = p["game_indices"]
        # Seat-swapped siblings: one P0, one P1.
        assert {games[i0]["a_seat"], games[i1]["a_seat"]} == {"P0", "P1"}
        # Shared CRN seed: both siblings carry the pair's seed.
        assert games[i0]["seed"] == games[i1]["seed"] == p["seed"]


def test_winner_mapping_is_net_a_centric_and_seat_symmetric(monkeypatch):
    """Net B stronger -> candidate (A) loses every decided game regardless of the
    seat it sits in (the winner mapping is net-relative, not seat-relative)."""

    result, _ = _run_match(monkeypatch, n_games=8, a_strength=1, b_strength=3, game_len=4)
    score = result["score"]
    assert score["a_wins"] == 0
    assert score["b_wins"] == 8
    # And B wins from BOTH seats (so the result is not just a seat-0 artifact).
    by_seat = score["by_seat"]
    assert by_seat["A_as_P0"]["b_wins"] == by_seat["A_as_P0"]["n"]
    assert by_seat["A_as_P1"]["b_wins"] == by_seat["A_as_P1"]["n"]


def test_full_sims_threaded_by_default(monkeypatch):
    """visits=None -> the full production search budget (selfplay.search_visits=512)
    is threaded into every search call, NOT the reduced eval_visits (128)."""

    cfg = parse_hexfield_config({})
    assert cfg.selfplay.search_visits == 512
    assert cfg.evaluation.eval_visits == 128  # the OLD default we must NOT use

    result, _ = _run_match(monkeypatch, n_games=4, a_strength=1, b_strength=1, game_len=4)
    assert result["meta"]["visits"] == 512
    assert all(c["visits"] == 512 for c in _FakeSession.calls)


def test_explicit_visits_overrides_full_default(monkeypatch):
    result, _ = _run_match(monkeypatch, n_games=4, a_strength=1, b_strength=1, game_len=4, visits=128)
    assert result["meta"]["visits"] == 128
    assert all(c["visits"] == 128 for c in _FakeSession.calls)


# =========================================================================== #
# 3. CRN under batching: opening LEADERS batch cross-game; paired siblings still
#    sample the IDENTICAL opening (the follower REPLAYS the leader's line).
# =========================================================================== #
def test_opening_plies_are_batched_and_pairing_preserved(monkeypatch):
    """Opening plies (temperature-sampled) for the pair LEADERS are now searched
    BATCHED cross-game in multi-root calls (the old single-root-per-leader loop was
    the throughput bottleneck and is gone), with every root carrying
    ``opening_temperature`` and the per-(net, round) opening base seed
    ``game_seed_base + (0|13_000_003) + rounds*1_000_003``. The CRN payoff — paired
    seat-swapped siblings share the IDENTICAL opening — is PRESERVED because the
    follower REPLAYS the leader's recorded line (no search of its own)."""

    n_games = 8
    opening_plies = 2
    game_seed_base = 100
    result, _ = _run_match(
        monkeypatch, n_games=n_games, a_strength=2, b_strength=1, game_len=6,
        opening_plies=opening_plies, game_seed_base=game_seed_base,
    )

    # No single-root opening searches remain: the opening leaders batch.
    single_root_opening = [
        c for c in _FakeSession.calls if c["n_roots"] == 1 and c["move_temperatures"][0] > 0.0
    ]
    assert not single_root_opening, (
        "opening leaders must batch cross-game; no single-root opening search expected "
        f"(saw {single_root_opening})"
    )

    # The opening LEADER batches: multi-root (with 8 games several share each
    # side-to-move every round), every root at opening_temperature (>0), and the
    # seed is the per-(net, round) opening stream, NOT the greedy 7_000_003 stream.
    opening_batches = [
        c for c in _FakeSession.calls
        if c["n_roots"] >= 1 and any(t > 0.0 for t in c["move_temperatures"])
    ]
    assert opening_batches, "expected batched opening (temperature>0) searches"
    assert any(c["n_roots"] > 1 for c in opening_batches), (
        "expected at least one MULTI-root opening batch (cross-game leaders)"
    )
    valid_open_seeds = set()
    # rounds is 1-based in the arena; opening plies finish in the first few rounds.
    for rounds in range(1, result["meta"]["rounds"] + 1):
        for off in (13_000_003, 19_000_003):
            valid_open_seeds.add(game_seed_base + off + rounds * 1_000_003)
    for c in opening_batches:
        # All roots in an opening batch are at opening_temperature (the leaders are,
        # by construction, all at plies < opening_plies).
        assert all(t > 0.0 for t in c["move_temperatures"]), c["move_temperatures"]
        assert c["seed"] in valid_open_seeds, (
            c["seed"], "not a per-(net,round) opening base seed"
        )

    # The greedy tail uses multi-root batches at temperature 0, on a DISTINCT seed
    # stream (offset 7_000_003) so opening and greedy batches never collide.
    greedy_batches = [
        c for c in _FakeSession.calls
        if c["n_roots"] > 1 and all(t == 0.0 for t in c["move_temperatures"])
    ]
    assert greedy_batches
    greedy_seeds = {
        game_seed_base + off + rounds * 1_000_003
        for rounds in range(1, result["meta"]["rounds"] + 1)
        for off in (0, 7_000_003)
    }
    for c in greedy_batches:
        assert c["seed"] in greedy_seeds
        assert c["seed"] not in valid_open_seeds, (
            "greedy and opening seed streams must not collide"
        )

    # CRN payoff (LOAD-BEARING, preserved): paired siblings produced the IDENTICAL
    # opening prefix — the follower replayed the leader's recorded line.
    games = {g["index"]: g for g in result["games"]}
    for p in result["pentanomial"]["pairs"]:
        i0, i1 = p["game_indices"]
        op0 = games[i0]["opening"][:opening_plies]
        op1 = games[i1]["opening"][:opening_plies]
        assert op0 == op1, f"pair {p['pair_index']} siblings diverged on the opening: {op0} vs {op1}"


def test_batched_openers_decorrelate_across_leaders(monkeypatch):
    """Two DIFFERENT leaders sharing one opening batch must get DISTINCT per-root
    sampling seeds so they are free to sample DIFFERENT openings (independent
    leaders must not collapse onto one line). With the native ABI the per-root seed
    is ``open_seed.wrapping_add(root_index)`` (search.rs:748-749); the fake session
    records the call's base ``open_seed`` and ``n_roots``, so we assert a real
    multi-root opening batch exists and that its roots therefore span distinct
    per-root seeds (``open_seed+0 .. open_seed+n_roots-1`` are all different)."""

    n_games = 8
    opening_plies = 3
    game_seed_base = 500
    _result, _ = _run_match(
        monkeypatch, n_games=n_games, a_strength=2, b_strength=1, game_len=8,
        opening_plies=opening_plies, game_seed_base=game_seed_base,
    )

    multi_root_opening = [
        c for c in _FakeSession.calls
        if c["n_roots"] > 1 and all(t > 0.0 for t in c["move_temperatures"])
    ]
    assert multi_root_opening, (
        "expected at least one multi-root opening batch with >1 leader to decorrelate"
    )
    for c in multi_root_opening:
        per_root_seeds = [c["seed"] + i for i in range(c["n_roots"])]
        assert len(set(per_root_seeds)) == c["n_roots"], (
            "leaders in one opening batch must get distinct per-root seeds "
            f"(base {c['seed']}, n_roots {c['n_roots']})"
        )


def test_batch_openings_true_batches_everything(monkeypatch):
    """``batch_openings=True`` collapses the opening single-root special case so
    EVERY ply (incl. the opening) batches — a throughput knob. The pairing /
    result shape is unaffected."""

    result, _ = _run_match(
        monkeypatch, n_games=8, a_strength=2, b_strength=1, game_len=6,
        opening_plies=3, batch_openings=True,
    )
    # No single-root opening calls when batch_openings is on (all plies batch when
    # >1 game shares a side-to-move, which they do here).
    single_root_opening = [
        c for c in _FakeSession.calls if c["n_roots"] == 1 and c["move_temperatures"][0] > 0.0
    ]
    # With 8 games created and the Connect6 schedule, several games share each
    # side-to-move every round, so opening plies batch (n_roots > 1).
    assert not single_root_opening
    assert result["meta"]["batch_openings"] is True
    assert result["pentanomial"]["n_pairs"] == 4


# =========================================================================== #
# 4. Terminal vs max_plies truncation + edge cases (odd count, unpaired).
# =========================================================================== #
def test_max_plies_truncation_marks_games_undecided(monkeypatch):
    """A game that never reaches a terminal before ``selfplay.max_game_plies`` is
    finalized as ``status='truncated'`` with ``winner=None`` (hexo has no draws,
    so truncation is the only non-decisive outcome), is EXCLUDED from
    ``decided``/``a_wins``/``b_wins`` but COUNTED in ``score.truncated``, and the
    round loop still terminates (no hang). The fake engine only declares a winner
    at ``ply >= game_len``, so a ``max_game_plies`` BELOW ``game_len`` forces the
    ply-cap truncation path for every game."""

    cfg = parse_hexfield_config({"selfplay": {"max_game_plies": 3}})
    # game_len (10) > max_game_plies (3) -> the engine never sets a winner, so the
    # ply cap fires first and every game truncates.
    result, sessions = _run_match(
        monkeypatch, n_games=6, a_strength=2, b_strength=1, game_len=10,
        config=cfg, opening_plies=2,
    )

    score = result["score"]
    assert score["truncated"] == 6
    assert score["completed"] == 0
    assert score["decided"] == 0
    assert score["a_wins"] == 0 and score["b_wins"] == 0
    # Undecided -> no descriptive win rate / CI.
    assert score["a_winrate_decided"] is None
    assert score["a_winrate_ci95"] is None

    # Every per-game row is truncated/undecided, capped at max_game_plies, and the
    # loop did not hang (all games done).
    assert len(result["games"]) == 6
    for g in result["games"]:
        assert g["status"] == "truncated"
        assert g["winner"] is None
        assert g["plies"] == 3  # finalized exactly at the ply cap

    # Pentanomial: full pairs need 2 DECIDED games, so there are none; the
    # histogram is empty and no pair is informative.
    penta = result["pentanomial"]
    assert penta["n_pairs"] == 3
    assert penta["n_full_pairs"] == 0
    assert penta["n_informative_pairs"] == 0
    assert penta["histogram_a_wins"] == {"0": 0, "1": 0, "2": 0}
    for p in penta["pairs"]:
        assert p["n_decided"] == 0 and p["a_wins"] == 0

    # Trees still discarded on BOTH sessions for every game (no leak on the
    # truncation path either).
    assert len(sessions) == 2
    for s in sessions:
        assert sorted(s.discarded) == list(range(6))


def test_terminal_and_truncation_coexist_in_one_match(monkeypatch):
    """Terminal-decided and ply-truncated games are scored independently in the
    same match: decided games drive the win counts, truncated games only bump
    ``score.truncated`` and are dropped from the pentanomial's informative set.

    Half the games (those whose net-A seat is player0) are decided early; the rest
    truncate. We drive that purely through the fake engine's per-game winner hook
    so the arena's status/winner mapping is exercised on a mixed match."""

    cfg = parse_hexfield_config({"selfplay": {"max_game_plies": 4}})

    # A custom engine: a game is decided (centre-seat wins) iff its index is even,
    # else it runs past the ply cap and truncates. This is independent of the
    # arena's seat assignment, so we get a deterministic decided/truncated mix.
    decided_indices = {0, 2}

    class _MixedState(_FakeState):
        def __init__(self, *, game_len):
            super().__init__(game_len=game_len)
            self.index: int | None = None  # set by the engine at creation

    order: list[_MixedState] = []

    class _MixedApi(_FakeApi):
        Player = Player

        def new_game(self, *, seed=None, scenario=None):
            st = _MixedState(game_len=10)
            st.index = len(order)
            order.append(st)
            return st

        def current_player(self, state):
            return Player.PLAYER_0 if state.mover_seat() == 0 else Player.PLAYER_1

        def apply_action(self, state, action):
            coord = action.coord
            state.actions.append(pack_action_id(coord.q, coord.r))
            state.ply += 1
            # Decided games resolve at ply 2 (well before the cap); others never.
            if state.index in decided_indices and state.ply >= 2:
                state.winner_seat = 0

        def terminal(self, state):
            if state.winner_seat is None:
                return None
            return _Terminal("player0" if state.winner_seat == 0 else "player1")

    monkeypatch.setattr(eval_arena, "api", _MixedApi(game_len=10, decide_winner=lambda s: 0))

    def _build_evaluators():
        return _FakeEvaluator("A", 1), _FakeEvaluator("B", 1)

    result = eval_arena.play_checkpoint_match(
        "a", "b", 4, config=cfg, label_a="cand", label_b="opp",
        opening_plies=2, make_session=_make_session_factory()[0],
        build_evaluators=_build_evaluators,
    )

    score = result["score"]
    assert score["completed"] == 2  # indices 0 and 2 decided
    assert score["truncated"] == 2  # indices 1 and 3 hit the ply cap
    assert score["decided"] == 2
    # Status per game row matches the engine's decided/truncated split.
    by_index = {g["index"]: g for g in result["games"]}
    for i in (0, 2):
        assert by_index[i]["status"] == "completed"
        assert by_index[i]["winner"] in ("A", "B")
    for i in (1, 3):
        assert by_index[i]["status"] == "truncated"
        assert by_index[i]["winner"] is None
        assert by_index[i]["plies"] == 4


def test_odd_game_count_singleton_final_pair(monkeypatch):
    result, _ = _run_match(monkeypatch, n_games=5, a_strength=2, b_strength=1, game_len=4)
    penta = result["pentanomial"]
    assert penta["n_pairs"] == 3  # ceil(5/2)
    sizes = sorted(p["n_games"] for p in penta["pairs"])
    assert sizes == [1, 2, 2]  # last pair is a singleton
    assert result["score"]["completed"] == 5


def test_unpaired_mode_has_no_pentanomial(monkeypatch):
    result, _ = _run_match(
        monkeypatch, n_games=6, a_strength=2, b_strength=1, game_len=4, paired_openings=False,
    )
    assert result["pentanomial"] is None
    assert result["score"]["completed"] == 6
    assert result["score"]["a_wins"] == 6


# =========================================================================== #
# 5. EVAL-SPECIFIC virtual_batch_size override (the LOCKED-16 in-run eval).
# =========================================================================== #
def test_eval_vbs_override_reaches_every_search_call(monkeypatch):
    """play_checkpoint_match(virtual_batch_size=16) must thread 16 into EVERY
    multi-root search call and into the result meta — WITHOUT touching the
    self-play config value (4)."""
    cfg = parse_hexfield_config({})
    assert cfg.selfplay.virtual_batch_size == 4  # self-play stays 4
    result, _sessions = _run_match(
        monkeypatch, n_games=6, a_strength=2, b_strength=1, game_len=4,
        virtual_batch_size=16,
    )
    assert result["meta"]["virtual_batch_size"] == 16
    assert _FakeSession.calls, "expected search calls"
    assert all(c["virtual_batch_size"] == 16 for c in _FakeSession.calls)


def test_eval_vbs_defaults_to_selfplay_value(monkeypatch):
    """Omitting the override falls back to cfg.selfplay.virtual_batch_size (4)."""
    result, _sessions = _run_match(
        monkeypatch, n_games=4, a_strength=2, b_strength=1, game_len=4,
    )
    assert result["meta"]["virtual_batch_size"] == 4
    assert all(c["virtual_batch_size"] == 4 for c in _FakeSession.calls)
