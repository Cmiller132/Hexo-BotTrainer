#!/usr/bin/env python3
"""Capture the CURRENT (pre-refactor) full result dicts of the four eval arenas.

This is a golden-baseline helper for the centralized-eval-runner refactor
(plan: lovely-sniffing-castle). It drives each arena
(``play_checkpoint_match`` / ``play_multi_checkpoint_match`` /
``play_strix_match`` / ``play_sealbot_match``) through the SAME CPU injection
seams the migration verifiers use, at a FIXED ``game_seed_base``, and dumps the
full result dict (json, ``sort_keys=True``) so later migrations can be proven
bit-identical.

CPU-only: no GPU, no real checkpoints, no STRIX_CKPT. Every arena is driven with
the deterministic fakes the existing arena tests use:

  * checkpoint / multi -> ``hexfield_eval_kit`` fakes + the tracking fake engine
    from ``test_hexfield_eval_arena`` (monkeypatched onto ``eval_arena.api``).
  * strix             -> the first-move / last-move fakes from
    ``test_hexfield_strix_arena_paired`` (real ``hexo_engine`` engine).
  * sealbot           -> a fake ``hexo_runner.adapters.sealbot`` module + fake
    engine/session/evaluator (play_sealbot_match has only the ``build_opponent``
    seam, so the hexfield-side session/evaluator/net loaders are monkeypatched).

Volatile (timing / environment) meta keys are NOT stripped here — the full dict
is dumped as ground truth. The equivalence verifier is responsible for ignoring
the known-volatile keys documented at the bottom of this file.

Run (WSL):
    /root/.venvs/hexgt-build/bin/python scripts/_eval_golden_capture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- repo package + test-kit paths (mirror the test modules' sys.path prep). --
_REPO = Path(__file__).resolve().parent.parent
for _p in (
    _REPO / "packages" / "hexo_engine" / "python",
    _REPO / "packages" / "hexfield" / "python",
    _REPO / "tests",  # hexfield_eval_kit lives here
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hexo_engine import api  # noqa: E402
from hexo_engine.types import Player  # noqa: E402

from hexfield import eval_arena  # noqa: E402  (torch-free import)
from hexfield.config import parse_hexfield_config  # noqa: E402

from hexfield_eval_kit import (  # noqa: E402
    _FakeApi,
    _FakeEvaluator,
    _FakeSession,
    _FakeState,
    _make_session_factory,
)

# Output directory (WSL path; this helper is only ever run in the WSL venv).
OUT_DIR = Path(
    "/mnt/c/Users/epicm/AppData/Local/Temp/claude/"
    "E--Hexo-BotTrainer-hexgt/7bcece5b-76b5-44fd-bed2-6ab286787cb4/"
    "scratchpad/eval_baselines"
)

# One fixed seed for every arena so the goldens are reproducible.
GAME_SEED_BASE = 100


# --------------------------------------------------------------------------- #
# Arena 1: play_checkpoint_match  (mirrors test_hexfield_eval_arena._run_match)
# --------------------------------------------------------------------------- #
def capture_checkpoint() -> dict:
    n_games, a_strength, b_strength, game_len, opening_plies = 8, 2, 1, 6, 2

    _FakeSession.calls = []
    factory, _sessions = _make_session_factory()

    # Per-state net-A seat, filled as games are created in arena order.
    seat_iter: list[bool] = []
    n_pairs = (n_games + 1) // 2
    for p in range(n_pairs):
        seat_iter.append(True)
        if 2 * p + 1 < n_games:
            seat_iter.append(False)

    created: list[_FakeState] = []
    seat_of_state: dict[int, bool] = {}

    def decide(state: _FakeState) -> int:
        a_is_p0 = seat_of_state[id(state)]
        a_seat = 0 if a_is_p0 else 1
        b_seat = 1 - a_seat
        if a_strength == b_strength:
            return 0
        return a_seat if a_strength > b_strength else b_seat

    base_api = _FakeApi(game_len=game_len, decide_winner=decide)

    class _TrackingApi(_FakeApi):
        Player = Player

        def new_game(self, *, seed=None, scenario=None):
            st = _FakeState(game_len=game_len)
            created.append(st)
            seat_of_state[id(st)] = seat_iter[len(created) - 1]
            return st

        def current_player(self, state):
            return base_api.current_player(state)

        def apply_action(self, state, action):
            return base_api.apply_action(state, action)

        def terminal(self, state):
            return base_api.terminal(state)

    orig_api = eval_arena.api
    eval_arena.api = _TrackingApi(game_len=game_len, decide_winner=decide)
    try:
        return eval_arena.play_checkpoint_match(
            "ckpt_a.pt", "ckpt_b.pt", n_games,
            config=parse_hexfield_config({}),
            label_a="cand", label_b="opp",
            make_session=factory,
            build_evaluators=lambda: (
                _FakeEvaluator("A", a_strength), _FakeEvaluator("B", b_strength)
            ),
            opening_plies=opening_plies,
            game_seed_base=GAME_SEED_BASE,
        )
    finally:
        eval_arena.api = orig_api


# --------------------------------------------------------------------------- #
# Arena 2: play_multi_checkpoint_match  (mirrors ._run_concurrent)
# --------------------------------------------------------------------------- #
def _seat_iter(n_games: int) -> list[bool]:
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
        return 0
    return a_seat if a_strength > b_strength else b_seat


class _MultiTrackingApi:
    """Stand-in for hexo_engine.api; tags each game with its (opponent, seat)."""

    Player = Player

    def __init__(self, *, game_len, plan, strength_by_opp):
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
        from hexfield.geometry import pack_action_id

        coord = action.coord
        state.actions.append(pack_action_id(coord.q, coord.r))
        state.ply += 1
        if state.ply >= state.game_len:
            opp, a_is_p0 = self._tag[id(state)]
            a_str, b_str = self._strength[opp]
            state.winner_seat = _winner_seat_for(a_str, b_str, a_is_p0)

    def terminal(self, state):
        from hexfield_eval_kit import _Terminal

        if state.winner_seat is None:
            return None
        return _Terminal("player0" if state.winner_seat == 0 else "player1")


def capture_multi() -> dict:
    opponents = [("opp_weak", 1), ("opp_strong", 5), ("opp_even", 3)]
    n_games, candidate_strength, game_len, opening_plies = 8, 3, 6, 2
    cfg = parse_hexfield_config({})

    _FakeSession.calls = []
    strength = {label: (candidate_strength, opp) for label, opp in opponents}
    plan: list[tuple[str, bool]] = []
    for label, _ in opponents:
        plan.extend((label, a_is_p0) for a_is_p0 in _seat_iter(n_games))
    engine = _MultiTrackingApi(game_len=game_len, plan=plan, strength_by_opp=strength)

    def factory():
        return _FakeSession()

    opp_strength = dict(opponents)

    orig_api = eval_arena.api
    eval_arena.api = engine
    try:
        return eval_arena.play_multi_checkpoint_match(
            "cand.pt",
            [(label, f"{label}.pt") for label, _ in opponents],
            n_games,
            config=cfg, candidate_label="cand",
            make_session=factory,
            build_candidate_evaluator=lambda: _FakeEvaluator("A", candidate_strength),
            build_opponent_evaluator=lambda label, ckpt: _FakeEvaluator(
                label, opp_strength[label]
            ),
            opening_plies=opening_plies,
            game_seed_base=GAME_SEED_BASE,
        )
    finally:
        eval_arena.api = orig_api


# --------------------------------------------------------------------------- #
# Arena 3: play_strix_match  (mirrors test_hexfield_strix_arena_paired)
# --------------------------------------------------------------------------- #
def capture_strix() -> dict:
    import dataclasses
    from types import SimpleNamespace

    from hexo_engine.types import pack_coord_id as _pack

    class _StrixFakeSession:
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

    class _StrixFakeStrix:
        """Strix player: plays the LAST legal move (distinct from hexfield's)."""

        def __init__(self, seed: int, is_leader: bool) -> None:
            self.seed = seed
            self.is_leader = is_leader

        def decide(self, state):
            return SimpleNamespace(action=api.legal_actions(state)[-1])

    class _StrixFakeServer:
        def __init__(self) -> None:
            self.closed = 0
            self.n_forwards = 0
            self.n_graphs = 0
            self.max_seen_batch = 0

        def close(self) -> None:
            self.closed += 1

    cfg = parse_hexfield_config({})
    cfg = dataclasses.replace(
        cfg, selfplay=dataclasses.replace(cfg.selfplay, max_game_plies=16)
    )
    session = _StrixFakeSession()
    server = _StrixFakeServer()

    return eval_arena.play_strix_match(
        "hexfield.pt", "strix.pt", 4,
        config=cfg,
        opening_plies=4,
        paired_openings=True,
        diagnostics_dir=None,  # keep hxr_record None -> path-independent golden
        game_seed_base=GAME_SEED_BASE,
        build_hexfield_evaluator=lambda: object(),
        make_session=lambda: session,
        build_strix_player=lambda seed, is_leader: _StrixFakeStrix(seed, is_leader),
        make_batch_server=lambda: server,
    )


# --------------------------------------------------------------------------- #
# Arena 4: play_sealbot_match  (unpaired; only a build_opponent seam exists, so
# the hexfield-side net/session/evaluator loaders are monkeypatched too).
# --------------------------------------------------------------------------- #
def capture_sealbot() -> dict:
    import types as _types
    from types import SimpleNamespace

    n_games, game_len, opening_plies = 6, 6, 2

    # A fake hexo_runner.adapters.sealbot module so no SealBot checkout / minimax
    # .so is needed and SealBotConfig.validate() is a no-op.
    fake_mod = _types.ModuleType("hexo_runner.adapters.sealbot")

    class _SealBotUnavailableError(RuntimeError):
        pass

    class _SealBotConfig:
        def __init__(self, *, path=None, variant="best", time_limit=0.05, **kw):
            self.path = path
            self.variant = variant
            self.time_limit = time_limit

        def validate(self):  # no-op: bot presence is faked
            return None

    class _SealBotPlayer:  # unused (build_opponent seam supplies the opponent)
        def __init__(self, *a, **k):
            pass

    fake_mod.SealBotConfig = _SealBotConfig
    fake_mod.SealBotPlayer = _SealBotPlayer
    fake_mod.SealBotUnavailableError = _SealBotUnavailableError

    class _FakeSealbot:
        """Deterministic opponent: move is a pure function of the ply."""

        def decide(self, state):
            q = (state.ply % 3) - 1
            r = ((state.ply // 3) % 3) - 1
            return SimpleNamespace(action=SimpleNamespace(coord=SimpleNamespace(q=q, r=r)))

        def close(self):
            return None

    class _FakeHexEvaluator:
        tag = "hex"

        def __init__(self, *a, **k):
            pass

    # Winner is always engine seat 0 -> even-index games (hex is player0) are
    # hexfield wins, odd-index games are sealbot wins: a deterministic split.
    decide_winner = lambda state: 0
    fake_engine = _FakeApi(game_len=game_len, decide_winner=decide_winner)

    import hexfield.inference as _inf  # torch present in this venv; import is fine

    saved = []

    def _patch(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    prev_mod = sys.modules.get("hexo_runner.adapters.sealbot")
    sys.modules["hexo_runner.adapters.sealbot"] = fake_mod
    _patch(eval_arena, "api", fake_engine)
    _patch(eval_arena, "_load_hexfield_net", lambda ckpt: object())
    _patch(eval_arena, "_new_rust_session", lambda max_states: _FakeSession())
    _patch(_inf, "HexfieldEvaluator", _FakeHexEvaluator)
    try:
        return eval_arena.play_sealbot_match(
            "hexfield.pt", n_games,
            config=parse_hexfield_config({}),
            label="hexfield",
            opening_plies=opening_plies,
            diagnostics_dir=None,
            game_seed_base=GAME_SEED_BASE,
            build_opponent=lambda sealbot_config: _FakeSealbot(),
        )
    finally:
        for obj, name, val in reversed(saved):
            setattr(obj, name, val)
        if prev_mod is None:
            sys.modules.pop("hexo_runner.adapters.sealbot", None)
        else:
            sys.modules["hexo_runner.adapters.sealbot"] = prev_mod


# --------------------------------------------------------------------------- #
def _dump(name: str, result: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    text = json.dumps(result, sort_keys=True, indent=2)
    path.write_text(text, encoding="utf-8")

    # Summary: top-level key count (+ per-opponent for the multi arena).
    if name == "multi_checkpoint":
        summary = ", ".join(
            f"{opp}:{len(match)}keys" for opp, match in sorted(result.items())
        )
        print(f"[{name}] wrote {path} ({len(text)} bytes); opponents -> {summary}")
        print(f"    top-level keys (opponent labels): {sorted(result)}")
    else:
        print(f"[{name}] wrote {path} ({len(text)} bytes); {len(result)} top-level keys")
        print(f"    top-level keys: {sorted(result)}")


def main() -> int:
    arenas = {
        "checkpoint": capture_checkpoint,
        "multi_checkpoint": capture_multi,
        "strix": capture_strix,
        "sealbot": capture_sealbot,
    }
    ok = True
    for name, fn in arenas.items():
        try:
            result = fn()
            _dump(name, result)
        except Exception as exc:  # keep going so one failure does not hide others
            ok = False
            import traceback

            print(f"[{name}] FAILED: {exc!r}")
            traceback.print_exc()
    print("\nAll baselines captured." if ok else "\nSOME baselines FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
