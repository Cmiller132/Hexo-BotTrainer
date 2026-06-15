"""Real-ABI golden tests for the CONCURRENT checkpoint-vs-checkpoint arena
(``eval_arena.play_checkpoint_match``) driven against the REAL native
``HexfieldMctsSession`` with a numpy STUB evaluator — no torch, no GPU.

The sibling ``test_hexfield_eval_arena_concurrent.py`` exercises the loop's
bookkeeping with a fully-faked engine + session (pure Python, always runs).
This file complements it by running the loop through the ACTUAL multi-root
``search`` ABI + real engine, so it pins the things only the real search can
prove:

  * NATIVE-EQUIVALENCE-OF-PAIRING + DETERMINISM. The opening LEADERS now batch
    cross-game (each leader root seeded ``open_seed+index`` via the native per-root
    offset), so the leader's specific opening LINE differs from a single-root
    serial replay — that is fine; the load-bearing invariant is the PAIRING, not
    byte-equivalence to the old single-root line. So we pin: (a) the seed-INDEPENDENT
    pairing structure (game count + per-game seat) matches a serial reference,
    (b) within each pair the follower replays the leader (siblings share the
    opening line), and (c) two concurrent runs are BYTE-IDENTICAL (batching changes
    the evaluator-call ORDER, never the game). Both the opening leaders and the
    greedy tail are batched (forward_batches << n_games*plies).

  * CRN under real batching. With two IDENTICAL stub evaluators the two
    seat-swapped games of a pair play the IDENTICAL line through real MCTS, so
    every decided full pair SPLITS (``pentanomial_a_score == 1``) and the paired
    siblings' opening prefixes match ply-for-ply (the follower replays the leader).

The evaluator is a deterministic, seat-symmetric numpy stub speaking the §5.2
ABI; it is injected via ``build_evaluators`` so no checkpoint is loaded.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

_PACKAGES = Path(__file__).resolve().parent.parent / "packages"
for _p in ("hexfield/python", "hexo_engine/python"):
    _src = str(_PACKAGES / _p)
    if _src not in sys.path:
        sys.path.insert(0, _src)

from hexo_engine import api  # noqa: E402
from hexo_engine.types import AxialCoord, PlacementAction  # noqa: E402

from hexfield import eval_arena  # noqa: E402  (torch-free import)
from hexfield.config import build_divergence_overrides, parse_hexfield_config  # noqa: E402
from hexfield.geometry import unpack_action_id  # noqa: E402

try:
    from hexfield import _rust as hexfield_rust
except ImportError:  # pragma: no cover
    hexfield_rust = None

needs_native = pytest.mark.skipif(
    hexfield_rust is None, reason="hexfield native module not built"
)


# --------------------------------------------------------------------------- #
# Deterministic, seat-symmetric numpy stub evaluator (speaks the §5.2 ABI).
# Pure function of the position (legal-coordinate prefix), so two paired siblings
# at the same position get the IDENTICAL value/priors regardless of seat — what
# the CRN guarantee needs. ``salt`` lets two stubs differ (asymmetric strengths).
# --------------------------------------------------------------------------- #
def _hash_coords(coords) -> int:
    h = 1469598103934665603
    for q, r in coords:
        h = (h ^ (int(q) & 0xFFFF)) * 1099511628211 % (1 << 61)
        h = (h ^ (int(r) & 0xFFFF)) * 1099511628211 % (1 << 61)
    return h


class _StubEvaluator:
    def __init__(self, salt: int = 0) -> None:
        self.salt = int(salt)
        self.calls = 0

    def __call__(self, payload: dict) -> dict:
        import numpy as np

        b, total = (int(x) for x in payload["shape"])
        self.calls += 1
        legal_counts = np.frombuffer(bytes(payload["legal_counts"]), dtype=np.int32)
        offsets = np.asarray(payload["node_row_offsets"], dtype=np.int64)
        qr = np.frombuffer(bytes(payload["node_qr"]), dtype=np.int16).reshape(total, 2)

        values: list[float] = []
        priors: list[float] = []
        for g in range(b):
            o = int(offsets[g])
            ln = int(legal_counts[g])
            legal = [(int(qr[o + i, 0]), int(qr[o + i, 1])) for i in range(ln)]
            rh = _hash_coords(legal) ^ (self.salt * 0x9E3779B97F4A7C15)
            values.append(((rh % 2001) - 1000) / 1000.0)  # [-1, 1], position-pure
            for i, (q, r) in enumerate(legal):
                priors.append(float((q * 2654435761 + r * 40503 + rh + i) % 997 + 1))
        reply = {
            "values_bytes": struct.pack(f"<{b}f", *values),
            "priors_bytes": struct.pack(f"<{len(priors)}f", *priors),
        }
        if payload.get("request_moves_left"):
            reply["moves_left_bytes"] = struct.pack(f"<{b}f", *([100.0] * b))
        return reply


def _cfg(*, visits: int, max_plies: int, vbs: int = 4):
    return parse_hexfield_config(
        {
            "device": "cpu",
            "selfplay": {
                "search_visits": visits,
                "virtual_batch_size": vbs,
                "max_game_plies": max_plies,
                "active_root_limit": 64,
            },
            # Deliberately huge so a regression that uses eval_visits is caught.
            "evaluation": {"eval_visits": visits + 1000},
        }
    )


def _serial_reference(cfg, eval_a, eval_b, *, n_games, opening_plies,
                      opening_temperature, game_seed_base, visits):
    """Serial reference: replay each game one ply at a time, single-root, no
    greedy batching, recording per-game winner/status/plies/opening line + seat.

    SCOPE NOTE: since the runner now BATCHES the opening leaders (each leader root
    sampled with the native per-root ``open_seed+index`` rather than a single-root
    ``seed``), this single-root serial replay no longer reproduces the runner's
    leader opening LINE byte-for-byte — and therefore not the winner/status/plies
    that depend on it. The caller uses this reference ONLY for the seed-INDEPENDENT
    pairing structure (game count + per-game seat assignment); the opening-line /
    winner equivalence is checked instead via within-pair pairing (follower==leader)
    and concurrent self-determinism. The leader's opening RNG here is kept as
    ``pair_seed*5003+ply`` (the historical single-root stream) purely so the
    forced-opening REPLAY mechanics below are exercised faithfully.

    FORCED-OPENING CRN (L-1): within a pair the LEADER (``a_is_p0=True``, game 0)
    searches its opening and its opening line is recorded; the FOLLOWER
    (``a_is_p0=False``, game 1) does NOT search the opening — it REPLAYS the
    leader's recorded action for each opening ply, so the pair shares the real
    opening LINE (the seat swap means a shared seed alone would NOT, because a
    different net moves at ply 0). If the leader ended its game before
    ``opening_plies`` (fewer recorded actions), the follower falls back to a
    single-root search for the remaining opening plies — the same fallback
    ``eval_arena`` uses (the follower shares the leader's seed)."""

    sp = cfg.selfplay
    ov = build_divergence_overrides(sp)
    rows = []
    n_pairs = (n_games + 1) // 2
    for pair_index in range(n_pairs):
        pair_seed = game_seed_base + pair_index
        leader_line: list[int] = []  # the leader's recorded opening, replayed below
        for a_is_p0 in (True, False):
            game_index = pair_index * 2 + (0 if a_is_p0 else 1)
            if game_index >= n_games:
                continue
            is_leader = a_is_p0  # game 0 leads, game 1 (seat-swapped) follows
            s_a = hexfield_rust.HexfieldMctsSession(max_states=4096)
            s_b = hexfield_rust.HexfieldMctsSession(max_states=4096)
            state = api.new_game()
            line: list[int] = []
            ply = 0
            winner = None
            status = "truncated"
            while ply < sp.max_game_plies:
                in_opening = ply < opening_plies
                # FOLLOWER opening: replay the leader's recorded action (no search)
                # when one exists for this ply; otherwise fall back to a search.
                if (not is_leader) and in_opening and ply < len(leader_line):
                    aid = int(leader_line[ply])
                else:
                    a_to_move = (api.current_player(state) == api.Player.PLAYER_0) == a_is_p0
                    session = s_a if a_to_move else s_b
                    evaluator = eval_a if a_to_move else eval_b
                    temperature = opening_temperature if in_opening else 0.0
                    out = session.search(
                        [game_index], (state,),
                        visits=visits, c_puct=sp.c_puct, temperature=temperature,
                        seed=pair_seed * 5003 + ply, evaluator=evaluator,
                        virtual_batch_size=sp.virtual_batch_size,
                        move_temperatures=[temperature],
                        widening_policy_mass=sp.widening_policy_mass,
                        widening_max_children=sp.widening_max_children,
                        widening_min_children=sp.widening_min_children,
                        fpu_reduction=sp.fpu_reduction, tss_enabled=sp.tss_enabled,
                        search_parity_mode=sp.search_parity_mode,
                        divergence_overrides=ov,
                    )[0]
                    aid = int(out["action_id"])
                if len(line) < opening_plies:
                    line.append(aid)
                q, r = unpack_action_id(aid)
                result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
                ply += 1
                if result.terminal:
                    terminal = api.terminal(state)
                    won_p0 = str(terminal.winner) == "player0"
                    winner = "A" if (won_p0 == a_is_p0) else "B"
                    status = "completed"
                    break
            if is_leader:
                leader_line = line  # record for the follower's replay
            rows.append({"index": game_index, "winner": winner, "status": status,
                         "plies": ply, "opening": line, "a_seat": "P0" if a_is_p0 else "P1"})
    return rows


@needs_native
def test_concurrent_pairing_matches_serial_and_is_deterministic() -> None:
    """Native-equivalence-of-PAIRING + batched-opening determinism.

    The opening LEADERS now batch cross-game, so each leader root samples with the
    native per-root seed ``open_seed+index`` instead of a single-root ``seed`` —
    the leader's specific opening LINE therefore differs from a single-root serial
    replay, which is FINE (the load-bearing invariant is the PAIRING, not byte
    equivalence to the old single-root line). So this test pins what IS invariant:

      1. PAIRING STRUCTURE matches a serial reference: same game count, same seat
         assignment per game (seed-independent), same pair membership.
      2. WITHIN-PAIR PAIRING: the follower replays the leader, so the two siblings
         of every pair share the IDENTICAL opening line and (under these stubs) a
         consistent pair outcome.
      3. DETERMINISM: two concurrent runs with identical inputs produce
         BYTE-IDENTICAL per-game rows — batching changes the evaluator-call ORDER,
         never the game (the meaningful "concurrent == itself" equivalence now that
         the single-root serial baseline plays a different opening line).

    Two different stubs make the strengths asymmetric (a stronger discriminator)."""

    vbs = 8
    visits = 16
    cfg = _cfg(visits=visits, max_plies=24, vbs=vbs)
    n_games = 6
    opening_plies, opening_temperature, seed_base = 4, 1.0, 4242

    # Serial reference: used here ONLY for the seed-independent pairing structure
    # (game count + per-game seat assignment), NOT for the opening line / winner,
    # which legitimately diverge now that leaders batch.
    serial_rows = _serial_reference(
        cfg, _StubEvaluator(salt=1), _StubEvaluator(salt=2),
        n_games=n_games, opening_plies=opening_plies,
        opening_temperature=opening_temperature, game_seed_base=seed_base, visits=visits,
    )

    def _run():
        return eval_arena.play_checkpoint_match(
            "a", "b", n_games,
            config=cfg, label_a="A", label_b="B",
            paired_openings=True, opening_plies=opening_plies,
            opening_temperature=opening_temperature, game_seed_base=seed_base,
            build_evaluators=lambda: (_StubEvaluator(salt=1), _StubEvaluator(salt=2)),
        )

    res = _run()
    conc = {g["index"]: g for g in res["games"]}

    # (1) Pairing STRUCTURE matches the serial reference: count + seat per game.
    assert len(serial_rows) == n_games
    assert len(res["games"]) == n_games
    for sref in serial_rows:
        cg = conc[sref["index"]]
        assert cg["a_seat"] == sref["a_seat"], (sref["index"], "seat")

    # (2) Within-pair PAIRING: siblings seat-swapped, share the opening line.
    for p in res["pentanomial"]["pairs"]:
        if p["n_games"] != 2:
            continue
        i0, i1 = p["game_indices"]
        assert {conc[i0]["a_seat"], conc[i1]["a_seat"]} == {"P0", "P1"}
        assert conc[i0]["opening"][:opening_plies] == conc[i1]["opening"][:opening_plies], (
            f"pair {p['pair_index']} siblings diverged on the opening (replay broke)"
        )

    # (3) DETERMINISM: a second identical run reproduces every game byte-for-byte.
    res2 = _run()
    conc2 = {g["index"]: g for g in res2["games"]}
    for idx, cg in conc.items():
        cg2 = conc2[idx]
        for key in ("winner", "status", "plies", "opening", "a_seat", "seed"):
            assert cg[key] == cg2[key], (idx, key, cg[key], cg2[key])

    # Anti-vacuity: the opening lines must be real (the sampled prefix has more
    # than just the forced centre stone), the opening LEADERS actually BATCHED
    # (a multi-root opening forward), and the greedy tail batched too.
    assert any(len(g["opening"]) >= opening_plies for g in res["games"])
    assert res["meta"]["rounds"] >= opening_plies
    # forward_batches counts every search call; with batched openings AND a batched
    # greedy tail it is far below "one per ply per game" (n_games * plies).
    total_plies = sum(g["plies"] for g in res["games"])
    assert res["meta"]["forward_batches"] < total_plies


@needs_native
def test_crn_paired_siblings_share_line_and_split() -> None:
    """Two IDENTICAL stubs -> the seat-swapped games of every pair play the same
    real-MCTS line, so paired opening prefixes match and every DECIDED full pair
    splits (pentanomial_a_score == 1)."""

    cfg = _cfg(visits=16, max_plies=24)
    n_games = 8
    res = eval_arena.play_checkpoint_match(
        "x", "x", n_games,
        config=cfg, label_a="A", label_b="B",
        paired_openings=True, opening_plies=4, opening_temperature=1.0,
        game_seed_base=7,
        build_evaluators=lambda: (_StubEvaluator(salt=0), _StubEvaluator(salt=0)),
    )
    games = res["games"]
    for pi in range(n_games // 2):
        g0, g1 = games[2 * pi], games[2 * pi + 1]
        assert g0["opening"] == g1["opening"], (
            f"pair {pi} opening diverged: {g0['opening']} vs {g1['opening']}"
        )
    for p in res["pentanomial"]["pairs"]:
        if p["n_games"] == 2 and p["n_decided"] == 2:
            assert p["pentanomial_a_score"] == 1, f"pair did not split: {p}"
    hist = res["pentanomial"]["histogram_a_wins"]
    assert hist["0"] == 0 and hist["2"] == 0


def _opening_led_by(cfg, p0_eval, p1_eval, *, seed_base, opening_plies):
    """Search a full opening LINE with ``p0_eval`` as the net at engine seat P0
    and ``p1_eval`` at P1, using the serial CRN RNG (``seed_base*5003+ply``).
    Returns the opening action-id line. This reconstructs, for the FOLLOWER, the
    line it WOULD have searched without forced-opening replay (its net sits at P0
    swapped vs the leader), so the test can prove replay actually changed it."""

    sp = cfg.selfplay
    ov = build_divergence_overrides(sp)
    state = api.new_game()
    line: list[int] = []
    s0 = hexfield_rust.HexfieldMctsSession(max_states=4096)
    s1 = hexfield_rust.HexfieldMctsSession(max_states=4096)
    for ply in range(opening_plies):
        p0_to_move = api.current_player(state) == api.Player.PLAYER_0
        evaluator = p0_eval if p0_to_move else p1_eval
        session = s0 if p0_to_move else s1
        out = session.search(
            [0], (state,),
            visits=sp.search_visits, c_puct=sp.c_puct, temperature=1.0,
            seed=seed_base * 5003 + ply, evaluator=evaluator,
            virtual_batch_size=sp.virtual_batch_size, move_temperatures=[1.0],
            widening_policy_mass=sp.widening_policy_mass,
            widening_max_children=sp.widening_max_children,
            widening_min_children=sp.widening_min_children,
            fpu_reduction=sp.fpu_reduction, tss_enabled=sp.tss_enabled,
            search_parity_mode=sp.search_parity_mode, divergence_overrides=ov,
        )[0]
        aid = int(out["action_id"])
        line.append(aid)
        q, r = unpack_action_id(aid)
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    return line


@needs_native
def test_forced_opening_replay_shares_line_under_asymmetric_nets() -> None:
    """FORCED-OPENING CRN (L-1): with ASYMMETRIC nets the paired siblings STILL
    share the identical opening LINE — proving the share comes from REPLAYING the
    leader's recorded actions, not from net symmetry.

    The seat swap means a different net moves at the first real decision ply in
    each sibling (the leader has net A at P0, the follower has net B at P0), so if
    the pair only shared the RNG STREAM (the pre-L-1 behavior) the asymmetric nets
    would sample a DIFFERENT opening and the lines would diverge after the forced
    centre stone. We first prove that divergence directly (the line the follower
    WOULD have searched, net B leading from P0, differs from the leader's within
    the opening), then assert the runner's actual siblings agree ply-for-ply."""

    cfg = _cfg(visits=16, max_plies=24)
    opening_plies, seed_base = 4, 9999
    sa, sb = 11, 29  # asymmetric salts

    # Anti-vacuity: the leader line (net A at P0) and the line the FOLLOWER would
    # have searched WITHOUT replay (net B at P0 — its swapped seat) genuinely
    # diverge somewhere within the opening. (Ply 0 is the forced centre stone, so
    # the divergence appears from ply 1 on; we only require SOME divergence.)
    leader_line = _opening_led_by(
        cfg, _StubEvaluator(salt=sa), _StubEvaluator(salt=sb),
        seed_base=seed_base, opening_plies=opening_plies,
    )
    follower_would_be = _opening_led_by(
        cfg, _StubEvaluator(salt=sb), _StubEvaluator(salt=sa),
        seed_base=seed_base, opening_plies=opening_plies,
    )
    assert leader_line != follower_would_be, (
        "stubs are not asymmetric on the opening; absent replay the siblings would "
        "still match, making the replay test vacuous"
    )

    n_games = 8
    res = eval_arena.play_checkpoint_match(
        "a", "b", n_games,
        config=cfg, label_a="A", label_b="B",
        paired_openings=True, opening_plies=opening_plies, opening_temperature=1.0,
        game_seed_base=seed_base,
        build_evaluators=lambda: (_StubEvaluator(salt=sa), _StubEvaluator(salt=sb)),
    )
    games = {g["index"]: g for g in res["games"]}
    for p in res["pentanomial"]["pairs"]:
        i0, i1 = p["game_indices"]
        # The follower (the seat-swapped P1 sibling) replayed the LEADER's (P0)
        # opening, so its prefix equals the leader's exactly — even though, left to
        # search on its own (``follower_would_be``), it would have diverged.
        leader = games[i0] if games[i0]["a_seat"] == "P0" else games[i1]
        follower = games[i1] if leader is games[i0] else games[i0]
        assert follower["opening"][:opening_plies] == leader["opening"][:opening_plies], (
            f"pair {p['pair_index']} follower diverged from the leader despite "
            f"replay: {follower['opening']} vs {leader['opening']}"
        )

    # Pair 0's leader is seat P0, follower seat P1, and the leader drove a REAL
    # sampled line (more than the forced centre stone) which the follower replayed.
    # We deliberately do NOT compare ``leader0["opening"]`` to ``leader_line`` (the
    # single-root ``_opening_led_by`` reconstruction): the leader now samples via
    # the native per-root ``open_seed+index`` in a cross-game batch, so its specific
    # line legitimately differs from the single-root stream — the load-bearing
    # invariant is the PAIRING (follower replays leader, asserted above), not byte
    # equivalence to the old single-root opening line. The anti-vacuity check
    # (``leader_line != follower_would_be``) above already proves the asymmetric
    # stubs produce seat-divergent openings, so the follower-replays-leader match is
    # non-vacuous regardless of which specific line the leader sampled.
    pair0 = res["pentanomial"]["pairs"][0]
    leader0 = games[pair0["game_indices"][0]]
    follower0 = games[pair0["game_indices"][1]]
    assert leader0["a_seat"] == "P0" and follower0["a_seat"] == "P1"
    assert len(leader0["opening"]) >= opening_plies


@needs_native
def test_full_sims_default_through_real_search() -> None:
    """visits=None runs the real search at selfplay.search_visits (full sims);
    the returned root visit count reflects the full budget, and an explicit
    visits overrides it."""

    cfg = _cfg(visits=24, max_plies=10)
    res = eval_arena.play_checkpoint_match(
        "a", "b", 2, config=cfg, paired_openings=True,
        opening_plies=2, opening_temperature=1.0,
        build_evaluators=lambda: (_StubEvaluator(), _StubEvaluator()),
    )
    assert res["meta"]["visits"] == 24  # == search_visits, not eval_visits (1024)

    res2 = eval_arena.play_checkpoint_match(
        "a", "b", 2, config=cfg, visits=9, paired_openings=True,
        opening_plies=2, opening_temperature=1.0,
        build_evaluators=lambda: (_StubEvaluator(), _StubEvaluator()),
    )
    assert res2["meta"]["visits"] == 9
