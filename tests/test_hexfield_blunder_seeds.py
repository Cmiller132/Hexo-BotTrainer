"""Tests for blunder-seeded self-play (hexfield.blunder_seeds + selfplay seeding).

The load-bearing test is the golden equivalence one: a seeded game and a
live-style game that reach the SAME position must emit bit-identical rows,
because the seeded tape must be indistinguishable from one reached by live play.
record_player / record_phase / window_scan are pure functions of the ply counter
and records list (NOT the engine state), so a tape whose ply/records/player-parity
are consistent produces correct labels.

These tests drive ContinuousDriver.__call__ directly with synthetic full-search
payloads (no Rust search session): each decision picks a legal move from the
engine's own legal set, and the driver builds and finalizes the exact same
HexfieldSampleData rows the live path would. The engine + label functions are
real (no mocks).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

# Imported for its side effect: prepends the working-tree packages/hexfield/python
# to sys.path so `hexfield` resolves to the source under test, not a stale
# site-packages install (which may predate modules like blunder_seeds).
import hexfield_testkit  # noqa: F401

from hexfield import blunder_seeds as bs
from hexfield.config import SelfplayConfig
from hexfield.features import record_phase, record_player
from hexfield.geometry import pack_action_id, unpack_action_id
from hexfield.samples import finalize_game_samples
from hexfield.selfplay import ContinuousDriver, seed_game_tape
from hexfield.shards import read_compact_shard, write_compact_shard


# ---------------------------------------------------------------------------
# Synthetic full-search payload + deterministic move choice.
# ---------------------------------------------------------------------------

def _legal_qr(state) -> list[tuple[int, int]]:
    """Ordered list of legal (q, r) at a state (via the engine's own legals)."""
    return sorted(unpack_action_id(int(a)) for a in api.legal_action_ids(state))


def _pick_move(state, salt: int) -> tuple[int, int]:
    """Deterministically pick a legal move: index (salt) into the sorted legal
    set. Same salt sequence -> same game regardless of seeding, so live-style
    and seeded games can be driven down identical move lines."""
    legals = _legal_qr(state)
    return legals[salt % len(legals)]


def _full_payload(state, q: int, r: int) -> dict:
    """A minimal FULL-search on_move payload for move (q, r).

    Carries a one-hot visit policy on the chosen action and a matching prior, so
    _policy_surprise_kl is well defined; root_value is a deterministic function
    of the action so rows differ per move. No gumbel keys (visit target path)."""
    aid = pack_action_id(q, r)
    ids = np.asarray([aid], dtype=np.uint32)
    weights = np.asarray([1.0], dtype=np.float32)
    qs = np.asarray([0.25], dtype=np.float32)
    # Prior slightly off the visit so surprise is non-zero and finite.
    prior_ids = np.asarray([aid], dtype=np.uint32)
    prior_weights = np.asarray([0.5], dtype=np.float32)
    root_value = float(((q * 7 + r * 3) % 11) - 5) / 10.0
    return {
        "action_id": int(aid),
        "pcr_full": True,
        "policy_init": False,
        "visit_policy_action_ids_bytes": ids.tobytes(),
        "visit_policy_weights_bytes": weights.tobytes(),
        "visit_policy_q_bytes": qs.tobytes(),
        "root_prior_policy_action_ids_bytes": prior_ids.tobytes(),
        "root_prior_policy_weights_bytes": prior_weights.tobytes(),
        "root_value": root_value,
    }


def _make_driver(tmp_path, *, seeds=None, fraction=0.0, base=1234, epoch=7,
                 games_target=1, max_plies=64) -> ContinuousDriver:
    return ContinuousDriver(
        epoch=epoch, games_target=games_target, max_plies=max_plies,
        out_dir=tmp_path, diag_dir=None, active_limit=games_target,
        blunder_seeds=list(seeds or []), blunder_seed_fraction=fraction,
        blunder_base_seed=base,
    )


def _drive_game(driver: ContinuousDriver, tape, salts) -> list:
    """Feed the tape's game a fixed move line (by salt), capturing pending
    samples. Returns the FINALIZED rows (same finalize the writer runs), stopping
    at terminal or when the salt list is exhausted. Uses the driver's own
    __call__ so records/ply/labels are produced exactly as in production."""
    key = tape.key
    winner = None
    truncated = False
    for salt in salts:
        # Pick a legal move from the CURRENT engine state.
        q, r = _pick_move(driver.games[key].state, salt)
        payload = _full_payload(driver.games[key].state, q, r)
        # Detect terminal by pre-checking: apply is inside __call__, so we look
        # at the return value.
        ret = driver(key, payload)
        if key not in driver.games:  # game finished inside __call__
            # Recover winner/truncated from the queued writer item.
            item = driver._write_queue.get_nowait()
            _t, winner, truncated = item
            tape = _t
            break
    else:
        # Line exhausted without terminal: finalize the in-flight tape as-is.
        tape = driver.games[key]
    finalized = finalize_game_samples(
        tape.pending, winner, driver.horizons,
        truncated=truncated, mask_opp_from_fast=True,
    )
    return finalized


# ---------------------------------------------------------------------------
# 1. GOLDEN EQUIVALENCE (the load-bearing test).
# ---------------------------------------------------------------------------

def _fields_of(sample) -> dict:
    """The label-bearing fields that a corrupt seeded tape would silently break."""
    return {
        "turn_index": sample.turn_index,
        "current_player": sample.current_player,
        "phase": sample.phase,
        "records": sample.records,
        "first_stone": sample.first_stone,
        "own_hot": sample.own_hot,
        "opp_hot": sample.opp_hot,
        "own_win": sample.own_win,
        "opp_win": sample.opp_win,
    }


@pytest.mark.parametrize("prefix_len", [3, 4, 5, 6])
def test_golden_equivalence_next_row(tmp_path, prefix_len):
    """Seeding a prefix then emitting the NEXT row == driving that same prefix
    live then emitting the next row: identical label fields."""
    # Move line is deterministic in the salts, so both games follow it.
    salts = list(range(12))

    # (1) LIVE-STYLE: drive prefix_len moves through the normal path, then one
    # more decision -> capture that row.
    live = _make_driver(tmp_path / "live")
    (ltape,) = live.start_games(1)
    live_rows = _drive_game(live, ltape, salts[: prefix_len + 1])
    # The row emitted AT ply == prefix_len is the "next row" after the prefix.
    live_next = next(s for s in live_rows if s.turn_index == prefix_len)

    # Build the seed prefix by replaying the SAME move line on a scratch engine.
    scratch = api.new_game()
    prefix = []
    for salt in salts[:prefix_len]:
        q, r = _pick_move(scratch, salt)
        prefix.append((q, r))
        api.apply_action(scratch, PlacementAction(AxialCoord(q=q, r=r)))

    # (2) SEEDED: seed with that prefix, apply ZERO further moves, then emit the
    # next row (the decision at ply == prefix_len).
    seed = bs.BlunderSeed(
        move_prefix=tuple(prefix), seed_ply=prefix_len, surprise=1.0,
        source_epoch=0, source_game_key=0, source_row=0,
    )
    seeded = _make_driver(tmp_path / "seeded")
    # Build a tape via start_games (no auto-seed since fraction=0), then seed it.
    (stape,) = seeded.start_games(1)
    seed_game_tape(stape, seed.move_prefix)
    assert stape.seed_ply == prefix_len
    assert stape.ply == prefix_len
    seeded_rows = _drive_game(seeded, stape, salts[prefix_len : prefix_len + 1])
    seeded_next = next(s for s in seeded_rows if s.turn_index == prefix_len)

    assert _fields_of(seeded_next) == _fields_of(live_next)


def test_golden_equivalence_further_play_bit_equal(tmp_path):
    """A seeded game played several plies further emits rows bit-equal (through
    the shard writer path) to the live-style equivalent, for every shared ply."""
    prefix_len = 4
    salts = list(range(16))

    live = _make_driver(tmp_path / "live2")
    (ltape,) = live.start_games(1)
    live_rows = _drive_game(live, ltape, salts)

    scratch = api.new_game()
    prefix = []
    for salt in salts[:prefix_len]:
        q, r = _pick_move(scratch, salt)
        prefix.append((q, r))
        api.apply_action(scratch, PlacementAction(AxialCoord(q=q, r=r)))

    seeded = _make_driver(tmp_path / "seeded2")
    (stape,) = seeded.start_games(1)
    seed_game_tape(stape, tuple(prefix))
    # Seeded game plays the SAME continuation from ply==prefix_len onward.
    seeded_rows = _drive_game(seeded, stape, salts[prefix_len:])

    # Compare the full rows that both games emit at the same ply, THROUGH the
    # writer serialization (bit-equal features), for plies both cover.
    live_by_ply = {s.turn_index: s for s in live_rows if s.metadata.get("pcr_full")}
    seeded_by_ply = {s.turn_index: s for s in seeded_rows if s.metadata.get("pcr_full")}
    shared = sorted(set(live_by_ply) & set(seeded_by_ply))
    assert shared, "no shared full-row plies to compare"

    for ply in shared:
        lp = tmp_path / f"live_{ply}.npz"
        sp = tmp_path / f"seed_{ply}.npz"
        write_compact_shard(lp, [live_by_ply[ply]], sidecar={})
        write_compact_shard(sp, [seeded_by_ply[ply]], sidecar={})
        lrow = read_compact_shard(lp)[0]
        srow = read_compact_shard(sp)[0]
        assert _fields_of(lrow) == _fields_of(srow), f"row mismatch at ply {ply}"
        assert lrow.policy == srow.policy
        assert lrow.value == srow.value


# ---------------------------------------------------------------------------
# 5. PARITY / PLAYER CONSISTENCY (odd + even seed lengths).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix_len", [1, 2, 3, 4, 5, 6, 7, 8])
def test_seed_player_phase_matches_engine(tmp_path, prefix_len):
    """After seeding an odd- or even-length prefix, the NEXT decision's
    current_player and phase (pure functions of ply) match the hexo 1-2-2 turn
    structure the engine actually reproduces when replaying the same moves."""
    salts = list(range(prefix_len + 1))
    scratch = api.new_game()
    prefix = []
    for salt in salts[:prefix_len]:
        q, r = _pick_move(scratch, salt)
        prefix.append((q, r))
        api.apply_action(scratch, PlacementAction(AxialCoord(q=q, r=r)))

    seeded = _make_driver(tmp_path)
    (tape,) = seeded.start_games(1)
    seed_game_tape(tape, tuple(prefix))
    assert tape.ply == prefix_len
    # The next decision is at ply == prefix_len; its labels come from the pure
    # functions, which encode the 1-2-2 structure.
    expected_player = record_player(prefix_len)
    expected_phase = record_phase(prefix_len)
    rows = _drive_game(seeded, tape, [salts[prefix_len]])
    nxt = next(s for s in rows if s.turn_index == prefix_len)
    assert nxt.current_player == expected_player
    assert nxt.phase == expected_phase
    # SecondStone rows must carry a first_stone (the previous placement).
    if expected_phase == "SecondStone":
        assert nxt.first_stone == (prefix[-1][0], prefix[-1][1])
    else:
        assert nxt.first_stone is None


# ---------------------------------------------------------------------------
# 2. MINER: determinism, max_ply / quantile, malformed-shard handling.
# ---------------------------------------------------------------------------

def _write_fake_shard(path, rows):
    """Write a shard from (turn_index, surprise, prefix, is_full) rows using real
    records so the miner's history reconstruction is exercised. ``is_full`` sets
    the row's pcr_full metadata and whether it carries a policy (fast rows carry
    none); the main_9 shard schema has no policy_valid column."""
    samples = []
    for (ti, surprise, prefix, pv) in rows:
        records = tuple(
            (q, r, record_player(k), k + 1) for k, (q, r) in enumerate(prefix)
        )
        meta = {"pcr_full": bool(pv)}
        from hexfield.samples import HexfieldSampleData
        samples.append(
            HexfieldSampleData(
                game_id="", turn_index=ti, current_player=record_player(ti),
                phase=record_phase(ti), records=records, first_stone=None,
                own_hot=(), opp_hot=(), own_win=(), opp_win=(),
                policy=((pack_action_id(0, 0), 1.0),) if pv else (),
                policy_surprise=float(surprise), metadata=meta,
            )
        )
    write_compact_shard(path, samples, sidecar={})


def _real_prefix(n):
    """n legal (q,r) moves from a fresh game."""
    st = api.new_game()
    out = []
    for salt in range(n):
        legals = sorted(unpack_action_id(int(a)) for a in api.legal_action_ids(st))
        q, r = legals[salt % len(legals)]
        out.append((q, r))
        api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
    return out


def test_miner_deterministic_and_thresholds(tmp_path):
    samples_dir = tmp_path / "samples"
    ep = samples_dir / "epoch_000005"
    ep.mkdir(parents=True)
    # Row surprises: two low, two high; one high row is DEEP (ply>max_ply).
    p3 = _real_prefix(3)
    p4 = _real_prefix(4)
    p5 = _real_prefix(5)
    p50 = _real_prefix(50)
    _write_fake_shard(ep / "game_5000000.npz", [
        (3, 0.1, p3, True),
        (4, 0.2, p4, True),
        (5, 9.0, p5, True),     # high surprise, shallow -> candidate
        (50, 9.5, p50, True),   # high surprise but ply>max_ply -> excluded
    ])
    out1 = bs.mine_blunder_seeds(
        samples_dir, current_epoch=10, recent_epochs=5, max_ply=40,
        surprise_quantile=0.9,
    )
    out2 = bs.mine_blunder_seeds(
        samples_dir, current_epoch=10, recent_epochs=5, max_ply=40,
        surprise_quantile=0.9,
    )
    # Deterministic.
    assert [s.key for s in out1] == [s.key for s in out2]
    # Only the shallow high-surprise row clears q90 AND max_ply.
    assert len(out1) == 1
    seed = out1[0]
    assert seed.seed_ply == 5
    assert seed.move_prefix == tuple(p5)  # ordered prefix recovered
    # max_ply excludes the ply-50 row even though it is the highest surprise.
    assert all(s.seed_ply <= 40 for s in out1)


def test_miner_respects_recent_window(tmp_path):
    # main_9: fast (PCR value-only) rows are excluded at the SELF-PLAY WRITER, so
    # a real shard on disk contains only FULL rows -- the miner never sees a fast
    # row (the policy_valid gate was removed with the column). This shard mirrors
    # that: every written row is full. The out-of-window epoch is still ignored.
    samples_dir = tmp_path / "samples"
    # Epoch too old (outside recent window) -> ignored.
    old = samples_dir / "epoch_000001"
    old.mkdir(parents=True)
    _write_fake_shard(old / "game_1000000.npz", [(3, 9.0, _real_prefix(3), True)])
    # Recent epoch: full rows only (as the main_9 writer produces).
    rec = samples_dir / "epoch_000008"
    rec.mkdir(parents=True)
    _write_fake_shard(rec / "game_8000000.npz", [
        (4, 8.0, _real_prefix(4), True),
    ])
    seeds = bs.mine_blunder_seeds(
        samples_dir, current_epoch=9, recent_epochs=3, max_ply=40,
        surprise_quantile=0.0,  # accept all qualifying full rows
    )
    # Only the recent full row survives (the old epoch is out of the window).
    assert len(seeds) == 1
    assert seeds[0].source_epoch == 8
    assert seeds[0].seed_ply == 4


def test_miner_malformed_shard_skipped(tmp_path):
    samples_dir = tmp_path / "samples"
    ep = samples_dir / "epoch_000005"
    ep.mkdir(parents=True)
    _write_fake_shard(ep / "game_5000000.npz", [(4, 8.0, _real_prefix(4), True)])
    # A malformed npz (not a real shard) and a truncated file.
    (ep / "game_5000001.npz").write_bytes(b"not an npz")
    np.savez(ep / "game_5000002.npz", junk=np.arange(3))  # valid npz, wrong schema
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seeds = bs.mine_blunder_seeds(
            samples_dir, current_epoch=9, recent_epochs=5, max_ply=40,
            surprise_quantile=0.0,
        )
    # The good shard's row is mined; the two bad shards are skipped with warnings.
    assert len(seeds) == 1
    assert any("skipping" in str(w.message) for w in caught)


def test_miner_no_data_returns_empty(tmp_path):
    assert bs.mine_blunder_seeds(
        tmp_path / "does_not_exist", current_epoch=5, recent_epochs=5,
        max_ply=40, surprise_quantile=0.9,
    ) == []
    empty = tmp_path / "samples_empty"
    empty.mkdir()
    assert bs.mine_blunder_seeds(
        empty, current_epoch=5, recent_epochs=5, max_ply=40,
        surprise_quantile=0.9,
    ) == []


# ---------------------------------------------------------------------------
# 3. FRACTION=0.0 BIT-IDENTITY.
# ---------------------------------------------------------------------------

def test_fraction_zero_no_seeding_and_identical_states(tmp_path):
    """With fraction=0.0 (default), start_games seeds nothing and produces the
    SAME game keys/states as a driver constructed with no seed pool at all — the
    seeding RNG is never drawn, so no existing stream is perturbed."""
    seeds = [
        bs.BlunderSeed(
            move_prefix=tuple(_real_prefix(4)), seed_ply=4, surprise=5.0,
            source_epoch=1, source_game_key=1, source_row=0,
        )
    ]
    # Driver A: has a pool but fraction=0.0.
    a = _make_driver(tmp_path / "a", seeds=seeds, fraction=0.0, base=999)
    a_tapes = a.start_games(8)
    # Driver B: no pool at all (pre-feature behavior).
    b = _make_driver(tmp_path / "b", seeds=None, fraction=0.0, base=999)
    b_tapes = b.start_games(8)

    assert [t.key for t in a_tapes] == [t.key for t in b_tapes]
    assert all(t.seed_ply == 0 for t in a_tapes)
    assert all(t.ply == 0 for t in a_tapes)
    assert all(len(t.records) == 0 for t in a_tapes)
    assert a.games_seeded == 0 and b.games_seeded == 0


def test_fraction_zero_stats_have_null_seed_fields(tmp_path):
    d = _make_driver(tmp_path, fraction=0.0)
    d.start_games(4)
    stats = d.stats()
    assert stats["games_seeded"] == 0
    assert stats["seed_ply_mean"] is None
    assert stats["unique_openings_seeded"] == 0


# ---------------------------------------------------------------------------
# 4. SEEDED-GAME TELEMETRY + UNIQUE-OPENINGS STRATIFICATION.
# ---------------------------------------------------------------------------

def test_seeding_telemetry_and_opening_stratification(tmp_path):
    """A pool + fraction=1.0 seeds every game; telemetry counters populate and
    seeded openings are counted separately from the diversity tripwire."""
    prefix = _real_prefix(4)
    seeds = [
        bs.BlunderSeed(
            move_prefix=tuple(prefix), seed_ply=4, surprise=5.0,
            source_epoch=1, source_game_key=1, source_row=0,
        )
    ]
    d = _make_driver(tmp_path, seeds=seeds, fraction=1.0, base=42, games_target=6)
    tapes = d.start_games(6)
    # fraction=1.0 -> every game seeded.
    assert d.games_seeded == 6
    assert all(t.seed_ply == 4 for t in tapes)
    assert d.seed_plies == [4] * 6

    # Play each seeded game one full decision then finish it so _finish tallies
    # openings. We drive to terminal-ish by exhausting a short salt line.
    for t in list(tapes):
        # finish the game as truncated by pushing ply to max quickly is heavy;
        # instead invoke _finish directly with a synthetic winner to exercise the
        # opening stratification path.
        d._finish(t, winner=0, truncated=False)
        d._write_queue.get_nowait()  # drain (no writer thread running)

    stats = d.stats()
    assert stats["games_seeded"] == 6
    assert stats["seed_ply_mean"] == 4.0
    # Seeded games share the SAME stored opening -> 1 distinct seeded opening,
    # and ZERO self-generated openings feed the tripwire.
    assert stats["unique_openings_seeded"] == 1
    assert stats["unique_openings"]["10"] == 0
    assert stats["unique_openings"]["16"] == 0
    assert stats["unique_openings"]["20"] == 0


def test_seeded_rows_carry_metadata_flag(tmp_path):
    """Emitted rows of a seeded game carry metadata['seeded']=True + seed_ply."""
    prefix = _real_prefix(4)
    seeds = [
        bs.BlunderSeed(
            move_prefix=tuple(prefix), seed_ply=4, surprise=5.0,
            source_epoch=1, source_game_key=1, source_row=0,
        )
    ]
    d = _make_driver(tmp_path, seeds=seeds, fraction=1.0, base=7)
    (tape,) = d.start_games(1)
    assert tape.seed_ply == 4
    rows = _drive_game(d, tape, list(range(4, 8)))
    full_rows = [s for s in rows if s.metadata.get("pcr_full")]
    assert full_rows
    for s in full_rows:
        assert s.metadata.get("seeded") is True
        assert s.metadata.get("seed_ply") == 4


def test_max_plies_applies_to_total_ply(tmp_path):
    """A seeded game's truncation counts TOTAL ply (seed_ply + played): with
    max_plies just above seed_ply, the game truncates after few played moves."""
    prefix = _real_prefix(6)
    seeds = [
        bs.BlunderSeed(
            move_prefix=tuple(prefix), seed_ply=6, surprise=5.0,
            source_epoch=1, source_game_key=1, source_row=0,
        )
    ]
    d = _make_driver(tmp_path, seeds=seeds, fraction=1.0, base=3, max_plies=8)
    (tape,) = d.start_games(1)
    assert tape.ply == 6
    # Drive moves; the game must truncate once tape.ply reaches 8 (2 played).
    salts = list(range(6, 20))
    _drive_game(d, tape, salts)
    # After truncation the tape left driver.games; recover from finished counters.
    assert d.games_truncated == 1
    # Game length recorded == max_plies (total ply), proving seed_ply counted.
    assert d.game_lengths[0] == 8
