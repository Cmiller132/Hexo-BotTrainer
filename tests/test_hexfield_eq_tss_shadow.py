"""hexfield_eq TSS Stage-0 shadow tests (docs/PLAN_TSS_DEEPENING.md §10 Stage 0).

Four gates:

  1. λ¹ FIXTURES — engine-built positions with known threat analysis, driven
     through the new ``hexfield_eq_threat_analysis`` probe (the shared
     ``analysis_pydict`` builder): quiet, forced-defense (min_hitting_set == B,
     the Lever-0 boundary), win-now (verdict +1), and forced-loss (verdict −1).
  2. SHADOW PAYLOAD — a lockstep search on the fixtures carries the λ¹ class
     column + proof scalar + the diagnostics tss block, and play respects the
     (pre-existing) tactical guard: the played move is never a proven loss, and
     is a proven win when one exists.
  3. SHARD v5 ROUND-TRIP — policy_class / tss_proof / target_regime survive
     write→read; a v4-style shard (columns stripped) reads back with empty/zero
     defaults.
  4. STAGE-0 DIGEST — a fixed-seed ``run_continuous`` self-play run over a
     deterministic stub evaluator, digesting the pre-TSS-v2 payload surface
     (played actions + recorded targets + root values). The stored golden was
     generated on the PARENT commit (pre-refactor build), so a digest match IS
     the bit-identical differential; it stays as the regression anchor for
     every later increment that claims "flag-off identical".

Runs in the hexgt-build venv (PYTHONPATH=packages/hexfield_eq/python). CPU-only,
torch-free (the evaluator is a numpy stub).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield_eq.geometry import unpack_action_id
from hexfield_eq.samples import STV_HORIZONS, HexfieldSampleData
from hexfield_eq.shards import read_compact_shard, write_compact_shard

try:
    from hexfield_eq import _rust
except ImportError:  # pragma: no cover
    _rust = None

needs_rust = pytest.mark.skipif(
    _rust is None, reason="hexfield_eq._rust not built (see the Phase-1 build gate)"
)

GOLDEN_PATH = Path(__file__).parent / "data" / "hexfield_eq_tss_stage0_golden.json"


# --- fixture positions (scripted placements; api enforces legality) ----------


def _play(state, coords):
    for q, r in coords:
        result = api.apply_action(state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
        assert result is not None
    return state


def quiet_state():
    """Opening + one quiet turn: no ≥4 window anywhere."""
    state = api.new_game()
    return _play(state, [(0, 0), (0, 8), (2, 7)])


def forced_defense_state():
    """p0 owns a 5-in-line (q=0..4, r=0); p1 to move at FirstStone B=2.

    The 5-line's window family needs exactly 2 hitting cells (e.g. (-1,0) and
    (5,0)), so min_hitting_set == B == 2: the fully-forced Lever-0 boundary.
    p1's stones ride the (2,-1) direction — not a window axis — so p1 has no
    threats of its own.
    """
    state = api.new_game()
    return _play(
        state,
        [
            (0, 0),                # p0 opening
            (0, 8), (2, 7),        # p1
            (1, 0), (2, 0),        # p0
            (4, 6), (6, 5),        # p1
            (3, 0), (4, 0),        # p0 → five in line
        ],
    )


def win_now_state():
    """forced_defense_state + a p1 turn that ignores the threat: p0 to move
    with a live count-5 → own_win_now, verdict +1."""
    return _play(forced_defense_state(), [(8, 4), (10, 3)])


def forced_loss_state():
    """p0 owns TWO disjoint 4-in-lines (r=0 and r=4, q=0..3 each); p1 to move.

    Each 4-line's window family needs 2 hitting cells, so min_hitting_set = 4
    > B = 2 → a proven one-turn forced loss (verdict −1) for p1. All p1 stones
    stay on the non-axis (2,-1) line; p0's final filler stone (16,0) shares no
    6-window with either line.
    """
    state = api.new_game()
    return _play(
        state,
        [
            (0, 0),                 # p0 opening
            (0, 8), (2, 7),         # p1
            (1, 0), (2, 0),         # p0
            (4, 6), (6, 5),         # p1
            (3, 0), (0, 4),         # p0 → line A complete (q=0..3, r=0); line B starts
            (8, 4), (10, 3),        # p1
            (1, 4), (2, 4),         # p0
            (12, 2), (14, 1),       # p1
            (3, 4), (16, 0),        # p0 → line B complete; harmless filler
        ],
    )


# --- 1. λ¹ fixtures through the probe ----------------------------------------


@needs_rust
def test_threat_analysis_quiet():
    d = _rust.hexfield_eq_threat_analysis(quiet_state())
    assert d["own_win_now"] is False
    assert d["opp_threat_count"] == 0
    assert d["min_hitting_set"] == 0
    assert d["forced_loss"] is False
    assert d["verdict"] is None
    assert d["tactical_cells"] == []


@needs_rust
def test_threat_analysis_forced_defense():
    d = _rust.hexfield_eq_threat_analysis(forced_defense_state())
    assert d["b"] == 2
    assert d["own_win_now"] is False
    # 5-line window family: [-2..3](4), [-1..4](5), [0..5](5), [1..6](4).
    assert d["opp_threat_count"] == 4
    assert d["min_hitting_set"] == 2  # the k == B fully-forced boundary
    assert d["forced_loss"] is False
    assert d["verdict"] is None
    cells = {tuple(c) for c in d["tactical_cells"]}
    assert {(-2, 0), (-1, 0), (5, 0), (6, 0)} == cells


@needs_rust
def test_threat_analysis_win_now():
    d = _rust.hexfield_eq_threat_analysis(win_now_state())
    assert d["b"] == 2
    assert d["own_win_now"] is True
    assert d["verdict"] == 1.0
    cells = {tuple(c) for c in d["tactical_cells"]}
    # Own winning completions lead the tactical set.
    assert {(-1, 0), (5, 0)} <= cells


@needs_rust
def test_threat_analysis_forced_loss():
    d = _rust.hexfield_eq_threat_analysis(forced_loss_state())
    assert d["b"] == 2
    assert d["own_win_now"] is False
    assert d["min_hitting_set"] == -1  # None: needs 4 > B=2
    assert d["forced_loss"] is True
    assert d["verdict"] == -1.0


# --- 2. shadow payload + guard-consistent play --------------------------------


class StubEvaluator:
    """Deterministic torch-free evaluator. Priors/logits are a fixed pattern
    over each row's legal prefix; the value is a pure function of the row's
    legal count. Bitwise-reproducible across runs and builds."""

    def __call__(self, payload: dict) -> dict:
        legal = np.frombuffer(bytes(payload["legal_counts"]), dtype=np.int32)
        total = int(legal.sum())
        logits = np.empty(total, dtype=np.float32)
        pos = 0
        for length in legal.tolist():
            k = np.arange(int(length), dtype=np.float64)
            row = ((k * 2654435761.0) % 97.0) / 97.0 * 2.0 - 1.0
            logits[pos : pos + int(length)] = row.astype(np.float32)
            pos += int(length)
        reply = {
            "values_bytes": np.asarray(
                [np.tanh(((int(n) % 7) - 3) * 0.1) for n in legal.tolist()],
                dtype=np.float32,
            ).tobytes(),
            "priors_bytes": np.exp(logits, dtype=np.float32).tobytes(),
        }
        if payload.get("request_moves_left"):
            reply["moves_left_bytes"] = np.full(len(legal), 30.0, dtype=np.float32).tobytes()
        if payload.get("request_logits"):
            reply["priors_logits_bytes"] = logits.tobytes()
        return reply


def _lockstep_payload(state, *, seed=42, visits=48):
    session = _rust.HexfieldMctsSession(max_states=4096)
    payloads = session.search(
        [0],
        (state,),
        visits,
        1.5,
        1.0,
        seed,
        StubEvaluator(),
    )
    assert len(payloads) == 1
    return payloads[0]


def _classes(payload) -> dict[int, int]:
    if "tss_class_action_ids_bytes" not in payload:
        return {}
    ids = np.frombuffer(bytes(payload["tss_class_action_ids_bytes"]), dtype=np.uint32)
    vals = np.frombuffer(bytes(payload["tss_class_bytes"]), dtype=np.int8)
    return {int(a): int(v) for a, v in zip(ids, vals)}


@needs_rust
def test_payload_class_column_forced_defense():
    payload = _lockstep_payload(forced_defense_state())
    tss = payload["diagnostics"]["tss"]
    assert tss["b"] == 2
    assert tss["k"] == 2
    assert tss["opp_threats"] == 4
    assert payload["tss_proof"] == 0
    classes = _classes(payload)
    assert classes, "threatful root must export a class map"
    assert set(classes.values()) <= {-1, 0, 1}
    # No proven win exists here, and every non-hitting move is a proven loss:
    # the (pre-existing) guard must never play a class = -1 move.
    played = int(payload["action_id"])
    assert classes.get(played, 0) != -1
    # The played move defends: it is one of the four hitting cells.
    q, r = unpack_action_id(played)
    assert (q, r) in {(-2, 0), (-1, 0), (5, 0), (6, 0)}


@needs_rust
def test_payload_win_now_plays_the_win():
    payload = _lockstep_payload(win_now_state())
    assert payload["tss_proof"] == 1
    classes = _classes(payload)
    played = int(payload["action_id"])
    # At B=2 with a live count-5, ANY safe first stone is a proven win (the
    # second placement completes the six) — so class 1 covers far more than
    # the completion cells. The guard must still only play a proven winner,
    # and the direct completions must be among the proven winners.
    assert classes.get(played, 0) == 1, "a proven winning move exists; play one"
    from hexfield_eq.geometry import pack_action_id

    for cell in ((-1, 0), (5, 0)):
        assert classes.get(pack_action_id(*cell), 0) == 1


@needs_rust
def test_payload_quiet_root_has_no_class_map():
    payload = _lockstep_payload(quiet_state())
    assert "tss_class_action_ids_bytes" not in payload
    assert payload["tss_proof"] == 0
    tss = payload["diagnostics"]["tss"]
    assert tss["opp_threats"] == 0
    assert tss["root_tactical"] == 0


# --- 3. shard v5 round-trip ----------------------------------------------------


def _sample_row(policy_class=(), tss_proof=0) -> HexfieldSampleData:
    return HexfieldSampleData(
        game_id="g",
        turn_index=3,
        current_player=1,
        phase="FirstStone",
        records=((0, 0, 0, 0), (1, 0, 0, 1), (0, 8, 1, 2)),
        first_stone=None,
        policy=((7, 0.5), (9, 0.25), (11, 0.25)),
        q_policy=((7, 0.1), (9, -0.2), (11, 0.0)),
        policy_surprise=0.3,
        policy_class=policy_class,
        tss_proof=tss_proof,
        metadata={"pcr_full": True},
    )


def test_shard_v5_roundtrip(tmp_path):
    rows = [
        _sample_row(policy_class=((7, 1), (9, -1)), tss_proof=1),
        _sample_row(),
    ]
    path = tmp_path / "game_1.npz"
    assert write_compact_shard(path, rows, short_term_value_horizons=STV_HORIZONS) == 2
    back = read_compact_shard(path)
    assert back[0].policy_class == ((7, 1), (9, -1))
    assert back[0].tss_proof == 1
    assert back[1].policy_class == ()
    assert back[1].tss_proof == 0
    with np.load(path) as data:
        assert int(data["schema_version"]) == 5
        assert int(data["target_regime"]) == 0
        assert data["pol_class"].dtype == np.int8
        assert data["tss_proof"].dtype == np.int8


def test_shard_v4_reads_with_empty_tss(tmp_path):
    rows = [_sample_row(policy_class=((7, 1),), tss_proof=-1)]
    path = tmp_path / "game_2.npz"
    write_compact_shard(path, rows, short_term_value_horizons=STV_HORIZONS)
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    for key in ("pol_class", "tss_proof", "target_regime"):
        arrays.pop(key)
    arrays["schema_version"] = np.asarray(4, dtype=np.int32)
    stripped = tmp_path / "game_2_v4.npz"
    with open(stripped, "wb") as f:
        np.savez_compressed(f, **arrays)
    back = read_compact_shard(stripped)
    assert back[0].policy_class == ()
    assert back[0].tss_proof == 0


# --- 4. fixed-seed Stage-0 digest (the bit-identity differential) --------------

# The pre-TSS-v2 payload surface: played move + every recorded-target column +
# root value + telemetry scalars that existed before Stage 0. New tss_* keys are
# deliberately EXCLUDED so the digest is comparable against pre-Stage-0 builds.
_CORE_KEYS = (
    "action_id",
    "action_selection",
    "lcb_override",
    "early_stopped",
    "play_pruned",
    "play_winner",
    "pcr_full",
    "policy_init",
    "visit_policy_action_ids_bytes",
    "visit_policy_weights_bytes",
    "visit_policy_q_bytes",
    "visit_policy_count",
    "root_prior_policy_action_ids_bytes",
    "root_prior_policy_weights_bytes",
    "root_prior_policy_count",
    "gumbel_policy_action_ids_bytes",
    "gumbel_policy_weights_bytes",
    "gumbel_policy_count",
    "root_prior_logits_bytes",
    "root_value",
    "visits",
)


class DigestHarness:
    """Minimal on_move driver: digests the core payload surface, applies the
    played action, ends each game at terminal or after max_plies."""

    def __init__(self, states: dict[int, object], max_plies: int = 60):
        self.states = states
        self.max_plies = max_plies
        self.plies = {key: 0 for key in states}
        self.digest = hashlib.sha256()
        self.moves = 0

    def __call__(self, game_key: int, payload: dict):
        self.moves += 1
        self.digest.update(b"|move|%d|" % int(game_key))
        for key in _CORE_KEYS:
            if key not in payload:
                continue
            value = payload[key]
            self.digest.update(key.encode())
            if isinstance(value, (bytes, bytearray)):
                self.digest.update(bytes(value))
            elif isinstance(value, float):
                self.digest.update(np.float32(value).tobytes())
            else:
                self.digest.update(repr(value).encode())
        state = self.states[game_key]
        q, r = unpack_action_id(int(payload["action_id"]))
        api.apply_action(state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
        self.plies[game_key] += 1
        if api.terminal(state) is not None or self.plies[game_key] >= self.max_plies:
            self.digest.update(b"|end|%d|" % int(game_key))
            return None
        return ("advance", state)


def _run_stage0_digest() -> tuple[str, int]:
    states = {1: api.new_game(), 2: api.new_game(), 3: api.new_game()}
    harness = DigestHarness(states)
    session = _rust.HexfieldMctsSession(max_states=8192)
    session.run_continuous(
        list(states.keys()),
        tuple(states.values()),
        evaluator=StubEvaluator(),
        on_move=harness,
        visits=24,
        c_puct=1.5,
        base_seed=987_654_321,
        virtual_batch_size=8,
        flush_target=16,
        active_root_limit=4,
        temperature_by_ply=[1.0, 0.9, 0.8],
        tss_enabled=True,
    )
    return harness.digest.hexdigest(), harness.moves


@needs_rust
def test_mini_selfplay_driver_end_to_end(tmp_path):
    """The REAL ContinuousDriver over the stub evaluator: covers the Stage-0
    telemetry accumulation in __call__, the writer-thread proof/disagreement
    counting, the stats() tss block, v5 shard writes through the production
    writer, and the crash-resume merge — the exact code paths a main_3 deploy
    exercises."""
    from hexfield_eq.selfplay import ContinuousDriver, _merge_epoch_diag

    driver = ContinuousDriver(
        epoch=1, games_target=2, max_plies=40, out_dir=tmp_path, active_limit=2
    )
    tapes = driver.start_games(2)
    driver._start_writer()
    session = _rust.HexfieldMctsSession(max_states=8192)
    session.run_continuous(
        [t.key for t in tapes],
        tuple(t.state for t in tapes),
        evaluator=StubEvaluator(),
        on_move=driver,
        visits=24,
        c_puct=1.5,
        base_seed=777,
        virtual_batch_size=8,
        flush_target=16,
        active_root_limit=2,
        temperature_by_ply=[1.0],
        tss_enabled=True,
    )
    driver._stop_writer()
    stats = driver.stats()
    tss = stats["tss"]
    assert tss["moves"] == driver.decisions > 0
    assert 0.0 <= (tss["threat_move_fraction"] or 0.0) <= 1.0
    assert tss["proof_disagreements"] <= tss["proof_rows"]
    shards = sorted(tmp_path.glob("game_*.npz"))
    assert shards, "the production writer must have written v5 shards"
    rows = read_compact_shard(shards[0])
    assert rows
    with np.load(shards[0]) as data:
        assert int(data["schema_version"]) == 5
    # Crash-resume merge: integer counters sum, rates recompute from the sums.
    merged = _merge_epoch_diag([stats, stats])
    assert merged["tss"]["moves"] == 2 * tss["moves"]
    if tss["injection_fire_rate"] is not None:
        assert merged["tss"]["injection_fire_rate"] == pytest.approx(
            tss["injection_fire_rate"]
        )


@needs_rust
def test_stage0_digest_matches_golden():
    digest, moves = _run_stage0_digest()
    assert moves > 30, f"self-play run too short to be meaningful: {moves} moves"
    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps({"digest": digest, "moves": moves}, indent=2), encoding="utf-8"
        )
        pytest.skip(f"golden digest written ({moves} moves); re-run to compare")
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert moves == golden["moves"], "move count drifted from the golden run"
    assert digest == golden["digest"], (
        "Stage-0 payload surface drifted from the golden build — the refactor "
        "is no longer bit-identical (or an intentional change needs a new golden)"
    )
