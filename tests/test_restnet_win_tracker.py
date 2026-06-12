"""Frozen-win override foundations for dense_cnn_restnet (main_4 prep).

Covers: the IncrementalWinTracker (standing win-in-1 cells, incremental 6-cell
windows) against an independent brute-force scan and the real engine, the
frozen-win override decision function (in-crop vs out-of-crop standing wins,
min-packed-id selection, clone-verify fall-through), and a golden replay of the
longest main_3 ep9 marathon game pinned to the engine-verified analysis
artifacts (tests/data/main3_frozen_win_{structure,verify}.json, 47/47 method).
"""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine", "hexo_utils", "hexo_runner", "hexo_train"):
    path = ROOT / "packages" / package / "python"
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
_RESTNET = ROOT / "packages" / "dense_cnn_restnet" / "python"
if str(_RESTNET) not in sys.path:
    sys.path.insert(0, str(_RESTNET))

torch = pytest.importorskip("torch")
engine = importlib.import_module("hexo_engine")
engine_types = importlib.import_module("hexo_engine.types")
win_tracker_module = importlib.import_module("dense_cnn_restnet.win_tracker")
geometry = importlib.import_module("dense_cnn_restnet.geometry")
constants = importlib.import_module("dense_cnn_restnet.constants")
selfplay = importlib.import_module("dense_cnn_restnet.selfplay")

IncrementalWinTracker = win_tracker_module.IncrementalWinTracker
AXES = win_tracker_module.AXES
HALF = constants.BOARD_SIZE // 2

RUN_DIR = Path("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3")
STRUCTURE_JSON = ROOT / "tests" / "data" / "main3_frozen_win_structure.json"
VERIFY_JSON = ROOT / "tests" / "data" / "main3_frozen_win_verify.json"


def _pack(q: int, r: int) -> int:
    return int(engine_types.pack_coord_id(engine_types.AxialCoord(q, r)))


def _unpack(action_id: int) -> tuple[int, int]:
    coord = engine_types.unpack_coord_id(int(action_id))
    return int(coord.q), int(coord.r)


def _mover(ply: int) -> int:
    # Opening ply 0 -> P0; then two-stone Connect6 turns: (1,2)->P1, (3,4)->P0...
    return 0 if ply == 0 else (1 if ((ply - 1) // 2) % 2 == 0 else 0)


def _brute_force_standing(stones: dict[tuple[int, int], int]) -> tuple[set, set]:
    """Independent standing-win scan: enumerate every 6-cell window touching a
    stone from scratch (no incremental bookkeeping)."""

    standing: tuple[set, set] = (set(), set())
    seen: set[tuple[int, int, int]] = set()
    for (q, r) in stones:
        for axis_index, (vq, vr) in enumerate(AXES):
            for k in range(6):
                key = (axis_index, q - k * vq, r - k * vr)
                if key in seen:
                    continue
                seen.add(key)
                cells = [(key[1] + j * vq, key[2] + j * vr) for j in range(6)]
                owners = [stones.get(cell) for cell in cells]
                empties = [cell for cell, owner in zip(cells, owners) if owner is None]
                if len(empties) != 1:
                    continue
                count0 = sum(1 for owner in owners if owner == 0)
                count1 = sum(1 for owner in owners if owner == 1)
                if count0 == 5 and count1 == 0:
                    standing[0].add(empties[0])
                elif count1 == 5 and count0 == 0:
                    standing[1].add(empties[0])
    return standing


# --------------------------- tracker vs brute force --------------------------- #


def test_tracker_matches_brute_force_on_dense_random_boards() -> None:
    # Dense random fills of a small box: plenty of 4/5-in-line patterns, standing
    # cells appearing AND disappearing (occupied / opponent-poisoned windows).
    rng = random.Random(20260612)
    cells_box = [(q, r) for q in range(-4, 5) for r in range(-4, 5)]
    checked_nonempty = 0
    for _game in range(150):
        order = rng.sample(cells_box, 40)
        tracker = IncrementalWinTracker()
        stones: dict[tuple[int, int], int] = {}
        for ply, cell in enumerate(order):
            owner = _mover(ply)
            tracker.add_stone(cell[0], cell[1], owner)
            stones[cell] = owner
            expected = _brute_force_standing(stones)
            assert tracker.standing_win_cells(0) == expected[0]
            assert tracker.standing_win_cells(1) == expected[1]
            if expected[0] or expected[1]:
                checked_nonempty += 1
    # The comparison must have exercised real standing-win states, not just
    # empty-vs-empty agreement.
    assert checked_nonempty > 100


def test_tracker_matches_brute_force_and_engine_on_random_legal_games() -> None:
    # Engine-legal random games: origin opening, distance-8 legality, real
    # terminality. Where the side to move has a standing cell that is legal,
    # placing it on a cloned engine state must be terminal for that side.
    rng = random.Random(7)
    engine_verified = 0
    for index in range(60):
        state = engine.new_game(seed=1000 + index)
        tracker = IncrementalWinTracker()
        stones: dict[tuple[int, int], int] = {}
        for _ply in range(48):
            if engine.terminal(state) is not None:
                break
            mover_label = str(engine.current_player(state))
            mover_index = 0 if mover_label.endswith("0") else 1
            standing = tracker.standing_win_cells(mover_index)
            assert standing == _brute_force_standing(stones)[mover_index]
            if standing and engine_verified < 25:
                legal = set(int(a) for a in engine.legal_action_ids(state))
                cell = sorted(standing)[0]
                if _pack(*cell) in legal:
                    clone = engine.clone_state(state)
                    engine.apply_action(
                        clone, engine.PlacementAction(engine_types.AxialCoord(cell[0], cell[1]))
                    )
                    terminal = engine.terminal(clone)
                    assert terminal is not None and str(terminal.winner) == mover_label
                    engine_verified += 1
            action_id = int(rng.choice(engine.legal_action_ids(state)))
            engine.apply_action(state, engine.PlacementAction(engine_types.unpack_coord_id(action_id)))
            q, r = _unpack(action_id)
            tracker.add_stone(q, r, mover_label)
            stones[(q, r)] = mover_index


def test_tracker_standing_cells_appear_and_clear_exactly() -> None:
    # Explicit 5-in-row: standing cells are exactly the two completion ends; an
    # opponent stone on one end clears that end only.
    tracker = IncrementalWinTracker()
    for q in range(5):
        tracker.add_stone(q, 0, 0)
        # Opponent stones far away and OFF the three win axes (no interference,
        # no opponent line).
        tracker.add_stone(2 * q, 10 + q, 1)
    assert tracker.standing_win_cells(0) == {(-1, 0), (5, 0)}
    assert tracker.standing_win_cells(1) == set()
    tracker.add_stone(5, 0, 1)  # opponent blocks one end
    assert tracker.standing_win_cells(0) == {(-1, 0)}
    tracker.add_stone(-1, 0, 1)  # ... and the other
    assert tracker.standing_win_cells(0) == set()


# --------------------------- override decision function --------------------------- #


def _state_with_p0_five_in_row() -> object:
    """Engine state where P0 holds (0,0)..(4,0) and it is P0's turn (ply 11).

    Standing win cells for P0: (-1, 0) and (5, 0). P1 stones are scattered and
    harmless. Plies follow the engine's Connect6 parity exactly (validated by
    apply_action succeeding for the intended movers).
    """

    moves = [
        (0, 0),   # ply 0  P0 (forced origin)
        (0, 5),   # ply 1  P1
        (1, 5),   # ply 2  P1
        (1, 0),   # ply 3  P0
        (2, 0),   # ply 4  P0
        (-5, 5),  # ply 5  P1
        (-4, 5),  # ply 6  P1
        (3, 0),   # ply 7  P0
        (4, 0),   # ply 8  P0
        (2, 7),   # ply 9  P1
        (3, 7),   # ply 10 P1
    ]
    state = engine.new_game(seed=1)
    tracker = IncrementalWinTracker()
    for ply, (q, r) in enumerate(moves):
        mover_label = str(engine.current_player(state))
        assert mover_label == f"player{_mover(ply)}"
        engine.apply_action(state, engine.PlacementAction(engine_types.AxialCoord(q, r)))
        tracker.add_stone(q, r, mover_label)
    assert str(engine.current_player(state)) == "player0"
    assert tracker.standing_win_cells(0) == {(-1, 0), (5, 0)}
    return state, tracker


def test_override_skips_in_crop_standing_wins() -> None:
    state, tracker = _state_with_p0_five_in_row()
    # (a) center at the blob: both standing cells in-crop -> never override.
    action_id, failed = selfplay._frozen_win_override_action(
        state, tracker.standing_win_cells(0), center=(0, 0), mover="player0"
    )
    assert action_id is None and failed is False


def test_override_plays_min_packed_id_when_all_out_of_crop() -> None:
    state, tracker = _state_with_p0_five_in_row()
    # (b) center far away: dist((-1,0),(30,0))=31, dist((5,0),(30,0))=25, both
    # > 20 -> override with the min-packed-id cell, verified on a clone.
    action_id, failed = selfplay._frozen_win_override_action(
        state, tracker.standing_win_cells(0), center=(30, 0), mover="player0"
    )
    assert failed is False
    assert action_id == min(_pack(-1, 0), _pack(5, 0)) == _pack(-1, 0)
    # The returned move really ends the game with the mover winning.
    clone = engine.clone_state(state)
    engine.apply_action(clone, engine.PlacementAction(engine_types.unpack_coord_id(action_id)))
    terminal = engine.terminal(clone)
    assert terminal is not None and str(terminal.winner) == "player0"


def test_override_skips_mixed_in_and_out_of_crop() -> None:
    state, tracker = _state_with_p0_five_in_row()
    # (c) center=(25,0): dist((5,0))=20 (in-crop), dist((-1,0))=26 (out) -> mixed
    # -> no override.
    action_id, failed = selfplay._frozen_win_override_action(
        state, tracker.standing_win_cells(0), center=(25, 0), mover="player0"
    )
    assert action_id is None and failed is False


def test_override_no_standing_wins_is_noop() -> None:
    state, _tracker = _state_with_p0_five_in_row()
    # (d) empty standing set -> no override, no failure.
    action_id, failed = selfplay._frozen_win_override_action(
        state, set(), center=(30, 0), mover="player0"
    )
    assert action_id is None and failed is False


def test_override_verify_failure_falls_through() -> None:
    state, _tracker = _state_with_p0_five_in_row()
    # (e) bogus standing cells must NEVER be played: a legal-but-not-winning
    # cell (terminal stays None), an illegal far cell (apply raises), and a
    # winning cell attributed to the wrong mover all report verify failure.
    for cells, center, mover in (
        ({(6, 0)}, (60, 60), "player0"),    # legal, not a win
        ({(50, 50)}, (0, 0), "player0"),    # illegal (outside distance-8 legality)
        ({(-1, 0)}, (60, 60), "player1"),   # real win cell, wrong mover claimed
    ):
        # Each bogus cell is out-of-crop for its center, so the decision reaches
        # the clone-verify step and must fail there.
        action_id, failed = selfplay._frozen_win_override_action(
            state, cells, center=center, mover=mover
        )
        assert action_id is None and failed is True


# --------------------------- golden marathon replay --------------------------- #


@pytest.mark.skipif(
    not RUN_DIR.exists() or not STRUCTURE_JSON.exists() or not VERIFY_JSON.exists(),
    reason="main_3 run dir / engine-verified analysis fixtures not available",
)
def test_golden_marathon_replay_matches_engine_verified_analysis() -> None:
    """Replay the longest completed ep9 game through IncrementalWinTracker and
    pin it to the engine-verified _wf_r4 analysis (the 47/47 method).

    Asserts: (1) the tracker's missed-standing-win plies match
    _wf_r4_structure.json exactly; (2) every missed-win ply has ALL standing win
    cells out-of-crop (the frozen-game signature the override keys on); (3) >=20
    sampled standing-win plies engine-verify (clone, place min-packed-id cell,
    terminal with the mover winning); (4) the _wf_r4_verify.json engine-checked
    cells are present in the tracker's standing sets at those plies.
    """

    records_module = importlib.import_module("hexo_utils.records")
    with records_module.HexoRecordFile.open(RUN_DIR / "selfplay" / "epoch_000009.hxr") as record_file:
        records = [record for record in record_file.iter_records() if record.winner is not None]
    records.sort(key=lambda record: len(record.action_ids))
    record = records[-1]
    coords = [_unpack(action_id) for action_id in record.action_ids]
    assert record.game_id == "epoch-000009-selfplay-000056"
    assert len(coords) == 855

    structure = json.loads(STRUCTURE_JSON.read_text(encoding="utf-8"))
    golden = next(g for g in structure["games"] if g["game_id"] == record.game_id)

    tracker = IncrementalWinTracker()
    stones_so_far: list[tuple[int, int]] = []
    flagged: list[tuple[int, int, list[tuple[int, int]]]] = []
    all_out_of_crop = 0
    for ply, cell in enumerate(coords):
        mover_index = _mover(ply)
        standing = tracker.standing_win_cells(mover_index)
        if standing and cell not in standing:
            center = geometry.crop_center(stones_so_far)
            if all(
                geometry.hex_distance(win_cell, (center.q, center.r)) > HALF
                for win_cell in standing
            ):
                all_out_of_crop += 1
            flagged.append((ply, mover_index, sorted(standing)))
        tracker.add_stone(cell[0], cell[1], mover_index)
        stones_so_far.append(cell)

    # (1) exact match with the engine-verified analyzer output.
    assert len(flagged) == golden["missed_wins"] == 188
    assert [ply for ply, _mv, _cells in flagged[:50]] == golden["missed_win_plies"]

    # (2) every missed standing win in this marathon is fully out-of-crop.
    assert all_out_of_crop == len(flagged)

    # (3) engine-verify >= 20 sampled standing-win plies (mirrors the 47/47
    # method: replay to the ply, clone, place the cell, assert terminal+winner).
    sample_indices = sorted({int(i * (len(flagged) - 1) / 23) for i in range(24)})
    picks = [flagged[i] for i in sample_indices]
    assert len(picks) >= 20
    state = engine.new_game()
    ply_now = 0
    for ply, mover_index, cells in picks:
        while ply_now < ply:
            q, r = coords[ply_now]
            engine.apply_action(state, engine.PlacementAction(engine_types.AxialCoord(q, r)))
            ply_now += 1
        mover_label = str(engine.current_player(state))
        assert mover_label == f"player{mover_index}"
        cell = min(cells, key=lambda c: _pack(*c))
        clone = engine.clone_state(state)
        engine.apply_action(clone, engine.PlacementAction(engine_types.AxialCoord(cell[0], cell[1])))
        terminal = engine.terminal(clone)
        assert terminal is not None and str(terminal.winner) == mover_label

    # (4) the independently engine-verified cells from _wf_r4_verify.json sit in
    # the tracker's standing set at their plies.
    verify = json.loads(VERIFY_JSON.read_text(encoding="utf-8"))
    checks = [c for c in verify["engine_checks"] if c["game_id"] == record.game_id]
    assert checks, "verify artifact must contain checks for the longest game"
    flagged_by_ply = {ply: set(map(tuple, cells)) for ply, _mv, cells in flagged}
    for check in checks:
        assert check["verified_win"] is True
        assert tuple(check["cell"]) in flagged_by_ply[check["ply"]]
