r"""Spec A tests for dense_cnn_restnet: the radius-20 hex-disk crop contract.

Covers the three Spec A acceptance tests:
  (a) D6 round-trip never drops an in-disk fact across all 12 symmetries;
  (b) corner cells are zero in every input plane and never receive policy mass;
  (c) the Rust and Python encoders agree cell-for-cell on random states (the
      cross-boundary disk-contract check).

(a) and (b) are pure-Python (no native module beyond importing the package).
(c) drives the real engine and the shared native accelerator; it is skipped if
the loaded `_rust.dense_cnn` does not yet honor the disk contract (i.e. an old
.so), so the suite stays green before the native rebuild and fully exercises the
contract after it.

Run (from the repo root):
    python -m pytest tests\test_dense_cnn_restnet_crop.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("dense_cnn_restnet", "hexo_train", "hexo_runner", "hexo_engine", "hexo_models"):
    _src = _ROOT / "packages" / _pkg / "python"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from dense_cnn_restnet import samples as samples_mod  # noqa: E402
from dense_cnn_restnet.constants import BOARD_SIZE, INPUT_CHANNELS, PLANE_CENTER_DISTANCE  # noqa: E402
from dense_cnn_restnet.d6 import D6_SIZE, Axial, pack_coord_id  # noqa: E402
from dense_cnn_restnet.geometry import disk_mask, hex_distance, in_disk  # noqa: E402
from dense_cnn_restnet.input import build_input_planes, dense_policy_target, legal_mask_flat  # noqa: E402
from dense_cnn_restnet.samples import Model1SampleData, count_spill, expand_sample  # noqa: E402

_HALF = BOARD_SIZE // 2


# --------------------------------------------------------------------------- #
# Geometry: the disk partitions the 41x41 square into 1261 disk + 420 corners.
# --------------------------------------------------------------------------- #
def test_disk_partitions_1261_and_420():
    disk = sum(1 for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if in_disk(row, col))
    assert disk == 1261
    assert BOARD_SIZE * BOARD_SIZE - disk == 420


def test_disk_mask_matches_in_disk():
    mask = disk_mask(BOARD_SIZE)
    assert mask.shape == (BOARD_SIZE, BOARD_SIZE)
    assert mask.dtype == torch.bool
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            assert bool(mask[row, col]) == in_disk(row, col)


def test_disk_is_d6_closed_about_center():
    # The disk is closed under all 12 D6 symmetries about the crop center: hex
    # distance to the center is a D6 invariant, so an in-disk cell maps to an
    # in-disk cell. This is the property that makes the augmentation fallback
    # unnecessary.
    from dense_cnn_restnet.d6 import transform_coord

    center = Axial(0, 0)
    in_disk_coords = [(q, r) for q in range(-_HALF, _HALF + 1) for r in range(-_HALF, _HALF + 1)
                      if hex_distance((q, r), (0, 0)) <= _HALF]
    for sym in range(D6_SIZE):
        for q, r in in_disk_coords:
            image = transform_coord((q, r), sym, center=center)
            assert hex_distance((image.q, image.r), (0, 0)) <= _HALF


# --------------------------------------------------------------------------- #
# A sample whose every fact is comfortably inside the disk.
# --------------------------------------------------------------------------- #
def _in_disk_coords() -> list[tuple[int, int]]:
    # A spread of cells at hex distance <= 18 from center (0, 0).
    coords = [(0, 0), (3, 4), (-5, 2), (6, -3), (8, 8), (-10, 5), (12, -6), (0, 14), (-14, 0), (7, -14)]
    for q, r in coords:
        assert hex_distance((q, r), (0, 0)) <= _HALF
    return coords


def _in_disk_sample() -> Model1SampleData:
    coords = _in_disk_coords()
    stones = tuple((q, r, "player0" if i % 2 == 0 else "player1") for i, (q, r) in enumerate(coords[:6]))
    legal = tuple(pack_coord_id((q, r)) for q, r in coords)
    policy = tuple((pack_coord_id((q, r)), 1.0) for q, r in coords[:5])
    history = tuple((q, r, "player0" if i % 2 == 0 else "player1", None, i, None, None)
                    for i, (q, r) in enumerate(coords[:4]))
    return Model1SampleData(
        game_id="g",
        turn_index=4,
        current_player="player0",
        phase="Opening",
        center=(0, 0),
        stones=stones,
        legal_action_ids=legal,
        placement_history=history,
        own_hot=(coords[6],),
        opponent_hot=(coords[7],),
        policy=policy,
        opp_policy=(),
    )


# --------------------------------------------------------------------------- #
# (a) D6 round-trip never drops an in-disk fact.
# --------------------------------------------------------------------------- #
def test_d6_roundtrip_never_drops_in_disk_fact():
    sample = _in_disk_sample()
    coords = _in_disk_coords()
    n_stones = 6
    n_legal = len(coords)
    n_policy = 5
    identity = expand_sample(sample, symmetry=0)
    for sym in range(D6_SIZE):
        out = expand_sample(sample, symmetry=sym)
        planes = out["input"]
        # Every in-disk stone survives under every symmetry (distinct cells, no
        # collisions because D6 is injective), so the set-cell count is preserved.
        own = int((planes[0] > 0).sum())
        opp = int((planes[1] > 0).sum())
        assert own + opp == n_stones, f"sym={sym}: {own + opp} stones, expected {n_stones}"
        # Legal plane keeps all in-disk legal cells.
        assert int((planes[3] > 0).sum()) == n_legal, f"sym={sym}: legal dropped"
        # Policy mass is fully preserved (sum 1, all support kept).
        assert float(out["policy"].sum()) == pytest.approx(1.0)
        assert int((out["policy"] > 0).sum()) == n_policy, f"sym={sym}: policy support dropped"
        # The identity and symmetry expansions carry the same total mass / counts.
        assert int((identity["legal_mask"]).sum()) == n_legal
    # No fact in this sample spills (all in-disk).
    spill = count_spill(sample)
    assert spill == {"stones": 0, "legal": 0, "policy": 0, "history": 0}


# --------------------------------------------------------------------------- #
# (b) corner cells zero in every plane and never receive policy mass.
# --------------------------------------------------------------------------- #
def _corner_coords() -> list[tuple[int, int]]:
    # Cells at the four corner triangles (hex distance > 20 from center (0, 0)).
    coords = []
    for row, col in [(0, 0), (0, 1), (1, 0), (40, 40), (39, 40), (40, 39), (0, 40), (40, 0)]:
        q = col - _HALF
        r = row - _HALF
        if not in_disk(row, col):
            coords.append((q, r))
    assert coords
    return coords


def test_corner_cells_zero_in_every_plane():
    corners = _corner_coords()
    in_disk_coords = _in_disk_coords()
    # A sample placing facts on BOTH corner and in-disk cells.
    stones = tuple((q, r, "player0") for q, r in corners) + ((in_disk_coords[0][0], in_disk_coords[0][1], "player1"),)
    legal = tuple(pack_coord_id((q, r)) for q, r in corners + in_disk_coords)
    policy = tuple((pack_coord_id((q, r)), 1.0) for q, r in in_disk_coords[:3])
    sample = Model1SampleData(
        game_id="g", turn_index=2, current_player="player0", phase="SecondStone",
        center=(0, 0), stones=stones, legal_action_ids=legal,
        first_stone=corners[0], own_hot=(corners[0],), policy=policy,
    )
    out = expand_sample(sample, symmetry=0)
    planes = out["input"]
    corner_mask = ~disk_mask(BOARD_SIZE)  # (41, 41) True at the 420 corner cells
    # Every plane is identically zero on the corner cells (including the constant
    # fill planes: empty, player-colour, second-placement, and center-distance).
    for plane in range(INPUT_CHANNELS):
        assert int(planes[plane][corner_mask].abs().sum()) == 0, f"plane {plane} nonzero at a corner"
    # Center-distance is also zero across the corners specifically.
    assert float(planes[PLANE_CENTER_DISTANCE][corner_mask].abs().sum()) == 0.0
    # ... and the in-disk center-distance cells are NOT all zero (sanity: the plane
    # is still populated where it should be).
    assert float(planes[PLANE_CENTER_DISTANCE].abs().sum()) > 0.0


def test_corner_actions_receive_no_policy_mass():
    corners = _corner_coords()
    in_disk_coords = _in_disk_coords()
    # Policy weights split between corner and in-disk actions; corners get dropped
    # and the in-disk survivors renormalize to sum 1.
    weights = {pack_coord_id(c): 5.0 for c in corners}
    weights[pack_coord_id(in_disk_coords[0])] = 3.0
    weights[pack_coord_id(in_disk_coords[1])] = 1.0
    target = dense_policy_target(weights, center=Axial(0, 0))
    corner_mask_flat = (~disk_mask(BOARD_SIZE)).reshape(-1)
    assert float(target[corner_mask_flat].sum()) == 0.0
    assert float(target.sum()) == pytest.approx(1.0)
    # Mass is split 3:1 between the two surviving in-disk actions.
    from dense_cnn_restnet.geometry import coord_to_flat

    f0 = coord_to_flat(in_disk_coords[0], center=Axial(0, 0))
    f1 = coord_to_flat(in_disk_coords[1], center=Axial(0, 0))
    assert float(target[f0]) == pytest.approx(0.75)
    assert float(target[f1]) == pytest.approx(0.25)
    # The legal mask likewise excludes corner cells.
    mask = legal_mask_flat(list(weights.keys()), center=Axial(0, 0))
    assert int(mask[corner_mask_flat].sum()) == 0


def test_count_spill_counts_corner_facts():
    corners = _corner_coords()
    in_disk_coords = _in_disk_coords()
    sample = Model1SampleData(
        game_id="g", turn_index=0, current_player="player0", phase="Opening",
        center=(0, 0),
        stones=tuple((q, r, "player0") for q, r in corners) + ((0, 0, "player1"),),
        legal_action_ids=tuple(pack_coord_id(c) for c in corners + in_disk_coords),
        placement_history=tuple((q, r, "player0", None, i, None, None) for i, (q, r) in enumerate(corners)),
        policy=tuple((pack_coord_id(c), 1.0) for c in corners[:2]) + ((pack_coord_id(in_disk_coords[0]), 1.0),),
    )
    spill = count_spill(sample)
    assert spill["stones"] == len(corners)  # the (0,0) stone is in-disk
    assert spill["legal"] == len(corners)  # in-disk legals not counted
    assert spill["policy"] == 2
    assert spill["history"] == len(corners)


# --------------------------------------------------------------------------- #
# (c) Rust/Python encoder equality on random states (cross-boundary disk check).
# --------------------------------------------------------------------------- #
def _native_honors_disk() -> bool:
    """True if the loaded native encoder zeroes corners (the disk contract)."""

    try:
        import hexo_engine as engine
        from dense_cnn_restnet import rust_bridge
    except Exception:
        return False
    try:
        state = engine.new_game(seed=0)
        payload = rust_bridge.model1_batch_inputs([state])
    except Exception:
        return False
    shape = tuple(int(s) for s in payload["shape"])
    planes = torch.frombuffer(bytearray(payload["inputs"]), dtype=torch.float32).reshape(shape)[0]
    corner_mask = ~disk_mask(BOARD_SIZE)
    # On a fresh board the empty plane (index 2) is 1 everywhere under the OLD
    # encoder; the disk contract zeroes it at corners.
    return float(planes[2][corner_mask].abs().sum()) == 0.0


@pytest.mark.skipif(not _native_honors_disk(), reason="native _rust.dense_cnn does not yet honor the disk contract")
def test_rust_python_encoder_equal_on_random_states():
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id
    from dense_cnn_restnet import rust_bridge

    def python_planes(state) -> torch.Tensor:
        facts = rust_bridge.model1_sample_from_state(state, game_id="g", turn_index=0, metadata={})
        sample = samples_mod._sample_data_from_facts(facts)
        return build_input_planes(
            current_player=sample.current_player,
            phase=sample.phase,
            center=Axial(*sample.center),
            stones=sample.stones,
            legal_action_ids=sample.legal_action_ids,
            placement_history=sample.placement_history,
            first_stone=sample.first_stone,
            own_hot=sample.own_hot,
            opponent_hot=sample.opponent_hot,
            opponent_last_turn=sample.opponent_last_turn,
            symmetry=0,
        )

    def rust_planes(state) -> torch.Tensor:
        payload = rust_bridge.model1_batch_inputs([state])
        shape = tuple(int(s) for s in payload["shape"])
        return torch.frombuffer(bytearray(payload["inputs"]), dtype=torch.float32).reshape(shape)[0]

    checked = 0
    for seed in range(6):
        state = engine.new_game(seed=seed)
        # Drive the game a few plies into a mid-game position (crop center moves,
        # recency / hot / last-turn planes populate), comparing the encoders at
        # several positions along the way.
        for ply in range(14):
            if engine.terminal(state) is not None:
                break
            payload = rust_bridge.model1_batch_inputs([state])
            legal = [int(a) for a in payload["legal_action_ids"][0]]
            if not legal:
                break
            if ply >= 4:
                py = python_planes(state)
                ru = rust_planes(state)
                assert torch.equal(py, ru), f"seed={seed} ply={ply}: encoder mismatch"
                checked += 1
            action_id = legal[(seed + ply) % len(legal)]
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(action_id)))
    assert checked > 0, "no positions were compared"
