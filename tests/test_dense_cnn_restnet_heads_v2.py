"""heads_v2 surface for dense_cnn_restnet (2026-06-09).

Covers: the adaptive half-life temperature decay, the moves-left auxiliary
target/head plumbing (finalize -> shard -> expansion -> loss, with old-shard
back-compat), the horizon-UNION shuffle (mid-run horizon changes), and the
shared ValueReduction head layout.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
_RESTNET = ROOT / "packages" / "dense_cnn_restnet" / "python"
if str(_RESTNET) not in sys.path:
    sys.path.insert(0, str(_RESTNET))

torch = pytest.importorskip("torch")
architecture = importlib.import_module("dense_cnn_restnet.architecture")
compact_io = importlib.import_module("dense_cnn_restnet.compact_io")
constants = importlib.import_module("dense_cnn_restnet.constants")
d6 = importlib.import_module("dense_cnn_restnet.d6")
losses = importlib.import_module("dense_cnn_restnet.losses")
replay = importlib.import_module("dense_cnn_restnet.replay")
samples = importlib.import_module("dense_cnn_restnet.samples")
selfplay = importlib.import_module("dense_cnn_restnet.selfplay")


def _aid(q: int, r: int) -> int:
    return int(d6.pack_coord_id(d6.Axial(q, r)))


def _mk_sample(turn: int, *, moves_left: float = -1.0, horizons=((1, 0.5),)):
    legal = tuple((q, r) for q in range(-2, 3) for r in range(-2, 3) if abs(q + r) <= 2)
    return samples.Model1SampleData(
        game_id="g",
        turn_index=turn,
        current_player="player0",
        phase="Placement",
        center=(0, 0),
        stones=(),
        legal_action_ids=tuple(_aid(q, r) for q, r in legal),
        policy=tuple((_aid(q, r), 1.0) for q, r in legal[:2]),
        root_prior_policy=tuple((_aid(q, r), 1.0) for q, r in legal),
        value=1.0,
        short_term_value=tuple(horizons),
        moves_left=moves_left,
    )


# --- adaptive temperature half-life -------------------------------------------


def test_halflife_temperature_decay():
    kwargs = dict(initial=1.0, final=1.0, decay_moves=0, floor=0.1, halflife_plies=8.0)
    assert selfplay._move_temperature(0, **kwargs) == pytest.approx(1.0)
    assert selfplay._move_temperature(8, **kwargs) == pytest.approx(0.5)
    assert selfplay._move_temperature(16, **kwargs) == pytest.approx(0.25)
    assert selfplay._move_temperature(64, **kwargs) == pytest.approx(0.1)  # floor


def test_halflife_takes_precedence_over_schedule():
    temp = selfplay._move_temperature(
        10, initial=1.0, final=1.0, decay_moves=0,
        schedule=((0, 0.9), (30, 0.9)), floor=0.0, halflife_plies=10.0,
    )
    assert temp == pytest.approx(0.5)


def test_length_ema_roundtrip(tmp_path):
    assert selfplay._read_length_ema(tmp_path, 32.0) == pytest.approx(32.0)  # prior
    selfplay._write_length_ema(tmp_path, 40.0, epoch=3, latest_mean=44.0)
    assert selfplay._read_length_ema(tmp_path, 32.0) == pytest.approx(40.0)


# --- moves-left target plumbing ------------------------------------------------


def test_finalize_sets_moves_left_and_masks_truncated():
    pending = [("player0", _mk_sample(i, horizons=()), 0.0) for i in range(4)]
    done = samples.finalize_game_samples(pending, winner="player0", horizons=())
    assert [s.moves_left for s in done] == [3.0, 2.0, 1.0, 0.0]
    truncated = samples.finalize_game_samples(pending, winner=None, horizons=(), truncated=True)
    assert all(s.moves_left == -1.0 for s in truncated)


def test_expand_normalizes_moves_left():
    cap = constants.MOVES_LEFT_CAP
    expanded = samples.expand_sample(_mk_sample(0, moves_left=cap / 4))
    assert float(expanded["moves_left"]) == pytest.approx(2.0 * 0.25 - 1.0)
    assert "moves_left" not in samples.expand_sample(_mk_sample(0, moves_left=-1.0))
    capped = samples.expand_sample(_mk_sample(0, moves_left=cap * 3))
    assert float(capped["moves_left"]) == pytest.approx(1.0)  # clamped at cap


def test_shard_roundtrip_and_old_shard_backcompat(tmp_path):
    rows = [_mk_sample(0, moves_left=5.0), _mk_sample(1, moves_left=-1.0)]
    shard = tmp_path / "rows.npz"
    compact_io.write_compact_shard(shard, rows, short_term_value_horizons=(1,))
    arrays = compact_io.expand_shard_to_arrays(shard, np.zeros(2, dtype=np.int64), (1,))
    assert arrays["moves_left_mask"].tolist() == [1.0, 0.0]
    assert arrays["moves_left"][0] == pytest.approx(2.0 * 5.0 / constants.MOVES_LEFT_CAP - 1.0)

    # Old-shard back-compat: strip the moves_left column and re-read.
    with np.load(shard, allow_pickle=True) as data:
        stripped = {k: data[k] for k in data.files if k != "moves_left"}
    np.savez_compressed(shard, **stripped)
    restored = compact_io.read_compact_shard(shard)
    assert all(s.moves_left == -1.0 for s in restored)
    arrays = compact_io.expand_shard_to_arrays(shard, np.zeros(2, dtype=np.int64), (1,))
    assert arrays["moves_left_mask"].tolist() == [0.0, 0.0]


# --- turn-aligned lookahead targets (even-offset EMA) --------------------------


def test_stvalue_targets_step_over_full_turns():
    # Decision sequence after P0's opening: P1 FS, P1 SS, P0 FS, P0 SS, P1 FS ...
    movers = ["player0", "player1", "player1", "player0", "player0", "player1", "player1"]
    roots = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    pending = [(mover, _mk_sample(i, horizons=()), roots[i]) for i, mover in enumerate(movers)]
    done = samples.finalize_game_samples(pending, winner="player0", horizons=(2,))
    # From decision 0 (player0): even offsets land on future indices 1, 3, 5 ->
    # decisions 2, 4, 6 with movers P1, P0, P1 -> perspective -0.2, +0.4, -0.6.
    # Horizon 2 => per-step decay (2-1)/(2+1) = 1/3.
    w = [1.0, 1.0 / 3.0, 1.0 / 9.0]
    expected = (-0.2 * w[0] + 0.4 * w[1] - 0.6 * w[2]) / sum(w)
    target = dict(done[0].short_term_value)
    assert target[2] == pytest.approx(expected)
    # The decision 1 ply before the end has no even-offset future -> masked.
    assert done[-2].short_term_value == ()
    assert done[-1].short_term_value == ()  # final decision: no future at all


def test_config_rejects_odd_horizons():
    importlib.import_module("dense_cnn_restnet.config")
    config_mod = sys.modules["dense_cnn_restnet.config"]
    with pytest.raises(ValueError, match="even"):
        config_mod.parse_model1_config({"architecture": {"short_term_value_horizons": [1, 4]}})
    parsed = config_mod.parse_model1_config({"architecture": {"short_term_value_horizons": [2, 6, 16]}})
    assert parsed.architecture.short_term_value_horizons == (2, 6, 16)


# --- horizon-union shuffle (mid-run horizon change) ----------------------------


def test_shuffle_unions_mixed_horizons(tmp_path):
    selfplay_dir = tmp_path / "selfplay"
    old = [_mk_sample(i, moves_left=float(i), horizons=((1, 0.2), (8, 0.4))) for i in range(8)]
    new = [_mk_sample(i, moves_left=float(i), horizons=((1, 0.3), (16, 0.6))) for i in range(8)]
    replay.write_selfplay_npz(
        selfplay_dir / "old.npz", old, raw_rows=8, epoch=0, game_id="old", short_term_value_horizons=(1, 8)
    )
    replay.write_selfplay_npz(
        selfplay_dir / "new.npz", new, raw_rows=8, epoch=1, game_id="new", short_term_value_horizons=(1, 16)
    )
    result = replay.build_katago_shuffle(
        selfplay_dir=selfplay_dir, shuffled_root=tmp_path / "shuffled", scratch_dir=tmp_path / "scratch",
        epoch=1, seed=0, min_rows=1, keep_target_rows=64, taper_window_exponent=1.0,
        expand_window_per_row=1.0, taper_window_scale=None, approx_rows_per_out_file=64,
        batch_size=4, worker_group_size=64,
    )
    assert result.status == "completed"
    out = result.output_files[0]
    assert compact_io.read_shard_horizons(out) == (1, 8, 16)
    arrays = compact_io.expand_shard_to_arrays(out, np.zeros(result.output_rows, dtype=np.int64), (1, 4, 16))
    # Every row carries horizon 1; only the 'new' rows carry 16; nobody has 4.
    assert arrays["stvalue_1_mask"].sum() == result.output_rows
    assert 0 < arrays["stvalue_16_mask"].sum() < result.output_rows
    assert arrays["stvalue_4_mask"].sum() == 0


# --- shared reduction + moves-left head + loss ---------------------------------


def test_network_emits_moves_left_and_splits_reductions():
    net = architecture.RestnetNetwork(
        channels=16, blocks_type="R_T", attention_heads=4, short_term_value_horizons=(1, 16)
    )
    x = torch.zeros(2, constants.INPUT_CHANNELS, 41, 41)
    x[:, 3, 20, 20] = 1.0
    out = net(x)
    assert out["moves_left"].shape == (2, constants.VALUE_BINS)
    assert out["stvalue_16"].shape == (2, constants.VALUE_BINS)
    assert set(net.forward_policy_value(x)) == {"policy", "value"}
    # heads_v3: the MAIN value head owns a private reduction and the auxiliary
    # heads (stvalue/moves_left) share a second one -> exactly two
    # BOARD_AREA->64 Linears in the whole model.
    big = [m for m in net.modules() if isinstance(m, torch.nn.Linear) and m.in_features == constants.BOARD_AREA]
    assert len(big) == 2
    assert net.aux_value_reduction is not None
    # With no auxiliary heads at all there is no aux reduction.
    plain = architecture.RestnetNetwork(
        channels=16, blocks_type="R_T", attention_heads=4, moves_left_head=False
    )
    assert "moves_left" not in plain(x)
    assert plain.aux_value_reduction is None
    plain_big = [
        m for m in plain.modules() if isinstance(m, torch.nn.Linear) and m.in_features == constants.BOARD_AREA
    ]
    assert len(plain_big) == 1


def test_model1_loss_moves_left_component():
    generator = torch.Generator().manual_seed(3)
    outputs = {
        "policy": torch.randn(2, 8, generator=generator),
        "value": torch.randn(2, constants.VALUE_BINS, generator=generator),
        "moves_left": torch.randn(2, constants.VALUE_BINS, generator=generator),
    }
    batch = {
        "policy": torch.rand(2, 8, generator=generator),
        "value": torch.zeros(2),
        "moves_left": torch.tensor([0.5, 0.0]),
        "moves_left_mask": torch.tensor([1.0, 0.0]),
    }
    total, components = losses.model1_loss(outputs, batch, moves_left_weight=0.5)
    assert "moves_left" in components
    base, _ = losses.model1_loss({k: v for k, v in outputs.items() if k != "moves_left"}, batch)
    assert float(total) == pytest.approx(float(base) + 0.5 * float(components["moves_left"]), rel=1e-5)
    # Fully-masked moves_left contributes exactly nothing.
    all_masked = {**batch, "moves_left_mask": torch.zeros(2)}
    masked_total, _ = losses.model1_loss(outputs, all_masked, moves_left_weight=0.5)
    assert float(masked_total) == pytest.approx(float(base), rel=1e-6)
