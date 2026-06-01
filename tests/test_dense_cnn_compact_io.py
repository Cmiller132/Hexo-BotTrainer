"""Compact-shard (de)serialization round-trip and train-time D6 augmentation.

The compact shard must reconstruct samples that expand BYTE-FOR-BYTE the same as
the originals (except the deliberately-dropped root-policy output) under every D6
symmetry, since training expands compact rows with a fresh per-epoch symmetry.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

torch = pytest.importorskip("torch")
d6 = importlib.import_module("hexo_models.dense_cnn.d6")
samples = importlib.import_module("hexo_models.dense_cnn.samples")
compact_io = importlib.import_module("hexo_models.dense_cnn.compact_io")
constants = importlib.import_module("hexo_models.dense_cnn.constants")

TRANSFORMS = tuple(d6.D6Symmetry(i) for i in range(d6.D6_SIZE))
HORIZONS = (1, 4, 8)
# Every expand_sample key the trainer consumes. root_policy is intentionally
# excluded: compact shards drop root_prior_policy (the trainer ignores it).
COMPARE_KEYS = ("input", "policy", "opp_policy", "legal_mask", "value", "stvalue_1", "stvalue_4", "stvalue_8")


def _aid(q: int, r: int) -> int:
    return int(d6.pack_coord_id(d6.Axial(q, r)))


def _sample(
    *,
    current_player: str,
    phase: str,
    center: tuple[int, int],
    stones: tuple[tuple[int, int, str], ...],
    legal: tuple[tuple[int, int], ...],
    history: tuple[tuple[int, int, str, int], ...],
    first_stone: tuple[int, int] | None,
    hot: tuple[tuple[int, int], ...],
    policy: tuple[tuple[tuple[int, int], float], ...],
    opp_policy: tuple[tuple[tuple[int, int], float], ...],
    value: float,
    stvalue: tuple[tuple[int, float], ...],
) -> Any:
    return samples.Model1SampleData(
        game_id="g",
        turn_index=len(history),
        current_player=current_player,
        phase=phase,
        center=center,
        stones=stones,
        legal_action_ids=tuple(_aid(q, r) for q, r in legal),
        placement_history=tuple((q, r, p, None, idx, None, None) for q, r, p, idx in history),
        first_stone=first_stone,
        own_hot=hot,
        opponent_hot=tuple((r, q) for q, r in hot),
        opponent_last_turn=hot[:1],
        policy=tuple((_aid(q, r), w) for (q, r), w in policy),
        root_prior_policy=tuple((_aid(q, r), w) for (q, r), w in policy),
        opp_policy=tuple((_aid(q, r), w) for (q, r), w in opp_policy),
        value=value,
        short_term_value=stvalue,
    )


def _varied_samples() -> list[Any]:
    interior = _sample(
        current_player="player0", phase="FirstStone", center=(0, 0),
        stones=((1, 0, "player0"), (-1, 2, "player1")),
        legal=((0, 0), (2, -3), (-4, 1), (5, 5)),
        history=((1, 0, "player0", 0), (-1, 2, "player1", 1)),
        first_stone=(1, 0), hot=((1, 0), (2, -3)),
        policy=(((2, -3), 0.7), ((-4, 1), 0.3)),
        opp_policy=(((0, 0), 1.0),),
        value=-0.375, stvalue=((1, 0.125), (4, -0.75), (8, 0.5)),
    )
    # Boundary: facts near the +-20 square edge / corner that a rotation moves out of crop.
    boundary = _sample(
        current_player="player1", phase="Placement", center=(0, 0),
        stones=((20, -10, "player1"), (19, 20, "player0"), (-20, -20, "player0")),
        legal=((20, 20), (-20, 5), (18, -19)),
        history=((20, -10, "player1", 0), (19, 20, "player0", 1), (-20, -20, "player0", 2)),
        first_stone=(20, -10), hot=((20, 20), (-20, -20)),
        policy=(((20, 20), 0.5), ((-20, 5), 0.5)),
        opp_policy=(((18, -19), 1.0),),
        value=1.0, stvalue=((1, -1.0), (4, 0.25)),  # horizon 8 deliberately absent
    )
    # Late-game-ish: many stones / history / legal, off-center crop.
    many_stones = tuple((q, q - 5, "player0" if (q % 2 == 0) else "player1") for q in range(-6, 7))
    many_hist = tuple((q, q - 5, "player0" if (q % 2 == 0) else "player1", i) for i, q in enumerate(range(-6, 7)))
    late = _sample(
        current_player="player0", phase="Placement", center=(3, -2),
        stones=many_stones,
        legal=tuple((q, -q) for q in range(-8, 9)),
        history=many_hist,
        first_stone=(-6, -11), hot=((3, -2), (4, -3), (5, -4)),
        policy=(((1, -1), 0.4), ((2, -2), 0.35), ((3, -3), 0.25)),
        opp_policy=(((0, 0), 0.6), ((1, 1), 0.4)),
        value=0.0, stvalue=((1, 0.0), (4, 0.1), (8, -0.2)),
    )
    # Minimal: empty board, single legal move, no history/hot, no stvalue.
    minimal = _sample(
        current_player="player1", phase="FirstStone", center=(0, 0),
        stones=(), legal=((0, 0),), history=(), first_stone=None, hot=(),
        policy=(((0, 0), 1.0),), opp_policy=(), value=0.0, stvalue=(),
    )
    return [interior, boundary, late, minimal]


def _assert_expand_equal(original: Any, restored: Any) -> None:
    for sym in TRANSFORMS:
        a = samples.expand_sample(original, symmetry=sym)
        b = samples.expand_sample(restored, symmetry=sym)
        for key in COMPARE_KEYS:
            if key not in a:  # stvalue horizon absent in this sample
                assert key not in b, f"{key} present in restored but not original"
                continue
            assert key in b, f"{key} missing from restored expansion"
            ta, tb = a[key], b[key]
            if ta.dtype == torch.bool:
                assert torch.equal(ta, tb), f"{key} mismatch under sym {sym.index}"
            else:
                assert torch.allclose(ta, tb, atol=1e-6), f"{key} mismatch under sym {sym.index}"


def test_compact_shard_round_trip_expands_identically(tmp_path: Path) -> None:
    originals = _varied_samples()
    path = tmp_path / "shard.npz"
    rows = compact_io.write_compact_shard(path, originals, short_term_value_horizons=HORIZONS)
    assert rows == len(originals)
    assert compact_io.compact_row_count(path) == len(originals)

    restored = compact_io.read_compact_shard(path)
    assert len(restored) == len(originals)
    for original, rebuilt in zip(originals, restored):
        _assert_expand_equal(original, rebuilt)


def test_compact_shard_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    assert compact_io.write_compact_shard(path, [], short_term_value_horizons=HORIZONS) == 0
    assert compact_io.compact_row_count(path) == 0
    assert compact_io.read_compact_shard(path) == []


def test_expand_shard_to_arrays_matches_per_row_expand(tmp_path: Path) -> None:
    originals = _varied_samples()
    path = tmp_path / "shard.npz"
    compact_io.write_compact_shard(path, originals, short_term_value_horizons=HORIZONS)
    syms = np.array([i % d6.D6_SIZE for i in range(len(originals))], dtype=np.int64)

    arrays = compact_io.expand_shard_to_arrays(path, syms, HORIZONS)
    restored = compact_io.read_compact_shard(path)
    assert arrays["input"].shape[0] == len(originals)
    for i, (sample, sym) in enumerate(zip(restored, syms)):
        # expand_shard_to_arrays falls a row back to identity when the drawn symmetry
        # would drop any in-crop fact (the square crop is not D6-closed), so compare
        # against the SAME effective symmetry, not the raw drawn one.
        eff = 0 if (int(sym) != 0 and samples.symmetry_drops_support(sample, int(sym))) else int(sym)
        ref = samples.expand_sample(sample, symmetry=eff)
        assert np.allclose(arrays["input"][i], ref["input"].numpy(), atol=1e-6)
        assert np.allclose(arrays["policy"][i], ref["policy"].numpy(), atol=1e-6)
        assert np.allclose(arrays["opp_policy"][i], ref["opp_policy"].numpy(), atol=1e-6)
        assert arrays["value"][i] == pytest.approx(float(ref["value"]), abs=1e-6)
    for h in HORIZONS:
        assert f"stvalue_{h}" in arrays and f"stvalue_{h}_mask" in arrays


def test_expand_shard_to_arrays_runs_in_spawn_pool(tmp_path: Path) -> None:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    originals = _varied_samples()
    path = tmp_path / "shard.npz"
    compact_io.write_compact_shard(path, originals, short_term_value_horizons=HORIZONS)
    syms = np.zeros(len(originals), dtype=np.int64)

    serial = compact_io.expand_shard_to_arrays(path, syms, HORIZONS)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        parallel = pool.submit(compact_io.expand_shard_to_arrays, path, syms, HORIZONS).result()
    assert set(serial) == set(parallel)
    for key in serial:
        assert np.allclose(serial[key], parallel[key], atol=1e-6), key


def test_value_and_stvalue_survive_round_trip(tmp_path: Path) -> None:
    originals = _varied_samples()
    path = tmp_path / "shard.npz"
    compact_io.write_compact_shard(path, originals, short_term_value_horizons=HORIZONS)
    restored = compact_io.read_compact_shard(path)
    for original, rebuilt in zip(originals, restored):
        base = samples.expand_sample(original, symmetry=TRANSFORMS[0])
        got = samples.expand_sample(rebuilt, symmetry=TRANSFORMS[0])
        assert float(got["value"]) == pytest.approx(float(base["value"]))
        for h in HORIZONS:
            key = f"stvalue_{h}"
            assert (key in got) == (key in base)
            if key in base:
                assert float(got[key]) == pytest.approx(float(base[key]), abs=1e-6)


def _corner_policy_sample() -> Any:
    # Entire policy mass at the +-20 square corner. Valid at identity, but a hex D6
    # rotation sends (20,20) -> (-20,40), outside the 41x41 crop, so the dense policy
    # target is all-zeros and expand_sample raises. (opp_policy uses allow_empty.)
    return _sample(
        current_player="player0", phase="FirstStone", center=(0, 0),
        stones=(), legal=((20, 20),), history=(), first_stone=None, hot=(),
        policy=(((20, 20), 1.0),), opp_policy=(), value=0.25, stvalue=((1, 0.5),),
    )


def test_expand_shard_falls_back_to_identity_when_symmetry_empties_policy(tmp_path: Path) -> None:
    corner = _corner_policy_sample()
    # The sample must actually trigger the path: >=1 symmetry rotates the lone corner
    # action out of the crop (raises), while identity keeps it in.
    raising = []
    for sym in TRANSFORMS:
        try:
            samples.expand_sample(corner, symmetry=sym)
        except ValueError:
            raising.append(sym.index)
    assert raising, "test sample did not empty the policy under any symmetry"
    assert 0 not in raising, "identity must keep the corner action in-crop"

    path = tmp_path / "corner.npz"
    compact_io.write_compact_shard(path, [corner], short_term_value_horizons=HORIZONS)
    identity_policy = samples.expand_sample(corner, symmetry=0)["policy"].numpy()

    # Under every empties-the-policy symmetry, expand_shard_to_arrays must NOT raise,
    # must keep the row (not drop it), and must produce the identity expansion.
    for k in raising:
        arrays = compact_io.expand_shard_to_arrays(path, np.array([k], dtype=np.int64), HORIZONS)
        assert arrays["input"].shape[0] == 1, f"row dropped under sym {k} instead of identity-fallback"
        assert float(arrays["policy"][0].sum()) > 0.0, f"policy still empty under sym {k}"
        assert np.allclose(arrays["policy"][0], identity_policy, atol=1e-6), f"sym {k} not re-expanded at identity"


def test_expand_shard_drops_row_degenerate_even_at_identity(tmp_path: Path) -> None:
    good = _varied_samples()[0]  # interior sample, valid under any symmetry
    empty_policy = _sample(
        current_player="player0", phase="FirstStone", center=(0, 0),
        stones=(), legal=((0, 0),), history=(), first_stone=None, hot=(),
        policy=(), opp_policy=(), value=0.0, stvalue=(),  # empty policy -> raises even at identity
    )
    path = tmp_path / "mix.npz"
    compact_io.write_compact_shard(path, [good, empty_policy], short_term_value_horizons=HORIZONS)
    # Identity syms: the good row expands; the degenerate row is dropped, not fatal.
    arrays = compact_io.expand_shard_to_arrays(path, np.zeros(2, dtype=np.int64), HORIZONS)
    assert arrays["input"].shape[0] == 1
    assert float(arrays["policy"][0].sum()) > 0.0


def test_expand_shard_falls_back_on_partial_policy_spill(tmp_path: Path) -> None:
    # #1 (HIGH): policy mass split between a CENTER action (never leaves the crop) and a
    # CORNER action (leaves under some symmetries). A symmetry that drops only the corner
    # is a PARTIAL spill: expand_sample does NOT raise -- it silently truncates 2 actions
    # to 1 and renormalizes. The coverage guard must fall the row back to identity.
    sample = _sample(
        current_player="player0", phase="FirstStone", center=(0, 0),
        stones=(), legal=((0, 0), (20, 20)), history=(), first_stone=None, hot=(),
        policy=(((0, 0), 0.5), ((20, 20), 0.5)), opp_policy=(), value=0.0, stvalue=(),
    )
    id_pol = samples.expand_sample(sample, symmetry=0)["policy"].numpy()
    assert int((id_pol > 0).sum()) == 2  # both actions in-crop at identity

    partial = []
    for k in range(1, d6.D6_SIZE):
        if not samples.symmetry_drops_support(sample, k):
            continue
        raw = samples.expand_sample(sample, symmetry=k)["policy"].numpy()  # never raises (center survives)
        partial.append((k, int((raw > 0).sum())))
    assert partial, "test sample did not produce a partial-spill symmetry"
    # PROVE THE BUG: raw expand_sample silently truncates the corner action (2 -> 1).
    assert all(n == 1 for _k, n in partial), f"expected silent truncation to 1 action, got {partial}"

    # PROVE THE FIX: expand_shard_to_arrays falls back to identity (both actions kept).
    path = tmp_path / "partial.npz"
    compact_io.write_compact_shard(path, [sample], short_term_value_horizons=HORIZONS)
    for k, _n in partial:
        arr = compact_io.expand_shard_to_arrays(path, np.array([k], dtype=np.int64), HORIZONS)
        assert int((arr["policy"][0] > 0).sum()) == 2, f"sym {k}: still truncated (coverage guard failed)"
        assert np.allclose(arr["policy"][0], id_pol, atol=1e-6)


def _stone_count(planes) -> int:
    """Number of in-crop stones encoded in an input tensor (own + opponent planes)."""
    import numpy as _np

    arr = planes.numpy() if hasattr(planes, "numpy") else _np.asarray(planes)
    return int(arr[constants.PLANE_OWN_STONES].sum() + arr[constants.PLANE_OPPONENT_STONES].sum())


def test_coverage_guard_preserves_corner_stone(tmp_path: Path) -> None:
    # #3 (MED): a CORNER stone is silently dropped from the input planes under a spilling
    # symmetry while the central policy survives -> counterfactual board, no exception.
    sample = _sample(
        current_player="player0", phase="Placement", center=(0, 0),
        stones=((0, 0, "player0"), (20, 20, "player1")),
        legal=((0, 0),), history=(), first_stone=None, hot=(),
        policy=(((0, 0), 1.0),), opp_policy=(), value=0.0, stvalue=(),
    )
    assert _stone_count(samples.expand_sample(sample, symmetry=0)["input"]) == 2

    partial = [k for k in range(1, d6.D6_SIZE) if samples.symmetry_drops_support(sample, k)]
    assert partial, "test sample did not produce a stone-spilling symmetry"
    # PROVE THE BUG: raw expand drops the corner stone (2 -> 1) with the policy intact.
    raw_counts = {k: _stone_count(samples.expand_sample(sample, symmetry=k)["input"]) for k in partial}
    assert all(c < 2 for c in raw_counts.values()), f"expected a dropped stone, got {raw_counts}"
    # PROVE THE FIX: shard expansion falls back to identity -> both stones present.
    path = tmp_path / "stone.npz"
    compact_io.write_compact_shard(path, [sample], short_term_value_horizons=HORIZONS)
    for k in partial:
        arr = compact_io.expand_shard_to_arrays(path, np.array([k], dtype=np.int64), HORIZONS)
        n = int(arr["input"][0, constants.PLANE_OWN_STONES].sum() + arr["input"][0, constants.PLANE_OPPONENT_STONES].sum())
        assert n == 2, f"sym {k}: stone still dropped (coverage guard failed)"


def test_coverage_guard_preserves_opp_policy(tmp_path: Path) -> None:
    # #7/#8 (LOW): opp_policy uses allow_empty=True, so a corner opp action is silently
    # dropped (or zeroed) under a spilling symmetry with NO exception and no fallback.
    sample = _sample(
        current_player="player0", phase="FirstStone", center=(0, 0),
        stones=(), legal=((0, 0),), history=(), first_stone=None, hot=(),
        policy=(((0, 0), 1.0),), opp_policy=(((20, 20), 1.0),), value=0.0, stvalue=(),
    )
    id_opp = samples.expand_sample(sample, symmetry=0)["opp_policy"].numpy()
    assert float(id_opp.sum()) > 0.0  # opp target present at identity

    partial = [k for k in range(1, d6.D6_SIZE) if samples.symmetry_drops_support(sample, k)]
    assert partial, "test sample did not produce an opp-policy-spilling symmetry"
    # PROVE THE BUG: raw expand silently zeroes the opp target (allow_empty, no raise).
    assert all(float(samples.expand_sample(sample, symmetry=k)["opp_policy"].sum()) == 0.0 for k in partial)
    # PROVE THE FIX: shard expansion falls back to identity -> opp target retained.
    path = tmp_path / "opp.npz"
    compact_io.write_compact_shard(path, [sample], short_term_value_horizons=HORIZONS)
    for k in partial:
        arr = compact_io.expand_shard_to_arrays(path, np.array([k], dtype=np.int64), HORIZONS)
        assert float(arr["opp_policy"][0].sum()) > 0.0, f"sym {k}: opp target still dropped (coverage guard failed)"
