from __future__ import annotations

import importlib
import math
import numbers
import sys
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


try:
    importlib.import_module("hexo_models")
except ModuleNotFoundError as exc:
    if exc.name != "hexo_models":
        raise
    HEXO_MODELS = None
else:
    HEXO_MODELS = sys.modules["hexo_models"]


def _import_first_available(*module_names: str) -> Any:
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
    return None


if HEXO_MODELS is None:
    D6 = None
    SAMPLES = None
else:
    D6 = _import_first_available("hexo_models.dense_cnn.d6")
    SAMPLES = _import_first_available("hexo_models.dense_cnn.samples")

HAS_HEXO_MODELS_D6 = D6 is not None
HAS_HEXO_MODELS_SAMPLES = SAMPLES is not None


BOARD_SIZE = 41


def _required_api(name: str) -> Any:
    if SAMPLES is None:
        raise AssertionError("hexo_models.samples API is not available")
    value = getattr(SAMPLES, name, None)
    if value is None:
        raise AssertionError(f"hexo_models.samples must expose {name}")
    return value


def _d6_transforms() -> tuple[Any, ...]:
    transforms = getattr(SAMPLES, "D6_TRANSFORMS", None) if SAMPLES is not None else None
    if transforms is None:
        factory = getattr(SAMPLES, "d6_transforms", None) if SAMPLES is not None else None
        if factory is not None:
            transforms = factory()
    if transforms is None and D6 is not None:
        symmetry_class = getattr(D6, "D6Symmetry", None)
        size = int(getattr(D6, "D6_SIZE", 12))
        transforms = tuple(symmetry_class(index) if symmetry_class is not None else index for index in range(size))
    if transforms is None:
        raise AssertionError("hexo_models must expose D6 transforms")
    return tuple(transforms)


def _coord_tuple(coord: Any) -> tuple[int, int]:
    if isinstance(coord, Mapping):
        return int(coord["q"]), int(coord["r"])
    if hasattr(coord, "q") and hasattr(coord, "r"):
        return int(coord.q), int(coord.r)
    if isinstance(coord, Sequence) and not isinstance(coord, (str, bytes, bytearray)):
        return int(coord[0]), int(coord[1])
    raise AssertionError(f"cannot interpret axial coordinate {coord!r}")


def _transform_coord(transform: Any, coord: tuple[int, int]) -> tuple[int, int]:
    transform_axial = getattr(SAMPLES, "transform_axial", None) if SAMPLES is not None else None
    if transform_axial is not None:
        try:
            return _coord_tuple(transform_axial(coord, transform))
        except TypeError:
            return _coord_tuple(transform_axial(q=coord[0], r=coord[1], transform=transform))

    if D6 is None or not hasattr(D6, "transform_coord"):
        raise AssertionError("hexo_models must expose a D6 coordinate transform")
    return _coord_tuple(D6.transform_coord(coord, transform))


def _inverse_transform(transform: Any) -> Any:
    inverse_d6 = getattr(SAMPLES, "inverse_d6", None) if SAMPLES is not None else None
    if inverse_d6 is not None:
        return inverse_d6(transform)

    if D6 is None or not hasattr(D6, "inverse_index"):
        raise AssertionError("hexo_models must expose inverse D6 transforms")
    index = transform.index if hasattr(transform, "index") else int(transform)
    inverse = D6.inverse_index(index)
    symmetry_class = getattr(D6, "D6Symmetry", None)
    return symmetry_class(inverse) if symmetry_class is not None else inverse


def _expected_d6_orbit(coord: tuple[int, int]) -> set[tuple[int, int]]:
    q, r = coord
    return {
        (q, r),
        (-r, q + r),
        (-q - r, q),
        (-q, -r),
        (r, -q - r),
        (q + r, -q),
        (r, q),
        (-q - r, r),
        (q, -q - r),
        (-r, -q),
        (q + r, -r),
        (-q, q + r),
    }


def _identity_transform(transforms: Sequence[Any]) -> Any:
    probes = ((2, 5), (-3, 1), (4, -2))
    for transform in transforms:
        if all(_transform_coord(transform, coord) == coord for coord in probes):
            return transform
    raise AssertionError("D6 transforms must include identity")


def _moving_transform(transforms: Sequence[Any]) -> Any:
    probe = (2, -3)
    for transform in transforms:
        if _transform_coord(transform, probe) != probe:
            return transform
    raise AssertionError("D6 transforms must include a non-identity transform")


def _raw_sample(sample_id: str = "sample", sequence: int = 0) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "sequence": sequence,
        "board_size": BOARD_SIZE,
        "center": (0, 0),
        "policy": (((2, -3), 0.7), ((-4, 1), 0.3)),
        "opp_policy": (((1, 3), 0.6), ((-2, -1), 0.4)),
        "value": -0.375,
        "lookahead": {1: 0.125, 4: -0.75},
    }


def _pack_coord_id(coord: tuple[int, int]) -> int:
    if D6 is not None and hasattr(D6, "pack_coord_id"):
        return int(D6.pack_coord_id(coord))
    q, r = coord
    offset = 1 << 15
    return ((int(q) + offset) << 16) | (int(r) + offset)


def _as_model_sample(sample: Mapping[str, Any]) -> Any:
    sample_class = getattr(SAMPLES, "Model1SampleData", None) if SAMPLES is not None else None
    if sample_class is None:
        return sample

    policy = tuple((_pack_coord_id(_coord_tuple(coord)), float(weight)) for coord, weight in sample["policy"])
    opp_policy = tuple((_pack_coord_id(_coord_tuple(coord)), float(weight)) for coord, weight in sample["opp_policy"])
    legal_action_ids = tuple(action_id for action_id, _weight in policy + opp_policy)
    lookahead = sample["lookahead"].items() if isinstance(sample["lookahead"], Mapping) else sample["lookahead"]

    return sample_class(
        game_id=str(sample["sample_id"]),
        turn_index=int(sample["sequence"]),
        current_player="player0",
        phase="FirstStone",
        center=tuple(sample["center"]),
        stones=((-1, 0, "player0"), (0, 1, "player1")),
        legal_action_ids=legal_action_ids,
        policy=policy,
        opp_policy=opp_policy,
        value=float(sample["value"]),
        lookahead=tuple((int(horizon), float(value)) for horizon, value in lookahead),
        metadata={"sample_id": str(sample["sample_id"])},
    )


def _encode(sample: Mapping[str, Any]) -> Any:
    encoder = getattr(SAMPLES, "encode_compact_sample", None) if SAMPLES is not None else None
    if encoder is not None:
        try:
            return encoder(sample)
        except (AttributeError, TypeError, ValueError):
            return encoder(_as_model_sample(sample))

    compressed_class = getattr(SAMPLES, "CompressedSample", None) if SAMPLES is not None else None
    if compressed_class is not None and hasattr(compressed_class, "from_data"):
        return compressed_class.from_data(_as_model_sample(sample))

    raise AssertionError("hexo_models.samples must expose compact sample encoding")


def _decode(compact_sample: Any, transform: Any) -> Any:
    decode = getattr(SAMPLES, "decode_compact_sample", None) if SAMPLES is not None else None
    if decode is not None:
        for kwargs in ({"transform": transform}, {"symmetry": transform}, {"d6": transform}):
            try:
                return decode(compact_sample, **kwargs)
            except TypeError:
                continue

    expand = getattr(SAMPLES, "expand_sample", None) if SAMPLES is not None else None
    if expand is not None:
        for kwargs in ({"transform": transform}, {"symmetry": transform}, {"d6": transform}):
            try:
                return expand(compact_sample, **kwargs)
            except TypeError:
                continue

    raise AssertionError("hexo_models.samples must decode compact samples with a D6 transform")


def _field(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AssertionError(f"decoded sample is missing one of {names!r}")


def _flatten(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return list(value.reshape(-1).tolist())
    if hasattr(value, "flatten") and hasattr(value, "tolist"):
        return list(value.flatten().tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        flattened: list[Any] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                flattened.extend(_flatten(item))
            else:
                flattened.append(item)
        return flattened
    raise AssertionError(f"cannot flatten dense target {value!r}")


def _dense_index(coord: tuple[int, int], *, center: tuple[int, int] = (0, 0)) -> int:
    indexer = getattr(SAMPLES, "dense_index_for_coord", None)
    if indexer is not None:
        for kwargs in (
            {"center": center, "board_size": BOARD_SIZE},
            {"center": center, "size": BOARD_SIZE},
            {"crop_center": center, "crop_size": BOARD_SIZE},
        ):
            try:
                return int(indexer(coord, **kwargs))
            except TypeError:
                continue

    q, r = coord
    center_q, center_r = center
    radius = BOARD_SIZE // 2
    col = q - center_q + radius
    row = r - center_r + radius
    if not 0 <= row < BOARD_SIZE or not 0 <= col < BOARD_SIZE:
        raise AssertionError(f"coordinate {coord!r} fell outside the dense crop")
    return row * BOARD_SIZE + col


def _normal_form(value: Any) -> Any:
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "detach") or (hasattr(value, "tolist") and hasattr(value, "reshape")):
        return tuple(float(item) for item in _flatten(value))
    if isinstance(value, Mapping):
        return tuple(sorted((key, _normal_form(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_normal_form(item) for item in value)
    return value


def _lookahead_targets(decoded: Any) -> Any:
    try:
        return _field(decoded, "lookahead_targets", "lookahead")
    except AssertionError:
        if isinstance(decoded, Mapping):
            lookahead = {key: value for key, value in decoded.items() if str(key).startswith("lookahead_")}
            if lookahead:
                return lookahead
        raise


def _has_dense_targets(value: Any) -> bool:
    target_names = {
        "policy",
        "policy_target",
        "dense_policy",
        "opp_policy",
        "opp_policy_target",
        "dense_opp_policy",
    }
    if isinstance(value, Mapping):
        return any(name in value for name in target_names)
    return any(hasattr(value, name) for name in target_names)


def _is_compressed_payload(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if isinstance(value, Mapping):
        if value.get("compressed") is True:
            return True
        return any(isinstance(value.get(name), (bytes, bytearray, memoryview)) for name in ("payload", "blob", "data"))
    if getattr(value, "compressed", False) is True:
        return True
    return any(
        isinstance(getattr(value, name, None), (bytes, bytearray, memoryview))
        for name in ("payload", "blob", "data")
    )


def _make_buffer(*, capacity: int, recency_decay: float, seed: int) -> Any:
    buffer_class = _required_api("SampleBuffer")
    halflife = max(float(recency_decay), 1.0e-6)
    for kwargs in (
        {"capacity": capacity, "recency_decay": recency_decay, "seed": seed},
        {"max_size": capacity, "recency_decay": recency_decay, "seed": seed},
        {"capacity": capacity, "decay": recency_decay, "seed": seed},
        {"capacity": capacity, "recency_halflife": halflife},
        {"capacity": capacity},
    ):
        try:
            return buffer_class(**kwargs)
        except TypeError:
            continue
    raise AssertionError("SampleBuffer must accept configurable capacity and recency decay")


def _buffer_add(buffer: Any, sample: Mapping[str, Any]) -> None:
    for name in ("add", "append", "push"):
        method = getattr(buffer, name, None)
        if method is not None:
            try:
                method(sample)
                return
            except (AttributeError, TypeError, ValueError):
                method(_as_model_sample(sample))
                return
    raise AssertionError("SampleBuffer must expose add, append, or push")


def _buffer_entries(buffer: Any) -> list[Any]:
    for name in ("compact_samples", "entries", "records", "samples", "_compact_samples", "_entries", "_records", "_samples"):
        value = getattr(buffer, name, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        return list(value)
    raise AssertionError("SampleBuffer must expose compact in-memory entries")


def _buffer_sample(buffer: Any, count: int) -> list[Any]:
    for name in ("sample", "sample_batch", "random_sample", "draw"):
        method = getattr(buffer, name, None)
        if method is None:
            continue
        for args, kwargs in (((count,), {}), ((), {"count": count}), ((), {"n": count}), ((), {"batch_size": count})):
            try:
                batch = method(*args, **kwargs)
                break
            except TypeError:
                continue
        else:
            continue

        if hasattr(batch, "records"):
            return list(batch.records)
        if isinstance(batch, Mapping) and "records" in batch:
            return list(batch["records"])
        return list(batch)
    raise AssertionError("SampleBuffer must expose random sample selection")


def _sample_id(sample: Any, identity: Any) -> str:
    if hasattr(sample, "decode") and not isinstance(sample, Mapping):
        sample = sample.decode()
    if _is_compressed_payload(sample):
        sample = _decode(sample, identity)
    if isinstance(sample, Mapping):
        if "sample_id" in sample:
            return str(sample["sample_id"])
        metadata = sample.get("metadata")
        if isinstance(metadata, Mapping) and "sample_id" in metadata:
            return str(metadata["sample_id"])
    if hasattr(sample, "sample_id"):
        return str(sample.sample_id)
    if hasattr(sample, "metadata") and "sample_id" in sample.metadata:
        return str(sample.metadata["sample_id"])
    if hasattr(sample, "game_id"):
        return str(sample.game_id)
    raise AssertionError(f"sample id is not visible on sampled record {sample!r}")


@unittest.skipUnless(HAS_HEXO_MODELS_D6, "hexo_models.d6 API is not available yet")
class HexoModelsD6SampleTests(unittest.TestCase):
    def test_axial_d6_exposes_twelve_unique_true_hex_transforms(self) -> None:
        transforms = _d6_transforms()
        self.assertEqual(len(transforms), 12)

        probe = (2, 5)
        transformed = {_transform_coord(transform, probe) for transform in transforms}

        self.assertEqual(len(transformed), 12)
        self.assertEqual(transformed, _expected_d6_orbit(probe))

    def test_axial_d6_transforms_are_bijective_and_round_trip_through_inverse(self) -> None:
        transforms = _d6_transforms()
        domain = tuple((q, r) for q in range(-3, 4) for r in range(-3, 4))

        for transform in transforms:
            image = tuple(_transform_coord(transform, coord) for coord in domain)
            self.assertEqual(len(set(image)), len(domain))

            inverse = _inverse_transform(transform)
            for coord in domain:
                transformed = _transform_coord(transform, coord)
                self.assertEqual(_transform_coord(inverse, transformed), coord)
                self.assertEqual(_transform_coord(transform, _transform_coord(inverse, coord)), coord)


@unittest.skipUnless(HAS_HEXO_MODELS_SAMPLES, "hexo_models.samples API is not available yet")
class HexoModelsSampleBufferTests(unittest.TestCase):

    def test_dense_policy_and_opp_policy_targets_move_with_axial_d6_coordinates(self) -> None:
        transforms = _d6_transforms()
        transform = _moving_transform(transforms)
        raw = _raw_sample()

        decoded = _decode(_encode(raw), transform)
        policy = _flatten(_field(decoded, "policy_target", "dense_policy", "policy"))
        opp_policy = _flatten(_field(decoded, "opp_policy_target", "dense_opp_policy", "opp_policy"))

        self.assertEqual(len(policy), BOARD_SIZE * BOARD_SIZE)
        self.assertEqual(len(opp_policy), BOARD_SIZE * BOARD_SIZE)

        for coord, weight in raw["policy"]:
            index = _dense_index(_transform_coord(transform, coord))
            self.assertAlmostEqual(float(policy[index]), weight, places=6)

        for coord, weight in raw["opp_policy"]:
            index = _dense_index(_transform_coord(transform, coord))
            self.assertAlmostEqual(float(opp_policy[index]), weight, places=6)

        self.assertTrue(math.isclose(sum(float(value) for value in policy), 1.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(sum(float(value) for value in opp_policy), 1.0, abs_tol=1e-6))

    def test_scalar_value_and_lookahead_targets_are_invariant_under_d6(self) -> None:
        transforms = _d6_transforms()
        identity = _identity_transform(transforms)
        compact = _encode(_raw_sample())
        baseline = _decode(compact, identity)
        baseline_value = _normal_form(_field(baseline, "value_target", "value"))
        baseline_lookahead = _normal_form(_lookahead_targets(baseline))

        for transform in transforms:
            decoded = _decode(compact, transform)
            self.assertEqual(_normal_form(_field(decoded, "value_target", "value")), baseline_value)
            self.assertEqual(_normal_form(_lookahead_targets(decoded)), baseline_lookahead)

    def test_compact_samples_remain_compressed_until_decode_expands_dense_targets(self) -> None:
        transforms = _d6_transforms()
        compact = _encode(_raw_sample())

        self.assertTrue(_is_compressed_payload(compact))
        self.assertFalse(_has_dense_targets(compact))

        decoded = _decode(compact, _identity_transform(transforms))

        self.assertTrue(_has_dense_targets(decoded))

    def test_sample_buffer_capacity_is_configurable_to_at_least_200k(self) -> None:
        buffer = _make_buffer(capacity=200_000, recency_decay=0.95, seed=7)
        capacity = getattr(buffer, "capacity", getattr(buffer, "max_size", None))

        self.assertIsNotNone(capacity)
        self.assertGreaterEqual(int(capacity), 200_000)

        _buffer_add(buffer, _raw_sample())
        entries = _buffer_entries(buffer)

        self.assertEqual(len(entries), 1)
        self.assertTrue(_is_compressed_payload(entries[0]))

    def test_random_sampling_prefers_recent_samples_under_decay(self) -> None:
        transforms = _d6_transforms()
        identity = _identity_transform(transforms)
        buffer = _make_buffer(capacity=200_000, recency_decay=0.05, seed=123)

        for index in range(20):
            _buffer_add(buffer, _raw_sample(sample_id=f"old-{index}", sequence=index))
        for index in range(20):
            _buffer_add(buffer, _raw_sample(sample_id=f"recent-{index}", sequence=20 + index))

        draws = 500
        recent = 0
        for _ in range(draws):
            picked = _buffer_sample(buffer, 1)
            self.assertEqual(len(picked), 1)
            if _sample_id(picked[0], identity).startswith("recent-"):
                recent += 1

        self.assertGreater(recent, int(draws * 0.8))


if __name__ == "__main__":
    unittest.main()
