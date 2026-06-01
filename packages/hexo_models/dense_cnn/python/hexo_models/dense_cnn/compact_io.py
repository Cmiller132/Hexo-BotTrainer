"""Columnar (de)serialization of compact Model1 sample shards.

Self-play stores compact sample *facts* (no tensors) so the training read path
can expand each row with a fresh, per-epoch D6 symmetry via
``samples.expand_sample(sample, symmetry=...)``. One shard is one ``.npz`` holding
``N`` rows: fixed scalar-per-row arrays plus, for each variable-length field, a
concatenated data array and an ``int64`` offsets array of length ``N + 1``.

Only the facts ``expand_sample`` consumes for *training* are stored. The root
prior policy, ``policy_surprise`` and ``frequency_weight`` are intentionally
dropped: the loss is unweighted (surprise weighting is already baked into row
duplication by ``materialize_policy_surprise_rows`` before write) and the trainer
never uses the root-policy output. Reconstruction yields the same iterable shapes
self-play produced, reusing ``_PackedStones`` / ``_PackedHistory`` so no per-row
Python tuples are built at read time.
"""

from __future__ import annotations

import zipfile
from array import array
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.lib import format as np_format

from .constants import BOARD_SIZE, INPUT_CHANNELS
from .samples import Model1SampleData, _PackedHistory, _PackedStones, expand_sample

_BOARD_AREA = BOARD_SIZE * BOARD_SIZE

COMPACT_SCHEMA_VERSION = 1

# Owner index <-> engine player, mirroring samples._PLAYERS so stones/history
# round-trip to the exact ``(q, r, player)`` tuples build_input_planes expects.
_PLAYERS = ("player0", "player1")
_PLAYER_INDEX = {"player0": 0, "player1": 1}


def _concat_offsets(lengths: Sequence[int]) -> np.ndarray:
    """Return the cumulative ``int64`` offsets array (length ``N + 1``)."""

    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
    return offsets


def write_compact_shard(
    path: Path,
    samples: Sequence[Model1SampleData],
    *,
    short_term_value_horizons: Sequence[int],
) -> int:
    """Serialize compact samples into one columnar ``.npz`` shard. Returns rows."""

    horizons = tuple(int(h) for h in short_term_value_horizons)
    n = len(samples)
    h = len(horizons)

    turn_index = np.empty(n, dtype=np.int32)
    current_player = np.empty(n, dtype=np.uint8)
    phase = np.empty(n, dtype=object)
    center_q = np.empty(n, dtype=np.int16)
    center_r = np.empty(n, dtype=np.int16)
    value = np.empty(n, dtype=np.float32)
    first_q = np.zeros(n, dtype=np.int16)
    first_r = np.zeros(n, dtype=np.int16)
    first_present = np.zeros(n, dtype=np.uint8)
    stvalue = np.zeros((n, h), dtype=np.float32)
    stvalue_mask = np.zeros((n, h), dtype=np.float32)
    horizon_index = {hz: i for i, hz in enumerate(horizons)}

    # Variable-length accumulators.
    stones_qr: list[np.ndarray] = []
    stones_owner: list[np.ndarray] = []
    stones_len: list[int] = []
    legal_ids: list[np.ndarray] = []
    legal_len: list[int] = []
    hist_qr: list[np.ndarray] = []
    hist_owner: list[np.ndarray] = []
    hist_idx: list[np.ndarray] = []
    hist_len: list[int] = []
    hot_qr: dict[str, list[np.ndarray]] = {"own": [], "opp": [], "last": []}
    hot_len: dict[str, list[int]] = {"own": [], "opp": [], "last": []}
    pol_act: list[np.ndarray] = []
    pol_w: list[np.ndarray] = []
    pol_len: list[int] = []
    opp_act: list[np.ndarray] = []
    opp_w: list[np.ndarray] = []
    opp_len: list[int] = []

    def _pack_qr(points: Sequence[tuple[int, int]]) -> np.ndarray:
        if not points:
            return np.empty(0, dtype=np.int16)
        flat = np.empty(2 * len(points), dtype=np.int16)
        flat[0::2] = [int(q) for q, _ in points]
        flat[1::2] = [int(r) for _, r in points]
        return flat

    def _pack_policy(pairs: Sequence[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
        pairs = tuple(pairs)
        if not pairs:
            return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.float32)
        acts = np.fromiter((int(a) for a, _ in pairs), dtype=np.uint32, count=len(pairs))
        ws = np.fromiter((float(w) for _, w in pairs), dtype=np.float32, count=len(pairs))
        return acts, ws

    for i, sample in enumerate(samples):
        turn_index[i] = int(sample.turn_index)
        current_player[i] = _PLAYER_INDEX[str(sample.current_player)]
        phase[i] = str(sample.phase)
        center_q[i] = int(sample.center[0])
        center_r[i] = int(sample.center[1])
        value[i] = float(sample.value)
        if sample.first_stone is not None:
            first_q[i] = int(sample.first_stone[0])
            first_r[i] = int(sample.first_stone[1])
            first_present[i] = 1
        for hz, val in sample.short_term_value:
            col = horizon_index.get(int(hz))
            if col is not None:
                stvalue[i, col] = float(val)
                stvalue_mask[i, col] = 1.0

        s_qr = array("h")
        s_owner = bytearray()
        for q, r, player in sample.stones:
            s_qr.append(int(q))
            s_qr.append(int(r))
            s_owner.append(_PLAYER_INDEX[str(player)])
        stones_qr.append(np.frombuffer(bytes(s_qr), dtype=np.int16))
        stones_owner.append(np.frombuffer(bytes(s_owner), dtype=np.uint8))
        stones_len.append(len(s_owner))

        legal = np.fromiter((int(a) for a in sample.legal_action_ids), dtype=np.uint32)
        legal_ids.append(legal)
        legal_len.append(int(legal.shape[0]))

        h_qr = array("h")
        h_owner = bytearray()
        h_idx = array("i")
        for q, r, player, _phase, idx, _fq, _fr in sample.placement_history:
            h_qr.append(int(q))
            h_qr.append(int(r))
            h_owner.append(_PLAYER_INDEX[str(player)])
            h_idx.append(int(idx))
        hist_qr.append(np.frombuffer(bytes(h_qr), dtype=np.int16))
        hist_owner.append(np.frombuffer(bytes(h_owner), dtype=np.uint8))
        hist_idx.append(np.frombuffer(bytes(h_idx), dtype=np.int32))
        hist_len.append(len(h_owner))

        for key, points in (
            ("own", sample.own_hot),
            ("opp", sample.opponent_hot),
            ("last", sample.opponent_last_turn),
        ):
            packed = _pack_qr(tuple(points))
            hot_qr[key].append(packed)
            hot_len[key].append(packed.shape[0] // 2)

        pa, pw = _pack_policy(sample.policy)
        pol_act.append(pa)
        pol_w.append(pw)
        pol_len.append(int(pa.shape[0]))
        oa, ow = _pack_policy(sample.opp_policy)
        opp_act.append(oa)
        opp_w.append(ow)
        opp_len.append(int(oa.shape[0]))

    def _cat(parts: list[np.ndarray], dtype) -> np.ndarray:
        if not parts:
            return np.empty(0, dtype=dtype)
        return np.concatenate(parts).astype(dtype, copy=False)

    arrays = {
        "schema_version": np.asarray(COMPACT_SCHEMA_VERSION, dtype=np.int32),
        "num_rows": np.asarray(n, dtype=np.int64),
        "horizons": np.asarray(horizons, dtype=np.int32),
        "turn_index": turn_index,
        "current_player": current_player,
        "phase": phase,
        "center_q": center_q,
        "center_r": center_r,
        "value": value,
        "first_q": first_q,
        "first_r": first_r,
        "first_present": first_present,
        "stvalue": stvalue,
        "stvalue_mask": stvalue_mask,
        "stones_qr": _cat(stones_qr, np.int16),
        "stones_owner": _cat(stones_owner, np.uint8),
        "stones_off": _concat_offsets(stones_len),
        "legal_ids": _cat(legal_ids, np.uint32),
        "legal_off": _concat_offsets(legal_len),
        "hist_qr": _cat(hist_qr, np.int16),
        "hist_owner": _cat(hist_owner, np.uint8),
        "hist_idx": _cat(hist_idx, np.int32),
        "hist_off": _concat_offsets(hist_len),
        "own_hot_qr": _cat(hot_qr["own"], np.int16),
        "own_hot_off": _concat_offsets(hot_len["own"]),
        "opp_hot_qr": _cat(hot_qr["opp"], np.int16),
        "opp_hot_off": _concat_offsets(hot_len["opp"]),
        "last_hot_qr": _cat(hot_qr["last"], np.int16),
        "last_hot_off": _concat_offsets(hot_len["last"]),
        "pol_act": _cat(pol_act, np.uint32),
        "pol_w": _cat(pol_w, np.float32),
        "pol_off": _concat_offsets(pol_len),
        "opp_act": _cat(opp_act, np.uint32),
        "opp_w": _cat(opp_w, np.float32),
        "opp_off": _concat_offsets(opp_len),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return n


def compact_row_count(path: Path) -> int:
    """Rows in a compact shard, read from the array header (no full decompress).

    Reads the shape of the per-row ``turn_index`` array straight from the zip
    member header so scanning the replay window stays cheap.
    """

    with zipfile.ZipFile(path) as archive:
        with archive.open("turn_index.npy") as handle:
            version = np_format.read_magic(handle)
            if version == (1, 0):
                shape, _fortran, _dtype = np_format.read_array_header_1_0(handle)
            else:
                shape, _fortran, _dtype = np_format.read_array_header_2_0(handle)
    return int(shape[0]) if shape else 0


def read_shard_horizons(path: Path) -> tuple[int, ...]:
    """Short-term-value horizons a compact shard was written with."""

    with np.load(path, allow_pickle=True) as data:
        return tuple(int(h) for h in data["horizons"])


def expand_shard_to_arrays(
    path: Path,
    symmetries: "np.ndarray | Sequence[int]",
    short_term_value_horizons: Sequence[int],
) -> dict[str, np.ndarray]:
    """Read a compact shard and expand every row to stacked dense numpy arrays.

    Row ``i`` is expanded under ``symmetries[i]`` via the tested
    ``samples.expand_sample``. Returns plain numpy (picklable, so this runs in a
    worker process) with exactly the keys the trainer's optimizer step consumes:
    ``input`` (N,C,41,41), ``policy``/``opp_policy`` (N,1681), ``value`` (N,), and
    ``stvalue_<h>`` + ``stvalue_<h>_mask`` (N,) per horizon (missing horizons
    masked to 0). This is the parallelizable unit of train-read expansion.
    """

    horizons = tuple(int(h) for h in short_term_value_horizons)

    def _empty() -> dict[str, np.ndarray]:
        out = {
            "input": np.zeros((0, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "policy": np.zeros((0, _BOARD_AREA), dtype=np.float32),
            "opp_policy": np.zeros((0, _BOARD_AREA), dtype=np.float32),
            "value": np.zeros((0,), dtype=np.float32),
        }
        for horizon in horizons:
            out[f"stvalue_{horizon}"] = np.zeros((0,), dtype=np.float32)
            out[f"stvalue_{horizon}_mask"] = np.zeros((0,), dtype=np.float32)
        return out

    samples = read_compact_shard(path)
    if not samples:
        return _empty()

    # Expand each row under its drawn D6 symmetry. The 41x41 *square* crop is NOT
    # closed under hexagonal D6: a symmetry can rotate a row's entire policy mass
    # outside the crop (e.g. corner offset (20,20) -> (-20,40) under a 60deg turn),
    # so dense_policy_target raises "positive probability mass". When that happens we
    # re-expand the whole row at identity (symmetry 0), which the sample was authored
    # to fit, so the row trains un-augmented instead of crashing the epoch's training
    # pass. A row degenerate even at identity (genuine bad data) is dropped. Both are
    # rare; the count is logged once per shard for visibility.
    expanded = []
    aug_fallbacks = 0
    dropped = 0
    for i, sample in enumerate(samples):
        sym = int(symmetries[i])
        try:
            expanded.append(expand_sample(sample, symmetry=sym))
            continue
        except ValueError:
            pass
        if sym != 0:
            try:
                expanded.append(expand_sample(sample, symmetry=0))
                aug_fallbacks += 1
                continue
            except ValueError:
                pass
        dropped += 1
    if aug_fallbacks or dropped:
        import sys

        print(
            f"[compact_io] D6 square-crop guard in {getattr(path, 'name', path)}: "
            f"{aug_fallbacks} row(s) -> identity, {dropped} dropped (of {len(samples)})",
            file=sys.stderr,
            flush=True,
        )
    if not expanded:
        return _empty()

    out = {
        "input": np.stack([row["input"].numpy() for row in expanded]).astype(np.float32, copy=False),
        "policy": np.stack([row["policy"].numpy() for row in expanded]).astype(np.float32, copy=False),
        "opp_policy": np.stack([row["opp_policy"].numpy() for row in expanded]).astype(np.float32, copy=False),
        "value": np.asarray([float(row["value"]) for row in expanded], dtype=np.float32),
    }
    for horizon in horizons:
        key = f"stvalue_{horizon}"
        present = [key in row for row in expanded]
        out[key] = np.asarray(
            [float(row[key]) if ok else 0.0 for row, ok in zip(expanded, present)], dtype=np.float32
        )
        out[f"{key}_mask"] = np.asarray([1.0 if ok else 0.0 for ok in present], dtype=np.float32)
    return out


def read_compact_shard(path: Path) -> list[Model1SampleData]:
    """Decode a compact shard back into ``Model1SampleData`` rows.

    Stones / placement history are rebuilt as ``_PackedStones`` / ``_PackedHistory``
    (the same types self-play produced) so iteration matches ``build_input_planes``
    exactly with no per-row Python tuple construction. ``root_prior_policy`` is
    restored as empty — the trainer does not use the root-policy output.
    """

    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    n = int(arrays["num_rows"])
    horizons = [int(h) for h in arrays["horizons"]]
    out: list[Model1SampleData] = []

    stones_qr = arrays["stones_qr"]
    stones_owner = arrays["stones_owner"]
    stones_off = arrays["stones_off"]
    legal_ids = arrays["legal_ids"]
    legal_off = arrays["legal_off"]
    hist_qr = arrays["hist_qr"]
    hist_owner = arrays["hist_owner"]
    hist_idx = arrays["hist_idx"]
    hist_off = arrays["hist_off"]
    pol_act = arrays["pol_act"]
    pol_w = arrays["pol_w"]
    pol_off = arrays["pol_off"]
    opp_act = arrays["opp_act"]
    opp_w = arrays["opp_w"]
    opp_off = arrays["opp_off"]

    def _qr_pairs(flat: np.ndarray, off: np.ndarray, i: int) -> tuple[tuple[int, int], ...]:
        a, b = int(off[i]), int(off[i + 1])
        seg = flat[2 * a : 2 * b]
        return tuple((int(seg[2 * k]), int(seg[2 * k + 1])) for k in range(b - a))

    for i in range(n):
        s0, s1 = int(stones_off[i]), int(stones_off[i + 1])
        stones = _PackedStones(
            coords=array("h", stones_qr[2 * s0 : 2 * s1].tobytes()),
            owners=stones_owner[s0:s1].tobytes(),
        )
        h0, h1 = int(hist_off[i]), int(hist_off[i + 1])
        history = _PackedHistory(
            coords=array("h", hist_qr[2 * h0 : 2 * h1].tobytes()),
            owners=hist_owner[h0:h1].tobytes(),
            indices=array("i", hist_idx[h0:h1].tobytes()),
        )
        l0, l1 = int(legal_off[i]), int(legal_off[i + 1])
        legal = array("q", (int(a) for a in legal_ids[l0:l1]))
        p0, p1 = int(pol_off[i]), int(pol_off[i + 1])
        policy = [(int(pol_act[k]), float(pol_w[k])) for k in range(p0, p1)]
        o0, o1 = int(opp_off[i]), int(opp_off[i + 1])
        opp_policy = tuple((int(opp_act[k]), float(opp_w[k])) for k in range(o0, o1))
        stval = tuple(
            (int(horizons[c]), float(arrays["stvalue"][i, c]))
            for c in range(len(horizons))
            if arrays["stvalue_mask"][i, c] > 0.0
        )
        first = (
            (int(arrays["first_q"][i]), int(arrays["first_r"][i]))
            if int(arrays["first_present"][i]) == 1
            else None
        )
        out.append(
            Model1SampleData(
                game_id="",
                turn_index=int(arrays["turn_index"][i]),
                current_player=_PLAYERS[int(arrays["current_player"][i])],
                phase=str(arrays["phase"][i]),
                center=(int(arrays["center_q"][i]), int(arrays["center_r"][i])),
                stones=stones,
                legal_action_ids=legal,
                placement_history=history,
                first_stone=first,
                own_hot=_qr_pairs(arrays["own_hot_qr"], arrays["own_hot_off"], i),
                opponent_hot=_qr_pairs(arrays["opp_hot_qr"], arrays["opp_hot_off"], i),
                opponent_last_turn=_qr_pairs(arrays["last_hot_qr"], arrays["last_hot_off"], i),
                policy=policy,
                root_prior_policy=(),
                opp_policy=opp_policy,
                value=float(arrays["value"][i]),
                short_term_value=stval,
            )
        )
    return out
