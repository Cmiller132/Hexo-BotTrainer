"""hexfield_compact_v1 shard (de)serialization + the legacy restnet adapter.

Spec §6.1. One columnar ``.npz`` + JSON sidecar per game; raw
representation-agnostic facts; encoders expand at train read. Hygiene changes
vs the legacy layout: NO legal-id column (legality is closed-form from
stones, CI-pinned to the engine); stones and history UNIFIED into one column
``(q i16, r i16, owner u8, placement_index u16)``; ``phase`` stored u8 enum;
standing-win cell columns added.

The legacy adapter reads existing restnet compact-v1 shards by re-implementing
their column layout here (dense_cnn_restnet is never imported at runtime —
its writer is the test oracle only), IGNORING the stored crop-restricted
legal_ids and deriving legality from stones; win-now cells are derived from
the stored stones via the same window scan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import window_scan
from .samples import STV_HORIZONS, HexfieldSampleData

SCHEMA = "hexfield_compact_v1"
SCHEMA_VERSION = 1
# The restnet compact-v1 layout the Phase-B adapter reads
# (dense_cnn_restnet.compact_io.COMPACT_SCHEMA_VERSION). Older restnet shards
# predate the column entirely, so the adapter is lenient when it is absent but
# loud on a present-but-wrong version (mirrors read_compact_shard's guard).
LEGACY_RESTNET_SCHEMA_VERSION = 1
_PHASES = ("Opening", "FirstStone", "SecondStone")
_PHASE_INDEX = {name: i for i, name in enumerate(_PHASES)}


def _concat_offsets(lengths: Sequence[int]) -> np.ndarray:
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
    return offsets


def _pack_qr(points: Sequence[tuple[int, int]]) -> np.ndarray:
    if not points:
        return np.empty(0, dtype=np.int16)
    flat = np.empty(2 * len(points), dtype=np.int16)
    flat[0::2] = [int(q) for q, _ in points]
    flat[1::2] = [int(r) for _, r in points]
    return flat


def _unpack_qr(flat: np.ndarray, off: np.ndarray, i: int) -> tuple[tuple[int, int], ...]:
    a, b = int(off[i]), int(off[i + 1])
    seg = flat[2 * a : 2 * b]
    return tuple((int(seg[2 * k]), int(seg[2 * k + 1])) for k in range(b - a))


def write_compact_shard(
    path: Path,
    samples: Sequence[HexfieldSampleData],
    *,
    short_term_value_horizons: Sequence[int] = STV_HORIZONS,
    sidecar: dict | None = None,
) -> int:
    """Serialize rows into one hexfield_compact_v1 ``.npz`` + JSON sidecar."""

    horizons = tuple(int(h) for h in short_term_value_horizons)
    n = len(samples)
    h = len(horizons)
    horizon_index = {hz: i for i, hz in enumerate(horizons)}

    turn_index = np.empty(n, dtype=np.int32)
    current_player = np.empty(n, dtype=np.uint8)
    phase = np.empty(n, dtype=np.uint8)
    value = np.empty(n, dtype=np.float32)
    moves_left = np.full(n, -1.0, dtype=np.float32)
    first_q = np.zeros(n, dtype=np.int16)
    first_r = np.zeros(n, dtype=np.int16)
    first_present = np.zeros(n, dtype=np.uint8)
    stvalue = np.zeros((n, h), dtype=np.float32)
    stvalue_mask = np.zeros((n, h), dtype=np.float32)

    hist_qr: list[np.ndarray] = []
    hist_owner: list[np.ndarray] = []
    hist_pidx: list[np.ndarray] = []
    hist_len: list[int] = []
    cell_cols: dict[str, tuple[list[np.ndarray], list[int]]] = {
        key: ([], []) for key in ("own_hot", "opp_hot", "own_win", "opp_win")
    }
    pol_act: list[np.ndarray] = []
    pol_w: list[np.ndarray] = []
    pol_len: list[int] = []
    opp_act: list[np.ndarray] = []
    opp_w: list[np.ndarray] = []
    opp_len: list[int] = []

    for i, sample in enumerate(samples):
        turn_index[i] = int(sample.turn_index)
        current_player[i] = int(sample.current_player)
        phase[i] = _PHASE_INDEX[str(sample.phase)]
        value[i] = float(sample.value)
        moves_left[i] = float(sample.moves_left)
        if sample.first_stone is not None:
            first_q[i] = int(sample.first_stone[0])
            first_r[i] = int(sample.first_stone[1])
            first_present[i] = 1
        for hz, val in sample.short_term_value:
            col = horizon_index.get(int(hz))
            if col is not None:
                stvalue[i, col] = float(val)
                stvalue_mask[i, col] = 1.0

        qr = _pack_qr([(q, r) for q, r, _o, _p in sample.records])
        hist_qr.append(qr)
        hist_owner.append(np.asarray([o for _q, _r, o, _p in sample.records], dtype=np.uint8))
        hist_pidx.append(np.asarray([p for _q, _r, _o, p in sample.records], dtype=np.uint16))
        hist_len.append(len(sample.records))

        for key, cells in (
            ("own_hot", sample.own_hot),
            ("opp_hot", sample.opp_hot),
            ("own_win", sample.own_win),
            ("opp_win", sample.opp_win),
        ):
            packed = _pack_qr(tuple(cells))
            cell_cols[key][0].append(packed)
            cell_cols[key][1].append(packed.shape[0] // 2)

        pa = np.fromiter((int(a) for a, _ in sample.policy), dtype=np.uint32, count=len(sample.policy))
        pw = np.fromiter((float(w) for _, w in sample.policy), dtype=np.float32, count=len(sample.policy))
        pol_act.append(pa)
        pol_w.append(pw)
        pol_len.append(int(pa.shape[0]))
        oa = np.fromiter((int(a) for a, _ in sample.opp_policy), dtype=np.uint32, count=len(sample.opp_policy))
        ow = np.fromiter((float(w) for _, w in sample.opp_policy), dtype=np.float32, count=len(sample.opp_policy))
        opp_act.append(oa)
        opp_w.append(ow)
        opp_len.append(int(oa.shape[0]))

    def _cat(parts: list[np.ndarray], dtype) -> np.ndarray:
        if not parts:
            return np.empty(0, dtype=dtype)
        return np.concatenate(parts).astype(dtype, copy=False)

    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "num_rows": np.asarray(n, dtype=np.int64),
        "horizons": np.asarray(horizons, dtype=np.int32),
        "turn_index": turn_index,
        "current_player": current_player,
        "phase": phase,
        "value": value,
        "moves_left": moves_left,
        "first_q": first_q,
        "first_r": first_r,
        "first_present": first_present,
        "stvalue": stvalue,
        "stvalue_mask": stvalue_mask,
        "hist_qr": _cat(hist_qr, np.int16),
        "hist_owner": _cat(hist_owner, np.uint8),
        "hist_pidx": _cat(hist_pidx, np.uint16),
        "hist_off": _concat_offsets(hist_len),
        "pol_act": _cat(pol_act, np.uint32),
        "pol_w": _cat(pol_w, np.float32),
        "pol_off": _concat_offsets(pol_len),
        "opp_act": _cat(opp_act, np.uint32),
        "opp_w": _cat(opp_w, np.float32),
        "opp_off": _concat_offsets(opp_len),
    }
    for key, (parts, lens) in cell_cols.items():
        arrays[f"{key}_qr"] = _cat(parts, np.int16)
        arrays[f"{key}_off"] = _concat_offsets(lens)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    meta = {
        "lineage": "hexfield",
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "rows": n,
        "horizons": list(horizons),
        **(sidecar or {}),
    }
    sidecar_path = path.with_suffix(".json") if path.suffix == ".npz" else Path(str(path) + ".json")
    sidecar_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return n


def read_compact_shard(path: Path) -> list[HexfieldSampleData]:
    """Decode a hexfield_compact_v1 shard back into sample rows."""

    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    if int(arrays["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"unsupported hexfield shard schema {int(arrays['schema_version'])}")

    n = int(arrays["num_rows"])
    horizons = [int(h) for h in arrays["horizons"]]
    out: list[HexfieldSampleData] = []
    for i in range(n):
        h0, h1 = int(arrays["hist_off"][i]), int(arrays["hist_off"][i + 1])
        qr = arrays["hist_qr"][2 * h0 : 2 * h1]
        records = tuple(
            (
                int(qr[2 * k]),
                int(qr[2 * k + 1]),
                int(arrays["hist_owner"][h0 + k]),
                int(arrays["hist_pidx"][h0 + k]),
            )
            for k in range(h1 - h0)
        )
        p0, p1 = int(arrays["pol_off"][i]), int(arrays["pol_off"][i + 1])
        policy = tuple(
            (int(arrays["pol_act"][k]), float(arrays["pol_w"][k])) for k in range(p0, p1)
        )
        o0, o1 = int(arrays["opp_off"][i]), int(arrays["opp_off"][i + 1])
        opp_policy = tuple(
            (int(arrays["opp_act"][k]), float(arrays["opp_w"][k])) for k in range(o0, o1)
        )
        stval = tuple(
            (horizons[c], float(arrays["stvalue"][i, c]))
            for c in range(len(horizons))
            if arrays["stvalue_mask"][i, c] > 0.0
        )
        first = (
            (int(arrays["first_q"][i]), int(arrays["first_r"][i]))
            if int(arrays["first_present"][i]) == 1
            else None
        )
        out.append(
            HexfieldSampleData(
                game_id="",
                turn_index=int(arrays["turn_index"][i]),
                current_player=int(arrays["current_player"][i]),
                phase=_PHASES[int(arrays["phase"][i])],
                records=records,
                first_stone=first,
                own_hot=_unpack_qr(arrays["own_hot_qr"], arrays["own_hot_off"], i),
                opp_hot=_unpack_qr(arrays["opp_hot_qr"], arrays["opp_hot_off"], i),
                own_win=_unpack_qr(arrays["own_win_qr"], arrays["own_win_off"], i),
                opp_win=_unpack_qr(arrays["opp_win_qr"], arrays["opp_win_off"], i),
                policy=policy,
                opp_policy=opp_policy,
                value=float(arrays["value"][i]),
                short_term_value=stval,
                moves_left=float(arrays["moves_left"][i]),
            )
        )
    return out


def read_legacy_restnet_shard(path: Path) -> list[HexfieldSampleData]:
    """Read a restnet compact-v1 shard as hexfield rows (Phase-B adapter).

    Stored legal_ids (crop-restricted at the source) and the crop center are
    IGNORED — legality re-derives from stones at expansion, so crop-clipped
    marathon rows re-expand with full supports. Stored hot lists are raw
    engine coords and read as-is; standing-win cells are derived from the
    stored stones via the shared window scan. The stored visit policies
    remain crop-limited at the source and are disclosed as such
    (`source=legacy_shard`).
    """

    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    # Lenient on absence (pre-versioning restnet shards), loud on drift: mirrors
    # read_compact_shard's guard so a layout change in the legacy writer cannot
    # be misread as the current restnet column order.
    legacy_version = arrays.get("schema_version")
    if legacy_version is not None and int(legacy_version) != LEGACY_RESTNET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported legacy restnet shard schema {int(legacy_version)} "
            f"(adapter expects {LEGACY_RESTNET_SCHEMA_VERSION})"
        )

    n = int(arrays["num_rows"])
    horizons = [int(h) for h in arrays["horizons"]]
    moves_left = arrays.get("moves_left")
    out: list[HexfieldSampleData] = []
    for i in range(n):
        h0, h1 = int(arrays["hist_off"][i]), int(arrays["hist_off"][i + 1])
        qr = arrays["hist_qr"][2 * h0 : 2 * h1]
        records = tuple(
            (
                int(qr[2 * k]),
                int(qr[2 * k + 1]),
                int(arrays["hist_owner"][h0 + k]),
                int(arrays["hist_idx"][h0 + k]),
            )
            for k in range(h1 - h0)
        )
        s0, s1 = int(arrays["stones_off"][i]), int(arrays["stones_off"][i + 1])
        if (s1 - s0) != len(records):
            raise ValueError(
                f"legacy row {i}: stones ({s1 - s0}) != history ({len(records)}) — "
                "the unified-records assumption does not hold"
            )
        current = int(arrays["current_player"][i])
        own_win, opp_win = window_scan(records, current, len(records))[2:]
        p0, p1 = int(arrays["pol_off"][i]), int(arrays["pol_off"][i + 1])
        policy = tuple(
            (int(arrays["pol_act"][k]), float(arrays["pol_w"][k])) for k in range(p0, p1)
        )
        o0, o1 = int(arrays["opp_off"][i]), int(arrays["opp_off"][i + 1])
        opp_policy = tuple(
            (int(arrays["opp_act"][k]), float(arrays["opp_w"][k])) for k in range(o0, o1)
        )
        stval = tuple(
            (horizons[c], float(arrays["stvalue"][i, c]))
            for c in range(len(horizons))
            if arrays["stvalue_mask"][i, c] > 0.0
        )
        first = (
            (int(arrays["first_q"][i]), int(arrays["first_r"][i]))
            if int(arrays["first_present"][i]) == 1
            else None
        )
        out.append(
            HexfieldSampleData(
                game_id="",
                turn_index=int(arrays["turn_index"][i]),
                current_player=current,
                phase=str(arrays["phase"][i]),
                records=records,
                first_stone=first,
                own_hot=_unpack_qr(arrays["own_hot_qr"], arrays["own_hot_off"], i),
                opp_hot=_unpack_qr(arrays["opp_hot_qr"], arrays["opp_hot_off"], i),
                own_win=own_win,
                opp_win=opp_win,
                policy=policy,
                opp_policy=opp_policy,
                value=float(arrays["value"][i]),
                short_term_value=stval,
                moves_left=float(moves_left[i]) if moves_left is not None else -1.0,
                metadata={"source": "legacy_shard"},
            )
        )
    return out
