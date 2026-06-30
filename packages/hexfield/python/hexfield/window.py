"""Packed columnar in-RAM replay window.

The on-disk ``hexfield_compact_v1`` shard (see ``shards.write_compact_shard``) is
already a flat columnar layout: per-row scalar arrays + ``(n,H)`` blocks + a
handful of CSR ``data``/``off`` group pairs. ``shards.read_compact_shard`` is
correct but eagerly explodes every row into a frozen
:class:`~hexfield.samples.HexfieldSampleData` (tuples of boxed Python scalars;
~1-2 GB heap at 500k rows). This module keeps every column **packed** — it never
materializes the boxed-tuple representation.

:class:`PackedWindow` holds the exact compact column set concatenated across
shards (with CSR offsets rebased to one global index) plus a per-row
``generation`` and ``row_shard_id`` tag. :class:`PackedRowView` hands back the
zero-copy slices for ONE row in precisely the shape one
:func:`~hexfield.samples.expand_sample` call consumes.

``shards.read_compact_shard`` is no longer on the hot path but is kept intact as
the parity oracle this packed path is validated row-for-row against.

Design notes:
- ``horizons`` on a :class:`PackedWindow` is the **union** across concatenated
  shards. Expansion passes the CONFIG horizons; the stored
  ``stvalue``/``stvalue_mask`` columns are preserved verbatim. In practice every
  hexfield shard is written with the same ``STV_HORIZONS=(2,6,16)`` so the union
  is a no-op, but :func:`concat_packed` asserts identical horizons rather than
  silently merging mismatched blocks (a mismatch would mean ``stvalue`` columns
  are not comparable and must not be concatenated).
- :func:`concat_packed` is **streaming**: it pre-sizes every output array from
  the per-shard counts, fills in place, rebases CSR offsets to ``int64``, and
  frees each part right after its copy — so the transient peak is ~1x the final
  window plus one shard, not the ~2x of ``np.concatenate(parts)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .shards import SCHEMA_VERSION

if TYPE_CHECKING:
    from .buffer_manifest import ShardEntry

# --- column taxonomy (mirrors shards.write_compact_shard exactly) ------------

# Per-row scalar columns: indexed directly by the row index ``i``.
SCALAR_COLS: tuple[str, ...] = (
    "turn_index",
    "current_player",
    "phase",
    "value",
    "moves_left",
    "outcome_valid",
    "policy_valid",
    "policy_surprise",
    "first_q",
    "first_r",
    "first_present",
)

# Per-row ``(n, H)`` block columns: indexed ``[i, :]``.
BLOCK_COLS: tuple[str, ...] = ("stvalue", "stvalue_mask")

# CSR groups. Each entry is ``(off_col, data_cols, qr_doubled)`` where:
#   - ``off_col``    is the ``int64[n+1]`` offsets array for the group;
#   - ``data_cols``  are the flat data arrays governed by that offsets array;
#   - ``qr_doubled`` is True for the packed-(q,r) int16 arrays, where ``off``
#     counts *pairs* and the flat slice for row ``i`` is
#     ``data[2*off[i] : 2*off[i+1]]`` (shards._unpack_qr semantics). When False
#     (pol/opp/hist_owner/hist_pidx) the slice is ``data[off[i] : off[i+1]]``.
#
# The ``hist`` group is special: ``hist_qr`` is qr-doubled while ``hist_owner``
# and ``hist_pidx`` share the SAME ``hist_off`` but are NOT doubled. It is split
# into two pseudo-groups that share one offsets array so the per-column copy +
# rebase logic stays uniform.
CSR_GROUPS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("hist_off", ("hist_qr",), True),
    ("hist_off", ("hist_owner", "hist_pidx"), False),
    ("pol_off", ("pol_act", "pol_w", "q_pol_q"), False),
    ("opp_off", ("opp_act", "opp_w"), False),
    ("own_hot_off", ("own_hot_qr",), True),
    ("opp_hot_off", ("opp_hot_qr",), True),
    ("own_win_off", ("own_win_qr",), True),
    ("opp_win_off", ("opp_win_qr",), True),
)

# Distinct offset arrays (each appears once even though hist_off backs two
# pseudo-groups). Order is fixed for deterministic concat.
OFF_COLS: tuple[str, ...] = (
    "hist_off",
    "pol_off",
    "opp_off",
    "own_hot_off",
    "opp_hot_off",
    "own_win_off",
    "opp_win_off",
)

# Map an offsets column to the data columns it governs and whether each is
# qr-doubled. Built once from CSR_GROUPS.
_OFF_TO_DATA: dict[str, list[tuple[str, bool]]] = {}
for _off, _datas, _doubled in CSR_GROUPS:
    bucket = _OFF_TO_DATA.setdefault(_off, [])
    for _d in _datas:
        bucket.append((_d, _doubled))


@dataclass
class PackedRowView:
    """Zero-copy slices for ONE row — exactly what one ``expand_sample`` needs.

    Every array attribute is a *view* into the parent :class:`PackedWindow`
    columns (no copy). The qr arrays are the flat ``int16`` pair-packed segments
    (``shards._unpack_qr`` reads ``(seg[2k], seg[2k+1])``); the owner/pidx and
    policy/value arrays are the plain per-row segments.
    """

    # scalars (python-native, cheap to box for one row)
    turn_index: int
    current_player: int
    phase: int
    value: float
    moves_left: float
    outcome_valid: int  # 1 completed / 0 truncated (gates value/stvalue/cell_q)
    policy_valid: int  # 1 full / 0 fast (gates policy/opp/soft/cell_q)
    policy_surprise: float
    first_q: int
    first_r: int
    first_present: int
    # blocks
    stvalue: np.ndarray  # (H,) f32 view
    stvalue_mask: np.ndarray  # (H,) f32 view
    # history CSR (owner/pidx aligned with hist_qr pairs)
    hist_qr: np.ndarray  # (2L,) i16 view
    hist_owner: np.ndarray  # (L,) u8 view
    hist_pidx: np.ndarray  # (L,) u16 view
    # policy / opp-policy CSR
    pol_act: np.ndarray  # (P,) u32 view
    pol_w: np.ndarray  # (P,) f32 view
    q_pol_q: np.ndarray  # (P,) f32 view; one child Q per recorded action (== pol_act)
    opp_act: np.ndarray  # (O,) u32 view
    opp_w: np.ndarray  # (O,) f32 view
    # standing-cell qr CSR (flat pair-packed i16 views)
    own_hot_qr: np.ndarray
    opp_hot_qr: np.ndarray
    own_win_qr: np.ndarray
    opp_win_qr: np.ndarray
    # tags
    horizons: tuple[int, ...]
    generation: int
    row_shard_id: int

    def records(self) -> tuple[tuple[int, int, int, int], ...]:
        """``(q, r, owner, placement_index)`` tuples — the ``records`` field."""
        qr = self.hist_qr
        owner = self.hist_owner
        pidx = self.hist_pidx
        return tuple(
            (int(qr[2 * k]), int(qr[2 * k + 1]), int(owner[k]), int(pidx[k]))
            for k in range(owner.shape[0])
        )

    @staticmethod
    def _qr_pairs(flat: np.ndarray) -> tuple[tuple[int, int], ...]:
        m = flat.shape[0] // 2
        return tuple((int(flat[2 * k]), int(flat[2 * k + 1])) for k in range(m))

    def own_hot(self) -> tuple[tuple[int, int], ...]:
        return self._qr_pairs(self.own_hot_qr)

    def opp_hot(self) -> tuple[tuple[int, int], ...]:
        return self._qr_pairs(self.opp_hot_qr)

    def own_win(self) -> tuple[tuple[int, int], ...]:
        return self._qr_pairs(self.own_win_qr)

    def opp_win(self) -> tuple[tuple[int, int], ...]:
        return self._qr_pairs(self.opp_win_qr)

    def policy(self) -> tuple[tuple[int, float], ...]:
        return tuple((int(self.pol_act[k]), float(self.pol_w[k])) for k in range(self.pol_act.shape[0]))

    def q_policy(self) -> tuple[tuple[int, float], ...]:
        return tuple((int(self.pol_act[k]), float(self.q_pol_q[k])) for k in range(self.pol_act.shape[0]))

    def opp_policy(self) -> tuple[tuple[int, float], ...]:
        return tuple((int(self.opp_act[k]), float(self.opp_w[k])) for k in range(self.opp_act.shape[0]))

    def first_stone(self) -> tuple[int, int] | None:
        return (int(self.first_q), int(self.first_r)) if int(self.first_present) == 1 else None

    def short_term_value(self) -> tuple[tuple[int, float], ...]:
        mask = self.stvalue_mask
        vals = self.stvalue
        return tuple(
            (int(self.horizons[c]), float(vals[c]))
            for c in range(len(self.horizons))
            if mask[c] > 0.0
        )


@dataclass
class PackedWindow:
    """Concatenated packed columns for a whole replay window.

    ``cols`` holds every ``hexfield_compact_v1`` column kept PACKED: the per-row
    scalar arrays, the ``(n,H)`` blocks, the flat CSR data arrays, and the
    ``int64[n+1]`` CSR offsets (one global offsets array per group, rebased by
    :func:`concat_packed`). ``generation`` and ``row_shard_id`` are ``int32[n]``
    per-row tags.

    The window deliberately exposes neither ``window_size`` nor an ``index`` with
    ``sample_count`` (the opaque-window guard) so the framework's
    ``D6SymmetrySelector`` treats it as opaque (``_sample_count`` -> 0) and does
    NOT blake2b-hash every row each epoch.
    """

    n: int
    cols: dict[str, np.ndarray]
    horizons: tuple[int, ...]
    generation: np.ndarray  # int32[n]
    row_shard_id: np.ndarray  # int32[n]

    @classmethod
    def empty(cls) -> "PackedWindow":
        """A zero-row window. ``train_passes`` already handles n==0."""
        cols: dict[str, np.ndarray] = {}
        for name in SCALAR_COLS:
            cols[name] = np.empty(0, dtype=_SCALAR_DTYPES[name])
        for name in BLOCK_COLS:
            cols[name] = np.empty((0, 0), dtype=np.float32)
        for off in OFF_COLS:
            cols[off] = np.zeros(1, dtype=np.int64)
        for _off, datas, _doubled in CSR_GROUPS:
            for d in datas:
                cols[d] = np.empty(0, dtype=_CSR_DTYPES[d])
        return cls(
            n=0,
            cols=cols,
            horizons=(),
            generation=np.empty(0, dtype=np.int32),
            row_shard_id=np.empty(0, dtype=np.int32),
        )

    def row_view(self, i: int) -> PackedRowView:
        """Zero-copy slices for row ``i``. Feeds one ``expand_sample``."""
        if i < 0 or i >= self.n:
            raise IndexError(f"row {i} out of range for PackedWindow(n={self.n})")
        c = self.cols
        h0, h1 = int(c["hist_off"][i]), int(c["hist_off"][i + 1])
        p0, p1 = int(c["pol_off"][i]), int(c["pol_off"][i + 1])
        o0, o1 = int(c["opp_off"][i]), int(c["opp_off"][i + 1])

        def qr_slice(key: str) -> np.ndarray:
            off = c[key + "_off"]
            a, b = int(off[i]), int(off[i + 1])
            return c[key + "_qr"][2 * a : 2 * b]

        return PackedRowView(
            turn_index=int(c["turn_index"][i]),
            current_player=int(c["current_player"][i]),
            phase=int(c["phase"][i]),
            value=float(c["value"][i]),
            moves_left=float(c["moves_left"][i]),
            outcome_valid=int(c["outcome_valid"][i]),
            policy_valid=int(c["policy_valid"][i]),
            policy_surprise=float(c["policy_surprise"][i]),
            first_q=int(c["first_q"][i]),
            first_r=int(c["first_r"][i]),
            first_present=int(c["first_present"][i]),
            stvalue=c["stvalue"][i],
            stvalue_mask=c["stvalue_mask"][i],
            hist_qr=c["hist_qr"][2 * h0 : 2 * h1],
            hist_owner=c["hist_owner"][h0:h1],
            hist_pidx=c["hist_pidx"][h0:h1],
            pol_act=c["pol_act"][p0:p1],
            pol_w=c["pol_w"][p0:p1],
            q_pol_q=c["q_pol_q"][p0:p1],
            opp_act=c["opp_act"][o0:o1],
            opp_w=c["opp_w"][o0:o1],
            own_hot_qr=qr_slice("own_hot"),
            opp_hot_qr=qr_slice("opp_hot"),
            own_win_qr=qr_slice("own_win"),
            opp_win_qr=qr_slice("opp_win"),
            horizons=self.horizons,
            generation=int(self.generation[i]),
            row_shard_id=int(self.row_shard_id[i]),
        )


# Expected dtypes per column, used by empty() and as a load-time sanity guard.
# These mirror the writer (shards.write_compact_shard).
_SCALAR_DTYPES: dict[str, np.dtype] = {
    "turn_index": np.dtype(np.int32),
    "current_player": np.dtype(np.uint8),
    "phase": np.dtype(np.uint8),
    "value": np.dtype(np.float32),
    "moves_left": np.dtype(np.float32),
    "outcome_valid": np.dtype(np.uint8),
    "policy_valid": np.dtype(np.uint8),
    "policy_surprise": np.dtype(np.float32),
    "first_q": np.dtype(np.int16),
    "first_r": np.dtype(np.int16),
    "first_present": np.dtype(np.uint8),
}
_CSR_DTYPES: dict[str, np.dtype] = {
    "hist_qr": np.dtype(np.int16),
    "hist_owner": np.dtype(np.uint8),
    "hist_pidx": np.dtype(np.uint16),
    "pol_act": np.dtype(np.uint32),
    "pol_w": np.dtype(np.float32),
    "q_pol_q": np.dtype(np.float32),
    "opp_act": np.dtype(np.uint32),
    "opp_w": np.dtype(np.float32),
    "own_hot_qr": np.dtype(np.int16),
    "opp_hot_qr": np.dtype(np.int16),
    "own_win_qr": np.dtype(np.int16),
    "opp_win_qr": np.dtype(np.int16),
}


def _shard_generation(path: Path, num_rows: int) -> int:
    """Producing epoch for the shard. Key-derived epoch is authoritative
    (``game_key // 1_000_000``, structurally guaranteed by the selfplay shard
    writer); the sidecar ``epoch`` only cross-checks (warns on mismatch); the
    directory name is the last-resort fallback. Never uses mtime.
    """
    stem = path.stem  # e.g. "game_1000000"
    key_epoch: int | None = None
    if "_" in stem:
        try:
            game_key = int(stem.split("_", 1)[1])
            key_epoch = game_key // 1_000_000
        except (ValueError, IndexError):
            key_epoch = None

    sidecar = path.with_suffix(".json")
    side_epoch: int | None = None
    if sidecar.exists():
        try:
            import json

            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if "epoch" in meta:
                side_epoch = int(meta["epoch"])
        except (ValueError, OSError):
            side_epoch = None

    if key_epoch is not None:
        if side_epoch is not None and side_epoch != key_epoch:
            import warnings

            warnings.warn(
                f"shard {path.name}: sidecar epoch {side_epoch} != key-derived "
                f"epoch {key_epoch}; trusting key-derived",
                RuntimeWarning,
                stacklevel=2,
            )
        return key_epoch
    if side_epoch is not None:
        return side_epoch
    # Last resort: parent dir name "epoch_NNNNNN".
    parent = path.parent.name
    if parent.startswith("epoch_"):
        try:
            return int(parent.split("_", 1)[1])
        except (ValueError, IndexError):
            pass
    return 0


def load_packed_shard(path: Path) -> PackedWindow:
    """Load ONE ``hexfield_compact_v1`` shard with columns kept PACKED.

    ``np.load`` then validate ``schema_version`` (loud on drift, mirroring
    ``read_compact_shard``); columns are NOT exploded into
    :class:`HexfieldSampleData`. Legacy restnet shards are out of scope here —
    they defer to ``shards.read_legacy_restnet_shard`` (a separate compat island)
    and are not packed by this loader.
    """
    path = Path(path)
    with np.load(path) as data:
        files = set(data.files)
        if "schema_version" not in files:
            raise ValueError(f"{path.name}: not a hexfield_compact_v1 shard (no schema_version)")
        version = int(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported hexfield shard schema {version} (loader expects {SCHEMA_VERSION})"
            )
        n = int(data["num_rows"])
        horizons = tuple(int(h) for h in data["horizons"])

        cols: dict[str, np.ndarray] = {}
        # Materialize each needed column out of the npz mmap into a real array
        # (np.load arrays are lazy/closed-on-exit; force the read while open).
        for name in SCALAR_COLS:
            if name in ("outcome_valid", "policy_valid") and name not in files:
                # Backward-compatible: shards predating the truncated-game
                # (outcome_valid) / PCR value-row (policy_valid) features lack
                # this column → default all-1 (every row completed / full).
                cols[name] = np.ones(n, dtype=_SCALAR_DTYPES[name])
                continue
            cols[name] = np.ascontiguousarray(data[name])
        for name in BLOCK_COLS:
            cols[name] = np.ascontiguousarray(data[name])
        for off in OFF_COLS:
            cols[off] = np.ascontiguousarray(data[off]).astype(np.int64, copy=False)
        for _off, datas, _doubled in CSR_GROUPS:
            for d in datas:
                cols[d] = np.ascontiguousarray(data[d])

    generation = np.full(n, _shard_generation(path, n), dtype=np.int32)
    row_shard_id = np.zeros(n, dtype=np.int32)
    return PackedWindow(
        n=n,
        cols=cols,
        horizons=horizons,
        generation=generation,
        row_shard_id=row_shard_id,
    )


def concat_packed(parts: Sequence[PackedWindow]) -> PackedWindow:
    """Streaming concat of packed shards into one window.

    Pre-sizes every output array from the per-shard counts (no list-of-parts +
    ``np.concatenate``, which peaks ~2x), fills in place, rebases each CSR
    offsets array to ONE global ``int64`` index, and **frees each part right
    after its copy** so the transient stays ~1x the final window plus one shard.

    Empty parts (n==0) are tolerated and skipped. ``horizons`` is the union and
    must be identical across non-empty parts — a mismatch means
    the ``stvalue`` block columns are not comparable and must not be glued.
    """
    parts = [p for p in parts]
    nonempty = [p for p in parts if p.n > 0]
    if not nonempty:
        return PackedWindow.empty()

    # Validate consistent horizons / block widths across parts (S4).
    horizons = nonempty[0].horizons
    h_width = nonempty[0].cols["stvalue"].shape[1]
    for p in nonempty[1:]:
        if p.horizons != horizons:
            raise ValueError(
                f"concat_packed: horizon mismatch {p.horizons} != {horizons}; "
                "stvalue blocks are not concatenatable"
            )
        if p.cols["stvalue"].shape[1] != h_width:
            raise ValueError("concat_packed: stvalue block width mismatch")

    total_n = int(sum(p.n for p in nonempty))

    # --- pre-size outputs from counts (no transient 2x) ----------------------
    out: dict[str, np.ndarray] = {}
    for name in SCALAR_COLS:
        out[name] = np.empty(total_n, dtype=_SCALAR_DTYPES[name])
    for name in BLOCK_COLS:
        out[name] = np.empty((total_n, h_width), dtype=np.float32)
    # CSR data totals (sum of each data array length across parts).
    data_totals: dict[str, int] = {}
    for _off, datas, _doubled in CSR_GROUPS:
        for d in datas:
            data_totals[d] = int(sum(p.cols[d].shape[0] for p in nonempty))
    for d, tot in data_totals.items():
        out[d] = np.empty(tot, dtype=_CSR_DTYPES[d])
    for off in OFF_COLS:
        out[off] = np.empty(total_n + 1, dtype=np.int64)
        out[off][0] = 0
    out_gen = np.empty(total_n, dtype=np.int32)
    out_sid = np.empty(total_n, dtype=np.int32)

    # --- fill in place, rebasing CSR offsets ---------------------------------
    row_cursor = 0
    data_cursor: dict[str, int] = {d: 0 for d in data_totals}
    off_base: dict[str, int] = {off: 0 for off in OFF_COLS}

    for shard_idx, part in enumerate(nonempty):
        pc = part.cols
        pn = part.n
        r0, r1 = row_cursor, row_cursor + pn

        for name in SCALAR_COLS:
            out[name][r0:r1] = pc[name]
        for name in BLOCK_COLS:
            out[name][r0:r1, :] = pc[name]
        out_gen[r0:r1] = part.generation
        # Preserve the source-shard identity as a global running index across the
        # whole window (diagnostics); part.row_shard_id is per-load 0.
        out_sid[r0:r1] = np.int32(shard_idx)

        for off in OFF_COLS:
            src_off = pc[off]  # int64[pn+1], starts at 0
            base = off_base[off]
            # Global offsets for this part's rows: src_off[1:] + base.
            out[off][r0 + 1 : r1 + 1] = src_off[1:] + base
            # Advance base by this part's total count for the group.
            off_base[off] = base + int(src_off[pn])
            # Copy each data array governed by this offsets group.
            for d, doubled in _OFF_TO_DATA[off]:
                src = pc[d]
                m = src.shape[0]
                dc = data_cursor[d]
                out[d][dc : dc + m] = src
                data_cursor[d] = dc + m

        row_cursor = r1
        # Free the part's columns now (free each part right after its copy).
        part.cols.clear()
        part.generation = np.empty(0, dtype=np.int32)
        part.row_shard_id = np.empty(0, dtype=np.int32)
        part.n = 0

    # Sanity: every CSR data array filled exactly, and each offsets array ends at
    # the accumulated element/pair count for its group.
    for d, tot in data_totals.items():
        assert data_cursor[d] == tot, f"CSR data {d} fill mismatch {data_cursor[d]} != {tot}"
    for off in OFF_COLS:
        assert int(out[off][total_n]) == off_base[off]

    return PackedWindow(
        n=total_n,
        cols=out,
        horizons=horizons,
        generation=out_gen,
        row_shard_id=out_sid,
    )


# =============================================================================
# KataGo / dense_cnn_restnet window mathematics, md5 split, and the overshoot-skip
# file selection. Each function below is a faithful port of its dense twin (named
# inline); the divergences are only the hexfield row container (``ShardEntry``
# carries ``.rows`` / ``.generation`` / ``.game_key`` / ``.rel_path`` where
# dense's ``ShuffleFileInfo`` carries ``.rows`` / ``.mtime`` / ``.path``) and the
# in-RAM ``PackedWindow`` build (hexfield shuffles the window in RAM; dense
# re-shards to disk).
# =============================================================================


def compute_katago_window_rows(
    usable_rows: int,
    *,
    min_rows: int,
    expand_window_per_row: float,
    taper_window_exponent: float,
    taper_window_scale: float | None,
) -> int:
    """Power-law taper window size.

    Ported **verbatim** from
    ``dense_cnn_restnet.replay.compute_katago_window_rows``: same float operation
    order, same ``int()`` truncation (NOT ``round``). As ``usable_rows ->
    min_rows`` the window collapses to ``min_rows``; ``taper_window_exponent < 1``
    gives the sublinear KataGo taper. The caller clamps ``max(window, min_rows)``.
    """
    offset = float(taper_window_scale if taper_window_scale is not None else min_rows)
    power_law_x = float(usable_rows) - float(min_rows) + offset
    unscaled = power_law_x ** taper_window_exponent - offset ** taper_window_exponent
    scaled = unscaled / (taper_window_exponent * (offset ** (taper_window_exponent - 1.0)))
    return int(scaled * expand_window_per_row + float(min_rows))


def keep_prob(used_rows: int, keep_target_rows: int) -> float:
    """Uniform-subsample probability toward ``keep_target_rows``.

    Ported verbatim from ``replay.keep_prob``:
    ``min(keep_target_rows, used_rows) / used_rows``. ``1.0`` when the window is
    already at or below the target (no subsample); else the down-sample ratio.
    ``used_rows`` is always > 0 at the call site (an empty window is rejected
    earlier), but guard against a zero divide defensively.
    """
    if used_rows <= 0:
        return 1.0
    return min(float(keep_target_rows), float(used_rows)) / float(used_rows)


def select_recent_window(
    entries: Sequence["ShardEntry"], desired_rows: int
) -> tuple[list["ShardEntry"], int]:
    """Newest->oldest whole-shard accumulation until ``used_rows >= desired_rows``.

    Ported from ``replay._select_recent_window``. dense's ``files`` are
    mtime-ascending, so ``reversed`` walks newest-first; hexfield's
    ``entries`` arrive **(generation, game_key)-ascending** from the manifest, so
    ``reversed`` likewise walks newest-first (mtime-free — that is the whole point
    of the port). Whole-shard granularity overshoots ``desired_rows`` by < one
    shard. The selected list is re-sorted ascending on return.
    """
    selected: list["ShardEntry"] = []
    used_rows = 0
    for info in reversed(entries):
        selected.append(info)
        used_rows += int(info.rows)
        if used_rows >= desired_rows:
            break
    selected.reverse()
    return selected, used_rows


def _md5_path_fraction(value: str) -> float:
    """Stable [0, 1) fraction from the md5 of a path.

    Ported verbatim from ``replay._md5_path_fraction``: the first 13 hex digits
    of ``md5(value)`` as an int over ``2**52``. Seed-
    independent (a pure function of the path string), so the train/val partition
    and any md5 sub-range filter are stable across epochs and runs.
    """
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:13]
    return int("0x" + digest, 16) / float(2**52)


def _split_by_md5(
    selected: Sequence["ShardEntry"],
    *,
    validation_fraction: float,
) -> tuple[list["ShardEntry"], list["ShardEntry"]]:
    """Per-file md5 train/val split.

    Ported from ``replay._split_by_md5``, keyed on ``str(entry.rel_path)`` (the
    portable, stable shard key) where dense keys on ``str(info.path)``.
    ``validation_fraction <= 0`` ⇒ all-train,
    empty val (the hexfield default). Otherwise a file goes to val iff its md5
    fraction is ``>= 1 - validation_fraction`` (a fixed, path-stable cut).
    """
    if validation_fraction <= 0.0:
        return list(selected), []
    train_upper = 1.0 - float(validation_fraction)
    train_infos: list["ShardEntry"] = []
    val_infos: list["ShardEntry"] = []
    for info in selected:
        fraction = _md5_path_fraction(str(info.rel_path))
        if fraction < train_upper:
            train_infos.append(info)
        else:
            val_infos.append(info)
    return train_infos, val_infos


def _select_files_for_rows(
    entries: Sequence["ShardEntry"],
    requested_rows: int,
    rng: np.random.Generator,
) -> tuple[list["ShardEntry"], int]:
    """Overshoot-skip single-pass file selection capped near ``requested_rows``.

    Ported from ``trainer._select_files_for_rows``. dense reads each candidate's
    row count via ``npz_row_count(path)``; hexfield reads it straight off
    ``ShardEntry.rows`` (already in the manifest — no re-stat). The overshoot-skip
    logic is identical: shuffle the candidates, greedily accumulate, and a shard
    that would overshoot is *probabilistically* skipped (``skip_prob = overshoot
    / row_count``) and deferred; deferred shards are added back if still short.
    Unbiasedly lands near (not far past) ``requested_rows``.

    Determinism: ``rng`` is the caller's pre-seeded ``np.random.default_rng(seed +
    epoch*65537)``; every draw happens here on the main thread.
    """
    candidates: list[tuple["ShardEntry", int]] = [(info, int(info.rows)) for info in entries]
    rng.shuffle(candidates)
    selected: list["ShardEntry"] = []
    deferred: list[tuple["ShardEntry", int]] = []
    rows = 0
    for info, row_count in candidates:
        if rows > 0 and rows + row_count > requested_rows:
            overshoot = rows + row_count - requested_rows
            skip_prob = min(1.0, max(0.0, overshoot / max(1, row_count)))
            if rng.random() < skip_prob:
                deferred.append((info, row_count))
                continue
        selected.append(info)
        rows += row_count
        if rows >= requested_rows:
            return selected, rows
    for info, row_count in deferred:
        selected.append(info)
        rows += row_count
        if rows >= requested_rows:
            break
    return selected, rows


def build_window_split(
    selected: Sequence["ShardEntry"],
    *,
    keep_prob: float,
    rng: np.random.Generator,
    samples_dir: Path,
) -> PackedWindow:
    """Load the selected shards, per-row Bernoulli subsample, and concat into one
    packed in-RAM window.

    The hexfield equivalent of dense's ``_build_compact_split`` minus the disk
    re-shard: hexfield keeps the window PACKED in RAM, so there is no
    ``data*.npz`` write and no fixed-batch alignment here — the permute +
    ``effective_rows`` truncation live in the consumer.

    Subsample fidelity: the per-row keep is an independent ``Bernoulli(keep_prob)``
    drawn from the **single shared** ``rng`` consumed in deterministic
    ``(generation, game_key)`` shard order (the manifest order), and within a
    shard in stored row order — exactly ``rng.random(len(shard)) < keep_prob`` per
    shard. ``keep_prob >= 1.0`` keeps every row with no RNG draw, so the stream is
    identical whether or not a subsample is needed.

    Memory: survivors are concatenated with the streaming :func:`concat_packed`
    (pre-size + fill + free-each-part), so the transient peak stays ~1x the final
    window plus one shard rather than ~2x.
    """
    # Consume the keep mask in deterministic (generation, game_key) order so the
    # single shared rng stream is reproducible regardless of how `selected` was
    # ordered upstream. select_recent_window already returns ascending order, but
    # we re-sort defensively to pin the contract.
    ordered = sorted(selected, key=lambda e: (int(e.generation), int(e.game_key)))

    survivors: list[PackedWindow] = []
    for entry in ordered:
        shard = load_packed_shard(samples_dir / entry.rel_path)
        if keep_prob >= 1.0:
            survivors.append(shard)
            continue
        # Independent per-row Bernoulli(keep_prob). One vectorized draw per shard,
        # in stored row order (rng.random releases the GIL internally).
        mask = rng.random(shard.n) < keep_prob
        survivors.append(_subset_packed(shard, mask))

    return concat_packed(survivors)


def _subset_packed(window: PackedWindow, mask: np.ndarray) -> PackedWindow:
    """Return a new :class:`PackedWindow` keeping only the rows where ``mask`` is
    True, rebuilding every CSR group's offsets/data for the survivor rows.

    Used by :func:`build_window_split` for the keep_prob subsample. The kept count
    is ``int(mask.sum())``; an all-False mask yields a valid empty window
    (:func:`concat_packed` tolerates and skips it). Block/scalar columns slice
    directly; CSR columns are rebuilt by walking the kept rows and copying each
    row's flat segment (qr groups copy pair-doubled segments and rebuild the
    pair-counting offsets).
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(np.bool_)
    if mask.shape[0] != window.n:
        raise ValueError(f"_subset_packed: mask length {mask.shape[0]} != window.n {window.n}")
    keep_idx = np.nonzero(mask)[0]
    kept = int(keep_idx.shape[0])
    if kept == 0:
        return PackedWindow.empty()
    if kept == window.n:
        return window  # nothing dropped

    c = window.cols
    out: dict[str, np.ndarray] = {}
    # Scalars + blocks: fancy-index the kept rows.
    for name in SCALAR_COLS:
        out[name] = np.ascontiguousarray(c[name][keep_idx])
    for name in BLOCK_COLS:
        out[name] = np.ascontiguousarray(c[name][keep_idx, :])

    # CSR groups: rebuild offsets + data over the kept rows. Each distinct offsets
    # array is rebuilt once; its governed data arrays (qr-doubled or not) are
    # gathered alongside.
    for off in OFF_COLS:
        src_off = c[off]
        datas = _OFF_TO_DATA[off]
        # New offsets: cumulative kept segment lengths (in *group units*, i.e.
        # pairs for qr-doubled groups since src_off already counts pairs).
        seg_lens = (src_off[keep_idx + 1] - src_off[keep_idx]).astype(np.int64)
        new_off = np.empty(kept + 1, dtype=np.int64)
        new_off[0] = 0
        np.cumsum(seg_lens, out=new_off[1:])
        out[off] = new_off
        for d, doubled in datas:
            src = c[d]
            tot = int(new_off[kept])
            elems = 2 * tot if doubled else tot
            dst = np.empty(elems, dtype=_CSR_DTYPES[d])
            wcur = 0
            for row in keep_idx:
                a = int(src_off[row])
                b = int(src_off[row + 1])
                if doubled:
                    seg = src[2 * a : 2 * b]
                else:
                    seg = src[a:b]
                m = seg.shape[0]
                dst[wcur : wcur + m] = seg
                wcur += m
            assert wcur == elems, f"_subset_packed: {d} fill {wcur} != {elems}"
            out[d] = dst

    return PackedWindow(
        n=kept,
        cols=out,
        horizons=window.horizons,
        generation=np.ascontiguousarray(window.generation[keep_idx]),
        row_shard_id=np.ascontiguousarray(window.row_shard_id[keep_idx]),
    )
