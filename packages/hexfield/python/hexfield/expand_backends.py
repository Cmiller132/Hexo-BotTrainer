"""Train-read row expansion backends (PLAN §4.3 fallback ladder).

The dominant per-epoch train-read cost is the pure-Python ``expand_sample`` loop
(D6 transform + depth-9 support BFS + feature build + legal-slot policy
projection — ``samples.py``). This module factors that loop out of
``trainer.train_passes`` and dispatches it across a configurable backend so the
GPU is fed instead of starved.

Backends (``backend=`` argument / ``config.training.expand_backend`` /
``HEXFIELD_EXPAND`` env), the PLAN §4.3 ladder:

* ``"serial"`` — the Phase-5 path, refactored verbatim here: one
  ``expand_sample`` call per window row on the main thread. The PARITY ORACLE and
  the safe default.
* ``"pool"`` — dense's exact pattern (``dense_cnn_restnet/trainer.py:320-355``):
  a persistent ``ProcessPoolExecutor(mp_context="spawn")``; the unit of work is
  one contiguous CHUNK of rows shipped to the TOP-LEVEL picklable
  :func:`expand_chunk_to_arrays`; ``workers+2`` inflight; ``wait(FIRST_COMPLETED)``;
  serial below :data:`_PARALLEL_MIN_ROWS`. Strictly inferior to the eventual Rust
  kernel on this host (spawn is mandatory on WSL/Windows) but proven and low-risk.
* ``"rust"`` — the Phase-7 production path: the GIL-free rayon kernel
  (``replay_expand.rs::expand_shard_train``). One parallel call expands the whole
  window under the pre-drawn D6, returning zero-copy buffers reassembled here into
  the SAME ``(rows, valid)`` shape as serial/pool (Rust↔serial element-wise parity
  is gated by ``tests/katago_buffer/test_p7_rust_parity.py``). ``workers``/``pool``
  are ignored (rayon owns its thread pool).

DETERMINISM (the load-bearing contract, PLAN §4.4/§4.5): **all randomness is
pre-drawn on the main thread** (the per-row ``d6`` vector) and passed positionally
into the workers. Workers are PURE functions — they import only numpy / the
torch-free hexfield expansion chain (``samples`` → ``features`` / ``geometry`` /
``support``), NEVER torch, and make NO ``rng`` call. Results are reassembled in
the ORIGINAL row order (chunks are written back at their start offset), so the
output is element-for-element identical to the serial path regardless of worker
count or future-completion order. ``expand_rows`` therefore satisfies, by
construction: ``pool == serial`` and ``workers=1 == workers=N``.

OFF-LEGAL (PLAN §4.5 step 1): an off-legal row (``expand_sample`` raises
"... off the legal set ...") is flagged INVALID in the returned ``valid`` mask —
NOT dropped in-worker. The caller (``train_passes``) does the survivor filter →
permute → truncate on the main thread (steps 2-5), so the survivor SET is
identical across backends. A non-off-legal ``ValueError`` (e.g. zero policy mass)
propagates exactly as in the serial v1 path.

This module is intentionally torch-free so a spawn worker re-importing it stays
lean (no per-worker torch RSS for the expansion itself).
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Sequence

import numpy as np

from .constants import LEGAL_RADIUS, NUM_FEATURES
from .samples import STV_HORIZONS, ExpandedRow, HexfieldSampleData, expand_sample
from .shards import _PHASES
from .support import Support
from .window import PackedRowView, PackedWindow

# Columns the Rust kernel reads off the PackedWindow (PLAN §4.2). The scalar /
# block / CSR-data / CSR-offset arrays of hexfield_compact_v1 — passed as raw
# bytes (LE/native; build+train share the host) plus the explicit row count.
_RUST_SCALAR_COLS = (
    "current_player",
    "phase",
    "value",
    "moves_left",
    "first_q",
    "first_r",
    "first_present",
)
_RUST_BLOCK_COLS = ("stvalue", "stvalue_mask")
_RUST_CSR_DATA = (
    "hist_qr",
    "hist_owner",
    "hist_pidx",
    "pol_act",
    "pol_w",
    "opp_act",
    "opp_w",
    "own_hot_qr",
    "opp_hot_qr",
    "own_win_qr",
    "opp_win_qr",
)
_RUST_OFF_COLS = (
    "hist_off",
    "pol_off",
    "opp_off",
    "own_hot_off",
    "opp_hot_off",
    "own_win_off",
    "opp_win_off",
)

# Off-legal sentinel substring raised by ``expand_sample`` (``samples.py:206``).
# Mirrors the predicate the Phase-5 ``train_passes`` loop used to gate the skip.
_OFF_LEGAL_MARKER = "off the legal set"

# Expand serially below this many rows (tests / tiny epochs) to skip the
# spawn-pool startup + pickling overhead. Mirrors dense's ``_PARALLEL_MIN_ROWS``
# (``dense_cnn_restnet/trainer.py:584``).
_PARALLEL_MIN_ROWS = 2048

# Rows per chunk shipped to a pool worker. Picked so a 32-CPU box at the default
# 100k-row epoch yields many more chunks than workers (good load balance) while
# keeping each pickle payload modest. Not a determinism knob — any positive value
# yields the same reassembled output (chunks are written back at their offset).
_CHUNK_ROWS = 256


# ---------------------------------------------------------------------------
# PackedRowView -> HexfieldSampleData shim (moved here from trainer.py so the
# pool path can build the picklable per-row facts WITHOUT importing the
# torch-pulling trainer module; trainer.py re-exports it for back-compat).
# ---------------------------------------------------------------------------
def _row_view_to_sample(view: PackedRowView) -> HexfieldSampleData:
    """Adapt a zero-copy :class:`~hexfield.window.PackedRowView` into the
    :class:`~hexfield.samples.HexfieldSampleData` that ``expand_sample`` consumes
    (the serial-phase shim, PLAN §6/Phase-5).

    The ``PackedRowView`` already exposes every field as an accessor whose return
    shape matches the dataclass (``records()`` → ``(q,r,owner,idx)`` tuples,
    ``policy()`` / ``opp_policy()`` → ``(action_id, weight)`` tuples,
    ``own_hot()`` / ``own_win()`` / … → ``(q,r)`` tuples, ``first_stone()`` →
    ``(q,r)|None``, ``short_term_value()`` → ``(horizon, value)`` tuples). The one
    representation gap is ``phase``: the packed column stores the u8 enum index
    while ``HexfieldSampleData.phase`` (and ``build_features``) want the STRING
    name — so it is mapped through ``shards._PHASES`` here exactly as
    ``read_compact_shard`` does (``shards.py:238``). ``game_id`` is unused by
    expansion and left empty; ``metadata`` defaults empty (the opp-policy source
    was already resolved at write time)."""
    return HexfieldSampleData(
        game_id="",
        turn_index=view.turn_index,
        current_player=view.current_player,
        phase=_PHASES[view.phase],
        records=view.records(),
        first_stone=view.first_stone(),
        own_hot=view.own_hot(),
        opp_hot=view.opp_hot(),
        own_win=view.own_win(),
        opp_win=view.opp_win(),
        policy=view.policy(),
        opp_policy=view.opp_policy(),
        value=view.value,
        short_term_value=view.short_term_value(),
        moves_left=view.moves_left,
    )


def _expand_one(
    sample: HexfieldSampleData,
    sym: int,
    horizons: tuple[int, ...],
    tolerate_off_legal: bool,
) -> tuple[ExpandedRow | None, bool]:
    """Expand ONE row under its pre-drawn symmetry, returning ``(row, valid)``.

    The single point where the off-legal skip is realized identically for both
    backends (PLAN §4.5): an off-legal target raises ``ValueError`` and — when
    ``tolerate_off_legal`` (a radius-transition is in effect) — the row is flagged
    invalid (``(None, False)``) rather than dropped, so the survivor SET stays a
    pure function of ``(row, sym, radius)``. Any OTHER ``ValueError`` (e.g. zero
    policy mass) propagates exactly as in the serial v1 path.
    """
    try:
        return expand_sample(sample, symmetry=int(sym), horizons=horizons), True
    except ValueError as exc:
        if tolerate_off_legal and _OFF_LEGAL_MARKER in str(exc):
            return None, False
        raise


# ---------------------------------------------------------------------------
# Top-level picklable worker unit (PLAN §4.3 rung 2): one contiguous chunk.
# ---------------------------------------------------------------------------
def expand_chunk_to_arrays(
    samples: list[HexfieldSampleData],
    d6: np.ndarray,
    horizons: tuple[int, ...],
    tolerate_off_legal: bool,
) -> tuple[list[ExpandedRow | None], list[bool]]:
    """Expand a CHUNK of pre-extracted rows to ``ExpandedRow`` objects.

    The pool's unit of work — TOP-LEVEL and picklable so it runs unchanged in a
    spawn worker. PURE: numpy + the torch-free hexfield expansion chain only, and
    NO rng (every symmetry arrives pre-drawn in ``d6``). Returns the per-row
    expanded rows (``None`` for an off-legal-skipped row) plus the aligned
    ``valid`` mask. Returning a list of (ragged-support) ``ExpandedRow`` rather
    than stacked planes is the hexfield divergence the PLAN calls out (§4.2): the
    per-row support graph is ragged and cannot be stacked into fixed arrays.

    ``samples`` and ``d6`` are positionally aligned and 1:1; the caller ships
    contiguous chunks and stitches the results back at their start offset, so the
    full result is identical to the serial expansion regardless of how the chunks
    were scheduled.
    """
    horizons = tuple(int(h) for h in horizons)
    rows: list[ExpandedRow | None] = []
    valid: list[bool] = []
    for sample, sym in zip(samples, d6):
        row, ok = _expand_one(sample, int(sym), horizons, tolerate_off_legal)
        rows.append(row)
        valid.append(ok)
    return rows, valid


# ---------------------------------------------------------------------------
# Worker-count resolution (PLAN §4.3): min(8, max(1, cpu//4)), overridable.
# ---------------------------------------------------------------------------
def resolve_expand_workers(configured: int | None = None) -> int:
    """Resolve the pool worker count (PLAN §4.3 / §7).

    Precedence: ``HEXFIELD_EXPAND_WORKERS`` env (an explicit operator override)
    > the ``configured`` value (``config.training.expand_workers``, ``0`` ⇒ auto)
    > the auto default ``min(8, max(1, cpu_count // 4))``. The ``//4`` + cap-8
    leaves headroom for the main process feeding the GPU and bounds idle-pool RAM,
    mirroring dense's ``_resolve_expand_workers`` (``trainer.py:562-579``).
    Always ``>= 1``.
    """
    env = os.environ.get("HEXFIELD_EXPAND_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if configured is not None and int(configured) > 0:
        return max(1, int(configured))
    return min(8, max(1, (os.cpu_count() or 4) // 4))


def _chunk_bounds(n: int, chunk_rows: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` chunk bounds covering ``range(n)``."""
    step = max(1, int(chunk_rows))
    return [(s, min(s + step, n)) for s in range(0, n, step)]


# ---------------------------------------------------------------------------
# Rust backend (PLAN §4.2/§4.5): the GIL-free rayon kernel + zero-copy reassembly.
# ---------------------------------------------------------------------------
def _resolve_support_radius() -> int:
    """The model support radius (``HEXFIELD_SUPPORT_RADIUS``), passed explicitly
    to the Rust kernel. Mirrors ``support.py``'s import-time read EXACTLY so the
    Rust twin and the serial Python oracle build the same support: a value in
    ``[1, HALO_DIST]`` wins, anything else falls back to ``LEGAL_RADIUS`` (==8).
    """
    raw = os.environ.get("HEXFIELD_SUPPORT_RADIUS")
    if raw is None:
        return LEGAL_RADIUS
    try:
        r = int(raw)
    except ValueError:
        return LEGAL_RADIUS
    # support.py uses int(os.environ.get(..., LEGAL_RADIUS)) with no clamp; the
    # Rust support_radius() clamps to [1, HALO_DIST]. Match support.py (Python is
    # the oracle): pass the raw int through; out-of-range values would diverge,
    # but production only ever sets 4 or 8.
    return r


def _window_columns_as_bytes(window: PackedWindow) -> dict[str, bytes]:
    """Pack the PackedWindow columns the Rust kernel needs into a ``{name: bytes}``
    dict (PLAN §4.2). Each array is forced C-contiguous + its writer dtype, then
    ``.tobytes()`` (a single bulk copy per column — the BFS/feature cost the kernel
    parallelizes dwarfs it). Offsets stay ``int64``; the kernel reinterprets bytes.
    """
    c = window.cols
    out: dict[str, bytes] = {}
    for name in _RUST_SCALAR_COLS + _RUST_BLOCK_COLS + _RUST_CSR_DATA + _RUST_OFF_COLS:
        arr = np.ascontiguousarray(c[name])
        out[name] = arr.tobytes()
    return out


def _reassemble_rust_rows(
    result: dict, horizons_len: int
) -> tuple[list[ExpandedRow | None], np.ndarray]:
    """Turn the kernel's zero-copy buffers back into the ``(rows, valid)`` shape the
    serial/pool backends return (PLAN §4.2/§4.5).

    Each per-row segment is sliced out of the flat ``coords``/``dist``/``nbr``/
    ``feats``/``policy``/``opp_policy`` buffers via the returned CSR offsets
    (``node_off`` over support nodes, ``pol_off`` over the legal prefix). An invalid
    (off-legal-flagged) row has a zero-length segment and maps to ``None`` — the
    caller's survivor filter then drops it, so the survivor SET is backend-invariant.

    The reassembled ``Support`` carries an EMPTY ``index`` dict: it is consumed only
    by ``collate_rows`` (reads ``coords``/``nbr``/``legal_count``/``num_nodes``) and
    the parity test (reads ``coords``/``nbr``/``dist``/counts) — neither touches
    ``index``, which the serial path needs only DURING expansion (already done here).
    """
    r = int(result["num_rows"])
    valid_bytes = bytes(result["valid"])
    valid = np.frombuffer(valid_bytes, dtype=np.uint8, count=r).astype(bool)

    legal_count = np.frombuffer(bytes(result["legal_count"]), dtype=np.int32, count=r)
    stone_count = np.frombuffer(bytes(result["stone_count"]), dtype=np.int32, count=r)
    halo_count = np.frombuffer(bytes(result["halo_count"]), dtype=np.int32, count=r)
    node_off = np.frombuffer(bytes(result["node_off"]), dtype=np.int64, count=r + 1)
    pol_off = np.frombuffer(bytes(result["pol_off"]), dtype=np.int64, count=r + 1)

    total_nodes = int(node_off[r])
    total_legal = int(pol_off[r])
    coords = np.frombuffer(bytes(result["coords"]), dtype=np.int32, count=2 * total_nodes).reshape(-1, 2)
    dist = np.frombuffer(bytes(result["dist"]), dtype=np.int32, count=total_nodes)
    nbr = np.frombuffer(bytes(result["nbr"]), dtype=np.int32, count=6 * total_nodes).reshape(-1, 6)
    feats = np.frombuffer(bytes(result["feats"]), dtype=np.float32, count=NUM_FEATURES * total_nodes).reshape(-1, NUM_FEATURES)
    policy = np.frombuffer(bytes(result["policy"]), dtype=np.float32, count=total_legal)
    opp_policy = np.frombuffer(bytes(result["opp_policy"]), dtype=np.float32, count=total_legal)
    # opp_coverage is f64 (matches the serial oracle's Python-float coverage exactly,
    # so a strict 1e-12 parity check holds — see replay_expand.rs RowOut::opp_coverage).
    opp_coverage = np.frombuffer(bytes(result["opp_coverage"]), dtype=np.float64, count=r)
    value = np.frombuffer(bytes(result["value"]), dtype=np.float32, count=r)
    moves_left = np.frombuffer(bytes(result["moves_left"]), dtype=np.float32, count=r)
    moves_left_mask = np.frombuffer(bytes(result["moves_left_mask"]), dtype=np.float32, count=r)
    stvalue = np.frombuffer(bytes(result["stvalue"]), dtype=np.float32, count=r * horizons_len).reshape(r, horizons_len)
    stvalue_mask = np.frombuffer(bytes(result["stvalue_mask"]), dtype=np.float32, count=r * horizons_len).reshape(r, horizons_len)

    # Per-row slices are COPIED (`.copy()`) so each ExpandedRow owns independent,
    # WRITABLE memory — matching the serial path (fresh per-row arrays) and avoiding
    # torch.from_numpy's read-only-tensor warning. `np.frombuffer` arrays are
    # read-only views over the bytes copies; a contiguous slice of them stays a
    # read-only view, so the explicit copy is required, not np.ascontiguousarray.
    rows: list[ExpandedRow | None] = []
    for k in range(r):
        if not valid[k]:
            rows.append(None)
            continue
        a, b = int(node_off[k]), int(node_off[k + 1])
        pa, pb = int(pol_off[k]), int(pol_off[k + 1])
        lc = int(legal_count[k])
        sup = Support(
            coords=coords[a:b].copy(),
            legal_count=lc,
            stone_count=int(stone_count[k]),
            halo_count=int(halo_count[k]),
            dist=dist[a:b].copy(),
            nbr=nbr[a:b].copy(),
            index={},  # see docstring — never read on the assembled (post-expand) row
        )
        rows.append(
            ExpandedRow(
                support=sup,
                feats=feats[a:b].copy(),
                policy=policy[pa:pb].copy(),
                opp_policy=opp_policy[pa:pb].copy(),
                opp_coverage=float(opp_coverage[k]),
                value=float(value[k]),
                stvalue=stvalue[k].copy(),
                stvalue_mask=stvalue_mask[k].copy(),
                moves_left=float(moves_left[k]),
                moves_left_mask=float(moves_left_mask[k]),
            )
        )
    return rows, valid


def _expand_rows_rust(
    window: PackedWindow,
    index: list[int],
    d6: np.ndarray,
    horizons: tuple[int, ...],
    tolerate_off_legal: bool,
) -> tuple[list[ExpandedRow | None], np.ndarray]:
    """Dispatch ``index`` of ``window`` to the Rust rayon kernel and reassemble.

    The per-row D6 vector is pre-drawn (passed positionally — no rng in the kernel,
    PLAN §4.4). ``horizons`` is the CONFIG horizon set (S4); the kernel copies the
    stored ``stvalue`` columns and only uses ``len(horizons)`` to slice the block.
    """
    from . import _rust  # local import: the package imports fine without the .so

    horizons_len = len(horizons)
    if window.cols["stvalue"].shape[1] != horizons_len:
        # The stored block width must equal the requested horizon count; the kernel
        # slices [:horizons_len] from each row's block. A mismatch would silently
        # misalign the STV target, so fail loudly (matches concat_packed's S4 guard).
        raise ValueError(
            f"rust backend: stvalue block width {window.cols['stvalue'].shape[1]} "
            f"!= len(horizons) {horizons_len}"
        )
    columns = _window_columns_as_bytes(window)
    row_index = np.asarray(index, dtype=np.int64)
    # d6 may be longer than the expanded set (contract: len(d6) >= len(index));
    # the kernel requires len(d6) == len(row_index), so slice to the aligned head.
    d6_i32 = np.asarray(d6, dtype=np.int32)[: row_index.shape[0]]
    result = _rust.expand_shard_train(
        columns,
        int(window.n),
        row_index.tolist(),
        d6_i32.tolist(),
        horizons_len,
        int(_resolve_support_radius()),
        bool(tolerate_off_legal),
    )
    return _reassemble_rust_rows(result, horizons_len)


# ---------------------------------------------------------------------------
# Public dispatch entry point.
# ---------------------------------------------------------------------------
def expand_rows(
    window: PackedWindow,
    survivor_index: Sequence[int] | np.ndarray | None,
    d6: np.ndarray,
    horizons: Sequence[int] = STV_HORIZONS,
    *,
    support_radius: int | None = None,
    tolerate_off_legal: bool = False,
    backend: str = "serial",
    workers: int = 0,
    pool: Any | None = None,
) -> tuple[list[ExpandedRow | None], np.ndarray]:
    """Expand a window's rows under their pre-drawn D6 symmetries (PLAN §4.3/§4.5).

    Parameters
    ----------
    window
        The packed in-RAM replay window.
    survivor_index
        The rows to expand, in expansion order. ``None`` ⇒ all rows
        ``range(window.n)`` (the Phase-5 default: expand ALL, filter later). When
        given, ``d6`` is indexed by the SAME positions (``d6[k]`` is the symmetry
        for ``survivor_index[k]``), so the result is positionally aligned to
        ``survivor_index``.
    d6
        Pre-drawn per-row symmetry vector (main-thread draw, PLAN §4.4). Length
        must cover the expanded rows (``window.n`` when ``survivor_index`` is
        ``None``; otherwise ``len(survivor_index)``).
    horizons
        STV horizons passed verbatim to ``expand_sample`` (config horizons, S4).
    support_radius
        Diagnostic only here — the support radius is read from the
        ``HEXFIELD_SUPPORT_RADIUS`` env at ``support`` import time (shared by the
        serial main thread AND every spawn worker, since the pool inherits the
        environment). Passed through for symmetry with the future Rust kernel,
        which takes it explicitly; accepted to pin the contract but not re-read.
    tolerate_off_legal
        When True, an off-legal row is flagged invalid (mask ``False``) instead of
        raising — the radius-transition behavior (PLAN §4.5).
    backend
        ``"serial"`` | ``"pool"`` | ``"rust"`` (the implemented Phase-7 rayon kernel).
    workers
        Pool worker count (``0`` ⇒ auto via :func:`resolve_expand_workers`); only
        consulted for ``backend="pool"`` when no ``pool`` is supplied.
    pool
        An optional pre-built persistent ``ProcessPoolExecutor`` (the trainer owns
        one across epochs). When ``None`` and ``backend="pool"`` needs a pool, an
        ephemeral pool is created and torn down within the call.

    Returns
    -------
    ``(rows, valid)``
        ``rows`` is the list of ``ExpandedRow`` (``None`` for an off-legal-skipped
        row), aligned 1:1 to the expansion order; ``valid`` is the ``bool``
        numpy mask of the same length. The caller applies the survivor filter,
        survivor permutation, and ``effective_rows`` truncation (PLAN §4.5
        steps 2-5).
    """
    _ = support_radius  # see docstring: env-sourced at support import, not re-read
    horizons = tuple(int(h) for h in horizons)

    if survivor_index is None:
        index = list(range(int(window.n)))
    else:
        index = [int(i) for i in survivor_index]
    n = len(index)

    d6 = np.asarray(d6)
    if d6.shape[0] < n:
        raise ValueError(
            f"expand_rows: d6 vector length {d6.shape[0]} < rows to expand {n}"
        )

    if n == 0:
        return [], np.zeros(0, dtype=bool)

    if backend == "rust":
        # The production GIL-free rayon kernel (PLAN §4.2): one parallel
        # expand_shard_train call, zero-copy buffers reassembled into the same
        # (rows, valid) shape the serial/pool paths return. Off-legal rows are
        # flagged invalid in the mask (NOT dropped in-worker, PLAN §4.5 step 1);
        # the caller does the survivor filter / permute / truncate. `workers` /
        # `pool` are ignored (rayon manages its own thread pool internally).
        return _expand_rows_rust(window, index, d6, horizons, tolerate_off_legal)
    if backend not in ("serial", "pool"):
        raise ValueError(f"unknown expand_backend {backend!r} (serial|pool|rust)")

    # --- serial: the Phase-5 path, refactored verbatim ----------------------
    # Also the small-workload fast path for the pool backend (avoid the spawn +
    # pickle cost below _PARALLEL_MIN_ROWS), exactly like dense.
    if backend == "serial" or n < _PARALLEL_MIN_ROWS:
        rows: list[ExpandedRow | None] = []
        valid = np.zeros(n, dtype=bool)
        for k, row_i in enumerate(index):
            sample = _row_view_to_sample(window.row_view(row_i))
            row, ok = _expand_one(sample, int(d6[k]), horizons, tolerate_off_legal)
            rows.append(row)
            valid[k] = ok
        return rows, valid

    # --- pool: persistent spawn ProcessPool, dense's exact dispatch ----------
    # (1) Build the picklable per-row facts on the MAIN thread (the PackedWindow
    # is in-RAM here, unlike dense which reads from a path in-worker). This keeps
    # the row->sample shim identical to the serial path, so pool == serial.
    samples = [_row_view_to_sample(window.row_view(row_i)) for row_i in index]

    owns_pool = pool is None
    if pool is None:
        n_workers = resolve_expand_workers(workers)
        import multiprocessing as mp

        pool = ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context("spawn")
        )
    else:
        n_workers = getattr(pool, "_max_workers", resolve_expand_workers(workers))

    rows = [None] * n  # type: ignore[assignment]
    valid = np.zeros(n, dtype=bool)
    bounds = _chunk_bounds(n, _CHUNK_ROWS)
    try:
        pending = iter(bounds)
        inflight: dict[Any, tuple[int, int]] = {}

        def _submit_one() -> bool:
            for start, stop in pending:
                fut = pool.submit(
                    expand_chunk_to_arrays,
                    samples[start:stop],
                    np.asarray(d6[start:stop]),
                    horizons,
                    tolerate_off_legal,
                )
                inflight[fut] = (start, stop)
                return True
            return False

        # Prime workers+2 chunks inflight (dense's depth), then refill on each
        # completion. FIRST_COMPLETED ordering is irrelevant: each chunk's rows
        # are written back at their [start, stop) offset, so the final arrays are
        # in original order no matter how the futures finished.
        for _ in range(n_workers + 2):
            if not _submit_one():
                break
        while inflight:
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                start, stop = inflight.pop(fut)
                chunk_rows, chunk_valid = fut.result()
                rows[start:stop] = chunk_rows
                valid[start:stop] = chunk_valid
                _submit_one()
    finally:
        if owns_pool:
            pool.shutdown(wait=False, cancel_futures=True)

    return rows, valid
