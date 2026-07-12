"""Lazy chunked train-phase expansion == legacy upfront expansion.

The 2026-07-11 lazy-expand rewrite of ``trainer.train_passes`` (production
``tolerate_off_legal=False`` path) replaces the expand-everything-upfront step
with per-chunk expansion via ``make_chunk_expander``: the survivor permutation
is drawn over ``range(window.n)`` directly (valid == all-True is guaranteed in
this mode because an off-legal row raises in every backend), and chunks are an
exact multiple of ``batch_rows``. The contract this suite pins: the SEQUENCE OF
OPTIMIZER BATCHES — each batch's ``ExpandedRow`` contents in order — is
identical to the legacy shape for any (seed, d6, batch_rows, chunk multiple,
effective_rows truncation).

Pure CPU + tmp-dir IO (serial backend); a needs_rust leg pins the rust
expander closure against both expand_rows(rust) and the serial reference.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest

from hexfield_eq.buffer_manifest import ShardEntry
from hexfield_eq.expand_backends import expand_rows, make_chunk_expander
from hexfield_eq.features import build_position
from hexfield_eq.geometry import pack_action_id, unpack_action_id
from hexfield_eq.samples import STV_HORIZONS, ExpandedRow, HexfieldSampleData
from hexfield_eq.shards import write_compact_shard
from hexfield_eq.window import build_window_split

try:
    from hexfield_eq import _rust
except ImportError:  # pragma: no cover
    _rust = None

needs_rust = pytest.mark.skipif(
    _rust is None, reason="hexfield_eq._rust not built (Windows CPU lane)"
)


# --- synthetic shard fixtures (same shape as test_hexfield_eq_window_streaming) --


def _legal_ids(stub: HexfieldSampleData) -> list[int]:
    """The position's legal action ids, from the SAME machinery expansion uses
    (build_position over the row's facts; legal slots are the sup prefix)."""
    sup, _feats = build_position(stub.facts())
    return sorted(
        int(pack_action_id(q, r))
        for (q, r), slot in sup.index.items()
        if slot < sup.legal_count
    )


def _rows(seed: int, n_moves: int, game_id: str) -> list[HexfieldSampleData]:
    # Policy/gumbel/opp targets are drawn from the position's OWN legal set
    # (expansion hard-errors on off-legal targets at the default support
    # radius), and stones are placed on legal cells, so every synthetic row
    # expands cleanly by construction.
    rng = random.Random(seed)
    placed: list[tuple[int, int, int, int]] = []
    rows: list[HexfieldSampleData] = []
    for turn in range(n_moves):
        stub = HexfieldSampleData(
            game_id=game_id,
            turn_index=turn,
            current_player=turn % 2,
            phase="Opening" if turn == 0 else "FirstStone",
            records=tuple(placed),
            first_stone=None,
            policy=(),
        )
        legal = _legal_ids(stub)
        if not legal:
            break
        k = min(1 + rng.randrange(4), len(legal))
        support = rng.sample(legal, k)
        raw = [rng.random() + 1e-3 for _ in support]
        tot = sum(raw)
        policy = tuple((a, w / tot) for a, w in zip(support, raw))
        gumbel = ()
        if rng.random() < 0.67:
            g_support = support[: max(1, k - rng.randrange(2))]
            g_raw = [rng.random() + 1e-3 for _ in g_support]
            g_tot = sum(g_raw)
            gumbel = tuple((a, w / g_tot) for a, w in zip(g_support, g_raw))
        rows.append(
            HexfieldSampleData(
                game_id=game_id,
                turn_index=turn,
                current_player=turn % 2,
                phase=stub.phase,
                records=tuple(placed),
                first_stone=None,
                policy=policy,
                q_policy=tuple((a, rng.uniform(-1.0, 1.0)) for a in support),
                prior_logit=tuple((a, rng.uniform(-4.0, 4.0)) for a in support),
                gumbel_policy=gumbel,
                opp_policy=tuple((a, 1.0 / k) for a in support) if rng.random() < 0.5 else (),
                short_term_value=tuple(
                    (int(h), rng.uniform(-1.0, 1.0)) for h in STV_HORIZONS if rng.random() < 0.7
                ),
                value=rng.uniform(-1.0, 1.0),
                moves_left=float(rng.randrange(80)),
                policy_surprise=rng.random() * 3.0,
            )
        )
        q, r = unpack_action_id(rng.choice(legal))
        placed.append((q, r, turn % 2, turn))
    assert rows
    return rows


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    samples_dir = tmp_path_factory.mktemp("samples")
    entries = []
    for i, n in enumerate((11, 17, 8, 14)):
        game_key = 1_000_000 + i
        rel = f"epoch_000001/game_{game_key}.npz"
        write_compact_shard(samples_dir / rel, _rows(50 + i, n, f"g{i}"), sidecar={"epoch": 1})
        entries.append(ShardEntry(rel_path=rel, rows=n, generation=1, game_key=game_key))
    return build_window_split(
        entries, keep_prob=1.0, rng=np.random.default_rng(0), samples_dir=samples_dir
    )


# --- equivalence harness ---------------------------------------------------------


def _assert_equal_value(
    va, vb, ctx: str,
    ignore_suffixes: tuple[str, ...] = (),
    float_rtol: float | None = None,
) -> None:
    if any(ctx.endswith(s) for s in ignore_suffixes):
        return
    if isinstance(va, np.ndarray):
        if float_rtol is not None and va.dtype.kind == "f":
            np.testing.assert_allclose(va, vb, rtol=float_rtol, atol=1e-7, err_msg=ctx)
        else:
            np.testing.assert_array_equal(va, vb, err_msg=ctx)
    elif dataclasses.is_dataclass(va) and not isinstance(va, type):
        assert type(va) is type(vb), f"{ctx}: {type(va)} != {type(vb)}"
        for f in dataclasses.fields(va):
            _assert_equal_value(
                getattr(va, f.name), getattr(vb, f.name), f"{ctx}.{f.name}",
                ignore_suffixes, float_rtol,
            )
    elif isinstance(va, dict):
        assert set(va.keys()) == set(vb.keys()), f"{ctx}: key sets differ"
        for k in va:
            _assert_equal_value(va[k], vb[k], f"{ctx}[{k!r}]", ignore_suffixes, float_rtol)
    elif isinstance(va, (list, tuple)):
        assert len(va) == len(vb), f"{ctx}: length {len(va)} != {len(vb)}"
        for i, (x, y) in enumerate(zip(va, vb)):
            _assert_equal_value(x, y, f"{ctx}[{i}]", ignore_suffixes, float_rtol)
    elif isinstance(va, float) and float_rtol is not None:
        assert va == pytest.approx(vb, rel=float_rtol, abs=1e-7), f"{ctx}: {va!r} != {vb!r}"
    else:
        assert va == vb, f"{ctx}: {va!r} != {vb!r}"


def _assert_rows_equal(
    a: ExpandedRow, b: ExpandedRow, ctx: str,
    ignore_suffixes: tuple[str, ...] = (),
    float_rtol: float | None = None,
) -> None:
    _assert_equal_value(a, b, ctx, ignore_suffixes, float_rtol)


def _legacy_batches(window, d6, *, seed_perm, batch_rows, effective_rows):
    """Verbatim legacy shape: expand ALL -> filter valid -> permute -> truncate
    -> slice into nominal batches."""
    expanded_rows, valid = expand_rows(
        window, None, d6, STV_HORIZONS, tolerate_off_legal=False, backend="serial"
    )
    survivors = [row for row, ok in zip(expanded_rows, valid) if ok]
    perm = np.random.default_rng(seed_perm).permutation(len(survivors))
    keep = perm[: max(0, int(effective_rows))]
    ordered = [survivors[int(j)] for j in keep]
    return [ordered[s : s + batch_rows] for s in range(0, len(ordered), batch_rows)]


def _lazy_batches(window, d6, *, seed_perm, batch_rows, chunk_batches, effective_rows):
    """The trainer's lazy production path, extracted verbatim."""
    perm = np.random.default_rng(seed_perm).permutation(int(window.n))
    keep = perm[: max(0, int(effective_rows))]
    expander = make_chunk_expander(
        window, STV_HORIZONS, tolerate_off_legal=False, backend="serial"
    )
    chunk_rows = batch_rows * max(1, int(chunk_batches))
    batches = []
    for cstart in range(0, len(keep), chunk_rows):
        cidx = keep[cstart : cstart + chunk_rows]
        crows, cvalid = expander(cidx, d6[cidx])
        assert bool(np.all(cvalid))
        for bstart in range(0, len(crows), batch_rows):
            batches.append(crows[bstart : bstart + batch_rows])
    return batches


@pytest.mark.parametrize("chunk_batches", [1, 3, 100])
@pytest.mark.parametrize("effective_rows", [10_000, 30, 0])
def test_lazy_batches_equal_legacy(window, chunk_batches, effective_rows) -> None:
    d6 = np.random.default_rng(11).integers(0, 12, size=int(window.n), dtype=np.int64)
    want = _legacy_batches(
        window, d6, seed_perm=77, batch_rows=8, effective_rows=effective_rows
    )
    got = _lazy_batches(
        window, d6, seed_perm=77, batch_rows=8,
        chunk_batches=chunk_batches, effective_rows=effective_rows,
    )
    assert len(got) == len(want)
    for bi, (gb, wb) in enumerate(zip(got, want)):
        assert len(gb) == len(wb), f"batch {bi} size"
        for ri, (gr, wr) in enumerate(zip(gb, wb)):
            _assert_rows_equal(gr, wr, f"batch{bi}.row{ri}")


def test_identity_d6_matches_production_draw(window) -> None:
    """GROUP_ORDER > 1 production zeroes d6; equivalence must hold there too."""
    d6 = np.zeros(int(window.n), dtype=np.int64)
    want = _legacy_batches(window, d6, seed_perm=3, batch_rows=16, effective_rows=999)
    got = _lazy_batches(
        window, d6, seed_perm=3, batch_rows=16, chunk_batches=2, effective_rows=999
    )
    assert len(got) == len(want)
    for bi, (gb, wb) in enumerate(zip(got, want)):
        for ri, (gr, wr) in enumerate(zip(gb, wb)):
            _assert_rows_equal(gr, wr, f"batch{bi}.row{ri}")


@needs_rust
def test_rust_chunk_expander_matches_expand_rows(window, tmp_path) -> None:
    """The hoisted-columns rust closure == one-shot expand_rows(rust) == serial,
    on an arbitrary row subset — and the closure must keep working after the
    window's numpy columns are dropped (the trainer's lazy path clears them
    once the closure owns its serialized copy)."""
    idx = np.asarray([3, 0, 17, 9, 25, 1], dtype=np.int64)
    idx = idx[idx < int(window.n)]
    d6 = np.random.default_rng(4).integers(0, 12, size=idx.shape[0], dtype=np.int64)
    # Reference results BEFORE building the closure (fresh loads of the same
    # shards would work too; the module-scoped window is shared, so re-derive
    # a private copy for the destructive part below).
    import copy as _copy

    win2 = _copy.copy(window)
    win2.cols = dict(window.cols)
    exp = make_chunk_expander(win2, STV_HORIZONS, tolerate_off_legal=False, backend="rust")
    # Destructive: the trainer clears the columns after building the expander.
    win2.cols.clear()
    got_rows, got_valid = exp(idx, d6)
    ref_rows, ref_valid = expand_rows(
        window, idx, d6, STV_HORIZONS, tolerate_off_legal=False, backend="rust"
    )
    ser_rows, ser_valid = expand_rows(
        window, idx, d6, STV_HORIZONS, tolerate_off_legal=False, backend="serial"
    )
    np.testing.assert_array_equal(got_valid, ref_valid)
    np.testing.assert_array_equal(got_valid, ser_valid)
    for i in range(len(idx)):
        # Closure vs one-shot rust: same kernel + reassembly => strict equality.
        _assert_rows_equal(got_rows[i], ref_rows[i], f"rust-closure vs rust row{i}")
        # Rust vs serial: three known pre-existing representational
        # differences — the rust reassembly leaves support.index empty (slot
        # mapping already happened in-kernel), the rust kernel eagerly builds
        # raylen while serial defers it to collate, and scalar floats round
        # through f32 in the kernel (~1e-7 rel). Everything else must match.
        _assert_rows_equal(
            got_rows[i], ser_rows[i], f"rust-closure vs serial row{i}",
            ignore_suffixes=(".support.index", ".raylen"),
            float_rtol=1e-5,
        )
