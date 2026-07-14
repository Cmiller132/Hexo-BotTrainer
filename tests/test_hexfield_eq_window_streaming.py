"""Streaming ``build_window_split`` == legacy accumulate-then-``concat_packed``.

The 2026-07-11 streaming rewrite of ``hexfield_eq.window.build_window_split``
(two passes: size+mask, then preallocate+fill) exists purely to cut the
select-phase RAM transient from ~2x the window to ~1x + one shard — the 2x
shape OOM'd the main_2 train driver. The output contract is BIT-IDENTICAL
arrays to the legacy shape, including the rng stream consumed for the
keep_prob masks (drawn once per cleanly-loading shard, in
``(generation, game_key)`` order). This suite pins that equivalence against a
verbatim copy of the legacy algorithm built from the still-exported primitives
(``load_packed_shard`` + ``_subset_packed`` + ``concat_packed``).

Pure CPU + tmp-dir IO; no GPU, no rust, no live run touched.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hexfield_eq.buffer_manifest import ShardEntry
from hexfield_eq.geometry import pack_action_id
from hexfield_eq.samples import STV_HORIZONS, HexfieldSampleData
from hexfield_eq.shards import write_compact_shard
from hexfield_eq.window import (
    PackedWindow,
    _subset_packed,
    build_window_split,
    concat_packed,
    load_packed_shard,
)


# --- row / shard fixtures -------------------------------------------------------


def _rows(seed: int, n_moves: int, game_id: str) -> list[HexfieldSampleData]:
    """Synthetic engine-free rows with every CSR group populated on a varying
    subset of rows: growing history records, multi-action policy (+ parallel
    q_policy / prior_logit), gumbel π' on its own support, opponent policy, and
    short-term values. No hexo_engine dependency, so this runs on the
    Windows-python CPU lane too."""
    rng = random.Random(seed)
    coords = [(q, r) for q in range(-6, 7) for r in range(-6, 7) if abs(q + r) <= 6]
    rng.shuffle(coords)
    placed: list[tuple[int, int, int, int]] = []  # (q, r, owner, placement_index)
    rows: list[HexfieldSampleData] = []
    for turn in range(n_moves):
        legal = coords[len(placed) :]
        if not legal:
            break
        k = min(1 + rng.randrange(4), len(legal))
        support_qr = rng.sample(legal, k)
        support = [int(pack_action_id(q, r)) for q, r in support_qr]
        raw = [rng.random() + 1e-3 for _ in support]
        tot = sum(raw)
        policy = tuple((a, w / tot) for a, w in zip(support, raw))
        q_policy = tuple((a, rng.uniform(-1.0, 1.0)) for a in support)
        prior = tuple((a, rng.uniform(-4.0, 4.0)) for a in support)
        # π' present on ~2/3 of rows, sometimes on a truncated support.
        if rng.random() < 0.67:
            g_support = support[: max(1, k - rng.randrange(2))]
            g_raw = [rng.random() + 1e-3 for _ in g_support]
            g_tot = sum(g_raw)
            gumbel = tuple((a, w / g_tot) for a, w in zip(g_support, g_raw))
        else:
            gumbel = ()
        opp = tuple((a, 1.0 / k) for a in support) if rng.random() < 0.5 else ()
        stv = tuple(
            (int(h), rng.uniform(-1.0, 1.0))
            for h in STV_HORIZONS
            if rng.random() < 0.7
        )
        phase = "Opening" if turn == 0 else ("FirstStone" if turn % 2 else "SecondStone")
        rows.append(
            HexfieldSampleData(
                game_id=game_id,
                turn_index=turn,
                current_player=turn % 2,
                phase=phase,
                records=tuple(placed),
                first_stone=(placed[0][0], placed[0][1]) if placed else None,
                policy=policy,
                q_policy=q_policy,
                prior_logit=prior,
                gumbel_policy=gumbel,
                opp_policy=opp,
                short_term_value=stv,
                value=rng.uniform(-1.0, 1.0),
                moves_left=float(rng.randrange(80)),
                policy_surprise=rng.random() * 3.0,
            )
        )
        q, r = legal[0]
        placed.append((q, r, turn % 2, turn))
    assert rows, "fixture generated no rows"
    return rows


def _write_shard(samples_dir: Path, epoch: int, idx: int, samples) -> ShardEntry:
    game_key = epoch * 1_000_000 + idx
    rel = f"epoch_{epoch:06d}/game_{game_key}.npz"
    write_compact_shard(samples_dir / rel, samples, sidecar={"epoch": epoch})
    return ShardEntry(rel_path=rel, rows=len(samples), generation=epoch, game_key=game_key)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[Path, list[ShardEntry]]:
    """Six shards across two generations, varying sizes (5..24 rows)."""
    samples_dir = tmp_path_factory.mktemp("samples")
    entries = []
    sizes = (9, 24, 5, 17, 12, 8)
    for i, n in enumerate(sizes):
        epoch = 1 + (i // 3)
        entries.append(
            _write_shard(samples_dir, epoch, i % 3, _rows(1000 * epoch + i, n, f"g{i}"))
        )
    return samples_dir, entries


# --- legacy reference (verbatim pre-streaming algorithm) -------------------------


def _legacy_build(entries, *, keep_prob: float, rng, samples_dir: Path):
    ordered = sorted(entries, key=lambda e: (int(e.generation), int(e.game_key)))
    survivors: list[PackedWindow] = []
    skipped: list[str] = []
    rows_loaded = 0
    for entry in ordered:
        path = samples_dir / entry.rel_path
        try:
            shard = load_packed_shard(path)
        except Exception:
            skipped.append(str(path))
            continue
        rows_loaded += int(shard.n)
        if keep_prob >= 1.0:
            survivors.append(shard)
            continue
        mask = rng.random(shard.n) < keep_prob
        survivors.append(_subset_packed(shard, mask))
    return concat_packed(survivors), rows_loaded, skipped


def _assert_windows_identical(a: PackedWindow, b: PackedWindow) -> None:
    assert a.n == b.n
    assert a.horizons == b.horizons
    assert set(a.cols) == set(b.cols)
    np.testing.assert_array_equal(a.generation, b.generation)
    np.testing.assert_array_equal(a.row_shard_id, b.row_shard_id)
    for name in sorted(a.cols):
        av, bv = a.cols[name], b.cols[name]
        assert av.dtype == bv.dtype, name
        assert av.shape == bv.shape, name
        np.testing.assert_array_equal(av, bv, err_msg=name)


# --- equivalence ------------------------------------------------------------------


def test_keep_all_equals_legacy(corpus) -> None:
    samples_dir, entries = corpus
    diag: dict = {}
    got = build_window_split(
        entries, keep_prob=1.0, rng=np.random.default_rng(7), samples_dir=samples_dir, diag=diag
    )
    want, rows_loaded, _ = _legacy_build(
        entries, keep_prob=1.0, rng=np.random.default_rng(7), samples_dir=samples_dir
    )
    _assert_windows_identical(got, want)
    assert got.n == sum(e.rows for e in entries)
    assert diag["rows_loaded"] == rows_loaded
    assert diag["rows_post_thin"] == want.n
    assert diag["shards_selected"] == len(entries)
    assert diag["shards_skipped"] == 0
    assert diag["shards_reload_failed"] == 0


@pytest.mark.parametrize("kp", [0.6, 0.25])
def test_thinned_equals_legacy(corpus, kp) -> None:
    """keep_prob < 1: identical rng stream => identical masks => identical arrays.
    The generators' end states are compared too, so consuming extra draws (or
    fewer) than legacy cannot pass on mask luck alone."""
    samples_dir, entries = corpus
    rng_got = np.random.default_rng(1234)
    rng_want = np.random.default_rng(1234)
    got = build_window_split(
        entries, keep_prob=kp, rng=rng_got, samples_dir=samples_dir
    )
    want, _, _ = _legacy_build(
        entries, keep_prob=kp, rng=rng_want, samples_dir=samples_dir
    )
    _assert_windows_identical(got, want)
    assert 0 < got.n < sum(e.rows for e in entries)
    assert rng_got.bit_generator.state == rng_want.bit_generator.state


def test_zero_kept_shard_excluded_equals_legacy(corpus) -> None:
    """A keep_prob low enough that some shard keeps zero rows: the empty part is
    skipped in both shapes and row_shard_id enumerates the nonempty survivors
    identically."""
    samples_dir, entries = corpus
    kp = 0.05
    got = build_window_split(
        entries, keep_prob=kp, rng=np.random.default_rng(99), samples_dir=samples_dir
    )
    want, _, _ = _legacy_build(
        entries, keep_prob=kp, rng=np.random.default_rng(99), samples_dir=samples_dir
    )
    _assert_windows_identical(got, want)


def test_torn_shard_skip_equals_legacy(corpus, tmp_path) -> None:
    """A torn npz mid-list: skipped with the same warning, and the rng stream for
    the SUBSEQUENT shards stays aligned with legacy (no draw for the torn one)."""
    samples_dir, entries = corpus
    # Rebuild the corpus in a fresh dir so the module-scoped fixture stays intact.
    local = tmp_path / "samples"
    local_entries = []
    for i, e in enumerate(entries):
        src = (samples_dir / e.rel_path).read_bytes()
        dst = local / e.rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src)
        side = (samples_dir / e.rel_path).with_suffix(".json")
        if side.exists():
            dst.with_suffix(".json").write_bytes(side.read_bytes())
        local_entries.append(e)
    torn = local_entries[2]
    (local / torn.rel_path).write_bytes(b"torn by a power cut, sidecar intact")

    diag: dict = {}
    with pytest.warns(RuntimeWarning, match="unreadable shard"):
        got = build_window_split(
            local_entries, keep_prob=0.55, rng=np.random.default_rng(42),
            samples_dir=local, diag=diag,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want, rows_loaded, skipped = _legacy_build(
            local_entries, keep_prob=0.55, rng=np.random.default_rng(42), samples_dir=local
        )
    _assert_windows_identical(got, want)
    assert diag["shards_skipped"] == 1
    assert diag["shards_selected"] == len(local_entries) - 1
    assert diag["rows_loaded"] == rows_loaded == sum(
        e.rows for e in local_entries if e is not torn
    )
    assert diag["skipped_paths"] == sorted(skipped)


def test_pass2_reload_failure_compacts_shortfall(corpus, tmp_path, monkeypatch) -> None:
    """A shard that reads cleanly in pass 1 but fails on the pass-2 reload is
    dropped with a warning; the preallocated window is compacted by view-slice
    and stays fully self-consistent (offsets endpoints == data lengths, row_view
    walkable end to end)."""
    import hexfield_eq.window as W

    samples_dir, entries = corpus
    ordered = sorted(entries, key=lambda e: (int(e.generation), int(e.game_key)))
    victim_rel = ordered[1].rel_path
    real_load = W.load_packed_shard
    calls: dict[str, int] = {}

    def flaky_load(path):
        p = str(path).replace("\\", "/")
        if p.endswith(victim_rel.replace("\\", "/")):
            calls[victim_rel] = calls.get(victim_rel, 0) + 1
            if calls[victim_rel] >= 2:  # pass 1 ok, pass 2 fails
                raise OSError("simulated read failure between passes")
        return real_load(path)

    monkeypatch.setattr(W, "load_packed_shard", flaky_load)
    diag: dict = {}
    with pytest.warns(RuntimeWarning, match="became unreadable"):
        got = W.build_window_split(
            entries, keep_prob=1.0, rng=np.random.default_rng(5),
            samples_dir=samples_dir, diag=diag,
        )
    monkeypatch.undo()

    assert diag["shards_reload_failed"] == 1
    assert diag["shards_skipped"] == 1
    expected_rows = sum(e.rows for e in entries) - ordered[1].rows
    assert got.n == expected_rows == diag["rows_post_thin"]
    # Self-consistency of the compacted window: every offsets array ends exactly
    # at its data array's (possibly qr-doubled) length, and each row slices.
    for off in ("hist_off", "pol_off", "gumbel_off", "opp_off"):
        assert got.cols[off].shape[0] == got.n + 1
        assert int(got.cols[off][0]) == 0
    assert got.cols["hist_qr"].shape[0] == 2 * int(got.cols["hist_off"][got.n])
    assert got.cols["hist_owner"].shape[0] == int(got.cols["hist_off"][got.n])
    assert got.cols["pol_act"].shape[0] == int(got.cols["pol_off"][got.n])
    assert got.cols["gumbel_act"].shape[0] == int(got.cols["gumbel_off"][got.n])
    assert got.cols["opp_act"].shape[0] == int(got.cols["opp_off"][got.n])
    for i in range(got.n):
        got.row_view(i)
    # And the surviving rows equal a legacy build over the surviving shards.
    survivors = [e for e in ordered if e.rel_path != victim_rel]
    want, _, _ = _legacy_build(
        survivors, keep_prob=1.0, rng=np.random.default_rng(5), samples_dir=samples_dir
    )
    assert got.n == want.n
    for name in sorted(want.cols):
        np.testing.assert_array_equal(got.cols[name], want.cols[name], err_msg=name)


def test_pass2_shard_swap_detected(corpus, tmp_path) -> None:
    """A shard REPLACED between passes with same-row-count different-CSR content
    is rejected (not silently mixed): swap in a same-n shard with different
    policy support sizes via a monkeypatched loader-level file redirect."""
    import hexfield_eq.window as W

    samples_dir, entries = corpus
    ordered = sorted(entries, key=lambda e: (int(e.generation), int(e.game_key)))
    victim = ordered[0]
    victim_pol = load_packed_shard(samples_dir / victim.rel_path).cols["pol_act"].shape[0]
    # A replacement shard with the same row count but different CSR lengths
    # (seed-searched so the pol_act length provably differs — the deep guard,
    # not the row-count check, must be what rejects it).
    replacement_dir = tmp_path / "swap"
    replacement = None
    for seed in range(777, 787):
        cand_rows = _rows(seed, victim.rows, "swap")
        if len(cand_rows) != victim.rows:
            continue
        cand = _write_shard(replacement_dir, victim.generation, 900 + seed, cand_rows)
        if load_packed_shard(
            replacement_dir / cand.rel_path
        ).cols["pol_act"].shape[0] != victim_pol:
            replacement = cand
            break
    assert replacement is not None, "no same-n different-CSR replacement found"
    assert replacement.rows == victim.rows
    real_load = W.load_packed_shard
    state = {"count": 0}

    def swapping_load(path):
        p = str(path).replace("\\", "/")
        if p.endswith(victim.rel_path.replace("\\", "/")):
            state["count"] += 1
            if state["count"] >= 2:  # pass 2 sees different content
                swapped = real_load(replacement_dir / replacement.rel_path)
                return swapped
        return real_load(path)

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setattr(W, "load_packed_shard", swapping_load)
    try:
        with pytest.warns(RuntimeWarning, match="changed between"):
            got = W.build_window_split(
                entries, keep_prob=1.0, rng=np.random.default_rng(6),
                samples_dir=samples_dir,
            )
    finally:
        mp.undo()
    assert got.n == sum(e.rows for e in entries) - victim.rows
    for i in range(got.n):
        got.row_view(i)


def test_empty_selection_returns_empty_window() -> None:
    got = build_window_split(
        [], keep_prob=1.0, rng=np.random.default_rng(0), samples_dir=Path(".")
    )
    assert got.n == 0
    assert isinstance(got, PackedWindow)


def test_row_view_roundtrip(corpus) -> None:
    """Spot-check row_view slicing on the streamed window against a straight
    per-shard load (CSR rebasing correctness beyond raw array equality)."""
    samples_dir, entries = corpus
    window = build_window_split(
        entries, keep_prob=1.0, rng=np.random.default_rng(3), samples_dir=samples_dir
    )
    ordered = sorted(entries, key=lambda e: (int(e.generation), int(e.game_key)))
    row = 0
    for entry in ordered:
        shard = load_packed_shard(samples_dir / entry.rel_path)
        for i in range(shard.n):
            got_v = window.row_view(row)
            want_v = shard.row_view(i)
            assert got_v.policy() == want_v.policy()
            assert got_v.gumbel_policy() == want_v.gumbel_policy()
            assert got_v.records() == want_v.records()
            assert got_v.opp_policy() == want_v.opp_policy()
            np.testing.assert_array_equal(got_v.stvalue, want_v.stvalue)
            assert got_v.generation == want_v.generation
            row += 1
    assert row == window.n


def test_window_preserves_tss_proof_metadata(tmp_path) -> None:
    """The packed window must carry pol_class / tss_proof / target_regime
    through load + concat + row_view (they were silently dropped pre-review):
    proof provenance and target-semantics tags survive the replay round trip."""
    samples_dir = tmp_path / "samples"
    base = _rows(4242, 6, "gproof")
    tagged = []
    for i, s in enumerate(base):
        support = [a for a, _ in s.policy]
        # Class the first support action as a proven winner on even rows.
        pol_class = ((support[0], 1),) if i % 2 == 0 else ()
        tagged.append(
            replace(s, policy_class=pol_class, tss_proof=(1 if i % 2 == 0 else 0))
        )
    entries = [_write_shard(samples_dir, 1, 0, tagged)]
    # Also a regime-1 (sharpened) shard to prove mixed windows stay labeled.
    from hexfield_eq.shards import write_compact_shard as _w

    rel2 = "epoch_000001/game_1000001.npz"
    _w(samples_dir / rel2, base, sidecar={"epoch": 1}, target_regime=1)
    entries.append(
        ShardEntry(rel_path=rel2, rows=len(base), generation=1, game_key=1000001)
    )

    window = build_window_split(
        entries, keep_prob=1.0, rng=np.random.default_rng(5), samples_dir=samples_dir
    )
    assert window.n == len(tagged) + len(base)
    for name in ("tss_proof", "target_regime", "pol_class"):
        assert name in window.cols, name
    for i, s in enumerate(tagged):
        v = window.row_view(i)
        assert v.tss_proof == (1 if i % 2 == 0 else 0)
        assert v.target_regime == 0
        assert v.policy_class() == tuple((int(a), int(c)) for a, c in s.policy_class)
    for j in range(len(base)):
        v = window.row_view(len(tagged) + j)
        assert v.target_regime == 1
        assert v.policy_class() == ()
