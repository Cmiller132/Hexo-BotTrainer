"""Phase 4 self-test — KataGo taper window + train-bucket governor +
select_training_samples rewrite (PLAN §3.1-3.8, §5, §6).

CPU-only, no GPU, no model, no live-run interaction. The live ``hexfield_main_2``
tree is read READ-ONLY (we copy a handful of real shards out); every write
(``scan_or_update_manifest`` -> ``.buffer_manifest.json``, the per-epoch select
diag, the keep_prob-subsampled window) lands ONLY under ``_scratch/`` — never
under ``runs/*``.

Gates:
  1. WINDOW-MATH BYTE PARITY vs dense (PLAN §9 test 1): ``compute_katago_window_rows``,
     ``keep_prob`` (== ``replay.py:404``), and ``_md5_path_fraction`` are
     byte-equal to the dense functions on synthetic row-count / path vectors,
     across the radius-independent knob grid.
  2. RECENT-WINDOW: ``select_recent_window`` picks the NEWEST shards covering
     ``desired_rows`` (overshoot < one shard), re-sorted ascending; parity with
     dense ``_select_recent_window`` on the same row vector.
  3. SPLIT + SELECTION: ``_split_by_md5`` matches dense (per-rel_path); the
     ``_select_files_for_rows`` overshoot-skip lands near ``requested_rows`` and
     is deterministic under a fixed rng.
  4. GOVERNOR: ``_update_train_bucket`` accrual / cap / monotone-reload branch
     match dense ``_update_train_bucket`` line-for-line on a scripted row stream.
  5. build_window_split: keep_prob=1.0 keeps every row (decode-parity preserved);
     keep_prob<1.0 yields a deterministic subset whose survivors still decode
     field-identically (the _subset_packed CSR rebuild is correct).
  6. DRY-RUN select_training_samples against a COPY of real main_2 shards with a
     minimal fake ctx/components: a plausible return dict (window_rows>0,
     effective_rows>0, monotone cumulative) AND
     ``components.shared.sample_window`` is a PackedWindow; a second call advances
     the governor deterministically.

Run:
  PYTHONPATH=packages/hexfield/python:packages/dense_cnn_restnet/python \
    python tests/katago_buffer/test_p4_windowmath.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dense_cnn_restnet.replay as dense_replay
from hexfield import shards as hex_shards
from hexfield.buffer_manifest import ShardEntry, scan_or_update_manifest
from hexfield.trainer import HexfieldTrainer
from hexfield.config import HexfieldConfig, TrainingSection
from hexfield.window import (
    PackedWindow,
    _md5_path_fraction,
    _select_files_for_rows,
    _split_by_md5,
    build_window_split,
    compute_katago_window_rows,
    keep_prob,
    load_packed_shard,
    select_recent_window,
)

LIVE_SAMPLES = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2/samples")
SCRATCH = Path(__file__).resolve().parent / "_scratch"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _fresh(name: str) -> Path:
    root = SCRATCH / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _copy_real_epochs(dst_samples: Path, epoch_names: list[str], per_epoch: int) -> int:
    """Copy (npz+json) pairs from the named live epochs into ``dst_samples``.
    READ-only on the live tree (copies out). Returns shards copied."""
    copied = 0
    for ep in epoch_names:
        src_dir = LIVE_SAMPLES / ep
        if not src_dir.exists():
            continue
        dst_dir = dst_samples / ep
        dst_dir.mkdir(parents=True, exist_ok=True)
        for npz in sorted(src_dir.glob("game_*.npz"))[:per_epoch]:
            side = npz.with_suffix(".json")
            if not side.exists():
                continue
            shutil.copy2(npz, dst_dir / npz.name)
            shutil.copy2(side, dst_dir / side.name)
            copied += 1
    return copied


class _FakeEntry:
    """Stand-in carrying just the attrs select_recent_window / _split_by_md5 /
    _select_files_for_rows touch (rows, generation, game_key, rel_path)."""

    def __init__(self, rows, generation, game_key, rel_path):
        self.rows = rows
        self.generation = generation
        self.game_key = game_key
        self.rel_path = rel_path


class _DenseInfo:
    """dense ShuffleFileInfo stand-in (rows + path) for the dense oracle calls."""

    def __init__(self, rows, path):
        self.rows = rows
        self.path = path


# ----------------------------------------------------------------------
# 1. window-math byte parity vs dense
# ----------------------------------------------------------------------


def test_window_math_parity() -> None:
    # compute_katago_window_rows across a knob grid. The call-site invariant is
    # usable_rows >= min_rows (dense guards total_rows < min_rows -> skip at
    # replay.py:386; the hexfield caller clamps max(window, min_rows)), so the
    # parity grid spans that real domain. The degenerate out-of-domain point
    # (negative base ** fractional exponent -> complex) is checked separately
    # below: ours and dense BOTH raise the identical TypeError there, which is
    # itself proof the port is byte-faithful (not a silent divergence).
    exps = [0.5, 0.65, 0.8, 1.0]
    expands = [0.2, 0.4, 0.7]
    scales = [None, 10_000.0, 20_000.0, 50_000.0]
    min_rows_grid = [1, 20_000, 100_000]
    n = 0
    for mr in min_rows_grid:
        for e in exps:
            for ex in expands:
                for sc in scales:
                    # usable_rows >= min_rows (the real domain) + the floor itself.
                    for ur in [mr, mr + 1, mr + 5_000, mr + 280_000, mr + 980_000, mr + 5_000_000]:
                        ours = compute_katago_window_rows(
                            ur,
                            min_rows=mr,
                            expand_window_per_row=ex,
                            taper_window_exponent=e,
                            taper_window_scale=sc,
                        )
                        theirs = dense_replay.compute_katago_window_rows(
                            ur,
                            min_rows=mr,
                            expand_window_per_row=ex,
                            taper_window_exponent=e,
                            taper_window_scale=sc,
                        )
                        assert ours == theirs, (
                            f"compute_katago_window_rows mismatch ur={ur} mr={mr} e={e} "
                            f"ex={ex} sc={sc}: {ours} != {theirs}"
                        )
                        assert isinstance(ours, int)
                        n += 1
    print(f"  compute_katago_window_rows: {n} grid points byte-equal dense (usable>=min domain)")

    # As usable_rows -> min_rows the window collapses to min_rows (clamp floor).
    for mr in (20_000, 100_000):
        assert compute_katago_window_rows(
            mr, min_rows=mr, expand_window_per_row=0.4, taper_window_exponent=0.65,
            taper_window_scale=20_000.0,
        ) == mr, "window must equal min_rows at the floor"

    # Degenerate (out-of-call-domain) point: ours and dense raise the SAME error.
    deg = dict(min_rows=100_000, expand_window_per_row=0.4, taper_window_exponent=0.65,
               taper_window_scale=10_000.0)
    ours_raised = dense_raised = None
    try:
        compute_katago_window_rows(0, **deg)
    except TypeError as ex:
        ours_raised = type(ex).__name__
    try:
        dense_replay.compute_katago_window_rows(0, **deg)
    except TypeError as ex:
        dense_raised = type(ex).__name__
    assert ours_raised == dense_raised == "TypeError", (
        f"degenerate-point behavior diverges: ours={ours_raised} dense={dense_raised}"
    )
    print("  compute_katago_window_rows: floor collapses to min_rows; "
          "degenerate point raises identically to dense")

    # keep_prob == replay.py:404 formula on a used/target grid.
    kpn = 0
    for used in [1, 100, 50_000, 299_999, 300_000, 300_001, 1_000_000]:
        for target in [1, 50_000, 300_000, 600_000]:
            ours = keep_prob(used, target)
            theirs = min(float(target), float(used)) / float(used)  # replay.py:404
            assert ours == theirs, f"keep_prob({used},{target}) {ours} != {theirs}"
            assert 0.0 < ours <= 1.0
            kpn += 1
    # used<=0 guard (not exercised by dense; hexfield defends a zero-divide).
    assert keep_prob(0, 100) == 1.0
    print(f"  keep_prob: {kpn} (used,target) points == replay.py:404 formula")

    # _md5_path_fraction byte-equal dense over many path strings.
    paths = [f"epoch_{e:06d}/game_{e*1_000_000 + i}.npz" for e in range(1, 30) for i in range(40)]
    paths += ["", "x", "a/b/c.npz", "/mnt/e/abs/path.npz"]
    for p in paths:
        ours = _md5_path_fraction(p)
        theirs = dense_replay._md5_path_fraction(p)
        assert ours == theirs, f"_md5_path_fraction({p!r}) {ours!r} != {theirs!r}"
        assert 0.0 <= ours < 1.0
    print(f"  _md5_path_fraction: {len(paths)} paths byte-equal dense")


# ----------------------------------------------------------------------
# 2. recent-window cut
# ----------------------------------------------------------------------


def test_select_recent_window() -> None:
    # Ascending-(generation,game_key) entries; each shard 100 rows. Newest first.
    entries = [_FakeEntry(100, g, g * 1_000_000 + i, f"e{g}/g{i}") for g in range(1, 6) for i in range(4)]
    # 20 shards * 100 = 2000 rows total. Ask for 350 -> 4 newest shards (400 rows).
    selected, used = select_recent_window(entries, 350)
    assert used == 400, f"used {used} != 400"
    assert len(selected) == 4, f"selected {len(selected)} != 4"
    # Newest 4 shards are the LAST 4 of the ascending list.
    assert [s.rel_path for s in selected] == [e.rel_path for e in entries[-4:]], "wrong shards"
    # Re-sorted ascending on return.
    keys = [(s.generation, s.game_key) for s in selected]
    assert keys == sorted(keys), "selected not ascending"

    # Parity vs dense _select_recent_window on the same row sequence (mtime asc ==
    # our (gen,game_key) asc): build dense infos in the SAME order.
    dinfos = [_DenseInfo(100, e.rel_path) for e in entries]
    dsel, dused = dense_replay._select_recent_window(dinfos, 350)
    assert dused == used and len(dsel) == len(selected), "dense recent-window parity"
    assert [d.path for d in dsel] == [s.rel_path for s in selected], "dense recent-window order parity"

    # desired larger than total -> take everything.
    sel_all, used_all = select_recent_window(entries, 999_999)
    assert used_all == 2000 and len(sel_all) == 20
    # empty entries -> empty.
    assert select_recent_window([], 100) == ([], 0)
    print(f"  select_recent_window: newest-covering cut (used=400 for 4 shards) + dense parity")


# ----------------------------------------------------------------------
# 3. md5 split + overshoot-skip selection
# ----------------------------------------------------------------------


def test_split_and_selection() -> None:
    entries = [_FakeEntry(100, g, g * 1_000_000 + i, f"epoch_{g:06d}/game_{g*1_000_000+i}.npz")
               for g in range(1, 6) for i in range(6)]

    # validation_fraction=0 -> all-train, empty val (default).
    tr, va = _split_by_md5(entries, validation_fraction=0.0)
    assert len(tr) == len(entries) and va == [], "vf=0 must be all-train"

    # vf>0 -> partition matches dense (keyed on the SAME rel_path string).
    tr2, va2 = _split_by_md5(entries, validation_fraction=0.1)
    dtr, dva = dense_replay._split_by_md5([_DenseInfo(e.rows, e.rel_path) for e in entries],
                                          validation_fraction=0.1)
    assert [e.rel_path for e in tr2] == [d.path for d in dtr], "md5 split train parity"
    assert [e.rel_path for e in va2] == [d.path for d in dva], "md5 split val parity"
    assert len(tr2) + len(va2) == len(entries)
    print(f"  _split_by_md5: vf=0 all-train; vf=0.1 partition byte-parity w/ dense "
          f"(train={len(tr2)} val={len(va2)})")

    # overshoot-skip: requested 350, shards of 100 each -> lands near 350 (>=).
    rng = np.random.default_rng(12345)
    sel, rows = _select_files_for_rows(entries, 350, rng)
    assert rows >= 350, f"selection {rows} short of requested 350"
    assert rows <= 350 + 100, f"selection {rows} overshoots by > one shard"
    # deterministic under a fixed seed.
    sel_a, rows_a = _select_files_for_rows(entries, 350, np.random.default_rng(7))
    sel_b, rows_b = _select_files_for_rows(entries, 350, np.random.default_rng(7))
    assert rows_a == rows_b and [s.rel_path for s in sel_a] == [s.rel_path for s in sel_b], \
        "selection not deterministic for fixed seed"
    # requested larger than available -> take all.
    sel_all, rows_all = _select_files_for_rows(entries, 10_000, np.random.default_rng(1))
    assert rows_all == 100 * len(entries) and len(sel_all) == len(entries)
    print(f"  _select_files_for_rows: lands at {rows} for req=350 (<= +1 shard), deterministic")


# ----------------------------------------------------------------------
# 3b. keep_prob double-subsample accounting (bug fix)
# ----------------------------------------------------------------------


def test_keep_prob_selection_accounting() -> None:
    """When keep_prob<1.0, select_training_samples inflates the file-selection
    request by 1/kp so build_window_split's per-row Bernoulli(kp) thinning leaves
    ~requested_rows survivors, and debits/accounts effective_rows = the post-thin
    expectation (min(requested, round(selected*kp))). This mirrors the exact
    arithmetic of the fixed select_training_samples without touching a GPU/model."""
    import math

    # Big window of 100-row shards so the inflated (1/kp) request has headroom.
    entries = [_FakeEntry(100, g, g * 1_000_000 + i, f"epoch_{g:06d}/game_{g*1_000_000+i}.npz")
               for g in range(1, 40) for i in range(40)]  # 1560 shards * 100 = 156_000 rows
    total = sum(e.rows for e in entries)
    requested = 6_000

    # --- kp == 1.0: request and accounting are BIT-IDENTICAL to the old path. ----
    kp1 = keep_prob(used_rows=requested, keep_target_rows=10_000_000)  # >= target -> 1.0
    assert kp1 == 1.0
    select_request1 = requested if kp1 >= 1.0 else int(math.ceil(requested / kp1))
    assert select_request1 == requested  # no inflation
    _sel1, selected1 = _select_files_for_rows(entries, select_request1, np.random.default_rng(1))
    eff1_new = min(requested, int(round(selected1 * kp1)))
    eff1_old = min(requested, selected1)  # pre-fix formula
    assert eff1_new == eff1_old, "kp=1.0 accounting must be bit-identical to the old path"

    # --- kp < 1.0: inflate the request, account the post-thin survivors. --------
    used = 60_000  # window larger than the keep target -> kp = target/used < 1
    kp = keep_prob(used_rows=used, keep_target_rows=30_000)
    assert kp == pytest.approx(0.5)
    select_request = requested if kp >= 1.0 else int(math.ceil(requested / kp))
    assert select_request == int(math.ceil(requested / kp)) == 12_000  # inflated by 1/kp

    sel, selected = _select_files_for_rows(entries, select_request, np.random.default_rng(2025))
    assert selected >= select_request, "inflated selection must cover the 1/kp-scaled request"

    effective_rows = min(requested, int(round(selected * kp)))
    # The expected trained rows (post-thin) should be ~requested, NOT requested*kp.
    assert effective_rows == requested, (
        f"effective_rows {effective_rows} != requested {requested}; the 1/kp inflation "
        f"should make post-thin survivors reach the target"
    )

    # Simulate build_window_split's per-shard Bernoulli(kp) thinning and confirm the
    # actually-trained survivor count lands near effective_rows (the debited value) —
    # i.e. accounting matches what will be trained in expectation, closing the bug.
    thin_rng = np.random.default_rng(2025)
    ordered = sorted(sel, key=lambda e: (int(e.generation), int(e.game_key)))
    survivors = int(sum((thin_rng.random(int(e.rows)) < kp).sum() for e in ordered))
    # perm[:effective_rows] truncation in train_passes caps the trained rows at
    # effective_rows; survivors >= effective_rows means the epoch trains exactly
    # effective_rows rows. Assert survivors is close to (and at least ~) the target.
    assert survivors == pytest.approx(requested, rel=0.1), (
        f"post-thin survivors {survivors} not within 10% of requested {requested}"
    )
    assert survivors >= effective_rows * 0.9, (
        f"survivors {survivors} far below debited effective_rows {effective_rows}"
    )

    # --- window smaller than requested: exact behavior (train what exists). -----
    small = [_FakeEntry(100, 1, i, f"epoch_000001/game_{i}.npz") for i in range(20)]  # 2000 rows
    kp_s = keep_prob(used_rows=2_000, keep_target_rows=1_000)  # 0.5
    assert kp_s == pytest.approx(0.5)
    req_s = 5_000  # more than the (thinned) window can supply
    select_request_s = int(math.ceil(req_s / kp_s))  # 10_000 > 2_000 available
    sel_s, selected_s = _select_files_for_rows(small, select_request_s, np.random.default_rng(9))
    assert selected_s == 2_000, "short window returns all rows"
    effective_s = min(req_s, int(round(selected_s * kp_s)))
    assert effective_s == 1_000, (
        f"short-window effective_rows {effective_s} != round(2000*0.5)=1000"
    )
    assert effective_s < req_s, "when the window is smaller than requested, train what exists"
    print("  keep_prob accounting: kp=1.0 bit-identical; kp=0.5 inflates request 2x, "
          f"effective_rows={requested} post-thin survivors~{survivors}; "
          "short window trains round(selected*kp)")


# ----------------------------------------------------------------------
# 4. train-bucket governor
# ----------------------------------------------------------------------


def _make_trainer(**training_overrides) -> HexfieldTrainer:
    """A trainer with a CPU config and a tiny real model/optimizer (we only call
    the pure window/governor/selection methods, never train_passes).

    HexfieldTrainer.__init__ partitions params via ``model.named_parameters()``
    for per-group grad-norm logging, so the model must be a real ``nn.Module``
    (a bare SimpleNamespace has no ``named_parameters``); the linear layer is never
    forwarded/stepped by the paths these tests exercise.
    """
    import torch

    base = dict(
        max_train_bucket_size=500_000.0,
        train_samples_per_epoch=100_000,
        max_train_bucket_per_new_data=8.0,
    )
    base.update(training_overrides)
    cfg = HexfieldConfig(device="cpu", training=TrainingSection(**base))
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return HexfieldTrainer(model=model, config=cfg, optimizer=opt)


def test_update_train_bucket() -> None:
    tr = _make_trainer()
    # Reference a fresh dense trainer-equivalent by replicating the dense math
    # directly (dense's DenseCNNTrainer needs torch model; we replicate the
    # documented formula instead, which the function is a verbatim port of).
    cap = max(500_000.0, 100_000.0)

    # Step 1: 1000 new rows -> +8000 level, watermark -> 1000.
    tr._update_train_bucket(1000, window_start=0)
    assert tr.train_state.train_bucket_level == min(cap, 0.0 + 1000 * 8.0) == 8000.0
    assert tr.train_state.train_bucket_level_at_row == 1000
    assert tr.train_state.total_num_data_rows == 1000

    # Step 2: +500 rows -> +4000 -> 12000.
    tr._update_train_bucket(1500, window_start=10)
    assert tr.train_state.train_bucket_level == 12000.0
    assert tr.train_state.train_bucket_level_at_row == 1500
    assert tr.train_state.window_start_data_row_idx == 10

    # Step 3: same total -> no change (neither branch).
    before = tr.train_state.train_bucket_level
    tr._update_train_bucket(1500, window_start=10)
    assert tr.train_state.train_bucket_level == before

    # Step 4: total DECREASES (window regenerated) -> rebase watermark, zero reload.
    tr.train_state.train_steps_since_last_reload = 5
    tr._update_train_bucket(1200, window_start=3)
    assert tr.train_state.train_bucket_level_at_row == 1200
    assert tr.train_state.train_steps_since_last_reload == 0
    assert tr.train_state.train_bucket_level == min(before, cap) == before  # level clamped, not zeroed

    # Cap: a huge accrual saturates at cap.
    tr2 = _make_trainer(max_train_bucket_size=1000.0, train_samples_per_epoch=100)
    tr2._update_train_bucket(10_000_000, window_start=0)
    assert tr2.train_state.train_bucket_level == max(1000.0, 100.0) == 1000.0
    print("  _update_train_bucket: accrual / cap / no-op / monotone-reload branches match dense")


# ----------------------------------------------------------------------
# 5. build_window_split keep_prob behaviour
# ----------------------------------------------------------------------


def test_build_window_split(samples_dir: Path, entries: list[ShardEntry]) -> None:
    # keep_prob=1.0 -> every row kept; decode-parity preserved per row.
    win_full = build_window_split(entries, keep_prob=1.0, rng=np.random.default_rng(0),
                                  samples_dir=samples_dir)
    total = sum(e.rows for e in entries)
    assert win_full.n == total, f"keep_prob=1.0 kept {win_full.n} != {total}"
    assert isinstance(win_full, PackedWindow)

    # Row-for-row decode parity of the full window vs the per-shard oracle (rows
    # are concatenated in (generation, game_key) order).
    ordered = sorted(entries, key=lambda e: (e.generation, e.game_key))
    base = 0
    checked = 0
    for e in ordered:
        oracle = hex_shards.read_compact_shard(samples_dir / e.rel_path)
        for k in range(len(oracle)):
            v = win_full.row_view(base + k)
            assert v.records() == oracle[k].records, f"records mismatch row {base+k}"
            assert v.value == oracle[k].value, f"value mismatch row {base+k}"
            assert v.policy() == oracle[k].policy, f"policy mismatch row {base+k}"
            assert v.short_term_value() == oracle[k].short_term_value, f"stv mismatch row {base+k}"
            checked += 1
        base += len(oracle)
    assert base == win_full.n
    print(f"  build_window_split keep_prob=1.0: n={win_full.n}, {checked} rows decode-parity vs oracle")

    # keep_prob<1.0 -> deterministic subset; survivors must still decode-parity.
    kp = 0.5
    win_a = build_window_split(entries, keep_prob=kp, rng=np.random.default_rng(99),
                               samples_dir=samples_dir)
    win_b = build_window_split(entries, keep_prob=kp, rng=np.random.default_rng(99),
                               samples_dir=samples_dir)
    assert win_a.n == win_b.n, f"keep_prob subsample not deterministic: {win_a.n} != {win_b.n}"
    assert 0 < win_a.n < total, f"subsample {win_a.n} not strictly between 0 and {total}"

    # Verify the SURVIVORS decode field-identically: reconstruct the exact keep
    # mask the function used (single shared rng, per-shard, (gen,game_key) order)
    # and compare survivor rows against the oracle.
    rng = np.random.default_rng(99)
    base = 0
    sv_checked = 0
    for e in ordered:
        oracle = hex_shards.read_compact_shard(samples_dir / e.rel_path)
        shard = load_packed_shard(samples_dir / e.rel_path)
        mask = rng.random(shard.n) < kp
        for k in range(len(oracle)):
            if not mask[k]:
                continue
            v = win_a.row_view(base)
            assert v.records() == oracle[k].records, f"survivor records mismatch at out-row {base}"
            assert v.value == oracle[k].value, f"survivor value mismatch at out-row {base}"
            assert v.policy() == oracle[k].policy, f"survivor policy mismatch at out-row {base}"
            assert v.short_term_value() == oracle[k].short_term_value, f"survivor stv mismatch"
            base += 1
            sv_checked += 1
    assert base == win_a.n, f"reconstructed survivor count {base} != window n {win_a.n}"
    print(f"  build_window_split keep_prob=0.5: deterministic n={win_a.n}; "
          f"{sv_checked} survivors decode-parity (CSR rebuild correct)")


# ----------------------------------------------------------------------
# 6. dry-run select_training_samples against copied real shards
# ----------------------------------------------------------------------


def _fake_ctx(samples_dir: Path, diag_dir: Path, seed: int = 7) -> SimpleNamespace:
    """Minimal RunContext stand-in: only the attrs select_training_samples reads
    (ctx.config.run.seed, ctx.samples_dir, ctx.diagnostics_dir)."""
    return SimpleNamespace(
        config=SimpleNamespace(run=SimpleNamespace(seed=seed)),
        samples_dir=samples_dir,
        diagnostics_dir=diag_dir,
    )


def _fake_components() -> SimpleNamespace:
    return SimpleNamespace(shared=SimpleNamespace(sample_window=None))


def test_select_training_samples_dryrun(samples_dir: Path, diag_dir: Path) -> None:
    # Tune the taper low so the modest copied window (a few hundred rows) clears
    # min_rows and produces a real window. requested small so effective_rows>0.
    tr = _make_trainer(
        shuffle_min_rows=1,
        shuffle_taper_window_scale=10.0,
        shuffle_keep_target_rows=10_000,
        train_samples_per_epoch=200,
        max_train_bucket_size=500_000.0,
        max_train_bucket_per_new_data=8.0,
    )
    ctx = _fake_ctx(samples_dir, diag_dir)
    comp = _fake_components()

    out = tr.select_training_samples(ctx=ctx, components=comp, epoch=1)
    assert out["status"] == "completed", f"epoch1 status {out['status']}: {out.get('reason')}"
    # plausible dict (PLAN §6 reference return shape).
    for key in ("total_rows", "live_total_rows", "desired_rows", "used_rows", "keep_prob",
                "effective_rows", "window_rows", "window_start", "train_bucket_level",
                "reuse_ratio"):
        assert key in out, f"return dict missing {key}"
    assert out["window_rows"] > 0, f"window_rows {out['window_rows']} not > 0"
    assert out["effective_rows"] > 0, f"effective_rows {out['effective_rows']} not > 0"
    assert out["total_rows"] >= out["live_total_rows"], "cumulative < live (monotone broken)"
    assert out["desired_rows"] >= 1
    assert 0.0 < out["keep_prob"] <= 1.0
    # the window handle is a PackedWindow on components.shared.
    assert isinstance(comp.shared.sample_window, PackedWindow), (
        f"sample_window is {type(comp.shared.sample_window).__name__}, not PackedWindow"
    )
    assert comp.shared.sample_window.n == out["window_rows"]
    # governor was credited (cumulative>0) and debited by effective_rows.
    cap = max(500_000.0, 200.0)
    expected_after = min(cap, out["total_rows"] * 8.0) - out["effective_rows"]
    assert abs(tr.train_state.train_bucket_level - expected_after) < 1e-6, (
        f"bucket level {tr.train_state.train_bucket_level} != expected {expected_after}"
    )
    assert tr.train_state.train_steps_since_last_reload == 1
    # the select diag was written under the (scratch) diagnostics dir.
    assert (diag_dir / "hexfield.select.epoch_000001.json").exists(), "select diag not written"
    print(f"  dry-run epoch 1: status=completed window_rows={out['window_rows']} "
          f"effective_rows={out['effective_rows']} total_rows={out['total_rows']} "
          f"keep_prob={out['keep_prob']:.3f} bucket={tr.train_state.train_bucket_level:.0f}")

    # A second epoch: same total (no new shards) -> no accrual; bucket debited
    # again by effective_rows; steps_since_last_reload advances to 2.
    bucket_before = tr.train_state.train_bucket_level
    comp2 = _fake_components()
    out2 = tr.select_training_samples(ctx=ctx, components=comp2, epoch=2)
    assert out2["status"] == "completed", f"epoch2 status {out2['status']}: {out2.get('reason')}"
    assert out2["total_rows"] == out["total_rows"], "cumulative changed with no new shards"
    assert tr.train_state.train_bucket_level == bucket_before - out2["effective_rows"], \
        "second-epoch debit wrong"
    assert tr.train_state.train_steps_since_last_reload == 2
    assert isinstance(comp2.shared.sample_window, PackedWindow)
    print(f"  dry-run epoch 2: no new rows -> no accrual; bucket {bucket_before:.0f} -> "
          f"{tr.train_state.train_bucket_level:.0f}; steps_since_reload=2")

    # Bucket-limited branch: a fresh trainer whose bucket can't cover effective_rows.
    tr_lim = _make_trainer(
        shuffle_min_rows=1,
        shuffle_taper_window_scale=10.0,
        shuffle_keep_target_rows=10_000,
        train_samples_per_epoch=200,
        max_train_bucket_size=500_000.0,
        max_train_bucket_per_new_data=0.0,  # never credits -> always limited
    )
    comp3 = _fake_components()
    out3 = tr_lim.select_training_samples(ctx=ctx, components=comp3, epoch=1)
    assert out3["status"] == "train_bucket_limited", f"expected limited, got {out3['status']}"
    assert isinstance(comp3.shared.sample_window, PackedWindow) and comp3.shared.sample_window.n == 0
    print(f"  dry-run bucket-limited: status=train_bucket_limited, empty PackedWindow set")


def main() -> int:
    # --- pure unit gates (no live data) -------------------------------------
    test_window_math_parity()
    test_select_recent_window()
    test_split_and_selection()
    test_keep_prob_selection_accounting()
    test_update_train_bucket()

    # --- gates needing real shards (copied out, mutate only copies) ----------
    root = _fresh("p4_select")
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_real_epochs(samples_dir, ["epoch_000001", "epoch_000002", "epoch_000003"], per_epoch=4)
    if copied == 0:
        print("FAIL: no real shards copied from", LIVE_SAMPLES)
        return 1

    # Build the manifest over the copy (writes .buffer_manifest.json under scratch).
    manifest = scan_or_update_manifest(samples_dir)
    assert manifest.entries, "manifest empty after copy"
    assert manifest.total_rows > 0
    assert manifest.cumulative_rows_ever >= manifest.total_rows
    print(f"  manifest (scratch copy): {len(manifest.entries)} shards, "
          f"total_rows={manifest.total_rows}, cumulative={manifest.cumulative_rows_ever}")

    test_build_window_split(samples_dir, list(manifest.entries))
    test_select_training_samples_dryrun(samples_dir, diag_dir)

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
