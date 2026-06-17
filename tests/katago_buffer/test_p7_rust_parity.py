"""Phase 7 — Rust rayon GIL-free train-read expand kernel parity (PLAN §4.2/§4.5,
§9 test 3 + test 7). CPU-ONLY, no GPU, no live-run interaction.

The live ``hexfield_main_2`` tree is READ-ONLY; a few shards are COPIED into
``_scratch/p5/samples`` (by ``_p5_setup_scratch.sh``, reused here) and only that
copy is read. Off-legal injection writes COPIES under ``_scratch/p7_offlegal``.

What this verifies (each block prints its own line; ``main`` prints PASS):

  1. RUST == SERIAL across ALL 12 D6 — for a sample of real main_2 rows, expand
     EACH row under EVERY symmetry 0..11 and assert ``expand_shard_train`` (Rust)
     == per-row ``expand_sample`` (Python serial oracle) element-wise on the
     support graph (coords/dist/nbr/legal_count/stone/halo), features, self/opp
     policy, opp_coverage, value, stvalue(+mask), moves_left(+mask). Every D6 value
     is exercised on every sampled row (the full cross-product, not a sweep).

  2. OFF-LEGAL @ radius 4 (subprocess; radius is import-time) — an INJECTED
     off-legal policy action (a huge action id, off the legal set at any radius;
     real radius-8 shards have nothing naturally off-legal at radius 4) is flagged
     INVALID by BOTH the Rust kernel and the serial oracle, with identical ``valid``
     masks (the mask is a pure function of (row, d6, radius), PLAN §4.5) and
     surviving rows element-identical. Pins the kernel's off-legal flagging against
     the Python skip.

  3. DETERMINISM — the Rust kernel run twice on the same (window, d6) is
     element-identical, and ``expand_shard_train`` over a permuted row order
     reproduces the per-row results of the natural order (order-preserving
     ``collect``: ``workers=1 == workers=N``, PLAN §4.4). Also prints a rows/s line
     for serial vs rust.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hexfield.config import HexfieldConfig, TrainingSection
from hexfield.expand_backends import _PARALLEL_MIN_ROWS, expand_rows
from hexfield.model import HexfieldNet
from hexfield.samples import ExpandedRow, expand_sample
from hexfield.trainer import HexfieldTrainer
from hexfield.window import concat_packed, load_packed_shard

LIVE = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2/samples")
SCRATCH = Path(__file__).resolve().parent / "_scratch"
SAMPLES = SCRATCH / "p5" / "samples"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_window(limit: int | None = None):
    npzs = sorted(SAMPLES.glob("epoch_*/game_*.npz"))
    assert npzs, f"no scratch shards under {SAMPLES}; run _p5_setup_scratch.sh"
    if limit is not None:
        npzs = npzs[:limit]
    return concat_packed([load_packed_shard(p) for p in npzs])


def _rows_equal(a: ExpandedRow | None, b: ExpandedRow | None) -> bool:
    """Element-wise equality of two expanded rows (or both-None).

    fp note (PLAN §9.3 "within fp tolerance"): coords/dist/nbr are exact ints;
    feats/policy/opp/stvalue are f32 arrays computed identically (the Rust twin
    matches numpy's f32 arithmetic, and the recency/moves_left f64-then-f32 double
    rounding). opp_coverage is compared STRICTLY (<= 1e-12): the kernel emits it as
    f64 (RowOut::opp_coverage) accumulated in the SAME order as the serial oracle's
    Python-float coverage, so the ratios are bit-identical, not merely close.
    value/moves_left/moves_left_mask are exactly representable in f32 (value widens
    losslessly; moves_left = 2*(ml/512)-1 is dyadic for integer ml), so exact
    ``float() == float()`` holds.
    """
    if a is None or b is None:
        return a is None and b is None
    return (
        a.support.num_nodes == b.support.num_nodes
        and a.support.legal_count == b.support.legal_count
        and a.support.stone_count == b.support.stone_count
        and a.support.halo_count == b.support.halo_count
        and np.array_equal(a.support.coords, b.support.coords)
        and np.array_equal(a.support.nbr, b.support.nbr)
        and np.array_equal(a.support.dist, b.support.dist)
        and a.policy.shape == b.policy.shape
        and np.array_equal(a.feats, b.feats)
        and np.array_equal(a.policy, b.policy)
        and np.array_equal(a.opp_policy, b.opp_policy)
        and abs(float(a.opp_coverage) - float(b.opp_coverage)) <= 1e-12
        and float(a.value) == float(b.value)
        and np.array_equal(a.stvalue, b.stvalue)
        and np.array_equal(a.stvalue_mask, b.stvalue_mask)
        and float(a.moves_left) == float(b.moves_left)
        and float(a.moves_left_mask) == float(b.moves_left_mask)
    )


def _describe_mismatch(i: int, sym: int, a: ExpandedRow, b: ExpandedRow) -> str:
    parts = [f"row {i} sym {sym}:"]
    if a is None or b is None:
        return f"row {i} sym {sym}: None mismatch serial={a is None} rust={b is None}"
    parts.append(f"N s/r={a.support.num_nodes}/{b.support.num_nodes}")
    parts.append(f"coords={np.array_equal(a.support.coords, b.support.coords)}")
    parts.append(f"dist={np.array_equal(a.support.dist, b.support.dist)}")
    parts.append(f"nbr={np.array_equal(a.support.nbr, b.support.nbr)}")
    parts.append(f"feats={np.array_equal(a.feats, b.feats)}")
    if not np.array_equal(a.feats, b.feats):
        d = np.abs(a.feats.astype(np.float64) - b.feats.astype(np.float64))
        idx = np.unravel_index(int(np.argmax(d)), d.shape)
        parts.append(f"maxfeatΔ={float(d.max()):.3e}@{idx}")
    parts.append(f"pol={np.array_equal(a.policy, b.policy)}")
    parts.append(f"opp={np.array_equal(a.opp_policy, b.opp_policy)}")
    parts.append(f"cov s/r={a.opp_coverage}/{b.opp_coverage}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 1. rust == serial across ALL 12 D6 (full cross-product over sampled rows)
# ---------------------------------------------------------------------------
def test_rust_equals_serial_all_d6() -> None:
    window = _load_window()
    n = window.n
    # Sample up to ~400 rows spread across the window (every shard contributes via
    # the stride); expanding each under 12 syms is ~5k expansions — fast on CPU and
    # exhaustive on the D6 axis.
    sample_rows = list(range(0, n, max(1, n // 400)))[:400]
    assert sample_rows, "no rows sampled"

    seen_syms: set[int] = set()
    total_pairs = 0
    for sym in range(12):
        d6 = np.full(len(sample_rows), sym, dtype=np.int64)
        rows_s, valid_s = expand_rows(window, sample_rows, d6, backend="serial")
        rows_r, valid_r = expand_rows(window, sample_rows, d6, backend="rust")
        assert np.array_equal(valid_s, valid_r), f"valid mask differs at sym {sym}"
        for k, (a, b) in enumerate(zip(rows_s, rows_r)):
            assert _rows_equal(a, b), _describe_mismatch(sample_rows[k], sym, a, b)
        seen_syms.add(sym)
        total_pairs += len(sample_rows)

    assert seen_syms == set(range(12)), f"missing D6 values: {set(range(12)) - seen_syms}"
    print(
        f"  1. RUST==SERIAL across ALL 12 D6: {len(sample_rows)} sampled rows x 12 syms "
        f"= {total_pairs} expansions element-identical (coords/dist/nbr/feats/policy/"
        f"opp/cov/value/stvalue/moves_left)"
    )


# ---------------------------------------------------------------------------
# 2. off-legal under radius 4 (subprocess — radius is import-time in support.py)
# ---------------------------------------------------------------------------
_OFF_ID = 200_000  # an action id guaranteed off the legal set at any radius
OFFLEGAL_SAMPLES = SCRATCH / "p7_offlegal" / "samples"

_RADIUS_PROBE = r"""
import numpy as np
from pathlib import Path
from hexfield.window import concat_packed, load_packed_shard
from hexfield.expand_backends import expand_rows

SAMPLES = Path("__SAMPLES__")
npzs = sorted(SAMPLES.glob("epoch_*/game_*.npz"))
window = concat_packed([load_packed_shard(p) for p in npzs])
# Sweep every row across all 12 syms folded by index so off-legal rows are hit
# under several orientations (the injected id is off-legal under ALL syms).
d6 = (np.arange(window.n) % 12).astype(np.int64)

rows_s, valid_s = expand_rows(window, None, d6, backend="serial", tolerate_off_legal=True)
rows_r, valid_r = expand_rows(window, None, d6, backend="rust", tolerate_off_legal=True)

n_skip = int((~valid_s).sum())
mask_equal = bool(np.array_equal(valid_s, valid_r))
ok = True
for a, b in zip(rows_s, rows_r):
    if (a is None) != (b is None):
        ok = False; break
    if a is None:
        continue
    if not (np.array_equal(a.support.coords, b.support.coords)
            and np.array_equal(a.support.dist, b.support.dist)
            and np.array_equal(a.support.nbr, b.support.nbr)
            and np.array_equal(a.feats, b.feats)
            and np.array_equal(a.policy, b.policy)
            and np.array_equal(a.opp_policy, b.opp_policy)):
        ok = False; break
print(f"RADIUS4 n={window.n} n_skip={n_skip} mask_equal={mask_equal} rows_equal={ok}")
"""


def _make_offlegal_samples() -> int:
    """Copy a handful of scratch shards into ``p7_offlegal`` and rewrite row 0's
    first policy action to an off-legal id in EACH copied shard (the P6 technique).
    Returns the count of injected (off-legal) rows so the test can pin ``n_skip``.
    """
    import shutil

    if OFFLEGAL_SAMPLES.exists():
        shutil.rmtree(OFFLEGAL_SAMPLES)
    srcs = sorted(SAMPLES.glob("epoch_*/game_*.npz"))[:6]
    assert srcs, "no scratch shards to inject"
    injected = 0
    for src in srcs:
        side = src.with_suffix(".json")
        dst_dir = OFFLEGAL_SAMPLES / src.parent.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        with np.load(src) as z:
            arrays = {k: z[k] for k in z.files}
        if int(arrays["pol_off"][1]) <= int(arrays["pol_off"][0]):
            continue  # row 0 has no policy entries; skip (rare)
        pol_act = arrays["pol_act"].copy()
        pol_act[int(arrays["pol_off"][0])] = _OFF_ID
        arrays["pol_act"] = pol_act
        np.savez_compressed(dst, **arrays)
        dst.with_suffix(".json").write_text(side.read_text(), encoding="utf-8")
        injected += 1
    assert injected > 0, "failed to inject any off-legal rows"
    return injected


def test_offlegal_radius4_subprocess() -> None:
    injected = _make_offlegal_samples()
    probe = _RADIUS_PROBE.replace("__SAMPLES__", str(OFFLEGAL_SAMPLES))
    env = dict(os.environ)
    env["HEXFIELD_SUPPORT_RADIUS"] = "4"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            "packages/hexfield/python",
            "packages/dense_cnn_restnet/python",
            env.get("PYTHONPATH", ""),
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd="/mnt/e/hexgt-katago",
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout.strip()
    assert proc.returncode == 0, f"radius-4 subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    line = [ln for ln in out.splitlines() if ln.startswith("RADIUS4")]
    assert line, f"probe produced no RADIUS4 line:\n{out}\n{proc.stderr}"
    fields = dict(tok.split("=", 1) for tok in line[-1].split()[1:])
    n_skip = int(fields["n_skip"])
    assert fields["mask_equal"] == "True", f"radius-4 valid mask differs rust vs serial: {line[-1]}"
    assert fields["rows_equal"] == "True", f"radius-4 surviving rows differ rust vs serial: {line[-1]}"
    assert n_skip == injected, (
        f"expected exactly {injected} off-legal rows flagged invalid, got {n_skip}: {line[-1]}"
    )
    print(
        f"  2. OFF-LEGAL @radius4 (injected): the Rust kernel flags the same {n_skip} rows "
        "invalid as the serial oracle; surviving rows element-identical (validity mask is "
        "backend-invariant, PLAN §4.5)"
    )


# ---------------------------------------------------------------------------
# 3. determinism: rust run-twice + permuted-order invariance (+ throughput)
# ---------------------------------------------------------------------------
def test_determinism_and_order_invariance() -> None:
    window = _load_window()
    n = window.n
    d6 = np.random.default_rng(70707).integers(0, 12, size=n, dtype=np.int64)

    # (a) run twice -> element-identical (rayon scheduling is irrelevant; the
    # order-preserving collect makes the output worker-count-invariant: workers=1
    # == workers=N, PLAN §4.4).
    t0 = time.perf_counter()
    rows_a, valid_a = expand_rows(window, None, d6, backend="rust")
    t_rust = time.perf_counter() - t0
    rows_b, valid_b = expand_rows(window, None, d6, backend="rust")
    assert np.array_equal(valid_a, valid_b), "valid mask differs across two rust runs"
    assert all(_rows_equal(a, b) for a, b in zip(rows_a, rows_b)), "rust not deterministic run-to-run"

    # (b) permuted input order -> same per-row result at the corresponding position
    # (proves the kernel result for a row depends only on (row, d6), not on its
    # neighbours / scheduling). Build a permutation of the row index, expand it,
    # and check each permuted output equals the natural-order output for that row.
    perm = np.random.default_rng(99).permutation(n)
    d6_perm = d6[perm]
    rows_p, valid_p = expand_rows(window, list(perm), d6_perm, backend="rust")
    for k in range(n):
        src = int(perm[k])
        assert bool(valid_p[k]) == bool(valid_a[src]), f"perm valid mismatch at {k}"
        assert _rows_equal(rows_p[k], rows_a[src]), f"perm row mismatch at pos {k} (src row {src})"

    # throughput line (serial vs rust) for the record.
    t0 = time.perf_counter()
    expand_rows(window, None, d6, backend="serial")
    t_serial = time.perf_counter() - t0
    print(
        f"  3. DETERMINISM: rust run-twice identical AND permuted-order invariant "
        f"(order-preserving collect, workers=1==N). throughput: "
        f"serial={n / max(t_serial, 1e-9):.0f} rows/s, rust={n / max(t_rust, 1e-9):.0f} rows/s (n={n})"
    )


# ---------------------------------------------------------------------------
# 4. train_passes integration: rust vs serial bit-identical downstream
# ---------------------------------------------------------------------------
def _build_trainer(backend: str) -> HexfieldTrainer:
    # A SMALL, FAST config: min_rows spans the whole scratch corpus so selection is
    # deterministic, but keep_target=600 subsamples the window to ~600 rows and
    # train_samples_per_epoch=300 caps the (slow, CPU) training to ~300 rows — both
    # backends run the identical light load. The rust kernel engages at ANY window
    # size (no _PARALLEL_MIN_ROWS gate, unlike pool), so this still exercises it; the
    # point of this test is rust==serial through the WHOLE dispatch, which holds at
    # any size. The expand still runs over the full ~600-row window each epoch.
    base = dict(
        shuffle_min_rows=8000,
        shuffle_taper_window_scale=8000.0,
        shuffle_keep_target_rows=600,  # subsample window -> ~600 rows (fast expand)
        train_samples_per_epoch=300,  # cap trained rows -> fast CPU training
        max_train_bucket_size=500_000.0,
        max_train_bucket_per_new_data=8.0,
        batch_rows=16,
        expand_backend=backend,
        expand_workers=0,
    )
    cfg = HexfieldConfig(device="cpu", training=TrainingSection(**base))
    model = HexfieldNet()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "bias_table" not in name and name != "tokens":
            decay.append(p)
        else:
            no_decay.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.training.learning_rate,
    )
    return HexfieldTrainer(model=model, config=cfg, optimizer=opt)


def _run_epoch(backend: str, diag_dir: Path, *, seed: int, model_seed: int, epoch: int = 1):
    torch.manual_seed(model_seed)  # identical initial weights across backends
    tr = _build_trainer(backend)
    try:
        diag_dir.mkdir(parents=True, exist_ok=True)
        ctx = SimpleNamespace(
            config=SimpleNamespace(run=SimpleNamespace(seed=seed)),
            samples_dir=SAMPLES,
            diagnostics_dir=diag_dir,
        )
        comp = SimpleNamespace(shared=SimpleNamespace(sample_window=None, sample_symmetries=None))
        sel = tr.select_training_samples(ctx=ctx, components=comp, epoch=epoch)
        assert sel["status"] == "completed", sel
        out = tr.train_passes(
            passes=1,
            sample_window=comp.shared.sample_window,
            sample_symmetries=comp.shared.sample_symmetries,
            ctx=ctx,
            components=comp,
            epoch=epoch,
        )
        return sel, out
    finally:
        tr.close()


def test_train_passes_rust_eq_serial() -> None:
    sel_s, out_s = _run_epoch("serial", SCRATCH / "p5" / "diag_p7_serial", seed=77, model_seed=7)
    sel_r, out_r = _run_epoch("rust", SCRATCH / "p5" / "diag_p7_rust", seed=77, model_seed=7)

    assert out_s["status"] == "completed" and out_r["status"] == "completed", (out_s, out_r)
    # The window selection is backend-independent (same seed) — pin it. (No
    # _PARALLEL_MIN_ROWS gate here: the rust kernel engages at any window size, so a
    # small window is a valid exercise of the dispatch.)
    assert sel_s["window_rows"] == sel_r["window_rows"], (sel_s["window_rows"], sel_r["window_rows"])
    assert sel_s["window_rows"] > 0, sel_s["window_rows"]
    # The parallel expansion must change NOTHING downstream (same model init + same
    # pre-drawn RNG => bit-identical trained_rows / steps / loss / grad norms).
    assert out_s["trained_rows"] == out_r["trained_rows"], (out_s["trained_rows"], out_r["trained_rows"])
    assert out_s["steps"] == out_r["steps"], (out_s["steps"], out_r["steps"])
    assert out_s["rows_skipped_off_legal"] == out_r["rows_skipped_off_legal"]
    for k in out_s:
        if k.startswith("loss_"):
            assert out_s[k] == out_r[k], f"loss {k} differs (rust vs serial): {out_s[k]} vs {out_r[k]}"
    assert out_s["grad_norm_mean"] == out_r["grad_norm_mean"], (
        out_s["grad_norm_mean"], out_r["grad_norm_mean"]
    )
    assert out_s["grad_norm_p95"] == out_r["grad_norm_p95"]
    print(
        f"  4. train_passes(rust) == train_passes(serial): window_rows={sel_s['window_rows']}, "
        f"trained_rows={out_s['trained_rows']}, steps={out_s['steps']}, "
        f"grad_mean={out_s['grad_norm_mean']:.6g} (bit-identical downstream)"
    )


def main() -> int:
    if not LIVE.exists():
        print("FAIL: live samples tree missing", LIVE)
        return 1
    if not SAMPLES.exists() or not any(SAMPLES.glob("epoch_*/game_*.npz")):
        print("FAIL: scratch samples missing; run tests/katago_buffer/_p5_setup_scratch.sh first", SAMPLES)
        return 1
    os.environ.pop("HEXFIELD_EXPAND", None)
    os.environ.pop("HEXFIELD_EXPAND_WORKERS", None)
    torch.use_deterministic_algorithms(True, warn_only=True)
    test_rust_equals_serial_all_d6()
    test_offlegal_radius4_subprocess()
    test_determinism_and_order_invariance()
    test_train_passes_rust_eq_serial()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
