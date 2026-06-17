"""Phase 6 — spawn ProcessPool expand backend (PLAN §4.3 ladder, ``backend="pool"``).

CPU-ONLY, no GPU, no live-run interaction. The live ``hexfield_main_2`` tree is
READ-ONLY; a few shards are COPIED into ``_scratch/p5/samples`` (by
``_p5_setup_scratch.sh``, reused here) and only that copy is read. Every write
lands under ``_scratch/``.

What this verifies (each block prints its own line; ``main`` prints PASS):

  1. POOL == SERIAL element-wise — ``expand_rows(backend="pool")`` reproduces
     ``backend="serial"`` for EVERY row of a real main_2 window: identical
     ``valid`` mask and, per surviving row, identical support graph
     (coords/nbr/dist/legal_count), features, self/opp policy, opp_coverage,
     value, stvalue(+mask), moves_left(+mask). The window is sized above
     ``_PARALLEL_MIN_ROWS`` so the pool path actually engages.

  2. workers=1 == workers=4 — the pool output is invariant to worker count
     (commutativity: chunks are reassembled at their offset, so future-completion
     order is irrelevant — PLAN §4.4). Also prints a rows/s throughput line for
     serial, workers=1, and workers=4.

  3. OFF-LEGAL under HEXFIELD_SUPPORT_RADIUS=4 — in a fresh subprocess (the
     support radius is import-time), an INJECTED off-legal row (rewrite one row's
     first policy action id to a huge value, off the legal set at any radius —
     real radius-8 shards have nothing naturally off-legal at radius 4) is flagged
     invalid by BOTH serial and pool, with identical ``valid`` masks (the mask is
     a pure function of (row, d6, radius), PLAN §4.5) and surviving rows
     element-identical. Proves the worker's off-legal flagging matches the main
     thread.

  4. train_passes integration — the full consumer with ``expand_backend="pool"``
     yields bit-identical trained_rows / steps / loss components / grad norms to
     ``expand_backend="serial"`` (same model init + same pre-drawn RNG ⇒ the
     parallel expansion changes nothing downstream).

  5. rust hook — ``backend="rust"`` raises ``NotImplementedError`` (the Phase-7
     placeholder), and an unknown backend raises ``ValueError``.
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
from hexfield.expand_backends import _PARALLEL_MIN_ROWS, expand_rows, resolve_expand_workers
from hexfield.model import HexfieldNet
from hexfield.samples import ExpandedRow
from hexfield.trainer import HexfieldTrainer
from hexfield.window import PackedWindow, concat_packed, load_packed_shard

LIVE = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2/samples")
SCRATCH = Path(__file__).resolve().parent / "_scratch" / "p5"
SAMPLES = SCRATCH / "samples"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_window() -> PackedWindow:
    npzs = sorted(SAMPLES.glob("epoch_*/game_*.npz"))
    assert npzs, f"no scratch shards under {SAMPLES}; run _p5_setup_scratch.sh"
    return concat_packed([load_packed_shard(p) for p in npzs])


def _rows_equal(a: ExpandedRow | None, b: ExpandedRow | None) -> bool:
    """Element-wise equality of two expanded rows (or both-None)."""
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
        and abs(a.opp_coverage - b.opp_coverage) <= 1e-12
        and float(a.value) == float(b.value)
        and np.array_equal(a.stvalue, b.stvalue)
        and np.array_equal(a.stvalue_mask, b.stvalue_mask)
        and float(a.moves_left) == float(b.moves_left)
        and float(a.moves_left_mask) == float(b.moves_left_mask)
    )


def _lists_equal(ra, rb) -> tuple[bool, int]:
    """(all rows element-equal, count of mismatches) over two aligned lists."""
    if len(ra) != len(rb):
        return False, abs(len(ra) - len(rb))
    bad = sum(0 if _rows_equal(a, b) else 1 for a, b in zip(ra, rb))
    return bad == 0, bad


# ---------------------------------------------------------------------------
# 1. pool == serial element-wise
# ---------------------------------------------------------------------------
def test_pool_equals_serial() -> None:
    window = _load_window()
    assert window.n > _PARALLEL_MIN_ROWS, (
        f"window.n={window.n} <= _PARALLEL_MIN_ROWS={_PARALLEL_MIN_ROWS}; the pool "
        "path would silently fall back to serial and this test would be vacuous"
    )
    d6 = np.random.default_rng(20240617).integers(0, 12, size=window.n, dtype=np.int64)

    rows_s, valid_s = expand_rows(window, None, d6, backend="serial")
    rows_p, valid_p = expand_rows(window, None, d6, backend="pool", workers=4)

    assert len(rows_s) == window.n and len(rows_p) == window.n
    assert valid_s.dtype == bool and valid_p.dtype == bool
    assert np.array_equal(valid_s, valid_p), "valid mask differs (pool vs serial)"
    same, bad = _lists_equal(rows_s, rows_p)
    assert same, f"{bad} of {window.n} rows differ element-wise (pool vs serial)"
    print(
        f"  1. POOL==SERIAL: {window.n} rows expand element-identical "
        f"(valid mask equal, {int(valid_s.sum())} valid); pool path engaged "
        f"(n>{_PARALLEL_MIN_ROWS})"
    )


# ---------------------------------------------------------------------------
# 2. workers=1 == workers=4 (+ throughput)
# ---------------------------------------------------------------------------
def test_worker_count_invariant() -> None:
    window = _load_window()
    d6 = np.random.default_rng(555).integers(0, 12, size=window.n, dtype=np.int64)

    def _timed(backend: str, workers: int):
        t0 = time.perf_counter()
        rows, valid = expand_rows(window, None, d6, backend=backend, workers=workers)
        return rows, valid, time.perf_counter() - t0

    rows_serial, valid_serial, t_serial = _timed("serial", 0)
    rows_w1, valid_w1, t_w1 = _timed("pool", 1)
    rows_w4, valid_w4, t_w4 = _timed("pool", 4)

    same14, bad14 = _lists_equal(rows_w1, rows_w4)
    assert same14, f"workers=1 vs workers=4 differ ({bad14} rows)"
    assert np.array_equal(valid_w1, valid_w4), "valid mask differs (w1 vs w4)"
    # And both equal serial (already covered in test 1 for w4, re-pinned here for w1).
    same_s, bad_s = _lists_equal(rows_serial, rows_w1)
    assert same_s, f"workers=1 differs from serial ({bad_s} rows)"

    n = window.n
    print(
        f"  2. workers=1 == workers=4 == serial (element-wise). throughput: "
        f"serial={n / max(t_serial, 1e-9):.0f} rows/s, "
        f"pool(w1)={n / max(t_w1, 1e-9):.0f} rows/s, "
        f"pool(w4)={n / max(t_w4, 1e-9):.0f} rows/s  (n={n})"
    )


# ---------------------------------------------------------------------------
# 3. off-legal under radius 4 (subprocess — radius is import-time)
# ---------------------------------------------------------------------------
_OFF_ID = 200_000  # an action id guaranteed off the legal set at any radius
OFFLEGAL_SAMPLES = SCRATCH.parent / "p6_offlegal" / "samples"

# Probe run at HEXFIELD_SUPPORT_RADIUS=4 in a fresh process (radius is import-time
# in support.py). It expands the INJECTED-off-legal window under BOTH serial and
# pool, with tolerate_off_legal=True, and reports whether the masks + surviving
# rows agree and how many rows were skipped.
_RADIUS_PROBE = r"""
import numpy as np
from pathlib import Path
from hexfield.window import concat_packed, load_packed_shard
from hexfield.expand_backends import expand_rows

SAMPLES = Path("__SAMPLES__")
npzs = sorted(SAMPLES.glob("epoch_*/game_*.npz"))
window = concat_packed([load_packed_shard(p) for p in npzs])
d6 = np.random.default_rng(31337).integers(0, 12, size=window.n, dtype=np.int64)

rows_s, valid_s = expand_rows(window, None, d6, backend="serial", tolerate_off_legal=True)
rows_p, valid_p = expand_rows(window, None, d6, backend="pool", workers=4, tolerate_off_legal=True)

n_skip = int((~valid_s).sum())
mask_equal = bool(np.array_equal(valid_s, valid_p))
ok = True
for a, b in zip(rows_s, rows_p):
    if (a is None) != (b is None):
        ok = False; break
    if a is None:
        continue
    if not (np.array_equal(a.support.coords, b.support.coords)
            and np.array_equal(a.feats, b.feats)
            and np.array_equal(a.policy, b.policy)
            and np.array_equal(a.opp_policy, b.opp_policy)):
        ok = False; break
print(f"RADIUS4 n={window.n} n_skip={n_skip} mask_equal={mask_equal} rows_equal={ok}")
"""


def _make_offlegal_samples() -> int:
    """Copy a handful of scratch shards into ``p6_offlegal`` and rewrite row 0's
    first policy action to an off-legal id in EACH copied shard.

    Real radius-8 shards have nothing naturally off-legal at radius 4 (confirmed
    in the Phase-5 verifier), so the off-legal path must be exercised with an
    injected row — exactly the Phase-5 ``_make_offlegal_shard`` technique. Returns
    the number of injected (off-legal) rows so the test can pin ``n_skip``.
    """
    if OFFLEGAL_SAMPLES.exists():
        import shutil

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
        # pol_off[0]..pol_off[1] is row 0's policy slice; force its first action
        # off-legal (a huge action id is off the legal set at any radius).
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
    assert fields["mask_equal"] == "True", f"radius-4 valid mask differs pool vs serial: {line[-1]}"
    assert fields["rows_equal"] == "True", f"radius-4 surviving rows differ pool vs serial: {line[-1]}"
    # Every injected off-legal row must be flagged invalid by BOTH backends.
    assert n_skip == injected, (
        f"expected exactly {injected} off-legal rows skipped, got {n_skip}: {line[-1]}"
    )
    print(
        f"  3. OFF-LEGAL @radius4 (injected): pool flags the same {n_skip} rows "
        "invalid as serial; surviving rows element-identical (validity mask is "
        "backend-invariant, PLAN §4.5)"
    )


# ---------------------------------------------------------------------------
# 4. train_passes integration: pool vs serial bit-identical downstream
# ---------------------------------------------------------------------------
def _build_trainer(backend: str, workers: int) -> HexfieldTrainer:
    # min_rows >= the whole scratch corpus (~3.5k rows) so the taper +
    # recent-window cut select EVERY shard; keep_target huge ⇒ keep_prob=1.0 (no
    # subsample). The window then holds all ~3.5k rows (> _PARALLEL_MIN_ROWS), so
    # the pool path actually engages instead of falling back to serial.
    base = dict(
        shuffle_min_rows=8000,
        shuffle_taper_window_scale=8000.0,
        shuffle_keep_target_rows=1_000_000,  # keep every row (no subsample) so n>2048
        train_samples_per_epoch=8000,
        max_train_bucket_size=500_000.0,
        max_train_bucket_per_new_data=8.0,
        batch_rows=16,
        expand_backend=backend,
        expand_workers=workers,
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


def _ctx(diag_dir: Path, seed: int) -> SimpleNamespace:
    diag_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        config=SimpleNamespace(run=SimpleNamespace(seed=seed)),
        samples_dir=SAMPLES,
        diagnostics_dir=diag_dir,
    )


def _components() -> SimpleNamespace:
    return SimpleNamespace(shared=SimpleNamespace(sample_window=None, sample_symmetries=None))


def _run_epoch(backend: str, workers: int, diag_dir: Path, *, seed: int, model_seed: int, epoch: int = 1):
    torch.manual_seed(model_seed)  # identical initial weights across backends
    tr = _build_trainer(backend, workers)
    try:
        ctx = _ctx(diag_dir, seed)
        comp = _components()
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
        tr.close()  # tear down the spawn pool


def test_train_passes_pool_eq_serial() -> None:
    sel_s, out_s = _run_epoch("serial", 0, SCRATCH / "diag_p6_serial", seed=77, model_seed=7)
    sel_p, out_p = _run_epoch("pool", 4, SCRATCH / "diag_p6_pool", seed=77, model_seed=7)

    assert out_s["status"] == "completed" and out_p["status"] == "completed", (out_s, out_p)
    # The window selection is backend-independent (same seed) — pin it.
    assert sel_s["window_rows"] == sel_p["window_rows"], (sel_s["window_rows"], sel_p["window_rows"])
    assert sel_s["window_rows"] > _PARALLEL_MIN_ROWS, (
        f"integration window.n={sel_s['window_rows']} <= {_PARALLEL_MIN_ROWS}; pool path "
        "would fall back to serial and the integration test would be vacuous"
    )
    # The parallel expansion must change NOTHING downstream.
    assert out_s["trained_rows"] == out_p["trained_rows"], (out_s["trained_rows"], out_p["trained_rows"])
    assert out_s["steps"] == out_p["steps"], (out_s["steps"], out_p["steps"])
    assert out_s["rows_skipped_off_legal"] == out_p["rows_skipped_off_legal"]
    for k in out_s:
        if k.startswith("loss_"):
            assert out_s[k] == out_p[k], f"loss {k} differs (pool vs serial): {out_s[k]} vs {out_p[k]}"
    assert out_s["grad_norm_mean"] == out_p["grad_norm_mean"], (
        out_s["grad_norm_mean"], out_p["grad_norm_mean"]
    )
    assert out_s["grad_norm_p95"] == out_p["grad_norm_p95"]
    print(
        f"  4. train_passes(pool) == train_passes(serial): window_rows={sel_s['window_rows']}, "
        f"trained_rows={out_s['trained_rows']}, steps={out_s['steps']}, "
        f"grad_mean={out_s['grad_norm_mean']:.6g} (bit-identical downstream)"
    )


# ---------------------------------------------------------------------------
# 5. rust hook + unknown backend
# ---------------------------------------------------------------------------
def test_backend_guards() -> None:
    window = _load_window()
    d6 = np.zeros(window.n, dtype=np.int64)
    try:
        expand_rows(window, None, d6, backend="rust")
        raised = False
    except NotImplementedError:
        raised = True
    assert raised, "backend='rust' must raise NotImplementedError (Phase-7 hook)"
    try:
        expand_rows(window, None, d6, backend="bogus")
        bad_raised = False
    except ValueError:
        bad_raised = True
    assert bad_raised, "an unknown backend must raise ValueError"
    # resolve_expand_workers precedence + bounds.
    assert resolve_expand_workers(0) >= 1
    assert resolve_expand_workers(3) == 3  # explicit config wins (no env override set below)
    print("  5. backend='rust' -> NotImplementedError; unknown backend -> ValueError; "
          f"resolve_expand_workers(0)={resolve_expand_workers(0)} (auto), (3)=3 (explicit)")


def main() -> int:
    if not LIVE.exists():
        print("FAIL: live samples tree missing", LIVE)
        return 1
    if not SAMPLES.exists() or not any(SAMPLES.glob("epoch_*/game_*.npz")):
        print("FAIL: scratch samples missing; run tests/katago_buffer/_p5_setup_scratch.sh first", SAMPLES)
        return 1
    # Ensure the env override does not leak in and skew resolve_expand_workers(3)==3.
    os.environ.pop("HEXFIELD_EXPAND_WORKERS", None)
    os.environ.pop("HEXFIELD_EXPAND", None)
    torch.use_deterministic_algorithms(True, warn_only=True)
    test_pool_equals_serial()
    test_worker_count_invariant()
    test_offlegal_radius4_subprocess()
    test_train_passes_pool_eq_serial()
    test_backend_guards()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
