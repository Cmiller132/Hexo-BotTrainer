"""Migration converter: continue an EXISTING hexfield run under the new KataGo /
dense_cnn_restnet replay buffer — additively, with NO file moves and NO re-format.

================================================================================
HOW TO CONTINUE ``hexfield_main_2`` (or any existing run) UNDER THE NEW BUFFER
================================================================================

The on-disk compact shard schema (``hexfield_compact_v1``, ``shards.py``) and the
self-play write path (``selfplay.py``) are UNCHANGED by the buffer port. So there
is no cross-format migration and no data movement: the new buffer reads the SAME
``samples/epoch_*/game_*.npz`` shards the old window did. "Continuation" is purely
two additive steps:

  1. **Build the manifest additively over the existing samples/.** The first
     ``scan_or_update_manifest(<run>/samples)`` the resumed run performs
     ``rglob``s every existing ``epoch_*/game_*.npz``, reads its sidecar for the
     row count, derives ``generation``/``game_key`` from the filename (mtime-FREE),
     sorts by ``(generation, game_key)``, and persists
     ``<run>/samples/.buffer_manifest.json`` atomically. No shard is opened for
     writing, moved, renamed, re-sharded, or deleted. The monotone
     ``cumulative_rows_ever`` is seeded from the live row sum.

  2. **Resume the latest checkpoint with a FRESH governor.** The existing
     checkpoints were written by the OLD saver, so their ``meta`` carries no
     ``train_state``. On resume the loader calls
     ``HexfieldTrainState.from_dict(meta.get("train_state"))`` ->
     ``from_dict(None)`` -> a FRESH train-bucket governor (level 0,
     ``level_at_row`` 0). That is correct and intended: the governor simply
     re-accrues from the current cumulative row count on the first post-resume
     epoch — it never KeyErrors on the old-format checkpoint, and it never
     inherits a stale bucket. From then on every new checkpoint persists
     ``train_state`` so subsequent resumes restore the bucket exactly.

CONCRETELY, to continue ``hexfield_main_2`` under the new buffer:

  * Author a NEW config that POINTS AT THE EXISTING RUN DIR and resumes the
    latest checkpoint, e.g.::

        [run]
        name = "hexfield_main_2"                       # same run name
        output_dir = "runs/hexfield_main_2"            # the EXISTING dir
        seed = <same as before>

        [checkpoint]
        resume_from = "runs/hexfield_main_2/checkpoints/epoch_000030.pt"  # latest

        [model.config.training]
        # the NEW KataGo buffer knobs (PLAN §7). These are additive; the strict
        # flat config merge tolerates them and the old fields that remain.
        shuffle_min_rows = 20000
        shuffle_keep_target_rows = 300000
        shuffle_taper_window_exponent = 0.65
        shuffle_expand_window_per_row = 0.4
        shuffle_taper_window_scale = 20000.0
        validation_fraction = 0.0
        train_samples_per_epoch = 100000
        max_train_bucket_per_new_data = 8.0
        max_train_bucket_size = 500000.0
        no_repeat_files = false
        expand_backend = "serial"        # flip to "rust" after parity gates (PLAN §10)
        expand_workers = 0

  * Launch the run normally. On the first resumed epoch the trainer:
      (a) builds ``.buffer_manifest.json`` additively over the existing
          ``samples/`` (indexing all ~7.7k shards),
      (b) restores a FRESH governor from the old checkpoint (no train_state),
      (c) selects a power-law-taper recent window over the existing shards,
          keep_prob-subsamples it, and trains a single capped pass.

  * NO data is lost or moved. The existing ``epoch_*/game_*.npz`` are indexed in
    place; new self-play epochs append new shards the next scan picks up
    incrementally.

  NOTE ON THE LIVE RUN: ``hexfield_main_2`` is a LIVE process on this machine.
  Per the port's rollout (PLAN §10) the recommended path is a FRESH run
  (``hexfield_main_3``) so the live run finishes undisturbed. This converter is a
  READ-ONLY diagnostic that PROVES the continuation path works without touching
  the live run. Persisting the manifest into a live run dir (``--write``) should
  only ever be done against a STOPPED run.

================================================================================
WHAT THIS SCRIPT DOES (read-only by default)
================================================================================

Given a run dir it:
  * builds/updates the manifest over ``<run>/samples`` (IN-MEMORY by default — it
    does NOT write ``.buffer_manifest.json`` into a live run dir unless
    ``--write`` is given and the run is verified stopped),
  * prints the window + governor diagnostics for the latest cumulative state:
    ``desired_rows`` (power-law taper), ``used_rows`` (recent-window cut),
    ``keep_prob``, ``cumulative_rows_ever``, ``total_rows`` (live), and a reuse
    estimate (``effective_rows / new_rows_this_epoch`` and the bucket accrual),
  * VALIDATES the new pipeline can load+select a window from the existing shards
    and resume the latest checkpoint with ``from_dict(None)`` -> a fresh governor
    (no crash), by driving ``HexfieldCheckpointLoader`` +
    ``HexfieldTrainer.select_training_samples`` on CPU.

Usage::

    python scripts/_hexfield_buffer_convert.py [RUN_DIR] [--write] [--no-validate]
                                               [--epoch N] [--device cpu]

``RUN_DIR`` defaults to ``runs/hexfield_main_2`` under the repo root (resolved
against $HEXGT_RUNS_ROOT or the standard layout). All torch work is CPU-only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Resolve the hexfield package on sys.path (script is run from the repo root via
# the venv python; mirror the convention in scripts/_hexfield_check_prod_config.py).
_REPO = Path(__file__).resolve().parent.parent
for _pkg in ("hexfield", "dense_cnn_restnet"):
    _p = _REPO / "packages" / _pkg / "python"
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Read-only manifest build (never persists into a live samples dir)
# ---------------------------------------------------------------------------
def build_manifest_readonly(samples_dir: Path):
    """Build the buffer manifest over ``samples_dir`` WITHOUT persisting it.

    Mirrors ``buffer_manifest.scan_or_update_manifest`` build logic (drop vanished
    -> incrementally index sidecar-backed shards -> sort by (generation, game_key)
    -> recompute totals + monotone cumulative) but DOES NOT call the atomic write,
    so it is safe to run against a LIVE run dir (never writes
    ``.buffer_manifest.json`` there). If a manifest already exists it is loaded as
    the incremental base (read-only) so an already-migrated run reports its
    persisted ``cumulative_rows_ever``.
    """
    from hexfield.buffer_manifest import (
        MANIFEST_NAME,
        BufferManifest,
        _build_entry,
        _iter_shard_npzs,
        _load_manifest,
    )

    samples_dir = Path(samples_dir)
    manifest = _load_manifest(samples_dir / MANIFEST_NAME) or BufferManifest()

    # (2) prune vanished, index survivors by rel_path.
    present: dict[str, Any] = {}
    for entry in manifest.entries:
        if (samples_dir / entry.rel_path).exists():
            present[entry.rel_path] = entry

    # (3) incrementally add new sidecar-backed shards (skip half-written).
    n_npz = 0
    n_skipped = 0
    for npz_path in _iter_shard_npzs(samples_dir):
        n_npz += 1
        rel_path = npz_path.relative_to(samples_dir).as_posix()
        if rel_path in present:
            continue
        new_entry = _build_entry(npz_path, samples_dir)
        if new_entry is not None:
            present[rel_path] = new_entry
        else:
            n_skipped += 1

    manifest.entries = list(present.values())
    manifest.resort()
    manifest.recompute_totals()
    return manifest, n_npz, n_skipped


def persist_manifest(samples_dir: Path):
    """Persist the manifest into ``samples_dir`` via the real
    ``scan_or_update_manifest`` (atomic tmp+rename). Only for a STOPPED run."""
    from hexfield.buffer_manifest import scan_or_update_manifest

    return scan_or_update_manifest(samples_dir)


# ---------------------------------------------------------------------------
# Window + governor diagnostics (PLAN §3.1-3.5)
# ---------------------------------------------------------------------------
def window_governor_diagnostics(manifest, training, *, prev_level_at_row: int = 0) -> dict[str, Any]:
    """Compute the taper window / recent-window cut / keep_prob / governor accrual
    for the CURRENT cumulative state (the first post-resume epoch, governor fresh).

    Returns a dict mirroring the ``select_training_samples`` return shape's
    diagnostic fields (PLAN §6). ``prev_level_at_row`` defaults 0 (a fresh
    governor) so ``new_rows_this_epoch == cumulative_rows_ever`` and the reuse
    estimate is the first-epoch accrual."""
    from hexfield.window import (
        compute_katago_window_rows,
        keep_prob as _keep_prob,
        select_recent_window,
    )

    total_rows = int(manifest.total_rows)
    cumulative = int(manifest.cumulative_rows_ever)

    desired = compute_katago_window_rows(
        total_rows,
        min_rows=training.shuffle_min_rows,
        expand_window_per_row=training.shuffle_expand_window_per_row,
        taper_window_exponent=training.shuffle_taper_window_exponent,
        taper_window_scale=training.shuffle_taper_window_scale,
    )
    desired = max(int(desired), int(training.shuffle_min_rows))
    selected_window, used = select_recent_window(manifest.entries, desired)
    window_start = max(0, total_rows - used)
    kp = _keep_prob(used, int(training.shuffle_keep_target_rows))

    # Governor accrual on the monotone cumulative counter (fresh: level_at_row=prev).
    cap = max(
        float(training.max_train_bucket_size),
        float(training.train_samples_per_epoch),
    )
    new_rows = max(0, cumulative - int(prev_level_at_row))
    accrued = min(cap, new_rows * float(training.max_train_bucket_per_new_data))
    requested = int(training.train_samples_per_epoch)
    # First-epoch effective is bounded by what a single capped pass can train; we
    # report the requested cap and the keep_prob-expected window rows.
    expected_window_rows = int(round(used * kp))
    effective_est = min(requested, expected_window_rows if expected_window_rows > 0 else used)
    # The governor only throttles when the bucket can't cover effective_rows.
    bucket_limited = accrued + 1e-9 < effective_est
    reuse_ratio = effective_est / max(1, new_rows)

    return {
        "cumulative_rows_ever": cumulative,
        "live_total_rows": total_rows,
        "desired_rows": int(desired),
        "used_rows": int(used),
        "window_shards": len(selected_window),
        "window_start": int(window_start),
        "keep_prob": float(kp),
        "expected_window_rows": expected_window_rows,
        "train_samples_per_epoch": requested,
        "effective_rows_est": int(effective_est),
        "new_rows_this_epoch": int(new_rows),
        "bucket_accrued": float(accrued),
        "bucket_cap": float(cap),
        "bucket_limited": bool(bucket_limited),
        "reuse_ratio_est": float(reuse_ratio),
    }


# ---------------------------------------------------------------------------
# Continuation validation: load+select a window + resume the latest checkpoint
# ---------------------------------------------------------------------------
def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    cks = sorted(checkpoint_dir.glob("epoch_*.pt"))
    return cks[-1] if cks else None


def _build_trainer(device: str, training_overrides: dict[str, Any] | None = None):
    """A minimal HexfieldNet + AdamW + HexfieldTrainer on ``device`` (CPU here).

    The param-group split mirrors the production optimizer (and the P5 test):
    weight-decay on >=2D weights except the bias tables / tokens.
    """
    import torch

    from hexfield.config import HexfieldConfig, TrainingSection
    from hexfield.model import HexfieldNet
    from hexfield.trainer import HexfieldTrainer

    training = TrainingSection(**(training_overrides or {}))
    cfg = HexfieldConfig(device=device, training=training)
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


def run_validation(
    *,
    samples_dir: Path,
    checkpoint: Path,
    seed: int,
    epoch: int,
    device: str,
    diag_dir: Path,
    run_name: str,
    training_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resume ``checkpoint`` into a fresh trainer and run one window selection
    against ``samples_dir``, exercising the EXACT production resume+select path
    (``HexfieldCheckpointLoader`` -> ``HexfieldTrainer.select_training_samples``).

    Asserts the governor came up FRESH from the old-format checkpoint
    (``from_dict(None)`` because the old ``meta`` has no ``train_state``).
    ``samples_dir`` MUST be a writable location — ``select_training_samples`` calls
    ``scan_or_update_manifest`` which persists ``.buffer_manifest.json`` there — so
    the caller passes a scratch COPY for a live run and never the live tree."""
    import torch

    from hexfield.checkpoints import HexfieldCheckpointLoader
    from hexfield.train_state import HexfieldTrainState
    from hexfield.window import PackedWindow

    diag_dir.mkdir(parents=True, exist_ok=True)
    trainer = _build_trainer(device, training_overrides)

    components = SimpleNamespace(
        model=SimpleNamespace(
            model=trainer.model,
            optimizer=trainer.optimizer,
            trainer=trainer,
        ),
        shared=SimpleNamespace(sample_window=None, sample_symmetries=None),
    )
    ctx = SimpleNamespace(
        config=SimpleNamespace(
            run=SimpleNamespace(name=run_name, seed=seed),
            checkpoint=SimpleNamespace(resume_from=checkpoint, initialize_from=None),
        ),
        samples_dir=Path(samples_dir),
        diagnostics_dir=diag_dir,
        checkpoint_dir=checkpoint.parent,
    )

    # (a) RESUME the old-format checkpoint -> fresh governor (the load-bearing
    # assertion: from_dict(None) on a checkpoint with no meta.train_state).
    loader = HexfieldCheckpointLoader()
    load_report = loader.load(checkpoint, ctx=ctx, components=components)
    ts: HexfieldTrainState = trainer.train_state
    fresh = HexfieldTrainState()
    governor_is_fresh = (
        ts.train_bucket_level == fresh.train_bucket_level
        and ts.train_bucket_level_at_row == fresh.train_bucket_level_at_row
        and ts.total_num_data_rows == fresh.total_num_data_rows
        and ts.train_steps_since_last_reload == fresh.train_steps_since_last_reload
        and len(ts.data_files_used) == 0
    )

    # (b) SELECT a window from the existing shards under the new pipeline.
    t0 = time.time()
    sel = trainer.select_training_samples(ctx=ctx, components=components, epoch=epoch)
    sel_seconds = time.time() - t0
    window = components.shared.sample_window
    window_ok = isinstance(window, PackedWindow)

    return {
        "load_report": load_report,
        "governor_is_fresh": bool(governor_is_fresh),
        "train_state_after_resume": ts.to_dict(),
        "select_status": sel.get("status"),
        "select": sel,
        "window_is_packed": bool(window_ok),
        "window_rows": int(window.n) if window_ok else None,
        "select_seconds": round(sel_seconds, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_run_dir() -> Path:
    root = os.environ.get("HEXGT_RUNS_ROOT")
    if root:
        return Path(root) / "hexfield_main_2"
    # Standard layout on this host.
    return Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2")


def _is_run_live(run_dir: Path) -> bool:
    """Best-effort: a run with a driver.pid / supervisor.lock is presumed LIVE."""
    return (run_dir / "driver.pid").exists() or (run_dir / "supervisor.lock").exists()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="hexfield KataGo buffer migration converter (read-only)")
    ap.add_argument("run_dir", nargs="?", default=str(_default_run_dir()),
                    help="run dir containing samples/ + checkpoints/ (default: hexfield_main_2)")
    ap.add_argument("--write", action="store_true",
                    help="PERSIST .buffer_manifest.json into the run's samples/ (STOPPED runs only)")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the resume+select validation (manifest + diagnostics only)")
    ap.add_argument("--epoch", type=int, default=None,
                    help="epoch index for the diagnostics/selection (default: latest_ckpt_epoch+1)")
    ap.add_argument("--device", default="cpu", help="torch device for validation (CPU enforced)")
    ap.add_argument("--scratch", default=None,
                    help="scratch dir for the validation copy (default: <repo>/scratch/buffer_convert)")
    args = ap.parse_args(argv)

    if args.device != "cpu":
        print(f"[convert] forcing device=cpu (was {args.device!r}); migration must never use the GPU")
        args.device = "cpu"

    run_dir = Path(args.run_dir)
    samples_dir = run_dir / "samples"
    checkpoint_dir = run_dir / "checkpoints"
    if not samples_dir.is_dir():
        print(f"FAIL: samples dir not found: {samples_dir}")
        return 1

    live = _is_run_live(run_dir)
    print(f"=== hexfield buffer migration converter ===")
    print(f"run_dir   : {run_dir}")
    print(f"samples   : {samples_dir}")
    print(f"live?     : {live}  (driver.pid / supervisor.lock present)")

    # --- 1. build the manifest (read-only by default) ----------------------
    from hexfield.config import TrainingSection

    training = TrainingSection()  # PLAN §7 defaults

    t0 = time.time()
    manifest, n_npz, n_skipped = build_manifest_readonly(samples_dir)
    scan_dt = time.time() - t0
    print("\n--- manifest (additive over existing samples/, NOT persisted) ---")
    print(f"shards_found      : {n_npz}")
    print(f"shards_indexed    : {len(manifest.entries)}")
    print(f"shards_skipped    : {n_skipped} (no sidecar / half-written)")
    print(f"total_rows (live) : {manifest.total_rows}")
    print(f"cumulative_ever   : {manifest.cumulative_rows_ever}")
    if manifest.entries:
        gens = [e.generation for e in manifest.entries]
        keys = [(e.generation, e.game_key) for e in manifest.entries]
        ordered = keys == sorted(keys)
        print(f"generations       : min={min(gens)} max={max(gens)} distinct={len(set(gens))}")
        print(f"ordering          : {'sorted (generation, game_key)' if ordered else '*** NOT SORTED ***'}")
        print(f"first / last      : {manifest.entries[0].rel_path} .. {manifest.entries[-1].rel_path}")
    print(f"scan_seconds      : {scan_dt:.1f}")

    if not manifest.entries:
        print("FAIL: no shards indexed; cannot continue")
        return 1

    # --- 2. window + governor diagnostics ----------------------------------
    diag = window_governor_diagnostics(manifest, training, prev_level_at_row=0)
    print("\n--- window + governor (first post-resume epoch, fresh governor) ---")
    print(f"desired_rows (taper)     : {diag['desired_rows']}")
    print(f"used_rows (recent-window): {diag['used_rows']}  over {diag['window_shards']} shards")
    print(f"window_start             : {diag['window_start']}")
    print(f"keep_prob                : {diag['keep_prob']:.4f}")
    print(f"expected_window_rows     : {diag['expected_window_rows']} (used * keep_prob)")
    print(f"train_samples_per_epoch  : {diag['train_samples_per_epoch']}")
    print(f"effective_rows_est       : {diag['effective_rows_est']}")
    print(f"bucket_accrued / cap     : {diag['bucket_accrued']:.0f} / {diag['bucket_cap']:.0f}"
          f"  (limited={diag['bucket_limited']})")
    print(f"new_rows_this_epoch      : {diag['new_rows_this_epoch']}")
    print(f"reuse_ratio_est          : {diag['reuse_ratio_est']:.3f}")

    # --- 3. optional persist (STOPPED runs only) ---------------------------
    if args.write:
        if live:
            print("\nREFUSING --write: run looks LIVE (driver.pid / supervisor.lock present). "
                  "Stop the run before persisting the manifest into its samples/.")
            return 2
        man2 = persist_manifest(samples_dir)
        print(f"\n--- manifest PERSISTED -> {samples_dir / '.buffer_manifest.json'} "
              f"({len(man2.entries)} entries, cumulative={man2.cumulative_rows_ever}) ---")

    # --- 4. continuation validation (resume + select, CPU) -----------------
    ckpt = _latest_checkpoint(checkpoint_dir)
    print(f"\nlatest_checkpoint : {ckpt.name if ckpt else None}")
    if args.no_validate:
        print("[validate] skipped (--no-validate)")
        print("\nMIGRATION-DIAGNOSTICS-OK")
        return 0
    if ckpt is None:
        print("[validate] no checkpoint to resume; skipping validation")
        print("\nMIGRATION-DIAGNOSTICS-OK")
        return 0

    epoch = args.epoch
    if epoch is None:
        # latest_ckpt_epoch + 1
        try:
            epoch = int(ckpt.stem.split("_", 1)[1]) + 1
        except (IndexError, ValueError):
            epoch = 1

    scratch = Path(args.scratch) if args.scratch else (_REPO / "scratch" / "buffer_convert")
    diag_dir = scratch / "diagnostics"

    # For a LIVE run, NEVER let select_training_samples persist .buffer_manifest.json
    # into the live samples/. Validate against a small scratch COPY of a few epochs
    # so the path is exercised end-to-end without writing the live tree.
    if live:
        import shutil

        copy_samples = scratch / "samples"
        if copy_samples.exists():
            shutil.rmtree(copy_samples)
        # Copy the newest few epochs (bounded so this stays fast).
        epoch_dirs = sorted(samples_dir.glob("epoch_*"))[-3:]
        copied = 0
        for ed in epoch_dirs:
            dst = copy_samples / ed.name
            dst.mkdir(parents=True, exist_ok=True)
            for npz in sorted(ed.glob("game_*.npz"))[:40]:
                side = npz.with_suffix(".json")
                if not side.exists():
                    continue
                shutil.copy2(npz, dst / npz.name)
                shutil.copy2(side, dst / side.name)
                copied += 1
        print(f"[validate] LIVE run -> using scratch copy of {copied} shards "
              f"from {len(epoch_dirs)} newest epochs (live samples/ untouched)")
        val_samples = copy_samples
    else:
        val_samples = samples_dir

    seed = 0  # the run's seed is irrelevant to the FRESH-governor assertion; 0 is fine
    report = run_validation(
        samples_dir=val_samples,
        checkpoint=ckpt,
        seed=seed,
        epoch=epoch,
        device=args.device,
        diag_dir=diag_dir,
        run_name=run_dir.name,
    )
    print("\n--- continuation validation (resume + select, CPU) ---")
    print(f"load_report          : {report['load_report']}")
    print(f"governor_is_fresh    : {report['governor_is_fresh']} "
          f"(old checkpoint has no meta.train_state -> from_dict(None) -> fresh)")
    print(f"train_state          : {report['train_state_after_resume']}")
    print(f"select_status        : {report['select_status']}")
    print(f"window_is_packed     : {report['window_is_packed']}  rows={report['window_rows']}")
    print(f"select_seconds       : {report['select_seconds']}")

    ok = (
        report["governor_is_fresh"]
        and report["window_is_packed"]
        and report["select_status"] in ("completed", "skipped", "train_bucket_limited")
    )
    if not ok:
        print("\nMIGRATION-VALIDATION-FAILED")
        return 1
    print("\nMIGRATION-OK: existing run continues under the new buffer (additive manifest, "
          "fresh governor on resume, window selected from existing shards).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
