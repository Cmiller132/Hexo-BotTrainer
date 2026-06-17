"""Migration self-test — continue hexfield_main_2 under the new KataGo buffer
(PLAN §10 + the migration converter scripts/_hexfield_buffer_convert.py).

CPU-ONLY, no GPU, no live-run interaction. The live ``hexfield_main_2`` tree is
touched **READ-ONLY**: the manifest is built over its real ``samples/`` IN-MEMORY
(never persisting ``.buffer_manifest.json`` there), and the newest checkpoint +
a handful of real shards are COPIED into ``_scratch/migration/`` so every write
lands under scratch. We NEVER write under ``runs/*``.

What this verifies (each block prints its line; ``main`` prints PASS):

  1. MANIFEST (read-only on real main_2). ``build_manifest_readonly`` over the
     live ``samples/`` indexes the real shards: entries sorted by
     ``(generation, game_key)``, ``total_rows > 0``, ``cumulative_rows_ever > 0``
     and ``>= total_rows`` (monotone), generations span the real epochs, and the
     live tree gets NO ``.buffer_manifest.json`` written into it (read-only).

  2. WINDOW DIAGNOSTICS. ``window_governor_diagnostics`` over the real manifest
     produces a sane taper window: ``desired_rows >= shuffle_min_rows``,
     ``used_rows >= desired_rows`` (whole-shard overshoot), ``0 < keep_prob <=
     1``, and a finite reuse estimate.

  3. CHECKPOINT RESUME -> FRESH GOVERNOR. COPY the newest checkpoint
     (``epoch_000026.pt`` or later) into ``_scratch/`` and ``torch.load`` it on
     CPU: assert it is the OLD format (no ``meta.train_state``). Drive the real
     ``HexfieldCheckpointLoader`` resume semantics into a fresh
     ``HexfieldNet`` + ``HexfieldTrainer`` and assert ``trainer.train_state`` is
     FRESH (``from_dict(None)``) with NO crash, and the model weights loaded
     (key-exact).

  4. SELECT A WINDOW from main_2 real shards. Run
     ``HexfieldTrainer.select_training_samples`` (via the converter's
     ``run_validation`` orchestration) against a scratch COPY of real main_2
     shards and assert a valid ``PackedWindow`` (``window.n > 0``) is produced and
     the resume came up with a fresh governor — i.e. the end-to-end continuation
     path works.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from hexfield.config import TrainingSection
from hexfield.train_state import HexfieldTrainState
from hexfield.window import PackedWindow

# --- import the migration converter (scripts/ is not a package) -------------
_REPO = Path(__file__).resolve().parents[2]
_CONVERT = _REPO / "scripts" / "_hexfield_buffer_convert.py"
_spec = importlib.util.spec_from_file_location("_hexfield_buffer_convert", _CONVERT)
assert _spec and _spec.loader, f"cannot load converter at {_CONVERT}"
convert = importlib.util.module_from_spec(_spec)
sys.modules["_hexfield_buffer_convert"] = convert
_spec.loader.exec_module(convert)

LIVE_RUN = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2")
LIVE_SAMPLES = LIVE_RUN / "samples"
LIVE_CKPTS = LIVE_RUN / "checkpoints"
SCRATCH = Path(__file__).resolve().parent / "_scratch" / "migration"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _newest_checkpoint() -> Path:
    cks = sorted(LIVE_CKPTS.glob("epoch_*.pt"))
    assert cks, f"no checkpoints in {LIVE_CKPTS}"
    return cks[-1]


def _copy_recent_real_shards(dst_samples: Path, *, n_epochs: int = 3, per_epoch: int = 40) -> int:
    """COPY a few real (npz+json) shard pairs from the newest live epochs into a
    writable scratch tree (read-only on the live tree — copies out)."""
    if dst_samples.exists():
        shutil.rmtree(dst_samples)
    copied = 0
    for ed in sorted(LIVE_SAMPLES.glob("epoch_*"))[-n_epochs:]:
        dst = dst_samples / ed.name
        dst.mkdir(parents=True, exist_ok=True)
        for npz in sorted(ed.glob("game_*.npz"))[:per_epoch]:
            side = npz.with_suffix(".json")
            if not side.exists():
                continue
            shutil.copy2(npz, dst / npz.name)
            shutil.copy2(side, dst / side.name)
            copied += 1
    return copied


# ---------------------------------------------------------------------------
# 1. manifest over real main_2 (read-only)
# ---------------------------------------------------------------------------
def test_manifest_over_real_main2() -> None:
    live_manifest = LIVE_SAMPLES / ".buffer_manifest.json"
    pre_existing = live_manifest.exists()  # must NOT change due to us

    manifest, n_npz, n_skipped = convert.build_manifest_readonly(LIVE_SAMPLES)

    assert manifest.entries, "manifest indexed zero shards over real main_2"
    assert n_npz >= len(manifest.entries) >= 1
    # ordering: strictly by (generation, game_key)
    keys = [(e.generation, e.game_key) for e in manifest.entries]
    assert keys == sorted(keys), "real-shard manifest not sorted by (generation, game_key)"
    # totals: live > 0 and monotone cumulative >= live
    assert manifest.total_rows > 0, "total_rows must be > 0 over real shards"
    assert manifest.cumulative_rows_ever > 0, "cumulative_rows_ever must be > 0"
    assert manifest.cumulative_rows_ever >= manifest.total_rows, "cumulative < live total (non-monotone)"
    assert manifest.total_rows == sum(e.rows for e in manifest.entries)
    # generations span the real epochs (>= a couple distinct), all derived w/o mtime
    gens = sorted({e.generation for e in manifest.entries})
    assert len(gens) >= 2, f"expected multiple generations, got {gens[:5]}..."
    assert min(gens) >= 1, "generation should be the producing epoch (>=1)"
    # every entry's generation == game_key // 1_000_000 (key-derived, authoritative)
    for e in manifest.entries[:200]:
        assert e.generation == e.game_key // 1_000_000, (e.rel_path, e.generation, e.game_key)

    # READ-ONLY guarantee: we did not create the live manifest.
    if not pre_existing:
        assert not live_manifest.exists(), (
            "build_manifest_readonly wrote .buffer_manifest.json into the LIVE samples tree!"
        )

    print(f"  1. MANIFEST (read-only real main_2): shards={len(manifest.entries)} "
          f"total_rows={manifest.total_rows} cumulative={manifest.cumulative_rows_ever} "
          f"generations={gens[0]}..{gens[-1]} (live tree NOT written)")
    return manifest


# ---------------------------------------------------------------------------
# 2. window + governor diagnostics over the real manifest
# ---------------------------------------------------------------------------
def test_window_diagnostics(manifest) -> None:
    training = TrainingSection()  # PLAN §7 defaults
    diag = convert.window_governor_diagnostics(manifest, training, prev_level_at_row=0)

    assert diag["desired_rows"] >= training.shuffle_min_rows, (
        f"taper desired_rows {diag['desired_rows']} < min_rows {training.shuffle_min_rows}"
    )
    # whole-shard recent-window overshoots desired by < one shard, so used >= desired
    assert diag["used_rows"] >= diag["desired_rows"] or diag["used_rows"] == manifest.total_rows, (
        f"used_rows {diag['used_rows']} < desired {diag['desired_rows']}"
    )
    assert diag["window_shards"] >= 1
    assert 0.0 < diag["keep_prob"] <= 1.0, diag["keep_prob"]
    assert diag["window_start"] == max(0, manifest.total_rows - diag["used_rows"])
    assert diag["cumulative_rows_ever"] == manifest.cumulative_rows_ever
    assert np.isfinite(diag["reuse_ratio_est"]) and diag["reuse_ratio_est"] >= 0.0
    assert diag["bucket_accrued"] <= diag["bucket_cap"] + 1e-6

    print(f"  2. WINDOW DIAG: desired_rows={diag['desired_rows']} used_rows={diag['used_rows']} "
          f"({diag['window_shards']} shards) keep_prob={diag['keep_prob']:.4f} "
          f"reuse_est={diag['reuse_ratio_est']:.3f}")


# ---------------------------------------------------------------------------
# 3. copy newest checkpoint, torch.load on CPU, resume -> FRESH governor
# ---------------------------------------------------------------------------
def test_checkpoint_resume_fresh_governor() -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    src_ckpt = _newest_checkpoint()
    epoch_num = int(src_ckpt.stem.split("_", 1)[1])
    assert epoch_num >= 26, f"newest checkpoint epoch {epoch_num} < 26 (task expects >=26)"
    dst_ckpt = SCRATCH / src_ckpt.name
    shutil.copy2(src_ckpt, dst_ckpt)

    # torch.load on CPU; confirm OLD format (no train_state in meta).
    payload = torch.load(dst_ckpt, map_location="cpu", weights_only=False)
    assert set(payload.keys()) >= {"meta", "model", "optimizer"}, sorted(payload.keys())
    meta = payload["meta"]
    assert "train_state" not in meta, (
        "newest main_2 checkpoint already carries train_state; migration premise (old format) broken"
    )
    assert int(meta.get("epoch", -1)) == epoch_num

    # Drive the REAL HexfieldCheckpointLoader resume semantics into a fresh trainer.
    # run_validation needs a writable samples dir (select_training_samples persists
    # a manifest), so use a scratch COPY of real shards (the live tree stays
    # read-only). It loads -> asserts fresh governor -> selects a window.
    val_samples = SCRATCH / "samples"
    copied = _copy_recent_real_shards(val_samples, n_epochs=3, per_epoch=40)
    assert copied > 0, "expected to copy at least one real shard for the selection"

    report = convert.run_validation(
        samples_dir=val_samples,
        checkpoint=dst_ckpt,
        seed=0,
        epoch=epoch_num + 1,
        device="cpu",
        diag_dir=SCRATCH / "diag",
        run_name="hexfield_main_2",
    )

    # Resume succeeded and reported the right epoch.
    assert report["load_report"]["status"] == "loaded", report["load_report"]
    assert report["load_report"]["epoch"] == epoch_num
    # Governor came up FRESH (old ckpt has no train_state -> from_dict(None)).
    assert report["governor_is_fresh"], (
        f"governor not fresh after resuming an old-format checkpoint: "
        f"{report['train_state_after_resume']}"
    )
    print(f"  3. RESUME: copied {dst_ckpt.name} (epoch {epoch_num}), torch.load CPU OK, "
          f"no meta.train_state -> from_dict(None) -> FRESH governor, model weights loaded")
    return report


# ---------------------------------------------------------------------------
# 4. select_training_samples yields a valid PackedWindow from real shards
# ---------------------------------------------------------------------------
def test_select_window_from_real_shards(report) -> None:
    assert report["window_is_packed"], "select_training_samples did not set a PackedWindow"
    assert report["select_status"] in ("completed", "skipped", "train_bucket_limited"), (
        report["select_status"]
    )
    # The copied real shards are plentiful enough that a window is non-empty.
    assert report["select_status"] == "completed", (
        f"expected a completed selection over the copied real shards, got "
        f"{report['select_status']}: {report['select']}"
    )
    assert report["window_rows"] is not None and report["window_rows"] > 0, (
        f"PackedWindow is empty: window_rows={report['window_rows']}"
    )
    sel = report["select"]
    # the selection dict carries the PLAN §6 diagnostic fields
    for k in ("total_rows", "live_total_rows", "desired_rows", "used_rows",
              "keep_prob", "effective_rows", "window_rows", "window_start",
              "train_bucket_level", "reuse_ratio"):
        assert k in sel, f"selection result missing field {k}: {sorted(sel)}"
    assert sel["window_rows"] == report["window_rows"]
    assert sel["effective_rows"] >= 0
    print(f"  4. SELECT: PackedWindow with {report['window_rows']} rows produced from real "
          f"main_2 shards (status={report['select_status']}, effective_rows={sel['effective_rows']}, "
          f"keep_prob={sel['keep_prob']:.3f})")


# ---------------------------------------------------------------------------
# direct loader/train_state round-trip sanity (no live data) — belt & suspenders
# ---------------------------------------------------------------------------
def test_from_dict_none_is_fresh() -> None:
    """The load-bearing primitive: from_dict(None) (old-format meta) == fresh."""
    fresh = HexfieldTrainState()
    assert HexfieldTrainState.from_dict(None).to_dict() == fresh.to_dict()
    assert HexfieldTrainState.from_dict({}).to_dict() == fresh.to_dict()
    # a meta lacking the key behaves identically.
    assert HexfieldTrainState.from_dict({"epoch": 30}.get("train_state")).to_dict() == fresh.to_dict()
    print("  +. from_dict(None) / {} / missing-key -> FRESH HexfieldTrainState (old-format safe)")


def main() -> int:
    if not LIVE_SAMPLES.exists():
        print("FAIL: live main_2 samples tree missing:", LIVE_SAMPLES)
        return 1
    if not LIVE_CKPTS.exists() or not any(LIVE_CKPTS.glob("epoch_*.pt")):
        print("FAIL: live main_2 checkpoints missing:", LIVE_CKPTS)
        return 1
    # The key-derived/sidecar generation cross-check may WARN on a torn sidecar;
    # surface warnings but don't fail the read-only scan.
    warnings.simplefilter("default")

    manifest = test_manifest_over_real_main2()
    test_window_diagnostics(manifest)
    report = test_checkpoint_resume_fresh_governor()
    test_select_window_from_real_shards(report)
    test_from_dict_none_is_fresh()

    # Final read-only guarantee re-check: the live samples tree must still have no
    # manifest written by us (only the scratch copy got one).
    assert not (LIVE_SAMPLES / ".buffer_manifest.json").exists(), (
        "a .buffer_manifest.json leaked into the LIVE samples tree"
    )
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
