#!/usr/bin/env python3
"""Build the main_9 BC-prefit dataset from OLD main_8 ep7 shards.

main_9 NEVER records/trains on Fast (PCR value-only) rows, and the shard schema
has had its ``policy_valid`` column removed entirely. The old main_8 shards
still carry ``policy_valid`` (1 == full search / ``pcr_full``, 0 == fast). This
script bridges the two schemas: it selects only the FULL rows from the old
shards and rewrites them through the CURRENT (post-strip) shard writer, so the
prefit dataset contains no fast rows and no ``policy_valid`` column.

It is schema-agnostic on the read side: rows are decoded with
``read_compact_shard`` (which accepts the old v3 shards and simply ignores the
unknown ``policy_valid`` column), while ``policy_valid`` itself is pulled from
the raw ``.npz`` and paired to the decoded rows by index. The write side uses
``write_compact_shard`` unchanged, which emits the new schema.

Mirrors the intent of scripts/_main7_prefit_data.sh (newest-epoch shards, ~5%
held out to val/), but does row-level FULL selection + a deterministic 20k cap
and re-serializes instead of symlinking (the schema changed, so symlinks to old
shards would carry the dropped column).

Run (WSL):
    PYTHONPATH=packages/hexfield/python \
      /root/.venvs/hexgt-build/bin/python scripts/_main9_prefit_data.py

Defaults match the task spec; override with flags if needed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

# Path-shim so the script runs without installing the package (mirrors
# scripts/_hexfield_prefit.py).
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages" / "hexfield" / "python")
)

from hexfield.samples import STV_HORIZONS  # noqa: E402
from hexfield.shards import read_compact_shard, write_compact_shard  # noqa: E402

DEFAULT_SRC = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_8/samples/epoch_000007"
DEFAULT_OUT = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_9_prefit"
DEFAULT_CAP = 20_000
DEFAULT_SEED = 1234
# Rows per output shard. Old main_8 shards are ~one game each (~170 rows); we
# repack into fixed-size shards so the count is independent of source game
# lengths.
ROWS_PER_SHARD = 512
# 1-in-VAL_EVERY output shards go to val/ (mirrors _main7_prefit_data.sh's ~5%).
VAL_EVERY = 20


def _load_full_rows(shard: Path) -> tuple[int, int, list]:
    """Decode ``shard`` and return (n_total, n_full, full_rows).

    Rows are kept where ``policy_valid == 1`` (full search / pcr_full). Shards
    lacking the ``policy_valid`` column (there should be none here, but be
    defensive) are treated as all-full, matching read_compact_shard semantics.
    """

    rows = read_compact_shard(shard)
    with np.load(shard) as data:
        policy_valid = data["policy_valid"] if "policy_valid" in data.files else None
    if policy_valid is None:
        return len(rows), len(rows), list(rows)
    policy_valid = np.asarray(policy_valid).reshape(-1)
    if policy_valid.shape[0] != len(rows):
        raise ValueError(
            f"{shard.name}: policy_valid len {policy_valid.shape[0]} != rows {len(rows)}"
        )
    full = [r for r, v in zip(rows, policy_valid) if int(v) == 1]
    return len(rows), len(full), full


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=DEFAULT_SRC, help="old main_8 epoch dir with game_*.npz")
    parser.add_argument("--out", default=DEFAULT_OUT, help="prefit run dir (samples/ + train/ + val/)")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="max full rows to keep")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="deterministic subsample seed")
    parser.add_argument("--rows-per-shard", type=int, default=ROWS_PER_SHARD)
    args = parser.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    shard_paths = sorted(src.glob("game_*.npz"))
    if not shard_paths:
        raise SystemExit(f"no game_*.npz under {src}")

    rows_read = 0
    full_rows: list = []
    for path in shard_paths:
        n_total, n_full, full = _load_full_rows(path)
        rows_read += n_total
        full_rows.extend(full)
    full_read = len(full_rows)

    # Deterministic cap: fixed-seed permutation, then keep the first `cap`. The
    # permutation (rather than a head slice) avoids biasing toward the earliest
    # games/opening plies. Sorted afterward is unnecessary — order within the
    # dataset does not matter to the trainer's per-epoch reshuffle.
    rng = np.random.default_rng(args.seed)
    if full_read > args.cap:
        keep_idx = rng.permutation(full_read)[: args.cap]
        keep_idx.sort()
        selected = [full_rows[i] for i in keep_idx]
    else:
        selected = full_rows

    # Fresh output tree (samples/ is the schema-agnostic sink named by the task;
    # train/ + val/ are the split the prefit launcher globs).
    samples_dir = out / "samples"
    train_dir = out / "train"
    val_dir = out / "val"
    for d in (samples_dir, train_dir, val_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    starts = list(range(0, len(selected), args.rows_per_shard))
    n_shards = len(starts)
    # Decide the train/val split up front (mirrors _main7_prefit_data.sh's ~5%
    # to val). With few output shards the 1-in-VAL_EVERY rule can select none, so
    # force the LAST shard to val when it otherwise would have picked zero — the
    # prefit launcher raises on an empty val/.
    val_ids = {i for i in range(n_shards) if i % VAL_EVERY == VAL_EVERY - 1}
    if not val_ids and n_shards > 1:
        val_ids = {n_shards - 1}

    written = 0
    n_val_shards = 0
    for shard_i, start in enumerate(starts):
        chunk = selected[start : start + args.rows_per_shard]
        name = f"shard_{shard_i:06d}.npz"
        # Primary sink (new-schema shards, no policy_valid).
        written += write_compact_shard(
            samples_dir / name, chunk, short_term_value_horizons=STV_HORIZONS,
            sidecar={"source": "main_8_ep7_full_only", "lineage_note": "main_9 prefit"},
        )
        # Mirror into the train/val split the prefit launcher expects. Reuse the
        # already-written npz+sidecar via copy (identical bytes, no re-encode).
        split_dir = val_dir if shard_i in val_ids else train_dir
        if split_dir is val_dir:
            n_val_shards += 1
        for suffix in (".npz", ".json"):
            shutil.copy2(
                (samples_dir / name).with_suffix(suffix),
                (split_dir / name).with_suffix(suffix),
            )

    print("=== main_9 prefit data ===")
    print(f"source epoch dir : {src}")
    print(f"source shards    : {len(shard_paths)}")
    print(f"rows read        : {rows_read}")
    print(f"full rows        : {full_read}  ({100.0 * full_read / max(rows_read, 1):.1f}% of read)")
    print(f"cap              : {args.cap}  (seed {args.seed})")
    print(f"rows written     : {written}")
    print(f"output shards    : {n_shards}  ({n_shards - n_val_shards} train / {n_val_shards} val)")
    print(f"samples dir      : {samples_dir}")
    print(f"train/val root   : {out}")
    if written != len(selected):
        raise SystemExit(f"BUG: wrote {written} != selected {len(selected)}")


if __name__ == "__main__":
    main()
