#!/usr/bin/env python3
"""Convert hexfield-lineage compact shards (schema v1-v3) to hexfield_eq schema v4.

The two schemas are identical except that hexfield_eq v4 retires the four
hot/standing-win cell CSR groups (own_hot/opp_hot/own_win/opp_win, one *_qr +
one *_off array each) — the eq expand path recomputes the graded per-axis
window planes (and ray lengths) from the placement history, so nothing needs
to be featurized here. Conversion is therefore a PURE COLUMN TRANSFORM:

  * drop the 8 hot/win arrays,
  * rewrite the schema_version scalar to 4,
  * preserve every other column BYTE-EXACT (same dtype, same values).

Deliberately implemented with numpy only (no hexfield / hexfield_eq import):

  * byte-exact preservation is stronger than an object-level decode/re-encode
    round trip — and the round trip is actually LOSSY: ``read_compact_shard``
    (both lineages) does not read ``policy_surprise`` back into the sample, so
    re-serializing through ``HexfieldSampleData`` would silently zero the
    self-policy CE weights. The column transform keeps them.
  * no Rust extension / torch dependency, so the converter runs on any host
    (the VALIDATION script is the piece that needs the eq package under WSL).

Output layout (what ``hexfield_eq.prefit`` globs: ``<out>/{train,val}/shard_*.npz``):

  <out>/train/shard_game_<key>.npz + .json
  <out>/val/shard_game_<key>.npz   + .json     (game_key % VAL_EVERY == VAL_EVERY-1)

The val split is keyed on the source ``game_key`` (globally unique across
epochs: epoch*1e6 + index), so the assignment is deterministic and independent
of enumeration order / partial runs. Output naming is derived purely from the
source filename, so re-running is idempotent: existing outputs (npz AND
sidecar present) are skipped.

Sidecars are copied with ``schema_version`` rewritten to 4 plus provenance
keys; the npz-first-then-sidecar atomic commit order of shards.py is kept.

Run (WSL, from the dev repo; pure numpy so the venv only needs numpy):

  /root/.venvs/hexgt-build/bin/python \
      /mnt/e/Hexo-BotTrainer-hexgt/scripts/convert_hexfield_shards_to_eq.py \
      --src /mnt/e/Hexo-BotTrainer/runs/hexfield_main_11/samples \
      --out /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit_main11

Sampling (e.g. 2 shards for the conversion validation):  add ``--limit 2``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_SRC = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_11/samples"
DEFAULT_OUT = "/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit_main11"

# The four hot/standing-win CSR groups retired by hexfield_eq schema v4
# (hexfield_eq/shards.py SCHEMA_VERSION docstring).
DROPPED_KEYS = tuple(
    f"{group}_{part}"
    for group in ("own_hot", "opp_hot", "own_win", "opp_win")
    for part in ("qr", "off")
)

# Source schema versions this converter accepts (hexfield shards.py
# _ACCEPTED_SCHEMA_VERSIONS). main_11 is v3 throughout.
ACCEPTED_SRC_VERSIONS = (1, 2, 3)
EQ_SCHEMA_VERSION = 4

# 1-in-VAL_EVERY games go to val/ (~5%, mirrors scripts/_main9_prefit_data.py).
VAL_EVERY = 20

_GAME_KEY_RE = re.compile(r"game_(\d+)")


def _atomic_write_npz(path: Path, arrays: dict) -> None:
    """np.savez_compressed via tmp handle + os.replace (shards.py convention:
    savez appends `.npz` to suffix-less names, so pass the file HANDLE)."""

    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.unlink()
    except OSError:
        pass
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.unlink()
    except OSError:
        pass
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def convert_one(src_npz: Path, dst_npz: Path) -> dict:
    """Transform one shard; returns per-shard stats. Raises on schema mismatch."""

    with np.load(src_npz) as data:
        arrays = {key: data[key] for key in data.files}
    src_version = int(arrays["schema_version"])
    if src_version not in ACCEPTED_SRC_VERSIONS:
        raise ValueError(
            f"{src_npz.name}: unsupported source schema_version {src_version} "
            f"(accepted: {ACCEPTED_SRC_VERSIONS})"
        )

    dropped = [k for k in DROPPED_KEYS if k in arrays]
    out_arrays = {k: v for k, v in arrays.items() if k not in DROPPED_KEYS}
    out_arrays["schema_version"] = np.asarray(EQ_SCHEMA_VERSION, dtype=np.int32)

    rows = int(arrays["num_rows"])
    gumbel_rows = (
        int(np.asarray(arrays["gumbel_present"]).sum())
        if "gumbel_present" in arrays
        else 0
    )

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    # Commit order: npz first, sidecar last (the sidecar is the commit marker —
    # matches shards.py write_compact_shard).
    _atomic_write_npz(dst_npz, out_arrays)

    sidecar_src = src_npz.with_suffix(".json")
    meta = json.loads(sidecar_src.read_text(encoding="utf-8"))
    meta["schema_version"] = EQ_SCHEMA_VERSION
    meta["converted_from"] = str(src_npz)
    meta["converter"] = "scripts/convert_hexfield_shards_to_eq.py"
    meta["source_schema_version"] = src_version
    _atomic_write_json(dst_npz.with_suffix(".json"), meta)

    return {"rows": rows, "gumbel_rows": gumbel_rows, "dropped": len(dropped)}


def game_key_of(path: Path) -> int:
    m = _GAME_KEY_RE.search(path.stem)
    if m is None:
        raise ValueError(f"cannot derive game_key from shard name {path.name!r}")
    return int(m.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=DEFAULT_SRC, help="samples root with epoch_*/game_*.npz")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output root (train/ + val/ created)")
    parser.add_argument("--epoch-glob", default="epoch_*", help="epoch dir filter under --src")
    parser.add_argument("--limit", type=int, default=0, help="stop after N shards (0 = all)")
    parser.add_argument("--force", action="store_true", help="rewrite outputs that already exist")
    args = parser.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    epoch_dirs = sorted(d for d in src.glob(args.epoch_glob) if d.is_dir())
    if not epoch_dirs:
        print(f"ERROR: no {args.epoch_glob!r} dirs under {src}", file=sys.stderr)
        return 2

    converted = skipped = orphans = 0
    rows_total = gumbel_total = 0
    val_shards = 0
    processed = 0
    for epoch_dir in epoch_dirs:
        for src_npz in sorted(epoch_dir.glob("game_*.npz")):
            if args.limit and processed >= args.limit:
                break
            # Sidecar-less npz = torn write from a power cut; never counted
            # downstream (buffer_manifest convention) — skip it here too.
            if not src_npz.with_suffix(".json").exists():
                orphans += 1
                continue
            key = game_key_of(src_npz)
            split = "val" if key % VAL_EVERY == VAL_EVERY - 1 else "train"
            dst_npz = out / split / f"shard_{src_npz.stem}.npz"
            processed += 1
            if split == "val":
                val_shards += 1
            if (
                not args.force
                and dst_npz.exists()
                and dst_npz.with_suffix(".json").exists()
            ):
                skipped += 1
                continue
            stats = convert_one(src_npz, dst_npz)
            converted += 1
            rows_total += stats["rows"]
            gumbel_total += stats["gumbel_rows"]
        if args.limit and processed >= args.limit:
            break

    print("=== hexfield -> hexfield_eq shard conversion ===")
    print(f"source root      : {src}  ({len(epoch_dirs)} epoch dirs)")
    print(f"output root      : {out}  (train/ + val/, val = game_key % {VAL_EVERY} == {VAL_EVERY - 1})")
    print(f"processed        : {processed}  ({val_shards} assigned to val/)")
    print(f"converted        : {converted}")
    print(f"skipped existing : {skipped}")
    print(f"orphan npz       : {orphans}  (no sidecar; ignored)")
    print(f"rows written     : {rows_total}")
    if converted:
        print(
            f"gumbel_present   : {gumbel_total}/{rows_total} rows "
            f"({100.0 * gumbel_total / max(rows_total, 1):.1f}%) carry the Gumbel "
            "completedQ target"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
