#!/usr/bin/env python3
"""Validate hexfield -> hexfield_eq shard conversion (see convert_hexfield_shards_to_eq.py).

For each sampled (source, converted) shard pair this checks, HARD (exit 1 on
any failure):

  1. Column transform: converted key set == source keys minus the 8 hot/win
     arrays; schema_version == 4; every preserved column byte-identical
     (dtype + values).
  2. Reader equivalence: ``hexfield_eq.shards.read_compact_shard`` decodes BOTH
     files (the eq reader accepts v3 sources) and every decoded row field is
     identical — policy / q_policy / gumbel_policy / prior_logit / opp_policy /
     value / short_term_value / moves_left / records / metadata / phase /
     turn_index / current_player / first_stone.
  3. Expand: ``expand_sample`` on converted rows yields 25-plane features
     (NUM_FEATURES), a (N, RAYLEN_SLOTS) uint8 ``raylen`` array, and expanded
     policy/value/gumbel targets numerically identical to expanding the source
     read (byte-identical inputs => this is the end-to-end confirmation).

SOFT (reported, not failed): whether ``collate_training`` threads ``raylen``
into the model batch (the ray-attention Phase L threading may land after the
converter; the plan's L2 says the mask is rebuilt from raylen at the model).

Also prints the prefit policy-target status: main_11 rows carry the Gumbel
completedQ improved-policy target (gumbel_present); ``hexfield_eq.prefit``
selects it via ``--policy-target gumbel`` (checklist B1, landed 2026-07-08) —
ladder launches must pass that flag.

Run (WSL; needs numpy + torch + the eq package on PYTHONPATH — no Rust needed,
the eq read/expand chain is pure Python; set the deployment support radius so
the expand matches the prefit env):

  HEXFIELD_EQ_SUPPORT_RADIUS=4 HEXFIELD_EQ_CHANNELS=192 \
  PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield_eq/python \
      /root/.venvs/hexgt-build/bin/python \
      /mnt/e/Hexo-BotTrainer-hexgt/scripts/validate_eq_shard_conversion.py \
      --src /mnt/e/Hexo-BotTrainer/runs/hexfield_main_11/samples \
      --out /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit_main11

Full-corpus check after the full conversion: add ``--all`` (column + reader
checks on every pair; expand stays sampled via --expand-rows).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_SRC = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_11/samples"
DEFAULT_OUT = "/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit_main11"

DROPPED_KEYS = tuple(
    f"{group}_{part}"
    for group in ("own_hot", "opp_hot", "own_win", "opp_win")
    for part in ("qr", "off")
)

ROW_FIELDS = (
    "turn_index",
    "current_player",
    "phase",
    "records",
    "first_stone",
    "policy",
    "q_policy",
    "gumbel_policy",
    "prior_logit",
    "opp_policy",
    "value",
    "short_term_value",
    "moves_left",
)

_failures: list[str] = []


def _check(ok: bool, label: str) -> bool:
    if not ok:
        _failures.append(label)
        print(f"  FAIL  {label}")
    return ok


def source_of(converted: Path, src_root: Path) -> Path:
    """shard_game_<key>.npz -> <src>/epoch_<key//1e6>/game_<key>.npz."""

    stem = converted.stem
    if not stem.startswith("shard_"):
        raise ValueError(f"unexpected converted shard name {converted.name!r}")
    game = stem[len("shard_") :]
    key = int(game.split("_")[1])
    return src_root / f"epoch_{key // 1_000_000:06d}" / f"{game}.npz"


def check_columns(src_npz: Path, dst_npz: Path) -> bool:
    with np.load(src_npz) as d:
        src = {k: d[k] for k in d.files}
    with np.load(dst_npz) as d:
        dst = {k: d[k] for k in d.files}
    ok = True
    expected = set(src) - set(DROPPED_KEYS)
    ok &= _check(set(dst) == expected, f"{dst_npz.name}: key set == source minus hot/win")
    ok &= _check(int(dst["schema_version"]) == 4, f"{dst_npz.name}: schema_version == 4")
    for key in sorted(expected - {"schema_version"}):
        same = src[key].dtype == dst[key].dtype and np.array_equal(src[key], dst[key])
        if not same:
            ok &= _check(False, f"{dst_npz.name}: column {key!r} byte-identical")
    return ok


def check_reader(src_npz: Path, dst_npz: Path, read_compact_shard) -> tuple[bool, list, list]:
    rows_src = read_compact_shard(src_npz)
    rows_dst = read_compact_shard(dst_npz)
    ok = _check(len(rows_src) == len(rows_dst), f"{dst_npz.name}: row count matches")
    for i, (a, b) in enumerate(zip(rows_src, rows_dst)):
        for f in ROW_FIELDS:
            if getattr(a, f) != getattr(b, f):
                ok &= _check(False, f"{dst_npz.name} row {i}: field {f!r} identical")
        if dict(a.metadata) != dict(b.metadata):
            ok &= _check(False, f"{dst_npz.name} row {i}: metadata identical")
    return ok, rows_src, rows_dst


def check_expand(rows_src, rows_dst, n_rows: int) -> bool:
    from hexfield_eq import samples as _samples
    from hexfield_eq.batching import collate_training
    from hexfield_eq.constants import NUM_FEATURES, RAYLEN_SLOTS
    from hexfield_eq.samples import expand_sample

    # Validation exercises the full data path, so force the serial-expand raylen
    # oracle on even when the ambient TRUNK_LAYOUT has no L blocks (spec D-S29
    # gates it off for C/A arms as a prefit-worker perf optimization).
    _samples._EXPAND_RAYLEN = True

    ok = True
    expanded = []
    for i, (a, b) in enumerate(zip(rows_src[:n_rows], rows_dst[:n_rows])):
        ea = expand_sample(a, symmetry=0)
        eb = expand_sample(b, symmetry=0)
        expanded.append(eb)
        ok &= _check(
            eb.feats.shape[1] == NUM_FEATURES,
            f"expand row {i}: feats width {eb.feats.shape[1]} == NUM_FEATURES ({NUM_FEATURES})",
        )
        rl = getattr(eb, "raylen", None)
        ok &= _check(
            rl is not None
            and rl.shape == (eb.support.num_nodes, RAYLEN_SLOTS)
            and rl.dtype == np.uint8,
            f"expand row {i}: raylen (N, {RAYLEN_SLOTS}) uint8 present",
        )
        for f in ("policy", "opp_policy", "gumbel_policy", "cell_q", "prior_logit",
                  "stvalue", "stvalue_mask", "feats"):
            if not np.array_equal(getattr(ea, f), getattr(eb, f)):
                ok &= _check(False, f"expand row {i}: {f} identical source vs converted")
        for f in ("value", "value_mask", "moves_left", "moves_left_mask",
                  "gumbel_policy_valid", "policy_surprise", "opp_coverage"):
            if float(getattr(ea, f)) != float(getattr(eb, f)):
                ok &= _check(False, f"expand row {i}: {f} identical source vs converted")

    batch = collate_training(expanded)
    print(f"  collate_training keys: {sorted(batch.keys())}")
    ok &= _check(batch["feats"].shape[-1] == NUM_FEATURES, "batch feats last dim == NUM_FEATURES")
    ok &= _check("gumbel_policy" in batch and "gumbel_policy_valid" in batch,
                 "batch carries gumbel_policy + gumbel_policy_valid")
    if "raylen" in batch:
        print("  [ok]   collate_training threads 'raylen' into the model batch")
    else:
        print(
            "  [soft] collate_training does NOT yet thread 'raylen' into the model "
            "batch (expand carries it; Phase L threading pending — fine for C/A-only arms)"
        )
    n_gumbel = int((batch["gumbel_policy_valid"] > 0).sum())
    print(f"  gumbel_policy_valid rows in expand sample: {n_gumbel}/{len(expanded)}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--pairs", type=int, default=4, help="shard pairs to check (ignored with --all)")
    parser.add_argument("--expand-rows", type=int, default=8, help="rows for the expand check")
    parser.add_argument("--all", action="store_true", help="column+reader checks on every converted shard")
    args = parser.parse_args(argv)

    from hexfield_eq.constants import NUM_FEATURES
    from hexfield_eq.shards import read_compact_shard
    from hexfield_eq.support import _SUPPORT_RADIUS

    print(f"effective NUM_FEATURES={NUM_FEATURES} HEXFIELD_EQ_SUPPORT_RADIUS={_SUPPORT_RADIUS}")

    out = Path(args.out)
    converted = sorted((out / "train").glob("shard_*.npz")) + sorted(
        (out / "val").glob("shard_*.npz")
    )
    if not converted:
        print(f"ERROR: no converted shards under {out}/{{train,val}}", file=sys.stderr)
        return 2
    if not args.all:
        converted = converted[: args.pairs]

    src_root = Path(args.src)
    first_rows: tuple | None = None
    for dst_npz in converted:
        src_npz = source_of(dst_npz, src_root)
        if not _check(src_npz.exists(), f"{dst_npz.name}: source {src_npz} exists"):
            continue
        print(f"pair: {src_npz.relative_to(src_root)} -> {dst_npz.relative_to(out)}")
        check_columns(src_npz, dst_npz)
        _, rows_src, rows_dst = check_reader(src_npz, dst_npz, read_compact_shard)
        if first_rows is None:
            first_rows = (rows_src, rows_dst)

    if first_rows is not None:
        print(f"expand check ({args.expand_rows} rows of {converted[0].name}):")
        check_expand(first_rows[0], first_rows[1], args.expand_rows)

    print()
    print(
        "NOTE: main_11 rows carry the Gumbel completedQ target (gumbel_present); "
        "ladder launches must pass --policy-target gumbel to hexfield_eq.prefit "
        "(checklist B1, landed 2026-07-08)."
    )
    if _failures:
        print(f"\nRESULT: FAIL ({len(_failures)} failed checks)")
        return 1
    print(f"\nRESULT: PASS ({len(converted)} shard pairs validated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
