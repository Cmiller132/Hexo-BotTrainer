"""Build and verify the frozen production-throughput seed set.

``BENCH-SEEDS-V1`` contains 256 mid-game positions from
``raws/selfplay_positions.jsonl``: 64 positions from each width-10 placement
band 3, 4, 5, and 6 (30--69 placements).  Selection is by a canonical stable
hash, not input order or ambient RNG, so rebuilding from the same logical input
is deterministic.  Each row carries its selection pin and the first JSONL line
is a manifest whose ``sha256`` pins the complete ordered position payload.

Usage (plain Python; no torch/GPU environment required)::

    python scripts/tss_harness/bench_seeds.py build
    python scripts/tss_harness/bench_seeds.py verify
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:  # Package import (tests / ``python -m``).
    from .contract import SCHEMA_VERSION, stable_hash
except ImportError:  # Direct script execution from the repository root.
    from contract import SCHEMA_VERSION, stable_hash


SET_NAME = "BENCH-SEEDS-V1"
SET_VERSION = 1
BANDS = (3, 4, 5, 6)
PER_BAND = 64
SET_SIZE = len(BANDS) * PER_BAND
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "raws" / "selfplay_positions.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sets" / "bench_seeds_v1.jsonl"


class SeedSetError(ValueError):
    """The source or frozen seed set violates BENCH-SEEDS-V1's contract."""


def _canonical_moves(value: Any, *, row_number: int) -> list[list[int]]:
    if not isinstance(value, list):
        raise SeedSetError(f"source row {row_number}: moves must be a list")
    moves: list[list[int]] = []
    for index, move in enumerate(value):
        if not isinstance(move, (list, tuple)) or len(move) != 2:
            raise SeedSetError(
                f"source row {row_number}: move {index} must be a [q, r] pair"
            )
        q, r = move
        if isinstance(q, bool) or isinstance(r, bool):
            raise SeedSetError(f"source row {row_number}: boolean coordinate")
        try:
            moves.append([int(q), int(r)])
        except (TypeError, ValueError) as exc:
            raise SeedSetError(
                f"source row {row_number}: non-integer coordinate at move {index}"
            ) from exc
    return moves


def normalize_source_row(raw: dict[str, Any], row_number: int) -> dict[str, Any] | None:
    """Normalize one source row, returning ``None`` outside bands 3--6."""

    if not isinstance(raw, dict):
        raise SeedSetError(f"source row {row_number}: expected JSON object")
    moves = _canonical_moves(raw.get("moves"), row_number=row_number)
    placements = int(raw.get("placements", len(moves)))
    if placements != len(moves):
        raise SeedSetError(
            f"source row {row_number}: placements={placements} but has {len(moves)} moves"
        )
    band = placements // 10
    if band not in BANDS:
        return None
    source_id = str(raw.get("id") or f"source-row-{row_number}")
    source = str(raw.get("source") or "selfplay")
    identity = {
        "source_id": source_id,
        "source": source,
        "placements": placements,
        "moves": moves,
    }
    pin = stable_hash({"set": SET_NAME, "position": identity})
    return {
        "id": source_id,
        "source": source,
        "band": band,
        "placements": placements,
        "moves": moves,
        "stable_hash": pin,
    }


def select_seed_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the frozen set from decoded source rows.

    Duplicate positions (identical move prefixes) are collapsed before ranking.
    The rank key is the per-position stable hash with the source id as a final
    deterministic collision tie-break.
    """

    by_band: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_moves: set[str] = set()
    for row_number, raw in enumerate(rows, 1):
        normalized = normalize_source_row(raw, row_number)
        if normalized is None:
            continue
        move_key = stable_hash(normalized["moves"])
        if move_key in seen_moves:
            continue
        seen_moves.add(move_key)
        by_band[normalized["band"]].append(normalized)

    selected: list[dict[str, Any]] = []
    for band in BANDS:
        candidates = sorted(
            by_band[band], key=lambda row: (row["stable_hash"], row["id"])
        )
        if len(candidates) < PER_BAND:
            raise SeedSetError(
                f"band {band} has {len(candidates)} unique positions; need {PER_BAND}"
            )
        selected.extend(candidates[:PER_BAND])
    return selected


def build_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(int(row["band"]) for row in rows)
    return {
        "type": "manifest",
        "schema": SCHEMA_VERSION,
        "set": SET_NAME,
        "version": SET_VERSION,
        "count": len(rows),
        "bands": {str(band): counts[band] for band in BANDS},
        "sha256": stable_hash(rows),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SeedSetError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise SeedSetError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def build(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    source_rows = _read_jsonl(input_path)
    selected = select_seed_rows(source_rows)
    manifest = build_manifest(selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp_path.replace(output_path)
    return manifest


def load_and_verify(path: Path = DEFAULT_OUTPUT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_jsonl(path)
    if not payload:
        raise SeedSetError(f"{path}: empty seed set")
    manifest, rows = payload[0], payload[1:]
    expected_header = {
        "type": "manifest",
        "schema": SCHEMA_VERSION,
        "set": SET_NAME,
        "version": SET_VERSION,
        "count": SET_SIZE,
        "bands": {str(band): PER_BAND for band in BANDS},
    }
    for key, want in expected_header.items():
        if manifest.get(key) != want:
            raise SeedSetError(
                f"{path}: manifest {key}={manifest.get(key)!r}, expected {want!r}"
            )
    if len(rows) != SET_SIZE:
        raise SeedSetError(f"{path}: has {len(rows)} positions, expected {SET_SIZE}")
    counts = Counter()
    seen_moves: set[str] = set()
    for index, row in enumerate(rows, 1):
        normalized = normalize_source_row(row, index + 1)
        if normalized is None:
            raise SeedSetError(f"{path}: position {index} is outside bands 3--6")
        for key in ("id", "source", "band", "placements", "moves", "stable_hash"):
            if row.get(key) != normalized[key]:
                raise SeedSetError(f"{path}: position {index} has invalid {key}")
        counts[normalized["band"]] += 1
        move_key = stable_hash(normalized["moves"])
        if move_key in seen_moves:
            raise SeedSetError(f"{path}: duplicate move prefix at position {index}")
        seen_moves.add(move_key)
    want_counts = Counter({band: PER_BAND for band in BANDS})
    if counts != want_counts:
        raise SeedSetError(f"{path}: band counts {dict(counts)}, expected {dict(want_counts)}")
    actual_hash = stable_hash(rows)
    if manifest.get("sha256") != actual_hash:
        raise SeedSetError(
            f"{path}: sha256 mismatch: manifest {manifest.get('sha256')}, actual {actual_hash}"
        )
    return manifest, rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build", help="deterministically build BENCH-SEEDS-V1")
    build_p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build_p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    verify_p = sub.add_parser("verify", help="verify pins, hash, size and stratification")
    verify_p.add_argument("--path", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build(args.input, args.out)
            path = args.out
        else:
            manifest, _rows = load_and_verify(args.path)
            path = args.path
    except (OSError, SeedSetError) as exc:
        print(f"BENCH-SEEDS-V1 ERROR: {exc}")
        return 2
    print(
        f"BENCH-SEEDS-V1 OK: {manifest['count']} positions, "
        f"sha256={manifest['sha256']} -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
