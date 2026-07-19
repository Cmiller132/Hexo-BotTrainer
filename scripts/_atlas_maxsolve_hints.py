#!/usr/bin/env python3
"""Emit the minimal raw-row fields consumed by the atlas parent-lift harness.

Published decisive rows and newly certified raw rows are combined additively.
The Rust harness treats these only as scheduling hints: it re-solves the child,
strict-verifies that certificate, reconstructs the parent certificate, and
strict-verifies the parent before emitting a verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def field(line: str, name: str) -> str | None:
    prefix = f"{name}="
    for token in line.split():
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--upgrade-raw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    atlas = json.loads(Path(args.atlas).read_text(encoding="utf-8"))
    rows: dict[str, tuple[int, str, str]] = {}
    for row in atlas["rows"]:
        if row["status"] in {"WIN", "LOSS"}:
            rows[row["id"]] = (
                int(row["source_prefix"]),
                row["status"],
                row["claimant"],
            )

    if args.upgrade_raw:
        for line in Path(args.upgrade_raw).read_text(encoding="utf-8-sig").splitlines():
            if not line.startswith("ATLAS_ROW ") or field(line, "certified") != "1":
                continue
            status = field(line, "status")
            if status not in {"WIN", "LOSS"}:
                continue
            row_id = field(line, "id")
            depth = field(line, "source_prefix")
            claimant = field(line, "claimant")
            if row_id is None or depth is None or claimant is None:
                raise ValueError(f"incomplete decisive raw row: {line}")
            value = (int(depth), status, claimant)
            previous = rows.setdefault(row_id, value)
            if previous != value:
                raise ValueError(f"verdict drift for {row_id}: {previous} != {value}")

    output = ["ATLAS_HINTS schema=1"]
    for row_id, (depth, status, claimant) in sorted(rows.items()):
        output.append(
            " ".join(
                (
                    "ATLAS_ROW",
                    f"id={row_id}",
                    f"source_prefix={depth}",
                    f"status={status}",
                    "certified=1",
                    f"claimant={claimant}",
                )
            )
        )
    output.append(f"ATLAS_HINTS_DONE decisive={len(rows)}")
    Path(args.out).write_text("\n".join(output) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"decisive": len(rows), "out": str(Path(args.out).resolve())}))


if __name__ == "__main__":
    main()
