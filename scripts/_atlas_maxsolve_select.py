"""Select prior atlas rows for a targeted escalation pass."""

from __future__ import annotations

import argparse
import glob
import os


def read_text(path: str) -> str:
    data = open(path, "rb").read()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def fields(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.split() if "=" in token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", action="append", required=True, dest="patterns")
    parser.add_argument("--out", required=True)
    parser.add_argument("--status", default="UNKNOWN")
    parser.add_argument("--min-nodes", type=int, default=0)
    parser.add_argument("--max-nodes", type=int)
    args = parser.parse_args()

    selected: dict[str, str] = {}
    seen = 0
    for path in sorted({p for pattern in args.patterns for p in glob.glob(pattern)}):
        for line in read_text(path).splitlines():
            if not line.startswith("ATLAS_ROW "):
                continue
            seen += 1
            rec = fields(line)
            nodes = int(rec["nodes"])
            if rec["status"] != args.status or nodes < args.min_nodes:
                continue
            if args.max_nodes is not None and nodes > args.max_nodes:
                continue
            old = selected.get(rec["id"])
            assert old is None or old == line, f"row drift for {rec['id']}"
            selected[rec["id"]] = line

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for row in sorted(selected.values(), key=lambda value: fields(value)["id"]):
            fh.write(row + "\n")
        fh.write(
            f"ATLAS_SELECT_DONE seen={seen} selected={len(selected)} "
            f"status={args.status} min_nodes={args.min_nodes} "
            f"max_nodes={args.max_nodes if args.max_nodes is not None else 'NA'}\n"
        )
    print({"seen": seen, "selected": len(selected), "out": os.path.abspath(args.out)})


if __name__ == "__main__":
    main()
