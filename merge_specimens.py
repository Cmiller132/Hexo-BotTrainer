#!/usr/bin/env python3
"""Merge the per-source QL_*.jsonl into one canonical specimen file, adding a
stable `record_id` and dropping exact duplicate lines. Prints a per-source
count summary.

Usage: python merge_specimens.py OUT.jsonl IN1.jsonl [IN2.jsonl ...]
"""
import json
import sys
from collections import Counter


def main():
    out_path = sys.argv[1]
    ins = sys.argv[2:]
    seen = set()
    rows = []
    for p in ins:
        try:
            fh = open(p, "r", encoding="utf-8")
        except FileNotFoundError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                key = line
                if key in seen:
                    continue
                seen.add(key)
                rows.append(json.loads(line))
    for i, r in enumerate(rows):
        r["record_id"] = i
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} records to {out_path}")
    print("by_source:", dict(Counter(r["source"] for r in rows)))
    print("distinct spec_ids:", len(set(r["spec_id"] for r in rows)))


if __name__ == "__main__":
    main()
