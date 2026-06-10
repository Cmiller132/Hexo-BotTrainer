"""Attribute a torch.cuda memory snapshot's live allocations to call sites.

Loads a `torch.cuda.memory._dump_snapshot()` pickle and aggregates the currently
ALLOCATED blocks (live tensors at dump time) by the most-relevant user stack
frame, so we can see which code path holds the live working set. Also reports the
reserved-vs-allocated (fragmentation) split from the segments.

Usage: python scripts/_vram_snapshot_attr.py /tmp/snap_default.pickle
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict

MB = 1024 * 1024


def frame_label(frames):
    """Pick the most informative user frame (skip torch internals)."""
    if not frames:
        return "<no frames>"
    for fr in frames:
        fn = fr.get("filename", "")
        name = fr.get("name", "")
        if "hexo_models" in fn or "/scripts/" in fn or "_vram" in fn:
            short = fn.split("/")[-1]
            return f"{short}:{fr.get('line','?')} {name}"
    fr = frames[0]
    return f"{fr.get('filename','?').split('/')[-1]}:{fr.get('line','?')} {fr.get('name','')}"


def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        snap = pickle.load(f)

    segments = snap.get("segments", [])
    total_reserved = 0
    total_active = 0
    by_site = defaultdict(lambda: [0, 0])  # bytes, count
    biggest = []  # (size, label)
    for seg in segments:
        total_reserved += seg.get("total_size", 0)
        for blk in seg.get("blocks", []):
            if blk.get("state") == "active_allocated":
                sz = blk.get("size", 0)
                total_active += sz
                frames = blk.get("frames", [])
                lbl = frame_label(frames)
                by_site[lbl][0] += sz
                by_site[lbl][1] += 1
                biggest.append((sz, lbl))

    print(f"snapshot: {path}")
    print(f"  segments reserved total = {total_reserved/MB:9.1f} MiB")
    print(f"  active (live) total      = {total_active/MB:9.1f} MiB")
    print(f"  reserved-minus-active    = {(total_reserved-total_active)/MB:9.1f} MiB (free-in-reserved / frag)")
    print(f"\n  LIVE allocation by call site (top 25):")
    print(f"  {'bytes(MiB)':>11} {'count':>6}  site")
    for lbl, (b, c) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"  {b/MB:>11.1f} {c:>6}  {lbl}")

    print(f"\n  Largest single live blocks (top 15):")
    for sz, lbl in sorted(biggest, key=lambda x: -x[0])[:15]:
        print(f"  {sz/MB:>11.1f}  {lbl}")

    # Segment-size histogram (fragmentation signature): how many reserved
    # segments of each rounded size, and how full each is.
    print(f"\n  Reserved segments (size / how-full):")
    seg_rows = []
    for seg in segments:
        tot = seg.get("total_size", 0)
        act = sum(b.get("size", 0) for b in seg.get("blocks", []) if b.get("state") == "active_allocated")
        seg_rows.append((tot, act))
    for tot, act in sorted(seg_rows, key=lambda x: -x[0])[:20]:
        print(f"    seg {tot/MB:9.1f} MiB  active {act/MB:9.1f} MiB  ({100*act/max(1,tot):4.0f}% used)")
    print(f"  total segments = {len(seg_rows)}")


if __name__ == "__main__":
    main()
