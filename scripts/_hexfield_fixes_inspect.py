"""Inspect the fix-workflow result: synthesis (apply_order, conflicts,
do_not_apply) + per-group edits/new-files, so we can apply skeptically."""

import json
from pathlib import Path

raw = json.load(open(Path(__file__).resolve().parent.parent / "runs" / "_fixes_result.json"))
d = raw.get("result", raw)
if isinstance(d, str):
    d = json.loads(d)

print("top-level keys:", list(d.keys()))

syn = d.get("synthesis", {})
if isinstance(syn, str):
    try:
        syn = json.loads(syn)
    except Exception:
        pass
print("\n===== SYNTHESIS =====")
print("synthesis type:", type(syn).__name__)
if isinstance(syn, dict):
    print("apply_order:", syn.get("apply_order"))
    print("\nconflicts:")
    for c in syn.get("conflicts", []):
        print("  -", c)
    print("\ndo_not_apply:")
    for c in syn.get("do_not_apply", []):
        print("  -", c)
    # any other keys
    for k, v in syn.items():
        if k not in ("apply_order", "conflicts", "do_not_apply"):
            print(f"\n[{k}]:", repr(v)[:500])
else:
    print(repr(syn)[:1000])

print("\n===== FIX GROUPS =====")
fixes = d.get("fixes", [])
print("num fix groups:", len(fixes))
for i, fx in enumerate(fixes):
    if not isinstance(fx, dict):
        print(f"\n[group {i}] (non-dict):", repr(fx)[:200])
        continue
    gid = fx.get("group") or fx.get("id") or fx.get("name") or f"group{i}"
    print(f"\n----- [{i}] {gid} -----")
    print("  summary:", (fx.get("summary") or fx.get("description") or "")[:300])
    edits = fx.get("edits", [])
    print(f"  edits: {len(edits)}")
    for j, e in enumerate(edits):
        f = e.get("file", "?")
        old = (e.get("old_string") or "")
        new = (e.get("new_string") or "")
        print(f"    [{j}] {f}  (old {len(old)}ch -> new {len(new)}ch)")
    nf = fx.get("new_files", [])
    if nf:
        print(f"  new_files: {len(nf)}")
        for n in nf:
            path = n.get("path") if isinstance(n, dict) else n
            content = n.get("content", "") if isinstance(n, dict) else ""
            print(f"    + {path}  ({len(content)}ch)")
