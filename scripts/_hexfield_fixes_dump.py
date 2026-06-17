"""Dump full edit content for a given fix group so we can review old/new
strings before applying. Usage: python _hexfield_fixes_dump.py <group_index>"""

import json
import sys
from pathlib import Path

raw = json.load(open(Path(__file__).resolve().parent.parent / "runs" / "_fixes_result.json"))
d = raw.get("result", raw)
if isinstance(d, str):
    d = json.loads(d)

gi = int(sys.argv[1]) if len(sys.argv) > 1 else 0
fx = d["fixes"][gi]
print("=" * 70)
print(f"GROUP [{gi}]:", fx.get("group") or fx.get("id") or fx.get("name"))
print("=" * 70)
for j, e in enumerate(fx.get("edits", [])):
    print(f"\n########## EDIT [{j}] FILE: {e.get('file')} ##########")
    print("---------- OLD ----------")
    print(e.get("old_string"))
    print("---------- NEW ----------")
    print(e.get("new_string"))
for n in fx.get("new_files", []):
    path = n.get("path") if isinstance(n, dict) else n
    content = n.get("content", "") if isinstance(n, dict) else ""
    print(f"\n########## NEW FILE: {path} ({len(content)}ch) ##########")
    print(content)
