"""Inspect the audit result structure + roll up findings directly from the
per-subsystem audits (in case synthesis is empty)."""

import json
from pathlib import Path

raw = json.load(open(Path(__file__).resolve().parent.parent / "runs" / "_audit_result.json"))
d = raw.get("result", raw)
if isinstance(d, str):
    d = json.loads(d)
print("top-level keys:", list(d.keys()))
print("synthesis type:", type(d.get("synthesis")).__name__,
      "value:", repr(d.get("synthesis"))[:200])
audits = d.get("audits", [])
print("num audits:", len(audits))

order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "nit": 4}
allf = []
for a in audits:
    sub = (a.get("subsystem", "")[:40]) if isinstance(a, dict) else str(a)[:40]
    for f in (a.get("findings", []) if isinstance(a, dict) else []):
        allf.append((order.get(f.get("severity"), 9), f.get("severity"), sub,
                     f.get("title", ""), f.get("file_line", ""), f.get("confirmed")))

allf.sort(key=lambda x: x[0])
from collections import Counter
sev_counts = Counter(f[1] for f in allf)
print("finding severity counts:", dict(sev_counts))
print("\n===== CRITICAL / HIGH findings =====")
for _, sev, sub, title, fl, conf in allf:
    if sev in ("critical", "high"):
        c = "CONFIRMED" if conf else "suspected"
        print(f"[{sev}|{c}] {sub} :: {title}\n     @ {fl}")
print("\n===== MEDIUM findings =====")
for _, sev, sub, title, fl, conf in allf:
    if sev == "medium":
        c = "CONF" if conf else "susp"
        print(f"[{c}] {sub} :: {title}  @ {fl}")
