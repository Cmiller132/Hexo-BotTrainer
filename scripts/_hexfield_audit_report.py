"""Print the audit synthesis (bugs/deviations/gaps/checks/verdict) compactly."""

import json
from pathlib import Path

d = json.load(open(Path(__file__).resolve().parent.parent / "runs" / "_audit_result.json"))
s = d.get("synthesis", {})

print("===== VERDICT =====")
print(s.get("verdict", "(none)"))

bugs = s.get("confirmed_bugs", [])
print(f"\n===== CONFIRMED BUGS ({len(bugs)}) =====")
order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
for b in sorted(bugs, key=lambda x: order.get(x.get("severity"), 9)):
    print(f"[{b.get('severity','?').upper()}] {b.get('subsystem','')}: {b.get('title','')}")
    print(f"   @ {b.get('file_line','')}")
    print(f"   why: {b.get('why','')}")
    print(f"   fix: {b.get('fix','')}")

dev = s.get("spec_deviations", [])
print(f"\n===== SPEC DEVIATIONS ({len(dev)}) =====")
for x in dev:
    print(" -", x)

gaps = s.get("coverage_gaps", [])
print(f"\n===== COVERAGE GAPS ({len(gaps)}) =====")
for x in gaps:
    print(" -", x)

checks = s.get("runtime_checks", [])
print(f"\n===== RUNTIME CHECKS ({len(checks)}) =====")
for x in checks:
    print(" -", x)

print(f"\n===== CLEAN SUBSYSTEMS =====")
print(", ".join(s.get("clean_subsystems", [])))
