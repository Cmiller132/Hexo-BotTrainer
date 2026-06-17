"""Inspect the dashboard-visibility workflow result: synthesis edits + followups."""
import json
from pathlib import Path

p = Path(r"C:\Users\epicm\AppData\Local\Temp\claude\E--Hexo-BotTrainer-hexgt\5e52b0ab-2ba2-4a10-980e-cc8d0599726e\tasks\wq6vj5gqk.output")
raw = json.load(open(p, encoding="utf-8"))
d = raw.get("result", raw)
if isinstance(d, str):
    d = json.loads(d)
impl = d.get("impl", {})
if isinstance(impl, str):
    impl = json.loads(impl)
print("=== SUMMARY ==="); print(impl.get("summary", "")[:1500])
print("\n=== RISK NOTES ==="); print(impl.get("risk_notes", "")[:1000])
print("\n=== HEXFIELD-SIDE FOLLOWUPS ===")
for x in impl.get("hexfield_side_followups", []):
    print(" -", x)
edits = impl.get("edits", [])
print(f"\n=== EDITS ({len(edits)}) ===")
for i, e in enumerate(edits):
    print(f"\n######### EDIT [{i}] file={e.get('file')} #########")
    print("--- rationale:", e.get("rationale", "")[:300])
    print("--- OLD ({} ch) ---".format(len(e.get("old_string", ""))))
    print(e.get("old_string"))
    print("--- NEW ({} ch) ---".format(len(e.get("new_string", ""))))
    print(e.get("new_string"))
