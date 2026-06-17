import json
from pathlib import Path
raw = json.load(open(r"C:\Users\epicm\AppData\Local\Temp\claude\E--Hexo-BotTrainer-hexgt\5e52b0ab-2ba2-4a10-980e-cc8d0599726e\tasks\wzwx6t164.output", encoding="utf-8"))
d = raw.get("result", raw)
if isinstance(d, str): d = json.loads(d)
plan = d.get("plan", {})
if isinstance(plan, str): plan = json.loads(plan)
impl = d.get("impl", {})
if isinstance(impl, str): impl = json.loads(impl)
print("=== PLAN apply_order ==="); [print(" -", x) for x in plan.get("apply_order", [])]
print("\n=== PLAN do_not_auto_apply ==="); [print(" -", x) for x in plan.get("do_not_auto_apply", [])]
print("\n=== IMPL summary ==="); print(impl.get("summary","")[:800])
print("\n=== VALIDATION COMMANDS ==="); [print(" -", x) for x in impl.get("validation_commands", [])]
edits = impl.get("edits", [])
print(f"\n=== EDITS ({len(edits)}) ===")
for i, e in enumerate(edits):
    print(f"\n###### EDIT [{i}] {e.get('file')}  [{e.get('parity_class')}] ######")
    print("rationale:", e.get("rationale","")[:260])
    print("--- OLD ({}ch) ---".format(len(e.get("old_string",""))))
    print(e.get("old_string"))
    print("--- NEW ({}ch) ---".format(len(e.get("new_string",""))))
    print(e.get("new_string"))
print("\n=== DEFERRED PIPELINING DESIGN (not applied) ===")
print((impl.get("deferred_pipelining_design","") or "")[:1200])
