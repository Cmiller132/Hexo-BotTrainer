"""Finish applying the debug_infer hexfield edits (mixed/partial state): apply
each edit whose old_string is uniquely present; skip those already applied
(new_string present). Abort if an edit is neither applicable nor already done.
py_compile-gates the result before writing."""
import json, py_compile, sys, tempfile
from pathlib import Path
raw = json.load(open(r"C:\Users\epicm\AppData\Local\Temp\claude\E--Hexo-BotTrainer-hexgt\5e52b0ab-2ba2-4a10-980e-cc8d0599726e\tasks\wdtpfmc48.output", encoding="utf-8"))
d = raw.get("result", raw)
if isinstance(d, str): d = json.loads(d)
impl = d["impl"]
if isinstance(impl, str): impl = json.loads(impl)
edits = impl["edits"]
fp = Path(r"E:\Hexo-BotTrainer-hexgt\packages\hexo_frontend\python\hexo_frontend\debug_infer.py")
t = fp.read_text(encoding="utf-8")

applied, skipped, problems = [], [], []
for i, e in enumerate(edits):
    old, new = e["old_string"], e["new_string"]
    if t.count(new) >= 1 and t.count(old) == 0:
        skipped.append(i); continue          # already applied
    if t.count(old) == 1:
        t = t.replace(old, new, 1); applied.append(i); continue
    if t.count(new) >= 1 and t.count(old) >= 1:
        # both present: new already in file, old still elsewhere — apply remaining old
        t = t.replace(old, new, 1); applied.append(i); continue
    problems.append(f"edit {i}: old occurs {t.count(old)} (need 1), new occurs {t.count(new)} :: {e.get('rationale','')[:80]}")

if problems:
    print("ABORT — unresolved edits:"); [print("  -", p) for p in problems]; sys.exit(1)

with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
    tf.write(t); tmp = tf.name
try:
    py_compile.compile(tmp, doraise=True)
except py_compile.PyCompileError as exc:
    print("ABORT — py_compile FAILED:\n", exc); sys.exit(1)
fp.write_text(t, encoding="utf-8", newline="\n")
print(f"applied={applied} skipped(already)={skipped}  py_compile OK  WROTE debug_infer.py")
