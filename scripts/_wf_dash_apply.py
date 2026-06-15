"""Apply the dashboard-visibility workflow edits to web.py with verification.
Each edit's old_string must appear EXACTLY ONCE; aborts (no write) otherwise.
Then py_compile validates the result before it is saved."""
import json
import py_compile
import sys
import tempfile
from pathlib import Path

OUT = Path(r"C:\Users\epicm\AppData\Local\Temp\claude\E--Hexo-BotTrainer-hexgt\5e52b0ab-2ba2-4a10-980e-cc8d0599726e\tasks\wq6vj5gqk.output")
raw = json.load(open(OUT, encoding="utf-8"))
d = raw.get("result", raw)
if isinstance(d, str):
    d = json.loads(d)
impl = d["impl"]
if isinstance(impl, str):
    impl = json.loads(impl)
edits = impl["edits"]

web = Path(r"E:\Hexo-BotTrainer-hexgt\packages\hexo_frontend\python\hexo_frontend\web.py")
text = web.read_text(encoding="utf-8")
orig = text

# Verify all old_strings unique+present BEFORE mutating.
problems = []
for i, e in enumerate(edits):
    if e["file"].replace("\\", "/").split("/")[-1] != "web.py":
        problems.append(f"edit {i}: unexpected file {e['file']}")
        continue
    n = text.count(e["old_string"])
    if n != 1:
        problems.append(f"edit {i}: old_string occurs {n}x (need exactly 1) :: {e['rationale'][:80]}")
if problems:
    print("ABORT — anchor problems:")
    for p in problems:
        print("  -", p)
    sys.exit(1)

# Apply sequentially (re-check uniqueness against the evolving text).
for i, e in enumerate(edits):
    if text.count(e["old_string"]) != 1:
        print(f"ABORT mid-apply: edit {i} old_string no longer unique")
        sys.exit(1)
    text = text.replace(e["old_string"], e["new_string"], 1)
    print(f"applied edit {i}")

# Validate syntax in a temp file before overwriting.
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
    tf.write(text)
    tmp = tf.name
try:
    py_compile.compile(tmp, doraise=True)
    print("py_compile OK")
except py_compile.PyCompileError as exc:
    print("ABORT — py_compile FAILED:\n", exc)
    sys.exit(1)

web.write_text(text, encoding="utf-8")
print(f"WROTE {web} ({len(orig)} -> {len(text)} chars, {len(edits)} edits)")
