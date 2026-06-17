"""Apply the parity-safe perf edits (search.rs + inference.py) with uniqueness
verification. Aborts (no writes) if any old_string is missing/non-unique."""
import json, py_compile, sys
from pathlib import Path
ROOT = Path(r"E:\Hexo-BotTrainer-hexgt")
raw = json.load(open(r"C:\Users\epicm\AppData\Local\Temp\claude\E--Hexo-BotTrainer-hexgt\5e52b0ab-2ba2-4a10-980e-cc8d0599726e\tasks\wzwx6t164.output", encoding="utf-8"))
d = raw.get("result", raw)
if isinstance(d, str): d = json.loads(d)
impl = d["impl"]
if isinstance(impl, str): impl = json.loads(impl)
edits = impl["edits"]

# group edits by file, verify all anchors first
files = {}
for i, e in enumerate(edits):
    rel = e["file"].replace("\\", "/")
    rel = rel.split("packages/")[-1]
    fp = ROOT / "packages" / rel if "packages/" in e["file"].replace("\\","/") else ROOT / e["file"]
    files.setdefault(str(fp), []).append((i, e))

problems = []
texts = {}
for fp, es in files.items():
    if not Path(fp).exists():
        problems.append(f"{fp}: MISSING FILE"); continue
    t = Path(fp).read_text(encoding="utf-8"); texts[fp] = t
    for i, e in es:
        n = t.count(e["old_string"])
        if n != 1:
            problems.append(f"edit {i} {Path(fp).name}: old_string occurs {n}x (need 1)")
if problems:
    print("ABORT — anchor problems:"); [print("  -", p) for p in problems]; sys.exit(1)

# apply
for fp, es in files.items():
    t = texts[fp]
    for i, e in es:
        t = t.replace(e["old_string"], e["new_string"], 1); print(f"applied edit {i} -> {Path(fp).name}")
    texts[fp] = t

# py_compile python files in a temp before writing
import tempfile
for fp, t in texts.items():
    if fp.endswith(".py"):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(t); tmp = tf.name
        try:
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError as exc:
            print(f"ABORT — py_compile FAIL {Path(fp).name}:\n{exc}"); sys.exit(1)

for fp, t in texts.items():
    Path(fp).write_text(t, encoding="utf-8", newline="\n"); print(f"WROTE {Path(fp).name}")
print("ALL EDITS APPLIED")
