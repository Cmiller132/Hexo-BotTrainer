"""Split the stage-3 feasibility result into per-section files."""
import json
from pathlib import Path

SRC = Path("/mnt/c/Users/epicm/AppData/Local/Temp/claude/E--Hexo-BotTrainer-hexgt/5d8f6648-12b9-4654-b27f-c9b52d623d0c/tasks/wq9qdrgcq.output")
OUT = Path("/mnt/e/Hexo-BotTrainer-hexgt/scripts")

raw = SRC.read_text(encoding="utf-8", errors="replace")
data = json.loads(raw[raw.find("{"):])
if "design" not in data:
    for v in data.values():
        if isinstance(v, dict) and "design" in v:
            data = v
            break

for i, p in enumerate(data.get("probes") or []):
    (OUT / f"_wf_s3_RESULT_probe{i}.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
(OUT / "_wf_s3_RESULT_design.json").write_text(json.dumps(data.get("design"), indent=1), encoding="utf-8")
(OUT / "_wf_s3_RESULT_verdict.json").write_text(json.dumps(data.get("verdict"), indent=1), encoding="utf-8")
for f in sorted(OUT.glob("_wf_s3_RESULT_*.json")):
    print(f.name, f.stat().st_size)
