#!/usr/bin/env bash
cd /mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
sleep 5
curl -s "localhost:8080/api/training/history-page?run=hexgt_rl_main2&source=selfplay&limit=400" > /tmp/sp.json
python3 - <<'PY'
import json
import hexo_engine as engine, hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id
from hexo_runner.records import HexoRecordFile
h = json.load(open("/tmp/sp.json"))
items = h.get("items", [])
from collections import Counter
by_ep = Counter(i.get("epoch") for i in items)
print("self-play games per DISPLAY epoch:", dict(sorted(by_ep.items())))
# Inspect the CURRENT display epoch (max) — should be FULL real games now.
cur = max(by_ep)
cur_items = [i for i in items if i.get("epoch") == cur][:5]
print(f"\nCURRENT display epoch {cur} sample game labels:")
for i in cur_items:
    print("  ", i.get("game_id"), "| winner", i.get("winner"))
# Decisive: open the served mirror .hxr for the current epoch and check terminal.
mirror = f"/mnt/e/Hexo-BotTrainer/runs/hexgt_rl_main2/selfplay/epoch_{cur:06d}.hxr"
recs = list(HexoRecordFile.open(mirror).iter_records())
term = 0
for r in recs[:30]:
    s = eng.new_game()
    for aid in r.action_ids:
        eng.apply_action(s, engine.PlacementAction(unpack_coord_id(int(aid))))
    if eng.terminal(s) is not None:
        term += 1
print(f"\nserved mirror epoch_{cur:06d}.hxr: {len(recs)} games; first 30 reach TERMINAL: {term}/30")
preview = sum(1 for i in items if "sampled preview" in str(i.get("game_id","")))
print(f"games still labeled 'sampled preview' (old pre-fix epochs): {preview}")
PY
