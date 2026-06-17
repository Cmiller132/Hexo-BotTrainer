"""Smoke: build the FULL /api/training/run payload for the live run through the
worktree web.py (read-only) and assert the D1/D2 fixes are present end-to-end."""
import sys
from pathlib import Path

from hexo_frontend import web

assert web.__file__.startswith("/mnt/e/hexgt-evaldash/"), web.__file__

# Force the resolver to see the live runs root regardless of cwd.
RUN_DIR = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2")
epoch_history = web._epoch_history(RUN_DIR)
evaluation_history = web._evaluation_history(RUN_DIR)
ms = web._multistage_eval_history(RUN_DIR)
live = web._training_live_status(RUN_DIR)
pool = web._eval_pool_summary(RUN_DIR)

# D1: learning_health driven by the REAL multistage eval (no false 'no eval').
health = web._learning_health(epoch_history, evaluation_history, live, ms)
msgs = health.get("messages") or []
assert not any("No SealBot evaluation result yet" in m for m in msgs), msgs
assert not any("D6 augmentation preview is missing" in m for m in msgs), msgs
assert health.get("latest_verdict") is not None, "eval chip has a real verdict"
assert health.get("latest_sealbot_winrate") is not None, "sealbot winrate present"
print(f"[smoke-D1] status={health['status']} verdict={health['latest_verdict']} "
      f"cand_elo={health['latest_cand_elo']} sb_wr={health['latest_sealbot_winrate']:.3f}")

# D2: pool edges carry slim raw physical counts for the W-L matrix.
chk = [e for e in pool["edges"] if e.get("kind") == "checkpoint"]
assert chk and all("raw" in e and "physical_wins_a" in e["raw"] for e in chk), \
    "checkpoint edges must carry slim physical_wins_*"
print(f"[smoke-D2] {len(pool['edges'])} pool edges, "
      f"{len(chk)} checkpoint edges carry physical_wins_*")

# D3: roster + dropped anchors flow to the frontend payload.
latest = ms[-1]
roster = latest["roster"]
assert "permanent_anchors" in roster and "opponents" in roster
print(f"[smoke-D3] roster perms={roster['permanent_anchors']} "
      f"opps={[o['label'] for o in roster['opponents']]}")

print("PAYLOAD SMOKE OK")
