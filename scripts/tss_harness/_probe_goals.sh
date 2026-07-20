#!/bin/bash
# One-shot: quick-tier probes of the goal=loss and goal=both arms.
cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/v1-soak || exit 1
PY=/root/.venvs/harness-dev/bin/python
for g in loss both; do
  echo "=== goal=$g ==="
  $PY scripts/tss_harness/runner.py run --label "probe_goal_$g" --tier quick \
    --config-json "{\"goal\": \"$g\"}" 2>&1 | grep -E "canaries:|GATES|FAIL|_v1:"
done
