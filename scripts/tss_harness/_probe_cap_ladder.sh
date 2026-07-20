#!/bin/bash
# Owner-directed 2026-07-20: raise the node cap under BOTH (wins AND losses)
# + the two_pass protocol arm. Quick tiers vs the standing anchor.
cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/v1-soak || exit 1
PY=/root/.venvs/harness-dev/bin/python
BASE=scripts/tss_harness/harness_runs/20260720_211755_baseline_production_v2
run () {
  echo "=== $1 ==="
  $PY scripts/tss_harness/runner.py run --label "$1" --tier quick \
    --config-json "$2" --baseline "$BASE" 2>&1 \
    | grep -E "canaries:|GATES|FAIL|_v1:|diff "
}
run cap1000_both '{"node_cap": 1000}'
run cap2000_both '{"node_cap": 2000}'
run cap4000_both '{"node_cap": 4000}'
run twopass_cap500 '{"goal": "two_pass"}'
run twopass_cap2000 '{"goal": "two_pass", "node_cap": 2000}'
