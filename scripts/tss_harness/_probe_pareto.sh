#!/bin/bash
# Cap x dual_pass Pareto sweep + dual-pass adoption (dev+holdout) run.
cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/order-prior || exit 1
PY=/root/.venvs/order-dev/bin/python
echo "=== adoption (dev+holdout, dual_pass) ==="
$PY scripts/tss_harness/runner.py run --label dualpass_adoption --tier standard \
  --adoption --no-bench --config-json '{"dual_pass": true}' 2>&1 \
  | grep -E "GATES|FAIL|_v1:|_v3:"
for cap in 250 1000 2000 4000; do
  echo "=== cap $cap + dual_pass ==="
  $PY scripts/tss_harness/runner.py run --label "pareto_cap${cap}_dp" --tier quick \
    --config-json "{\"node_cap\": $cap, \"dual_pass\": true}" 2>&1 \
    | grep -E "GATES|FAIL|_v1:|_v3:"
done
