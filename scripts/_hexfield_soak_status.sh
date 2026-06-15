#!/bin/bash
RD=/mnt/e/Hexo-BotTrainer/runs/hexfield_soak
echo "=== epoch-2 selfplay ==="
grep -E 'rows_written|mean_game_length|root_policy_entropy|games_finished|truncated_games' \
  "$RD/diagnostics/hexfield.selfplay.epoch_000002.json" 2>/dev/null || echo "  (not yet)"
echo "=== epoch-2 training ==="
grep -E 'status|window_rows|steps|loss_total|loss_value' \
  "$RD/diagnostics/hexfield.training.epoch_000002.json" 2>/dev/null || echo "  (not yet)"
echo "=== epoch-2 eval arena ==="
cat "$RD/diagnostics/hexfield.evaluation.epoch_000002.json" 2>/dev/null || echo "  (not yet)"
echo "=== checkpoints ==="
ls "$RD/checkpoints/" 2>/dev/null || echo "  (none)"
echo "=== events tail ==="
tail -3 "$RD/diagnostics/events.jsonl" 2>/dev/null | cut -c1-150
