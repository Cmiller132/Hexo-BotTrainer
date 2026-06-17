#!/usr/bin/env bash
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
DI="$RD/diagnostics"
echo "NOW: $(date -u +%FT%TZ)"
o=$(ls -t "$DI"/trainer.*.out.log | head -1)
echo "NEWEST out log: $o  (mtime $(stat -c %y "$o"))"
echo "--- adopt lines in NEWEST out log ---"; grep -n 'adopted TRT FP16' "$o" || echo "(none in newest out log yet)"
echo "--- events last 2 ---"; tail -2 "$DI/events.jsonl"
echo "--- live file ---"; stat -c 'mtime=%y' "$DI/dense_cnn.selfplay.live.json"
python3 -c "import json;d=json.load(open('$DI/dense_cnn.selfplay.live.json'));print('  epoch=%s status=%s pos/s=%.1f games=%s/%s active=%s elapsed=%.0fs'%(d.get('epoch'),d.get('status'),d.get('search_positions_per_second',0),d.get('games_finished'),d.get('requested_games'),d.get('active_games'),d.get('elapsed_seconds',0)))" 2>/dev/null
echo "--- epoch3 shards: count + newest mtime ---"
ls "$RD"/selfplay/epoch_000003_game_*.npz 2>/dev/null | wc -l
ls -t "$RD"/selfplay/epoch_000003_game_*.npz 2>/dev/null | head -1 | xargs -r stat -c '  newest %y %n'
echo "--- trainer 441 alive? ---"; ps -eo pid,etime,cmd | grep -E 'train_model' | grep -v grep
