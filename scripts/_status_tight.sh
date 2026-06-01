#!/usr/bin/env bash
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
DI="$RD/diagnostics"
echo "NOW: $(date -u +%FT%TZ)"
echo "=== procs ==="; ps -eo pid,etime,cmd | grep -E 'train_model|supervise_target_96x6_wsl' | grep -v grep || echo NONE
echo "=== latest checkpoint ==="; ls -t "$RD"/checkpoints/epoch_*.pt | head -1 | xargs stat -c '%y %n'
echo "=== current stage (events last 2) ==="; tail -2 "$DI/events.jsonl"
echo "=== live ==="; python3 -c "import json;d=json.load(open('$DI/dense_cnn.selfplay.live.json'));print('epoch=%s status=%s pos/s=%.1f games=%s/%s active=%s elapsed=%.0fs'%(d.get('epoch'),d.get('status'),d.get('search_positions_per_second',0),d.get('games_finished'),d.get('requested_games'),d.get('active_games'),d.get('elapsed_seconds',0)))"
echo "  live mtime: $(stat -c %y "$DI/dense_cnn.selfplay.live.json")"
echo "=== newest selfplay shard (freshness) ==="; ls -t "$RD"/selfplay/epoch_*_game_*.npz 2>/dev/null | head -1 | xargs stat -c '%y %n'
echo "=== latest TRT adopt line (newest out log) ==="; o=$(ls -t "$DI"/trainer.*.out.log | head -1); grep -h 'adopted TRT FP16' "$o" | tail -1; echo "  (out log: $(basename "$o"))"
echo "=== TRT failures anywhere? ==="; grep -hcE 'NOT ADOPTED|TRTAdoptError' "$DI"/trainer.*.out.log "$DI"/trainer.*.err.log 2>/dev/null | paste -sd+ | bc 2>/dev/null; echo "(0 = none)"
echo "=== supervisor: starts / launches / exits / relaunches / halts ==="; grep -E 'SUPERVISOR start|LAUNCH pid|EXIT pid|RELAUNCH|halt' "$DI/supervisor_wsl.log" | tail -12
echo "=== flags ==="; ls "$DI"/*.flag 2>/dev/null || echo "none"
echo "=== EVAL TREND (all epochs) ==="; for f in $(ls "$DI"/dense_cnn.evaluation.epoch_*.json 2>/dev/null | sort); do python3 -c "import json,sys;d=json.load(open(sys.argv[1]));e=sys.argv[1].split('.epoch_')[1].split('.')[0];print('  e%s: %s-%s  mean_turns=%.1f  (games=%s)'%(int(e),d.get('wins'),d.get('losses'),d.get('mean_turns') or 0,d.get('games')))" "$f"; done
echo "=== backstop_log tail (what the scheduled task did) ==="; tail -8 "$DI/backstop_log.md" 2>/dev/null || echo "(no backstop_log.md yet)"
