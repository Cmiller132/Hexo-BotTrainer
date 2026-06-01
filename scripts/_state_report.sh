#!/usr/bin/env bash
D=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
DIAG="$D/diagnostics"
echo "===== NOW ====="; date -u +%FT%TZ
echo "===== PROCS ====="
ps -eo pid,etime,cmd | grep -E 'train_model|supervise_target_96x6_wsl' | grep -v grep
echo "===== CHECKPOINTS ====="
ls -l --time-style=+%FT%TZ "$D"/checkpoints/epoch_*.pt 2>/dev/null
ls -l --time-style=+%FT%TZ "$D"/checkpoints/*.pt 2>/dev/null | tail -5
echo "===== NEWEST SELFPLAY SHARDS (epoch2) ====="
ls -lt --time-style=+%FT%TZ "$D"/selfplay/epoch_000002_game_*.npz 2>/dev/null | head -3
echo "epoch2 npz count: $(ls "$D"/selfplay/ 2>/dev/null | grep -c epoch_000002_game_.*npz)"
echo "epoch3 npz count: $(ls "$D"/selfplay/ 2>/dev/null | grep -c epoch_000003_game_.*npz)"
echo "===== LIVE POS/S ====="
python3 -c "import json;L=json.load(open('$DIAG/dense_cnn.selfplay.live.json'));print('epoch=%s status=%s search_pos_s=%.2f full_pos_s=%.2f searched=%s games_fin=%s/%s active=%s elapsed=%.0fs ts_age=?'%(L.get('epoch'),L.get('status'),L.get('search_positions_per_second',0),L.get('positions_per_second',0),L.get('searched_positions'),L.get('games_finished'),L.get('requested_games'),L.get('active_games'),L.get('elapsed_seconds',0)))" 2>/dev/null
echo "live file mtime:"; stat -c '%y' "$DIAG/dense_cnn.selfplay.live.json" 2>/dev/null
echo "===== TRT ADOPT (last 2 across trainers) ====="
grep -h 'adopted TRT FP16\|NOT ADOPTED\|TRTAdoptError\|falling back to torch\|torch fallback' "$DIAG"/trainer.*.out.log "$DIAG"/trainer.*.err.log 2>/dev/null | tail -3
echo "===== EVENTS (last 4) ====="
tail -4 "$DIAG/events.jsonl" 2>/dev/null
echo "===== SUPERVISOR LOG (last 12) ====="
tail -12 "$DIAG/supervisor_wsl.log" 2>/dev/null
echo "===== FLAGS ====="
ls -l "$DIAG"/*.flag 2>/dev/null || echo "no halt/completed flags"
echo "===== EVAL DIAGNOSTICS ====="
ls -l --time-style=+%FT%TZ "$DIAG"/dense_cnn.evaluation.epoch_*.json 2>/dev/null || echo "no evaluation diagnostics yet"
echo "===== CALIBRATION (current) ====="
python3 -c "import json;c=json.load(open('$DIAG/dense_cnn.performance_calibration.json'));print({k:c.get(k) for k in ['measured_selfplay_positions_per_second','selected_mcts_virtual_batch_size','selected_selfplay_batch_size','selected_inference_batch_size','meets_target']})" 2>/dev/null
echo "===== GPU ====="
nvidia-smi dmon -s u -c 3 2>/dev/null | tail -3
