#!/usr/bin/env bash
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
DI="$RD/diagnostics"
echo "NOW                : $(date -u +%FT%TZ)"
echo "1. newest_checkpoint: $(ls -t "$RD"/checkpoints/epoch_*.pt 2>/dev/null | head -1 | xargs -r stat -c '%n  (mtime %y)')"
# 2: stage name ONLY from last events line (no payload), + events.jsonl mtime as the timestamp
st=$(tail -1 "$DI/events.jsonl" | python3 -c "import json,sys;d=json.loads(sys.stdin.read());print(d.get('event'), d.get('payload',{}).get('stage'))" 2>/dev/null)
echo "2. last_event_stage : $st   (events.jsonl mtime $(stat -c %y "$DI/events.jsonl"))"
# 3: newest selfplay shard mtime + age in seconds
sh=$(ls -t "$RD"/selfplay/*.npz 2>/dev/null | head -1)
if [ -n "$sh" ]; then age=$(( $(date +%s) - $(stat -c %Y "$sh") )); echo "3. newest_shard     : $(basename "$sh")  mtime $(stat -c %y "$sh")  age=${age}s  alive=$([ $age -lt 600 ] && echo YES || echo 'STALE>10min')"; else echo "3. newest_shard     : none"; fi
# 4: PIDs
tp=$(pgrep -f 'train_model .*dense_cnn_model1_target_96x6'); sp=$(pgrep -f 'supervise_target_96x6_wsl.sh' | tr '\n' ',')
echo "4. trainer_pid      : ${tp:-NONE} (alive=$([ -n "$tp" ] && echo yes || echo no))   supervisor_pids: ${sp:-NONE}"
# 5: latest TRT adopt line (newest out log)
o=$(ls -t "$DI"/trainer.*.out.log 2>/dev/null | head -1)
echo "5. latest_trt_adopt : $(grep -h 'adopted TRT FP16' "$o" 2>/dev/null | tail -1)"
# 6: eval win-rate per epoch
echo "6. eval_winrate:"
for f in $(ls "$DI"/dense_cnn.evaluation.epoch_*.json 2>/dev/null | sort -V); do
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));e=int(sys.argv[1].split('.epoch_')[1].split('.')[0]);print('   epoch%d: %s/%s'%(e,d.get('wins'),d.get('games')))" "$f"
done
# 7: flags + crash/halt counts in supervisor log
echo "7. flags            : $(ls "$DI"/*.flag 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ' || echo none)$([ -z "$(ls "$DI"/*.flag 2>/dev/null)" ] && echo none)"
echo "   supervisor EXITs=$(grep -c 'EXIT pid' "$DI/supervisor_wsl.log") RELAUNCHes=$(grep -c 'RELAUNCH' "$DI/supervisor_wsl.log") halts=$(grep -c 'halt' "$DI/supervisor_wsl.log")  (since last SUPERVISOR start 05:03:56Z: EXITs=$(awk '/SUPERVISOR start/{c=0} /EXIT pid/{c++} END{print c}' "$DI/supervisor_wsl.log"))"
