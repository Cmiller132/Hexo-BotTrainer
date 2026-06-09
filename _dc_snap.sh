RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== now: $(date '+%H:%M:%S') | driver etime: $(ps -eo etime,cmd | grep [h]exo_train.cli.train_model | grep -oE '^ *[0-9:]+' | head -1) ==="
echo "=== live self-play ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('status','epoch','requested_games','games_started','games_finished','active_games','positions_per_second','elapsed_seconds')})" 2>/dev/null
echo "=== checkpoints (epochs done) ==="
ls "$RD/checkpoints"/epoch_*.pt 2>/dev/null | wc -l
ls -t "$RD/checkpoints"/*.pt 2>/dev/null | head -3 | xargs -n1 basename 2>/dev/null
echo "=== events.jsonl: train losses + SealBot eval (tail) ==="
[ -f "$RD/diagnostics/events.jsonl" ] && tail -25 "$RD/diagnostics/events.jsonl" | $V -c "import sys,json
for ln in sys.stdin:
    try: e=json.loads(ln)
    except: continue
    s=str(e.get('stage','') or e.get('event','') or e.get('type',''))
    keys=[k for k in ('epoch','policy_loss','value_loss','loss','total_loss','sealbot','win','wins','losses','draws','winrate','win_rate','vs_sealbot','eval') if k in e]
    if keys or 'eval' in s.lower() or 'train' in s.lower() or 'seal' in str(e).lower():
        print(s, {k:e.get(k) for k in keys}, str(e)[:160] if not keys else '')" 2>/dev/null | tail -15
echo "=== any SealBot/eval lines in train.out.log ==="
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1)
grep -iE "sealbot|eval|win|loss=|policy|value" "$LOG" 2>/dev/null | tail -12
echo "=== eval diagnostic files ==="
ls "$RD/diagnostics"/ 2>/dev/null | grep -iE "eval|sealbot"
echo "=== crashes/relaunches ==="
grep -cE "RELAUNCH|HALT|EXCEPTION|EXIT code=[^0]" "$RD/supervisor.log" 2>/dev/null
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null
