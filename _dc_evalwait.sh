RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1)
for i in $(seq 1 40); do
  # eval result signals: an eval diag file, an events line with win/sealbot result, epoch_000002 ckpt, or a vs-sealbot line in log
  res=$(grep -rhiE "[0-9]+ ?W ?/ ?[0-9]+ ?L|win[_ ]?rate|vs.?sealbot|sealbot.*=.*%|\"wins\"|\"win_rate\"|evaluation.*win" "$RD/diagnostics/events.jsonl" "$LOG" "$RD"/diagnostics/*eval*.json 2>/dev/null | grep -viE "winners" | tail -3)
  [ -n "$res" ] && { echo "EVAL RESULT:"; echo "$res" | cut -c1-220; break; }
  [ -f "$RD/checkpoints/epoch_000002.pt" ] && { echo "epoch_000002.pt appeared (epoch advanced)"; break; }
  if ! ps -eo cmd|grep -q "[h]exo_train.cli.train_model"; then echo "DRIVER GONE"; break; fi
  sleep 15
done
echo "=== eval diag files ==="; ls "$RD/diagnostics"/ | grep -iE "eval" || echo "(none)"
echo "=== latest events.jsonl entries (parsed for win/eval) ==="
$V - "$RD/diagnostics/events.jsonl" <<'PY'
import sys,json
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for e in rows[-6:]:
    p=e.get('payload',e); s=p.get('stage') or e.get('event')
    b=json.dumps(p).lower()
    print(s, '| has eval/win:', any(t in b for t in ('eval','win','seal')))
PY
echo "=== checkpoints + GPU + crashes ==="
ls "$RD/checkpoints"/*.pt 2>/dev/null | xargs -n1 basename
echo "crashes: $(grep -cE 'RELAUNCH|HALT|EXIT code=[^0]' "$RD/supervisor.log" 2>/dev/null)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
