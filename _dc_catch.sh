RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
for i in $(seq 1 80); do
  gf=$($V -c "import json;print(json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json')).get('games_finished',0))" 2>/dev/null)
  if [ "${gf:-0}" -ge 1 ]; then echo "FIRST GAMES at $(date '+%H:%M:%S'): games_finished=$gf"; break; fi
  sleep 20
done
echo "=== live status ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('games_finished','completed_games','active_games','searched_positions','positions_per_second','elapsed_seconds')})" 2>/dev/null
echo ".hxr size: $(stat -c %s "$RD/selfplay/epoch_000001.hxr" 2>/dev/null) B"
echo "=== dashboard history-count API (does it now show games?) ==="
$V - <<'PY'
import urllib.request, json
try:
    for ep in ("history-count","history"):
        u=f"http://127.0.0.1:8080/api/training/{ep}?name=dense_cnn_rl_main1"
        try:
            r=json.loads(urllib.request.urlopen(u,timeout=8).read().decode())
            print(ep,"->", json.dumps(r)[:200])
        except Exception as e: print(ep,"ERR",e)
    r=json.loads(urllib.request.urlopen("http://127.0.0.1:8080/api/training/run?name=dense_cnn_rl_main1",timeout=8).read().decode())
    print("run.games=",r.get("games"),"run.history.games=",(r.get("history") or {}).get("games"))
except Exception as e: print("API ERR",e)
PY
