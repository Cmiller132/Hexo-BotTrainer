echo "=== /api/training/runs (run list shown in the DOM) ==="
curl -s "http://localhost:8080/api/training/runs" | python3 -c "import sys,json; d=json.load(sys.stdin); rl=d if isinstance(d,list) else d.get('runs',d); print(json.dumps(rl,indent=0)[:600])" 2>/dev/null || curl -s "http://localhost:8080/api/training/runs" | head -c 500
echo; echo "=== /api/training/run?name=hexgnn_rl_main1 (epochs/status the page renders) ==="
curl -s "http://localhost:8080/api/training/run?name=hexgnn_rl_main1" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('name:',d.get('name'),'status:',d.get('status'))
eh=d.get('epoch_history') or d.get('histories') or {}
ev=d.get('evaluation_history')
dbe=d.get('diagnostics_by_epoch') or {}
ls=d.get('live_status') or {}
print('diagnostics_by_epoch keys:', sorted(list(dbe.keys()))[:6])
print('evaluation_history entries:', len(ev) if isinstance(ev,list) else ev)
print('live_status:', json.dumps(ls)[:200])
" 2>/dev/null || echo "(parse failed) raw:" && curl -s "http://localhost:8080/api/training/run?name=hexgnn_rl_main1" | head -c 400
echo; echo "=== /api/debug/checkpoints (debug tab via HEXO_DEBUG_RUN_ROOT) ==="
curl -s "http://localhost:8080/api/debug/checkpoints" | head -c 300
