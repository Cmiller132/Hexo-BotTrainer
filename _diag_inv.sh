DR=/mnt/e/Hexo-BotTrainer/runs
V=/root/.venvs/hexgt-build/bin/python
echo "===== diagnostics ON DISK: hexgnn (mirror) vs dense_cnn ====="
echo "--- hexgnn_rl_main1/diagnostics ---"; ls "$DR/hexgnn_rl_main1/diagnostics" 2>/dev/null | sed 's/epoch_0*[0-9]*/epoch_N/' | sort -u | head -30
echo "--- dense_cnn_rl_main1/diagnostics ---"; ls "$DR/dense_cnn_rl_main1/diagnostics" 2>/dev/null | sed 's/epoch_0*[0-9]*/epoch_N/' | sort -u | head -30
echo ""
echo "===== /api/training/run payload TOP-LEVEL keys (both) ====="
for r in hexgnn_rl_main1 dense_cnn_rl_main1; do
  echo "--- $r ---"
  $V - "$r" <<'PY'
import sys,urllib.request,json
r=sys.argv[1]
try:
    d=json.loads(urllib.request.urlopen(f"http://127.0.0.1:8080/api/training/run?name={r}",timeout=10).read())
    for k,v in d.items():
        if isinstance(v,dict): print(f"  {k}: dict keys={list(v.keys())[:8]}")
        elif isinstance(v,list): print(f"  {k}: list[{len(v)}]")
        else: print(f"  {k}: {str(v)[:60]}")
except Exception as e: print("ERR",e)
PY
done
